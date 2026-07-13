// Notes reactive store (Svelte 5 runes, module singleton). Mirrors
// chat-state.svelte.ts's shape (module-level $state + `ensure()` + exported
// getter object).
//
// `savedMessageIds` (derived from `source_message_id`) is the SINGLE source of
// truth for a chat message's saved-state — MessageActions reads it, never a
// separately-tracked flag.

import {
  saveChatNote,
  saveManualNote,
  listNotes,
  deleteNote,
  updateNote,
  setNotePinned
} from './ipc.js';
import { parseCitations } from '$lib/chat/citations.js';
import type { Note } from './types.js';
import type { ChatMessage } from '$lib/chat/types.js';

/** Rendering-only grouping: notes sharing the ordinal-1 citation's `source_id`. */
export interface NoteGroup {
  sourceId: string | null;
  sourceTitle: string | null;
  notes: Note[];
}

let byNotebook = $state<Record<string, Note[]>>({});
/** Per-notebook hydrate generation (mirrors chat-state's streamGeneration): a
 * stale `listNotes` resolving after a newer hydrate/toggle must not clobber.
 * Plain (non-reactive) Map — only `byNotebook` needs to drive the UI. */
const hydrateGeneration = new Map<string, number>();
/** In-flight `toggleSave` message ids per notebook — guards a check-then-act
 * race where two fast clicks both see "not saved" and insert two notes. */
const pendingToggles = new Map<string, Set<string>>();

function ensure(notebookId: string): Note[] {
  let notes = byNotebook[notebookId];
  if (!notes) {
    notes = [];
    byNotebook[notebookId] = notes;
  }
  return notes;
}

function ensurePending(notebookId: string): Set<string> {
  let pending = pendingToggles.get(notebookId);
  if (!pending) {
    pending = new Set();
    pendingToggles.set(notebookId, pending);
  }
  return pending;
}

/** Bumps and returns a notebook's generation counter (hydrate + toggleSave share it). */
function bumpGeneration(notebookId: string): number {
  const gen = (hydrateGeneration.get(notebookId) ?? 0) + 1;
  hydrateGeneration.set(notebookId, gen);
  return gen;
}

/** Ordinal-1 citation's `source_id` for a note, or `null` when uncited. */
function primarySourceId(note: Note): string | null {
  const citations = parseCitations(note.citations);
  return citations?.find((c) => c.ordinal === 1)?.source_id ?? null;
}

/**
 * Client-side ordering mirroring the engine's `list_notes` ORDER BY
 * (`pinned DESC, created_at DESC, id DESC`) so an optimistic pin toggle re-floats
 * the note without a re-hydrate. Stable, non-mutating (returns a sorted copy).
 */
function sortNotes(notes: Note[]): Note[] {
  return [...notes].sort((a, b) => {
    if (a.pinned !== b.pinned) return a.pinned ? -1 : 1;
    if (a.created_at !== b.created_at) return a.created_at < b.created_at ? 1 : -1;
    return a.id < b.id ? 1 : a.id > b.id ? -1 : 0;
  });
}

/** Hydrates a notebook's saved notes from `notes` (newest-first, as returned). */
export async function hydrate(notebookId: string): Promise<void> {
  const gen = bumpGeneration(notebookId);
  const rows = await listNotes(notebookId);
  if (hydrateGeneration.get(notebookId) !== gen) return;
  byNotebook[notebookId] = rows;
}

/**
 * Toggles save-state for a chat message (AC24): if already saved, deletes the
 * backing note; otherwise saves a new snapshot and prepends it (newest-first).
 * No-ops for a synthetic streaming-bubble id and while a toggle for this
 * message is already in flight (belt-and-suspenders alongside the UI gate).
 */
export async function toggleSave(notebookId: string, message: ChatMessage): Promise<void> {
  if (message.id.endsWith('-streaming')) return;
  const pending = ensurePending(notebookId);
  if (pending.has(message.id)) return;
  pending.add(message.id);
  try {
    const gen = bumpGeneration(notebookId);
    const notes = ensure(notebookId);
    const existing = notes.find((n) => n.source_message_id === message.id);
    if (existing) {
      await deleteNote(existing.id);
      if (hydrateGeneration.get(notebookId) !== gen) return;
      byNotebook[notebookId] = notes.filter((n) => n.id !== existing.id);
      return;
    }
    const citations = parseCitations(message.citations);
    const saved = await saveChatNote(notebookId, message.content, citations, message.id);
    if (hydrateGeneration.get(notebookId) !== gen) return;
    byNotebook[notebookId] = [saved, ...notes];
  } finally {
    pending.delete(message.id);
  }
}

/**
 * Saves a user-authored manual note and prepends it (newest-first). No-ops on
 * empty/whitespace content (belt-and-suspenders alongside the engine's guard and
 * the composer's disabled state). Shares the hydrate generation guard so a stale
 * `listNotes` cannot clobber the freshly-prepended note.
 */
export async function addManualNote(notebookId: string, content: string): Promise<void> {
  if (content.trim().length === 0) return;
  const gen = bumpGeneration(notebookId);
  const notes = ensure(notebookId);
  const saved = await saveManualNote(notebookId, content);
  if (hydrateGeneration.get(notebookId) !== gen) return;
  byNotebook[notebookId] = [saved, ...notes];
}

/** Removes a note by id (idempotent — no-op if the row is absent). */
export async function remove(notebookId: string, noteId: string): Promise<void> {
  await deleteNote(noteId);
  const notes = byNotebook[notebookId];
  if (!notes) return;
  byNotebook[notebookId] = notes.filter((n) => n.id !== noteId);
}

/**
 * Edits a note's content in place, replacing the row with the returned snapshot
 * (bumped `updated_at`, grounding cols preserved by the engine). No-ops on
 * empty/whitespace content (belt-and-suspenders alongside the engine guard).
 * Shares the hydrate generation guard so a stale `listNotes` cannot clobber.
 */
export async function editNote(notebookId: string, noteId: string, content: string): Promise<void> {
  if (content.trim().length === 0) return;
  const gen = bumpGeneration(notebookId);
  const updated = await updateNote(noteId, content);
  if (hydrateGeneration.get(notebookId) !== gen) return;
  const notes = byNotebook[notebookId];
  if (!notes) return;
  byNotebook[notebookId] = notes.map((n) => (n.id === noteId ? updated : n));
}

/**
 * Pins/unpins a note, replacing the row with the returned snapshot and re-sorting
 * so pinned notes float to the top (mirrors the engine ORDER BY). Shares the
 * hydrate generation guard so a stale `listNotes` cannot clobber the new order.
 */
export async function setPinned(
  notebookId: string,
  noteId: string,
  pinned: boolean
): Promise<void> {
  const gen = bumpGeneration(notebookId);
  const updated = await setNotePinned(noteId, pinned);
  if (hydrateGeneration.get(notebookId) !== gen) return;
  const notes = byNotebook[notebookId];
  if (!notes) return;
  byNotebook[notebookId] = sortNotes(notes.map((n) => (n.id === noteId ? updated : n)));
}

/**
 * Case-insensitive substring match over a note's `content` + `source_title`
 * (both sections). Pure — used by the ⌘K palette's Notes-tab branch. An empty
 * query matches everything (mirrors the notebook-title branch).
 */
export function noteMatchesQuery(note: Note, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (q.length === 0) return true;
  if (note.content.toLowerCase().includes(q)) return true;
  return note.source_title?.toLowerCase().includes(q) ?? false;
}

export const notesStore = {
  notes(notebookId: string): Note[] {
    return byNotebook[notebookId] ?? [];
  },
  /** Chat message ids with a backing note — the sole source of truth for saved-state. */
  savedMessageIds(notebookId: string): Set<string> {
    const notes = byNotebook[notebookId] ?? [];
    const ids = new Set<string>();
    for (const note of notes) {
      if (note.source_message_id) ids.add(note.source_message_id);
    }
    return ids;
  },
  /** User-authored manual notes only, pinned-first then newest-first. */
  manualNotes(notebookId: string): Note[] {
    return sortNotes((byNotebook[notebookId] ?? []).filter((n) => n.origin === 'manual'));
  },
  /** Notes grouped by the ordinal-1 citation's `source_id`; uncited notes group under `null`. Pinned-first then newest-first within each group. Chat notes only. */
  groupedBySource(notebookId: string): NoteGroup[] {
    const notes = sortNotes((byNotebook[notebookId] ?? []).filter((n) => n.origin === 'chat'));
    const order: Array<string | null> = [];
    const groups = new Map<string | null, NoteGroup>();
    for (const note of notes) {
      const sourceId = primarySourceId(note);
      let group = groups.get(sourceId);
      if (!group) {
        group = { sourceId, sourceTitle: note.source_title, notes: [] };
        groups.set(sourceId, group);
        order.push(sourceId);
      }
      group.notes.push(note);
    }
    return order.map((key) => groups.get(key)!);
  }
};

/** Reset all state. Call in `afterEach` to prevent cross-test bleed. */
export function resetNotesStore(): void {
  byNotebook = {};
  hydrateGeneration.clear();
  pendingToggles.clear();
}
