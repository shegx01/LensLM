// Notes reactive store (Svelte 5 runes, module singleton). Mirrors
// chat-state.svelte.ts's shape (module-level $state + `ensure()` + exported
// getter object).
//
// `savedMessageIds` (derived from `source_message_id`) is the SINGLE source of
// truth for a chat message's saved-state — MessageActions reads it, never a
// separately-tracked flag.

import { saveChatNote, listNotes, deleteNote } from './ipc.js';
import type { Note } from './types.js';
import type { ChatMessage, Citation } from '$lib/chat/types.js';

/** Rendering-only grouping: notes sharing the ordinal-1 citation's `source_id`. */
export interface NoteGroup {
  sourceId: string | null;
  sourceTitle: string | null;
  notes: Note[];
}

let byNotebook = $state<Record<string, Note[]>>({});

function ensure(notebookId: string): Note[] {
  let notes = byNotebook[notebookId];
  if (!notes) {
    notes = [];
    byNotebook[notebookId] = notes;
  }
  return notes;
}

function parseCitations(json: string | null): Citation[] | null {
  if (json === null) return null;
  try {
    return JSON.parse(json) as Citation[];
  } catch (err) {
    console.warn('notes-state: failed to parse citations JSON', err);
    return null;
  }
}

/** Ordinal-1 citation's `source_id` for a note, or `null` when uncited. */
function primarySourceId(note: Note): string | null {
  const citations = parseCitations(note.citations);
  return citations?.find((c) => c.ordinal === 1)?.source_id ?? null;
}

/** Hydrates a notebook's saved notes from `notes` (newest-first, as returned). */
export async function hydrate(notebookId: string): Promise<void> {
  const rows = await listNotes(notebookId);
  byNotebook[notebookId] = rows;
}

/**
 * Toggles save-state for a chat message (AC24): if already saved, deletes the
 * backing note; otherwise saves a new snapshot and prepends it (newest-first).
 */
export async function toggleSave(notebookId: string, message: ChatMessage): Promise<void> {
  const notes = ensure(notebookId);
  const existing = notes.find((n) => n.source_message_id === message.id);
  if (existing) {
    await deleteNote(existing.id);
    byNotebook[notebookId] = notes.filter((n) => n.id !== existing.id);
    return;
  }
  const citations = parseCitations(message.citations);
  const saved = await saveChatNote(notebookId, message.content, citations, message.id);
  byNotebook[notebookId] = [saved, ...notes];
}

/** Removes a note by id (idempotent — no-op if the row is absent). */
export async function remove(notebookId: string, noteId: string): Promise<void> {
  await deleteNote(noteId);
  const notes = byNotebook[notebookId];
  if (!notes) return;
  byNotebook[notebookId] = notes.filter((n) => n.id !== noteId);
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
  /** Notes grouped by the ordinal-1 citation's `source_id`; uncited notes group under `null`. Newest-first preserved. */
  groupedBySource(notebookId: string): NoteGroup[] {
    const notes = byNotebook[notebookId] ?? [];
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
}
