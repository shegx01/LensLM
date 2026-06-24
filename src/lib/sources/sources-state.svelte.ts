// Sources reactive store (Svelte 5 runes, module singleton).
//
// Module-level `$state` singleton — mirrors `notebooks-state.svelte.ts`.
// All SourcesRail consumers read from a single source of truth without prop drilling.
//
// LOADING LIFECYCLE: every CRUD action wraps its IPC call with `loading = true`
// before the call and `loading = false` in a `finally` block.
//
// ERROR HANDLING: try/catch on every action; console.error on failure; transient
// `error` field for future surfacing. Polished error UI is M9 scope.

import { listSources, ingestSource, setSourceSelected, trashSource, restoreSource } from './ipc.js';
import type { Source, SourceStatus, StreamEvent, IngestProgress } from './types.js';
import { notebookStore } from '$lib/notebooks/notebooks-state.svelte.js';

// ---------------------------------------------------------------------------
// Module-level reactive state
// ---------------------------------------------------------------------------

let sources = $state<Source[]>([]);
// TODO(M9): single `loading` boolean flickers under concurrent/compound actions — replace with a counter when wiring loading UI.
let loading = $state(false);
// TODO(M9): `error` is written but not yet surfaced in UI (polished error states are M9).
let error = $state<string | null>(null); // transient; polished surfacing deferred to M9

// ---------------------------------------------------------------------------
// Recently-trashed stash — holds the last soft-deleted source + its original
// list index so undoRemove() can re-insert it at the right position.
// ---------------------------------------------------------------------------

interface TrashStash {
  source: Source;
  index: number;
  timeoutId: ReturnType<typeof setTimeout>;
}

let trashStash = $state<TrashStash | null>(null);

// ---------------------------------------------------------------------------
// Exported store object (getter/setter pairs — project pattern)
// ---------------------------------------------------------------------------

export const sourcesStore = {
  get sources() {
    return sources;
  },
  get loading() {
    return loading;
  },
  get error() {
    return error;
  },
  set error(e: string | null) {
    error = e;
  },
  /** Whether a soft-deleted source is currently pending undo (drives the Undo bar). */
  get recentlyTrashed(): boolean {
    return trashStash !== null;
  }
};

// ---------------------------------------------------------------------------
// Actions (exported top-level functions)
// ---------------------------------------------------------------------------

/** Fetch all sources for the given notebook and populate the store. */
export async function loadSources(notebookId: string): Promise<void> {
  error = null;
  loading = true;
  try {
    sources = await listSources(notebookId);
  } catch (err) {
    console.error('loadSources: failed', err);
    error = String(err);
  } finally {
    loading = false;
  }
}

/**
 * Ingest a source by id. Subscribes to the progress channel and updates
 * the matching row's status optimistically as events arrive.
 *
 * Status transitions: queued → parsing → embedding → indexed | error.
 */
/** Maps the Rust IngestProgress.phase string to the SourceStatus union. */
function phaseToStatus(phase: string): SourceStatus {
  switch (phase) {
    case 'started':
      return 'parsing';
    case 'parsing':
    case 'chunking':
      return 'parsing';
    case 'model_download':
    case 'embedding':
    case 'indexing':
      return 'embedding';
    case 'done':
      return 'indexed';
    case 'failed':
      return 'error';
    default:
      return 'parsing';
  }
}

export async function ingest(sourceId: string): Promise<void> {
  function handleEvent(e: StreamEvent<IngestProgress>): void {
    if (e.type === 'chunk' && e.data) {
      // 'chunk' carries IngestProgress { phase, done, total } — update row status.
      const status = phaseToStatus(e.data.phase);
      sources = sources.map((s) => (s.id === sourceId ? { ...s, status } : s));
    } else if (e.type === 'progress') {
      // 'progress' carries only { done, total } — no phase; reserved for future progress bar.
    } else if (e.type === 'done') {
      sources = sources.map((s) => (s.id === sourceId ? { ...s, status: 'indexed' } : s));
    } else if (e.type === 'failed') {
      sources = sources.map((s) => (s.id === sourceId ? { ...s, status: 'error' } : s));
    } else if (e.type === 'started') {
      sources = sources.map((s) => (s.id === sourceId ? { ...s, status: 'parsing' } : s));
    }
  }

  try {
    await ingestSource(sourceId, handleEvent);
  } catch (err) {
    console.error('ingest: failed for source', sourceId, err);
    sources = sources.map((s) => (s.id === sourceId ? { ...s, status: 'error' } : s));
    error = String(err);
  }
}

/**
 * Toggle a source's selected state. Persists via setSourceSelected IPC
 * and updates local state immediately (optimistic).
 */
export async function toggleSelected(sourceId: string): Promise<void> {
  const current = sources.find((s) => s.id === sourceId);
  if (!current) return;
  const newSelected = current.selected === 1 ? 0 : 1;
  // Optimistic update
  sources = sources.map((s) => (s.id === sourceId ? { ...s, selected: newSelected } : s));
  try {
    // Rust serde expects JSON bool — convert the INTEGER column value to boolean.
    await setSourceSelected(sourceId, newSelected !== 0);
  } catch (err) {
    // Revert on failure
    sources = sources.map((s) => (s.id === sourceId ? { ...s, selected: current.selected } : s));
    console.error('toggleSelected: failed for source', sourceId, err);
    error = String(err);
  }
}

/** How long (ms) the Undo bar is shown before the stash is cleared. */
const TRASH_UNDO_TTL_MS = 6_000;

/**
 * Soft-delete a source (move to trash). Optimistic: the row is removed from
 * local state immediately, then the IPC `trash_source` command is awaited.
 * On failure the row is restored and `error` is set.
 *
 * A "recently trashed" stash is held for TRASH_UNDO_TTL_MS so the caller
 * (SourcesRail) can render an Undo bar. Initiating another delete clears
 * the previous stash immediately.
 */
export async function removeSource(sourceId: string): Promise<void> {
  const snapshot = sources.find((s) => s.id === sourceId);
  if (!snapshot) return;
  const originalIndex = sources.indexOf(snapshot);

  // Clear any pre-existing stash + its timeout before starting a new one.
  if (trashStash !== null) {
    clearTimeout(trashStash.timeoutId);
    trashStash = null;
  }

  // Optimistic remove
  sources = sources.filter((s) => s.id !== sourceId);

  try {
    await trashSource(sourceId);

    // Stash for undo — auto-clears after TTL.
    const timeoutId = setTimeout(() => {
      trashStash = null;
    }, TRASH_UNDO_TTL_MS);
    trashStash = { source: snapshot, index: originalIndex, timeoutId };
  } catch (err) {
    // Revert on failure — re-insert at original position.
    if (originalIndex >= sources.length) {
      sources = [...sources, snapshot];
    } else {
      sources = [...sources.slice(0, originalIndex), snapshot, ...sources.slice(originalIndex)];
    }
    console.error('removeSource: failed for source', sourceId, err);
    error = String(err);
  }
}

/**
 * Undo the most recent soft-delete. Calls `restore_source` IPC and re-inserts
 * the source at its original list position. Clears the stash on success.
 * No-op if there is nothing to undo.
 */
export async function undoRemove(): Promise<void> {
  if (trashStash === null) return;
  const { source, index, timeoutId } = trashStash;

  // Clear the timeout so the stash doesn't auto-expire mid-flight.
  clearTimeout(timeoutId);
  trashStash = null;

  // Optimistic re-insert at original index.
  if (index >= sources.length) {
    sources = [...sources, source];
  } else {
    sources = [...sources.slice(0, index), source, ...sources.slice(index)];
  }

  try {
    await restoreSource(source.id);
  } catch (err) {
    // Revert the optimistic re-insert.
    sources = sources.filter((s) => s.id !== source.id);
    console.error('undoRemove: failed for source', source.id, err);
    error = String(err);
  }
}

/**
 * Restore every field to its initial value. Call in `afterEach` of component
 * tests to prevent cross-test bleed from module-level `$state` globals.
 * Analogous to `resetNotebookStore()` in notebooks-state.svelte.ts.
 */
export function resetSourcesStore(): void {
  sources = [];
  loading = false;
  error = null;
  if (trashStash !== null) {
    clearTimeout(trashStash.timeoutId);
    trashStash = null;
  }
}

// ---------------------------------------------------------------------------
// Auto-refresh on active notebook change
// ---------------------------------------------------------------------------
// When `activeNotebookId` changes, reload sources for the new notebook.
// Uses Svelte 5 `$effect.root` so this runs outside a component lifecycle.

$effect.root(() => {
  $effect(() => {
    const id = notebookStore.activeNotebookId;
    if (id) {
      void loadSources(id);
    } else {
      // No active notebook — clear the list.
      sources = [];
    }
  });
});
