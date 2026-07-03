// Sources reactive store (Svelte 5 runes, module singleton).
// Every CRUD action sets `loading = true` before IPC and `loading = false` in `finally`; errors are transient.

import {
  listSources,
  ingestSource,
  retryIngestSource,
  retryAllFailedSources,
  setSourceSelected,
  trashSource,
  restoreSource
} from './ipc.js';
import type { Source, SourceStatus, StreamEvent, IngestProgress, ErrorMeta } from './types.js';
import { notebookStore, refreshTrashedSources } from '$lib/notebooks/notebooks-state.svelte.js';

// ---------------------------------------------------------------------------
// Module-level reactive state
// ---------------------------------------------------------------------------

let sources = $state<Source[]>([]);
// TODO(M9): single `loading` boolean flickers under concurrent/compound actions — replace with a counter when wiring loading UI.
let loading = $state(false);
// TODO(M9): `error` is written but not yet surfaced in UI.
let error = $state<string | null>(null);

// ---------------------------------------------------------------------------
// Recently-trashed stash — QUEUE of soft-deleted sources.
// LIFO; each entry holds the source, its prior sibling id (for optimistic re-insert), and a timeout.
// On undo, optimistic re-insert then `loadSources` reconciles canonical backend order.
// ---------------------------------------------------------------------------

interface TrashEntry {
  source: Source;
  /** id of the source that was immediately before this one when it was trashed; null if it was at index 0. */
  prevSiblingId: string | null;
  timeoutId: ReturnType<typeof setTimeout>;
}

// LIFO queue — newest entry at the tail. undoRemove() pops from the tail.
let trashQueue = $state<TrashEntry[]>([]);

// ---------------------------------------------------------------------------
// Exported store object
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
  /** Whether any soft-deleted source is currently pending undo (drives the Undo bar). */
  get recentlyTrashed(): boolean {
    return trashQueue.length > 0;
  }
};

/**
 * Cancel all pending trash timers and drain the queue.
 * Call from `onDestroy` so timers don't fire after the component unmounts on a different notebook session.
 */
export function disposeTrashTimers(): void {
  for (const entry of trashQueue) {
    clearTimeout(entry.timeoutId);
  }
  trashQueue = [];
}

/**
 * Remove the undo-bar queue entry for a source. Clears its timeout.
 * Call before modal restore to avoid a stale entry calling `restore_source` on a live source.
 */
export function drainTrashQueueEntry(sourceId: string): void {
  const entry = trashQueue.find((e) => e.source.id === sourceId);
  if (entry) {
    clearTimeout(entry.timeoutId);
    trashQueue = trashQueue.filter((e) => e.source.id !== sourceId);
  }
}

// ---------------------------------------------------------------------------
// Actions (exported top-level functions)
// ---------------------------------------------------------------------------

/** Normalize `error_meta` from the backend: serde may return a JSON string or a nested object. */
function isErrorMetaShape(v: unknown): v is ErrorMeta {
  return (
    typeof v === 'object' &&
    v !== null &&
    'kind' in v &&
    'message' in v &&
    typeof (v as ErrorMeta).message === 'string'
  );
}

function parseErrorMeta(raw: unknown): ErrorMeta | null {
  if (raw === null || raw === undefined) return null;
  const obj =
    typeof raw === 'string'
      ? (() => {
          try {
            return JSON.parse(raw) as unknown;
          } catch {
            return null;
          }
        })()
      : raw;
  return isErrorMetaShape(obj) ? obj : null;
}

/** Fetch all sources for the given notebook and populate the store. */
export async function loadSources(notebookId: string): Promise<void> {
  error = null;
  loading = true;
  try {
    const raw = await listSources(notebookId);
    sources = raw.map((s) => ({ ...s, error_meta: parseErrorMeta(s.error_meta) }));
  } catch (err) {
    console.error('loadSources: failed', err);
    error = String(err);
  } finally {
    loading = false;
  }
}

/**
 * Optimistically insert a source without a backend round-trip.
 * Prepends so the row appears at the top before ingest progress arrives (matches DESC order).
 */
export function addSourceLocal(source: Source): void {
  if (sources.some((s) => s.id === source.id)) return;
  sources = [source, ...sources];
}

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

function updateSourceStatus(
  sourceId: string,
  status: SourceStatus,
  errorMeta?: ErrorMeta | null
): void {
  // Mutate by index to avoid replacing the whole array on every progress event.
  const i = sources.findIndex((s) => s.id === sourceId);
  if (i < 0) return;
  const update: Source = { ...sources[i], status };
  if (errorMeta !== undefined) update.error_meta = errorMeta;
  sources[i] = update;
}

/** Shared event handler for ingest + retry streams. */
function makeIngestHandler(sourceId: string): (e: StreamEvent<IngestProgress>) => void {
  return function handleEvent(e: StreamEvent<IngestProgress>): void {
    if (e.type === 'chunk' && e.data) {
      updateSourceStatus(sourceId, phaseToStatus(e.data.phase));
    } else if (e.type === 'progress') {
      // reserved for future progress bar
    } else if (e.type === 'done') {
      updateSourceStatus(sourceId, 'indexed', null);
    } else if (e.type === 'failed') {
      // Derive attempt_count by incrementing the prior in-memory value; loadSources reconciles later.
      const prior = sources.find((s) => s.id === sourceId)?.error_meta?.attempt_count ?? 0;
      const partialMeta: ErrorMeta = {
        kind: e.data.kind,
        message: e.data.message,
        timestamp: new Date().toISOString(),
        attempt_count: prior + 1
      };
      updateSourceStatus(sourceId, 'error', partialMeta);
    } else if (e.type === 'started') {
      updateSourceStatus(sourceId, 'parsing');
    }
  };
}

export async function ingest(sourceId: string): Promise<void> {
  try {
    await ingestSource(sourceId, makeIngestHandler(sourceId));
  } catch (err) {
    console.error('ingest: failed for source', sourceId, err);
    updateSourceStatus(sourceId, 'error');
    error = String(err);
  }
}

/** Retry a single errored source. Backend rejects non-error or trashed sources. */
export async function retrySource(sourceId: string): Promise<void> {
  // Leave error_meta intact so the failed handler can increment attempt_count from the prior value.
  updateSourceStatus(sourceId, 'parsing');
  try {
    await retryIngestSource(sourceId, makeIngestHandler(sourceId));
  } catch (err) {
    console.error('retrySource: failed for source', sourceId, err);
    updateSourceStatus(sourceId, 'error');
    error = String(err);
  }
}

/** Retry every non-trashed errored source. Continue-on-failure; reconciles via loadSources when done. */
export async function retryAllFailed(notebookId: string): Promise<void> {
  const errorIds = sources.filter((s) => s.status === 'error' && !s.trashed_at).map((s) => s.id);
  for (const id of errorIds) {
    updateSourceStatus(id, 'parsing');
  }
  try {
    // Bulk stream events don't carry sourceId — reload after completion to reconcile final states.
    await retryAllFailedSources(notebookId, (_e) => {});
  } catch (err) {
    console.error('retryAllFailed: failed for notebook', notebookId, err);
    error = String(err);
  } finally {
    await loadSources(notebookId);
  }
}

/** Toggle selected state; optimistic update with revert on failure. */
export async function toggleSelected(sourceId: string): Promise<void> {
  const current = sources.find((s) => s.id === sourceId);
  if (!current) return;
  const newSelected = current.selected === 1 ? 0 : 1;
  sources = sources.map((s) => (s.id === sourceId ? { ...s, selected: newSelected } : s));
  try {
    await setSourceSelected(sourceId, newSelected !== 0);
  } catch (err) {
    sources = sources.map((s) => (s.id === sourceId ? { ...s, selected: current.selected } : s));
    console.error('toggleSelected: failed for source', sourceId, err);
    error = String(err);
  }
}

/** How long (ms) the Undo bar is shown before a stash entry is cleared. */
const TRASH_UNDO_TTL_MS = 6_000;

/**
 * Soft-delete a source. Optimistic removal with revert on failure.
 * Pushes onto trashQueue for TRASH_UNDO_TTL_MS; each delete gets an independent timeout.
 */
export async function removeSource(sourceId: string): Promise<void> {
  const snapshot = sources.find((s) => s.id === sourceId);
  if (!snapshot) return;

  const idx = sources.indexOf(snapshot);
  const prevSiblingId = idx > 0 ? sources[idx - 1].id : null;
  sources = sources.filter((s) => s.id !== sourceId);

  try {
    await trashSource(sourceId);
    const timeoutId = setTimeout(() => {
      trashQueue = trashQueue.filter((e) => e.source.id !== sourceId);
    }, TRASH_UNDO_TTL_MS);
    trashQueue = [...trashQueue, { source: snapshot, prevSiblingId, timeoutId }];
    void refreshTrashedSources().catch(() => {});
  } catch (err) {
    if (idx >= sources.length) {
      sources = [...sources, snapshot];
    } else {
      sources = [...sources.slice(0, idx), snapshot, ...sources.slice(idx)];
    }
    console.error('removeSource: failed for source', sourceId, err);
    error = String(err);
  }
}

/**
 * Undo the most recent soft-delete (LIFO). Optimistic re-insert for responsiveness;
 * `loadSources` reconciles canonical order after the IPC call.
 */
export async function undoRemove(notebookId?: string): Promise<void> {
  if (trashQueue.length === 0) return;

  const entry = trashQueue[trashQueue.length - 1];
  clearTimeout(entry.timeoutId);
  trashQueue = trashQueue.slice(0, -1);

  const { source, prevSiblingId } = entry;

  // Identity anchor: insert after prevSiblingId if present, else at 0, else append.
  let insertAt: number;
  if (prevSiblingId === null) {
    insertAt = 0;
  } else {
    const siblingIdx = sources.findIndex((s) => s.id === prevSiblingId);
    insertAt = siblingIdx >= 0 ? siblingIdx + 1 : sources.length;
  }
  sources = [...sources.slice(0, insertAt), source, ...sources.slice(insertAt)];

  try {
    await restoreSource(source.id);
    if (notebookId) {
      await loadSources(notebookId);
    }
    void refreshTrashedSources().catch(() => {});
  } catch (err) {
    sources = sources.filter((s) => s.id !== source.id);
    console.error('undoRemove: failed for source', source.id, err);
    error = String(err);
  }
}

/** Reset all state. Call in `afterEach` to prevent cross-test bleed from module-level `$state` globals. */
export function resetSourcesStore(): void {
  sources = [];
  loading = false;
  error = null;
  for (const entry of trashQueue) {
    clearTimeout(entry.timeoutId);
  }
  trashQueue = [];
}

// ---------------------------------------------------------------------------
// Auto-refresh on active notebook change
// ---------------------------------------------------------------------------

/** Dispose fn from `$effect.root`. Exposed so tests can stop the auto-refresh effect. */
export const disposeAutoRefresh: () => void = $effect.root(() => {
  $effect(() => {
    const id = notebookStore.activeNotebookId;
    if (id) {
      void loadSources(id);
    } else {
      sources = [];
    }
  });
});
