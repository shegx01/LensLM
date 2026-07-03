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
// TODO(M9): `error` is written but not yet surfaced in UI (polished error states are M9).
let error = $state<string | null>(null); // transient; polished surfacing deferred to M9

// ---------------------------------------------------------------------------
// Recently-trashed stash — QUEUE of soft-deleted sources.
//
// Each entry records the source, the id of the source that preceded it at
// trash time (prevSiblingId), and its own auto-clear timeout. This allows
// multiple in-window deletes to be each independently undoable (LIFO).
//
// Re-insertion strategy: on undo, reconcile via loadSources(activeNotebookId)
// so the list always reflects the backend's canonical ordering. The optimistic
// re-insert before the await keeps the UI responsive, but a successful
// restoreSource is always followed by a canonical reload.
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
  /** Whether any soft-deleted source is currently pending undo (drives the Undo bar). */
  get recentlyTrashed(): boolean {
    return trashQueue.length > 0;
  }
};

/**
 * Cancel all pending trash auto-clear timers and drain the queue.
 *
 * Call from a component's `onDestroy` (or equivalent) so orphan timers cannot
 * fire after the component unmounts and mutate state that belongs to a
 * different notebook session. Without this, a 6 s timer armed just before
 * navigation fires after the rail unmounts and silently clears a trashQueue
 * entry for a different notebook's session.
 */
export function disposeTrashTimers(): void {
  for (const entry of trashQueue) {
    clearTimeout(entry.timeoutId);
  }
  trashQueue = [];
}

/**
 * Drain (remove) the undo-bar queue entry for a specific source id if present.
 * Clears its timeout so it cannot fire after the source is already restored
 * via the Trash modal. Call before restoring via the modal to avoid a stale
 * undo entry calling `restore_source` on an already-live source.
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

/**
 * Parse a raw `error_meta` value from the backend.
 * The backend persists `error_meta` as a JSON TEXT column string; serde may
 * deserialise it as a string (if the column is TEXT) or as a nested object
 * depending on the sqlx mapping. We normalise both shapes here.
 */
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
  // Validate the shape so a malformed row falls back to the null-meta UI path
  // ("Ingest failed (no details captured)") rather than rendering `undefined`.
  return isErrorMetaShape(obj) ? obj : null;
}

/** Fetch all sources for the given notebook and populate the store. */
export async function loadSources(notebookId: string): Promise<void> {
  error = null;
  loading = true;
  try {
    const raw = await listSources(notebookId);
    // Parse the error_meta JSON string at the IPC boundary.
    sources = raw.map((s) => ({ ...s, error_meta: parseErrorMeta(s.error_meta) }));
  } catch (err) {
    console.error('loadSources: failed', err);
    error = String(err);
  } finally {
    loading = false;
  }
}

/**
 * Optimistically insert a source into the store WITHOUT a round-trip to the
 * backend. Called immediately after addFileSource/addTextSource returns so
 * the row exists in the store before ingest progress events arrive.
 *
 * If a source with the same id already exists (e.g. a concurrent loadSources
 * already reconciled), this is a no-op.
 */
export function addSourceLocal(source: Source): void {
  if (sources.some((s) => s.id === source.id)) return;
  // Prepend so the optimistic row appears at the top, matching backend newest-first ordering
  // (ORDER BY created_at DESC). Without this the row appears at the bottom and then jumps
  // to the top when loadSources reconciles — a visible flicker on the row the user just created.
  sources = [source, ...sources];
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

function updateSourceStatus(
  sourceId: string,
  status: SourceStatus,
  errorMeta?: ErrorMeta | null
): void {
  // Mutate in place by index — avoids replacing the whole array on every
  // progress event (prevents full-list re-renders / derived recomputes per tick).
  const i = sources.findIndex((s) => s.id === sourceId);
  if (i < 0) return;
  const update: Source = { ...sources[i], status };
  // Merge error_meta when explicitly provided (undefined = leave as-is).
  if (errorMeta !== undefined) update.error_meta = errorMeta;
  sources[i] = update;
}

/** Shared event handler for ingest + retry streams. */
function makeIngestHandler(sourceId: string): (e: StreamEvent<IngestProgress>) => void {
  return function handleEvent(e: StreamEvent<IngestProgress>): void {
    if (e.type === 'chunk' && e.data) {
      // 'chunk' carries IngestProgress { phase, done, total } — update row status.
      updateSourceStatus(sourceId, phaseToStatus(e.data.phase));
    } else if (e.type === 'progress') {
      // 'progress' carries only { done, total } — no phase; reserved for future progress bar.
    } else if (e.type === 'done') {
      // On success, clear any stale error_meta so no stale reason lingers.
      updateSourceStatus(sourceId, 'indexed', null);
    } else if (e.type === 'failed') {
      // Capture the {kind, message} payload and merge it as error_meta.
      // The stream event doesn't carry attempt_count, so derive it by
      // incrementing the prior in-memory value (null ⇒ 0). This mirrors the
      // backend's read-prior-then-+1 and stays accurate across repeated retries
      // without a DB round-trip; a later loadSources reconciles the timestamp.
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

/**
 * Retry a single errored source. Resets status to `parsing` optimistically,
 * streams progress, and updates the row in place (same id/order/selected).
 * The backend rejects non-error and trashed sources.
 */
export async function retrySource(sourceId: string): Promise<void> {
  // Optimistic transition: error → parsing so the dot starts animating immediately.
  // Note: error_meta is intentionally left intact so the failed handler can
  // increment attempt_count from the prior value (matching the backend).
  updateSourceStatus(sourceId, 'parsing');
  try {
    await retryIngestSource(sourceId, makeIngestHandler(sourceId));
  } catch (err) {
    console.error('retrySource: failed for source', sourceId, err);
    updateSourceStatus(sourceId, 'error');
    error = String(err);
  }
}

/**
 * Retry every non-trashed errored source in the given notebook.
 * Sequentially streams each source; continue-on-failure — one failure does not
 * abort the rest. Progress for each individual source flows through the shared
 * handler, updating dots in real time.
 */
export async function retryAllFailed(notebookId: string): Promise<void> {
  // Optimistically flip all error sources to parsing so their dots animate.
  const errorIds = sources.filter((s) => s.status === 'error' && !s.trashed_at).map((s) => s.id);
  for (const id of errorIds) {
    updateSourceStatus(id, 'parsing');
  }
  try {
    // The backend runs each source sequentially with continue-on-failure.
    // The shared handler updates the correct row via sourceId captured in the closure.
    // Since we cannot inject a per-source handler into the bulk command, we reload
    // after the bulk completes to reconcile final states from the DB.
    await retryAllFailedSources(notebookId, (_e) => {
      // Bulk stream events carry progress but not sourceId — reload on done/failed.
    });
  } catch (err) {
    console.error('retryAllFailed: failed for notebook', notebookId, err);
    error = String(err);
  } finally {
    // Reconcile with the backend to get final statuses + updated error_meta.
    await loadSources(notebookId);
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

/** How long (ms) the Undo bar is shown before a stash entry is cleared. */
const TRASH_UNDO_TTL_MS = 6_000;

/**
 * Soft-delete a source (move to trash). Optimistic: the row is removed from
 * local state immediately, then the IPC `trash_source` command is awaited.
 * On failure the row is restored and `error` is set.
 *
 * A "recently trashed" entry is pushed onto trashQueue for TRASH_UNDO_TTL_MS
 * so the caller (SourcesRail) can render an Undo bar. Multiple in-window
 * deletes each get their own queue entry and independent timeout — no entry
 * is discarded by a subsequent delete.
 */
export async function removeSource(sourceId: string): Promise<void> {
  const snapshot = sources.find((s) => s.id === sourceId);
  if (!snapshot) return;

  // Capture the id of the preceding sibling for identity-anchored re-insertion.
  const idx = sources.indexOf(snapshot);
  const prevSiblingId = idx > 0 ? sources[idx - 1].id : null;

  // Optimistic remove
  sources = sources.filter((s) => s.id !== sourceId);

  try {
    await trashSource(sourceId);

    // Push a new queue entry — auto-removes itself after TTL.
    const timeoutId = setTimeout(() => {
      trashQueue = trashQueue.filter((e) => e.source.id !== sourceId);
    }, TRASH_UNDO_TTL_MS);
    trashQueue = [...trashQueue, { source: snapshot, prevSiblingId, timeoutId }];
    // Refresh sidebar trash badge without blocking or hijacking the trash flow.
    void refreshTrashedSources().catch(() => {});
  } catch (err) {
    // Revert on failure — re-insert at original position.
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
 * Undo the most recent soft-delete (LIFO). Calls `restore_source` IPC, then
 * reconciles the list via `loadSources` so ordering matches the backend's
 * canonical newest-first order.
 *
 * Identity re-anchor: on successful restore, `loadSources` provides the
 * canonical list — no stale-index fragility. The optimistic re-insert before
 * the await is only for perceived responsiveness; the canonical reload
 * overwrites it.
 *
 * No-op if there is nothing to undo.
 */
export async function undoRemove(notebookId?: string): Promise<void> {
  if (trashQueue.length === 0) return;

  // Pop the most-recent entry (LIFO — tail of the queue).
  const entry = trashQueue[trashQueue.length - 1];
  clearTimeout(entry.timeoutId);
  trashQueue = trashQueue.slice(0, -1);

  const { source, prevSiblingId } = entry;

  // Optimistic re-insert using identity anchor:
  // - If prevSiblingId is null (was at index 0), insert at position 0.
  // - If prevSiblingId exists in current list, insert immediately after it.
  // - If prevSiblingId is set but no longer in the list (was also deleted), append.
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
    // Reconcile with backend canonical order if the caller provides the notebookId.
    if (notebookId) {
      await loadSources(notebookId);
    }
    // Refresh sidebar trash badge without blocking or hijacking the restore flow.
    void refreshTrashedSources().catch(() => {});
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
  for (const entry of trashQueue) {
    clearTimeout(entry.timeoutId);
  }
  trashQueue = [];
}

// ---------------------------------------------------------------------------
// Auto-refresh on active notebook change
// ---------------------------------------------------------------------------
// When `activeNotebookId` changes, reload sources for the new notebook.
// Uses Svelte 5 `$effect.root` so this runs outside a component lifecycle.

/**
 * Dispose function returned by `$effect.root`. Exposed so tests (and future
 * teardown paths) can stop the auto-refresh effect cleanly.
 */
export const disposeAutoRefresh: () => void = $effect.root(() => {
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
