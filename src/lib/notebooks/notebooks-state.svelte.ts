// Notebooks reactive store (Svelte 5 runes, module singleton).
//
// Module-level `$state` singleton — same pattern as `onboarding-state.svelte.ts`.
// All sidebar, center top-bar, trash-view, and create-dialog consumers read from
// a single source of truth without prop drilling.
//
// SESSION-ONLY state: sidebarCollapsed, activeNotebookId, activeTab are not
// persisted to config or localStorage in M3 (deferred to a follow-up).
//
// LOADING LIFECYCLE: every CRUD action wraps its IPC call with `loading = true`
// before the call and `loading = false` in a `finally` block.
//
// ERROR HANDLING: try/catch on every action; console.error on failure; transient
// `error` field for future surfacing. Polished error UI is M9 scope.

import {
  listNotebooks,
  createNotebook,
  renameNotebook,
  trashNotebook,
  restoreNotebook,
  listTrashed,
  purgeNotebook,
  touchNotebookActivity
} from './ipc.js';
import { listTrashedSources, restoreSource, purgeSource } from '$lib/sources/ipc.js';
import type { TrashedSource } from '$lib/sources/types.js';
// NOTE: loadSources + drainTrashQueueEntry are imported lazily inside actions
// to break the circular dependency:
//   notebooks-state → sources-state → notebooks-state (for activeNotebookId)
// Static import would initialise sources-state before notebookStore is ready,
// causing the $effect.root auto-refresh to read an undefined notebookStore.
import type { Notebook, NotebookSummary } from './types.js';
import { NOTEBOOK_PALETTE, notebookAccentClass } from './notebook-color.js';

// ---------------------------------------------------------------------------
// Module-level reactive state
// ---------------------------------------------------------------------------

let notebooks = $state<NotebookSummary[]>([]);
let trashedNotebooks = $state<NotebookSummary[]>([]);
let trashedSources = $state<TrashedSource[]>([]);
let activeNotebookId = $state<string | null>(null); // session-only
let activeTab = $state<'chat' | 'notes'>('chat'); // session-only
let trashOpen = $state(false); // Trash modal visibility (centered dialog)
let inspectorOpen = $state(false); // dev/QA Embeddings Inspector overlay visibility
let settingsOpen = $state(false); // global Preferences shell (Settings>Embeddings) visibility
let notebookSettingsOpen = $state(false); // per-notebook "{notebook} settings" sheet visibility
let sidebarCollapsed = $state(false); // session-only; localStorage deferred to follow-up
let rightRailCollapsed = $state(false); // session-only; persisted same way as sidebarCollapsed
let paletteOpen = $state(false); // command palette visibility
let paletteQuery = $state(''); // search query (palette-scoped, reset on close)
// TODO(M9): single `loading` boolean flickers under concurrent/compound actions — replace with a counter when wiring loading UI.
let loading = $state(false);
// TODO(M9): `error` is written but not yet surfaced in UI (polished error states are M9).
let error = $state<string | null>(null); // transient; polished surfacing deferred to M9

// ---------------------------------------------------------------------------
// Derived state
// ---------------------------------------------------------------------------

const paletteResults = $derived(
  paletteQuery
    ? notebooks.filter((n) => n.title.toLowerCase().includes(paletteQuery.toLowerCase()))
    : notebooks
);

const activeNotebook = $derived(notebooks.find((n) => n.id === activeNotebookId) ?? null);

const trashCount = $derived(trashedNotebooks.length + trashedSources.length);

// Rank-based decorative color assignment.
//
// Notebooks are sorted by `id` ASCENDING (UUIDv7 ids are creation-ordered, so
// this is creation order) and assigned `NOTEBOOK_PALETTE[rank % length]`. This
// guarantees the first 10 notebooks get 10 DISTINCT hues — no birthday-paradox
// collisions like the old hash-into-6 approach. The map is stable when
// notebooks are appended; it only reshuffles ranks after a deletion, which is
// acceptable for a decorative cue.
const notebookColorMap = $derived.by(() => {
  const map = new Map<string, string>();
  const sorted = [...notebooks].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  sorted.forEach((n, i) => {
    map.set(n.id, `nb-${NOTEBOOK_PALETTE[i % NOTEBOOK_PALETTE.length]}`);
  });
  return map;
});

// ---------------------------------------------------------------------------
// Exported store object (getter/setter pairs — project pattern)
// ---------------------------------------------------------------------------

export const notebookStore = {
  get notebooks() {
    return notebooks;
  },
  get trashedNotebooks() {
    return trashedNotebooks;
  },
  get trashedSources() {
    return trashedSources;
  },
  get trashCount() {
    return trashCount;
  },
  get activeNotebook() {
    return activeNotebook;
  },
  get activeNotebookId() {
    return activeNotebookId;
  },
  set activeNotebookId(id: string | null) {
    activeNotebookId = id;
  },
  get activeTab() {
    return activeTab;
  },
  set activeTab(tab: 'chat' | 'notes') {
    activeTab = tab;
  },
  get trashOpen() {
    return trashOpen;
  },
  set trashOpen(v: boolean) {
    trashOpen = v;
  },
  get inspectorOpen() {
    return inspectorOpen;
  },
  set inspectorOpen(v: boolean) {
    inspectorOpen = v;
  },
  get settingsOpen() {
    return settingsOpen;
  },
  set settingsOpen(v: boolean) {
    settingsOpen = v;
  },
  get notebookSettingsOpen() {
    return notebookSettingsOpen;
  },
  set notebookSettingsOpen(v: boolean) {
    notebookSettingsOpen = v;
  },
  get sidebarCollapsed() {
    return sidebarCollapsed;
  },
  set sidebarCollapsed(v: boolean) {
    sidebarCollapsed = v;
  },
  get rightRailCollapsed() {
    return rightRailCollapsed;
  },
  set rightRailCollapsed(v: boolean) {
    rightRailCollapsed = v;
  },
  get paletteOpen() {
    return paletteOpen;
  },
  set paletteOpen(v: boolean) {
    paletteOpen = v;
    if (!v) paletteQuery = ''; // auto-reset query on close
  },
  get paletteQuery() {
    return paletteQuery;
  },
  set paletteQuery(q: string) {
    paletteQuery = q;
  },
  get paletteResults() {
    return paletteResults;
  },
  get loading() {
    return loading;
  },
  get error() {
    return error;
  },
  set error(e: string | null) {
    error = e;
  }
};

// ---------------------------------------------------------------------------
// CRUD actions (exported top-level functions)
// ---------------------------------------------------------------------------

// Internal refresh helpers. These ONLY fetch + assign; they do NOT manage the
// loading/error lifecycle so that compound actions can own a single
// loading/error scope and run both refreshes concurrently without one wiping
// the other's error or flickering `loading`.
async function refreshNotebooks(): Promise<void> {
  notebooks = await listNotebooks();
}

export async function refreshTrashed(): Promise<void> {
  trashedNotebooks = await listTrashed();
}

// Coalescing serial refresh for trashed sources. Concurrent callers coalesce
// onto the in-flight promise; a queued flag ensures a final fetch always runs
// AFTER the last trigger, so stale earlier responses can never overwrite newer
// ones (fixes multi-delete race / badge undercount).
let _trashSourcesRefreshInFlight: Promise<void> | null = null;
let _trashSourcesRefreshQueued = false;

export async function refreshTrashedSources(): Promise<void> {
  if (_trashSourcesRefreshInFlight) {
    _trashSourcesRefreshQueued = true;
    return _trashSourcesRefreshInFlight;
  }
  _trashSourcesRefreshInFlight = (async () => {
    try {
      do {
        _trashSourcesRefreshQueued = false;
        trashedSources = await listTrashedSources();
      } while (_trashSourcesRefreshQueued);
    } finally {
      _trashSourcesRefreshInFlight = null;
    }
  })();
  return _trashSourcesRefreshInFlight;
}

/** Fetch all non-trashed notebooks and populate the store. */
export async function loadNotebooks(): Promise<void> {
  error = null;
  loading = true;
  try {
    await refreshNotebooks();
  } catch (err) {
    console.error('loadNotebooks: failed', err);
    error = String(err);
  } finally {
    loading = false;
  }
}

/** Fetch all trashed notebooks and populate the trashed list. */
export async function loadTrashed(): Promise<void> {
  error = null;
  loading = true;
  try {
    await refreshTrashed();
  } catch (err) {
    console.error('loadTrashed: failed', err);
    error = String(err);
  } finally {
    loading = false;
  }
}

/**
 * Create a notebook, refresh the list, and auto-select the new notebook.
 * Returns the created `Notebook` on success, or `null` on failure (with the
 * store `error` field set) so callers can distinguish success from failure.
 */
export async function createNotebookAction(
  title: string,
  description?: string | null,
  focusMode?: string | null
): Promise<Notebook | null> {
  error = null;
  loading = true;
  try {
    const created = await createNotebook(title, description, focusMode);
    await refreshNotebooks();
    activeNotebookId = created.id;
    return created;
  } catch (err) {
    console.error('createNotebookAction: failed', err);
    error = String(err);
    return null;
  } finally {
    loading = false;
  }
}

/** Rename a notebook and refresh the list. */
export async function renameNotebookAction(id: string, title: string): Promise<void> {
  error = null;
  loading = true;
  try {
    await renameNotebook(id, title);
    await refreshNotebooks();
  } catch (err) {
    console.error('renameNotebookAction: failed', err);
    error = String(err);
  } finally {
    loading = false;
  }
}

/**
 * Soft-delete a notebook (move to trash). Refreshes both lists and clears
 * `activeNotebookId` if the trashed notebook was the active one.
 */
export async function trashNotebookAction(id: string): Promise<void> {
  error = null;
  loading = true;
  try {
    await trashNotebook(id);
    if (activeNotebookId === id) {
      activeNotebookId = null;
    }
    await Promise.all([refreshNotebooks(), refreshTrashed(), refreshTrashedSources()]);
  } catch (err) {
    console.error('trashNotebookAction: failed', err);
    error = String(err);
  } finally {
    loading = false;
  }
}

/** Restore a trashed notebook. Refreshes both lists. */
export async function restoreNotebookAction(id: string): Promise<void> {
  error = null;
  loading = true;
  try {
    await restoreNotebook(id);
    await Promise.all([refreshNotebooks(), refreshTrashed(), refreshTrashedSources()]);
  } catch (err) {
    console.error('restoreNotebookAction: failed', err);
    error = String(err);
  } finally {
    loading = false;
  }
}

/** Permanently delete a trashed notebook. Refreshes both lists. */
export async function purgeNotebookAction(id: string): Promise<void> {
  error = null;
  loading = true;
  try {
    await purgeNotebook(id);
    await Promise.all([refreshNotebooks(), refreshTrashed(), refreshTrashedSources()]);
  } catch (err) {
    console.error('purgeNotebookAction: failed', err);
    error = String(err);
  } finally {
    loading = false;
  }
}

/**
 * Select a notebook by id and close the command palette. The center pane is
 * driven by `activeNotebook`, so no view-mode switch is needed. Closing the
 * palette mirrors the `notebookStore.paletteOpen` setter semantics (clears the
 * query too) so a stale palette query can't linger. Idempotent when the palette
 * is already closed.
 *
 * Fire-and-forget activity touch: records the open for MRU ordering. A DB write
 * failure is intentionally swallowed — it must not block selection.
 */
export function selectNotebook(id: string): void {
  activeNotebookId = id;
  paletteOpen = false;
  paletteQuery = '';
  void touchNotebookActivity(id).catch(() => {});
}

/**
 * Return the decorative `nb-{hue}` class for a notebook id.
 *
 * For LIVE notebooks this is the rank-based, collision-free assignment from
 * {@link notebookColorMap}. For ids not in the live set (e.g. trashed notebooks
 * rendered in TrashView), it falls back to the pure hash {@link notebookAccentClass}
 * so the call never crashes and still yields a stable, distinct-ish hue.
 */
export function notebookColorClass(id: string): string {
  return notebookColorMap.get(id) ?? notebookAccentClass(id);
}

/** Fetch all individually-trashed sources and populate the store. */
export async function loadTrashedSources(): Promise<void> {
  error = null;
  loading = true;
  try {
    await refreshTrashedSources();
  } catch (err) {
    console.error('loadTrashedSources: failed', err);
    error = String(err);
  } finally {
    loading = false;
  }
}

/**
 * Restore a trashed source from the Trash modal. Drains any matching undo-bar
 * entry for that source so the stale undo cannot call `restore_source` on an
 * already-live source. Refreshes the trashed-sources list, and if the source
 * belonged to the active notebook, also refreshes the active source list.
 */
export async function restoreSourceFromTrash(sourceId: string): Promise<void> {
  // Look up notebook_id BEFORE the IPC call (row is removed after).
  const source = trashedSources.find((s) => s.id === sourceId);
  const notebookId = source?.notebook_id ?? null;

  // Drain any pending undo-bar entry for this source.
  // Dynamic import to avoid circular dependency: sources-state ↔ notebooks-state.
  const { drainTrashQueueEntry } = await import('$lib/sources/sources-state.svelte.js');
  drainTrashQueueEntry(sourceId);

  error = null;
  loading = true;
  try {
    await restoreSource(sourceId);
    await refreshTrashedSources();
    // Only reload the active source list if this source belonged to it.
    if (notebookId && notebookId === activeNotebookId) {
      const { loadSources } = await import('$lib/sources/sources-state.svelte.js');
      await loadSources(notebookId);
    }
  } catch (err) {
    console.error('restoreSourceFromTrash: failed', err);
    error = String(err);
  } finally {
    loading = false;
  }
}

/**
 * Permanently delete a trashed source. Refreshes the trashed-sources list, and
 * if the source belonged to the active notebook, also refreshes the active
 * source list (in case it was temporarily visible via optimistic state).
 */
export async function purgeSourceAction(sourceId: string): Promise<void> {
  // Look up notebook_id BEFORE the IPC call (row is removed after).
  const source = trashedSources.find((s) => s.id === sourceId);
  const notebookId = source?.notebook_id ?? null;

  // Drain any pending undo-bar entry so a stale undo cannot try to restore a
  // source we just permanently purged.
  // Dynamic import to avoid circular dependency: sources-state ↔ notebooks-state.
  const { drainTrashQueueEntry } = await import('$lib/sources/sources-state.svelte.js');
  drainTrashQueueEntry(sourceId);

  error = null;
  loading = true;
  try {
    await purgeSource(sourceId);
    await refreshTrashedSources();
    // Only reload the active source list if this source belonged to it.
    if (notebookId && notebookId === activeNotebookId) {
      // Dynamic import to avoid circular dependency: sources-state ↔ notebooks-state.
      const { loadSources } = await import('$lib/sources/sources-state.svelte.js');
      await loadSources(notebookId);
    }
  } catch (err) {
    console.error('purgeSourceAction: failed', err);
    error = String(err);
  } finally {
    loading = false;
  }
}

/**
 * Open the Trash modal and load both trashed notebooks and trashed sources.
 * Uses Promise.allSettled so a failure in one fetch still renders the other.
 */
export async function openTrash(): Promise<void> {
  trashOpen = true;
  error = null;
  loading = true;
  try {
    const results = await Promise.allSettled([refreshTrashed(), refreshTrashedSources()]);
    for (const result of results) {
      if (result.status === 'rejected') {
        console.error('openTrash: a fetch failed', result.reason);
        error = String(result.reason);
      }
    }
  } finally {
    loading = false;
  }
}

/**
 * Restore every field to its initial value. Call in `afterEach` of component
 * tests to prevent cross-test bleed from module-level `$state` globals.
 * Analogous to `resetDraft()` in `onboarding-state.svelte.ts`.
 */
export function resetNotebookStore(): void {
  notebooks = [];
  trashedNotebooks = [];
  trashedSources = [];
  activeNotebookId = null;
  activeTab = 'chat';
  trashOpen = false;
  inspectorOpen = false;
  sidebarCollapsed = false;
  rightRailCollapsed = false;
  paletteOpen = false;
  paletteQuery = '';
  loading = false;
  error = null;
  _trashSourcesRefreshInFlight = null;
  _trashSourcesRefreshQueued = false;
}
