// Notebooks reactive store (Svelte 5 runes, module singleton).
//
// SESSION-ONLY: sidebarCollapsed, activeNotebookId, activeTab are not persisted.
// Every CRUD action guards loading/error via try/finally. Error UI is M9 scope.

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
// Lazy import: breaks the circular dependency notebooks-state ↔ sources-state.
// Static import would initialise sources-state before notebookStore is ready.
import type { Notebook, NotebookSummary } from './types.js';
import { NOTEBOOK_PALETTE, notebookAccentClass } from './notebook-color.js';

// Palette results are a discriminated union: notebook-title results (Chat tab or
// no active notebook) or per-note results (Notes tab). The `note`-kind branch is
// computed in CommandPalette, which can import `notes-state` freely — a static
// import here would risk a circular dependency (notebooks-state ↔ notes-state).
export type NotebookPaletteResult = { kind: 'notebook'; notebook: NotebookSummary };
export type NotePaletteResult = {
  kind: 'note';
  noteId: string;
  title: string;
  snippet: string;
  sourceTitle: string | null;
};
export type PaletteResult = NotebookPaletteResult | NotePaletteResult;

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
let settingsSection = $state<string | null>(null); // deep-link target section for the Preferences shell
let notebookSettingsOpen = $state(false); // per-notebook "{notebook} settings" sheet visibility
let sidebarCollapsed = $state(false); // session-only
let rightRailCollapsed = $state(false); // session-only
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

// Rank-based color: ids sorted ascending (UUIDv7 = creation order), assigned
// NOTEBOOK_PALETTE[rank % length]. First 10 notebooks get 10 DISTINCT hues.
const notebookColorMap = $derived.by(() => {
  const map = new Map<string, string>();
  const sorted = [...notebooks].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  sorted.forEach((n, i) => {
    map.set(n.id, `nb-${NOTEBOOK_PALETTE[i % NOTEBOOK_PALETTE.length]}`);
  });
  return map;
});

// ---------------------------------------------------------------------------
// Exported store object
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
    // The two in-place settings surfaces are mutually exclusive — opening one closes the
    // other, so closing it never re-reveals a stale sibling underneath.
    if (v) notebookSettingsOpen = false;
    // A deep-link target only lives for the duration a surface is open.
    if (!v) settingsSection = null;
  },
  get settingsSection() {
    return settingsSection;
  },
  set settingsSection(id: string | null) {
    settingsSection = id;
  },
  /** Open the global Preferences shell, optionally deep-linked to a section (e.g. `'ai'`). */
  openSettings(section?: string) {
    settingsSection = section ?? null;
    settingsOpen = true;
    notebookSettingsOpen = false;
  },
  get notebookSettingsOpen() {
    return notebookSettingsOpen;
  },
  set notebookSettingsOpen(v: boolean) {
    notebookSettingsOpen = v;
    if (v) settingsOpen = false;
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
// CRUD actions
// ---------------------------------------------------------------------------

// Internal helpers — fetch + assign only; callers own the loading/error scope.
async function refreshNotebooks(): Promise<void> {
  notebooks = await listNotebooks();
}

export async function refreshTrashed(): Promise<void> {
  trashedNotebooks = await listTrashed();
}

// Coalescing serial refresh: concurrent callers share the in-flight promise;
// a queued flag ensures a final fetch runs after the last trigger, preventing
// stale responses overwriting newer ones (fixes multi-delete race).
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

/** Create a notebook, refresh the list, and auto-select it. Returns `null` on failure. */
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

/** Soft-delete a notebook. Clears `activeNotebookId` if it was the active one. */
export async function trashNotebookAction(id: string): Promise<void> {
  error = null;
  loading = true;
  try {
    await trashNotebook(id);
    if (activeNotebookId === id) {
      activeNotebookId = null;
      // The active notebook is gone; drop its settings surface so it can't rebind.
      notebookSettingsOpen = false;
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
 * Select a notebook and close the command palette (clears query too).
 * Fire-and-forget activity touch for MRU ordering — DB failures swallowed.
 */
export function selectNotebook(id: string): void {
  activeNotebookId = id;
  // Never let the per-notebook settings surface stay bound to the notebook we left.
  notebookSettingsOpen = false;
  paletteOpen = false;
  paletteQuery = '';
  void touchNotebookActivity(id).catch(() => {});
}

/** Decorative `nb-{hue}` class: rank-based for live notebooks, hash fallback for trashed. */
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
 * Restore a trashed source. Drains any pending undo-bar entry first so a stale
 * undo can't call `restore_source` on an already-live source.
 */
export async function restoreSourceFromTrash(sourceId: string): Promise<void> {
  // Look up notebook_id BEFORE the IPC call (row is removed after restore).
  const source = trashedSources.find((s) => s.id === sourceId);
  const notebookId = source?.notebook_id ?? null;

  // Dynamic import to avoid circular dependency: sources-state ↔ notebooks-state.
  const { drainTrashQueueEntry } = await import('$lib/sources/sources-state.svelte.js');
  drainTrashQueueEntry(sourceId);

  error = null;
  loading = true;
  try {
    await restoreSource(sourceId);
    await refreshTrashedSources();
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

/** Permanently delete a trashed source. Also drains any pending undo-bar entry. */
export async function purgeSourceAction(sourceId: string): Promise<void> {
  // Look up notebook_id BEFORE the IPC call (row is removed after purge).
  const source = trashedSources.find((s) => s.id === sourceId);
  const notebookId = source?.notebook_id ?? null;

  // Dynamic import to avoid circular dependency: sources-state ↔ notebooks-state.
  const { drainTrashQueueEntry } = await import('$lib/sources/sources-state.svelte.js');
  drainTrashQueueEntry(sourceId);

  error = null;
  loading = true;
  try {
    await purgeSource(sourceId);
    await refreshTrashedSources();
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

/** Open the Trash modal. Uses `Promise.allSettled` so one fetch failure doesn't block the other. */
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

/** Reset all fields to initial values. Call in `afterEach` to prevent cross-test bleed. */
export function resetNotebookStore(): void {
  notebooks = [];
  trashedNotebooks = [];
  trashedSources = [];
  activeNotebookId = null;
  activeTab = 'chat';
  settingsOpen = false;
  settingsSection = null;
  notebookSettingsOpen = false;
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
