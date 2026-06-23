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
  purgeNotebook
} from './ipc.js';
import type { NotebookSummary } from './types.js';

// ---------------------------------------------------------------------------
// Module-level reactive state
// ---------------------------------------------------------------------------

let notebooks = $state<NotebookSummary[]>([]);
let trashedNotebooks = $state<NotebookSummary[]>([]);
let activeNotebookId = $state<string | null>(null); // session-only
let activeTab = $state<'chat' | 'notes'>('chat'); // session-only
let viewMode = $state<'notebook' | 'trash'>('notebook'); // center pane view
let sidebarCollapsed = $state(false); // session-only; localStorage deferred to follow-up
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

const trashCount = $derived(trashedNotebooks.length);

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
  get viewMode() {
    return viewMode;
  },
  set viewMode(mode: 'notebook' | 'trash') {
    viewMode = mode;
  },
  get sidebarCollapsed() {
    return sidebarCollapsed;
  },
  set sidebarCollapsed(v: boolean) {
    sidebarCollapsed = v;
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

/** Fetch all non-trashed notebooks and populate the store. */
export async function loadNotebooks(): Promise<void> {
  error = null;
  loading = true;
  try {
    notebooks = await listNotebooks();
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
    trashedNotebooks = await listTrashed();
  } catch (err) {
    console.error('loadTrashed: failed', err);
    error = String(err);
  } finally {
    loading = false;
  }
}

/** Create a notebook, refresh the list, and auto-select the new notebook. */
export async function createNotebookAction(
  title: string,
  description?: string | null,
  focusMode?: string | null
): Promise<void> {
  error = null;
  loading = true;
  try {
    const created = await createNotebook(title, description, focusMode);
    await loadNotebooks();
    activeNotebookId = created.id;
    viewMode = 'notebook';
  } catch (err) {
    console.error('createNotebookAction: failed', err);
    error = String(err);
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
    await loadNotebooks();
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
    await loadNotebooks();
    await loadTrashed();
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
    await loadNotebooks();
    await loadTrashed();
  } catch (err) {
    console.error('restoreNotebookAction: failed', err);
    error = String(err);
  } finally {
    loading = false;
  }
}

/** Permanently delete a trashed notebook. Refreshes the trashed list. */
export async function purgeNotebookAction(id: string): Promise<void> {
  error = null;
  loading = true;
  try {
    await purgeNotebook(id);
    await loadTrashed();
  } catch (err) {
    console.error('purgeNotebookAction: failed', err);
    error = String(err);
  } finally {
    loading = false;
  }
}

/**
 * Select a notebook by id, switch viewMode to 'notebook', and close the command
 * palette. Setting `paletteOpen = false` is idempotent, so calling this when the
 * palette is already closed is harmless.
 */
export function selectNotebook(id: string): void {
  activeNotebookId = id;
  viewMode = 'notebook';
  paletteOpen = false;
}

/**
 * Switch to the Trash view in the center pane and load the trashed list.
 */
export async function openTrash(): Promise<void> {
  viewMode = 'trash';
  await loadTrashed();
}

/**
 * Restore every field to its initial value. Call in `afterEach` of component
 * tests to prevent cross-test bleed from module-level `$state` globals.
 * Analogous to `resetDraft()` in `onboarding-state.svelte.ts`.
 */
export function resetNotebookStore(): void {
  notebooks = [];
  trashedNotebooks = [];
  activeNotebookId = null;
  activeTab = 'chat';
  viewMode = 'notebook';
  sidebarCollapsed = false;
  paletteOpen = false;
  paletteQuery = '';
  loading = false;
  error = null;
}
