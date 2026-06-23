// Store unit tests for notebooks-state.svelte.ts.
//
// The IPC module is mocked so tests run without a Tauri host.
// `resetNotebookStore()` is called in afterEach to prevent cross-test bleed
// from module-level $state globals — same pattern as onboarding tests.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  notebookStore,
  resetNotebookStore,
  loadNotebooks,
  loadTrashed,
  createNotebookAction,
  renameNotebookAction,
  trashNotebookAction,
  restoreNotebookAction,
  purgeNotebookAction,
  selectNotebook,
  openTrash
} from './notebooks-state.svelte.js';

// ---------------------------------------------------------------------------
// Mock the IPC layer
// ---------------------------------------------------------------------------

vi.mock('./ipc.js', () => ({
  listNotebooks: vi.fn(),
  createNotebook: vi.fn(),
  renameNotebook: vi.fn(),
  trashNotebook: vi.fn(),
  restoreNotebook: vi.fn(),
  listTrashed: vi.fn(),
  purgeNotebook: vi.fn()
}));

// Import the mocked functions so we can configure return values per test.
import {
  listNotebooks,
  createNotebook,
  renameNotebook,
  trashNotebook,
  restoreNotebook,
  listTrashed,
  purgeNotebook
} from './ipc.js';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

import type { NotebookSummary, Notebook } from './types.js';

function makeNotebookSummary(overrides?: Partial<NotebookSummary>): NotebookSummary {
  return {
    id: 'nb-001',
    title: 'My Notebook',
    description: null,
    focus_mode: 'research',
    created_at: new Date(Date.now() - 3600_000).toISOString(),
    updated_at: new Date(Date.now() - 3600_000).toISOString(),
    trashed_at: null,
    source_count: 0,
    ...overrides
  };
}

function makeNotebook(overrides?: Partial<Notebook>): Notebook {
  return {
    id: 'nb-001',
    title: 'My Notebook',
    description: null,
    focus_mode: 'research',
    created_at: new Date(Date.now() - 3600_000).toISOString(),
    updated_at: new Date(Date.now() - 3600_000).toISOString(),
    trashed_at: null,
    ...overrides
  };
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
  resetNotebookStore();
});

afterEach(() => {
  resetNotebookStore();
});

// ---------------------------------------------------------------------------
// resetNotebookStore
// ---------------------------------------------------------------------------

describe('resetNotebookStore', () => {
  it('resets all fields to initial values', async () => {
    vi.mocked(listNotebooks).mockResolvedValue([makeNotebookSummary()]);
    await loadNotebooks();
    notebookStore.activeNotebookId = 'nb-001';
    notebookStore.sidebarCollapsed = true;
    notebookStore.paletteOpen = true;
    notebookStore.trashOpen = true;

    resetNotebookStore();

    expect(notebookStore.notebooks).toHaveLength(0);
    expect(notebookStore.activeNotebookId).toBeNull();
    expect(notebookStore.trashOpen).toBe(false);
    expect(notebookStore.sidebarCollapsed).toBe(false);
    expect(notebookStore.paletteOpen).toBe(false);
    expect(notebookStore.paletteQuery).toBe('');
    expect(notebookStore.loading).toBe(false);
    expect(notebookStore.error).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// loadNotebooks
// ---------------------------------------------------------------------------

describe('loadNotebooks', () => {
  it('populates notebooks from listNotebooks()', async () => {
    const data = [makeNotebookSummary({ id: 'nb-001' }), makeNotebookSummary({ id: 'nb-002' })];
    vi.mocked(listNotebooks).mockResolvedValue(data);

    await loadNotebooks();

    expect(notebookStore.notebooks).toEqual(data);
  });

  it('sets loading to false after success', async () => {
    vi.mocked(listNotebooks).mockResolvedValue([]);
    await loadNotebooks();
    expect(notebookStore.loading).toBe(false);
  });

  it('sets loading to false and sets error on failure', async () => {
    vi.mocked(listNotebooks).mockRejectedValue(new Error('DB error'));
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await loadNotebooks();

    expect(notebookStore.loading).toBe(false);
    expect(notebookStore.error).toBeTruthy();
    consoleSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// createNotebookAction
// ---------------------------------------------------------------------------

describe('createNotebookAction', () => {
  it('creates a notebook, refreshes the list, and auto-selects the new notebook', async () => {
    const created = makeNotebook({ id: 'nb-new' });
    vi.mocked(createNotebook).mockResolvedValue(created);
    vi.mocked(listNotebooks).mockResolvedValue([makeNotebookSummary({ id: 'nb-new' })]);

    await createNotebookAction('New Notebook');

    expect(notebookStore.notebooks).toHaveLength(1);
    expect(notebookStore.activeNotebookId).toBe('nb-new');
  });

  it('sets loading to false after success', async () => {
    vi.mocked(createNotebook).mockResolvedValue(makeNotebook());
    vi.mocked(listNotebooks).mockResolvedValue([]);

    await createNotebookAction('Test');

    expect(notebookStore.loading).toBe(false);
  });

  it('sets error and loading=false on failure', async () => {
    vi.mocked(createNotebook).mockRejectedValue(new Error('title too long'));
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await createNotebookAction('x'.repeat(600));

    expect(notebookStore.loading).toBe(false);
    expect(notebookStore.error).toBeTruthy();
    consoleSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// trashNotebookAction
// ---------------------------------------------------------------------------

describe('trashNotebookAction', () => {
  it('removes notebook from active list after trashing', async () => {
    const nb = makeNotebookSummary({ id: 'nb-001' });
    vi.mocked(listNotebooks).mockResolvedValue([nb]);
    await loadNotebooks();

    vi.mocked(trashNotebook).mockResolvedValue(undefined);
    vi.mocked(listNotebooks).mockResolvedValue([]);
    vi.mocked(listTrashed).mockResolvedValue([{ ...nb, trashed_at: new Date().toISOString() }]);

    await trashNotebookAction('nb-001');

    expect(notebookStore.notebooks).toHaveLength(0);
    expect(notebookStore.trashedNotebooks).toHaveLength(1);
  });

  it('clears activeNotebookId when the active notebook is trashed', async () => {
    const nb = makeNotebookSummary({ id: 'nb-001' });
    vi.mocked(listNotebooks).mockResolvedValue([nb]);
    await loadNotebooks();
    notebookStore.activeNotebookId = 'nb-001';

    vi.mocked(trashNotebook).mockResolvedValue(undefined);
    vi.mocked(listNotebooks).mockResolvedValue([]);
    vi.mocked(listTrashed).mockResolvedValue([{ ...nb, trashed_at: new Date().toISOString() }]);

    await trashNotebookAction('nb-001');

    expect(notebookStore.activeNotebookId).toBeNull();
  });

  it('does NOT clear activeNotebookId when a different notebook is trashed', async () => {
    vi.mocked(listNotebooks).mockResolvedValue([
      makeNotebookSummary({ id: 'nb-001' }),
      makeNotebookSummary({ id: 'nb-002' })
    ]);
    await loadNotebooks();
    notebookStore.activeNotebookId = 'nb-001';

    vi.mocked(trashNotebook).mockResolvedValue(undefined);
    vi.mocked(listNotebooks).mockResolvedValue([makeNotebookSummary({ id: 'nb-001' })]);
    vi.mocked(listTrashed).mockResolvedValue([
      makeNotebookSummary({ id: 'nb-002', trashed_at: new Date().toISOString() })
    ]);

    await trashNotebookAction('nb-002');

    expect(notebookStore.activeNotebookId).toBe('nb-001');
  });
});

// ---------------------------------------------------------------------------
// restoreNotebookAction
// ---------------------------------------------------------------------------

describe('restoreNotebookAction', () => {
  it('moves notebook from trashed back to active list', async () => {
    const nb = makeNotebookSummary({ id: 'nb-001', trashed_at: new Date().toISOString() });
    vi.mocked(listTrashed).mockResolvedValue([nb]);
    await loadTrashed();

    vi.mocked(restoreNotebook).mockResolvedValue(undefined);
    vi.mocked(listNotebooks).mockResolvedValue([{ ...nb, trashed_at: null }]);
    vi.mocked(listTrashed).mockResolvedValue([]);

    await restoreNotebookAction('nb-001');

    expect(notebookStore.notebooks).toHaveLength(1);
    expect(notebookStore.trashedNotebooks).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// purgeNotebookAction
// ---------------------------------------------------------------------------

describe('purgeNotebookAction', () => {
  it('removes notebook from trashed list permanently', async () => {
    const nb = makeNotebookSummary({ id: 'nb-001', trashed_at: new Date().toISOString() });
    vi.mocked(listTrashed).mockResolvedValue([nb]);
    await loadTrashed();

    vi.mocked(purgeNotebook).mockResolvedValue(undefined);
    vi.mocked(listTrashed).mockResolvedValue([]);

    await purgeNotebookAction('nb-001');

    expect(notebookStore.trashedNotebooks).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// selectNotebook
// ---------------------------------------------------------------------------

describe('selectNotebook', () => {
  it('sets activeNotebookId', () => {
    selectNotebook('nb-abc');

    expect(notebookStore.activeNotebookId).toBe('nb-abc');
  });
});

// ---------------------------------------------------------------------------
// openTrash
// ---------------------------------------------------------------------------

describe('openTrash', () => {
  it('opens the Trash modal and loads trashed notebooks', async () => {
    const trashed = makeNotebookSummary({ id: 'nb-trashed', trashed_at: new Date().toISOString() });
    vi.mocked(listTrashed).mockResolvedValue([trashed]);

    await openTrash();

    expect(notebookStore.trashOpen).toBe(true);
    expect(notebookStore.trashedNotebooks).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// Command palette — paletteResults
// ---------------------------------------------------------------------------

describe('notebookStore.paletteResults', () => {
  beforeEach(async () => {
    vi.mocked(listNotebooks).mockResolvedValue([
      makeNotebookSummary({ id: 'nb-001', title: 'Alpha Research' }),
      makeNotebookSummary({ id: 'nb-002', title: 'Beta Coding' }),
      makeNotebookSummary({ id: 'nb-003', title: 'alpha Notes' })
    ]);
    await loadNotebooks();
  });

  it('returns all notebooks when paletteQuery is empty', () => {
    notebookStore.paletteQuery = '';
    expect(notebookStore.paletteResults).toHaveLength(3);
  });

  it('filters by case-insensitive title substring', () => {
    notebookStore.paletteQuery = 'alpha';
    const results = notebookStore.paletteResults;
    expect(results).toHaveLength(2);
    expect(results.map((r) => r.id)).toContain('nb-001');
    expect(results.map((r) => r.id)).toContain('nb-003');
  });

  it('returns empty array when no notebooks match the query', () => {
    notebookStore.paletteQuery = 'zzz-no-match';
    expect(notebookStore.paletteResults).toHaveLength(0);
  });

  it('resets paletteQuery to "" when paletteOpen is set to false', () => {
    notebookStore.paletteQuery = 'alpha';
    notebookStore.paletteOpen = true;
    notebookStore.paletteOpen = false;
    expect(notebookStore.paletteQuery).toBe('');
  });
});

// ---------------------------------------------------------------------------
// renameNotebookAction
// ---------------------------------------------------------------------------

describe('renameNotebookAction', () => {
  it('calls renameNotebook IPC and refreshes the list', async () => {
    vi.mocked(renameNotebook).mockResolvedValue(undefined);
    vi.mocked(listNotebooks).mockResolvedValue([makeNotebookSummary({ title: 'Renamed' })]);

    await renameNotebookAction('nb-001', 'Renamed');

    expect(renameNotebook).toHaveBeenCalledWith('nb-001', 'Renamed');
    expect(notebookStore.notebooks[0].title).toBe('Renamed');
  });
});

// ---------------------------------------------------------------------------
// trashCount derived
// ---------------------------------------------------------------------------

describe('notebookStore.trashCount', () => {
  it('reflects the number of trashed notebooks', async () => {
    vi.mocked(listTrashed).mockResolvedValue([
      makeNotebookSummary({ id: 'nb-001', trashed_at: new Date().toISOString() }),
      makeNotebookSummary({ id: 'nb-002', trashed_at: new Date().toISOString() })
    ]);
    await loadTrashed();
    expect(notebookStore.trashCount).toBe(2);
  });

  it('is 0 initially', () => {
    expect(notebookStore.trashCount).toBe(0);
  });
});
