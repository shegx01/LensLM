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
  loadTrashedSources,
  refreshTrashed,
  refreshTrashedSources,
  createNotebookAction,
  renameNotebookAction,
  trashNotebookAction,
  restoreNotebookAction,
  purgeNotebookAction,
  restoreSourceFromTrash,
  purgeSourceAction,
  selectNotebook,
  openTrash,
  notebookColorClass
} from './notebooks-state.svelte.js';
import { NOTEBOOK_PALETTE, notebookAccentClass } from './notebook-color.js';

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
  purgeNotebook: vi.fn(),
  touchNotebookActivity: vi.fn().mockResolvedValue(undefined)
}));

// Mock the sources IPC layer (used by notebooks-state for trashed sources).
vi.mock('$lib/sources/ipc.js', () => ({
  listTrashedSources: vi.fn(),
  purgeSource: vi.fn(),
  restoreSource: vi.fn()
}));

// Mock the sources store (used for loadSources + drainTrashQueueEntry).
vi.mock('$lib/sources/sources-state.svelte.js', () => ({
  loadSources: vi.fn(),
  drainTrashQueueEntry: vi.fn()
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
import { listTrashedSources, purgeSource, restoreSource } from '$lib/sources/ipc.js';
import { loadSources, drainTrashQueueEntry } from '$lib/sources/sources-state.svelte.js';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

import type { NotebookSummary, Notebook } from './types.js';
import type { TrashedSource } from '$lib/sources/types.js';

function makeTrashedSource(overrides?: Partial<TrashedSource>): TrashedSource {
  return {
    id: 'src-001',
    notebook_id: 'nb-001',
    kind: 'pdf',
    title: 'My Report.pdf',
    status: 'indexed',
    locator: '/path/to/my-report.pdf',
    selected: 1,
    created_at: new Date(Date.now() - 7200_000).toISOString(),
    token_count: 1024,
    content_hash: 'abc123',
    raw_content_hash: null,
    trashed_at: new Date(Date.now() - 3600_000).toISOString(),
    enrichment_status: null,
    enrichment_meta: null,
    force_js_render: 0,
    notebook_title: 'My Notebook',
    ...overrides
  };
}

function makeNotebookSummary(overrides?: Partial<NotebookSummary>): NotebookSummary {
  return {
    id: 'nb-001',
    title: 'My Notebook',
    description: null,
    focus_mode: 'research',
    created_at: new Date(Date.now() - 3600_000).toISOString(),
    updated_at: new Date(Date.now() - 3600_000).toISOString(),
    trashed_at: null,
    last_activity_at: null,
    embedding_model: null,
    embedding_backend: null,
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
    last_activity_at: null,
    embedding_model: null,
    embedding_backend: null,
    ...overrides
  };
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
  resetNotebookStore();
  // Default mocks for sources dependencies
  vi.mocked(listTrashedSources).mockResolvedValue([]);
  vi.mocked(loadSources).mockResolvedValue(undefined);
  vi.mocked(drainTrashQueueEntry).mockReturnValue(undefined);
  vi.mocked(purgeSource).mockResolvedValue(undefined);
  vi.mocked(restoreSource).mockResolvedValue(undefined);
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
    notebookStore.rightRailCollapsed = true;
    notebookStore.paletteOpen = true;
    notebookStore.trashOpen = true;

    resetNotebookStore();

    expect(notebookStore.notebooks).toHaveLength(0);
    expect(notebookStore.activeNotebookId).toBeNull();
    expect(notebookStore.trashOpen).toBe(false);
    expect(notebookStore.sidebarCollapsed).toBe(false);
    expect(notebookStore.rightRailCollapsed).toBe(false);
    expect(notebookStore.paletteOpen).toBe(false);
    expect(notebookStore.paletteQuery).toBe('');
    expect(notebookStore.loading).toBe(false);
    expect(notebookStore.error).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// loadNotebooks
// ---------------------------------------------------------------------------

describe('rightRailCollapsed', () => {
  it('defaults to false and round-trips through the setter', () => {
    expect(notebookStore.rightRailCollapsed).toBe(false);
    notebookStore.rightRailCollapsed = true;
    expect(notebookStore.rightRailCollapsed).toBe(true);
    notebookStore.rightRailCollapsed = false;
    expect(notebookStore.rightRailCollapsed).toBe(false);
  });
});

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
    vi.mocked(listTrashedSources).mockResolvedValue([]);

    await openTrash();

    expect(notebookStore.trashOpen).toBe(true);
    expect(notebookStore.trashedNotebooks).toHaveLength(1);
  });

  it('also loads trashed sources when opening', async () => {
    vi.mocked(listTrashed).mockResolvedValue([]);
    vi.mocked(listTrashedSources).mockResolvedValue([makeTrashedSource()]);

    await openTrash();

    expect(notebookStore.trashOpen).toBe(true);
    expect(notebookStore.trashedSources).toHaveLength(1);
  });

  it('still renders trashed notebooks when trashed-sources fetch rejects (Promise.allSettled)', async () => {
    const trashed = makeNotebookSummary({ id: 'nb-trashed', trashed_at: new Date().toISOString() });
    vi.mocked(listTrashed).mockResolvedValue([trashed]);
    vi.mocked(listTrashedSources).mockRejectedValue(new Error('sources fetch failed'));
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await openTrash();

    // Notebooks still rendered despite sources error
    expect(notebookStore.trashedNotebooks).toHaveLength(1);
    // Error is set from the rejection
    expect(notebookStore.error).toBeTruthy();
    consoleSpy.mockRestore();
  });

  it('still renders trashed sources when trashed-notebooks fetch rejects (Promise.allSettled)', async () => {
    vi.mocked(listTrashed).mockRejectedValue(new Error('notebooks fetch failed'));
    vi.mocked(listTrashedSources).mockResolvedValue([makeTrashedSource()]);
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await openTrash();

    // Sources still rendered despite notebooks error
    expect(notebookStore.trashedSources).toHaveLength(1);
    expect(notebookStore.error).toBeTruthy();
    consoleSpy.mockRestore();
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
// notebookColorClass — rank-based decorative assignment
// ---------------------------------------------------------------------------

describe('notebookColorClass (rank-based)', () => {
  // Helper: build N notebooks with ascending ids (UUIDv7-like ordering).
  function ascendingIds(n: number): NotebookSummary[] {
    return Array.from({ length: n }, (_, i) =>
      makeNotebookSummary({ id: `018f4c7e-0000-7b00-0000-${String(i).padStart(12, '0')}` })
    );
  }

  it('assigns 10 DISTINCT classes to the first 10 notebooks', async () => {
    vi.mocked(listNotebooks).mockResolvedValue(ascendingIds(10));
    await loadNotebooks();

    const classes = notebookStore.notebooks.map((n) => notebookColorClass(n.id));
    expect(new Set(classes).size).toBe(10);
    // Every palette hue is represented exactly once.
    expect(new Set(classes)).toEqual(new Set(NOTEBOOK_PALETTE.map((p) => `nb-${p}`)));
  });

  it('wraps the 11th notebook back to the first palette hue', async () => {
    vi.mocked(listNotebooks).mockResolvedValue(ascendingIds(11));
    await loadNotebooks();

    const sorted = [...notebookStore.notebooks].sort((a, b) => (a.id < b.id ? -1 : 1));
    const first = notebookColorClass(sorted[0].id);
    const eleventh = notebookColorClass(sorted[10].id);
    expect(eleventh).toBe(first); // 10 % 10 === 0
  });

  it('is stable on append: existing ranks do not shift when a later id is added', async () => {
    vi.mocked(listNotebooks).mockResolvedValue(ascendingIds(3));
    await loadNotebooks();
    const before = notebookStore.notebooks.map((n) => [n.id, notebookColorClass(n.id)] as const);

    // Append a notebook with a strictly greater id (UUIDv7 = creation-ordered).
    vi.mocked(listNotebooks).mockResolvedValue([
      ...ascendingIds(3),
      makeNotebookSummary({ id: '018f4c7e-0000-7b00-0000-zzzzzzzzzzzz' })
    ]);
    await loadNotebooks();

    for (const [id, cls] of before) {
      expect(notebookColorClass(id)).toBe(cls);
    }
  });

  it('falls back to the pure hash for ids not in the live set', async () => {
    vi.mocked(listNotebooks).mockResolvedValue(ascendingIds(2));
    await loadNotebooks();

    const trashedId = 'not-in-live-set-trashed';
    expect(notebookColorClass(trashedId)).toBe(notebookAccentClass(trashedId));
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

  it('includes trashed sources in the count', async () => {
    vi.mocked(listTrashedSources).mockResolvedValue([
      makeTrashedSource({ id: 'src-001' }),
      makeTrashedSource({ id: 'src-002' })
    ]);
    await loadTrashedSources();
    expect(notebookStore.trashCount).toBe(2);
  });

  it('sums trashed notebooks AND trashed sources', async () => {
    vi.mocked(listTrashed).mockResolvedValue([
      makeNotebookSummary({ id: 'nb-001', trashed_at: new Date().toISOString() })
    ]);
    vi.mocked(listTrashedSources).mockResolvedValue([
      makeTrashedSource({ id: 'src-001' }),
      makeTrashedSource({ id: 'src-002' })
    ]);
    await loadTrashed();
    await loadTrashedSources();
    expect(notebookStore.trashCount).toBe(3);
  });

  it('resets to 0 after resetNotebookStore()', async () => {
    vi.mocked(listTrashed).mockResolvedValue([
      makeNotebookSummary({ id: 'nb-001', trashed_at: new Date().toISOString() })
    ]);
    vi.mocked(listTrashedSources).mockResolvedValue([makeTrashedSource({ id: 'src-001' })]);
    await loadTrashed();
    await loadTrashedSources();
    expect(notebookStore.trashCount).toBe(2);

    resetNotebookStore();

    expect(notebookStore.trashCount).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// loadTrashedSources
// ---------------------------------------------------------------------------

describe('loadTrashedSources', () => {
  it('populates trashedSources from listTrashedSources()', async () => {
    const data = [makeTrashedSource({ id: 'src-001' }), makeTrashedSource({ id: 'src-002' })];
    vi.mocked(listTrashedSources).mockResolvedValue(data);

    await loadTrashedSources();

    expect(notebookStore.trashedSources).toEqual(data);
  });

  it('sets loading to false after success', async () => {
    vi.mocked(listTrashedSources).mockResolvedValue([]);
    await loadTrashedSources();
    expect(notebookStore.loading).toBe(false);
  });

  it('sets loading to false and sets error on failure', async () => {
    vi.mocked(listTrashedSources).mockRejectedValue(new Error('DB error'));
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await loadTrashedSources();

    expect(notebookStore.loading).toBe(false);
    expect(notebookStore.error).toBeTruthy();
    consoleSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// restoreSourceFromTrash
// ---------------------------------------------------------------------------

describe('restoreSourceFromTrash', () => {
  it('calls restoreSource IPC and refreshes trashedSources', async () => {
    const src = makeTrashedSource({ id: 'src-001', notebook_id: 'nb-999' });
    vi.mocked(listTrashedSources).mockResolvedValue([src]);
    await loadTrashedSources();

    vi.mocked(restoreSource).mockResolvedValue(undefined);
    vi.mocked(listTrashedSources).mockResolvedValue([]);

    await restoreSourceFromTrash('src-001');

    expect(restoreSource).toHaveBeenCalledWith('src-001');
    expect(notebookStore.trashedSources).toHaveLength(0);
  });

  it('drains the trashQueue entry for the source before restoring', async () => {
    const src = makeTrashedSource({ id: 'src-001', notebook_id: 'nb-999' });
    vi.mocked(listTrashedSources).mockResolvedValue([src]);
    await loadTrashedSources();
    vi.mocked(restoreSource).mockResolvedValue(undefined);
    vi.mocked(listTrashedSources).mockResolvedValue([]);

    await restoreSourceFromTrash('src-001');

    expect(drainTrashQueueEntry).toHaveBeenCalledWith('src-001');
  });

  it('refreshes active source list when the source belongs to the active notebook', async () => {
    notebookStore.activeNotebookId = 'nb-active';
    const src = makeTrashedSource({ id: 'src-001', notebook_id: 'nb-active' });
    vi.mocked(listTrashedSources).mockResolvedValue([src]);
    await loadTrashedSources();
    vi.mocked(restoreSource).mockResolvedValue(undefined);
    vi.mocked(listTrashedSources).mockResolvedValue([]);

    await restoreSourceFromTrash('src-001');

    expect(loadSources).toHaveBeenCalledWith('nb-active');
  });

  it('does NOT refresh active source list when source belongs to a different notebook', async () => {
    notebookStore.activeNotebookId = 'nb-active';
    const src = makeTrashedSource({ id: 'src-001', notebook_id: 'nb-other' });
    vi.mocked(listTrashedSources).mockResolvedValue([src]);
    await loadTrashedSources();
    vi.mocked(restoreSource).mockResolvedValue(undefined);
    vi.mocked(listTrashedSources).mockResolvedValue([]);

    await restoreSourceFromTrash('src-001');

    expect(loadSources).not.toHaveBeenCalled();
  });

  it('does NOT refresh active source list when no notebook is active', async () => {
    notebookStore.activeNotebookId = null;
    const src = makeTrashedSource({ id: 'src-001', notebook_id: 'nb-001' });
    vi.mocked(listTrashedSources).mockResolvedValue([src]);
    await loadTrashedSources();
    vi.mocked(restoreSource).mockResolvedValue(undefined);
    vi.mocked(listTrashedSources).mockResolvedValue([]);

    await restoreSourceFromTrash('src-001');

    expect(loadSources).not.toHaveBeenCalled();
  });

  it('sets error on failure and loading stays false', async () => {
    const src = makeTrashedSource({ id: 'src-001' });
    vi.mocked(listTrashedSources).mockResolvedValue([src]);
    await loadTrashedSources();
    vi.mocked(restoreSource).mockRejectedValue(new Error('restore failed'));
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await restoreSourceFromTrash('src-001');

    expect(notebookStore.loading).toBe(false);
    expect(notebookStore.error).toBeTruthy();
    consoleSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// purgeSourceAction
// ---------------------------------------------------------------------------

describe('purgeSourceAction', () => {
  it('calls purgeSource IPC and refreshes trashedSources', async () => {
    const src = makeTrashedSource({ id: 'src-001', notebook_id: 'nb-999' });
    vi.mocked(listTrashedSources).mockResolvedValue([src]);
    await loadTrashedSources();

    vi.mocked(purgeSource).mockResolvedValue(undefined);
    vi.mocked(listTrashedSources).mockResolvedValue([]);

    await purgeSourceAction('src-001');

    expect(purgeSource).toHaveBeenCalledWith('src-001');
    expect(notebookStore.trashedSources).toHaveLength(0);
  });

  it('drains the trashQueue entry for the source before purging', async () => {
    const src = makeTrashedSource({ id: 'src-001', notebook_id: 'nb-999' });
    vi.mocked(listTrashedSources).mockResolvedValue([src]);
    await loadTrashedSources();
    vi.mocked(purgeSource).mockResolvedValue(undefined);
    vi.mocked(listTrashedSources).mockResolvedValue([]);

    await purgeSourceAction('src-001');

    expect(drainTrashQueueEntry).toHaveBeenCalledWith('src-001');
  });

  it('refreshes active source list when the source belongs to the active notebook', async () => {
    notebookStore.activeNotebookId = 'nb-active';
    const src = makeTrashedSource({ id: 'src-001', notebook_id: 'nb-active' });
    vi.mocked(listTrashedSources).mockResolvedValue([src]);
    await loadTrashedSources();
    vi.mocked(purgeSource).mockResolvedValue(undefined);
    vi.mocked(listTrashedSources).mockResolvedValue([]);

    await purgeSourceAction('src-001');

    expect(loadSources).toHaveBeenCalledWith('nb-active');
  });

  it('does NOT refresh active source list when source belongs to a different notebook', async () => {
    notebookStore.activeNotebookId = 'nb-active';
    const src = makeTrashedSource({ id: 'src-001', notebook_id: 'nb-other' });
    vi.mocked(listTrashedSources).mockResolvedValue([src]);
    await loadTrashedSources();
    vi.mocked(purgeSource).mockResolvedValue(undefined);
    vi.mocked(listTrashedSources).mockResolvedValue([]);

    await purgeSourceAction('src-001');

    expect(loadSources).not.toHaveBeenCalled();
  });

  it('sets error on failure and loading stays false', async () => {
    const src = makeTrashedSource({ id: 'src-001' });
    vi.mocked(listTrashedSources).mockResolvedValue([src]);
    await loadTrashedSources();
    vi.mocked(purgeSource).mockRejectedValue(new Error('purge failed'));
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await purgeSourceAction('src-001');

    expect(notebookStore.loading).toBe(false);
    expect(notebookStore.error).toBeTruthy();
    consoleSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// FIX A — refreshTrashedSources coalescing serial pattern
// ---------------------------------------------------------------------------

describe('refreshTrashedSources (coalescing race guard)', () => {
  it('concurrent calls coalesce: final state reflects the LATEST fetch, not an earlier stale one', async () => {
    // Deferred promise helpers so we control resolution order.
    let resolveFirst!: (v: ReturnType<typeof makeTrashedSource>[]) => void;
    let resolveSecond!: (v: ReturnType<typeof makeTrashedSource>[]) => void;
    const firstResult = [makeTrashedSource({ id: 'src-STALE' })];
    const secondResult = [
      makeTrashedSource({ id: 'src-LATEST-1' }),
      makeTrashedSource({ id: 'src-LATEST-2' })
    ];

    const firstFetch = new Promise<ReturnType<typeof makeTrashedSource>[]>((res) => {
      resolveFirst = res;
    });
    const secondFetch = new Promise<ReturnType<typeof makeTrashedSource>[]>((res) => {
      resolveSecond = res;
    });

    vi.mocked(listTrashedSources)
      .mockReturnValueOnce(firstFetch as unknown as Promise<TrashedSource[]>)
      .mockReturnValueOnce(secondFetch as unknown as Promise<TrashedSource[]>);

    // Fire both calls concurrently. The second call should queue and NOT start
    // a second fetch until the first fetch settles.
    const p1 = refreshTrashedSources();
    const p2 = refreshTrashedSources();

    // listTrashedSources should have been called only once so far (the second
    // call is coalesced onto the in-flight promise).
    expect(vi.mocked(listTrashedSources)).toHaveBeenCalledTimes(1);

    // Now resolve the first (stale) fetch — the loop should notice the queued
    // flag and immediately run a second fetch.
    resolveFirst(firstResult);
    // Let microtasks settle so the do-while loop can check _trashSourcesRefreshQueued.
    await Promise.resolve();
    await Promise.resolve();

    // A second fetch should now be in-flight.
    expect(vi.mocked(listTrashedSources)).toHaveBeenCalledTimes(2);

    // Resolve the second (latest) fetch.
    resolveSecond(secondResult);
    await Promise.all([p1, p2]);

    // Final state must reflect the second (latest) result.
    expect(notebookStore.trashedSources).toHaveLength(2);
    expect(notebookStore.trashedSources.map((s) => s.id)).toContain('src-LATEST-1');
    expect(notebookStore.trashedSources.map((s) => s.id)).toContain('src-LATEST-2');
  });

  it('resets guard flags via resetNotebookStore so subsequent calls start fresh', async () => {
    // Start a refresh, let it complete, then reset.
    vi.mocked(listTrashedSources).mockResolvedValueOnce([
      makeTrashedSource({ id: 'before-reset' })
    ]);
    await refreshTrashedSources();
    expect(notebookStore.trashedSources).toHaveLength(1);

    // After reset, guards are cleared and state is zeroed.
    resetNotebookStore();
    expect(notebookStore.trashedSources).toHaveLength(0);

    // A fresh call after reset should trigger a new independent fetch (no coalescing bleed).
    vi.mocked(listTrashedSources).mockResolvedValueOnce([
      makeTrashedSource({ id: 'after-reset-1' }),
      makeTrashedSource({ id: 'after-reset-2' })
    ]);
    await refreshTrashedSources();
    expect(notebookStore.trashedSources).toHaveLength(2);
    expect(notebookStore.trashedSources[0].id).toBe('after-reset-1');
  });
});

// ---------------------------------------------------------------------------
// FIX B — notebook actions must refresh trashed sources
// ---------------------------------------------------------------------------

describe('trashNotebookAction — refreshes trashed sources', () => {
  it('calls listTrashedSources after trashing a notebook', async () => {
    vi.mocked(trashNotebook).mockResolvedValue(undefined);
    vi.mocked(listNotebooks).mockResolvedValue([]);
    vi.mocked(listTrashed).mockResolvedValue([]);
    vi.mocked(listTrashedSources).mockResolvedValue([]);

    await trashNotebookAction('nb-001');

    expect(vi.mocked(listTrashedSources)).toHaveBeenCalled();
  });
});

describe('restoreNotebookAction — refreshes trashed sources', () => {
  it('calls listTrashedSources after restoring a notebook', async () => {
    vi.mocked(restoreNotebook).mockResolvedValue(undefined);
    vi.mocked(listNotebooks).mockResolvedValue([]);
    vi.mocked(listTrashed).mockResolvedValue([]);
    vi.mocked(listTrashedSources).mockResolvedValue([]);

    await restoreNotebookAction('nb-001');

    expect(vi.mocked(listTrashedSources)).toHaveBeenCalled();
  });
});

describe('purgeNotebookAction — refreshes trashed sources', () => {
  it('calls listTrashedSources after purging a notebook', async () => {
    vi.mocked(purgeNotebook).mockResolvedValue(undefined);
    vi.mocked(listNotebooks).mockResolvedValue([]);
    vi.mocked(listTrashed).mockResolvedValue([]);
    vi.mocked(listTrashedSources).mockResolvedValue([]);

    await purgeNotebookAction('nb-001');

    expect(vi.mocked(listTrashedSources)).toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// FIX C — refreshTrashed is exported and populates trashedNotebooks
// ---------------------------------------------------------------------------

describe('refreshTrashed (exported)', () => {
  it('is exported from notebooks-state and populates trashedNotebooks', async () => {
    const nb = makeNotebookSummary({ id: 'nb-trashed', trashed_at: new Date().toISOString() });
    vi.mocked(listTrashed).mockResolvedValue([nb]);

    await refreshTrashed();

    expect(notebookStore.trashedNotebooks).toHaveLength(1);
    expect(notebookStore.trashedNotebooks[0].id).toBe('nb-trashed');
    // Crucially, it must NOT toggle the shared loading flag (no UI flash).
    expect(notebookStore.loading).toBe(false);
  });
});
