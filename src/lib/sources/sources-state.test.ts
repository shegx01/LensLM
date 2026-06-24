// Store unit tests for sources-state.svelte.ts.
//
// The IPC module is mocked so tests run without a Tauri host.
// `resetSourcesStore()` is called in afterEach to prevent cross-test bleed
// from module-level $state globals — same pattern as notebooks-state tests.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  sourcesStore,
  resetSourcesStore,
  loadSources,
  ingest,
  toggleSelected,
  removeSource,
  undoRemove
} from './sources-state.svelte.js';

// ---------------------------------------------------------------------------
// Mock the IPC layer
// ---------------------------------------------------------------------------

vi.mock('./ipc.js', () => ({
  listSources: vi.fn(),
  addTextSource: vi.fn(),
  addFileSource: vi.fn(),
  ingestSource: vi.fn(),
  setSourceSelected: vi.fn(),
  trashSource: vi.fn(),
  restoreSource: vi.fn()
}));

// Mock @tauri-apps/api/core so the $effect.root auto-refresh does not error.
vi.mock('@tauri-apps/api/core', () => ({
  isTauri: () => false,
  invoke: vi.fn(),
  Channel: vi.fn()
}));

// Mock the notebooks store to avoid cross-module $state side effects.
vi.mock('$lib/notebooks/notebooks-state.svelte.js', () => ({
  notebookStore: {
    get activeNotebookId() {
      return null;
    }
  }
}));

// Import the mocked functions so we can configure return values per test.
import { listSources, ingestSource, setSourceSelected, trashSource, restoreSource } from './ipc.js';
import type { Source } from './types.js';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function makeSource(overrides?: Partial<Source>): Source {
  return {
    id: 'src-001',
    notebook_id: 'nb-001',
    kind: 'file',
    title: 'My Document.md',
    status: 'indexed',
    locator: '/path/to/my-document.md',
    selected: 1,
    created_at: new Date(Date.now() - 3600_000).toISOString(),
    token_count: 512,
    content_hash: 'abc123',
    ...overrides
  };
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
  resetSourcesStore();
});

afterEach(() => {
  resetSourcesStore();
});

// ---------------------------------------------------------------------------
// resetSourcesStore
// ---------------------------------------------------------------------------

describe('resetSourcesStore', () => {
  it('resets all fields to initial values', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource()]);
    await loadSources('nb-001');
    sourcesStore.error = 'some error';

    resetSourcesStore();

    expect(sourcesStore.sources).toHaveLength(0);
    expect(sourcesStore.loading).toBe(false);
    expect(sourcesStore.error).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// loadSources
// ---------------------------------------------------------------------------

describe('loadSources', () => {
  it('populates sources from listSources()', async () => {
    const data = [makeSource({ id: 'src-001' }), makeSource({ id: 'src-002' })];
    vi.mocked(listSources).mockResolvedValue(data);

    await loadSources('nb-001');

    expect(sourcesStore.sources).toEqual(data);
  });

  it('sets loading to false after success', async () => {
    vi.mocked(listSources).mockResolvedValue([]);
    await loadSources('nb-001');
    expect(sourcesStore.loading).toBe(false);
  });

  it('sets loading to false and sets error on failure', async () => {
    vi.mocked(listSources).mockRejectedValue(new Error('DB error'));
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await loadSources('nb-001');

    expect(sourcesStore.loading).toBe(false);
    expect(sourcesStore.error).toBeTruthy();
    consoleSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// toggleSelected
// ---------------------------------------------------------------------------

describe('toggleSelected', () => {
  it('flips selected from 1 to 0 and persists via setSourceSelected', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001', selected: 1 })]);
    await loadSources('nb-001');
    vi.mocked(setSourceSelected).mockResolvedValue(undefined);

    await toggleSelected('src-001');

    expect(setSourceSelected).toHaveBeenCalledWith('src-001', false);
    expect(sourcesStore.sources[0].selected).toBe(0);
  });

  it('flips selected from 0 to 1 and persists via setSourceSelected', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001', selected: 0 })]);
    await loadSources('nb-001');
    vi.mocked(setSourceSelected).mockResolvedValue(undefined);

    await toggleSelected('src-001');

    expect(setSourceSelected).toHaveBeenCalledWith('src-001', true);
    expect(sourcesStore.sources[0].selected).toBe(1);
  });

  // BUG 2 regression: setSourceSelected must receive a boolean, not 0/1
  it('calls setSourceSelected with a boolean (not a number)', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001', selected: 1 })]);
    await loadSources('nb-001');
    vi.mocked(setSourceSelected).mockResolvedValue(undefined);

    await toggleSelected('src-001');

    const calledWith = vi.mocked(setSourceSelected).mock.calls[0][1];
    expect(typeof calledWith).toBe('boolean');
  });

  it('reverts optimistic update on setSourceSelected failure', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001', selected: 1 })]);
    await loadSources('nb-001');
    vi.mocked(setSourceSelected).mockRejectedValue(new Error('network error'));
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await toggleSelected('src-001');

    // Reverted back to original value
    expect(sourcesStore.sources[0].selected).toBe(1);
    expect(sourcesStore.error).toBeTruthy();
    consoleSpy.mockRestore();
  });

  it('is a no-op for an unknown sourceId', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001' })]);
    await loadSources('nb-001');

    await toggleSelected('src-unknown');

    expect(setSourceSelected).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// ingest — status updates from progress events
// ---------------------------------------------------------------------------

describe('ingest', () => {
  // BUG 1 regression: phase updates must come from the 'chunk' event, not 'progress'
  it('updates status to "embedding" on chunk event with phase:"embedding"', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001', status: 'queued' })]);
    await loadSources('nb-001');

    let capturedHandler: ((e: unknown) => void) | null = null;
    vi.mocked(ingestSource).mockImplementation(async (_id, onProgress) => {
      capturedHandler = onProgress as (e: unknown) => void;
    });

    const ingestPromise = ingest('src-001');
    if (capturedHandler) {
      (capturedHandler as (e: unknown) => void)({
        type: 'chunk',
        data: { phase: 'embedding', done: 2, total: 5 }
      });
    }
    await ingestPromise;

    expect(sourcesStore.sources[0].status).toBe('embedding');
  });

  it('does NOT update status on progress event (no phase field)', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001', status: 'queued' })]);
    await loadSources('nb-001');

    let capturedHandler: ((e: unknown) => void) | null = null;
    vi.mocked(ingestSource).mockImplementation(async (_id, onProgress) => {
      capturedHandler = onProgress as (e: unknown) => void;
    });

    const ingestPromise = ingest('src-001');
    if (capturedHandler) {
      // 'progress' has no phase — status must stay unchanged
      (capturedHandler as (e: unknown) => void)({ type: 'progress', data: { done: 1, total: 5 } });
    }
    await ingestPromise;

    expect(sourcesStore.sources[0].status).toBe('queued');
  });

  it('updates status to "parsing" on started event', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001', status: 'queued' })]);
    await loadSources('nb-001');

    // Capture the onProgress handler and drive it manually
    let capturedHandler: ((e: unknown) => void) | null = null;
    vi.mocked(ingestSource).mockImplementation(async (_id, onProgress) => {
      capturedHandler = onProgress as (e: unknown) => void;
    });

    const ingestPromise = ingest('src-001');
    if (capturedHandler) {
      (capturedHandler as (e: unknown) => void)({ type: 'started' });
    }
    await ingestPromise;

    expect(sourcesStore.sources[0].status).toBe('parsing');
  });

  it('updates status to "indexed" on done event', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001', status: 'embedding' })]);
    await loadSources('nb-001');

    let capturedHandler: ((e: unknown) => void) | null = null;
    vi.mocked(ingestSource).mockImplementation(async (_id, onProgress) => {
      capturedHandler = onProgress as (e: unknown) => void;
    });

    const ingestPromise = ingest('src-001');
    if (capturedHandler) {
      (capturedHandler as (e: unknown) => void)({ type: 'done' });
    }
    await ingestPromise;

    expect(sourcesStore.sources[0].status).toBe('indexed');
  });

  it('updates status to "error" on failed event', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001', status: 'parsing' })]);
    await loadSources('nb-001');

    let capturedHandler: ((e: unknown) => void) | null = null;
    vi.mocked(ingestSource).mockImplementation(async (_id, onProgress) => {
      capturedHandler = onProgress as (e: unknown) => void;
    });

    const ingestPromise = ingest('src-001');
    if (capturedHandler) {
      (capturedHandler as (e: unknown) => void)({
        type: 'failed',
        data: { kind: 'Internal', message: 'boom' }
      });
    }
    await ingestPromise;

    expect(sourcesStore.sources[0].status).toBe('error');
  });

  it('sets status to "error" and sets store error when ingestSource throws', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001', status: 'queued' })]);
    await loadSources('nb-001');
    vi.mocked(ingestSource).mockRejectedValue(new Error('channel closed'));
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await ingest('src-001');

    expect(sourcesStore.sources[0].status).toBe('error');
    expect(sourcesStore.error).toBeTruthy();
    consoleSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// removeSource — soft-delete (trash) with optimistic remove + undo stash
// ---------------------------------------------------------------------------

describe('removeSource', () => {
  it('removes the row from sources immediately (optimistic)', async () => {
    vi.mocked(listSources).mockResolvedValue([
      makeSource({ id: 'src-001' }),
      makeSource({ id: 'src-002' })
    ]);
    await loadSources('nb-001');
    vi.mocked(trashSource).mockResolvedValue(undefined);

    await removeSource('src-001');

    expect(sourcesStore.sources).toHaveLength(1);
    expect(sourcesStore.sources[0].id).toBe('src-002');
  });

  it('calls trashSource with the correct sourceId', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001' })]);
    await loadSources('nb-001');
    vi.mocked(trashSource).mockResolvedValue(undefined);

    await removeSource('src-001');

    expect(trashSource).toHaveBeenCalledWith('src-001');
  });

  it('sets recentlyTrashed to true after a successful soft-delete', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001' })]);
    await loadSources('nb-001');
    vi.mocked(trashSource).mockResolvedValue(undefined);

    await removeSource('src-001');

    expect(sourcesStore.recentlyTrashed).toBe(true);
  });

  it('reverts the optimistic remove when trashSource fails', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001' })]);
    await loadSources('nb-001');
    vi.mocked(trashSource).mockRejectedValue(new Error('DB error'));
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await removeSource('src-001');

    // Row should be restored after failure
    expect(sourcesStore.sources).toHaveLength(1);
    expect(sourcesStore.sources[0].id).toBe('src-001');
    expect(sourcesStore.error).toBeTruthy();
    consoleSpy.mockRestore();
  });

  it('does not set recentlyTrashed when trashSource fails', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001' })]);
    await loadSources('nb-001');
    vi.mocked(trashSource).mockRejectedValue(new Error('DB error'));
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await removeSource('src-001');

    expect(sourcesStore.recentlyTrashed).toBe(false);
    consoleSpy.mockRestore();
  });

  it('is a no-op for an unknown sourceId', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001' })]);
    await loadSources('nb-001');

    await removeSource('src-unknown');

    expect(trashSource).not.toHaveBeenCalled();
    expect(sourcesStore.sources).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// undoRemove — re-inserts at original index + calls restoreSource
// ---------------------------------------------------------------------------

describe('undoRemove', () => {
  it('is a no-op when there is nothing to undo', async () => {
    vi.mocked(restoreSource).mockResolvedValue(undefined);

    await undoRemove();

    expect(restoreSource).not.toHaveBeenCalled();
  });

  it('re-inserts the source at its original list index', async () => {
    vi.mocked(listSources).mockResolvedValue([
      makeSource({ id: 'src-001' }),
      makeSource({ id: 'src-002' }),
      makeSource({ id: 'src-003' })
    ]);
    await loadSources('nb-001');
    vi.mocked(trashSource).mockResolvedValue(undefined);
    vi.mocked(restoreSource).mockResolvedValue(undefined);

    // Remove the middle item (index 1)
    await removeSource('src-002');
    expect(sourcesStore.sources).toHaveLength(2);

    // Undo — should re-insert src-002 at index 1
    await undoRemove();

    expect(sourcesStore.sources).toHaveLength(3);
    expect(sourcesStore.sources[1].id).toBe('src-002');
  });

  it('calls restoreSource with the correct sourceId', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001' })]);
    await loadSources('nb-001');
    vi.mocked(trashSource).mockResolvedValue(undefined);
    vi.mocked(restoreSource).mockResolvedValue(undefined);

    await removeSource('src-001');
    await undoRemove();

    expect(restoreSource).toHaveBeenCalledWith('src-001');
  });

  it('clears recentlyTrashed after a successful undo', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001' })]);
    await loadSources('nb-001');
    vi.mocked(trashSource).mockResolvedValue(undefined);
    vi.mocked(restoreSource).mockResolvedValue(undefined);

    await removeSource('src-001');
    expect(sourcesStore.recentlyTrashed).toBe(true);

    await undoRemove();

    expect(sourcesStore.recentlyTrashed).toBe(false);
  });

  it('reverts the re-insert when restoreSource fails', async () => {
    vi.mocked(listSources).mockResolvedValue([makeSource({ id: 'src-001' })]);
    await loadSources('nb-001');
    vi.mocked(trashSource).mockResolvedValue(undefined);
    vi.mocked(restoreSource).mockRejectedValue(new Error('restore failed'));
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await removeSource('src-001');
    await undoRemove();

    // Re-insert was reverted — list should be empty again
    expect(sourcesStore.sources).toHaveLength(0);
    expect(sourcesStore.error).toBeTruthy();
    consoleSpy.mockRestore();
  });

  it('re-inserts the first item (index 0) correctly after removal', async () => {
    vi.mocked(listSources).mockResolvedValue([
      makeSource({ id: 'src-001' }),
      makeSource({ id: 'src-002' })
    ]);
    await loadSources('nb-001');
    vi.mocked(trashSource).mockResolvedValue(undefined);
    vi.mocked(restoreSource).mockResolvedValue(undefined);

    await removeSource('src-001');
    await undoRemove();

    expect(sourcesStore.sources[0].id).toBe('src-001');
    expect(sourcesStore.sources[1].id).toBe('src-002');
  });
});
