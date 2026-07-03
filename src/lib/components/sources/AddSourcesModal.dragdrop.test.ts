// AddSourcesModal.dragdrop.test.ts
//
// Regression tests for the onMount → $effect fix (issue #95).
//
// THE BUG: AddSourcesModal registers its native drop zone once in onMount with
// `if (!dropZoneEl) return`. When the component mounts while `open=false`, the
// drop-zone element has not rendered yet (it lives behind `{#if open}` +
// `{#if activeTab === 'upload'}`), so dropZoneEl is undefined and registration
// is skipped — FOREVER. Subsequent re-renders with open=true never re-trigger
// onMount.
//
// THE FIX: Registration was moved into an `$effect` keyed on `dropZoneEl`.
// Svelte 5's $effect re-runs whenever the reactive binding changes, so when the
// drop-zone element mounts (open flips true) the effect fires and registers; when
// it unmounts (open flips false or tab changes) the returned cleanup fn
// unregisters.
//
// These tests verify:
//   1. Mounting closed → open correctly registers the drop listener ($effect fix guard).
//   2. A drop on the registered zone calls addFileSource and dismisses the modal.
//   3. A drop of only unsupported files does NOT call addFileSource or dismiss.
//
// NOTE: getBoundingClientRect stubs, devicePixelRatio setup, and the `position`
// field in simulated drop events have been removed — coordinate hit-testing no
// longer exists in dragDrop.ts. Drops are routed to the last-registered target
// unconditionally.

import { render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Source } from '$lib/sources/types.js';

const { mockSourcesStore, mockNotebookStore } = vi.hoisted(() => {
  let _sources: Source[] = [];

  const mockSourcesStore = {
    get sources() {
      return _sources;
    },
    get loading() {
      return false;
    },
    get error() {
      return null;
    },
    set error(_: string | null) {},
    get recentlyTrashed() {
      return false;
    },
    _setSources(s: Source[]) {
      _sources = s;
    }
  };

  const mockNotebookStore = {
    get activeNotebookId() {
      return 'nb-001';
    },
    get activeNotebook() {
      return { id: 'nb-001', title: 'Test Notebook' };
    },
    get rightRailCollapsed() {
      return false;
    },
    set rightRailCollapsed(_: boolean) {}
  };

  return { mockSourcesStore, mockNotebookStore };
});

vi.mock('$lib/sources/sources-state.svelte.js', () => ({
  sourcesStore: mockSourcesStore,
  loadSources: vi.fn().mockResolvedValue(undefined),
  ingest: vi.fn().mockResolvedValue(undefined),
  addSourceLocal: vi.fn(),
  toggleSelected: vi.fn().mockResolvedValue(undefined),
  removeSource: vi.fn().mockResolvedValue(undefined),
  undoRemove: vi.fn().mockResolvedValue(undefined),
  resetSourcesStore: vi.fn(),
  disposeTrashTimers: vi.fn()
}));

vi.mock('$lib/sources/ipc.js', () => ({
  listSources: vi.fn().mockResolvedValue([]),
  addTextSource: vi
    .fn()
    .mockResolvedValue({ source: { id: 'src-new', status: 'pending' }, wasExisting: false }),
  addFileSource: vi.fn().mockResolvedValue({
    source: {
      id: 'src-new',
      notebook_id: 'nb-001',
      kind: 'file',
      title: 'a.pdf',
      status: 'pending',
      locator: '/tmp/a.pdf',
      selected: 1,
      created_at: new Date().toISOString(),
      token_count: null,
      content_hash: null,
      raw_content_hash: null,
      trashed_at: null,
      enrichment_status: null,
      enrichment_meta: null,
      force_js_render: 0,
      error_meta: null
    } satisfies Source,
    wasExisting: false
  }),
  ingestSource: vi.fn().mockResolvedValue(undefined),
  setSourceSelected: vi.fn().mockResolvedValue(undefined),
  trashSource: vi.fn().mockResolvedValue(undefined),
  restoreSource: vi.fn().mockResolvedValue(undefined)
}));

vi.mock('$lib/notebooks/notebooks-state.svelte.js', () => ({
  notebookStore: mockNotebookStore
}));

// The component imports notebookStore from '$lib/notebooks/index.js' (barrel),
// which re-exports from notebooks-state.svelte.js. Mock the barrel too so both
// import paths resolve to the same stub.
vi.mock('$lib/notebooks/index.js', () => ({
  notebookStore: mockNotebookStore
}));

vi.mock('@tauri-apps/api/core', () => ({
  // isTauri must return true so dragDrop.ts actually calls registerDropTarget.
  isTauri: () => true,
  invoke: vi.fn()
}));

vi.mock('@tauri-apps/plugin-dialog', () => ({
  open: vi.fn().mockResolvedValue(null)
}));

// Toast — referenced by dragDrop.ts for rejected-extension notifications.
vi.mock('$lib/sources/toast.svelte.js', () => ({
  showToast: vi.fn()
}));

// Capture the onDragDropEvent handler so tests can simulate native drop events.
// Must be defined at module-evaluation time so the dynamic import in registerDropTarget
// always resolves to this mock.
let capturedHandler: ((e: unknown) => void) | null = null;

vi.mock('@tauri-apps/api/webview', () => ({
  getCurrentWebview: () => ({
    onDragDropEvent: (h: (e: unknown) => void) => {
      capturedHandler = h;
      return Promise.resolve(() => {});
    }
  })
}));

import AddSourcesModal from './AddSourcesModal.svelte';
import { addFileSource } from '$lib/sources/ipc.js';
import { open as openFilePicker } from '@tauri-apps/plugin-dialog';
import { addSourceLocal, ingest } from '$lib/sources/sources-state.svelte.js';
import { showToast } from '$lib/sources/toast.svelte.js';

const mockAddFileSource = vi.mocked(addFileSource);
const mockOpenFilePicker = vi.mocked(openFilePicker);
const mockAddSourceLocal = vi.mocked(addSourceLocal);
const mockIngest = vi.mocked(ingest);
const mockShowToast = vi.mocked(showToast);

/** Build the `{ source, wasExisting }` outcome the backend returns. */
function outcome(id: string, wasExisting: boolean) {
  return {
    source: {
      id,
      notebook_id: 'nb-001',
      kind: 'file',
      title: `${id}.pdf`,
      status: 'pending',
      locator: `/tmp/${id}.pdf`,
      selected: 1,
      created_at: new Date().toISOString(),
      token_count: null,
      content_hash: null,
      raw_content_hash: null,
      trashed_at: null,
      enrichment_status: null,
      enrichment_meta: null,
      force_js_render: 0,
      error_meta: null
    } satisfies Source,
    wasExisting
  };
}

/** Tick the microtask queue N times to let Svelte $effect / promises settle. */
async function flushEffects(ticks = 5): Promise<void> {
  for (let i = 0; i < ticks; i++) {
    await Promise.resolve();
  }
}

beforeEach(() => {
  vi.clearAllMocks();
  capturedHandler = null;
  mockSourcesStore._setSources([]);
});

afterEach(() => {
  capturedHandler = null;
});

describe('AddSourcesModal — native drag-drop registration ($effect fix)', () => {
  // REGRESSION GUARD: onMount fires once while open=false; dropZoneEl is undefined
  // and registration is skipped forever. $effect keyed on dropZoneEl re-runs when
  // the element mounts, so capturedHandler becomes non-null.
  it('registers the drop listener when the modal transitions from closed to open', async () => {
    const { rerender } = render(AddSourcesModal, { open: false });
    await flushEffects();
    expect(capturedHandler).toBeNull();

    await rerender({ open: true });
    await flushEffects(10);

    await waitFor(
      () => {
        expect(capturedHandler).not.toBeNull();
      },
      { timeout: 500 }
    );
  });

  it('registers the drop listener when rendered with open=true from the start', async () => {
    render(AddSourcesModal, { open: true });
    await flushEffects(10);

    await waitFor(
      () => {
        expect(capturedHandler).not.toBeNull();
      },
      { timeout: 500 }
    );
  });

  it('calls addFileSource and fires onclose when a supported file is dropped', async () => {
    const onclose = vi.fn();
    render(AddSourcesModal, { open: true, onclose });
    await flushEffects(10);

    await waitFor(
      () => {
        expect(capturedHandler).not.toBeNull();
      },
      { timeout: 500 }
    );

    capturedHandler!({
      payload: {
        type: 'drop',
        paths: ['/tmp/a.pdf']
      }
    });

    await waitFor(
      () => {
        expect(addFileSource).toHaveBeenCalledWith('nb-001', 'a.pdf', '/tmp/a.pdf');
      },
      { timeout: 1000 }
    );

    await waitFor(
      () => {
        expect(onclose).toHaveBeenCalledOnce();
      },
      { timeout: 1000 }
    );
  });

  it('does NOT call addFileSource or onclose when only unsupported files are dropped', async () => {
    const onclose = vi.fn();
    render(AddSourcesModal, { open: true, onclose });
    await flushEffects(10);

    await waitFor(
      () => {
        expect(capturedHandler).not.toBeNull();
      },
      { timeout: 500 }
    );

    capturedHandler!({
      payload: {
        type: 'drop',
        paths: ['/tmp/a.mp3']
      }
    });

    await flushEffects(10);

    expect(addFileSource).not.toHaveBeenCalled();
    expect(onclose).not.toHaveBeenCalled();
  });
});

/** Render an open modal and wait until the native drop handler is registered. */
async function renderOpenWithDrop(onclose = vi.fn()) {
  render(AddSourcesModal, { open: true, onclose });
  await flushEffects(10);
  await waitFor(() => expect(capturedHandler).not.toBeNull(), { timeout: 500 });
  return { onclose };
}

/** Fire a native drop of the given (already-supported) paths. */
function fireDrop(paths: string[]): void {
  capturedHandler!({ payload: { type: 'drop', paths } });
}

describe('AddSourcesModal — ingestPaths batch dispatch (#96, via drop)', () => {
  it('all-success: adds every path, ingests each, closes the modal, no toast', async () => {
    mockAddFileSource
      .mockResolvedValueOnce(outcome('s1', false))
      .mockResolvedValueOnce(outcome('s2', false))
      .mockResolvedValueOnce(outcome('s3', false));
    const { onclose } = await renderOpenWithDrop();

    fireDrop(['/tmp/a.pdf', '/tmp/b.txt', '/tmp/c.md']);

    await waitFor(() => expect(mockAddFileSource).toHaveBeenCalledTimes(3), { timeout: 1000 });
    await waitFor(() => expect(onclose).toHaveBeenCalledOnce(), { timeout: 1000 });
    expect(mockAddSourceLocal).toHaveBeenCalledTimes(3);
    expect(mockIngest).toHaveBeenCalledTimes(3);
    expect(mockShowToast).not.toHaveBeenCalled();
  });

  it('partial-failure: batch continues, closes (added>0), toast reports the failure', async () => {
    mockAddFileSource
      .mockResolvedValueOnce(outcome('s1', false))
      .mockRejectedValueOnce(new Error('boom'))
      .mockResolvedValueOnce(outcome('s3', false));
    const { onclose } = await renderOpenWithDrop();

    fireDrop(['/tmp/a.pdf', '/tmp/b.txt', '/tmp/c.md']);

    await waitFor(() => expect(mockAddFileSource).toHaveBeenCalledTimes(3), { timeout: 1000 });
    await waitFor(() => expect(onclose).toHaveBeenCalledOnce(), { timeout: 1000 });
    expect(mockAddSourceLocal).toHaveBeenCalledTimes(2);
    expect(mockIngest).toHaveBeenCalledTimes(2);
    await waitFor(() => expect(mockShowToast).toHaveBeenCalledWith('2 added, 1 failed'), {
      timeout: 1000
    });
  });

  it('all-fail: nothing added, modal stays open, toast reports failures', async () => {
    mockAddFileSource
      .mockRejectedValueOnce(new Error('boom'))
      .mockRejectedValueOnce(new Error('boom'))
      .mockRejectedValueOnce(new Error('boom'));
    const { onclose } = await renderOpenWithDrop();

    fireDrop(['/tmp/a.pdf', '/tmp/b.txt', '/tmp/c.md']);

    await waitFor(() => expect(mockAddFileSource).toHaveBeenCalledTimes(3), { timeout: 1000 });
    await waitFor(() => expect(mockShowToast).toHaveBeenCalledWith('3 failed'), { timeout: 1000 });
    expect(mockAddSourceLocal).not.toHaveBeenCalled();
    expect(mockIngest).not.toHaveBeenCalled();
    expect(onclose).not.toHaveBeenCalled();
  });

  it('duplicate-skip: wasExisting path is skipped (no addSourceLocal/ingest)', async () => {
    mockAddFileSource
      .mockResolvedValueOnce(outcome('s1', false))
      .mockResolvedValueOnce(outcome('s2', true));
    const { onclose } = await renderOpenWithDrop();

    fireDrop(['/tmp/a.pdf', '/tmp/b.txt']);

    await waitFor(() => expect(mockAddFileSource).toHaveBeenCalledTimes(2), { timeout: 1000 });
    await waitFor(() => expect(onclose).toHaveBeenCalledOnce(), { timeout: 1000 });
    expect(mockAddSourceLocal).toHaveBeenCalledTimes(1);
    expect(mockIngest).toHaveBeenCalledTimes(1);
    await waitFor(
      () => expect(mockShowToast).toHaveBeenCalledWith('1 added, 1 already in notebook'),
      {
        timeout: 1000
      }
    );
  });

  it('all-skipped: modal stays open (added=0), toast shows "already in notebook"', async () => {
    mockAddFileSource
      .mockResolvedValueOnce(outcome('s1', true))
      .mockResolvedValueOnce(outcome('s2', true))
      .mockResolvedValueOnce(outcome('s3', true));
    const { onclose } = await renderOpenWithDrop();

    fireDrop(['/tmp/a.pdf', '/tmp/b.txt', '/tmp/c.md']);

    await waitFor(() => expect(mockAddFileSource).toHaveBeenCalledTimes(3), { timeout: 1000 });
    await waitFor(() => expect(mockShowToast).toHaveBeenCalledWith('3 already in notebook'), {
      timeout: 1000
    });
    expect(mockAddSourceLocal).not.toHaveBeenCalled();
    expect(mockIngest).not.toHaveBeenCalled();
    expect(onclose).not.toHaveBeenCalled();
  });

  it('batch summary toast: mixed added + skipped', async () => {
    mockAddFileSource
      .mockResolvedValueOnce(outcome('s1', false))
      .mockResolvedValueOnce(outcome('s2', false))
      .mockResolvedValueOnce(outcome('s3', true));
    await renderOpenWithDrop();

    fireDrop(['/tmp/a.pdf', '/tmp/b.txt', '/tmp/c.md']);

    await waitFor(
      () => expect(mockShowToast).toHaveBeenCalledWith('2 added, 1 already in notebook'),
      { timeout: 1000 }
    );
  });
});

describe('AddSourcesModal — multi-select browse picker (#96)', () => {
  it('multi-select: picker returns 3 paths → addFileSource called 3x with multiple:true', async () => {
    mockOpenFilePicker.mockResolvedValueOnce(['/tmp/a.pdf', '/tmp/b.txt', '/tmp/c.md']);
    mockAddFileSource
      .mockResolvedValueOnce(outcome('s1', false))
      .mockResolvedValueOnce(outcome('s2', false))
      .mockResolvedValueOnce(outcome('s3', false));
    render(AddSourcesModal, { open: true });
    await flushEffects(10);

    const browseBtn = await screen.findByRole('button', { name: /browse your computer/i });
    browseBtn.click();

    await waitFor(() => expect(mockAddFileSource).toHaveBeenCalledTimes(3), { timeout: 1000 });
    expect(mockOpenFilePicker).toHaveBeenCalledWith(expect.objectContaining({ multiple: true }));
    expect(mockAddFileSource).toHaveBeenNthCalledWith(1, 'nb-001', 'a.pdf', '/tmp/a.pdf');
    expect(mockAddFileSource).toHaveBeenNthCalledWith(3, 'nb-001', 'c.md', '/tmp/c.md');
  });

  it('single-select compat: picker returns a bare string → normalized to one path', async () => {
    mockOpenFilePicker.mockResolvedValueOnce('/tmp/only.pdf');
    mockAddFileSource.mockResolvedValueOnce(outcome('s1', false));
    render(AddSourcesModal, { open: true });
    await flushEffects(10);

    const browseBtn = await screen.findByRole('button', { name: /browse your computer/i });
    browseBtn.click();

    await waitFor(() => expect(mockAddFileSource).toHaveBeenCalledTimes(1), { timeout: 1000 });
    expect(mockAddFileSource).toHaveBeenCalledWith('nb-001', 'only.pdf', '/tmp/only.pdf');
  });
});
