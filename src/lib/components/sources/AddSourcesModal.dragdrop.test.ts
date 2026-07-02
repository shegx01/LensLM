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

// ---------------------------------------------------------------------------
// Hoisted mocks
// ---------------------------------------------------------------------------

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
      force_js_render: 0
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

// ---------------------------------------------------------------------------
// @tauri-apps/api/webview mock — CAPTURE the handler
// ---------------------------------------------------------------------------
//
// The drag-drop manager (dragDrop.ts) dynamically imports this module and calls
// getCurrentWebview().onDragDropEvent(handler). We intercept the call so tests
// can invoke `capturedHandler` to simulate native drop events.
//
// The mock is defined at module-evaluation time (before any test runs) so the
// dynamic import inside registerDropTarget always gets this mock.

let capturedHandler: ((e: unknown) => void) | null = null;

vi.mock('@tauri-apps/api/webview', () => ({
  getCurrentWebview: () => ({
    onDragDropEvent: (h: (e: unknown) => void) => {
      capturedHandler = h;
      return Promise.resolve(() => {});
    }
  })
}));

// ---------------------------------------------------------------------------
// Import component + mocked dependencies after mocks are established
// ---------------------------------------------------------------------------

import AddSourcesModal from './AddSourcesModal.svelte';
import { addFileSource } from '$lib/sources/ipc.js';
import { open as openFilePicker } from '@tauri-apps/plugin-dialog';
import { addSourceLocal, ingest } from '$lib/sources/sources-state.svelte.js';
import { showToast } from '$lib/sources/toast.svelte.js';

// Typed handles onto the mocked functions (each is a vi.fn()).
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
      force_js_render: 0
    } satisfies Source,
    wasExisting
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Tick the microtask queue N times to let Svelte $effect / promises settle. */
async function flushEffects(ticks = 5): Promise<void> {
  for (let i = 0; i < ticks; i++) {
    await Promise.resolve();
  }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
  capturedHandler = null;
  mockSourcesStore._setSources([]);
});

afterEach(() => {
  // @testing-library/svelte cleanup() is called globally by vitest-setup.ts.
  // Reset capturedHandler after each test to prevent cross-test leakage.
  capturedHandler = null;
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('AddSourcesModal — native drag-drop registration ($effect fix)', () => {
  // ─────────────────────────────────────────────────────────────────────────
  // Case 1: Mounting closed → open registers the drop listener
  //
  // REGRESSION GUARD — this test would FAIL with the old onMount approach:
  //   onMount fires once while open=false; dropZoneEl is undefined; the guard
  //   `if (!dropZoneEl) return` bails out immediately. Subsequent renders with
  //   open=true never re-trigger onMount, so capturedHandler stays null.
  //
  // With $effect keyed on dropZoneEl: when open flips true, the drop-zone div
  // mounts, Svelte updates the $state binding, the $effect re-runs, and
  // registerDropTarget is called → capturedHandler becomes non-null.
  // ─────────────────────────────────────────────────────────────────────────

  it('registers the drop listener when the modal transitions from closed to open', async () => {
    // Render with open=false — the drop zone is not in the DOM yet.
    const { rerender } = render(AddSourcesModal, { open: false });
    await flushEffects();

    // The drop zone has not mounted; the handler must not be registered yet.
    expect(capturedHandler).toBeNull();

    // Flip open → true — the drop zone div enters the DOM.
    await rerender({ open: true });
    await flushEffects(10);

    // The $effect must have fired and called registerDropTarget, which called
    // onDragDropEvent and stored the handler. This assertion is the regression
    // guard — it would be null under the old onMount approach.
    await waitFor(
      () => {
        expect(capturedHandler).not.toBeNull();
      },
      { timeout: 500 }
    );
  });

  // ─────────────────────────────────────────────────────────────────────────
  // Case 1b: Rendering open=true from the start also registers
  // ─────────────────────────────────────────────────────────────────────────

  it('registers the drop listener when rendered with open=true from the start', async () => {
    render(AddSourcesModal, { open: true });
    await flushEffects(10);

    // The drop zone is rendered immediately; registration must happen.
    await waitFor(
      () => {
        expect(capturedHandler).not.toBeNull();
      },
      { timeout: 500 }
    );
  });

  // ─────────────────────────────────────────────────────────────────────────
  // Case 2: A drop on the zone ingests files and dismisses the modal
  // ─────────────────────────────────────────────────────────────────────────

  it('calls addFileSource and fires onclose when a supported file is dropped', async () => {
    const onclose = vi.fn();
    render(AddSourcesModal, { open: true, onclose });
    await flushEffects(10);

    // Wait for registration.
    await waitFor(
      () => {
        expect(capturedHandler).not.toBeNull();
      },
      { timeout: 500 }
    );

    // Simulate a native drop event — no position field; the active target
    // receives the drop unconditionally (no coordinate hit-test).
    capturedHandler!({
      payload: {
        type: 'drop',
        paths: ['/tmp/a.pdf']
      }
    });

    // The onDrop handler is async; wait for all async work to settle.
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

  // ─────────────────────────────────────────────────────────────────────────
  // Case 3: A drop of only unsupported files does NOT dismiss the modal
  // ─────────────────────────────────────────────────────────────────────────

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

    // Drop an unsupported audio file — dragDrop.ts will partition it to
    // `rejected`; onDrop is never called because accepted.length === 0.
    capturedHandler!({
      payload: {
        type: 'drop',
        paths: ['/tmp/a.mp3']
      }
    });

    // Give async paths time to settle (they should NOT fire here).
    await flushEffects(10);

    expect(addFileSource).not.toHaveBeenCalled();
    expect(onclose).not.toHaveBeenCalled();
  });
});

// ===========================================================================
// #96 — shared ingestPaths dispatch (exercised via the drop handler) + the
// multi-select browse picker. ingestPaths is an internal helper; it is tested
// through its two real entry points (drop + browse) — the same surface the app
// uses — asserting on addFileSource / addSourceLocal / ingest / showToast /
// onclose, which fully pins its behavior.
// ===========================================================================

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
    // Clean batch (all added) → no summary toast.
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
    // Only the two successes were inserted + ingested.
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
    // Only the non-duplicate was inserted + ingested.
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
    // The picker must be opened in multi-select mode.
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
