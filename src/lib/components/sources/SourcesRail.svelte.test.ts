// SourcesRail.svelte.test.ts — Component tests for SourcesRail and AddSourcesModal.
// All IPC and Tauri modules are mocked so tests run without a native host.

import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { tick } from 'svelte';
import type { Source } from '$lib/sources/types.js';

const { mockSourcesStore, mockNotebookStore } = vi.hoisted(() => {
  let _sources: Source[] = [];
  let _recentlyTrashed = false;
  let _focusedSourceId: string | null = null;
  let _focusNonce = 0;

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
      return _recentlyTrashed;
    },
    get focusedSourceId() {
      return _focusedSourceId;
    },
    get focusNonce() {
      return _focusNonce;
    },
    _setSources(s: Source[]) {
      _sources = s;
    },
    _setRecentlyTrashed(v: boolean) {
      _recentlyTrashed = v;
    },
    _focus(id: string | null) {
      _focusedSourceId = id;
      _focusNonce += 1;
    },
    _resetFocus() {
      _focusedSourceId = null;
      _focusNonce = 0;
    }
  };

  let _rightRailCollapsed = false;

  const mockNotebookStore = {
    get activeNotebookId() {
      return 'nb-001';
    },
    get activeNotebook() {
      return { id: 'nb-001', title: 'iGaming Market Analysis' };
    },
    get rightRailCollapsed() {
      return _rightRailCollapsed;
    },
    set rightRailCollapsed(v: boolean) {
      _rightRailCollapsed = v;
    },
    _setRightRailCollapsed(v: boolean) {
      _rightRailCollapsed = v;
    }
  };

  return { mockSourcesStore, mockNotebookStore };
});

vi.mock('$lib/sources/sources-state.svelte.js', () => ({
  sourcesStore: mockSourcesStore,
  addSourceLocal: vi.fn(),
  loadSources: vi.fn().mockResolvedValue(undefined),
  ingest: vi.fn().mockResolvedValue(undefined),
  toggleSelected: vi.fn().mockResolvedValue(undefined),
  removeSource: vi.fn().mockResolvedValue(undefined),
  undoRemove: vi.fn().mockResolvedValue(undefined),
  retrySource: vi.fn().mockResolvedValue(undefined),
  resetSourcesStore: vi.fn(),
  disposeTrashTimers: vi.fn(),
  focusSource: vi.fn()
}));

vi.mock('$lib/sources/ipc.js', () => ({
  listSources: vi.fn().mockResolvedValue([]),
  addTextSource: vi
    .fn()
    .mockResolvedValue({ source: { id: 'src-new', status: 'pending' }, wasExisting: false }),
  addFileSource: vi
    .fn()
    .mockResolvedValue({ source: { id: 'src-new', status: 'pending' }, wasExisting: false }),
  addUrlSource: vi
    .fn()
    .mockResolvedValue({ source: { id: 'src-url', status: 'queued' }, wasExisting: false }),
  ingestSource: vi.fn().mockResolvedValue(undefined),
  setSourceSelected: vi.fn().mockResolvedValue(undefined),
  trashSource: vi.fn().mockResolvedValue(undefined),
  restoreSource: vi.fn().mockResolvedValue(undefined)
}));

vi.mock('$lib/notebooks/notebooks-state.svelte.js', () => ({
  notebookStore: mockNotebookStore
}));

vi.mock('@tauri-apps/api/core', () => ({
  isTauri: () => false,
  invoke: vi.fn()
}));

vi.mock('@tauri-apps/plugin-dialog', () => ({
  open: vi.fn().mockResolvedValue(null)
}));

import SourcesRail from './SourcesRail.svelte';
import AddSourcesModal from './AddSourcesModal.svelte';
import {
  removeSource,
  undoRemove,
  disposeTrashTimers,
  ingest
} from '$lib/sources/sources-state.svelte.js';
import { addUrlSource } from '$lib/sources/ipc.js';

function makeSource(overrides?: Partial<Source>): Source {
  return {
    id: 'src-001',
    notebook_id: 'nb-001',
    kind: 'file',
    title: 'Market Analysis Report.md',
    status: 'indexed',
    locator: '/docs/Market Analysis Report.md',
    selected: 1,
    created_at: new Date().toISOString(),
    token_count: 2048,
    content_hash: 'abc123',
    raw_content_hash: null,
    trashed_at: null,
    enrichment_status: null,
    enrichment_meta: null,
    force_js_render: 0,
    error_meta: null,
    ...overrides
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockSourcesStore._setSources([]);
  mockSourcesStore._setRecentlyTrashed(false);
  mockSourcesStore._resetFocus();
  mockNotebookStore._setRightRailCollapsed(false);
});

afterEach(() => {
  mockSourcesStore._setSources([]);
  mockSourcesStore._setRecentlyTrashed(false);
  mockSourcesStore._resetFocus();
  mockNotebookStore._setRightRailCollapsed(false);
});

describe('SourcesRail', () => {
  it('renders the "Sources" heading', () => {
    render(SourcesRail);
    expect(screen.getByText('Sources')).toBeInTheDocument();
  });

  it('header row has data-tauri-drag-region', () => {
    const { container } = render(SourcesRail);
    const dragRow = container.querySelector('[data-tauri-drag-region]') as HTMLElement;
    expect(dragRow).not.toBeNull();
    expect(dragRow.textContent).toContain('Sources');
  });

  it('renders the empty state when no sources', () => {
    render(SourcesRail);
    expect(screen.getByText('No sources yet')).toBeInTheDocument();
    expect(screen.getByText(/add a file or paste text/i)).toBeInTheDocument();
  });

  it('does NOT render the empty state when sources exist', () => {
    mockSourcesStore._setSources([makeSource()]);
    render(SourcesRail);
    expect(screen.queryByText('No sources yet')).not.toBeInTheDocument();
  });

  it('renders source title', () => {
    mockSourcesStore._setSources([makeSource({ title: 'My Research Doc.md' })]);
    render(SourcesRail);
    expect(screen.getByText('My Research Doc.md')).toBeInTheDocument();
  });

  it('renders type badge derived from extension', () => {
    mockSourcesStore._setSources([
      makeSource({ title: 'Report.pdf', locator: '/docs/Report.pdf', kind: 'file' })
    ]);
    render(SourcesRail);
    expect(screen.getByText('PDF')).toBeInTheDocument();
  });

  it('renders MD badge for .md files', () => {
    mockSourcesStore._setSources([makeSource({ locator: '/docs/Notes.md', title: 'Notes.md' })]);
    render(SourcesRail);
    expect(screen.getByText('MD')).toBeInTheDocument();
  });

  it('renders TXT badge for kind=text sources', () => {
    mockSourcesStore._setSources([makeSource({ kind: 'text', locator: '', title: 'My paste' })]);
    render(SourcesRail);
    expect(screen.getByText('TXT')).toBeInTheDocument();
  });

  it('renders URL badge for kind=url sources', () => {
    mockSourcesStore._setSources([
      makeSource({ kind: 'url', locator: 'https://example.com', title: 'Example' })
    ]);
    render(SourcesRail);
    expect(screen.getByText('URL')).toBeInTheDocument();
  });

  it('renders selected/total counter when sources exist', () => {
    mockSourcesStore._setSources([
      makeSource({ id: 'src-001', selected: 1 }),
      makeSource({ id: 'src-002', selected: 1 }),
      makeSource({ id: 'src-003', selected: 0 })
    ]);
    render(SourcesRail);
    expect(screen.getByText('2/3')).toBeInTheDocument();
  });

  it('renders all-selected counter correctly', () => {
    mockSourcesStore._setSources([
      makeSource({ id: 'src-001', selected: 1 }),
      makeSource({ id: 'src-002', selected: 1 }),
      makeSource({ id: 'src-003', selected: 1 })
    ]);
    render(SourcesRail);
    expect(screen.getByText('3/3')).toBeInTheDocument();
  });

  it('source checkbox has aria-pressed=true when selected=1', () => {
    mockSourcesStore._setSources([makeSource({ selected: 1, title: 'Doc.md' })]);
    render(SourcesRail);
    const checkbox = screen.getByRole('button', { name: /deselect source doc\.md/i });
    expect(checkbox).toHaveAttribute('aria-pressed', 'true');
  });

  it('source checkbox has aria-pressed=false when selected=0', () => {
    mockSourcesStore._setSources([makeSource({ selected: 0, title: 'Doc.md' })]);
    render(SourcesRail);
    const checkbox = screen.getByRole('button', { name: /select source doc\.md/i });
    expect(checkbox).toHaveAttribute('aria-pressed', 'false');
  });

  it('"Add source" header button has correct aria-label', () => {
    render(SourcesRail);
    // "Add source" is distinct from the empty-state "Add first source" button.
    expect(screen.getByRole('button', { name: 'Add source' })).toBeInTheDocument();
  });

  it('clicking header "Add source" button opens the modal', async () => {
    render(SourcesRail);
    const addBtn = screen.getByRole('button', { name: 'Add source' });
    await fireEvent.click(addBtn);
    expect(screen.getByRole('dialog', { name: /add sources/i })).toBeInTheDocument();
  });

  it('sources list has role=list and aria-label', () => {
    mockSourcesStore._setSources([makeSource()]);
    render(SourcesRail);
    expect(screen.getByRole('list', { name: /sources/i })).toBeInTheDocument();
  });

  it('status dot is present for an indexed source', () => {
    mockSourcesStore._setSources([makeSource({ status: 'indexed' })]);
    const { container } = render(SourcesRail);
    const dot = container.querySelector('span.bg-green-primary') as HTMLElement;
    expect(dot).not.toBeNull();
  });

  it('status dot has animate-pulse for parsing status', () => {
    mockSourcesStore._setSources([makeSource({ status: 'parsing' })]);
    const { container } = render(SourcesRail);
    const dot = container.querySelector('span.animate-pulse') as HTMLElement;
    expect(dot).not.toBeNull();
  });

  it('status dot has bg-destructive for error status', () => {
    mockSourcesStore._setSources([makeSource({ status: 'error' })]);
    const { container } = render(SourcesRail);
    const dot = container.querySelector('span.bg-destructive') as HTMLElement;
    expect(dot).not.toBeNull();
  });

  it('the source list lives in a hidden-scroll (no-scrollbar) container', () => {
    mockSourcesStore._setSources([makeSource()]);
    const { container } = render(SourcesRail);
    const scroll = container.querySelector('[data-sources-scroll]') as HTMLElement;
    expect(scroll).not.toBeNull();
    expect(scroll.className).toContain('no-scrollbar');
    expect(scroll.className).toContain('overflow-y-auto');
    expect(scroll.className).toContain('flex-1');
  });
});

describe('SourcesRail — delete button', () => {
  it('renders a delete button for each source row', () => {
    mockSourcesStore._setSources([
      makeSource({ id: 'src-001', title: 'Doc A.md' }),
      makeSource({ id: 'src-002', title: 'Doc B.pdf' })
    ]);
    render(SourcesRail);
    const deleteBtns = screen.getAllByRole('button', { name: /delete source/i });
    expect(deleteBtns).toHaveLength(2);
  });

  it('delete button has aria-label="Delete source"', () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-001', title: 'Doc.md' })]);
    render(SourcesRail);
    const deleteBtn = screen.getByRole('button', { name: 'Delete source' });
    expect(deleteBtn).toBeInTheDocument();
  });

  it('delete button carries -webkit-app-region: no-drag', () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-001' })]);
    const { container } = render(SourcesRail);
    const deleteBtn = container.querySelector('[data-delete-source-btn]') as HTMLElement;
    expect(deleteBtn).not.toBeNull();
    expect(deleteBtn.getAttribute('style') ?? '').toContain('-webkit-app-region: no-drag');
  });

  it('clicking the delete button calls removeSource with the source id', async () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-001', title: 'Doc.md' })]);
    render(SourcesRail);
    const deleteBtn = screen.getByRole('button', { name: 'Delete source' });
    await fireEvent.click(deleteBtn);
    expect(removeSource).toHaveBeenCalledWith('src-001');
  });

  it('delete button is NOT rendered in the empty state', () => {
    render(SourcesRail);
    expect(screen.queryByRole('button', { name: /delete source/i })).not.toBeInTheDocument();
  });
});

describe('SourcesRail — Undo bar', () => {
  it('is not visible when recentlyTrashed is false', () => {
    render(SourcesRail);
    expect(screen.queryByText('Source moved to trash')).not.toBeInTheDocument();
  });

  it('appears when recentlyTrashed is true', () => {
    mockSourcesStore._setRecentlyTrashed(true);
    render(SourcesRail);
    expect(screen.getByText('Source moved to trash')).toBeInTheDocument();
  });

  it('renders an "Undo" button when recentlyTrashed is true', () => {
    mockSourcesStore._setRecentlyTrashed(true);
    render(SourcesRail);
    expect(screen.getByRole('button', { name: /^undo$/i })).toBeInTheDocument();
  });

  it('clicking the Undo button calls undoRemove', async () => {
    mockSourcesStore._setRecentlyTrashed(true);
    render(SourcesRail);
    await fireEvent.click(screen.getByRole('button', { name: /^undo$/i }));
    expect(undoRemove).toHaveBeenCalledOnce();
  });

  it('clicking the Undo button passes activeNotebookId to undoRemove (fix #3)', async () => {
    // activeNotebookId is 'nb-001' in the mock — undoRemove must receive it so
    // the canonical loadSources reconcile actually runs after restore.
    mockSourcesStore._setRecentlyTrashed(true);
    render(SourcesRail);
    await fireEvent.click(screen.getByRole('button', { name: /^undo$/i }));
    expect(undoRemove).toHaveBeenCalledWith('nb-001');
  });

  it('Undo bar has -webkit-app-region: no-drag on the outer element', () => {
    mockSourcesStore._setRecentlyTrashed(true);
    const { container } = render(SourcesRail);
    const bar = container.querySelector('[role="status"]') as HTMLElement;
    expect(bar).not.toBeNull();
    expect(bar.getAttribute('style') ?? '').toContain('-webkit-app-region: no-drag');
  });

  it('Undo bar is not visible in the collapsed state', () => {
    mockSourcesStore._setRecentlyTrashed(true);
    mockNotebookStore._setRightRailCollapsed(true);
    render(SourcesRail);
    expect(screen.queryByText('Source moved to trash')).not.toBeInTheDocument();
  });
});

describe('SourcesRail — onDestroy disposeTrashTimers wiring', () => {
  it('disposeTrashTimers is called when the component is unmounted', () => {
    const { unmount } = render(SourcesRail);
    expect(disposeTrashTimers).not.toHaveBeenCalled();
    unmount();
    expect(disposeTrashTimers).toHaveBeenCalledOnce();
  });
});

describe('SourcesRail — collapse toggle', () => {
  it('renders the "Collapse sources" toggle in the expanded header', () => {
    render(SourcesRail);
    expect(screen.getByRole('button', { name: /collapse sources/i })).toBeInTheDocument();
  });

  it('clicking the toggle flips rightRailCollapsed to true', async () => {
    render(SourcesRail);
    expect(mockNotebookStore.rightRailCollapsed).toBe(false);
    await fireEvent.click(screen.getByRole('button', { name: /collapse sources/i }));
    expect(mockNotebookStore.rightRailCollapsed).toBe(true);
  });

  it('renders the collapsed icon strip (Expand affordance) when collapsed', () => {
    mockNotebookStore._setRightRailCollapsed(true);
    render(SourcesRail);
    expect(screen.getByRole('button', { name: /expand sources/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /collapse sources/i })).not.toBeInTheDocument();
  });

  it('collapsed strip shows a Sources count badge and a Studio icon', () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-001' }), makeSource({ id: 'src-002' })]);
    mockNotebookStore._setRightRailCollapsed(true);
    render(SourcesRail);
    expect(screen.getByRole('button', { name: /sources \(2\)/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^studio$/i })).toBeInTheDocument();
  });

  it('clicking the collapsed Expand button flips rightRailCollapsed back to false', async () => {
    mockNotebookStore._setRightRailCollapsed(true);
    render(SourcesRail);
    await fireEvent.click(screen.getByRole('button', { name: /expand sources/i }));
    expect(mockNotebookStore.rightRailCollapsed).toBe(false);
  });

  it('every collapsed-strip button carries -webkit-app-region: no-drag', () => {
    mockNotebookStore._setRightRailCollapsed(true);
    const { container } = render(SourcesRail);
    const buttons = Array.from(container.querySelectorAll('button')) as HTMLElement[];
    expect(buttons.length).toBeGreaterThan(0);
    for (const btn of buttons) {
      expect(btn.getAttribute('style') ?? '').toContain('-webkit-app-region: no-drag');
    }
  });
});

describe('SourcesRail — Studio shell', () => {
  it('renders the Studio header with a RESEARCH tag', () => {
    render(SourcesRail);
    expect(screen.getByText('Studio')).toBeInTheDocument();
    expect(screen.getByText('Research')).toBeInTheDocument();
  });

  it('renders the Audio Overview card with the selected/total line', () => {
    mockSourcesStore._setSources([
      makeSource({ id: 'src-001', selected: 1 }),
      makeSource({ id: 'src-002', selected: 0 })
    ]);
    render(SourcesRail);
    expect(screen.getByText('Audio Overview')).toBeInTheDocument();
    expect(screen.getByText(/1 of 2 sources selected/i)).toBeInTheDocument();
  });

  it('the Generate Audio Overview button is disabled', () => {
    render(SourcesRail);
    const gen = screen.getByRole('button', { name: /generate audio overview/i });
    expect(gen).toBeDisabled();
  });

  it('renders the study-tool actions and they are all disabled', () => {
    render(SourcesRail);
    for (const label of ['Study Guide', 'Briefing Doc', 'Report', 'Slide Deck', 'Flashcards']) {
      const btn = screen.getByRole('button', { name: new RegExp(label, 'i') });
      expect(btn).toBeDisabled();
    }
  });

  it('the Studio section is NOT rendered when the rail is collapsed', () => {
    mockNotebookStore._setRightRailCollapsed(true);
    render(SourcesRail);
    expect(screen.queryByText('Audio Overview')).not.toBeInTheDocument();
  });
});

describe('SourcesRail — data-source-id anchors', () => {
  it('tags each source <li> with its data-source-id', () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-001' }), makeSource({ id: 'src-002' })]);
    const { container } = render(SourcesRail);
    expect(container.querySelector('[data-source-id="src-001"]')).not.toBeNull();
    expect(container.querySelector('[data-source-id="src-002"]')).not.toBeNull();
  });
});

describe('SourcesRail — reveal-in-rail (focusSource)', () => {
  it('does NOT scroll on mount when focusedSourceId is null', async () => {
    const scrollSpy = vi
      .spyOn(HTMLElement.prototype, 'scrollIntoView')
      .mockImplementation(() => {});
    mockSourcesStore._setSources([makeSource({ id: 'src-001' })]);

    render(SourcesRail);
    await tick();

    expect(scrollSpy).not.toHaveBeenCalled();
    scrollSpy.mockRestore();
  });

  it('scrolls the focused row into view and applies the pulse ring', async () => {
    const scrollSpy = vi
      .spyOn(HTMLElement.prototype, 'scrollIntoView')
      .mockImplementation(() => {});
    mockSourcesStore._setSources([makeSource({ id: 'src-001' }), makeSource({ id: 'src-002' })]);
    // Set focus BEFORE render so the effect picks it up on its first (mount) run.
    mockSourcesStore._focus('src-002');

    const { container } = render(SourcesRail);
    await tick();
    await tick();

    expect(scrollSpy).toHaveBeenCalledWith({ block: 'nearest' });
    const li = container.querySelector('[data-source-id="src-002"]') as HTMLElement;
    expect(li.className).toContain('ring-2');
    scrollSpy.mockRestore();
  });

  it('expands a collapsed rail before revealing the source', async () => {
    vi.spyOn(HTMLElement.prototype, 'scrollIntoView').mockImplementation(() => {});
    mockSourcesStore._setSources([makeSource({ id: 'src-001' })]);
    mockNotebookStore._setRightRailCollapsed(true);
    mockSourcesStore._focus('src-001');

    render(SourcesRail);
    await tick();

    expect(mockNotebookStore.rightRailCollapsed).toBe(false);
  });
});

describe('SourcesRail — trash-in-session (AC5 ≡ AC6 invariant)', () => {
  it('a trashed cited source is absent from the rail (its <li> is gone)', () => {
    // The store is exactly the live set: a trashed source is removed from it, so
    // the rail has no <li> for it — matching a stale chip having no scroll target.
    mockSourcesStore._setSources([makeSource({ id: 'src-live' })]);
    const { container } = render(SourcesRail);
    expect(container.querySelector('[data-source-id="src-live"]')).not.toBeNull();
    expect(container.querySelector('[data-source-id="src-trashed"]')).toBeNull();
  });
});

describe('AddSourcesModal', () => {
  it('does NOT render when open=false', () => {
    render(AddSourcesModal, { open: false });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('renders when open=true', () => {
    render(AddSourcesModal, { open: true });
    expect(screen.getByRole('dialog', { name: /add sources/i })).toBeInTheDocument();
  });

  it('renders the modal title "Add sources"', () => {
    render(AddSourcesModal, { open: true });
    expect(screen.getByText('Add sources')).toBeInTheDocument();
  });

  it('renders the active notebook name as subtitle', () => {
    render(AddSourcesModal, { open: true });
    expect(screen.getByText('iGaming Market Analysis')).toBeInTheDocument();
  });

  it('renders all three tabs', () => {
    render(AddSourcesModal, { open: true });
    expect(screen.getByRole('tab', { name: /upload/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /url/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /paste text/i })).toBeInTheDocument();
  });

  it('Upload tab is active by default (aria-selected)', () => {
    render(AddSourcesModal, { open: true });
    expect(screen.getByRole('tab', { name: /upload/i })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: /url/i })).toHaveAttribute('aria-selected', 'false');
    expect(screen.getByRole('tab', { name: /paste text/i })).toHaveAttribute(
      'aria-selected',
      'false'
    );
  });

  it('Upload tab shows drop zone text', () => {
    render(AddSourcesModal, { open: true });
    expect(screen.getByText('Drop files here')).toBeInTheDocument();
    expect(screen.getByText('browse your computer')).toBeInTheDocument();
  });

  it('Upload tab shows only backend-supported format categories', () => {
    render(AddSourcesModal, { open: true });
    // Trimmed to the backend-accepted set (issue #95): DOCUMENTS + JSON only.
    expect(screen.getByText(/DOCUMENTS/)).toBeInTheDocument();
    expect(screen.getAllByText(/JSON/).length).toBeGreaterThan(0);
  });

  it('Upload tab no longer advertises unsupported formats', () => {
    render(AddSourcesModal, { open: true });
    // Regression guard (AC-7): no audio/video/Whisper/spreadsheet/presentation copy.
    expect(screen.queryByText(/AUDIO/)).not.toBeInTheDocument();
    expect(screen.queryByText(/VIDEO/)).not.toBeInTheDocument();
    expect(screen.queryByText(/whisper/i)).not.toBeInTheDocument();
  });

  it('clicking URL tab switches to URL panel', async () => {
    render(AddSourcesModal, { open: true });
    await fireEvent.click(screen.getByRole('tab', { name: /url/i }));
    expect(screen.getByRole('tab', { name: /url/i })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByLabelText(/web page url/i)).toBeInTheDocument();
  });

  it('URL tab no longer shows the deferral notice', async () => {
    render(AddSourcesModal, { open: true });
    await fireEvent.click(screen.getByRole('tab', { name: /url/i }));
    expect(screen.queryByText(/available in the next update/i)).not.toBeInTheDocument();
  });

  it('"Add to notebook →" is disabled on the URL tab when the field is empty', async () => {
    render(AddSourcesModal, { open: true });
    await fireEvent.click(screen.getByRole('tab', { name: /url/i }));
    const addBtn = screen.getByRole('button', { name: /add to notebook/i });
    expect(addBtn).toBeDisabled();
  });

  it('"Add to notebook →" is enabled once a valid URL is entered', async () => {
    render(AddSourcesModal, { open: true });
    await fireEvent.click(screen.getByRole('tab', { name: /url/i }));
    const input = screen.getByLabelText(/web page url/i);
    await fireEvent.input(input, { target: { value: 'https://example.com/article' } });
    const addBtn = screen.getByRole('button', { name: /add to notebook/i });
    expect(addBtn).not.toBeDisabled();
  });

  it('"Add to notebook →" stays disabled for an invalid URL', async () => {
    render(AddSourcesModal, { open: true });
    await fireEvent.click(screen.getByRole('tab', { name: /url/i }));
    const input = screen.getByLabelText(/web page url/i);
    await fireEvent.input(input, { target: { value: 'not a url' } });
    const addBtn = screen.getByRole('button', { name: /add to notebook/i });
    expect(addBtn).toBeDisabled();
  });

  it('submitting a URL calls addUrlSource then ingests', async () => {
    const onclose = vi.fn();
    render(AddSourcesModal, { open: true, onclose });
    await fireEvent.click(screen.getByRole('tab', { name: /url/i }));
    const input = screen.getByLabelText(/web page url/i);
    await fireEvent.input(input, { target: { value: 'https://example.com/article' } });
    await fireEvent.click(screen.getByRole('button', { name: /add to notebook/i }));
    expect(addUrlSource).toHaveBeenCalledWith(
      'nb-001',
      expect.any(String),
      'https://example.com/article',
      false
    );
    expect(ingest).toHaveBeenCalledWith('src-url');
    expect(onclose).toHaveBeenCalled();
  });

  it('checking the SPA checkbox then submitting calls addUrlSource with forceJsRender=true', async () => {
    const onclose = vi.fn();
    render(AddSourcesModal, { open: true, onclose });
    await fireEvent.click(screen.getByRole('tab', { name: /url/i }));
    const input = screen.getByLabelText(/web page url/i);
    await fireEvent.input(input, { target: { value: 'https://example.com/spa' } });
    await fireEvent.click(screen.getByLabelText(/needs javascript to load/i));
    await fireEvent.click(screen.getByRole('button', { name: /add to notebook/i }));
    expect(addUrlSource).toHaveBeenCalledWith(
      'nb-001',
      expect.any(String),
      'https://example.com/spa',
      true
    );
    expect(ingest).toHaveBeenCalledWith('src-url');
    expect(onclose).toHaveBeenCalled();
  });

  it('clicking Paste text tab switches to paste panel', async () => {
    render(AddSourcesModal, { open: true });
    await fireEvent.click(screen.getByRole('tab', { name: /paste text/i }));
    expect(screen.getByRole('tab', { name: /paste text/i })).toHaveAttribute(
      'aria-selected',
      'true'
    );
    expect(screen.getByLabelText(/content/i)).toBeInTheDocument();
  });

  it('Paste tab has TITLE and CONTENT fields', async () => {
    render(AddSourcesModal, { open: true });
    await fireEvent.click(screen.getByRole('tab', { name: /paste text/i }));
    expect(screen.getByLabelText(/title/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/content/i)).toBeInTheDocument();
  });

  it('"Add to notebook →" is disabled when paste content is empty', async () => {
    render(AddSourcesModal, { open: true });
    await fireEvent.click(screen.getByRole('tab', { name: /paste text/i }));
    const addBtn = screen.getByRole('button', { name: /add to notebook/i });
    expect(addBtn).toBeDisabled();
  });

  it('"Add to notebook →" is enabled when paste content is filled', async () => {
    render(AddSourcesModal, { open: true });
    await fireEvent.click(screen.getByRole('tab', { name: /paste text/i }));
    const textarea = screen.getByLabelText(/content/i);
    await fireEvent.input(textarea, { target: { value: 'Some text content here' } });
    const addBtn = screen.getByRole('button', { name: /add to notebook/i });
    expect(addBtn).not.toBeDisabled();
  });

  it('Cancel button calls onclose', async () => {
    const onclose = vi.fn();
    render(AddSourcesModal, { open: true, onclose });
    await fireEvent.click(screen.getByRole('button', { name: /cancel/i }));
    expect(onclose).toHaveBeenCalledOnce();
  });

  it('X close button calls onclose', async () => {
    const onclose = vi.fn();
    render(AddSourcesModal, { open: true, onclose });
    await fireEvent.click(screen.getByRole('button', { name: /close/i }));
    expect(onclose).toHaveBeenCalledOnce();
  });

  it('modal has aria-modal=true', () => {
    render(AddSourcesModal, { open: true });
    const dialog = screen.getByRole('dialog', { name: /add sources/i });
    expect(dialog).toHaveAttribute('aria-modal', 'true');
  });

  it('tablist has correct aria-label', () => {
    render(AddSourcesModal, { open: true });
    expect(screen.getByRole('tablist', { name: /source type/i })).toBeInTheDocument();
  });
});
