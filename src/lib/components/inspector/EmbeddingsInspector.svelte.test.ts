// EmbeddingsInspector component tests (dev/QA Embeddings Inspector).
//
// Covers:
//   - renders the Dialog when notebookStore.inspectorOpen is true (and not otherwise)
//   - lists sources from sourcesStore in the left pane
//   - selecting a source calls listSourceChunks(sourceId, activeNotebookId)
//   - renders chunk text / section_path / block_type in the right pane
//   - header renders one badge per EmbeddingStats entry (incl. a 2-entry case)
//   - empty chunks → "No chunks found"
//   - empty stats → "Not yet embedded"
//   - loading indicator while the IPC promise is pending
//   - status dot reflects the source status
//
// Mocks the inspector ipc, the sources store, the notebooks store, and the
// Tauri core module so tests run without a native host. Pattern mirrors
// SourcesRail.svelte.test.ts.

import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Source } from '$lib/sources/types.js';
import type { InspectorResponse } from '$lib/inspector/types.js';

// ---------------------------------------------------------------------------
// Hoisted mocks
// ---------------------------------------------------------------------------

const { mockSourcesStore, mockNotebookStore } = vi.hoisted(() => {
  let _sources: Source[] = [];

  const mockSourcesStore = {
    get sources() {
      return _sources;
    },
    _setSources(s: Source[]) {
      _sources = s;
    }
  };

  let _inspectorOpen = true;
  let _activeNotebookId: string | null = 'nb-001';

  const mockNotebookStore = {
    get inspectorOpen() {
      return _inspectorOpen;
    },
    set inspectorOpen(v: boolean) {
      _inspectorOpen = v;
    },
    get activeNotebookId() {
      return _activeNotebookId;
    },
    _setInspectorOpen(v: boolean) {
      _inspectorOpen = v;
    },
    _setActiveNotebookId(v: string | null) {
      _activeNotebookId = v;
    }
  };

  return { mockSourcesStore, mockNotebookStore };
});

vi.mock('$lib/inspector/ipc.js', () => ({
  listSourceChunks: vi.fn()
}));

vi.mock('$lib/sources/sources-state.svelte.js', () => ({
  sourcesStore: mockSourcesStore
}));

vi.mock('$lib/notebooks/notebooks-state.svelte.js', () => ({
  notebookStore: mockNotebookStore
}));

vi.mock('$lib/notebooks/index.js', () => ({
  notebookStore: mockNotebookStore
}));

vi.mock('@tauri-apps/api/core', () => ({
  isTauri: () => false,
  invoke: vi.fn()
}));

// Import after mocks.
import EmbeddingsInspector from './EmbeddingsInspector.svelte';
import { listSourceChunks } from '$lib/inspector/ipc.js';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function makeSource(overrides?: Partial<Source>): Source {
  return {
    id: 'src-001',
    notebook_id: 'nb-001',
    kind: 'text',
    title: 'Market Analysis.md',
    status: 'indexed',
    locator: '/docs/Market Analysis.md',
    selected: 1,
    created_at: new Date().toISOString(),
    token_count: 2048,
    content_hash: 'abc123',
    raw_content_hash: null,
    trashed_at: null,
    enrichment_status: null,
    enrichment_meta: null,
    ...overrides
  };
}

function makeResponse(overrides?: Partial<InspectorResponse>): InspectorResponse {
  return {
    chunks: [
      {
        id: 'chunk-001',
        parent_id: null,
        kind: 'parent',
        level: 0,
        section_path: 'Introduction > Overview',
        text: 'This is the canonical chunk text body.',
        block_type: 'paragraph',
        char_start: 0,
        char_end: 38,
        source_anchor: null,
        embedding_text: 'Context: Introduction. This is the canonical chunk text body.'
      }
    ],
    stats: [{ model: 'nomic-embed-text-v1.5', dim: 768, status: 'active' }],
    ...overrides
  };
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
  mockSourcesStore._setSources([]);
  mockNotebookStore._setInspectorOpen(true);
  mockNotebookStore._setActiveNotebookId('nb-001');
  vi.mocked(listSourceChunks).mockResolvedValue(makeResponse());
});

afterEach(() => {
  mockSourcesStore._setSources([]);
  mockNotebookStore._setInspectorOpen(true);
  mockNotebookStore._setActiveNotebookId('nb-001');
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('EmbeddingsInspector — visibility', () => {
  it('renders the Dialog when inspectorOpen is true', () => {
    render(EmbeddingsInspector);
    expect(screen.getByRole('dialog', { name: /embeddings inspector/i })).toBeInTheDocument();
  });

  it('does NOT render the Dialog when inspectorOpen is false', () => {
    mockNotebookStore._setInspectorOpen(false);
    render(EmbeddingsInspector);
    expect(screen.queryByRole('dialog', { name: /embeddings inspector/i })).not.toBeInTheDocument();
  });
});

describe('EmbeddingsInspector — source list', () => {
  it('lists sources from the sources store', () => {
    mockSourcesStore._setSources([
      makeSource({ id: 'src-001', title: 'Doc A.md' }),
      makeSource({ id: 'src-002', title: 'Doc B.pdf' })
    ]);
    render(EmbeddingsInspector);
    expect(screen.getByText('Doc A.md')).toBeInTheDocument();
    expect(screen.getByText('Doc B.pdf')).toBeInTheDocument();
  });

  it('renders a status dot reflecting the source status (indexed → green)', () => {
    mockSourcesStore._setSources([makeSource({ status: 'indexed' })]);
    render(EmbeddingsInspector);
    // The Dialog content renders in a portal under document.body, not the
    // render container — query the whole document.
    expect(document.querySelector('span.bg-green-primary')).not.toBeNull();
  });

  it('renders a destructive status dot for an error source', () => {
    mockSourcesStore._setSources([makeSource({ status: 'error' })]);
    render(EmbeddingsInspector);
    expect(document.querySelector('span.bg-destructive')).not.toBeNull();
  });
});

describe('EmbeddingsInspector — selecting a source', () => {
  it('selecting a source calls listSourceChunks with sourceId + activeNotebookId', async () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-001', title: 'Doc A.md' })]);
    render(EmbeddingsInspector);
    await fireEvent.click(screen.getByRole('button', { name: /Doc A\.md/i }));
    expect(listSourceChunks).toHaveBeenCalledWith('src-001', 'nb-001');
  });

  it('renders chunk text, section_path and block_type after selection', async () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-001', title: 'Doc A.md' })]);
    render(EmbeddingsInspector);
    await fireEvent.click(screen.getByRole('button', { name: /Doc A\.md/i }));
    await waitFor(() =>
      expect(screen.getByText('This is the canonical chunk text body.')).toBeInTheDocument()
    );
    expect(screen.getByText(/Introduction > Overview/)).toBeInTheDocument();
    expect(screen.getByText('paragraph')).toBeInTheDocument();
  });
});

describe('EmbeddingsInspector — stats header', () => {
  it('renders one badge per stats entry (single)', async () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-001', title: 'Doc A.md' })]);
    render(EmbeddingsInspector);
    await fireEvent.click(screen.getByRole('button', { name: /Doc A\.md/i }));
    await waitFor(() => expect(screen.getByText(/nomic-embed-text-v1\.5/)).toBeInTheDocument());
    expect(screen.getByText(/768/)).toBeInTheDocument();
  });

  it('renders one badge per stats entry (two entries — both models appear)', async () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-001', title: 'Doc A.md' })]);
    vi.mocked(listSourceChunks).mockResolvedValue(
      makeResponse({
        stats: [
          { model: 'nomic-embed-text-v1.5', dim: 768, status: 'active' },
          { model: 'bge-small-en-v1.5', dim: 384, status: 'active' }
        ]
      })
    );
    render(EmbeddingsInspector);
    await fireEvent.click(screen.getByRole('button', { name: /Doc A\.md/i }));
    await waitFor(() => expect(screen.getByText(/nomic-embed-text-v1\.5/)).toBeInTheDocument());
    expect(screen.getByText(/bge-small-en-v1\.5/)).toBeInTheDocument();
  });

  it('shows "Not yet embedded" when stats is empty', async () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-001', title: 'Doc A.md' })]);
    vi.mocked(listSourceChunks).mockResolvedValue(makeResponse({ stats: [] }));
    render(EmbeddingsInspector);
    await fireEvent.click(screen.getByRole('button', { name: /Doc A\.md/i }));
    await waitFor(() => expect(screen.getByText(/not yet embedded/i)).toBeInTheDocument());
  });
});

describe('EmbeddingsInspector — chunk states', () => {
  it('shows "No chunks found" when chunks is empty', async () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-001', title: 'Doc A.md' })]);
    vi.mocked(listSourceChunks).mockResolvedValue(makeResponse({ chunks: [] }));
    render(EmbeddingsInspector);
    await fireEvent.click(screen.getByRole('button', { name: /Doc A\.md/i }));
    await waitFor(() => expect(screen.getByText(/no chunks found/i)).toBeInTheDocument());
  });

  it('shows a loading indicator while the IPC promise is pending', async () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-001', title: 'Doc A.md' })]);
    // Never-resolving promise keeps the component in the loading state.
    vi.mocked(listSourceChunks).mockReturnValue(new Promise<InspectorResponse>(() => {}));
    render(EmbeddingsInspector);
    await fireEvent.click(screen.getByRole('button', { name: /Doc A\.md/i }));
    await waitFor(() =>
      expect(screen.getByRole('status', { name: /loading/i })).toBeInTheDocument()
    );
  });
});
