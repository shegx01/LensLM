// Component tests for CitationChips: resolves titles from the sources store,
// orders by ordinal, falls back for stale sources, renders nothing when empty,
// and routes a live click through focusSource.

import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Source } from '$lib/sources/types.js';
import type { Citation } from '$lib/chat/types.js';

const { mockSourcesStore } = vi.hoisted(() => {
  let _sources: Source[] = [];
  const mockSourcesStore = {
    get sources() {
      return _sources;
    },
    _setSources(s: Source[]) {
      _sources = s;
    }
  };
  return { mockSourcesStore };
});

vi.mock('$lib/sources/sources-state.svelte.js', () => ({
  sourcesStore: mockSourcesStore,
  focusSource: vi.fn()
}));

import CitationChips from './CitationChips.svelte';
import { focusSource } from '$lib/sources/sources-state.svelte.js';

function makeSource(overrides?: Partial<Source>): Source {
  return {
    id: 'src-001',
    notebook_id: 'nb-001',
    kind: 'file',
    title: 'Doc.md',
    status: 'indexed',
    locator: '/docs/Doc.md',
    selected: 1,
    created_at: new Date().toISOString(),
    token_count: 100,
    content_hash: 'h',
    raw_content_hash: null,
    trashed_at: null,
    enrichment_status: null,
    enrichment_meta: null,
    force_js_render: 0,
    error_meta: null,
    ...overrides
  };
}

function cite(source_id: string, ordinal: number): Citation {
  return { source_id, ordinal, locators: [] };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockSourcesStore._setSources([]);
});

afterEach(() => {
  mockSourcesStore._setSources([]);
});

describe('CitationChips', () => {
  it('renders one chip per citation with the resolved source title', () => {
    mockSourcesStore._setSources([
      makeSource({ id: 'src-a', title: 'Alpha.pdf' }),
      makeSource({ id: 'src-b', title: 'Beta.md' })
    ]);
    render(CitationChips, { citations: [cite('src-a', 1), cite('src-b', 2)] });
    expect(screen.getByText('Alpha.pdf')).toBeInTheDocument();
    expect(screen.getByText('Beta.md')).toBeInTheDocument();
  });

  it('orders chips by ordinal ascending regardless of citation order', () => {
    mockSourcesStore._setSources([
      makeSource({ id: 'src-a', title: 'Alpha.pdf' }),
      makeSource({ id: 'src-b', title: 'Beta.md' })
    ]);
    render(CitationChips, { citations: [cite('src-b', 2), cite('src-a', 1)] });
    const btns = screen.getAllByRole('button');
    expect(btns[0]).toHaveAccessibleName('Source 1: Alpha.pdf');
    expect(btns[1]).toHaveAccessibleName('Source 2: Beta.md');
  });

  it('renders a disabled fallback chip for a stale (missing) source without renumbering siblings', () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-a', title: 'Alpha.pdf' })]);
    render(CitationChips, { citations: [cite('src-a', 1), cite('src-gone', 2)] });
    const live = screen.getByRole('button', { name: 'Source 1: Alpha.pdf' });
    const stale = screen.getByRole('button', { name: /source 2: removed source \(unavailable\)/i });
    expect(live).not.toBeDisabled();
    expect(stale).toBeDisabled();
  });

  it('renders nothing when there are no citations', () => {
    const { container } = render(CitationChips, { citations: [] });
    expect(container.querySelector('button')).toBeNull();
    expect(screen.queryByLabelText(/sources cited/i)).not.toBeInTheDocument();
  });

  it('clicking a live chip calls focusSource with its source_id', async () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-a', title: 'Alpha.pdf' })]);
    render(CitationChips, { citations: [cite('src-a', 1)] });
    await fireEvent.click(screen.getByRole('button', { name: 'Source 1: Alpha.pdf' }));
    expect(focusSource).toHaveBeenCalledWith('src-a');
  });
});
