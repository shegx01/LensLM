// Component tests for AssistantMessage: inline [n] markers stripped from the
// rendered prose; footer chips present for persisted citations, absent when
// citations is null; copy passes the RAW content (with markers) while the
// displayed prose is stripped (deliberate divergence).

import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Source } from '$lib/sources/types.js';
import type { Citation, ChatMessage } from '$lib/chat/types.js';

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

import AssistantMessage from './AssistantMessage.svelte';

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

function makeAssistant(content: string, citations: Citation[] | null): ChatMessage {
  return {
    id: 'a1',
    notebook_id: 'nb-001',
    turn_id: 't1',
    role: 'assistant',
    content,
    citations: citations === null ? null : JSON.stringify(citations),
    feedback: null,
    tokens_used: 10,
    created_at: '2026-07-12T00:00:00Z'
  };
}

const baseProps = {
  notebookId: 'nb-001',
  oncopy: vi.fn(),
  onregenerate: vi.fn(),
  onfeedback: vi.fn()
};

beforeEach(() => {
  vi.clearAllMocks();
  mockSourcesStore._setSources([]);
});

afterEach(() => {
  mockSourcesStore._setSources([]);
});

describe('AssistantMessage — marker strip + chips', () => {
  it('strips inline [n] markers from the rendered prose', () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-a', title: 'Alpha' })]);
    const msg = makeAssistant('Revenue grew 34%[1]. Solid.', [
      { source_id: 'src-a', ordinal: 1, locators: [] }
    ]);
    const { container } = render(AssistantMessage, { versions: [msg], ...baseProps });
    const prose = container.querySelector('.chat-markdown') as HTMLElement;
    expect(prose.textContent).toContain('Revenue grew 34%. Solid.');
    expect(prose.textContent).not.toContain('[1]');
  });

  it('renders footer chips for a persisted answer with citations', () => {
    mockSourcesStore._setSources([
      makeSource({ id: 'src-a', title: 'Alpha' }),
      makeSource({ id: 'src-b', title: 'Beta' })
    ]);
    const msg = makeAssistant('See [1] and [2].', [
      { source_id: 'src-a', ordinal: 1, locators: [] },
      { source_id: 'src-b', ordinal: 2, locators: [] }
    ]);
    render(AssistantMessage, { versions: [msg], ...baseProps });
    expect(screen.getByRole('button', { name: 'Source 1: Alpha' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Source 2: Beta' })).toBeInTheDocument();
  });

  it('renders no footer row when citations is null (streaming/zero-citation)', () => {
    const msg = makeAssistant('Just prose, no citations.', null);
    render(AssistantMessage, { versions: [msg], ...baseProps });
    expect(screen.queryByLabelText(/sources cited/i)).not.toBeInTheDocument();
  });

  it('copy passes the RAW content (with markers) while the display is stripped', async () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-a', title: 'Alpha' })]);
    const oncopy = vi.fn();
    const raw = 'Grew 34%[1].';
    const msg = makeAssistant(raw, [{ source_id: 'src-a', ordinal: 1, locators: [] }]);
    const { container } = render(AssistantMessage, { versions: [msg], ...baseProps, oncopy });

    const prose = container.querySelector('.chat-markdown') as HTMLElement;
    expect(prose.textContent).not.toContain('[1]');

    await fireEvent.click(screen.getByLabelText('Copy answer'));
    expect(oncopy).toHaveBeenCalledWith(raw);
  });
});
