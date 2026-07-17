// Component tests for AssistantMessage: inline [n] markers become compact numbered
// chips at their cited spot (no footer); a live chip reveals its source, a removed
// source renders a disabled chip; streaming (citations null) shows none; copy passes
// the RAW content (markers intact).

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
    state: null,
    error_kind: null,
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

describe('AssistantMessage — inline citation chips', () => {
  it('replaces each [n] marker with a compact numbered chip at its spot', async () => {
    mockSourcesStore._setSources([
      makeSource({ id: 'src-a', title: 'Alpha' }),
      makeSource({ id: 'src-b', title: 'Beta' })
    ]);
    const msg = makeAssistant('See [1] and [2].', [
      { source_id: 'src-a', ordinal: 1, locators: [] },
      { source_id: 'src-b', ordinal: 2, locators: [] }
    ]);
    const { container } = render(AssistantMessage, { versions: [msg], ...baseProps });

    const chip1 = await screen.findByRole('button', { name: 'Source 1: Alpha' });
    const chip2 = await screen.findByRole('button', { name: 'Source 2: Beta' });
    expect(chip1).toHaveClass('citation-chip');
    expect(chip1.textContent).toBe('1');
    expect(chip2.textContent).toBe('2');

    const prose = container.querySelector('.chat-markdown') as HTMLElement;
    // Raw bracket markers are gone; the chip numbers remain in the flow.
    expect(prose.textContent).not.toContain('[1]');
    expect(prose.textContent).not.toContain('[2]');
  });

  it('reveals the source in the rail when a live chip is clicked', async () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-a', title: 'Alpha' })]);
    const msg = makeAssistant('Grew 34%[1].', [{ source_id: 'src-a', ordinal: 1, locators: [] }]);
    render(AssistantMessage, { versions: [msg], ...baseProps });

    await fireEvent.click(await screen.findByRole('button', { name: 'Source 1: Alpha' }));
    expect(focusSource).toHaveBeenCalledWith('src-a');
  });

  it('renders a disabled chip for a citation whose source was removed', async () => {
    mockSourcesStore._setSources([]); // src-a not present
    const msg = makeAssistant('Gone [1].', [{ source_id: 'src-a', ordinal: 1, locators: [] }]);
    render(AssistantMessage, { versions: [msg], ...baseProps });

    const chip = await screen.findByRole('button', {
      name: /Source 1: Removed source \(unavailable\)/
    });
    expect(chip).toBeDisabled();

    await fireEvent.click(chip);
    expect(focusSource).not.toHaveBeenCalled();
  });

  it('renders no chips when citations is null (streaming/zero-citation)', () => {
    const msg = makeAssistant('Just prose, no citations [1].', null);
    const { container } = render(AssistantMessage, { versions: [msg], ...baseProps });
    expect(container.querySelector('.citation-chip')).toBeNull();
  });

  it('copy passes the RAW content (markers intact)', async () => {
    mockSourcesStore._setSources([makeSource({ id: 'src-a', title: 'Alpha' })]);
    const oncopy = vi.fn();
    const raw = 'Grew 34%[1].';
    const msg = makeAssistant(raw, [{ source_id: 'src-a', ordinal: 1, locators: [] }]);
    render(AssistantMessage, { versions: [msg], ...baseProps, oncopy });

    await fireEvent.click(screen.getByLabelText('Copy answer'));
    expect(oncopy).toHaveBeenCalledWith(raw);
  });

  it('does not render Save for the streaming bubble (finalized=false)', () => {
    const msg = makeAssistant('partial content...', null);
    render(AssistantMessage, { versions: [msg], ...baseProps, finalized: false });
    expect(screen.queryByLabelText('Save to notes')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Remove from notes')).not.toBeInTheDocument();
  });

  it('renders Save for a finalized answer (default finalized=true)', () => {
    const msg = makeAssistant('a finished answer', null);
    render(AssistantMessage, { versions: [msg], ...baseProps });
    expect(screen.getByLabelText('Save to notes')).toBeInTheDocument();
  });
});
