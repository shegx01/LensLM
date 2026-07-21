// CitationSnippetPopover component tests (issue #237). Covers: lazy-fetch on
// hover/focus (driven through the real citation-preview store — chips publish to
// it from outside Svelte), the marked span rendered inside a <mark>, ellipsis on
// truncation, one card per locator (a citation may cite multiple spans), the
// null-offset degradation (R4 — no fetch, still offers "view in source"), the
// "View in source" action wiring to openSourceViewer, and a friendly error state.

import { render, screen, waitFor, fireEvent } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { CitationTarget } from '$lib/chat/citation-inline.js';
import type { SnippetSegments } from '$lib/chat/types.js';

const { mockResolveSnippet, mockOpenSourceViewer } = vi.hoisted(() => ({
  mockResolveSnippet: vi.fn(),
  mockOpenSourceViewer: vi.fn()
}));

vi.mock('$lib/sources/source-text.js', () => ({
  resolveCitationSnippet: mockResolveSnippet
}));

vi.mock('$lib/sources/sources-state.svelte.js', () => ({
  openSourceViewer: mockOpenSourceViewer
}));

import CitationSnippetPopover from './CitationSnippetPopover.svelte';
import { showCitationPreview, resetCitationPreview } from '$lib/chat/citation-preview.svelte.js';

function target(overrides: Partial<CitationTarget> = {}): CitationTarget {
  return {
    source_id: 'src-1',
    title: 'Doc A',
    live: true,
    locators: [
      { chunk_id: 'c1', anchor: null, section_path: null, page: null, char_start: 10, char_end: 20 }
    ],
    ...overrides
  };
}

function segments(overrides: Partial<SnippetSegments> = {}): SnippetSegments {
  return {
    before: 'lead-in ',
    marked: 'the cited words',
    after: ' trailing',
    truncated_before: false,
    truncated_after: false,
    ...overrides
  };
}

beforeEach(() => {
  mockResolveSnippet.mockReset();
  mockOpenSourceViewer.mockReset();
  resetCitationPreview();
});

afterEach(() => {
  resetCitationPreview();
});

describe('CitationSnippetPopover', () => {
  it('renders nothing until a chip has published a preview request', () => {
    render(CitationSnippetPopover);
    expect(screen.queryByText('Doc A')).not.toBeInTheDocument();
  });

  it('fetches and renders the snippet with the marked span highlighted', async () => {
    mockResolveSnippet.mockResolvedValue(segments());
    render(CitationSnippetPopover);
    const anchor = document.createElement('button');
    document.body.append(anchor);

    showCitationPreview(anchor, target());

    await waitFor(() => expect(screen.getByText('Doc A')).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText('the cited words')).toBeInTheDocument());
    expect(mockResolveSnippet).toHaveBeenCalledWith('src-1', 10, 20);
    expect(screen.getByText('the cited words').tagName).toBe('MARK');

    anchor.remove();
  });

  it('keeps the marked span visible when `before` is a large leading window (regression, #237)', async () => {
    // The engine's WINDOW snap always returns a large leading `before` (~240+
    // chars once a citation lands past the doc start). A whole-paragraph
    // line-clamp used to push the highlighted span below the visible lines;
    // the fix trims `before`/`after` for display so the mark is guaranteed
    // to render — with an ellipsis in front, but never hidden.
    const longBefore = 'x'.repeat(500);
    mockResolveSnippet.mockResolvedValue(
      segments({ before: longBefore, marked: 'the cited words', after: 'y'.repeat(500) })
    );
    render(CitationSnippetPopover);
    showCitationPreview(document.createElement('button'), target());

    await waitFor(() => expect(screen.getByText('the cited words')).toBeInTheDocument());
    const mark = screen.getByText('the cited words');
    expect(mark.tagName).toBe('MARK');

    const excerpt = mark.closest('p');
    expect(excerpt).not.toBeNull();
    expect(excerpt?.textContent).toContain('the cited words');
    // The visible text before the mark must be far shorter than the full
    // 500-char `before` — otherwise the mark would again be pushed off-screen.
    const beforeMark = excerpt!.textContent!.split('the cited words')[0];
    expect(beforeMark.length).toBeLessThan(150);
    expect(beforeMark.startsWith('…')).toBe(true);
  });

  it('trims before/after by code point so an astral emoji at the cut boundary is not split (regression, #237)', async () => {
    // '\u{1F600}' (😀) is a surrogate pair in UTF-16; str.slice(-N) can land
    // mid-pair and corrupt it. Placing several right at the trim boundary
    // catches a naive UTF-16 slice, which would render U+FFFD (�).
    const emoji = '\u{1F600}'.repeat(60);
    mockResolveSnippet.mockResolvedValue(
      segments({ before: emoji + 'tail-text', marked: 'm', after: 'head-text' + emoji })
    );
    render(CitationSnippetPopover);
    showCitationPreview(document.createElement('button'), target());

    await waitFor(() => expect(screen.getByText('m', { selector: 'mark' })).toBeInTheDocument());
    const excerpt = screen.getByText('m', { selector: 'mark' }).closest('p');
    expect(excerpt?.textContent).not.toContain('�');
  });

  it('shows an ellipsis on each truncated side of the snippet', async () => {
    mockResolveSnippet.mockResolvedValue(
      segments({
        before: 'lead',
        marked: 'span',
        after: 'trail',
        truncated_before: true,
        truncated_after: true
      })
    );
    render(CitationSnippetPopover);
    showCitationPreview(document.createElement('button'), target());

    await waitFor(() => expect(screen.getByText('span')).toBeInTheDocument());
    const excerpt = screen.getByText('span').closest('p');
    expect(excerpt?.textContent).toBe('…leadspantrail…');
  });

  it('lists one card per locator — a citation may cite several spans', async () => {
    mockResolveSnippet.mockResolvedValue(segments({ before: '', marked: 'x', after: '' }));
    render(CitationSnippetPopover);
    showCitationPreview(
      document.createElement('button'),
      target({
        locators: [
          { chunk_id: 'c1', anchor: null, section_path: null, page: 2, char_start: 0, char_end: 1 },
          { chunk_id: 'c2', anchor: null, section_path: null, page: 5, char_start: 2, char_end: 3 }
        ]
      })
    );

    await waitFor(() => expect(mockResolveSnippet).toHaveBeenCalledTimes(2));
    expect(screen.getByText('Page 2')).toBeInTheDocument();
    expect(screen.getByText('Page 5')).toBeInTheDocument();
    expect(screen.getAllByText('View in source')).toHaveLength(2);
  });

  it('degrades a null-offset locator (R4): no fetch, still offers "view in source"', async () => {
    render(CitationSnippetPopover);
    showCitationPreview(
      document.createElement('button'),
      target({
        locators: [
          {
            chunk_id: 'c1',
            anchor: null,
            section_path: null,
            page: null,
            char_start: null,
            char_end: null
          }
        ]
      })
    );

    await waitFor(() =>
      expect(screen.getByText('No excerpt available for this reference.')).toBeInTheDocument()
    );
    expect(mockResolveSnippet).not.toHaveBeenCalled();
    expect(screen.getByText('View in source')).toBeInTheDocument();
  });

  it('"View in source" opens the viewer with that locator\'s span', async () => {
    mockResolveSnippet.mockResolvedValue(segments({ before: '', marked: 'x', after: '' }));
    render(CitationSnippetPopover);
    showCitationPreview(document.createElement('button'), target());

    await waitFor(() => expect(screen.getByText('View in source')).toBeInTheDocument());
    await fireEvent.click(screen.getByText('View in source'));

    expect(mockOpenSourceViewer).toHaveBeenCalledWith('src-1', 10, 20);
  });

  it('shows a friendly message when the snippet fetch fails', async () => {
    mockResolveSnippet.mockRejectedValue(new Error('boom'));
    render(CitationSnippetPopover);
    showCitationPreview(document.createElement('button'), target());

    await waitFor(() =>
      expect(screen.getByText("Couldn't load this excerpt.")).toBeInTheDocument()
    );
  });
});
