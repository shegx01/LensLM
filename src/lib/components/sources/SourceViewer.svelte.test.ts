// SourceViewer component tests (issue #237, AC2).
//
// Drives the REAL $state-backed sources store (openSourceViewer/sourceViewerNonce
// + focusSource/focusNonce, mirroring SourcesRail.reveal.svelte.test.ts) so the
// viewer's open $effect is exercised through genuine Svelte reactivity, not a
// static pre-render snapshot — a mocked store's plain-object mutation would never
// re-fire a mounted $effect. Only IPC and `source-text.js` are mocked.
//
// Covers: opening + loading state, highlighting + scrolling the cited span into
// view, the truncated-document notice, the null-offset (whole-text, no highlight)
// path, the friendly "no longer available" message on a rejected load (never a
// raw error), Reveal-in-sources wiring back to focusSource without regressing it,
// and re-opening the same source+span re-firing via the nonce.

import { render, screen, waitFor, fireEvent } from '@testing-library/svelte';
import { tick } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { SourceView } from '$lib/chat/types.js';

vi.mock('@tauri-apps/api/core', () => ({
  isTauri: () => false,
  invoke: vi.fn(),
  Channel: vi.fn()
}));

const { mockLoadSourceView } = vi.hoisted(() => ({
  mockLoadSourceView: vi.fn()
}));

vi.mock('$lib/sources/source-text.js', () => ({
  loadSourceView: mockLoadSourceView
}));

// The REAL sources store — genuine $state, not a mock (mirrors reveal-in-rail's
// own test file for the same reason: mutating a plain-object mock post-render
// would never re-fire the component's mounted $effect).
import {
  sourcesStore,
  openSourceViewer,
  resetSourcesStore
} from '$lib/sources/sources-state.svelte.js';
import SourceViewer from './SourceViewer.svelte';

function view(overrides: Partial<SourceView> = {}): SourceView {
  return {
    before: 'lead-in text ',
    marked: 'the cited span',
    after: ' trailing text',
    title: 'Doc A',
    kind: 'pdf',
    truncated: false,
    ...overrides
  };
}

beforeEach(() => {
  mockLoadSourceView.mockReset();
  resetSourcesStore();
  Element.prototype.scrollIntoView = vi.fn();
});

afterEach(() => {
  resetSourcesStore();
});

describe('SourceViewer', () => {
  it('does not open a dialog before any viewer request', () => {
    render(SourceViewer);
    expect(screen.queryByText('Loading source…')).not.toBeInTheDocument();
  });

  it('opens and shows a loading state, then the highlighted span once loaded', async () => {
    let resolveLoad!: (v: SourceView) => void;
    mockLoadSourceView.mockReturnValue(
      new Promise<SourceView>((resolve) => {
        resolveLoad = resolve;
      })
    );
    render(SourceViewer);

    openSourceViewer('src-1', 10, 24);
    await tick();
    await waitFor(() => expect(screen.getByText('Loading source…')).toBeInTheDocument());
    expect(mockLoadSourceView).toHaveBeenCalledWith('src-1', 10, 24);

    resolveLoad(view());
    await waitFor(() => expect(screen.getByText('the cited span')).toBeInTheDocument());
    expect(screen.getByText('the cited span').tagName).toBe('MARK');
    expect(screen.getByText('Doc A')).toBeInTheDocument();
  });

  it('shows the truncated-document notice when the view was windowed', async () => {
    mockLoadSourceView.mockResolvedValue(view({ truncated: true }));
    render(SourceViewer);

    openSourceViewer('src-1', 10, 24);
    await tick();
    await waitFor(() => expect(screen.getByText(/showing the area around/i)).toBeInTheDocument());
  });

  it('renders the whole text unhighlighted when offsets are null (R4)', async () => {
    mockLoadSourceView.mockResolvedValue(
      view({ before: 'the whole document body', marked: '', after: '' })
    );
    render(SourceViewer);

    openSourceViewer('src-1');
    await tick();
    expect(mockLoadSourceView).toHaveBeenCalledWith('src-1', null, null);
    await waitFor(() => expect(screen.getByText(/the whole document body/)).toBeInTheDocument());
    expect(document.querySelector('mark')).toBeNull();
  });

  it('shows a friendly message (not a raw error) when the source is purged/unavailable', async () => {
    mockLoadSourceView.mockRejectedValue(new Error('citation source is unavailable'));
    render(SourceViewer);

    openSourceViewer('src-purged', 0, 5);
    await tick();
    await waitFor(() =>
      expect(screen.getByText('This source is no longer available.')).toBeInTheDocument()
    );
    expect(screen.queryByText(/citation source is unavailable/)).not.toBeInTheDocument();
  });

  it('"Reveal in sources" calls focusSource and closes the dialog', async () => {
    mockLoadSourceView.mockResolvedValue(view());
    render(SourceViewer);

    openSourceViewer('src-1', 10, 24);
    await tick();
    await waitFor(() => expect(screen.getByText('Doc A')).toBeInTheDocument());

    await fireEvent.click(screen.getByRole('button', { name: /reveal in sources/i }));
    expect(sourcesStore.focusedSourceId).toBe('src-1');
    await waitFor(() => expect(screen.queryByText('Doc A')).not.toBeInTheDocument());
  });

  it('re-opening the same source+span re-fires (nonce bump), not just a request-identity change', async () => {
    mockLoadSourceView.mockResolvedValue(view());
    render(SourceViewer);

    openSourceViewer('src-1', 10, 24);
    await tick();
    await waitFor(() => expect(mockLoadSourceView).toHaveBeenCalledTimes(1));

    openSourceViewer('src-1', 10, 24);
    await tick();
    await waitFor(() => expect(mockLoadSourceView).toHaveBeenCalledTimes(2));
  });
});
