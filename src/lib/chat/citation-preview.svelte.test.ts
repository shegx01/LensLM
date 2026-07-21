// citation-preview store bridge (issue #237): chips are built OUTSIDE Svelte
// (citation-inline.ts), so they publish {anchor, target} here; a single mounted
// CitationSnippetPopover reacts. Covers the show/grace-hide/cancel/reset contract.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  citationPreviewStore,
  showCitationPreview,
  scheduleHideCitationPreview,
  cancelHideCitationPreview,
  hideCitationPreviewNow,
  resetCitationPreview
} from './citation-preview.svelte.js';
import type { CitationTarget } from './citation-inline.js';

const target: CitationTarget = {
  source_id: 's1',
  title: 'Alpha',
  live: true,
  locators: [
    { chunk_id: 'c1', anchor: null, section_path: null, page: null, char_start: 0, char_end: 5 }
  ]
};

function anchor(): HTMLElement {
  return document.createElement('button');
}

beforeEach(() => {
  vi.useFakeTimers();
  resetCitationPreview();
});

afterEach(() => {
  resetCitationPreview();
  vi.useRealTimers();
});

describe('citationPreviewStore', () => {
  it('starts closed with no request', () => {
    expect(citationPreviewStore.open).toBe(false);
    expect(citationPreviewStore.request).toBeNull();
  });

  it('showCitationPreview opens immediately and publishes the anchor + target', () => {
    const el = anchor();
    showCitationPreview(el, target);

    expect(citationPreviewStore.open).toBe(true);
    expect(citationPreviewStore.request).toEqual({ anchor: el, target });
  });

  it('scheduleHideCitationPreview closes only after the grace period elapses', () => {
    showCitationPreview(anchor(), target);
    scheduleHideCitationPreview();

    expect(citationPreviewStore.open).toBe(true);
    vi.advanceTimersByTime(179);
    expect(citationPreviewStore.open).toBe(true);
    vi.advanceTimersByTime(1);
    expect(citationPreviewStore.open).toBe(false);
  });

  it('cancelHideCitationPreview aborts a pending grace-period close', () => {
    showCitationPreview(anchor(), target);
    scheduleHideCitationPreview();
    cancelHideCitationPreview();

    vi.advanceTimersByTime(1000);
    expect(citationPreviewStore.open).toBe(true);
  });

  it('hideCitationPreviewNow closes immediately, bypassing the grace period', () => {
    showCitationPreview(anchor(), target);
    hideCitationPreviewNow();

    expect(citationPreviewStore.open).toBe(false);
  });

  it('re-showing cancels a pending hide from a prior chip', () => {
    showCitationPreview(anchor(), target);
    scheduleHideCitationPreview();

    const el2 = anchor();
    showCitationPreview(el2, { ...target, source_id: 's2' });

    vi.advanceTimersByTime(1000);
    expect(citationPreviewStore.open).toBe(true);
    expect(citationPreviewStore.request?.anchor).toBe(el2);
  });

  it('resetCitationPreview clears request/open and cancels pending timers', () => {
    showCitationPreview(anchor(), target);
    scheduleHideCitationPreview();

    resetCitationPreview();

    expect(citationPreviewStore.open).toBe(false);
    expect(citationPreviewStore.request).toBeNull();
    vi.advanceTimersByTime(1000);
    expect(citationPreviewStore.open).toBe(false);
  });
});
