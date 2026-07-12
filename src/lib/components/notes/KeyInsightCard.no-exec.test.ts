// @vitest-environment jsdom
//
// No-exec sanitization proof for note-body rendering (security invariant:
// model-generated markdown/code is display-only, never executed). happy-dom
// mis-reports DOMPurify behavior (strips <pre>, mis-flags URI sanitization),
// so this file runs under jsdom to match the real Chromium webview.

import { render } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import type { Note } from '$lib/notes/types.js';

import KeyInsightCard from './KeyInsightCard.svelte';

function makeNote(overrides?: Partial<Note>): Note {
  return {
    id: 'note-001',
    notebook_id: 'nb-1',
    origin: 'chat',
    content: 'safe content',
    citations: null,
    source_title: null,
    source_message_id: 'msg-001',
    created_at: '2026-07-12T00:00:00Z',
    updated_at: '2026-07-12T00:00:00Z',
    ...overrides
  };
}

describe('KeyInsightCard — no-exec sanitization', () => {
  it('never executes a <script> tag in note content', () => {
    const onExec = vi.fn();
    (globalThis as unknown as { __onExec?: () => void }).__onExec = onExec;
    const note = makeNote({ content: '<script>window.__onExec && window.__onExec()</script>' });

    const { container } = render(KeyInsightCard, { props: { note } });

    expect(container.querySelector('script')).toBeNull();
    expect(onExec).not.toHaveBeenCalled();
  });

  it('neutralizes an onerror image handler in note content', () => {
    const onExec = vi.fn();
    (globalThis as unknown as { __onExec?: () => void }).__onExec = onExec;
    const note = makeNote({
      content: '<img src="x" onerror="window.__onExec && window.__onExec()">'
    });

    const { container } = render(KeyInsightCard, { props: { note } });

    const img = container.querySelector('img');
    if (img) {
      expect(img.getAttribute('onerror')).toBeNull();
      img.dispatchEvent(new Event('error'));
    }
    expect(onExec).not.toHaveBeenCalled();
  });

  it('renders plain markdown content unaffected', () => {
    const note = makeNote({ content: '**bold insight**' });
    const { container } = render(KeyInsightCard, { props: { note } });
    expect(container.querySelector('strong')?.textContent).toBe('bold insight');
  });
});
