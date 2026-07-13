// @vitest-environment jsdom
//
// No-exec sanitization proof for the note editor's live PREVIEW (edited content is
// display-only, never executed). The preview goes through the same render seam as
// the rest of the app, so a `<script>`/`onerror` payload must render inert. Runs
// under jsdom (not happy-dom, which mis-reports DOMPurify) to match the real
// Chromium webview.

import { render, waitFor } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import type { Note } from '$lib/notes/types.js';

vi.mock('$lib/notes/notes-state.svelte.js', () => ({
  editNote: vi.fn().mockResolvedValue(undefined)
}));

import NoteEditor from './NoteEditor.svelte';

function makeNote(overrides?: Partial<Note>): Note {
  return {
    id: 'note-001',
    notebook_id: 'nb-1',
    origin: 'manual',
    content: 'safe content',
    citations: null,
    source_title: null,
    source_message_id: null,
    created_at: '2026-07-12T00:00:00Z',
    updated_at: '2026-07-12T00:00:00Z',
    pinned: false,
    ...overrides
  };
}

describe('NoteEditor preview — no-exec sanitization', () => {
  it('never executes a <script> tag in edited content', async () => {
    const onExec = vi.fn();
    (globalThis as unknown as { __onExec?: () => void }).__onExec = onExec;
    const note = makeNote({ content: '<script>window.__onExec && window.__onExec()</script>' });

    const { container } = render(NoteEditor, {
      props: { note, notebookId: 'nb-1', onclose: () => {} }
    });

    await waitFor(() => {
      // The CodeMirror editor host mounts; the preview is the sibling render target.
      expect(container.querySelector('.note-preview')).not.toBeNull();
    });
    // No live <script> node exists in the preview and nothing executed.
    expect(container.querySelector('.note-preview script')).toBeNull();
    expect(onExec).not.toHaveBeenCalled();
  });

  it('neutralizes an onerror image handler in edited content', async () => {
    const onExec = vi.fn();
    (globalThis as unknown as { __onExec?: () => void }).__onExec = onExec;
    const note = makeNote({
      content: '<img src="x" onerror="window.__onExec && window.__onExec()">'
    });

    const { container } = render(NoteEditor, {
      props: { note, notebookId: 'nb-1', onclose: () => {} }
    });

    await waitFor(() => expect(container.querySelector('.note-preview')).not.toBeNull());
    const img = container.querySelector('.note-preview img');
    if (img) {
      expect(img.getAttribute('onerror')).toBeNull();
      img.dispatchEvent(new Event('error'));
    }
    expect(onExec).not.toHaveBeenCalled();
  });

  it('renders plain markdown in the preview unaffected', async () => {
    const note = makeNote({ content: '**bold edit**' });
    const { container } = render(NoteEditor, {
      props: { note, notebookId: 'nb-1', onclose: () => {} }
    });
    await waitFor(() =>
      expect(container.querySelector('.note-preview strong')?.textContent).toBe('bold edit')
    );
  });
});
