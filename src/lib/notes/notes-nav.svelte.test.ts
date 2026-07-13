// Unit tests for the notes-nav "jump to a note" seam (no pane mount required —
// this is why a store was chosen over a pane-registered callback).

import { afterEach, describe, expect, it } from 'vitest';
import { notesNav, requestScrollTo, resetNotesNav } from './notes-nav.svelte.js';

afterEach(() => {
  resetNotesNav();
});

describe('notes-nav', () => {
  it('starts with no pending request', () => {
    expect(notesNav.request).toBeNull();
  });

  it('requestScrollTo publishes the note id', () => {
    requestScrollTo('note-xyz');
    expect(notesNav.request?.noteId).toBe('note-xyz');
  });

  it('bumps the nonce so repeating the same note id still fires', () => {
    requestScrollTo('note-1');
    const first = notesNav.request?.nonce;
    requestScrollTo('note-1');
    const second = notesNav.request?.nonce;
    expect(first).toBeDefined();
    expect(second).toBeDefined();
    expect(second).not.toBe(first);
    expect(notesNav.request?.noteId).toBe('note-1');
  });

  it('resetNotesNav clears the pending request', () => {
    requestScrollTo('note-1');
    resetNotesNav();
    expect(notesNav.request).toBeNull();
  });
});
