// Cross-component seam for "jump to a note": the ⌘K palette pushes a requested
// note id here; NotesPane observes it and scrolls to that note. A store (not a
// pane-registered callback) so it stays testable without a mounted pane and
// avoids lifecycle coupling between the palette and the pane.

/** Monotonic ticket so requesting the SAME note id twice still fires an observer. */
let request = $state<{ noteId: string; nonce: number } | null>(null);
let nonce = 0;

/** Signals that a note should be scrolled into view (idempotent per distinct call). */
export function requestScrollTo(noteId: string): void {
  request = { noteId, nonce: ++nonce };
}

export const notesNav = {
  /** The pending scroll request, or `null`. Read inside an `$effect` to subscribe. */
  get request() {
    return request;
  }
};

/** Reset state. Call in `afterEach` to prevent cross-test bleed. */
export function resetNotesNav(): void {
  request = null;
  nonce = 0;
}
