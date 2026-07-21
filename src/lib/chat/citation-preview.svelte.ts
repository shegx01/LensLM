// Bridges inline citation chips (built OUTSIDE Svelte, see citation-inline.ts) to a
// single mounted CitationSnippetPopover (issue #237). A chip publishes its DOM node
// + resolved target on hover/focus; the popover reacts via `customAnchor`. Mirrors
// the `focusSource`/`focusNonce` bridge in sources-state.svelte.ts.

import type { CitationTarget } from './citation-inline.js';

/** Grace period before a hover/focus-out actually closes the popover, so the
 * pointer can travel from the chip onto the panel (e.g. to click "View in source"). */
const HIDE_GRACE_MS = 180;

interface CitationPreviewRequest {
  anchor: HTMLElement;
  target: CitationTarget;
}

let request = $state<CitationPreviewRequest | null>(null);
let open = $state(false);
let hideTimer: ReturnType<typeof setTimeout> | undefined;

export const citationPreviewStore = {
  get request(): CitationPreviewRequest | null {
    return request;
  },
  get open(): boolean {
    return open;
  }
};

/** Shows the preview anchored to `anchor` (the chip's own button element). */
export function showCitationPreview(anchor: HTMLElement, target: CitationTarget): void {
  clearTimeout(hideTimer);
  request = { anchor, target };
  open = true;
}

/** Starts the grace-period close; cancel with `cancelHideCitationPreview`. */
export function scheduleHideCitationPreview(): void {
  clearTimeout(hideTimer);
  hideTimer = setTimeout(() => {
    open = false;
  }, HIDE_GRACE_MS);
}

/** Cancels a pending grace-period close (pointer/focus entered the panel). */
export function cancelHideCitationPreview(): void {
  clearTimeout(hideTimer);
}

/** Closes immediately — no grace period (e.g. Escape, or "View in source" clicked). */
export function hideCitationPreviewNow(): void {
  clearTimeout(hideTimer);
  open = false;
}

/** Test-only: resets module state so cases don't bleed into each other. */
export function resetCitationPreview(): void {
  clearTimeout(hideTimer);
  hideTimer = undefined;
  request = null;
  open = false;
}
