// Reactive "which model is the active chat/notes/audio-overview pin?" signal (Svelte 5
// runes, module singleton) — mirrors chat-provider.svelte.ts's shape. Backed by
// list_active_model_candidates(); refreshed on the picker's mount and after every AI
// Model settings persist so the pin never goes stale mid-session.

import { EMPTY_ACTIVE_SELECTION, listActiveModelCandidates } from './catalog.js';
import type { ActiveModelSelection } from './types.js';

let selection = $state<ActiveModelSelection>(EMPTY_ACTIVE_SELECTION);

export const activeModelStore = {
  get active() {
    return selection.active;
  },
  get candidates() {
    return selection.candidates;
  }
};

/** Re-query the engine and update the signal. Never throws (resolves to empty on failure). */
export async function refreshActiveModel(): Promise<void> {
  selection = await listActiveModelCandidates();
}

/** Reset to the default. Call in `afterEach` to prevent cross-test bleed. */
export function resetActiveModel(): void {
  selection = EMPTY_ACTIVE_SELECTION;
}
