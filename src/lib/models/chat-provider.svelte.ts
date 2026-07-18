// Reactive "is a chat provider usable?" signal (Svelte 5 runes, module singleton).
//
// Backed by the read-only `has_chat_provider` command via `hasChatProvider()`, which
// mirrors the engine's `usable` gate — the only safe signal for AC-11 (a
// present-but-unusable models[] entry must NOT enable Send). Refreshed on notebook
// mount and after each AI Model settings persist.

import { hasChatProvider } from './catalog.js';

// Default false so Send is never wrongly enabled before the first check resolves.
let available = $state(false);

export const chatProviderStore = {
  get available() {
    return available;
  }
};

/** Re-query the engine and update the signal. Never throws (resolves false on failure). */
export async function refreshChatProvider(): Promise<void> {
  available = await hasChatProvider();
}

/** Reset to the default. Call in `afterEach` to prevent cross-test bleed. */
export function resetChatProvider(): void {
  available = false;
}
