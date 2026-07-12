// Shared citation JSON parsing. Used by chat-state.svelte.ts and
// notes-state.svelte.ts — keep behavior identical across both stores.

import type { Citation } from './types.js';

/** Parses a `citations` JSON string (`chat_messages`/`notes` column) into `Citation[]`, or `null`. */
export function parseCitations(json: string | null): Citation[] | null {
  if (json === null) return null;
  try {
    return JSON.parse(json) as Citation[];
  } catch (err) {
    console.warn('parseCitations: failed to parse citations JSON', err);
    return null;
  }
}
