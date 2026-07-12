// Shared ChatMessage/Turn test fixtures.

import type { ChatMessage, Turn } from './types.js';

/** Minimal valid ChatMessage (user role by default), all required fields. */
export function makeChatMessage(overrides?: Partial<ChatMessage>): ChatMessage {
  return {
    id: 'msg-001',
    notebook_id: 'nb-001',
    turn_id: 'turn-001',
    role: 'user',
    content: 'hello',
    citations: null,
    feedback: null,
    tokens_used: null,
    created_at: '2026-07-12T00:00:00Z',
    ...overrides
  };
}

/** A single Turn wrapping `user` plus its assistant `versions`. */
export function makeTurn(overrides?: Partial<Turn>): Turn {
  return {
    turn_id: 'turn-001',
    user: makeChatMessage(),
    versions: [],
    ...overrides
  };
}
