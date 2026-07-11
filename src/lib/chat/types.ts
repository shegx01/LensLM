// SYNC-CHECK: must match lens-core/src/answer.rs AnswerEvent / lens-core/src/citation.rs
// Citation/Locator / lens-core/src/chat.rs ChatMessage — update all together.

export type { StreamEvent } from '$lib/sources/types.js';

/** Coarse pipeline phase. Mirrors `lens-core/src/answer.rs` `AnswerStage`. */
export type AnswerStage = 'Retrieving' | 'Thinking' | 'Answering';

// SYNC-CHECK: must match lens-core/src/citation.rs Locator struct.
export interface Locator {
  chunk_id: string;
  anchor: string | null;
  section_path: string | null;
  page: number | null;
  char_start: number | null;
  char_end: number | null;
}

// SYNC-CHECK: must match lens-core/src/citation.rs Citation struct.
export interface Citation {
  source_id: string;
  /** 1-based first-appearance rank among surviving citations. */
  ordinal: number;
  locators: Locator[];
}

/**
 * One event streamed inside a `StreamEvent<AnswerEvent>` `chunk` frame. Externally
 * tagged (serde default) by variant name — mirrors `lens-core/src/answer.rs`
 * `AnswerEvent`. NOT the same framing as the outer `StreamEvent`: see
 * `src/lib/chat/ipc.ts` for the two-layer terminator contract.
 */
export type AnswerEvent =
  | { Stage: AnswerStage }
  | { ThinkingDelta: string }
  | { TextDelta: string }
  | { Citations: Citation[] }
  | { Done: { tokens_used: number } };

/** `chat_messages.role`. Mirrors `lens-core/src/chat.rs` `ChatRole`. */
export type ChatRole = 'user' | 'assistant';

/** `chat_messages.feedback`. Mirrors `lens-core/src/chat.rs` `ChatFeedback`; `null` = no feedback. */
export type ChatFeedback = 'up' | 'down' | null;

/**
 * Wire shape returned by `save_chat_user` / `save_chat_assistant` / `list_chat_messages`.
 * NOTE the asymmetry: `citations` is a JSON-ENCODED string on this wire type — parse
 * it with `JSON.parse` (into `Citation[]`) when hydrating; `save_chat_assistant` TAKES
 * a typed `Citation[]` array as an argument, it is only the return/list shape that is
 * a raw string.
 */
export interface ChatMessage {
  id: string;
  notebook_id: string;
  turn_id: string;
  role: ChatRole;
  content: string;
  /** Raw JSON `Citation[]` (assistant rows only); `null` for user rows / no citations. */
  citations: string | null;
  feedback: ChatFeedback;
  tokens_used: number | null;
  created_at: string;
}

/**
 * Logical grouping (NOT a wire type): one user message plus its assistant
 * `versions` sharing `turn_id`, in insertion/creation order. `versions.length === 0`
 * is a legal state (a reloaded cancelled/errored turn — nothing was persisted).
 */
export interface Turn {
  turn_id: string;
  user: ChatMessage;
  versions: ChatMessage[];
}
