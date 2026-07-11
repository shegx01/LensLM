// Typed IPC wrappers for the chat Tauri commands. Guarded with `isTauri()`.
//
// Two-frame terminator (highest-risk seam — see the plan's Risk R9): the outer
// `StreamEvent<AnswerEvent>` wraps `started|chunk|done|failed`; the INNER
// `AnswerEvent` (Stage/ThinkingDelta/TextDelta/Citations/Done) is delivered only
// inside `chunk` frames. `askNotebook` forwards each layer separately and does NOT
// treat the outer `done` as a finalize trigger — unlike `embeddings/ipc.ts`, which
// finalizes on outer `done`. The caller (chat-state store) decides what to persist;
// this module only relays frames.

import { Channel, invoke, isTauri } from '@tauri-apps/api/core';
import type { StreamEvent } from '$lib/sources/types.js';
import type { AnswerEvent, ChatMessage, ChatFeedback, Citation } from './types.js';

export interface AskNotebookHandlers {
  /** One `AnswerEvent` per `chunk` frame (Stage/ThinkingDelta/TextDelta/Citations/Done). */
  onEvent: (event: AnswerEvent) => void;
  /** Outer stream-closed terminator (`{"type":"done"}`). No-op for persistence. */
  onStreamDone: () => void;
  /** Outer `failed` frame — sanitized `{kind,message}` `LensError`. */
  onFailed: (error: { kind: string; message: string }) => void;
}

/**
 * Streams a grounded answer for `question` via `ask_notebook`. Resolves once the
 * IPC call itself returns (which happens after the stream is fully drained
 * server-side); the caller drives its own state machine off `handlers`, not off
 * this promise's resolution.
 */
export async function askNotebook(
  notebookId: string,
  question: string,
  handlers: AskNotebookHandlers
): Promise<void> {
  if (!isTauri()) return;
  const channel = new Channel<StreamEvent<AnswerEvent>>();
  channel.onmessage = (ev) => {
    if (ev.type === 'chunk') handlers.onEvent(ev.data);
    else if (ev.type === 'done') handlers.onStreamDone();
    else if (ev.type === 'failed') handlers.onFailed(ev.data);
  };
  await invoke<void>('ask_notebook', { notebookId, question, onAnswer: channel });
}

/** Cancels the in-flight grounded answer for a notebook. Returns `false` outside Tauri. */
export async function cancelAsk(notebookId: string): Promise<boolean> {
  if (!isTauri()) return false;
  return invoke<boolean>('cancel_ask', { notebookId });
}

/** Persists a user chat message on send. `turnId` is minted by the frontend. */
export async function saveChatUser(
  notebookId: string,
  turnId: string,
  content: string
): Promise<ChatMessage> {
  if (!isTauri()) throw new Error('saveChatUser: not running under Tauri');
  return invoke<ChatMessage>('save_chat_user', { notebookId, turnId, content });
}

/** Persists an assistant chat message on inner `AnswerEvent::Done`. */
export async function saveChatAssistant(
  notebookId: string,
  turnId: string,
  content: string,
  citations: Citation[] | null,
  tokensUsed: number
): Promise<ChatMessage> {
  if (!isTauri()) throw new Error('saveChatAssistant: not running under Tauri');
  return invoke<ChatMessage>('save_chat_assistant', {
    notebookId,
    turnId,
    content,
    citations,
    tokensUsed
  });
}

/** Sets or clears (`null`) feedback on a chat message. */
export async function setChatFeedback(messageId: string, feedback: ChatFeedback): Promise<void> {
  if (!isTauri()) return;
  return invoke<void>('set_chat_feedback', { messageId, feedback });
}

/** Lists a notebook's chat messages as flat rows in transcript order. Returns `[]` outside Tauri. */
export async function listChatMessages(notebookId: string): Promise<ChatMessage[]> {
  if (!isTauri()) return [];
  return invoke<ChatMessage[]>('list_chat_messages', { notebookId });
}
