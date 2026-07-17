// Typed IPC wrappers for the chat Tauri commands. Guarded with `isTauri()`.
//
// Two-frame terminator (Risk R9): see chat-state.svelte.ts header for the full
// persistence contract — this module only relays frames, it decides nothing.

import { Channel, invoke, isTauri } from '@tauri-apps/api/core';
import type { StreamEvent } from '$lib/sources/types.js';
import type { AnswerEvent, ChatMessage, ChatFeedback, ChatState, Citation } from './types.js';

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
  turnId: string,
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
  await invoke<void>('ask_notebook', { notebookId, turnId, question, onAnswer: channel });
}

/** Cancels the in-flight grounded answer for a notebook. Returns `false` outside Tauri. */
export async function cancelAsk(notebookId: string): Promise<boolean> {
  if (!isTauri()) return false;
  return invoke<boolean>('cancel_ask', { notebookId });
}

/** `turnId` is minted by the frontend. */
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

/**
 * Persists a terminal-state marker for a cancelled/errored turn (Plan 2 / PC-1).
 * `content` may carry the partial answer streamed so far. Throws outside Tauri (the
 * caller only invokes it after a real stream).
 */
export async function saveChatMarker(
  notebookId: string,
  turnId: string,
  content: string,
  state: Exclude<ChatState, null>,
  errorKind: string | null
): Promise<ChatMessage> {
  if (!isTauri()) throw new Error('saveChatMarker: not running under Tauri');
  return invoke<ChatMessage>('save_chat_marker', {
    notebookId,
    turnId,
    content,
    state,
    errorKind
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
