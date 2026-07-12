// Chat reactive store (Svelte 5 runes, module singleton). Mirrors
// notebooks-state.svelte.ts's shape (module-level $state + exported getters).
//
// Persistence contract (plan Step 6 / Risk R9 — the highest-risk seam):
// the INNER `AnswerEvent::Done{tokens_used}` delivered inside a `chunk` frame is
// the SOLE finalize+persist trigger — it is where `saveChatAssistant` is called,
// exactly once. The OUTER stream-closed terminator (`StreamEvent::Done`, surfaced
// here as `onStreamDone`) is a no-op for persistence; it must never re-trigger a
// save (that would double-persist or persist without tokens_used). A cancelled or
// errored turn persists nothing — only the user row (written on send) survives.

import {
  askNotebook,
  cancelAsk,
  saveChatUser,
  saveChatAssistant,
  setChatFeedback,
  listChatMessages
} from './ipc.js';
import type {
  AnswerEvent,
  AnswerStage,
  ChatMessage,
  ChatFeedback,
  Citation,
  Turn
} from './types.js';

interface NotebookChatState {
  turns: Turn[];
  streaming: boolean;
  stage: AnswerStage | null;
  thinkingBuffer: string;
  answerBuffer: string;
  pendingCitations: Citation[] | null;
  currentTurnId: string | null;
  error: { kind: string; message: string } | null;
  pinnedToBottom: boolean;
  streamGeneration: number;
}

function emptyNotebookState(): NotebookChatState {
  return {
    turns: [],
    streaming: false,
    stage: null,
    thinkingBuffer: '',
    answerBuffer: '',
    pendingCitations: null,
    currentTurnId: null,
    error: null,
    pinnedToBottom: true,
    streamGeneration: 0
  };
}

let byNotebook = $state<Record<string, NotebookChatState>>({});

function ensure(notebookId: string): NotebookChatState {
  let s = byNotebook[notebookId];
  if (!s) {
    s = emptyNotebookState();
    byNotebook[notebookId] = s;
  }
  return s;
}

// ---------------------------------------------------------------------------
// Grouping (Option A — plan :62-72): fold flat rows into ordered Turn[], each
// turn = one user row + its assistant versions (creation order), newest last.
// ---------------------------------------------------------------------------

function groupIntoTurns(rows: ChatMessage[]): Turn[] {
  const order: string[] = [];
  const users = new Map<string, ChatMessage>();
  const versions = new Map<string, ChatMessage[]>();

  for (const row of rows) {
    if (!order.includes(row.turn_id)) order.push(row.turn_id);
    if (row.role === 'user') {
      users.set(row.turn_id, row);
    } else {
      const list = versions.get(row.turn_id) ?? [];
      list.push(row);
      versions.set(row.turn_id, list);
    }
  }

  const turns: Turn[] = [];
  for (const turnId of order) {
    const user = users.get(turnId);
    if (!user) continue; // an assistant row with no user row is not a legal turn
    turns.push({ turn_id: turnId, user, versions: versions.get(turnId) ?? [] });
  }
  return turns;
}

function parseCitations(json: string | null): Citation[] | null {
  if (json === null) return null;
  try {
    return JSON.parse(json) as Citation[];
  } catch (err) {
    console.warn('parseCitations: failed to parse citations JSON', err);
    return null;
  }
}

/** Hydrates a notebook's transcript from `chat_messages` (AC17). */
export async function hydrate(notebookId: string): Promise<void> {
  const rows = await listChatMessages(notebookId);
  const state = ensure(notebookId);
  state.turns = groupIntoTurns(rows);
}

/** Newest assistant version of a turn, or `null` for a versions-empty (cancelled/errored) turn. */
export function latestVersion(turn: Turn): ChatMessage | null {
  return turn.versions.length > 0 ? turn.versions[turn.versions.length - 1] : null;
}

/** Parses a `ChatMessage.citations` JSON string into `Citation[]`, or `null`. */
export function messageCitations(message: ChatMessage): Citation[] | null {
  return parseCitations(message.citations);
}

function resetStreamBuffers(state: NotebookChatState): void {
  state.streaming = false;
  state.stage = null;
  state.thinkingBuffer = '';
  state.answerBuffer = '';
  state.pendingCitations = null;
  state.currentTurnId = null;
}

async function runStream(notebookId: string, turnId: string, question: string): Promise<void> {
  const state = ensure(notebookId);
  // Bump so late callbacks from a superseded/cancelled stream no-op.
  const gen = ++state.streamGeneration;
  state.streaming = true;
  state.stage = null;
  state.thinkingBuffer = '';
  state.answerBuffer = '';
  state.pendingCitations = null;
  state.currentTurnId = turnId;
  state.error = null;
  state.pinnedToBottom = true;

  let persisted = false;

  const persistOnce = async (tokensUsed: number): Promise<void> => {
    if (state.streamGeneration !== gen) return;
    if (persisted) return;
    persisted = true;
    const saved = await saveChatAssistant(
      notebookId,
      turnId,
      state.answerBuffer,
      state.pendingCitations,
      tokensUsed
    );
    if (state.streamGeneration !== gen) return;
    const turn = state.turns.find((t) => t.turn_id === turnId);
    if (turn) turn.versions.push(saved);
    state.streaming = false;
  };

  await askNotebook(notebookId, question, {
    onEvent: (event: AnswerEvent) => {
      if (state.streamGeneration !== gen) return;
      if ('Stage' in event) {
        state.stage = event.Stage;
      } else if ('ThinkingDelta' in event) {
        state.thinkingBuffer += event.ThinkingDelta;
      } else if ('TextDelta' in event) {
        state.answerBuffer += event.TextDelta;
      } else if ('Citations' in event) {
        state.pendingCitations = event.Citations;
      } else if ('Done' in event) {
        // Sole finalize+persist trigger (see header). Fire-and-forget from the
        // sync handler; errors surface as a store error, not thrown to askNotebook.
        void persistOnce(event.Done.tokens_used).catch((err) => {
          if (state.streamGeneration !== gen) return;
          state.error = { kind: 'Internal', message: String(err) };
          resetStreamBuffers(state);
        });
      }
    },
    onStreamDone: () => {
      if (state.streamGeneration !== gen) return;
      // Outer terminator — no-op for persistence (see header). Only clear
      // streaming if the inner Done never arrived.
      if (!persisted) {
        resetStreamBuffers(state);
      }
    },
    onFailed: (err) => {
      if (state.streamGeneration !== gen) return;
      if (err.kind === 'Cancelled') {
        // AC8: keep partial answerBuffer in-session, persist nothing.
        state.streaming = false;
        state.stage = null;
      } else {
        // AC7/AC21: sanitized error, persist nothing.
        state.error = err;
        state.streaming = false;
        state.stage = null;
      }
    }
  });
}

/**
 * Sends a new question (AC1, AC9). Cancels any in-flight turn for this notebook
 * first (single-flight, AC10, Risk R7). Generates a frontend `turn_id`
 * (crypto.randomUUID — an opaque grouping key, not a row id) and persists the
 * user row immediately.
 */
export async function send(notebookId: string, question: string): Promise<void> {
  const state = ensure(notebookId);
  if (state.streaming) {
    await cancelAsk(notebookId);
  }

  const turnId = crypto.randomUUID();
  const userRow = await saveChatUser(notebookId, turnId, question);
  state.turns.push({ turn_id: turnId, user: userRow, versions: [] });
  state.pinnedToBottom = true;

  await runStream(notebookId, turnId, question);
}

/** Stops the in-flight turn for a notebook (AC10). */
export async function stop(notebookId: string): Promise<void> {
  await cancelAsk(notebookId);
}

/**
 * Re-asks the question of an existing turn (AC13). Reads the question text from
 * the USER row (never an assistant version). Appends a new assistant version on
 * inner `Done` under the SAME `turn_id`. Cancels any in-flight turn first
 * (single-flight, Risk R7).
 */
export async function regenerate(notebookId: string, turnId: string): Promise<void> {
  const state = ensure(notebookId);
  const turn = state.turns.find((t) => t.turn_id === turnId);
  if (!turn) return;
  if (state.streaming) {
    await cancelAsk(notebookId);
  }
  await runStream(notebookId, turnId, turn.user.content);
}

/**
 * Toggles feedback on an assistant message (AC14, AC22): clicking the active
 * thumb clears it to `null`; clicking the opposite switches it. Optimistic update.
 */
export async function setFeedback(
  notebookId: string,
  messageId: string,
  next: ChatFeedback
): Promise<void> {
  const state = ensure(notebookId);
  for (const turn of state.turns) {
    const version = turn.versions.find((v) => v.id === messageId);
    if (version) {
      const prev = version.feedback;
      const resolved = version.feedback === next ? null : next;
      version.feedback = resolved;
      try {
        await setChatFeedback(messageId, resolved);
      } catch {
        version.feedback = prev;
      }
      return;
    }
  }
}

/** Copies an assistant message's markdown source to the clipboard (AC11, AC22). */
export async function copyMessage(content: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(content);
  } catch (err) {
    console.warn('copyMessage: clipboard write failed', err);
  }
}

/** Re-pins the transcript to the bottom (AC19 "Jump to latest"). */
export function jumpToLatest(notebookId: string): void {
  ensure(notebookId).pinnedToBottom = true;
}

/** Marks the transcript as unpinned (user scrolled up, AC19). */
export function unpin(notebookId: string): void {
  ensure(notebookId).pinnedToBottom = false;
}

export const chatStore = {
  turns(notebookId: string): Turn[] {
    return byNotebook[notebookId]?.turns ?? [];
  },
  streaming(notebookId: string): boolean {
    return byNotebook[notebookId]?.streaming ?? false;
  },
  stage(notebookId: string): AnswerStage | null {
    return byNotebook[notebookId]?.stage ?? null;
  },
  thinkingBuffer(notebookId: string): string {
    return byNotebook[notebookId]?.thinkingBuffer ?? '';
  },
  answerBuffer(notebookId: string): string {
    return byNotebook[notebookId]?.answerBuffer ?? '';
  },
  currentTurnId(notebookId: string): string | null {
    return byNotebook[notebookId]?.currentTurnId ?? null;
  },
  error(notebookId: string): { kind: string; message: string } | null {
    return byNotebook[notebookId]?.error ?? null;
  },
  pinnedToBottom(notebookId: string): boolean {
    return byNotebook[notebookId]?.pinnedToBottom ?? true;
  }
};

/** Reset all state. Call in `afterEach` to prevent cross-test bleed. */
export function resetChatStore(): void {
  byNotebook = {};
}
