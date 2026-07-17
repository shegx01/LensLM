// Chat reactive store (Svelte 5 runes, module singleton). Mirrors
// notebooks-state.svelte.ts's shape (module-level $state + exported getters).
//
// Persistence contract (plan Step 6 / Risk R9 — the highest-risk seam):
// the INNER `AnswerEvent::Done{...}` delivered inside a `chunk` frame is the SOLE
// finalize+persist trigger for a SUCCESSFUL answer — it is where `saveChatAssistant`
// is called, exactly once. The OUTER stream-closed terminator (`StreamEvent::Done`,
// surfaced here as `onStreamDone`) is a no-op for persistence; it must never
// re-trigger a save.
//
// Plan 2 (PC-1) change: a cancelled/errored turn no longer persists NOTHING — it
// writes ONE terminal marker row (partial answer as content, `state` =
// cancelled/errored) via `saveChatMarker`, so a reload renders a "Stopped"/"Couldn't
// complete" line instead of a bare, dangling question. `terminalSettled` dedupes the
// marker across the two paths that can fire for one turn (see its declaration below).

import {
  askNotebook,
  cancelAsk,
  saveChatUser,
  saveChatAssistant,
  saveChatMarker,
  setChatFeedback,
  listChatMessages
} from './ipc.js';
import { parseCitations } from './citations.js';
import type {
  AnswerEvent,
  AnswerStage,
  ChatMessage,
  ChatFeedback,
  ChatState,
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
  /** Turn whose just-finished answer was ungrounded (text, zero citations). Drives
   * a subtle live badge (Plan 2 / SP-3); `null` when the last answer was grounded
   * or none is fresh. Live-only — cleared on the next send, not persisted. */
  ungroundedTurnId: string | null;
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
    streamGeneration: 0,
    ungroundedTurnId: null
  };
}

let byNotebook = $state<Record<string, NotebookChatState>>({});

// Turn ids that already have a persisted terminal marker (Plan 2). Guards against a
// double marker when both the stream callback and `markSuperseded` fire for one turn
// (the check+add is synchronous, so the two callbacks can't both pass it). Module-
// scoped and small (opaque UUIDs); cleared in `resetChatStore`.
const terminalSettled = new Set<string>();

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

/** Passes a `{kind,message}` LensError through; wraps anything else without leaking a raw `Error:` prefix. */
function toLensError(err: unknown): { kind: string; message: string } {
  if (err && typeof err === 'object' && 'kind' in err && 'message' in err) {
    return err as { kind: string; message: string };
  }
  return { kind: 'Internal', message: err instanceof Error ? err.message : String(err) };
}

/**
 * Persists a terminal marker for `turnId` (Plan 2 / PC-1); guarded by `terminalSettled`
 * (see its declaration above). `pushInSession` appends the returned row as a version so
 * the partial answer stays visible immediately — `true` for cancel (the marker IS the
 * in-session rendering), `false` for stream errors (the ErrorCard already renders
 * in-session; the marker is only for reload). On IPC failure the guard is released so a
 * later path may retry, and the error is swallowed (never throw from a stream callback).
 */
async function persistTerminalMarker(
  state: NotebookChatState,
  notebookId: string,
  turnId: string,
  markerState: Exclude<ChatState, null>,
  errorKind: string | null,
  content: string,
  pushInSession: boolean
): Promise<void> {
  if (terminalSettled.has(turnId)) return;
  terminalSettled.add(turnId);
  let saved: ChatMessage;
  try {
    saved = await saveChatMarker(notebookId, turnId, content, markerState, errorKind);
  } catch (err) {
    terminalSettled.delete(turnId);
    console.warn('persistTerminalMarker: save failed', err);
    return;
  }
  if (!pushInSession) return;
  const turn = state.turns.find((t) => t.turn_id === turnId);
  if (turn && !turn.versions.some((v) => v.id === saved.id)) {
    turn.versions.push(saved);
  }
}

/**
 * When a new send/regenerate supersedes an in-flight turn, capture its partial answer
 * into a cancelled marker BEFORE the new stream resets the buffers, so the superseded
 * turn is never left a bare question (FE-2). The later `onFailed('Cancelled')` from
 * the cancelled stream is then a no-op (already marked).
 */
async function markSuperseded(state: NotebookChatState, notebookId: string): Promise<void> {
  if (!state.streaming || !state.currentTurnId) return;
  await persistTerminalMarker(
    state,
    notebookId,
    state.currentTurnId,
    'cancelled',
    null,
    state.answerBuffer,
    true
  );
}

async function runStream(notebookId: string, turnId: string, question: string): Promise<void> {
  const state = ensure(notebookId);
  // Fresh attempt for this turn — clear any prior terminal marker (e.g. regenerating
  // a previously-cancelled turn) so this run can record its own outcome.
  terminalSettled.delete(turnId);
  // Bump so late callbacks from a superseded/cancelled stream no-op.
  const gen = ++state.streamGeneration;
  state.streaming = true;
  state.stage = null;
  state.thinkingBuffer = '';
  state.answerBuffer = '';
  state.pendingCitations = null;
  state.currentTurnId = turnId;
  state.error = null;
  state.ungroundedTurnId = null;
  state.pinnedToBottom = true;

  let persisted = false;

  // rAF-coalesce streaming text (FE-3): buffer deltas and flush to the reactive
  // answerBuffer at most once per frame, so the markdown re-render runs ~60fps
  // instead of once per token (previously O(n²) over the growing answer).
  let pendingDelta = '';
  let rafHandle: number | null = null;
  const hasRaf = typeof requestAnimationFrame !== 'undefined';
  const flushDelta = (): void => {
    rafHandle = null;
    if (!pendingDelta) return;
    if (state.streamGeneration !== gen) {
      pendingDelta = '';
      return;
    }
    state.answerBuffer += pendingDelta;
    pendingDelta = '';
  };
  const scheduleFlush = (): void => {
    if (!hasRaf) {
      flushDelta();
      return;
    }
    if (rafHandle === null) rafHandle = requestAnimationFrame(flushDelta);
  };
  const flushNow = (): void => {
    if (rafHandle !== null && typeof cancelAnimationFrame !== 'undefined') {
      cancelAnimationFrame(rafHandle);
      rafHandle = null;
    }
    flushDelta();
  };

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

  // A pre-stream ctx/provider error rejects the invoke AND delivers a `failed`
  // frame; `onFailed` owns the state transition, so swallow the rejection here to
  // avoid an unhandled-rejection surfacing through the caller's `void send(...)`.
  await askNotebook(notebookId, turnId, question, {
    onEvent: (event: AnswerEvent) => {
      if (state.streamGeneration !== gen) return;
      if ('Stage' in event) {
        state.stage = event.Stage;
      } else if ('ThinkingDelta' in event) {
        state.thinkingBuffer += event.ThinkingDelta;
      } else if ('TextDelta' in event) {
        pendingDelta += event.TextDelta;
        scheduleFlush();
      } else if ('Citations' in event) {
        state.pendingCitations = event.Citations;
      } else if ('Done' in event) {
        flushNow(); // ensure the whole answer is in answerBuffer before persisting
        // Ungrounded flag (SP-3): substantive text that cited nothing.
        if (!event.Done.grounded) state.ungroundedTurnId = turnId;
        // Claim the turn synchronously so a concurrent markSuperseded cannot slip a
        // cancelled marker in during the save's IPC window (a successful answer needs
        // no marker). The save-failure catch releases the claim to record an error.
        terminalSettled.add(turnId);
        // Sole finalize+persist trigger for a SUCCESSFUL answer (see header).
        void persistOnce(event.Done.tokens_used).catch((err) => {
          const lensErr = toLensError(err);
          // Persist the errored marker FIRST, independent of the generation guard:
          // marker persistence is reload-safety, and a resend during the save window
          // (which bumps the generation) must not leave a bare question (PC-1).
          terminalSettled.delete(turnId); // release the success-claim so the marker records
          void persistTerminalMarker(
            state,
            notebookId,
            turnId,
            'errored',
            lensErr.kind,
            state.answerBuffer,
            false
          );
          // In-session UI mutations only apply to the still-current turn.
          if (state.streamGeneration !== gen) return;
          state.error = lensErr;
          state.streaming = false;
          state.stage = null;
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
      flushNow();
      if (err.kind === 'Cancelled') {
        // Stop button (FE-1/PC-1): keep the partial answer as a cancelled marker so
        // it survives reload, and surface it in-session as a "Stopped" version.
        void persistTerminalMarker(
          state,
          notebookId,
          turnId,
          'cancelled',
          null,
          state.answerBuffer,
          true
        );
        state.streaming = false;
        state.stage = null;
      } else {
        // Stream error (AC7/AC21): sanitized error → ErrorCard (retry) in-session;
        // persist an errored marker (backend only) so reload shows a terminal line.
        void persistTerminalMarker(
          state,
          notebookId,
          turnId,
          'errored',
          err.kind,
          state.answerBuffer,
          false
        );
        state.error = err;
        state.streaming = false;
        state.stage = null;
      }
    }
  }).catch(() => {
    // Rejection already reflected via onFailed above (or a superseded gen). Swallow.
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
    // Mark the superseded turn (FE-2) BEFORE cancel resets its buffers.
    await markSuperseded(state, notebookId);
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
    await markSuperseded(state, notebookId);
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
  pendingCitations(notebookId: string): Citation[] | null {
    return byNotebook[notebookId]?.pendingCitations ?? null;
  },
  currentTurnId(notebookId: string): string | null {
    return byNotebook[notebookId]?.currentTurnId ?? null;
  },
  error(notebookId: string): { kind: string; message: string } | null {
    return byNotebook[notebookId]?.error ?? null;
  },
  pinnedToBottom(notebookId: string): boolean {
    return byNotebook[notebookId]?.pinnedToBottom ?? true;
  },
  ungroundedTurnId(notebookId: string): string | null {
    return byNotebook[notebookId]?.ungroundedTurnId ?? null;
  }
};

/** Reset all state. Call in `afterEach` to prevent cross-test bleed. */
export function resetChatStore(): void {
  byNotebook = {};
  terminalSettled.clear();
}
