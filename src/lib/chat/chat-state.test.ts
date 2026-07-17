// Store unit tests for chat-state.svelte.ts (IPC mocked, no Tauri host).

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./ipc.js', () => ({
  askNotebook: vi.fn(),
  cancelAsk: vi.fn().mockResolvedValue(true),
  saveChatUser: vi.fn(),
  saveChatAssistant: vi.fn(),
  saveChatMarker: vi.fn(),
  setChatFeedback: vi.fn().mockResolvedValue(undefined),
  listChatMessages: vi.fn()
}));

import {
  chatStore,
  resetChatStore,
  hydrate,
  send,
  regenerate,
  setFeedback,
  latestVersion,
  messageCitations
} from './chat-state.svelte.js';
import {
  askNotebook,
  cancelAsk,
  saveChatUser,
  saveChatAssistant,
  saveChatMarker,
  listChatMessages
} from './ipc.js';
import type { AskNotebookHandlers } from './ipc.js';
import type { Citation } from './types.js';
import { makeChatMessage } from './test-fixtures.js';

const NB = 'nb-001';

beforeEach(() => {
  vi.clearAllMocks();
  resetChatStore();
  // rAF-coalesced deltas (FE-3): stub requestAnimationFrame to run synchronously so
  // TextDelta events land in answerBuffer deterministically without a real frame tick.
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback): number => {
    cb(0);
    return 0;
  });
  vi.stubGlobal('cancelAnimationFrame', (): void => {});
});

afterEach(() => {
  resetChatStore();
  vi.unstubAllGlobals();
});

describe('hydrate / grouping', () => {
  it('groups flat rows into turns with newest version last', async () => {
    const user = makeChatMessage({ id: 'u1', turn_id: 't1', role: 'user', content: 'q1' });
    const v1 = makeChatMessage({
      id: 'a1',
      turn_id: 't1',
      role: 'assistant',
      content: 'a1',
      created_at: new Date(Date.now() - 1000).toISOString()
    });
    const v2 = makeChatMessage({
      id: 'a2',
      turn_id: 't1',
      role: 'assistant',
      content: 'a2',
      created_at: new Date().toISOString()
    });
    vi.mocked(listChatMessages).mockResolvedValue([user, v1, v2]);

    await hydrate(NB);

    const turns = chatStore.turns(NB);
    expect(turns).toHaveLength(1);
    expect(turns[0].user.id).toBe('u1');
    expect(turns[0].versions.map((v) => v.id)).toEqual(['a1', 'a2']);
    expect(latestVersion(turns[0])?.id).toBe('a2');
  });

  it('handles a reload-after-cancel turn: user row with zero assistant versions', async () => {
    const user = makeChatMessage({ id: 'u1', turn_id: 't1', role: 'user' });
    vi.mocked(listChatMessages).mockResolvedValue([user]);

    await hydrate(NB);

    const turns = chatStore.turns(NB);
    expect(turns).toHaveLength(1);
    expect(turns[0].versions).toEqual([]);
    expect(latestVersion(turns[0])).toBeNull();
  });

  it('parses citations JSON on the assistant row', async () => {
    const citations: Citation[] = [{ source_id: 'src-1', ordinal: 1, locators: [] }];
    const assistant = makeChatMessage({
      id: 'a1',
      role: 'assistant',
      citations: JSON.stringify(citations)
    });
    expect(messageCitations(assistant)).toEqual(citations);
  });
});

describe('send / persist-exactly-once fixture', () => {
  it('persists exactly once on inner Done; outer done triggers no second save', async () => {
    vi.mocked(saveChatUser).mockResolvedValue(makeChatMessage({ id: 'u1', turn_id: 'ignored' }));
    const savedAssistant = makeChatMessage({ id: 'a1', role: 'assistant', content: 'ab' });
    vi.mocked(saveChatAssistant).mockResolvedValue(savedAssistant);

    const citations: Citation[] = [{ source_id: 'src-1', ordinal: 1, locators: [] }];

    vi.mocked(askNotebook).mockImplementation(
      async (_nb: string, _turnId: string, _q: string, handlers: AskNotebookHandlers) => {
        handlers.onEvent({ TextDelta: 'a' });
        handlers.onEvent({ TextDelta: 'b' });
        handlers.onEvent({ Citations: citations });
        handlers.onEvent({ Done: { tokens_used: 42, grounded: true, citation_count: 0 } });
        handlers.onStreamDone();
      }
    );

    await send(NB, 'what is x?');

    expect(saveChatAssistant).toHaveBeenCalledTimes(1);
    const [notebookId, turnId, content, cites, tokens] = vi.mocked(saveChatAssistant).mock.calls[0];
    expect(notebookId).toBe(NB);
    expect(typeof turnId).toBe('string');
    expect(content).toBe('ab');
    expect(cites).toEqual(citations);
    expect(tokens).toBe(42);

    const turns = chatStore.turns(NB);
    expect(turns).toHaveLength(1);
    expect(turns[0].versions).toHaveLength(1);
    expect(chatStore.streaming(NB)).toBe(false);
  });

  it('freshly-answered parity: the pushed version round-trips citations without a reload (S6.5)', async () => {
    vi.mocked(saveChatUser).mockResolvedValue(makeChatMessage({ id: 'u1', turn_id: 'ignored' }));
    const citations: Citation[] = [
      { source_id: 'src-1', ordinal: 1, locators: [] },
      { source_id: 'src-2', ordinal: 2, locators: [] }
    ];
    // Mirror insert_assistant (chat.rs:282): the saved row carries citations as a JSON string.
    vi.mocked(saveChatAssistant).mockResolvedValue(
      makeChatMessage({
        id: 'a1',
        role: 'assistant',
        content: 'answer',
        citations: JSON.stringify(citations)
      })
    );

    vi.mocked(askNotebook).mockImplementation(
      async (_nb: string, _turnId: string, _q: string, handlers: AskNotebookHandlers) => {
        handlers.onEvent({ TextDelta: 'answer' });
        handlers.onEvent({ Citations: citations });
        handlers.onEvent({ Done: { tokens_used: 7, grounded: true, citation_count: 0 } });
        handlers.onStreamDone();
      }
    );

    await send(NB, 'q?');

    const turns = chatStore.turns(NB);
    const saved = latestVersion(turns[0]);
    expect(saved).not.toBeNull();
    expect(messageCitations(saved!)).toEqual(citations);
  });

  it('cancels an in-flight turn before starting a new send (single-flight)', async () => {
    vi.mocked(saveChatUser).mockResolvedValue(makeChatMessage({ id: 'u1' }));
    // First send's stream never reaches Done/onStreamDone/onFailed, so `streaming`
    // stays true after it resolves — simulating a still-in-flight turn.
    vi.mocked(askNotebook).mockImplementationOnce(
      async (_nb: string, _turnId: string, _q: string, handlers: AskNotebookHandlers) => {
        handlers.onEvent({ Stage: 'Retrieving' });
      }
    );

    await send(NB, 'first');
    expect(chatStore.streaming(NB)).toBe(true);

    vi.mocked(askNotebook).mockImplementationOnce(
      async (_nb: string, _turnId: string, _q: string, handlers: AskNotebookHandlers) => {
        handlers.onEvent({ Done: { tokens_used: 1, grounded: true, citation_count: 0 } });
        handlers.onStreamDone();
      }
    );
    vi.mocked(saveChatAssistant).mockResolvedValue(
      makeChatMessage({ id: 'a-second', role: 'assistant' })
    );

    await send(NB, 'second');
    expect(cancelAsk).toHaveBeenCalledWith(NB);
  });

  it('ignores a stale onFailed({Cancelled}) from a superseded stream (generation guard)', async () => {
    vi.mocked(saveChatUser).mockResolvedValue(makeChatMessage({ id: 'u1' }));

    let staleHandlers: AskNotebookHandlers | undefined;
    // First send's askNotebook captures its handlers but never calls them —
    // simulating a cancel command that resolves before the prior stream drains.
    vi.mocked(askNotebook).mockImplementationOnce(
      async (_nb: string, _turnId: string, _q: string, handlers: AskNotebookHandlers) => {
        staleHandlers = handlers;
        handlers.onEvent({ TextDelta: 'stale-first-turn-text' });
      }
    );

    await send(NB, 'first');
    expect(chatStore.streaming(NB)).toBe(true);

    // Second send's stream stays in-flight (never reaches Done) so we can
    // observe corruption from the stale handler if the guard were absent.
    vi.mocked(askNotebook).mockImplementationOnce(
      async (_nb: string, _turnId: string, _q: string, handlers: AskNotebookHandlers) => {
        handlers.onEvent({ TextDelta: 'second-turn-text' });
      }
    );

    await send(NB, 'second');
    expect(chatStore.streaming(NB)).toBe(true);
    expect(chatStore.answerBuffer(NB)).toBe('second-turn-text');

    // The stale first-turn stream's onFailed arrives late, after the second
    // runStream has already started. It must be inert.
    staleHandlers?.onFailed({ kind: 'Cancelled', message: 'answer generation cancelled' });

    expect(chatStore.streaming(NB)).toBe(true);
    expect(chatStore.answerBuffer(NB)).toBe('second-turn-text');
    expect(chatStore.error(NB)).toBeNull();
  });
});

describe('cancel path', () => {
  it('keeps partial answerBuffer in-session and persists nothing', async () => {
    vi.mocked(saveChatUser).mockResolvedValue(makeChatMessage({ id: 'u1' }));
    vi.mocked(askNotebook).mockImplementation(
      async (_nb: string, _turnId: string, _q: string, handlers: AskNotebookHandlers) => {
        handlers.onEvent({ TextDelta: 'partial' });
        handlers.onFailed({ kind: 'Cancelled', message: 'answer generation cancelled' });
      }
    );

    await send(NB, 'q');

    expect(saveChatAssistant).not.toHaveBeenCalled();
    expect(chatStore.answerBuffer(NB)).toBe('partial');
    expect(chatStore.streaming(NB)).toBe(false);
    expect(chatStore.error(NB)).toBeNull();
  });

  it('persists a cancelled marker with the partial answer and pushes it as a version (PC-1)', async () => {
    vi.mocked(saveChatUser).mockResolvedValue(makeChatMessage({ id: 'u1' }));
    vi.mocked(saveChatMarker).mockResolvedValue(
      makeChatMessage({
        id: 'm1',
        role: 'assistant',
        state: 'cancelled',
        content: 'partial answer'
      })
    );
    vi.mocked(askNotebook).mockImplementation(
      async (_nb: string, _turnId: string, _q: string, handlers: AskNotebookHandlers) => {
        handlers.onEvent({ TextDelta: 'partial' });
        handlers.onEvent({ TextDelta: ' answer' });
        handlers.onFailed({ kind: 'Cancelled', message: 'answer generation cancelled' });
        // persistTerminalMarker is fire-and-forget from onFailed; flush microtasks
        // so its saveChatMarker call + in-session version push settle before we assert.
        await Promise.resolve();
        await Promise.resolve();
      }
    );

    await send(NB, 'q');

    expect(saveChatMarker).toHaveBeenCalledTimes(1);
    const [notebookId, turnId, content, state, errorKind] = vi.mocked(saveChatMarker).mock.calls[0];
    expect(notebookId).toBe(NB);
    expect(typeof turnId).toBe('string');
    expect(content).toBe('partial answer');
    expect(state).toBe('cancelled');
    expect(errorKind).toBeNull();

    const turns = chatStore.turns(NB);
    expect(turns[0].versions).toHaveLength(1);
    expect(turns[0].versions[0].state).toBe('cancelled');
  });
});

describe('error path', () => {
  it('sets error state and persists nothing on a non-cancel failure', async () => {
    vi.mocked(saveChatUser).mockResolvedValue(makeChatMessage({ id: 'u1' }));
    vi.mocked(askNotebook).mockImplementation(
      async (_nb: string, _turnId: string, _q: string, handlers: AskNotebookHandlers) => {
        handlers.onEvent({ TextDelta: 'x' });
        handlers.onFailed({ kind: 'Internal', message: 'boom' });
      }
    );

    await send(NB, 'q');

    expect(saveChatAssistant).not.toHaveBeenCalled();
    expect(chatStore.error(NB)).toEqual({ kind: 'Internal', message: 'boom' });
    expect(chatStore.streaming(NB)).toBe(false);
  });

  it('sets error state and stops streaming when saveChatAssistant rejects on inner Done', async () => {
    vi.mocked(saveChatUser).mockResolvedValue(makeChatMessage({ id: 'u1' }));
    vi.mocked(saveChatAssistant).mockRejectedValue(new Error('save failed'));
    vi.mocked(askNotebook).mockImplementation(
      async (_nb: string, _turnId: string, _q: string, handlers: AskNotebookHandlers) => {
        handlers.onEvent({ TextDelta: 'partial answer' });
        handlers.onEvent({ Done: { tokens_used: 5, grounded: true, citation_count: 0 } });
        handlers.onStreamDone();
        // persistOnce is fire-and-forget from the Done handler; flush microtasks
        // so its rejection settles before we assert.
        await Promise.resolve();
        await Promise.resolve();
      }
    );

    await send(NB, 'q');

    expect(chatStore.error(NB)).toEqual({ kind: 'Internal', message: 'save failed' });
    expect(chatStore.streaming(NB)).toBe(false);
    // currentTurnId MUST survive so the ErrorCard gate (error && currentTurnId === turn)
    // matches on this path too — otherwise the error/Retry UI is silently unreachable.
    const turnId = chatStore.turns(NB)[0]?.turn_id;
    expect(turnId).toBeTruthy();
    expect(chatStore.currentTurnId(NB)).toBe(turnId);
  });

  it('persists an errored marker with the errorKind and the partial answer (PC-1)', async () => {
    vi.mocked(saveChatUser).mockResolvedValue(makeChatMessage({ id: 'u1' }));
    vi.mocked(saveChatMarker).mockResolvedValue(
      makeChatMessage({
        id: 'm2',
        role: 'assistant',
        state: 'errored',
        error_kind: 'Model',
        content: 'partial'
      })
    );
    vi.mocked(askNotebook).mockImplementation(
      async (_nb: string, _turnId: string, _q: string, handlers: AskNotebookHandlers) => {
        handlers.onEvent({ TextDelta: 'partial' });
        handlers.onFailed({ kind: 'Model', message: 'the model backend failed' });
        await Promise.resolve();
        await Promise.resolve();
      }
    );

    await send(NB, 'q');

    expect(saveChatMarker).toHaveBeenCalledTimes(1);
    const [notebookId, turnId, content, state, errorKind] = vi.mocked(saveChatMarker).mock.calls[0];
    expect(notebookId).toBe(NB);
    expect(typeof turnId).toBe('string');
    expect(content).toBe('partial');
    expect(state).toBe('errored');
    expect(errorKind).toBe('Model');

    expect(chatStore.error(NB)).toEqual({ kind: 'Model', message: 'the model backend failed' });
  });
});

describe('regenerate', () => {
  it('reads the question from the user row and appends a new version under the same turn_id', async () => {
    const user = makeChatMessage({ id: 'u1', turn_id: 't1', role: 'user', content: 'original q' });
    vi.mocked(listChatMessages).mockResolvedValue([user]);
    await hydrate(NB);

    const savedAssistant = makeChatMessage({
      id: 'a1',
      turn_id: 't1',
      role: 'assistant',
      content: 'answer v1'
    });
    vi.mocked(saveChatAssistant).mockResolvedValue(savedAssistant);

    vi.mocked(askNotebook).mockImplementation(
      async (_nb: string, _turnId: string, question: string, handlers: AskNotebookHandlers) => {
        expect(question).toBe('original q');
        handlers.onEvent({ TextDelta: 'answer v1' });
        handlers.onEvent({ Done: { tokens_used: 10, grounded: true, citation_count: 0 } });
        handlers.onStreamDone();
      }
    );

    await regenerate(NB, 't1');

    expect(saveChatAssistant).toHaveBeenCalledTimes(1);
    expect(saveChatAssistant).toHaveBeenCalledWith(NB, 't1', 'answer v1', null, 10);

    const turns = chatStore.turns(NB);
    expect(turns[0].versions).toHaveLength(1);
    expect(turns[0].versions[0].id).toBe('a1');
  });
});

describe('feedback toggle', () => {
  it('sets up, then clears to null when clicking the active thumb again', async () => {
    const user = makeChatMessage({ id: 'u1', turn_id: 't1', role: 'user' });
    const assistant = makeChatMessage({ id: 'a1', turn_id: 't1', role: 'assistant' });
    vi.mocked(listChatMessages).mockResolvedValue([user, assistant]);
    await hydrate(NB);

    await setFeedback(NB, 'a1', 'up');
    let turns = chatStore.turns(NB);
    expect(turns[0].versions[0].feedback).toBe('up');
    expect(vi.mocked((await import('./ipc.js')).setChatFeedback)).toHaveBeenCalledWith('a1', 'up');

    await setFeedback(NB, 'a1', 'up');
    turns = chatStore.turns(NB);
    expect(turns[0].versions[0].feedback).toBeNull();
  });

  it('switches from up to down directly', async () => {
    const user = makeChatMessage({ id: 'u1', turn_id: 't1', role: 'user' });
    const assistant = makeChatMessage({
      id: 'a1',
      turn_id: 't1',
      role: 'assistant',
      feedback: 'up'
    });
    vi.mocked(listChatMessages).mockResolvedValue([user, assistant]);
    await hydrate(NB);

    await setFeedback(NB, 'a1', 'down');
    const turns = chatStore.turns(NB);
    expect(turns[0].versions[0].feedback).toBe('down');
  });

  it('rolls back the optimistic write when the IPC call rejects', async () => {
    const user = makeChatMessage({ id: 'u1', turn_id: 't1', role: 'user' });
    const assistant = makeChatMessage({
      id: 'a1',
      turn_id: 't1',
      role: 'assistant',
      feedback: 'up'
    });
    vi.mocked(listChatMessages).mockResolvedValue([user, assistant]);
    await hydrate(NB);

    const { setChatFeedback } = await import('./ipc.js');
    vi.mocked(setChatFeedback).mockRejectedValueOnce(new Error('boom'));

    await setFeedback(NB, 'a1', 'down');

    const turns = chatStore.turns(NB);
    expect(turns[0].versions[0].feedback).toBe('up');
  });
});

describe('superseded turn (FE-2)', () => {
  it('persists a cancelled marker for the in-flight turn before the new turn starts', async () => {
    vi.mocked(saveChatUser)
      .mockResolvedValueOnce(makeChatMessage({ id: 'u1', turn_id: 't1' }))
      .mockResolvedValueOnce(makeChatMessage({ id: 'u2', turn_id: 't2' }));
    vi.mocked(saveChatMarker).mockResolvedValue(
      makeChatMessage({
        id: 'm1',
        role: 'assistant',
        state: 'cancelled',
        content: 'partial-first'
      })
    );

    let firstTurnId = '';
    // First send's stream never reaches Done/onStreamDone/onFailed — it stays
    // in-flight (streaming stays true) so the second send must supersede it.
    vi.mocked(askNotebook).mockImplementationOnce(
      async (_nb: string, turnId: string, _q: string, handlers: AskNotebookHandlers) => {
        firstTurnId = turnId;
        handlers.onEvent({ TextDelta: 'partial-first' });
      }
    );

    await send(NB, 'first');
    expect(chatStore.streaming(NB)).toBe(true);

    vi.mocked(askNotebook).mockImplementationOnce(
      async (_nb: string, _turnId: string, _q: string, handlers: AskNotebookHandlers) => {
        handlers.onEvent({ Done: { tokens_used: 1, grounded: true, citation_count: 0 } });
        handlers.onStreamDone();
      }
    );
    vi.mocked(saveChatAssistant).mockResolvedValue(
      makeChatMessage({ id: 'a-second', role: 'assistant' })
    );

    await send(NB, 'second');

    expect(saveChatMarker).toHaveBeenCalledTimes(1);
    const [notebookId, turnId, content, state, errorKind] = vi.mocked(saveChatMarker).mock.calls[0];
    expect(notebookId).toBe(NB);
    expect(turnId).toBe(firstTurnId);
    expect(content).toBe('partial-first');
    expect(state).toBe('cancelled');
    expect(errorKind).toBeNull();

    // markSuperseded runs BEFORE cancelAsk in `send` (see chat-state.svelte.ts),
    // so the marker call must be ordered ahead of the cancel call.
    const markerOrder = vi.mocked(saveChatMarker).mock.invocationCallOrder[0];
    const cancelOrder = vi.mocked(cancelAsk).mock.invocationCallOrder[0];
    expect(markerOrder).toBeLessThan(cancelOrder);
  });
});

describe('marker idempotency (terminalMarked dedupe)', () => {
  it('persists the marker at most once even if the supersede path and a later cancel callback both fire for the same turn', async () => {
    vi.mocked(saveChatUser).mockResolvedValue(makeChatMessage({ id: 'u1' }));
    vi.mocked(saveChatMarker).mockResolvedValue(
      makeChatMessage({ id: 'm1', role: 'assistant', state: 'cancelled' })
    );

    let staleHandlers: AskNotebookHandlers | undefined;
    // First send's askNotebook captures its handlers but never resolves the
    // terminal callbacks itself — simulating a still-in-flight stream.
    vi.mocked(askNotebook).mockImplementationOnce(
      async (_nb: string, _turnId: string, _q: string, handlers: AskNotebookHandlers) => {
        staleHandlers = handlers;
        handlers.onEvent({ TextDelta: 'stale-first-turn-text' });
      }
    );

    await send(NB, 'first');
    expect(chatStore.streaming(NB)).toBe(true);

    vi.mocked(askNotebook).mockImplementationOnce(
      async (_nb: string, _turnId: string, _q: string, handlers: AskNotebookHandlers) => {
        handlers.onEvent({ TextDelta: 'second-turn-text' });
      }
    );

    // Supersede path (FE-2): marks+persists the first turn as cancelled.
    await send(NB, 'second');
    expect(saveChatMarker).toHaveBeenCalledTimes(1);

    // The stale first-turn stream's own onFailed('Cancelled') arrives late, after
    // the supersede already marked+persisted this turn — it must be a no-op, so
    // the marker is never persisted twice for the same turn_id.
    staleHandlers?.onFailed({ kind: 'Cancelled', message: 'answer generation cancelled' });
    await Promise.resolve();
    await Promise.resolve();

    expect(saveChatMarker).toHaveBeenCalledTimes(1);
  });
});

describe('grounded flag (SP-3)', () => {
  it('sets ungroundedTurnId to the turn id when Done arrives with grounded: false', async () => {
    vi.mocked(saveChatUser).mockResolvedValue(makeChatMessage({ id: 'u1' }));
    vi.mocked(saveChatAssistant).mockResolvedValue(
      makeChatMessage({ id: 'a1', role: 'assistant' })
    );
    let capturedTurnId = '';
    vi.mocked(askNotebook).mockImplementation(
      async (_nb: string, turnId: string, _q: string, handlers: AskNotebookHandlers) => {
        capturedTurnId = turnId;
        handlers.onEvent({ TextDelta: 'answer' });
        handlers.onEvent({ Done: { tokens_used: 3, grounded: false, citation_count: 0 } });
        handlers.onStreamDone();
      }
    );

    await send(NB, 'q');

    expect(capturedTurnId).not.toBe('');
    expect(chatStore.ungroundedTurnId(NB)).toBe(capturedTurnId);
  });

  it('leaves ungroundedTurnId null when Done arrives with grounded: true', async () => {
    vi.mocked(saveChatUser).mockResolvedValue(makeChatMessage({ id: 'u1' }));
    vi.mocked(saveChatAssistant).mockResolvedValue(
      makeChatMessage({ id: 'a1', role: 'assistant' })
    );
    vi.mocked(askNotebook).mockImplementation(
      async (_nb: string, _turnId: string, _q: string, handlers: AskNotebookHandlers) => {
        handlers.onEvent({ TextDelta: 'answer' });
        handlers.onEvent({ Done: { tokens_used: 3, grounded: true, citation_count: 0 } });
        handlers.onStreamDone();
      }
    );

    await send(NB, 'q');

    expect(chatStore.ungroundedTurnId(NB)).toBeNull();
  });
});
