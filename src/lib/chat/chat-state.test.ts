// Store unit tests for chat-state.svelte.ts (IPC mocked, no Tauri host).

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./ipc.js', () => ({
  askNotebook: vi.fn(),
  cancelAsk: vi.fn().mockResolvedValue(true),
  saveChatUser: vi.fn(),
  saveChatAssistant: vi.fn(),
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
  listChatMessages
} from './ipc.js';
import type { AskNotebookHandlers } from './ipc.js';
import type { ChatMessage, Citation } from './types.js';

const NB = 'nb-001';

function makeChatMessage(overrides?: Partial<ChatMessage>): ChatMessage {
  return {
    id: 'msg-001',
    notebook_id: NB,
    turn_id: 'turn-001',
    role: 'user',
    content: 'hello',
    citations: null,
    feedback: null,
    tokens_used: null,
    created_at: new Date().toISOString(),
    ...overrides
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  resetChatStore();
});

afterEach(() => {
  resetChatStore();
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
      async (_nb: string, _q: string, handlers: AskNotebookHandlers) => {
        handlers.onEvent({ TextDelta: 'a' });
        handlers.onEvent({ TextDelta: 'b' });
        handlers.onEvent({ Citations: citations });
        handlers.onEvent({ Done: { tokens_used: 42 } });
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

  it('cancels an in-flight turn before starting a new send (single-flight)', async () => {
    vi.mocked(saveChatUser).mockResolvedValue(makeChatMessage({ id: 'u1' }));
    // First send's stream never reaches Done/onStreamDone/onFailed, so `streaming`
    // stays true after it resolves — simulating a still-in-flight turn.
    vi.mocked(askNotebook).mockImplementationOnce(
      async (_nb: string, _q: string, handlers: AskNotebookHandlers) => {
        handlers.onEvent({ Stage: 'Retrieving' });
      }
    );

    await send(NB, 'first');
    expect(chatStore.streaming(NB)).toBe(true);

    vi.mocked(askNotebook).mockImplementationOnce(
      async (_nb: string, _q: string, handlers: AskNotebookHandlers) => {
        handlers.onEvent({ Done: { tokens_used: 1 } });
        handlers.onStreamDone();
      }
    );
    vi.mocked(saveChatAssistant).mockResolvedValue(
      makeChatMessage({ id: 'a-second', role: 'assistant' })
    );

    await send(NB, 'second');
    expect(cancelAsk).toHaveBeenCalledWith(NB);
  });
});

describe('cancel path', () => {
  it('keeps partial answerBuffer in-session and persists nothing', async () => {
    vi.mocked(saveChatUser).mockResolvedValue(makeChatMessage({ id: 'u1' }));
    vi.mocked(askNotebook).mockImplementation(
      async (_nb: string, _q: string, handlers: AskNotebookHandlers) => {
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
});

describe('error path', () => {
  it('sets error state and persists nothing on a non-cancel failure', async () => {
    vi.mocked(saveChatUser).mockResolvedValue(makeChatMessage({ id: 'u1' }));
    vi.mocked(askNotebook).mockImplementation(
      async (_nb: string, _q: string, handlers: AskNotebookHandlers) => {
        handlers.onEvent({ TextDelta: 'x' });
        handlers.onFailed({ kind: 'Internal', message: 'boom' });
      }
    );

    await send(NB, 'q');

    expect(saveChatAssistant).not.toHaveBeenCalled();
    expect(chatStore.error(NB)).toEqual({ kind: 'Internal', message: 'boom' });
    expect(chatStore.streaming(NB)).toBe(false);
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
      async (_nb: string, question: string, handlers: AskNotebookHandlers) => {
        expect(question).toBe('original q');
        handlers.onEvent({ TextDelta: 'answer v1' });
        handlers.onEvent({ Done: { tokens_used: 10 } });
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
});
