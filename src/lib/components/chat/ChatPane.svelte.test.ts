import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { makeTurn } from '$lib/chat/test-fixtures.js';

const failedTurn = makeTurn({
  turn_id: 'turn-failed',
  user: {
    id: 'u1',
    notebook_id: 'nb-1',
    turn_id: 'turn-failed',
    role: 'user',
    content: 'what happened?',
    citations: null,
    feedback: null,
    tokens_used: null,
    state: null,
    error_kind: null,
    created_at: '2026-07-12T00:00:00Z'
  },
  versions: []
});

vi.mock('$lib/chat/chat-state.svelte.js', () => ({
  chatStore: {
    turns: vi.fn(() => [failedTurn]),
    // Real post-failure state: streaming=false, error set, currentTurnId matches
    // the failed turn (onFailed's non-cancel path — see chat-state.svelte.ts).
    streaming: vi.fn(() => false),
    stage: vi.fn(() => null),
    thinkingBuffer: vi.fn(() => ''),
    answerBuffer: vi.fn(() => ''),
    pendingCitations: vi.fn(() => null),
    currentTurnId: vi.fn(() => 'turn-failed'),
    ungroundedTurnId: vi.fn(() => null),
    error: vi.fn(() => ({ kind: 'Internal', message: 'stream failed' })),
    reindexing: vi.fn(() => false),
    pinnedToBottom: vi.fn(() => true)
  },
  hydrate: vi.fn().mockResolvedValue(undefined),
  send: vi.fn(),
  stop: vi.fn(),
  regenerate: vi.fn(),
  setFeedback: vi.fn(),
  copyMessage: vi.fn(),
  jumpToLatest: vi.fn(),
  unpin: vi.fn()
}));

import ChatPane from './ChatPane.svelte';
import { send, regenerate } from '$lib/chat/chat-state.svelte.js';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('ChatPane retry wiring', () => {
  it('routes Retry through regenerate(notebookId, failedTurnId), not send', async () => {
    render(ChatPane, { props: { notebookId: 'nb-1' } });

    const retryButton = await screen.findByText('Retry');
    await fireEvent.click(retryButton);

    expect(regenerate).toHaveBeenCalledWith('nb-1', 'turn-failed');
    expect(send).not.toHaveBeenCalled();
  });
});
