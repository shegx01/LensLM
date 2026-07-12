import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ChatTranscript from './ChatTranscript.svelte';
import type { Turn } from '$lib/chat/types.js';
import { makeChatMessage } from '$lib/chat/test-fixtures.js';

afterEach(() => {
  vi.restoreAllMocks();
});

function userRow(overrides: Partial<ReturnType<typeof makeChatMessage>> = {}) {
  return makeChatMessage({
    id: 'msg-user-1',
    notebook_id: 'nb-1',
    turn_id: 'turn-1',
    role: 'user',
    content: 'What were the key drivers?',
    created_at: '2026-07-12T00:00:00Z',
    ...overrides
  });
}

function assistantRow(overrides: Partial<ReturnType<typeof makeChatMessage>> = {}) {
  return makeChatMessage({
    id: 'msg-assistant-1',
    notebook_id: 'nb-1',
    turn_id: 'turn-1',
    role: 'assistant',
    content: 'Three primary drivers stand out.',
    tokens_used: 42,
    created_at: '2026-07-12T00:00:01Z',
    ...overrides
  });
}

const baseProps = {
  streaming: false,
  stage: null,
  thinkingBuffer: '',
  answerBuffer: '',
  currentTurnId: null,
  error: null,
  pinnedToBottom: true,
  oncopy: vi.fn(),
  onregenerate: vi.fn(),
  onfeedback: vi.fn(),
  onretry: vi.fn(),
  onunpin: vi.fn(),
  onjumptolatest: vi.fn()
};

describe('ChatTranscript', () => {
  it('shows the empty state when there are no turns and not streaming (AC20)', () => {
    render(ChatTranscript, { props: { ...baseProps, turns: [] } });
    expect(screen.getByText('Ask anything about your sources')).toBeInTheDocument();
    expect(screen.getByText(/grounded in this notebook's selected sources/i)).toBeInTheDocument();
  });

  it('does not show the empty state while streaming even with no turns yet', () => {
    render(ChatTranscript, { props: { ...baseProps, turns: [], streaming: true } });
    expect(screen.queryByText('Ask anything about your sources')).not.toBeInTheDocument();
  });

  it('a zero-version turn renders the user bubble alone (no assistant slot / pager / actions)', () => {
    const turns: Turn[] = [{ turn_id: 'turn-1', user: userRow(), versions: [] }];
    render(ChatTranscript, { props: { ...baseProps, turns } });

    expect(screen.getByText('What were the key drivers?')).toBeInTheDocument();
    expect(screen.queryByLabelText('Copy answer')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Regenerate answer')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Good response')).not.toBeInTheDocument();
  });

  it('a turn with one assistant version renders content but no pager', () => {
    const turns: Turn[] = [{ turn_id: 'turn-1', user: userRow(), versions: [assistantRow()] }];
    render(ChatTranscript, { props: { ...baseProps, turns } });

    expect(screen.getByText(/three primary drivers/i)).toBeInTheDocument();
    expect(screen.queryByLabelText(/previous version/i)).not.toBeInTheDocument();
  });

  it('a turn with multiple assistant versions shows the < n/m > pager', () => {
    const turns: Turn[] = [
      {
        turn_id: 'turn-1',
        user: userRow(),
        versions: [assistantRow({ id: 'v1' }), assistantRow({ id: 'v2', content: 'Second take.' })]
      }
    ];
    render(ChatTranscript, { props: { ...baseProps, turns } });
    expect(screen.getByText('2/2')).toBeInTheDocument();
  });

  it('shows "Jump to latest" when unpinned, and clicking it calls onjumptolatest', async () => {
    const onjumptolatest = vi.fn();
    const turns: Turn[] = [{ turn_id: 'turn-1', user: userRow(), versions: [assistantRow()] }];
    render(ChatTranscript, {
      props: { ...baseProps, turns, pinnedToBottom: false, onjumptolatest }
    });

    const jumpButton = screen.getByLabelText('Jump to latest message');
    expect(jumpButton).toBeInTheDocument();
    await fireEvent.click(jumpButton);
    expect(onjumptolatest).toHaveBeenCalledOnce();
  });

  it('does not show "Jump to latest" while pinned', () => {
    const turns: Turn[] = [{ turn_id: 'turn-1', user: userRow(), versions: [assistantRow()] }];
    render(ChatTranscript, { props: { ...baseProps, turns, pinnedToBottom: true } });
    expect(screen.queryByLabelText('Jump to latest message')).not.toBeInTheDocument();
  });

  it('renders an ErrorCard with Retry in the real post-failure state (streaming=false, error set)', async () => {
    const onretry = vi.fn();
    const turns: Turn[] = [{ turn_id: 'turn-1', user: userRow(), versions: [] }];
    render(ChatTranscript, {
      props: {
        ...baseProps,
        turns,
        streaming: false,
        currentTurnId: 'turn-1',
        error: { kind: 'Internal', message: 'stream failed' },
        onretry
      }
    });

    expect(screen.getByText('stream failed')).toBeInTheDocument();
    await fireEvent.click(screen.getByText('Retry'));
    expect(onretry).toHaveBeenCalledOnce();
  });

  it('does not render an ErrorCard for a turn other than the failed currentTurnId', () => {
    const turns: Turn[] = [
      { turn_id: 'turn-1', user: userRow(), versions: [] },
      { turn_id: 'turn-2', user: userRow({ id: 'msg-user-2', turn_id: 'turn-2' }), versions: [] }
    ];
    render(ChatTranscript, {
      props: {
        ...baseProps,
        turns,
        streaming: false,
        currentTurnId: 'turn-1',
        error: { kind: 'Internal', message: 'stream failed' }
      }
    });

    expect(screen.getByText('stream failed')).toBeInTheDocument();
    expect(screen.getAllByText('stream failed')).toHaveLength(1);
  });
});
