import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import type { Turn, ChatMessage } from '$lib/chat/types.js';
import ChatScrubber from './ChatScrubber.svelte';

function userMsg(turnId: string, content: string): ChatMessage {
  return {
    id: `${turnId}-u`,
    notebook_id: 'nb1',
    turn_id: turnId,
    role: 'user',
    content,
    citations: null,
    feedback: null,
    tokens_used: null,
    created_at: '2026-07-12T00:00:00Z'
  };
}

function turn(turnId: string, content: string): Turn {
  return { turn_id: turnId, user: userMsg(turnId, content), versions: [] };
}

const turns = [turn('t1', 'What are the H2 risks?'), turn('t2', 'How do I define the form?')];

describe('ChatScrubber', () => {
  it('renders one tick per turn with a question preview label', () => {
    render(ChatScrubber, { turns, activeTurnId: 't1', onjump: vi.fn() });
    expect(
      screen.getByRole('button', { name: 'Jump to: What are the H2 risks?' })
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'Jump to: How do I define the form?' })
    ).toBeInTheDocument();
  });

  it('marks the active turn with aria-current', () => {
    render(ChatScrubber, { turns, activeTurnId: 't2', onjump: vi.fn() });
    const active = screen.getByRole('button', { name: 'Jump to: How do I define the form?' });
    expect(active).toHaveAttribute('aria-current', 'true');
    const inactive = screen.getByRole('button', { name: 'Jump to: What are the H2 risks?' });
    expect(inactive).not.toHaveAttribute('aria-current');
  });

  it('fires onjump with the turn id when a tick is clicked', async () => {
    const onjump = vi.fn();
    render(ChatScrubber, { turns, activeTurnId: 't1', onjump });
    await fireEvent.click(
      screen.getByRole('button', { name: 'Jump to: How do I define the form?' })
    );
    expect(onjump).toHaveBeenCalledWith('t2');
  });

  it('renders nothing for a single turn (nothing to navigate)', () => {
    render(ChatScrubber, {
      turns: [turn('t1', 'only one')],
      activeTurnId: 't1',
      onjump: vi.fn()
    });
    // No nav landmark and no jump targets when there's only one turn.
    expect(screen.queryByRole('navigation', { name: 'Conversation timeline' })).toBeNull();
    expect(screen.queryByRole('button', { name: /^Jump to:/ })).toBeNull();
  });
});
