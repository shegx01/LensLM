import { fireEvent, render, screen } from '@testing-library/svelte';
import type { ComponentProps } from 'svelte';
import { describe, expect, it, vi } from 'vitest';
import { parse } from '$lib/shortcuts/binding.js';
import ShortcutRow from './ShortcutRow.svelte';

type Props = ComponentProps<typeof ShortcutRow>;

function props(overrides: Partial<Props> = {}): Props {
  return {
    label: 'Toggle command palette',
    description: 'Opens quick search across notebooks and notes.',
    chips: [
      {
        id: 'palette.toggle',
        action: 'Toggle command palette',
        token: 'Mod+K',
        overridden: false,
        remappable: true
      }
    ],
    platform: 'darwin',
    armedId: null,
    candidate: null,
    message: null,
    onarm: () => {},
    oncapture: () => {},
    ondisarm: () => {},
    onreset: () => {},
    ...overrides
  };
}

const pairedChips: Props['chips'] = [
  {
    id: 'player.seekBack',
    action: 'Seek back',
    token: 'ArrowLeft',
    overridden: false,
    remappable: true
  },
  {
    id: 'player.seekFwd',
    action: 'Seek forward',
    token: 'ArrowRight',
    overridden: false,
    remappable: true
  }
];

describe('ShortcutRow', () => {
  it('renders the label, description and the effective chip when idle', () => {
    render(ShortcutRow, { props: props() });

    expect(screen.getByText('Toggle command palette')).toBeInTheDocument();
    expect(screen.getByText('⌘K')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('offers a per-chip edit affordance for every remappable id in the row', () => {
    render(ShortcutRow, { props: props({ label: 'Seek', chips: pairedChips }) });

    expect(
      screen.getByRole('button', { name: /change shortcut for seek back/i })
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /change shortcut for seek forward/i })
    ).toBeInTheDocument();
  });

  it('arms the clicked chip, not the row', async () => {
    const onarm = vi.fn();
    render(ShortcutRow, { props: props({ label: 'Seek', chips: pairedChips, onarm }) });

    await fireEvent.click(
      screen.getByRole('button', { name: /change shortcut for seek forward/i })
    );

    expect(onarm).toHaveBeenCalledWith('player.seekFwd');
  });

  it('shows the candidate in place of the current chip while armed', () => {
    render(ShortcutRow, {
      props: props({ armedId: 'palette.toggle', candidate: parse('Mod+Shift+P') })
    });

    expect(screen.getByText('⌘⇧P')).toBeInTheDocument();
    expect(screen.queryByText('⌘K')).not.toBeInTheDocument();
  });

  it('prompts for a keystroke while armed with nothing captured yet', () => {
    render(ShortcutRow, { props: props({ armedId: 'palette.toggle' }) });

    expect(screen.getByText(/press a key/i)).toBeInTheDocument();
    expect(screen.queryByText('⌘K')).not.toBeInTheDocument();
  });

  it('forwards keystrokes and blur while armed', async () => {
    const oncapture = vi.fn();
    const ondisarm = vi.fn();
    render(ShortcutRow, { props: props({ armedId: 'palette.toggle', oncapture, ondisarm }) });

    const button = screen.getByRole('button', {
      name: /change shortcut for toggle command palette/i
    });
    await fireEvent.keyDown(button, { key: 'p', metaKey: true });
    expect(oncapture).toHaveBeenCalledTimes(1);

    await fireEvent.blur(button);
    expect(ondisarm).toHaveBeenCalledTimes(1);
  });

  it('does not forward keystrokes while idle', async () => {
    const oncapture = vi.fn();
    render(ShortcutRow, { props: props({ oncapture }) });

    await fireEvent.keyDown(
      screen.getByRole('button', { name: /change shortcut for toggle command palette/i }),
      { key: 'p', metaKey: true }
    );

    expect(oncapture).not.toHaveBeenCalled();
  });

  it('announces the inline message', () => {
    render(ShortcutRow, {
      props: props({
        armedId: 'palette.toggle',
        candidate: parse('Space'),
        message: 'Already used by “Play or pause”.'
      })
    });

    expect(screen.getByRole('alert')).toHaveTextContent(/already used by/i);
  });

  it('renders a reserved row as static, with no edit affordance and a conventional label', () => {
    render(ShortcutRow, {
      props: props({
        label: 'Send message',
        description: 'Sends the current message.',
        chips: [
          {
            id: 'chat.send',
            action: 'Send message',
            token: 'Enter',
            overridden: false,
            remappable: false
          }
        ]
      })
    });

    expect(screen.getByText('Enter')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /change shortcut/i })).not.toBeInTheDocument();
    expect(screen.getByText(/conventional/i)).toBeInTheDocument();
  });

  it('offers a reset affordance only for an overridden chip', async () => {
    const onreset = vi.fn();
    const { rerender } = render(ShortcutRow, { props: props({ onreset }) });

    expect(screen.queryByRole('button', { name: /reset/i })).not.toBeInTheDocument();

    await rerender(
      props({
        onreset,
        chips: [
          {
            id: 'palette.toggle',
            action: 'Toggle command palette',
            token: 'Mod+P',
            overridden: true,
            remappable: true
          }
        ]
      })
    );

    const reset = screen.getByRole('button', { name: /reset toggle command palette/i });
    await fireEvent.click(reset);
    expect(onreset).toHaveBeenCalledWith('palette.toggle');
  });
});
