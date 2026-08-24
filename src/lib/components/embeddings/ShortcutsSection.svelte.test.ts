import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { clearMocks, mockIPC } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { appConfigStore, resetConfig } from '$lib/models/app-config.svelte.js';
import { resolve } from '$lib/shortcuts/dispatcher.js';
import { setPlatform } from '$lib/shortcuts/platform.js';
import { ROWS } from '$lib/shortcuts/registry.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import type { AppConfig } from '$lib/theme/types.js';
import ShortcutsSection from './ShortcutsSection.svelte';

// persist() and load() both early-return on !isTauri(), so without this flag every
// "writes nothing" assertion below passes trivially.
beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
  // Chips are platform-rendered; the frontend CI job runs on ubuntu, and happy-dom's
  // navigator is not macOS-shaped even on a mac, so the glyphs need the platform pinned.
  setPlatform('darwin');
  mockIPC((cmd) => (cmd === 'get_config' ? baseAppConfig() : undefined));
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
  setPlatform(null);
  resetConfig();
});

/** mockIPC that serves `keymap` and records the last written config. */
function recordingIpc(keymap: AppConfig['keymap'] = {}) {
  const written: { config: AppConfig | null } = { config: null };
  let current = baseAppConfig({ keymap });
  mockIPC((cmd, args) => {
    if (cmd === 'get_config') return current;
    if (cmd === 'set_config') {
      current = (args as { config: AppConfig }).config;
      written.config = current;
    }
    return undefined;
  });
  return written;
}

function chip(action: string): HTMLElement {
  return screen.getByRole('button', { name: new RegExp(`change shortcut for ${action}`, 'i') });
}

describe('ShortcutsSection', () => {
  it('renders a heading for every group present in the registry', () => {
    render(ShortcutsSection);

    for (const group of ['Global', 'Chat', 'Audio player']) {
      expect(screen.getByText(group)).toBeInTheDocument();
    }
  });

  it('renders representative keys from each group', () => {
    render(ShortcutsSection);

    expect(screen.getByText('⌘K')).toBeInTheDocument();
    expect(screen.getByText('Space')).toBeInTheDocument();
    expect(screen.getByText('J')).toBeInTheDocument();
    expect(screen.getByText('L')).toBeInTheDocument();
  });

  it('renders exactly one row per display row (drift guard)', () => {
    const { container } = render(ShortcutsSection);

    const rows = container.querySelectorAll('[data-shortcut-row]');
    expect(rows.length).toBe(ROWS.length);
  });

  it('renders persisted overrides instead of the shipped defaults', async () => {
    recordingIpc({ 'player.skipFwd': 'Q' });

    render(ShortcutsSection);

    await waitFor(() => expect(screen.getByText('Q')).toBeInTheDocument());
    expect(screen.queryByText('L')).not.toBeInTheDocument();
  });

  it('renders the three reserved actions as static rows with no edit affordance', () => {
    render(ShortcutsSection);

    expect(
      screen.queryByRole('button', { name: /change shortcut for close command palette/i })
    ).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /change shortcut for send message/i })).toBeNull();
    expect(
      screen.queryByRole('button', { name: /change shortcut for insert newline/i })
    ).toBeNull();
    expect(screen.getAllByText(/conventional/i)).toHaveLength(3);
  });

  it('has no Save button', () => {
    render(ShortcutsSection);

    expect(screen.queryByRole('button', { name: /^save/i })).toBeNull();
  });
});

// Every negative write assertion below sits beside a passing positive one, so a missing
// `globalThis.isTauri` cannot turn the negatives green.
describe('ShortcutsSection rebinding', () => {
  it('accepting a candidate with Enter writes the new entry and spreads the rest of the config verbatim', async () => {
    const written = recordingIpc();
    render(ShortcutsSection);
    await waitFor(() => expect(appConfigStore.keymap).toEqual({}));

    await fireEvent.click(chip('skip forward'));
    await fireEvent.keyDown(chip('skip forward'), { key: 'q' });
    await waitFor(() => expect(screen.getByText('Q')).toBeInTheDocument());
    await fireEvent.keyDown(chip('skip forward'), { key: 'Enter' });

    await waitFor(() => expect(written.config).not.toBeNull());
    expect(written.config?.keymap).toEqual({ 'player.skipFwd': 'Q' });
    expect(written.config?.theme).toBe('dark');
    expect(written.config?.accent).toBe('purple');
    expect(written.config?.animations).toBe('system');
  });

  it('cancelling with Escape writes nothing and restores the chip', async () => {
    const written = recordingIpc();
    render(ShortcutsSection);
    await waitFor(() => expect(appConfigStore.keymap).toEqual({}));

    await fireEvent.click(chip('skip forward'));
    await fireEvent.keyDown(chip('skip forward'), { key: 'q' });
    await waitFor(() => expect(screen.getByText('Q')).toBeInTheDocument());
    await fireEvent.keyDown(chip('skip forward'), { key: 'Escape' });

    await waitFor(() => expect(screen.getByText('L')).toBeInTheDocument());
    expect(written.config).toBeNull();
  });

  it('blocks a player-vs-player collision, naming the occupying action, and writes nothing', async () => {
    const written = recordingIpc();
    render(ShortcutsSection);
    await waitFor(() => expect(appConfigStore.keymap).toEqual({}));

    await fireEvent.click(chip('skip forward'));
    await fireEvent.keyDown(chip('skip forward'), { key: ' ' });

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(/play or pause/i);

    await fireEvent.keyDown(chip('skip forward'), { key: 'Enter' });
    expect(written.config).toBeNull();
  });

  it('blocks a window-vs-player collision (window is a universal conflict domain)', async () => {
    const written = recordingIpc();
    render(ShortcutsSection);
    await waitFor(() => expect(appConfigStore.keymap).toEqual({}));

    await fireEvent.click(chip('toggle command palette'));
    await fireEvent.keyDown(chip('toggle command palette'), { key: ' ' });

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(/play or pause/i);

    await fireEvent.keyDown(chip('toggle command palette'), { key: 'Enter' });
    expect(written.config).toBeNull();
  });

  it('rejects a typeable window candidate and writes nothing', async () => {
    const written = recordingIpc();
    render(ShortcutsSection);
    await waitFor(() => expect(appConfigStore.keymap).toEqual({}));

    await fireEvent.click(chip('toggle command palette'));
    await fireEvent.keyDown(chip('toggle command palette'), { key: 'q' });

    expect(await screen.findByRole('alert')).toHaveTextContent(/modifier/i);

    // Shift+Q is how a capital Q is typed, so it is rejected on the same grounds.
    await fireEvent.keyDown(chip('toggle command palette'), { key: 'Q', shiftKey: true });
    expect(screen.getByRole('alert')).toHaveTextContent(/modifier/i);

    await fireEvent.keyDown(chip('toggle command palette'), { key: 'Enter' });
    expect(written.config).toBeNull();
  });

  it('accepts a modified window candidate', async () => {
    const written = recordingIpc();
    render(ShortcutsSection);
    await waitFor(() => expect(appConfigStore.keymap).toEqual({}));

    await fireEvent.click(chip('toggle command palette'));
    await fireEvent.keyDown(chip('toggle command palette'), { key: 'p', metaKey: true });
    await fireEvent.keyDown(chip('toggle command palette'), { key: 'Enter' });

    await waitFor(() => expect(written.config?.keymap).toEqual({ 'palette.toggle': 'Mod+P' }));
  });

  it('arms one row at a time', async () => {
    render(ShortcutsSection);

    await fireEvent.click(chip('skip forward'));
    expect(screen.getAllByText(/press a key/i)).toHaveLength(1);

    await fireEvent.click(chip('play or pause'));

    expect(screen.getAllByText(/press a key/i)).toHaveLength(1);
    expect(chip('play or pause')).toHaveAttribute('aria-pressed', 'true');
    expect(chip('skip forward')).toHaveAttribute('aria-pressed', 'false');
  });

  it('focuses the armed chip, so its element-scoped listener actually receives the keystrokes', async () => {
    render(ShortcutsSection);

    await fireEvent.click(chip('skip forward'));

    expect(document.activeElement).toBe(chip('skip forward'));
  });

  it('disarms on blur so a live-looking row can never record through the window listener', async () => {
    render(ShortcutsSection);

    await fireEvent.click(chip('skip forward'));
    expect(screen.getByText(/press a key/i)).toBeInTheDocument();

    await fireEvent.blur(chip('skip forward'));

    expect(screen.queryByText(/press a key/i)).toBeNull();
    expect(screen.getByText('L')).toBeInTheDocument();
  });

  it('stops an armed keystroke from reaching the window listener', async () => {
    let reachedWindow = 0;
    const spy = () => {
      reachedWindow += 1;
    };
    window.addEventListener('keydown', spy);
    try {
      render(ShortcutsSection);

      await fireEvent.click(chip('skip forward'));
      await fireEvent.keyDown(chip('skip forward'), { key: 'k', metaKey: true });

      expect(reachedWindow).toBe(0);
    } finally {
      window.removeEventListener('keydown', spy);
    }
  });

  it('leaves Tab uncaptured so an armed row cannot trap focus', async () => {
    render(ShortcutsSection);

    await fireEvent.click(chip('skip forward'));
    const event = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    chip('skip forward').dispatchEvent(event);

    expect(event.defaultPrevented).toBe(false);
    expect(screen.getByText(/press a key/i)).toBeInTheDocument();
  });

  it('takes effect for the dispatcher without a remount', async () => {
    recordingIpc();
    render(ShortcutsSection);
    await waitFor(() => expect(appConfigStore.keymap).toEqual({}));

    await fireEvent.click(chip('skip forward'));
    await fireEvent.keyDown(chip('skip forward'), { key: 'q' });
    await fireEvent.keyDown(chip('skip forward'), { key: 'Enter' });

    await waitFor(() =>
      expect(resolve({ key: 'q' }, 'player', appConfigStore.keymap, 'darwin')).toBe(
        'player.skipFwd'
      )
    );
  });
});

describe('ShortcutsSection resets', () => {
  it('per-row reset deletes only that key from the keymap', async () => {
    const written = recordingIpc({ 'player.skipFwd': 'Q', 'player.skipBack': 'B' });
    render(ShortcutsSection);
    await waitFor(() => expect(screen.getByText('Q')).toBeInTheDocument());

    await fireEvent.click(screen.getByRole('button', { name: /reset skip forward to default/i }));

    await waitFor(() => expect(written.config).not.toBeNull());
    expect(written.config?.keymap).toEqual({ 'player.skipBack': 'B' });
    await waitFor(() => expect(screen.getByText('L')).toBeInTheDocument());
  });

  it('offers no reset affordance for a row that is not overridden', async () => {
    recordingIpc({ 'player.skipFwd': 'Q' });
    render(ShortcutsSection);
    await waitFor(() => expect(screen.getByText('Q')).toBeInTheDocument());

    expect(
      screen.getByRole('button', { name: /reset skip forward to default/i })
    ).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /reset play or pause to default/i })).toBeNull();
  });

  it('Reset all clears the whole keymap and reverts every chip', async () => {
    const written = recordingIpc({ 'player.skipFwd': 'Q', 'palette.toggle': 'Mod+P' });
    render(ShortcutsSection);
    await waitFor(() => expect(screen.getByText('Q')).toBeInTheDocument());

    await fireEvent.click(screen.getByRole('button', { name: /reset all/i }));

    await waitFor(() => expect(written.config?.keymap).toEqual({}));
    await waitFor(() => expect(screen.getByText('L')).toBeInTheDocument());
    expect(screen.getByText('⌘K')).toBeInTheDocument();
  });

  it('disables Reset all when nothing is overridden', async () => {
    recordingIpc();
    render(ShortcutsSection);
    await waitFor(() => expect(appConfigStore.keymap).toEqual({}));

    expect(screen.getByRole('button', { name: /reset all/i })).toBeDisabled();
  });
});

describe('ShortcutsSection persist failures', () => {
  it('disarms, reverts the chip and surfaces the error when set_config rejects', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'set_config') throw new Error('write failed');
      return undefined;
    });
    render(ShortcutsSection);
    await waitFor(() => expect(appConfigStore.keymap).toEqual({}));

    await fireEvent.click(chip('skip forward'));
    await fireEvent.keyDown(chip('skip forward'), { key: 'q' });
    await fireEvent.keyDown(chip('skip forward'), { key: 'Enter' });

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/write failed/i));
    expect(screen.queryByText(/press a key/i)).toBeNull();
    expect(screen.getByText('L')).toBeInTheDocument();
    expect(screen.queryByText('Q')).toBeNull();
  });

  it('disarms and surfaces persistError when the write lands but the confirming re-read rejects', async () => {
    let written: AppConfig | undefined;
    let rereadAttempted = false;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') {
        if (written !== undefined && !rereadAttempted) {
          rereadAttempted = true;
          throw new Error('reread failed');
        }
        return written ?? baseAppConfig();
      }
      if (cmd === 'set_config') written = (args as { config: AppConfig }).config;
      return undefined;
    });
    render(ShortcutsSection);
    await waitFor(() => expect(appConfigStore.keymap).toEqual({}));

    await fireEvent.click(chip('skip forward'));
    await fireEvent.keyDown(chip('skip forward'), { key: 'q' });
    await fireEvent.keyDown(chip('skip forward'), { key: 'Enter' });

    await waitFor(() => expect(rereadAttempted).toBe(true));
    expect(written?.keymap).toEqual({ 'player.skipFwd': 'Q' });
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/reread failed/i));
    expect(screen.queryByText(/press a key/i)).toBeNull();
    // The write DID land, so the chip keeps the optimistic value rather than lying
    // about what is on disk — same contract as PrivacySection's consent toggles.
    expect(screen.getByText('Q')).toBeInTheDocument();
    expect(appConfigStore.loadError).toBeNull();
  });
});
