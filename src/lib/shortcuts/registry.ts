// Descriptive mirror of shortcuts hard-coded in their handlers below — not an
// authoritative binding source. #239 will introduce a central dispatcher +
// AppConfig keymap that binds to this registry.

export interface ShortcutEntry {
  group: 'Global' | 'Chat' | 'Audio player';
  action: string;
  keys: string[];
  description: string;
}

export const SHORTCUTS: readonly ShortcutEntry[] = [
  // AppShell.svelte:55-68 — ⌘K toggles the command palette open/closed.
  {
    group: 'Global',
    action: 'Toggle command palette',
    keys: ['⌘K'],
    description: 'Opens quick search across notebooks and notes, or closes it if already open.'
  },
  // CommandPalette.svelte:100-106 — Escape closes the palette while it has focus.
  {
    group: 'Global',
    action: 'Close command palette',
    keys: ['Escape'],
    description: 'Closes the command palette.'
  },
  // ChatComposer.svelte:74-79 — Enter sends unless Shift is held.
  {
    group: 'Chat',
    action: 'Send message',
    keys: ['Enter'],
    description: 'Sends the current message (ignored while empty or whitespace-only).'
  },
  {
    group: 'Chat',
    action: 'Insert newline',
    keys: ['Shift+Enter'],
    description: 'Adds a line break in the composer without sending.'
  },
  // AudioPlayer.svelte:98-123 — transport shortcuts scoped to the player's focus.
  {
    group: 'Audio player',
    action: 'Play or pause',
    keys: ['Space'],
    description: 'Toggles playback of the audio overview.'
  },
  {
    group: 'Audio player',
    action: 'Seek',
    keys: ['←', '→'],
    description: 'Seeks 5 seconds back or forward.'
  },
  {
    group: 'Audio player',
    action: 'Skip',
    keys: ['J', 'L'],
    description: 'Skips 15 seconds back or forward.'
  },
  {
    group: 'Audio player',
    action: 'Playback speed',
    keys: ['[', ']'],
    description: 'Decreases or increases playback speed.'
  }
];
