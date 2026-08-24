export const ACTION_IDS = [
  'palette.toggle',
  'palette.close',
  'chat.send',
  'chat.newline',
  'player.playPause',
  'player.seekBack',
  'player.seekFwd',
  'player.skipBack',
  'player.skipFwd',
  'player.rateDown',
  'player.rateUp'
] as const;

export type ActionId = (typeof ACTION_IDS)[number];

export type Scope = 'window' | 'palette' | 'player' | 'composer';

export const GROUP_ORDER = ['Global', 'Chat', 'Audio player'] as const;

export type Group = (typeof GROUP_ORDER)[number];

export interface ShortcutEntry {
  id: ActionId;
  group: Group;
  scope: Scope;
  action: string;
  defaultBinding: string;
  description: string;
  remappable: boolean;
}

// Conflict detection blocks only *user* edits, so nothing at runtime guards these:
// the shipped defaults must stay collision-free across `window` ∪ every other scope,
// because player keys bubble to the window listener.
const SHORTCUTS_BY_ID = {
  'palette.toggle': {
    id: 'palette.toggle',
    group: 'Global',
    scope: 'window',
    action: 'Toggle command palette',
    defaultBinding: 'Mod+K',
    description: 'Opens quick search across notebooks and notes, or closes it if already open.',
    remappable: true
  },
  'palette.close': {
    id: 'palette.close',
    group: 'Global',
    scope: 'palette',
    action: 'Close command palette',
    defaultBinding: 'Escape',
    description: 'Closes the command palette.',
    remappable: false
  },
  'chat.send': {
    id: 'chat.send',
    group: 'Chat',
    scope: 'composer',
    action: 'Send message',
    defaultBinding: 'Enter',
    description: 'Sends the current message (ignored while empty or whitespace-only).',
    remappable: false
  },
  'chat.newline': {
    id: 'chat.newline',
    group: 'Chat',
    scope: 'composer',
    action: 'Insert newline',
    defaultBinding: 'Shift+Enter',
    description: 'Adds a line break in the composer without sending.',
    remappable: false
  },
  'player.playPause': {
    id: 'player.playPause',
    group: 'Audio player',
    scope: 'player',
    action: 'Play or pause',
    defaultBinding: 'Space',
    description: 'Toggles playback of the audio overview.',
    remappable: true
  },
  'player.seekBack': {
    id: 'player.seekBack',
    group: 'Audio player',
    scope: 'player',
    action: 'Seek back',
    defaultBinding: 'ArrowLeft',
    description: 'Seeks 5 seconds back.',
    remappable: true
  },
  'player.seekFwd': {
    id: 'player.seekFwd',
    group: 'Audio player',
    scope: 'player',
    action: 'Seek forward',
    defaultBinding: 'ArrowRight',
    description: 'Seeks 5 seconds forward.',
    remappable: true
  },
  'player.skipBack': {
    id: 'player.skipBack',
    group: 'Audio player',
    scope: 'player',
    action: 'Skip back',
    defaultBinding: 'J',
    description: 'Skips 15 seconds back.',
    remappable: true
  },
  'player.skipFwd': {
    id: 'player.skipFwd',
    group: 'Audio player',
    scope: 'player',
    action: 'Skip forward',
    defaultBinding: 'L',
    description: 'Skips 15 seconds forward.',
    remappable: true
  },
  'player.rateDown': {
    id: 'player.rateDown',
    group: 'Audio player',
    scope: 'player',
    action: 'Decrease playback speed',
    defaultBinding: '[',
    description: 'Steps playback speed down.',
    remappable: true
  },
  'player.rateUp': {
    id: 'player.rateUp',
    group: 'Audio player',
    scope: 'player',
    action: 'Increase playback speed',
    defaultBinding: ']',
    description: 'Steps playback speed up.',
    remappable: true
  }
} satisfies Record<ActionId, ShortcutEntry>;

export const SHORTCUTS: readonly ShortcutEntry[] = Object.values(SHORTCUTS_BY_ID);

export interface ShortcutRow {
  group: Group;
  label: string;
  description: string;
  ids: readonly ActionId[];
}

export const ROWS: readonly ShortcutRow[] = [
  {
    group: 'Global',
    label: 'Toggle command palette',
    description: 'Opens quick search across notebooks and notes, or closes it if already open.',
    ids: ['palette.toggle']
  },
  {
    group: 'Global',
    label: 'Close command palette',
    description: 'Closes the command palette.',
    ids: ['palette.close']
  },
  {
    group: 'Chat',
    label: 'Send message',
    description: 'Sends the current message (ignored while empty or whitespace-only).',
    ids: ['chat.send']
  },
  {
    group: 'Chat',
    label: 'Insert newline',
    description: 'Adds a line break in the composer without sending.',
    ids: ['chat.newline']
  },
  {
    group: 'Audio player',
    label: 'Play or pause',
    description: 'Toggles playback of the audio overview.',
    ids: ['player.playPause']
  },
  {
    group: 'Audio player',
    label: 'Seek',
    description: 'Seeks 5 seconds back or forward.',
    ids: ['player.seekBack', 'player.seekFwd']
  },
  {
    group: 'Audio player',
    label: 'Skip',
    description: 'Skips 15 seconds back or forward.',
    ids: ['player.skipBack', 'player.skipFwd']
  },
  {
    group: 'Audio player',
    label: 'Playback speed',
    description: 'Decreases or increases playback speed.',
    ids: ['player.rateDown', 'player.rateUp']
  }
];
