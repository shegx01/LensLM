import { describe, expect, it } from 'vitest';
import { parse, type Binding } from './binding.js';
import { findConflict, isValidForScope } from './conflicts.js';

function b(token: string): Binding {
  const parsed = parse(token);
  if (parsed === null) throw new Error(`unparseable test token: ${token}`);
  return parsed;
}

describe('findConflict', () => {
  it('reports a collision inside the candidate own scope', () => {
    expect(findConflict('player', b('Space'), 'player.skipFwd', {})?.id).toBe('player.playPause');
  });

  it('reports a player collision for a window candidate (window is a universal domain)', () => {
    expect(findConflict('window', b('Space'), 'palette.toggle', {})?.id).toBe('player.playPause');
  });

  it('reports a window collision for a player candidate (the other direction)', () => {
    expect(findConflict('player', b('Mod+K'), 'player.skipFwd', {})?.id).toBe('palette.toggle');
  });

  it('keeps non-window scopes mutually disjoint: palette never collides with player', () => {
    expect(findConflict('palette', b('Space'), 'palette.close', {})).toBeNull();
  });

  it('never reports the action being edited against itself', () => {
    expect(findConflict('player', b('Space'), 'player.playPause', {})).toBeNull();
  });

  it('sees reserved actions as occupied keys (pure property; unreachable from the panel)', () => {
    expect(findConflict('composer', b('Enter'), 'chat.newline', {})?.id).toBe('chat.send');
  });

  it('compares against the effective keymap, not the shipped defaults', () => {
    const keymap = { 'player.skipFwd': 'Q' } as const;

    expect(findConflict('player', b('Q'), 'player.skipBack', keymap)?.id).toBe('player.skipFwd');
    expect(findConflict('player', b('L'), 'player.skipBack', keymap)).toBeNull();
  });

  it('names the colliding action so the panel can quote it', () => {
    expect(findConflict('player', b('Space'), 'player.skipFwd', {})?.action).toBe('Play or pause');
  });

  it('ignores an unparseable stored token instead of throwing', () => {
    expect(
      findConflict('player', b('Space'), 'player.skipFwd', { 'player.skipBack': '???' })?.id
    ).toBe('player.playPause');
  });
});

describe('isValidForScope', () => {
  it('accepts a window candidate carrying Mod or Alt', () => {
    expect(isValidForScope('window', parse('Mod+Q'))).toBe(true);
    expect(isValidForScope('window', parse('Alt+Q'))).toBe(true);
  });

  it('rejects a typeable window candidate, Shift included', () => {
    expect(isValidForScope('window', parse('Q'))).toBe(false);
    expect(isValidForScope('window', parse('ArrowDown'))).toBe(false);
    expect(isValidForScope('window', parse('Shift+Q'))).toBe(false);
  });

  it('leaves bare keys valid outside window scope', () => {
    expect(isValidForScope('player', parse('Space'))).toBe(true);
    expect(isValidForScope('composer', parse('Enter'))).toBe(true);
  });

  it('rejects an unrecordable keystroke', () => {
    expect(isValidForScope('player', null)).toBe(false);
  });
});
