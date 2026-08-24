import { beforeEach, describe, expect, it } from 'vitest';
import { resolve } from './dispatcher.js';
import { setPlatform } from './platform.js';
import type { ActionId } from './registry.js';
import type { KeyEventLike } from './binding.js';

// Ambient detection under happy-dom resolves to non-darwin even on macOS; every case
// here turns on modifier resolution, so pin it.
beforeEach(() => {
  setPlatform('darwin');
});

type Keymap = Partial<Record<ActionId, string>>;

const ev = (key: string, mods: Omit<KeyEventLike, 'key'> = {}): KeyEventLike => ({ key, ...mods });

describe('resolve', () => {
  it('matches an in-scope default binding', () => {
    expect(resolve(ev('k', { metaKey: true }), 'window', {}, 'darwin')).toBe('palette.toggle');
  });

  it('resolves the same action from ctrlKey on linux', () => {
    expect(resolve(ev('k', { ctrlKey: true }), 'window', {}, 'linux')).toBe('palette.toggle');
  });

  it('ignores entries outside the requested scope', () => {
    expect(resolve(ev('k', { metaKey: true }), 'player', {}, 'darwin')).toBeNull();
    expect(resolve(ev('l'), 'window', {}, 'darwin')).toBeNull();
  });

  it('matches a bare-key binding in its own scope', () => {
    expect(resolve(ev('l'), 'player', {}, 'darwin')).toBe('player.skipFwd');
  });

  it('requires exact modifier equality — Mod+L is not player.skipFwd', () => {
    expect(resolve(ev('l', { metaKey: true }), 'player', {}, 'darwin')).toBeNull();
  });

  // Deliberate behaviour change: AudioPlayer.svelte:100 lowercases e.key, so Shift+J
  // fires skipBack today. Exact-modifier equality removes that.
  it('requires exact modifier equality — Shift+J is not player.skipBack', () => {
    expect(resolve(ev('J', { shiftKey: true }), 'player', {}, 'darwin')).toBeNull();
    expect(resolve(ev('j'), 'player', {}, 'darwin')).toBe('player.skipBack');
  });

  it('lets a keymap override beat the default binding', () => {
    const keymap: Keymap = { 'palette.toggle': 'Mod+P' };
    expect(resolve(ev('p', { metaKey: true }), 'window', keymap, 'darwin')).toBe('palette.toggle');
    expect(resolve(ev('k', { metaKey: true }), 'window', keymap, 'darwin')).toBeNull();
  });

  it('honours a bare-letter override', () => {
    const keymap: Keymap = { 'palette.toggle': 'P' };
    expect(resolve(ev('p'), 'window', keymap, 'darwin')).toBe('palette.toggle');
  });

  it('ignores an unknown ActionId in the keymap without throwing', () => {
    const keymap = { 'not.an.action': 'Mod+K' } as unknown as Keymap;
    expect(() => resolve(ev('k', { metaKey: true }), 'window', keymap, 'darwin')).not.toThrow();
    expect(resolve(ev('k', { metaKey: true }), 'window', keymap, 'darwin')).toBe('palette.toggle');
  });

  it('ignores a malformed token without throwing and without matching', () => {
    const keymap: Keymap = { 'palette.toggle': '???' };
    expect(() => resolve(ev('k', { metaKey: true }), 'window', keymap, 'darwin')).not.toThrow();
    expect(resolve(ev('k', { metaKey: true }), 'window', keymap, 'darwin')).toBeNull();
  });

  it('returns null when fromEvent cannot represent the event', () => {
    expect(resolve(ev('Shift', { shiftKey: true }), 'window', {}, 'darwin')).toBeNull();
    expect(resolve(ev('F5'), 'window', {}, 'darwin')).toBeNull();
    expect(resolve(ev('k', { metaKey: true, ctrlKey: true }), 'window', {}, 'darwin')).toBeNull();
    expect(resolve(ev('k', { metaKey: true, ctrlKey: true }), 'window', {}, 'linux')).toBeNull();
  });

  it('returns null when nothing matches', () => {
    expect(resolve(ev('z', { metaKey: true }), 'window', {}, 'darwin')).toBeNull();
  });
});
