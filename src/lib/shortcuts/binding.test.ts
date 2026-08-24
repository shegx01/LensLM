import { describe, expect, it } from 'vitest';
import {
  eq,
  fromEvent,
  parse,
  render,
  describe as spoken,
  toToken,
  type Binding,
  type KeyEventLike,
  type Platform
} from './binding.js';

const binding = (key: string, mods: Partial<Omit<Binding, 'key'>> = {}): Binding => ({
  key,
  mod: false,
  shift: false,
  alt: false,
  ...mods
});

describe('parse', () => {
  it.each([
    ['Mod+K', binding('K', { mod: true })],
    ['Mod+Shift+P', binding('P', { mod: true, shift: true })],
    ['Shift+Mod+P', binding('P', { mod: true, shift: true })],
    ['Alt+K', binding('K', { alt: true })],
    ['Space', binding('Space')],
    ['Enter', binding('Enter')],
    ['Escape', binding('Escape')],
    ['ArrowLeft', binding('ArrowLeft')],
    ['ArrowRight', binding('ArrowRight')],
    ['ArrowUp', binding('ArrowUp')],
    ['ArrowDown', binding('ArrowDown')],
    ['[', binding('[')],
    [']', binding(']')],
    ['Shift+Enter', binding('Enter', { shift: true })],
    ['j', binding('J')],
    ['mod+shift+p', binding('P', { mod: true, shift: true })],
    ['space', binding('Space')]
  ])('canonicalizes %s', (token, expected) => {
    expect(parse(token)).toEqual(expected);
  });

  it.each(['', '+', 'Mod+', 'Mod+Shift', '???', 'Mod+K+L', 'Mod+Mod+K', 'F5', 'Ctrl+K'])(
    'returns null for %s',
    (token) => {
      expect(parse(token)).toBeNull();
    }
  );
});

describe('render', () => {
  it.each([
    ['Mod+K', 'darwin', '⌘K'],
    ['Mod+K', 'win32', 'Ctrl+K'],
    ['Mod+K', 'linux', 'Ctrl+K'],
    ['Shift+Enter', 'darwin', '⇧Enter'],
    ['Shift+Enter', 'win32', 'Shift+Enter'],
    ['Shift+Enter', 'linux', 'Shift+Enter'],
    ['Alt+K', 'darwin', '⌥K'],
    ['Alt+K', 'win32', 'Alt+K'],
    ['Alt+K', 'linux', 'Alt+K'],
    ['Mod+Shift+P', 'darwin', '⌘⇧P'],
    ['Mod+Shift+P', 'win32', 'Ctrl+Shift+P'],
    ['Mod+Shift+Alt+P', 'darwin', '⌘⇧⌥P'],
    ['Mod+Shift+Alt+P', 'linux', 'Ctrl+Shift+Alt+P']
  ] as [string, Platform, string][])('renders %s on %s as %s', (token, platform, expected) => {
    expect(render(token, platform)).toBe(expected);
  });

  it.each([
    ['Space', 'Space'],
    ['Enter', 'Enter'],
    ['Escape', 'Escape'],
    ['ArrowLeft', '←'],
    ['ArrowRight', '→'],
    ['ArrowUp', '↑'],
    ['ArrowDown', '↓'],
    ['[', '['],
    [']', ']'],
    ['J', 'J'],
    ['j', 'J']
  ])('renders the key %s identically on every platform as %s', (token, expected) => {
    for (const platform of ['darwin', 'win32', 'linux'] as Platform[]) {
      expect(render(token, platform)).toBe(expected);
    }
  });

  it('returns an unparseable token verbatim', () => {
    expect(render('???', 'darwin')).toBe('???');
  });
});

describe('describe', () => {
  it.each([
    ['Mod+K', 'darwin', 'Command plus K'],
    ['Mod+K', 'win32', 'Control plus K'],
    ['Mod+K', 'linux', 'Control plus K'],
    ['Shift+Enter', 'darwin', 'Shift plus Enter'],
    ['Shift+Enter', 'win32', 'Shift plus Enter'],
    ['Alt+K', 'darwin', 'Option plus K'],
    ['Alt+K', 'win32', 'Alt plus K'],
    ['Alt+K', 'linux', 'Alt plus K'],
    ['Mod+Shift+P', 'darwin', 'Command plus Shift plus P'],
    ['Mod+Shift+P', 'linux', 'Control plus Shift plus P']
  ] as [string, Platform, string][])('describes %s on %s as %s', (token, platform, expected) => {
    expect(spoken(token, platform)).toBe(expected);
  });

  it.each([
    ['Space', 'Space'],
    ['Enter', 'Enter'],
    ['Escape', 'Escape'],
    ['ArrowLeft', 'Left arrow'],
    ['ArrowRight', 'Right arrow'],
    ['ArrowUp', 'Up arrow'],
    ['ArrowDown', 'Down arrow'],
    ['[', 'Left bracket'],
    [']', 'Right bracket'],
    ['J', 'J']
  ])('describes the key %s identically on every platform as %s', (token, expected) => {
    for (const platform of ['darwin', 'win32', 'linux'] as Platform[]) {
      expect(spoken(token, platform)).toBe(expected);
    }
  });

  it('returns an unparseable token verbatim', () => {
    expect(spoken('???', 'linux')).toBe('???');
  });
});

describe('toToken', () => {
  it.each([
    ['Mod+K', 'Mod+K'],
    ['Mod+Shift+P', 'Mod+Shift+P'],
    ['Shift+Mod+P', 'Mod+Shift+P'],
    ['Shift+Alt+ArrowLeft', 'Shift+Alt+ArrowLeft'],
    ['Alt+Shift+ArrowLeft', 'Shift+Alt+ArrowLeft'],
    ['Mod+Shift+Alt+P', 'Mod+Shift+Alt+P'],
    ['[', '['],
    [']', ']'],
    ['Space', 'Space'],
    ['Enter', 'Enter'],
    ['Escape', 'Escape'],
    ['ArrowRight', 'ArrowRight'],
    ['ArrowUp', 'ArrowUp'],
    ['ArrowDown', 'ArrowDown'],
    ['j', 'J'],
    ['shift+enter', 'Shift+Enter']
  ])('serializes %s to the canonical token %s', (token, expected) => {
    expect(toToken(parse(token)!)).toBe(expected);
  });

  it('emits a storage form with no platform glyphs', () => {
    for (const token of ['Mod+K', 'Mod+Shift+Alt+P', 'Shift+Enter']) {
      expect(toToken(parse(token)!)).not.toMatch(/[⌘⇧⌥]/);
    }
  });

  it.each(['Mod+K', 'Mod+Shift+P', 'Shift+Alt+ArrowLeft', '[', 'Space'])(
    'round-trips %s through parse',
    (token) => {
      const canonical = toToken(parse(token)!);
      expect(eq(parse(canonical), parse(token))).toBe(true);
      expect(toToken(parse(canonical)!)).toBe(canonical);
    }
  );

  it.each([
    [{ key: 'p', metaKey: true, shiftKey: true }, 'darwin'],
    [{ key: 'k', ctrlKey: true }, 'linux'],
    [{ key: ' ' }, 'linux'],
    [{ key: 'ArrowLeft', altKey: true }, 'win32']
  ] as [KeyEventLike, Platform][])(
    'closes the capture loop parse(toToken(fromEvent(%o, %s)))',
    (event, platform) => {
      const captured = fromEvent(event, platform);
      expect(captured).not.toBeNull();
      expect(parse(toToken(captured!))).toEqual(captured);
    }
  );
});

describe('eq', () => {
  it('ignores modifier order in the source tokens', () => {
    expect(eq(parse('Shift+Mod+P'), parse('Mod+Shift+P'))).toBe(true);
  });

  it.each([
    ['J', 'Shift+J'],
    ['J', 'Mod+J'],
    ['J', 'Alt+J'],
    ['J', 'L'],
    ['Mod+J', 'Mod+Shift+J']
  ])('treats %s and %s as different', (a, b) => {
    expect(eq(parse(a), parse(b))).toBe(false);
  });

  it.each([
    [null, null],
    [null, 'J'],
    ['J', null]
  ])('is never true when either side is null (%s, %s)', (a, b) => {
    expect(eq(a === null ? null : parse(a), b === null ? null : parse(b))).toBe(false);
  });
});

describe('fromEvent', () => {
  it.each([
    [' ', 'Space'],
    ['Spacebar', 'Space'],
    ['ArrowLeft', 'ArrowLeft'],
    ['j', 'J'],
    ['[', '[']
  ])('maps event key %s to %s', (key, expected) => {
    expect(fromEvent({ key }, 'linux')).toEqual(binding(expected));
  });

  it('distinguishes a bare letter from its shifted form', () => {
    const bare = fromEvent({ key: 'j' }, 'darwin');
    const shifted = fromEvent({ key: 'J', shiftKey: true }, 'darwin');

    expect(bare).toEqual(binding('J'));
    expect(shifted).toEqual(binding('J', { shift: true }));
    expect(eq(bare, shifted)).toBe(false);
  });

  it('reads mod from metaKey on darwin and ctrlKey elsewhere', () => {
    expect(fromEvent({ key: 'k', metaKey: true }, 'darwin')).toEqual(binding('K', { mod: true }));
    expect(fromEvent({ key: 'k', ctrlKey: true }, 'linux')).toEqual(binding('K', { mod: true }));
    expect(fromEvent({ key: 'k', ctrlKey: true }, 'win32')).toEqual(binding('K', { mod: true }));
  });

  it.each([
    [{ key: ' ', ctrlKey: true }, 'darwin'],
    [{ key: 'k', ctrlKey: true }, 'darwin'],
    [{ key: 'k', metaKey: true }, 'linux'],
    [{ key: 'k', metaKey: true }, 'win32']
  ] as [KeyEventLike, Platform][])(
    'rejects %o on %s rather than degrading to a weaker binding',
    (event, platform) => {
      expect(fromEvent(event, platform)).toBeNull();
    }
  );

  it('still accepts the representable primary modifier on each platform', () => {
    expect(fromEvent({ key: 'k', ctrlKey: true }, 'linux')).toEqual(binding('K', { mod: true }));
    expect(fromEvent({ key: 'k', metaKey: true }, 'darwin')).toEqual(binding('K', { mod: true }));
  });

  it('reads altKey', () => {
    expect(fromEvent({ key: 'k', altKey: true }, 'darwin')).toEqual(binding('K', { alt: true }));
  });

  it.each(['Shift', 'Alt', 'Meta', 'Control', 'F5', 'Dead'])(
    'returns null for the non-binding key %s',
    (key) => {
      expect(fromEvent({ key }, 'darwin')).toBeNull();
    }
  );
});
