import { describe, expect, it } from 'vitest';
import { eq, parse, render, describe as spoken, toToken, type Platform } from './binding.js';
import { ACTION_IDS, GROUP_ORDER, ROWS, SHORTCUTS, type ActionId, type Scope } from './registry.js';

const PLATFORMS: Platform[] = ['darwin', 'win32', 'linux'];
const RESERVED: ActionId[] = ['palette.close', 'chat.send', 'chat.newline'];

describe('ACTION_IDS', () => {
  it('is the closed set of 11 ids', () => {
    expect(ACTION_IDS).toEqual([
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
    ]);
  });
});

describe('SHORTCUTS', () => {
  it('carries exactly one entry per action id, keyed by that id', () => {
    expect(SHORTCUTS.map((entry) => entry.id)).toEqual([...ACTION_IDS]);
  });

  it.each([...ACTION_IDS])('has a parseable default binding for %s', (id) => {
    const entry = SHORTCUTS.find((e) => e.id === id);
    expect(entry).toBeDefined();
    expect(parse(entry!.defaultBinding)).not.toBeNull();
  });

  it('renders and describes every default binding on every platform', () => {
    for (const entry of SHORTCUTS) {
      for (const platform of PLATFORMS) {
        expect(render(entry.defaultBinding, platform)).not.toBe('');
        expect(spoken(entry.defaultBinding, platform)).not.toBe('');
      }
    }
  });

  it.each([...ACTION_IDS])('ships %s already in canonical token form', (id) => {
    const entry = SHORTCUTS.find((e) => e.id === id)!;
    expect(toToken(parse(entry.defaultBinding)!)).toBe(entry.defaultBinding);
  });

  it('reserves exactly the three capture-widget bindings', () => {
    const reserved = SHORTCUTS.filter((entry) => !entry.remappable).map((entry) => entry.id);
    expect(reserved).toEqual(RESERVED);
  });

  it('gives every entry a non-empty action name and description', () => {
    for (const entry of SHORTCUTS) {
      expect(entry.action.length).toBeGreaterThan(0);
      expect(entry.description.length).toBeGreaterThan(0);
    }
  });

  it.each(['palette', 'player', 'composer'] as Scope[])(
    'ships defaults that are collision-free across window ∪ %s',
    (scope) => {
      const domain = SHORTCUTS.filter((e) => e.scope === 'window' || e.scope === scope);
      const collisions = domain.flatMap((a, i) =>
        domain
          .slice(i + 1)
          .filter((b) => eq(parse(a.defaultBinding), parse(b.defaultBinding)))
          .map((b) => `${a.id} vs ${b.id}`)
      );
      expect(collisions).toEqual([]);
    }
  );
});

describe('ROWS', () => {
  it('covers all 11 action ids exactly once', () => {
    expect(ROWS.flatMap((row) => row.ids).sort()).toEqual([...ACTION_IDS].sort());
  });

  it('is the 8 display rows, three of which pair two ids', () => {
    expect(ROWS.length).toBe(8);
    expect(ROWS.filter((row) => row.ids.length === 2).map((row) => row.label)).toEqual([
      'Seek',
      'Skip',
      'Playback speed'
    ]);
  });

  it('gives every row a group from GROUP_ORDER matching its entries', () => {
    for (const row of ROWS) {
      expect(GROUP_ORDER).toContain(row.group);
      for (const id of row.ids) {
        expect(SHORTCUTS.find((e) => e.id === id)?.group).toBe(row.group);
      }
    }
  });

  it('gives every row a non-empty label and description', () => {
    for (const row of ROWS) {
      expect(row.label.length).toBeGreaterThan(0);
      expect(row.description.length).toBeGreaterThan(0);
    }
  });
});

describe('GROUP_ORDER', () => {
  it('is the three display groups in order', () => {
    expect(GROUP_ORDER).toEqual(['Global', 'Chat', 'Audio player']);
  });
});
