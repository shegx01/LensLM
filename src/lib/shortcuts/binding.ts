export type Platform = 'darwin' | 'win32' | 'linux';

/**
 * Canonical, fully-specified keystroke. `mod` is the platform-abstract primary
 * modifier: Command on darwin, Control elsewhere.
 */
export interface Binding {
  key: string;
  mod: boolean;
  shift: boolean;
  alt: boolean;
}

export interface KeyEventLike {
  key: string;
  ctrlKey?: boolean;
  metaKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
}

interface ModifierLabels {
  mod: string;
  shift: string;
  alt: string;
}

const NAMED_KEYS = new Map([
  ['space', 'Space'],
  ['enter', 'Enter'],
  ['escape', 'Escape'],
  ['arrowleft', 'ArrowLeft'],
  ['arrowright', 'ArrowRight'],
  ['arrowup', 'ArrowUp'],
  ['arrowdown', 'ArrowDown']
]);

const EVENT_KEY_ALIASES = new Map([
  [' ', 'Space'],
  ['Spacebar', 'Space']
]);

const GLYPH_MODIFIERS: Record<Platform, ModifierLabels> = {
  darwin: { mod: '⌘', shift: '⇧', alt: '⌥' },
  win32: { mod: 'Ctrl', shift: 'Shift', alt: 'Alt' },
  linux: { mod: 'Ctrl', shift: 'Shift', alt: 'Alt' }
};

const WORD_MODIFIERS: Record<Platform, ModifierLabels> = {
  darwin: { mod: 'Command', shift: 'Shift', alt: 'Option' },
  win32: { mod: 'Control', shift: 'Shift', alt: 'Alt' },
  linux: { mod: 'Control', shift: 'Shift', alt: 'Alt' }
};

const KEY_GLYPHS = new Map([
  ['ArrowLeft', '←'],
  ['ArrowRight', '→'],
  ['ArrowUp', '↑'],
  ['ArrowDown', '↓']
]);

const KEY_WORDS = new Map([
  ['ArrowLeft', 'Left arrow'],
  ['ArrowRight', 'Right arrow'],
  ['ArrowUp', 'Up arrow'],
  ['ArrowDown', 'Down arrow'],
  ['[', 'Left bracket'],
  [']', 'Right bracket']
]);

function canonicalKey(raw: string): string | null {
  const named = NAMED_KEYS.get(raw.toLowerCase());
  if (named !== undefined) return named;
  if (raw === '[' || raw === ']') return raw;
  if (/^[a-z0-9]$/i.test(raw)) return raw.toUpperCase();
  return null;
}

export function parse(token: string): Binding | null {
  let mod = false;
  let shift = false;
  let alt = false;
  let key: string | null = null;

  for (const part of token.split('+')) {
    if (key !== null) return null;

    switch (part.toLowerCase()) {
      case 'mod':
        if (mod) return null;
        mod = true;
        continue;
      case 'shift':
        if (shift) return null;
        shift = true;
        continue;
      case 'alt':
        if (alt) return null;
        alt = true;
        continue;
    }

    key = canonicalKey(part);
    if (key === null) return null;
  }

  return key === null ? null : { key, mod, shift, alt };
}

// An unparseable token is echoed verbatim so a stale or hand-edited config.json
// entry stays visible to the user instead of rendering as an empty chip.
function present(
  token: string,
  modifiers: ModifierLabels,
  keyLabel: (key: string) => string,
  joiner: string
): string {
  const b = parse(token);
  if (b === null) return token;

  const parts: string[] = [];
  if (b.mod) parts.push(modifiers.mod);
  if (b.shift) parts.push(modifiers.shift);
  if (b.alt) parts.push(modifiers.alt);
  parts.push(keyLabel(b.key));
  return parts.join(joiner);
}

export function render(token: string, platform: Platform): string {
  return present(
    token,
    GLYPH_MODIFIERS[platform],
    (key) => KEY_GLYPHS.get(key) ?? key,
    platform === 'darwin' ? '' : '+'
  );
}

export function describe(token: string, platform: Platform): string {
  return present(token, WORD_MODIFIERS[platform], (key) => KEY_WORDS.get(key) ?? key, ' plus ');
}

/** Canonical storage form — platform-independent, never glyphs. */
export function toToken(binding: Binding): string {
  const parts: string[] = [];
  if (binding.mod) parts.push('Mod');
  if (binding.shift) parts.push('Shift');
  if (binding.alt) parts.push('Alt');
  parts.push(binding.key);
  return parts.join('+');
}

export function eq(a: Binding | null, b: Binding | null): boolean {
  return (
    a !== null &&
    b !== null &&
    a.key === b.key &&
    a.mod === b.mod &&
    a.shift === b.shift &&
    a.alt === b.alt
  );
}

export function fromEvent(event: KeyEventLike, platform: Platform): Binding | null {
  const onDarwin = platform === 'darwin';

  // A modifier the canonical form cannot express must reject, never degrade to a
  // weaker binding: swallowing Control on darwin would make Ctrl+Space (the
  // input-source switcher) equal to a bare Space binding.
  if (onDarwin ? event.ctrlKey === true : event.metaKey === true) return null;

  const key = canonicalKey(EVENT_KEY_ALIASES.get(event.key) ?? event.key);
  if (key === null) return null;

  return {
    key,
    mod: onDarwin ? event.metaKey === true : event.ctrlKey === true,
    shift: event.shiftKey === true,
    alt: event.altKey === true
  };
}
