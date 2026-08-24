import { eq, fromEvent, parse, type KeyEventLike, type Platform } from './binding.js';
import { SHORTCUTS, type ActionId, type Scope } from './registry.js';

/**
 * Pure keystroke → action lookup. No DOM, no listeners, no ambient platform read:
 * callers pass `currentPlatform()` so this stays cheap on the keydown hot path.
 */
export function resolve(
  event: KeyEventLike,
  scope: Scope,
  keymap: Partial<Record<ActionId, string>>,
  platform: Platform
): ActionId | null {
  const pressed = fromEvent(event, platform);
  if (pressed === null) return null;

  for (const entry of SHORTCUTS) {
    if (entry.scope !== scope) continue;
    if (eq(parse(keymap[entry.id] ?? entry.defaultBinding), pressed)) return entry.id;
  }
  return null;
}
