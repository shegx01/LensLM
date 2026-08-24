import { eq, parse, type Binding } from './binding.js';
import { SHORTCUTS, type ActionId, type Scope, type ShortcutEntry } from './registry.js';

// Element-scoped handlers do not stop propagation on a miss, so every element-scoped
// key also reaches AppShell's window listener. `window` is therefore a universal
// domain; the other scopes stay mutually disjoint.
function sharesDomain(candidateScope: Scope, other: Scope): boolean {
  return candidateScope === 'window' || other === 'window' || other === candidateScope;
}

/** The entry already occupying `candidate` in its conflict domain, or `null`. */
export function findConflict(
  scope: Scope,
  candidate: Binding,
  forId: ActionId,
  keymap: Partial<Record<ActionId, string>>
): ShortcutEntry | null {
  for (const entry of SHORTCUTS) {
    if (entry.id === forId) continue;
    if (!sharesDomain(scope, entry.scope)) continue;
    if (eq(parse(keymap[entry.id] ?? entry.defaultBinding), candidate)) return entry;
  }
  return null;
}

/**
 * A `window` binding needs Mod or Alt: AppShell's close branch is unguarded, so a
 * typeable global shortcut swallows characters no typing guard can protect. Shift
 * does not qualify — `Shift+Q` is how a capital Q is typed.
 */
export function isValidForScope(scope: Scope, binding: Binding | null): boolean {
  if (binding === null) return false;
  if (scope !== 'window') return true;
  return binding.mod || binding.alt;
}
