// Deterministic notebook icon color utility (FALLBACK path).
//
// The primary color source is rank-based assignment in the store
// (`notebookColorClass` in notebooks-state.svelte.ts), which guarantees the
// first N notebooks (N = palette size) are always distinct. This module
// provides the canonical decorative palette and a pure hash-based fallback for
// ids that are NOT in the live notebook set (e.g. trashed notebooks rendered in
// TrashView), where no stable rank exists.
//
// NOTE (visual overlap): the decorative notebook-icon palette deliberately
// includes hues that also appear in the user accent selector (`ACCENT_IDS`). A
// notebook icon may coincidentally match the user's current accent. This is
// intentional and acceptable — these classes never touch [data-accent] tokens.

/**
 * Canonical decorative palette ids (10 hues). Order is the rank order used by
 * the store's rank-based assignment, interleaving warm/cool so adjacent ranks
 * stay visually distinct. Each id maps to a `.nb-{id}` rule in app.css.
 */
export const NOTEBOOK_PALETTE = [
  'purple',
  'blue',
  'green',
  'teal',
  'amber',
  'orange',
  'rose',
  'pink',
  'indigo',
  'graphite'
] as const;

export type NotebookPaletteId = (typeof NOTEBOOK_PALETTE)[number];

/**
 * djb2 hash — fast, low collision rate, deterministic for equal strings.
 * Returns a non-negative integer.
 */
function djb2(str: string): number {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = (hash * 33) ^ str.charCodeAt(i);
  }
  // Force to unsigned 32-bit int.
  return hash >>> 0;
}

/**
 * Pure fallback: return a deterministic CSS class name for a notebook icon
 * based on its id, hashing into the 10-hue {@link NOTEBOOK_PALETTE}.
 *
 * Returns a class of the form `nb-{paletteId}` (e.g. `nb-purple`). The same id
 * ALWAYS produces the same class; color is purely decorative.
 *
 * Prefer the store's `notebookColorClass(id)` for live notebooks — it is
 * collision-free by rank. Use this only when no live rank exists.
 */
export function notebookAccentClass(id: string): string {
  const index = djb2(id) % NOTEBOOK_PALETTE.length;
  return `nb-${NOTEBOOK_PALETTE[index]}`;
}
