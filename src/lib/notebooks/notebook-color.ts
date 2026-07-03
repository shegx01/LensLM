// Deterministic notebook icon color utility (FALLBACK path).
//
// The primary source is rank-based assignment in the store; this module provides
// the palette and a pure hash-based fallback for ids not in the live set (e.g.
// trashed notebooks in TrashView). Palette hues may coincide with accent hues —
// intentional; these classes never touch [data-accent] tokens.

/** 10-hue decorative palette; interleaved warm/cool so adjacent ranks stay distinct. Maps to `.nb-{id}` in app.css. */
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

function djb2(str: string): number {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = (hash * 33) ^ str.charCodeAt(i);
  }
  return hash >>> 0; // unsigned 32-bit
}

/**
 * Hash-based fallback returning a stable `nb-{paletteId}` class for ids not in
 * the live set. Prefer `notebookColorClass(id)` for live notebooks.
 */
export function notebookAccentClass(id: string): string {
  const index = djb2(id) % NOTEBOOK_PALETTE.length;
  return `nb-${NOTEBOOK_PALETTE[index]}`;
}
