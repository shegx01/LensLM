// Deterministic notebook icon color utility.
//
// Maps a notebook id to one of the 6 accent palette ids via a djb2 hash.
// The result is a CSS class string that consumers apply to the notebook icon
// element — design-system tokens handle the actual color values.
//
// NOTE (visual overlap): the deterministic notebook-icon palette reuses the
// same 6 accent hues as the user accent selector (`ACCENT_IDS`). A notebook
// icon may coincidentally match the user's current accent. This is intentional
// and acceptable — documented here so it is not flagged as a bug.

import { ACCENT_IDS } from '$lib/theme/accents.js';

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
 * Return a deterministic CSS class name for a notebook icon based on its id.
 * The class encodes which accent hue to use — always one of the 6 values in
 * `ACCENT_IDS` source order: `['purple', 'green', 'blue', 'amber', 'rose', 'graphite']`.
 *
 * Returns a class of the form `nb-{accentId}` (e.g. `nb-purple`). Components
 * must map this class to an appropriate token-based style rule.
 *
 * The same id ALWAYS produces the same class; color is purely decorative.
 */
export function notebookAccentClass(id: string): string {
  const index = djb2(id) % ACCENT_IDS.length;
  return `nb-${ACCENT_IDS[index]}`;
}
