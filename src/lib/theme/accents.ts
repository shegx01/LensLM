// Canonical accent definitions — single source of truth for the accent id set
// and the rich swatch list. The [data-accent] token layer (app.css) keys off
// these ids; +layout.svelte validates persisted config against ACCENT_IDS and
// MakeItYours renders the swatch grid from ACCENTS. Keep order/labels in sync
// with the design spec.

export const ACCENT_IDS = ['purple', 'green', 'blue', 'amber', 'rose', 'graphite'] as const;

export type AccentId = (typeof ACCENT_IDS)[number];

/** The six accent swatches; `solid` values are design-spec fixed colors. */
export const ACCENTS: { id: AccentId; label: string; solid: string }[] = [
  { id: 'purple', label: 'Violet', solid: '#7c3aed' },
  { id: 'blue', label: 'Blue', solid: '#2563eb' },
  { id: 'green', label: 'Green', solid: '#16a34a' },
  { id: 'amber', label: 'Amber', solid: '#d97706' },
  { id: 'rose', label: 'Rose', solid: '#e11d48' },
  { id: 'graphite', label: 'Graphite', solid: '#52525b' }
];
