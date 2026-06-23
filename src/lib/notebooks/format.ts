// Shared display-formatting helpers for the notebooks module.
//
// Pure functions — no external deps.

/**
 * Derive up-to-two-character uppercase initials from a display name.
 *
 * Trims, splits on whitespace, drops empty segments, takes the first letter of
 * the first two words, and upper-cases the result. Falls back to `"?"` when the
 * name yields no usable characters (empty / whitespace-only).
 */
export function getInitials(name: string): string {
  return (
    name
      .trim()
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((word) => word[0].toUpperCase())
      .join('') || '?'
  );
}
