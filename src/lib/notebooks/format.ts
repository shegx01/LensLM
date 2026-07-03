/** Up-to-two-character uppercase initials from a name. Falls back to `"?"`. */
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

/**
 * Format a source count as a pluralized label: `"1 source"` for exactly one,
 * `"N sources"` otherwise (including zero).
 */
export function formatSourceCount(count: number): string {
  return count === 1 ? '1 source' : `${count} sources`;
}
