// Relative-time formatter for notebook timestamps.
//
// Pure function — no external deps, no Date.now() side-channel injection
// (callers can control `now` via the optional second arg for tests).

/**
 * Format an RFC3339 / ISO 8601 timestamp as a human-readable relative time
 * string (e.g. "just now", "2h ago", "3d ago", "1w ago", "3mo ago").
 *
 * @param isoString - An RFC3339 / ISO 8601 date string (as returned by Rust chrono)
 * @param now       - Reference timestamp in ms (defaults to Date.now()). Inject in tests.
 */
export function formatRelativeTime(isoString: string, now: number = Date.now()): string {
  const then = Date.parse(isoString);
  if (Number.isNaN(then)) return '';

  const diffMs = now - then;
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHours = Math.floor(diffMin / 60);
  const diffDays = Math.floor(diffHours / 24);
  const diffWeeks = Math.floor(diffDays / 7);
  const diffMonths = Math.floor(diffDays / 30);

  if (diffSec < 60) return 'just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  if (diffWeeks < 5) return `${diffWeeks}w ago`;
  return `${diffMonths}mo ago`;
}

/**
 * Format a source count as a pluralized label: `"1 source"` for exactly one,
 * `"N sources"` otherwise (including zero).
 */
export function formatSourceCount(count: number): string {
  return count === 1 ? '1 source' : `${count} sources`;
}
