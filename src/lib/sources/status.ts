// Shared source-status presentation helpers (single source of truth for status dots).

import type { SourceStatus } from './types.js';

/**
 * Map a `SourceStatus` to its dot color class.
 * indexed→green, error→destructive, queued/pending/parsing/embedding→amber pulse,
 * needs_js→amber static, render_failed→destructive/60, unknown→muted.
 */
export function statusDotClass(status: SourceStatus): string {
  switch (status) {
    case 'indexed':
      return 'bg-green-primary';
    case 'error':
      return 'bg-destructive';
    case 'parsing':
    case 'embedding':
    case 'queued':
    case 'pending':
      return 'bg-amber-500 animate-pulse';
    case 'needs_js':
      return 'bg-amber-500';
    case 'render_failed':
      return 'bg-destructive/60';
    default:
      return 'bg-muted-foreground/40';
  }
}
