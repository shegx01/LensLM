// Shared source-status presentation helpers.
//
// Extracted so SourcesRail (the product sources list) and the dev/QA
// EmbeddingsInspector render identical status dots from one source of truth.

import type { SourceStatus } from './types.js';

/**
 * Map a {@link SourceStatus} to its dot color class (semantic tokens only).
 *
 * indexed → green, error → destructive/red,
 * queued/pending/parsing/embedding → amber (pulsing), unknown → muted.
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
    default:
      return 'bg-muted-foreground/40';
  }
}
