// IPC wrappers for citation read-back (issue #237): resolve a cited span into a
// bounded context snippet, or load a source for the "view in source" viewer.
// SYNC-CHECK: mirrors lens-core `SnippetSegments`/`SourceView`.

import { invoke, isTauri } from '@tauri-apps/api/core';
import type { SnippetSegments, SourceView } from '$lib/chat/types.js';

/** Cap on memoized snippets so a long session's hover history can't grow unbounded. */
const SNIPPET_CACHE_MAX = 200;

const snippetCache = new Map<string, Promise<SnippetSegments>>();

function snippetKey(sourceId: string, charStart: number, charEnd: number): string {
  return `${sourceId}:${charStart}:${charEnd}`;
}

/** Evicts the oldest (insertion-order) entry once the cache exceeds its cap. */
function capSnippetCache(): void {
  if (snippetCache.size <= SNIPPET_CACHE_MAX) return;
  const oldest = snippetCache.keys().next().value;
  if (oldest !== undefined) snippetCache.delete(oldest);
}

/**
 * Resolves a cited span into bounded display segments for the inline popover.
 * Memoized per `(sourceId, charStart, charEnd)` so repeated hovers over the same
 * chip don't re-invoke the IPC bridge. A rejected fetch is evicted so a transient
 * failure (e.g. a source removed mid-hover) can be retried on the next hover.
 */
export function resolveCitationSnippet(
  sourceId: string,
  charStart: number,
  charEnd: number
): Promise<SnippetSegments> {
  if (!isTauri()) {
    return Promise.reject(new Error('resolveCitationSnippet: not running under Tauri'));
  }
  const key = snippetKey(sourceId, charStart, charEnd);
  const cached = snippetCache.get(key);
  if (cached) return cached;

  const pending = invoke<SnippetSegments>('resolve_citation_snippet', {
    sourceId,
    charStart,
    charEnd
  }).catch((err) => {
    snippetCache.delete(key);
    throw err;
  });
  snippetCache.set(key, pending);
  capSnippetCache();
  return pending;
}

/**
 * Loads a source for the "view in source" viewer. `charStart`/`charEnd` are
 * omitted or `null` when the citation carries no offsets (R4) — the viewer then
 * gets the whole text back in `before`, unhighlighted. Not memoized: the viewer
 * is opened rarely, and a stale cache could show outdated truncation state.
 */
export function loadSourceView(
  sourceId: string,
  charStart: number | null = null,
  charEnd: number | null = null
): Promise<SourceView> {
  if (!isTauri()) {
    return Promise.reject(new Error('loadSourceView: not running under Tauri'));
  }
  return invoke<SourceView>('load_source_view', { sourceId, charStart, charEnd });
}

/** Test-only: clears the snippet memo cache so cases don't bleed into each other. */
export function resetCitationSnippetCache(): void {
  snippetCache.clear();
}
