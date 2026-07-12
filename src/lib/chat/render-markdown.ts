// Sanitized markdown rendering for assistant chat messages. `marked` parses,
// `DOMPurify` sanitizes — both are bundled (no CDN) and free of eval/new
// Function, satisfying the CSP `script-src 'self' ...` policy (scripts/verify-csp-hash.mjs
// only guards the inline pre-paint script and is unaffected by this dependency).
//
// Inline-only: no remote images/fonts/scripts are fetched by either library.

import { marked } from 'marked';
import DOMPurify from 'dompurify';

marked.setOptions({ breaks: true, gfm: true });

/**
 * Removes inline `[n]` citation markers from `source`, but ONLY when `n` is a real
 * citation ordinal — the payload's ordinal set disambiguates a marker from a literal
 * numeric bracket in prose (`arr[0]`, out-of-range). `(?!\()` spares markdown links
 * `[1](url)`; `(?!\s*:)` spares reference-style definitions `[1]: url`. Empty set ⇒
 * nothing to strip.
 */
export function stripCitationMarkers(source: string, ordinals: Set<number>): string {
  if (ordinals.size === 0) return source;
  return source.replace(/[ \t]?\[(\d+)\](?!\(|\s*:)/g, (m, num) =>
    ordinals.has(Number(num)) ? '' : m
  );
}

/** Parses `source` as GFM markdown and sanitizes the resulting HTML for safe `{@html}` use. */
export function renderMarkdown(source: string): string {
  const html = marked.parse(source, { async: false }) as string;
  return DOMPurify.sanitize(html, {
    ALLOWED_URI_REGEXP: /^(?:https?:|mailto:)/i,
    FORBID_ATTR: ['style', 'onerror', 'onload']
  });
}
