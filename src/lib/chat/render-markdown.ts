// Sanitized markdown rendering for assistant chat messages. `marked` parses,
// `DOMPurify` sanitizes — both are bundled (no CDN) and free of eval/new
// Function, satisfying the CSP `script-src 'self' ...` policy (scripts/verify-csp-hash.mjs
// only guards the inline pre-paint script and is unaffected by this dependency).
//
// Inline-only: no remote images/fonts/scripts are fetched by either library.

import { marked } from 'marked';
import DOMPurify from 'dompurify';

marked.setOptions({ breaks: true, gfm: true });

/** Parses `source` as GFM markdown and sanitizes the resulting HTML for safe `{@html}` use. */
export function renderMarkdown(source: string): string {
  const html = marked.parse(source, { async: false }) as string;
  return DOMPurify.sanitize(html, {
    ALLOWED_URI_REGEXP: /^(?:https?:|mailto:)/i,
    FORBID_ATTR: ['style', 'onerror', 'onload']
  });
}
