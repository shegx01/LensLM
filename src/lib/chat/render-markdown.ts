// Sanitized markdown rendering for assistant chat messages. `marked` parses,
// `highlight.js` colorizes fenced code (pure JS, no eval/new Function/WASM — CSP
// `script-src 'self'`-safe; colors come from bundled .hljs-* token CSS, not
// inline styles), and `DOMPurify` sanitizes — all bundled (no CDN), so
// scripts/verify-csp-hash.mjs (inline pre-paint script only) is unaffected.
//
// Inline-only: no remote images/fonts/scripts are fetched by any library.

import { Marked } from 'marked';
import { markedHighlight } from 'marked-highlight';
import hljs from 'highlight.js/lib/common';
import DOMPurify from 'dompurify';

// Fence-hint priority, auto-detect fallback: honor a known language hint, else
// let hljs guess over the ~37-language common preset.
const marked = new Marked(
  markedHighlight({
    langPrefix: 'hljs language-',
    highlight(code, lang) {
      if (lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value;
      }
      return hljs.highlightAuto(code).value;
    }
  })
);

marked.setOptions({ breaks: true, gfm: true });

/** Parses `source` as GFM markdown and sanitizes the resulting HTML for safe `{@html}` use. */
export function renderMarkdown(source: string): string {
  const html = marked.parse(source, { async: false }) as string;
  return DOMPurify.sanitize(html, {
    ALLOWED_URI_REGEXP: /^(?:https?:|mailto:)/i,
    FORBID_ATTR: ['style', 'onerror', 'onload']
  });
}
