// Sanitized markdown rendering for assistant chat messages. `marked` parses,
// `highlight.js` colorizes fenced code (pure JS, no eval/new Function/WASM — CSP
// `script-src 'self'`-safe; colors come from bundled .hljs-* token CSS, not
// inline styles), and `DOMPurify` sanitizes — all bundled (no CDN), so
// scripts/verify-csp-hash.mjs (inline pre-paint script only) is unaffected.
//
// Inline-only: no remote images/fonts/scripts are fetched by any library.
//
// hljs is a static import (in the boot bundle) so renderMarkdown stays
// synchronous — no highlight flash; acceptable for a local-first desktop app
// (assets on disk, no network fetch).

import { Marked } from 'marked';
import { markedHighlight } from 'marked-highlight';
import hljs from 'highlight.js/lib/common';
import DOMPurify from 'dompurify';

// Fence-hint priority, auto-detect fallback: honor a known language hint, else
// let hljs guess over the ~37-language common preset.
const highlightMarked = new Marked(
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
highlightMarked.setOptions({ breaks: true, gfm: true });

// Plain instance for the live streaming path: no hljs, so the growing buffer
// isn't re-tokenized per token (O(n²)) and highlightAuto never runs on an
// open fence. Same GFM options and DOMPurify pass as the highlighting path.
const plainMarked = new Marked();
plainMarked.setOptions({ breaks: true, gfm: true });

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
export function renderMarkdown(source: string, opts?: { highlight?: boolean }): string {
  const instance = opts?.highlight === false ? plainMarked : highlightMarked;
  const html = instance.parse(source, { async: false }) as string;
  return DOMPurify.sanitize(html, {
    ALLOWED_URI_REGEXP: /^(?:https?:|mailto:)/i,
    FORBID_ATTR: ['style', 'onerror', 'onload']
  });
}
