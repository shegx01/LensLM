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
//
// KaTeX ($…$ inline, $$…$$ / \[…\] block) renders synchronously on the final
// (highlight) instance only — NOT on the streaming `plainMarked` instance, so a
// growing buffer isn't re-tokenized per token (mirrors the hljs-vs-plain split);
// math shows raw while streaming, then renders on final render. KaTeX is pure JS
// (no eval/WASM) with `output:'html'` (HTML-only, no MathML) to keep the sanitize
// allow-list small; its CSS+fonts are bundled locally below (no CDN).

import { Marked, type TokenizerAndRendererExtension } from 'marked';
import { markedHighlight } from 'marked-highlight';
import markedKatex from 'marked-katex-extension';
import katex from 'katex';
import hljs from 'highlight.js/lib/common';
import DOMPurify from 'dompurify';
import 'katex/dist/katex.min.css';

// marked-katex-extension only knows `$…$`/`$$…$$`; this pair of inline extensions
// adds the LaTeX `\[…\]` (display) and `\(…\)` (inline) delimiters (AC16). Both are
// inline-level so they match wherever they appear (a bare `\[…\]` line is otherwise
// a paragraph, so a block-level ext would never fire); `\[` still renders in KaTeX
// `displayMode`. Same HTML-only output so the sanitize profile is unchanged;
// `throwOnError` off so malformed math degrades to KaTeX error markup, never throws.
function katexDelimiter(
  name: string,
  open: string,
  close: string,
  displayMode: boolean
): TokenizerAndRendererExtension {
  const esc = (s: string): string => s.replace(/[[\]()\\]/g, '\\$&');
  const rule = new RegExp(`^${esc(open)}([\\s\\S]+?)${esc(close)}`);
  return {
    name,
    level: 'inline',
    start(src: string) {
      return src.indexOf(open);
    },
    tokenizer(src: string) {
      const m = rule.exec(src);
      if (!m) return undefined;
      return { type: name, raw: m[0], text: m[1].trim() };
    },
    renderer(token) {
      return katex.renderToString(token.text ?? '', {
        output: 'html',
        throwOnError: false,
        displayMode
      });
    }
  };
}

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
highlightMarked.use(markedKatex({ output: 'html', throwOnError: false, nonStandard: true }));
highlightMarked.use({
  extensions: [
    katexDelimiter('katexBlockBracket', '\\[', '\\]', true),
    katexDelimiter('katexInlineParen', '\\(', '\\)', false)
  ]
});

// Plain instance for the live streaming path: no hljs and no KaTeX, so the
// growing buffer isn't re-tokenized per token (O(n²)), highlightAuto never runs
// on an open fence, and `$…$` shows raw until the final render. Same GFM options
// and DOMPurify pass as the highlighting path.
const plainMarked = new Marked();
plainMarked.setOptions({ breaks: true, gfm: true });

// KaTeX lays out glyphs with inline `style` (e.g. height/margin on class-bearing
// spans), so `style` cannot be globally stripped for math output. Under the app
// CSP (no `unsafe-inline` for scripts; no eval/WASM) an inline `style` attribute
// cannot execute JS — it is pure presentation — so allowing it ONLY on KaTeX's
// own class-bearing elements is safe. This hook permits `style` on `.katex*`
// nodes and strips it everywhere else, while event handlers and `javascript:`
// URIs stay forbidden by the sanitize config below (proven by the no-exec tests).
let katexHookRegistered = false;
function ensureKatexStyleHook(): void {
  if (katexHookRegistered) return;
  katexHookRegistered = true;
  DOMPurify.addHook('uponSanitizeAttribute', (node, data) => {
    if (data.attrName !== 'style') return;
    const el = node as Element;
    const cls = typeof el.className === 'string' ? el.className : '';
    if (/\bkatex\b/.test(cls) || (el.closest && el.closest('.katex'))) {
      data.forceKeepAttr = true;
    }
  });
}

/** Parses `source` as GFM markdown and sanitizes the resulting HTML for safe `{@html}` use. */
export function renderMarkdown(source: string, opts?: { highlight?: boolean }): string {
  ensureKatexStyleHook();
  const instance = opts?.highlight === false ? plainMarked : highlightMarked;
  const html = instance.parse(source, { async: false }) as string;
  // `style` stays in FORBID_ATTR (stripped by default); the hook above re-keeps it
  // ONLY on KaTeX nodes. `onerror`/`onload` and non-http(s)/mailto URIs stay blocked.
  return DOMPurify.sanitize(html, {
    ALLOWED_URI_REGEXP: /^(?:https?:|mailto:)/i,
    FORBID_ATTR: ['style', 'onerror', 'onload']
  });
}
