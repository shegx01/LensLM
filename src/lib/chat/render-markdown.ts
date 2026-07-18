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

// KaTeX lays out glyphs with inline `style` (lengths + display/position:relative on
// its own spans), so `style` cannot be globally stripped for math output. We must NOT
// decide by class — `marked` passes model/user-authored HTML through, so a forged
// `class="katex"` would otherwise smuggle an arbitrary `style` (e.g. a full-viewport
// `position:fixed` clickjacking overlay). Instead we keep `style` only when EVERY
// declaration is a safe presentational one KaTeX emits (em/px/%/pt/rem lengths,
// `display`, `vertical-align`, `position:relative`). Viewport units, `position:fixed`,
// and any `url(...)`/function are rejected regardless of class. `\color` math loses
// its color (graceful degradation) — an acceptable trade for closing the overlay
// vector. Event handlers and `javascript:` URIs stay forbidden by the config below.
const KATEX_STYLE_PROP =
  /^(?:height|width|min-width|max-width|margin(?:-(?:left|right|top|bottom))?|padding(?:-(?:left|right|top|bottom))?|top|bottom|left|right|vertical-align|display|position)$/;
const KATEX_STYLE_VALUE =
  /^(?:-?[0-9.]+(?:em|ex|px|pt|rem|%)?|inline-block|inline|block|middle|baseline|text-top|text-bottom|top|bottom|relative)$/;

function isSafeKatexStyle(value: string): boolean {
  const decls = value
    .split(';')
    .map((d) => d.trim())
    .filter(Boolean);
  if (decls.length === 0) return false;
  for (const decl of decls) {
    const idx = decl.indexOf(':');
    if (idx < 0) return false;
    const prop = decl.slice(0, idx).trim().toLowerCase();
    const val = decl
      .slice(idx + 1)
      .trim()
      .toLowerCase();
    if (!KATEX_STYLE_PROP.test(prop)) return false;
    if (val.includes('(')) return false; // no url()/calc()/any function
    if (prop === 'position' && val !== 'relative') return false; // never fixed/absolute/sticky
    if (!KATEX_STYLE_VALUE.test(val)) return false;
  }
  return true;
}

let katexHookRegistered = false;
function ensureKatexStyleHook(): void {
  if (katexHookRegistered) return;
  katexHookRegistered = true;
  DOMPurify.addHook('uponSanitizeAttribute', (node, data) => {
    if (data.attrName !== 'style') return;
    if (isSafeKatexStyle(data.attrValue)) data.forceKeepAttr = true;
  });
}

/**
 * SECURITY INVARIANT: model/assistant output is display-only and MUST NOT execute —
 * it reaches `{@html}` only after this sanitize pass. Locked by *.no-exec.test.ts.
 * Parses `source` as GFM markdown and sanitizes the resulting HTML for safe `{@html}` use.
 */
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
