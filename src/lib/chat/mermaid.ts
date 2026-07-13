// Async, two-pass mermaid hydration. `renderMarkdown` stays synchronous and emits
// ```mermaid fences as ordinary highlighted code blocks; this pass upgrades the
// SUPPORTED subset (flowchart + sequence) into sanitized inline SVG, and leaves
// every other fence as its already-highlighted raw code (never blank). Inert by
// construction — see `sanitizeSvg` for the security posture. Re-runnable /
// restore-safe like enhanceCitations: each call re-reads the original source from
// the (preserved) fence, so re-render/theme-change is idempotent.

import DOMPurify from 'dompurify';

// Supported subset, in two vocabularies that MUST stay in sync (keep both edited
// together): `SUPPORTED_KEYWORDS` is the cheap leading-keyword pre-filter run
// before mermaid; `SUPPORTED_TYPES` is the `mermaid.parse` `diagramType` gate run
// after. Anything not in both keeps its raw fence.
const SUPPORTED_TYPES: ReadonlySet<string> = new Set(['flowchart', 'flowchart-v2', 'sequence']);
const SUPPORTED_KEYWORDS = /^\s*(?:flowchart|graph|sequenceDiagram)\b/;

let mermaidReady: Promise<typeof import('mermaid').default> | null = null;

async function loadMermaid(theme: 'default' | 'dark'): Promise<typeof import('mermaid').default> {
  if (!mermaidReady) {
    mermaidReady = import('mermaid').then((m) => m.default);
  }
  const mermaid = await mermaidReady;
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',
    theme,
    fontFamily: 'var(--font-sans, inherit)'
  });
  return mermaid;
}

// Never-execute-model-output posture: mermaid runs `securityLevel:'strict'`
// (no HTML labels / click handlers) and its SVG is re-sanitized here — `<script>`,
// `<foreignObject>` (no HTML/img-onerror ride-along), `on*` handlers and
// `javascript:` URIs are all dropped, so an SVG from a malicious diagram source is
// inert (proven by mermaid.no-exec.test). `ALLOWED_URI_REGEXP` allows ONLY
// same-document refs — a bare `#id` fragment or the functional `url(#id)` form that
// flowchart arrowheads/edge markers and gradient/clip fills depend on — while still
// rejecting external `http(s):`/`data:`/`mailto:` URIs so no diagram can fetch or
// exfil (defense-in-depth beyond CSP img-src).
const SAME_DOCUMENT_URI = /^(?:#|url\(['"]?#)/;

function sanitizeSvg(svg: string): string {
  return DOMPurify.sanitize(svg, {
    USE_PROFILES: { svg: true, svgFilters: true },
    FORBID_TAGS: ['script', 'foreignObject', 'image', 'feImage'],
    ALLOWED_URI_REGEXP: SAME_DOCUMENT_URI
  });
}

let idCounter = 0;

/**
 * Upgrades supported ```mermaid fences inside `container` to sanitized inline SVG.
 * Idempotent and restore-safe; any unsupported/malformed/failed block keeps its
 * original highlighted code fence. Never throws into the caller's render path.
 */
export async function hydrateMermaid(container: HTMLElement): Promise<void> {
  const blocks = Array.from(
    container.querySelectorAll<HTMLElement>(
      'code.language-mermaid, code[class*="language-mermaid"]'
    )
  );
  if (blocks.length === 0) return;

  const isDark =
    typeof document !== 'undefined' && document.documentElement.classList.contains('dark');
  const theme = isDark ? 'dark' : 'default';

  let mermaid: typeof import('mermaid').default;
  try {
    mermaid = await loadMermaid(theme);
  } catch {
    return;
  }

  for (const code of blocks) {
    // `textContent` reassembles the escaped source across hljs token spans, giving
    // back the exact diagram source the user typed.
    const source = (code.textContent ?? '').trim();
    if (!source || !SUPPORTED_KEYWORDS.test(source)) continue;

    const pre = code.closest('pre');
    if (!pre) continue;

    try {
      const parsed = await mermaid.parse(source, { suppressErrors: true });
      if (!parsed || !SUPPORTED_TYPES.has(parsed.diagramType)) continue;

      const { svg } = await mermaid.render(`lens-mermaid-${idCounter++}`, source);
      const clean = sanitizeSvg(svg);
      if (!clean.includes('<svg')) continue;

      const figure = document.createElement('div');
      figure.className = 'mermaid-figure';
      // `clean` is inert per sanitizeSvg (see its doc); safe for innerHTML.
      figure.innerHTML = clean;
      pre.replaceWith(figure);
    } catch {
      // Parse/layout/CSP failure: leave the highlighted raw fence untouched.
      continue;
    }
  }
}
