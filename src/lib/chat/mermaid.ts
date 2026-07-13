// Async, two-pass mermaid hydration. `renderMarkdown` stays synchronous and emits
// ```mermaid fences as ordinary highlighted code blocks; this pass upgrades the
// SUPPORTED subset (flowchart + sequence) into sanitized inline SVG, and leaves
// every other fence as its already-highlighted raw code (never blank).
//
// Security posture (never-execute-model-output): mermaid runs in
// `securityLevel:'strict'` (no HTML labels, no click handlers) with
// `startOnLoad:false`, and its SVG output is re-sanitized through a DEDICATED
// DOMPurify profile that drops `<script>`/`<foreignObject>` and any `on*`
// handler while still blocking `javascript:` URIs — so an SVG synthesized from a
// malicious diagram source is inert (proven by the no-exec test). mermaid itself
// is lazy-imported ONLY when a mermaid fence is present, keeping it off the boot
// bundle and off the synchronous render path.
//
// Re-runnable / restore-safe like enhanceCitations: each call re-reads the
// original source from the (preserved) fence, so re-render/theme-change is
// idempotent and a previously-hydrated block never double-renders.

import DOMPurify from 'dompurify';

// The eval-free subset we commit to rendering. `mermaid.parse` reports these
// `diagramType` values; anything else (or a parse failure) keeps the raw fence.
const SUPPORTED_TYPES: ReadonlySet<string> = new Set(['flowchart', 'flowchart-v2', 'sequence']);

// Leading keyword → diagram family, checked BEFORE invoking mermaid (Interpretation
// B: allowlist gate first, runtime try/catch as the catch-all). Cheap pre-filter so
// unsupported types never reach the layout engine.
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

// The SVG profile already drops every `on*` event handler, `<script>` and
// `javascript:` URIs; `foreignObject` is forbidden explicitly so no HTML (and thus
// no img/onerror) can ride inside the SVG. Verified inert by the no-exec test.
function sanitizeSvg(svg: string): string {
  return DOMPurify.sanitize(svg, {
    USE_PROFILES: { svg: true, svgFilters: true },
    // flowchart/sequence output never needs external-image elements or remote URIs;
    // forbid them so exfil beacons don't rely on CSP img-src alone (defense-in-depth).
    FORBID_TAGS: ['script', 'foreignObject', 'image', 'feImage'],
    ALLOWED_URI_REGEXP: /^#/
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
      // Sanitized SVG string from the dedicated profile above — inert, no <script>,
      // no foreignObject, no on* handlers (proven by mermaid no-exec test).
      figure.innerHTML = clean;
      pre.replaceWith(figure);
    } catch {
      // Parse/layout/CSP failure: leave the highlighted raw fence untouched.
      continue;
    }
  }
}
