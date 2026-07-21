// Build-vendored provider brand-mark SVGs (see scripts/fetch-provider-logos.sh).
// Eager raw import so lookup is synchronous — no runtime fetch, no CSP impact.

// New logos MUST be added via `scripts/fetch-provider-logos.sh --record` (sanitized + checksummed), never hand-dropped.
const logos = import.meta.glob('$lib/assets/provider-logos/*.svg', {
  query: '?raw',
  import: 'default',
  eager: true
}) as Record<string, string>;

/** Provider id → raw SVG markup, keyed by the vendored filename (no extension). */
const logosById: Record<string, string> = Object.fromEntries(
  Object.entries(logos).map(([path, svg]) => {
    const id =
      path
        .split('/')
        .pop()
        ?.replace(/\.svg$/, '') ?? path;
    return [id, svg];
  })
);

// Belt-and-suspenders: reject a hand-dropped SVG that bypassed the fetch-time sanitizer before it reaches {@html}.
const UNSAFE_SVG =
  /<script|\son\w+=|<foreignObject|<use\b|<image\b|<style|(?:xlink:)?href\s*=\s*["']?\s*(?:javascript|data):/i;

/** Raw inline SVG markup for a bundled provider mark, or `null` if not vendored (monogram fallback). */
export function providerLogo(id: string): string | null {
  const svg = logosById[id] ?? null;
  if (svg !== null && UNSAFE_SVG.test(svg)) {
    return null;
  }
  return svg;
}

/** Monogram fallback for providers with no bundled mark. */
export function providerMonogram(name: string): string {
  return name.trim().charAt(0).toUpperCase();
}
