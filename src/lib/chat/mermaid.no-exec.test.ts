// @vitest-environment jsdom
//
// No-exec + fallback proof for mermaid hydration (security invariant:
// model-generated diagrams are display-only, never executed). happy-dom
// mis-reports DOMPurify, so this runs under jsdom to match the real Chromium
// webview.
//
// mermaid's layout engine can't measure SVG under jsdom (no getBBox/getComputed
// TextLength), so the actual browser render path is exercised by e2e, not here.
// These tests mock the lazy `import('mermaid')` to (a) return a HOSTILE SVG so the
// dedicated SVG sanitize profile is proven to strip script/foreignObject/handlers
// even when mermaid "succeeds", and (b) drive the allowlist + fallback gates that
// are pure JS and DON'T need a real layout engine.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderMarkdown } from './render-markdown.js';

const HOSTILE_SVG =
  '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">' +
  '<script>window.__onExec && window.__onExec()</script>' +
  '<rect width="10" height="10" onload="window.__onExec && window.__onExec()"/>' +
  '<foreignObject><body xmlns="http://www.w3.org/1999/xhtml">' +
  '<img src="x" onerror="window.__onExec && window.__onExec()"></body></foreignObject>' +
  '<a xlink:href="javascript:window.__onExec && window.__onExec()">link</a>' +
  '</svg>';

const parse = vi.fn(async (src: string) => {
  if (/^\s*(?:flowchart|graph)\b/.test(src)) return { diagramType: 'flowchart', config: {} };
  if (/^\s*sequenceDiagram\b/.test(src)) return { diagramType: 'sequence', config: {} };
  return false; // unsupported (e.g. pie) — mermaid.parse returns false w/ suppressErrors
});
const render = vi.fn(async () => ({ svg: HOSTILE_SVG, diagramType: 'flowchart' }));
const initialize = vi.fn();

vi.mock('mermaid', () => ({
  default: { parse, render, initialize }
}));

let hydrateMermaid: (c: HTMLElement) => Promise<void>;

beforeEach(async () => {
  vi.clearAllMocks();
  ({ hydrateMermaid } = await import('./mermaid.js'));
});

afterEach(() => {
  document.body.innerHTML = '';
});

function mount(markdown: string): HTMLElement {
  const el = document.createElement('div');
  el.innerHTML = renderMarkdown(markdown);
  document.body.appendChild(el);
  return el;
}

describe('hydrateMermaid — no-exec + fallback', () => {
  it('AC15: a supported diagram is swapped to inert SVG — no script/foreignObject/on* handler, nothing executes', async () => {
    const onExec = vi.fn();
    (globalThis as unknown as { __onExec?: () => void }).__onExec = onExec;
    const el = mount('```mermaid\nflowchart TD\n  A --> B\n```');

    await hydrateMermaid(el);

    const fig = el.querySelector('.mermaid-figure');
    expect(fig).not.toBeNull();
    expect(fig?.querySelector('svg')).not.toBeNull();
    // The hostile parts of the mocked mermaid SVG are all stripped by the SVG profile.
    expect(fig?.querySelector('script')).toBeNull();
    expect(fig?.querySelector('foreignObject')).toBeNull();
    expect(fig?.innerHTML).not.toMatch(/javascript:/i);
    for (const node of Array.from(fig?.querySelectorAll('*') ?? [])) {
      for (const attr of Array.from(node.attributes)) {
        expect(attr.name.toLowerCase().startsWith('on')).toBe(false);
      }
    }
    // Fire load/error on everything left in the tree — still inert.
    for (const node of Array.from(el.querySelectorAll('*'))) {
      node.dispatchEvent(new Event('error'));
      node.dispatchEvent(new Event('load'));
    }
    expect(onExec).not.toHaveBeenCalled();
  });

  it('AC14: an unsupported diagram type (pie) keeps the raw highlighted code fence', async () => {
    const el = mount('```mermaid\npie title Pets\n  "Dogs" : 40\n```');

    await hydrateMermaid(el);

    expect(el.querySelector('.mermaid-figure')).toBeNull();
    expect(
      el.querySelector('code.language-mermaid, code[class*="language-mermaid"]')
    ).not.toBeNull();
    // render() is never reached for an unsupported type (gated before layout).
    expect(render).not.toHaveBeenCalled();
  });

  it('AC14: a render failure keeps the raw fence (never blank)', async () => {
    render.mockRejectedValueOnce(new Error('layout blew up'));
    const el = mount('```mermaid\nflowchart TD\n  A --> B\n```');

    await hydrateMermaid(el);

    expect(el.querySelector('.mermaid-figure')).toBeNull();
    expect(el.querySelector('code[class*="language-mermaid"]')).not.toBeNull();
    expect(el.textContent).toContain('flowchart TD');
  });

  it('does nothing when there is no mermaid fence (mermaid is never imported)', async () => {
    const el = mount('just **text** and `code`');
    await hydrateMermaid(el);
    expect(parse).not.toHaveBeenCalled();
    expect(render).not.toHaveBeenCalled();
  });

  it('is idempotent / restore-safe: re-running does not double-render', async () => {
    const el = mount('```mermaid\nflowchart TD\n  A --> B\n```');

    await hydrateMermaid(el);
    await hydrateMermaid(el);

    expect(el.querySelectorAll('.mermaid-figure').length).toBe(1);
  });
});
