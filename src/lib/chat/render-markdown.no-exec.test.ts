// @vitest-environment jsdom
//
// No-exec proof for KaTeX math output (AC17): the math sanitize path re-allows
// inline `style` ONLY on KaTeX nodes, so this proves that even with `style`
// re-enabled a `$…$` payload carrying an injected handler renders inert (no
// executable markup, no event handler fires). Runs under jsdom (not happy-dom,
// which mis-reports DOMPurify) to match the real Chromium webview.

import { describe, expect, it, vi } from 'vitest';
import { renderMarkdown } from './render-markdown.js';

function mount(markdown: string): HTMLElement {
  const el = document.createElement('div');
  el.innerHTML = renderMarkdown(markdown);
  document.body.appendChild(el);
  return el;
}

describe('renderMarkdown math — no-exec sanitization (AC17)', () => {
  it('renders KaTeX math but neutralizes an injected handler in the same content', () => {
    const onExec = vi.fn();
    (globalThis as unknown as { __onExec?: () => void }).__onExec = onExec;
    const el = mount('math $x^2$ and <img src=x onerror="window.__onExec && window.__onExec()">');

    expect(el.querySelector('.katex')).not.toBeNull();
    const img = el.querySelector('img');
    if (img) {
      expect(img.getAttribute('onerror')).toBeNull();
      img.dispatchEvent(new Event('error'));
    }
    expect(onExec).not.toHaveBeenCalled();
  });

  it('keeps KaTeX inline style but no on* handler survives on any math node', () => {
    const el = mount('$\\frac{a}{b} + c^2$');
    const katex = el.querySelector('.katex');
    expect(katex).not.toBeNull();
    // A style-bearing strut proves the sanitize hook re-kept KaTeX inline style.
    expect(el.querySelector('.katex [style]')).not.toBeNull();
    for (const node of Array.from(el.querySelectorAll('.katex *'))) {
      for (const attr of Array.from(node.attributes)) {
        expect(attr.name.toLowerCase().startsWith('on')).toBe(false);
      }
    }
  });

  it('does not execute a <script> smuggled next to math', () => {
    const onExec = vi.fn();
    (globalThis as unknown as { __onExec?: () => void }).__onExec = onExec;
    const el = mount('$a+b$ <script>window.__onExec && window.__onExec()</script>');
    expect(el.querySelector('.katex')).not.toBeNull();
    expect(el.querySelector('script')).toBeNull();
    expect(onExec).not.toHaveBeenCalled();
  });
});
