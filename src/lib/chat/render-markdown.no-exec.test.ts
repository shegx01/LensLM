// @vitest-environment jsdom
//
// No-exec proof for KaTeX math output (AC17): the math sanitize path re-allows
// inline `style` ONLY on KaTeX nodes, so this proves that even with `style`
// re-enabled a `$…$` payload carrying an injected handler renders inert (no
// executable markup, no event handler fires). Runs under jsdom (not happy-dom,
// which mis-reports DOMPurify) to match the real Chromium webview.

import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderMarkdown } from './render-markdown.js';

function mount(markdown: string): HTMLElement {
  const el = document.createElement('div');
  el.innerHTML = renderMarkdown(markdown);
  document.body.appendChild(el);
  return el;
}

afterEach(() => {
  delete (globalThis as unknown as { __onExec?: () => void }).__onExec;
});

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

describe('renderMarkdown links — no-exec sanitization (#182)', () => {
  it('keeps the anchor but strips a javascript: href, and no injected handler fires', () => {
    const onExec = vi.fn();
    (globalThis as unknown as { __onExec?: () => void }).__onExec = onExec;
    const a = mount('[x](javascript:window.__onExec&&window.__onExec())').querySelector('a');
    expect(a).not.toBeNull();
    // Anchor text survives, the javascript: href is dropped entirely (asserting
    // absence, not just "not javascript:", so a stripped-to-empty href fails too).
    // The click is belt-and-suspenders — jsdom won't navigate a javascript: URL.
    expect(a?.getAttribute('href')).toBeNull();
    a?.dispatchEvent(new MouseEvent('click'));
    expect(onExec).not.toHaveBeenCalled();
  });

  it('keeps the anchor but strips a data: href', () => {
    const a = mount('[x](data:text/html,<b>x</b>)').querySelector('a');
    expect(a).not.toBeNull();
    expect(a?.getAttribute('href')).toBeNull();
  });

  it('drops a scheme allowed by default DOMPurify but not by our ALLOWED_URI_REGEXP', () => {
    // Pins the CUSTOM regexp: tel: passes DOMPurify's built-in URI filter, so this
    // fails if our http(s)/mailto-only config is ever loosened back to the default.
    const a = mount('[call](tel:+15551234567)').querySelector('a');
    expect(a).not.toBeNull();
    expect(a?.getAttribute('href')).toBeNull();
  });

  it('preserves benign https: and mailto: hrefs (positive control)', () => {
    expect(mount('[s](https://example.com/x)').querySelector('a')?.getAttribute('href')).toBe(
      'https://example.com/x'
    );
    expect(mount('[m](mailto:a@b.com)').querySelector('a')?.getAttribute('href')).toBe(
      'mailto:a@b.com'
    );
  });
});

describe('renderMarkdown — dangerous tags stripped (#182 invariant lock)', () => {
  it('removes iframe, object, and svg <script> under the custom DOMPurify config', () => {
    const el = mount(
      '<iframe src="https://evil.example"></iframe>' +
        '<object data="x"></object>' +
        '<svg><script>window.__x=1</script></svg>'
    );
    expect(el.querySelector('iframe')).toBeNull();
    expect(el.querySelector('object')).toBeNull();
    expect(el.querySelector('script')).toBeNull();
  });
});

describe('renderMarkdown — forged class="katex" cannot smuggle style (no clickjacking overlay)', () => {
  it('strips a full-viewport position:fixed overlay style even under a forged katex class', () => {
    const el = mount(
      '<span class="katex"><span style="position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:9999;background:red">x</span></span>'
    );
    for (const node of Array.from(el.querySelectorAll('[style]'))) {
      const style = node.getAttribute('style') ?? '';
      expect(style).not.toMatch(/position\s*:\s*fixed/i);
      expect(style).not.toMatch(/100v[wh]/i);
    }
  });

  it('strips style on a forged top-level class="katex" element', () => {
    const el = mount('<div class="katex" style="position:fixed;inset:0">overlay</div>');
    const div = el.querySelector('div.katex');
    expect(div?.getAttribute('style')).toBeNull();
  });

  it('keeps a genuine KaTeX-shaped length style (allow-list positive case)', () => {
    const el = mount(
      '<span class="katex"><span style="height:0.8em;vertical-align:-0.2em">x</span></span>'
    );
    expect(el.querySelector('.katex [style]')).not.toBeNull();
  });
});
