// Unit tests for render-markdown: the citation-marker strip (AC1 matrix) and
// the sanitized markdown renderer.

import { describe, expect, it } from 'vitest';
import { renderMarkdown, stripCitationMarkers } from './render-markdown.js';

const ORD = new Set([1, 2]);

describe('stripCitationMarkers — AC1 matrix', () => {
  it('removes a marker plus its single leading space (no double space)', () => {
    expect(stripCitationMarkers('a [1] b', ORD)).toBe('a b');
  });

  it('removes an adjacent-to-word marker (no leading space)', () => {
    expect(stripCitationMarkers('grew 34%[1]. Margins[2].', ORD)).toBe('grew 34%. Margins.');
  });

  it('strips adjacent markers both', () => {
    expect(stripCitationMarkers('[1][2] done', ORD)).toBe(' done');
  });

  it('preserves a literal numeric bracket whose number is not an ordinal', () => {
    expect(stripCitationMarkers('arr[0] holds 5', ORD)).toBe('arr[0] holds 5');
  });

  it('preserves an out-of-range marker (number not an ordinal)', () => {
    expect(stripCitationMarkers('see [9] appendix', ORD)).toBe('see [9] appendix');
  });

  it('preserves a reference-style definition [1]: url', () => {
    expect(stripCitationMarkers('[1]: http://x', ORD)).toBe('[1]: http://x');
  });

  it('preserves a markdown link [1](url) and a non-numeric labelled link', () => {
    expect(stripCitationMarkers('[1](http://y) and [see](http://x)', ORD)).toBe(
      '[1](http://y) and [see](http://x)'
    );
  });

  it('strips a real-ordinal marker inside fenced code (documented accepted behavior)', () => {
    expect(stripCitationMarkers('```\ncode [1]\n```', ORD)).toBe('```\ncode\n```');
  });

  it('returns the input unchanged when the ordinal set is empty', () => {
    const src = 'a [1] b [2] c';
    expect(stripCitationMarkers(src, new Set())).toBe(src);
  });
});

describe('renderMarkdown', () => {
  it('renders GFM markdown to sanitized HTML', () => {
    const html = renderMarkdown('**bold** and _em_');
    expect(html).toContain('<strong>bold</strong>');
    expect(html).toContain('<em>em</em>');
  });

  it('highlights a fenced block with an explicit language hint', () => {
    const html = renderMarkdown('```js\nconst x = 1;\n```');
    expect(html).toContain('language-js');
    expect(html).toContain('hljs');
    expect(html).toMatch(/class="hljs-\w/);
  });

  it('auto-detects and highlights a fenced block with no language', () => {
    const html = renderMarkdown('```\nfunction greet() { return "hi"; }\n```');
    expect(html).toContain('hljs');
    expect(html).toMatch(/class="hljs-\w/);
  });

  it('falls back to auto-detect for an unknown language hint without throwing', () => {
    const render = () => renderMarkdown('```notalang\nlet y = 2;\n```');
    expect(render).not.toThrow();
    expect(render()).toContain('hljs');
  });

  it('does not wrap inline code in hljs spans', () => {
    const html = renderMarkdown('Use `const x = 1` inline.');
    expect(html).toContain('<code>');
    expect(html).not.toMatch(/class="hljs-\w/);
  });

  it('neutralizes dangerous markup inside a code block', () => {
    const html = renderMarkdown('```\n<script>alert(1)</script>\n```');
    expect(html).not.toContain('<script>');
    expect(html).not.toContain('</script>');
    expect(html).toContain('&lt;');
    expect(html).toContain('&gt;');
  });

  it('neutralizes an onerror image inside a code block', () => {
    const html = renderMarkdown('```html\n<img src=x onerror=alert(1)>\n```');
    expect(html).not.toMatch(/<img[^>]*onerror/i);
    expect(html).not.toMatch(/\sonerror=/i);
    expect(html).toContain('&lt;');
  });

  it('renders the onerror payload only as highlighted text, never a live attribute', () => {
    const html = renderMarkdown('```html\n<img src=x onerror=alert(1)>\n```');
    // No live element/attribute survived: the dangerous markup is escaped text.
    expect(html).not.toMatch(/<img\b/i);
    expect(html).not.toMatch(/\bonerror\s*=/i);
    // The `onerror` word appears only inside an hljs-classed span (escaped source).
    const onerrorSpan = /<span class="hljs-[\w-]+"[^>]*>[^<]*onerror[^<]*<\/span>/i;
    expect(html).toMatch(onerrorSpan);
  });

  it('leaves a plain markdown paragraph unaffected', () => {
    const html = renderMarkdown('Hello **world**.');
    expect(html).toContain('<strong>world</strong>');
    expect(html).not.toContain('hljs');
  });

  it('skips hljs spans on a fenced block when highlight is disabled', () => {
    const html = renderMarkdown('```js\nconst x = 1;\n```', { highlight: false });
    expect(html).not.toMatch(/class="hljs-\w/);
    expect(html).not.toContain('hljs-');
    expect(html).toContain('<code');
  });

  it('emits hljs spans on a fenced block by default', () => {
    const html = renderMarkdown('```js\nconst x = 1;\n```', { highlight: true });
    expect(html).toMatch(/class="hljs-\w/);
  });

  it('escapes dangerous markup even on the un-highlighted path', () => {
    const html = renderMarkdown('```html\n<img src=x onerror=alert(1)>\n```', { highlight: false });
    // The whole payload is escaped text inside <code>, not a live element.
    expect(html).not.toMatch(/<img\b/i);
    expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;');
  });

  it('renders an empty fenced block without throwing', () => {
    expect(() => renderMarkdown('```js\n```')).not.toThrow();
    expect(() => renderMarkdown('```js\n```', { highlight: false })).not.toThrow();
  });
});
