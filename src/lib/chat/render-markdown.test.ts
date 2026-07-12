// Unit tests for render-markdown: the sanitized markdown renderer.
// (Inline `[n]` citation markers are no longer stripped here — they are converted
// to inline chips post-render by enhanceCitations; see citation-inline.test.ts.)

import { describe, expect, it } from 'vitest';
import { renderMarkdown } from './render-markdown.js';

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
