import { describe, expect, it } from 'vitest';
import { renderMarkdown } from './render-markdown.js';

describe('renderMarkdown', () => {
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

  it('leaves a plain markdown paragraph unaffected', () => {
    const html = renderMarkdown('Hello **world**.');
    expect(html).toContain('<strong>world</strong>');
    expect(html).not.toContain('hljs');
  });
});
