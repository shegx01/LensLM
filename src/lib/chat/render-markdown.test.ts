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
});
