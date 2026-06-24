import { describe, expect, it } from 'vitest';
import { notebookAccentClass, NOTEBOOK_PALETTE } from './notebook-color.js';

describe('NOTEBOOK_PALETTE', () => {
  it('has 10 distinct decorative hues', () => {
    expect(NOTEBOOK_PALETTE).toHaveLength(10);
    expect(new Set(NOTEBOOK_PALETTE).size).toBe(10);
  });

  it('contains the four added hues (teal, orange, pink, indigo)', () => {
    for (const hue of ['teal', 'orange', 'pink', 'indigo'] as const) {
      expect(NOTEBOOK_PALETTE).toContain(hue);
    }
  });
});

describe('notebookAccentClass (hash fallback)', () => {
  describe('determinism', () => {
    it('returns the same class for the same id every time', () => {
      const id = '018f4c7e-dead-7b00-beef-000000000001';
      const first = notebookAccentClass(id);
      const second = notebookAccentClass(id);
      expect(first).toBe(second);
    });

    it('returns a class in the form nb-{paletteId}', () => {
      const result = notebookAccentClass('some-id');
      expect(result).toMatch(
        /^nb-(purple|blue|green|teal|amber|orange|rose|pink|indigo|graphite)$/
      );
    });
  });

  describe('distribution', () => {
    it('only returns classes based on the 10-hue NOTEBOOK_PALETTE', () => {
      const valid = new Set(NOTEBOOK_PALETTE.map((p) => `nb-${p}`));
      for (let i = 0; i < 200; i++) {
        expect(valid).toContain(notebookAccentClass(`test-id-${i}`));
      }
    });

    it('produces multiple distinct classes across a varied set of ids', () => {
      const ids = Array.from({ length: 50 }, (_, i) => `018f4c7e-aaa0-7b00-0000-${i}`);
      const seen = new Set(ids.map(notebookAccentClass));
      // With 50 varied ids over 10 buckets we expect broad coverage.
      expect(seen.size).toBeGreaterThan(5);
    });
  });
});
