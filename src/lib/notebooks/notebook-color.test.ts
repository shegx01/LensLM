import { describe, expect, it } from 'vitest';
import { notebookAccentClass } from './notebook-color.js';
import { ACCENT_IDS } from '$lib/theme/accents.js';

describe('notebookAccentClass', () => {
  describe('determinism', () => {
    it('returns the same class for the same id every time', () => {
      const id = '018f4c7e-dead-7b00-beef-000000000001';
      const first = notebookAccentClass(id);
      const second = notebookAccentClass(id);
      expect(first).toBe(second);
    });

    it('returns a class in the form nb-{accentId}', () => {
      const result = notebookAccentClass('some-id');
      expect(result).toMatch(/^nb-(purple|green|blue|amber|rose|graphite)$/);
    });
  });

  describe('distribution', () => {
    it('produces all 6 accent classes across a varied set of ids', () => {
      // Use sufficiently varied ids to hit all 6 buckets.
      const ids = [
        'aaaa',
        'bbbb',
        'cccc',
        'dddd',
        'eeee',
        'ffff',
        '1111',
        '2222',
        '3333',
        '4444',
        '5555',
        '6666',
        'uuid-alpha',
        'uuid-beta',
        'uuid-gamma',
        'uuid-delta',
        'uuid-epsilon',
        'uuid-zeta',
        '018f4c7e-aaa0-7b00-0000-000000000001',
        '018f4c7e-bbb0-7b00-0000-000000000002',
        '018f4c7e-ccc0-7b00-0000-000000000003',
        '018f4c7e-ddd0-7b00-0000-000000000004',
        '018f4c7e-eee0-7b00-0000-000000000005',
        '018f4c7e-fff0-7b00-0000-000000000006'
      ];

      const seen = new Set<string>();
      for (const id of ids) {
        seen.add(notebookAccentClass(id));
      }

      // All 6 accent class names should appear.
      const expected = new Set(ACCENT_IDS.map((a) => `nb-${a}`));
      for (const cls of expected) {
        expect(seen, `expected "${cls}" to appear in distribution`).toContain(cls);
      }
    });

    it('different ids produce different classes (not all identical)', () => {
      const classes = ['a', 'b', 'c', 'd', 'e', 'f'].map(notebookAccentClass);
      const unique = new Set(classes);
      // At least 2 distinct values in a set of 6 short ids.
      expect(unique.size).toBeGreaterThan(1);
    });
  });

  describe('accent alignment', () => {
    it('only returns classes based on ACCENT_IDS', () => {
      const validClasses = new Set(ACCENT_IDS.map((a) => `nb-${a}`));
      for (let i = 0; i < 100; i++) {
        const result = notebookAccentClass(`test-id-${i}`);
        expect(validClasses).toContain(result);
      }
    });
  });
});
