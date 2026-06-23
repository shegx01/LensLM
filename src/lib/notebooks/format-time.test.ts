import { describe, expect, it } from 'vitest';
import { formatRelativeTime, formatSourceCount } from './format-time.js';

/** Build an ISO string that is `deltaMs` milliseconds before `now`. */
function ago(now: number, deltaMs: number): string {
  return new Date(now - deltaMs).toISOString();
}

const SEC = 1000;
const MIN = 60 * SEC;
const HOUR = 60 * MIN;
const DAY = 24 * HOUR;
const WEEK = 7 * DAY;

const NOW = Date.now();

describe('formatRelativeTime', () => {
  describe('just now (< 60 seconds)', () => {
    it('returns "just now" for 0 seconds', () => {
      expect(formatRelativeTime(ago(NOW, 0), NOW)).toBe('just now');
    });

    it('returns "just now" for 30 seconds', () => {
      expect(formatRelativeTime(ago(NOW, 30 * SEC), NOW)).toBe('just now');
    });

    it('returns "just now" for 59 seconds', () => {
      expect(formatRelativeTime(ago(NOW, 59 * SEC), NOW)).toBe('just now');
    });
  });

  describe('minutes ago (1m – 59m)', () => {
    it('returns "1m ago" at exactly 60 seconds', () => {
      expect(formatRelativeTime(ago(NOW, 60 * SEC), NOW)).toBe('1m ago');
    });

    it('returns "2m ago" at 2 minutes', () => {
      expect(formatRelativeTime(ago(NOW, 2 * MIN), NOW)).toBe('2m ago');
    });

    it('returns "59m ago" at 59 minutes', () => {
      expect(formatRelativeTime(ago(NOW, 59 * MIN), NOW)).toBe('59m ago');
    });
  });

  describe('hours ago (1h – 23h)', () => {
    it('returns "1h ago" at exactly 60 minutes', () => {
      expect(formatRelativeTime(ago(NOW, 60 * MIN), NOW)).toBe('1h ago');
    });

    it('returns "2h ago" at 2 hours', () => {
      expect(formatRelativeTime(ago(NOW, 2 * HOUR), NOW)).toBe('2h ago');
    });

    it('returns "23h ago" at 23h 59m', () => {
      expect(formatRelativeTime(ago(NOW, 23 * HOUR + 59 * MIN), NOW)).toBe('23h ago');
    });
  });

  describe('days ago (1d – 6d)', () => {
    it('returns "1d ago" at exactly 24 hours', () => {
      expect(formatRelativeTime(ago(NOW, 24 * HOUR), NOW)).toBe('1d ago');
    });

    it('returns "3d ago" at 3 days', () => {
      expect(formatRelativeTime(ago(NOW, 3 * DAY), NOW)).toBe('3d ago');
    });

    it('returns "6d ago" at 6 days 23h', () => {
      expect(formatRelativeTime(ago(NOW, 6 * DAY + 23 * HOUR), NOW)).toBe('6d ago');
    });
  });

  describe('weeks ago (1w – 4w)', () => {
    it('returns "1w ago" at exactly 7 days', () => {
      expect(formatRelativeTime(ago(NOW, WEEK), NOW)).toBe('1w ago');
    });

    it('returns "3w ago" at 3 weeks', () => {
      expect(formatRelativeTime(ago(NOW, 3 * WEEK), NOW)).toBe('3w ago');
    });

    it('returns "4w ago" at 29 days', () => {
      expect(formatRelativeTime(ago(NOW, 29 * DAY), NOW)).toBe('4w ago');
    });
  });

  describe('months ago (>= 5 weeks)', () => {
    it('returns "1mo ago" at 35 days (5 weeks)', () => {
      expect(formatRelativeTime(ago(NOW, 35 * DAY), NOW)).toBe('1mo ago');
    });

    it('returns "3mo ago" at 90 days', () => {
      expect(formatRelativeTime(ago(NOW, 90 * DAY), NOW)).toBe('3mo ago');
    });
  });

  describe('edge / error cases', () => {
    it('returns empty string for an invalid date string', () => {
      expect(formatRelativeTime('not-a-date', NOW)).toBe('');
    });
  });
});

describe('formatSourceCount', () => {
  it('returns singular "1 source" for exactly one', () => {
    expect(formatSourceCount(1)).toBe('1 source');
  });

  it('returns plural "0 sources" for zero', () => {
    expect(formatSourceCount(0)).toBe('0 sources');
  });

  it('returns plural "N sources" for more than one', () => {
    expect(formatSourceCount(3)).toBe('3 sources');
    expect(formatSourceCount(42)).toBe('42 sources');
  });
});
