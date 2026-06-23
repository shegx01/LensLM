import { describe, expect, it } from 'vitest';
import { getInitials } from './format.js';

describe('getInitials', () => {
  it('returns the first two initials of a two-word name, uppercased', () => {
    expect(getInitials('Ada Lovelace')).toBe('AL');
  });

  it('returns a single initial for a one-word name', () => {
    expect(getInitials('Grace')).toBe('G');
  });

  it('uses only the first two words when more are present', () => {
    expect(getInitials('John Ronald Reuel Tolkien')).toBe('JR');
  });

  it('collapses repeated whitespace between words', () => {
    expect(getInitials('  Alan   Turing  ')).toBe('AT');
  });

  it('falls back to "?" for an empty string', () => {
    expect(getInitials('')).toBe('?');
  });

  it('falls back to "?" for whitespace-only input', () => {
    expect(getInitials('   ')).toBe('?');
  });
});
