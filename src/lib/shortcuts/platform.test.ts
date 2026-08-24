import { afterEach, describe, expect, it } from 'vitest';
import { currentPlatform, detectPlatform, setPlatform } from './platform.js';

const MAC_UA =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15';
const WINDOWS_UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';
const LINUX_UA =
  'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';

afterEach(() => {
  setPlatform(null);
});

describe('detectPlatform', () => {
  it.each([
    [MAC_UA, 'darwin'],
    ['Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)', 'darwin'],
    [WINDOWS_UA, 'win32'],
    [LINUX_UA, 'linux'],
    ['', 'linux']
  ])('maps a %s user agent', (userAgent, expected) => {
    expect(detectPlatform(userAgent)).toBe(expected);
  });
});

describe('currentPlatform', () => {
  it('returns a known platform without an override', () => {
    expect(['darwin', 'win32', 'linux']).toContain(currentPlatform());
  });

  it.each(['darwin', 'win32', 'linux'] as const)('lets setPlatform(%s) win over detection', (p) => {
    setPlatform(p);
    expect(currentPlatform()).toBe(p);
  });

  it('falls back to detection once the override is cleared', () => {
    setPlatform('win32');
    expect(currentPlatform()).toBe('win32');

    setPlatform(null);
    expect(currentPlatform()).toBe(detectPlatform(navigator.userAgent));
  });

  it('memoizes detection', () => {
    expect(currentPlatform()).toBe(currentPlatform());
  });
});
