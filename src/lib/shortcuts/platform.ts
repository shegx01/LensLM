import type { Platform } from './binding.js';

let override: Platform | null = null;
let detected: Platform | null = null;

export function detectPlatform(userAgent: string): Platform {
  if (/mac|iphone|ipad|ipod/i.test(userAgent)) return 'darwin';
  if (/windows|win32|win64/i.test(userAgent)) return 'win32';
  return 'linux';
}

/**
 * Test-only override: happy-dom's userAgent is built from `process.platform`, so
 * ambient detection resolves to 'linux' even when the test host is macOS/Windows.
 * Pass `null` to clear the override and fall back to detection.
 */
export function setPlatform(platform: Platform | null): void {
  override = platform;
}

export function currentPlatform(): Platform {
  if (override !== null) return override;
  detected ??= detectPlatform(typeof navigator === 'undefined' ? '' : navigator.userAgent);
  return detected;
}
