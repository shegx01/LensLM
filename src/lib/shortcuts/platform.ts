import type { Platform } from './binding.js';

let override: Platform | null = null;
let detected: Platform | null = null;

export function detectPlatform(userAgent: string): Platform {
  if (/mac|iphone|ipad|ipod/i.test(userAgent)) return 'darwin';
  if (/windows|win32|win64/i.test(userAgent)) return 'win32';
  return 'linux';
}

/**
 * Test-only override: happy-dom's userAgent is (X11; Darwin arm64 …), which matches
 * none of detectPlatform's mac/windows patterns, so ambient detection always lands
 * on 'linux'.
 */
export function setPlatform(platform: Platform | null): void {
  override = platform;
}

export function currentPlatform(): Platform {
  if (override !== null) return override;
  detected ??= detectPlatform(typeof navigator === 'undefined' ? '' : navigator.userAgent);
  return detected;
}
