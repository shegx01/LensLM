import type { Platform } from './binding.js';

let override: Platform | null = null;
let detected: Platform | null = null;

export function detectPlatform(userAgent: string): Platform {
  if (/mac|iphone|ipad|ipod/i.test(userAgent)) return 'darwin';
  if (/windows|win32|win64/i.test(userAgent)) return 'win32';
  return 'linux';
}

export function setPlatform(platform: Platform | null): void {
  override = platform;
}

export function currentPlatform(): Platform {
  if (override !== null) return override;
  detected ??= detectPlatform(typeof navigator === 'undefined' ? '' : navigator.userAgent);
  return detected;
}
