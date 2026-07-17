// Motion-preference resolution tests.
//
// prefersReducedMotion() lets the app `animations` preference (mirrored to
// `data-motion` on <html>) override the OS `prefers-reduced-motion` query.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { prefersReducedMotion } from './index.js';

let originalMatchMedia: typeof window.matchMedia;

function setMotion(value: 'on' | 'off' | 'system' | undefined): void {
  if (value === undefined) delete document.documentElement.dataset.motion;
  else document.documentElement.dataset.motion = value;
}

function mockMatchMedia(matches: boolean): void {
  window.matchMedia = vi.fn().mockReturnValue({ matches }) as unknown as typeof window.matchMedia;
}

beforeEach(() => {
  originalMatchMedia = window.matchMedia;
});

afterEach(() => {
  setMotion(undefined);
  window.matchMedia = originalMatchMedia;
  vi.restoreAllMocks();
});

describe('prefersReducedMotion', () => {
  it("returns false when data-motion='on', even if the OS asks to reduce motion", () => {
    setMotion('on');
    mockMatchMedia(true);
    expect(prefersReducedMotion()).toBe(false);
  });

  it("returns true when data-motion='off', even if the OS allows motion", () => {
    setMotion('off');
    mockMatchMedia(false);
    expect(prefersReducedMotion()).toBe(true);
  });

  it("defers to the OS query when data-motion='system'", () => {
    setMotion('system');
    mockMatchMedia(true);
    expect(prefersReducedMotion()).toBe(true);
    mockMatchMedia(false);
    expect(prefersReducedMotion()).toBe(false);
  });

  it('defers to the OS query when data-motion is unset', () => {
    setMotion(undefined);
    mockMatchMedia(true);
    expect(prefersReducedMotion()).toBe(true);
  });

  it('returns false when matchMedia is unavailable (SSR / no-matchMedia guard)', () => {
    setMotion(undefined);
    // @ts-expect-error simulate an environment without matchMedia
    window.matchMedia = undefined;
    expect(prefersReducedMotion()).toBe(false);
  });
});
