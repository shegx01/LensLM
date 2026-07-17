// Project animation toolkit.
//
// Uses the `motion` imperative API (not `<motion.div>`) because the wrapper
// ships uncompiled TS `.svelte` files the Vite/rolldown bundler can't process.
// SYNC-CHECK: add shared transitions/actions here, not inline in components.

import type { Action } from 'svelte/action';
import type { TransitionConfig } from 'svelte/transition';
import { cubicOut } from 'svelte/easing';
import { animate, inView } from 'motion';

const REDUCE_MOTION_QUERY = '(prefers-reduced-motion: reduce)';

/**
 * True when motion should be reduced. The app `animations` preference (mirrored
 * to `data-motion` on `<html>`) overrides the OS query: `'on'` forces motion,
 * `'off'` forces calm, `'system'`/unset defers to the OS `prefers-reduced-motion`.
 */
function prefersReducedMotion(): boolean {
  if (typeof document !== 'undefined') {
    const motion = document.documentElement.dataset.motion;
    if (motion === 'on') return false;
    if (motion === 'off') return true;
  }
  if (typeof window === 'undefined' || !window.matchMedia) return false;
  return window.matchMedia(REDUCE_MOTION_QUERY).matches;
}

export interface FadeRiseParams {
  /** Vertical offset (px) the element rises from. Default 8. */
  y?: number;
  /** Duration in seconds. Default 0.4. */
  duration?: number;
  /** Delay in seconds — pass `index * step` for a staggered group. Default 0. */
  delay?: number;
  /**
   * When true, animate the first time the element scrolls into view instead of
   * on mount. Useful for content below the fold. Default false (animate on mount).
   */
  whenInView?: boolean;
}

const EASE_OUT: [number, number, number, number] = [0.16, 1, 0.3, 1]; // ≈ easeOutExpo

/** Svelte action: fade + rise. Honors OS "reduce motion" (snaps, no animation). */
export const fadeRise: Action<HTMLElement, FadeRiseParams | undefined> = (node, params) => {
  const { y = 8, duration = 0.4, delay = 0, whenInView = false } = params ?? {};

  // WAAPI guard: happy-dom and reduced-motion environments lack `element.animate`.
  if (prefersReducedMotion() || typeof node.animate !== 'function') {
    node.style.opacity = '1';
    return {};
  }

  node.style.opacity = '0'; // hide synchronously to prevent flash before first frame

  const play = () =>
    animate(node, { opacity: [0, 1], y: [y, 0] }, { duration, delay, ease: EASE_OUT });

  let controls: ReturnType<typeof animate> | undefined;
  let stopInView: (() => void) | undefined;

  if (whenInView) {
    stopInView = inView(node, () => {
      controls = play();
      return () => {}; // animate once, then stop observing
    });
  } else {
    controls = play();
  }

  return {
    destroy() {
      controls?.stop();
      stopInView?.();
    }
  };
};

export interface ExpandFadeParams {
  /** Duration in ms. Default 300. */
  duration?: number;
  /** Easing function. Default cubicOut (smooth decelerate, no flat tail). */
  easing?: (t: number) => number;
}

/**
 * Svelte transition for `{#if}` content: height + opacity in one tween so there
 * are no competing animations to desync. Lazy-mounted (a11y). Honors "reduce motion".
 */
export function expandFade(
  node: HTMLElement,
  { duration = 300, easing = cubicOut }: ExpandFadeParams = {}
): TransitionConfig {
  if (prefersReducedMotion()) return { duration: 0 };

  const height = node.scrollHeight;
  return {
    duration,
    easing,
    css: (t) => `overflow: hidden; height: ${t * height}px; opacity: ${Math.min(1, t * 1.4)};`
  };
}
