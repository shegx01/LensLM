// Project animation toolkit.
//
// Uses the `motion` imperative API (not `<motion.div>`) because the wrapper
// ships uncompiled TS `.svelte` files the Vite/rolldown bundler can't process.
// SYNC-CHECK: add shared transitions/actions here, not inline in components.

import type { Action } from 'svelte/action';
import type { TransitionConfig } from 'svelte/transition';
import { cubicOut, expoOut } from 'svelte/easing';
import { animate, inView } from 'motion';

const REDUCE_MOTION_QUERY = '(prefers-reduced-motion: reduce)';

/**
 * True when motion should be reduced. `data-motion` on `<html>` overrides the OS
 * query: `'on'` forces motion, `'off'` forces calm, `'system'`/unset defers to it.
 */
export function prefersReducedMotion(): boolean {
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

export interface EnterRiseParams {
  /** Vertical offset (px) the element rises from. Default 10. */
  y?: number;
  /** Duration in ms. Default 420. */
  duration?: number;
  /** Delay in ms — pass `index * step` for a staggered group. Default 0. */
  delay?: number;
}

/**
 * Svelte `in:` transition — transform+opacity rise. As an intro on a keyed
 * `{#each}`, Svelte skips it for items present at initial mount and plays it only
 * for items added later, which is exactly what chat wants: history renders
 * instantly, new messages ease in. Transform/opacity only (never height), so it
 * can't fight the transcript's scroll-height autoscroll. Honors "reduce motion"
 * with a bare opacity fade (no movement).
 */
export function enterRise(
  _node: HTMLElement,
  { y = 10, duration = 420, delay = 0 }: EnterRiseParams = {}
): TransitionConfig {
  if (prefersReducedMotion()) return { duration: 160, delay, css: (t) => `opacity: ${t};` };
  return {
    duration,
    delay,
    easing: expoOut,
    css: (t) => `opacity: ${t}; transform: translateY(${(1 - t) * y}px);`
  };
}

/**
 * Svelte `in:` transition — scale+opacity pop from 0.8 (never from 0; nothing in
 * the real world appears from nothing). For swap flourishes like the composer's
 * send↔stop morph. Skipped on initial mount by Svelte; plays on later swaps.
 */
export function popIn(
  _node: HTMLElement,
  { duration = 220 }: { duration?: number } = {}
): TransitionConfig {
  if (prefersReducedMotion()) return { duration: 120, css: (t) => `opacity: ${t};` };
  return {
    duration,
    easing: expoOut,
    css: (t) => `opacity: ${t}; transform: scale(${0.8 + 0.2 * t});`
  };
}
