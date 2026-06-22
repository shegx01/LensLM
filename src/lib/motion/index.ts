// Project animation toolkit. Two kinds of primitive live here:
//   • Svelte actions built on the `motion` engine (framer-motion's vanilla
//     `animate`/`inView`) — for mount/scroll entrances on persistent elements.
//     We use the imperative API, not the `<motion.div>` wrapper, because that
//     wrapper ships uncompiled TS `.svelte` files our Vite 8 / rolldown bundler
//     can't process.
//   • Svelte transitions — for conditionally-rendered ({#if}) content, where a
//     single coordinated tween (and proper outro-before-unmount) matters.
//
// SYNC-CHECK: add new shared transitions/actions here, never inline in a
// component, so timing/easing stays consistent across the app.

import type { Action } from 'svelte/action';
import type { TransitionConfig } from 'svelte/transition';
import { cubicOut } from 'svelte/easing';
import { animate, inView } from 'motion';

const REDUCE_MOTION_QUERY = '(prefers-reduced-motion: reduce)';

/** True when the OS asks for reduced motion (or we're not in a browser). */
function prefersReducedMotion(): boolean {
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

// A premium ease-out curve (≈ easeOutExpo): quick start, gentle settle.
const EASE_OUT: [number, number, number, number] = [0.16, 1, 0.3, 1];

/**
 * Svelte action: fade + rise an element in. Honors the OS "reduce motion"
 * setting (snaps straight to the final state, no animation).
 *
 *   <div use:fadeRise={{ delay: i * 0.06 }}>…</div>
 */
export const fadeRise: Action<HTMLElement, FadeRiseParams | undefined> = (
  node,
  params
) => {
  const { y = 8, duration = 0.4, delay = 0, whenInView = false } = params ?? {};

  // Snap straight to visible (no animation) when the user asked for reduced
  // motion, or when the Web Animations API isn't available — e.g. the test DOM
  // (happy-dom) or any environment lacking `element.animate`. motion's
  // `animate()` relies on WAAPI, so guard before touching it.
  if (prefersReducedMotion() || typeof node.animate !== 'function') {
    node.style.opacity = '1';
    return {};
  }

  // Hide synchronously so there is no flash of the final state before the
  // animation's first frame.
  node.style.opacity = '0';

  const play = () =>
    animate(node, { opacity: [0, 1], y: [y, 0] }, { duration, delay, ease: EASE_OUT });

  let controls: ReturnType<typeof animate> | undefined;
  let stopInView: (() => void) | undefined;

  if (whenInView) {
    stopInView = inView(node, () => {
      controls = play();
      // Animate once, then stop observing.
      return () => {};
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
 * Svelte transition for conditionally-rendered ({#if}) content: expand/collapse
 * by animating height AND opacity in ONE coordinated tween. Doing both in a
 * single transition (rather than a height `slide` + a separate opacity action)
 * is what makes it smooth — there are no competing animations on nested nodes
 * to desync. Lazy-mounted, so collapsed content stays out of the DOM + focus
 * order (a11y). Honors the OS "reduce motion" setting (snaps, no animation).
 *
 *   {#if open}
 *     <div transition:expandFade>…panel…</div>
 *   {/if}
 */
export function expandFade(
  node: HTMLElement,
  { duration = 300, easing = cubicOut }: ExpandFadeParams = {}
): TransitionConfig {
  if (prefersReducedMotion()) return { duration: 0 };

  const height = node.scrollHeight;
  // Opacity reaches full a little before height does, so the content is solid
  // as the panel finishes opening (and clears early as it closes).
  return {
    duration,
    easing,
    css: (t) =>
      `overflow: hidden; height: ${t * height}px; opacity: ${Math.min(1, t * 1.4)};`
  };
}
