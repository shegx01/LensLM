// Project motion primitives, built on the `motion` engine (framer-motion's
// vanilla `animate`/`inView`). We use the imperative API via Svelte actions
// rather than the declarative `<motion.div>` wrapper, because that wrapper
// ships uncompiled TS `.svelte` files that our Vite 8 / rolldown bundler can't
// process. The actions below give the same fade/rise/stagger entrances and
// build cleanly.
//
// SYNC-CHECK: keep params here in sync with any callers (see usages of
// `use:fadeRise`). Add new shared transitions to this module, not inline.

import type { Action } from 'svelte/action';
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
