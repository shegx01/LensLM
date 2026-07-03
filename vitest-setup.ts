import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/svelte';
import { randomFillSync } from 'node:crypto';
import { afterEach, beforeAll, beforeEach } from 'vitest';

// ── Pending-timer tracking ────────────────────────────────────────────────────
//
// Root fix for the intermittent "ReferenceError: document is not defined" CI
// flake: bits-ui Dialog (focus-restore / scroll-lock cleanup) schedules a
// `setTimeout(fn, >0)` during component teardown. Under parallel Vitest with
// happy-dom, that callback can fire AFTER happy-dom has already torn down the
// *next* file's `document`, throwing at the process level and failing the run.
//
// Two layers, because one alone is incomplete:
//
//  1. Tracking + clearing: wrap globalThis.{setTimeout,setInterval} to register
//     every pending id in a Set; wrap clearTimeout/clearInterval to deregister;
//     auto-deregister a setTimeout id when its callback fires. In afterEach,
//     after cleanup() + the 0-delay drain, forcibly clear any remaining ids so
//     no *already-scheduled* callback can outlive the file boundary.
//
//  2. Callback guard: clearing can only cancel timers that exist when the hook
//     runs — it cannot touch a timer scheduled AFTER afterEach (a promise/
//     microtask continuation from the test, or the cross-file boundary itself).
//     That residual window is exactly when bits-ui's focus-restore fires into a
//     torn-down document. So every wrapped callback runs inside a guard that
//     swallows ONLY the benign post-teardown "document/window is not defined"
//     ReferenceError. It is provably safe to swallow: a real bug touching the
//     DOM while it still exists throws a different error (not "not defined"),
//     and the leaked callback is pure teardown (focus restore / scroll unlock)
//     with nothing left to act on. Any other error still propagates.
//
// The internal drain uses the REAL setTimeout (captured before wrapping) so it
// is never tracked and never self-cleared.
//
// vi.useFakeTimers() — tests that invoke this replace globalThis.setTimeout
// temporarily and restore it afterwards. Our wrapper is the baseline real impl
// and is NOT affected: if fake timers are active during afterEach our Set only
// holds real-timer ids from before/after the fake-timer window; clearing them
// is safe. Fake-timer ids managed by Vitest are outside our Set entirely.

const _realSetTimeout = globalThis.setTimeout;
const _realSetInterval = globalThis.setInterval;
const _realClearTimeout = globalThis.clearTimeout;
const _realClearInterval = globalThis.clearInterval;

const _pendingTimeouts = new Set<ReturnType<typeof _realSetTimeout>>();
const _pendingIntervals = new Set<ReturnType<typeof _realSetInterval>>();

// A timer callback that fires after happy-dom has torn down the document throws
// `ReferenceError: document is not defined` (or `window …`). That is benign —
// the callback is leftover teardown (bits-ui focus restore / scroll unlock) with
// nothing to act on. Swallow ONLY that; anything else must propagate.
const _isPostTeardownDomError = (err: unknown): boolean =>
  err instanceof ReferenceError && /\b(?:document|window)\b is not defined/.test(err.message);

globalThis.setTimeout = (<TArgs extends unknown[]>(
  handler: (...args: TArgs) => void,
  delay?: number,
  ...args: TArgs
): ReturnType<typeof _realSetTimeout> => {
  let id: ReturnType<typeof _realSetTimeout>;
  id = _realSetTimeout(
    (...a: TArgs) => {
      _pendingTimeouts.delete(id);
      try {
        handler(...a);
      } catch (err) {
        if (!_isPostTeardownDomError(err)) throw err;
      }
    },
    delay,
    ...args
  );
  _pendingTimeouts.add(id);
  return id;
}) as typeof globalThis.setTimeout;

globalThis.setInterval = (<TArgs extends unknown[]>(
  handler: (...args: TArgs) => void,
  delay?: number,
  ...args: TArgs
): ReturnType<typeof _realSetInterval> => {
  const id = _realSetInterval(
    (...a: TArgs) => {
      try {
        handler(...a);
      } catch (err) {
        if (!_isPostTeardownDomError(err)) throw err;
      }
    },
    delay,
    ...args
  );
  _pendingIntervals.add(id);
  return id;
}) as typeof globalThis.setInterval;

globalThis.clearTimeout = (id?: ReturnType<typeof _realSetTimeout>): void => {
  if (id !== undefined) _pendingTimeouts.delete(id);
  _realClearTimeout(id);
};

globalThis.clearInterval = (id?: ReturnType<typeof _realSetInterval>): void => {
  if (id !== undefined) _pendingIntervals.delete(id);
  _realClearInterval(id);
};

// ── afterEach: cleanup → 0-delay drain → clear all surviving timers ───────────
//
// Unmount any components rendered during the test, THEN drain pending macrotasks
// + microtasks. Unmounting fires each component's `onDestroy`, which clears any
// live timers (e.g. the embeddings install phase ticker, a 1200ms `setInterval`)
// so they can't fire after happy-dom's `document` is torn down. The subsequent
// drain flushes deferred 0-delay callbacks (component focus `setTimeout(…, 0)`,
// Svelte transition `onfinish` microtasks, bits-ui focus/scroll-lock restore,
// etc.) WHILE `document` still exists.
//
// Finally, forcibly clear every timer id still in the tracking Sets: any timer
// that survives both cleanup() and the drain is a leaked >0-delay callback (e.g.
// bits-ui Dialog focus-restore) that must not be allowed to fire after teardown.
// The drain uses _realSetTimeout so it is never entered into the tracking Sets.
afterEach(async () => {
  cleanup();
  await new Promise<void>((resolve) => _realSetTimeout(resolve, 0));

  // Clear all surviving tracked timeouts.
  for (const id of _pendingTimeouts) {
    _realClearTimeout(id);
  }
  _pendingTimeouts.clear();

  // Clear all surviving tracked intervals.
  for (const id of _pendingIntervals) {
    _realClearInterval(id);
  }
  _pendingIntervals.clear();
});

// happy-dom v20 exposes `localStorage` but its `getItem`/`setItem` are not
// callable in this context, which breaks mode-watcher — it captures the
// `localStorage` reference at *module-import* time (before any hook runs), so
// this polyfill must execute at setup-file top level, not inside beforeAll.
if (typeof window.localStorage?.getItem !== 'function') {
  const store = new Map<string, string>();
  const storage: Storage = {
    get length() {
      return store.size;
    },
    clear: () => store.clear(),
    getItem: (k: string) => (store.has(k) ? (store.get(k) as string) : null),
    key: (i: number) => Array.from(store.keys())[i] ?? null,
    removeItem: (k: string) => void store.delete(k),
    setItem: (k: string, v: string) => void store.set(String(k), String(v))
  };
  Object.defineProperty(window, 'localStorage', { configurable: true, value: storage });
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: storage });
}

// happy-dom doesn't implement the Web Animations API (`element.animate`), which
// both Svelte 5 transitions (e.g. `slide`) and the `motion` engine rely on.
// Stub it so animated components don't throw in tests — it completes instantly
// (resolves `finished` + fires `onfinish`) so intros/outros settle immediately.
if (typeof Element !== 'undefined' && typeof Element.prototype.animate !== 'function') {
  Element.prototype.animate = function animateStub(): Animation {
    const animation = {
      finished: Promise.resolve(),
      currentTime: 0,
      playState: 'finished',
      effect: null,
      onfinish: null as null | (() => void),
      play() {},
      pause() {},
      finish() {},
      cancel() {},
      reverse() {},
      commitStyles() {},
      persist() {},
      updatePlaybackRate() {},
      addEventListener() {},
      removeEventListener() {}
    };
    queueMicrotask(() => animation.onfinish?.());
    return animation as unknown as Animation;
  };
}

// happy-dom backs `requestAnimationFrame` with a timer, so a rAF scheduled during
// a test (CommandPalette focus, bits-ui Dialog focus/scroll-lock restore on close,
// etc.) can fire AFTER the DOM is torn down → "ReferenceError: document is not
// defined" surfaces as an unhandled error and fails the run. Run rAF callbacks
// synchronously (while the DOM still exists) to make teardown deterministic.
globalThis.requestAnimationFrame = ((cb: FrameRequestCallback): number => {
  cb(0);
  return 0;
}) as typeof globalThis.requestAnimationFrame;
globalThis.cancelAnimationFrame = (() => {}) as typeof globalThis.cancelAnimationFrame;

// `@tauri-apps/api/mocks`' mockIPC needs `window.crypto.getRandomValues`, which
// is not guaranteed in the simulated DOM. Polyfill it only if it's missing.
beforeAll(() => {
  if (!globalThis.window?.crypto?.getRandomValues) {
    Object.defineProperty(globalThis.window, 'crypto', {
      configurable: true,
      value: {
        getRandomValues: (buffer: Uint8Array) => {
          randomFillSync(buffer);
          return buffer;
        }
      }
    });
  }
});

// Reset persisted UI state between tests so theme tests don't leak.
beforeEach(() => {
  globalThis.localStorage?.clear();
});
