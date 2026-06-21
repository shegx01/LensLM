import '@testing-library/jest-dom/vitest';
import { randomFillSync } from 'node:crypto';
import { beforeAll, beforeEach } from 'vitest';

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
