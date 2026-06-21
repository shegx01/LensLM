import '@testing-library/jest-dom/vitest';
import { randomFillSync } from 'node:crypto';
import { beforeAll } from 'vitest';

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
