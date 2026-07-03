import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { persistTheme, __flushNow, setPersistErrorHandler, PERSIST_DEBOUNCE_MS } from './index.js';
import type { AppConfig } from './types.js';
import { fullAppConfig } from '$lib/test-fixtures.js';

// A fully-populated AppConfig so tests can assert the WHOLE struct survives the
// read-modify-write (only `.theme` should ever change).
function fullConfig(theme: string): AppConfig {
  return fullAppConfig({ theme });
}

beforeEach(() => {
  // isTauri() reads globalThis.isTauri; mockIPC only wires __TAURI_INTERNALS__.
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  setPersistErrorHandler(null);
  vi.useRealTimers();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('persistTheme (debounced read-modify-write)', () => {
  it('writes the FULL config with only `theme` changed', async () => {
    const stored = fullConfig('light');
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return stored;
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return undefined;
      }
    });

    persistTheme('dark');
    await __flushNow();

    expect(written).not.toBeNull();
    const w = written as unknown as AppConfig;
    expect(w.theme).toBe('dark');
    expect(w.models).toEqual(stored.models);
    expect(w.endpoints).toEqual(stored.endpoints);
    expect(w.voices).toEqual(stored.voices);
    expect(w.paths).toEqual(stored.paths);
    expect(w.tier_thresholds).toEqual(stored.tier_thresholds);
    expect(w.onboarding_complete).toBe(true);
  });

  it('coalesces rapid toggles into one write and reads config at flush time', async () => {
    vi.useFakeTimers();
    let getCalls = 0;
    let setCalls = 0;
    let lastWritten: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') {
        getCalls++;
        return fullConfig('light');
      }
      if (cmd === 'set_config') {
        setCalls++;
        lastWritten = (args as { config: AppConfig }).config;
        return undefined;
      }
    });

    persistTheme('dark');
    persistTheme('light');
    persistTheme('system');

    // Nothing read or written yet — config read happens at flush, not at call.
    expect(getCalls).toBe(0);
    expect(setCalls).toBe(0);

    await vi.advanceTimersByTimeAsync(PERSIST_DEBOUNCE_MS + 10);

    expect(getCalls).toBe(1); // single coalesced read at flush
    expect(setCalls).toBe(1); // single coalesced write
    expect((lastWritten as unknown as AppConfig).theme).toBe('system'); // last value wins
  });

  it('surfaces persist failure without reverting live state', async () => {
    const errors: unknown[] = [];
    setPersistErrorHandler((e) => errors.push(e));
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    mockIPC((cmd) => {
      if (cmd === 'get_config') return fullConfig('light');
      if (cmd === 'set_config') throw new Error('disk full');
    });

    persistTheme('dark');
    await __flushNow();

    expect(errors).toHaveLength(1);
    expect((errors[0] as Error).message).toMatch(/disk full/);
    consoleSpy.mockRestore();
  });
});
