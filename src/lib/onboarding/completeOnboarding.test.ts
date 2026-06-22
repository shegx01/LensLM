import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { completeOnboarding, resetOnboarding } from './completeOnboarding.js';
import type { AppConfig } from '$lib/theme/types.js';

// A fully-populated AppConfig so we can assert the WHOLE struct survives the
// read-modify-write (only `onboarding_complete` should ever change).
function fullConfig(onboarding_complete: boolean): AppConfig {
  return {
    theme: 'dark',
    accent: 'purple',
    models: [
      {
        provider: 'ollama',
        base_url: 'http://localhost:11434/v1',
        model: 'llama3.2:3b',
        context: 8000,
        temperature: 0.7,
        api_key: 'secret-key'
      }
    ],
    endpoints: { local: 'http://localhost:11434' },
    voices: { host: 'host-voice', guest: 'guest-voice' },
    paths: { data_dir: '/Users/x/Library/Application Support/Lens' },
    tier_thresholds: { tier1_token_cap: 4000, tier2_token_cap: 16000 },
    onboarding_complete
  };
}

beforeEach(() => {
  // isTauri() reads globalThis.isTauri; mockIPC only wires __TAURI_INTERNALS__.
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('completeOnboarding (read-modify-write)', () => {
  it('writes the FULL config with only `onboarding_complete` flipped to true', async () => {
    const stored = fullConfig(false);
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return stored;
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return undefined;
      }
    });

    await completeOnboarding();

    expect(written).not.toBeNull();
    const w = written as unknown as AppConfig;
    expect(w.onboarding_complete).toBe(true);
    // Every other field is preserved verbatim (no theme/model clobber).
    expect(w.theme).toBe(stored.theme);
    expect(w.models).toEqual(stored.models);
    expect(w.endpoints).toEqual(stored.endpoints);
    expect(w.voices).toEqual(stored.voices);
    expect(w.paths).toEqual(stored.paths);
    expect(w.tier_thresholds).toEqual(stored.tier_thresholds);
  });

  it('is a no-op when not running under Tauri', async () => {
    delete (globalThis as { isTauri?: boolean }).isTauri;
    let setCalled = false;
    mockIPC((cmd) => {
      if (cmd === 'set_config') setCalled = true;
    });

    await completeOnboarding();

    expect(setCalled).toBe(false);
  });
});

describe('resetOnboarding (read-modify-write)', () => {
  it('writes the FULL config with only `onboarding_complete` flipped to false', async () => {
    const stored = fullConfig(true);
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return stored;
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return undefined;
      }
    });

    await resetOnboarding();

    expect(written).not.toBeNull();
    const w = written as unknown as AppConfig;
    expect(w.onboarding_complete).toBe(false);
    expect(w.theme).toBe(stored.theme);
    expect(w.models).toEqual(stored.models);
  });
});
