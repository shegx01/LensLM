import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { saveLlmProvider } from './llm-config.js';
import type { AppConfig } from '$lib/theme/types.js';

// A base AppConfig carrying an EXISTING ollama model entry so we can assert the
// upsert REPLACES it (rather than appending a duplicate).
function configWithOllama(): AppConfig {
  return {
    theme: 'dark',
    accent: 'purple',
    models: [
      {
        provider: 'ollama',
        base_url: 'http://old-host:11434',
        model: 'old-model',
        context: 4096,
        temperature: 0.5,
        api_key: 'stale'
      }
    ],
    endpoints: {},
    voices: { host: '', guest: '' },
    paths: { data_dir: '' },
    tier_thresholds: { tier1_token_cap: 4000, tier2_token_cap: 16000 },
    onboarding_complete: false,
    embedding_model: ''
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

describe('saveLlmProvider (upsert into models[])', () => {
  it('REPLACES an existing entry for the same provider rather than appending', async () => {
    const stored = configWithOllama();
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return stored;
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return undefined;
      }
    });

    await saveLlmProvider({
      provider: 'ollama',
      base_url: 'http://new-host:11434',
      model: 'new-model',
      api_key: '',
      context: 16384
    });

    expect(written).not.toBeNull();
    const w = written as unknown as AppConfig;
    // Still exactly ONE ollama entry — replaced in place, not appended.
    expect(w.models).toHaveLength(1);
    expect(w.models.filter((m) => m.provider === 'ollama')).toHaveLength(1);
    expect(w.models[0]).toMatchObject({
      provider: 'ollama',
      base_url: 'http://new-host:11434',
      model: 'new-model'
    });
    // The stale fields from the previous entry are gone.
    expect(w.models[0].base_url).not.toBe('http://old-host:11434');
    expect(w.models[0].model).not.toBe('old-model');
  });

  it('persists the supplied context window (no longer hardcoded to 8192)', async () => {
    const stored = configWithOllama();
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return stored;
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return undefined;
      }
    });

    await saveLlmProvider({
      provider: 'ollama',
      base_url: 'http://new-host:11434',
      model: 'new-model',
      api_key: '',
      context: 32768
    });

    const w = written as unknown as AppConfig;
    expect(w.models[0].context).toBe(32768);
  });
});
