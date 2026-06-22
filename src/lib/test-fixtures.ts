// Shared AppConfig test fixtures.
//
// `baseAppConfig` is the minimal valid AppConfig (all required fields, empty
// collections) for tests that only care about a couple of fields.
// `fullAppConfig` adds a populated model/endpoint/voices set for tests that
// assert the WHOLE struct round-trips through a read-modify-write. Both accept
// `overrides` so a test can pin just the fields it asserts on.

import type { AppConfig } from '$lib/theme/types.js';

/** Minimal valid AppConfig: all required fields, empty collections. */
export function baseAppConfig(overrides?: Partial<AppConfig>): AppConfig {
  return {
    theme: 'dark',
    accent: 'purple',
    user_name: '',
    models: [],
    endpoints: {},
    voices: { host: '', guest: '' },
    tts: { provider: '', api_key: '' },
    paths: { data_dir: '' },
    tier_thresholds: { tier1_token_cap: 4000, tier2_token_cap: 16000 },
    onboarding_complete: false,
    embedding_model: '',
    ...overrides
  };
}

/** Fully-populated AppConfig (models/endpoints/voices/paths) for whole-struct
 *  round-trip assertions. */
export function fullAppConfig(overrides?: Partial<AppConfig>): AppConfig {
  return baseAppConfig({
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
    onboarding_complete: true,
    ...overrides
  });
}
