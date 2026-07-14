// Shared AppConfig test fixtures.

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
    tts: { version: 1, backend: 'orpheus', model: '', cloud: null },
    enrichment: { enabled: false, coref_strategy: 'llm_inline', cloud_consent: false },
    paths: { data_dir: '' },
    tier_thresholds: { tier1_token_cap: 4000, tier2_token_cap: 16000 },
    onboarding_complete: false,
    embedding_model: '',
    embedding_backend: '',
    max_source_mb: '',
    asr: {
      backend: '',
      whisper_model: 'base',
      language: null,
      translate: false,
      cloud_provider: null,
      cloud_base_url: '',
      cloud_model: '',
      cloud_api_key: ''
    },
    audio_cloud_consent: false,
    js_render_enabled: true,
    reopen_last_notebook: true,
    ...overrides
  };
}

/** Fully-populated AppConfig for whole-struct round-trip assertions. */
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
