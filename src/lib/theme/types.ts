// SYNC-CHECK: must match lens-core/src/config.rs AppConfig — update both together.
// serde uses verbatim snake_case; set_config replaces the whole struct so all fields must round-trip.

export interface ModelConfig {
  provider: string;
  base_url: string;
  model: string;
  context: number;
  temperature: number;
  api_key: string;
}

export interface VoiceConfig {
  host: string;
  guest: string;
}

// SYNC-CHECK: must match lens-core/src/config.rs TtsConfig — update both together.
export interface TtsConfig {
  provider: string;
  api_key: string;
}

// SYNC-CHECK: must match lens-core/src/enrichment/embedding_text.rs CorefStrategy
// (snake_case serde strings; legacy "dedicated_model" reads back as "llm_inline" on the Rust side).
export type CorefStrategy = 'none' | 'llm_inline';

// SYNC-CHECK: must match lens-core/src/llm.rs LlmRouting — internally tagged on `kind` (snake_case).
// `cloud_first` prefers cloud-then-local; `local_first` is the inverse; `explicit` pins a (provider, model).
export type LlmRouting =
  | { kind: 'cloud_first' }
  | { kind: 'local_first' }
  | { kind: 'explicit'; provider: string; model: string };

// SYNC-CHECK: must match lens-core/src/config.rs TaskModel — flat {provider, model}.
// Cloud pairs are catalog-validated on the Rust side; local Ollama is exempt.
export interface TaskModel {
  provider: string;
  model: string;
}

// SYNC-CHECK: must match lens-core/src/config.rs EnrichmentConfig — update both together.
// Older configs have no `enrichment` key; Rust defaults via `#[serde(default)]`.
export interface EnrichmentConfig {
  enabled: boolean;
  coref_strategy: CorefStrategy;
  // Cloud enrichment never dispatches without explicit consent.
  cloud_consent: boolean;
  // Absent in older configs; Rust defaults to `{ kind: 'cloud_first' }`.
  routing?: LlmRouting;
  // `null`/absent ⇒ use routing default; set ⇒ pin this task. Rust defaults each to `None`.
  coref_model?: TaskModel | null;
  map_model?: TaskModel | null;
  chat_model?: TaskModel | null;
}

export interface PathConfig {
  data_dir: string;
}

export interface TierThresholds {
  tier1_token_cap: number;
  tier2_token_cap: number;
}

export interface AppConfig {
  theme: string;
  accent: string;
  // SYNC-CHECK: must match lens-core/src/config.rs AppConfig.user_name (default "").
  user_name: string;
  models: ModelConfig[];
  endpoints: Record<string, string>;
  voices: VoiceConfig;
  // SYNC-CHECK: must match lens-core/src/config.rs AppConfig.tts (default empty).
  tts: TtsConfig;
  // SYNC-CHECK: must match lens-core/src/config.rs AppConfig.enrichment (default disabled).
  enrichment: EnrichmentConfig;
  paths: PathConfig;
  tier_thresholds: TierThresholds;
  onboarding_complete: boolean;
  // SYNC-CHECK: must match lens-core/src/config.rs AppConfig.embedding_model (default "").
  embedding_model: string;
  // SYNC-CHECK: must match lens-core/src/config.rs AppConfig.embedding_backend (default "").
  embedding_backend: string;
  // SYNC-CHECK: must match lens-core/src/config.rs AppConfig.js_render_enabled (default true).
  js_render_enabled: boolean;
  // SYNC-CHECK: must match lens-core/src/config.rs AppConfig.reopen_last_notebook (default true).
  reopen_last_notebook: boolean;
}
