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

// SYNC-CHECK: must match lens-core/src/config.rs VoiceRef — `#[serde(untagged)]`, so
// `Named(String)` round-trips as a bare string and `Reference` as the object form.
export type VoiceRef = string | { clip_path: string; transcript: string };

export interface VoiceConfig {
  host: VoiceRef;
  guest: VoiceRef;
}

// SYNC-CHECK: must match lens-core/src/tts/mod.rs CloudTtsKind — snake_case serde.
export type CloudTtsKind = 'open_ai_compatible' | 'deepgram' | 'eleven_labs';

// SYNC-CHECK: must match lens-core/src/tts/mod.rs TtsBackend — externally tagged: unit
// variants round-trip as bare strings, `Cloud(CloudTtsKind)` as `{ cloud: CloudTtsKind }`.
export type TtsBackend = 'orpheus' | 'qwen3_local' | { cloud: CloudTtsKind };

// SYNC-CHECK: must match lens-core/src/config.rs CloudTtsConfig — update both together.
export interface CloudTtsConfig {
  kind: CloudTtsKind;
  api_key: string;
  base_url: string;
}

// SYNC-CHECK: must match lens-core/src/config.rs TtsConfig — update both together.
export interface TtsConfig {
  version: number;
  backend: TtsBackend;
  model: string;
  cloud: CloudTtsConfig | null;
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

// SYNC-CHECK: must match lens-core/src/config.rs CloudAsrProvider — snake_case serde.
export type CloudAsrProvider = 'open_ai_compatible' | 'deepgram';

// SYNC-CHECK: must match lens-core/src/config.rs AsrConfig — update both together.
// Older configs have no `asr` key; Rust defaults via `#[serde(default)]`.
export interface AsrConfig {
  backend: string;
  whisper_model: string;
  language?: string | { Other: string } | null;
  translate: boolean;
  cloud_provider?: CloudAsrProvider | null;
  cloud_base_url: string;
  cloud_model: string;
  cloud_api_key: string;
}

export interface PathConfig {
  data_dir: string;
}

// SYNC-CHECK: must match the `get_storage_stats` Tauri command's return shape
// (src-tauri/src/commands/system.rs) — not a persisted AppConfig field.
export interface StorageStats {
  corpus_bytes: number;
  reclaimable_cache_bytes: number;
  retained_bytes: number;
  total_bytes: number;
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
  // SYNC-CHECK: must match lens-core/src/config.rs AppConfig.tts (default backend Orpheus).
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
  // SYNC-CHECK: must match lens-core/src/config.rs AppConfig.max_source_mb (default "").
  max_source_mb: string;
  // SYNC-CHECK: must match lens-core/src/config.rs AppConfig.asr (default AsrConfig::default).
  asr: AsrConfig;
  // SYNC-CHECK: must match lens-core/src/config.rs AppConfig.audio_cloud_consent (default false).
  audio_cloud_consent: boolean;
  // SYNC-CHECK: must match lens-core/src/config.rs AppConfig.js_render_enabled (default true).
  js_render_enabled: boolean;
  // SYNC-CHECK: must match lens-core/src/config.rs AppConfig.reopen_last_notebook (default true).
  reopen_last_notebook: boolean;
  // SYNC-CHECK: must match lens-core/src/config.rs AppConfig.animations (default "system").
  // Applied as `data-motion` on <html>: 'system' | 'on' | 'off'.
  animations: string;
}
