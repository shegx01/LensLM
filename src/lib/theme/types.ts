// SYNC-CHECK: must match lens-core/src/config.rs AppConfig — update both together.
//
// TypeScript mirror of the Rust `AppConfig`, used so the theme read-modify-write
// preserves EVERY field across set_config (which replaces the whole struct).
// serde uses the field names verbatim (snake_case), so this shape must match
// exactly. M1-0 only mutates `theme`; all other fields are round-tripped untouched.

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

// SYNC-CHECK: must match lens-core/src/enrichment/embedding_text.rs CorefStrategy —
// the snake_case serde strings are the on-disk JSON + cache-key contract; update both.
//
// The coreference-resolution strategy applied while composing `embedding_text`.
// Only the two strategies that ship: `none` or `llm_inline`. (A `dedicated_model`
// stub was removed — it only fell back to `llm_inline`; the Rust side still reads
// a legacy `"dedicated_model"` string back as `llm_inline` for config round-trip.)
export type CorefStrategy = 'none' | 'llm_inline';

// SYNC-CHECK: must match lens-core/src/llm.rs LlmRouting — the serde shape is
// internally tagged on `kind` (snake_case); update both together.
//
// Typed routing policy for selecting the enrichment LLM. `cloud_first` (default)
// prefers a consented cloud provider then local; `local_first` is the inverse;
// `explicit` pins one exact (provider, model).
export type LlmRouting =
  | { kind: 'cloud_first' }
  | { kind: 'local_first' }
  | { kind: 'explicit'; provider: string; model: string };

// SYNC-CHECK: must match lens-core/src/config.rs TaskModel — flat {provider, model}.
//
// A per-task model pin (M4 Phase 3, Stage 3): one exact (provider, model) for a
// single enrichment task (coref / map / chat). Cloud pairs are catalog-validated
// on the Rust side; local Ollama is exempt (user-pulled models aren't in models.dev).
export interface TaskModel {
  provider: string;
  model: string;
}

// SYNC-CHECK: must match lens-core/src/config.rs EnrichmentConfig — update both together.
//
// Optional, additive background-enrichment config (M4 Phase 3). An older config
// written before Phase 3 has no `enrichment` key and reads back as the Rust
// `EnrichmentConfig::default` via `#[serde(default)]`.
export interface EnrichmentConfig {
  // Master toggle. When false, enrichment never runs (sources stay on raw vectors).
  enabled: boolean;
  // Coref strategy (snake_case to match the Rust serde mirror). Default 'llm_inline'.
  coref_strategy: CorefStrategy;
  // Explicit consent to send document text to a CLOUD LLM. Default false (local-first);
  // cloud enrichment never dispatches without it.
  cloud_consent: boolean;
  // Typed routing policy (Stage 2). Optional in the TS mirror: an older config has
  // no `routing` key and the Rust side defaults it to `{ kind: 'cloud_first' }` via
  // `#[serde(default)]`. When present, it round-trips verbatim.
  routing?: LlmRouting;
  // OPTIONAL per-task model overrides (Stage 3). `null`/absent ⇒ the task uses the
  // routing default; set ⇒ that task is pinned to the named (provider, model).
  // The Rust side `#[serde(default)]`s each to `None`, so omitting them is safe.
  coref_model?: TaskModel | null;
  map_model?: TaskModel | null;
  // chat_model is M5's concern (reserved for symmetry; no chat wiring in Phase 3).
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
  // Empty resolves to the default backend ("fastembed") at the resolver boundary.
  embedding_backend: string;
  // SYNC-CHECK: must match lens-core/src/config.rs AppConfig.js_render_enabled (default true).
  // On by default; user may opt out via the Settings > Ingestion toggle.
  js_render_enabled: boolean;
  // SYNC-CHECK: must match lens-core/src/config.rs AppConfig.reopen_last_notebook (default true).
  // On by default; user may opt out via Settings > General toggle.
  reopen_last_notebook: boolean;
}
