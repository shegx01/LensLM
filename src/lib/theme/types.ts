// TypeScript mirror of the Rust `AppConfig` (lens-core/src/config.rs), used so the
// theme read-modify-write preserves EVERY field across set_config (which replaces
// the whole struct). serde uses the field names verbatim (snake_case), so this
// shape must match exactly. M1-0 only mutates `theme`; all other fields are
// round-tripped untouched.

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

export interface PathConfig {
  data_dir: string;
}

export interface TierThresholds {
  tier1_token_cap: number;
  tier2_token_cap: number;
}

export interface AppConfig {
  theme: string;
  models: ModelConfig[];
  endpoints: Record<string, string>;
  voices: VoiceConfig;
  paths: PathConfig;
  tier_thresholds: TierThresholds;
  onboarding_complete: boolean;
}
