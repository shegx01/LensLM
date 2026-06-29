//! Application configuration, persisted as disk-only `config.json`.
//!
//! There is intentionally NO `app_config` table in SQLite (see plan §2): the
//! config shape is still evolving, and disk JSON avoids checksum-locking a guess
//! into a migration. If DB-resident flags are ever needed, add an `app_flags`
//! table as an additive migration in the milestone that requires it.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::LensError;

/// File name for the on-disk config, relative to the engine data directory.
const CONFIG_FILE_NAME: &str = "config.json";

/// Per-provider model endpoint configuration (LLM or embedding backend).
///
/// `Debug` is implemented MANUALLY (not derived) so the `api_key` is never
/// echoed verbatim into logs / panic messages: it is redacted to `"***"` when
/// present and shown as `""` when empty. See the `impl Debug` below.
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Provider identifier (e.g. `"ollama"`, `"openai"`).
    pub provider: String,
    /// Base URL of the provider API.
    pub base_url: String,
    /// Model name/id to request.
    pub model: String,
    /// Maximum context window in tokens.
    pub context: u32,
    /// Sampling temperature.
    pub temperature: f32,
    /// Optional API key (empty when not required, e.g. local providers).
    ///
    /// NOTE: stored in PLAINTEXT inside `config.json`. The file is written with
    /// restrictive `0o600` permissions on Unix (owner read/write only) as a
    /// stopgap. Migrating secrets into the OS keychain is deferred to M2; do not
    /// rely on this field being secure at rest.
    pub api_key: String,
}

impl std::fmt::Debug for ModelConfig {
    /// Redacts `api_key` so a secret can never leak through a `{:?}` format (logs,
    /// panic messages, tracing). A non-empty key prints as `"***"`; an empty key
    /// prints as `""`. All other fields are shown verbatim.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let api_key = if self.api_key.is_empty() { "" } else { "***" };
        f.debug_struct("ModelConfig")
            .field("provider", &self.provider)
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("context", &self.context)
            .field("temperature", &self.temperature)
            .field("api_key", &api_key)
            .finish()
    }
}

/// Host/guest voice identifiers used by the (future) TTS subsystem.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// Voice id for the "host" speaker.
    pub host: String,
    /// Voice id for the "guest" speaker.
    pub guest: String,
}

/// Cloud text-to-speech provider configuration (e.g. ElevenLabs).
///
/// Empty `provider`/`api_key` means no cloud TTS is configured (the default);
/// the system-check then falls back to the local Kokoro-on-disk gate.
///
/// `Debug` is implemented MANUALLY (not derived), exactly like [`ModelConfig`],
/// so the `api_key` is never echoed verbatim into logs / panic messages: it is
/// redacted to `"***"` when present and shown as `""` when empty.
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TtsConfig {
    /// Cloud TTS provider identifier (e.g. `"elevenlabs"`). Empty when unset.
    pub provider: String,
    /// Optional API key for the cloud provider (empty when not configured).
    ///
    /// NOTE: stored in PLAINTEXT inside `config.json` — see [`ModelConfig::api_key`]
    /// for the at-rest caveat. Migrating secrets into the OS keychain is deferred.
    pub api_key: String,
}

impl std::fmt::Debug for TtsConfig {
    /// Redacts `api_key` so a secret can never leak through a `{:?}` format (logs,
    /// panic messages, tracing). A non-empty key prints as `"***"`; an empty key
    /// prints as `""`. All other fields are shown verbatim.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let api_key = if self.api_key.is_empty() { "" } else { "***" };
        f.debug_struct("TtsConfig")
            .field("provider", &self.provider)
            .field("api_key", &api_key)
            .finish()
    }
}

/// A per-task model pin (M4 Phase 3, Stage 3).
///
/// Names one exact `(provider, model)` for a single enrichment task (coref / map /
/// chat). `provider` is a canonical id (`"anthropic"`, `"openai-compatible"`,
/// `"ollama"`, …) and `model` is a catalog model id. For CLOUD providers the pair
/// is validated against the [`crate::model_catalog::ModelCatalog`] before dispatch
/// (anti-free-string guard); local Ollama is exempt (user-pulled models aren't in
/// models.dev). Serde-stable snake_case so it round-trips in `config.json` and the
/// TS mirror without leaking a Rust shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskModel {
    /// Canonical provider id (matches `ModelConfig.provider`).
    pub provider: String,
    /// The model id to pin for this task.
    pub model: String,
}

/// Optional, additive background-enrichment configuration (M4 Phase 3).
///
/// Enrichment uses the configured LLM to improve RETRIEVAL after a source is
/// `Indexed` — it never mutates canonical text and never blocks the demo. This
/// section is OPTIONAL in `config.json`: an older config written before Phase 3
/// has no `enrichment` key and reads back as [`EnrichmentConfig::default`] via
/// the `#[serde(default)]` on the `AppConfig::enrichment` field (mirroring the
/// `tts`/[`TtsConfig`] backward-compat pattern).
///
/// `coref_strategy` is the typed [`CorefStrategy`] (single source of truth shared
/// with the enrichment worker — no stringly-typed config); it serializes to the
/// stable snake_case strings (`none`/`llm_inline`) so the on-disk JSON and the
/// TS mirror stay byte-compatible.
///
/// ## Per-task model overrides (M4 Phase 3, Stage 3)
/// `coref_model`/`map_model`/`chat_model` are OPTIONAL per-task pins ([`TaskModel`]).
/// When unset (`None`, the default) the task uses the routing default; when set, the
/// enrichment worker builds a cheap per-task provider pinned to that exact
/// `(provider, model)`. They are `#[serde(default)]` so an older `config.json`
/// written before Stage 3 (no per-task keys) reads back as `None` for each.
///
/// ## Provider-driven defaults
/// The onboarding LLM step picks defaults from the configured provider (see
/// [`EnrichmentConfig::for_provider`]): a reachable LOCAL provider (Ollama) →
/// `enabled=true`, `coref=LlmInline`; a CLOUD provider → `enabled=false` and
/// `cloud_consent` must be explicitly granted (document text leaving the machine
/// is a privacy/cost decision); NO provider → effectively `none`/disabled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnrichmentConfig {
    /// Master toggle. When `false`, enrichment never runs (sources stay on raw
    /// vectors — `none`/`pending`, the same graceful path as no provider).
    pub enabled: bool,
    /// OPTIONAL per-task model override for the COREF pass (M4 Phase 3, Stage 3).
    /// When `Some`, the enrichment worker pins the coref pass to this exact
    /// `(provider, model)` (catalog-validated for cloud providers; Ollama-local
    /// exempt); when `None` the coref pass uses the routing default. The product
    /// ask: a user may pick a coder model for coref while enrichment/chat default
    /// to a generalist. `#[serde(default)]` for backward-compat.
    #[serde(default)]
    pub coref_model: Option<TaskModel>,
    /// OPTIONAL per-task model override for the structural-MAP pass (Stage 3).
    /// Same semantics as [`coref_model`](Self::coref_model): `Some` pins the map
    /// pass to that model, `None` falls back to the routing default.
    #[serde(default)]
    pub map_model: Option<TaskModel>,
    /// OPTIONAL per-task model override for the CHAT task (M5's concern; added now
    /// only for symmetry). Defaulted `None`; NO chat wiring exists in Phase 3 —
    /// the field is reserved so the config shape is stable when M5 lands chat.
    #[serde(default)]
    pub chat_model: Option<TaskModel>,
    /// Coreference-resolution strategy applied while composing `embedding_text`.
    /// Typed (the canonical [`CorefStrategy`]); defaults to `LlmInline`. Read
    /// through [`deserialize_coref_strategy`] so a legacy `"dedicated_model"`
    /// string (the now-removed stub) round-trips to `LlmInline` instead of
    /// failing to deserialize.
    #[serde(deserialize_with = "deserialize_coref_strategy")]
    pub coref_strategy: CorefStrategy,
    /// Explicit consent to send document text to a CLOUD LLM. Defaults to `false`
    /// (local-first): cloud enrichment is gated on this and never dispatches
    /// without it (AC11). Local (Ollama) enrichment ignores this flag.
    pub cloud_consent: bool,
    /// Typed routing policy for selecting the enrichment LLM from `models[]`
    /// (M4 Phase 3, Stage 2). Defaults to [`LlmRouting::CloudFirst`] (prefer a
    /// configured + consented cloud provider, else local Ollama) per product
    /// direction. An absent field in an older `config.json` reads back as the
    /// default via `#[serde(default)]` (backward compatibility, mirroring the
    /// other additive enrichment fields).
    #[serde(default)]
    pub routing: LlmRouting,
}

/// Re-export of the typed routing policy so `config::LlmRouting` resolves at the
/// config layer without callers reaching into the `llm` module, and so there is
/// exactly ONE definition (it lives in [`crate::llm`] with the provider factory
/// that consumes it).
pub use crate::llm::LlmRouting;

/// Re-export of the canonical coref enum so `config::CorefStrategy` resolves at
/// the config layer without callers reaching into the enrichment module, and so
/// there is exactly ONE definition (de-duplicated — the enum lives in
/// [`crate::enrichment::embedding_text`] and is the worker's runtime strategy).
pub use crate::enrichment::CorefStrategy;

/// Tolerant deserializer for [`EnrichmentConfig::coref_strategy`]: parses the
/// snake_case string via [`CorefStrategy::from_config`], so the canonical
/// `"none"`/`"llm_inline"` values AND a legacy `"dedicated_model"` string (the
/// removed Phase-3 stub) all deserialize without error — the latter mapping to
/// `LlmInline`. Keeps an old `config.json` round-tripping instead of panicking.
fn deserialize_coref_strategy<'de, D>(deserializer: D) -> Result<CorefStrategy, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = String::deserialize(deserializer)?;
    Ok(CorefStrategy::from_config(&raw))
}

impl Default for EnrichmentConfig {
    /// The conservative default: enrichment OFF, `LlmInline` coref, cloud consent
    /// withheld. An older `config.json` with no `enrichment` key reads back as
    /// this via `#[serde(default)]`, and the demo degrades to raw vectors until
    /// the user opts in through onboarding.
    fn default() -> Self {
        Self {
            enabled: false,
            coref_model: None,
            map_model: None,
            chat_model: None,
            coref_strategy: CorefStrategy::default(),
            cloud_consent: false,
            routing: LlmRouting::default(),
        }
    }
}

impl EnrichmentConfig {
    /// Provider-driven defaults for the onboarding LLM step.
    ///
    /// * a LOCAL provider (Ollama) is configured → `enabled=true`,
    ///   `coref=LlmInline`, `cloud_consent` preserved/false (local never sends
    ///   text off-machine);
    /// * a CLOUD provider is configured → `enabled=false` (explicit-enable) and
    ///   `cloud_consent=false` — the onboarding step surfaces the consent note;
    /// * NO usable provider → disabled, `coref` unchanged (effectively `none`).
    ///
    /// This encodes the spec's "local Ollama → on + LlmInline; cloud → off +
    /// consent; no LLM → none/disabled" rule in one place so the engine and the
    /// onboarding UI agree.
    pub fn for_provider(has_local: bool, has_cloud: bool) -> Self {
        if has_local {
            Self {
                enabled: true,
                coref_strategy: CorefStrategy::LlmInline,
                cloud_consent: false,
                routing: LlmRouting::default(),
                ..Self::default()
            }
        } else if has_cloud {
            // Cloud configured but consent must be explicit; default OFF.
            Self {
                enabled: false,
                coref_strategy: CorefStrategy::LlmInline,
                cloud_consent: false,
                routing: LlmRouting::default(),
                ..Self::default()
            }
        } else {
            Self::default()
        }
    }
}

/// Filesystem paths the engine cares about.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PathConfig {
    /// Root data directory (where `lens.db` and `config.json` live).
    pub data_dir: String,
}

/// Token-budget thresholds controlling the tiered retrieval/synthesis pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TierThresholds {
    /// Upper token cap for tier-1 (cheap/fast) context assembly.
    pub tier1_token_cap: u32,
    /// Upper token cap for tier-2 (expanded) context assembly.
    pub tier2_token_cap: u32,
}

impl Default for TierThresholds {
    fn default() -> Self {
        // Conservative defaults; tuned per milestone as the pipeline lands.
        Self {
            tier1_token_cap: 4_000,
            tier2_token_cap: 16_000,
        }
    }
}

/// Default accent token name. Drives the `[data-accent]` token layer in the UI.
const DEFAULT_ACCENT: &str = "purple";

/// The serde default for [`AppConfig::accent`]: configs written before the
/// `accent` field existed (or with it omitted) deserialize to `"purple"` rather
/// than the empty string, so the persisted accent always resolves to a real
/// token name.
fn default_accent() -> String {
    DEFAULT_ACCENT.to_string()
}

/// Top-level application configuration.
///
/// Loaded from / saved to `{data_dir}/config.json`. A missing file yields
/// [`AppConfig::default`] (and is written back); a malformed file yields
/// [`LensError::Parse`] rather than panicking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    /// UI theme name (e.g. `"system"`, `"light"`, `"dark"`).
    pub theme: String,
    /// UI accent token name (drives the `[data-accent]` layer). Defaults to
    /// `"purple"`; an absent field in an older `config.json` reads back as
    /// `"purple"` via [`default_accent`].
    #[serde(default = "default_accent")]
    pub accent: String,
    /// User-supplied display name captured during onboarding ("Make it yours").
    /// Defaults to `""`; an absent field in an older `config.json` reads back as
    /// `""` via `#[serde(default)]` (forward compatibility, mirroring the
    /// `accent`/[`default_accent`] pattern above).
    #[serde(default)]
    pub user_name: String,
    /// Selected embedding model id (e.g. `"nomic-embed-text"`). Empty when the
    /// user has not yet chosen one. Defaults to `""`; an absent field in an older
    /// `config.json` reads back as `""` via `#[serde(default)]`.
    #[serde(default)]
    pub embedding_model: String,
    /// Selected embedding *backend* (`"fastembed"` | `"ollama"`). Empty when the
    /// user has not yet chosen one. Defaults to `""`; an absent field in an older
    /// `config.json` reads back as `""` via `#[serde(default)]`. An empty value
    /// resolves to the default backend (`fastembed`) via
    /// [`crate::embedder::EmbeddingBackend::from_opt_str`] — the SAME
    /// empty-string-resolves-to-default pattern as `embedding_model`. This is the
    /// app-wide default applied to NEW notebooks (M4 Phase 4b-B, R5/m2); it lives
    /// here, NOT in SQL.
    #[serde(default)]
    pub embedding_backend: String,
    /// Maximum accepted source size, in **megabytes**, for non-PDF formats
    /// (issue #71). Empty when the user has not configured it; an absent field in
    /// an older `config.json` reads back as `""` via `#[serde(default)]`. An empty
    /// (or unparseable / non-positive) value resolves to the 50 MB default via
    /// [`crate::ingest::resolve_max_source_bytes`] — the SAME
    /// empty-string-resolves-to-default pattern as `embedding_backend`. PDF sources
    /// are EXEMPT from this cap (they stream into a building table); only paste,
    /// raw-bytes, extracted-text, and URL enforcement sites read it. Stored as a
    /// `String` (not a number) to match the config's stringly-typed user-editable
    /// fields and keep an absent/empty value forward-compatible. Config-file-only;
    /// no settings UI (deferred).
    #[serde(default)]
    pub max_source_mb: String,
    /// Configured chat/inference models keyed by role.
    pub models: Vec<ModelConfig>,
    /// Arbitrary named endpoints (label -> URL).
    pub endpoints: BTreeMap<String, String>,
    /// Host/guest TTS voices.
    pub voices: VoiceConfig,
    /// Cloud TTS provider config (e.g. ElevenLabs). Defaults to empty; an absent
    /// field in an older `config.json` reads back as the default via
    /// `#[serde(default)]` (backward compatibility).
    #[serde(default)]
    pub tts: TtsConfig,
    /// Optional background-enrichment config (M4 Phase 3). Defaults to disabled;
    /// an absent field in an older `config.json` reads back as the default via
    /// `#[serde(default)]` (backward compatibility, mirroring `tts`).
    #[serde(default)]
    pub enrichment: EnrichmentConfig,
    /// Filesystem paths.
    pub paths: PathConfig,
    /// Tier token thresholds.
    pub tier_thresholds: TierThresholds,
    /// Whether first-run onboarding has completed.
    pub onboarding_complete: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            theme: String::default(),
            accent: default_accent(),
            user_name: String::default(),
            embedding_model: String::default(),
            embedding_backend: String::default(),
            max_source_mb: String::default(),
            models: Vec::default(),
            endpoints: BTreeMap::default(),
            voices: VoiceConfig::default(),
            tts: TtsConfig::default(),
            enrichment: EnrichmentConfig::default(),
            paths: PathConfig::default(),
            tier_thresholds: TierThresholds::default(),
            onboarding_complete: false,
        }
    }
}

impl AppConfig {
    /// Loads config from `{dir}/config.json`.
    ///
    /// * Missing file -> returns [`AppConfig::default`] and writes it to disk.
    /// * Malformed JSON -> returns [`LensError::Parse`].
    #[tracing::instrument(skip_all, fields(dir = %dir.as_ref().display()))]
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, LensError> {
        let path = dir.as_ref().join(CONFIG_FILE_NAME);
        match std::fs::read_to_string(&path) {
            Ok(contents) => {
                tracing::debug!("loading config from {}", path.display());
                let config = serde_json::from_str(&contents).map_err(|e| {
                    tracing::error!("malformed config at {}: {e}", path.display());
                    LensError::Parse(format!("{CONFIG_FILE_NAME}: {e}"))
                })?;
                Ok(config)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                tracing::debug!("no config at {}, writing default", path.display());
                let config = AppConfig::default();
                config.save(dir)?;
                Ok(config)
            }
            Err(e) => {
                tracing::error!("failed to read config at {}: {e}", path.display());
                Err(LensError::Io(format!(
                    "failed to read {CONFIG_FILE_NAME}: {e}"
                )))
            }
        }
    }

    /// Serializes and writes this config to `{dir}/config.json` (pretty JSON).
    ///
    /// On Unix the file is given `0o600` permissions (owner read/write only)
    /// because it may hold a plaintext `api_key`. This is a stopgap until M2
    /// moves secrets into the OS keychain.
    #[tracing::instrument(skip_all, fields(dir = %dir.as_ref().display()))]
    pub fn save(&self, dir: impl AsRef<Path>) -> Result<(), LensError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(|e| {
            tracing::error!("failed to create config dir {}: {e}", dir.display());
            LensError::Io(format!("failed to create config directory: {e}"))
        })?;
        let path = dir.join(CONFIG_FILE_NAME);
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json).map_err(|e| {
            tracing::error!("failed to write config at {}: {e}", path.display());
            LensError::Io(format!("failed to write {CONFIG_FILE_NAME}: {e}"))
        })?;
        Self::restrict_permissions(&path)?;
        tracing::debug!("saved config to {}", path.display());
        Ok(())
    }

    /// Restricts `config.json` to owner read/write (`0o600`) on Unix; a no-op on
    /// other platforms (Windows ACLs are not addressed in M0).
    #[cfg(unix)]
    fn restrict_permissions(path: &Path) -> Result<(), LensError> {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(path, perms).map_err(|e| {
            tracing::error!("failed to set permissions on {}: {e}", path.display());
            LensError::Io(format!("failed to secure {CONFIG_FILE_NAME}: {e}"))
        })?;
        Ok(())
    }

    /// Permission-tightening is Unix-only; elsewhere this is intentionally inert.
    #[cfg(not(unix))]
    fn restrict_permissions(_path: &Path) -> Result<(), LensError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_config_debug_redacts_api_key() {
        let cfg = ModelConfig {
            provider: "openai".to_string(),
            base_url: "https://api.openai.com".to_string(),
            model: "gpt-4".to_string(),
            context: 8192,
            temperature: 0.2,
            api_key: "sk-supersecret-token-value".to_string(),
        };
        let debug = format!("{cfg:?}");
        // The secret must never appear in the Debug output.
        assert!(!debug.contains("sk-supersecret-token-value"));
        assert!(!debug.contains("supersecret"));
        // Redaction marker present; other fields still visible.
        assert!(debug.contains("***"));
        assert!(debug.contains("openai"));
        assert!(debug.contains("gpt-4"));
    }

    #[test]
    fn model_config_debug_shows_empty_api_key_as_empty() {
        let cfg = ModelConfig {
            provider: "ollama".to_string(),
            ..ModelConfig::default()
        };
        let debug = format!("{cfg:?}");
        // An empty key is shown as "" (not redacted to ***).
        assert!(debug.contains("api_key: \"\""));
        assert!(!debug.contains("***"));
    }

    #[test]
    fn default_accent_is_purple() {
        assert_eq!(AppConfig::default().accent, "purple");
    }

    #[test]
    fn missing_accent_deserializes_to_purple() {
        // A config.json written before the `accent` field existed has no
        // `accent` key; it must read back as the default rather than failing.
        let json = r#"{
            "theme": "dark",
            "models": [],
            "endpoints": {},
            "voices": { "host": "", "guest": "" },
            "paths": { "data_dir": "" },
            "tier_thresholds": { "tier1_token_cap": 4000, "tier2_token_cap": 16000 },
            "onboarding_complete": true
        }"#;
        let config: AppConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.accent, "purple");
        assert_eq!(config.theme, "dark");
    }

    #[test]
    fn default_user_name_is_empty() {
        assert_eq!(AppConfig::default().user_name, "");
    }

    #[test]
    fn missing_user_name_deserializes_to_empty() {
        // A config.json written before the `user_name` field existed has no
        // `user_name` key; it must read back as the empty string (the serde
        // default) rather than failing to deserialize (forward compatibility).
        let json = r#"{
            "theme": "dark",
            "accent": "purple",
            "embedding_model": "",
            "models": [],
            "endpoints": {},
            "voices": { "host": "", "guest": "" },
            "paths": { "data_dir": "" },
            "tier_thresholds": { "tier1_token_cap": 4000, "tier2_token_cap": 16000 },
            "onboarding_complete": true
        }"#;
        let config: AppConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.user_name, "");
    }

    #[test]
    fn explicit_user_name_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let config = AppConfig {
            user_name: "Jamie".to_string(),
            ..AppConfig::default()
        };
        config.save(dir.path()).unwrap();
        let loaded = AppConfig::load(dir.path()).unwrap();
        assert_eq!(loaded.user_name, "Jamie");
    }

    #[test]
    fn default_embedding_model_is_empty() {
        assert_eq!(AppConfig::default().embedding_model, "");
    }

    #[test]
    fn missing_embedding_model_deserializes_to_empty() {
        // A config.json written before the `embedding_model` field existed has no
        // `embedding_model` key; it must read back as the empty string (the serde
        // default) rather than failing to deserialize (backward compatibility).
        let json = r#"{
            "theme": "dark",
            "accent": "purple",
            "models": [],
            "endpoints": {},
            "voices": { "host": "", "guest": "" },
            "paths": { "data_dir": "" },
            "tier_thresholds": { "tier1_token_cap": 4000, "tier2_token_cap": 16000 },
            "onboarding_complete": true
        }"#;
        let config: AppConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.embedding_model, "");
    }

    #[test]
    fn explicit_embedding_model_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let config = AppConfig {
            embedding_model: "nomic-embed-text".to_string(),
            ..AppConfig::default()
        };
        config.save(dir.path()).unwrap();
        let loaded = AppConfig::load(dir.path()).unwrap();
        assert_eq!(loaded.embedding_model, "nomic-embed-text");
    }

    #[test]
    fn default_embedding_backend_is_empty() {
        assert_eq!(AppConfig::default().embedding_backend, "");
    }

    #[test]
    fn missing_embedding_backend_deserializes_to_empty() {
        // A config.json written before the `embedding_backend` field existed has
        // no `embedding_backend` key; it must read back as the empty string (the
        // serde default) rather than failing to deserialize (backward
        // compatibility — the SAME pattern as `embedding_model`). An empty value
        // resolves to the default backend (`fastembed`) at the resolver boundary.
        let json = r#"{
            "theme": "dark",
            "accent": "purple",
            "embedding_model": "",
            "models": [],
            "endpoints": {},
            "voices": { "host": "", "guest": "" },
            "paths": { "data_dir": "" },
            "tier_thresholds": { "tier1_token_cap": 4000, "tier2_token_cap": 16000 },
            "onboarding_complete": true
        }"#;
        let config: AppConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.embedding_backend, "");
    }

    #[test]
    fn explicit_embedding_backend_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let config = AppConfig {
            embedding_backend: "ollama".to_string(),
            ..AppConfig::default()
        };
        config.save(dir.path()).unwrap();
        let loaded = AppConfig::load(dir.path()).unwrap();
        assert_eq!(loaded.embedding_backend, "ollama");
    }

    // ── max_source_mb (issue #71: configurable ingest cap) ────────────────

    #[test]
    fn test_max_source_mb_default() {
        // Empty by default — the resolver (`ingest::resolve_max_source_bytes`)
        // maps an empty value to the 50 MB default, mirroring the
        // empty-string-resolves-to-default pattern of `embedding_backend`.
        assert_eq!(AppConfig::default().max_source_mb, "");
    }

    #[test]
    fn test_max_source_mb_missing_deserializes_to_empty() {
        // A config.json written before the `max_source_mb` field existed has no
        // `max_source_mb` key; it must read back as the empty string (the serde
        // default) rather than failing to deserialize (backward compatibility —
        // the SAME pattern as `embedding_backend`).
        let json = r#"{
            "theme": "dark",
            "accent": "purple",
            "embedding_model": "",
            "embedding_backend": "",
            "models": [],
            "endpoints": {},
            "voices": { "host": "", "guest": "" },
            "paths": { "data_dir": "" },
            "tier_thresholds": { "tier1_token_cap": 4000, "tier2_token_cap": 16000 },
            "onboarding_complete": true
        }"#;
        let config: AppConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.max_source_mb, "");
    }

    #[test]
    fn test_max_source_mb_explicit_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let config = AppConfig {
            max_source_mb: "100".to_string(),
            ..AppConfig::default()
        };
        config.save(dir.path()).unwrap();
        let loaded = AppConfig::load(dir.path()).unwrap();
        assert_eq!(loaded.max_source_mb, "100");
    }

    #[test]
    fn tts_config_debug_redacts_api_key() {
        let cfg = TtsConfig {
            provider: "elevenlabs".to_string(),
            api_key: "sk-elevenlabs-supersecret".to_string(),
        };
        let debug = format!("{cfg:?}");
        // The secret must never appear in the Debug output.
        assert!(!debug.contains("sk-elevenlabs-supersecret"));
        assert!(!debug.contains("supersecret"));
        // Redaction marker present; provider still visible.
        assert!(debug.contains("***"));
        assert!(debug.contains("elevenlabs"));
    }

    #[test]
    fn tts_config_debug_shows_empty_api_key_as_empty() {
        let cfg = TtsConfig::default();
        let debug = format!("{cfg:?}");
        // An empty key is shown as "" (not redacted to ***).
        assert!(debug.contains("api_key: \"\""));
        assert!(!debug.contains("***"));
    }

    #[test]
    fn default_tts_is_empty() {
        assert_eq!(AppConfig::default().tts, TtsConfig::default());
        assert_eq!(AppConfig::default().tts.provider, "");
        assert_eq!(AppConfig::default().tts.api_key, "");
    }

    #[test]
    fn missing_tts_deserializes_to_default() {
        // A config.json written before the `tts` field existed has no `tts` key;
        // it must read back as the default (empty) rather than failing to
        // deserialize (backward compatibility).
        let json = r#"{
            "theme": "dark",
            "accent": "purple",
            "embedding_model": "",
            "models": [],
            "endpoints": {},
            "voices": { "host": "", "guest": "" },
            "paths": { "data_dir": "" },
            "tier_thresholds": { "tier1_token_cap": 4000, "tier2_token_cap": 16000 },
            "onboarding_complete": true
        }"#;
        let config: AppConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.tts, TtsConfig::default());
    }

    #[test]
    fn explicit_tts_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let config = AppConfig {
            tts: TtsConfig {
                provider: "elevenlabs".to_string(),
                api_key: "sk-elevenlabs".to_string(),
            },
            ..AppConfig::default()
        };
        config.save(dir.path()).unwrap();
        let loaded = AppConfig::load(dir.path()).unwrap();
        assert_eq!(loaded.tts.provider, "elevenlabs");
        assert_eq!(loaded.tts.api_key, "sk-elevenlabs");
    }

    #[test]
    fn explicit_accent_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let config = AppConfig {
            accent: "emerald".to_string(),
            ..AppConfig::default()
        };
        config.save(dir.path()).unwrap();
        let loaded = AppConfig::load(dir.path()).unwrap();
        assert_eq!(loaded.accent, "emerald");
    }

    // ── EnrichmentConfig (M4 Phase 3, AC14) ────────────────────────────────

    #[test]
    fn default_enrichment_is_disabled_llm_inline_no_consent() {
        let e = AppConfig::default().enrichment;
        assert!(!e.enabled, "enrichment defaults OFF (raw-vector floor)");
        assert_eq!(e.coref_strategy, CorefStrategy::LlmInline);
        assert!(!e.cloud_consent, "cloud consent defaults withheld");
        assert_eq!(
            e.routing,
            LlmRouting::CloudFirst,
            "routing defaults to cloud-first per product direction"
        );
    }

    #[test]
    fn missing_routing_deserializes_to_cloud_first() {
        // An enrichment section written before the Stage-2 `routing` field existed
        // has no `routing` key; it must read back as the default (cloud-first)
        // rather than failing to deserialize (backward compatibility).
        let json = r#"{
            "enabled": true,
            "coref_strategy": "llm_inline",
            "cloud_consent": false
        }"#;
        let e: EnrichmentConfig = serde_json::from_str(json).unwrap();
        assert_eq!(e.routing, LlmRouting::CloudFirst);
    }

    #[test]
    fn coref_strategy_serializes_to_stable_snake_case() {
        // The on-disk JSON + the TS mirror both depend on these exact strings.
        assert_eq!(
            serde_json::to_string(&CorefStrategy::None).unwrap(),
            "\"none\""
        );
        assert_eq!(
            serde_json::to_string(&CorefStrategy::LlmInline).unwrap(),
            "\"llm_inline\""
        );
        // …and deserialize back (only `none`/`llm_inline` are emitted now).
        let s: CorefStrategy = serde_json::from_str("\"llm_inline\"").unwrap();
        assert_eq!(s, CorefStrategy::LlmInline);
    }

    #[test]
    fn legacy_dedicated_model_coref_deserializes_to_llm_inline() {
        // An older build shipped a `dedicated_model` coref stub that has since been
        // removed (no stub ships). A `config.json` written by that build must still
        // load: the legacy string round-trips to `LlmInline` rather than failing to
        // deserialize the enrichment section.
        let json = r#"{
            "enabled": true,
            "coref_strategy": "dedicated_model",
            "cloud_consent": false
        }"#;
        let e: EnrichmentConfig = serde_json::from_str(json).unwrap();
        assert_eq!(e.coref_strategy, CorefStrategy::LlmInline);
        assert!(e.enabled);
    }

    #[test]
    fn missing_enrichment_deserializes_to_default() {
        // A config.json written before the `enrichment` field existed has no
        // `enrichment` key; it must read back as the default (disabled) rather
        // than failing to deserialize (backward compatibility — the AC14 core
        // round-trip). This mirrors `missing_tts_deserializes_to_default`.
        let json = r#"{
            "theme": "dark",
            "accent": "purple",
            "user_name": "",
            "embedding_model": "",
            "models": [],
            "endpoints": {},
            "voices": { "host": "", "guest": "" },
            "tts": { "provider": "", "api_key": "" },
            "paths": { "data_dir": "" },
            "tier_thresholds": { "tier1_token_cap": 4000, "tier2_token_cap": 16000 },
            "onboarding_complete": true
        }"#;
        let config: AppConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.enrichment, EnrichmentConfig::default());
        assert!(!config.enrichment.enabled);
        assert_eq!(config.enrichment.coref_strategy, CorefStrategy::LlmInline);
    }

    #[test]
    fn explicit_enrichment_round_trips() {
        // serialize → deserialize is stable, AND the snake_case coref string is
        // what lands on disk (so an existing config.json keeps working).
        let dir = tempfile::tempdir().unwrap();
        let config = AppConfig {
            enrichment: EnrichmentConfig {
                enabled: true,
                coref_strategy: CorefStrategy::None,
                cloud_consent: true,
                routing: LlmRouting::LocalFirst,
                ..EnrichmentConfig::default()
            },
            ..AppConfig::default()
        };
        config.save(dir.path()).unwrap();
        let loaded = AppConfig::load(dir.path()).unwrap();
        assert!(loaded.enrichment.enabled);
        assert_eq!(loaded.enrichment.coref_strategy, CorefStrategy::None);
        assert!(loaded.enrichment.cloud_consent);
        // The typed routing policy round-trips intact.
        assert_eq!(loaded.enrichment.routing, LlmRouting::LocalFirst);

        // The persisted JSON carries the snake_case string, not a struct variant.
        let on_disk = std::fs::read_to_string(dir.path().join(CONFIG_FILE_NAME)).unwrap();
        assert!(on_disk.contains("\"coref_strategy\": \"none\""));
    }

    #[test]
    fn provider_driven_defaults_match_the_locked_rule() {
        // local Ollama → enabled + LlmInline; cloud → disabled + (no consent yet);
        // no LLM → disabled / default.
        let local = EnrichmentConfig::for_provider(true, false);
        assert!(local.enabled);
        assert_eq!(local.coref_strategy, CorefStrategy::LlmInline);
        assert!(!local.cloud_consent);

        let cloud = EnrichmentConfig::for_provider(false, true);
        assert!(!cloud.enabled, "cloud is explicit-enable (off by default)");
        assert!(!cloud.cloud_consent, "consent withheld until explicit");

        let none = EnrichmentConfig::for_provider(false, false);
        assert_eq!(none, EnrichmentConfig::default());
        assert!(!none.enabled);

        // local takes precedence when both are present (local-first).
        let both = EnrichmentConfig::for_provider(true, true);
        assert!(both.enabled);
        assert_eq!(both.coref_strategy, CorefStrategy::LlmInline);
    }

    // ── Per-task model overrides (M4 Phase 3, Stage 3) ─────────────────────

    #[test]
    fn default_per_task_models_are_none() {
        // The conservative default: no per-task pins ⇒ every task uses the routing
        // default. This is what an onboarding "use the configured model" choice maps to.
        let e = EnrichmentConfig::default();
        assert_eq!(e.coref_model, None);
        assert_eq!(e.map_model, None);
        assert_eq!(e.chat_model, None);
    }

    #[test]
    fn missing_per_task_models_deserialize_to_none() {
        // An enrichment section written before Stage 3 (no `coref_model`/`map_model`/
        // `chat_model` keys) must read back as `None` for each rather than failing
        // to deserialize (backward compatibility — the Stage-3 additive shape).
        let json = r#"{
            "enabled": true,
            "coref_strategy": "llm_inline",
            "cloud_consent": false,
            "routing": { "kind": "cloud_first" }
        }"#;
        let e: EnrichmentConfig = serde_json::from_str(json).unwrap();
        assert_eq!(e.coref_model, None);
        assert_eq!(e.map_model, None);
        assert_eq!(e.chat_model, None);
    }

    #[test]
    fn explicit_per_task_models_round_trip() {
        // serialize → deserialize preserves the per-task pins exactly.
        let dir = tempfile::tempdir().unwrap();
        let config = AppConfig {
            enrichment: EnrichmentConfig {
                enabled: true,
                coref_model: Some(TaskModel {
                    provider: "ollama".to_string(),
                    model: "qwen2.5-coder".to_string(),
                }),
                map_model: Some(TaskModel {
                    provider: "anthropic".to_string(),
                    model: "claude-sonnet-4-5".to_string(),
                }),
                ..EnrichmentConfig::default()
            },
            ..AppConfig::default()
        };
        config.save(dir.path()).unwrap();
        let loaded = AppConfig::load(dir.path()).unwrap();
        assert_eq!(
            loaded.enrichment.coref_model,
            Some(TaskModel {
                provider: "ollama".to_string(),
                model: "qwen2.5-coder".to_string(),
            })
        );
        assert_eq!(
            loaded.enrichment.map_model,
            Some(TaskModel {
                provider: "anthropic".to_string(),
                model: "claude-sonnet-4-5".to_string(),
            })
        );
        // chat_model stays None (M5's concern; not set here).
        assert_eq!(loaded.enrichment.chat_model, None);
    }

    #[test]
    fn task_model_serializes_to_flat_snake_case() {
        // The on-disk JSON + TS mirror depend on the flat `{provider, model}` shape.
        let tm = TaskModel {
            provider: "anthropic".to_string(),
            model: "claude-sonnet-4-5".to_string(),
        };
        assert_eq!(
            serde_json::to_value(&tm).unwrap(),
            serde_json::json!({ "provider": "anthropic", "model": "claude-sonnet-4-5" })
        );
    }

    #[test]
    fn per_task_model_validates_against_catalog() {
        // A cloud per-task model must be a real catalog entry (anti-free-string).
        let catalog = crate::model_catalog::ModelCatalog::bundled();
        let coref = TaskModel {
            provider: "anthropic".to_string(),
            model: "claude-sonnet-4-5".to_string(),
        };
        assert!(catalog.validate(&coref.provider, &coref.model).is_ok());
        // A made-up cloud model is rejected by the same guard.
        assert!(catalog.validate("anthropic", "totally-made-up").is_err());
        // Local Ollama is exempt (user-pulled): any non-empty id is valid.
        assert!(catalog.is_valid("ollama", "qwen2.5-coder"));
    }
}
