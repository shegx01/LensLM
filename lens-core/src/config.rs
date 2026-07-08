//! Application configuration, persisted as disk-only `config.json`.
//! There is intentionally NO `app_config` table in SQLite (see plan §2): the
//! config shape is still evolving, and disk JSON avoids checksum-locking a guess
//! into a migration. If DB-resident flags are ever needed, add an `app_flags`
//! table as an additive migration in the milestone that requires it.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::LensError;
use crate::asr::Lang;

/// File name for the on-disk config, relative to the engine data directory.
const CONFIG_FILE_NAME: &str = "config.json";

/// Per-provider model endpoint configuration (LLM or embedding backend).
/// `Debug` is manual so `api_key` is redacted in logs (`"***"` or `""`).
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub provider: String,
    pub base_url: String,
    pub model: String,
    pub context: u32,
    pub temperature: f32,
    /// Empty for local providers. Stored in PLAINTEXT in `config.json` (written
    /// with `0o600` on Unix); keychain migration is deferred to M2.
    pub api_key: String,
}

impl std::fmt::Debug for ModelConfig {
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

/// Host/guest voice identifiers used by the TTS subsystem.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VoiceConfig {
    pub host: String,
    pub guest: String,
}

/// Cloud TTS provider configuration (e.g. ElevenLabs). Empty means no cloud TTS.
/// `Debug` is manual so `api_key` is redacted, exactly like [`ModelConfig`].
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TtsConfig {
    pub provider: String,
    /// Stored in PLAINTEXT — see [`ModelConfig::api_key`] for the at-rest caveat.
    pub api_key: String,
}

impl std::fmt::Debug for TtsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let api_key = if self.api_key.is_empty() { "" } else { "***" };
        f.debug_struct("TtsConfig")
            .field("provider", &self.provider)
            .field("api_key", &api_key)
            .finish()
    }
}

/// Default Whisper model id for a fresh [`AsrConfig`] — the multilingual base
/// model (mirrors the registry default; #42).
fn default_whisper_model() -> String {
    "base".to_string()
}

/// Cloud ASR provider wire format (#45). Strong-typed, not a magic string:
/// `OpenAiCompatible` covers OpenAI/Groq/self-hosted (WAV multipart), `Deepgram`
/// is the raw-PCM `linear32` path. Serialized snake_case in `config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CloudAsrProvider {
    OpenAiCompatible,
    Deepgram,
}

/// Optional additive ASR configuration (#42). Absent from older `config.json`
/// files reads back as [`AsrConfig::default`]. `backend` is a `String` — not the
/// [`AsrBackend`](crate::asr::AsrBackend) enum — intentionally mirroring
/// `AppConfig::embedding_backend`: an empty string means router-decided, and a
/// non-empty token (`"apple_native"` | `"local_whisper"`) is resolved at the
/// engine boundary via `AsrBackend::from_opt_str`.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct AsrConfig {
    /// `""` → router-decided; `"apple_native"` | `"local_whisper"` | `"cloud"` → explicit override.
    #[serde(default)]
    pub backend: String,
    /// Whisper model id (`"tiny"` | `"base"` | `"small"`); defaults to `"base"`.
    #[serde(default = "default_whisper_model")]
    pub whisper_model: String,
    /// Forced source language; `None` ⇒ auto-detect.
    #[serde(default)]
    pub language: Option<Lang>,
    /// `true` ⇒ translate to English (the Whisper translate task).
    #[serde(default)]
    pub translate: bool,
    /// Cloud ASR provider (#45); `None` ⇒ not configured. Gated by
    /// `AppConfig::audio_cloud_consent` at the engine boundary.
    #[serde(default)]
    pub cloud_provider: Option<CloudAsrProvider>,
    /// Cloud ASR endpoint base URL (e.g. `https://api.openai.com`).
    #[serde(default)]
    pub cloud_base_url: String,
    /// Cloud ASR model id (e.g. `whisper-1`, `nova-3`).
    #[serde(default)]
    pub cloud_model: String,
    /// Cloud ASR API key. Stored in PLAINTEXT — see [`ModelConfig::api_key`].
    #[serde(default)]
    pub cloud_api_key: String,
}

impl std::fmt::Debug for AsrConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cloud_api_key = if self.cloud_api_key.is_empty() {
            ""
        } else {
            "***"
        };
        f.debug_struct("AsrConfig")
            .field("backend", &self.backend)
            .field("whisper_model", &self.whisper_model)
            .field("language", &self.language)
            .field("translate", &self.translate)
            .field("cloud_provider", &self.cloud_provider)
            .field("cloud_base_url", &self.cloud_base_url)
            .field("cloud_model", &self.cloud_model)
            .field("cloud_api_key", &cloud_api_key)
            .finish()
    }
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            backend: String::default(),
            whisper_model: default_whisper_model(),
            language: None,
            translate: false,
            cloud_provider: None,
            cloud_base_url: String::default(),
            cloud_model: String::default(),
            cloud_api_key: String::default(),
        }
    }
}

/// Per-task model pin (M4 Phase 3, Stage 3): one exact `(provider, model)` for
/// a single enrichment task. Cloud pairs are validated against the catalog;
/// Ollama is exempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskModel {
    pub provider: String,
    pub model: String,
}

/// Optional additive background-enrichment configuration (M4 Phase 3). Absent
/// from older `config.json` files reads back as [`EnrichmentConfig::default`].
/// Per-task model pins (`coref_model`/`map_model`/`chat_model`) default to `None`
/// (routing default) and are backward-compatible via `#[serde(default)]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnrichmentConfig {
    pub enabled: bool,
    /// Pins the coref pass to a specific `(provider, model)`; `None` uses the
    /// routing default. `#[serde(default)]` for backward-compat.
    #[serde(default)]
    pub coref_model: Option<TaskModel>,
    /// Pins the structural-map pass; `None` uses the routing default.
    #[serde(default)]
    pub map_model: Option<TaskModel>,
    /// Reserved for M5 chat; no wiring exists in Phase 3. `#[serde(default)]`.
    #[serde(default)]
    pub chat_model: Option<TaskModel>,
    /// Coref strategy. Legacy `"dedicated_model"` round-trips to `LlmInline`
    /// via [`deserialize_coref_strategy`] so old configs don't fail.
    #[serde(deserialize_with = "deserialize_coref_strategy")]
    pub coref_strategy: CorefStrategy,
    /// Opt-in LLM relation-extraction strategy (#154). `#[serde(default)]` +
    /// tolerant deserializer so old configs read back as `Off`.
    #[serde(default, deserialize_with = "deserialize_relations_strategy")]
    pub relations_strategy: RelationsStrategy,
    /// Explicit consent to send document text to a cloud LLM (AC11). Ollama
    /// enrichment ignores this flag.
    pub cloud_consent: bool,
    /// Routing policy for selecting the enrichment LLM. `#[serde(default)]`.
    #[serde(default)]
    pub routing: LlmRouting,
}

/// Re-export so `config::LlmRouting` resolves without callers reaching into `llm`.
pub use crate::llm::LlmRouting;

/// Re-export so `config::CorefStrategy` resolves without callers reaching into
/// the enrichment module. Single definition lives in `crate::enrichment`.
pub use crate::enrichment::CorefStrategy;

/// Re-export so `config::RelationsStrategy` resolves without reaching into the
/// enrichment module. Single definition lives in `crate::enrichment`.
pub use crate::enrichment::RelationsStrategy;

/// Tolerant deserializer for [`EnrichmentConfig::coref_strategy`]. Legacy
/// `"dedicated_model"` maps to `LlmInline` so old `config.json` files survive.
fn deserialize_coref_strategy<'de, D>(deserializer: D) -> Result<CorefStrategy, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = String::deserialize(deserializer)?;
    Ok(CorefStrategy::from_config(&raw))
}

/// Tolerant deserializer for [`EnrichmentConfig::relations_strategy`]. Unknown
/// values map to `Off` so old/garbled `config.json` files survive.
fn deserialize_relations_strategy<'de, D>(deserializer: D) -> Result<RelationsStrategy, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = String::deserialize(deserializer)?;
    Ok(RelationsStrategy::from_config(&raw))
}

impl Default for EnrichmentConfig {
    /// Conservative default: enrichment OFF, cloud consent withheld. An older
    /// `config.json` with no `enrichment` key reads back as this via `#[serde(default)]`.
    fn default() -> Self {
        Self {
            enabled: false,
            coref_model: None,
            map_model: None,
            chat_model: None,
            coref_strategy: CorefStrategy::default(),
            relations_strategy: RelationsStrategy::default(),
            cloud_consent: false,
            routing: LlmRouting::default(),
        }
    }
}

impl EnrichmentConfig {
    /// Provider-driven defaults: local (Ollama) → `enabled=true` + `LlmInline`;
    /// cloud → `enabled=false` (requires explicit consent); no provider → default.
    pub fn for_provider(has_local: bool, has_cloud: bool) -> Self {
        if has_local {
            Self {
                enabled: true,
                coref_strategy: CorefStrategy::LlmInline,
                relations_strategy: RelationsStrategy::default(),
                cloud_consent: false,
                routing: LlmRouting::default(),
                ..Self::default()
            }
        } else if has_cloud {
            Self {
                enabled: false,
                coref_strategy: CorefStrategy::LlmInline,
                relations_strategy: RelationsStrategy::default(),
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
    pub data_dir: String,
}

/// Token-budget thresholds for the tiered retrieval/synthesis pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TierThresholds {
    pub tier1_token_cap: u32,
    pub tier2_token_cap: u32,
}

impl Default for TierThresholds {
    fn default() -> Self {
        Self {
            tier1_token_cap: 4_000,
            tier2_token_cap: 16_000,
        }
    }
}

/// Cross-encoder reranker model (issue #39). An ENUM, not a magic string —
/// following the workspace "enums over strings" rule. Only the MIT-licensed
/// `BgeRerankerBase` is surfaced for the MVP (the v2-m3 mirror is out of scope).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RerankerModel {
    /// BGE reranker base (MIT). The default and only surfaced model.
    #[default]
    BgeRerankerBase,
}

/// Serde default for [`RerankerConfig::timeout_ms`] (3s).
fn default_reranker_timeout_ms() -> u64 {
    3_000
}

/// Opt-in cross-encoder reranker (issue #39). `enabled=false` by default — the
/// reranker is a strictly optional accelerator; the RRF query path is correct
/// and fast without it. Every field is `#[serde(default)]` so an old `config.json`
/// lacking the key (or a sub-field) deserializes to defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankerConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub model: RerankerModel,
    #[serde(default = "default_reranker_timeout_ms")]
    pub timeout_ms: u64,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: RerankerModel::default(),
            timeout_ms: default_reranker_timeout_ms(),
        }
    }
}

/// Serde default for [`RetrievalConfig::hybrid_enabled`] (`true`).
fn default_hybrid_enabled() -> bool {
    true
}

/// Hybrid-retrieval configuration (issue #39). Additive `#[serde(default)]` struct;
/// an absent `retrieval` key in an old `config.json` reads back as this default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Fuse dense + BM25 via RRF. `false` degrades to dense-only. Defaults `true`.
    #[serde(default = "default_hybrid_enabled")]
    pub hybrid_enabled: bool,
    #[serde(default)]
    pub reranker: RerankerConfig,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            hybrid_enabled: default_hybrid_enabled(),
            reranker: RerankerConfig::default(),
        }
    }
}

const DEFAULT_ACCENT: &str = "purple";

/// Serde default: configs without an `accent` key read back as `"purple"`.
fn default_accent() -> String {
    DEFAULT_ACCENT.to_string()
}

/// Serde default: `js_render_enabled` is `true` when absent (#78).
fn default_js_render_enabled() -> bool {
    true
}

/// Serde default: `reopen_last_notebook` is `true` when absent.
fn default_reopen_last_notebook() -> bool {
    true
}

/// Top-level application configuration. Loaded from / saved to `{data_dir}/config.json`;
/// missing file writes the default back; malformed file yields [`LensError::Parse`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    pub theme: String,
    /// Drives the `[data-accent]` token layer. Absent key reads as `"purple"`.
    #[serde(default = "default_accent")]
    pub accent: String,
    #[serde(default)]
    pub user_name: String,
    /// Empty string resolves to the registry default at the embedder boundary.
    #[serde(default)]
    pub embedding_model: String,
    /// `"fastembed"` | `"ollama"`. Empty resolves to `fastembed`. App-wide default
    /// for new notebooks (M4 Phase 4b-B); lives here, NOT in SQL.
    #[serde(default)]
    pub embedding_backend: String,
    /// Max source size in MB for non-PDF formats (#71). Empty resolves to 50 MB.
    #[serde(default)]
    pub max_source_mb: String,
    pub models: Vec<ModelConfig>,
    pub endpoints: BTreeMap<String, String>,
    pub voices: VoiceConfig,
    #[serde(default)]
    pub tts: TtsConfig,
    #[serde(default)]
    pub enrichment: EnrichmentConfig,
    /// Speech-to-text configuration (#42). Absent key reads as [`AsrConfig::default`].
    #[serde(default)]
    pub asr: AsrConfig,
    /// Hybrid-retrieval configuration (#39). Absent key reads as
    /// [`RetrievalConfig::default`] (hybrid on, reranker off).
    #[serde(default)]
    pub retrieval: RetrievalConfig,
    /// Explicit consent to upload raw audio to a cloud ASR provider (#45). A NEW
    /// flag, SEPARATE from `EnrichmentConfig::cloud_consent` (audio is more
    /// sensitive than text). Cloud ASR is refused (falls back to local) unless true.
    #[serde(default)]
    pub audio_cloud_consent: bool,
    /// SPA JS-render fallback (#78). Defaults to `true`; absent key reads as `true`.
    #[serde(default = "default_js_render_enabled")]
    pub js_render_enabled: bool,
    /// Auto-open most-recently-active notebook on cold launch. Defaults to `true`.
    #[serde(default = "default_reopen_last_notebook")]
    pub reopen_last_notebook: bool,
    pub paths: PathConfig,
    pub tier_thresholds: TierThresholds,
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
            asr: AsrConfig::default(),
            retrieval: RetrievalConfig::default(),
            audio_cloud_consent: false,
            js_render_enabled: default_js_render_enabled(),
            reopen_last_notebook: default_reopen_last_notebook(),
            paths: PathConfig::default(),
            tier_thresholds: TierThresholds::default(),
            onboarding_complete: false,
        }
    }
}

impl AppConfig {
    /// Loads config from `{dir}/config.json`. Missing file → writes the default.
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

    /// Writes config to `{dir}/config.json` (pretty JSON) with `0o600` permissions
    /// on Unix (plaintext `api_key` stopgap until M2).
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
        assert!(!debug.contains("sk-supersecret-token-value"));
        assert!(!debug.contains("supersecret"));
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
        assert!(debug.contains("api_key: \"\""));
        assert!(!debug.contains("***"));
    }

    #[test]
    fn default_accent_is_purple() {
        assert_eq!(AppConfig::default().accent, "purple");
    }

    #[test]
    fn missing_accent_deserializes_to_purple() {
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

    #[test]
    fn test_max_source_mb_default() {
        assert_eq!(AppConfig::default().max_source_mb, "");
    }

    #[test]
    fn test_max_source_mb_missing_deserializes_to_empty() {
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
        assert!(!debug.contains("sk-elevenlabs-supersecret"));
        assert!(!debug.contains("supersecret"));
        assert!(debug.contains("***"));
        assert!(debug.contains("elevenlabs"));
    }

    #[test]
    fn tts_config_debug_shows_empty_api_key_as_empty() {
        let cfg = TtsConfig::default();
        let debug = format!("{cfg:?}");
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
    fn default_asr_is_router_decided() {
        let asr = AppConfig::default().asr;
        assert_eq!(asr.backend, "");
        assert_eq!(asr.whisper_model, "base");
        assert_eq!(asr.language, None);
        assert!(!asr.translate);
        assert_eq!(asr, AsrConfig::default());
    }

    #[test]
    fn missing_asr_deserializes_to_default() {
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
        assert_eq!(config.asr, AsrConfig::default());
        assert_eq!(config.asr.whisper_model, "base");
    }

    #[test]
    fn partial_asr_fills_whisper_model_default() {
        let json = r#"{ "backend": "local_whisper" }"#;
        let asr: AsrConfig = serde_json::from_str(json).unwrap();
        assert_eq!(asr.backend, "local_whisper");
        assert_eq!(asr.whisper_model, "base");
        assert_eq!(asr.language, None);
        assert!(!asr.translate);
    }

    #[test]
    fn explicit_asr_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let config = AppConfig {
            asr: AsrConfig {
                backend: "apple_native".to_string(),
                whisper_model: "small".to_string(),
                language: Some(Lang::De),
                translate: true,
                ..AsrConfig::default()
            },
            ..AppConfig::default()
        };
        config.save(dir.path()).unwrap();
        let loaded = AppConfig::load(dir.path()).unwrap();
        assert_eq!(loaded.asr, config.asr);
    }

    #[test]
    fn missing_tts_deserializes_to_default() {
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
        assert_eq!(
            serde_json::to_string(&CorefStrategy::None).unwrap(),
            "\"none\""
        );
        assert_eq!(
            serde_json::to_string(&CorefStrategy::LlmInline).unwrap(),
            "\"llm_inline\""
        );
        let s: CorefStrategy = serde_json::from_str("\"llm_inline\"").unwrap();
        assert_eq!(s, CorefStrategy::LlmInline);
    }

    #[test]
    fn legacy_dedicated_model_coref_deserializes_to_llm_inline() {
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
        assert_eq!(loaded.enrichment.routing, LlmRouting::LocalFirst);

        let on_disk = std::fs::read_to_string(dir.path().join(CONFIG_FILE_NAME)).unwrap();
        assert!(on_disk.contains("\"coref_strategy\": \"none\""));
    }

    #[test]
    fn provider_driven_defaults_match_the_locked_rule() {
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

        let both = EnrichmentConfig::for_provider(true, true);
        assert!(both.enabled);
        assert_eq!(both.coref_strategy, CorefStrategy::LlmInline);
    }

    #[test]
    fn default_per_task_models_are_none() {
        let e = EnrichmentConfig::default();
        assert_eq!(e.coref_model, None);
        assert_eq!(e.map_model, None);
        assert_eq!(e.chat_model, None);
    }

    #[test]
    fn default_relations_strategy_is_off() {
        assert_eq!(
            EnrichmentConfig::default().relations_strategy,
            RelationsStrategy::Off
        );
        // AC3 compile-coverage: `for_provider` populates the field (defaults to Off).
        assert_eq!(
            EnrichmentConfig::for_provider(true, false).relations_strategy,
            RelationsStrategy::Off
        );
    }

    #[test]
    fn missing_per_task_models_deserialize_to_none() {
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
        // AC1: a config without `relations_strategy` reads back as `Off`.
        assert_eq!(e.relations_strategy, RelationsStrategy::Off);
    }

    #[test]
    fn explicit_per_task_models_round_trip() {
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
        assert_eq!(loaded.enrichment.chat_model, None);
    }

    #[test]
    fn chat_model_round_trips_persist_and_reload() {
        let dir = tempfile::tempdir().unwrap();
        let config = AppConfig {
            enrichment: EnrichmentConfig {
                enabled: true,
                chat_model: Some(TaskModel {
                    provider: "openai".to_string(),
                    model: "gpt-4o".to_string(),
                }),
                ..EnrichmentConfig::default()
            },
            ..AppConfig::default()
        };
        config.save(dir.path()).unwrap();
        let loaded = AppConfig::load(dir.path()).unwrap();
        assert_eq!(
            loaded.enrichment.chat_model,
            Some(TaskModel {
                provider: "openai".to_string(),
                model: "gpt-4o".to_string(),
            }),
            "chat_model must survive a save→reload cycle intact"
        );
        assert_eq!(loaded.enrichment.coref_model, None);
        assert_eq!(loaded.enrichment.map_model, None);
        let on_disk = std::fs::read_to_string(dir.path().join(CONFIG_FILE_NAME)).unwrap();
        assert!(
            on_disk.contains("\"chat_model\""),
            "chat_model key must appear in config.json"
        );
        assert!(
            on_disk.contains("\"gpt-4o\""),
            "chat_model.model must appear in config.json"
        );
    }

    #[test]
    fn task_model_serializes_to_flat_snake_case() {
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
        let catalog = crate::model_catalog::ModelCatalog::bundled();
        let coref = TaskModel {
            provider: "anthropic".to_string(),
            model: "claude-sonnet-4-5".to_string(),
        };
        assert!(catalog.validate(&coref.provider, &coref.model).is_ok());
        assert!(catalog.validate("anthropic", "totally-made-up").is_err());
        assert!(catalog.is_valid("ollama", "qwen2.5-coder"));
    }

    #[test]
    fn default_js_render_enabled_is_true() {
        assert!(
            AppConfig::default().js_render_enabled,
            "js_render_enabled must default ON"
        );
    }

    #[test]
    fn missing_js_render_enabled_deserializes_to_true() {
        let json = r#"{
            "theme": "dark",
            "accent": "purple",
            "user_name": "",
            "embedding_model": "",
            "embedding_backend": "",
            "models": [],
            "endpoints": {},
            "voices": { "host": "", "guest": "" },
            "tts": { "provider": "", "api_key": "" },
            "paths": { "data_dir": "" },
            "tier_thresholds": { "tier1_token_cap": 4000, "tier2_token_cap": 16000 },
            "onboarding_complete": true
        }"#;
        let config: AppConfig = serde_json::from_str(json).unwrap();
        assert!(
            config.js_render_enabled,
            "absent js_render_enabled key must read back as true"
        );
    }

    #[test]
    fn default_retrieval_is_hybrid_on_reranker_off() {
        let r = AppConfig::default().retrieval;
        assert!(r.hybrid_enabled, "hybrid retrieval defaults ON");
        assert!(!r.reranker.enabled, "reranker defaults OFF (opt-in)");
        assert_eq!(r.reranker.model, RerankerModel::BgeRerankerBase);
        assert_eq!(r.reranker.timeout_ms, 3_000);
        assert_eq!(r, RetrievalConfig::default());
    }

    #[test]
    fn missing_retrieval_deserializes_to_default() {
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
        assert_eq!(config.retrieval, RetrievalConfig::default());
        assert!(config.retrieval.hybrid_enabled);
        assert!(!config.retrieval.reranker.enabled);
    }

    #[test]
    fn partial_retrieval_fills_sub_field_defaults() {
        // An old config with `reranker.enabled` set but no `timeout_ms`/`model`
        // fills the missing sub-fields from defaults.
        let json = r#"{ "reranker": { "enabled": true } }"#;
        let r: RetrievalConfig = serde_json::from_str(json).unwrap();
        assert!(r.hybrid_enabled, "absent hybrid_enabled reads back true");
        assert!(r.reranker.enabled);
        assert_eq!(r.reranker.model, RerankerModel::BgeRerankerBase);
        assert_eq!(r.reranker.timeout_ms, 3_000);
    }

    #[test]
    fn reranker_model_serializes_to_stable_snake_case() {
        assert_eq!(
            serde_json::to_string(&RerankerModel::BgeRerankerBase).unwrap(),
            "\"bge_reranker_base\""
        );
        let m: RerankerModel = serde_json::from_str("\"bge_reranker_base\"").unwrap();
        assert_eq!(m, RerankerModel::BgeRerankerBase);
    }

    #[test]
    fn explicit_retrieval_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let config = AppConfig {
            retrieval: RetrievalConfig {
                hybrid_enabled: false,
                reranker: RerankerConfig {
                    enabled: true,
                    model: RerankerModel::BgeRerankerBase,
                    timeout_ms: 1_500,
                },
            },
            ..AppConfig::default()
        };
        config.save(dir.path()).unwrap();
        let loaded = AppConfig::load(dir.path()).unwrap();
        assert_eq!(loaded.retrieval, config.retrieval);
        assert!(!loaded.retrieval.hybrid_enabled);
        assert!(loaded.retrieval.reranker.enabled);
        assert_eq!(loaded.retrieval.reranker.timeout_ms, 1_500);

        let on_disk = std::fs::read_to_string(dir.path().join(CONFIG_FILE_NAME)).unwrap();
        assert!(on_disk.contains("\"model\": \"bge_reranker_base\""));
    }

    #[test]
    fn default_reopen_last_notebook_is_true() {
        assert!(
            AppConfig::default().reopen_last_notebook,
            "reopen_last_notebook must default ON"
        );
    }

    #[test]
    fn missing_reopen_last_notebook_deserializes_to_true() {
        let json = r#"{
            "theme": "dark",
            "accent": "purple",
            "user_name": "",
            "embedding_model": "",
            "embedding_backend": "",
            "models": [],
            "endpoints": {},
            "voices": { "host": "", "guest": "" },
            "tts": { "provider": "", "api_key": "" },
            "paths": { "data_dir": "" },
            "tier_thresholds": { "tier1_token_cap": 4000, "tier2_token_cap": 16000 },
            "onboarding_complete": true
        }"#;
        let config: AppConfig = serde_json::from_str(json).unwrap();
        assert!(
            config.reopen_last_notebook,
            "absent reopen_last_notebook key must read back as true"
        );
    }
}
