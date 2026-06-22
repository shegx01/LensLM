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
    /// Selected embedding model id (e.g. `"nomic-embed-text"`). Empty when the
    /// user has not yet chosen one. Defaults to `""`; an absent field in an older
    /// `config.json` reads back as `""` via `#[serde(default)]`.
    #[serde(default)]
    pub embedding_model: String,
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
            embedding_model: String::default(),
            models: Vec::default(),
            endpoints: BTreeMap::default(),
            voices: VoiceConfig::default(),
            tts: TtsConfig::default(),
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
}
