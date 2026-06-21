//! Application configuration, persisted as disk-only `config.json`.
//!
//! There is intentionally NO `app_config` table in SQLite (see plan §2): the
//! config shape is still evolving, and disk JSON avoids checksum-locking a guess
//! into a migration. If DB-resident flags are ever needed, add an `app_flags`
//! table as an additive migration in the milestone that requires it.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::LensError;

/// File name for the on-disk config, relative to the engine data directory.
const CONFIG_FILE_NAME: &str = "config.json";

/// Per-provider model endpoint configuration (LLM or embedding backend).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
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
    pub api_key: String,
}

/// Host/guest voice identifiers used by the (future) TTS subsystem.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// Voice id for the "host" speaker.
    pub host: String,
    /// Voice id for the "guest" speaker.
    pub guest: String,
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

/// Top-level application configuration.
///
/// Loaded from / saved to `{data_dir}/config.json`. A missing file yields
/// [`AppConfig::default`] (and is written back); a malformed file yields
/// [`LensError::Parse`] rather than panicking.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    /// UI theme name (e.g. `"system"`, `"light"`, `"dark"`).
    pub theme: String,
    /// Configured chat/inference models keyed by role.
    pub models: Vec<ModelConfig>,
    /// Arbitrary named endpoints (label -> URL).
    pub endpoints: std::collections::BTreeMap<String, String>,
    /// Host/guest TTS voices.
    pub voices: VoiceConfig,
    /// Filesystem paths.
    pub paths: PathConfig,
    /// Tier token thresholds.
    pub tier_thresholds: TierThresholds,
    /// Whether first-run onboarding has completed.
    pub onboarding_complete: bool,
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
                let config = serde_json::from_str(&contents)
                    .map_err(|e| LensError::Parse(format!("{}: {e}", path.display())))?;
                Ok(config)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                tracing::debug!("no config at {}, writing default", path.display());
                let config = AppConfig::default();
                config.save(dir)?;
                Ok(config)
            }
            Err(e) => Err(LensError::Io(format!("{}: {e}", path.display()))),
        }
    }

    /// Serializes and writes this config to `{dir}/config.json` (pretty JSON).
    #[tracing::instrument(skip_all, fields(dir = %dir.as_ref().display()))]
    pub fn save(&self, dir: impl AsRef<Path>) -> Result<(), LensError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)
            .map_err(|e| LensError::Io(format!("{}: {e}", dir.display())))?;
        let path = dir.join(CONFIG_FILE_NAME);
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)
            .map_err(|e| LensError::Io(format!("{}: {e}", path.display())))?;
        tracing::debug!("saved config to {}", path.display());
        Ok(())
    }
}
