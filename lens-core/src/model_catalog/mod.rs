//! Typed model catalog sourced from models.dev.
//!
//! Parses the [models.dev](https://models.dev/api.json) catalog and exposes [`ModelCatalog::validate`]
//! — the anti-free-string guard. Schema is tolerant: unknown providers, extra fields, and
//! unrecognized `reasoning_options` variants never fail the parse; missing optional fields
//! degrade to `Option`/`Default`.
//!
//! The bundled fallback is the FULL catalog (~2.4 MB raw, ~200 KB gzipped), vendored by
//! `scripts/fetch-models-catalog.sh` and decompressed via `flate2`. [`refresh_if_stale`]
//! replaces it at runtime so online users always converge to the current list.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error::LensError;

/// Catalog endpoint; a constant so tests can substitute a mock server URL.
pub const MODELS_CATALOG_URL: &str = "https://models.dev/api.json";

pub const MODELS_CATALOG_RELPATH: &str = "models/models-catalog.json";
pub const MODELS_CATALOG_FILENAME: &str = "models-catalog.json";

/// ~8 hours → re-fetched at most ~3×/day (checked on app start or picker open).
pub const MODELS_CATALOG_REFRESH_INTERVAL: Duration = Duration::from_secs(8 * 60 * 60);

const CATALOG_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Body is ~2.4 MB, so the read timeout is longer than a system-check probe but still bounded.
const CATALOG_FETCH_TIMEOUT: Duration = Duration::from_secs(30);
/// Defense-in-depth cap: the real catalog is ~2.4 MB; 16 MB leaves headroom for growth.
const MAX_CATALOG_BODY_BYTES: u64 = 16 * 1024 * 1024;

/// Full `https://models.dev/api.json` catalog gzipped at build time
/// (redistributed under models.dev's MIT license, github.com/sst/models.dev).
const BUNDLED_CATALOG_GZ: &[u8] = include_bytes!("bundled-catalog.json.gz");

// ---------------------------------------------------------------------------
// Typed schema (tolerant subset of models.dev)
// ---------------------------------------------------------------------------

/// Full parsed catalog: provider key → entry. `BTreeMap` for deterministic iteration order.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelCatalog {
    pub providers: BTreeMap<String, ProviderEntry>,
}

/// One provider's entry. Extra catalog fields (`npm`, `api`, …) are tolerated — no
/// `deny_unknown_fields`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ProviderEntry {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub env: Vec<String>,
    #[serde(default)]
    pub doc: Option<String>,
    #[serde(default)]
    pub models: BTreeMap<String, ModelInfo>,
}

/// One model's capabilities + economics. All optional fields degrade to `Option`/`Default`
/// rather than failing the parse. `context_limit`/`output_limit` are flattened from the
/// catalog's nested `limit: { context, output }` by a manual [`Deserialize`] impl, then
/// serialized FLAT — what the TS mirror and Svelte picker consume (`.toLocaleString()`).
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    pub reasoning: bool,
    /// Tolerant: unrecognized variants become [`ReasoningOption::Other`] rather than failing.
    pub reasoning_options: Vec<ReasoningOption>,
    pub tool_call: bool,
    pub temperature: bool,
    pub modalities: Modalities,
    /// Serialized FLAT as `context_limit` for the TS mirror / IPC.
    pub context_limit: Option<u32>,
    /// Serialized FLAT as `output_limit` for the TS mirror / IPC.
    pub output_limit: Option<u32>,
    pub open_weights: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<Cost>,
    /// ISO `YYYY-MM-DD`. Picker sorts cloud options by this (newest first).
    pub last_updated: Option<String>,
    /// ISO `YYYY-MM-DD`. Tiebreaker when `last_updated` matches.
    pub release_date: Option<String>,
}

impl<'de> Deserialize<'de> for ModelInfo {
    fn deserialize<D>(de: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Inner struct mirrors the catalog's verbatim nested shape; flattened into ModelInfo.
        #[derive(Deserialize)]
        struct Raw {
            #[serde(default)]
            id: String,
            #[serde(default)]
            name: String,
            #[serde(default)]
            family: Option<String>,
            #[serde(default)]
            reasoning: bool,
            #[serde(default)]
            reasoning_options: Vec<ReasoningOption>,
            #[serde(default)]
            tool_call: bool,
            #[serde(default)]
            temperature: bool,
            #[serde(default)]
            modalities: Modalities,
            #[serde(default)]
            limit: Option<Limit>,
            #[serde(default)]
            open_weights: bool,
            #[serde(default)]
            cost: Option<Cost>,
            #[serde(default)]
            last_updated: Option<String>,
            #[serde(default)]
            release_date: Option<String>,
        }
        let raw = Raw::deserialize(de)?;
        Ok(ModelInfo {
            id: raw.id,
            name: raw.name,
            family: raw.family,
            reasoning: raw.reasoning,
            reasoning_options: raw.reasoning_options,
            tool_call: raw.tool_call,
            temperature: raw.temperature,
            modalities: raw.modalities,
            context_limit: raw.limit.as_ref().and_then(|l| l.context),
            output_limit: raw.limit.and_then(|l| l.output),
            open_weights: raw.open_weights,
            cost: raw.cost,
            last_updated: raw.last_updated,
            release_date: raw.release_date,
        })
    }
}

/// A single reasoning control on a model. Tagged on the catalog's `type` field.
/// An unrecognized `type` falls through to `Other` rather than failing the parse.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningOption {
    Effort {
        /// Nulls in the array are dropped (the catalog occasionally seeds these).
        #[serde(default, deserialize_with = "deserialize_tolerant_strings")]
        values: Vec<String>,
    },
    BudgetTokens {
        /// Sentinel negatives (e.g. `-1`) degrade to `None` rather than failing.
        #[serde(default, deserialize_with = "deserialize_tolerant_u32")]
        min: Option<u32>,
        #[serde(default, deserialize_with = "deserialize_tolerant_u32")]
        max: Option<u32>,
    },
    Toggle,
    #[serde(other)]
    Other,
}

/// Input/output modalities a model accepts/produces.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Modalities {
    #[serde(default)]
    pub input: Vec<String>,
    #[serde(default)]
    pub output: Vec<String>,
}

/// Per-token cost in USD per 1M tokens. All fields optional — providers report different subsets.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Cost {
    #[serde(default)]
    pub input: Option<f64>,
    #[serde(default)]
    pub output: Option<f64>,
    #[serde(default)]
    pub cache_read: Option<f64>,
    #[serde(default)]
    pub cache_write: Option<f64>,
}

/// Catalog's nested `limit: { context, output }`. Internal only — flattened into
/// [`ModelInfo::context_limit`] / [`ModelInfo::output_limit`] by the manual deserializer.
#[derive(Debug, Deserialize)]
struct Limit {
    #[serde(default, deserialize_with = "deserialize_tolerant_u32")]
    context: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_tolerant_u32")]
    output: Option<u32>,
}

/// Tolerates out-of-range values (e.g. sentinel `-1`) that degrade to `None` rather than
/// failing the parse. A clean in-range integer parses normally; JSON `null` stays `None`.
fn deserialize_tolerant_u32<'de, D>(de: D) -> Result<Option<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw: Option<serde_json::Value> = Option::deserialize(de)?;
    Ok(raw
        .and_then(|v| v.as_u64())
        .and_then(|n| u32::try_from(n).ok()))
}

/// Tolerates `null` array elements (the catalog occasionally seeds effort `values` with nulls);
/// they are dropped rather than failing the parse.
fn deserialize_tolerant_strings<'de, D>(de: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw: Vec<Option<String>> = Vec::deserialize(de)?;
    Ok(raw.into_iter().flatten().collect())
}

// ---------------------------------------------------------------------------
// SupportedProvider — the curated default surface
// ---------------------------------------------------------------------------

/// Providers LensLM surfaces by default. `Other` keeps any catalog provider addressable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupportedProvider {
    Anthropic,
    OpenAI,
    Google,
    /// Local runtime; models are user-pulled and NOT necessarily in models.dev.
    Ollama,
    OllamaCloud,
    Zai,
    Other(String),
}

impl SupportedProvider {
    pub fn catalog_key(&self) -> &str {
        match self {
            SupportedProvider::Anthropic => "anthropic",
            SupportedProvider::OpenAI => "openai",
            SupportedProvider::Google => "google",
            SupportedProvider::Ollama => "ollama",
            SupportedProvider::OllamaCloud => "ollama-cloud",
            SupportedProvider::Zai => "zai",
            SupportedProvider::Other(key) => key,
        }
    }

    pub fn from_catalog_key(key: &str) -> Self {
        match key {
            "anthropic" => SupportedProvider::Anthropic,
            "openai" => SupportedProvider::OpenAI,
            "google" => SupportedProvider::Google,
            "ollama" => SupportedProvider::Ollama,
            "ollama-cloud" => SupportedProvider::OllamaCloud,
            "zai" => SupportedProvider::Zai,
            other => SupportedProvider::Other(other.to_string()),
        }
    }

    /// Single locality predicate shared by the catalog-validation exemption here and the
    /// consent-gate exemption in `crate::llm` — they MUST agree.
    pub(crate) fn is_local(key: &str) -> bool {
        key == SupportedProvider::Ollama.catalog_key()
    }
}

// ---------------------------------------------------------------------------
// Validation — the anti-free-string guard
// ---------------------------------------------------------------------------

impl ModelCatalog {
    pub fn from_json(bytes: &[u8]) -> Result<Self, LensError> {
        serde_json::from_slice(bytes).map_err(LensError::from)
    }

    /// Decompresses and parses `bundled-catalog.json.gz`, memoized via [`LazyLock`].
    /// Called on the enrichment hot path; the `expect`s stay (a malformed bundle is a
    /// build-time catastrophe caught by tests).
    pub fn bundled() -> Self {
        static BUNDLED: std::sync::LazyLock<ModelCatalog> = std::sync::LazyLock::new(|| {
            use std::io::Read;
            let mut decoder = flate2::read::GzDecoder::new(BUNDLED_CATALOG_GZ);
            let mut json = Vec::new();
            decoder
                .read_to_end(&mut json)
                .expect("bundled model catalog must gunzip");
            ModelCatalog::from_json(&json).expect("bundled model catalog must be valid JSON")
        });
        BUNDLED.clone()
    }

    pub fn provider(&self, provider_key: &str) -> Option<&ProviderEntry> {
        self.providers.get(provider_key)
    }

    /// Anti-free-string guard: validates `(provider_key, model_id)` against the catalog.
    /// Local Ollama exception: any non-empty id is accepted (user-pulled models may not be listed).
    /// TODO(stage2): validate Ollama-local ids against the live `/api/tags`.
    pub fn validate(&self, provider_key: &str, model_id: &str) -> Result<&ModelInfo, LensError> {
        if model_id.is_empty() {
            return Err(LensError::Validation("model id must not be empty".into()));
        }

        if SupportedProvider::is_local(provider_key) {
            if let Some(info) = self
                .providers
                .get(provider_key)
                .and_then(|p| p.models.get(model_id))
            {
                return Ok(info);
            }
            // TODO(stage2): validate against the live Ollama `/api/tags`.
            return Err(LensError::Validation(format!(
                "ollama-local model '{model_id}' is not in the catalog; live /api/tags \
                 validation lands in stage 2 (TODO)"
            )));
        }

        let provider = self.providers.get(provider_key).ok_or_else(|| {
            LensError::Validation(format!("unknown model provider: '{provider_key}'"))
        })?;
        provider.models.get(model_id).ok_or_else(|| {
            LensError::Validation(format!(
                "model '{model_id}' is not in the catalog for provider '{provider_key}'"
            ))
        })
    }

    /// Convenience over [`validate`](Self::validate). For local Ollama, any non-empty id is valid.
    pub fn is_valid(&self, provider_key: &str, model_id: &str) -> bool {
        if SupportedProvider::is_local(provider_key) {
            return !model_id.is_empty();
        }
        self.validate(provider_key, model_id).is_ok()
    }
}

// ---------------------------------------------------------------------------
// Staleness policy
// ---------------------------------------------------------------------------

/// Returns whether `mtime` is stale relative to `now`. A future `mtime` (clock skew) is
/// treated as fresh. A zero `interval` makes everything stale.
pub fn is_stale(mtime: SystemTime, now: SystemTime, interval: Duration) -> bool {
    match now.duration_since(mtime) {
        Ok(age) => age >= interval,
        Err(_) => false, // mtime in the future (clock skew): treat as fresh
    }
}

// ---------------------------------------------------------------------------
// Cache / fetch / refresh
// ---------------------------------------------------------------------------

pub fn catalog_cache_path(data_dir: &Path) -> PathBuf {
    data_dir.join("models").join(MODELS_CATALOG_FILENAME)
}

/// Loads the model catalog, never failing hard. Falls back to the bundled snapshot on any
/// read or parse error so the returned catalog is always usable for validation.
pub fn load_catalog(data_dir: &Path) -> ModelCatalog {
    let path = catalog_cache_path(data_dir);
    match std::fs::read(&path) {
        Ok(bytes) => match ModelCatalog::from_json(&bytes) {
            Ok(catalog) => catalog,
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    "cached model catalog failed to parse; using bundled snapshot: {e}"
                );
                ModelCatalog::bundled()
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => ModelCatalog::bundled(),
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                "cached model catalog unreadable; using bundled snapshot: {e}"
            );
            ModelCatalog::bundled()
        }
    }
}

/// Re-fetches the catalog from `url` when stale or absent, atomically replacing the cache.
/// Returns `Ok(true)` when refreshed, `Ok(false)` when still fresh. Best-effort: the caller
/// can fire-and-forget; errors leave the existing cache (or bundled snapshot) in place.
pub async fn refresh_if_stale(
    data_dir: &Path,
    url: &str,
    client: &reqwest::Client,
) -> Result<bool, LensError> {
    let path = catalog_cache_path(data_dir);

    if let Ok(meta) = std::fs::metadata(&path)
        && let Ok(mtime) = meta.modified()
        && !is_stale(mtime, SystemTime::now(), MODELS_CATALOG_REFRESH_INTERVAL)
    {
        return Ok(false);
    }

    let bytes = fetch_catalog_bytes(url, client).await?;
    // Validate before overwriting — never replace a good cache with a corrupt body.
    ModelCatalog::from_json(&bytes)?;
    write_cache_atomic(&path, &bytes)?;
    tracing::info!(
        path = %path.display(),
        bytes = bytes.len(),
        "refreshed model catalog cache"
    );
    Ok(true)
}

pub fn catalog_client() -> reqwest::Client {
    crate::http::hardened_client(CATALOG_CONNECT_TIMEOUT, CATALOG_FETCH_TIMEOUT)
}

/// Streams the catalog body, aborting if it exceeds [`MAX_CATALOG_BODY_BYTES`].
async fn fetch_catalog_bytes(url: &str, client: &reqwest::Client) -> Result<Vec<u8>, LensError> {
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| LensError::Network(format!("catalog fetch failed: {e}")))?;
    if !resp.status().is_success() {
        return Err(LensError::Network(format!(
            "catalog fetch returned HTTP {}",
            resp.status()
        )));
    }
    if let Some(len) = resp.content_length()
        && len > MAX_CATALOG_BODY_BYTES
    {
        return Err(LensError::Network(
            "catalog body exceeds size cap".to_string(),
        ));
    }

    let mut body: Vec<u8> = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| LensError::Network(format!("catalog stream error: {e}")))?;
        if body.len() as u64 + chunk.len() as u64 > MAX_CATALOG_BODY_BYTES {
            return Err(LensError::Network(
                "catalog body exceeds size cap".to_string(),
            ));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Atomically writes via `.part` + rename so a partial write never masquerades as a complete
/// cache.
fn write_cache_atomic(path: &Path, bytes: &[u8]) -> Result<(), LensError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| LensError::Io(format!("create {}: {e}", parent.display())))?;
    }
    let tmp = path.with_extension("part");
    std::fs::write(&tmp, bytes)
        .map_err(|e| LensError::Io(format!("write {}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        LensError::Io(format!("finalize {}: {e}", path.display()))
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// A representative slice of the real models.dev schema: an unknown provider,
    /// extra provider/model fields, every `reasoning_options` variant, and models
    /// with missing optional fields.
    const FIXTURE: &str = r#"{
        "anthropic": {
            "id": "anthropic",
            "name": "Anthropic",
            "npm": "@ai-sdk/anthropic",
            "api": "https://api.anthropic.com",
            "env": ["ANTHROPIC_API_KEY"],
            "doc": "https://docs.anthropic.com",
            "models": {
                "claude-sonnet-4-5": {
                    "id": "anthropic/claude-sonnet-4-5",
                    "name": "Claude Sonnet 4.5",
                    "family": "claude-sonnet",
                    "attachment": true,
                    "reasoning": true,
                    "reasoning_options": [],
                    "tool_call": true,
                    "temperature": true,
                    "knowledge": "2025-07-31",
                    "release_date": "2025-09-29",
                    "last_updated": "2025-09-29",
                    "modalities": { "input": ["text", "image", "pdf"], "output": ["text"] },
                    "open_weights": false,
                    "limit": { "context": 1000000, "output": 64000 },
                    "cost": { "input": 3, "output": 15, "cache_read": 0.3, "cache_write": 3.75 }
                }
            }
        },
        "openai": {
            "id": "openai",
            "name": "OpenAI",
            "env": ["OPENAI_API_KEY"],
            "models": {
                "o4-mini": {
                    "id": "openai/o4-mini",
                    "name": "o4-mini",
                    "reasoning": true,
                    "reasoning_options": [{ "type": "effort", "values": ["low", "medium", "high"] }],
                    "tool_call": true,
                    "temperature": false,
                    "modalities": { "input": ["text", "image"], "output": ["text"] },
                    "open_weights": false,
                    "limit": { "context": 200000, "output": 100000 }
                }
            }
        },
        "google": {
            "id": "google",
            "name": "Google",
            "env": ["GEMINI_API_KEY"],
            "models": {
                "gemini-2.5-pro": {
                    "id": "google/gemini-2.5-pro",
                    "name": "Gemini 2.5 Pro",
                    "reasoning": true,
                    "reasoning_options": [{ "type": "budget_tokens", "min": 128, "max": 32768 }],
                    "tool_call": true,
                    "temperature": true,
                    "modalities": { "input": ["text"], "output": ["text"] }
                }
            }
        },
        "zai": {
            "id": "zai",
            "name": "Z.ai",
            "models": {
                "glm-4.6": {
                    "id": "zai/glm-4.6",
                    "name": "GLM-4.6",
                    "reasoning": true,
                    "reasoning_options": [{ "type": "toggle" }],
                    "tool_call": true
                }
            }
        },
        "ollama": {
            "id": "ollama",
            "name": "Ollama",
            "models": {
                "llama3.1": { "id": "ollama/llama3.1", "name": "Llama 3.1" }
            }
        },
        "mystery-provider-9000": {
            "id": "mystery-provider-9000",
            "name": "Future Provider",
            "totally_new_field": 42,
            "models": {
                "future-model": {
                    "id": "mystery/future-model",
                    "name": "Future Model",
                    "reasoning": true,
                    "reasoning_options": [{ "type": "warp_drive", "factor": 9 }],
                    "brand_new_capability": "yes"
                }
            }
        }
    }"#;

    fn fixture() -> ModelCatalog {
        ModelCatalog::from_json(FIXTURE.as_bytes()).expect("fixture parses")
    }

    #[test]
    fn parses_representative_schema_slice() {
        let catalog = fixture();
        let anthropic = catalog.provider("anthropic").expect("anthropic present");
        assert_eq!(anthropic.name, "Anthropic");
        assert_eq!(anthropic.env, vec!["ANTHROPIC_API_KEY"]);
        assert_eq!(anthropic.doc.as_deref(), Some("https://docs.anthropic.com"));

        let sonnet = anthropic
            .models
            .get("claude-sonnet-4-5")
            .expect("model present");
        assert_eq!(sonnet.name, "Claude Sonnet 4.5");
        assert_eq!(sonnet.family.as_deref(), Some("claude-sonnet"));
        assert!(sonnet.reasoning);
        assert!(sonnet.tool_call);
        assert!(sonnet.temperature);
        assert_eq!(sonnet.context_limit, Some(1_000_000));
        assert_eq!(sonnet.output_limit, Some(64_000));
        assert_eq!(sonnet.modalities.input, vec!["text", "image", "pdf"]);
        let cost = sonnet.cost.as_ref().expect("cost present");
        assert_eq!(cost.input, Some(3.0));
        assert_eq!(cost.cache_write, Some(3.75));
        assert_eq!(sonnet.last_updated.as_deref(), Some("2025-09-29"));
        assert_eq!(sonnet.release_date.as_deref(), Some("2025-09-29"));
    }

    #[test]
    fn parses_all_reasoning_option_variants() {
        let catalog = fixture();

        let effort = &catalog.providers["openai"].models["o4-mini"].reasoning_options;
        assert_eq!(
            effort,
            &vec![ReasoningOption::Effort {
                values: vec!["low".into(), "medium".into(), "high".into()]
            }]
        );

        let budget = &catalog.providers["google"].models["gemini-2.5-pro"].reasoning_options;
        assert_eq!(
            budget,
            &vec![ReasoningOption::BudgetTokens {
                min: Some(128),
                max: Some(32768)
            }]
        );

        let toggle = &catalog.providers["zai"].models["glm-4.6"].reasoning_options;
        assert_eq!(toggle, &vec![ReasoningOption::Toggle]);

        let unknown =
            &catalog.providers["mystery-provider-9000"].models["future-model"].reasoning_options;
        assert_eq!(unknown, &vec![ReasoningOption::Other]);
    }

    #[test]
    fn tolerates_out_of_range_reasoning_budget_and_limits() {
        // Regression guard: sentinel negatives must degrade to `None`, never fail the parse
        // (which would poison the bundled() LazyLock).
        const SLICE: &str = r#"{
            "weird": {
                "id": "weird",
                "name": "Weird",
                "models": {
                    "m": {
                        "id": "weird/m",
                        "name": "M",
                        "reasoning": true,
                        "reasoning_options": [
                            { "type": "budget_tokens", "min": -1, "max": 32768 },
                            { "type": "effort", "values": [null, "low", "high"] }
                        ],
                        "limit": { "context": -5, "output": 64000 }
                    }
                }
            }
        }"#;
        let catalog = ModelCatalog::from_json(SLICE.as_bytes()).expect("parses despite negatives");
        let m = &catalog.providers["weird"].models["m"];
        assert_eq!(
            m.reasoning_options,
            vec![
                ReasoningOption::BudgetTokens {
                    min: None,
                    max: Some(32768)
                },
                ReasoningOption::Effort {
                    values: vec!["low".into(), "high".into()]
                }
            ]
        );
        assert_eq!(m.context_limit, None);
        assert_eq!(m.output_limit, Some(64_000));
    }

    #[test]
    fn tolerates_unknown_providers_and_extra_fields() {
        let catalog = fixture();
        let mystery = catalog
            .provider("mystery-provider-9000")
            .expect("unknown provider tolerated");
        assert_eq!(mystery.name, "Future Provider");
        assert!(mystery.models.contains_key("future-model"));
    }

    #[test]
    fn model_info_serializes_flat_numeric_limits() {
        // Fix #2: must serialize as flat numbers (TS mirror calls `.toLocaleString()`),
        // not a nested `{ "context": N }` object and not drop `output_limit`.
        let catalog = fixture();
        let sonnet = &catalog.providers["anthropic"].models["claude-sonnet-4-5"];
        let value = serde_json::to_value(sonnet).expect("serializes");

        assert_eq!(
            value.get("context_limit"),
            Some(&serde_json::json!(1_000_000)),
            "context_limit must be a flat number"
        );
        assert_eq!(
            value.get("output_limit"),
            Some(&serde_json::json!(64_000)),
            "output_limit must be a flat number, not dropped"
        );
        assert!(
            value.get("limit").is_none(),
            "must not emit a nested `limit` object"
        );

        // Catalog dates serialize FLAT as strings (the picker sorts cloud options
        // by `last_updated`; the TS mirror types them `string | null`).
        assert_eq!(
            value.get("last_updated"),
            Some(&serde_json::json!("2025-09-29")),
            "last_updated must be a flat string"
        );
        assert_eq!(
            value.get("release_date"),
            Some(&serde_json::json!("2025-09-29")),
            "release_date must be a flat string"
        );

        let glm = &catalog.providers["zai"].models["glm-4.6"];
        let glm_value = serde_json::to_value(glm).expect("serializes");
        assert_eq!(
            glm_value.get("context_limit"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(
            glm_value.get("output_limit"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(
            glm_value.get("last_updated"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(
            glm_value.get("release_date"),
            Some(&serde_json::Value::Null)
        );
    }

    #[test]
    fn handles_missing_optional_fields_with_defaults() {
        let catalog = fixture();
        let glm = &catalog.providers["zai"].models["glm-4.6"];
        assert_eq!(glm.family, None);
        assert!(!glm.temperature);
        assert!(!glm.open_weights);
        assert_eq!(glm.context_limit, None);
        assert_eq!(glm.output_limit, None);
        assert!(glm.cost.is_none());
        assert!(glm.modalities.input.is_empty());
        assert_eq!(glm.last_updated, None);
        assert_eq!(glm.release_date, None);
    }

    #[test]
    fn validate_accepts_known_model() {
        let catalog = fixture();
        let info = catalog
            .validate("anthropic", "claude-sonnet-4-5")
            .expect("known model accepted");
        assert_eq!(info.id, "anthropic/claude-sonnet-4-5");
        assert!(catalog.is_valid("openai", "o4-mini"));
    }

    #[test]
    fn validate_rejects_unknown_model_and_provider() {
        let catalog = fixture();
        // Unknown model under a known provider.
        let err = catalog
            .validate("anthropic", "claude-does-not-exist")
            .unwrap_err();
        assert!(matches!(err, LensError::Validation(_)), "got {err:?}");
        assert!(!catalog.is_valid("anthropic", "claude-does-not-exist"));

        // Unknown provider entirely.
        let err = catalog.validate("nope", "whatever").unwrap_err();
        assert!(matches!(err, LensError::Validation(_)), "got {err:?}");

        assert!(catalog.validate("anthropic", "").is_err());
    }

    #[test]
    fn validate_ollama_local_accepts_any_nonempty_id() {
        let catalog = fixture();
        assert!(catalog.validate("ollama", "llama3.1").is_ok());
        assert!(catalog.is_valid("ollama", "my-custom-pull:latest"));
        assert!(!catalog.is_valid("ollama", ""));
    }

    #[test]
    fn is_local_is_the_single_locality_predicate() {
        // Fix #3: ollama-cloud is hosted, not local; `llm::is_local_provider` delegates here.
        assert!(SupportedProvider::is_local("ollama"));
        for key in [
            "anthropic",
            "openai",
            "google",
            "zai",
            "ollama-cloud",
            "openai-compatible",
        ] {
            assert!(!SupportedProvider::is_local(key), "{key} must not be local");
        }
    }

    #[test]
    fn supported_provider_round_trips_catalog_keys() {
        for (variant, key) in [
            (SupportedProvider::Anthropic, "anthropic"),
            (SupportedProvider::OpenAI, "openai"),
            (SupportedProvider::Google, "google"),
            (SupportedProvider::Ollama, "ollama"),
            (SupportedProvider::OllamaCloud, "ollama-cloud"),
            (SupportedProvider::Zai, "zai"),
        ] {
            assert_eq!(variant.catalog_key(), key);
            assert_eq!(SupportedProvider::from_catalog_key(key), variant);
        }
        let other = SupportedProvider::from_catalog_key("cohere");
        assert_eq!(other, SupportedProvider::Other("cohere".into()));
        assert_eq!(other.catalog_key(), "cohere");
    }

    #[test]
    fn bundled_catalog_decompresses_parses_and_covers_supported_providers() {
        let catalog = ModelCatalog::bundled();
        for key in ["anthropic", "openai", "google", "zai", "ollama-cloud"] {
            assert!(
                catalog.provider(key).is_some(),
                "bundled catalog must cover supported cloud provider {key}"
            );
        }

        // Lower bounds only — not a frozen list (model ids change per release).
        assert!(
            catalog.providers.len() >= 50,
            "full catalog should carry many providers, got {}",
            catalog.providers.len()
        );
        let total_models: usize = catalog.providers.values().map(|p| p.models.len()).sum();
        assert!(
            total_models >= 1000,
            "full catalog should carry thousands of models, got {total_models}"
        );
        for key in ["anthropic", "openai", "google"] {
            let n = catalog.provider(key).unwrap().models.len();
            assert!(
                n >= 5,
                "{key} should carry many models in the full catalog, got {n}"
            );
        }
    }

    // --- staleness policy ---------------------------------------------------

    #[test]
    fn is_stale_distinguishes_fresh_from_stale() {
        let now = SystemTime::now();
        let interval = MODELS_CATALOG_REFRESH_INTERVAL;

        let fresh = now - Duration::from_secs(60 * 60);
        assert!(!is_stale(fresh, now, interval));

        let stale = now - Duration::from_secs(9 * 60 * 60);
        assert!(is_stale(stale, now, interval));

        // Exactly at the boundary is stale (>=).
        let boundary = now - interval;
        assert!(is_stale(boundary, now, interval));

        let future = now + Duration::from_secs(60 * 60);
        assert!(!is_stale(future, now, interval));
    }

    // --- load / fallback ----------------------------------------------------

    #[test]
    fn load_catalog_falls_back_to_bundled_when_cache_absent() {
        let dir = tempfile::tempdir().unwrap();
        let catalog = load_catalog(dir.path());
        assert!(catalog.provider("anthropic").is_some());
    }

    #[test]
    fn load_catalog_reads_cache_when_present() {
        let dir = tempfile::tempdir().unwrap();
        let path = catalog_cache_path(dir.path());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, FIXTURE).unwrap();

        let catalog = load_catalog(dir.path());
        assert!(catalog.provider("mystery-provider-9000").is_some()); // fixture-only provider proves cache was read
    }

    #[test]
    fn load_catalog_falls_back_when_cache_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let path = catalog_cache_path(dir.path());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"{ this is not valid json").unwrap();

        let catalog = load_catalog(dir.path());
        assert!(catalog.provider("anthropic").is_some());
        assert!(catalog.provider("mystery-provider-9000").is_none());
    }

    // --- refresh_if_stale (mock server) -------------------------------------

    #[tokio::test]
    async fn refresh_fetches_and_writes_cache_when_absent() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_string(FIXTURE))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let refreshed = refresh_if_stale(dir.path(), &server.uri(), &catalog_client())
            .await
            .unwrap();
        assert!(refreshed, "absent cache must be fetched");

        let cached = load_catalog(dir.path());
        assert!(cached.provider("mystery-provider-9000").is_some());
    }

    #[tokio::test]
    async fn refresh_skips_when_cache_fresh() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let path = catalog_cache_path(dir.path());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, FIXTURE).unwrap(); // just written → fresh

        let refreshed = refresh_if_stale(dir.path(), &server.uri(), &catalog_client())
            .await
            .unwrap();
        assert!(!refreshed, "fresh cache must skip the fetch");
    }

    #[tokio::test]
    async fn refresh_degrades_on_fetch_failure_keeping_existing_cache() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let err = refresh_if_stale(dir.path(), &server.uri(), &catalog_client())
            .await
            .unwrap_err();
        assert!(matches!(err, LensError::Network(_)), "got {err:?}");
        assert!(!catalog_cache_path(dir.path()).exists());
        // The app still has a usable catalog via the bundled fallback.
        assert!(load_catalog(dir.path()).provider("anthropic").is_some());
    }

    #[tokio::test]
    async fn refresh_parse_guard_rejects_corrupt_body_before_cache_write() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{ broken json"))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let err = refresh_if_stale(dir.path(), &server.uri(), &catalog_client())
            .await
            .unwrap_err();
        assert!(matches!(err, LensError::Parse(_)), "got {err:?}");
        assert!(
            !catalog_cache_path(dir.path()).exists(),
            "corrupt body must not be written to the cache"
        );
        assert!(load_catalog(dir.path()).provider("anthropic").is_some());
    }
}
