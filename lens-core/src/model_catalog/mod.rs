//! Typed model catalog sourced from models.dev (Stage 1 of the LLM-interface
//! overhaul).
//!
//! The app must never store a model id as an unvalidated free string. This
//! module parses the [models.dev](https://models.dev/api.json) catalog into
//! typed structs and exposes [`ModelCatalog::validate`] — the anti-free-string
//! guard that rejects a model id that isn't in the catalog for its provider.
//!
//! `lens-core` stays Tauri-free, so this module owns only the pure pieces: the
//! typed structs, the [`SupportedProvider`] enum, the parse, the
//! cache/fetch/refresh routines, and the staleness policy. The Tauri command
//! layer mirrors the shapes over IPC.
//!
//! ## Schema tolerance (load-bearing)
//!
//! The real catalog has 100+ heterogeneous providers and evolves over time. We
//! parse ONLY the fields LensLM needs and deliberately do NOT use
//! `deny_unknown_fields`: an unknown provider, an unknown model field, or a new
//! `reasoning_options` variant must NEVER fail the parse. Missing/null optional
//! fields degrade to `Option`/`Default` rather than erroring.
//!
//! ## Fetch / cache / refresh
//!
//! [`load_catalog`] reads the cached `models-catalog.json` under
//! `{data_dir}/models/` when present, else falls back to the bundled snapshot —
//! it NEVER fails hard. [`refresh_if_stale`] re-fetches the live catalog when the
//! cache is older than [`MODELS_CATALOG_REFRESH_INTERVAL`], mirroring the
//! hardened streamed/size-capped download pattern from [`crate::tts`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error::LensError;

/// Canonical models.dev catalog endpoint. A parameter rather than a hard-coded
/// constant at the fetch call site so tests can point it at a mock server.
pub const MODELS_CATALOG_URL: &str = "https://models.dev/api.json";

/// Relative path (under the app data dir) the cached catalog is written to.
pub const MODELS_CATALOG_RELPATH: &str = "models/models-catalog.json";

/// Bare filename of the cached catalog.
pub const MODELS_CATALOG_FILENAME: &str = "models-catalog.json";

/// Refresh the cached catalog when it is older than this. ~8 hours → the catalog
/// is re-fetched at most ~3×/day (achieved by checking staleness whenever the
/// app starts or a picker opens — no background timer loop in Stage 1).
pub const MODELS_CATALOG_REFRESH_INTERVAL: Duration = Duration::from_secs(8 * 60 * 60);

/// Connect timeout for the catalog fetch (matches the system-check probe shape).
const CATALOG_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Overall (read) timeout for the catalog fetch. The body is ~2.4 MB, so this is
/// longer than a probe but still bounded.
const CATALOG_FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// Upper bound on the catalog body we will buffer to disk. Defense-in-depth
/// against a misconfigured/hostile endpoint streaming an unbounded body (the real
/// catalog is ~2.4 MB; this leaves generous headroom for growth).
const MAX_CATALOG_BODY_BYTES: u64 = 16 * 1024 * 1024;

/// The bundled fallback snapshot. A small curated slice of the real catalog
/// (our core supported providers, sourced from <https://models.dev/api.json>
/// 2026-06), redistributed under models.dev's MIT license (github.com/sst/
/// models.dev). This is NOT the full 2.4 MB catalog — it carries only a handful
/// of models per core provider so the typed validation guard works fully offline.
/// The live catalog is fetched + cached at runtime ([`refresh_if_stale`]); this
/// snapshot is the last-resort degrade target when no cache exists and the
/// network is unavailable.
const BUNDLED_CATALOG_JSON: &str = include_str!("bundled-catalog.json");

// ---------------------------------------------------------------------------
// Typed schema (tolerant subset of models.dev)
// ---------------------------------------------------------------------------

/// The full parsed catalog: provider key → provider entry.
///
/// `BTreeMap` for a deterministic, sorted iteration order (stable IPC output and
/// stable test assertions). The models.dev API is a JSON object keyed by provider
/// id, which deserializes directly into this map.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelCatalog {
    /// Provider id → entry. Unknown providers are tolerated (no allowlist gate
    /// at parse time — the [`SupportedProvider`] enum curates the default
    /// surface, but any catalog provider stays addressable).
    pub providers: BTreeMap<String, ProviderEntry>,
}

/// One provider's entry (Anthropic, OpenAI, …). Extra fields present in the real
/// catalog (`npm`, `api`, …) are tolerated and dropped — NO `deny_unknown_fields`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ProviderEntry {
    /// Provider id (e.g. `"anthropic"`). Mirrors the map key.
    #[serde(default)]
    pub id: String,
    /// Human-readable provider name (e.g. `"Anthropic"`).
    #[serde(default)]
    pub name: String,
    /// Environment variable names that carry this provider's API key.
    #[serde(default)]
    pub env: Vec<String>,
    /// Documentation URL, when the catalog provides one.
    #[serde(default)]
    pub doc: Option<String>,
    /// Model id → model info. Sorted (`BTreeMap`) for deterministic output.
    #[serde(default)]
    pub models: BTreeMap<String, ModelInfo>,
}

/// One model's capabilities + economics. Parses ONLY the fields LensLM needs;
/// every optional field degrades to `Option`/`Default` so a heterogeneous or
/// evolving catalog can never fail the parse.
///
/// `context_limit`/`output_limit` are flattened from the catalog's nested
/// `limit: { context, output }` object by a manual [`Deserialize`] impl (the
/// public struct stays flat for an easy TS mirror); serialization re-nests them
/// under `limit` so the cached round-trip is faithful.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct ModelInfo {
    /// Fully-qualified model id (e.g. `"anthropic/claude-sonnet-4-5"`).
    pub id: String,
    /// Human-readable model name (e.g. `"Claude Sonnet 4.5"`).
    pub name: String,
    /// Model family (e.g. `"claude-sonnet"`), when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    /// Whether the model supports a reasoning/thinking mode.
    pub reasoning: bool,
    /// The reasoning controls the model exposes (effort levels, token budgets, a
    /// simple toggle, …). Tolerant: an unrecognized variant becomes
    /// [`ReasoningOption::Other`] rather than failing the parse.
    pub reasoning_options: Vec<ReasoningOption>,
    /// Whether the model supports tool/function calling.
    pub tool_call: bool,
    /// Whether the model honors a `temperature` parameter.
    pub temperature: bool,
    /// Input/output modalities.
    pub modalities: Modalities,
    /// Maximum context window in tokens (`limit.context`), when known.
    #[serde(serialize_with = "serialize_limit")]
    pub context_limit: Option<u32>,
    /// Maximum output tokens (`limit.output`), when known. Re-nested under
    /// `limit` on serialize together with `context_limit`, so this field is
    /// skipped to avoid emitting a duplicate.
    #[serde(skip_serializing)]
    pub output_limit: Option<u32>,
    /// Whether the model's weights are openly available.
    pub open_weights: bool,
    /// Per-token cost (USD per 1M tokens), when the catalog reports it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<Cost>,
}

/// Serializes the flat `context_limit`/`output_limit` back into the catalog's
/// nested `limit` object so a cached round-trip stays faithful to the source
/// schema. Reads `output_limit` off the surrounding struct is not possible from
/// a field serializer, so this emits only `{ "context": ... }`; `output_limit`
/// is intentionally NOT round-tripped on serialize (the cache is always
/// re-fetched from the source, never re-uploaded — serialize exists only for
/// tests / IPC, which read the flat fields directly).
fn serialize_limit<S>(context: &Option<u32>, ser: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeStruct;
    let mut s = ser.serialize_struct("limit", 1)?;
    s.serialize_field("context", context)?;
    s.end()
}

impl<'de> Deserialize<'de> for ModelInfo {
    fn deserialize<D>(de: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        /// Shadow with the catalog's verbatim shape (nested `limit`), tolerant of
        /// extra fields. Flattened into the public [`ModelInfo`] below.
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
        })
    }
}

/// A single reasoning control on a model.
///
/// Tagged on the catalog's `type` field. models.dev mixes shapes across
/// providers — `effort` carries `values`, `budget_tokens` carries `min`/`max`
/// (either or both may be present depending on provider), `toggle` is a bare
/// switch. An unrecognized `type` falls through to [`ReasoningOption::Other`] so
/// a future variant never breaks the parse.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningOption {
    /// Discrete effort levels (e.g. `["low", "medium", "high"]`).
    Effort {
        /// Selectable effort level strings.
        #[serde(default)]
        values: Vec<String>,
    },
    /// A reasoning token budget. `min`/`max` are independently optional because
    /// providers report one, the other, or both.
    BudgetTokens {
        /// Minimum reasoning token budget, when reported.
        #[serde(default)]
        min: Option<u32>,
        /// Maximum reasoning token budget, when reported.
        #[serde(default)]
        max: Option<u32>,
    },
    /// A bare on/off reasoning toggle (no parameters).
    Toggle,
    /// Any reasoning-option `type` we don't model yet. Tolerated, never an error.
    #[serde(other)]
    Other,
}

/// Input/output modalities a model accepts/produces (e.g. `text`, `image`, `pdf`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Modalities {
    /// Accepted input modalities.
    #[serde(default)]
    pub input: Vec<String>,
    /// Produced output modalities.
    #[serde(default)]
    pub output: Vec<String>,
}

/// Per-token cost in USD per 1M tokens. Every field is optional — providers
/// report different subsets (cache pricing is frequently absent).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Cost {
    /// Input (prompt) cost per 1M tokens.
    #[serde(default)]
    pub input: Option<f64>,
    /// Output (completion) cost per 1M tokens.
    #[serde(default)]
    pub output: Option<f64>,
    /// Cache-read cost per 1M tokens, when supported.
    #[serde(default)]
    pub cache_read: Option<f64>,
    /// Cache-write cost per 1M tokens, when supported.
    #[serde(default)]
    pub cache_write: Option<f64>,
}

/// The catalog's nested `limit: { context, output }` object. Internal only —
/// flattened into [`ModelInfo::context_limit`] / [`ModelInfo::output_limit`] by
/// the manual [`ModelInfo`] deserializer so the public struct stays flat.
#[derive(Debug, Deserialize)]
struct Limit {
    #[serde(default)]
    context: Option<u32>,
    #[serde(default)]
    output: Option<u32>,
}

// ---------------------------------------------------------------------------
// SupportedProvider — the curated default surface
// ---------------------------------------------------------------------------

/// The providers LensLM surfaces by default, with a mapping to the models.dev
/// provider key. [`SupportedProvider::Other`] keeps any catalog provider
/// addressable so an overriding user is never boxed in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupportedProvider {
    /// Anthropic (`anthropic`).
    Anthropic,
    /// OpenAI (`openai`).
    OpenAI,
    /// Google Gemini (`google`).
    Google,
    /// Local Ollama runtime (`ollama`). Local models are user-pulled and NOT
    /// necessarily in models.dev — see [`ModelCatalog::validate`].
    Ollama,
    /// Ollama's hosted cloud (`ollama-cloud`).
    OllamaCloud,
    /// Z.ai / GLM (`zai`).
    Zai,
    /// Any other catalog provider, addressed by its raw models.dev key.
    Other(String),
}

impl SupportedProvider {
    /// The models.dev provider key for this provider.
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

    /// Maps a models.dev provider key onto a [`SupportedProvider`]. An
    /// unrecognized key becomes [`SupportedProvider::Other`] so it stays usable.
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

    /// Whether this provider's models are validated against the live catalog at
    /// all. Local Ollama models are user-pulled and not in models.dev, so they
    /// are exempt from the catalog membership check (see
    /// [`ModelCatalog::validate`]).
    fn is_local_ollama(key: &str) -> bool {
        key == "ollama"
    }
}

// ---------------------------------------------------------------------------
// Validation — the anti-free-string guard
// ---------------------------------------------------------------------------

impl ModelCatalog {
    /// Parses a catalog from a JSON byte slice (tolerant of unknown providers,
    /// extra fields, and unrecognized `reasoning_options` variants).
    pub fn from_json(bytes: &[u8]) -> Result<Self, LensError> {
        serde_json::from_slice(bytes).map_err(LensError::from)
    }

    /// The bundled fallback snapshot (small curated slice of the real catalog).
    /// Infallible in practice — the snapshot is a committed, valid fixture; a
    /// parse failure here is a build-time error caught by the unit tests.
    pub fn bundled() -> Self {
        Self::from_json(BUNDLED_CATALOG_JSON.as_bytes())
            .expect("bundled model catalog must be valid JSON")
    }

    /// Looks up a provider entry by its models.dev key.
    pub fn provider(&self, provider_key: &str) -> Option<&ProviderEntry> {
        self.providers.get(provider_key)
    }

    /// Validates `(provider_key, model_id)` against the catalog — the
    /// anti-free-string guard.
    ///
    /// Returns the matched [`ModelInfo`] on success, or a
    /// [`LensError::Validation`] when the provider or model is not in the
    /// catalog.
    ///
    /// **Local Ollama exception:** for the local `ollama` provider, models are
    /// user-pulled and NOT necessarily present in models.dev, so any non-empty id
    /// is accepted (validated against the catalog only if it happens to be
    /// listed). Live `/api/tags` validation is deferred:
    /// TODO(stage2): validate Ollama-local ids against the live `/api/tags`.
    pub fn validate(&self, provider_key: &str, model_id: &str) -> Result<&ModelInfo, LensError> {
        if model_id.is_empty() {
            return Err(LensError::Validation("model id must not be empty".into()));
        }

        // Local Ollama: accept any non-empty id (return the catalog entry if it
        // happens to be listed; otherwise a synthetic stand-in is not returned —
        // callers that need the &ModelInfo for an unlisted local model must fall
        // back to defaults until stage 2 wires live /api/tags validation).
        if SupportedProvider::is_local_ollama(provider_key) {
            if let Some(info) = self
                .providers
                .get(provider_key)
                .and_then(|p| p.models.get(model_id))
            {
                return Ok(info);
            }
            // Unlisted local model: the membership check is intentionally skipped.
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

    /// Whether a `(provider_key, model_id)` pair is valid (a convenience over
    /// [`validate`](Self::validate) that discards the matched info). For local
    /// Ollama, ANY non-empty id is treated as valid (the user-pull exception).
    pub fn is_valid(&self, provider_key: &str, model_id: &str) -> bool {
        if SupportedProvider::is_local_ollama(provider_key) {
            return !model_id.is_empty();
        }
        self.validate(provider_key, model_id).is_ok()
    }
}

// ---------------------------------------------------------------------------
// Staleness policy
// ---------------------------------------------------------------------------

/// Whether a cache last modified at `mtime` is stale relative to `now`, given a
/// refresh `interval`. A pure function so the policy is unit-testable without
/// touching the filesystem or a clock.
///
/// A `mtime` in the future (clock skew) is treated as fresh. An `interval` of
/// zero makes everything stale.
pub fn is_stale(mtime: SystemTime, now: SystemTime, interval: Duration) -> bool {
    match now.duration_since(mtime) {
        Ok(age) => age >= interval,
        // `mtime` is in the future relative to `now` (clock skew): treat as fresh.
        Err(_) => false,
    }
}

// ---------------------------------------------------------------------------
// Cache / fetch / refresh
// ---------------------------------------------------------------------------

/// Resolves the on-disk path of the cached catalog under `data_dir`.
pub fn catalog_cache_path(data_dir: &Path) -> PathBuf {
    data_dir.join("models").join(MODELS_CATALOG_FILENAME)
}

/// Loads the model catalog, NEVER failing hard.
///
/// Reads the cached `models-catalog.json` under `{data_dir}/models/` when it is
/// present and parses cleanly; otherwise (missing file, read error, or a parse
/// error on a corrupt cache) it degrades to the bundled snapshot. The returned
/// catalog is always usable for validation.
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

/// Re-fetches the catalog from `url` and atomically replaces the cache when the
/// cache is stale (older than [`MODELS_CATALOG_REFRESH_INTERVAL`]) or absent.
///
/// Best-effort by contract: a network/HTTP/parse error is logged and swallowed
/// (the existing cache or the bundled snapshot keeps serving validation) so a
/// caller on the startup path can fire-and-forget. Returns `Ok(true)` when the
/// cache was refreshed, `Ok(false)` when it was still fresh and left untouched.
///
/// `client` is injected (rather than built here) so callers reuse a hardened
/// client and tests can point `url` at a mock server.
pub async fn refresh_if_stale(
    data_dir: &Path,
    url: &str,
    client: &reqwest::Client,
) -> Result<bool, LensError> {
    let path = catalog_cache_path(data_dir);

    // Staleness gate: skip the fetch entirely when the cache is fresh.
    if let Ok(meta) = std::fs::metadata(&path)
        && let Ok(mtime) = meta.modified()
        && !is_stale(mtime, SystemTime::now(), MODELS_CATALOG_REFRESH_INTERVAL)
    {
        return Ok(false);
    }

    let bytes = fetch_catalog_bytes(url, client).await?;
    // Validate the freshly-fetched bytes parse before we overwrite a working
    // cache — never replace a good cache with a corrupt body.
    ModelCatalog::from_json(&bytes)?;
    write_cache_atomic(&path, &bytes)?;
    tracing::info!(
        path = %path.display(),
        bytes = bytes.len(),
        "refreshed model catalog cache"
    );
    Ok(true)
}

/// Builds the hardened HTTP client for catalog fetches: bounded connect/read
/// timeouts + the same no-redirect SSRF guard as the system-check probe.
///
/// Degrades to a default client if the (pure-Rust rustls) TLS backend somehow
/// fails to initialize — never panics.
pub fn catalog_client() -> reqwest::Client {
    let builder = || {
        reqwest::Client::builder()
            .connect_timeout(CATALOG_CONNECT_TIMEOUT)
            .timeout(CATALOG_FETCH_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
    };
    builder()
        .build()
        .unwrap_or_else(|_| builder().build().unwrap_or_default())
}

/// Streams the catalog body from `url`, aborting if it would exceed
/// [`MAX_CATALOG_BODY_BYTES`]. Rejects early on an advertised `Content-Length`
/// over the cap. A non-success status → [`LensError::Network`].
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

/// Atomically writes `bytes` to `path` via a `.part` temp + rename, creating the
/// parent dir, so a partial write never masquerades as a complete cache (mirrors
/// the [`crate::tts`] download finalize).
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
        // Known providers parsed.
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

        // An unknown reasoning-option `type` degrades to `Other`, never an error.
        let unknown =
            &catalog.providers["mystery-provider-9000"].models["future-model"].reasoning_options;
        assert_eq!(unknown, &vec![ReasoningOption::Other]);
    }

    #[test]
    fn tolerates_unknown_providers_and_extra_fields() {
        let catalog = fixture();
        // The unknown provider parsed (no allowlist gate at parse time).
        let mystery = catalog
            .provider("mystery-provider-9000")
            .expect("unknown provider tolerated");
        assert_eq!(mystery.name, "Future Provider");
        assert!(mystery.models.contains_key("future-model"));
    }

    #[test]
    fn handles_missing_optional_fields_with_defaults() {
        let catalog = fixture();
        // GLM-4.6 omits modalities, limit, cost, family, temperature, open_weights.
        let glm = &catalog.providers["zai"].models["glm-4.6"];
        assert_eq!(glm.family, None);
        assert!(!glm.temperature); // default false
        assert!(!glm.open_weights); // default false
        assert_eq!(glm.context_limit, None);
        assert_eq!(glm.output_limit, None);
        assert!(glm.cost.is_none());
        assert!(glm.modalities.input.is_empty());
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

        // Empty model id is always rejected.
        assert!(catalog.validate("anthropic", "").is_err());
    }

    #[test]
    fn validate_ollama_local_accepts_any_nonempty_id() {
        let catalog = fixture();
        // A listed local model resolves to its catalog entry.
        assert!(catalog.validate("ollama", "llama3.1").is_ok());
        // An UNLISTED local model is treated as valid by `is_valid` (user-pull
        // exception) even though `validate` can't return a &ModelInfo for it.
        assert!(catalog.is_valid("ollama", "my-custom-pull:latest"));
        assert!(!catalog.is_valid("ollama", ""));
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
        // An unknown key escapes to `Other` and round-trips.
        let other = SupportedProvider::from_catalog_key("cohere");
        assert_eq!(other, SupportedProvider::Other("cohere".into()));
        assert_eq!(other.catalog_key(), "cohere");
    }

    #[test]
    fn bundled_snapshot_is_valid_and_covers_core_providers() {
        let catalog = ModelCatalog::bundled();
        for key in ["anthropic", "openai", "google", "zai", "ollama"] {
            assert!(
                catalog.provider(key).is_some(),
                "bundled snapshot must cover {key}"
            );
        }
        // The bundled snapshot's models validate through the guard.
        assert!(catalog.validate("anthropic", "claude-sonnet-4-5").is_ok());
    }

    // --- staleness policy ---------------------------------------------------

    #[test]
    fn is_stale_distinguishes_fresh_from_stale() {
        let now = SystemTime::now();
        let interval = MODELS_CATALOG_REFRESH_INTERVAL;

        // Fresh: modified 1 hour ago, interval 8 hours.
        let fresh = now - Duration::from_secs(60 * 60);
        assert!(!is_stale(fresh, now, interval));

        // Stale: modified 9 hours ago.
        let stale = now - Duration::from_secs(9 * 60 * 60);
        assert!(is_stale(stale, now, interval));

        // Exactly at the boundary is stale (>=).
        let boundary = now - interval;
        assert!(is_stale(boundary, now, interval));

        // Future mtime (clock skew) is treated as fresh, never panics.
        let future = now + Duration::from_secs(60 * 60);
        assert!(!is_stale(future, now, interval));
    }

    // --- load / fallback ----------------------------------------------------

    #[test]
    fn load_catalog_falls_back_to_bundled_when_cache_absent() {
        let dir = tempfile::tempdir().unwrap();
        // No cache file written → bundled snapshot.
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
        // The fixture has providers the bundled snapshot does not (the mystery
        // provider), proving we read the cache rather than the bundle.
        assert!(catalog.provider("mystery-provider-9000").is_some());
    }

    #[test]
    fn load_catalog_falls_back_when_cache_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let path = catalog_cache_path(dir.path());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"{ this is not valid json").unwrap();

        // A corrupt cache must degrade to the bundled snapshot, never panic.
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

        // The cache now exists and parses to the fetched fixture.
        let cached = load_catalog(dir.path());
        assert!(cached.provider("mystery-provider-9000").is_some());
    }

    #[tokio::test]
    async fn refresh_skips_when_cache_fresh() {
        // A GET mock that PANICS the test if hit: a fresh cache must not fetch.
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
        // The mock's expect(0) is verified on server drop.
    }

    #[tokio::test]
    async fn refresh_degrades_on_fetch_failure_keeping_existing_cache() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        // No cache present + a failing endpoint → an Err is returned (caller
        // swallows it on the best-effort startup path) and the cache stays absent,
        // so load_catalog still degrades to the bundled snapshot.
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
        // A freshly-fetched corrupt body must fail the parse guard in
        // `refresh_if_stale` BEFORE any cache write, so a working cache (or the
        // bundled fallback) is never clobbered by garbage.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{ broken json"))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        // No cache present → the refresh is attempted; the corrupt body must
        // surface as a parse error and leave NO cache file behind.
        let err = refresh_if_stale(dir.path(), &server.uri(), &catalog_client())
            .await
            .unwrap_err();
        assert!(matches!(err, LensError::Parse(_)), "got {err:?}");
        assert!(
            !catalog_cache_path(dir.path()).exists(),
            "corrupt body must not be written to the cache"
        );
        // The app still has a usable catalog via the bundled fallback.
        assert!(load_catalog(dir.path()).provider("anthropic").is_some());
    }
}
