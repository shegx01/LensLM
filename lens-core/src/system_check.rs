//! First-run system check: honest readiness gates for the local intelligence stack.
//!
//! This module defines the FROZEN IPC contract ([`CheckResult`]) returned by
//! [`crate::LensEngine::run_system_check`], plus the three probes that populate
//! it. The contract is consumed verbatim by the Tauri command layer and mirrored
//! in the Svelte UI; do not reshape it without updating every mirror.
//!
//! Each of the three checks (LLM runtime, embedding model, text-to-speech) is a
//! real readiness GATE: it reports [`CheckStatus::Pass`] only when the subsystem
//! is genuinely usable (a local runtime is reachable OR an equivalent cloud
//! provider is configured), and [`CheckStatus::Fail`] otherwise. The frontend
//! disables "Continue to setup" until all three pass.
//!
//! Probes never surface an expected-absent subsystem as a [`crate::LensError`]:
//! absence is a `Fail` status. `LensError` is reserved for genuinely unexpected
//! failures.

use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::AppConfig;

/// Connect timeout for a single runtime-detection HTTP request.
const PROBE_CONNECT_TIMEOUT: Duration = Duration::from_secs(1);
/// Overall (read) timeout for a single runtime-detection HTTP request.
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
/// Default Ollama base URL when none is configured.
const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434";
/// Default LM Studio OpenAI-compatible base URL.
const DEFAULT_LMSTUDIO_BASE_URL: &str = "http://localhost:1234";
/// Allowlisted embedding model ids the embedding-model gate accepts. Single
/// source of truth: the Tauri install command imports this same slice, and the
/// UI's `EMBEDDING_MODELS` list mirrors it (see the SYNC-CHECK there).
pub const ALLOWED_EMBEDDING_MODELS: &[&str] = &[
    "nomic-embed-text",
    "mxbai-embed-large",
    "all-minilm",
    "bge-m3",
];
/// Upper bound on a probe response body we will buffer + deserialize. A version
/// string or a model list is tiny; this cap (1 MiB) is defense-in-depth so a
/// malicious/misconfigured endpoint can't stream an unbounded body into memory.
const MAX_PROBE_BODY_BYTES: usize = 1024 * 1024;

/// Status of a single system-check row.
///
/// Serializes lowercase: `pass` | `fail`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    /// The subsystem is present and healthy.
    Pass,
    /// The subsystem is expected but absent / unhealthy.
    Fail,
}

/// Stable identifier for each system-check row. Drives UI row ordering/mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckId {
    /// Local LLM runtime (Ollama / LM Studio) or a configured cloud provider.
    LlmRuntime,
    /// Embedding model availability (allowlisted model installed / configured).
    EmbeddingModel,
    /// Text-to-speech: local Kokoro engine on disk or a configured cloud provider.
    TextToSpeech,
}

/// Optional UI affordance attached to a check row.
///
/// Absence of an action is expressed ONLY by `Option::None` on
/// [`CheckResult::action`] — there is deliberately NO `None` variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckAction {
    /// Open configuration for this subsystem (e.g. set an LLM endpoint).
    Configure,
    /// Choose among options (e.g. pick an embedding model).
    Choose,
}

/// One row in the system-check screen.
///
/// THIS IS THE FROZEN IPC CONTRACT. It crosses the Tauri boundary verbatim and
/// is mirrored in the Svelte client; field names and the serde shape are locked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CheckResult {
    /// Stable row identifier.
    pub id: CheckId,
    /// Human-readable row label, e.g. "LLM runtime".
    pub label: String,
    /// Pass / fail.
    pub status: CheckStatus,
    /// Product-facing detail copy. NO internal milestone vocabulary.
    pub detail: String,
    /// Optional UI affordance; absence is `None` (no `CheckAction::None`).
    pub action: Option<CheckAction>,
}

/// Result of probing a single LLM endpoint via [`detect_llm`].
///
/// THIS IS THE FROZEN IPC CONTRACT for the "Configure → Auto-detect" flow.
/// It crosses the Tauri boundary verbatim and is mirrored in the Svelte client;
/// field names and the serde shape are locked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LlmDetection {
    /// Whether the endpoint answered with a successful (2xx) response.
    pub reachable: bool,
    /// Ollama version string, if the endpoint spoke the Ollama protocol.
    pub version: Option<String>,
    /// Model names/ids collected from the endpoint (Ollama + OpenAI-compatible).
    pub models: Vec<String>,
}

/// Probes `base_url` for both Ollama-style and OpenAI-compatible endpoints,
/// returning a [`LlmDetection`] that merges both responses.
///
/// Uses the shared [`probe_client`] (rustls, no-redirect, connect 1s / total 2s).
/// Both protocol probes run concurrently via [`tokio::join!`].
///
/// - Ollama style: `GET {base_url}/api/version` → version; `GET {base_url}/api/tags` → models.
/// - OpenAI-compatible: `GET {base_url}/v1/models` → models from `data[].id`.
/// - Any connect/timeout/non-200 response contributes nothing (not an error).
/// - If neither probe responds: `reachable=false`, `version=None`, `models=[]`.
/// - Never returns `Err` for "not reachable"; `LensError` is reserved for genuine
///   internal faults (which cannot realistically occur here).
pub async fn detect_llm(base_url: &str) -> LlmDetection {
    let base_url = base_url.trim_end_matches('/');

    // Defense-in-depth: only probe http/https schemes. This is a local-first app
    // where the user controls their own machine, so the threat is self-SSRF
    // (a typo or malicious config coaxing the probe into a non-HTTP scheme like
    // file://) rather than a multi-tenant SSRF — hence a scheme allowlist, not a
    // host blocklist. A non-http(s) scheme is reported as simply unreachable.
    let scheme_ok = base_url.split_once("://").is_some_and(|(scheme, _)| {
        scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https")
    });
    if !scheme_ok {
        return LlmDetection {
            reachable: false,
            version: None,
            models: vec![],
        };
    }

    let client = probe_client();

    let (ollama_result, openai_models) = tokio::join!(
        probe_ollama_endpoint(&client, base_url),
        probe_openai_endpoint(&client, base_url),
    );

    let (ollama_version, ollama_models) = ollama_result;

    // Merge + deduplicate models from both protocols.
    let mut models = ollama_models;
    for id in openai_models {
        if !models.contains(&id) {
            models.push(id);
        }
    }

    let reachable = ollama_version.is_some() || !models.is_empty();

    LlmDetection {
        reachable,
        version: ollama_version,
        models,
    }
}

/// Lists the LOCALLY-available Ollama models at `base_url` via `GET /api/tags`
/// (M4 Phase 3, Stage 3 — the per-provider picker for local models).
///
/// models.dev only catalogs CLOUD providers; the user's pulled local models are
/// known only to their running Ollama. This reuses the SAME hardened probe
/// (`probe_ollama_endpoint`) the system-check + `detect_llm` use, so the picker and
/// the readiness gate can never drift on how they speak Ollama.
///
/// **Graceful by contract:** when Ollama is unreachable (not running, wrong URL,
/// non-Ollama endpoint) this returns an EMPTY `Vec` — never an `Err`. The picker
/// surfaces that as a "no local models / Ollama not reachable" state rather than an
/// error toast (the plan's never-error-toast requirement).
pub async fn list_ollama_models(base_url: &str) -> Vec<String> {
    let base_url = base_url.trim_end_matches('/');

    // Scheme allowlist (self-SSRF defense-in-depth), mirroring `detect_llm`.
    let scheme_ok = base_url.split_once("://").is_some_and(|(scheme, _)| {
        scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https")
    });
    if !scheme_ok {
        return Vec::new();
    }

    let client = probe_client();
    let (_version, models) = probe_ollama_endpoint(&client, base_url).await;
    models
}

/// GETs `url` and deserializes a successful (2xx) response as `T`, capping the
/// buffered body at [`MAX_PROBE_BODY_BYTES`] before parsing. Returns `None` on a
/// connect/timeout error, a non-2xx status, an over-cap body, or a parse miss.
///
/// Reading `bytes()` (vs `resp.json()`) lets us reject an oversized body without
/// streaming it all into a `serde` deserializer — defense-in-depth against a
/// malicious endpoint that answers a probe with an unbounded stream.
async fn get_json_capped<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    url: &str,
) -> Option<T> {
    let resp = match client.get(url).send().await {
        Ok(resp) if resp.status().is_success() => resp,
        _ => return None,
    };
    let body = resp.bytes().await.ok()?;
    if body.len() > MAX_PROBE_BODY_BYTES {
        return None;
    }
    serde_json::from_slice::<T>(&body).ok()
}

/// Probes the Ollama protocol: `GET /api/version` then `GET /api/tags`.
/// Returns `(version, model_names)`.
async fn probe_ollama_endpoint(
    client: &reqwest::Client,
    base_url: &str,
) -> (Option<String>, Vec<String>) {
    let version_url = format!("{base_url}/api/version");
    let version = get_json_capped::<OllamaVersion>(client, &version_url)
        .await
        .map(|v| v.version);

    if version.is_none() {
        return (None, vec![]);
    }

    let tags_url = format!("{base_url}/api/tags");
    let models = get_json_capped::<OllamaTags>(client, &tags_url)
        .await
        .map(|tags| tags.models.into_iter().map(|m| m.name).collect())
        .unwrap_or_default();

    (version, models)
}

/// Probes the OpenAI-compatible protocol: `GET /v1/models`.
/// Returns `data[].id` strings on success, empty vec otherwise.
async fn probe_openai_endpoint(client: &reqwest::Client, base_url: &str) -> Vec<String> {
    let url = format!("{base_url}/v1/models");
    get_json_capped::<OpenAiModels>(client, &url)
        .await
        .map(|m| m.data.into_iter().map(|d| d.id).collect())
        .unwrap_or_default()
}

/// Shape of the OpenAI-compatible `GET /v1/models` response.
#[derive(Debug, Deserialize)]
struct OpenAiModels {
    #[serde(default)]
    data: Vec<OpenAiModelEntry>,
}

/// One entry from `GET /v1/models`.
#[derive(Debug, Deserialize)]
struct OpenAiModelEntry {
    id: String,
}

/// Shape of Ollama's `GET /api/version` response.
#[derive(Debug, Deserialize)]
struct OllamaVersion {
    version: String,
}

/// One model entry from Ollama's `GET /api/tags`.
#[derive(Debug, Deserialize)]
struct OllamaTagModel {
    #[serde(default)]
    name: String,
}

/// Shape of Ollama's `GET /api/tags` response.
#[derive(Debug, Deserialize)]
struct OllamaTags {
    #[serde(default)]
    models: Vec<OllamaTagModel>,
}

/// Outcome of probing the local LLM runtime, shared between the LLM-runtime gate
/// and the embedding-model gate so the latter can reuse the Ollama tags fetch.
struct LlmRuntimeProbe {
    /// The completed LLM-runtime check row.
    result: CheckResult,
    /// Whether a local Ollama runtime answered (gates the embedding probe).
    ollama_up: bool,
    /// The Ollama base URL we probed (for the embedding tags fetch).
    ollama_base_url: String,
}

/// Builds a short-timeout HTTP client for runtime detection via the one hardened
/// builder ([`crate::http::hardened_client`]): bounded connect/read timeouts plus
/// SSRF hardening (never follow a redirect — a malicious / misconfigured endpoint
/// could 30x a probe toward an internal host; a probe only ever inspects the
/// directly-addressed service). Centralized so the primary build and its fallback
/// can never drift apart, and so the fallback never degrades to a
/// redirect-following default.
fn probe_client() -> reqwest::Client {
    crate::http::hardened_client(PROBE_CONNECT_TIMEOUT, PROBE_TIMEOUT)
}

/// Resolves the configured Ollama base URL, defaulting to localhost.
///
/// Public so the embedding-model install command can target the SAME runtime
/// the system-check probe detected, rather than re-deriving the URL.
pub fn ollama_base_url(config: &AppConfig) -> String {
    provider_base_url(config, "ollama").unwrap_or_else(|| DEFAULT_OLLAMA_BASE_URL.to_string())
}

/// Resolves the configured LM Studio base URL, defaulting to localhost:1234.
///
/// Mirrors [`ollama_base_url`] so the LM Studio probe target is configurable
/// rather than hard-coded, and so the aggregate fallback can be tested via the
/// seam (point both seams at a mock server).
fn lmstudio_base_url(config: &AppConfig) -> String {
    provider_base_url(config, "lmstudio")
        .or_else(|| provider_base_url(config, "lm_studio"))
        .or_else(|| provider_base_url(config, "lm studio"))
        .unwrap_or_else(|| DEFAULT_LMSTUDIO_BASE_URL.to_string())
}

/// Finds the first configured model for `provider` with a non-empty base URL,
/// returning its trailing-slash-trimmed URL.
fn provider_base_url(config: &AppConfig, provider: &str) -> Option<String> {
    config
        .models
        .iter()
        .find(|m| m.provider.eq_ignore_ascii_case(provider) && !m.base_url.is_empty())
        .map(|m| m.base_url.trim_end_matches('/').to_string())
}

/// Canonical cloud LLM provider ids (real models.dev keys) plus the legacy/custom
/// `openai-compatible` endpoint id. Used by [`has_cloud_llm`] to recognize a
/// configured cloud provider regardless of which first-class card the user picked.
const CLOUD_LLM_PROVIDERS: &[&str] = &[
    "openai",
    "anthropic",
    "google",
    "zai",
    "glm",
    "groq",
    "deepseek",
    "xai",
    "cohere",
    "ollama-cloud",
    "openai-compatible",
];

/// Returns `true` when `config.models` carries a usable cloud LLM entry: a
/// recognized cloud provider (a real models.dev id `openai`/`anthropic`/`google`/
/// `zai`, or a custom `openai-compatible` endpoint) with a non-empty `api_key` AND
/// `model`. This is the cloud arm of the LLM-runtime gate (the local arm is
/// runtime detection).
fn has_cloud_llm(config: &AppConfig) -> bool {
    config.models.iter().any(|m| {
        CLOUD_LLM_PROVIDERS
            .iter()
            .any(|p| m.provider.eq_ignore_ascii_case(p))
            && !m.api_key.is_empty()
            && !m.model.is_empty()
    })
}

/// Detects Ollama via the shared [`probe_ollama_endpoint`] probe, taking just the
/// version (discarding the model list). Returns the parsed version on a 200,
/// `None` on a clean connect/timeout failure or a non-Ollama endpoint.
///
/// Delegates to the single Ollama probe implementation so the runtime-detection
/// row and the `detect_llm` command can never drift in how they speak Ollama.
async fn detect_ollama(client: &reqwest::Client, base_url: &str) -> Option<String> {
    let (version, _models) = probe_ollama_endpoint(client, base_url).await;
    version
}

/// Detects LM Studio via the shared [`probe_openai_endpoint`] probe. Returns
/// `true` when the endpoint advertises at least one model on `/v1/models`.
///
/// Delegates to the single OpenAI-compatible probe implementation so detection
/// behavior stays identical to the `detect_llm` command's OpenAI path.
async fn detect_lmstudio(client: &reqwest::Client, base_url: &str) -> bool {
    !probe_openai_endpoint(client, base_url).await.is_empty()
}

/// Probe 1 — LLM runtime readiness gate.
///
/// PASSES when EITHER a local runtime is reachable OR a cloud provider is
/// configured:
///
/// - local: Ollama (`/api/version`) or LM Studio (`/v1/models`), probed
///   CONCURRENTLY via [`tokio::join!`] so the wall-clock is one timeout window
///   (connect 1s + read 2s), NOT the ~4s of two sequential probes; OR
/// - cloud: a usable `openai-compatible` entry (see [`has_cloud_llm`]).
///
/// Otherwise `Fail`. A clean connect/timeout failure is not a [`crate::LensError`].
async fn probe_llm_runtime(config: &AppConfig) -> LlmRuntimeProbe {
    let client = probe_client();
    let ollama_base = ollama_base_url(config);
    let lmstudio_base = lmstudio_base_url(config);

    let (ollama_version, lmstudio_up) = tokio::join!(
        detect_ollama(&client, &ollama_base),
        detect_lmstudio(&client, &lmstudio_base),
    );

    let ollama_up = ollama_version.is_some();
    let cloud_ok = has_cloud_llm(config);

    let result = match (ollama_up, lmstudio_up, cloud_ok) {
        (true, _, _) | (_, true, _) => CheckResult {
            id: CheckId::LlmRuntime,
            label: "LLM runtime".to_string(),
            status: CheckStatus::Pass,
            detail: "Configure your preferred LLM".to_string(),
            action: Some(CheckAction::Configure),
        },
        (false, false, true) => CheckResult {
            id: CheckId::LlmRuntime,
            label: "LLM runtime".to_string(),
            status: CheckStatus::Pass,
            detail: "Cloud provider configured".to_string(),
            action: Some(CheckAction::Configure),
        },
        (false, false, false) => CheckResult {
            id: CheckId::LlmRuntime,
            label: "LLM runtime".to_string(),
            status: CheckStatus::Fail,
            detail: "No LLM runtime detected or configured".to_string(),
            action: Some(CheckAction::Configure),
        },
    };

    LlmRuntimeProbe {
        result,
        ollama_up,
        ollama_base_url: ollama_base,
    }
}

/// Returns `true` when an Ollama model name matches an allowlisted embedding
/// model (e.g. `"nomic-embed-text:latest"` matches `"nomic-embed-text"`) OR the
/// user's configured `embedding_model`. Matches on the bare name (ignoring an
/// `:tag` suffix), case-insensitively.
fn is_allowlisted_embedding(installed_name: &str, configured: &str) -> bool {
    let bare = installed_name
        .split_once(':')
        .map_or(installed_name, |(name, _tag)| name)
        .to_ascii_lowercase();
    ALLOWED_EMBEDDING_MODELS
        .iter()
        .any(|m| m.eq_ignore_ascii_case(&bare))
        || (!configured.is_empty() && configured.eq_ignore_ascii_case(&bare))
}

/// Returns `true` when the SPECIFIC `model_id`'s fastembed weights are already
/// cached on disk under `{data_dir}/models/fastembed/`.
///
/// R6 (M4 Phase 4b-B, verify-then-implement): the per-model hf-hub cache
/// subdirectory shape was OBSERVED empirically by constructing a real
/// `FastembedEmbedder::new_with_spec` for `all-minilm` into a temp data_dir and
/// walking the tree — it produced
/// `{data_dir}/models/fastembed/models--Qdrant--all-MiniLM-L6-v2-onnx/`
/// (standard `hf-hub` repo cache: `models--{org}--{model}` with `snapshots/`,
/// `blobs/`, `refs/` underneath). The per-model subdir is derived from the
/// registry spec ([`EmbeddingModelSpec::fastembed_cache_subdir`]), so the check
/// is PER-MODEL: a notebook on `mxbai-embed-large` does NOT pass merely because
/// `nomic`'s weights happen to be present.
///
/// Treats the model as "cached" when ITS subdir exists AND contains at least one
/// entry (a non-empty repo dir means the download completed or is in progress —
/// either way construction, not this probe, is the final arbiter). An unknown /
/// empty `model_id` resolves to the default via the registry.
///
/// `pub` (re-exported from `lib.rs`) so the per-model cache state can be surfaced
/// to the frontend (the `fastembed_models_cached` Tauri command) without forcing a
/// duplicate path-derivation that could drift from the gate's own definition of
/// "cached".
pub fn fastembed_weights_cached(data_dir: &Path, model_id: &str) -> bool {
    let spec = crate::embedder::resolve(model_id);
    let model_dir = data_dir
        .join("models")
        .join("fastembed")
        .join(spec.fastembed_cache_subdir());
    if !model_dir.is_dir() {
        return false;
    }
    // Non-empty repo dir → weights present (or mid-download) for THIS model.
    std::fs::read_dir(&model_dir)
        .ok()
        .and_then(|mut d| d.next())
        .is_some()
}

/// Per-backend embedding-readiness gate predicate (M4 Phase 4b-B, R6 / D2).
///
/// PASSES iff the SELECTED backend's own arm is satisfied:
/// `(backend == Fastembed && fastembed_cached) || (backend == Ollama &&
/// ollama_detected)`.
///
/// Pure (no I/O) so the truth table is exhaustively unit-testable. The
/// per-backend shape is the D2 showstopper guard: it NEVER requires Ollama for a
/// fastembed selection (a fresh fastembed-only machine with cached weights passes
/// even when Ollama is unreachable), and NEVER passes a fastembed selection on
/// the strength of an unrelated Ollama tag (or vice-versa). The two facts
/// (`fastembed_cached`, `ollama_detected`) are computed by the caller and fed in.
fn embedding_gate_passes(
    backend: crate::embedder::EmbeddingBackend,
    fastembed_cached: bool,
    ollama_detected: bool,
) -> bool {
    match backend {
        crate::embedder::EmbeddingBackend::Fastembed => fastembed_cached,
        crate::embedder::EmbeddingBackend::Ollama => ollama_detected,
    }
}

/// Probe 2 — embedding-model readiness gate (M4 Phase 4b-B: per-backend OR-gate).
///
/// Resolves the configured backend ([`AppConfig::embedding_backend`], empty →
/// the `fastembed` default) and PASSES iff that backend's own readiness arm is
/// satisfied ([`embedding_gate_passes`]):
/// - [`Fastembed`](crate::embedder::EmbeddingBackend::Fastembed): the SELECTED
///   model's weights are cached on disk under `{data_dir}/models/fastembed/`
///   ([`fastembed_weights_cached`], per-model). Ollama is NOT probed for a
///   fastembed selection — a fresh fastembed-only install passes with Ollama
///   unreachable (D2).
/// - [`Ollama`](crate::embedder::EmbeddingBackend::Ollama): Ollama is up and an
///   allowlisted/selected model is installed via `GET /api/tags`.
///
/// `Fail`s only when the selected backend's arm is unsatisfied, with a `Choose`
/// affordance.
async fn probe_embedding_model(
    client: &reqwest::Client,
    runtime: &LlmRuntimeProbe,
    config: &AppConfig,
    data_dir: &Path,
) -> CheckResult {
    let label = "Embedding model".to_string();
    let fail = || CheckResult {
        id: CheckId::EmbeddingModel,
        label: label.clone(),
        status: CheckStatus::Fail,
        detail: "No embedding model installed".to_string(),
        action: Some(CheckAction::Choose),
    };
    let pass = || CheckResult {
        id: CheckId::EmbeddingModel,
        label: label.clone(),
        status: CheckStatus::Pass,
        detail: "Embedding model installed".to_string(),
        action: Some(CheckAction::Choose),
    };

    let backend = crate::embedder::EmbeddingBackend::from_opt_str(Some(&config.embedding_backend));

    let fastembed_cached = fastembed_weights_cached(data_dir, &config.embedding_model);

    // Only probe Ollama when the selected backend is Ollama AND the runtime is up
    // — a fastembed selection must NEVER depend on (or wait on) Ollama (D2).
    let ollama_detected =
        if matches!(backend, crate::embedder::EmbeddingBackend::Ollama) && runtime.ollama_up {
            let url = format!("{}/api/tags", runtime.ollama_base_url);
            get_json_capped::<OllamaTags>(client, &url)
                .await
                .map(|tags| {
                    tags.models
                        .iter()
                        .any(|m| is_allowlisted_embedding(&m.name, &config.embedding_model))
                })
                .unwrap_or(false)
        } else {
            false
        };

    let found = embedding_gate_passes(backend, fastembed_cached, ollama_detected);

    if found { pass() } else { fail() }
}

/// Returns `true` when a usable cloud TTS provider is configured: ElevenLabs
/// with a non-empty `api_key`. This is the cloud arm of the TTS gate (the local
/// arm is the Kokoro-model-on-disk check).
fn has_cloud_tts(config: &AppConfig) -> bool {
    config.tts.provider.eq_ignore_ascii_case("elevenlabs") && !config.tts.api_key.is_empty()
}

/// Probe 3 — text-to-speech readiness gate.
///
/// PASSES when the user has genuinely COMPLETED TTS setup, via EITHER arm:
///
/// - local: the Kokoro ONNX model is on disk at
///   `{data_dir}/models/kokoro/model_q8f16.onnx` (the exact path the downloader
///   writes) AND the user has saved both a host and a guest voice
///   (`config.voices.host` / `config.voices.guest` non-empty). The model file
///   alone is NOT enough — downloading the engine without choosing voices leaves
///   TTS unconfigured, so the gate must still `Fail`; OR
/// - cloud: a cloud TTS provider is configured (see [`has_cloud_tts`]).
///
/// Otherwise `Fail` with a `Choose` affordance.
fn probe_text_to_speech(config: &AppConfig) -> CheckResult {
    let model_path = crate::tts::kokoro_model_path(Path::new(&config.paths.data_dir));
    let kokoro_on_disk = model_path.is_file();
    let voices_set = !config.voices.host.is_empty() && !config.voices.guest.is_empty();

    let (status, detail) = if kokoro_on_disk && voices_set {
        (CheckStatus::Pass, "Kokoro audio engine ready".to_string())
    } else if has_cloud_tts(config) {
        (CheckStatus::Pass, "Cloud voice configured".to_string())
    } else {
        (
            CheckStatus::Fail,
            "Download the engine and choose voices, or connect a cloud provider".to_string(),
        )
    };

    CheckResult {
        id: CheckId::TextToSpeech,
        label: "Text-to-speech".to_string(),
        status,
        detail,
        action: Some(CheckAction::Choose),
    }
}

/// Runs the three system-check probes and returns them in the fixed row order:
/// LlmRuntime, EmbeddingModel, TextToSpeech.
///
/// The probes run SEQUENTIALLY here: the LLM-runtime probe first (it concurrently
/// probes Ollama + LM Studio internally), then the embedding-model probe — which
/// REUSES the LLM probe's Ollama outcome (`ollama_up` + base URL), so it must run
/// after it — and finally the synchronous text-to-speech probe (a filesystem +
/// config check, no I/O). The dominant cost is the single bounded LLM timeout
/// window.
///
/// Takes a `&AppConfig` and `data_dir` — the caller clones the config cheaply
/// under the engine read guard and DROPS the guard before calling here, so the
/// multi-second HTTP probes never hold the engine lock (which would block
/// concurrent `get_config`/`set_config`). `data_dir` is used by the
/// fastembed-cache arm of the embedding-model gate (R6).
pub(crate) async fn run_system_check(config: &AppConfig, data_dir: &Path) -> Vec<CheckResult> {
    let embed_client = probe_client();

    // The embedding probe reuses the LLM-runtime outcome, so it is awaited after
    // the LLM probe within this future.
    let runtime = probe_llm_runtime(config).await;
    let embedding_model = probe_embedding_model(&embed_client, &runtime, config, data_dir).await;

    vec![
        runtime.result,
        embedding_model,
        probe_text_to_speech(config),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    use crate::config::ModelConfig;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config_with_ollama(base_url: &str) -> AppConfig {
        AppConfig {
            models: vec![ModelConfig {
                provider: "ollama".to_string(),
                base_url: base_url.to_string(),
                ..ModelConfig::default()
            }],
            ..AppConfig::default()
        }
    }

    /// Builds a config carrying both an Ollama and an LM Studio model entry so
    /// both probe seams can be pointed at mock servers (or dead URLs).
    fn config_with_runtimes(ollama_url: &str, lmstudio_url: &str) -> AppConfig {
        AppConfig {
            models: vec![
                ModelConfig {
                    provider: "ollama".to_string(),
                    base_url: ollama_url.to_string(),
                    ..ModelConfig::default()
                },
                ModelConfig {
                    provider: "lmstudio".to_string(),
                    base_url: lmstudio_url.to_string(),
                    ..ModelConfig::default()
                },
            ],
            ..AppConfig::default()
        }
    }

    #[test]
    fn lmstudio_base_url_defaults_then_reads_config() {
        // No lmstudio entry ⇒ the default seam.
        assert_eq!(
            lmstudio_base_url(&AppConfig::default()),
            DEFAULT_LMSTUDIO_BASE_URL
        );
        // A configured entry wins, trailing slash trimmed.
        let cfg = config_with_runtimes("", "http://127.0.0.1:9999/");
        assert_eq!(lmstudio_base_url(&cfg), "http://127.0.0.1:9999");
    }

    #[tokio::test]
    async fn aggregate_falls_back_to_lmstudio_via_seam() {
        // Ollama is down, but a configured LM Studio seam answers 200 on
        // /v1/models ⇒ the aggregate LLM probe reports Pass via the fallback.
        // Use a fixed always-refused port (1) rather than a dropped MockServer's
        // port — the freed port can be rebound by a parallel test's Ollama mock
        // and serve /api/version, flaking the `!ollama_up` assertion on CI.
        let dead_ollama = "http://127.0.0.1:1".to_string();

        let lmstudio = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "id": "local-model" }]
            })))
            .mount(&lmstudio)
            .await;

        let config = config_with_runtimes(&dead_ollama, &lmstudio.uri());
        let probe = probe_llm_runtime(&config).await;

        assert_eq!(probe.result.status, CheckStatus::Pass);
        assert!(!probe.ollama_up);
        assert_eq!(probe.result.detail, "Configure your preferred LLM");
        assert_eq!(probe.result.action, Some(CheckAction::Configure));
    }

    #[tokio::test]
    async fn llm_runtime_pass_when_ollama_responds() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/version"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "version": "0.3.2"
            })))
            .mount(&server)
            .await;

        let config = config_with_ollama(&server.uri());
        let probe = probe_llm_runtime(&config).await;

        assert_eq!(probe.result.status, CheckStatus::Pass);
        assert!(probe.ollama_up);
        assert_eq!(probe.result.detail, "Configure your preferred LLM");
    }

    #[tokio::test]
    async fn llm_runtime_fail_when_nothing_responds() {
        // Both runtimes pointed at a fixed, always-refused localhost port (1).
        // A dropped-MockServer port is NOT safe: cargo runs tests in parallel, so
        // the freed ephemeral port can be re-bound by a concurrent test's mock
        // server, which then answers 200 and flips this to Pass (observed in CI).
        // Nothing binds 127.0.0.1:1, so the connection is deterministically
        // refused → both runtimes absent + no cloud configured → Fail.
        let dead_url = "http://127.0.0.1:1";
        let config = config_with_runtimes(dead_url, dead_url);
        let probe = probe_llm_runtime(&config).await;

        assert_eq!(probe.result.status, CheckStatus::Fail);
        assert!(!probe.ollama_up);
        assert_eq!(probe.result.detail, "No LLM runtime detected or configured");
        assert_eq!(probe.result.action, Some(CheckAction::Configure));
    }

    #[tokio::test]
    async fn llm_runtime_pass_when_cloud_provider_configured() {
        // No local runtime reachable, but a usable openai-compatible cloud entry
        // (provider + api_key + model) satisfies the gate.
        let dead_url = "http://127.0.0.1:1";
        let mut config = config_with_runtimes(dead_url, dead_url);
        config.models.push(ModelConfig {
            provider: "openai-compatible".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-4o".to_string(),
            api_key: "sk-cloud".to_string(),
            ..ModelConfig::default()
        });

        let probe = probe_llm_runtime(&config).await;
        assert_eq!(probe.result.status, CheckStatus::Pass);
        assert_eq!(probe.result.detail, "Cloud provider configured");
    }

    #[test]
    fn has_cloud_llm_requires_key_and_model() {
        // Missing api_key ⇒ not usable.
        let mut config = AppConfig::default();
        config.models.push(ModelConfig {
            provider: "openai-compatible".to_string(),
            model: "gpt-4o".to_string(),
            ..ModelConfig::default()
        });
        assert!(!has_cloud_llm(&config));

        // Missing model ⇒ not usable.
        config.models[0].model = String::new();
        config.models[0].api_key = "sk-cloud".to_string();
        assert!(!has_cloud_llm(&config));

        // Both present ⇒ usable.
        config.models[0].model = "gpt-4o".to_string();
        assert!(has_cloud_llm(&config));
    }

    #[test]
    fn has_cloud_llm_recognizes_first_class_cloud_providers() {
        // Fix #1: a first-class cloud entry now carries its REAL provider id, not a
        // blanket 'openai-compatible'. The LLM-runtime cloud arm must still
        // recognize it (else a configured Anthropic/Google cloud provider would
        // fail the readiness gate).
        for provider in ["anthropic", "google", "openai", "zai"] {
            let mut config = AppConfig::default();
            config.models.push(ModelConfig {
                provider: provider.to_string(),
                model: "some-model".to_string(),
                api_key: "k".to_string(),
                ..ModelConfig::default()
            });
            assert!(
                has_cloud_llm(&config),
                "{provider} cloud entry must be recognized"
            );
        }
    }

    #[test]
    fn has_cloud_tts_requires_elevenlabs_and_key() {
        // Empty api_key ⇒ not usable, even with the right provider.
        let mut config = AppConfig::default();
        config.tts = crate::config::TtsConfig {
            provider: "elevenlabs".to_string(),
            api_key: String::new(),
        };
        assert!(!has_cloud_tts(&config));

        // Wrong provider + a valid key ⇒ not usable.
        config.tts = crate::config::TtsConfig {
            provider: "openai".to_string(),
            api_key: "sk-key".to_string(),
        };
        assert!(!has_cloud_tts(&config));

        // Mixed-case "ElevenLabs" + key ⇒ usable (provider match is case-insensitive).
        config.tts = crate::config::TtsConfig {
            provider: "ElevenLabs".to_string(),
            api_key: "sk-key".to_string(),
        };
        assert!(has_cloud_tts(&config));

        // Canonical "elevenlabs" + key ⇒ usable.
        config.tts.provider = "elevenlabs".to_string();
        assert!(has_cloud_tts(&config));
    }

    #[tokio::test]
    async fn llm_runtime_falls_back_to_lmstudio() {
        // Ollama path 404s (present server, wrong endpoint ⇒ no version), but the
        // LM Studio endpoint answers 200. Because the LM Studio probe targets a
        // fixed default port, this asserts the fallback branch via the server
        // responding 200 on /v1/models while /api/version is absent.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "id": "local-model" }]
            })))
            .mount(&server)
            .await;
        // No /api/version mock ⇒ wiremock returns 404 ⇒ Ollama treated as absent.

        // Point the (fixed-port) LM Studio detector at this server by detecting
        // directly; the aggregate probe uses the default port, so we assert the
        // building blocks compose into the fallback Pass.
        let client = probe_client();
        let ollama = detect_ollama(&client, &server.uri()).await;
        let lmstudio = detect_lmstudio(&client, &server.uri()).await;

        assert!(ollama.is_none(), "no /api/version ⇒ Ollama absent");
        assert!(lmstudio, "LM Studio /v1/models answered 200");
    }

    #[tokio::test]
    async fn llm_probe_stays_within_time_budget_when_offline() {
        // Fixed always-refused port (avoids the parallel-test port-reuse race; see
        // llm_runtime_fail_when_nothing_responds). Genuinely offline; never reaches
        // a real LM Studio on :1234.
        let dead_url = "http://127.0.0.1:1";
        let config = config_with_runtimes(dead_url, dead_url);
        let start = Instant::now();
        let _ = probe_llm_runtime(&config).await;
        let elapsed = start.elapsed();

        // Concurrent (not sequential) probing keeps the wall-clock to roughly
        // ONE timeout window: PROBE_CONNECT_TIMEOUT (1s) + PROBE_TIMEOUT (2s) =
        // 3s for the slowest single probe, NOT the ~6s of two sequential ones.
        // The 3500ms budget is that 3s window plus 500ms of slack for CI
        // scheduling jitter; bump it only if those two constants change.
        assert!(
            elapsed < Duration::from_millis(3_500),
            "llm probe took {elapsed:?}, exceeding the concurrent budget"
        );
    }

    #[tokio::test]
    async fn embedding_pass_when_allowlisted_model_present() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [{ "name": "nomic-embed-text:latest" }]
            })))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let client = probe_client();
        let runtime = LlmRuntimeProbe {
            result: llm_runtime_placeholder(),
            ollama_up: true,
            ollama_base_url: server.uri(),
        };
        // Ollama-backend selection + an allowlisted Ollama model present → Pass.
        let config = AppConfig {
            embedding_backend: "ollama".to_string(),
            ..AppConfig::default()
        };
        let result = probe_embedding_model(&client, &runtime, &config, dir.path()).await;

        assert_eq!(result.status, CheckStatus::Pass);
        assert_eq!(result.action, Some(CheckAction::Choose));
        assert_eq!(result.detail, "Embedding model installed");
    }

    #[tokio::test]
    async fn embedding_fail_when_no_allowlisted_model() {
        // An installed-but-not-allowlisted model (e.g. a chat model) does NOT
        // satisfy the gate.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [{ "name": "llama3:latest" }]
            })))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let client = probe_client();
        let runtime = LlmRuntimeProbe {
            result: llm_runtime_placeholder(),
            ollama_up: true,
            ollama_base_url: server.uri(),
        };
        // Ollama-backend selection but only a non-allowlisted (chat) model → Fail.
        let config = AppConfig {
            embedding_backend: "ollama".to_string(),
            ..AppConfig::default()
        };
        let result = probe_embedding_model(&client, &runtime, &config, dir.path()).await;

        assert_eq!(result.status, CheckStatus::Fail);
        assert_eq!(result.action, Some(CheckAction::Choose));
        assert_eq!(result.detail, "No embedding model installed");
    }

    #[tokio::test]
    async fn embedding_fail_when_ollama_down() {
        let dir = tempfile::tempdir().unwrap();
        let client = probe_client();
        let runtime = LlmRuntimeProbe {
            result: llm_runtime_placeholder(),
            ollama_up: false,
            ollama_base_url: DEFAULT_OLLAMA_BASE_URL.to_string(),
        };
        // No fastembed cache + Ollama down → Fail.
        let result =
            probe_embedding_model(&client, &runtime, &AppConfig::default(), dir.path()).await;

        assert_eq!(result.status, CheckStatus::Fail);
    }

    /// Creates a non-empty per-model fastembed cache subdir for `model_id`
    /// (OBSERVED hf-hub shape `models/fastembed/models--{org}--{model}/…`).
    fn seed_fastembed_cache(data_dir: &Path, model_id: &str) {
        let subdir = crate::embedder::resolve(model_id).fastembed_cache_subdir();
        let model_dir = data_dir.join("models").join("fastembed").join(subdir);
        std::fs::create_dir_all(model_dir.join("snapshots")).unwrap();
        std::fs::write(model_dir.join("snapshots").join("model.onnx"), b"fake").unwrap();
    }

    #[tokio::test]
    async fn embedding_pass_when_fastembed_weights_cached_ollama_down() {
        // R6: the SELECTED model's fastembed weights present + Ollama unreachable
        // → PASS (fastembed-backend default).
        let dir = tempfile::tempdir().unwrap();
        seed_fastembed_cache(dir.path(), "nomic-embed-text-v1.5");

        let client = probe_client();
        let runtime = LlmRuntimeProbe {
            result: llm_runtime_placeholder(),
            ollama_up: false,
            ollama_base_url: "http://127.0.0.1:1".to_string(),
        };
        let result =
            probe_embedding_model(&client, &runtime, &AppConfig::default(), dir.path()).await;

        assert_eq!(result.status, CheckStatus::Pass);
        assert_eq!(result.detail, "Embedding model installed");
    }

    #[tokio::test]
    async fn embedding_fail_when_no_cache_and_ollama_down() {
        // No fastembed cache AND Ollama down → Fail.
        let dir = tempfile::tempdir().unwrap();
        let client = probe_client();
        let runtime = LlmRuntimeProbe {
            result: llm_runtime_placeholder(),
            ollama_up: false,
            ollama_base_url: "http://127.0.0.1:1".to_string(),
        };
        let result =
            probe_embedding_model(&client, &runtime, &AppConfig::default(), dir.path()).await;

        assert_eq!(result.status, CheckStatus::Fail);
    }

    // --- Step 5 (4b-B): per-model fastembed cache check (R6) ---

    /// `fastembed_weights_cached` is PER-MODEL: caching nomic's weights does NOT
    /// make the mxbai model report cached, and vice-versa.
    #[test]
    fn fastembed_weights_cached_is_per_model() {
        let dir = tempfile::tempdir().unwrap();
        // Only the nomic subdir is present.
        seed_fastembed_cache(dir.path(), "nomic-embed-text-v1.5");

        // The SAME model that was seeded → cached.
        assert!(fastembed_weights_cached(
            dir.path(),
            "nomic-embed-text-v1.5"
        ));
        // A DIFFERENT model whose subdir is absent → NOT cached.
        assert!(!fastembed_weights_cached(dir.path(), "mxbai-embed-large"));

        // Now seed mxbai too → both report cached.
        seed_fastembed_cache(dir.path(), "mxbai-embed-large");
        assert!(fastembed_weights_cached(dir.path(), "mxbai-embed-large"));
        assert!(fastembed_weights_cached(
            dir.path(),
            "nomic-embed-text-v1.5"
        ));
    }

    /// An empty model subdir (created but no files) is NOT cached.
    #[test]
    fn fastembed_weights_cached_false_for_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let subdir = crate::embedder::resolve("nomic-embed-text-v1.5").fastembed_cache_subdir();
        std::fs::create_dir_all(dir.path().join("models").join("fastembed").join(subdir)).unwrap();
        assert!(!fastembed_weights_cached(
            dir.path(),
            "nomic-embed-text-v1.5"
        ));
    }

    /// The OBSERVED hf-hub subdir literal is the `models--{org}--{model}` form,
    /// pinning the empirically-observed shape (Step 5 R6 protocol).
    #[test]
    fn fastembed_cache_subdir_matches_observed_shape() {
        assert_eq!(
            crate::embedder::resolve("all-minilm").fastembed_cache_subdir(),
            "models--Qdrant--all-MiniLM-L6-v2-onnx"
        );
        assert_eq!(
            crate::embedder::resolve("nomic-embed-text-v1.5").fastembed_cache_subdir(),
            "models--nomic-ai--nomic-embed-text-v1.5"
        );
    }

    // --- Step 5 (4b-B): per-backend OR-gate predicate truth table ---

    #[test]
    fn gate_predicate_truth_table() {
        use crate::embedder::EmbeddingBackend::{Fastembed, Ollama};
        // Fastembed selection: passes IFF fastembed weights are cached; the
        // ollama_detected flag is irrelevant (never required for fastembed).
        assert!(embedding_gate_passes(Fastembed, true, false));
        assert!(embedding_gate_passes(Fastembed, true, true));
        assert!(!embedding_gate_passes(Fastembed, false, false));
        assert!(
            !embedding_gate_passes(Fastembed, false, true),
            "a fastembed selection must NEVER pass on the strength of an Ollama tag"
        );
        // Ollama selection: passes IFF an Ollama tag is detected; fastembed cache
        // is irrelevant.
        assert!(embedding_gate_passes(Ollama, false, true));
        assert!(embedding_gate_passes(Ollama, true, true));
        assert!(!embedding_gate_passes(Ollama, false, false));
        assert!(
            !embedding_gate_passes(Ollama, true, false),
            "an ollama selection must NEVER pass on the strength of a fastembed cache"
        );
    }

    /// THE D2 showstopper guard: a fresh fastembed-only install — fastembed
    /// selected (default config), the selected model's weights cached, Ollama
    /// UNREACHABLE — PASSES the gate. A wrong fix that ANDs Ollama or probes it
    /// for a fastembed selection would dead-end onboarding here.
    #[tokio::test]
    async fn fresh_install_fastembed_only_passes_gate() {
        let dir = tempfile::tempdir().unwrap();
        seed_fastembed_cache(dir.path(), "nomic-embed-text-v1.5");

        let client = probe_client();
        // Ollama is DOWN and points at a dead loopback port.
        let runtime = LlmRuntimeProbe {
            result: llm_runtime_placeholder(),
            ollama_up: false,
            ollama_base_url: "http://127.0.0.1:1".to_string(),
        };
        // Default config: empty embedding_backend → resolves to Fastembed.
        let result =
            probe_embedding_model(&client, &runtime, &AppConfig::default(), dir.path()).await;
        assert_eq!(
            result.status,
            CheckStatus::Pass,
            "a fresh fastembed-only install with cached weights must pass with Ollama unreachable"
        );
    }

    #[test]
    fn allowlisted_embedding_matches_bare_name_and_config() {
        // Tagged allowlist name matches its bare form.
        assert!(is_allowlisted_embedding("nomic-embed-text:latest", ""));
        assert!(is_allowlisted_embedding("BGE-M3", ""));
        // A non-allowlisted name only matches when it equals the configured id.
        assert!(!is_allowlisted_embedding("my-custom-embed:latest", ""));
        assert!(is_allowlisted_embedding(
            "my-custom-embed:latest",
            "my-custom-embed"
        ));
        // A non-embed chat model never matches.
        assert!(!is_allowlisted_embedding("llama3:latest", ""));
    }

    /// Test helper: materialize the Kokoro model file at the exact path the
    /// downloader writes, under `data_dir`.
    fn write_kokoro_model(data_dir: &Path) {
        let model_path = crate::tts::kokoro_model_path(data_dir);
        std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        std::fs::write(&model_path, b"fake-onnx-bytes").unwrap();
    }

    #[test]
    fn text_to_speech_fail_when_nothing_configured() {
        // A fresh data dir has no Kokoro model on disk + no cloud TTS ⇒ Fail.
        let dir = tempfile::tempdir().unwrap();
        let mut config = AppConfig::default();
        config.paths.data_dir = dir.path().display().to_string();

        let result = probe_text_to_speech(&config);
        assert_eq!(result.id, CheckId::TextToSpeech);
        assert_eq!(result.status, CheckStatus::Fail);
        assert_eq!(result.action, Some(CheckAction::Choose));
        assert_eq!(
            result.detail,
            "Download the engine and choose voices, or connect a cloud provider"
        );
    }

    #[test]
    fn text_to_speech_fail_when_model_present_but_voices_empty() {
        // The engine file alone does NOT satisfy the gate: without host/guest
        // voices the user hasn't completed TTS setup, so it must Fail. This is
        // the readiness-gate tightening — downloading without choosing voices.
        let dir = tempfile::tempdir().unwrap();
        write_kokoro_model(dir.path());

        let mut config = AppConfig::default();
        config.paths.data_dir = dir.path().display().to_string();
        // config.voices is empty by default.

        let result = probe_text_to_speech(&config);
        assert_eq!(result.status, CheckStatus::Fail);
        assert_eq!(
            result.detail,
            "Download the engine and choose voices, or connect a cloud provider"
        );
    }

    #[test]
    fn text_to_speech_pass_when_model_present_and_voices_set() {
        // Engine on disk AND both voices saved ⇒ local TTS is genuinely ready.
        let dir = tempfile::tempdir().unwrap();
        write_kokoro_model(dir.path());

        let mut config = AppConfig::default();
        config.paths.data_dir = dir.path().display().to_string();
        config.voices = crate::config::VoiceConfig {
            host: "am_michael".to_string(),
            guest: "af_heart".to_string(),
        };

        let result = probe_text_to_speech(&config);
        assert_eq!(result.status, CheckStatus::Pass);
        assert_eq!(result.detail, "Kokoro audio engine ready");
    }

    #[test]
    fn text_to_speech_fail_when_only_one_voice_set() {
        // Engine on disk but only the host voice saved (guest empty) ⇒ Fail.
        // Guards the AND-conjunction: a single voice must NOT satisfy the gate.
        let dir = tempfile::tempdir().unwrap();
        write_kokoro_model(dir.path());

        let mut config = AppConfig::default();
        config.paths.data_dir = dir.path().display().to_string();
        config.voices = crate::config::VoiceConfig {
            host: "am_michael".to_string(),
            guest: String::new(),
        };

        let result = probe_text_to_speech(&config);
        assert_eq!(result.status, CheckStatus::Fail);
    }

    #[test]
    fn text_to_speech_fail_when_voices_set_but_model_absent() {
        // Voices saved but the engine was never downloaded ⇒ still Fail.
        let dir = tempfile::tempdir().unwrap();
        let mut config = AppConfig::default();
        config.paths.data_dir = dir.path().display().to_string();
        config.voices = crate::config::VoiceConfig {
            host: "am_michael".to_string(),
            guest: "af_heart".to_string(),
        };

        let result = probe_text_to_speech(&config);
        assert_eq!(result.status, CheckStatus::Fail);
    }

    #[test]
    fn text_to_speech_pass_when_cloud_configured() {
        // No local model on disk, but an ElevenLabs cloud key satisfies the gate.
        let dir = tempfile::tempdir().unwrap();
        let mut config = AppConfig::default();
        config.paths.data_dir = dir.path().display().to_string();
        config.tts = crate::config::TtsConfig {
            provider: "elevenlabs".to_string(),
            api_key: "sk-elevenlabs".to_string(),
        };

        let result = probe_text_to_speech(&config);
        assert_eq!(result.status, CheckStatus::Pass);
        assert_eq!(result.detail, "Cloud voice configured");
    }

    #[tokio::test]
    async fn run_system_check_returns_three_rows_in_order() {
        let engine = crate::LensEngine::for_test().await;
        let dir = tempfile::tempdir().unwrap();
        {
            let mut guard = engine.write().await;
            guard.config.paths.data_dir = dir.path().display().to_string();
        }
        let results = engine.run_system_check().await.unwrap();

        let ids: Vec<CheckId> = results.iter().map(|r| r.id).collect();
        assert_eq!(
            ids,
            vec![
                CheckId::LlmRuntime,
                CheckId::EmbeddingModel,
                CheckId::TextToSpeech,
            ]
        );
    }

    /// Snapshot the exact serde wire-format of `CheckResult`. Locks the FROZEN
    /// IPC contract: snake_case fields, lowercase status, `action` omitted/`None`.
    #[test]
    fn check_result_serialized_shape() {
        let result = CheckResult {
            id: CheckId::LlmRuntime,
            label: "LLM runtime".to_string(),
            status: CheckStatus::Fail,
            detail: "No local LLM runtime detected".to_string(),
            action: Some(CheckAction::Configure),
        };
        insta::assert_json_snapshot!(result, @r#"
        {
          "id": "llm_runtime",
          "label": "LLM runtime",
          "status": "fail",
          "detail": "No local LLM runtime detected",
          "action": "configure"
        }
        "#);
    }

    /// `action: None` serializes as JSON `null` (the only way to express "no
    /// action" — there is no `CheckAction::None` variant).
    #[test]
    fn check_result_no_action_serializes_null() {
        let result = CheckResult {
            id: CheckId::EmbeddingModel,
            label: "Embedding model".to_string(),
            status: CheckStatus::Fail,
            detail: "Built-in".to_string(),
            action: None,
        };
        insta::assert_json_snapshot!(result, @r#"
        {
          "id": "embedding_model",
          "label": "Embedding model",
          "status": "fail",
          "detail": "Built-in",
          "action": null
        }
        "#);
    }

    /// Test helper: a throwaway `CheckResult` to fill the unused `result` field
    /// of an `LlmRuntimeProbe` fixture in the embedding-probe tests.
    fn llm_runtime_placeholder() -> CheckResult {
        CheckResult {
            id: CheckId::LlmRuntime,
            label: "LLM runtime".to_string(),
            status: CheckStatus::Pass,
            detail: "fixture".to_string(),
            action: None,
        }
    }

    // --- detect_llm tests ---

    #[tokio::test]
    async fn detect_llm_ollama_responds_version_and_tags() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/version"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "version": "0.4.1"
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [
                    { "name": "llama3:latest" },
                    { "name": "nomic-embed-text:latest" }
                ]
            })))
            .mount(&server)
            .await;

        let result = detect_llm(&server.uri()).await;

        assert!(result.reachable);
        assert_eq!(result.version, Some("0.4.1".to_string()));
        assert_eq!(
            result.models,
            vec!["llama3:latest", "nomic-embed-text:latest"]
        );
    }

    #[tokio::test]
    async fn detect_llm_openai_compatible_only() {
        let server = MockServer::start().await;
        // No /api/version or /api/tags — only OpenAI-compatible /v1/models.
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    { "id": "mistral-7b" },
                    { "id": "codellama-13b" }
                ]
            })))
            .mount(&server)
            .await;

        let result = detect_llm(&server.uri()).await;

        assert!(result.reachable);
        assert_eq!(result.version, None);
        assert_eq!(result.models, vec!["mistral-7b", "codellama-13b"]);
    }

    #[tokio::test]
    async fn detect_llm_nothing_responds_returns_unreachable() {
        // Fixed always-refused port — avoids the parallel-test port-reuse race
        // (see llm_runtime_fail_when_nothing_responds for the full rationale).
        let result = detect_llm("http://127.0.0.1:1").await;

        assert!(!result.reachable);
        assert_eq!(result.version, None);
        assert!(result.models.is_empty());
    }

    #[tokio::test]
    async fn detect_llm_dedupes_overlapping_models_across_protocols() {
        // The same server speaks BOTH Ollama (/api/version + /api/tags) and the
        // OpenAI-compatible protocol (/v1/models), advertising an OVERLAPPING
        // model name. The merged `models` must dedupe it to a single entry.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/version"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "version": "0.4.1"
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [
                    { "name": "llama3:latest" },
                    { "name": "shared-model" }
                ]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    { "id": "shared-model" },
                    { "id": "mistral-7b" }
                ]
            })))
            .mount(&server)
            .await;

        let result = detect_llm(&server.uri()).await;

        assert!(result.reachable);
        assert_eq!(result.version, Some("0.4.1".to_string()));
        // "shared-model" appears in BOTH protocols but only once in the merge,
        // and the Ollama-first ordering is preserved.
        assert_eq!(
            result.models,
            vec!["llama3:latest", "shared-model", "mistral-7b"]
        );
        // Defensively assert the dedupe: the overlapping name occurs exactly once.
        assert_eq!(
            result
                .models
                .iter()
                .filter(|m| *m == "shared-model")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn detect_llm_rejects_non_http_scheme() {
        // A non-http(s) scheme must short-circuit to unreachable WITHOUT probing
        // (SSRF defense-in-depth — see detect_llm's scheme allowlist).
        let result = detect_llm("file:///etc/passwd").await;
        assert!(!result.reachable);
        assert_eq!(result.version, None);
        assert!(result.models.is_empty());
    }

    // --- list_ollama_models tests (Stage 3 — live local picker) ------------

    #[tokio::test]
    async fn list_ollama_models_returns_pulled_models() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/version"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "version": "0.4.1"
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [
                    { "name": "qwen2.5-coder:latest" },
                    { "name": "qwen2.5:7b" }
                ]
            })))
            .mount(&server)
            .await;

        let models = list_ollama_models(&server.uri()).await;
        assert_eq!(models, vec!["qwen2.5-coder:latest", "qwen2.5:7b"]);
    }

    #[tokio::test]
    async fn list_ollama_models_empty_when_unreachable() {
        // Ollama not running (always-refused port) ⇒ empty list, NEVER an error.
        let models = list_ollama_models("http://127.0.0.1:1").await;
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn list_ollama_models_empty_for_non_http_scheme() {
        // A non-http(s) scheme short-circuits to empty (no probe), no panic/error.
        let models = list_ollama_models("file:///etc/passwd").await;
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn list_ollama_models_empty_when_not_ollama_endpoint() {
        // An endpoint with no /api/version (not Ollama) ⇒ empty list, never errors.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "id": "gpt-4o" }]
            })))
            .mount(&server)
            .await;
        let models = list_ollama_models(&server.uri()).await;
        assert!(models.is_empty(), "no /api/version ⇒ Ollama absent ⇒ empty");
    }

    /// Snapshot the exact serde wire-format of `LlmDetection`. Locks the FROZEN
    /// IPC contract for the "Configure → Auto-detect" feature.
    #[test]
    fn llm_detection_serialized_shape() {
        let result = LlmDetection {
            reachable: true,
            version: Some("0.4.1".to_string()),
            models: vec!["llama3:latest".to_string()],
        };
        insta::assert_json_snapshot!(result, @r#"
        {
          "reachable": true,
          "version": "0.4.1",
          "models": [
            "llama3:latest"
          ]
        }
        "#);
    }
}
