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
use std::sync::LazyLock;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::AppConfig;

const PROBE_CONNECT_TIMEOUT: Duration = Duration::from_secs(1);
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434";
const DEFAULT_LMSTUDIO_BASE_URL: &str = "http://localhost:1234";
/// Cloud enrichment-model live-probe timeout (issue #90). Caps a hanging
/// endpoint so a system check never stalls onboarding.
const CLOUD_PROBE_TIMEOUT: Duration = Duration::from_secs(10);
/// Canonical embedding-model ids derived from the registry at init time (NOT a
/// hand-maintained literal) so adding a model to the registry can never desync
/// this list. The Ollama alias `"nomic-embed-text"` is accepted separately via
/// the alias bridge in [`is_allowlisted_embedding_id`].
pub static ALLOWED_EMBEDDING_MODELS: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    crate::embedder::registry::REGISTRY
        .iter()
        .map(|s| s.id)
        .collect()
});

/// Returns `true` when `id` is an accepted embedding-model id — canonical
/// registry id or the legacy Ollama alias `"nomic-embed-text"`, both resolved
/// by the registry's `resolve_opt` alias bridge.
pub fn is_allowlisted_embedding_id(id: &str) -> bool {
    crate::embedder::registry::resolve_opt(id).is_some()
}
/// Defense-in-depth cap on buffered probe response bodies (1 MiB).
const MAX_PROBE_BODY_BYTES: usize = 1024 * 1024;

/// Status of a single system-check row. Serializes lowercase: `pass` | `fail`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Pass,
    Fail,
}

/// Stable identifier for each system-check row (drives UI row ordering/mapping).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckId {
    LlmRuntime,
    EmbeddingModel,
    TextToSpeech,
}

/// Optional UI affordance attached to a check row. Absence is `None` on
/// [`CheckResult::action`] — there is deliberately no `None` variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckAction {
    Configure,
    Choose,
}

/// One row in the system-check screen.
///
/// FROZEN IPC CONTRACT — crosses the Tauri boundary verbatim; field names and
/// serde shape are locked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CheckResult {
    pub id: CheckId,
    pub label: String,
    pub status: CheckStatus,
    /// Product-facing detail copy; no internal milestone vocabulary.
    pub detail: String,
    pub action: Option<CheckAction>,
}

/// Result of probing a single LLM endpoint via [`detect_llm`].
///
/// FROZEN IPC CONTRACT for the "Configure → Auto-detect" flow; field names
/// and serde shape are locked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LlmDetection {
    pub reachable: bool,
    /// Ollama version string when the endpoint spoke the Ollama protocol.
    pub version: Option<String>,
    pub models: Vec<String>,
}

/// Probes `base_url` for both Ollama-style and OpenAI-compatible endpoints
/// concurrently, returning a merged [`LlmDetection`]. Unreachable endpoints
/// contribute nothing; never returns `Err` for "not reachable".
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

/// Lists locally-pulled Ollama models via `GET /api/tags`. Returns an empty
/// `Vec` (never `Err`) when Ollama is unreachable.
pub async fn list_ollama_models(base_url: &str) -> Vec<String> {
    let base_url = base_url.trim_end_matches('/');

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

/// GETs `url`, caps the buffered body at [`MAX_PROBE_BODY_BYTES`], and
/// deserializes as `T`. Returns `None` on error, non-2xx, over-cap, or parse miss.
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

async fn probe_openai_endpoint(client: &reqwest::Client, base_url: &str) -> Vec<String> {
    let url = format!("{base_url}/v1/models");
    get_json_capped::<OpenAiModels>(client, &url)
        .await
        .map(|m| m.data.into_iter().map(|d| d.id).collect())
        .unwrap_or_default()
}

#[derive(Debug, Deserialize)]
struct OpenAiModels {
    #[serde(default)]
    data: Vec<OpenAiModelEntry>,
}

#[derive(Debug, Deserialize)]
struct OpenAiModelEntry {
    id: String,
}

#[derive(Debug, Deserialize)]
struct OllamaVersion {
    version: String,
}

#[derive(Debug, Deserialize)]
struct OllamaTagModel {
    #[serde(default)]
    name: String,
}

#[derive(Debug, Deserialize)]
struct OllamaTags {
    #[serde(default)]
    models: Vec<OllamaTagModel>,
}

/// LLM-runtime probe outcome shared with the embedding-model gate.
struct LlmRuntimeProbe {
    result: CheckResult,
    ollama_up: bool,
    ollama_base_url: String,
}

/// Short-timeout HTTP client for runtime detection: bounded timeouts, no redirects
/// (a redirect could 30x a probe toward an internal host — SSRF hardening).
fn probe_client() -> reqwest::Client {
    crate::http::hardened_client(PROBE_CONNECT_TIMEOUT, PROBE_TIMEOUT)
}

/// Resolves the configured Ollama base URL (public so the install command uses
/// the same URL the system-check probe detected).
pub fn ollama_base_url(config: &AppConfig) -> String {
    provider_base_url(config, "ollama").unwrap_or_else(|| DEFAULT_OLLAMA_BASE_URL.to_string())
}

fn lmstudio_base_url(config: &AppConfig) -> String {
    provider_base_url(config, "lmstudio")
        .or_else(|| provider_base_url(config, "lm_studio"))
        .or_else(|| provider_base_url(config, "lm studio"))
        .unwrap_or_else(|| DEFAULT_LMSTUDIO_BASE_URL.to_string())
}

fn provider_base_url(config: &AppConfig, provider: &str) -> Option<String> {
    config
        .models
        .iter()
        .find(|m| m.provider.eq_ignore_ascii_case(provider) && !m.base_url.is_empty())
        .map(|m| m.base_url.trim_end_matches('/').to_string())
}

/// Recognized cloud provider ids (models.dev keys + the `openai-compatible` alias).
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

/// Returns `true` when config carries a usable cloud LLM entry (recognized
/// provider + non-empty `api_key` + non-empty `model`).
fn has_cloud_llm(config: &AppConfig) -> bool {
    config.models.iter().any(|m| {
        CLOUD_LLM_PROVIDERS
            .iter()
            .any(|p| m.provider.eq_ignore_ascii_case(p))
            && !m.api_key.is_empty()
            && !m.model.is_empty()
    })
}

async fn detect_ollama(client: &reqwest::Client, base_url: &str) -> Option<String> {
    let (version, _models) = probe_ollama_endpoint(client, base_url).await;
    version
}

async fn detect_lmstudio(client: &reqwest::Client, base_url: &str) -> bool {
    !probe_openai_endpoint(client, base_url).await.is_empty()
}

/// Enrichment-model validation outcome (issue #90). Separate from [`CheckStatus`]
/// so the frozen IPC enum stays two-valued; `Invalid(reason)` carries the
/// actionable message that `probe_llm_runtime` folds into the row's `detail`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelValidation {
    Pass,
    Invalid(String),
}

/// Canonical "model not installed" message used at all three validation sites.
pub fn ollama_model_missing_reason(model: &str) -> String {
    format!("Model '{model}' is not installed in Ollama. Run `ollama pull {model}` to install it.")
}

/// Runs a `max_tokens:1` live probe against `provider` under `CLOUD_PROBE_TIMEOUT`.
async fn cloud_probe(provider: &dyn crate::llm::LlmProvider, err_prefix: &str) -> ModelValidation {
    let req = crate::llm::LlmRequest {
        system: None,
        prompt: "hi".to_string(),
        max_tokens: 1,
        temperature: 0.0,
        json: false,
        thinking: false,
        reasoning_effort: None,
    };
    match tokio::time::timeout(CLOUD_PROBE_TIMEOUT, provider.generate(&req)).await {
        Ok(Ok(_)) => ModelValidation::Pass,
        Ok(Err(e)) => ModelValidation::Invalid(format!("{err_prefix} probe failed: {e}")),
        Err(_) => ModelValidation::Invalid(format!(
            "{err_prefix} probe timed out after {}s",
            CLOUD_PROBE_TIMEOUT.as_secs()
        )),
    }
}

/// Validates the enrichment model via the same provider factory the engine uses,
/// so routing and consent are respected (issue #90 Principle 3). Disabled
/// enrichment always returns `Pass`; Ollama uses a tags check; cloud uses a live
/// probe; non-Ollama local falls through to `Pass` (reachability is sufficient).
pub async fn validate_enrichment_model(config: &AppConfig) -> ModelValidation {
    if !config.enrichment.enabled {
        return ModelValidation::Pass;
    }

    let provider = match crate::llm::provider_from_config(config, config.enrichment.cloud_consent) {
        Some(p) => p,
        None => {
            return ModelValidation::Invalid(
                "Enrichment is enabled but no LLM provider is configured for the current routing policy"
                    .to_string(),
            );
        }
    };

    let genai = provider
        .as_any()
        .downcast_ref::<crate::llm::GenaiProvider>();
    match genai {
        Some(g) if g.is_ollama() => {
            let model = provider.model_id().to_string();
            let installed = list_ollama_models(&ollama_base_url(config)).await;
            if installed.iter().any(|m| m == &model) {
                ModelValidation::Pass
            } else {
                ModelValidation::Invalid(format!(
                    "LLM runtime detected, but {}",
                    ollama_model_missing_reason(&model)
                ))
            }
        }
        Some(_) => cloud_probe(provider.as_ref(), "Cloud enrichment model").await,
        None => ModelValidation::Pass,
    }
}

/// Validates raw (not-yet-persisted) params for the interactive `LlmConfigPanel`
/// pre-save probe (issue #90). Ollama: tags-membership check; cloud: live probe.
pub async fn validate_model_interactive(
    provider: &str,
    model: &str,
    base_url: &str,
    api_key: &str,
) -> ModelValidation {
    if provider.eq_ignore_ascii_case("ollama") {
        let installed = list_ollama_models(base_url).await;
        return if installed.iter().any(|m| m == model) {
            ModelValidation::Pass
        } else {
            ModelValidation::Invalid(ollama_model_missing_reason(model))
        };
    }

    let Some(provider) = crate::llm::build_provider_raw(provider, model, base_url, api_key) else {
        return ModelValidation::Invalid(
            "Unrecognized provider or missing endpoint for the selected model".to_string(),
        );
    };
    cloud_probe(provider.as_ref(), "Cloud model").await
}

/// LLM-runtime readiness gate: passes when a local runtime (Ollama / LM Studio,
/// probed concurrently) or a configured cloud provider is reachable.
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

    let mut result = match (ollama_up, lmstudio_up, cloud_ok) {
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

    // Only override to Fail when the preliminary reachability check passed — a
    // pre-existing Fail (no runtime) is moot to augment with enrichment detail.
    if result.status == CheckStatus::Pass
        && let ModelValidation::Invalid(reason) = validate_enrichment_model(config).await
    {
        result.status = CheckStatus::Fail;
        result.detail = reason;
        result.action = Some(CheckAction::Configure);
    }

    LlmRuntimeProbe {
        result,
        ollama_up,
        ollama_base_url: ollama_base,
    }
}

/// Returns `true` when `installed_name` matches an allowlisted embedding model
/// or the user's configured `embedding_model`.
///
/// EXACT-TAG rule (issue #80): try the full name first (accepts colon-bearing ids
/// like `qwen3-embedding:4b` exactly), then fall back to tag-stripping for the
/// `nomic-embed-text:latest` case. Configured-id escape hatch handles models not
/// yet in the registry.
fn is_allowlisted_embedding(installed_name: &str, configured: &str) -> bool {
    let full = installed_name.to_ascii_lowercase();
    let bare = installed_name
        .split_once(':')
        .map_or(installed_name, |(name, _tag)| name)
        .to_ascii_lowercase();
    is_allowlisted_embedding_id(&full)
        || is_allowlisted_embedding_id(&bare)
        || (!configured.is_empty()
            && (configured.eq_ignore_ascii_case(&full) || configured.eq_ignore_ascii_case(&bare)))
}

/// Returns `true` when `model_id`'s fastembed weights are cached under
/// `{data_dir}/models/fastembed/`. The per-model subdir shape
/// (`models--{org}--{model}`) was observed empirically (R6, M4 Phase 4b-B).
/// A non-empty subdir is treated as cached; construction is the final arbiter.
/// Public so the `fastembed_models_cached` Tauri command can surface cache state.
pub fn fastembed_weights_cached(data_dir: &Path, model_id: &str) -> bool {
    let spec = crate::embedder::resolve(model_id);
    // Ollama-only models (issue #80) have no fastembed cache dir.
    let Some(subdir) = spec.fastembed_cache_subdir() else {
        return false;
    };
    let model_dir = data_dir.join("models").join("fastembed").join(subdir);
    if !model_dir.is_dir() {
        return false;
    }
    std::fs::read_dir(&model_dir)
        .ok()
        .and_then(|mut d| d.next())
        .is_some()
}

/// Whether the local (non-Ollama) embedding engine's weights are on disk.
/// On `native-ml-metal` (Apple Silicon), accepts EITHER candle or fastembed cache
/// (issue #91); other builds use the fastembed-only check.
fn local_embedding_weights_cached(data_dir: &Path, model_id: &str) -> bool {
    #[cfg(feature = "native-ml-metal")]
    {
        if let Some(subdir) = crate::embedder::candle_cache_subdir(model_id) {
            let candle_dir = data_dir.join("models").join("candle").join(subdir);
            let candle_cached = candle_dir.is_dir()
                && std::fs::read_dir(&candle_dir)
                    .ok()
                    .and_then(|mut d| d.next())
                    .is_some();
            if candle_cached {
                return true;
            }
        }
    }
    fastembed_weights_cached(data_dir, model_id)
}

/// Embedding-readiness gate predicate (M4 4b-B D2): each backend passes only on
/// its own arm — fastembed on cached weights, Ollama on a detected tag.
/// Pure (no I/O) so the truth table is exhaustively unit-testable.
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

/// Embedding-model readiness gate (M4 4b-B): resolves the configured backend and
/// passes iff its own arm is satisfied (fastembed → weights on disk; Ollama →
/// allowlisted tag detected). Ollama is never probed for a fastembed selection.
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

    let fastembed_cached = local_embedding_weights_cached(data_dir, &config.embedding_model);

    // Only probe Ollama for an Ollama-backend selection (D2: fastembed must never wait on Ollama).
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

fn has_cloud_tts(config: &AppConfig) -> bool {
    matches!(config.tts.backend, crate::tts::TtsBackend::Cloud(_))
        && config
            .tts
            .cloud
            .as_ref()
            .is_some_and(|c| !c.api_key.is_empty())
}

/// TTS readiness gate: passes when the Kokoro ONNX model is on disk AND both
/// voices are saved, OR a cloud TTS provider is configured.
fn probe_text_to_speech(config: &AppConfig) -> CheckResult {
    let model_path = crate::tts::kokoro_model_path(Path::new(&config.paths.data_dir));
    let kokoro_on_disk = model_path.is_file();
    let voices_set = !config.voices.host.is_unset() && !config.voices.guest.is_unset();

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

/// Runs the three system-check probes in fixed order: LlmRuntime, EmbeddingModel,
/// TextToSpeech. The embedding probe reuses the LLM-runtime Ollama outcome, so it
/// runs after. The caller drops the engine lock before calling so the HTTP probes
/// never block concurrent config reads/writes.
pub(crate) async fn run_system_check(config: &AppConfig, data_dir: &Path) -> Vec<CheckResult> {
    let embed_client = probe_client();
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
        assert_eq!(
            lmstudio_base_url(&AppConfig::default()),
            DEFAULT_LMSTUDIO_BASE_URL
        );
        let cfg = config_with_runtimes("", "http://127.0.0.1:9999/");
        assert_eq!(lmstudio_base_url(&cfg), "http://127.0.0.1:9999");
    }

    #[tokio::test]
    async fn aggregate_falls_back_to_lmstudio_via_seam() {
        // Fixed always-refused port avoids the parallel-test port-reuse race.
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
        // Port 1 is always refused — avoids the parallel-test port-reuse race.
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
        let mut config = AppConfig::default();
        config.models.push(ModelConfig {
            provider: "openai-compatible".to_string(),
            model: "gpt-4o".to_string(),
            ..ModelConfig::default()
        });
        assert!(!has_cloud_llm(&config));

        config.models[0].model = String::new();
        config.models[0].api_key = "sk-cloud".to_string();
        assert!(!has_cloud_llm(&config));

        config.models[0].model = "gpt-4o".to_string();
        assert!(has_cloud_llm(&config));
    }

    #[test]
    fn has_cloud_llm_recognizes_first_class_cloud_providers() {
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
    fn has_cloud_tts_requires_cloud_backend_and_key() {
        use crate::config::{CloudTtsCfg, TtsConfig};
        use crate::tts::{CloudTtsKind, TtsBackend};

        let mut config = AppConfig::default();
        // Kokoro (default) is not cloud.
        assert!(!has_cloud_tts(&config));

        // Cloud backend but empty key → not configured.
        config.tts = TtsConfig {
            version: 1,
            backend: TtsBackend::Cloud(CloudTtsKind::ElevenLabs),
            model: String::new(),
            cloud: Some(CloudTtsCfg {
                kind: CloudTtsKind::ElevenLabs,
                api_key: String::new(),
                base_url: String::new(),
            }),
        };
        assert!(!has_cloud_tts(&config));

        // Cloud backend + key → configured.
        config.tts.cloud = Some(CloudTtsCfg {
            kind: CloudTtsKind::ElevenLabs,
            api_key: "sk-key".to_string(),
            base_url: String::new(),
        });
        assert!(has_cloud_tts(&config));
    }

    #[tokio::test]
    async fn llm_runtime_falls_back_to_lmstudio() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "id": "local-model" }]
            })))
            .mount(&server)
            .await;

        let client = probe_client();
        let ollama = detect_ollama(&client, &server.uri()).await;
        let lmstudio = detect_lmstudio(&client, &server.uri()).await;

        assert!(ollama.is_none(), "no /api/version ⇒ Ollama absent");
        assert!(lmstudio, "LM Studio /v1/models answered 200");
    }

    #[tokio::test]
    async fn llm_probe_stays_within_time_budget_when_offline() {
        let dead_url = "http://127.0.0.1:1";
        let config = config_with_runtimes(dead_url, dead_url);
        let start = Instant::now();
        let _ = probe_llm_runtime(&config).await;
        let elapsed = start.elapsed();

        // Concurrent probing = one timeout window (1s connect + 2s read = 3s max);
        // 3500ms budget adds 500ms CI jitter slack.
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
        let result =
            probe_embedding_model(&client, &runtime, &AppConfig::default(), dir.path()).await;

        assert_eq!(result.status, CheckStatus::Fail);
    }

    fn seed_fastembed_cache(data_dir: &Path, model_id: &str) {
        let subdir = crate::embedder::resolve(model_id)
            .fastembed_cache_subdir()
            .expect("a fastembed model has a cache subdir");
        let model_dir = data_dir.join("models").join("fastembed").join(subdir);
        std::fs::create_dir_all(model_dir.join("snapshots")).unwrap();
        std::fs::write(model_dir.join("snapshots").join("model.onnx"), b"fake").unwrap();
    }

    #[tokio::test]
    async fn embedding_pass_when_fastembed_weights_cached_ollama_down() {
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

    /// `fastembed_weights_cached` is per-model: seeding nomic must not satisfy mxbai.
    #[test]
    fn fastembed_weights_cached_is_per_model() {
        let dir = tempfile::tempdir().unwrap();
        seed_fastembed_cache(dir.path(), "nomic-embed-text-v1.5");

        assert!(fastembed_weights_cached(
            dir.path(),
            "nomic-embed-text-v1.5"
        ));
        assert!(!fastembed_weights_cached(dir.path(), "mxbai-embed-large"));

        seed_fastembed_cache(dir.path(), "mxbai-embed-large");
        assert!(fastembed_weights_cached(dir.path(), "mxbai-embed-large"));
        assert!(fastembed_weights_cached(
            dir.path(),
            "nomic-embed-text-v1.5"
        ));
    }

    #[test]
    fn fastembed_weights_cached_false_for_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let subdir = crate::embedder::resolve("nomic-embed-text-v1.5")
            .fastembed_cache_subdir()
            .expect("nomic has a cache subdir");
        std::fs::create_dir_all(dir.path().join("models").join("fastembed").join(subdir)).unwrap();
        assert!(!fastembed_weights_cached(
            dir.path(),
            "nomic-embed-text-v1.5"
        ));
    }

    #[test]
    fn fastembed_cache_subdir_matches_observed_shape() {
        assert_eq!(
            crate::embedder::resolve("all-minilm")
                .fastembed_cache_subdir()
                .as_deref(),
            Some("models--Qdrant--all-MiniLM-L6-v2-onnx")
        );
        assert_eq!(
            crate::embedder::resolve("nomic-embed-text-v1.5")
                .fastembed_cache_subdir()
                .as_deref(),
            Some("models--nomic-ai--nomic-embed-text-v1.5")
        );
        assert_eq!(
            crate::embedder::resolve("qwen3-embedding:4b").fastembed_cache_subdir(),
            None
        );
    }

    #[test]
    fn gate_predicate_truth_table() {
        use crate::embedder::EmbeddingBackend::{Fastembed, Ollama};
        assert!(embedding_gate_passes(Fastembed, true, false));
        assert!(embedding_gate_passes(Fastembed, true, true));
        assert!(!embedding_gate_passes(Fastembed, false, false));
        assert!(
            !embedding_gate_passes(Fastembed, false, true),
            "a fastembed selection must NEVER pass on the strength of an Ollama tag"
        );
        assert!(embedding_gate_passes(Ollama, false, true));
        assert!(embedding_gate_passes(Ollama, true, true));
        assert!(!embedding_gate_passes(Ollama, false, false));
        assert!(
            !embedding_gate_passes(Ollama, true, false),
            "an ollama selection must NEVER pass on the strength of a fastembed cache"
        );
    }

    /// D2 showstopper guard: fastembed weights cached + Ollama unreachable must pass.
    #[tokio::test]
    async fn fresh_install_fastembed_only_passes_gate() {
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
        assert_eq!(
            result.status,
            CheckStatus::Pass,
            "a fresh fastembed-only install with cached weights must pass with Ollama unreachable"
        );
    }

    #[test]
    fn allowlisted_embedding_matches_bare_name_and_config() {
        assert!(is_allowlisted_embedding("nomic-embed-text:latest", ""));
        assert!(is_allowlisted_embedding("nomic-embed-text-v1.5", ""));
        assert!(is_allowlisted_embedding("nomic-embed-text-v1.5:latest", ""));
        assert!(is_allowlisted_embedding("BGE-M3", ""));
        assert!(!is_allowlisted_embedding("my-custom-embed:latest", ""));
        assert!(is_allowlisted_embedding(
            "my-custom-embed:latest",
            "my-custom-embed"
        ));
        assert!(!is_allowlisted_embedding("llama3:latest", ""));
    }

    #[test]
    fn allowlisted_embedding_exact_tag_for_colon_bearing_ids() {
        assert!(is_allowlisted_embedding("qwen3-embedding:4b", ""));
        assert!(!is_allowlisted_embedding("qwen3-embedding:0.6b", ""));
        assert!(!is_allowlisted_embedding("qwen3-embedding:8b", ""));
        assert!(!is_allowlisted_embedding("qwen3-embedding", ""));
        assert!(is_allowlisted_embedding("embeddinggemma", ""));
        assert!(is_allowlisted_embedding("embeddinggemma:latest", ""));
        assert!(is_allowlisted_embedding("nomic-embed-text-v2-moe", ""));
        assert!(is_allowlisted_embedding("snowflake-arctic-embed2", ""));
        assert!(is_allowlisted_embedding("nomic-embed-text:latest", ""));
    }

    #[test]
    fn new_ollama_ids_are_in_allowlist() {
        for id in [
            "embeddinggemma",
            "qwen3-embedding:4b",
            "nomic-embed-text-v2-moe",
            "snowflake-arctic-embed2",
        ] {
            assert!(
                ALLOWED_EMBEDDING_MODELS.contains(&id),
                "{id} must be in the registry-derived allowlist"
            );
        }
    }

    #[test]
    fn allowlist_is_derived_from_registry_and_accepts_canonical_and_alias() {
        assert!(ALLOWED_EMBEDDING_MODELS.contains(&"nomic-embed-text-v1.5"));
        assert!(!ALLOWED_EMBEDDING_MODELS.contains(&"nomic-embed-text"));
        assert_eq!(
            ALLOWED_EMBEDDING_MODELS.len(),
            crate::embedder::registry::REGISTRY.len(),
            "allowlist must not drift from the registry"
        );
        assert!(is_allowlisted_embedding_id("nomic-embed-text-v1.5"));
        assert!(is_allowlisted_embedding_id("nomic-embed-text"));
        assert!(is_allowlisted_embedding_id("bge-m3"));
        assert!(!is_allowlisted_embedding_id("totally-made-up-model"));
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
        let dir = tempfile::tempdir().unwrap();
        write_kokoro_model(dir.path());

        let mut config = AppConfig::default();
        config.paths.data_dir = dir.path().display().to_string();

        let result = probe_text_to_speech(&config);
        assert_eq!(result.status, CheckStatus::Fail);
        assert_eq!(
            result.detail,
            "Download the engine and choose voices, or connect a cloud provider"
        );
    }

    #[test]
    fn text_to_speech_pass_when_model_present_and_voices_set() {
        let dir = tempfile::tempdir().unwrap();
        write_kokoro_model(dir.path());

        let mut config = AppConfig::default();
        config.paths.data_dir = dir.path().display().to_string();
        config.voices = crate::config::VoiceConfig {
            host: crate::config::VoiceRef::Named("am_michael".to_string()),
            guest: crate::config::VoiceRef::Named("af_heart".to_string()),
        };

        let result = probe_text_to_speech(&config);
        assert_eq!(result.status, CheckStatus::Pass);
        assert_eq!(result.detail, "Kokoro audio engine ready");
    }

    #[test]
    fn text_to_speech_fail_when_only_one_voice_set() {
        let dir = tempfile::tempdir().unwrap();
        write_kokoro_model(dir.path());

        let mut config = AppConfig::default();
        config.paths.data_dir = dir.path().display().to_string();
        config.voices = crate::config::VoiceConfig {
            host: crate::config::VoiceRef::Named("am_michael".to_string()),
            guest: crate::config::VoiceRef::default(),
        };

        let result = probe_text_to_speech(&config);
        assert_eq!(result.status, CheckStatus::Fail);
    }

    #[test]
    fn text_to_speech_fail_when_voices_set_but_model_absent() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = AppConfig::default();
        config.paths.data_dir = dir.path().display().to_string();
        config.voices = crate::config::VoiceConfig {
            host: crate::config::VoiceRef::Named("am_michael".to_string()),
            guest: crate::config::VoiceRef::Named("af_heart".to_string()),
        };

        let result = probe_text_to_speech(&config);
        assert_eq!(result.status, CheckStatus::Fail);
    }

    #[test]
    fn text_to_speech_pass_when_cloud_configured() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = AppConfig::default();
        config.paths.data_dir = dir.path().display().to_string();
        config.tts = crate::config::TtsConfig {
            version: 1,
            backend: crate::tts::TtsBackend::Cloud(crate::tts::CloudTtsKind::ElevenLabs),
            model: String::new(),
            cloud: Some(crate::config::CloudTtsCfg {
                kind: crate::tts::CloudTtsKind::ElevenLabs,
                api_key: "sk-elevenlabs".to_string(),
                base_url: String::new(),
            }),
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
        let result = detect_llm("http://127.0.0.1:1").await;

        assert!(!result.reachable);
        assert_eq!(result.version, None);
        assert!(result.models.is_empty());
    }

    #[tokio::test]
    async fn detect_llm_dedupes_overlapping_models_across_protocols() {
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
        assert_eq!(
            result.models,
            vec!["llama3:latest", "shared-model", "mistral-7b"]
        );
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
        let result = detect_llm("file:///etc/passwd").await;
        assert!(!result.reachable);
        assert_eq!(result.version, None);
        assert!(result.models.is_empty());
    }

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
        let models = list_ollama_models("http://127.0.0.1:1").await;
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn list_ollama_models_empty_for_non_http_scheme() {
        let models = list_ollama_models("file:///etc/passwd").await;
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn list_ollama_models_empty_when_not_ollama_endpoint() {
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

    use crate::config::EnrichmentConfig;
    use crate::llm::LlmRouting;

    /// An OpenAI-compatible `/v1/chat/completions` success body (genai parses
    /// `choices[0].message.content` + `usage`).
    fn openai_chat_body(content: &str) -> serde_json::Value {
        serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-test",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
        })
    }

    /// Config with a single local Ollama enrichment entry, `enabled` + LocalFirst
    /// routing so the resolved provider is deterministically the Ollama entry.
    fn config_enrichment_ollama(base_url: &str, model: &str, enabled: bool) -> AppConfig {
        AppConfig {
            models: vec![ModelConfig {
                provider: "ollama".to_string(),
                base_url: base_url.to_string(),
                model: model.to_string(),
                ..ModelConfig::default()
            }],
            enrichment: EnrichmentConfig {
                enabled,
                routing: LlmRouting::LocalFirst,
                ..EnrichmentConfig::default()
            },
            ..AppConfig::default()
        }
    }

    /// Config with a single cloud (openai-compatible) enrichment entry pointed at
    /// `base_url`, consented so the provider resolves.
    fn config_enrichment_cloud(base_url: &str, model: &str) -> AppConfig {
        AppConfig {
            models: vec![ModelConfig {
                provider: "openai-compatible".to_string(),
                base_url: base_url.to_string(),
                model: model.to_string(),
                api_key: "sk-test".to_string(),
                ..ModelConfig::default()
            }],
            enrichment: EnrichmentConfig {
                enabled: true,
                cloud_consent: true,
                routing: LlmRouting::CloudFirst,
                ..EnrichmentConfig::default()
            },
            ..AppConfig::default()
        }
    }

    #[tokio::test]
    async fn enrichment_validation_opt_out() {
        let config = config_enrichment_ollama("http://127.0.0.1:1", "llama3.2:3b", false);
        assert!(matches!(
            validate_enrichment_model(&config).await,
            ModelValidation::Pass
        ));
    }

    #[tokio::test]
    async fn enrichment_validation_no_provider_but_enabled() {
        let config = AppConfig {
            enrichment: EnrichmentConfig {
                enabled: true,
                ..EnrichmentConfig::default()
            },
            ..AppConfig::default()
        };
        match validate_enrichment_model(&config).await {
            ModelValidation::Invalid(reason) => {
                assert!(
                    reason.to_lowercase().contains("no llm provider"),
                    "got {reason}"
                );
            }
            ModelValidation::Pass => panic!("expected Invalid when enabled + no provider"),
        }
    }

    #[tokio::test]
    async fn enrichment_validation_local_model_present() {
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
                "models": [{ "name": "llama3.2:3b" }]
            })))
            .mount(&server)
            .await;

        let config = config_enrichment_ollama(&server.uri(), "llama3.2:3b", true);
        assert!(matches!(
            validate_enrichment_model(&config).await,
            ModelValidation::Pass
        ));
    }

    #[tokio::test]
    async fn enrichment_validation_local_model_missing() {
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
                "models": [{ "name": "mistral:7b" }]
            })))
            .mount(&server)
            .await;

        let config = config_enrichment_ollama(&server.uri(), "llama3.2:3b", true);
        match validate_enrichment_model(&config).await {
            ModelValidation::Invalid(reason) => {
                assert!(reason.contains("llama3.2:3b"), "got {reason}");
            }
            ModelValidation::Pass => panic!("expected Invalid for a missing local model"),
        }
    }

    #[tokio::test]
    async fn enrichment_validation_cloud_probe_success() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_chat_body("ok")))
            .mount(&server)
            .await;

        let config = config_enrichment_cloud(&server.uri(), "gpt-test");
        assert!(
            matches!(
                validate_enrichment_model(&config).await,
                ModelValidation::Pass
            ),
            "a successful cloud probe passes"
        );
    }

    #[tokio::test]
    async fn enrichment_validation_cloud_probe_fail() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let config = config_enrichment_cloud(&server.uri(), "gpt-test");
        match validate_enrichment_model(&config).await {
            ModelValidation::Invalid(reason) => {
                assert!(
                    reason.to_lowercase().contains("cloud"),
                    "reason names the cloud probe: {reason}"
                );
            }
            ModelValidation::Pass => panic!("expected Invalid on a failing cloud probe"),
        }
    }

    #[tokio::test]
    async fn enrichment_validation_cloud_probe_timeout() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(openai_chat_body("ok"))
                    .set_delay(Duration::from_secs(11)),
            )
            .mount(&server)
            .await;

        let config = config_enrichment_cloud(&server.uri(), "gpt-test");
        match validate_enrichment_model(&config).await {
            ModelValidation::Invalid(reason) => {
                assert!(
                    reason.to_lowercase().contains("timed out")
                        || reason.to_lowercase().contains("timeout"),
                    "reason names the timeout: {reason}"
                );
            }
            ModelValidation::Pass => panic!("expected Invalid on a hanging cloud probe"),
        }
    }

    #[tokio::test]
    async fn enrichment_validation_routing_resolves_correct_model() {
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
                "models": [{ "name": "mistral:7b" }]
            })))
            .mount(&server)
            .await;

        let config = AppConfig {
            models: vec![
                ModelConfig {
                    provider: "ollama".to_string(),
                    base_url: server.uri(),
                    model: "llama3.2:3b".to_string(),
                    ..ModelConfig::default()
                },
                ModelConfig {
                    provider: "ollama".to_string(),
                    base_url: server.uri(),
                    model: "mistral:7b".to_string(),
                    ..ModelConfig::default()
                },
            ],
            enrichment: EnrichmentConfig {
                enabled: true,
                routing: LlmRouting::Explicit {
                    provider: "ollama".to_string(),
                    model: "mistral:7b".to_string(),
                },
                ..EnrichmentConfig::default()
            },
            ..AppConfig::default()
        };
        assert!(
            matches!(
                validate_enrichment_model(&config).await,
                ModelValidation::Pass
            ),
            "Explicit routing validates the pinned mistral:7b (present), not llama3.2:3b (absent)"
        );
    }

    #[tokio::test]
    async fn probe_llm_runtime_fails_enrichment_model_missing() {
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
                "models": [{ "name": "mistral:7b" }]
            })))
            .mount(&server)
            .await;

        let config = config_enrichment_ollama(&server.uri(), "llama3.2:3b", true);
        let probe = probe_llm_runtime(&config).await;
        assert_eq!(probe.result.status, CheckStatus::Fail);
        assert_eq!(probe.result.action, Some(CheckAction::Configure));
        assert!(
            probe.result.detail.contains("llama3.2:3b"),
            "detail names the missing model: {}",
            probe.result.detail
        );
    }

    #[tokio::test]
    async fn probe_llm_runtime_passes_enrichment_disabled() {
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
                "models": [{ "name": "mistral:7b" }]
            })))
            .mount(&server)
            .await;

        let config = config_enrichment_ollama(&server.uri(), "llama3.2:3b", false);
        let probe = probe_llm_runtime(&config).await;
        assert_eq!(probe.result.status, CheckStatus::Pass);
    }

    #[tokio::test]
    async fn probe_llm_runtime_passes_enrichment_model_valid() {
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
                "models": [{ "name": "llama3.2:3b" }]
            })))
            .mount(&server)
            .await;

        let config = config_enrichment_ollama(&server.uri(), "llama3.2:3b", true);
        let probe = probe_llm_runtime(&config).await;
        assert_eq!(probe.result.status, CheckStatus::Pass);
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
