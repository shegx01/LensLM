//! LLM provider seam for the M4 Phase-3 enrichment pass.
//!
//! Defines [`LlmProvider`] — an `async`, object-safe trait (held behind
//! `Arc<dyn LlmProvider>`) alongside [`crate::Embedder`] / [`crate::VectorStore`]
//! — plus three HTTP backends ([`OllamaProvider`], [`OpenAiCompatProvider`],
//! [`AnthropicProvider`]) and a [`provider_from_config`] factory.
//!
//! All three backends are HTTP, so the trait is async and reuses the same
//! reqwest patterns as [`crate::system_check`] (rustls, no-redirect SSRF guard,
//! bounded timeouts, body-capped JSON) rather than `spawn_blocking`.
//!
//! `reachable()` is a `system_check`-style probe used to decide whether the
//! enrichment worker should dispatch — a misconfigured or down provider lets a
//! source degrade gracefully to raw vectors instead of looping.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::AppConfig;
use crate::error::LensError;

/// Connect timeout for an LLM HTTP request (matches the system-check probe).
const LLM_CONNECT_TIMEOUT: Duration = Duration::from_secs(1);
/// Overall (read) timeout for an LLM generate/probe request.
///
/// Mirrors `ENRICHMENT_LLM_TIMEOUT_SECS` from the plan (Decision C); a generate
/// call can legitimately take many seconds, so this is far longer than the
/// 2s system-check probe window.
const LLM_TIMEOUT: Duration = Duration::from_secs(30);
/// Upper bound on a response body we will buffer + deserialize. Defense-in-depth
/// against a malicious/misconfigured endpoint streaming an unbounded body
/// (mirrors `system_check::MAX_PROBE_BODY_BYTES` but sized for completions).
const MAX_LLM_BODY_BYTES: usize = 8 * 1024 * 1024;
/// `anthropic-version` header value sent on every Anthropic request.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Canonical provider identifiers (match `ModelConfig.provider` /
/// the onboarding `LlmProviderInput.provider` strings).
const PROVIDER_OLLAMA: &str = "ollama";
const PROVIDER_OPENAI_COMPAT: &str = "openai-compatible";
const PROVIDER_ANTHROPIC: &str = "anthropic";

/// A single completion request to an [`LlmProvider`].
///
/// `system` is an optional system/instruction prompt; `prompt` is the user
/// turn; `max_tokens` caps the generated output. `temperature` and `json` are
/// the determinism knobs threaded into every backend so enrichment can pin
/// reproducible, machine-parseable output (the enrichment callers pass
/// `temperature: 0.0, json: true`).
///
/// `temperature` is `f32`, so the struct uses `PartialEq` only (no `Eq`/`Hash`):
/// `LlmRequest` is a transient request value, never a map key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmRequest {
    /// Optional system/instruction prompt.
    pub system: Option<String>,
    /// The user prompt.
    pub prompt: String,
    /// Maximum output tokens to generate.
    pub max_tokens: u32,
    /// Sampling temperature. `0.0` requests greedy/deterministic decoding;
    /// enrichment callers pin this to `0.0`. Sent on every backend.
    pub temperature: f32,
    /// Request strict JSON output. When `true`, each backend asks for JSON mode
    /// where supported (Ollama `format:"json"`, OpenAI `response_format`); the
    /// Anthropic API has NO such param, so JSON is enforced only via the prompt
    /// there (the strict serde parse + retries remain the real guard).
    pub json: bool,
}

/// A completion response from an [`LlmProvider`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmResponse {
    /// The generated text.
    pub text: String,
    /// Total tokens billed/consumed by the call (input + output where the
    /// provider reports it), used to drive the enrichment budget counters.
    pub tokens_used: u32,
}

/// An async, object-safe LLM backend.
///
/// `Send + Sync` and behind `Arc<dyn LlmProvider>` so the engine can hold the
/// active provider in `Arc<RwLock<Option<Arc<dyn LlmProvider>>>>` (rebindable on
/// an unreachable→reachable transition; see the plan's ratified deviation) and
/// the background worker can dispatch against the trait object.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// The model id this provider generates with. Stable for the lifetime of the
    /// provider; a component of the enrichment composite cache key (AC9).
    fn model_id(&self) -> &str;

    /// `system_check`-style reachability probe. `true` when the endpoint is
    /// usable for generation; `false` on a connection refusal, DNS/timeout
    /// failure, OR an auth error (`401`/`403`) — a misconfigured key counts as
    /// unreachable so the source degrades to raw vectors instead of looping.
    async fn reachable(&self) -> bool;

    /// Generate a completion. Returns [`LensError::Network`] on transport
    /// failure, [`LensError::Model`] on a non-success HTTP status, and
    /// [`LensError::Parse`] when the response body can't be decoded.
    async fn generate(&self, req: &LlmRequest) -> Result<LlmResponse, LensError>;
}

/// Builds the shared HTTP client for LLM calls: bounded connect/read timeouts +
/// the same SSRF hardening as the system-check probe (never follow a redirect).
fn llm_client_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .connect_timeout(LLM_CONNECT_TIMEOUT)
        .timeout(LLM_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
}

/// Builds the LLM HTTP client, degrading to a default client if the (pure-Rust
/// rustls) TLS backend somehow fails to initialize — never panics.
fn llm_client() -> reqwest::Client {
    llm_client_builder()
        .build()
        .unwrap_or_else(|_| llm_client_builder().build().unwrap_or_default())
}

/// Maps a reqwest send/transport error into [`LensError::Network`].
fn network_err(err: reqwest::Error) -> LensError {
    LensError::Network(err.to_string())
}

/// Reads a successful response body (stream-and-cap) and decodes it as `T`.
///
/// The body is read INCREMENTALLY via `resp.chunk()` and aborted the instant the
/// running total would exceed [`MAX_LLM_BODY_BYTES`] — a misconfigured/hostile
/// endpoint can never inflate the buffer past the cap (plus one in-flight chunk).
/// If `Content-Length` is present and already over the cap, we reject before
/// reading a single byte.
///
/// A non-success status → [`LensError::Model`]; a transport read error →
/// [`LensError::Network`]; an over-cap body or a decode miss →
/// [`LensError::Parse`].
async fn read_json_capped<T: serde::de::DeserializeOwned>(
    mut resp: reqwest::Response,
) -> Result<T, LensError> {
    let status = resp.status();
    if !status.is_success() {
        return Err(LensError::Model(format!("LLM returned HTTP {status}")));
    }
    // Reject early on an advertised length that already exceeds the cap, before
    // reading any body bytes at all.
    if let Some(len) = resp.content_length()
        && len > MAX_LLM_BODY_BYTES as u64
    {
        return Err(LensError::Parse("LLM response body exceeded cap".into()));
    }
    let mut body: Vec<u8> = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(network_err)? {
        // Abort as soon as the running total would exceed the cap — never let the
        // buffer grow unbounded behind a streamed body.
        if body.len() + chunk.len() > MAX_LLM_BODY_BYTES {
            return Err(LensError::Parse("LLM response body exceeded cap".into()));
        }
        body.extend_from_slice(&chunk);
    }
    serde_json::from_slice::<T>(&body).map_err(LensError::from)
}

/// Classifies a reachability HEAD/GET/POST response into a `bool` per Decision G.
///
/// `true` for any non-error HTTP response that isn't an auth failure; `false`
/// for `401`/`403`. Connection/timeout failures are handled by the caller
/// (the `Result::Err` arm) — they map to `false`.
fn response_is_reachable(status: reqwest::StatusCode) -> bool {
    !(status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN)
}

// ---------------------------------------------------------------------------
// Ollama — POST /api/chat (stream:false); reachability GET /api/version
// ---------------------------------------------------------------------------

/// Ollama backend. Generates via `POST /api/chat` (`stream:false`); reachability
/// reuses the existing `GET /api/version` probe shape.
pub struct OllamaProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaProvider {
    /// Builds an Ollama provider for `model` at `base_url` (trailing slash trimmed).
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            client: llm_client(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }
}

/// Ollama `/api/chat` (`stream:false`) response shape (the fields we read).
#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    message: OllamaChatMessage,
    #[serde(default)]
    prompt_eval_count: u32,
    #[serde(default)]
    eval_count: u32,
}

#[derive(Debug, Deserialize)]
struct OllamaChatMessage {
    #[serde(default)]
    content: String,
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    fn model_id(&self) -> &str {
        &self.model
    }

    async fn reachable(&self) -> bool {
        let url = format!("{}/api/version", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => response_is_reachable(resp.status()),
            Err(_) => false,
        }
    }

    async fn generate(&self, req: &LlmRequest) -> Result<LlmResponse, LensError> {
        let url = format!("{}/api/chat", self.base_url);
        let mut messages = Vec::with_capacity(2);
        if let Some(system) = &req.system {
            messages.push(serde_json::json!({ "role": "system", "content": system }));
        }
        messages.push(serde_json::json!({ "role": "user", "content": req.prompt }));

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": false,
            "options": {
                "num_predict": req.max_tokens,
                "temperature": req.temperature,
            },
        });
        // Ollama JSON mode is a top-level `format:"json"` constraint.
        if req.json {
            body["format"] = serde_json::Value::String("json".to_string());
        }

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(network_err)?;

        let parsed: OllamaChatResponse = read_json_capped(resp).await?;
        Ok(LlmResponse {
            text: parsed.message.content,
            tokens_used: parsed.prompt_eval_count.saturating_add(parsed.eval_count),
        })
    }
}

// ---------------------------------------------------------------------------
// OpenAI-compatible — POST /v1/chat/completions; reachability GET /v1/models
// ---------------------------------------------------------------------------

/// OpenAI-compatible backend (OpenAI / GLM / LM Studio). Generates via
/// `POST /v1/chat/completions` with a `Bearer` key (when configured);
/// reachability uses `GET /v1/models`.
pub struct OpenAiCompatProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: String,
}

impl OpenAiCompatProvider {
    /// Builds an OpenAI-compatible provider. `api_key` may be empty for a local
    /// runtime (e.g. LM Studio) that needs no auth.
    pub fn new(base_url: &str, model: &str, api_key: &str) -> Self {
        Self {
            client: llm_client(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key: api_key.to_string(),
        }
    }

    /// Applies the `Authorization: Bearer` header when a key is configured.
    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if self.api_key.is_empty() {
            req
        } else {
            req.bearer_auth(&self.api_key)
        }
    }
}

/// OpenAI `/v1/chat/completions` response shape (the fields we read).
#[derive(Debug, Deserialize)]
struct OpenAiChatResponse {
    #[serde(default)]
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    #[serde(default)]
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    #[serde(default)]
    total_tokens: u32,
}

#[async_trait]
impl LlmProvider for OpenAiCompatProvider {
    fn model_id(&self) -> &str {
        &self.model
    }

    async fn reachable(&self) -> bool {
        let url = format!("{}/v1/models", self.base_url);
        match self.auth(self.client.get(&url)).send().await {
            Ok(resp) => response_is_reachable(resp.status()),
            Err(_) => false,
        }
    }

    async fn generate(&self, req: &LlmRequest) -> Result<LlmResponse, LensError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let mut messages = Vec::with_capacity(2);
        if let Some(system) = &req.system {
            messages.push(serde_json::json!({ "role": "system", "content": system }));
        }
        messages.push(serde_json::json!({ "role": "user", "content": req.prompt }));

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        });
        // OpenAI-compatible JSON mode.
        if req.json {
            body["response_format"] = serde_json::json!({ "type": "json_object" });
        }

        let resp = self
            .auth(self.client.post(&url))
            .json(&body)
            .send()
            .await
            .map_err(network_err)?;

        let parsed: OpenAiChatResponse = read_json_capped(resp).await?;
        let text = parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();
        Ok(LlmResponse {
            text,
            tokens_used: parsed.usage.map(|u| u.total_tokens).unwrap_or(0),
        })
    }
}

// ---------------------------------------------------------------------------
// Anthropic — POST /v1/messages; reachability = max_tokens=1 ping (Decision G)
// ---------------------------------------------------------------------------

/// Anthropic backend. Generates via `POST /v1/messages` with the `x-api-key` +
/// `anthropic-version` headers. There is NO `GET /v1/models`, so reachability is
/// a `max_tokens=1` `POST /v1/messages` ping (Decision G): any non-connection
/// error that isn't a `401`/`403` → reachable.
pub struct AnthropicProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: String,
}

impl AnthropicProvider {
    /// Builds an Anthropic provider for `model` at `base_url`.
    pub fn new(base_url: &str, model: &str, api_key: &str) -> Self {
        Self {
            client: llm_client(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key: api_key.to_string(),
        }
    }

    /// Adds the Anthropic auth + version headers to a request.
    fn headers(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        req.header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
    }
}

/// Anthropic `/v1/messages` response shape (the fields we read).
#[derive(Debug, Deserialize)]
struct AnthropicMessageResponse {
    #[serde(default)]
    content: Vec<AnthropicContentBlock>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn model_id(&self) -> &str {
        &self.model
    }

    async fn reachable(&self) -> bool {
        // No GET /v1/models on Anthropic — ping with a max_tokens=1 message.
        let url = format!("{}/v1/messages", self.base_url);
        // A minimal valid request: max_tokens=1, temperature 0.0, no JSON mode
        // (Anthropic has no json param anyway — see `generate`).
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": 1,
            "temperature": 0.0,
            "messages": [{ "role": "user", "content": "ping" }],
        });
        match self
            .headers(self.client.post(&url))
            .json(&body)
            .send()
            .await
        {
            Ok(resp) => response_is_reachable(resp.status()),
            Err(_) => false,
        }
    }

    async fn generate(&self, req: &LlmRequest) -> Result<LlmResponse, LensError> {
        let url = format!("{}/v1/messages", self.base_url);
        // Anthropic exposes a top-level `temperature` but has NO JSON-mode /
        // `response_format` param — `req.json` is honored only via the prompt
        // here (the strict serde parse + reprompts in `enrichment::map` are the
        // real guard), so we never send an unsupported field.
        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": [{ "role": "user", "content": req.prompt }],
        });
        if let Some(system) = &req.system {
            body["system"] = serde_json::Value::String(system.clone());
        }

        let resp = self
            .headers(self.client.post(&url))
            .json(&body)
            .send()
            .await
            .map_err(network_err)?;

        let parsed: AnthropicMessageResponse = read_json_capped(resp).await?;
        let text = parsed
            .content
            .into_iter()
            .map(|b| b.text)
            .collect::<String>();
        let tokens_used = parsed
            .usage
            .map(|u| u.input_tokens.saturating_add(u.output_tokens))
            .unwrap_or(0);
        Ok(LlmResponse { text, tokens_used })
    }
}

// ---------------------------------------------------------------------------
// Factory (Decision H)
// ---------------------------------------------------------------------------

/// Builds the enrichment [`LlmProvider`] from `config.models[]` (Decision H).
///
/// Returns the FIRST configured model entry whose provider is recognized; cloud
/// providers (`openai-compatible`, `anthropic`) require `cloud_consent=true` or
/// they are skipped. Returns `None` when no usable entry exists.
///
/// This does NOT probe reachability — the caller (the worker / `init`) calls
/// [`LlmProvider::reachable`] on the returned provider to gate dispatch. Building
/// the provider is cheap (an HTTP client) and the reachable check is the gate.
///
/// **Step-boundary note:** the plan's `provider_from_config(&AppConfig)`
/// signature reads the cloud-consent flag from `AppConfig.enrichment`, which is
/// introduced in Step 6. To keep Step 2 from colliding with that struct, the
/// consent flag is threaded in as a parameter here; Step 3/6 pass
/// `config.enrichment.cloud_consent`.
pub fn provider_from_config(
    config: &AppConfig,
    cloud_consent: bool,
) -> Option<Arc<dyn LlmProvider>> {
    config
        .models
        .iter()
        .find_map(|m| build_provider(m, cloud_consent))
}

/// Builds a provider for a single [`crate::config::ModelConfig`] entry, applying
/// the recognized-provider + cloud-consent gates. Returns `None` to skip.
fn build_provider(
    model: &crate::config::ModelConfig,
    cloud_consent: bool,
) -> Option<Arc<dyn LlmProvider>> {
    if model.base_url.is_empty() || model.model.is_empty() {
        return None;
    }
    let provider = model.provider.to_ascii_lowercase();
    match provider.as_str() {
        PROVIDER_OLLAMA => Some(Arc::new(OllamaProvider::new(&model.base_url, &model.model))),
        PROVIDER_OPENAI_COMPAT if cloud_consent => Some(Arc::new(OpenAiCompatProvider::new(
            &model.base_url,
            &model.model,
            &model.api_key,
        ))),
        PROVIDER_ANTHROPIC if cloud_consent => Some(Arc::new(AnthropicProvider::new(
            &model.base_url,
            &model.model,
            &model.api_key,
        ))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::ModelConfig;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// A fixed always-refused localhost port. Nothing binds `127.0.0.1:1`, so the
    /// connection is deterministically refused (the parallel-test-safe pattern
    /// from `system_check.rs`).
    const DEAD_URL: &str = "http://127.0.0.1:1";

    fn req() -> LlmRequest {
        LlmRequest {
            system: Some("be terse".to_string()),
            prompt: "hello".to_string(),
            max_tokens: 64,
            // The enrichment determinism defaults: greedy decode + JSON mode.
            temperature: 0.0,
            json: true,
        }
    }

    // --- Ollama (drives the Arc<dyn LlmProvider> trait object) --------------

    #[tokio::test]
    async fn ollama_reachable_true_on_version_ok() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/version"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "version": "0.4.1"
            })))
            .mount(&server)
            .await;

        let provider: Arc<dyn LlmProvider> = Arc::new(OllamaProvider::new(&server.uri(), "llama3"));
        assert!(provider.reachable().await);
    }

    #[tokio::test]
    async fn ollama_reachable_false_on_connection_refused() {
        let provider: Arc<dyn LlmProvider> = Arc::new(OllamaProvider::new(DEAD_URL, "llama3"));
        assert!(!provider.reachable().await);
    }

    #[tokio::test]
    async fn ollama_generate_parses_completion() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            // The determinism knobs must be on the wire: temperature in
            // `options` and the top-level `format:"json"` when json-mode is set.
            .and(body_partial_json(serde_json::json!({
                "options": { "temperature": 0.0 },
                "format": "json"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "message": { "role": "assistant", "content": "hi there" },
                "prompt_eval_count": 10,
                "eval_count": 5
            })))
            .mount(&server)
            .await;

        let provider: Arc<dyn LlmProvider> = Arc::new(OllamaProvider::new(&server.uri(), "llama3"));
        let resp = provider.generate(&req()).await.unwrap();
        assert_eq!(resp.text, "hi there");
        assert_eq!(resp.tokens_used, 15);
    }

    // --- OpenAI-compatible --------------------------------------------------

    #[tokio::test]
    async fn openai_reachable_true_on_models_ok() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "id": "gpt-4o" }]
            })))
            .mount(&server)
            .await;

        let provider: Arc<dyn LlmProvider> = Arc::new(OpenAiCompatProvider::new(
            &server.uri(),
            "gpt-4o",
            "sk-test",
        ));
        assert!(provider.reachable().await);
    }

    #[tokio::test]
    async fn openai_reachable_false_on_401_auth() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let provider: Arc<dyn LlmProvider> = Arc::new(OpenAiCompatProvider::new(
            &server.uri(),
            "gpt-4o",
            "bad-key",
        ));
        assert!(!provider.reachable().await);
    }

    #[tokio::test]
    async fn openai_reachable_false_on_connection_refused() {
        let provider: Arc<dyn LlmProvider> =
            Arc::new(OpenAiCompatProvider::new(DEAD_URL, "gpt-4o", "sk-test"));
        assert!(!provider.reachable().await);
    }

    #[tokio::test]
    async fn openai_generate_parses_completion() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            // temperature + `response_format: json_object` must be on the wire.
            .and(body_partial_json(serde_json::json!({
                "temperature": 0.0,
                "response_format": { "type": "json_object" }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{ "message": { "role": "assistant", "content": "the answer" } }],
                "usage": { "total_tokens": 42 }
            })))
            .mount(&server)
            .await;

        let provider: Arc<dyn LlmProvider> = Arc::new(OpenAiCompatProvider::new(
            &server.uri(),
            "gpt-4o",
            "sk-test",
        ));
        let resp = provider.generate(&req()).await.unwrap();
        assert_eq!(resp.text, "the answer");
        assert_eq!(resp.tokens_used, 42);
    }

    // --- Anthropic (max_tokens=1 ping reachability — Decision G) ------------

    #[tokio::test]
    async fn anthropic_reachable_true_on_messages_ok() {
        let server = MockServer::start().await;
        // The max_tokens=1 ping returns a stubbed message → reachable.
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("anthropic-version", ANTHROPIC_VERSION))
            .and(header("x-api-key", "sk-ant"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{ "type": "text", "text": "ping" }],
                "usage": { "input_tokens": 1, "output_tokens": 1 }
            })))
            .mount(&server)
            .await;

        let provider: Arc<dyn LlmProvider> = Arc::new(AnthropicProvider::new(
            &server.uri(),
            "claude-opus-4-8",
            "sk-ant",
        ));
        assert!(provider.reachable().await);
    }

    #[tokio::test]
    async fn anthropic_reachable_false_on_401_auth() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "type": "error",
                "error": { "type": "authentication_error", "message": "invalid x-api-key" }
            })))
            .mount(&server)
            .await;

        let provider: Arc<dyn LlmProvider> = Arc::new(AnthropicProvider::new(
            &server.uri(),
            "claude-opus-4-8",
            "bad-key",
        ));
        assert!(!provider.reachable().await);
    }

    #[tokio::test]
    async fn anthropic_reachable_false_on_connection_refused() {
        let provider: Arc<dyn LlmProvider> = Arc::new(AnthropicProvider::new(
            DEAD_URL,
            "claude-opus-4-8",
            "sk-ant",
        ));
        assert!(!provider.reachable().await);
    }

    #[tokio::test]
    async fn anthropic_generate_parses_completion() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "sk-ant"))
            // Anthropic gets a top-level temperature.
            .and(body_partial_json(serde_json::json!({ "temperature": 0.0 })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [
                    { "type": "text", "text": "structured " },
                    { "type": "text", "text": "map" }
                ],
                "usage": { "input_tokens": 30, "output_tokens": 12 }
            })))
            .mount(&server)
            .await;

        let provider: Arc<dyn LlmProvider> = Arc::new(AnthropicProvider::new(
            &server.uri(),
            "claude-opus-4-8",
            "sk-ant",
        ));
        let resp = provider.generate(&req()).await.unwrap();
        assert_eq!(resp.text, "structured map");
        assert_eq!(resp.tokens_used, 42);

        // Anthropic has NO json-mode param even when `req.json` is true — assert
        // the body never carries `response_format`/`format` (relies on the prompt).
        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert!(body.get("response_format").is_none());
        assert!(body.get("format").is_none());
        assert_eq!(body["temperature"], serde_json::json!(0.0));
    }

    #[tokio::test]
    async fn generate_non_success_status_is_model_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let provider: Arc<dyn LlmProvider> = Arc::new(OllamaProvider::new(&server.uri(), "llama3"));
        let err = provider.generate(&req()).await.unwrap_err();
        assert!(matches!(err, LensError::Model(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn generate_over_cap_body_is_rejected() {
        // Serve a body larger than MAX_LLM_BODY_BYTES; the stream-and-cap reader
        // must abort with a Parse error rather than buffering the whole thing.
        let server = MockServer::start().await;
        let oversized = "x".repeat(MAX_LLM_BODY_BYTES + 1024);
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_string(oversized))
            .mount(&server)
            .await;

        let provider: Arc<dyn LlmProvider> = Arc::new(OllamaProvider::new(&server.uri(), "llama3"));
        let err = provider.generate(&req()).await.unwrap_err();
        assert!(matches!(err, LensError::Parse(_)), "got {err:?}");
    }

    // --- model_id accessor (cache-key component, AC9) -----------------------

    #[test]
    fn model_id_returns_configured_model() {
        let ollama = OllamaProvider::new("http://localhost:11434", "llama3.1");
        assert_eq!(ollama.model_id(), "llama3.1");
        let anthropic = AnthropicProvider::new("https://api.anthropic.com", "claude-opus-4-8", "k");
        assert_eq!(anthropic.model_id(), "claude-opus-4-8");
    }

    // --- factory (Decision H) -----------------------------------------------

    fn config_with(models: Vec<ModelConfig>) -> AppConfig {
        AppConfig {
            models,
            ..AppConfig::default()
        }
    }

    #[test]
    fn factory_selects_local_ollama_without_consent() {
        let cfg = config_with(vec![ModelConfig {
            provider: "ollama".to_string(),
            base_url: "http://localhost:11434".to_string(),
            model: "llama3".to_string(),
            ..ModelConfig::default()
        }]);
        let provider = provider_from_config(&cfg, false).expect("ollama selected");
        assert_eq!(provider.model_id(), "llama3");
    }

    #[test]
    fn factory_rejects_cloud_without_consent() {
        let cfg = config_with(vec![
            ModelConfig {
                provider: "anthropic".to_string(),
                base_url: "https://api.anthropic.com".to_string(),
                model: "claude-opus-4-8".to_string(),
                api_key: "sk-ant".to_string(),
                ..ModelConfig::default()
            },
            ModelConfig {
                provider: "openai-compatible".to_string(),
                base_url: "https://api.openai.com".to_string(),
                model: "gpt-4o".to_string(),
                api_key: "sk-oai".to_string(),
                ..ModelConfig::default()
            },
        ]);
        assert!(provider_from_config(&cfg, false).is_none());
    }

    #[test]
    fn factory_selects_cloud_with_consent() {
        let cfg = config_with(vec![ModelConfig {
            provider: "anthropic".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            model: "claude-opus-4-8".to_string(),
            api_key: "sk-ant".to_string(),
            ..ModelConfig::default()
        }]);
        let provider = provider_from_config(&cfg, true).expect("anthropic selected with consent");
        assert_eq!(provider.model_id(), "claude-opus-4-8");
    }

    #[test]
    fn factory_skips_incomplete_and_unknown_entries() {
        let cfg = config_with(vec![
            // Unknown provider → skipped.
            ModelConfig {
                provider: "mystery".to_string(),
                base_url: "http://x".to_string(),
                model: "m".to_string(),
                ..ModelConfig::default()
            },
            // Missing model → skipped.
            ModelConfig {
                provider: "ollama".to_string(),
                base_url: "http://localhost:11434".to_string(),
                model: String::new(),
                ..ModelConfig::default()
            },
            // Valid ollama → selected.
            ModelConfig {
                provider: "ollama".to_string(),
                base_url: "http://localhost:11434".to_string(),
                model: "llama3".to_string(),
                ..ModelConfig::default()
            },
        ]);
        let provider = provider_from_config(&cfg, false).expect("third entry selected");
        assert_eq!(provider.model_id(), "llama3");
    }

    #[test]
    fn factory_none_when_no_models() {
        assert!(provider_from_config(&AppConfig::default(), true).is_none());
    }
}
