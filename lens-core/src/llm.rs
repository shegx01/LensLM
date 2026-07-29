//! LLM provider seam for the M4 Phase-3 enrichment pass.
//!
//! Defines [`LlmProvider`] (object-safe, `Arc<dyn LlmProvider>`), a typed routing policy
//! ([`LlmRouting`]), and the [`provider_from_config`] factory. Providers are constructed with our
//! hardened reqwest client so SSRF policy and timeouts carry over; enrichment pins
//! `temperature: 0.0 + json: true` for deterministic output. The default
//! [`LlmProvider::generate_stream`] lets enrichment mocks (which only implement the three core
//! methods) compile untouched.
//!
//! genai → rig migration (epic #255): [`RigProvider`] over the `rig-core` crate is the DEFAULT
//! backend (Phase 2, #258); `--no-default-features` selects the legacy [`GenaiProvider`] over the
//! `genai` crate as the rollback (removed in Phase 3, #259).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::Stream;
use genai::Client;
use genai::ModelIden;
use genai::adapter::AdapterKind;
use genai::chat::{
    ChatMessage, ChatOptions, ChatRequest, ChatResponseFormat, ReasoningEffort as GenaiEffort,
};
use genai::resolver::{AuthData, Endpoint};
use genai::{ModelSpec, ServiceTarget};
use serde::{Deserialize, Serialize};

use crate::config::AppConfig;
use crate::error::LensError;
use crate::model_catalog::SupportedProvider;

/// Connect timeout for LLM HTTP requests (matches the system-check probe).
const LLM_CONNECT_TIMEOUT: Duration = Duration::from_secs(1);
/// Total timeout for the cheap unauthenticated Ollama `api/version` reachability probe.
const LLM_TIMEOUT: Duration = Duration::from_secs(30);
/// Idle read timeout for LLM generation: resets on each received chunk, not a total-
/// request deadline, so unbounded streaming on a small local model never times out —
/// yet a stalled/unreachable model still fails. Also bounds a buffered `generate`.
const LLM_GENERATION_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// Canonical provider ids matching `ModelConfig.provider`. First-class cloud providers use their
/// models.dev catalog key; `openai-compatible` covers self-hosted OpenAI-protocol endpoints
/// (LM Studio, proxies) where the user supplies the base URL and models are arbitrary.
const PROVIDER_OLLAMA: &str = "ollama";
const PROVIDER_OPENAI_COMPAT: &str = "openai-compatible";
const PROVIDER_OPENAI: &str = "openai";
const PROVIDER_ANTHROPIC: &str = "anthropic";
const PROVIDER_GOOGLE: &str = "google";
const PROVIDER_GLM: &str = "glm";
const PROVIDER_ZAI: &str = "zai";
const PROVIDER_OLLAMA_CLOUD: &str = "ollama-cloud";
const PROVIDER_GROQ: &str = "groq";
const PROVIDER_DEEPSEEK: &str = "deepseek";
const PROVIDER_XAI: &str = "xai";
const PROVIDER_COHERE: &str = "cohere";

/// Serde-stable mirror of genai's `ReasoningEffort` so the trait API and IPC shape
/// never leak a genai type. Enrichment never sets this; M5 chat opts in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    /// Light reasoning budget.
    Low,
    /// Balanced reasoning budget.
    Medium,
    /// Heavy reasoning budget.
    High,
}

impl ReasoningEffort {
    fn to_genai(self) -> GenaiEffort {
        match self {
            ReasoningEffort::Low => GenaiEffort::Low,
            ReasoningEffort::Medium => GenaiEffort::Medium,
            ReasoningEffort::High => GenaiEffort::High,
        }
    }
}

/// One prior conversation turn fed into a completion request as context (Plan 2 /
/// CX-1). Role reuses [`crate::chat::ChatRole`] so the wire strings stay `user`/
/// `assistant` and no stringly-typed role leaks in. Ordered oldest→newest; assembled
/// between the system message and the final user `prompt` in [`GenaiProvider::map_request`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: crate::chat::ChatRole,
    pub content: String,
}

/// A single completion request to an [`LlmProvider`].
/// `temperature` is `f32`, so only `PartialEq` (no `Eq`/`Hash`): transient value, never a
/// map key. Enrichment pins `temperature: 0.0, json: true, thinking: false`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmRequest {
    pub system: Option<String>,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub json: bool,
    /// Defaults to `false`; older IPC payloads without this key read back as `false` via
    /// `#[serde(default)]`. Enrichment keeps this OFF; M5 chat opts in.
    #[serde(default)]
    pub thinking: bool,
    /// Reasoning budget when `thinking` is `true`. Older payloads without this key read
    /// back as `None`.
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Prior conversation turns (oldest→newest) injected before the final user
    /// `prompt`. Empty for enrichment (single-shot) and legacy payloads via
    /// `#[serde(default)]`; chat populates it from the persisted transcript.
    #[serde(default)]
    pub messages: Vec<LlmMessage>,
}

/// A completion response from an [`LlmProvider`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmResponse {
    pub text: String,
    /// Input + output tokens consumed (where reported); drives enrichment budget counters.
    pub tokens_used: u32,
}

/// One event from a streamed generation ([`LlmProvider::generate_stream`]).
/// genai's richer stream (`Start`/`ToolCallChunk`/…) is collapsed onto these three
/// so the trait stays provider-agnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamChunk {
    TextDelta(String),
    ThinkingDelta(String),
    /// Terminal event. `tokens_used` is `0` when the provider did not report usage.
    Done {
        tokens_used: u32,
    },
}

/// An async, object-safe LLM backend held behind `Arc<dyn LlmProvider>`.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Stable model id; a component of the enrichment composite cache key (AC9).
    fn model_id(&self) -> &str;

    /// The hardened `reqwest::Client` this provider was built with, exposed so
    /// [`task_provider_from_config`] can build sibling per-task providers over the SAME client
    /// (shared connection pool) without downcasting to a concrete type. The default `None` makes
    /// mocks and clients that can't share fall back to reusing the base provider unchanged.
    fn shared_http_client(&self) -> Option<reqwest::Client> {
        None
    }

    /// Whether this provider runs on-device (local Ollama). Lets callers relax limits
    /// small local models can't meet — e.g. the dialogue min-turns floor (#26) — while
    /// keeping cloud strict.
    fn is_local(&self) -> bool {
        false
    }

    /// Whether this provider targets a local Ollama runtime. The enrichment preflight
    /// (#90) and system check run Ollama's `/api/tags` model-installed check off this
    /// capability, not a concrete-type downcast, so it fires regardless of backend.
    fn is_ollama(&self) -> bool {
        false
    }

    /// Reachability probe. `false` on connection refusal, DNS/timeout, or auth errors
    /// (`401`/`403`) — a misconfigured key is unreachable so sources degrade gracefully.
    async fn reachable(&self) -> bool;

    async fn generate(&self, req: &LlmRequest) -> Result<LlmResponse, LensError>;

    /// Stream a completion ending in [`StreamChunk::Done`]. Enrichment never streams (uses
    /// the deterministic `generate` path). The default buffers `generate` into a single
    /// `TextDelta + Done` so enrichment mocks compile without changes.
    async fn generate_stream(
        &self,
        req: &LlmRequest,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<StreamChunk, LensError>> + Send>>,
        LensError,
    > {
        let resp = self.generate(req).await?;
        let chunks = vec![
            Ok(StreamChunk::TextDelta(resp.text)),
            Ok(StreamChunk::Done {
                tokens_used: resp.tokens_used,
            }),
        ];
        Ok(Box::pin(futures_util::stream::iter(chunks)))
    }
}

fn llm_client() -> reqwest::Client {
    crate::http::hardened_client_idle(LLM_CONNECT_TIMEOUT, LLM_GENERATION_IDLE_TIMEOUT)
}

/// Whether an error's text reads like a transport failure (connection/timeout/dns) rather than a
/// model/semantic error. reqwest's transport-failure `Display` often lacks "timeout"/"connect": a
/// send failure reads "error sending request", an idle read-timeout "error reading
/// response"/"body", a deadline "deadline". Matching those keeps a genuine transport error from
/// being misclassified as a model (bad-output) error. Shared by [`genai_err`] and the rig backend.
fn looks_like_transport(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("connect")
        || lower.contains("connection")
        || lower.contains("timed out")
        || lower.contains("timeout")
        || lower.contains("dns")
        || lower.contains("refused")
        || lower.contains("sending request")
        || lower.contains("reading response")
        || lower.contains("response body")
        || lower.contains("deadline")
}

/// Maps a genai error onto [`LensError`], sanitizing the message before it crosses the IPC
/// boundary. genai wraps transport errors inside its own types with no public `reqwest::Error`
/// accessor, so we classify by `Display` text (connect/timeout → `Network`; everything else
/// → `Model`). The full error is logged server-side; only a generic message is surfaced over IPC.
fn genai_err(err: genai::Error) -> LensError {
    let is_transport = looks_like_transport(&err.to_string());
    // Log the full detail for operators; never surface it across IPC.
    tracing::error!(error = %err, transport = is_transport, "LLM request failed");
    if is_transport {
        LensError::Network(
            "couldn't reach the language model — check that your LLM provider \
             (e.g. local Ollama) is running and reachable"
                .to_string(),
        )
    } else {
        LensError::Model("LLM request failed (model)".to_string())
    }
}

/// Resolved genai [`ServiceTarget`] plus metadata for the trait accessor and `reachable` probe.
#[derive(Clone)]
struct ResolvedTarget {
    target: ServiceTarget,
    model_id: String,
    adapter: AdapterKind,
    /// Always ends in `/`. For local Ollama, the `api/version` probe appends to this base.
    endpoint_base: String,
    /// Whether a non-empty API key was configured (cloud reachability signal).
    has_key: bool,
}

/// The single LLM backend. Every call pins the fully-resolved `ServiceTarget` via
/// `ModelSpec::Target`; the provider/model is never re-inferred from the model name.
pub struct GenaiProvider {
    client: Client,
    resolved: ResolvedTarget,
    /// The hardened reqwest client backing `client`, kept so `shared_http_client` can hand it to
    /// sibling per-task providers (genai's `Client` does not re-expose its inner reqwest).
    http: reqwest::Client,
}

/// Normalizes a `base_url` into the endpoint base genai expects. genai concatenates a relative
/// path onto this base, so it must end in `/`. OpenAI/Anthropic adapters also need `/v1/`
/// (they append `chat/completions` / `messages` after the version segment); Ollama only needs
/// a trailing slash. Unused under `llm-backend-rig` (rig owns endpoint handling there).
#[cfg_attr(feature = "llm-backend-rig", allow(dead_code))]
fn normalize_endpoint(adapter: AdapterKind, base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    let needs_v1 = matches!(adapter, AdapterKind::OpenAI | AdapterKind::Anthropic);
    // Don't double `/v1` when the base already ends in it.
    if needs_v1 && !trimmed.ends_with("/v1") {
        format!("{trimmed}/v1/")
    } else {
        format!("{trimmed}/")
    }
}

/// Returns the canonical public endpoint for a native cloud adapter.
/// genai bakes endpoints into each native adapter but exposes no public accessor, so we mirror
/// them here. A configured non-empty `base_url` still wins (explicit override). Returns `None`
/// for `Ollama` and `openai-compatible` where the URL is always user-supplied.
/// **Pinned to genai 0.6.5.** On a bump, verify against
/// `grep 'const BASE_URL' <genai>/src/adapter/adapters/*/adapter_impl.rs`.
fn native_endpoint(adapter: AdapterKind) -> Option<Endpoint> {
    match adapter {
        AdapterKind::OpenAI => Some(Endpoint::from_static("https://api.openai.com/v1/")),
        AdapterKind::Anthropic => Some(Endpoint::from_static("https://api.anthropic.com/v1/")),
        AdapterKind::Gemini => Some(Endpoint::from_static(
            "https://generativelanguage.googleapis.com/v1beta/",
        )),
        AdapterKind::Groq => Some(Endpoint::from_static("https://api.groq.com/openai/v1/")),
        AdapterKind::DeepSeek => Some(Endpoint::from_static("https://api.deepseek.com/v1/")),
        AdapterKind::Xai => Some(Endpoint::from_static("https://api.x.ai/v1/")),
        AdapterKind::Cohere => Some(Endpoint::from_static("https://api.cohere.com/v1/")),
        AdapterKind::Zai => Some(Endpoint::from_static("https://api.z.ai/api/paas/v4/")),
        AdapterKind::OllamaCloud => Some(Endpoint::from_static("https://ollama.com/")),
        _ => None, // local Ollama / openai-compatible: URL is always user-supplied
    }
}

impl GenaiProvider {
    /// Builds a provider with its own hardened client. Lib code goes through
    /// [`new_with_http`](Self::new_with_http) (via the construction seam); this stays as a
    /// terse test constructor.
    #[cfg(test)]
    fn new(adapter: AdapterKind, model: &str, base_url: &str, api_key: &str) -> Self {
        Self::new_with_http(llm_client(), adapter, model, base_url, api_key)
    }

    /// Builds a provider over a given hardened reqwest client (only the pinned target differs),
    /// so sibling per-task providers reuse one connection pool. Unused under `llm-backend-rig`
    /// (rig is the active backend there) but retained as the default backend.
    #[cfg_attr(feature = "llm-backend-rig", allow(dead_code))]
    fn new_with_http(
        http: reqwest::Client,
        adapter: AdapterKind,
        model: &str,
        base_url: &str,
        api_key: &str,
    ) -> Self {
        let client = Client::builder().with_reqwest(http.clone()).build();
        let model_iden = ModelIden::new(adapter, model.to_string());
        // Configured base_url wins (custom/self-hosted or explicit override). With no base_url,
        // a native cloud adapter falls back to its canonical endpoint; otherwise normalize an
        // empty base so construction stays infallible.
        let normalized = normalize_endpoint(adapter, base_url);
        let endpoint = if base_url.is_empty() {
            native_endpoint(adapter).unwrap_or_else(|| Endpoint::from_owned(normalized.clone()))
        } else {
            Endpoint::from_owned(normalized.clone())
        };
        let auth = if api_key.is_empty() {
            AuthData::from_single(String::new()) // local runtimes need no key
        } else {
            AuthData::from_single(api_key.to_string())
        };
        let target = ServiceTarget {
            endpoint,
            auth,
            model: model_iden,
        };
        Self {
            client,
            resolved: ResolvedTarget {
                target,
                model_id: model.to_string(),
                adapter,
                endpoint_base: normalized,
                has_key: !api_key.is_empty(),
            },
            http,
        }
    }

    fn map_request(req: &LlmRequest) -> (ChatRequest, ChatOptions) {
        let mut chat = ChatRequest::default();
        if let Some(system) = &req.system {
            chat = chat.with_system(system.clone());
        }
        // Prior turns (oldest→newest) precede the final user prompt so the model
        // sees the conversation as [system, …history…, user(question)].
        for msg in &req.messages {
            chat = chat.append_message(match msg.role {
                crate::chat::ChatRole::User => ChatMessage::user(msg.content.clone()),
                crate::chat::ChatRole::Assistant => ChatMessage::assistant(msg.content.clone()),
            });
        }
        chat = chat.append_message(ChatMessage::user(req.prompt.clone()));

        let mut opts = ChatOptions::default()
            .with_temperature(req.temperature as f64)
            .with_max_tokens(req.max_tokens)
            .with_capture_usage(true);
        if req.json {
            opts = opts.with_response_format(ChatResponseFormat::JsonMode);
        }
        if req.thinking {
            let effort = req
                .reasoning_effort
                .unwrap_or(ReasoningEffort::Medium)
                .to_genai();
            opts = opts.with_reasoning_effort(effort);
        }
        (chat, opts)
    }

    fn model_spec(&self) -> ModelSpec {
        ModelSpec::Target(self.resolved.target.clone())
    }

    /// Unauthenticated GET to `{endpoint_base}api/version` — never bills a token unlike a
    /// `generate` ping. Returns `true` on HTTP success; `false` on refusal/timeout/non-success.
    async fn ollama_alive(&self) -> bool {
        let url = format!("{}api/version", self.resolved.endpoint_base);
        crate::http::hardened_client(LLM_CONNECT_TIMEOUT, LLM_TIMEOUT)
            .get(url)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

/// Collapses genai `Option<i32>` usage fields into a saturating `u32`;
/// falls back to `total_tokens` when prompt/completion are both absent.
fn usage_to_tokens(usage: &genai::chat::Usage) -> u32 {
    let nonneg = |v: Option<i32>| u32::try_from(v.unwrap_or(0).max(0)).unwrap_or(0);
    let prompt = nonneg(usage.prompt_tokens);
    let completion = nonneg(usage.completion_tokens);
    let summed = prompt.saturating_add(completion);
    if summed > 0 {
        summed
    } else {
        nonneg(usage.total_tokens)
    }
}

#[async_trait]
impl LlmProvider for GenaiProvider {
    fn model_id(&self) -> &str {
        &self.resolved.model_id
    }

    fn shared_http_client(&self) -> Option<reqwest::Client> {
        Some(self.http.clone())
    }

    fn is_local(&self) -> bool {
        self.is_ollama()
    }

    fn is_ollama(&self) -> bool {
        matches!(self.resolved.adapter, AdapterKind::Ollama)
    }

    async fn reachable(&self) -> bool {
        // Local Ollama: free unauthenticated GET to /api/version (no token cost).
        // Cloud: treat "key configured or keyless native endpoint" as reachable without any
        // network probe — a genuinely unreachable cloud host surfaces as an error from
        // generate(), which the worker already maps to failed/degrade.
        if matches!(self.resolved.adapter, AdapterKind::Ollama) {
            return self.ollama_alive().await;
        }
        self.resolved.has_key || native_endpoint(self.resolved.adapter).is_some()
    }

    async fn generate(&self, req: &LlmRequest) -> Result<LlmResponse, LensError> {
        let (chat, opts) = Self::map_request(req);
        let res = self
            .client
            .exec_chat(self.model_spec(), chat, Some(&opts))
            .await
            .map_err(genai_err)?;
        let text = res.first_text().unwrap_or_default().to_string();
        let tokens_used = usage_to_tokens(&res.usage);
        Ok(LlmResponse { text, tokens_used })
    }

    async fn generate_stream(
        &self,
        req: &LlmRequest,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<StreamChunk, LensError>> + Send>>,
        LensError,
    > {
        use genai::chat::ChatStreamEvent;

        let (chat, opts) = Self::map_request(req);
        let res = self
            .client
            .exec_chat_stream(self.model_spec(), chat, Some(&opts))
            .await
            .map_err(genai_err)?;

        let mapped = futures_util::StreamExt::filter_map(res.stream, |ev| async move {
            match ev {
                Ok(ChatStreamEvent::Chunk(c)) => Some(Ok(StreamChunk::TextDelta(c.content))),
                Ok(ChatStreamEvent::ReasoningChunk(c)) => {
                    Some(Ok(StreamChunk::ThinkingDelta(c.content)))
                }
                Ok(ChatStreamEvent::End(end)) => {
                    let tokens_used = end
                        .captured_usage
                        .as_ref()
                        .map(usage_to_tokens)
                        .unwrap_or(0);
                    Some(Ok(StreamChunk::Done { tokens_used }))
                }
                Ok(_) => None, // Start / ToolCallChunk / etc: not part of the text contract
                Err(e) => Some(Err(genai_err(e))),
            }
        });
        Ok(Box::pin(mapped))
    }
}

// ---------------------------------------------------------------------------
// rig backend (genai → rig migration, epic #255 / Phase 0 #256)
// ---------------------------------------------------------------------------

/// The rig-backed [`LlmProvider`], gated behind the default-on `llm-backend-rig` feature and
/// covering every provider id since Phase 1 (#257). Re-exported so tests and the factories can
/// construct it directly.
#[cfg(feature = "llm-backend-rig")]
pub use rig_backend::RigProvider;

#[cfg(feature = "llm-backend-rig")]
// The public `new_*` convenience constructors are exercised only by `#[cfg(test)]` code — the
// lib path builds through `from_id`→`*_with_http` — so a non-test rig build would flag them.
#[allow(dead_code)]
mod rig_backend {
    use std::pin::Pin;

    use async_trait::async_trait;
    use futures_util::{Stream, StreamExt};
    use rig_core::client::CompletionClient;
    use rig_core::completion::{
        AssistantContent, CompletionError, CompletionModel, CompletionRequest, GetTokenUsage,
        Message as RigMessage, Usage,
    };
    use rig_core::http_client;
    use rig_core::providers::{
        anthropic, cohere, deepseek, gemini, groq, ollama, openai, xai, zai,
    };
    use rig_core::schemars::Schema;
    use rig_core::streaming::StreamedAssistantContent;

    use super::{
        LLM_CONNECT_TIMEOUT, LLM_TIMEOUT, LlmProvider, LlmRequest, LlmResponse, ReasoningEffort,
        StreamChunk,
    };
    use crate::error::LensError;

    /// Ollama's canonical default endpoint, mirrored so an empty `base_url` stays construct-able.
    const OLLAMA_DEFAULT_BASE_URL: &str = "http://localhost:11434";

    /// Ollama's managed cloud endpoint, used when `new_ollama_cloud` gets an empty `base_url`.
    const OLLAMA_CLOUD_DEFAULT_BASE_URL: &str = "https://ollama.com";

    #[derive(Clone, Copy)]
    enum OllamaKind {
        Local,
        Cloud,
    }

    impl OllamaKind {
        fn default_base(self) -> &'static str {
            match self {
                OllamaKind::Local => OLLAMA_DEFAULT_BASE_URL,
                OllamaKind::Cloud => OLLAMA_CLOUD_DEFAULT_BASE_URL,
            }
        }

        fn is_local(self) -> bool {
            matches!(self, OllamaKind::Local)
        }
    }

    /// One variant per distinct rig concrete completion-model type — NOT per provider id: the
    /// provider ids that share a client (openai/openai-compatible, ollama/ollama-cloud, glm/zai)
    /// collapse onto one variant. rig's `CompletionModel` carries associated types (it is not
    /// object-safe), so the backends are enum-dispatched rather than held behind `dyn`.
    enum RigModel {
        Ollama(ollama::CompletionModel<reqwest::Client>),
        // Chat Completions API — NOT the Responses API model that `openai::Client` yields.
        OpenAi(openai::completion::CompletionModel<reqwest::Client>),
        Anthropic(anthropic::completion::CompletionModel<reqwest::Client>),
        Gemini(gemini::completion::CompletionModel<reqwest::Client>),
        Cohere(cohere::completion::CompletionModel<reqwest::Client>),
        Xai(xai::completion::CompletionModel<reqwest::Client>),
        Groq(groq::CompletionModel<reqwest::Client>),
        DeepSeek(deepseek::CompletionModel<reqwest::Client>),
        // Z.ai (GLM) has no provider-level `CompletionModel` alias; name the generic type directly.
        Zai(openai::completion::GenericCompletionModel<zai::ZAiExt, reqwest::Client>),
    }

    /// A single LLM backend over rig. Constructed with our hardened reqwest client injected via
    /// `ClientBuilder::http_client(...)`, so the SSRF/timeout policy carries over unchanged.
    pub struct RigProvider {
        model: RigModel,
        model_id: String,
        /// Always ends in `/`; only the local-Ollama reachability probe reads it (it appends
        /// `api/version`). Cloud backends never probe, so the field is unused for them.
        endpoint_base: String,
        /// true only for a local Ollama runtime; see `is_ollama`.
        is_local_ollama: bool,
        /// The hardened client this provider was built with, kept so `shared_http_client` can hand
        /// it to sibling per-task providers (rig's completion model does not re-expose it).
        http: reqwest::Client,
    }

    /// Maps a rig client `build()` failure onto a fixed generic [`LensError`], logging the full
    /// detail server-side. Never leaks the client's error string across IPC.
    fn map_build_err(err: http_client::Error) -> LensError {
        tracing::error!(error = %err, "failed to build rig LLM client");
        LensError::Model("failed to initialize the language model client".to_string())
    }

    /// The hardened reqwest client every fresh backend is built with (delegates to the parent).
    fn default_http() -> reqwest::Client {
        super::llm_client()
    }

    /// Generates a cloud-provider constructor pair (public `$name` builds a fresh client, private
    /// `$with` reuses a caller's). An empty api_key is allowed — reachability is decided elsewhere;
    /// base_url is set only when non-empty so an empty value keeps the provider default.
    macro_rules! cloud_ctor {
        ($(#[$m:meta])* $name:ident, $with:ident, $client:ty, $variant:ident) => {
            $(#[$m])*
            pub(crate) fn $name(
                model: &str,
                base_url: &str,
                api_key: &str,
            ) -> Result<Self, LensError> {
                Self::$with(model, base_url, api_key, default_http())
            }

            fn $with(
                model: &str,
                base_url: &str,
                api_key: &str,
                http: reqwest::Client,
            ) -> Result<Self, LensError> {
                let mut builder = <$client>::builder().api_key(api_key).http_client(http.clone());
                if !base_url.is_empty() {
                    builder = builder.base_url(base_url);
                }
                let client = builder.build().map_err(map_build_err)?;
                Ok(Self {
                    model: RigModel::$variant(client.completion_model(model)),
                    model_id: model.to_string(),
                    endpoint_base: String::new(),
                    is_local_ollama: false,
                    http,
                })
            }
        };
    }

    impl RigProvider {
        /// Builds an Ollama-backed provider. An empty `base_url` falls back to Ollama's canonical
        /// endpoint; an empty `api_key` means keyless (the default for a local runtime).
        pub(crate) fn new_ollama(
            model: &str,
            base_url: &str,
            api_key: &str,
        ) -> Result<Self, LensError> {
            Self::build_ollama(model, base_url, api_key, OllamaKind::Local, default_http())
        }

        /// Ollama's managed cloud endpoint. An empty `base_url` falls back to `https://ollama.com`.
        pub(crate) fn new_ollama_cloud(
            model: &str,
            base_url: &str,
            api_key: &str,
        ) -> Result<Self, LensError> {
            Self::build_ollama(model, base_url, api_key, OllamaKind::Cloud, default_http())
        }

        fn build_ollama(
            model: &str,
            base_url: &str,
            api_key: &str,
            kind: OllamaKind,
            http: reqwest::Client,
        ) -> Result<Self, LensError> {
            let base = if base_url.is_empty() {
                kind.default_base()
            } else {
                base_url
            };
            let client = ollama::Client::builder()
                .api_key(ollama::OllamaApiKey::from(api_key))
                .base_url(base)
                .http_client(http.clone())
                .build()
                .map_err(map_build_err)?;
            Ok(Self {
                model: RigModel::Ollama(client.completion_model(model)),
                model_id: model.to_string(),
                endpoint_base: normalize_base(base),
                is_local_ollama: kind.is_local(),
                http,
            })
        }

        cloud_ctor!(
            /// OpenAI's Chat Completions API. Uses `openai::CompletionsClient` (not the default
            /// `openai::Client`, which yields the Responses-API model).
            new_openai,
            openai_with_http,
            openai::CompletionsClient,
            OpenAi
        );
        cloud_ctor!(
            new_anthropic,
            anthropic_with_http,
            anthropic::Client,
            Anthropic
        );
        cloud_ctor!(new_gemini, gemini_with_http, gemini::Client, Gemini);
        cloud_ctor!(new_cohere, cohere_with_http, cohere::Client, Cohere);
        cloud_ctor!(new_xai, xai_with_http, xai::Client, Xai);
        cloud_ctor!(new_groq, groq_with_http, groq::Client, Groq);
        cloud_ctor!(new_deepseek, deepseek_with_http, deepseek::Client, DeepSeek);
        // glm and zai both build the Zai variant (GLM models are Z.ai's).
        cloud_ctor!(new_zai, zai_with_http, zai::Client, Zai);

        /// An OpenAI-wire-compatible provider at a user-supplied endpoint. The custom `base_url`
        /// is mandatory (there is no default host to fall back to); otherwise identical to
        /// [`Self::new_openai`].
        pub(crate) fn new_openai_compatible(
            model: &str,
            base_url: &str,
            api_key: &str,
        ) -> Result<Self, LensError> {
            Self::openai_compatible_with_http(model, base_url, api_key, default_http())
        }

        fn openai_compatible_with_http(
            model: &str,
            base_url: &str,
            api_key: &str,
            http: reqwest::Client,
        ) -> Result<Self, LensError> {
            if base_url.is_empty() {
                return Err(LensError::Validation(
                    "an OpenAI-compatible provider requires a base URL".to_string(),
                ));
            }
            Self::openai_with_http(model, base_url, api_key, http)
        }

        /// Dispatches a provider id onto its constructor, reusing `http` for every variant so a
        /// per-task sibling shares the base provider's client pool. Unknown ids error (the factory
        /// already gates recognition via [`super::adapter_for`], so this arm is defensive).
        pub(crate) fn from_id(
            provider: &str,
            model: &str,
            base_url: &str,
            api_key: &str,
            http: reqwest::Client,
        ) -> Result<Self, LensError> {
            match provider {
                super::PROVIDER_OLLAMA => {
                    Self::build_ollama(model, base_url, api_key, OllamaKind::Local, http)
                }
                super::PROVIDER_OLLAMA_CLOUD => {
                    Self::build_ollama(model, base_url, api_key, OllamaKind::Cloud, http)
                }
                super::PROVIDER_OPENAI => Self::openai_with_http(model, base_url, api_key, http),
                super::PROVIDER_OPENAI_COMPAT => {
                    Self::openai_compatible_with_http(model, base_url, api_key, http)
                }
                super::PROVIDER_ANTHROPIC => {
                    Self::anthropic_with_http(model, base_url, api_key, http)
                }
                super::PROVIDER_GOOGLE => Self::gemini_with_http(model, base_url, api_key, http),
                super::PROVIDER_ZAI | super::PROVIDER_GLM => {
                    Self::zai_with_http(model, base_url, api_key, http)
                }
                super::PROVIDER_GROQ => Self::groq_with_http(model, base_url, api_key, http),
                super::PROVIDER_DEEPSEEK => {
                    Self::deepseek_with_http(model, base_url, api_key, http)
                }
                super::PROVIDER_XAI => Self::xai_with_http(model, base_url, api_key, http),
                super::PROVIDER_COHERE => Self::cohere_with_http(model, base_url, api_key, http),
                other => Err(LensError::Validation(format!(
                    "unknown LLM provider id: {other}"
                ))),
            }
        }

        /// Test-only discriminant so id→variant mapping can be asserted without exposing the
        /// private [`RigModel`] type across the module boundary.
        #[cfg(test)]
        pub(crate) fn variant_name(&self) -> &'static str {
            match &self.model {
                RigModel::Ollama(_) => "ollama",
                RigModel::OpenAi(_) => "openai",
                RigModel::Anthropic(_) => "anthropic",
                RigModel::Gemini(_) => "gemini",
                RigModel::Cohere(_) => "cohere",
                RigModel::Xai(_) => "xai",
                RigModel::Groq(_) => "groq",
                RigModel::DeepSeek(_) => "deepseek",
                RigModel::Zai(_) => "zai",
            }
        }

        /// Unauthenticated GET to `{endpoint_base}api/version` — never bills a token, mirroring
        /// [`super::GenaiProvider`]'s cheap Ollama liveness probe.
        async fn ollama_alive(&self) -> bool {
            let url = format!("{}api/version", self.endpoint_base);
            crate::http::hardened_client(LLM_CONNECT_TIMEOUT, LLM_TIMEOUT)
                .get(url)
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false)
        }
    }

    fn normalize_base(base: &str) -> String {
        format!("{}/", base.trim_end_matches('/'))
    }

    /// How a provider family carries the token cap and reasoning knobs — rig exposes no universal
    /// reasoning field, so each family needs its own `additional_params` shape.
    #[derive(Clone, Copy)]
    enum ParamStyle {
        /// Ollama: token cap via `options.num_predict`; reasoning via `think`.
        Ollama,
        /// OpenAI wire (openai, openai-compatible, groq, deepseek, xai): reasoning via
        /// `reasoning_effort`; the cap is the honored top-level `max_tokens`.
        OpenAi,
        /// Anthropic: reasoning via `thinking: {type, budget_tokens}`.
        Anthropic,
        /// Gemini / Cohere / Z.ai: only the top-level `max_tokens`; no reasoning knob mapped
        /// (their native reasoning controls are out of scope for the migration).
        Plain,
    }

    impl ParamStyle {
        /// The `additional_params` object for `req`, or `None` when nothing extra is needed.
        fn additional_params(self, req: &LlmRequest) -> Option<serde_json::Value> {
            let mut extra = serde_json::Map::new();
            match self {
                ParamStyle::Ollama => {
                    if req.max_tokens > 0 {
                        extra.insert("num_predict".to_string(), serde_json::json!(req.max_tokens));
                    }
                    if req.thinking {
                        // rig lifts `think` back out to a top-level Ollama field.
                        let think = match req.reasoning_effort {
                            Some(ReasoningEffort::Low) => serde_json::json!("low"),
                            Some(ReasoningEffort::Medium) => serde_json::json!("medium"),
                            Some(ReasoningEffort::High) => serde_json::json!("high"),
                            None => serde_json::json!(true),
                        };
                        extra.insert("think".to_string(), think);
                    }
                }
                ParamStyle::OpenAi => {
                    if req.thinking {
                        let effort = match req.reasoning_effort.unwrap_or(ReasoningEffort::Medium) {
                            ReasoningEffort::Low => "low",
                            ReasoningEffort::Medium => "medium",
                            ReasoningEffort::High => "high",
                        };
                        extra.insert("reasoning_effort".to_string(), serde_json::json!(effort));
                    }
                }
                ParamStyle::Anthropic => {
                    // Enable extended thinking only when a valid budget fits under `max_tokens`
                    // (Anthropic rejects a budget ≥ the cap or below its 1024 floor).
                    if let Some(budget) = anthropic_thinking_budget(req) {
                        extra.insert(
                            "thinking".to_string(),
                            serde_json::json!({ "type": "enabled", "budget_tokens": budget }),
                        );
                    }
                }
                ParamStyle::Plain => {}
            }
            if extra.is_empty() {
                None
            } else {
                Some(serde_json::Value::Object(extra))
            }
        }

        /// The temperature to send. Anthropic requires `temperature == 1` whenever extended
        /// thinking is enabled (it rejects any other value), so override it there; every other
        /// path uses the request's configured temperature.
        fn temperature(self, req: &LlmRequest) -> f64 {
            let anthropic_thinking =
                matches!(self, ParamStyle::Anthropic) && anthropic_thinking_budget(req).is_some();
            if anthropic_thinking {
                1.0
            } else {
                f64::from(req.temperature)
            }
        }
    }

    /// The active [`ParamStyle`] for a backend variant.
    fn param_style(model: &RigModel) -> ParamStyle {
        match model {
            RigModel::Ollama(_) => ParamStyle::Ollama,
            RigModel::OpenAi(_) | RigModel::Groq(_) | RigModel::DeepSeek(_) | RigModel::Xai(_) => {
                ParamStyle::OpenAi
            }
            RigModel::Anthropic(_) => ParamStyle::Anthropic,
            RigModel::Gemini(_) | RigModel::Cohere(_) | RigModel::Zai(_) => ParamStyle::Plain,
        }
    }

    /// Anthropic extended-thinking budget, or `None` when thinking is off or `max_tokens` leaves
    /// no room for Anthropic's 1024-token floor below the cap (the API rejects a budget ≥ the
    /// cap). Scaled to effort and clamped strictly below `max_tokens`.
    fn anthropic_thinking_budget(req: &LlmRequest) -> Option<u32> {
        if !req.thinking {
            return None;
        }
        let ceiling = req.max_tokens.checked_sub(1)?;
        if ceiling < 1024 {
            return None;
        }
        let want = match req.reasoning_effort.unwrap_or(ReasoningEffort::Medium) {
            ReasoningEffort::Low => 1024,
            ReasoningEffort::Medium => 4096,
            ReasoningEffort::High => 8192,
        };
        Some(want.min(ceiling))
    }

    /// A permissive object schema standing in for Ollama's schemaless `format:"json"` — rig's
    /// `output_schema` (typed `Option<Schema>`) can't carry that bare string, but this is the
    /// only path that lands `format` at the request's TOP level, not inside `options` (#256 §0.1 #4).
    fn json_object_schema() -> Schema {
        let mut map = serde_json::Map::new();
        map.insert(
            "type".to_string(),
            serde_json::Value::String("object".to_string()),
        );
        Schema::from(map)
    }

    /// Maps [`LlmRequest`] onto a rig [`CompletionRequest`]: system→preamble, `messages`
    /// (oldest→newest)→history, `prompt`→final user turn, temperature, `json`→top-level output
    /// schema, and the token-cap / reasoning knobs → a per-provider `additional_params` object
    /// (rig has no universal reasoning field, so each family carries its own — see [`ParamStyle`]).
    fn map_request<M: CompletionModel>(
        model: &M,
        req: &LlmRequest,
        style: ParamStyle,
    ) -> CompletionRequest {
        let mut builder = model.completion_request(RigMessage::user(req.prompt.clone()));
        if let Some(system) = &req.system {
            builder = builder.preamble(system.clone());
        }
        let history = req.messages.iter().map(|msg| match msg.role {
            crate::chat::ChatRole::User => RigMessage::user(msg.content.clone()),
            crate::chat::ChatRole::Assistant => RigMessage::assistant(msg.content.clone()),
        });
        builder = builder
            .messages(history)
            .temperature(style.temperature(req))
            // OpenAI-family / Anthropic / Gemini honor this bare top-level field; Ollama ignores it
            // and reads the cap from `options.num_predict` instead (added below for that style).
            .max_tokens(u64::from(req.max_tokens));
        if req.json {
            builder = builder.output_schema(json_object_schema());
        }

        // Per-provider knobs go in ONE `additional_params` object — a second call would overwrite
        // rather than merge. `thinking` is inert today (no runtime caller sets it; M5 chat will).
        let extra = style.additional_params(req);
        if let Some(extra) = extra {
            builder = builder.additional_params(extra);
        }
        builder.build()
    }

    /// Collapses rig's `u64` token usage into a saturating `u32` — never a blind `as` cast.
    /// Falls back to `total_tokens` when the split input/output counts are both absent.
    fn usage_to_tokens(usage: &Usage) -> u32 {
        let summed = usage.input_tokens.saturating_add(usage.output_tokens);
        let total = if summed > 0 {
            summed
        } else {
            usage.total_tokens
        };
        u32::try_from(total).unwrap_or(u32::MAX)
    }

    /// Maps a rig [`CompletionError`] onto [`LensError`], sanitizing every embedded string
    /// (incl. the `#[non_exhaustive]` catch-all) to a fixed generic message and logging the full
    /// detail server-side (mirrors [`super::genai_err`]).
    fn rig_err(err: CompletionError) -> LensError {
        tracing::error!(error = %err, "LLM request failed (rig)");
        // Only `HttpError::Instance` is transport (→ Network); HTTP status codes are semantic
        // (→ Model), and `ProviderError`/`ProviderResponse` are classified by message text (shared
        // with `genai_err`). The response body never crosses IPC.
        let is_transport = match &err {
            CompletionError::HttpError(http_client::Error::Instance(_)) => true,
            CompletionError::ProviderError(msg) => super::looks_like_transport(msg),
            CompletionError::ProviderResponse(e) => super::looks_like_transport(&e.to_string()),
            _ => false,
        };
        if is_transport {
            LensError::Network(
                "couldn't reach the language model — check that your LLM provider \
                 (e.g. local Ollama) is running and reachable"
                    .to_string(),
            )
        } else {
            LensError::Model("LLM request failed (model)".to_string())
        }
    }

    /// Runs one buffered completion. Generic over the rig model so every [`RigModel`] variant
    /// shares this body via `dispatch!`.
    async fn rig_generate<M: CompletionModel>(
        model: &M,
        req: &LlmRequest,
        style: ParamStyle,
    ) -> Result<LlmResponse, LensError> {
        let request = map_request(model, req, style);
        let resp = model.completion(request).await.map_err(rig_err)?;
        let text = resp
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        let tokens_used = usage_to_tokens(&resp.usage);
        Ok(LlmResponse { text, tokens_used })
    }

    /// Runs one streaming completion, mapping rig's stream items onto the [`StreamChunk`] text
    /// contract. Generic over the rig model so every variant shares this body via `dispatch!`.
    async fn rig_stream<M>(
        model: &M,
        req: &LlmRequest,
        style: ParamStyle,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LensError>> + Send>>, LensError>
    where
        M: CompletionModel,
        M::StreamingResponse: 'static,
    {
        let request = map_request(model, req, style);
        let stream = model.stream(request).await.map_err(rig_err)?;
        let mapped = stream.filter_map(|ev| async move {
            match ev {
                Ok(StreamedAssistantContent::Text(t)) => Some(Ok(StreamChunk::TextDelta(t.text))),
                Ok(StreamedAssistantContent::ReasoningDelta { reasoning, .. }) => {
                    Some(Ok(StreamChunk::ThinkingDelta(reasoning)))
                }
                Ok(StreamedAssistantContent::Final(resp)) => Some(Ok(StreamChunk::Done {
                    tokens_used: usage_to_tokens(&resp.token_usage()),
                })),
                // Tool calls / full reasoning blocks / unknown items: not part of the text
                // contract. Ollama streaming never emits them for enrichment.
                Ok(_) => None,
                Err(e) => Some(Err(rig_err(e))),
            }
        });
        Ok(Box::pin(mapped))
    }

    /// Binds the inner model of any [`RigModel`] variant to `$m` and runs `$body` — the arms are
    /// identical modulo the model value, so enumerate the variants once here.
    macro_rules! dispatch {
        ($model:expr, $m:ident => $body:expr) => {
            match $model {
                RigModel::Ollama($m) => $body,
                RigModel::OpenAi($m) => $body,
                RigModel::Anthropic($m) => $body,
                RigModel::Gemini($m) => $body,
                RigModel::Cohere($m) => $body,
                RigModel::Xai($m) => $body,
                RigModel::Groq($m) => $body,
                RigModel::DeepSeek($m) => $body,
                RigModel::Zai($m) => $body,
            }
        };
    }

    #[async_trait]
    impl LlmProvider for RigProvider {
        fn model_id(&self) -> &str {
            &self.model_id
        }

        fn shared_http_client(&self) -> Option<reqwest::Client> {
            Some(self.http.clone())
        }

        fn is_local(&self) -> bool {
            self.is_ollama()
        }

        // Matches genai (adapter == `Ollama`): true ONLY for a local Ollama runtime, NOT
        // ollama-cloud. The `/api/tags` model-installed preflight targets the LOCAL runtime
        // (`ollama_base_url(config)`), so firing it for a keyed cloud model would falsely fail;
        // `adapter_for` likewise maps ollama-cloud to a distinct `OllamaCloud` adapter.
        fn is_ollama(&self) -> bool {
            self.is_local_ollama
        }

        async fn reachable(&self) -> bool {
            // Local Ollama probes network liveness; cloud/compat report reachable without a network
            // probe, mirroring genai — a genuinely unreachable host or bad key surfaces from
            // `generate`, never a billed token here.
            if self.is_local_ollama {
                self.ollama_alive().await
            } else {
                true
            }
        }

        async fn generate(&self, req: &LlmRequest) -> Result<LlmResponse, LensError> {
            let style = param_style(&self.model);
            dispatch!(&self.model, model => rig_generate(model, req, style).await)
        }

        async fn generate_stream(
            &self,
            req: &LlmRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LensError>> + Send>>, LensError>
        {
            let style = param_style(&self.model);
            dispatch!(&self.model, model => rig_stream(model, req, style).await)
        }
    }
}

// ---------------------------------------------------------------------------
// Routing / override policy (Stage 2)
// ---------------------------------------------------------------------------

/// Typed routing policy for selecting the enrichment LLM. Serde-stable (snake_case, internally
/// tagged on `kind`) so it round-trips in `config.json` without leaking a Rust enum shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LlmRouting {
    /// Prefer a consented cloud provider; fall back to local Ollama. Default.
    CloudFirst,
    /// Prefer local Ollama; fall back to a consented cloud provider.
    LocalFirst,
    /// Pin one exact `(provider, model)`. Cloud entries are still consent-gated.
    Explicit { provider: String, model: String },
}

impl Default for LlmRouting {
    /// Product-direction default: prefer cloud-when-available, else local.
    fn default() -> Self {
        LlmRouting::CloudFirst
    }
}

/// Delegates to [`SupportedProvider::is_local`] — the single locality predicate shared
/// by the consent-gate exemption here and the catalog-validation exemption in `model_catalog`.
fn is_local_provider(provider: &str) -> bool {
    SupportedProvider::is_local(provider)
}

/// Maps a `ModelConfig.provider` id to a genai [`AdapterKind`]. `glm` is an alias for `zai`
/// (GLM models are Z.ai's). `openai-compatible` maps to [`AdapterKind::OpenAI`] with the
/// user-supplied base URL. Returns `None` for unrecognized providers.
fn adapter_for(provider: &str) -> Option<AdapterKind> {
    match provider {
        PROVIDER_OLLAMA => Some(AdapterKind::Ollama),
        PROVIDER_OLLAMA_CLOUD => Some(AdapterKind::OllamaCloud),
        PROVIDER_ANTHROPIC => Some(AdapterKind::Anthropic),
        PROVIDER_GOOGLE => Some(AdapterKind::Gemini),
        PROVIDER_OPENAI => Some(AdapterKind::OpenAI),
        PROVIDER_ZAI | PROVIDER_GLM => Some(AdapterKind::Zai),
        PROVIDER_GROQ => Some(AdapterKind::Groq),
        PROVIDER_DEEPSEEK => Some(AdapterKind::DeepSeek),
        PROVIDER_XAI => Some(AdapterKind::Xai),
        PROVIDER_COHERE => Some(AdapterKind::Cohere),
        PROVIDER_OPENAI_COMPAT => Some(AdapterKind::OpenAI), // custom endpoint: OpenAI protocol
        _ => None,
    }
}

/// Builds the enrichment [`LlmProvider`] from `config.models[]` under the [`LlmRouting`]
/// policy. Cloud providers require `cloud_consent == true`; local Ollama is exempt.
/// Catalog membership is advisory metadata, not a usability gate. Does NOT probe
/// reachability — the caller does that separately.
pub fn provider_from_config(
    config: &AppConfig,
    cloud_consent: bool,
) -> Option<Arc<dyn LlmProvider>> {
    let routing = config.enrichment.routing.clone();
    select_provider(&config.models, &routing, cloud_consent)
}

/// Resolves the interactive-chat provider (Variant B). A purpose-built
/// `enrichment.chat_model` pin is authoritative when present: it builds a fresh provider
/// for the matching `models[]` entry under the same consent gate as routing, and does NOT
/// fall back to routing when the pin is unusable (so `has_chat_provider` reports absence).
/// With no pin, defers to the routing-based [`provider_from_config`].
pub fn chat_provider_from_config(
    config: &AppConfig,
    cloud_consent: bool,
) -> Option<Arc<dyn LlmProvider>> {
    match &config.enrichment.chat_model {
        Some(chat_model) => build_pinned_provider(
            &chat_model.provider,
            &chat_model.model,
            &config.models,
            cloud_consent,
        ),
        None => provider_from_config(config, cloud_consent),
    }
}

/// Resolves a per-task enrichment provider, reusing `base`'s genai client (M4 Phase 3).
/// When `task_model` resolves to a gated, usable entry, returns a sibling [`GenaiProvider`]
/// pinned to that `(provider, model)` over the same client. Falls back to `base.clone()` on
/// `None` or failed gates (unknown provider, no consent, empty model).
pub fn task_provider_from_config(
    base: &Arc<dyn LlmProvider>,
    task_model: Option<&crate::config::TaskModel>,
    models: &[crate::config::ModelConfig],
    cloud_consent: bool,
) -> Arc<dyn LlmProvider> {
    match task_model.and_then(|tm| build_task_provider(base, tm, models, cloud_consent)) {
        Some(p) => p,
        None => base.clone(),
    }
}

/// Builds a sibling provider pinned to `task_model`, reusing `base`'s hardened client (one
/// connection pool across coref/map). Returns `None` — so the caller reuses `base` unchanged —
/// when no matching config entry exists, the provider is ungated/unrecognized, or `base` does not
/// expose a shareable client (e.g. a test mock). Backend-agnostic: no concrete-type downcast.
fn build_task_provider(
    base: &Arc<dyn LlmProvider>,
    task_model: &crate::config::TaskModel,
    models: &[crate::config::ModelConfig],
    cloud_consent: bool,
) -> Option<Arc<dyn LlmProvider>> {
    let want_provider = task_model.provider.to_ascii_lowercase();
    // Recognized provider only (mirrors the factory gate); unknown → fall back to base.
    adapter_for(&want_provider)?;

    // Prefer the entry matching both provider AND override model (e.g. two Ollama endpoints,
    // instruct vs. coder); fall back to the first entry for that provider.
    let entry = models
        .iter()
        .find(|m| {
            m.provider.to_ascii_lowercase() == want_provider
                && m.model == task_model.model
                && has_endpoint(m)
        })
        .or_else(|| {
            models
                .iter()
                .find(|m| m.provider.to_ascii_lowercase() == want_provider && has_endpoint(m))
        })?;

    // Same consent gate as routing: local Ollama exempt; every cloud provider needs consent
    // and a non-empty model id. Catalog membership is advisory metadata, not a usability gate.
    if is_local_provider(&want_provider) {
        if task_model.model.is_empty() {
            return None;
        }
    } else if !cloud_consent || task_model.model.is_empty() {
        return None;
    }

    // Reuse the base's client via the trait, not a concrete downcast — `None` (mocks) falls back.
    let http = base.shared_http_client()?;
    construct_provider_with_http(
        &want_provider,
        &task_model.model,
        &entry.base_url,
        &entry.api_key,
        http,
    )
}

/// Routing-aware selection over configured model entries; split out for testability.
fn select_provider(
    models: &[crate::config::ModelConfig],
    routing: &LlmRouting,
    cloud_consent: bool,
) -> Option<Arc<dyn LlmProvider>> {
    let usable = |m: &crate::config::ModelConfig| {
        has_endpoint(m) && !m.model.is_empty() && build_eligible(m, cloud_consent)
    };

    match routing {
        LlmRouting::Explicit { provider, model } => {
            build_pinned_provider(provider, model, models, cloud_consent)
        }
        // Build usable candidates in priority order via find_map so an unbuildable preferred entry
        // can't strand the cloud/local fallback. Defensive: genai's build is infallible, and rig's
        // only formerly-reachable failure (empty-base openai-compatible) is now filtered as unusable
        // upstream by #273 — so no config-reachable usable candidate fails to build today.
        LlmRouting::CloudFirst => {
            let cloud = models
                .iter()
                .filter(|m| !is_local_provider(&m.provider.to_ascii_lowercase()) && usable(m));
            let local = models
                .iter()
                .filter(|m| is_local_provider(&m.provider.to_ascii_lowercase()) && usable(m));
            cloud.chain(local).find_map(build_provider)
        }
        LlmRouting::LocalFirst => {
            let local = models
                .iter()
                .filter(|m| is_local_provider(&m.provider.to_ascii_lowercase()) && usable(m));
            let cloud = models
                .iter()
                .filter(|m| !is_local_provider(&m.provider.to_ascii_lowercase()) && usable(m));
            local.chain(cloud).find_map(build_provider)
        }
    }
}

/// Resolves the `models[]` entry pinned to `(provider, model)` and builds a fresh
/// [`GenaiProvider`] for it under the same usable gates routing selection applies
/// (endpoint present, non-empty model, `build_eligible` consent gate). Shared by
/// `select_provider`'s Explicit arm and the chat-model pin in [`chat_provider_from_config`]
/// so the gate lives in exactly one place.
fn build_pinned_provider(
    provider: &str,
    model: &str,
    models: &[crate::config::ModelConfig],
    cloud_consent: bool,
) -> Option<Arc<dyn LlmProvider>> {
    let want_provider = provider.to_ascii_lowercase();
    models
        .iter()
        .find(|m| {
            m.provider.to_ascii_lowercase() == want_provider
                && m.model == *model
                && has_endpoint(m)
                && !m.model.is_empty()
                && build_eligible(m, cloud_consent)
        })
        .and_then(build_provider)
}

/// Whether an entry passes the consent gate. Local Ollama is exempt; every other
/// (cloud / `openai-compatible`) provider needs consent and a non-empty model id.
/// Catalog membership is advisory metadata — it lists known models but must NOT block
/// usability, so a model newer than the bundled snapshot stays usable. Unrecognized
/// providers are never eligible.
fn build_eligible(model: &crate::config::ModelConfig, cloud_consent: bool) -> bool {
    let provider = model.provider.to_ascii_lowercase();
    if adapter_for(&provider).is_none() {
        return false;
    }
    if is_local_provider(&provider) {
        return true;
    }
    if !cloud_consent {
        return false;
    }
    !model.model.is_empty()
}

/// Whether a provider needs an explicit `base_url` (no default host): local Ollama and
/// `openai-compatible`. Keyed on the provider id, NOT the adapter — `openai-compatible` shares
/// `AdapterKind::OpenAI`, so an adapter check would wrongly treat a blank-base entry as usable (#273).
fn requires_explicit_base_url(provider: &str) -> bool {
    provider == PROVIDER_OPENAI_COMPAT
        || adapter_for(provider).is_none_or(|a| native_endpoint(a).is_none())
}

/// Whether an entry has a usable endpoint: a non-empty `base_url`, or a provider with a default
/// host (see [`requires_explicit_base_url`]).
fn has_endpoint(model: &crate::config::ModelConfig) -> bool {
    !model.base_url.is_empty() || !requires_explicit_base_url(&model.provider.to_ascii_lowercase())
}

/// Builds the active [`LlmProvider`] for a recognized entry (caller applies [`build_eligible`]
/// first). Returns `None` for an unrecognized provider, empty model, or missing endpoint.
fn build_provider(model: &crate::config::ModelConfig) -> Option<Arc<dyn LlmProvider>> {
    if model.model.is_empty() {
        return None;
    }
    let provider = model.provider.to_ascii_lowercase();
    adapter_for(&provider)?;
    if model.base_url.is_empty() && requires_explicit_base_url(&provider) {
        return None;
    }
    construct_provider(&provider, &model.model, &model.base_url, &model.api_key)
}

/// Builds a provider directly from raw, unsaved params (issue #90 interactive validation).
/// Bypasses routing/consent/catalog gates entirely — validates the values the user typed
/// before saving. Returns `None` for an unrecognized provider or a local/custom endpoint
/// with an empty `base_url`.
pub fn build_provider_raw(
    provider: &str,
    model: &str,
    base_url: &str,
    api_key: &str,
) -> Option<Arc<dyn LlmProvider>> {
    if model.is_empty() {
        return None;
    }
    if !base_url.is_empty() {
        let scheme_ok = base_url.split_once("://").is_some_and(|(scheme, _)| {
            scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https")
        });
        if !scheme_ok {
            return None;
        }
    }
    let lower = provider.to_ascii_lowercase();
    adapter_for(&lower)?;
    if base_url.is_empty() && requires_explicit_base_url(&lower) {
        return None;
    }
    construct_provider(&lower, model, base_url, api_key)
}

/// The backend-construction seam. `rig` is the default; opting out of `llm-backend-rig` swaps in
/// the legacy [`GenaiProvider`] for every recognized id. Both backends share the id-recognition,
/// consent, and endpoint gates in the callers (via [`adapter_for`]/[`native_endpoint`]) — this
/// only builds the concrete provider once an entry is already deemed usable. A fresh build gets
/// its own hardened client; [`build_task_provider`] passes the base's so the sibling shares one
/// connection pool.
///
/// Endpoint contract (Phase-2 #258): a native cloud id with no `base_url` uses rig's own default
/// endpoint. For a custom openai/`openai-compatible` `base_url`, rig's openai-wire client posts it
/// VERBATIM (`<base_url>/chat/completions`) — unlike genai's force-appended `/v1/` — so it must
/// include the version segment. Other adapters (Anthropic, Gemini, Groq, …) inject their own path
/// segment onto the base (e.g. Anthropic → `/v1/messages`), matching genai.
fn construct_provider(
    provider: &str,
    model: &str,
    base_url: &str,
    api_key: &str,
) -> Option<Arc<dyn LlmProvider>> {
    construct_provider_with_http(provider, model, base_url, api_key, llm_client())
}

#[cfg(not(feature = "llm-backend-rig"))]
fn construct_provider_with_http(
    provider: &str,
    model: &str,
    base_url: &str,
    api_key: &str,
    http: reqwest::Client,
) -> Option<Arc<dyn LlmProvider>> {
    let adapter = adapter_for(provider)?;
    Some(Arc::new(GenaiProvider::new_with_http(
        http, adapter, model, base_url, api_key,
    )))
}

#[cfg(feature = "llm-backend-rig")]
fn construct_provider_with_http(
    provider: &str,
    model: &str,
    base_url: &str,
    api_key: &str,
    http: reqwest::Client,
) -> Option<Arc<dyn LlmProvider>> {
    match RigProvider::from_id(provider, model, base_url, api_key, http) {
        Ok(built) => Some(Arc::new(built)),
        Err(err) => {
            tracing::error!(error = %err, provider, "failed to build rig provider");
            None
        }
    }
}

/// A configured `(provider, model)` offered as the active chat model, with its computed
/// availability and a short reason when unavailable. Tauri-free: crosses IPC as-is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ActiveModelCandidate {
    pub provider: String,
    pub model: String,
    /// Display label, e.g. `"Ollama · llama3.2:3b"`.
    pub label: String,
    pub available: bool,
    /// Short human reason when unavailable; `None` when `available`.
    pub reason: Option<String>,
}

/// Enumerates the *pinnable-eligible* `config.models[]` entries (those with a non-empty
/// model) as active-chat-model candidates. Each entry's `available` mirrors exactly what a
/// chat-model pin would resolve to (same endpoint + consent + catalog gates as
/// [`build_pinned_provider`]), with a short `reason` otherwise. Credential-only entries
/// (empty model) are excluded — they are not pinnable, mirroring the build gates. Entries
/// whose provider has no genai adapter (e.g. an embedding backend) are omitted — they can
/// never back a chat model.
pub fn active_model_candidates(
    config: &AppConfig,
    cloud_consent: bool,
) -> Vec<ActiveModelCandidate> {
    config
        .models
        .iter()
        .filter(|m| !m.model.is_empty())
        .filter(|m| adapter_for(&m.provider.to_ascii_lowercase()).is_some())
        .map(|m| {
            let reason = candidate_unavailable_reason(m, cloud_consent);
            ActiveModelCandidate {
                provider: m.provider.clone(),
                model: m.model.clone(),
                label: candidate_label(&m.provider, &m.model),
                available: reason.is_none(),
                reason,
            }
        })
        .collect()
}

/// `None` when the entry would resolve as a chat pin; otherwise the first failing gate as a
/// short human reason. Mirrors the usable-gate order in [`build_pinned_provider`]: endpoint,
/// then [`build_eligible`] (the consent gate). Catalog membership is advisory metadata, not a
/// usability gate — a keyed + consented cloud model absent from the bundled snapshot reports
/// available. The caller already filters out empty-model (credential-only) entries, so the
/// model here is always non-empty.
fn candidate_unavailable_reason(
    model: &crate::config::ModelConfig,
    cloud_consent: bool,
) -> Option<String> {
    if !has_endpoint(model) {
        return Some("base URL required".to_string());
    }
    if build_eligible(model, cloud_consent) {
        return None;
    }
    // Endpoint present and model non-empty, so the only remaining gate is cloud consent.
    Some("cloud consent required".to_string())
}

/// Human-friendly provider name for a candidate label; falls back to the raw id for unknowns.
fn provider_display_name(provider: &str) -> &str {
    match provider.to_ascii_lowercase().as_str() {
        PROVIDER_OLLAMA => "Ollama",
        PROVIDER_OLLAMA_CLOUD => "Ollama Cloud",
        PROVIDER_OPENAI => "OpenAI",
        PROVIDER_ANTHROPIC => "Anthropic",
        PROVIDER_GOOGLE => "Google",
        PROVIDER_ZAI | PROVIDER_GLM => "Z.ai",
        PROVIDER_GROQ => "Groq",
        PROVIDER_DEEPSEEK => "DeepSeek",
        PROVIDER_XAI => "xAI",
        PROVIDER_COHERE => "Cohere",
        PROVIDER_OPENAI_COMPAT => "OpenAI-compatible",
        _ => provider,
    }
}

fn candidate_label(provider: &str, model: &str) -> String {
    let name = provider_display_name(provider);
    if model.is_empty() {
        name.to_string()
    } else {
        format!("{name} · {model}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::ModelConfig;
    use crate::model_catalog::ModelCatalog;
    use futures_util::StreamExt;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Nothing binds `127.0.0.1:1` — connection is deterministically refused.
    const DEAD_URL: &str = "http://127.0.0.1:1";

    fn req() -> LlmRequest {
        LlmRequest {
            system: Some("be terse".to_string()),
            prompt: "hello".to_string(),
            max_tokens: 64,
            temperature: 0.0,
            json: true,
            thinking: false,
            reasoning_effort: None,
            messages: Vec::new(),
        }
    }

    fn ollama_chat_body(content: &str) -> serde_json::Value {
        serde_json::json!({
            "model": "llama3",
            "message": { "role": "assistant", "content": content },
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 5
        })
    }

    // --- LlmRequest mapping (determinism contract) --------------------------

    #[test]
    fn map_request_sets_temperature_and_json_mode() {
        let (_chat, opts) = GenaiProvider::map_request(&req());
        assert_eq!(opts.temperature, Some(0.0));
        assert!(
            matches!(opts.response_format, Some(ChatResponseFormat::JsonMode)),
            "json:true must map to ChatResponseFormat::JsonMode"
        );
        assert_eq!(opts.max_tokens, Some(64));
        assert!(opts.reasoning_effort.is_none());
    }

    #[test]
    fn map_request_thinking_sets_reasoning_effort() {
        let r = LlmRequest {
            thinking: true,
            reasoning_effort: Some(ReasoningEffort::High),
            json: false,
            ..req()
        };
        let (_chat, opts) = GenaiProvider::map_request(&r);
        assert!(matches!(opts.reasoning_effort, Some(GenaiEffort::High)));
        assert!(opts.response_format.is_none());
    }

    #[test]
    fn llm_request_thinking_defaults_off_on_legacy_payload() {
        // An IPC/disk payload written before `thinking`/`reasoning_effort` existed
        // has neither key; both must read back as the off/none defaults.
        let json = r#"{
            "system": null,
            "prompt": "hi",
            "max_tokens": 32,
            "temperature": 0.0,
            "json": true
        }"#;
        let r: LlmRequest = serde_json::from_str(json).unwrap();
        assert!(!r.thinking);
        assert!(r.reasoning_effort.is_none());
    }

    // --- endpoint normalization ---------------------------------------------

    #[test]
    fn normalize_endpoint_ollama_just_trailing_slash() {
        assert_eq!(
            normalize_endpoint(AdapterKind::Ollama, "http://localhost:11434"),
            "http://localhost:11434/"
        );
        // An already-slashed base isn't doubled.
        assert_eq!(
            normalize_endpoint(AdapterKind::Ollama, "http://localhost:11434/"),
            "http://localhost:11434/"
        );
    }

    #[test]
    fn normalize_endpoint_openai_anthropic_get_v1() {
        assert_eq!(
            normalize_endpoint(AdapterKind::OpenAI, "http://localhost:1234"),
            "http://localhost:1234/v1/"
        );
        assert_eq!(
            normalize_endpoint(AdapterKind::Anthropic, "https://api.anthropic.com"),
            "https://api.anthropic.com/v1/"
        );
        // A base that already carries /v1 is not doubled.
        assert_eq!(
            normalize_endpoint(AdapterKind::OpenAI, "https://api.openai.com/v1"),
            "https://api.openai.com/v1/"
        );
    }

    // --- usage mapping ------------------------------------------------------

    #[test]
    fn usage_sums_prompt_and_completion() {
        let usage = genai::chat::Usage {
            prompt_tokens: Some(30),
            completion_tokens: Some(12),
            total_tokens: Some(42),
            ..Default::default()
        };
        assert_eq!(usage_to_tokens(&usage), 42);
    }

    #[test]
    fn usage_falls_back_to_total_when_split_absent() {
        let usage = genai::chat::Usage {
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: Some(99),
            ..Default::default()
        };
        assert_eq!(usage_to_tokens(&usage), 99);
    }

    // --- GenaiProvider round-trip via wiremock (Ollama adapter) -------------

    #[tokio::test]
    async fn genai_generate_round_trips_ollama() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ollama_chat_body("hi there")))
            .mount(&server)
            .await;

        let provider: Arc<dyn LlmProvider> = Arc::new(GenaiProvider::new(
            AdapterKind::Ollama,
            "llama3",
            &server.uri(),
            "",
        ));
        let resp = provider.generate(&req()).await.unwrap();
        assert_eq!(resp.text, "hi there");
        assert_eq!(resp.tokens_used, 15);
    }

    #[tokio::test]
    async fn genai_reachable_true_on_ok() {
        // The chat mock asserts expect(0): any billed generate dispatch would fail the test.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/version"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "version": "0.1.0"
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ollama_chat_body("ok")))
            .expect(0)
            .mount(&server)
            .await;

        let provider: Arc<dyn LlmProvider> = Arc::new(GenaiProvider::new(
            AdapterKind::Ollama,
            "llama3",
            &server.uri(),
            "",
        ));
        assert!(provider.reachable().await);
        drop(server); // verifies the chat endpoint was NEVER hit by the probe.
    }

    #[tokio::test]
    async fn genai_reachable_false_on_connection_refused() {
        let provider: Arc<dyn LlmProvider> = Arc::new(GenaiProvider::new(
            AdapterKind::Ollama,
            "llama3",
            DEAD_URL,
            "",
        ));
        assert!(!provider.reachable().await);
    }

    #[tokio::test]
    async fn cloud_reachable_does_not_perform_a_billed_generate() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;

        let provider: Arc<dyn LlmProvider> = Arc::new(GenaiProvider::new(
            AdapterKind::Anthropic,
            "claude-3-5-sonnet",
            &server.uri(),
            "sk-ant-key",
        ));
        assert!(
            provider.reachable().await,
            "a configured+consented cloud provider is reachable with no network probe"
        );
        drop(server); // verifies NO generate was dispatched (expect(0)).
    }

    #[tokio::test]
    async fn cloud_generate_failure_still_degrades_gracefully() {
        let provider: Arc<dyn LlmProvider> = Arc::new(GenaiProvider::new(
            AdapterKind::Anthropic,
            "claude-3-5-sonnet",
            DEAD_URL,
            "sk-ant-key",
        ));
        assert!(
            provider.reachable().await,
            "cloud reachable() is a cheap no-network signal"
        );
        let err = provider
            .generate(&req())
            .await
            .expect_err("a dead cloud endpoint must error on the real generate");
        assert!(
            matches!(err, LensError::Network(_) | LensError::Model(_)),
            "the real generate failure degrades gracefully; got {err:?}"
        );
    }

    #[tokio::test]
    async fn genai_reachable_false_on_500() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/version"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let provider: Arc<dyn LlmProvider> = Arc::new(GenaiProvider::new(
            AdapterKind::Ollama,
            "llama3",
            &server.uri(),
            "",
        ));
        assert!(!provider.reachable().await);
    }

    #[tokio::test]
    async fn genai_generate_non_success_is_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let provider: Arc<dyn LlmProvider> = Arc::new(GenaiProvider::new(
            AdapterKind::Ollama,
            "llama3",
            &server.uri(),
            "",
        ));
        let err = provider.generate(&req()).await.unwrap_err();
        assert!(
            matches!(err, LensError::Model(_) | LensError::Network(_)),
            "got {err:?}"
        );
    }

    // --- streaming surface --------------------------------------------------

    #[tokio::test]
    async fn default_generate_stream_yields_text_then_done() {
        struct Fixed;
        #[async_trait]
        impl LlmProvider for Fixed {
            fn model_id(&self) -> &str {
                "fixed"
            }
            async fn reachable(&self) -> bool {
                true
            }
            async fn generate(&self, _req: &LlmRequest) -> Result<LlmResponse, LensError> {
                Ok(LlmResponse {
                    text: "answer".to_string(),
                    tokens_used: 7,
                })
            }
        }

        let provider = Fixed;
        let stream = provider.generate_stream(&req()).await.unwrap();
        let events: Vec<_> = stream.collect().await;
        let events: Vec<StreamChunk> = events.into_iter().map(|e| e.unwrap()).collect();
        assert_eq!(
            events,
            vec![
                StreamChunk::TextDelta("answer".to_string()),
                StreamChunk::Done { tokens_used: 7 },
            ]
        );
    }

    #[tokio::test]
    async fn genai_generate_stream_yields_deltas_and_done() {
        // genai's Ollama adapter buffers a non-streamed body into a single chunk + End,
        // so this NDJSON-less round-trip still exercises our TextDelta + Done mapping.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ollama_chat_body("streamed")))
            .mount(&server)
            .await;

        let provider: Arc<dyn LlmProvider> = Arc::new(GenaiProvider::new(
            AdapterKind::Ollama,
            "llama3",
            &server.uri(),
            "",
        ));
        let stream = provider.generate_stream(&req()).await.unwrap();
        let events: Vec<StreamChunk> = stream.map(|e| e.unwrap()).collect().await;

        let text: String = events
            .iter()
            .filter_map(|e| match e {
                StreamChunk::TextDelta(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(text.contains("streamed"), "got deltas: {events:?}");
        assert!(
            matches!(events.last(), Some(StreamChunk::Done { .. })),
            "stream must end in Done; got {events:?}"
        );
    }

    // --- model_id accessor (cache-key component, AC9) -----------------------

    #[test]
    fn model_id_returns_configured_model() {
        let p = GenaiProvider::new(
            AdapterKind::Ollama,
            "llama3.1",
            "http://localhost:11434",
            "",
        );
        assert_eq!(p.model_id(), "llama3.1");
        let a = GenaiProvider::new(
            AdapterKind::Anthropic,
            "claude-opus-4-8",
            "https://api.anthropic.com",
            "k",
        );
        assert_eq!(a.model_id(), "claude-opus-4-8");
    }

    // --- routing / factory (Stage 2) ----------------------------------------

    fn config_with(models: Vec<ModelConfig>, routing: LlmRouting) -> AppConfig {
        AppConfig {
            models,
            enrichment: crate::config::EnrichmentConfig {
                routing,
                ..crate::config::EnrichmentConfig::default()
            },
            ..AppConfig::default()
        }
    }

    fn ollama_entry() -> ModelConfig {
        ModelConfig {
            provider: "ollama".to_string(),
            base_url: "http://localhost:11434".to_string(),
            model: "llama3".to_string(),
            ..ModelConfig::default()
        }
    }

    fn anthropic_entry(model: &str) -> ModelConfig {
        ModelConfig {
            provider: "anthropic".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            model: model.to_string(),
            api_key: "sk-ant".to_string(),
            ..ModelConfig::default()
        }
    }

    fn google_entry(model: &str) -> ModelConfig {
        ModelConfig {
            provider: "google".to_string(),
            base_url: "https://generativelanguage.googleapis.com/v1beta/openai".to_string(),
            model: model.to_string(),
            api_key: "g-key".to_string(),
            ..ModelConfig::default()
        }
    }

    fn custom_openai_entry(model: &str) -> ModelConfig {
        ModelConfig {
            provider: "openai-compatible".to_string(),
            base_url: "http://localhost:1234/v1".to_string(),
            model: model.to_string(),
            api_key: "sk-local".to_string(),
            ..ModelConfig::default()
        }
    }

    fn catalog_model(provider: &str) -> String {
        let catalog = ModelCatalog::bundled();
        catalog
            .provider(provider)
            .and_then(|p| p.models.keys().next())
            .cloned()
            .unwrap_or_else(|| panic!("bundled catalog has at least one {provider} model"))
    }

    fn catalog_anthropic_model() -> String {
        catalog_model("anthropic")
    }

    #[test]
    fn local_first_selects_ollama_without_consent() {
        let cfg = config_with(vec![ollama_entry()], LlmRouting::LocalFirst);
        let p = provider_from_config(&cfg, false).expect("ollama selected");
        assert_eq!(p.model_id(), "llama3");
    }

    #[test]
    fn cloud_first_prefers_consented_catalog_valid_cloud() {
        let model = catalog_anthropic_model();
        let cfg = config_with(
            vec![ollama_entry(), anthropic_entry(&model)],
            LlmRouting::CloudFirst,
        );
        let p = provider_from_config(&cfg, true).expect("cloud preferred");
        assert_eq!(p.model_id(), model);
    }

    #[test]
    fn cloud_first_falls_back_to_local_without_consent() {
        let model = catalog_anthropic_model();
        let cfg = config_with(
            vec![anthropic_entry(&model), ollama_entry()],
            LlmRouting::CloudFirst,
        );
        let p = provider_from_config(&cfg, false).expect("falls back to local");
        assert_eq!(p.model_id(), "llama3");
    }

    #[test]
    fn cloud_uncatalogued_model_resolves_when_keyed_and_consented() {
        // Catalog membership is advisory: a model newer than the bundled snapshot must
        // still resolve when keyed + consented.
        let cfg = config_with(
            vec![anthropic_entry("totally-made-up-model")],
            LlmRouting::CloudFirst,
        );
        let p = provider_from_config(&cfg, true).expect("uncatalogued cloud model is usable");
        assert_eq!(p.model_id(), "totally-made-up-model");
    }

    #[test]
    fn anthropic_provider_validates_against_own_namespace() {
        // Fix #1: must validate claude-* against ANTHROPIC namespace, not "openai".
        let model = catalog_model("anthropic");
        assert!(model.starts_with("claude"), "expected a claude-* model");
        let cfg = config_with(vec![anthropic_entry(&model)], LlmRouting::CloudFirst);
        let p = provider_from_config(&cfg, true).expect("anthropic (claude-*) must select");
        assert_eq!(p.model_id(), model);
    }

    #[test]
    fn google_provider_validates_against_own_namespace() {
        let model = catalog_model("google");
        assert!(model.starts_with("gemini"), "expected a gemini-* model");
        let cfg = config_with(vec![google_entry(&model)], LlmRouting::CloudFirst);
        let p = provider_from_config(&cfg, true).expect("google (gemini-*) must select");
        assert_eq!(p.model_id(), model);
    }

    #[test]
    fn custom_openai_compatible_is_consent_gated_but_unvalidated() {
        let cfg = config_with(
            vec![custom_openai_entry("some-self-hosted-model-v3")],
            LlmRouting::CloudFirst,
        );
        let p = provider_from_config(&cfg, true).expect("custom endpoint selects with consent");
        assert_eq!(p.model_id(), "some-self-hosted-model-v3");
        assert!(
            provider_from_config(&cfg, false).is_none(),
            "custom endpoint is consent-gated"
        );
    }

    #[test]
    fn legacy_openai_compatible_config_still_works_as_custom_endpoint() {
        let cfg = config_with(
            vec![custom_openai_entry("gpt-4o")],
            LlmRouting::Explicit {
                provider: "openai-compatible".to_string(),
                model: "gpt-4o".to_string(),
            },
        );
        let p = provider_from_config(&cfg, true).expect("legacy openai-compatible resolves");
        assert_eq!(p.model_id(), "gpt-4o");
    }

    // --- newly-surfaced native cloud providers (M4 Phase 3) -----------------

    #[test]
    fn adapter_for_maps_new_native_providers() {
        assert!(matches!(adapter_for("groq"), Some(AdapterKind::Groq)));
        assert!(matches!(
            adapter_for("deepseek"),
            Some(AdapterKind::DeepSeek)
        ));
        assert!(matches!(adapter_for("xai"), Some(AdapterKind::Xai)));
        assert!(matches!(adapter_for("cohere"), Some(AdapterKind::Cohere)));
    }

    #[test]
    fn native_endpoint_covers_new_providers_and_skips_custom_local() {
        for adapter in [
            AdapterKind::Groq,
            AdapterKind::DeepSeek,
            AdapterKind::Xai,
            AdapterKind::Cohere,
            AdapterKind::OpenAI,
            AdapterKind::Anthropic,
            AdapterKind::Gemini,
            AdapterKind::Zai,
            AdapterKind::OllamaCloud,
        ] {
            assert!(
                native_endpoint(adapter).is_some(),
                "{adapter:?} must have a canonical endpoint"
            );
        }
        assert!(native_endpoint(AdapterKind::Ollama).is_none());
    }

    fn native_cloud_entry(provider: &str, model: &str) -> ModelConfig {
        ModelConfig {
            provider: provider.to_string(),
            base_url: String::new(),
            model: model.to_string(),
            api_key: "k".to_string(),
            ..ModelConfig::default()
        }
    }

    #[test]
    fn groq_selects_and_validates_against_groq_namespace() {
        let model = catalog_model("groq");
        let cfg = config_with(
            vec![native_cloud_entry("groq", &model)],
            LlmRouting::CloudFirst,
        );
        let p = provider_from_config(&cfg, true).expect("groq must select with consent");
        assert_eq!(p.model_id(), model);
    }

    #[test]
    fn deepseek_selects_and_validates_against_deepseek_namespace() {
        let model = catalog_model("deepseek");
        let cfg = config_with(
            vec![native_cloud_entry("deepseek", &model)],
            LlmRouting::CloudFirst,
        );
        let p = provider_from_config(&cfg, true).expect("deepseek must select with consent");
        assert_eq!(p.model_id(), model);
    }

    #[test]
    fn new_native_cloud_provider_rejected_without_consent() {
        let model = catalog_model("groq");
        let cfg = config_with(
            vec![native_cloud_entry("groq", &model)],
            LlmRouting::CloudFirst,
        );
        assert!(
            provider_from_config(&cfg, false).is_none(),
            "groq is a cloud provider and must be consent-gated"
        );
    }

    #[test]
    fn new_native_cloud_uncatalogued_model_resolves_with_consent() {
        // Advisory catalog: an uncatalogued groq model is usable once keyed + consented.
        let cfg = config_with(
            vec![native_cloud_entry("groq", "totally-made-up-model")],
            LlmRouting::CloudFirst,
        );
        let p = provider_from_config(&cfg, true).expect("uncatalogued groq model is usable");
        assert_eq!(p.model_id(), "totally-made-up-model");
    }

    #[test]
    fn native_cloud_entry_with_empty_base_url_is_usable() {
        // Regression guard: old guards rejected an empty base_url; native cloud adapters need none.
        let model = catalog_model("xai");
        assert!(has_endpoint(&native_cloud_entry("xai", &model)));
        assert!(!has_endpoint(&ModelConfig {
            provider: "ollama".to_string(),
            base_url: String::new(),
            model: "llama3".to_string(),
            ..ModelConfig::default()
        }));
    }

    #[test]
    fn blank_base_openai_compatible_is_unavailable() {
        // #273: openai-compatible shares AdapterKind::OpenAI (which has a native endpoint), but the
        // builder requires an explicit base_url — availability must agree, not report it usable.
        let entry = ModelConfig {
            provider: "openai-compatible".to_string(),
            base_url: String::new(),
            model: "some-self-hosted-model".to_string(),
            ..ModelConfig::default()
        };
        assert!(!has_endpoint(&entry));
        assert_eq!(
            candidate_unavailable_reason(&entry, true),
            Some("base URL required".to_string())
        );
        assert!(build_provider(&entry).is_none());
    }

    #[test]
    fn build_provider_raw_rejects_blank_base_openai_compatible() {
        // #273: interactive validation (#90) must also fail-closed on a blank-base openai-compatible
        // entry — otherwise the genai backend silently probes api.openai.com with the typed api_key.
        assert!(build_provider_raw("openai-compatible", "some-model", "", "sk-typed").is_none());
        assert!(
            build_provider_raw(
                "openai-compatible",
                "some-model",
                "http://localhost:1234",
                "k"
            )
            .is_some()
        );
    }

    #[test]
    fn select_provider_skips_unavailable_cloud_and_falls_to_local() {
        // #273: a blank-base openai-compatible entry is now unusable, so CloudFirst skips it via the
        // find_map chain and falls to the buildable local Ollama instead of stranding on it.
        let entry = |provider: &str, base_url: &str, model: &str, api_key: &str| ModelConfig {
            provider: provider.to_string(),
            base_url: base_url.to_string(),
            model: model.to_string(),
            api_key: api_key.to_string(),
            ..ModelConfig::default()
        };
        let models = vec![
            entry("openai-compatible", "", "some-model", "k"),
            entry("ollama", "http://localhost:11434", "llama3", ""),
        ];
        let provider = select_provider(&models, &LlmRouting::CloudFirst, true)
            .expect("must fall through to the buildable local candidate");
        assert_eq!(provider.model_id(), "llama3");
    }

    #[test]
    fn existing_native_providers_still_use_configured_base_url() {
        let model = catalog_anthropic_model();
        let cfg = config_with(vec![anthropic_entry(&model)], LlmRouting::CloudFirst);
        let p = provider_from_config(&cfg, true).expect("anthropic with base_url still selects");
        assert_eq!(p.model_id(), model);
    }

    #[test]
    fn explicit_pins_exact_provider_model() {
        let model = catalog_anthropic_model();
        let cfg = config_with(
            vec![ollama_entry(), anthropic_entry(&model)],
            LlmRouting::Explicit {
                provider: "anthropic".to_string(),
                model: model.clone(),
            },
        );
        let p = provider_from_config(&cfg, true).expect("explicit anthropic");
        assert_eq!(p.model_id(), model);
    }

    #[test]
    fn explicit_local_does_not_require_consent_or_catalog() {
        let cfg = config_with(
            vec![ollama_entry()],
            LlmRouting::Explicit {
                provider: "ollama".to_string(),
                model: "llama3".to_string(),
            },
        );
        let p = provider_from_config(&cfg, false).expect("explicit local");
        assert_eq!(p.model_id(), "llama3");
    }

    #[test]
    fn none_when_no_usable_models() {
        let cfg = config_with(vec![], LlmRouting::CloudFirst);
        assert!(provider_from_config(&cfg, true).is_none());
    }

    #[test]
    fn skips_incomplete_and_unknown_entries() {
        let cfg = config_with(
            vec![
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
                ollama_entry(),
            ],
            LlmRouting::LocalFirst,
        );
        let p = provider_from_config(&cfg, false).expect("valid ollama selected");
        assert_eq!(p.model_id(), "llama3");
    }

    #[test]
    fn routing_default_is_cloud_first() {
        assert_eq!(LlmRouting::default(), LlmRouting::CloudFirst);
    }

    #[test]
    fn routing_serializes_snake_case_tagged() {
        assert_eq!(
            serde_json::to_value(LlmRouting::CloudFirst).unwrap(),
            serde_json::json!({ "kind": "cloud_first" })
        );
        assert_eq!(
            serde_json::to_value(LlmRouting::Explicit {
                provider: "anthropic".to_string(),
                model: "claude".to_string(),
            })
            .unwrap(),
            serde_json::json!({ "kind": "explicit", "provider": "anthropic", "model": "claude" })
        );
    }

    // --- chat_model seam (Variant B) ----------------------------------------

    fn config_with_chat(
        models: Vec<ModelConfig>,
        routing: LlmRouting,
        chat_model: Option<crate::config::TaskModel>,
    ) -> AppConfig {
        AppConfig {
            models,
            enrichment: crate::config::EnrichmentConfig {
                routing,
                chat_model,
                ..crate::config::EnrichmentConfig::default()
            },
            ..AppConfig::default()
        }
    }

    #[test]
    fn chat_model_pin_outranks_routing() {
        // chat_model pins local Ollama while routing=CloudFirst + a consented cloud entry
        // is present: the pin wins, not the routing-preferred cloud model.
        let cloud = catalog_anthropic_model();
        let cfg = config_with_chat(
            vec![ollama_entry(), anthropic_entry(&cloud)],
            LlmRouting::CloudFirst,
            Some(crate::config::TaskModel {
                provider: "ollama".to_string(),
                model: "llama3".to_string(),
            }),
        );
        let p = chat_provider_from_config(&cfg, true).expect("chat_model pin resolves");
        assert_eq!(p.model_id(), "llama3", "chat_model pin outranks CloudFirst");
    }

    #[test]
    fn chat_model_none_falls_back_to_routing() {
        let cloud = catalog_anthropic_model();
        let cfg = config_with_chat(
            vec![ollama_entry(), anthropic_entry(&cloud)],
            LlmRouting::CloudFirst,
            None,
        );
        let p = chat_provider_from_config(&cfg, true).expect("routing fallback resolves");
        assert_eq!(
            p.model_id(),
            cloud,
            "no pin → routing (CloudFirst) selects cloud"
        );
    }

    #[test]
    fn chat_model_unusable_cloud_without_consent_is_none() {
        let cloud = catalog_anthropic_model();
        let cfg = config_with_chat(
            vec![ollama_entry(), anthropic_entry(&cloud)],
            LlmRouting::CloudFirst,
            Some(crate::config::TaskModel {
                provider: "anthropic".to_string(),
                model: cloud,
            }),
        );
        assert!(
            chat_provider_from_config(&cfg, false).is_none(),
            "cloud pin without consent must not report a provider (no routing fallback)"
        );
    }

    #[test]
    fn chat_model_unusable_empty_model_is_none() {
        let cfg = config_with_chat(
            vec![ollama_entry()],
            LlmRouting::LocalFirst,
            Some(crate::config::TaskModel {
                provider: "ollama".to_string(),
                model: String::new(),
            }),
        );
        assert!(
            chat_provider_from_config(&cfg, true).is_none(),
            "empty-model pin must not report a provider"
        );
    }

    // --- per-task provider override (Stage 3) -------------------------------

    use crate::config::TaskModel;

    fn base_genai(model: &str) -> Arc<dyn LlmProvider> {
        Arc::new(GenaiProvider::new(
            AdapterKind::Ollama,
            model,
            "http://localhost:11434",
            "",
        ))
    }

    #[test]
    fn task_provider_falls_back_to_base_when_override_unset() {
        let base = base_genai("qwen2.5-instruct");
        let models = vec![ollama_entry()];
        let p = task_provider_from_config(&base, None, &models, false);
        assert_eq!(p.model_id(), "qwen2.5-instruct");
    }

    #[test]
    fn task_provider_pins_local_override_model() {
        let base = base_genai("qwen2.5-instruct");
        let models = vec![ModelConfig {
            provider: "ollama".to_string(),
            base_url: "http://localhost:11434".to_string(),
            model: "qwen2.5-instruct".to_string(),
            ..ModelConfig::default()
        }];
        let coref = TaskModel {
            provider: "ollama".to_string(),
            model: "qwen2.5-coder".to_string(),
        };
        let p = task_provider_from_config(&base, Some(&coref), &models, false);
        assert_eq!(p.model_id(), "qwen2.5-coder", "coref pins the coder model");
    }

    #[test]
    fn task_provider_pins_consented_catalog_valid_cloud_override() {
        let model = catalog_anthropic_model();
        let base = base_genai("qwen2.5-instruct");
        let models = vec![anthropic_entry(&model)];
        let map = TaskModel {
            provider: "anthropic".to_string(),
            model: model.clone(),
        };
        let p = task_provider_from_config(&base, Some(&map), &models, true);
        assert_eq!(p.model_id(), model);
    }

    #[test]
    fn task_provider_rejects_cloud_override_without_consent() {
        let model = catalog_anthropic_model();
        let base = base_genai("qwen2.5-instruct");
        let models = vec![anthropic_entry(&model)];
        let map = TaskModel {
            provider: "anthropic".to_string(),
            model,
        };
        // No consent ⇒ cloud override rejected ⇒ falls back to base.
        let p = task_provider_from_config(&base, Some(&map), &models, false);
        assert_eq!(p.model_id(), "qwen2.5-instruct");
    }

    #[test]
    fn task_provider_pins_uncatalogued_cloud_override_with_consent() {
        // Advisory catalog: an uncatalogued cloud override still pins once consented.
        let base = base_genai("qwen2.5-instruct");
        let models = vec![anthropic_entry("totally-made-up-model")];
        let map = TaskModel {
            provider: "anthropic".to_string(),
            model: "totally-made-up-model".to_string(),
        };
        let p = task_provider_from_config(&base, Some(&map), &models, true);
        assert_eq!(p.model_id(), "totally-made-up-model");
    }

    #[test]
    fn task_provider_falls_back_when_no_matching_config_entry() {
        let base = base_genai("qwen2.5-instruct");
        let models = vec![ollama_entry()];
        let coref = TaskModel {
            provider: "anthropic".to_string(),
            model: catalog_anthropic_model(),
        };
        let p = task_provider_from_config(&base, Some(&coref), &models, true);
        assert_eq!(p.model_id(), "qwen2.5-instruct");
    }

    // --- active_model_candidates (selector enumeration) ----------------------

    fn candidate<'a>(
        list: &'a [ActiveModelCandidate],
        provider: &str,
        model: &str,
    ) -> &'a ActiveModelCandidate {
        list.iter()
            .find(|c| c.provider == provider && c.model == model)
            .unwrap_or_else(|| panic!("candidate {provider}/{model} missing"))
    }

    #[test]
    fn candidates_mark_usable_local_available() {
        let cfg = config_with(vec![ollama_entry()], LlmRouting::CloudFirst);
        let out = active_model_candidates(&cfg, false);
        let c = candidate(&out, "ollama", "llama3");
        assert!(
            c.available,
            "usable local entry is available without consent"
        );
        assert_eq!(c.reason, None);
        assert_eq!(c.label, "Ollama · llama3");
    }

    #[test]
    fn candidates_gate_cloud_on_consent() {
        let model = catalog_anthropic_model();
        let cfg = config_with(vec![anthropic_entry(&model)], LlmRouting::CloudFirst);

        let denied = active_model_candidates(&cfg, false);
        let c = candidate(&denied, "anthropic", &model);
        assert!(!c.available, "cloud entry needs consent");
        assert_eq!(c.reason.as_deref(), Some("cloud consent required"));

        let granted = active_model_candidates(&cfg, true);
        let c = candidate(&granted, "anthropic", &model);
        assert!(
            c.available,
            "consent + catalog-valid cloud entry is available"
        );
        assert_eq!(c.reason, None);
    }

    #[test]
    fn candidates_accept_uncatalogued_cloud_model_with_consent() {
        // Advisory catalog: a keyed + consented model absent from the bundled snapshot is
        // reported available (reason=None), not a false-negative "not in catalog".
        let cfg = config_with(
            vec![anthropic_entry("totally-made-up-model")],
            LlmRouting::CloudFirst,
        );
        let out = active_model_candidates(&cfg, true);
        let c = candidate(&out, "anthropic", "totally-made-up-model");
        assert!(c.available);
        assert_eq!(c.reason, None);
    }

    #[test]
    fn uncatalogued_cloud_model_is_available_and_pinnable() {
        // Advisory catalog end-to-end: a keyed + consented cloud model newer than the
        // bundled snapshot reports available AND resolves to a real pinned provider.
        let cfg = config_with(
            vec![anthropic_entry("claude-future-99")],
            LlmRouting::CloudFirst,
        );
        let out = active_model_candidates(&cfg, true);
        let c = candidate(&out, "anthropic", "claude-future-99");
        assert!(c.available);
        assert_eq!(c.reason, None);

        let pinned = build_pinned_provider("anthropic", "claude-future-99", &cfg.models, true)
            .expect("uncatalogued keyed cloud model is pinnable");
        assert_eq!(pinned.model_id(), "claude-future-99");
    }

    // A-T2: credential-only (empty-model) entries are excluded from candidates; only
    // pinnable-eligible (non-empty model) entries appear.
    #[test]
    fn candidates_exclude_empty_model_entries() {
        let credential_only = ModelConfig {
            provider: "ollama".to_string(),
            base_url: "http://localhost:11434".to_string(),
            model: String::new(),
            ..ModelConfig::default()
        };
        let cfg = config_with(
            vec![credential_only, ollama_entry()],
            LlmRouting::CloudFirst,
        );
        let out = active_model_candidates(&cfg, false);
        assert_eq!(
            out.len(),
            1,
            "only the non-empty-model entry is a candidate"
        );
        assert_eq!(out[0].provider, "ollama");
        assert_eq!(out[0].model, "llama3");
        assert!(
            out.iter().all(|c| !c.model.is_empty()),
            "no empty-model candidate leaks through"
        );
    }

    // A-T5: a credential-only cloud entry (saved key, empty model) is not a usable chat
    // provider and is not pinnable — the chat gate and pin builder both return None.
    #[test]
    fn credential_only_cloud_entry_is_not_a_chat_provider() {
        let credential_only = ModelConfig {
            provider: "anthropic".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            model: String::new(),
            api_key: "sk-ant".to_string(),
            ..ModelConfig::default()
        };
        let cfg = config_with(vec![credential_only.clone()], LlmRouting::CloudFirst);

        // No pin → routing resolution; the empty-model entry is not usable (mirrors
        // src-tauri `has_chat_provider`, which builds no client either).
        assert!(chat_provider_from_config(&cfg, true).is_none());

        // A pin to the empty model does not resolve.
        assert!(build_pinned_provider("anthropic", "", &[credential_only], true).is_none());
    }

    #[test]
    fn candidates_omit_non_llm_providers() {
        let cfg = config_with(
            vec![
                ollama_entry(),
                ModelConfig {
                    provider: "fastembed".to_string(),
                    model: "bge-small".to_string(),
                    ..ModelConfig::default()
                },
            ],
            LlmRouting::CloudFirst,
        );
        let out = active_model_candidates(&cfg, false);
        assert_eq!(out.len(), 1, "the embedding entry has no genai adapter");
        assert_eq!(out[0].provider, "ollama");
    }

    // --- is_ollama trait capability (backend-agnostic preflight seam, #256 §0.1 #1) --------

    #[test]
    fn genai_is_ollama_via_trait() {
        // Exercised through `dyn LlmProvider` (as the preflight sites now are), not the concrete
        // type: an Ollama target reports `true`, a cloud target `false`.
        let ollama: Arc<dyn LlmProvider> = Arc::new(GenaiProvider::new(
            AdapterKind::Ollama,
            "llama3",
            "http://x",
            "",
        ));
        assert!(ollama.is_ollama());
        assert!(ollama.is_local());
        let cloud: Arc<dyn LlmProvider> = Arc::new(GenaiProvider::new(
            AdapterKind::Anthropic,
            "claude",
            "https://api.anthropic.com",
            "k",
        ));
        assert!(!cloud.is_ollama());
        assert!(!cloud.is_local());
    }
}

// ---------------------------------------------------------------------------
// rig backend tests (Phase 0 #256) — offline (wiremock); real-model behind
// LENS_RUN_MODEL_TESTS is a separate exit gate not run in CI.
// ---------------------------------------------------------------------------
#[cfg(all(test, feature = "llm-backend-rig"))]
mod rig_tests {
    use super::*;
    use futures_util::StreamExt;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Nothing binds `127.0.0.1:1` — connection is deterministically refused.
    const DEAD_URL: &str = "http://127.0.0.1:1";

    fn req() -> LlmRequest {
        LlmRequest {
            system: Some("be terse".to_string()),
            prompt: "hello".to_string(),
            max_tokens: 64,
            temperature: 0.0,
            json: true,
            thinking: false,
            reasoning_effort: None,
            messages: Vec::new(),
        }
    }

    /// A non-streaming Ollama `/api/chat` body. `created_at` is required by rig's typed
    /// `CompletionResponse`, unlike the genai adapter's shape.
    fn ollama_chat_body(content: &str) -> serde_json::Value {
        serde_json::json!({
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00Z",
            "message": { "role": "assistant", "content": content },
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 5
        })
    }

    /// A non-terminal (`done: false`) NDJSON line, as Ollama emits mid-stream.
    fn ollama_chat_body_in_progress(content: &str) -> serde_json::Value {
        serde_json::json!({
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00Z",
            "message": { "role": "assistant", "content": content },
            "done": false
        })
    }

    #[tokio::test]
    async fn rig_generate_round_trips_ollama() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ollama_chat_body("hi there")))
            .mount(&server)
            .await;

        let provider = RigProvider::new_ollama("llama3", &server.uri(), "").unwrap();
        let resp = provider.generate(&req()).await.unwrap();
        assert_eq!(resp.text, "hi there");
        assert_eq!(
            resp.tokens_used, 15,
            "10 prompt + 5 eval tokens (u64→u32 cast)"
        );
    }

    /// BLOCKING gate (#256 §0.1 #4): the `json` directive must land as Ollama's TOP-LEVEL
    /// `format` field — a JSON Schema object (rig cannot carry the bare string `"json"`) — and
    /// must NOT leak into `options`, where a naive `additional_params` passthrough would put it.
    #[tokio::test]
    async fn rig_json_directive_lands_top_level_format() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ollama_chat_body("{}")))
            .mount(&server)
            .await;

        let provider = RigProvider::new_ollama("llama3", &server.uri(), "").unwrap();
        provider.generate(&req()).await.unwrap();

        let requests = server
            .received_requests()
            .await
            .expect("wiremock request recording is on by default");
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert!(
            body.get("format").is_some(),
            "format missing at top level: {body}"
        );
        assert_eq!(
            body["format"]["type"], "object",
            "unexpected format schema: {body}"
        );
        assert!(
            body["options"].get("format").is_none(),
            "format leaked into options (wrong merge level): {body}"
        );
    }

    #[tokio::test]
    async fn rig_no_json_omits_format() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ollama_chat_body("plain")))
            .mount(&server)
            .await;

        let provider = RigProvider::new_ollama("llama3", &server.uri(), "").unwrap();
        provider
            .generate(&LlmRequest {
                json: false,
                ..req()
            })
            .await
            .unwrap();

        let requests = server.received_requests().await.expect("recording on");
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert!(
            body.get("format").is_none(),
            "format set without json: {body}"
        );
    }

    /// Ollama only honors a token cap via `options.num_predict` — the request builder's
    /// `max_tokens` maps to a bare top-level field the real server ignores — so this must be
    /// threaded through `additional_params` and land inside `options`, merged alongside a `json`
    /// or `thinking` directive rather than clobbering it.
    #[tokio::test]
    async fn rig_max_tokens_lands_in_options_num_predict() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ollama_chat_body("ok")))
            .mount(&server)
            .await;

        let provider = RigProvider::new_ollama("llama3", &server.uri(), "").unwrap();
        provider
            .generate(&LlmRequest {
                max_tokens: 256,
                thinking: true,
                ..req()
            })
            .await
            .unwrap();

        let requests = server.received_requests().await.expect("recording on");
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(
            body["options"]["num_predict"], 256,
            "max_tokens must reach options.num_predict: {body}"
        );
        assert!(
            body.get("think").is_some(),
            "think must still be set alongside num_predict: {body}"
        );
    }

    /// A minimal OpenAI chat-completions body — enough for these tests, which assert on the
    /// OUTBOUND request (`reasoning_effort` mapping), not the parsed response.
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

    /// Stage 4: OpenAI-family reasoning maps to a top-level `reasoning_effort` string — never
    /// Ollama's `think`/`num_predict`. Targets the outbound request; the response is irrelevant.
    #[tokio::test]
    async fn rig_openai_reasoning_effort_lands_top_level() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_chat_body("ok")))
            .mount(&server)
            .await;

        let provider = RigProvider::new_openai("gpt", &server.uri(), "k").unwrap();
        let _ = provider
            .generate(&LlmRequest {
                thinking: true,
                reasoning_effort: Some(ReasoningEffort::High),
                ..req()
            })
            .await;

        let requests = server.received_requests().await.expect("recording on");
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(
            body["reasoning_effort"], "high",
            "reasoning_effort missing or mis-leveled: {body}"
        );
        assert!(
            body.get("think").is_none() && body.get("num_predict").is_none(),
            "Ollama-only knobs must not appear on the OpenAI wire: {body}"
        );
    }

    /// Stage 5: on the OpenAI wire the `json` directive lands as a top-level `response_format`
    /// (rig maps `output_schema`→`response_format`), NOT Ollama's `format`; absent without `json`.
    #[tokio::test]
    async fn rig_openai_json_directive_lands_as_response_format() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_chat_body("ok")))
            .mount(&server)
            .await;

        let provider = RigProvider::new_openai("gpt", &server.uri(), "k").unwrap();
        let _ = provider.generate(&req()).await; // req() has json: true
        let _ = provider
            .generate(&LlmRequest {
                json: false,
                ..req()
            })
            .await;

        let requests = server.received_requests().await.expect("recording on");
        let with_json: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        let without_json: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
        assert!(
            with_json.get("response_format").is_some(),
            "json → response_format at top level: {with_json}"
        );
        assert!(
            with_json.get("format").is_none(),
            "OpenAI must not use Ollama's `format` key: {with_json}"
        );
        assert!(
            without_json.get("response_format").is_none(),
            "response_format set without json: {without_json}"
        );
    }

    #[tokio::test]
    async fn rig_openai_omits_reasoning_effort_without_thinking() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_chat_body("ok")))
            .mount(&server)
            .await;

        let provider = RigProvider::new_openai("gpt", &server.uri(), "k").unwrap();
        let _ = provider.generate(&req()).await;

        let requests = server.received_requests().await.expect("recording on");
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert!(
            body.get("reasoning_effort").is_none(),
            "reasoning_effort set without thinking: {body}"
        );
    }

    /// Stage 4: Anthropic extended thinking maps to `thinking: {type, budget_tokens}` and forces
    /// `temperature == 1` (Anthropic rejects any other value with thinking on).
    #[tokio::test]
    async fn rig_anthropic_thinking_lands_with_budget_and_temperature_one() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .mount(&server)
            .await;

        let provider = RigProvider::new_anthropic("claude", &server.uri(), "k").unwrap();
        let _ = provider
            .generate(&LlmRequest {
                thinking: true,
                reasoning_effort: Some(ReasoningEffort::High),
                max_tokens: 20_000,
                temperature: 0.0,
                ..req()
            })
            .await;

        let requests = server.received_requests().await.expect("recording on");
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(
            body["thinking"]["type"], "enabled",
            "thinking not enabled: {body}"
        );
        assert_eq!(
            body["thinking"]["budget_tokens"], 8192,
            "High effort budget: {body}"
        );
        assert_eq!(
            body["temperature"], 1.0,
            "Anthropic thinking requires temperature 1: {body}"
        );
    }

    /// A `max_tokens` too small for Anthropic's 1024 floor below the cap disables thinking (and
    /// leaves the configured temperature untouched).
    #[tokio::test]
    async fn rig_anthropic_thinking_omitted_when_cap_too_small() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .mount(&server)
            .await;

        let provider = RigProvider::new_anthropic("claude", &server.uri(), "k").unwrap();
        let _ = provider
            .generate(&LlmRequest {
                thinking: true,
                reasoning_effort: Some(ReasoningEffort::High),
                max_tokens: 64,
                temperature: 0.0,
                ..req()
            })
            .await;

        let requests = server.received_requests().await.expect("recording on");
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert!(
            body.get("thinking").is_none(),
            "thinking must be omitted when the cap can't fit the 1024 floor: {body}"
        );
        assert_eq!(
            body["temperature"], 0.0,
            "temperature must stay configured when thinking is disabled: {body}"
        );
    }

    #[tokio::test]
    async fn rig_generate_stream_yields_text_then_done() {
        let server = MockServer::start().await;
        // The Ollama streaming path parses NDJSON (newline-terminated lines); a single
        // done-terminated line yields one text delta + a final response.
        let ndjson = format!("{}\n", ollama_chat_body("streamed"));
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_string(ndjson))
            .mount(&server)
            .await;

        let provider = RigProvider::new_ollama("llama3", &server.uri(), "").unwrap();
        let stream = provider.generate_stream(&req()).await.unwrap();
        let events: Vec<StreamChunk> = stream.map(|e| e.unwrap()).collect().await;

        let text: String = events
            .iter()
            .filter_map(|e| match e {
                StreamChunk::TextDelta(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(text.contains("streamed"), "got deltas: {events:?}");
        assert!(
            matches!(events.last(), Some(StreamChunk::Done { tokens_used: 15 })),
            "stream must end in Done with usage; got {events:?}"
        );
    }

    /// A malformed NDJSON line after a valid partial chunk fails rig's internal
    /// `serde_json` parse mid-stream (the HTTP response itself is a 200 — this is distinct from
    /// `rig_generate_non_success_is_sanitized_model_error`, which fails before streaming starts).
    /// The stream must surface exactly one sanitized `Err` and never a `Done` after it.
    #[tokio::test]
    async fn rig_generate_stream_mid_stream_error_yields_err_no_done() {
        let server = MockServer::start().await;
        let body = format!(
            "{}\nnot valid json\n",
            ollama_chat_body_in_progress("partial")
        );
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .mount(&server)
            .await;

        let provider = RigProvider::new_ollama("llama3", &server.uri(), "").unwrap();
        let stream = provider.generate_stream(&req()).await.unwrap();
        let events: Vec<Result<StreamChunk, LensError>> = stream.collect().await;

        assert!(
            matches!(events.first(), Some(Ok(StreamChunk::TextDelta(t))) if t == "partial"),
            "expected the valid partial chunk first; got {events:?}"
        );
        assert!(
            matches!(events.last(), Some(Err(LensError::Model(_)))),
            "a mid-stream parse failure must surface as a single Err; got {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamChunk::Done { .. }))),
            "no Done may follow a mid-stream error: {events:?}"
        );
    }

    #[tokio::test]
    async fn rig_generate_non_success_is_sanitized_model_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom LEAK_SECRET_TOKEN"))
            .mount(&server)
            .await;

        let provider = RigProvider::new_ollama("llama3", &server.uri(), "").unwrap();
        let err = provider.generate(&req()).await.unwrap_err();
        assert!(
            matches!(err, LensError::Model(_)),
            "4xx/5xx → Model; got {err:?}"
        );
        assert!(
            !err.message().contains("LEAK_SECRET_TOKEN"),
            "provider body must not cross IPC: {err:?}"
        );
    }

    #[tokio::test]
    async fn rig_generate_transport_failure_is_network_error() {
        let provider = RigProvider::new_ollama("llama3", DEAD_URL, "").unwrap();
        let err = provider.generate(&req()).await.unwrap_err();
        assert!(
            matches!(err, LensError::Network(_)),
            "connection refused → Network; got {err:?}"
        );
    }

    /// D5: a *cloud* HTTP-status error (auth/rate-limit) is semantic, not transport — it must map
    /// to `Model` (never `Network`, which the old blanket `ProviderError` rule would have done)
    /// and never leak the provider's response body across IPC.
    #[tokio::test]
    async fn rig_cloud_status_error_is_sanitized_model_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401).set_body_string("bad key LEAK_SECRET_TOKEN"))
            .mount(&server)
            .await;

        let provider = RigProvider::new_openai("gpt", &server.uri(), "k").unwrap();
        let err = provider.generate(&req()).await.unwrap_err();
        assert!(
            matches!(err, LensError::Model(_)),
            "cloud 401 → Model; got {err:?}"
        );
        assert!(
            !err.message().contains("LEAK_SECRET_TOKEN"),
            "provider body must not cross IPC: {err:?}"
        );
    }

    #[tokio::test]
    async fn rig_reachable_true_without_billed_generate() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/version"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "version": "0.1.0" })),
            )
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ollama_chat_body("x")))
            .expect(0) // any billed generate dispatched by the probe fails the test
            .mount(&server)
            .await;

        let provider = RigProvider::new_ollama("llama3", &server.uri(), "").unwrap();
        assert!(provider.reachable().await);
        drop(server);
    }

    #[tokio::test]
    async fn rig_reachable_false_on_connection_refused() {
        let provider = RigProvider::new_ollama("llama3", DEAD_URL, "").unwrap();
        assert!(!provider.reachable().await);
    }

    #[tokio::test]
    async fn rig_is_ollama_capability() {
        let provider = RigProvider::new_ollama("llama3", "http://localhost:11434", "").unwrap();
        assert!(provider.is_ollama());
        assert!(provider.is_local());
    }

    /// A cloud base URL that never resolves — constructors must not touch the network, so this
    /// still builds Ok and reachability is decided without a probe.
    const CLOUD_URL: &str = "https://api.example.invalid";

    /// Every constructor builds offline and maps its id onto the expected concrete variant —
    /// including the shared-variant ids: openai-compatible reuses `OpenAi`, ollama-cloud reuses
    /// `Ollama`, and glm reuses `Zai` (glm is routed to `new_zai` at the factory stage).
    #[test]
    fn rig_constructors_build_offline_and_map_to_expected_variant() {
        let cases: &[(&str, Result<RigProvider, LensError>)] = &[
            ("openai", RigProvider::new_openai("gpt", CLOUD_URL, "k")),
            (
                "openai",
                RigProvider::new_openai_compatible("gpt", CLOUD_URL, "k"),
            ),
            (
                "anthropic",
                RigProvider::new_anthropic("claude", CLOUD_URL, "k"),
            ),
            ("gemini", RigProvider::new_gemini("gemini", CLOUD_URL, "k")),
            ("cohere", RigProvider::new_cohere("command", CLOUD_URL, "k")),
            ("xai", RigProvider::new_xai("grok", CLOUD_URL, "k")),
            ("groq", RigProvider::new_groq("llama", CLOUD_URL, "k")),
            (
                "deepseek",
                RigProvider::new_deepseek("deepseek", CLOUD_URL, "k"),
            ),
            ("zai", RigProvider::new_zai("glm-4", CLOUD_URL, "k")),
            ("ollama", RigProvider::new_ollama("llama3", CLOUD_URL, "")),
            (
                "ollama",
                RigProvider::new_ollama_cloud("llama3", CLOUD_URL, "k"),
            ),
        ];
        for (expected_variant, built) in cases {
            let provider = built
                .as_ref()
                .unwrap_or_else(|e| panic!("{expected_variant} constructor failed: {e:?}"));
            assert_eq!(&provider.variant_name(), expected_variant);
        }
    }

    #[test]
    fn rig_model_id_echoes_the_model_string() {
        assert_eq!(
            RigProvider::new_openai("gpt-4o-mini", CLOUD_URL, "k")
                .unwrap()
                .model_id(),
            "gpt-4o-mini"
        );
        assert_eq!(
            RigProvider::new_zai("glm-4.6", CLOUD_URL, "k")
                .unwrap()
                .model_id(),
            "glm-4.6"
        );
    }

    #[test]
    fn rig_is_local_is_true_only_for_local_ollama() {
        assert!(
            RigProvider::new_ollama("llama3", CLOUD_URL, "")
                .unwrap()
                .is_local()
        );
        for built in [
            RigProvider::new_ollama_cloud("llama3", CLOUD_URL, "k"),
            RigProvider::new_openai("gpt", CLOUD_URL, "k"),
            RigProvider::new_anthropic("claude", CLOUD_URL, "k"),
            RigProvider::new_groq("llama", CLOUD_URL, "k"),
            RigProvider::new_zai("glm-4", CLOUD_URL, "k"),
        ] {
            assert!(
                !built.unwrap().is_local(),
                "cloud backend must not be local"
            );
        }
    }

    #[test]
    fn rig_is_ollama_is_true_only_for_local_ollama() {
        assert!(
            RigProvider::new_ollama("llama3", CLOUD_URL, "")
                .unwrap()
                .is_ollama()
        );
        // ollama-cloud reports false too: the local-runtime preflight must not fire for it.
        for built in [
            RigProvider::new_ollama_cloud("llama3", CLOUD_URL, "k"),
            RigProvider::new_openai("gpt", CLOUD_URL, "k"),
            RigProvider::new_anthropic("claude", CLOUD_URL, "k"),
            RigProvider::new_gemini("gemini", CLOUD_URL, "k"),
            RigProvider::new_zai("glm-4", CLOUD_URL, "k"),
        ] {
            assert!(!built.unwrap().is_ollama());
        }
    }

    #[tokio::test]
    async fn rig_openai_compatible_requires_a_base_url() {
        assert!(matches!(
            RigProvider::new_openai_compatible("gpt", "", "k"),
            Err(LensError::Validation(_))
        ));
    }

    /// Cloud reachability is `true` without any network probe — keyed OR keyless — mirroring
    /// genai (a bad key surfaces from `generate`, not here). Uses an unroutable host to prove no
    /// request is made; a billed generate is separately locked out by the reachable-Ollama test.
    #[tokio::test]
    async fn rig_cloud_reachable_is_true_without_network() {
        assert!(
            RigProvider::new_openai("gpt", CLOUD_URL, "k")
                .unwrap()
                .reachable()
                .await
        );
        assert!(
            RigProvider::new_openai("gpt", CLOUD_URL, "")
                .unwrap()
                .reachable()
                .await,
            "keyless cloud is still reachable (mirrors genai)"
        );
        assert!(
            RigProvider::new_ollama_cloud("llama3", CLOUD_URL, "k")
                .unwrap()
                .reachable()
                .await,
            "ollama-cloud uses the no-network signal, not the local liveness probe"
        );
    }

    /// Regression: a keyless local OpenAI-compatible server (LM Studio, llama.cpp, vLLM without
    /// `--api-key`) must report reachable — the old keyed signal wrongly blocked it.
    #[tokio::test]
    async fn rig_keyless_openai_compatible_is_reachable() {
        assert!(
            RigProvider::new_openai_compatible("m", "http://127.0.0.1:1/v1", "")
                .unwrap()
                .reachable()
                .await
        );
    }

    /// Guards against `adapter_for`/`from_id` id-set drift: every recognized id must construct,
    /// and an unknown id must error.
    #[tokio::test]
    async fn rig_from_id_covers_every_adapter_for_id() {
        for id in [
            "ollama",
            "ollama-cloud",
            "openai",
            "openai-compatible",
            "anthropic",
            "google",
            "groq",
            "deepseek",
            "xai",
            "cohere",
            "zai",
            "glm",
        ] {
            assert!(
                RigProvider::from_id(id, "m", "http://127.0.0.1:1/v1", "k", llm_client()).is_ok(),
                "from_id must construct {id}"
            );
        }
        assert!(
            RigProvider::from_id("bogus", "m", "http://127.0.0.1:1/v1", "k", llm_client()).is_err()
        );
    }

    /// Characterizes rig's verbatim endpoint handling (see [`super::construct_provider`]): rig
    /// posts `<base_url>/chat/completions` and injects no extra `/v1`, unlike genai's force-append.
    #[tokio::test]
    async fn rig_openai_compatible_posts_base_verbatim() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_chat_body("ok")))
            .mount(&server)
            .await;

        let provider =
            RigProvider::new_openai_compatible("gpt", &format!("{}/v1", server.uri()), "k")
                .unwrap();
        let _ = provider.generate(&req()).await;

        let requests = server.received_requests().await.expect("recording on");
        assert_eq!(
            requests[0].url.path(),
            "/v1/chat/completions",
            "rig must append /chat/completions to the base verbatim, no extra /v1"
        );
    }

    /// Locks the verbatim openai-wire contract (see [`super::construct_provider`]): a custom base
    /// with no version segment posts to `/chat/completions`, not `/v1/chat/completions`.
    #[tokio::test]
    async fn rig_openai_custom_base_has_no_implicit_v1() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_chat_body("ok")))
            .mount(&server)
            .await;

        let provider = RigProvider::new_openai("gpt", &server.uri(), "k").unwrap();
        let _ = provider.generate(&req()).await;

        let requests = server.received_requests().await.expect("recording on");
        assert_eq!(
            requests[0].url.path(),
            "/chat/completions",
            "a custom openai base must be verbatim — no implicit /v1"
        );
    }

    /// Anthropic injects its own `/v1/messages` (see [`super::construct_provider`]) — so a custom
    /// base keeps `/v1`, matching genai with no parity break, unlike the verbatim openai case above.
    #[tokio::test]
    async fn rig_anthropic_custom_base_keeps_v1_matching_genai() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .mount(&server)
            .await;

        let provider = RigProvider::new_anthropic("claude", &server.uri(), "k").unwrap();
        let _ = provider.generate(&req()).await;

        let requests = server.received_requests().await.expect("recording on");
        assert_eq!(
            requests[0].url.path(),
            "/v1/messages",
            "rig injects /v1 for anthropic — parity with genai, no break"
        );
    }
}
