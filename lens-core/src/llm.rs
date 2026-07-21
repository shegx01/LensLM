//! LLM provider seam for the M4 Phase-3 enrichment pass.
//!
//! Defines [`LlmProvider`] (object-safe, `Arc<dyn LlmProvider>`) backed by [`GenaiProvider`]
//! over the `genai` crate (jeremychone/rust-genai 0.6.x), a typed routing policy ([`LlmRouting`]),
//! and the [`provider_from_config`] factory. genai is constructed with our hardened reqwest client
//! so SSRF policy and timeouts carry over; enrichment pins `temperature: 0.0 + json: true` for
//! deterministic output. The default [`LlmProvider::generate_stream`] lets enrichment mocks
//! (which only implement the three core methods) compile untouched.

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

    /// Upcast for downcasting to a concrete type. [`task_provider_from_config`] downcasts to
    /// [`GenaiProvider`] to borrow its client. The default returns a `()` reference that never
    /// downcasts, so mocks that don't override it cause the per-task path to fall back to base.
    fn as_any(&self) -> &dyn std::any::Any {
        &()
    }

    /// Whether this provider runs on-device (local Ollama). Lets callers relax limits
    /// small local models can't meet — e.g. the dialogue min-turns floor (#26) — while
    /// keeping cloud strict.
    fn is_local(&self) -> bool {
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

/// Maps a genai error onto [`LensError`], sanitizing the message before it crosses the IPC
/// boundary. genai wraps transport errors inside its own types with no public `reqwest::Error`
/// accessor, so we classify by `Display` text (connect/timeout → `Network`; everything else
/// → `Model`). The full error is logged server-side; only a generic message is surfaced over IPC.
fn genai_err(err: genai::Error) -> LensError {
    let lower = err.to_string().to_ascii_lowercase();
    // reqwest's transport-failure `Display` often lacks "timeout"/"connect": a send
    // failure reads "error sending request", an idle read-timeout "error reading
    // response"/"body", and a deadline "deadline". Match those too so a genuine
    // transport error is never misclassified as a model (bad-output) error.
    let is_transport = lower.contains("connect")
        || lower.contains("connection")
        || lower.contains("timed out")
        || lower.contains("timeout")
        || lower.contains("dns")
        || lower.contains("refused")
        || lower.contains("sending request")
        || lower.contains("reading response")
        || lower.contains("response body")
        || lower.contains("deadline");
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
}

/// Normalizes a `base_url` into the endpoint base genai expects. genai concatenates a relative
/// path onto this base, so it must end in `/`. OpenAI/Anthropic adapters also need `/v1/`
/// (they append `chat/completions` / `messages` after the version segment); Ollama only needs
/// a trailing slash.
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
    /// Builds a provider with its own genai client. Per-task providers use
    /// [`new_with_client`](Self::new_with_client) to share one client across coref/map.
    fn new(adapter: AdapterKind, model: &str, base_url: &str, api_key: &str) -> Self {
        let client = Client::builder().with_reqwest(llm_client()).build();
        Self::new_with_client(client, adapter, model, base_url, api_key)
    }

    /// Builds a provider reusing an existing genai client (only the pinned target differs).
    fn new_with_client(
        client: Client,
        adapter: AdapterKind,
        model: &str,
        base_url: &str,
        api_key: &str,
    ) -> Self {
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
        }
    }

    /// Returns a cheap `Arc`-backed clone of the genai client for sibling per-task providers.
    fn client_handle(&self) -> Client {
        self.client.clone()
    }

    /// Whether the resolved adapter is local Ollama, without leaking [`AdapterKind`] into the
    /// public API. Used by the enrichment-model preflight (issue #90).
    pub fn is_ollama(&self) -> bool {
        matches!(self.resolved.adapter, AdapterKind::Ollama)
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn is_local(&self) -> bool {
        self.is_ollama()
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

/// Builds a sibling [`GenaiProvider`] pinned to `task_model`, reusing `base`'s client.
/// Returns `None` when no matching config entry exists, the provider is ungated, or
/// `base` is not a [`GenaiProvider`] (e.g. a test mock).
fn build_task_provider(
    base: &Arc<dyn LlmProvider>,
    task_model: &crate::config::TaskModel,
    models: &[crate::config::ModelConfig],
    cloud_consent: bool,
) -> Option<Arc<dyn LlmProvider>> {
    let want_provider = task_model.provider.to_ascii_lowercase();
    let adapter = adapter_for(&want_provider)?;

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

    let base_genai = base.as_any().downcast_ref::<GenaiProvider>()?;
    let client = base_genai.client_handle();
    Some(Arc::new(GenaiProvider::new_with_client(
        client,
        adapter,
        &task_model.model,
        &entry.base_url,
        &entry.api_key,
    )))
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
        LlmRouting::CloudFirst => models
            .iter()
            .find(|m| !is_local_provider(&m.provider.to_ascii_lowercase()) && usable(m))
            .or_else(|| {
                models
                    .iter()
                    .find(|m| is_local_provider(&m.provider.to_ascii_lowercase()) && usable(m))
            })
            .and_then(build_provider),
        LlmRouting::LocalFirst => models
            .iter()
            .find(|m| is_local_provider(&m.provider.to_ascii_lowercase()) && usable(m))
            .or_else(|| {
                models
                    .iter()
                    .find(|m| !is_local_provider(&m.provider.to_ascii_lowercase()) && usable(m))
            })
            .and_then(build_provider),
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

/// Whether an entry has a usable endpoint: a configured `base_url`, or a native cloud adapter
/// with a canonical endpoint. Local Ollama and `openai-compatible` always require a `base_url`.
fn has_endpoint(model: &crate::config::ModelConfig) -> bool {
    if !model.base_url.is_empty() {
        return true;
    }
    adapter_for(&model.provider.to_ascii_lowercase()).is_some_and(|a| native_endpoint(a).is_some())
}

/// Builds a [`GenaiProvider`] for a recognized entry (caller applies [`build_eligible`] first).
/// Returns `None` for an unrecognized provider, empty model, or missing endpoint.
fn build_provider(model: &crate::config::ModelConfig) -> Option<Arc<dyn LlmProvider>> {
    if model.model.is_empty() {
        return None;
    }
    let provider = model.provider.to_ascii_lowercase();
    let adapter = adapter_for(&provider)?;
    // Local Ollama / openai-compatible have no canonical default — require a configured base_url.
    if model.base_url.is_empty() && native_endpoint(adapter).is_none() {
        return None;
    }
    Some(Arc::new(GenaiProvider::new(
        adapter,
        &model.model,
        &model.base_url,
        &model.api_key,
    )))
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
    let adapter = adapter_for(&provider.to_ascii_lowercase())?;
    if base_url.is_empty() && native_endpoint(adapter).is_none() {
        return None;
    }
    Some(Arc::new(GenaiProvider::new(
        adapter, model, base_url, api_key,
    )))
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
}
