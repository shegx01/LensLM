//! LLM provider seam for the M4 Phase-3 enrichment pass.
//!
//! Defines [`LlmProvider`] — an `async`, object-safe trait (held behind
//! `Arc<dyn LlmProvider>`) alongside [`crate::Embedder`] / [`crate::VectorStore`]
//! — backed by a single [`GenaiProvider`] over the [`genai`] crate, plus a typed
//! routing policy ([`LlmRouting`]) and the [`provider_from_config`] factory.
//!
//! ## Why genai
//! Stage 1 of the LLM-interface overhaul landed the typed [`ModelCatalog`]
//! anti-free-string guard. Stage 2 (this module) replaces the three hand-rolled
//! HTTP backends (Ollama / OpenAI-compatible / Anthropic) with a SINGLE
//! [`GenaiProvider`] that delegates protocol details to `genai`
//! (jeremychone/rust-genai, 0.6.x). The [`LlmProvider`] trait is UNCHANGED in
//! shape (`model_id` / `reachable` / `generate`) so the enrichment worker, coref,
//! and structural-map passes — and every mock that implements the trait — compile
//! untouched. A defaulted [`LlmProvider::generate_stream`] method is added on top
//! (the foundation M5 chat will use) so the existing 3-method mocks keep working.
//!
//! ## Hardening
//! genai is constructed with OUR reqwest client (`Client::builder().with_reqwest`)
//! so the SSRF no-redirect policy + bounded connect/read timeouts from the
//! system-check probe carry over verbatim — genai never opens an unhardened
//! socket. genai 0.6.x depends on `reqwest ^0.13`, the same major lens-core pins,
//! so the client type is compatible and the dependency dedupes.
//!
//! ## Determinism contract (PRESERVED)
//! Enrichment pins `temperature: 0.0` and `json: true` (and `thinking: false`);
//! the mapping below threads those into genai `ChatOptions` via
//! `.with_temperature(0.0)` plus `ChatResponseFormat::JsonMode`, producing strict
//! JSON exactly as the hand-rolled backends did. Thinking/streaming are OFF for
//! enrichment.

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
use crate::model_catalog::{ModelCatalog, SupportedProvider};

/// Connect timeout for an LLM HTTP request (matches the system-check probe).
const LLM_CONNECT_TIMEOUT: Duration = Duration::from_secs(1);
/// Overall (read) timeout for an LLM generate/probe request.
///
/// Mirrors `ENRICHMENT_LLM_TIMEOUT_SECS` from the plan (Decision C); a generate
/// call can legitimately take many seconds, so this is far longer than the
/// 2s system-check probe window.
const LLM_TIMEOUT: Duration = Duration::from_secs(30);

/// Canonical provider identifiers (match `ModelConfig.provider` /
/// the onboarding `LlmProviderInput.provider` strings). First-class cloud
/// providers carry their REAL models.dev catalog key (`anthropic`, `google`,
/// `openai`, `zai`); `openai-compatible` is reserved for a genuinely
/// custom/self-hosted OpenAI-protocol endpoint (LM Studio, a proxy, …) where the
/// user supplies the base URL and the served models are arbitrary.
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

/// The thinking/reasoning effort exposed on [`LlmRequest`].
///
/// A small, serde-stable mirror of genai's `ReasoningEffort` so the trait API and
/// the on-disk/IPC request shape never leak a genai type. Enrichment never sets
/// this (it stays `temperature 0 + json`, thinking OFF); it is the foundation the
/// M5 chat surface will use. Maps 1:1 onto genai via [`ReasoningEffort::to_genai`].
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
    /// Maps onto the genai effort level.
    fn to_genai(self) -> GenaiEffort {
        match self {
            ReasoningEffort::Low => GenaiEffort::Low,
            ReasoningEffort::Medium => GenaiEffort::Medium,
            ReasoningEffort::High => GenaiEffort::High,
        }
    }
}

/// A single completion request to an [`LlmProvider`].
///
/// `system` is an optional system/instruction prompt; `prompt` is the user
/// turn; `max_tokens` caps the generated output. `temperature` and `json` are
/// the determinism knobs threaded into every backend so enrichment can pin
/// reproducible, machine-parseable output (the enrichment callers pass
/// `temperature: 0.0, json: true, thinking: false`).
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
    /// Request strict JSON output. When `true`, the mapping asks genai for JSON
    /// mode (`ChatResponseFormat::JsonMode`) where the provider supports it; the
    /// strict serde parse + reprompts in `enrichment::map` remain the real guard.
    pub json: bool,
    /// Enable provider "thinking"/reasoning. Defaults to `false` (an older IPC
    /// payload with no `thinking` key reads back as `false` via `#[serde(default)]`).
    /// Enrichment keeps this OFF for determinism; M5 chat opts in. When `true`,
    /// `reasoning_effort` selects the budget (defaulting to `Medium` when unset).
    #[serde(default)]
    pub thinking: bool,
    /// Reasoning budget when `thinking` is enabled. Ignored when `thinking` is
    /// `false`. Defaults to `None` (an older payload reads back as `None`).
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
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

/// One event from a streamed generation ([`LlmProvider::generate_stream`]).
///
/// The minimal surface M5 chat consumes: incremental answer text, incremental
/// thinking/reasoning text (when enabled), and a terminal `Done` carrying the
/// usage once the provider reports it. genai's richer stream
/// (`Start`/`ToolCallChunk`/…) is collapsed onto these three so the trait stays
/// provider-agnostic and the enrichment path never sees a genai type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamChunk {
    /// An incremental piece of the answer text.
    TextDelta(String),
    /// An incremental piece of the model's thinking/reasoning text (emitted only
    /// when `thinking` is requested and the provider streams reasoning).
    ThinkingDelta(String),
    /// Terminal event: the stream is complete. `tokens_used` is the captured
    /// total when the provider reported usage, else `0`.
    Done {
        /// Total tokens consumed by the call, or `0` when the provider reported none.
        tokens_used: u32,
    },
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

    /// Upcast to [`std::any::Any`] for downcasting to a concrete provider type.
    ///
    /// The per-task enrichment wiring ([`task_provider_from_config`]) downcasts to
    /// [`GenaiProvider`] to borrow its shared genai client when building a sibling
    /// provider pinned to a per-task model. The DEFAULT returns a reference that
    /// never downcasts to a real backend, so a test mock that doesn't override it
    /// simply makes the per-task override fall back to the base provider.
    fn as_any(&self) -> &dyn std::any::Any {
        &()
    }

    /// `system_check`-style reachability probe. `true` when the endpoint is
    /// usable for generation; `false` on a connection refusal, DNS/timeout
    /// failure, OR an auth error (`401`/`403`) — a misconfigured key counts as
    /// unreachable so the source degrades to raw vectors instead of looping.
    async fn reachable(&self) -> bool;

    /// Generate a completion. Returns [`LensError::Network`] on transport
    /// failure, [`LensError::Model`] on a non-success / provider error, and
    /// [`LensError::Parse`] when the response can't be decoded.
    async fn generate(&self, req: &LlmRequest) -> Result<LlmResponse, LensError>;

    /// Stream a completion as a sequence of [`StreamChunk`]s, ending in
    /// [`StreamChunk::Done`].
    ///
    /// This is the foundation the M5 chat surface uses; ENRICHMENT NEVER STREAMS
    /// (it stays on the non-streaming [`generate`](Self::generate) for the
    /// deterministic temp-0 + JSON contract). A DEFAULT implementation is provided
    /// so the enrichment mocks (which implement only the three core methods) need
    /// no changes: it buffers a non-streaming [`generate`](Self::generate) into a
    /// single `TextDelta` + `Done`. Real backends ([`GenaiProvider`]) override it
    /// with true incremental streaming.
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

/// Builds the shared HTTP client for LLM calls via the one hardened builder
/// ([`crate::http::hardened_client`]): bounded connect/read timeouts + the same
/// SSRF no-redirect hardening as the system-check probe. This client is injected
/// into genai via `Client::builder().with_reqwest(..)`, so genai inherits the
/// hardening rather than opening its own raw socket.
fn llm_client() -> reqwest::Client {
    crate::http::hardened_client(LLM_CONNECT_TIMEOUT, LLM_TIMEOUT)
}

/// Maps a genai error onto a [`LensError`], SANITIZING the message before it can
/// cross the Tauri IPC boundary.
///
/// genai wraps the underlying reqwest/transport error inside its own
/// `webc::Error` (the `WebAdapterCall`/`WebModelCall`/`WebStream` variants), so
/// there is no public `reqwest::Error` to inspect for `is_connect()`/`is_timeout()`.
/// We classify by the error's `Display` text: a connection/timeout marker →
/// [`LensError::Network`]; everything else (a non-success status `HttpError`, an
/// auth/provider error, a decode miss) → [`LensError::Model`].
///
/// The raw genai `Display` text can carry the configured endpoint URL or (in
/// future request shapes) an echoed API key, so — mirroring the `sqlx`-error
/// sanitization in [`crate::error`] — the FULL error is logged server-side via
/// `tracing` and only a GENERIC, fixed message is surfaced over IPC. The
/// Network/Model classification (and thus enrichment control flow, which treats
/// both identically) is UNCHANGED.
fn genai_err(err: genai::Error) -> LensError {
    let lower = err.to_string().to_ascii_lowercase();
    let is_transport = lower.contains("connect")
        || lower.contains("connection")
        || lower.contains("timed out")
        || lower.contains("timeout")
        || lower.contains("dns")
        || lower.contains("refused");
    // Log the full detail for operators; never surface it across IPC.
    tracing::error!(error = %err, transport = is_transport, "LLM request failed");
    if is_transport {
        LensError::Network("LLM request failed (network)".to_string())
    } else {
        LensError::Model("LLM request failed (model)".to_string())
    }
}

/// A resolved generation target: the genai [`ServiceTarget`] (adapter + model +
/// endpoint + auth) plus a stable, cloned `model_id` string for the trait
/// accessor / enrichment cache key.
#[derive(Clone)]
struct ResolvedTarget {
    target: ServiceTarget,
    model_id: String,
}

/// The single LLM backend over [`genai`].
///
/// Holds a genai [`Client`] (built with our hardened reqwest client) and a fully
/// resolved [`ServiceTarget`] (adapter + model + endpoint + auth). Every call
/// pins that exact target via `ModelSpec::Target`, so the provider/model is never
/// re-inferred from the model name — a custom endpoint (local Ollama, LM Studio,
/// GLM, Ollama Cloud) and an explicit API key are honored exactly.
pub struct GenaiProvider {
    client: Client,
    resolved: ResolvedTarget,
}

/// Normalizes a configured `base_url` into the endpoint base genai's adapters
/// expect.
///
/// genai builds the request URL by CONCATENATING a relative path onto the
/// endpoint base, so the base MUST end in a trailing slash (`format!("{base}path")`)
/// or the join is malformed. Beyond that, the OpenAI and Anthropic adapters append
/// the path AFTER the `/v1/` version segment (`{base}chat/completions`,
/// `{base}messages`), so their base must end in `/v1/`. Our config stores the bare
/// host for those providers (the system-check probe appends `/v1/models` itself),
/// so we add `v1/` when it is absent. The Ollama adapter appends `api/chat`
/// directly, so it only needs a trailing slash.
fn normalize_endpoint(adapter: AdapterKind, base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    let needs_v1 = matches!(adapter, AdapterKind::OpenAI | AdapterKind::Anthropic);
    // Only inject `/v1` when the configured base doesn't already carry a version
    // segment (some configs store `…/v1`), so we never double it up.
    if needs_v1 && !trimmed.ends_with("/v1") {
        format!("{trimmed}/v1/")
    } else {
        format!("{trimmed}/")
    }
}

/// The genai-default public endpoint for a NATIVE cloud adapter.
///
/// genai bakes a fixed `BASE_URL` into each native adapter (e.g. Groq's
/// `https://api.groq.com/openai/v1/`) but exposes it only through the crate-private
/// `AdapterDispatcher::default_endpoint` — there is no public accessor. We
/// therefore mirror those defaults here so a native cloud provider configured with
/// NO `base_url` resolves to its canonical endpoint (genai uses our `ServiceTarget`
/// VERBATIM via `ModelSpec::Target`, so we must supply a concrete endpoint).
///
/// Returns `None` for adapters where the URL is user-supplied (the custom
/// `openai-compatible` endpoint) or local (`Ollama` — the user's `base_url` is the
/// runtime). When `Some`, a configured non-empty `base_url` still WINS (an explicit
/// override), so this only fills the gap when the config omits a base URL.
///
/// **Pinned to genai 0.6.5.** On a genai bump, re-verify against
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
        // Local Ollama + the custom openai-compatible endpoint: the URL is supplied
        // by the user's config, so there is no canonical default to fall back to.
        _ => None,
    }
}

impl GenaiProvider {
    /// Builds a provider from a fully-resolved adapter/model/endpoint/auth target,
    /// constructing its OWN genai [`Client`] (over our hardened reqwest client).
    ///
    /// Most call sites use this; the per-task enrichment wiring instead uses
    /// [`new_with_client`](Self::new_with_client) to SHARE one client across the
    /// coref/map providers (so the worker never spins up multiple genai clients —
    /// only the pinned target differs).
    fn new(adapter: AdapterKind, model: &str, base_url: &str, api_key: &str) -> Self {
        let client = Client::builder().with_reqwest(llm_client()).build();
        Self::new_with_client(client, adapter, model, base_url, api_key)
    }

    /// Builds a provider that REUSES an existing genai [`Client`], pinned to its own
    /// resolved `(adapter, model, endpoint, auth)` target. The per-task enrichment
    /// path builds the coref/map providers this way from ONE shared client so they
    /// differ only in the pinned target (the plan's "reuse ONE genai Client" rule).
    fn new_with_client(
        client: Client,
        adapter: AdapterKind,
        model: &str,
        base_url: &str,
        api_key: &str,
    ) -> Self {
        let model_iden = ModelIden::new(adapter, model.to_string());
        // A configured (non-empty) base_url WINS — it pins a custom/self-hosted
        // backend (the openai-compatible case, a proxy, a local runtime) or an
        // explicit override of a native provider. With NO base_url, a native cloud
        // adapter falls back to its canonical genai endpoint; if even that is
        // absent (a custom/local adapter with no URL), normalize an empty base
        // (yields the old behavior) so construction stays infallible.
        let endpoint = if base_url.is_empty() {
            native_endpoint(adapter)
                .unwrap_or_else(|| Endpoint::from_owned(normalize_endpoint(adapter, base_url)))
        } else {
            Endpoint::from_owned(normalize_endpoint(adapter, base_url))
        };
        let auth = if api_key.is_empty() {
            // Local runtimes (Ollama / LM Studio) need no key; genai tolerates an
            // empty single key for those adapters.
            AuthData::from_single(String::new())
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
            },
        }
    }

    /// Clones the underlying genai [`Client`] so a sibling per-task provider can be
    /// built against the SAME client with a different pinned target. genai's
    /// [`Client`] is a cheap `Arc`-backed handle, so this shares the one HTTP client
    /// rather than constructing a new one.
    fn client_handle(&self) -> Client {
        self.client.clone()
    }

    /// Maps an [`LlmRequest`] onto a genai `(ChatRequest, ChatOptions)` pair.
    ///
    /// PRESERVES the determinism contract: `temperature` is passed verbatim
    /// (enrichment pins `0.0`), `json: true` → `ChatResponseFormat::JsonMode`,
    /// usage capture is always on so `tokens_used` is populated. `thinking`
    /// (default OFF) maps to `with_reasoning_effort`; enrichment never sets it.
    fn map_request(req: &LlmRequest) -> (ChatRequest, ChatOptions) {
        let mut chat = ChatRequest::default();
        if let Some(system) = &req.system {
            chat = chat.with_system(system.clone());
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

    /// The pinned per-call model spec (the fully-resolved target).
    fn model_spec(&self) -> ModelSpec {
        ModelSpec::Target(self.resolved.target.clone())
    }
}

/// genai usage is `Option<i32>` per field; collapse `prompt + completion` (falling
/// back to `total`) into our saturating `u32`, treating absent/negative as `0`.
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

    async fn reachable(&self) -> bool {
        // A minimal generate ping (max_tokens=1, temp 0, no JSON/thinking). Any
        // success → reachable; a `401`/`403` or a connection/timeout failure →
        // unreachable (Decision G: a misconfigured key counts as unreachable so
        // the source degrades to raw vectors instead of looping).
        let ping = LlmRequest {
            system: None,
            prompt: "ping".to_string(),
            max_tokens: 1,
            temperature: 0.0,
            json: false,
            thinking: false,
            reasoning_effort: None,
        };
        let (chat, opts) = Self::map_request(&ping);
        self.client
            .exec_chat(self.model_spec(), chat, Some(&opts))
            .await
            .is_ok()
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

        // Map genai's stream onto our minimal `StreamChunk` surface: answer text →
        // `TextDelta`, reasoning text → `ThinkingDelta`, terminal `End` → `Done`
        // with the captured usage. `Start`/`ToolCallChunk`/other events are
        // dropped (not part of the chat-text contract). A genai stream error maps
        // through `genai_err`.
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
                // Start / ToolCallChunk / ThoughtSignatureChunk: not part of the
                // text contract — skip.
                Ok(_) => None,
                Err(e) => Some(Err(genai_err(e))),
            }
        });
        Ok(Box::pin(mapped))
    }
}

// ---------------------------------------------------------------------------
// Routing / override policy (Stage 2)
// ---------------------------------------------------------------------------

/// Typed routing policy for selecting the enrichment LLM from the config.
///
/// Replaces the hand-rolled "first reachable, local-first" selection with an
/// explicit policy. The product direction flips the DEFAULT toward cloud-when-
/// available: [`LlmRouting::CloudFirst`] prefers a configured + consented cloud
/// provider, falling back to a local Ollama entry. [`LlmRouting::LocalFirst`]
/// keeps the old local-first behavior; [`LlmRouting::Explicit`] pins one exact
/// `(provider, model)`.
///
/// Serde-stable (snake_case, internally tagged on `kind`) so it round-trips in
/// `config.json` and a TS mirror without leaking a Rust enum shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LlmRouting {
    /// Prefer a configured cloud provider (when present AND consented); otherwise
    /// fall back to a local Ollama entry. The default per product direction.
    CloudFirst,
    /// Prefer a local Ollama entry; otherwise fall back to a consented cloud
    /// provider (the pre-Stage-2 behavior).
    LocalFirst,
    /// Pin one exact provider+model. Validated against the catalog (cloud) before
    /// dispatch; a mismatch with the configured entries skips selection.
    Explicit {
        /// The canonical provider id (`"anthropic"`, `"openai-compatible"`, …).
        provider: String,
        /// The model id to pin.
        model: String,
    },
}

impl Default for LlmRouting {
    /// The product-direction default: prefer cloud-when-available, else local.
    fn default() -> Self {
        LlmRouting::CloudFirst
    }
}

/// Whether a canonical provider id denotes a local runtime (exempt from the
/// cloud-consent gate AND from catalog validation — local models are user-pulled).
///
/// Delegates to the SINGLE locality predicate ([`SupportedProvider::is_local`]) so
/// the consent-gate exemption here can never drift from the catalog-validation
/// exemption in `model_catalog` (they guard the same privacy/consent bypass).
fn is_local_provider(provider: &str) -> bool {
    SupportedProvider::is_local(provider)
}

/// Maps a canonical `ModelConfig.provider` id onto the genai [`AdapterKind`].
///
/// First-class cloud providers map to their NATIVE genai adapter (genai 0.6.5
/// ships `Anthropic`, `Gemini`, `OpenAI`, `Zai`, `Groq`, `DeepSeek`, `Xai`,
/// `Cohere`, `Ollama`, `OllamaCloud`, so no provider falls back to a generic
/// OpenAI-compatible adapter). `glm` is an alias
/// for `zai` (the GLM models are Z.ai's). The custom `openai-compatible` endpoint
/// speaks the OpenAI wire protocol, so it maps to [`AdapterKind::OpenAI`] with the
/// user-supplied base URL pinning the actual backend. Returns `None` for an
/// unrecognized provider so the entry is skipped.
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
        // A genuinely custom/self-hosted OpenAI-protocol endpoint (LM Studio, a
        // proxy): OpenAI adapter + the user's base URL.
        PROVIDER_OPENAI_COMPAT => Some(AdapterKind::OpenAI),
        _ => None,
    }
}

/// The models.dev catalog key used to validate a first-class cloud
/// `(provider, model)`. Each real provider id validates against its OWN namespace
/// (anthropic→anthropic, google→google, openai→openai, zai→zai, …), so a
/// `claude-*` model validates against `anthropic` and a `gemini-*` against
/// `google`. `glm` validates against `zai`.
///
/// Returns `None` for the custom `openai-compatible` endpoint case (and any local
/// runtime), signalling the caller to SKIP catalog validation entirely — that
/// endpoint serves arbitrary models with no catalog namespace.
fn catalog_key_for(provider: &str) -> Option<&str> {
    match provider {
        PROVIDER_OPENAI => Some(SupportedProvider::OpenAI.catalog_key()),
        PROVIDER_ANTHROPIC => Some(SupportedProvider::Anthropic.catalog_key()),
        PROVIDER_GOOGLE => Some(SupportedProvider::Google.catalog_key()),
        PROVIDER_ZAI | PROVIDER_GLM => Some(SupportedProvider::Zai.catalog_key()),
        PROVIDER_OLLAMA_CLOUD => Some(SupportedProvider::OllamaCloud.catalog_key()),
        // Custom self-hosted endpoint (or a local runtime): no catalog namespace.
        PROVIDER_OPENAI_COMPAT | PROVIDER_OLLAMA => None,
        // groq / deepseek / xai / cohere (and any other first-class provider id):
        // validate against the SAME-named models.dev namespace (anti-free-string).
        other => Some(other),
    }
}

/// Builds the enrichment [`LlmProvider`] from `config.models[]` under the typed
/// [`LlmRouting`] policy + cloud-consent gate + catalog validation.
///
/// Selection:
/// * [`LlmRouting::CloudFirst`] (default): the first consented cloud entry that
///   validates against the catalog, else the first local Ollama entry.
/// * [`LlmRouting::LocalFirst`]: the first local Ollama entry, else the first
///   consented + catalog-valid cloud entry.
/// * [`LlmRouting::Explicit`]: the configured entry matching `(provider, model)`,
///   subject to the same consent + validation gates.
///
/// Cloud providers (everything but local Ollama) require `cloud_consent == true`
/// AND must pass [`ModelCatalog::validate`]; local Ollama is exempt from both
/// (user-pulled models aren't in models.dev). Returns `None` when no usable entry
/// exists.
///
/// This does NOT probe reachability — the caller (the worker / `init`) calls
/// [`LlmProvider::reachable`] on the returned provider to gate dispatch.
///
/// **Step-boundary note:** the cloud-consent flag is threaded in as a parameter
/// (rather than read off `config.enrichment`) to keep this signature stable for
/// the existing callers in `lib.rs`.
pub fn provider_from_config(
    config: &AppConfig,
    cloud_consent: bool,
) -> Option<Arc<dyn LlmProvider>> {
    let routing = config.enrichment.routing.clone();
    let catalog = ModelCatalog::bundled();
    select_provider(&config.models, &routing, cloud_consent, &catalog)
}

/// Resolves the per-task enrichment provider for ONE task (coref / map), reusing
/// the base provider's genai [`Client`] (M4 Phase 3, Stage 3).
///
/// When `task_model` is `Some` AND it resolves to a usable, gated entry, this
/// returns a sibling [`GenaiProvider`] pinned to that exact `(provider, model)` —
/// built over the SAME genai client as `base` (only the pinned target differs, per
/// the plan). When `task_model` is `None`, or the override fails its gates
/// (unknown provider, cloud without consent, uncatalogued cloud model), this
/// returns a clone of `base` so the task falls back to the routing default.
///
/// `base` carries the shared genai client; the override's `base_url`/`api_key` are
/// sourced from the matching `config.models[]` entry (a `TaskModel` names only
/// `provider`+`model`). Cloud overrides are consent-gated AND catalog-validated;
/// local Ollama is exempt — mirroring [`build_eligible`].
pub fn task_provider_from_config(
    base: &Arc<dyn LlmProvider>,
    task_model: Option<&crate::config::TaskModel>,
    models: &[crate::config::ModelConfig],
    cloud_consent: bool,
) -> Arc<dyn LlmProvider> {
    let catalog = ModelCatalog::bundled();
    match task_model.and_then(|tm| build_task_provider(base, tm, models, cloud_consent, &catalog)) {
        Some(p) => p,
        None => base.clone(),
    }
}

/// Builds a sibling [`GenaiProvider`] pinned to `task_model`, reusing the genai
/// client from `base` (downcast to a [`GenaiProvider`]). Returns `None` when the
/// override isn't usable: no matching config entry, an unknown/ungated provider,
/// or `base` isn't a [`GenaiProvider`] to lend its client (e.g. a test mock).
fn build_task_provider(
    base: &Arc<dyn LlmProvider>,
    task_model: &crate::config::TaskModel,
    models: &[crate::config::ModelConfig],
    cloud_consent: bool,
    catalog: &ModelCatalog,
) -> Option<Arc<dyn LlmProvider>> {
    let want_provider = task_model.provider.to_ascii_lowercase();
    let adapter = adapter_for(&want_provider)?;

    // Source endpoint/key from the matching configured entry (the TaskModel names
    // only provider+model). Prefer the entry matching BOTH provider AND the exact
    // override model (so two entries for the same provider — e.g. an instruct + a
    // coder Ollama endpoint — resolve to the right base_url/api_key); otherwise
    // fall back to the first entry for that provider (the endpoint/key are shared
    // across that provider's models). The entry's base_url/api_key pin the backend.
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

    // Apply the same consent + catalog gates as routing selection (anti-free-string):
    // local Ollama is exempt; the custom openai-compatible endpoint is consent-gated
    // but NOT catalog-validated; a first-class cloud provider needs consent + a
    // catalog hit on the OVERRIDE model in its OWN namespace.
    if is_local_provider(&want_provider) {
        if task_model.model.is_empty() {
            return None;
        }
    } else if !cloud_consent {
        return None;
    } else {
        let ok = match catalog_key_for(&want_provider) {
            Some(key) => catalog.validate(key, &task_model.model).is_ok(),
            None => !task_model.model.is_empty(),
        };
        if !ok {
            return None;
        }
    }

    // Reuse the base provider's genai client (only the pinned target differs).
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

/// Routing-aware selection over the configured model entries. Split out from
/// [`provider_from_config`] so it is unit-testable with an injected catalog.
fn select_provider(
    models: &[crate::config::ModelConfig],
    routing: &LlmRouting,
    cloud_consent: bool,
    catalog: &ModelCatalog,
) -> Option<Arc<dyn LlmProvider>> {
    let usable = |m: &crate::config::ModelConfig| {
        has_endpoint(m) && !m.model.is_empty() && build_eligible(m, cloud_consent, catalog)
    };

    match routing {
        LlmRouting::Explicit { provider, model } => {
            let want_provider = provider.to_ascii_lowercase();
            models
                .iter()
                .find(|m| {
                    m.provider.to_ascii_lowercase() == want_provider
                        && m.model == *model
                        && usable(m)
                })
                .and_then(build_provider)
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

/// Whether an entry passes the consent + catalog-validation gates (independent of
/// routing). Local Ollama is exempt from both. The custom `openai-compatible`
/// endpoint is consent-gated but EXEMPT from catalog validation (it serves
/// arbitrary models). Every other (first-class) cloud entry needs consent AND a
/// catalog hit in its OWN namespace. An unrecognized provider id is never eligible.
fn build_eligible(
    model: &crate::config::ModelConfig,
    cloud_consent: bool,
    catalog: &ModelCatalog,
) -> bool {
    let provider = model.provider.to_ascii_lowercase();
    if adapter_for(&provider).is_none() {
        return false;
    }
    if is_local_provider(&provider) {
        return true;
    }
    // Cloud: always consent-gated.
    if !cloud_consent {
        return false;
    }
    match catalog_key_for(&provider) {
        // First-class cloud provider: validate against its OWN catalog namespace
        // (anti-free-string), so e.g. a `claude-*` validates against `anthropic`.
        Some(key) => catalog.validate(key, &model.model).is_ok(),
        // Custom self-hosted OpenAI-compatible endpoint: consent-gated above, but
        // NOT catalog-validated (arbitrary models). A non-empty model is required
        // (enforced by the caller's `usable` check / `build_provider`).
        None => !model.model.is_empty(),
    }
}

/// Builds a [`GenaiProvider`] for a single recognized entry (no gating — the
/// caller applies [`build_eligible`] first). Returns `None` for an unrecognized
/// provider or an empty endpoint/model.
/// Whether an entry has a usable endpoint: a configured `base_url`, OR a native
/// cloud adapter whose canonical endpoint [`native_endpoint`] supplies. Native
/// cloud providers (`groq`/`deepseek`/`xai`/`cohere`/…) need no `base_url`; local
/// Ollama and the custom `openai-compatible` endpoint still require one.
fn has_endpoint(model: &crate::config::ModelConfig) -> bool {
    if !model.base_url.is_empty() {
        return true;
    }
    adapter_for(&model.provider.to_ascii_lowercase()).is_some_and(|a| native_endpoint(a).is_some())
}

fn build_provider(model: &crate::config::ModelConfig) -> Option<Arc<dyn LlmProvider>> {
    if model.model.is_empty() {
        return None;
    }
    let provider = model.provider.to_ascii_lowercase();
    let adapter = adapter_for(&provider)?;
    // A native cloud adapter resolves its canonical endpoint from `native_endpoint`,
    // so an empty `base_url` is fine. Local Ollama / the custom openai-compatible
    // endpoint have NO canonical default — they still require a configured base_url.
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::ModelConfig;
    use futures_util::StreamExt;
    use wiremock::matchers::{method, path};
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
            thinking: false,
            reasoning_effort: None,
        }
    }

    /// An Ollama `/api/chat` non-streaming response body genai can parse.
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
        // temperature 0.0 + json mode must be on the options (the deterministic
        // enrichment contract). We assert via the public ChatOptions getters.
        assert_eq!(opts.temperature, Some(0.0));
        assert!(
            matches!(opts.response_format, Some(ChatResponseFormat::JsonMode)),
            "json:true must map to ChatResponseFormat::JsonMode"
        );
        assert_eq!(opts.max_tokens, Some(64));
        // thinking OFF (default) ⇒ no reasoning effort.
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
        // json:false ⇒ no forced JSON mode.
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
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ollama_chat_body("ok")))
            .mount(&server)
            .await;

        let provider: Arc<dyn LlmProvider> = Arc::new(GenaiProvider::new(
            AdapterKind::Ollama,
            "llama3",
            &server.uri(),
            "",
        ));
        assert!(provider.reachable().await);
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
    async fn genai_reachable_false_on_500() {
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
        // The DEFAULT trait impl (used by every enrichment mock) buffers a
        // non-streaming generate into TextDelta + Done.
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
        // genai's Ollama adapter falls back to a buffered single chunk + End when
        // the endpoint returns a non-streamed body, so a wiremock NDJSON-less
        // round-trip still exercises our event mapping (TextDelta + Done).
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

        // At least one text delta and a terminal Done.
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

    /// A catalog-valid Anthropic entry (the bundled catalog lists this model).
    fn anthropic_entry(model: &str) -> ModelConfig {
        ModelConfig {
            provider: "anthropic".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            model: model.to_string(),
            api_key: "sk-ant".to_string(),
            ..ModelConfig::default()
        }
    }

    /// A catalog-valid Google entry (the bundled catalog lists Gemini models).
    fn google_entry(model: &str) -> ModelConfig {
        ModelConfig {
            provider: "google".to_string(),
            base_url: "https://generativelanguage.googleapis.com/v1beta/openai".to_string(),
            model: model.to_string(),
            api_key: "g-key".to_string(),
            ..ModelConfig::default()
        }
    }

    /// A custom self-hosted OpenAI-compatible entry (e.g. LM Studio): a
    /// user-supplied base_url serving an ARBITRARY model that is NOT in any
    /// models.dev namespace.
    fn custom_openai_entry(model: &str) -> ModelConfig {
        ModelConfig {
            provider: "openai-compatible".to_string(),
            base_url: "http://localhost:1234/v1".to_string(),
            model: model.to_string(),
            api_key: "sk-local".to_string(),
            ..ModelConfig::default()
        }
    }

    /// The first model id present in the bundled catalog for `provider` (so
    /// catalog validation passes for the cloud-selection tests).
    fn catalog_model(provider: &str) -> String {
        let catalog = ModelCatalog::bundled();
        catalog
            .provider(provider)
            .and_then(|p| p.models.keys().next())
            .cloned()
            .unwrap_or_else(|| panic!("bundled catalog has at least one {provider} model"))
    }

    /// The first Anthropic model id present in the bundled catalog (so catalog
    /// validation passes for the cloud-selection tests).
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
        // Cloud present but consent withheld → cloud skipped → falls back to local.
        let cfg = config_with(
            vec![anthropic_entry(&model), ollama_entry()],
            LlmRouting::CloudFirst,
        );
        let p = provider_from_config(&cfg, false).expect("falls back to local");
        assert_eq!(p.model_id(), "llama3");
    }

    #[test]
    fn cloud_rejected_when_model_not_in_catalog() {
        // A free-string model id that isn't in the catalog must be rejected even
        // with consent (the Stage-1 anti-free-string guard).
        let cfg = config_with(
            vec![anthropic_entry("totally-made-up-model")],
            LlmRouting::CloudFirst,
        );
        assert!(
            provider_from_config(&cfg, true).is_none(),
            "uncatalogued cloud model must be rejected"
        );
    }

    #[test]
    fn anthropic_provider_validates_against_own_namespace() {
        // Fix #1 regression guard: a first-class Anthropic entry (provider id
        // "anthropic") must validate the claude-* model against the ANTHROPIC
        // catalog namespace and be selected — NOT validated against "openai" (the
        // old openai-compatible→openai mapping that silently broke routing).
        let model = catalog_model("anthropic");
        assert!(model.starts_with("claude"), "expected a claude-* model");
        let cfg = config_with(vec![anthropic_entry(&model)], LlmRouting::CloudFirst);
        let p = provider_from_config(&cfg, true).expect("anthropic (claude-*) must select");
        assert_eq!(p.model_id(), model);
    }

    #[test]
    fn google_provider_validates_against_own_namespace() {
        // Fix #1: a Google entry (provider id "google") validates its gemini-*
        // model against the GOOGLE namespace and selects (maps to genai's native
        // Gemini adapter).
        let model = catalog_model("google");
        assert!(model.starts_with("gemini"), "expected a gemini-* model");
        let cfg = config_with(vec![google_entry(&model)], LlmRouting::CloudFirst);
        let p = provider_from_config(&cfg, true).expect("google (gemini-*) must select");
        assert_eq!(p.model_id(), model);
    }

    #[test]
    fn custom_openai_compatible_is_consent_gated_but_unvalidated() {
        // Fix #1: the custom openai-compatible endpoint serves arbitrary models, so
        // a model id absent from any catalog still selects — PROVIDED consent is
        // granted (it is still consent-gated, unlike local Ollama).
        let cfg = config_with(
            vec![custom_openai_entry("some-self-hosted-model-v3")],
            LlmRouting::CloudFirst,
        );
        // With consent: selected despite not being in the catalog.
        let p = provider_from_config(&cfg, true).expect("custom endpoint selects with consent");
        assert_eq!(p.model_id(), "some-self-hosted-model-v3");
        // Without consent: rejected (still a cloud-bound endpoint).
        assert!(
            provider_from_config(&cfg, false).is_none(),
            "custom endpoint is consent-gated"
        );
    }

    #[test]
    fn legacy_openai_compatible_config_still_works_as_custom_endpoint() {
        // Backward-compat: a legacy config persisted with provider "openai-compatible"
        // + a base_url (the pre-fix blanket cloud id) must still resolve as the
        // custom-endpoint case — no catalog validation — so existing installs keep
        // enriching.
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
    fn catalog_key_for_maps_new_native_providers_to_own_namespace() {
        assert_eq!(catalog_key_for("groq"), Some("groq"));
        assert_eq!(catalog_key_for("deepseek"), Some("deepseek"));
        assert_eq!(catalog_key_for("xai"), Some("xai"));
        assert_eq!(catalog_key_for("cohere"), Some("cohere"));
    }

    #[test]
    fn native_endpoint_covers_new_providers_and_skips_custom_local() {
        // Native cloud adapters expose a canonical endpoint (genai built-in).
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
        // Local Ollama has no canonical default (the user's base_url IS the runtime).
        assert!(native_endpoint(AdapterKind::Ollama).is_none());
    }

    /// A native cloud entry with NO `base_url` (the new combobox path): the
    /// canonical genai endpoint is used; only the key + model are configured.
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
        // No base_url: the native Groq endpoint is resolved internally.
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
    fn new_native_cloud_model_rejected_when_not_in_catalog() {
        let cfg = config_with(
            vec![native_cloud_entry("groq", "totally-made-up-model")],
            LlmRouting::CloudFirst,
        );
        assert!(
            provider_from_config(&cfg, true).is_none(),
            "uncatalogued groq model must be rejected"
        );
    }

    #[test]
    fn native_cloud_entry_with_empty_base_url_is_usable() {
        // Regression: the old `usable`/`build_provider` guards rejected an empty
        // base_url outright. Native cloud adapters must now be usable without one.
        let model = catalog_model("xai");
        assert!(has_endpoint(&native_cloud_entry("xai", &model)));
        // Local Ollama with no base_url is still NOT usable (no canonical default).
        assert!(!has_endpoint(&ModelConfig {
            provider: "ollama".to_string(),
            base_url: String::new(),
            model: "llama3".to_string(),
            ..ModelConfig::default()
        }));
    }

    #[test]
    fn existing_native_providers_still_use_configured_base_url() {
        // The existing 3 (openai/anthropic/google) pass a full base_url from config;
        // a non-empty base_url must still WIN (explicit override), unchanged.
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

    // --- per-task provider override (Stage 3) -------------------------------

    use crate::config::TaskModel;

    /// A base [`GenaiProvider`] (over a dead URL — never dispatched in these
    /// selection tests) used as the client lender for per-task siblings.
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
        // No override ⇒ same model id as base (the routing default).
        assert_eq!(p.model_id(), "qwen2.5-instruct");
    }

    #[test]
    fn task_provider_pins_local_override_model() {
        // The product ask: coref pinned to a coder model while the base/default is a
        // generalist — coref must use the coder, not the configured default.
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
    fn task_provider_rejects_uncatalogued_cloud_override() {
        let base = base_genai("qwen2.5-instruct");
        let models = vec![anthropic_entry("totally-made-up-model")];
        let map = TaskModel {
            provider: "anthropic".to_string(),
            model: "totally-made-up-model".to_string(),
        };
        // Consent granted but the model isn't in the catalog ⇒ rejected ⇒ base.
        let p = task_provider_from_config(&base, Some(&map), &models, true);
        assert_eq!(p.model_id(), "qwen2.5-instruct");
    }

    #[test]
    fn task_provider_falls_back_when_no_matching_config_entry() {
        // An override naming a provider with no configured entry (no base_url/key to
        // source) falls back to the base provider.
        let base = base_genai("qwen2.5-instruct");
        let models = vec![ollama_entry()];
        let coref = TaskModel {
            provider: "anthropic".to_string(),
            model: catalog_anthropic_model(),
        };
        let p = task_provider_from_config(&base, Some(&coref), &models, true);
        assert_eq!(p.model_id(), "qwen2.5-instruct");
    }
}
