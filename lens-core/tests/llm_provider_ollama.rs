// issue #71: deep `Send` auto-trait evaluation can overflow the default 128-frame
// limit under stricter toolchains (genai's async chain).
#![recursion_limit = "256"]
//! Offline HAPPY-PATH tests for the REAL genai→Ollama LLM path (#29).
//!
//! Existing enrichment/dialogue tests mock the [`LlmProvider`] trait away, so they
//! never exercise the concrete `GenaiProvider` request/response wiring — which is
//! exactly where the #29 timeout + generation bugs lived. These stand up a fake
//! Ollama `/api/chat` endpoint (wiremock, mirroring `cloud_asr`/`cloud_tts`) and
//! drive a real provider built from an `AppConfig` via the public factory:
//!
//! 1. buffered `generate` returns the FULL model content verbatim,
//! 2. a slow-but-reasonable response still succeeds (30s-total-timeout regression),
//! 3. active chat-model resolution: routing fallback and explicit pin.
//!
//! All offline — no real model, no `LENS_RUN_MODEL_TESTS` gate.

use std::time::Duration;

use lens_core::config::{AppConfig, EnrichmentConfig, ModelConfig, TaskModel};
use lens_core::llm::chat_provider_from_config;
use lens_core::{LlmRequest, LlmRouting, provider_from_config};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A canonical `{"turns":[...]}` dialogue script with 8 alternating host/guest
/// turns — the shape #29's audio-overview generation asks the model to return.
/// Serialized compactly so the buffered `generate` round-trip can assert byte-exact
/// equality against `LlmResponse::text`.
fn dialogue_content() -> String {
    serde_json::json!({
        "turns": [
            {"speaker": "host",  "text": "Welcome to today's overview."},
            {"speaker": "guest", "text": "Glad to be here."},
            {"speaker": "host",  "text": "Let's start with the main idea."},
            {"speaker": "guest", "text": "Sure — it centers on local-first design."},
            {"speaker": "host",  "text": "Why does that matter?"},
            {"speaker": "guest", "text": "It keeps data private and on-device."},
            {"speaker": "host",  "text": "Any trade-offs?"},
            {"speaker": "guest", "text": "Mostly the engineering effort up front."}
        ]
    })
    .to_string()
}

/// A non-streamed Ollama `/api/chat` body. `prompt_eval_count + eval_count` is what
/// genai maps into `LlmResponse::tokens_used`.
fn ollama_chat_response(content: &str) -> serde_json::Value {
    serde_json::json!({
        "model": "llama3",
        "message": { "role": "assistant", "content": content },
        "done": true,
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 50
    })
}

fn ollama_model(base_url: &str) -> ModelConfig {
    ModelConfig {
        provider: "ollama".to_string(),
        base_url: base_url.to_string(),
        model: "llama3".to_string(),
        ..ModelConfig::default()
    }
}

/// AppConfig with a single usable local Ollama entry and `LocalFirst` routing, so
/// `provider_from_config` / `chat_provider_from_config` resolve to it without any
/// cloud consent or catalog gate.
fn config_with_ollama(base_url: &str, chat_model: Option<TaskModel>) -> AppConfig {
    AppConfig {
        models: vec![ollama_model(base_url)],
        enrichment: EnrichmentConfig {
            routing: LlmRouting::LocalFirst,
            chat_model,
            ..EnrichmentConfig::default()
        },
        ..AppConfig::default()
    }
}

fn json_request(prompt: &str) -> LlmRequest {
    LlmRequest {
        system: Some("Return a JSON dialogue script.".to_string()),
        prompt: prompt.to_string(),
        max_tokens: 2048,
        temperature: 0.0,
        json: true,
        thinking: false,
        reasoning_effort: None,
        messages: Vec::new(),
    }
}

// ===========================================================================
// 1. Happy path — buffered generate returns the FULL content verbatim.
// ===========================================================================

#[tokio::test]
async fn ollama_generate_returns_full_dialogue_content() {
    let server = MockServer::start().await;
    let content = dialogue_content();
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ollama_chat_response(&content)))
        .mount(&server)
        .await;

    let cfg = config_with_ollama(&server.uri(), None);
    let provider = provider_from_config(&cfg, false).expect("ollama provider builds from config");

    let resp = provider
        .generate(&json_request("Generate an audio overview script."))
        .await
        .expect("buffered generate over the real genai→Ollama path must succeed");

    assert_eq!(
        resp.text, content,
        "buffered generate must return the model's full content byte-for-byte"
    );
    assert_eq!(
        resp.tokens_used, 60,
        "tokens_used must sum prompt_eval_count (10) + eval_count (50)"
    );

    let calls = server.received_requests().await.unwrap();
    assert_eq!(calls.len(), 1, "exactly one /api/chat dispatch");
}

// ===========================================================================
// 2. Happy path — a slow-but-reasonable response still succeeds.
//    Regression guard for the #29 30s-total-timeout bug: the generation path
//    uses an idle (not total) read timeout, so a modest server delay must pass.
// ===========================================================================

#[tokio::test]
async fn ollama_generate_succeeds_under_slow_response() {
    let server = MockServer::start().await;
    let content = dialogue_content();
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(ollama_chat_response(&content))
                .set_delay(Duration::from_secs(1)),
        )
        .mount(&server)
        .await;

    let cfg = config_with_ollama(&server.uri(), None);
    let provider = provider_from_config(&cfg, false).expect("ollama provider builds from config");

    let resp = provider
        .generate(&json_request("Generate an audio overview script."))
        .await
        .expect("a slow-but-reasonable response must still succeed (no premature total timeout)");

    assert_eq!(
        resp.text, content,
        "the full content must survive a delayed response intact"
    );
}

// ===========================================================================
// 3. Happy path — active chat-model resolution (offline; construction only).
// ===========================================================================

#[test]
fn chat_provider_resolves_via_routing_when_no_pin() {
    // No `chat_model` pin + LocalFirst routing + one usable Ollama entry: resolution
    // must still yield a provider. This is the "after restart, routing still
    // resolves the active chat model" happy path.
    let cfg = config_with_ollama("http://localhost:11434", None);
    let provider = chat_provider_from_config(&cfg, false)
        .expect("routing must resolve the local Ollama entry");
    assert_eq!(provider.model_id(), "llama3");
    assert!(provider.is_local(), "the resolved provider is local Ollama");
}

#[test]
fn chat_provider_resolves_matching_pin() {
    // An explicit `chat_model` pin matching a configured entry resolves that exact pin.
    let pin = TaskModel {
        provider: "ollama".to_string(),
        model: "llama3".to_string(),
    };
    let cfg = config_with_ollama("http://localhost:11434", Some(pin));
    let provider =
        chat_provider_from_config(&cfg, false).expect("a matching chat_model pin must resolve");
    assert_eq!(provider.model_id(), "llama3");
    assert!(provider.is_local());
}
