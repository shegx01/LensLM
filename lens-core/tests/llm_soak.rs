//! Real-model LLM soak (#258 Phase-2 acceptance): drives the ACTIVE LLM backend against real
//! endpoints across the four migration-risk dimensions — reachability, buffered generate,
//! streaming, and structured (JSON) output. (`reachable()` is a live probe for local Ollama;
//! cloud providers report reachable without a network call by design, so there the assertion just
//! pins that contract.) Opt-in and offline-by-default: without `LENS_RUN_MODEL_TESTS=1` every
//! target skips; with it, a cloud target skips when its key is unset and Ollama skips when no local
//! runtime answers — so CI (which sets neither key nor runtime) runs nothing here.
//!
//! Run: `LENS_RUN_MODEL_TESTS=1 cargo test -p lens-core --test llm_soak -- --ignored`.
//! Ollama reads `LENS_SOAK_OLLAMA_URL` (default `http://localhost:11434`) + `LENS_SOAK_OLLAMA_MODEL`
//! (default `llama3.2:3b`). Cloud reads `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` and the optional
//! `LENS_SOAK_OPENAI_MODEL` / `LENS_SOAK_ANTHROPIC_MODEL` overrides.

use std::sync::Arc;

use futures_util::StreamExt;
use lens_core::{
    AppConfig, EnrichmentConfig, LlmProvider, LlmRequest, LlmRouting, StreamChunk,
    provider_from_config,
};

fn run_model_tests() -> bool {
    std::env::var("LENS_RUN_MODEL_TESTS").is_ok()
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// Builds the active provider for one target via the real config path (`provider_from_config`),
/// exactly as the app pins an enrichment model. `ModelConfig` is not a public type, so the entry
/// is deserialized from JSON rather than struct-constructed.
fn build(
    provider: &str,
    model: &str,
    base_url: &str,
    api_key: &str,
) -> Option<Arc<dyn LlmProvider>> {
    let cfg = AppConfig {
        models: serde_json::from_value(serde_json::json!([{
            "provider": provider,
            "base_url": base_url,
            "model": model,
            "context": 8192,
            "temperature": 0.0,
            "api_key": api_key,
        }]))
        .expect("model config deserializes"),
        enrichment: EnrichmentConfig {
            routing: LlmRouting::Explicit {
                provider: provider.to_string(),
                model: model.to_string(),
            },
            ..Default::default()
        },
        ..Default::default()
    };
    provider_from_config(&cfg, true)
}

fn base_req(prompt: &str) -> LlmRequest {
    LlmRequest {
        system: Some("You are terse.".to_string()),
        prompt: prompt.to_string(),
        max_tokens: 128,
        temperature: 0.0,
        json: false,
        thinking: false,
        reasoning_effort: None,
        messages: Vec::new(),
    }
}

/// Tolerant JSON-object acceptance for the structured-output dimension: the prompt asks for an
/// object, so a bare scalar/array does not count; some models fence or prefix it, so fall back to
/// the outermost `{..}` slice.
fn parse_jsonish(s: &str) -> Option<serde_json::Value> {
    let as_object = |v: serde_json::Value| v.is_object().then_some(v);
    if let Some(v) = serde_json::from_str::<serde_json::Value>(s.trim())
        .ok()
        .and_then(as_object)
    {
        return Some(v);
    }
    let start = s.find('{')?;
    let end = s.rfind('}')?;
    serde_json::from_str::<serde_json::Value>(&s[start..=end])
        .ok()
        .and_then(as_object)
}

/// The shared four-dimension soak: reachable -> generate -> stream -> structured JSON.
async fn soak(label: &str, provider: Arc<dyn LlmProvider>) {
    assert!(
        provider.reachable().await,
        "{label}: reachable() must be true for a live endpoint"
    );

    let resp = provider
        .generate(&base_req("Reply with the single word: pong."))
        .await
        .unwrap_or_else(|e| panic!("{label}: generate failed: {e}"));
    assert!(
        !resp.text.trim().is_empty(),
        "{label}: generate returned empty text"
    );

    let mut stream = provider
        .generate_stream(&base_req("Count from one to three."))
        .await
        .unwrap_or_else(|e| panic!("{label}: generate_stream failed: {e}"));
    let mut streamed = String::new();
    let mut saw_done = false;
    while let Some(item) = stream.next().await {
        match item.unwrap_or_else(|e| panic!("{label}: stream error: {e}")) {
            StreamChunk::TextDelta(t) => streamed.push_str(&t),
            StreamChunk::ThinkingDelta(_) => {}
            StreamChunk::Done { .. } => {
                saw_done = true;
                break;
            }
        }
    }
    assert!(saw_done, "{label}: stream never emitted Done");
    assert!(
        !streamed.trim().is_empty(),
        "{label}: stream produced no text"
    );

    let mut json_req = base_req("Return only a JSON object with a single key \"ok\" set to true.");
    json_req.json = true;
    let jr = provider
        .generate(&json_req)
        .await
        .unwrap_or_else(|e| panic!("{label}: structured generate failed: {e}"));
    assert!(
        parse_jsonish(&jr.text).is_some(),
        "{label}: structured output is not JSON: {}",
        jr.text
    );

    eprintln!("{label}: soak OK (reachable + generate + stream + json)");
}

#[tokio::test]
#[ignore = "needs a real local Ollama; run with LENS_RUN_MODEL_TESTS=1 --ignored"]
async fn soak_ollama() {
    if !run_model_tests() {
        eprintln!("skipping soak_ollama (set LENS_RUN_MODEL_TESTS=1)");
        return;
    }
    let url = env_or("LENS_SOAK_OLLAMA_URL", "http://localhost:11434");
    let model = env_or("LENS_SOAK_OLLAMA_MODEL", "llama3.2:3b");
    let provider = build("ollama", &model, &url, "").expect("ollama provider builds");
    if !provider.reachable().await {
        eprintln!("skipping soak_ollama (no reachable Ollama at {url})");
        return;
    }
    soak(&format!("ollama:{model}"), provider).await;
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY; run with LENS_RUN_MODEL_TESTS=1 --ignored"]
async fn soak_openai() {
    if !run_model_tests() {
        eprintln!("skipping soak_openai (set LENS_RUN_MODEL_TESTS=1)");
        return;
    }
    let key = env_or("OPENAI_API_KEY", "");
    if key.is_empty() {
        eprintln!("skipping soak_openai (no OPENAI_API_KEY)");
        return;
    }
    let model = env_or("LENS_SOAK_OPENAI_MODEL", "gpt-4o-mini");
    let provider = build("openai", &model, "", &key).expect("openai provider builds");
    soak(&format!("openai:{model}"), provider).await;
}

#[tokio::test]
#[ignore = "needs ANTHROPIC_API_KEY; run with LENS_RUN_MODEL_TESTS=1 --ignored"]
async fn soak_anthropic() {
    if !run_model_tests() {
        eprintln!("skipping soak_anthropic (set LENS_RUN_MODEL_TESTS=1)");
        return;
    }
    let key = env_or("ANTHROPIC_API_KEY", "");
    if key.is_empty() {
        eprintln!("skipping soak_anthropic (no ANTHROPIC_API_KEY)");
        return;
    }
    let model = env_or("LENS_SOAK_ANTHROPIC_MODEL", "claude-3-5-haiku-latest");
    let provider = build("anthropic", &model, "", &key).expect("anthropic provider builds");
    soak(&format!("anthropic:{model}"), provider).await;
}
