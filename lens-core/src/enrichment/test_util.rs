//! Shared test helpers for the enrichment passes.
//!
//! [`ScriptedProvider`] is the single source of truth for the mock
//! [`LlmProvider`](crate::llm::LlmProvider) used by the enrichment tests — both the
//! in-crate unit tests (structural-map [`super::map`], coref [`super::coref`]) and
//! the integration suites (`tests/enrichment_step4.rs`, `tests/enrichment_step5.rs`).
//! It is a call-counted provider with:
//!   * a configurable [`model_id`](crate::llm::LlmProvider::model_id) (a component
//!     of the AC9 cache key, so a model-id change re-runs enrichment);
//!   * a scripted response sequence (cycling the last entry once exhausted); and
//!   * a "dead" mode where `reachable()` is `false` and `generate()` errors
//!     (`LensError::Network`) — the LLM-down / 429 path.
//!
//! Exposed publicly under the `test-util` feature so the separate integration-test
//! crate can construct it; the in-crate unit tests use the same type via the
//! `#[cfg(test)]` build.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;

use crate::error::LensError;
use crate::llm::{LlmProvider, LlmRequest, LlmResponse};

/// A configurable mock [`LlmProvider`] with a call-counter.
///
/// Defaults to an always-reachable provider serving a scripted body sequence
/// (cycling the last entry once the script is exhausted). The builder-style setters
/// pin a custom `model_id` ([`with_model`](Self::with_model)) or flip it to the
/// "dead" mode ([`dead`](Self::dead)) where `reachable()` is `false` and every
/// `generate()` returns a transport error.
pub struct ScriptedProvider {
    calls: Arc<AtomicU32>,
    responses: Vec<String>,
    model: String,
    dead: bool,
}

impl ScriptedProvider {
    /// Builds an always-reachable provider over `responses` (with the default model
    /// id `"mock-model"`), returning it alongside the shared call-counter the tests
    /// assert on.
    pub fn new(responses: Vec<&str>) -> (Self, Arc<AtomicU32>) {
        let calls = Arc::new(AtomicU32::new(0));
        let provider = Self {
            calls: calls.clone(),
            responses: responses.into_iter().map(|s| s.to_string()).collect(),
            model: "mock-model".to_string(),
            dead: false,
        };
        (provider, calls)
    }

    /// A "dead" provider (LLM down / 429): `reachable()` is `false` and every
    /// `generate()` returns [`LensError::Network`] (the counter still increments so
    /// a test can assert how many dispatches were attempted). Returns it alongside
    /// the shared call-counter.
    pub fn dead() -> (Self, Arc<AtomicU32>) {
        let calls = Arc::new(AtomicU32::new(0));
        let provider = Self {
            calls: calls.clone(),
            responses: Vec::new(),
            model: "dead".to_string(),
            dead: true,
        };
        (provider, calls)
    }

    /// Overrides the `model_id` this mock reports (the AC9 cache-key component).
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }
}

#[async_trait]
impl LlmProvider for ScriptedProvider {
    fn model_id(&self) -> &str {
        &self.model
    }
    async fn reachable(&self) -> bool {
        !self.dead
    }
    async fn generate(&self, _req: &LlmRequest) -> Result<LlmResponse, LensError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) as usize;
        if self.dead {
            return Err(LensError::Network("connection refused".into()));
        }
        let text = self
            .responses
            .get(n)
            .or_else(|| self.responses.last())
            .cloned()
            .unwrap_or_default();
        Ok(LlmResponse {
            text,
            tokens_used: 10,
        })
    }
}
