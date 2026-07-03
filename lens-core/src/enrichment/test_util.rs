//! Shared enrichment test helpers. [`ScriptedProvider`] is the mock `LlmProvider`
//! used by unit and integration tests (structural-map, coref, step4, step5).
//! Exposed via `test-util` feature for the integration-test crate.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;

use crate::error::LensError;
use crate::llm::{LlmProvider, LlmRequest, LlmResponse};

/// Call-counted mock `LlmProvider`. Always reachable by default, serving scripted
/// responses (cycling the last entry when exhausted). `dead()` makes it unreachable
/// and returns `LensError::Network` on every `generate()`.
pub struct ScriptedProvider {
    calls: Arc<AtomicU32>,
    responses: Vec<String>,
    model: String,
    dead: bool,
}

impl ScriptedProvider {
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
