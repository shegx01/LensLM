//! Shared `#[cfg(test)]` helpers for the enrichment unit tests.
//!
//! [`ScriptedProvider`] is the single source of truth for the mock
//! [`LlmProvider`](crate::llm::LlmProvider) used by both the structural-map
//! ([`super::map`]) and coref ([`super::coref`]) tests — a call-counted provider
//! that returns a scripted response sequence (cycling the last entry once the
//! script is exhausted).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;

use crate::error::LensError;
use crate::llm::{LlmProvider, LlmRequest, LlmResponse};

/// A mock provider with a call-counter returning a scripted response sequence
/// (cycling the last entry once exhausted).
pub(crate) struct ScriptedProvider {
    calls: Arc<AtomicU32>,
    responses: Vec<String>,
}

impl ScriptedProvider {
    /// Builds a provider over `responses` plus the shared call-counter the tests
    /// assert on.
    pub(crate) fn new(responses: Vec<&str>) -> (Self, Arc<AtomicU32>) {
        let calls = Arc::new(AtomicU32::new(0));
        (
            Self {
                calls: calls.clone(),
                responses: responses.into_iter().map(|s| s.to_string()).collect(),
            },
            calls,
        )
    }
}

#[async_trait]
impl LlmProvider for ScriptedProvider {
    fn model_id(&self) -> &str {
        "mock-model"
    }
    async fn reachable(&self) -> bool {
        true
    }
    async fn generate(&self, _req: &LlmRequest) -> Result<LlmResponse, LensError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) as usize;
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
