//! Opt-in cross-encoder reranker (issue #39). A strictly optional accelerator:
//! every failure branch (disabled / model-absent / init error / inference error /
//! timeout) falls back to the input (RRF) order and the query still returns `Ok`.
//! The model never blocks the query — inference runs on `spawn_blocking` under a
//! `tokio::time::timeout`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fastembed::{RerankInitOptions, RerankerModel as FeRerankerModel, TextRerank};
use tokio::sync::OnceCell;

use crate::LensError;
use crate::config::{RerankerConfig, RerankerModel};

use super::RetrievalHit;

/// Lazily-initialized reranker handle. `TextRerank::rerank` takes `&mut self`, so
/// the model is wrapped in a `Mutex` (verified against fastembed 5.17.2). The
/// `OnceCell` guarantees a single ONNX init even under concurrent first calls; the
/// model is NEVER initialized when the reranker is disabled (no download).
pub struct Reranker {
    cache_dir: PathBuf,
    cell: OnceCell<Arc<Mutex<TextRerank>>>,
}

impl std::fmt::Debug for Reranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reranker")
            .field("cache_dir", &self.cache_dir)
            .field("initialized", &self.cell.initialized())
            .finish()
    }
}

impl Reranker {
    /// Builds a reranker bound to `{data_dir}/models/fastembed` (shared with the
    /// embedder cache; fastembed namespaces by model repo). Does NOT init the model.
    pub fn new(data_dir: &std::path::Path) -> Self {
        Self {
            cache_dir: data_dir.join("models").join("fastembed"),
            cell: OnceCell::new(),
        }
    }

    /// Re-scores `candidates` for `query` with the cross-encoder and returns them
    /// re-ordered top-`limit`. On ANY failure — disabled, model absent, init error,
    /// inference error, or timeout — returns the input order truncated to `limit`
    /// and logs the reason. `candidates` must be paired with their hydrated `texts`
    /// in the SAME order.
    pub async fn rerank_with_fallback(
        &self,
        query: &str,
        candidates: Vec<RetrievalHit>,
        texts: Vec<String>,
        config: &RerankerConfig,
        limit: usize,
    ) -> Vec<RetrievalHit> {
        if !config.enabled {
            return truncate(candidates, limit);
        }
        if candidates.is_empty() {
            return candidates;
        }
        if candidates.len() != texts.len() {
            tracing::warn!(
                candidates = candidates.len(),
                texts = texts.len(),
                "reranker fallback: candidate/text length mismatch; returning RRF order"
            );
            return truncate(candidates, limit);
        }

        let handle = match self.handle(config.model).await {
            Ok(h) => h,
            Err(e) => {
                tracing::warn!(error = %e, "reranker fallback: model init failed; returning RRF order");
                return truncate(candidates, limit);
            }
        };

        let query = query.to_string();
        let texts_for_job = texts.clone();
        let timeout = Duration::from_millis(config.timeout_ms);
        let job = tokio::task::spawn_blocking(move || {
            let mut model = handle
                .lock()
                .map_err(|e| LensError::Model(format!("reranker mutex poisoned: {e}")))?;
            let docs: Vec<&str> = texts_for_job.iter().map(String::as_str).collect();
            model
                .rerank(query.as_str(), docs, false, None)
                .map_err(|e| LensError::Model(format!("reranker inference failed: {e}")))
        });

        let results = match tokio::time::timeout(timeout, job).await {
            Ok(Ok(Ok(results))) => results,
            Ok(Ok(Err(e))) => {
                tracing::warn!(error = %e, "reranker fallback: inference error; returning RRF order");
                return truncate(candidates, limit);
            }
            Ok(Err(join_err)) => {
                tracing::warn!(error = %join_err, "reranker fallback: blocking task panicked; returning RRF order");
                return truncate(candidates, limit);
            }
            Err(_) => {
                tracing::warn!(
                    timeout_ms = config.timeout_ms,
                    "reranker fallback: timed out; returning RRF order"
                );
                return truncate(candidates, limit);
            }
        };

        // fastembed keys each result by its INDEX into the passed document list;
        // map back to candidate[index] and carry the reranker score.
        let mut reordered: Vec<RetrievalHit> = Vec::with_capacity(results.len());
        for r in results {
            if let Some(c) = candidates.get(r.index) {
                reordered.push(RetrievalHit {
                    chunk_id: c.chunk_id.clone(),
                    score: r.score,
                    source: c.source,
                });
            }
        }
        if reordered.is_empty() {
            tracing::warn!("reranker fallback: no results mapped back; returning RRF order");
            return truncate(candidates, limit);
        }
        reordered.truncate(limit);
        reordered
    }

    /// Resolves the lazily-initialized model handle, initializing it exactly once.
    async fn handle(&self, model: RerankerModel) -> Result<Arc<Mutex<TextRerank>>, LensError> {
        let cache_dir = self.cache_dir.clone();
        let handle = self
            .cell
            .get_or_try_init(|| async move {
                let fe_model = to_fastembed_model(model);
                tokio::task::spawn_blocking(move || {
                    let opts = RerankInitOptions::new(fe_model).with_cache_dir(cache_dir);
                    TextRerank::try_new(opts)
                        .map(|m| Arc::new(Mutex::new(m)))
                        .map_err(|e| LensError::Model(format!("reranker init failed: {e}")))
                })
                .await
                .map_err(|e| LensError::Model(format!("reranker init task panicked: {e}")))?
            })
            .await?;
        Ok(handle.clone())
    }
}

/// Maps the config enum to the fastembed reranker model. Only the MIT-licensed
/// base model is surfaced (plan Out-of-scope: v2-m3 mirror).
fn to_fastembed_model(model: RerankerModel) -> FeRerankerModel {
    match model {
        RerankerModel::BgeRerankerBase => FeRerankerModel::BGERerankerBase,
    }
}

fn truncate(mut hits: Vec<RetrievalHit>, limit: usize) -> Vec<RetrievalHit> {
    hits.truncate(limit);
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::HitSource;

    fn hits(ids: &[&str]) -> Vec<RetrievalHit> {
        ids.iter()
            .enumerate()
            .map(|(i, id)| RetrievalHit {
                chunk_id: id.to_string(),
                score: 1.0 / (i + 1) as f32,
                source: HitSource::Both,
            })
            .collect()
    }

    fn texts(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("doc {i}")).collect()
    }

    #[tokio::test]
    async fn disabled_returns_rrf_order_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let r = Reranker::new(dir.path());
        let cfg = RerankerConfig {
            enabled: false,
            ..RerankerConfig::default()
        };
        let input = hits(&["a", "b", "c"]);
        let out = r
            .rerank_with_fallback("q", input.clone(), texts(3), &cfg, 5)
            .await;
        assert_eq!(
            out.iter().map(|h| h.chunk_id.as_str()).collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
        assert!(!r.cell.initialized(), "disabled must NOT init the model");
    }

    #[tokio::test]
    async fn disabled_truncates_to_limit() {
        let dir = tempfile::tempdir().unwrap();
        let r = Reranker::new(dir.path());
        let cfg = RerankerConfig {
            enabled: false,
            ..RerankerConfig::default()
        };
        let out = r
            .rerank_with_fallback("q", hits(&["a", "b", "c", "d"]), texts(4), &cfg, 2)
            .await;
        assert_eq!(out.len(), 2);
    }

    #[tokio::test]
    async fn enabled_but_model_absent_falls_back_to_rrf_order() {
        // Point the cache at an empty temp dir with an unreachable HF endpoint so
        // init fails fast; the query must still return the RRF order, never error.
        let dir = tempfile::tempdir().unwrap();
        // SAFETY: single-threaded test setting an env var before the init call.
        unsafe {
            std::env::set_var("HF_ENDPOINT", "http://127.0.0.1:1");
        }
        let r = Reranker::new(dir.path());
        let cfg = RerankerConfig {
            enabled: true,
            timeout_ms: 2_000,
            ..RerankerConfig::default()
        };
        let input = hits(&["a", "b", "c"]);
        let out = r
            .rerank_with_fallback("q", input.clone(), texts(3), &cfg, 5)
            .await;
        unsafe {
            std::env::remove_var("HF_ENDPOINT");
        }
        assert_eq!(
            out.iter().map(|h| h.chunk_id.as_str()).collect::<Vec<_>>(),
            vec!["a", "b", "c"],
            "model-absent must fall back to the RRF order"
        );
    }

    #[tokio::test]
    async fn empty_candidates_return_empty() {
        let dir = tempfile::tempdir().unwrap();
        let r = Reranker::new(dir.path());
        let cfg = RerankerConfig {
            enabled: true,
            ..RerankerConfig::default()
        };
        let out = r
            .rerank_with_fallback("q", Vec::new(), Vec::new(), &cfg, 5)
            .await;
        assert!(out.is_empty());
        assert!(
            !r.cell.initialized(),
            "no candidates must NOT init the model"
        );
    }
}
