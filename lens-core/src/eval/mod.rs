//! M13 #158a: entity-graph retrieval eval — tool-level, agent-agnostic recall@5.
//!
//! One deterministic measurement core (`recall_at_k` + `graph_arm`/`hybrid_arm`)
//! fed two ways: the `--graph` CI gate in `bin/eval.rs` (static fixtures) and the
//! `LENS_RUN_MODEL_TESTS`-gated per-notebook runtime harness (`run_notebook_eval`).
//! Gold is generation-provenance: the LLM emits chunk ids from the fed corpus, so it
//! is independent of BOTH retrievers and makes the ≥5pp promotion bar reachable.

mod runtime;

pub use runtime::{
    EvalOutcome, EvalReport, QA_PROMPT_VERSION, QuestionKind, RunEvalDeps, SeedEntity,
    run_notebook_eval,
};

use sqlx::SqlitePool;

use crate::LensError;
use crate::config::RetrievalConfig;
use crate::graph::{EntityKind, NotebookGraph};
use crate::retrieval::router::graph_compose;
use crate::retrieval::{Reranker, hybrid_search};
use crate::vector_store::{Coordinate, VectorStore};

/// Fraction of the gold set retrieved in the top-`k` (`|gold ∩ top-k| / |gold|`).
/// Empty gold → `0.0` (a question with no gold cannot be scored; the runtime
/// harness drops such questions rather than letting them deflate the mean).
pub fn recall_at_k(retrieved: &[String], gold: &[String], k: usize) -> f32 {
    if gold.is_empty() {
        return 0.0;
    }
    let top: &[String] = &retrieved[..retrieved.len().min(k)];
    let hits = gold.iter().filter(|g| top.contains(g)).count();
    hits as f32 / gold.len() as f32
}

/// Mean of a slice; empty → `0.0` (an empty scored subset scores neutral, not NaN).
pub fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f32>() / xs.len() as f32
}

/// Tool-level graph retrieval for one question's seed entities, agent-agnostic.
/// Merge invariant (LOCKED, #158a): evidence (direct mentions, fixed conf 1.0)
/// precedes equal-confidence `ppr_expand` output. Delegates to the #21 router's
/// pair-preserving `graph_compose` and drops the confidence to keep this arm's
/// `Vec<String>` output byte-for-byte identical to before the extraction.
pub async fn graph_arm(
    pool: &SqlitePool,
    graph: &NotebookGraph,
    seeds: &[(String, EntityKind)],
    k: usize,
) -> Result<Vec<String>, LensError> {
    Ok(graph_compose(pool, graph, seeds, k)
        .await?
        .into_iter()
        .map(|(id, _)| id)
        .collect())
}

/// The hybrid baseline arm as bare `chunk_id`s — thin wrapper over [`hybrid_search`];
/// the caller owns the query embedding (both feeders already have an embedder).
#[allow(clippy::too_many_arguments)]
pub async fn hybrid_arm(
    pool_db: &SqlitePool,
    store: &dyn VectorStore,
    reranker: &Reranker,
    coord: &Coordinate,
    query_text: &str,
    query_vec: &[f32],
    k: usize,
    config: &RetrievalConfig,
) -> Result<Vec<String>, LensError> {
    let hits = hybrid_search(
        pool_db, store, reranker, coord, query_text, query_vec, None, None, k, config,
    )
    .await?;
    Ok(hits.into_iter().map(|h| h.chunk_id).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_counts_gold_in_top_k() {
        let retrieved = vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()];
        // 2 of 2 gold in top-5.
        assert_eq!(recall_at_k(&retrieved, &["a".into(), "c".into()], 5), 1.0);
        // 1 of 2 gold in top-5 (e is #5, z absent).
        assert_eq!(recall_at_k(&retrieved, &["e".into(), "z".into()], 5), 0.5);
        // gold present but outside top-k.
        assert_eq!(recall_at_k(&retrieved, &["d".into()], 3), 0.0);
    }

    #[test]
    fn recall_empty_gold_is_zero_not_nan() {
        let r = recall_at_k(&["a".into()], &[], 5);
        assert_eq!(r, 0.0, "empty gold must not divide by zero");
    }

    #[test]
    fn recall_k_larger_than_retrieved_is_safe() {
        let retrieved = vec!["a".into()];
        assert_eq!(recall_at_k(&retrieved, &["a".into()], 5), 1.0);
    }

}
