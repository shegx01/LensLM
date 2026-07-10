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

use std::collections::HashMap;

use sqlx::SqlitePool;

use crate::LensError;
use crate::config::RetrievalConfig;
use crate::graph::{EntityKind, NotebookGraph, entity_evidence, ppr_expand};
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
/// precedes equal-confidence `ppr_expand` output; see [`merge_ranked`].
// #21 should extract this seed→chunk composition into a shared module, not import
// graph_arm from eval.
pub async fn graph_arm(
    pool: &SqlitePool,
    graph: &NotebookGraph,
    seeds: &[(String, EntityKind)],
    k: usize,
) -> Result<Vec<String>, LensError> {
    let notebook_id = graph.notebook_id().to_string();
    let mut evidence = Vec::new();
    for (name, kind) in seeds {
        evidence.extend(entity_evidence(pool, &notebook_id, name, *kind, k).await?);
    }
    let expansion: Vec<(String, f32)> = ppr_expand(pool, graph, seeds, k)
        .await?
        .into_iter()
        .flat_map(|hit| {
            hit.chunk_ids
                .into_iter()
                .map(move |c| (c, hit.graph_confidence))
        })
        .collect();
    Ok(merge_ranked(evidence, expansion, k))
}

/// The LOCKED graph_arm merge (extracted pure for offline testing). `evidence` gets
/// fixed conf `1.0` in call order; `expansion` carries traversal `graph_confidence`.
/// Dedup by chunk id keeping the highest conf; stable sort conf DESC, first-seen
/// tie-break (evidence precedes equal-conf expansion); truncate to `k`.
fn merge_ranked(evidence: Vec<String>, expansion: Vec<(String, f32)>, k: usize) -> Vec<String> {
    // chunk_id -> (best confidence, first-seen ordinal).
    let mut best: HashMap<String, (f32, usize)> = HashMap::new();
    let mut seen = 0usize;
    for chunk_id in evidence {
        insert_ranked(&mut best, &mut seen, chunk_id, 1.0);
    }
    for (chunk_id, conf) in expansion {
        insert_ranked(&mut best, &mut seen, chunk_id, conf);
    }
    let mut ranked: Vec<(String, f32, usize)> =
        best.into_iter().map(|(id, (c, o))| (id, c, o)).collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.2.cmp(&b.2))
    });
    ranked.truncate(k);
    ranked.into_iter().map(|(id, _, _)| id).collect()
}

fn insert_ranked(
    best: &mut HashMap<String, (f32, usize)>,
    seen: &mut usize,
    chunk_id: String,
    conf: f32,
) {
    best.entry(chunk_id)
        .and_modify(|(c, _)| *c = c.max(conf))
        .or_insert_with(|| {
            let ord = *seen;
            *seen += 1;
            (conf, ord)
        });
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

    #[test]
    fn merge_ranks_evidence_before_equal_confidence_expansion() {
        // Evidence chunk "e" (conf 1.0) must outrank an expansion chunk "x" even
        // when expansion confidence also normalizes to 1.0 (first-seen tie-break).
        let out = merge_ranked(
            vec!["e".into()],
            vec![("x".into(), 1.0), ("y".into(), 0.4)],
            5,
        );
        assert_eq!(out, vec!["e", "x", "y"]);
    }

    #[test]
    fn merge_dedups_keeping_highest_confidence_and_truncates() {
        // "d" appears as both evidence (1.0) and low-conf expansion (0.2): one
        // entry at conf 1.0. Expansion sorts by confidence DESC. Truncate to k=2.
        let out = merge_ranked(
            vec!["d".into()],
            vec![("d".into(), 0.2), ("hi".into(), 0.9), ("lo".into(), 0.1)],
            2,
        );
        assert_eq!(out, vec!["d", "hi"]);
    }
}
