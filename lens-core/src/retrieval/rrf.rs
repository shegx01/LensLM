//! Reciprocal Rank Fusion (issue #39). Pure, offline-testable rank merge of the
//! dense and BM25 candidate lists into a single fused ranking.

use super::{HitSource, RetrievalHit};

/// The RRF constant. `k=60` is the canonical value from the original RRF paper
/// (Cormack et al.); it dampens the contribution of low-ranked items.
pub const RRF_K: usize = 60;

/// Fuses two rank-ordered chunk-id lists (each already best-first) via Reciprocal
/// Rank Fusion: `score(c) = Σ 1/(k + rank(c))` over the lists c appears in, with
/// `rank` 0-based. Returns hits sorted by fused score descending, truncated to
/// `limit`. `source` records which path(s) contributed (dense/bm25/both).
///
/// Rank-only by design: dense cosine distances and BM25 scores are not comparable,
/// so only the ordinal position feeds the fusion.
pub fn rrf_merge(dense: &[String], bm25: &[String], k: usize, limit: usize) -> Vec<RetrievalHit> {
    use std::collections::HashMap;

    struct Acc {
        score: f32,
        in_dense: bool,
        in_bm25: bool,
        // First-seen order as a stable tiebreaker for equal fused scores.
        first_seen: usize,
    }

    let mut acc: HashMap<&str, Acc> = HashMap::new();
    let mut order = 0usize;

    for (rank, id) in dense.iter().enumerate() {
        let e = acc.entry(id.as_str()).or_insert_with(|| {
            let seen = order;
            order += 1;
            Acc {
                score: 0.0,
                in_dense: false,
                in_bm25: false,
                first_seen: seen,
            }
        });
        e.score += 1.0 / (k + rank) as f32;
        e.in_dense = true;
    }
    for (rank, id) in bm25.iter().enumerate() {
        let e = acc.entry(id.as_str()).or_insert_with(|| {
            let seen = order;
            order += 1;
            Acc {
                score: 0.0,
                in_dense: false,
                in_bm25: false,
                first_seen: seen,
            }
        });
        e.score += 1.0 / (k + rank) as f32;
        e.in_bm25 = true;
    }

    let mut fused: Vec<(RetrievalHit, usize)> = acc
        .into_iter()
        .map(|(id, a)| {
            let source = match (a.in_dense, a.in_bm25) {
                (true, true) => HitSource::Both,
                (true, false) => HitSource::Dense,
                (false, true) => HitSource::Bm25,
                (false, false) => HitSource::Dense,
            };
            (
                RetrievalHit {
                    chunk_id: id.to_string(),
                    score: a.score,
                    source,
                },
                a.first_seen,
            )
        })
        .collect();

    // Sort by fused score desc, then by first-seen asc for determinism on ties.
    fused.sort_by(|a, b| {
        b.0.score
            .partial_cmp(&a.0.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    fused.truncate(limit);
    fused.into_iter().map(|(hit, _)| hit).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn item_in_both_lists_outranks_singletons() {
        // "b" is rank-1 in dense and rank-0 in bm25 → appears in both, so its
        // fused score exceeds any item present in only one list.
        let dense = ids(&["a", "b", "c"]);
        let bm25 = ids(&["b", "d", "e"]);
        let out = rrf_merge(&dense, &bm25, RRF_K, 10);
        assert_eq!(out[0].chunk_id, "b");
        assert_eq!(out[0].source, HitSource::Both);
        let b_score = out[0].score;
        for h in &out[1..] {
            assert!(
                h.score < b_score,
                "{} ({}) should score below b ({b_score})",
                h.chunk_id,
                h.score
            );
        }
    }

    #[test]
    fn rrf_uses_k_60_scoring() {
        // Single-list rank-0 item scores exactly 1/(60+0).
        let dense = ids(&["x"]);
        let bm25: Vec<String> = Vec::new();
        let out = rrf_merge(&dense, &bm25, RRF_K, 10);
        assert_eq!(out.len(), 1);
        assert!((out[0].score - 1.0 / 60.0).abs() < 1e-6);
        assert_eq!(out[0].source, HitSource::Dense);
    }

    #[test]
    fn ranks_preserve_order_within_a_single_list() {
        let dense = ids(&["a", "b", "c"]);
        let bm25: Vec<String> = Vec::new();
        let out = rrf_merge(&dense, &bm25, RRF_K, 10);
        assert_eq!(
            out.iter().map(|h| h.chunk_id.as_str()).collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
    }

    #[test]
    fn truncates_to_limit() {
        let dense = ids(&["a", "b", "c", "d", "e"]);
        let bm25: Vec<String> = Vec::new();
        let out = rrf_merge(&dense, &bm25, RRF_K, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].chunk_id, "a");
        assert_eq!(out[1].chunk_id, "b");
    }

    #[test]
    fn tie_breaks_deterministically_by_first_seen() {
        // Two items each at the same rank in one list have equal scores; the one
        // encountered first (dense before bm25, then in-list order) wins the tie.
        let dense = ids(&["a"]);
        let bm25 = ids(&["z"]);
        let out = rrf_merge(&dense, &bm25, RRF_K, 10);
        assert_eq!(out[0].chunk_id, "a", "dense rank-0 seen before bm25 rank-0");
        assert_eq!(out[1].chunk_id, "z");
        assert!((out[0].score - out[1].score).abs() < 1e-6);
    }

    #[test]
    fn empty_lists_yield_empty() {
        let out = rrf_merge(&[], &[], RRF_K, 10);
        assert!(out.is_empty());
    }
}
