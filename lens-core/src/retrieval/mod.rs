//! Hybrid retrieval primitive (issue #39): fuses dense LanceDB vector search with
//! SQLite FTS5 BM25 lexical search via Reciprocal Rank Fusion (RRF, k=60), with an
//! opt-in cross-encoder reranker over the fused top candidates. Delivered as the
//! seam the Tiered Context Router (#21) consumes — it does NOT embed the query
//! (that is #21's job) and does not do router/chat/LLM work.
//!
//! Both retrieval paths restrict to live, SELECTED sources (`trashed_at IS NULL
//! AND selected = 1` — the "retrieval only from selected sources" contract): BM25
//! via its `sources` JOIN; DENSE via a SQLite post-filter, because
//! `LanceVectorStore::search` scopes only by `notebook_id`.

pub mod bm25;
pub mod rerank;
pub mod router;
pub mod rrf;

use std::collections::HashMap;

use sqlx::SqlitePool;

use crate::LensError;
use crate::config::RetrievalConfig;
use crate::vector_store::{Coordinate, VectorStore};

pub use rerank::Reranker;
pub use rrf::RRF_K;

/// Default number of candidates over-fetched from each path before fusion. The
/// fused top of this pool also feeds the reranker; the final result is truncated
/// to the caller's requested `pool` (k).
pub const OVERFETCH: usize = 50;

/// Upper bound on the per-path fetch, so a large caller `pool` can't drive an
/// unbounded dense fetch + per-candidate hydrate (defensive; the router #21 caller
/// is expected to pass a small k).
pub const MAX_OVERFETCH: usize = 500;

/// Which retrieval path(s) surfaced a fused hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitSource {
    /// Dense (vector) search only.
    Dense,
    /// BM25 (lexical) search only.
    Bm25,
    /// Both dense and BM25 paths.
    Both,
    /// Surfaced only by the entity-graph arm (#21); present in neither dense nor BM25.
    Graph,
}

/// A fused retrieval hit. Distinct from [`crate::vector_store::Hit`] whose
/// `distance` is a cosine distance — this `score` is a fused RRF (or reranker)
/// score where HIGHER is better.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalHit {
    /// The matched `chunks.id`.
    pub chunk_id: String,
    /// Fused RRF score (or reranker score when reranking ran); higher is better.
    pub score: f32,
    /// Which path(s) contributed the hit.
    pub source: HitSource,
}

/// Runs hybrid retrieval for a pre-embedded query within a notebook-scoped
/// coordinate: dense (`store.search`) + BM25 (`chunks_fts`), RRF-merged (k=60),
/// then optionally reranked. Both paths exclude trashed sources. `pool` is the
/// number of hits to return. When `config.hybrid_enabled` is false it degrades to
/// dense-only (still trashed-filtered).
///
/// The caller supplies BOTH `query_text` (for BM25) and `query_vec` (for dense);
/// this primitive never embeds the query.
#[allow(clippy::too_many_arguments)]
pub async fn hybrid_search(
    pool_db: &SqlitePool,
    store: &dyn VectorStore,
    reranker: &Reranker,
    coord: &Coordinate,
    query_text: &str,
    query_vec: &[f32],
    source_id: Option<&str>,
    level: Option<i32>,
    pool: usize,
    config: &RetrievalConfig,
) -> Result<Vec<RetrievalHit>, LensError> {
    let overfetch = pool.clamp(OVERFETCH, MAX_OVERFETCH);

    // DENSE: search, then post-filter trashed + apply source_id/level.
    let dense_hits = store.search(coord, query_vec, overfetch).await?;
    let dense_ids: Vec<String> = dense_hits.into_iter().map(|h| h.chunk_id).collect();
    let dense_ids = live_chunk_ids(pool_db, &dense_ids, source_id, level).await?;

    // BM25: lexical path.
    let bm25_ids = if config.hybrid_enabled {
        bm25::bm25_search(
            pool_db,
            &coord.notebook,
            source_id,
            level,
            query_text,
            overfetch,
        )
        .await?
    } else {
        Vec::new()
    };

    tracing::debug!(
        dense = dense_ids.len(),
        bm25 = bm25_ids.len(),
        hybrid_enabled = config.hybrid_enabled,
        "hybrid_search: retrieved candidates"
    );

    let (hits, _texts) = fuse_and_rerank(
        pool_db,
        reranker,
        query_text,
        &dense_ids,
        &bm25_ids,
        &[],
        pool,
        config,
    )
    .await?;
    Ok(hits)
}

/// The shared fusion + rerank tail (issue #21). Both [`hybrid_search`] and the
/// tiered router (`router::tiered_search`) call this identical body, so graph-OFF
/// fusion-seam parity holds by construction: given the same dense/bm25 ids and an
/// empty `graph_ids`, the output is bitwise-identical regardless of caller. Mirrors
/// the former `hybrid_search` tail exactly: over-fetch clamp → three-list RRF →
/// non-reranker truncation → reranker hydrate+rerank.
/// Returns `(hits, texts_by_id)`. `texts_by_id` is populated ONLY on the reranker
/// path — it is the chunk text this call already hydrated to score the reranker, so
/// a downstream consumer (the router's Tier-2 assembly) can reuse it instead of
/// re-selecting the same rows (RQ-3). Empty on the non-reranker path.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn fuse_and_rerank(
    pool_db: &SqlitePool,
    reranker: &Reranker,
    query_text: &str,
    dense_ids: &[String],
    bm25_ids: &[String],
    graph_ids: &[String],
    pool: usize,
    config: &RetrievalConfig,
) -> Result<(Vec<RetrievalHit>, HashMap<String, String>), LensError> {
    let overfetch = pool.clamp(OVERFETCH, MAX_OVERFETCH);
    let fused = rrf::rrf_merge3(dense_ids, bm25_ids, graph_ids, RRF_K, overfetch);

    if !config.reranker.enabled || fused.is_empty() {
        let mut out = fused;
        out.truncate(pool);
        return Ok((out, HashMap::new()));
    }

    // Hydrate text in the SAME order as the fused list (reranker maps by index).
    let ids: Vec<String> = fused.iter().map(|h| h.chunk_id.clone()).collect();
    let texts_by_id = hydrate_texts_map(pool_db, &ids).await?;
    let texts: Vec<String> = ids
        .iter()
        .map(|id| texts_by_id.get(id).cloned().unwrap_or_default())
        .collect();
    let reranked = reranker
        .rerank_with_fallback(query_text, fused, texts, &config.reranker, pool)
        .await;
    tracing::debug!(reranked = reranked.len(), "fuse_and_rerank: reranked");
    Ok((reranked, texts_by_id))
}

/// Filters `chunk_ids` down to those whose source is live (`trashed_at IS NULL`)
/// AND selected (`selected = 1`), preserving the input order, optionally narrowing
/// by `source_id`/`level`. This is the dense-path scope filter (the vector store
/// scopes only by `notebook_id`, so trashed/deselected exclusion happens here).
pub(crate) async fn live_chunk_ids(
    pool: &SqlitePool,
    chunk_ids: &[String],
    source_id: Option<&str>,
    level: Option<i32>,
) -> Result<Vec<String>, LensError> {
    if chunk_ids.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = std::iter::repeat_n("?", chunk_ids.len())
        .collect::<Vec<_>>()
        .join(",");
    let mut sql = format!(
        "SELECT c.id FROM chunks c JOIN sources s ON s.id = c.source_id \
         WHERE c.id IN ({placeholders}) AND s.trashed_at IS NULL AND s.selected = 1"
    );
    sql.push_str(&bm25::scope_filter_sql(source_id, level));

    let mut q = sqlx::query_scalar::<_, String>(&sql);
    for id in chunk_ids {
        q = q.bind(id);
    }
    let q = bm25::bind_scope_filters(q, source_id, level);
    let live: std::collections::HashSet<String> = q.fetch_all(pool).await?.into_iter().collect();

    Ok(chunk_ids
        .iter()
        .filter(|id| live.contains(*id))
        .cloned()
        .collect())
}

/// Batched `id -> text` map over `chunk_ids`. Absent ids are simply missing (the
/// caller substitutes an empty string). Callers index by id, so order is irrelevant.
pub(crate) async fn hydrate_texts_map(
    pool: &SqlitePool,
    chunk_ids: &[String],
) -> Result<HashMap<String, String>, LensError> {
    let rows: Vec<(String, String)> = crate::db::fetch_batched(pool, chunk_ids, |ph| {
        format!("SELECT id, text FROM chunks WHERE id IN ({ph})")
    })
    .await?;
    Ok(rows.into_iter().collect())
}
