//! Hybrid retrieval primitive (issue #39): fuses dense LanceDB vector search with
//! SQLite FTS5 BM25 lexical search via Reciprocal Rank Fusion (RRF, k=60), with an
//! opt-in cross-encoder reranker over the fused top candidates. Delivered as the
//! seam the Tiered Context Router (#21) consumes — it does NOT embed the query
//! (that is #21's job) and does not do router/chat/LLM work.
//!
//! Both retrieval paths exclude trashed sources (invariant `lib.rs`): BM25 via a
//! `sources.trashed_at IS NULL` JOIN; DENSE via a SQLite post-filter because
//! `LanceVectorStore::search` scopes only by `notebook_id`.

pub mod bm25;
pub mod rerank;
pub mod rrf;

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

/// Which retrieval path(s) surfaced a fused hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitSource {
    /// Dense (vector) search only.
    Dense,
    /// BM25 (lexical) search only.
    Bm25,
    /// Both paths.
    Both,
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
    let overfetch = pool.max(OVERFETCH);

    // DENSE: search then post-filter out trashed sources (search scopes only by
    // notebook_id). Optional source_id/level narrowing is applied here too.
    let dense_hits = store.search(coord, query_vec, overfetch).await?;
    let dense_ids: Vec<String> = dense_hits.into_iter().map(|h| h.chunk_id).collect();
    let dense_ids = live_chunk_ids(pool_db, &dense_ids, source_id, level).await?;

    // BM25: notebook-scoped, trashed-excluded via the sources JOIN.
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

    let fused = rrf::rrf_merge(&dense_ids, &bm25_ids, RRF_K, overfetch);

    if !config.reranker.enabled || fused.is_empty() {
        let mut out = fused;
        out.truncate(pool);
        return Ok(out);
    }

    // Hydrate candidate text from chunks in the SAME order as the fused list, then
    // rerank; any failure falls back to the RRF order inside the reranker.
    let texts = hydrate_texts(pool_db, &fused).await?;
    let reranked = reranker
        .rerank_with_fallback(query_text, fused, texts, &config.reranker, pool)
        .await;
    tracing::debug!(reranked = reranked.len(), "hybrid_search: reranked");
    Ok(reranked)
}

/// Filters `chunk_ids` down to those whose source is live (`trashed_at IS NULL`),
/// preserving the input order, optionally narrowing by `source_id`/`level`. This
/// is the dense-path trashed exclusion (the vector store does not filter it).
async fn live_chunk_ids(
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
         WHERE c.id IN ({placeholders}) AND s.trashed_at IS NULL"
    );
    if source_id.is_some() {
        sql.push_str(" AND c.source_id = ?");
    }
    if level.is_some() {
        sql.push_str(" AND c.level = ?");
    }

    let mut q = sqlx::query_scalar::<_, String>(&sql);
    for id in chunk_ids {
        q = q.bind(id);
    }
    if let Some(sid) = source_id {
        q = q.bind(sid);
    }
    if let Some(lvl) = level {
        q = q.bind(lvl);
    }
    let live: std::collections::HashSet<String> = q.fetch_all(pool).await?.into_iter().collect();

    Ok(chunk_ids
        .iter()
        .filter(|id| live.contains(*id))
        .cloned()
        .collect())
}

/// Hydrates `chunks.text` for each hit, in the SAME order as `hits`. A chunk that
/// vanished between fusion and hydration is dropped from both lists' correspondence
/// by returning an empty string; the reranker keys results by index so alignment
/// must be preserved — we look each id up individually to guarantee ordering.
async fn hydrate_texts(pool: &SqlitePool, hits: &[RetrievalHit]) -> Result<Vec<String>, LensError> {
    let mut texts = Vec::with_capacity(hits.len());
    for h in hits {
        let text: Option<String> =
            sqlx::query_scalar::<_, String>("SELECT text FROM chunks WHERE id = ?")
                .bind(&h.chunk_id)
                .fetch_optional(pool)
                .await?;
        texts.push(text.unwrap_or_default());
    }
    Ok(texts)
}
