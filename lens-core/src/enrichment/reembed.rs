//! The Step-5 re-embed new-table-flip (the keystone of the enrichment pass).
//!
//! After Step 4 has written the contextual `embedding_text` columns and left the
//! source in the `enriching` handoff state, this module:
//!
//! 1. inserts the doc-summary RAPTOR node (AC6) — `kind="summary"`, `level=2`,
//!    `parent_id=NULL`, `source_id` SET — as a single `INSERT`;
//! 2. re-embeds EVERY chunk from `COALESCE(embedding_text, text)` PLUS the summary
//!    node into a PRIVATE, gen-suffixed `building` Lance table (lock-free — search
//!    resolves `status='active'` only, so a half-populated building table is
//!    unobservable; only the embedder `Mutex` serializes the populate);
//! 3. acquires `ingest_lock` for the FLIP-ONLY window (concurrency synthesis): the
//!    ONE SQLite flip txn (`active→stale`, `building→active`) + the stale Lance
//!    drop (sub-second);
//! 4. marks the source `enriched`.
//!
//! ## Crash-safety (AC7, pre-mortem 1)
//!
//! The `active` Lance table is NEVER mutated. All enriched vectors accumulate in
//! the building table; the swap is one SQLite txn. ANY failure BEFORE the flip txn
//! leaves the raw vectors untouched (the source degrades to `failed`); the orphan
//! `building` row + table are reclaimed by the Step-3 startup-GC. A crash AFTER the
//! flip-txn commit but BEFORE the stale Lance drop leaves a `stale` row + orphan
//! table that the startup-GC also reclaims (idempotently — a missing table is a
//! no-op). The `active` row always points at a COMPLETE table at every boundary.

use crate::embedder::{EMBED_DIM, EMBED_MODEL_ID};
use crate::notebooks::{EnrichmentStatus, NotebookRepo, ReembedChunk};
use crate::vector_store::{LanceVectorStore, VectorRow, VectorStore};
use crate::{LensEngine, LensError};

/// Batch size for the re-embed embedder loop (mirrors the ingest embed batch).
const REEMBED_BATCH: usize = 32;

/// Re-embeds an `enriching` source's contextual text (+ the summary node) into a
/// new gen-suffixed Lance table and flips it active (Step 5).
///
/// `notebook` is the source's owning notebook id; `doc_summary` is the structural
/// map's summary (empty for the fallback/prefix-only path — then NO summary node
/// is created). On success the source is left `enriched`. On ANY error before the
/// flip the caller is responsible for degrading the source to `failed`; this
/// function only mutates SQLite/Lance through crash-safe steps.
pub(crate) async fn reembed_and_flip(
    engine: &LensEngine,
    source_id: &str,
    notebook: &str,
    doc_summary: &str,
) -> Result<(), LensError> {
    let pool = engine.pool().await;
    let repo = NotebookRepo::new(&pool);

    // ── (1) Summary RAPTOR node (AC6). Only when the structural map produced a
    // non-empty summary; the prefix-only fallback skips it (no summary text).
    if !doc_summary.trim().is_empty() {
        repo.insert_summary_chunk(source_id, doc_summary).await?;
    }

    // ── (2) Read every chunk's embed text (COALESCE(embedding_text, text)) — this
    // now includes the freshly-inserted summary node.
    let chunks: Vec<ReembedChunk> = repo.list_chunks_for_reembed(source_id).await?;
    if chunks.is_empty() {
        // Nothing to embed (an empty source). Leave the active raw index as-is and
        // mark enriched (the text-column pass already ran). No flip needed.
        repo.update_enrichment_status(source_id, EnrichmentStatus::Enriched)
            .await?;
        return Ok(());
    }

    // ── (3) Embed into a PRIVATE building table — LOCK-FREE (search reads only the
    // active table). The embedder `Mutex` is the only serialization point.
    let embedder = engine.embedder().await?;
    let data_dir = engine.data_dir().await;
    let store = LanceVectorStore::new(&data_dir, pool.clone());

    let building_name = store
        .create_building_table(notebook, EMBED_MODEL_ID, EMBED_DIM)
        .await?;

    for batch in chunks.chunks(REEMBED_BATCH) {
        let texts: Vec<String> = batch.iter().map(|c| c.embed_text.clone()).collect();
        let embedder = embedder.clone();
        let vectors = tokio::task::spawn_blocking(move || embedder.embed_documents_owned(texts))
            .await
            .map_err(|e| LensError::Model(format!("re-embed task panicked: {e}")))??;
        if vectors.len() != batch.len() {
            return Err(LensError::Model(format!(
                "re-embed embedder returned {} vectors for {} inputs",
                vectors.len(),
                batch.len()
            )));
        }
        let rows: Vec<VectorRow> = batch
            .iter()
            .zip(vectors.into_iter())
            .map(|(chunk, vector)| VectorRow {
                chunk_id: chunk.id.clone(),
                source_id: source_id.to_string(),
                notebook_id: notebook.to_string(),
                level: chunk.level,
                vector,
            })
            .collect();
        store.add_to_table(&building_name, rows).await?;
    }

    // ── (4) FLIP-ONLY lock window (concurrency synthesis): hold `ingest_lock` ONLY
    // for the atomic flip txn + stale Lance drop (sub-second). The long populate
    // above ran lock-free.
    {
        let _permit = engine
            .ingest_lock()
            .acquire()
            .await
            .map_err(|e| LensError::Internal(format!("ingest semaphore closed: {e}")))?;
        store
            .flip_active(notebook, EMBED_MODEL_ID, EMBED_DIM, &building_name)
            .await?;
    }

    // ── (5) Terminal: enriched. (If a crash hit between the flip commit and here,
    // the source stays `enriching` and the next restart's recovery resets it to
    // `pending`; the cache-key short-circuit then makes the re-run cheap.)
    repo.update_enrichment_status(source_id, EnrichmentStatus::Enriched)
        .await?;
    Ok(())
}
