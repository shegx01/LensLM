//! Step-5 re-embed new-table-flip (AC6–AC8).
//!
//! After Step 4 writes `embedding_text` columns: (1) inserts the summary RAPTOR node
//! (AC6); (2) seeds a private gen-suffixed `building` table with other sources' vectors
//! then populates it with this source's new embeddings (lock-free); (3) acquires
//! `ingest_lock` for the atomic flip txn only; (4) marks `enriched`.
//!
//! Crash-safety: the `active` table is never mutated. Any failure before the flip
//! leaves raw vectors untouched. Orphan `building` rows/tables are reclaimed by startup-GC.

use crate::notebooks::{EnrichmentStatus, NotebookRepo, ReembedChunk};
use crate::vector_store::{LanceVectorStore, VectorRow, VectorStore};
use crate::{LensEngine, LensError};

const REEMBED_BATCH: usize = 32;

/// Re-embeds an `enriching` source's contextual text (+ summary node) into a new
/// gen-suffixed Lance table and flips it active (Step 5). On any error before the
/// flip the caller degrades to `failed`; only crash-safe SQLite/Lance steps are used.
pub(crate) async fn reembed_and_flip(
    engine: &LensEngine,
    source_id: &str,
    notebook: &str,
    doc_summary: &str,
) -> Result<(), LensError> {
    let pool = engine.pool().await;
    let repo = NotebookRepo::new(&pool);

    // AC6: insert summary RAPTOR node only when the map produced a non-empty summary.
    if !doc_summary.trim().is_empty() {
        repo.insert_summary_chunk(source_id, doc_summary).await?;
    }

    let chunks: Vec<ReembedChunk> = repo.list_chunks_for_reembed(source_id).await?;
    if chunks.is_empty() {
        // Empty source: leave the raw index as-is and mark enriched. No flip needed.
        repo.update_enrichment_status(source_id, EnrichmentStatus::Enriched)
            .await?;
        return Ok(());
    }

    // Use the notebook's own embedding coordinate (R1), not the global default.
    // Cross-backend or model switches are handled by `reembed_notebook`.
    let (embed_model, embed_dim, embed_backend) = engine
        .resolve_notebook_embedding(&crate::NotebookId::from(notebook.to_string()))
        .await?;
    let coord = crate::vector_store::Coordinate::new(
        notebook.to_string(),
        embed_backend,
        embed_model.clone(),
        embed_dim,
    );

    // Embed into a private building table — lock-free (search reads only the active table).
    let embedder = engine
        .embedder_for(
            &embed_model,
            embed_backend,
            crate::embedder::WorkloadKind::Bulk,
        )
        .await?;
    let data_dir = engine.data_dir().await;
    let store = LanceVectorStore::new(&data_dir, pool.clone());

    let building_name = store.create_building_table(&coord).await?;

    // Seed with every OTHER source's current vectors so the flip preserves them.
    // A no-op for a notebook's first source (no active table yet).
    // KNOWN NARROW RACE (tracked for M4 Phase 4b reindex): a source purged between
    // this seed and the flip leaves phantom vectors until the next re-seed. This is
    // strictly better than data loss and self-heals on the next flip; full closure
    // lands with 4b to keep the flip window sub-second.
    store
        .seed_building_from_active(&coord, &building_name, source_id)
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
        store.add_to_table(&building_name, rows, embed_dim).await?;
    }

    #[cfg(feature = "test-util")]
    engine.reembed_preflip_gate().await;

    // FLIP-ONLY lock window: hold `ingest_lock` only for the atomic flip txn (sub-second).
    {
        let _permit = engine
            .ingest_lock()
            .acquire()
            .await
            .map_err(|e| LensError::Internal(format!("ingest semaphore closed: {e}")))?;

        // Purge-vs-flip race guard: a completed purge before we acquired the lock
        // would leave vectors for a deleted source in the building table. Re-check
        // under the held lock; skip the flip if the source is gone.
        if repo.get_source(source_id).await?.is_none() {
            tracing::debug!(
                source_id,
                building = %building_name,
                "enrichment: source purged before flip; skipping flip, leaving building table for GC"
            );
            return Ok(());
        }

        store.flip_active(&coord, &building_name).await?;
    }

    // A crash between the flip commit and here leaves the source `enriching`; the
    // next restart resets it to `pending` and the cache-key short-circuit makes the
    // re-run cheap.
    repo.update_enrichment_status(source_id, EnrichmentStatus::Enriched)
        .await?;
    Ok(())
}

/// Outcome of a notebook-wide model-switch re-embed ([`reembed_notebook`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReembedOutcome {
    /// Already on the configured coordinate (or nothing to migrate).
    NoOp,
    /// Re-embedded into the new coordinate, flipped active, retired old coordinate(s).
    Switched {
        model: String,
        dim: usize,
        /// How many old coordinates were retired.
        retired: usize,
    },
    /// Configured model changed again during the build; flip skipped so the newer
    /// re-embed wins (R2 coordinate re-check guard).
    RaceAborted,
}

/// Re-embeds every chunk of `notebook_id` under the configured coordinate, flips it
/// active, and retires old coordinate(s) (Step 9 — model-switch re-embed).
///
/// The long populate runs lock-free; only the atomic flip + retirement run under
/// `ingest_lock`. Returns `NoOp` when already on the configured coordinate; returns
/// `RaceAborted` when a second model switch landed mid-build (R2).
/// `on_progress(done, total)` is called after each batch for progress streaming.
pub(crate) async fn reembed_notebook(
    engine: &LensEngine,
    notebook_id: &crate::NotebookId,
    mut on_progress: impl FnMut(usize, usize) + Send,
) -> Result<ReembedOutcome, LensError> {
    let pool = engine.pool().await;
    let repo = NotebookRepo::new(&pool);
    let nb = notebook_id.to_string();

    // R1 + R2: backend is a first-class axis — a same-(model, dim) backend switch is
    // a genuine coordinate change; the NoOp filter compares the full (model, dim, backend) triple.
    let (new_model, new_dim, new_backend) = engine.resolve_notebook_embedding(notebook_id).await?;
    let new_coord =
        crate::vector_store::Coordinate::new(nb.clone(), new_backend, new_model.clone(), new_dim);

    // Discover active coordinates excluding the new one (R2: backend included so a
    // same-(model, dim) backend switch is not wrongly collapsed to NoOp).
    let active: Vec<(String, i64, String)> = sqlx::query_as(
        "SELECT DISTINCT model, dim, backend FROM embedding_index \
         WHERE notebook_id = ? AND status = 'active'",
    )
    .bind(&nb)
    .fetch_all(&pool)
    .await?;
    let old_coords: Vec<(String, usize, crate::embedder::EmbeddingBackend)> = active
        .into_iter()
        .map(|(m, d, b)| {
            (
                m,
                d as usize,
                crate::embedder::EmbeddingBackend::from_opt_str(Some(&b)),
            )
        })
        .filter(|(m, d, b)| !(m == &new_model && *d == new_dim && *b == new_backend))
        .collect();
    if old_coords.is_empty() {
        return Ok(ReembedOutcome::NoOp);
    }

    let sources = repo.list_sources(notebook_id).await?;
    let mut items: Vec<(String, ReembedChunk)> = Vec::new();
    for src in &sources {
        for chunk in repo.list_chunks_for_reembed(&src.id).await? {
            items.push((src.id.clone(), chunk));
        }
    }
    if items.is_empty() {
        // Old coordinates exist but no chunks — leave the existing index untouched.
        return Ok(ReembedOutcome::NoOp);
    }
    let total = items.len();

    // Populate the new coordinate's building table lock-free. No seed: the new
    // coordinate has a different (model, dim), so every chunk is embedded fresh.
    let embedder = engine
        .embedder_for(&new_model, new_backend, crate::embedder::WorkloadKind::Bulk)
        .await?;
    let data_dir = engine.data_dir().await;
    let store = LanceVectorStore::new(&data_dir, pool.clone());
    let building_name = store.create_building_table(&new_coord).await?;

    let mut done = 0usize;
    for batch in items.chunks(REEMBED_BATCH) {
        let texts: Vec<String> = batch.iter().map(|(_, c)| c.embed_text.clone()).collect();
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
            .map(|((source_id, chunk), vector)| VectorRow {
                chunk_id: chunk.id.clone(),
                source_id: source_id.clone(),
                notebook_id: nb.clone(),
                level: chunk.level,
                vector,
            })
            .collect();
        store.add_to_table(&building_name, rows, new_dim).await?;
        done += batch.len();
        on_progress(done, total);
    }

    #[cfg(feature = "test-util")]
    engine.reembed_preflip_gate().await;

    // FLIP-ONLY lock window (sub-second): atomic flip txn + old-coordinate retirement.
    {
        let _permit = engine
            .ingest_lock()
            .acquire()
            .await
            .map_err(|e| LensError::Internal(format!("ingest semaphore closed: {e}")))?;

        // R2 coordinate re-check: a second model switch that committed before we
        // acquired the lock would make a stale coordinate active. Re-resolve under
        // the held lock; skip the flip if it no longer matches what we built.
        // Backend is included so a same-(model, dim) backend switch also aborts.
        let (configured_model, configured_dim, configured_backend) =
            engine.resolve_notebook_embedding(notebook_id).await?;
        if configured_model != new_model
            || configured_dim != new_dim
            || configured_backend != new_backend
        {
            tracing::debug!(
                notebook = %nb,
                building = %building_name,
                built = %format!("{}/{new_model}/{new_dim}", new_backend.as_str()),
                now = %format!("{}/{configured_model}/{configured_dim}", configured_backend.as_str()),
                "reembed_notebook: configured coordinate changed under flip; skipping flip, leaving building table for GC"
            );
            return Ok(ReembedOutcome::RaceAborted);
        }

        store.flip_active(&new_coord, &building_name).await?;
    }

    // Retire old coordinates (R3, idempotent): a crash mid-retire leaves a `stale`
    // row the startup-GC reclaims; the new coordinate already serves search.
    for (old_model, old_dim, old_backend) in &old_coords {
        let old_coord = crate::vector_store::Coordinate::new(
            nb.clone(),
            *old_backend,
            old_model.clone(),
            *old_dim,
        );
        store.retire_coordinate(&old_coord).await?;
    }

    Ok(ReembedOutcome::Switched {
        model: new_model,
        dim: new_dim,
        retired: old_coords.len(),
    })
}
