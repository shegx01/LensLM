//! The Step-5 re-embed new-table-flip (the keystone of the enrichment pass).
//!
//! After Step 4 has written the contextual `embedding_text` columns and left the
//! source in the `enriching` handoff state, this module:
//!
//! 1. inserts the doc-summary RAPTOR node (AC6) — `kind="summary"`, `level=2`,
//!    `parent_id=NULL`, `source_id` SET — as a single `INSERT`;
//! 2. SEEDS the gen-suffixed `building` Lance table with every OTHER source's
//!    current vectors copied from the active table (so the flip — which promotes
//!    the building table to be the notebook's WHOLE active table — preserves them),
//!    then re-embeds EVERY chunk of THIS source from `COALESCE(embedding_text,
//!    text)` PLUS the summary node into it (lock-free — search resolves
//!    `status='active'` only, so a half-populated building table is unobservable;
//!    only the embedder `Mutex` serializes the populate);
//! 3. acquires `ingest_lock` for the FLIP-ONLY window (concurrency synthesis): the
//!    ONE SQLite flip txn (`active→stale`, `building→active`) + the stale Lance
//!    drop (sub-second);
//! 4. marks the source `enriched`.
//!
//! ## Crash-safety (AC7, pre-mortem 1)
//!
//! The `active` Lance table is NEVER mutated. The other sources' copied vectors and
//! this source's enriched vectors accumulate in the building table; the swap is one
//! SQLite txn. ANY failure BEFORE the flip txn
//! leaves the raw vectors untouched (the source degrades to `failed`); the orphan
//! `building` row + table are reclaimed by the Step-3 startup-GC. A crash AFTER the
//! flip-txn commit but BEFORE the stale Lance drop leaves a `stale` row + orphan
//! table that the startup-GC also reclaims (idempotently — a missing table is a
//! no-op). The `active` row always points at a COMPLETE table at every boundary.

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

    // Resolve the OWNING notebook's embedding coordinate (R1) so this same-model
    // re-embed builds, seeds, populates, and flips the notebook's OWN coordinate
    // (model + dim) rather than the global default. This per-source enrichment
    // re-embed always stays on whatever model the notebook is currently configured
    // with; the explicit model/backend SWITCH (which retires the old coordinate
    // and builds a new one) is handled separately by `reembed_notebook`.
    let (embed_model, embed_dim, embed_backend) = engine
        .resolve_notebook_embedding(&crate::NotebookId::from(notebook.to_string()))
        .await?;
    // Full backend-aware coordinate (M4 Phase 4b-B): every build/seed/flip below
    // threads the notebook's OWN backend. The cross-backend switch logic lives in
    // `reembed_notebook`; here the source re-embeds under whatever backend it is
    // currently configured with.
    let coord = crate::vector_store::Coordinate::new(
        notebook.to_string(),
        embed_backend,
        embed_model.clone(),
        embed_dim,
    );

    // ── (3) Embed into a PRIVATE building table — LOCK-FREE (search reads only the
    // active table). The embedder `Mutex` is the only serialization point.
    let embedder = engine.embedder_for(&embed_model, embed_backend).await?;
    let data_dir = engine.data_dir().await;
    let store = LanceVectorStore::new(&data_dir, pool.clone());

    let building_name = store.create_building_table(&coord).await?;

    // Seed the building table with every OTHER source's current vectors so this
    // per-source flip PRESERVES them. The flip promotes the building table to be
    // the notebook's whole active table; populating it with only THIS source's
    // chunks would wipe every other source from search (the Phase-3 multi-source
    // data-loss bug). A no-op for a notebook's first source (no active table yet).
    //
    // KNOWN NARROW RACE (tracked for M4 Phase 4b reindex): the seed runs lock-free
    // (like the populate below). If a DIFFERENT source is purged in the window
    // between this seed and the flip, that source's just-copied vectors survive the
    // flip as phantom hits (dangling chunk_ids) until the next enrichment re-seeds.
    // This is strictly better than the data loss it replaces and self-heals on the
    // next flip; closing it fully (re-verify seeded sources under `ingest_lock`, or
    // drop purged source_ids from the building table pre-flip) lands with 4b, which
    // reworks this orchestration — kept out of this hotfix to preserve the
    // sub-second lock window (the seed copy is unbounded in size).
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

    // Test-only seam: pause here (after the lock-free populate, before the flip
    // window) so a fix #2 test can run a `purge_source` to completion — modeling
    // the sequential purge-then-flip race the in-lock re-check below closes. A
    // no-op in production (no gate installed) and compiled out without `test-util`.
    #[cfg(feature = "test-util")]
    engine.reembed_preflip_gate().await;

    // ── (4) FLIP-ONLY lock window (concurrency synthesis): hold `ingest_lock` ONLY
    // for the atomic flip txn + stale Lance drop (sub-second). The long populate
    // above ran lock-free.
    {
        let _permit = engine
            .ingest_lock()
            .acquire()
            .await
            .map_err(|e| LensError::Internal(format!("ingest semaphore closed: {e}")))?;

        // Purge-vs-flip race guard. `purge_source` deletes the source row + drops
        // its vectors from the ACTIVE table under this SAME `ingest_lock`. The
        // long lock-free populate above could have raced a purge that fully
        // COMPLETED (and released the lock) before we acquired it — in which case
        // the building table still holds vectors for a now-deleted source, and
        // promoting it would resurrect dangling `chunk_id`s into the active index
        // (search returns hits no source backs). Re-check the source still exists
        // NOW, while holding the lock: because the re-check and the flip are both
        // under the held lock (and purge also takes it), no purge can interleave
        // between them. If the source is gone, SKIP the flip entirely and leave the
        // building table for startup-GC to reclaim — never promote it.
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

    // ── (5) Terminal: enriched. (If a crash hit between the flip commit and here,
    // the source stays `enriching` and the next restart's recovery resets it to
    // `pending`; the cache-key short-circuit then makes the re-run cheap.)
    repo.update_enrichment_status(source_id, EnrichmentStatus::Enriched)
        .await?;
    Ok(())
}

/// Outcome of a notebook-wide model-switch re-embed ([`reembed_notebook`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReembedOutcome {
    /// The notebook is already embedded under its configured coordinate (or has no
    /// existing vectors to migrate) — nothing was changed.
    NoOp,
    /// Re-embedded every chunk into the new `(model, dim)` coordinate, flipped it
    /// active, and retired `retired` previous coordinate(s).
    Switched {
        /// The now-active model id.
        model: String,
        /// The now-active embedding dimension.
        dim: usize,
        /// How many OLD coordinates were retired.
        retired: usize,
    },
    /// The notebook's configured model changed AGAIN while the new coordinate was
    /// building; the flip was skipped (building table left for the startup-GC) so
    /// the newest model's own re-embed wins (R2 coordinate re-check guard).
    RaceAborted,
}

/// Re-embeds EVERY chunk of `notebook_id` into the notebook's currently-configured
/// embedding coordinate, flips it active, and retires the previous coordinate(s)
/// (M4 Phase 4b, Step 9 — the model-switch re-embed).
///
/// Background-safe and crash-safe, reusing the Phase-3 flip machinery: the (long)
/// embed + populate of the NEW coordinate's building table runs LOCK-FREE (search
/// resolves `status='active'` only, so a half-built coordinate is unobservable);
/// only the atomic flip + the old-coordinate retirement run under `ingest_lock`.
/// The OLD index keeps serving search until the flip.
///
/// Returns [`ReembedOutcome::NoOp`] when the configured model already matches the
/// active coordinate (or there is nothing to migrate), and
/// [`ReembedOutcome::RaceAborted`] when a second model switch landed mid-build (R2).
///
/// `on_progress(done, total)` is called after each populated batch (the Tauri
/// command streams it as `ReembedProgress`); pass a no-op for headless callers.
pub(crate) async fn reembed_notebook(
    engine: &LensEngine,
    notebook_id: &crate::NotebookId,
    mut on_progress: impl FnMut(usize, usize) + Send,
) -> Result<ReembedOutcome, LensError> {
    let pool = engine.pool().await;
    let repo = NotebookRepo::new(&pool);
    let nb = notebook_id.to_string();

    // The NEW configured coordinate (notebooks.embedding_model + embedding_backend
    // → registry, R1 + R2). The backend is a FIRST-CLASS axis here: a same-(model,
    // dim) BACKEND switch (fastembed/nomic/768 → ollama/nomic/768) is a genuine
    // coordinate change, so the OLD-coordinate discovery + NoOp filter below compare
    // the FULL `(model, dim, backend)` triple.
    let (new_model, new_dim, new_backend) = engine.resolve_notebook_embedding(notebook_id).await?;
    let new_coord =
        crate::vector_store::Coordinate::new(nb.clone(), new_backend, new_model.clone(), new_dim);

    // The OLD active coordinate(s) currently backing the notebook's search, minus
    // any that already equal the configured coordinate (the user re-picked the same
    // model AND backend). What remains are coordinates to migrate away from + retire.
    // R2: discover `backend` too, so a same-(model, dim) backend switch is NOT
    // wrongly collapsed into a NoOp.
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
        // Nothing embedded yet, or the active coordinate already matches the
        // configured model/dim — no re-embed needed.
        return Ok(ReembedOutcome::NoOp);
    }

    // Gather every chunk across the notebook's sources, tagged with its owning
    // source_id (per-source `list_chunks_for_reembed` carries id/level/embed_text;
    // the source_id comes from the iteration). COALESCE(embedding_text, text) means
    // enriched chunks re-embed from their contextual text.
    let sources = repo.list_sources(notebook_id).await?;
    let mut items: Vec<(String, ReembedChunk)> = Vec::new();
    for src in &sources {
        for chunk in repo.list_chunks_for_reembed(&src.id).await? {
            items.push((src.id.clone(), chunk));
        }
    }
    if items.is_empty() {
        // Old coordinate(s) registered but no chunks to embed — leave the existing
        // index untouched rather than promote an empty active table.
        return Ok(ReembedOutcome::NoOp);
    }
    let total = items.len();

    // Build + populate the NEW coordinate's building table LOCK-FREE. No seed copy:
    // the new coordinate has a different (model, dim), so every chunk is embedded
    // fresh by the new model's embedder (the cross-coordinate analogue of the
    // same-coordinate seed in `reembed_and_flip`).
    let embedder = engine.embedder_for(&new_model, new_backend).await?;
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

    // Test-only seam: park after the lock-free populate, before the flip window, so
    // an R2 test can change the notebook's configured model AGAIN and exercise the
    // in-lock coordinate re-check below. No-op in production / without `test-util`.
    #[cfg(feature = "test-util")]
    engine.reembed_preflip_gate().await;

    // FLIP-ONLY lock window: hold `ingest_lock` ONLY for the atomic flip txn + the
    // old-coordinate retirement (sub-second). The long populate above ran lock-free.
    {
        let _permit = engine
            .ingest_lock()
            .acquire()
            .await
            .map_err(|e| LensError::Internal(format!("ingest semaphore closed: {e}")))?;

        // ── R2 coordinate re-check guard. The long lock-free populate could have
        // raced a SECOND model switch that landed (and committed the new
        // `notebooks.embedding_model`) before we acquired the lock. Promoting our
        // building table would then make a STALE target coordinate active. Re-resolve
        // the configured coordinate NOW, under the held lock: if it no longer matches
        // what we built, SKIP the flip and leave the building table for the
        // startup-GC — the newer switch's own re-embed wins. Mirrors the purge-vs-flip
        // guard in `reembed_and_flip`.
        // R2: the guard compares `backend` too — a same-(model, dim) backend switch
        // landing under the flip must abort (otherwise a stale-backend building
        // table flips active).
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

    // Retire each OLD coordinate now that the NEW one is active. Each call is
    // idempotent + crash-recoverable (R3): a crash mid-retire leaves a `stale` row
    // the startup-GC reclaims, and the new coordinate already serves search.
    for (old_model, old_dim, old_backend) in &old_coords {
        // R2: each old coordinate carries its OWN backend (discovered above), so a
        // same-(model, dim) backend switch retires the CORRECT old coordinate
        // (fastembed) rather than the new one.
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
