//! The background cross-document resolution worker (#155, Step 4).
//!
//! Runs on its OWN channel, separate from enrichment: `process_job` fires a
//! [`ResolveNotebook`] only after a job fully succeeds (past the re-embed flip), and
//! the worker drains all pending messages, dedupes by notebook, and runs one holistic
//! [`resolve_one`] pass per distinct notebook. The pass owns a FRESH [`SessionBudget`]
//! (never touches the enrichment budget) and serializes against the enrichment writer
//! via the engine's per-notebook lock. The write is a single SQLite transaction.

use std::collections::HashSet;

use tokio::sync::mpsc;

use super::{
    RESOLUTION_MAX_CALLS_PER_NOTEBOOK, RESOLUTION_MAX_TOKENS_PER_NOTEBOOK, ResolveInput,
    SqliteAdjudicationCache, embedding_text, resolve_notebook,
};
use crate::enrichment::meta::{Budget, SessionBudget};
use crate::notebooks::{NotebookId, NotebookRepo};
use crate::vector_store::{Coordinate, EntityVectorRow, LanceVectorStore, VectorStore};
use crate::{LensEngine, LensError};

/// Resolution prompt/version tag stamped on every node a pass processes and keyed into
/// the adjudication cache. SEPARATE from `ENRICHMENT_PROMPT_VERSION`: bumping it
/// invalidates cached verdicts and forces a full re-resolve on the next pass.
pub const RESOLUTION_PROMPT_VERSION: &str = "res-v1";

/// A request to (re-)resolve a notebook's entity graph. Coalesced by `notebook_id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolveNotebook {
    pub notebook_id: String,
}

/// Spawns the dedicated background resolution worker task.
///
/// Drain-based coalescing: on each wake it takes the first message, then drains every
/// other currently-pending message, dedupes the notebook ids, and runs exactly one
/// [`resolve_one`] pass per distinct notebook — so a burst of N single-notebook triggers
/// (a multi-source import) collapses to one pass.
pub fn spawn_resolution_worker(engine: LensEngine, mut rx: mpsc::Receiver<ResolveNotebook>) {
    tokio::spawn(async move {
        tracing::debug!("resolution worker started");
        while let Some(first) = rx.recv().await {
            let mut set: HashSet<String> = HashSet::new();
            set.insert(first.notebook_id);
            while let Ok(m) = rx.try_recv() {
                set.insert(m.notebook_id);
            }
            for notebook_id in set {
                if let Err(e) = resolve_one(&engine, &notebook_id).await {
                    tracing::warn!(
                        notebook_id = %notebook_id,
                        "resolution pass failed: {e}"
                    );
                }
            }
        }
        tracing::debug!("resolution worker stopped (channel closed)");
    });
}

/// Runs one full-notebook resolution pass under the per-notebook lock (serialized
/// against the enrichment writer). Loads all nodes, (re)builds their entity vectors
/// into the coordinate's `ent__` table, runs the 3-tier cascade with a fresh budget,
/// and writes the canonical assignments + version stamp in one transaction.
pub(crate) async fn resolve_one(engine: &LensEngine, notebook_id: &str) -> Result<(), LensError> {
    #[cfg(feature = "test-util")]
    engine.note_resolution_pass();

    // Hold the per-notebook lock for the WHOLE pass so an enrichment write on the same
    // notebook cannot interleave with (and clobber) the resolution write, or vice versa.
    let lock = engine.notebook_lock(notebook_id);
    let _guard = lock.lock().await;

    let pool = engine.pool().await;
    let repo = NotebookRepo::new(&pool);

    // Skip when the notebook has no active embedding coordinate yet (nothing ingested /
    // embedded); resolution needs the same coordinate as chunk vectors.
    let has_active: Option<(i64,)> = sqlx::query_as(
        "SELECT 1 FROM embedding_index WHERE notebook_id = ? AND status = 'active' LIMIT 1",
    )
    .bind(notebook_id)
    .fetch_optional(&pool)
    .await?;
    if has_active.is_none() {
        tracing::debug!(
            notebook_id,
            "resolution: no active embedding coordinate, skipping"
        );
        return Ok(());
    }

    let (embed_model, embed_dim, embed_backend) = engine
        .resolve_notebook_embedding(&NotebookId::from(notebook_id.to_string()))
        .await?;
    let coord = Coordinate::new(
        notebook_id.to_string(),
        embed_backend,
        embed_model.clone(),
        embed_dim,
    );

    let nodes = repo.list_entity_nodes(notebook_id).await?;
    if nodes.len() < 2 {
        tracing::debug!(
            notebook_id,
            node_count = nodes.len(),
            "resolution: fewer than 2 nodes, nothing to resolve"
        );
        return Ok(());
    }

    // Embed each node's `embedding_text`. The embedder is SYNC — run it on the blocking
    // pool so it never stalls the async runtime. `embed_documents_owned` returns
    // L2-normalized vectors (cosine-ready).
    let embedder = engine
        .embedder_for(
            &embed_model,
            embed_backend,
            crate::embedder::WorkloadKind::Bulk,
        )
        .await?;
    let texts: Vec<String> = nodes.iter().map(embedding_text).collect();
    let embed_texts = texts.clone();
    let embedder_clone = embedder.clone();
    let vectors =
        tokio::task::spawn_blocking(move || embedder_clone.embed_documents_owned(embed_texts))
            .await
            .map_err(|e| LensError::Model(format!("resolution embed task panicked: {e}")))??;
    if vectors.len() != nodes.len() {
        return Err(LensError::Model(format!(
            "resolution embedder returned {} vectors for {} nodes",
            vectors.len(),
            nodes.len()
        )));
    }

    let data_dir = engine.data_dir().await;
    let store = LanceVectorStore::new(&data_dir, pool.clone());
    let mut vector_by_id = std::collections::HashMap::with_capacity(nodes.len());
    let mut rows: Vec<EntityVectorRow> = Vec::with_capacity(nodes.len());
    for (node, vector) in nodes.iter().zip(vectors.into_iter()) {
        rows.push(EntityVectorRow {
            entity_node_id: node.id.clone(),
            source_id: node.source_id.clone(),
            notebook_id: notebook_id.to_string(),
            kind: node.kind.as_str().to_string(),
            vector: vector.clone(),
        });
        vector_by_id.insert(node.id.clone(), vector);
    }
    store.upsert_entity_vectors(&coord, rows).await?;

    // `None` provider degrades the cascade to Tiers 1-2 (never fails the pass).
    let provider = engine.llm_provider().await;
    let cache = SqliteAdjudicationCache { pool: &pool };

    // Fresh, ISOLATED budget: a separate `SessionBudget` so a resolution pass never
    // decrements the enrichment worker's session counters.
    let session = SessionBudget::new();
    let mut budget = Budget::with_caps(
        session,
        RESOLUTION_MAX_TOKENS_PER_NOTEBOOK,
        RESOLUTION_MAX_CALLS_PER_NOTEBOOK,
    );

    // #155: cross-source coref-pair seeding is a future refinement; Tier-1 normalized
    // exact-name covers same-name cross-source today.
    let coref_pairs: Vec<(String, String)> = Vec::new();

    let input = ResolveInput {
        nodes: &nodes,
        vectors: &vector_by_id,
        store: &store,
        coord: &coord,
        provider: provider.as_deref(),
        cache: &cache,
        prompt_version: RESOLUTION_PROMPT_VERSION,
        coref_pairs: &coref_pairs,
        notebook_id,
    };
    let updates = resolve_notebook(input, &mut budget).await?;

    repo.write_resolution_updates(
        notebook_id,
        RESOLUTION_PROMPT_VERSION,
        &updates,
        engine.resolution_write_fault_armed(),
    )
    .await?;

    cache
        .gc_stale(notebook_id, RESOLUTION_PROMPT_VERSION)
        .await?;

    tracing::debug!(
        notebook_id,
        nodes = nodes.len(),
        resolved = updates.len(),
        calls = budget.job_calls(),
        "resolution: pass complete"
    );
    Ok(())
}
