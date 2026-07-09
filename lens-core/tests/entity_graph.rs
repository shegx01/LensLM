//! Integration tests for the M13 entity-graph persistence seam (issue #153).
//!
//! Two lanes:
//! * The write seam (`write_enrichment_and_graph`) driven DIRECTLY with hand-built
//!   [`EntityGraphRows`] — idempotency across all 3 tables, atomicity, and the
//!   `ON DELETE CASCADE` FKs. Driving the seam directly (not via the worker)
//!   bypasses the cache-key short-circuit so a second identical write really runs.
//! * The worker end-to-end with a call-counting `LlmProvider` — the graph step adds
//!   ZERO LLM calls (`zero_llm_delta`).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use lens_core::LensEngine;
use lens_core::enrichment::test_util::ScriptedProvider;
use lens_core::graph::{
    EntityEdge, EntityGraphRows, EntityKind, EntityMention, EntityNode, Relation,
};
use lens_core::llm::LlmProvider;
use lens_core::notebooks::{ChunkEnrichmentUpdate, NotebookRepo};
use lens_core::vector_store::{Coordinate, EntityVectorRow, LanceVectorStore, VectorStore};
use lens_core::{EmbeddingBackend, NotebookId};
use sqlx::Row;
use tempfile::TempDir;

mod support;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn file_engine() -> (TempDir, LensEngine) {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = LensEngine::init(dir.path()).await.expect("engine init");
    {
        let mut cfg = engine.config().await;
        cfg.enrichment.enabled = true;
        engine.set_config(cfg).await;
    }
    engine.disable_tokenizer_for_test();
    let embedder: Arc<dyn lens_core::Embedder> = Arc::new(lens_core::CountingEmbedder::new(
        Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    ));
    engine
        .set_embedder_for_test(embedder, lens_core::EmbeddingBackend::Fastembed)
        .expect("inject embedder");
    (dir, engine)
}

/// Seeds a notebook + `indexed` source + one prose child chunk. Returns
/// `(notebook_id, source_id, chunk_id)`.
async fn seed_source_with_chunk(engine: &LensEngine) -> (String, String, String) {
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;
    let source_id = uuid::Uuid::now_v7().to_string();
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
         content_hash, enrichment_status, created_at) \
         VALUES (?, ?, 'text', 'seed', 'indexed', '/tmp/seed.txt', 1, 'h', NULL, ?)",
    )
    .bind(&source_id)
    .bind(&nb)
    .bind(&now)
    .execute(&pool)
    .await
    .expect("insert source");

    let chunk_id = format!("{source_id}-chunk-0");
    sqlx::query(
        "INSERT INTO chunks \
         (id, source_id, parent_id, kind, level, section_path, text, \
          token_start, token_end, char_start, char_end, block_type, created_at) \
         VALUES (?, ?, NULL, 'child', 1, 'Intro', 'Alice met Bob.', 0, 1, 0, 14, 'paragraph', ?)",
    )
    .bind(&chunk_id)
    .bind(&source_id)
    .bind(&now)
    .execute(&pool)
    .await
    .expect("insert chunk");
    (nb, source_id, chunk_id)
}

/// Hand-builds a small graph (2 nodes, 1 edge, 2 mentions) over a seeded source.
fn sample_rows(nb: &str, source_id: &str, chunk_id: &str) -> EntityGraphRows {
    let now = "2026-01-01T00:00:00Z".to_string();
    let node_a = EntityNode {
        id: format!("{source_id}-na"),
        notebook_id: nb.to_string(),
        source_id: source_id.to_string(),
        kind: EntityKind::Concept,
        name: "Alice".to_string(),
        canonical_name: None,
        definition: None,
        resolution_conf: None,
        resolution_prompt_version: None,
        created_at: now.clone(),
    };
    let node_b = EntityNode {
        id: format!("{source_id}-nb"),
        notebook_id: nb.to_string(),
        source_id: source_id.to_string(),
        kind: EntityKind::Concept,
        name: "Bob".to_string(),
        canonical_name: None,
        definition: None,
        resolution_conf: None,
        resolution_prompt_version: None,
        created_at: now.clone(),
    };
    let edge = EntityEdge {
        id: format!("{source_id}-e"),
        notebook_id: nb.to_string(),
        source_id: source_id.to_string(),
        chunk_id: chunk_id.to_string(),
        from_node: node_a.id.clone(),
        to_node: node_b.id.clone(),
        relation: Relation::CoOccurs,
        weight: Some(1.0),
        confidence: None,
        created_at: now.clone(),
    };
    let mentions = vec![
        EntityMention {
            id: format!("{source_id}-m0"),
            notebook_id: nb.to_string(),
            entity_node_id: node_a.id.clone(),
            chunk_id: chunk_id.to_string(),
            char_start: 0,
            char_end: 5,
            created_at: now.clone(),
        },
        EntityMention {
            id: format!("{source_id}-m1"),
            notebook_id: nb.to_string(),
            entity_node_id: node_b.id.clone(),
            chunk_id: chunk_id.to_string(),
            char_start: 9,
            char_end: 12,
            created_at: now,
        },
    ];
    EntityGraphRows {
        source_id: source_id.to_string(),
        nodes: vec![node_a, node_b],
        edges: vec![edge],
        mentions,
        dropped_cooccurrence: 0,
    }
}

async fn count(engine: &LensEngine, table: &str) -> i64 {
    let pool = engine.pool().await;
    sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {table}"))
        .fetch_one(&pool)
        .await
        .unwrap_or_else(|_| panic!("count {table}"))
}

async fn ent_store(engine: &LensEngine, nb: &str) -> (LanceVectorStore, Coordinate) {
    let pool = engine.pool().await;
    let data_dir = engine.data_dir_for_test().await;
    let (model, dim, _backend) = engine
        .resolve_notebook_embedding(&NotebookId::from(nb.to_string()))
        .await
        .expect("resolve embedding");
    let store = LanceVectorStore::new(&data_dir, pool);
    let coord = Coordinate::new(nb.to_string(), EmbeddingBackend::Fastembed, model, dim);
    (store, coord)
}

/// Marks the notebook's default coordinate `active` in `embedding_index` — `purge_source`
/// only drops vectors for coordinates it finds there.
async fn seed_active_coord(engine: &LensEngine, nb: &str) {
    let pool = engine.pool().await;
    let (model, dim, backend) = engine
        .resolve_notebook_embedding(&NotebookId::from(nb.to_string()))
        .await
        .expect("resolve embedding");
    sqlx::query(
        "INSERT INTO embedding_index \
         (id, notebook_id, model, dim, prefix_convention, lance_table_name, status, backend, created_at) \
         VALUES (?, ?, ?, ?, 'nomic', ?, 'active', ?, ?)",
    )
    .bind(uuid::Uuid::now_v7().to_string())
    .bind(nb)
    .bind(&model)
    .bind(dim as i64)
    .bind(format!("chunks__{nb}"))
    .bind(backend.as_str())
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(&pool)
    .await
    .expect("seed active coord");
}

/// Seeds one entity vector into the `ent__` table for testing drop-path assertions.
async fn seed_ent_vector(
    store: &LanceVectorStore,
    coord: &Coordinate,
    entity_node_id: &str,
    source_id: &str,
    nb: &str,
) {
    let dim = coord.dim;
    store
        .upsert_entity_vectors(
            coord,
            vec![EntityVectorRow {
                entity_node_id: entity_node_id.to_string(),
                source_id: source_id.to_string(),
                notebook_id: nb.to_string(),
                kind: "concept".to_string(),
                vector: vec![0.1; dim],
            }],
        )
        .await
        .expect("seed entity vector");
}

async fn ent_rows_for_source(store: &LanceVectorStore, coord: &Coordinate) -> usize {
    let probe = vec![0.1; coord.dim];
    store
        .entity_ann(coord, &probe, 100, None)
        .await
        .expect("entity_ann")
        .len()
}

// ---------------------------------------------------------------------------
// AC6 — idempotency across all three tables
// ---------------------------------------------------------------------------

#[tokio::test]
async fn graph_idempotency_all_tables() {
    let (_dir, engine) = file_engine().await;
    let (nb, source_id, chunk_id) = seed_source_with_chunk(&engine).await;
    let rows = sample_rows(&nb, &source_id, &chunk_id);
    let pool = engine.pool().await;
    let repo = NotebookRepo::new(&pool);

    // Write twice with identical prebuilt rows (bypassing the worker cache-key path).
    repo.write_enrichment_and_graph(&[], &rows)
        .await
        .expect("first write");
    repo.write_enrichment_and_graph(&[], &rows)
        .await
        .expect("second write");

    assert_eq!(count(&engine, "entity_nodes").await, 2, "nodes stable");
    assert_eq!(count(&engine, "entity_edges").await, 1, "edges stable");
    assert_eq!(
        count(&engine, "entity_mentions").await,
        2,
        "mentions stable"
    );
}

// ---------------------------------------------------------------------------
// #154 (AC10) — semantic + co-occurrence edges coexist for the same node pair
// under UNIQUE(source_id, from_node, to_node, relation); duplicate co_occurs ignored.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn semantic_and_cooccurrence_edges_coexist_for_same_pair() {
    let (_dir, engine) = file_engine().await;
    let (nb, source_id, chunk_id) = seed_source_with_chunk(&engine).await;
    let mut rows = sample_rows(&nb, &source_id, &chunk_id);

    // Same A-B pair as the co-occurrence edge, but a `founded` semantic edge — a
    // distinct UNIQUE slot, so both must survive.
    let from = rows.edges[0].from_node.clone();
    let to = rows.edges[0].to_node.clone();
    rows.edges.push(EntityEdge {
        id: format!("{source_id}-e-sem"),
        notebook_id: nb.clone(),
        source_id: source_id.clone(),
        chunk_id: chunk_id.clone(),
        from_node: from.clone(),
        to_node: to.clone(),
        relation: Relation::Semantic("founded".to_string()),
        weight: None,
        confidence: Some(0.9),
        created_at: "2026-01-01T00:00:00Z".to_string(),
    });
    // A duplicate co_occurs for the same pair must be silently ignored (INSERT OR IGNORE).
    rows.edges.push(EntityEdge {
        id: format!("{source_id}-e-dup"),
        notebook_id: nb.clone(),
        source_id: source_id.clone(),
        chunk_id: chunk_id.clone(),
        from_node: from,
        to_node: to,
        relation: Relation::CoOccurs,
        weight: Some(1.0),
        confidence: None,
        created_at: "2026-01-01T00:00:00Z".to_string(),
    });

    let pool = engine.pool().await;
    NotebookRepo::new(&pool)
        .write_enrichment_and_graph(&[], &rows)
        .await
        .expect("write mixed edges");

    // 2 rows survive: one co_occurs + one founded; the duplicate co_occurs is ignored.
    assert_eq!(count(&engine, "entity_edges").await, 2);
    let relations: Vec<String> = sqlx::query_scalar(
        "SELECT relation FROM entity_edges WHERE source_id = ? ORDER BY relation",
    )
    .bind(&source_id)
    .fetch_all(&pool)
    .await
    .expect("relations");
    assert_eq!(
        relations,
        vec!["co_occurs".to_string(), "founded".to_string()]
    );

    // Re-writing the same rows is idempotent (self-replacing DELETE + reinsert).
    NotebookRepo::new(&pool)
        .write_enrichment_and_graph(&[], &rows)
        .await
        .expect("second write");
    assert_eq!(
        count(&engine, "entity_edges").await,
        2,
        "re-enrichment replaces both edge types cleanly"
    );
}

// ---------------------------------------------------------------------------
// AC7 — chunk enrichment + graph rows both land from one `write_enrichment_and_graph`
// call (rows absent before, present after).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn graph_atomicity() {
    let (_dir, engine) = file_engine().await;
    let (nb, source_id, chunk_id) = seed_source_with_chunk(&engine).await;

    // Before the write: zero graph rows.
    assert_eq!(count(&engine, "entity_nodes").await, 0);
    assert_eq!(count(&engine, "entity_edges").await, 0);
    assert_eq!(count(&engine, "entity_mentions").await, 0);

    let rows = sample_rows(&nb, &source_id, &chunk_id);
    let updates = vec![ChunkEnrichmentUpdate {
        chunk_id: chunk_id.clone(),
        embedding_text: "Intro | Alice met Bob.".to_string(),
        enrichment_json: None,
    }];
    let pool = engine.pool().await;
    let repo = NotebookRepo::new(&pool);
    repo.write_enrichment_and_graph(&updates, &rows)
        .await
        .expect("write");

    // After commit: both the chunk enrichment AND all graph rows are present.
    let et: Option<String> = sqlx::query("SELECT embedding_text FROM chunks WHERE id = ?")
        .bind(&chunk_id)
        .fetch_one(&pool)
        .await
        .unwrap()
        .get("embedding_text");
    assert_eq!(et.as_deref(), Some("Intro | Alice met Bob."));
    assert_eq!(count(&engine, "entity_nodes").await, 2);
    assert_eq!(count(&engine, "entity_edges").await, 1);
    assert_eq!(count(&engine, "entity_mentions").await, 2);
}

// ---------------------------------------------------------------------------
// AC7 (rollback) — a failing graph write rolls back the chunk-enrichment write in
// the SAME transaction: nothing partial persists.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn graph_write_failure_rolls_back_chunk_enrichment() {
    let (_dir, engine) = file_engine().await;
    let (nb, source_id, chunk_id) = seed_source_with_chunk(&engine).await;

    // Poison the graph: an edge whose `from_node` references a node that is never
    // inserted → FK violation on the edge INSERT (foreign_keys=ON), aborting the txn.
    let mut rows = sample_rows(&nb, &source_id, &chunk_id);
    rows.edges[0].from_node = "does-not-exist".to_string();

    let updates = vec![ChunkEnrichmentUpdate {
        chunk_id: chunk_id.clone(),
        embedding_text: "SHOULD NOT PERSIST".to_string(),
        enrichment_json: None,
    }];
    let pool = engine.pool().await;
    let repo = NotebookRepo::new(&pool);

    // The combined write must fail (the poisoned edge aborts the transaction).
    let res = repo.write_enrichment_and_graph(&updates, &rows).await;
    assert!(res.is_err(), "poisoned graph write must return an error");

    // Rollback proof: the chunk-enrichment write in the SAME txn did NOT persist.
    let et: Option<String> = sqlx::query("SELECT embedding_text FROM chunks WHERE id = ?")
        .bind(&chunk_id)
        .fetch_one(&pool)
        .await
        .unwrap()
        .get("embedding_text");
    assert_eq!(
        et, None,
        "chunk enrichment must be rolled back with the graph"
    );

    // And no partial graph rows leaked (the two nodes inserted before the bad edge
    // must roll back too).
    assert_eq!(count(&engine, "entity_nodes").await, 0, "nodes rolled back");
    assert_eq!(count(&engine, "entity_edges").await, 0, "edges rolled back");
    assert_eq!(
        count(&engine, "entity_mentions").await,
        0,
        "mentions rolled back"
    );
}

// ---------------------------------------------------------------------------
// AC8 — ON DELETE CASCADE
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cascade_delete_source_removes_nodes() {
    let (_dir, engine) = file_engine().await;
    let (nb, source_id, chunk_id) = seed_source_with_chunk(&engine).await;
    let rows = sample_rows(&nb, &source_id, &chunk_id);
    let pool = engine.pool().await;
    NotebookRepo::new(&pool)
        .write_enrichment_and_graph(&[], &rows)
        .await
        .expect("write");
    assert_eq!(count(&engine, "entity_nodes").await, 2);

    sqlx::query("DELETE FROM sources WHERE id = ?")
        .bind(&source_id)
        .execute(&pool)
        .await
        .expect("delete source");

    // Deleting the source cascades nodes (source FK) and edges (source FK); nodes
    // then cascade any remaining mentions.
    assert_eq!(count(&engine, "entity_nodes").await, 0, "nodes cascaded");
    assert_eq!(count(&engine, "entity_edges").await, 0, "edges cascaded");
    assert_eq!(
        count(&engine, "entity_mentions").await,
        0,
        "mentions cascaded"
    );
}

#[tokio::test]
async fn cascade_delete_chunk_removes_edges_mentions() {
    let (_dir, engine) = file_engine().await;
    let (nb, source_id, chunk_id) = seed_source_with_chunk(&engine).await;
    let rows = sample_rows(&nb, &source_id, &chunk_id);
    let pool = engine.pool().await;
    NotebookRepo::new(&pool)
        .write_enrichment_and_graph(&[], &rows)
        .await
        .expect("write");
    assert_eq!(count(&engine, "entity_edges").await, 1);
    assert_eq!(count(&engine, "entity_mentions").await, 2);

    sqlx::query("DELETE FROM chunks WHERE id = ?")
        .bind(&chunk_id)
        .execute(&pool)
        .await
        .expect("delete chunk");

    // Edges and mentions are chunk-keyed → cascade. Nodes are source-keyed → survive.
    assert_eq!(count(&engine, "entity_edges").await, 0, "edges cascaded");
    assert_eq!(
        count(&engine, "entity_mentions").await,
        0,
        "mentions cascaded"
    );
    assert_eq!(
        count(&engine, "entity_nodes").await,
        2,
        "source-keyed nodes survive a chunk delete"
    );
}

// ---------------------------------------------------------------------------
// AC3 — zero new LLM calls across the graph step (worker end-to-end)
// ---------------------------------------------------------------------------

fn mock_provider(model: &str, bodies: Vec<&str>) -> (Arc<dyn LlmProvider>, Arc<AtomicU32>) {
    let (provider, calls) = ScriptedProvider::new(bodies);
    (Arc::new(provider.with_model(model)), calls)
}

fn valid_map() -> &'static str {
    r#"{"entities":["Ada Lovelace"],"definitions":[{"term":"engine","definition":"a machine"}],"dates":["1843"],"summary":"A note about Ada Lovelace and the analytical engine."}"#
}

fn empty_coref() -> &'static str {
    r#"{"results":[]}"#
}

fn long_prose() -> String {
    "Ada ".repeat(2100).trim_end().to_string()
}

async fn wait_for_status(engine: &LensEngine, source_id: &str, want: &str) -> bool {
    let pool = engine.pool().await;
    for _ in 0..200 {
        let status: Option<String> =
            sqlx::query("SELECT enrichment_status FROM sources WHERE id = ?")
                .bind(source_id)
                .fetch_one(&pool)
                .await
                .expect("fetch source")
                .get("enrichment_status");
        if status.as_deref() == Some(want) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    false
}

/// The graph-persistence step consumes ONLY in-memory enrichment outputs: driving a
/// full enrichment with a call-counting provider, the total LLM-call count is exactly
/// the map + coref calls — the graph step adds zero. Snapshot the count when the map
/// JSON lands, then confirm it is unchanged after the graph rows are persisted.
#[tokio::test]
async fn zero_llm_delta() {
    let (_dir, engine) = file_engine().await;
    let body = long_prose();
    let (_nb, source_id, chunk_id) = {
        let nb = engine
            .create_notebook("nb", None, None)
            .await
            .expect("create notebook")
            .id
            .to_string();
        let pool = engine.pool().await;
        let source_id = uuid::Uuid::now_v7().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
             content_hash, enrichment_status, created_at) \
             VALUES (?, ?, 'text', 'seed', 'indexed', '/tmp/seed.txt', 1, 'hz', NULL, ?)",
        )
        .bind(&source_id)
        .bind(&nb)
        .bind(&now)
        .execute(&pool)
        .await
        .expect("insert source");
        let chunk_id = format!("{source_id}-chunk-0");
        sqlx::query(
            "INSERT INTO chunks \
             (id, source_id, parent_id, kind, level, section_path, text, \
              token_start, token_end, char_start, char_end, block_type, created_at) \
             VALUES (?, ?, NULL, 'parent', 0, 'Intro', ?, 0, 1, 0, 1, 'paragraph', ?)",
        )
        .bind(&chunk_id)
        .bind(&source_id)
        .bind(&body)
        .bind(&now)
        .execute(&pool)
        .await
        .expect("insert chunk");
        (nb, source_id, chunk_id)
    };
    let _ = chunk_id;

    let (provider, calls) = mock_provider("mock-z", vec![valid_map(), empty_coref()]);
    engine.set_llm_provider(Some(provider)).await;
    engine.enqueue_enrichment_for_test(&source_id);

    assert!(
        wait_for_status(&engine, &source_id, "enriched").await,
        "worker must reach `enriched`; got {:?}",
        {
            let pool = engine.pool().await;
            sqlx::query("SELECT enrichment_status FROM sources WHERE id = ?")
                .bind(&source_id)
                .fetch_one(&pool)
                .await
                .ok()
                .and_then(|r| r.get::<Option<String>, _>("enrichment_status"))
        }
    );

    // The graph rows were persisted (the pass ran).
    assert!(
        count(&engine, "entity_nodes").await >= 1,
        "graph nodes persisted"
    );

    // Total LLM calls = map (1) + coref (1). No third call for the graph step: the
    // pure builder has no provider in scope and CANNOT dispatch. `calls` is monotone
    // and settled at `enriched`, so this is the full-run delta.
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "exactly map + coref calls; the graph step adds ZERO LLM calls"
    );
}

/// Regression: a second enrichment over an already-enriched source (cache-key
/// mismatch forces a re-run) must complete. On the re-run the prior run's summary
/// RAPTOR node is a prose-leaf chunk, so the graph builder emits co-occurrence edges
/// + mentions. Node ids are DETERMINISTIC (a hash of source_id+name+kind), so the
/// regenerated nodes `INSERT OR IGNORE` back onto the SAME rows and the edges/mentions
/// FK-reference ids that exist — no "referenced record does not exist" failure, and
/// no row accumulation on any table.
#[tokio::test]
async fn rerun_over_summary_chunk_persists_without_fk_error() {
    let (_dir, engine) = file_engine().await;
    let body = long_prose();
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .unwrap()
        .id
        .to_string();
    let pool = engine.pool().await;
    let source_id = uuid::Uuid::now_v7().to_string();
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
         content_hash, enrichment_status, created_at) \
         VALUES (?, ?, 'text', 'seed', 'indexed', '/tmp/seed.txt', 1, 'hr', NULL, ?)",
    )
    .bind(&source_id)
    .bind(&nb)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();
    // A prose CHILD (level 1) so mentions + co-occurrence edges actually form.
    let cid = format!("{source_id}-c0");
    sqlx::query(
        "INSERT INTO chunks (id, source_id, parent_id, kind, level, section_path, text, \
          token_start, token_end, char_start, char_end, block_type, created_at) \
         VALUES (?, ?, NULL, 'parent', 0, 'Intro', ?, 0, 1, 0, 1, 'paragraph', ?)",
    )
    .bind(&cid)
    .bind(&source_id)
    .bind(&body)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();
    let _ = nb;

    let (pa, _ca) = mock_provider("model-A", vec![valid_map(), empty_coref()]);
    engine.set_llm_provider(Some(pa)).await;
    engine.enqueue_enrichment_for_test(&source_id);
    assert!(
        wait_for_status(&engine, &source_id, "enriched").await,
        "first run must reach enriched"
    );
    let nodes_1 = count(&engine, "entity_nodes").await;
    assert!(nodes_1 >= 1, "first run persisted nodes");

    // Different model id → cache-key mismatch → the source re-enriches. The summary
    // chunk from run 1's re-embed is now present, so the graph builder emits edges.
    let (pb, _cb) = mock_provider("model-B", vec![valid_map(), empty_coref()]);
    engine.set_llm_provider(Some(pb)).await;
    engine.enqueue_enrichment_for_test(&source_id);
    assert!(
        wait_for_status(&engine, &source_id, "enriched").await,
        "re-run must reach enriched (no FK failure on graph write)"
    );

    // Idempotent: node count is stable across the re-run (deterministic ids).
    assert_eq!(
        count(&engine, "entity_nodes").await,
        nodes_1,
        "re-run must not accumulate nodes"
    );
}

// ---------------------------------------------------------------------------
// #157 — index consistency: self-replacing graph write + wipe-path node delete
// ---------------------------------------------------------------------------

/// The self-replacing write's leading `DELETE FROM entity_nodes` is a harmless no-op
/// on a source that has no prior graph rows: the fresh write still lands in full.
#[tokio::test]
async fn first_time_enrichment_delete_is_noop() {
    let (_dir, engine) = file_engine().await;
    let (nb, source_id, chunk_id) = seed_source_with_chunk(&engine).await;
    let pool = engine.pool().await;
    NotebookRepo::new(&pool)
        .write_enrichment_and_graph(&[], &sample_rows(&nb, &source_id, &chunk_id))
        .await
        .expect("first write");
    assert_eq!(count(&engine, "entity_nodes").await, 2, "fresh write lands");
    assert_eq!(count(&engine, "entity_edges").await, 1);
    assert_eq!(count(&engine, "entity_mentions").await, 2);
}

/// The core #157 guarantee: when a re-enrichment's entity set SHRINKS/CHANGES, the
/// self-replacing write purges the stale nodes rather than leaving orphans. Write
/// {Alice, Bob}, then re-write the same source with only {Carol}; assert Alice/Bob
/// (and their cascaded edges/mentions) are gone and only Carol remains.
#[tokio::test]
async fn enrichment_with_changed_entities_purges_stale_nodes() {
    let (_dir, engine) = file_engine().await;
    let (nb, source_id, chunk_id) = seed_source_with_chunk(&engine).await;
    let pool = engine.pool().await;
    let repo = NotebookRepo::new(&pool);

    repo.write_enrichment_and_graph(&[], &sample_rows(&nb, &source_id, &chunk_id))
        .await
        .expect("first write {Alice, Bob}");
    assert_eq!(count(&engine, "entity_nodes").await, 2);
    assert_eq!(count(&engine, "entity_edges").await, 1);
    assert_eq!(count(&engine, "entity_mentions").await, 2);

    // Re-enrichment now yields a different entity set: only Carol.
    let carol = EntityNode {
        id: format!("{source_id}-carol"),
        notebook_id: nb.clone(),
        source_id: source_id.clone(),
        kind: EntityKind::Concept,
        name: "Carol".to_string(),
        canonical_name: None,
        definition: None,
        resolution_conf: None,
        resolution_prompt_version: None,
        created_at: "2026-01-02T00:00:00Z".to_string(),
    };
    let rows2 = EntityGraphRows {
        source_id: source_id.clone(),
        nodes: vec![carol],
        edges: vec![],
        mentions: vec![],
        dropped_cooccurrence: 0,
    };
    repo.write_enrichment_and_graph(&[], &rows2)
        .await
        .expect("second write {Carol}");

    assert_eq!(
        count(&engine, "entity_nodes").await,
        1,
        "stale Alice/Bob purged, only Carol remains"
    );
    assert_eq!(count(&engine, "entity_edges").await, 0, "stale edge purged");
    assert_eq!(
        count(&engine, "entity_mentions").await,
        0,
        "stale mentions purged"
    );
    let name: String = sqlx::query_scalar("SELECT name FROM entity_nodes WHERE source_id = ?")
        .bind(&source_id)
        .fetch_one(&pool)
        .await
        .expect("one node");
    assert_eq!(name, "Carol");
}

/// A content-changing re-ingest goes through the main inline re-ingest path
/// (`ingest.rs`), whose transaction now deletes the source's `entity_nodes` explicitly.
/// With enrichment DISABLED, that explicit delete is the ONLY thing that can clear a
/// pre-existing node — so this isolates the wipe-path delete (independent of any
/// re-enrichment). Needs a real tokenizer for chunking, so it is gated behind
/// `tokenizer_available()` and skips cleanly offline, like the rest of the ingest suite.
#[tokio::test]
async fn reingest_changed_content_purges_stale_nodes() {
    if !support::tokenizer_available().await {
        eprintln!("skipping: tokenizer unavailable (offline)");
        return;
    }
    let (dir, engine) = support::inject_counting_engine().await;
    support::seed_tokenizer_from_env(dir.path());
    {
        let mut cfg = engine.config().await;
        cfg.enrichment.enabled = false;
        engine.set_config(cfg).await;
    }
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook");

    let src = engine
        .add_text_source(&nb.id, "doc", "Alice met Bob at the fair in Paris.", "text")
        .await
        .expect("add text source")
        .source;
    engine
        .ingest_source(&src.id, |_p| {})
        .await
        .expect("first ingest");

    // Simulate a prior enrichment's node for this source (enrichment is disabled here).
    let pool = engine.pool().await;
    sqlx::query(
        "INSERT INTO entity_nodes (id, notebook_id, source_id, kind, name, created_at) \
         VALUES (?, ?, ?, 'concept', 'Alice', '2026-01-01T00:00:00Z')",
    )
    .bind(format!("{}-stale", src.id))
    .bind(nb.id.to_string())
    .bind(&src.id)
    .execute(&pool)
    .await
    .expect("seed stale node");
    let stale: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM entity_nodes WHERE source_id = ?")
        .bind(&src.id)
        .fetch_one(&pool)
        .await
        .expect("count");
    assert_eq!(stale, 1, "stale node seeded");

    // Change the managed file's content so the re-ingest is not a content-hash no-op,
    // then re-ingest through the main path.
    std::fs::write(
        &src.locator,
        "A totally different note about Carol in Berlin.",
    )
    .expect("overwrite managed file");
    engine
        .ingest_source(&src.id, |_p| {})
        .await
        .expect("re-ingest");

    let remaining: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM entity_nodes WHERE source_id = ?")
            .bind(&src.id)
            .fetch_one(&pool)
            .await
            .expect("count");
    assert_eq!(
        remaining, 0,
        "re-ingest's explicit entity_nodes delete cleared the stale node"
    );
}

/// `purge_source` (explicit `chunks_fts` delete + `DELETE FROM sources`) clears all
/// three graph tables via `ON DELETE CASCADE` on the source/chunk FKs. Behavioral
/// (count-based) so it fails loudly if the cascade ever regresses.
#[tokio::test]
async fn purge_source_cascades_graph_rows() {
    let (_dir, engine) = file_engine().await;
    let (nb, source_id, chunk_id) = seed_source_with_chunk(&engine).await;
    let pool = engine.pool().await;
    NotebookRepo::new(&pool)
        .write_enrichment_and_graph(&[], &sample_rows(&nb, &source_id, &chunk_id))
        .await
        .expect("write");
    assert_eq!(count(&engine, "entity_nodes").await, 2);

    engine.trash_source(&source_id).await.expect("trash");
    engine.purge_source(&source_id).await.expect("purge");

    assert_eq!(count(&engine, "entity_nodes").await, 0, "nodes cascaded");
    assert_eq!(count(&engine, "entity_edges").await, 0, "edges cascaded");
    assert_eq!(
        count(&engine, "entity_mentions").await,
        0,
        "mentions cascaded"
    );
}

/// `purge_notebook` (`DELETE FROM notebooks`) cascades sources → chunks → all graph
/// tables. Behavioral assertion.
#[tokio::test]
async fn purge_notebook_cascades_graph_rows() {
    let (_dir, engine) = file_engine().await;
    let (nb, source_id, chunk_id) = seed_source_with_chunk(&engine).await;
    let pool = engine.pool().await;
    NotebookRepo::new(&pool)
        .write_enrichment_and_graph(&[], &sample_rows(&nb, &source_id, &chunk_id))
        .await
        .expect("write");
    assert_eq!(count(&engine, "entity_nodes").await, 2);

    let nb_id = lens_core::NotebookId::from(nb);
    engine.trash_notebook(&nb_id).await.expect("trash notebook");
    engine.purge_notebook(&nb_id).await.expect("purge notebook");

    assert_eq!(count(&engine, "entity_nodes").await, 0, "nodes cascaded");
    assert_eq!(count(&engine, "entity_edges").await, 0, "edges cascaded");
    assert_eq!(
        count(&engine, "entity_mentions").await,
        0,
        "mentions cascaded"
    );
}

/// The cascade-based purge guarantees hold ONLY if `foreign_keys` is enforced at
/// runtime. Pin it against the production pool constructor so a future pool-config
/// change that drops the pragma fails loudly.
#[tokio::test]
async fn foreign_keys_pragma_enabled() {
    let (_dir, engine) = file_engine().await;
    let pool = engine.pool().await;
    let fk: i64 = sqlx::query_scalar("PRAGMA foreign_keys")
        .fetch_one(&pool)
        .await
        .expect("pragma");
    assert_eq!(fk, 1, "foreign_keys must be ON for cascade correctness");
}

// ---------------------------------------------------------------------------
// #155 — entity-vector zero-orphan on every chunk-vector drop site (S2/S4)
// ---------------------------------------------------------------------------

/// A content-changing re-ingest drops the source's chunk vectors before the SQLite
/// txn (`wipe_source_content`); #155 wires the entity-vector drop alongside it, so
/// the source's `ent__` rows must be gone after the re-ingest. Gated on a tokenizer
/// (real chunking) like the rest of the ingest suite; skips cleanly offline.
#[tokio::test]
async fn reingest_drops_entity_vectors() {
    if !support::tokenizer_available().await {
        eprintln!("skipping: tokenizer unavailable (offline)");
        return;
    }
    let (dir, engine) = support::inject_counting_engine().await;
    support::seed_tokenizer_from_env(dir.path());
    {
        let mut cfg = engine.config().await;
        cfg.enrichment.enabled = false;
        engine.set_config(cfg).await;
    }
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook");
    let nb_id = nb.id.to_string();

    let src = engine
        .add_text_source(&nb.id, "doc", "Alice met Bob at the fair in Paris.", "text")
        .await
        .expect("add text source")
        .source;
    engine
        .ingest_source(&src.id, |_p| {})
        .await
        .expect("first ingest");

    let (store, coord) = ent_store(&engine, &nb_id).await;
    seed_ent_vector(&store, &coord, &format!("{}-e0", src.id), &src.id, &nb_id).await;
    assert_eq!(
        ent_rows_for_source(&store, &coord).await,
        1,
        "entity vector seeded"
    );

    std::fs::write(
        &src.locator,
        "A totally different note about Carol in Berlin.",
    )
    .expect("overwrite managed file");
    engine
        .ingest_source(&src.id, |_p| {})
        .await
        .expect("re-ingest");

    assert_eq!(
        ent_rows_for_source(&store, &coord).await,
        0,
        "re-ingest must drop the source's ent__ rows"
    );
}

/// `purge_source` iterates active coordinates dropping chunk vectors; #155 adds the
/// entity-vector drop in the same loop (the SQLite cascade never reaches Lance).
#[tokio::test]
async fn purge_source_drops_entity_vectors() {
    let (_dir, engine) = file_engine().await;
    let (nb, source_id, chunk_id) = seed_source_with_chunk(&engine).await;
    seed_active_coord(&engine, &nb).await;
    let pool = engine.pool().await;
    NotebookRepo::new(&pool)
        .write_enrichment_and_graph(&[], &sample_rows(&nb, &source_id, &chunk_id))
        .await
        .expect("write");

    let (store, coord) = ent_store(&engine, &nb).await;
    seed_ent_vector(&store, &coord, &format!("{source_id}-na"), &source_id, &nb).await;
    assert_eq!(ent_rows_for_source(&store, &coord).await, 1);

    engine.trash_source(&source_id).await.expect("trash");
    engine.purge_source(&source_id).await.expect("purge");

    assert_eq!(count(&engine, "entity_nodes").await, 0, "nodes cascaded");
    assert_eq!(
        ent_rows_for_source(&store, &coord).await,
        0,
        "purge_source must drop the source's ent__ rows"
    );
}

/// `purge_notebook` drops the notebook's chunk-vector tables from the registry; #155
/// adds an explicit `ent__`-table drop (those tables carry no registry). Asserts the
/// physical `ent__` tables for the notebook are gone.
#[tokio::test]
async fn purge_notebook_drops_entity_tables() {
    let (_dir, engine) = file_engine().await;
    let (nb, source_id, chunk_id) = seed_source_with_chunk(&engine).await;
    let pool = engine.pool().await;
    NotebookRepo::new(&pool)
        .write_enrichment_and_graph(&[], &sample_rows(&nb, &source_id, &chunk_id))
        .await
        .expect("write");

    let (store, coord) = ent_store(&engine, &nb).await;
    seed_ent_vector(&store, &coord, &format!("{source_id}-na"), &source_id, &nb).await;
    assert_eq!(
        store
            .entity_table_names_for_notebook(&nb)
            .await
            .expect("list ent tables")
            .len(),
        1,
        "ent__ table exists before purge"
    );

    let nb_id = NotebookId::from(nb.clone());
    engine.trash_notebook(&nb_id).await.expect("trash notebook");
    engine.purge_notebook(&nb_id).await.expect("purge notebook");

    assert_eq!(count(&engine, "entity_nodes").await, 0, "nodes cascaded");
    assert!(
        store
            .entity_table_names_for_notebook(&nb)
            .await
            .expect("list ent tables")
            .is_empty(),
        "purge_notebook must physically drop the notebook's ent__ tables"
    );
}

/// #155: a resolution pass fully resets prior canonical assignments — a node aliased by
/// an earlier pass but no longer in a merged group must not keep a stale canonical_name.
#[tokio::test]
async fn resolution_write_clears_stale_canonical() {
    let (_dir, engine) = file_engine().await;
    let (nb, source_id, chunk_id) = seed_source_with_chunk(&engine).await;
    let pool = engine.pool().await;
    let repo = NotebookRepo::new(&pool);
    repo.write_enrichment_and_graph(&[], &sample_rows(&nb, &source_id, &chunk_id))
        .await
        .expect("write graph");

    // Simulate a prior resolution pass that aliased this source's nodes.
    sqlx::query(
        "UPDATE entity_nodes SET canonical_name = 'Alias', resolution_conf = 0.95, \
         resolution_prompt_version = 'res-v1' WHERE notebook_id = ?",
    )
    .bind(&nb)
    .execute(&pool)
    .await
    .expect("seed prior resolution");

    // A new pass that merges nothing (empty updates) must clear the stale aliases.
    repo.write_resolution_updates(&nb, "res-v1", &[], false)
        .await
        .expect("resolution write");

    let stale: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM entity_nodes WHERE notebook_id = ? AND canonical_name IS NOT NULL",
    )
    .bind(&nb)
    .fetch_one(&pool)
    .await
    .expect("count");
    assert_eq!(
        stale, 0,
        "prior-pass canonical_name must be cleared by a fresh pass"
    );
}
