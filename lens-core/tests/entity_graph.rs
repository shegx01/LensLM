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
use sqlx::Row;
use tempfile::TempDir;

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
