//! Integration tests for the #154 LLM relation-extraction pass (worker end-to-end).
//!
//! A call-counting/scripted `LlmProvider` serves map → coref → relations responses.
//! Covers: strategy-off yields zero semantic edges; the On path persists semantic
//! edges; the entity-density gate; mixed-confidence + non-existent-entity filtering;
//! and a tight-budget breach flipping the source to `failed`.

use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::time::Duration;

use lens_core::LensEngine;
use lens_core::enrichment::test_util::ScriptedProvider;
use lens_core::llm::LlmProvider;
use sqlx::Row;
use tempfile::TempDir;

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

async fn set_relations_on(engine: &LensEngine) {
    let mut cfg = engine.config().await;
    cfg.enrichment.relations_strategy = lens_core::config::RelationsStrategy::On;
    engine.set_config(cfg).await;
}

/// Seeds a notebook + `indexed` source with a level-0 parent (drives the structural
/// map / size gate) and a prose CHILD carrying the entity names (so the relations
/// pass has a prose chunk to sample). Returns `(source_id, child_chunk_id)`.
async fn seed_prose_source(engine: &LensEngine, content_hash: &str) -> (String, String) {
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
         VALUES (?, ?, 'text', 'seed', 'indexed', '/tmp/seed.txt', 1, ?, NULL, ?)",
    )
    .bind(&source_id)
    .bind(&nb)
    .bind(content_hash)
    .bind(&now)
    .execute(&pool)
    .await
    .expect("insert source");

    // Level-0 parent: long prose so the size gate passes and the map runs.
    let parent = format!("{source_id}-p0");
    let parent_text = "Ada Babbage Turing Hopper Lovelace Church Neumann Shannon ".repeat(400);
    sqlx::query(
        "INSERT INTO chunks (id, source_id, parent_id, kind, level, section_path, text, \
          token_start, token_end, char_start, char_end, block_type, created_at) \
         VALUES (?, ?, NULL, 'parent', 0, 'Intro', ?, 0, 1, 0, 1, 'paragraph', ?)",
    )
    .bind(&parent)
    .bind(&source_id)
    .bind(&parent_text)
    .bind(&now)
    .execute(&pool)
    .await
    .expect("insert parent");

    // Prose child (level 1) with the entity names for the relations sampler.
    let child = format!("{source_id}-c0");
    sqlx::query(
        "INSERT INTO chunks (id, source_id, parent_id, kind, level, section_path, text, \
          token_start, token_end, char_start, char_end, block_type, created_at) \
         VALUES (?, ?, ?, 'child', 1, 'Intro', 'Ada founded a lab with Babbage; Turing joined.', \
          0, 1, 0, 1, 'paragraph', ?)",
    )
    .bind(&child)
    .bind(&source_id)
    .bind(&parent)
    .bind(&now)
    .execute(&pool)
    .await
    .expect("insert child");
    (source_id, child)
}

/// A structural map with 8 entities (meets the density floor).
fn map_8_entities() -> &'static str {
    r#"{"entities":["Ada","Babbage","Turing","Hopper","Lovelace","Church","Neumann","Shannon"],"definitions":[],"dates":[],"summary":"A note about early computing pioneers."}"#
}

/// A structural map with only 5 entities (below the floor of 8).
fn map_5_entities() -> &'static str {
    r#"{"entities":["Ada","Babbage","Turing","Hopper","Lovelace"],"definitions":[],"dates":[],"summary":"A short note."}"#
}

fn empty_coref() -> &'static str {
    r#"{"results":[]}"#
}

fn mock_provider(model: &str, bodies: Vec<&str>) -> (Arc<dyn LlmProvider>, Arc<AtomicU32>) {
    let (provider, calls) = ScriptedProvider::new(bodies);
    (Arc::new(provider.with_model(model)), calls)
}

async fn count(engine: &LensEngine, table: &str) -> i64 {
    let pool = engine.pool().await;
    sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {table}"))
        .fetch_one(&pool)
        .await
        .unwrap_or_else(|_| panic!("count {table}"))
}

async fn semantic_edge_count(engine: &LensEngine, source_id: &str) -> i64 {
    let pool = engine.pool().await;
    sqlx::query_scalar(
        "SELECT COUNT(*) FROM entity_edges WHERE source_id = ? AND relation != 'co_occurs'",
    )
    .bind(source_id)
    .fetch_one(&pool)
    .await
    .expect("semantic edge count")
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

/// Relations OFF (default): even with a graph-worthy doc, zero semantic edges.
#[tokio::test]
async fn strategy_off_yields_zero_semantic_edges() {
    let (_dir, engine) = file_engine().await;
    let (source_id, _child) = seed_prose_source(&engine, "off").await;

    // Map + coref only; the relations pass must never run.
    let (provider, _calls) = mock_provider("m-off", vec![map_8_entities(), empty_coref()]);
    engine.set_llm_provider(Some(provider)).await;
    engine.enqueue_enrichment_for_test(&source_id);

    assert!(wait_for_status(&engine, &source_id, "enriched").await);
    assert_eq!(
        semantic_edge_count(&engine, &source_id).await,
        0,
        "strategy Off must produce no semantic edges"
    );
}

/// Relations ON + 8 entities + valid triples → semantic edges persist alongside
/// co-occurrence edges.
#[tokio::test]
async fn strategy_on_persists_semantic_edges() {
    let (_dir, engine) = file_engine().await;
    set_relations_on(&engine).await;
    let (source_id, _child) = seed_prose_source(&engine, "on").await;

    let relations = r#"{"relations":[{"from_entity":"Ada","to_entity":"Babbage","predicate":"founded","chunk_id":"1","confidence":0.9}]}"#;
    let (provider, _calls) =
        mock_provider("m-on", vec![map_8_entities(), empty_coref(), relations]);
    engine.set_llm_provider(Some(provider)).await;
    engine.enqueue_enrichment_for_test(&source_id);

    assert!(wait_for_status(&engine, &source_id, "enriched").await);
    assert_eq!(
        semantic_edge_count(&engine, &source_id).await,
        1,
        "one `founded` semantic edge persists"
    );
    let pool = engine.pool().await;
    let (relation, confidence): (String, Option<f64>) = sqlx::query_as(
        "SELECT relation, confidence FROM entity_edges \
         WHERE source_id = ? AND relation != 'co_occurs'",
    )
    .bind(&source_id)
    .fetch_one(&pool)
    .await
    .expect("semantic edge");
    assert_eq!(relation, "founded");
    assert_eq!(confidence, Some(0.9));
}

/// Entity-floor gate: ON + map OK but only 5 entities (< floor 8) → zero semantic edges.
#[tokio::test]
async fn entity_floor_gate_blocks_pass() {
    let (_dir, engine) = file_engine().await;
    set_relations_on(&engine).await;
    let (source_id, _child) = seed_prose_source(&engine, "floor").await;

    // Even if the (never-reached) relations response is scripted, the gate blocks it.
    let relations = r#"{"relations":[{"from_entity":"Ada","to_entity":"Babbage","predicate":"founded","chunk_id":"1","confidence":0.9}]}"#;
    let (provider, _calls) =
        mock_provider("m-floor", vec![map_5_entities(), empty_coref(), relations]);
    engine.set_llm_provider(Some(provider)).await;
    engine.enqueue_enrichment_for_test(&source_id);

    assert!(wait_for_status(&engine, &source_id, "enriched").await);
    assert_eq!(
        semantic_edge_count(&engine, &source_id).await,
        0,
        "sub-floor entity count must block the relations pass"
    );
}

/// Mixed-confidence + non-existent-entity filtering: only the >= 0.5, existing-endpoint
/// triple persists.
#[tokio::test]
async fn mixed_confidence_and_unknown_entity_filtered() {
    let (_dir, engine) = file_engine().await;
    set_relations_on(&engine).await;
    let (source_id, _child) = seed_prose_source(&engine, "mixed").await;

    let relations = r#"{"relations":[
        {"from_entity":"Ada","to_entity":"Babbage","predicate":"founded","chunk_id":"1","confidence":0.3},
        {"from_entity":"Ada","to_entity":"Turing","predicate":"influenced","chunk_id":"1","confidence":0.8},
        {"from_entity":"Nobody","to_entity":"Babbage","predicate":"founded","chunk_id":"1","confidence":0.9}
    ]}"#;
    let (provider, _calls) =
        mock_provider("m-mixed", vec![map_8_entities(), empty_coref(), relations]);
    engine.set_llm_provider(Some(provider)).await;
    engine.enqueue_enrichment_for_test(&source_id);

    assert!(wait_for_status(&engine, &source_id, "enriched").await);
    // Only the 0.8 `influenced` triple survives (0.3 dropped; unknown entity dropped).
    assert_eq!(semantic_edge_count(&engine, &source_id).await, 1);
    let pool = engine.pool().await;
    let relation: String = sqlx::query_scalar(
        "SELECT relation FROM entity_edges WHERE source_id = ? AND relation != 'co_occurs'",
    )
    .bind(&source_id)
    .fetch_one(&pool)
    .await
    .expect("one semantic edge");
    assert_eq!(relation, "influenced");
}

/// A tight per-job call budget breached during the relations pass flips the source to
/// `failed` with `budget_exceeded` (AC12). Map(1) + coref(1) consume the 2-call budget,
/// so the relations batch's pre-dispatch check breaks the circuit.
#[tokio::test]
async fn budget_breach_during_relations_fails_source() {
    let (_dir, engine) = file_engine().await;
    set_relations_on(&engine).await;
    engine.set_enrichment_max_calls_for_test(2);
    let (source_id, _child) = seed_prose_source(&engine, "budget").await;

    let relations = r#"{"relations":[{"from_entity":"Ada","to_entity":"Babbage","predicate":"founded","chunk_id":"1","confidence":0.9}]}"#;
    let (provider, _calls) =
        mock_provider("m-budget", vec![map_8_entities(), empty_coref(), relations]);
    engine.set_llm_provider(Some(provider)).await;
    engine.enqueue_enrichment_for_test(&source_id);

    assert!(
        wait_for_status(&engine, &source_id, "failed").await,
        "budget breach during relations must flip the source to failed"
    );
    let pool = engine.pool().await;
    let meta_json: Option<String> = sqlx::query("SELECT enrichment_meta FROM sources WHERE id = ?")
        .bind(&source_id)
        .fetch_one(&pool)
        .await
        .expect("fetch meta")
        .get("enrichment_meta");
    let meta: serde_json::Value =
        serde_json::from_str(&meta_json.expect("meta present")).expect("parse meta");
    assert_eq!(
        meta["budget_exceeded"], true,
        "budget_exceeded must be recorded"
    );
    assert_eq!(
        count(&engine, "entity_edges").await,
        0,
        "no edges on breach"
    );
}
