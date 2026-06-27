//! `LensEngine::reembed_notebook` — the model-switch re-embed (M4 Phase 4b,
//! Step 9): re-embed every chunk into the notebook's newly-configured coordinate,
//! flip it active, retire the old one. Plus the R2 coordinate re-check guard.
//!
//! Setup seeds a source + chunks + raw active vectors via the production `add`
//! path (no tokenizer / no real model download), then injects a model-free
//! `CountingEmbedder` for the TARGET model so the re-embed runs deterministically.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::Duration;

use lens_core::embedder::{CountingEmbedder, Embedder, resolve};
use lens_core::enrichment::reembed::ReembedOutcome;
use lens_core::vector_store::{Coordinate, LanceVectorStore, VectorRow, VectorStore};
use lens_core::{
    DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL_ID, EmbeddingBackend, LensEngine, NotebookId,
};

fn coord(nb: &str, model: &str, dim: usize) -> Coordinate {
    Coordinate::new(nb, EmbeddingBackend::Fastembed, model, dim)
}

const BGE: &str = "bge-m3";
const BGE_DIM: usize = 1024;

/// Injects a model-free `CountingEmbedder` keyed by `model_id`'s registry spec so
/// `embedder_for(model_id)` returns the right-dim deterministic embedder.
fn inject_embedder(engine: &LensEngine, model_id: &str) {
    let spec = resolve(model_id);
    let e: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new_with_dim(
        spec.dim,
        spec.id,
        spec.prefix_doc,
        spec.prefix_query,
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    ));
    engine
        .set_embedder_for_test(e, lens_core::EmbeddingBackend::Fastembed)
        .expect("inject embedder");
}

/// A raw active vector for a chunk (axis-aligned, default nomic dim).
fn row(chunk_id: &str, source_id: &str, nb: &str, level: i32) -> VectorRow {
    let mut v = vec![0.0_f32; DEFAULT_EMBED_DIM];
    v[(level as usize) % DEFAULT_EMBED_DIM] = 1.0;
    VectorRow {
        chunk_id: chunk_id.to_string(),
        source_id: source_id.to_string(),
        notebook_id: nb.to_string(),
        level,
        vector: v,
    }
}

/// Seeds a notebook with one source, two chunks, and a raw nomic-768 active
/// coordinate (via the production `add`). Returns `(notebook_id, source_id)`.
async fn seed_nomic_notebook(engine: &LensEngine) -> (String, String) {
    let nb = engine
        .create_notebook("reembed-nb", None, None)
        .await
        .unwrap()
        .id
        .to_string();
    let pool = engine.pool().await;
    let source_id = uuid::Uuid::now_v7().to_string();
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
         content_hash, enrichment_status, created_at) \
         VALUES (?, ?, 'text', 'seed', 'indexed', '/tmp/seed.txt', 1, ?, NULL, ?)",
    )
    .bind(&source_id)
    .bind(&nb)
    .bind(format!("hash-{source_id}"))
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(&pool)
    .await
    .expect("insert source");

    let now = chrono::Utc::now().to_rfc3339();
    let parent_id = format!("{source_id}-p0");
    let child_id = format!("{source_id}-c0");
    for (id, parent, kind, level, text) in [
        (
            &parent_id,
            None::<&str>,
            "parent",
            0_i32,
            "Parent body text.",
        ),
        (
            &child_id,
            Some(parent_id.as_str()),
            "child",
            1,
            "Child body text.",
        ),
    ] {
        sqlx::query(
            "INSERT INTO chunks \
             (id, source_id, parent_id, kind, level, section_path, text, \
              token_start, token_end, char_start, char_end, block_type, created_at) \
             VALUES (?, ?, ?, ?, ?, '[\"Intro\"]', ?, 0, 1, 0, ?, 'paragraph', ?)",
        )
        .bind(id)
        .bind(&source_id)
        .bind(parent)
        .bind(kind)
        .bind(level)
        .bind(text)
        .bind(text.len() as i64)
        .bind(&now)
        .execute(&pool)
        .await
        .expect("insert chunk");
    }

    let data_dir = engine.data_dir_for_test().await;
    let store = LanceVectorStore::new(&data_dir, pool.clone());
    store
        .add(
            &coord(&nb, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM),
            vec![
                row(&parent_id, &source_id, &nb, 0),
                row(&child_id, &source_id, &nb, 1),
            ],
        )
        .await
        .expect("seed raw active vectors");

    (nb, source_id)
}

async fn set_model(engine: &LensEngine, nb: &str, model: &str) {
    sqlx::query("UPDATE notebooks SET embedding_model = ? WHERE id = ?")
        .bind(model)
        .bind(nb)
        .execute(&engine.pool().await)
        .await
        .unwrap();
}

async fn coord_count(engine: &LensEngine, nb: &str, model: &str, dim: usize, status: &str) -> i64 {
    sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM embedding_index \
         WHERE notebook_id = ? AND model = ? AND dim = ? AND status = ?",
    )
    .bind(nb)
    .bind(model)
    .bind(dim as i64)
    .bind(status)
    .fetch_one(&engine.pool().await)
    .await
    .unwrap()
}

/// Switching a notebook from nomic-768 to bge-m3-1024 re-embeds every chunk into
/// the NEW coordinate, flips it active, and retires the OLD coordinate.
#[tokio::test]
async fn model_change_reembeds_into_new_coordinate_and_retires_old() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    inject_embedder(&engine, BGE);
    let (nb, _src) = seed_nomic_notebook(&engine).await;

    // Old coordinate is active before the switch.
    assert_eq!(
        coord_count(
            &engine,
            &nb,
            DEFAULT_EMBED_MODEL_ID,
            DEFAULT_EMBED_DIM,
            "active"
        )
        .await,
        1
    );

    set_model(&engine, &nb, BGE).await;
    let outcome = engine
        .reembed_notebook(&NotebookId::from(nb.clone()), |_, _| {})
        .await
        .expect("reembed");

    assert_eq!(
        outcome,
        ReembedOutcome::Switched {
            model: BGE.to_string(),
            dim: BGE_DIM,
            retired: 1,
        }
    );
    // New coordinate active; old coordinate fully gone (row + table).
    assert_eq!(coord_count(&engine, &nb, BGE, BGE_DIM, "active").await, 1);
    assert_eq!(
        coord_count(
            &engine,
            &nb,
            DEFAULT_EMBED_MODEL_ID,
            DEFAULT_EMBED_DIM,
            "active"
        )
        .await,
        0
    );
    let total_old: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM embedding_index WHERE notebook_id = ? AND model = ?",
    )
    .bind(&nb)
    .bind(DEFAULT_EMBED_MODEL_ID)
    .fetch_one(&engine.pool().await)
    .await
    .unwrap();
    assert_eq!(total_old, 0, "old coordinate row deleted, not just staled");

    // Search on the NEW coordinate returns hits (the new table serves).
    let store = LanceVectorStore::new(&engine.data_dir_for_test().await, engine.pool().await);
    let q = vec![0.1_f32; BGE_DIM];
    let hits = store
        .search(&coord(&nb, BGE, BGE_DIM), &q, 4)
        .await
        .unwrap();
    assert_eq!(hits.len(), 2, "both chunks searchable under the new model");
}

/// No-op when the notebook's configured model already matches its active
/// coordinate (re-selecting the same model triggers nothing).
#[tokio::test]
async fn noop_when_model_already_matches() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    inject_embedder(&engine, DEFAULT_EMBED_MODEL_ID);
    let (nb, _src) = seed_nomic_notebook(&engine).await;

    // Configured model is already nomic (the create default) → no-op.
    let outcome = engine
        .reembed_notebook(&NotebookId::from(nb.clone()), |_, _| {})
        .await
        .expect("reembed");
    assert_eq!(outcome, ReembedOutcome::NoOp);
    assert_eq!(
        coord_count(
            &engine,
            &nb,
            DEFAULT_EMBED_MODEL_ID,
            DEFAULT_EMBED_DIM,
            "active"
        )
        .await,
        1
    );
}

/// R2 coordinate re-check guard: when the notebook's configured model changes
/// AGAIN while the new coordinate is building, the flip is skipped (building left
/// for the GC) and the OLD coordinate keeps serving — the newest switch wins.
#[tokio::test]
async fn r2_second_switch_mid_build_aborts_flip() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    inject_embedder(&engine, BGE);
    inject_embedder(&engine, "all-minilm");
    let (nb, _src) = seed_nomic_notebook(&engine).await;

    // Park the re-embed just before the flip via the shared preflip gate.
    let gate = Arc::new(tokio::sync::Notify::new());
    engine
        .set_reembed_preflip_gate_for_test(Some(gate.clone()))
        .await;

    // First switch: nomic -> bge-m3. Run it in a task; it will build then park.
    set_model(&engine, &nb, BGE).await;
    let engine2 = engine.clone();
    let nb2 = nb.clone();
    let handle = tokio::spawn(async move {
        engine2
            .reembed_notebook(&NotebookId::from(nb2), |_, _| {})
            .await
    });

    // While parked, a SECOND switch lands: bge-m3 -> all-minilm.
    tokio::time::sleep(Duration::from_millis(150)).await;
    set_model(&engine, &nb, "all-minilm").await;

    // Release the gate; the in-lock re-check sees the changed coordinate and aborts.
    gate.notify_one();
    let outcome = handle.await.unwrap().expect("reembed");
    assert_eq!(outcome, ReembedOutcome::RaceAborted);

    // The OLD nomic coordinate was NEVER flipped away — it still serves.
    assert_eq!(
        coord_count(
            &engine,
            &nb,
            DEFAULT_EMBED_MODEL_ID,
            DEFAULT_EMBED_DIM,
            "active"
        )
        .await,
        1
    );
    // No bge-m3 active coordinate was promoted (the building row is left for GC).
    assert_eq!(coord_count(&engine, &nb, BGE, BGE_DIM, "active").await, 0);
}
