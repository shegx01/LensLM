//! `VectorStore::retire_coordinate` — OLD-coordinate retirement after a
//! model/dim-change re-embed flips the NEW coordinate active (M4 Phase 4b,
//! Step 8 / R3).
//!
//! Builds a FILE-BACKED engine via [`LensEngine::init`] so the startup-GC
//! (`gc_orphan_embedding_tables`, which sweeps `building`/`stale` rows) runs on
//! `reopen`, exercising the crash-recovery path. The external [`LanceVectorStore`]
//! shares the engine's pool + lancedb root (same `data_dir`), mirroring the
//! existing `vector_index`/`enrichment_step5` integration tests.

use std::sync::atomic::Ordering;

use lens_core::vector_store::{
    CRASH_AFTER_RETIRE_STALE_BEFORE_LANCE_DROP, LanceVectorStore, VectorRow, VectorStore,
};
use lens_core::{DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL_ID, LensEngine};

/// A deterministic unit vector of length `dim`.
fn unit_vector(seed: usize, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim)
        .map(|j| ((seed as f32) * 0.013 + (j as f32) * 0.0007).sin())
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// `n` rows of width `dim` for `notebook`.
fn make_rows(notebook: &str, n: usize, dim: usize) -> Vec<VectorRow> {
    (0..n)
        .map(|i| VectorRow {
            chunk_id: format!("chunk-{i}"),
            source_id: "src-1".to_string(),
            notebook_id: notebook.to_string(),
            level: 1,
            vector: unit_vector(i, dim),
        })
        .collect()
}

/// Fresh engine + a created notebook + its pool (registry FK satisfied).
async fn engine_and_notebook() -> (tempfile::TempDir, LensEngine, sqlx::SqlitePool, String) {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let nb = engine.create_notebook("retire", None, None).await.unwrap();
    let pool = engine.pool().await;
    (dir, engine, pool, nb.id.to_string())
}

/// COUNT of `embedding_index` rows for a coordinate filtered by status (`None` =
/// any status).
async fn row_count(
    pool: &sqlx::SqlitePool,
    nb: &str,
    model: &str,
    dim: usize,
    status: Option<&str>,
) -> i64 {
    match status {
        Some(s) => sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM embedding_index \
             WHERE notebook_id = ? AND model = ? AND dim = ? AND status = ?",
        )
        .bind(nb)
        .bind(model)
        .bind(dim as i64)
        .bind(s)
        .fetch_one(pool)
        .await
        .unwrap(),
        None => sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM embedding_index \
             WHERE notebook_id = ? AND model = ? AND dim = ?",
        )
        .bind(nb)
        .bind(model)
        .bind(dim as i64)
        .fetch_one(pool)
        .await
        .unwrap(),
    }
}

/// Whether a physical Lance table exists under `data_dir`.
async fn table_exists(data_dir: &std::path::Path, name: &str) -> bool {
    let root = data_dir.join("lancedb");
    let conn = lancedb::connect(root.to_string_lossy().as_ref())
        .execute()
        .await
        .unwrap();
    conn.table_names()
        .execute()
        .await
        .unwrap()
        .iter()
        .any(|t| t == name)
}

const BGE: &str = "bge-m3";
const BGE_DIM: usize = 1024;
const BGE_TABLE: &str = "bge_m3"; // model_slug("bge-m3")

/// Retiring one coordinate demotes+drops+deletes ONLY its active row; a second
/// coordinate's active row for the same notebook is untouched and still serves.
#[tokio::test]
async fn retire_drops_active_and_leaves_other_coordinate() {
    let (dir, _engine, pool, nb) = engine_and_notebook().await;
    let store = LanceVectorStore::new(dir.path(), pool.clone());

    // Two active coordinates coexist (different model/dim → partial-unique OK).
    store
        .add(
            &nb,
            DEFAULT_EMBED_MODEL_ID,
            DEFAULT_EMBED_DIM,
            make_rows(&nb, 4, DEFAULT_EMBED_DIM),
        )
        .await
        .unwrap();
    store
        .add(&nb, BGE, BGE_DIM, make_rows(&nb, 4, BGE_DIM))
        .await
        .unwrap();

    let nomic_table = format!("vec__{nb}__nomic_v15__d{DEFAULT_EMBED_DIM}");
    let bge_table = format!("vec__{nb}__{BGE_TABLE}__d{BGE_DIM}");
    assert!(table_exists(dir.path(), &nomic_table).await);
    assert!(table_exists(dir.path(), &bge_table).await);

    store
        .retire_coordinate(&nb, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM)
        .await
        .unwrap();

    // Old coordinate fully gone: no row, table dropped.
    assert_eq!(
        row_count(&pool, &nb, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM, None).await,
        0
    );
    assert!(!table_exists(dir.path(), &nomic_table).await);

    // Other coordinate intact and searchable.
    assert_eq!(row_count(&pool, &nb, BGE, BGE_DIM, Some("active")).await, 1);
    assert!(table_exists(dir.path(), &bge_table).await);
    let hits = store
        .search(&nb, BGE, BGE_DIM, &unit_vector(0, BGE_DIM), 4)
        .await
        .unwrap();
    assert!(!hits.is_empty(), "bge-m3 coordinate still serves search");
}

/// Retirement touches ONLY the `active` row of the coordinate — a transient
/// `building` row for the SAME coordinate survives (proves the dedicated
/// active-scoped demote, not a blanket `set_status`).
#[tokio::test]
async fn retire_leaves_building_row_untouched() {
    let (dir, _engine, pool, nb) = engine_and_notebook().await;
    let store = LanceVectorStore::new(dir.path(), pool.clone());

    store
        .add(
            &nb,
            DEFAULT_EMBED_MODEL_ID,
            DEFAULT_EMBED_DIM,
            make_rows(&nb, 4, DEFAULT_EMBED_DIM),
        )
        .await
        .unwrap();
    let building = store
        .create_building_table(&nb, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM)
        .await
        .unwrap();

    store
        .retire_coordinate(&nb, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM)
        .await
        .unwrap();

    // No active row remains; the building row + its table are untouched.
    assert_eq!(
        row_count(
            &pool,
            &nb,
            DEFAULT_EMBED_MODEL_ID,
            DEFAULT_EMBED_DIM,
            Some("active")
        )
        .await,
        0
    );
    assert_eq!(
        row_count(
            &pool,
            &nb,
            DEFAULT_EMBED_MODEL_ID,
            DEFAULT_EMBED_DIM,
            Some("building")
        )
        .await,
        1
    );
    assert!(table_exists(dir.path(), &building).await);
}

/// Retiring a coordinate with no active row is an idempotent no-op (e.g. a retry
/// after the GC already reclaimed it).
#[tokio::test]
async fn retire_idempotent_when_no_active() {
    let (dir, _engine, pool, nb) = engine_and_notebook().await;
    let store = LanceVectorStore::new(dir.path(), pool.clone());

    // Never added → no active row.
    store
        .retire_coordinate(&nb, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM)
        .await
        .expect("no-op, not an error");
    assert_eq!(
        row_count(&pool, &nb, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM, None).await,
        0
    );
}

/// A crash AFTER the active→stale demote commits but BEFORE the Lance drop leaves
/// a `stale` row + orphan table that the startup-GC reclaims on reopen.
#[tokio::test]
async fn retire_crash_before_drop_recovered_by_gc() {
    let (dir, engine, pool, nb) = engine_and_notebook().await;
    let store = LanceVectorStore::new(dir.path(), pool.clone());

    store
        .add(
            &nb,
            DEFAULT_EMBED_MODEL_ID,
            DEFAULT_EMBED_DIM,
            make_rows(&nb, 4, DEFAULT_EMBED_DIM),
        )
        .await
        .unwrap();
    let nomic_table = format!("vec__{nb}__nomic_v15__d{DEFAULT_EMBED_DIM}");

    // Arm the crash seam: demote commits, then retire returns early.
    CRASH_AFTER_RETIRE_STALE_BEFORE_LANCE_DROP.store(true, Ordering::SeqCst);
    store
        .retire_coordinate(&nb, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM)
        .await
        .unwrap();

    // Post-crash: a stale row + its table linger.
    assert_eq!(
        row_count(
            &pool,
            &nb,
            DEFAULT_EMBED_MODEL_ID,
            DEFAULT_EMBED_DIM,
            Some("stale")
        )
        .await,
        1
    );
    assert!(table_exists(dir.path(), &nomic_table).await);

    // Reopen → startup-GC sweeps building/stale rows + their physical tables.
    drop(engine);
    let _engine2 = LensEngine::init(dir.path()).await.unwrap();
    assert_eq!(
        row_count(&pool, &nb, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM, None).await,
        0
    );
    assert!(!table_exists(dir.path(), &nomic_table).await);
}
