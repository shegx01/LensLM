//! M4 Phase 4b-B Step 6 — crash recovery on a SAME-DIM cross-backend re-embed.
//!
//! A backend switch (fastembed/nomic-v1.5/768 → ollama/nomic-v2-moe/768) re-embeds
//! into the new coordinate, flips it active, then retires the OLD fastembed
//! coordinate. Both sides are 768-dim so the switch is same-dim, but issue #80's
//! strict model↔backend partition means the OLLAMA side must be an ollama-valid
//! id (`nomic-embed-text-v2-moe`, 768) — nomic-v1.5 is fastembed-only. This
//! test arms the global retire crash seam
//! (`CRASH_AFTER_RETIRE_STALE_BEFORE_LANCE_DROP`) so the old-coordinate retire
//! demotes to `stale` and returns early — modeling a process crash between the
//! stale-demote and the Lance table drop. The startup-GC must reclaim the lingering
//! stale fastembed row + table on reopen, and the new ollama coordinate must keep
//! serving search.
//!
//! It lives in its OWN test binary (separate process) so the process-global crash
//! flag never leaks into the parallel re-embed tests in `reembed_notebook.rs`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use lens_core::embedder::{CountingEmbedder, Embedder, resolve};
use lens_core::enrichment::reembed::ReembedOutcome;
use lens_core::vector_store::{
    CRASH_AFTER_RETIRE_STALE_BEFORE_LANCE_DROP, Coordinate, LanceVectorStore, VectorRow,
    VectorStore,
};
use lens_core::{
    DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL_ID, EmbeddingBackend, LensEngine, NotebookId,
};

fn inject_embedder_for(engine: &LensEngine, model_id: &str, backend: EmbeddingBackend) {
    let spec = resolve(model_id);
    let e: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new_with_dim(
        spec.dim,
        spec.id,
        spec.prefix_doc,
        spec.prefix_query,
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    ));
    engine.set_embedder_for_test(e, backend).expect("inject");
}

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

/// Seeds a notebook with one source, two chunks, and a raw fastembed/nomic/768
/// active coordinate.
async fn seed_nomic_notebook(engine: &LensEngine) -> (String, String) {
    let nb = engine
        .create_notebook("crash-nb", None, None)
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
        (&parent_id, None::<&str>, "parent", 0_i32, "Parent body."),
        (
            &child_id,
            Some(parent_id.as_str()),
            "child",
            1,
            "Child body.",
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
            &Coordinate::new(
                nb.clone(),
                EmbeddingBackend::Fastembed,
                DEFAULT_EMBED_MODEL_ID,
                DEFAULT_EMBED_DIM,
            ),
            vec![
                row(&parent_id, &source_id, &nb, 0),
                row(&child_id, &source_id, &nb, 1),
            ],
        )
        .await
        .expect("seed raw active vectors");
    (nb, source_id)
}

async fn coord_count(
    engine: &LensEngine,
    nb: &str,
    backend: EmbeddingBackend,
    model: &str,
    status: &str,
) -> i64 {
    sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM embedding_index \
         WHERE notebook_id = ? AND backend = ? AND model = ? AND dim = ? AND status = ?",
    )
    .bind(nb)
    .bind(backend.as_str())
    .bind(model)
    .bind(DEFAULT_EMBED_DIM as i64)
    .bind(status)
    .fetch_one(&engine.pool().await)
    .await
    .unwrap()
}

/// Crash injection on a SAME-DIM cross-backend switch: a crash between the
/// old-coordinate retire's stale-demote and its Lance table drop leaves a `stale`
/// fastembed row + orphan table. The startup-GC reclaims it on reopen, and the new
/// ollama coordinate keeps serving search.
#[tokio::test]
async fn backend_switch_crash_before_drop_recovered_by_gc() {
    // The ollama-valid, 768-dim target model (issue #80 strict partition: the
    // ollama side cannot be nomic-v1.5, which is fastembed-only).
    const OLLAMA_MODEL: &str = "nomic-embed-text-v2-moe";

    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    inject_embedder_for(&engine, OLLAMA_MODEL, EmbeddingBackend::Ollama);
    let (nb, _src) = seed_nomic_notebook(&engine).await;

    // Switch the notebook to the ollama backend AND the ollama-valid 768-dim model.
    sqlx::query(
        "UPDATE notebooks SET embedding_backend = 'ollama', embedding_model = ? WHERE id = ?",
    )
    .bind(OLLAMA_MODEL)
    .bind(&nb)
    .execute(&engine.pool().await)
    .await
    .unwrap();

    // Arm the retire crash seam: the old-coordinate retire demotes to stale, then
    // returns early BEFORE dropping its Lance table / deleting its row.
    CRASH_AFTER_RETIRE_STALE_BEFORE_LANCE_DROP.store(true, Ordering::SeqCst);
    let outcome = engine
        .reembed_notebook(&NotebookId::from(nb.clone()), |_, _| {})
        .await
        .expect("reembed");
    CRASH_AFTER_RETIRE_STALE_BEFORE_LANCE_DROP.store(false, Ordering::SeqCst);
    assert!(matches!(outcome, ReembedOutcome::Switched { .. }));

    // New ollama coordinate is active; the OLD fastembed row lingers as `stale`.
    assert_eq!(
        coord_count(
            &engine,
            &nb,
            EmbeddingBackend::Ollama,
            OLLAMA_MODEL,
            "active"
        )
        .await,
        1
    );
    assert_eq!(
        coord_count(
            &engine,
            &nb,
            EmbeddingBackend::Fastembed,
            DEFAULT_EMBED_MODEL_ID,
            "stale"
        )
        .await,
        1,
        "the old fastembed coordinate lingers as stale after the simulated crash"
    );

    // Reopen → startup-GC sweeps the lingering stale fastembed row + its table.
    let pool = engine.pool().await;
    drop(engine);
    let engine2 = LensEngine::init(dir.path()).await.unwrap();
    let old_total: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM embedding_index \
         WHERE notebook_id = ? AND backend = 'fastembed' AND model = ? AND dim = ?",
    )
    .bind(&nb)
    .bind(DEFAULT_EMBED_MODEL_ID)
    .bind(DEFAULT_EMBED_DIM as i64)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(old_total, 0, "startup-GC reclaimed the stale fastembed row");

    // The new ollama coordinate still serves search after recovery.
    let store = LanceVectorStore::new(&engine2.data_dir_for_test().await, engine2.pool().await);
    let q = vec![0.1_f32; DEFAULT_EMBED_DIM];
    let hits = store
        .search(
            &Coordinate::new(
                nb.clone(),
                EmbeddingBackend::Ollama,
                OLLAMA_MODEL,
                DEFAULT_EMBED_DIM,
            ),
            &q,
            4,
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 2, "new-backend coordinate serves after GC");
}
