//! #158b — engine-level tests for the eval read + on-demand run methods
//! (`latest_notebook_eval`, `run_graph_eval`). OFFLINE: an injected
//! [`CountingEmbedder`] (no ONNX download) and a mock [`ScriptedProvider`] for the
//! LLM; `notebook_eval_log` rows are inserted by hand for the read-back tests.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use lens_core::embedder::{CountingEmbedder, Embedder, EmbeddingBackend};
use lens_core::enrichment::test_util::ScriptedProvider;
use lens_core::eval::{EvalOutcome, EvalPhase};
use lens_core::{LensEngine, LensError, NotebookId};
use sqlx::SqlitePool;
use tempfile::TempDir;

async fn engine() -> (TempDir, LensEngine, SqlitePool, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = LensEngine::init(dir.path()).await.expect("engine init");
    let nb = engine
        .create_notebook("eval-nb", None, None)
        .await
        .expect("create notebook");
    let pool = engine.pool().await;
    (dir, engine, pool, nb.id.to_string())
}

/// Inserts one `notebook_eval_log` row for `notebook_id` at the given `ran_at`.
#[allow(clippy::too_many_arguments)]
async fn insert_log(
    pool: &SqlitePool,
    notebook_id: &str,
    ran_at: &str,
    graph_recall: f64,
    hybrid_recall: f64,
    delta_pp: f64,
    p95_ms: f64,
    passed: bool,
    sample_n: i64,
    dropped_n: i64,
    graph_enabled: bool,
) {
    sqlx::query(
        "INSERT INTO notebook_eval_log \
         (id, notebook_id, ran_at, graph_recall, hybrid_recall, delta_pp, p95_ms, passed, \
          sample_n, dropped_n, graph_enabled, prompt_version, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(uuid::Uuid::now_v7().to_string())
    .bind(notebook_id)
    .bind(ran_at)
    .bind(graph_recall)
    .bind(hybrid_recall)
    .bind(delta_pp)
    .bind(p95_ms)
    .bind(passed as i64)
    .bind(sample_n)
    .bind(dropped_n)
    .bind(graph_enabled as i64)
    .bind("158a-qa-v2")
    .bind(ran_at)
    .execute(pool)
    .await
    .expect("insert log");
}

async fn insert_source_with_chunks(
    pool: &SqlitePool,
    notebook_id: &str,
    source_tag: &str,
    chunk_count: usize,
) {
    let now = chrono::Utc::now().to_rfc3339();
    let source_id = format!("src-{source_tag}");
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
         content_hash, created_at) \
         VALUES (?, ?, 'text', ?, 'indexed', ?, 1, ?, ?)",
    )
    .bind(&source_id)
    .bind(notebook_id)
    .bind(source_tag)
    .bind(format!("/tmp/{source_tag}.md"))
    .bind(format!("hash-{source_id}"))
    .bind(&now)
    .execute(pool)
    .await
    .expect("insert source");

    for i in 0..chunk_count {
        let chunk_id = format!("{source_id}-c{i}");
        let text = format!("Chunk {i} of {source_tag}.");
        sqlx::query(
            "INSERT INTO chunks \
             (id, source_id, parent_id, kind, level, section_path, text, \
              token_start, token_end, char_start, char_end, block_type, created_at) \
             VALUES (?, ?, NULL, 'child', 1, 'Intro', ?, 0, 1, 0, ?, 'paragraph', ?)",
        )
        .bind(&chunk_id)
        .bind(&source_id)
        .bind(&text)
        .bind(text.len() as i64)
        .bind(&now)
        .execute(pool)
        .await
        .expect("insert chunk");
    }
}

/// Pins the notebook to an offline `all-minilm` embedder (accelerate_hint=false →
/// CPU for both Interactive and Bulk, so the Bulk-keyed injection is found by the
/// Interactive lookup `run_graph_eval` uses).
async fn inject_offline_embedder(engine: &LensEngine, nb: &NotebookId) {
    engine
        .set_notebook_embedding_model(nb, "all-minilm", EmbeddingBackend::Fastembed)
        .await
        .expect("pin embedding model");
    let (_model_id, dim, _backend) = engine
        .resolve_notebook_embedding(nb)
        .await
        .expect("resolve embedding");
    let injected: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new_with_dim(
        dim,
        "all-minilm",
        "",
        "",
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    ));
    engine
        .set_embedder_for_test(injected, EmbeddingBackend::Fastembed)
        .expect("inject embedder");
}

// ---------------------------------------------------------------------------
// (a) latest_notebook_eval read-back
// ---------------------------------------------------------------------------

#[tokio::test]
async fn latest_notebook_eval_reads_newest_row() {
    let (_dir, engine, pool, nb) = engine().await;
    let nb_id: NotebookId = nb.clone().into();

    // No rows → None.
    assert!(
        engine
            .latest_notebook_eval(&nb_id)
            .await
            .expect("read")
            .is_none(),
        "no eval row yet → None"
    );

    // Older row, then a newer one; the newest by `ran_at` must win.
    insert_log(
        &pool,
        &nb,
        "2026-07-10T00:00:00Z",
        0.10,
        0.20,
        -10.0,
        100.0,
        false,
        10,
        1,
        false,
    )
    .await;
    insert_log(
        &pool,
        &nb,
        "2026-07-12T00:00:00Z",
        0.80,
        0.55,
        25.0,
        420.0,
        true,
        24,
        3,
        true,
    )
    .await;

    let latest = engine
        .latest_notebook_eval(&nb_id)
        .await
        .expect("read")
        .expect("row present");
    assert_eq!(latest.ran_at, "2026-07-12T00:00:00Z", "newest by ran_at");
    assert_eq!(latest.report.sample_n, 24);
    assert_eq!(latest.report.dropped_n, 3);
    assert!(latest.report.passed);
    assert!(latest.report.graph_enabled);
    assert!((latest.report.graph_recall - 0.80).abs() < 1e-6);
    assert!((latest.report.hybrid_recall - 0.55).abs() < 1e-6);
    assert!((latest.report.delta_pp - 25.0).abs() < 1e-4);
    assert!((latest.report.p95_ms - 420.0).abs() < 1e-4);
    assert_eq!(latest.report.prompt_version, "158a-qa-v2");
}

// ---------------------------------------------------------------------------
// (b) run_graph_eval with no provider → typed error, nothing written
// ---------------------------------------------------------------------------

#[tokio::test]
async fn run_graph_eval_without_provider_returns_typed_error() {
    let (_dir, engine, pool, nb) = engine().await;
    let nb_id: NotebookId = nb.clone().into();

    let mut phases: Vec<EvalPhase> = Vec::new();
    let err = engine
        .run_graph_eval(&nb_id, |p| phases.push(p))
        .await
        .expect_err("no provider must error");
    assert!(
        matches!(err, LensError::Model(_)),
        "no-provider must be a Model error, got {err:?}"
    );
    assert!(phases.is_empty(), "no phase emitted before the pre-check");

    // Nothing was written.
    let logged: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM notebook_eval_log WHERE notebook_id = ?")
            .bind(&nb)
            .fetch_one(&pool)
            .await
            .expect("count");
    assert_eq!(logged, 0, "no log row on the no-provider path");
}

#[tokio::test]
async fn run_graph_eval_with_unreachable_provider_returns_typed_error() {
    let (_dir, engine, _pool, nb) = engine().await;
    let nb_id: NotebookId = nb.clone().into();

    let (dead, _calls) = ScriptedProvider::dead();
    engine.set_llm_provider(Some(Arc::new(dead))).await;

    let mut phases: Vec<EvalPhase> = Vec::new();
    let err = engine
        .run_graph_eval(&nb_id, |p| phases.push(p))
        .await
        .expect_err("unreachable provider must error");
    assert!(
        matches!(err, LensError::Model(_)),
        "unreachable provider must be a Model error, got {err:?}"
    );
    assert!(
        phases.is_empty(),
        "no phase emitted before the reachability pre-check"
    );
}

// ---------------------------------------------------------------------------
// (c) Skipped path — under the sample floor, provider present + reachable
// ---------------------------------------------------------------------------

#[tokio::test]
async fn run_graph_eval_below_floor_is_skipped() {
    let (_dir, engine, _pool, nb) = engine().await;
    let nb_id: NotebookId = nb.clone().into();

    inject_offline_embedder(&engine, &nb_id).await;

    // Two sources (< 3): the sample floor rejects inside run_notebook_eval.
    insert_source_with_chunks(&engine.pool().await, &nb, "s0", 5).await;
    insert_source_with_chunks(&engine.pool().await, &nb, "s1", 5).await;

    let (provider, _calls) = ScriptedProvider::new(vec!["[]"]);
    engine.set_llm_provider(Some(Arc::new(provider))).await;

    let mut phases: Vec<EvalPhase> = Vec::new();
    let outcome = engine
        .run_graph_eval(&nb_id, |p| phases.push(p))
        .await
        .expect("eval runs");
    assert!(
        matches!(outcome, EvalOutcome::Skipped { .. }),
        "sub-floor notebook must be Skipped, got {outcome:?}"
    );
    // Both phases fire even on the Skipped path (GeneratingQa before, Done after).
    assert_eq!(phases, vec![EvalPhase::GeneratingQa, EvalPhase::Done]);
}
