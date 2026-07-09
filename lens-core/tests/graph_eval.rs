//! M13 #158a — entity-graph eval harness + `graph_retrieval_enabled` flag.
//!
//! Integration guards for the per-notebook opt-in override. The measurement-core
//! unit tests (`recall_at_k`, `graph_arm` merge order, non-circular gold) live
//! inline in `lens_core::eval`; the eval always runs both arms and never mutates
//! the flag.

use lens_core::LensEngine;

/// Deleting a notebook cascades away its #158a eval rows (`ON DELETE CASCADE` on
/// the `notebook_id` FKs) — the repo's zero-orphan convention (cf.
/// `purge_notebook_cascades_graph_rows`).
#[tokio::test]
async fn purge_notebook_cascades_eval_rows() {
    let engine = LensEngine::for_test().await;
    let nb = engine
        .create_notebook("eval-cascade", None, None)
        .await
        .unwrap();
    let pool = engine.pool().await;
    let now = chrono::Utc::now().to_rfc3339();

    sqlx::query(
        "INSERT INTO eval_questions \
         (id, notebook_id, kind, question, seed_entities, gold_chunk_ids, prompt_version, created_at) \
         VALUES (?, ?, 'bridging', 'q?', '[]', '[\"c1\"]', '158a-qa-v2', ?)",
    )
    .bind("eq-1")
    .bind(nb.id.as_str())
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO notebook_eval_log \
         (id, notebook_id, ran_at, graph_recall, hybrid_recall, delta_pp, p95_ms, \
          passed, sample_n, dropped_n, graph_enabled, prompt_version, created_at) \
         VALUES (?, ?, ?, 0.9, 0.7, 20.0, 42.0, 1, 20, 1, 0, '158a-qa-v2', ?)",
    )
    .bind("log-1")
    .bind(nb.id.as_str())
    .bind(&now)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    engine.trash_notebook(&nb.id).await.unwrap();
    engine.purge_notebook(&nb.id).await.unwrap();

    let q: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM eval_questions")
        .fetch_one(&pool)
        .await
        .unwrap();
    let l: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM notebook_eval_log")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(q, 0, "eval_questions cascaded on notebook purge");
    assert_eq!(l, 0, "notebook_eval_log cascaded on notebook purge");
}

/// The per-notebook override round-trips and `None` inherits the app-wide default
/// (which is OFF by default, #158a). Setting `Some(true)`/`Some(false)` overrides;
/// clearing with `None` reverts to inheritance.
#[tokio::test]
async fn graph_retrieval_flag_round_trips_and_null_inherits_app_default() {
    let engine = LensEngine::for_test().await;
    let nb = engine
        .create_notebook("graph-flag", None, None)
        .await
        .unwrap();

    // Fresh notebook: column NULL → inherit app default (false).
    assert!(
        !engine
            .notebook_graph_retrieval_enabled(&nb.id)
            .await
            .unwrap(),
        "unset override inherits the OFF app default"
    );

    // Override ON, then OFF.
    engine
        .set_notebook_graph_retrieval_enabled(&nb.id, Some(true))
        .await
        .unwrap();
    assert!(
        engine
            .notebook_graph_retrieval_enabled(&nb.id)
            .await
            .unwrap()
    );
    engine
        .set_notebook_graph_retrieval_enabled(&nb.id, Some(false))
        .await
        .unwrap();
    assert!(
        !engine
            .notebook_graph_retrieval_enabled(&nb.id)
            .await
            .unwrap()
    );

    // Clearing the override reverts to inheriting the app default.
    engine
        .set_notebook_graph_retrieval_enabled(&nb.id, None)
        .await
        .unwrap();
    assert!(
        !engine
            .notebook_graph_retrieval_enabled(&nb.id)
            .await
            .unwrap(),
        "cleared override re-inherits the app default"
    );
}
