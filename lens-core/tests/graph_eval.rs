//! M13 #158a — entity-graph eval harness + `graph_retrieval_enabled` flag.
//!
//! Integration guards for the per-notebook opt-in override. The measurement-core
//! unit tests (`recall_at_k`, `graph_arm` merge order, non-circular gold) live
//! inline in `lens_core::eval`; the eval always runs both arms and never mutates
//! the flag.

use lens_core::LensEngine;

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
