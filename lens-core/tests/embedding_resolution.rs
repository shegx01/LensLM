//! Engine-level embedding model resolution + keyed embedder cache (M4 Phase 4b,
//! Steps 5 & 6).
//!
//! Exercises the keyed embedder cache (`embedder_for`) and the read-path
//! resolver (`resolve_notebook_embedding`) through the `test-util` seams, using
//! the model-free `CountingEmbedder` so no ONNX weights are downloaded.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use lens_core::LensEngine;
use lens_core::embedder::{CountingEmbedder, Embedder, resolve};

// ---------------------------------------------------------------------------
// Step 5 — keyed embedder cache (R8)
// ---------------------------------------------------------------------------

/// Two `embedder_for(model_id)` calls for the SAME model id return the SAME
/// `Arc` (pointer-equal): the cache holds exactly one instance per key.
#[tokio::test]
async fn embedder_for_caches_same_model_id() {
    let engine = LensEngine::for_test().await;

    // Inject a default (nomic) CountingEmbedder; it registers under its own
    // model_id key.
    let load = Arc::new(AtomicUsize::new(0));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let injected: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new(load, in_flight));
    engine
        .set_embedder_for_test(injected)
        .expect("inject default embedder");

    let a = engine
        .embedder_for_test_get("nomic-embed-text-v1.5")
        .await
        .expect("embedder_for nomic");
    let b = engine
        .embedder_for_test_get("nomic-embed-text-v1.5")
        .await
        .expect("embedder_for nomic again");

    assert!(
        Arc::ptr_eq(&a, &b),
        "same model_id must return the same cached Arc"
    );
    assert_eq!(a.model_id(), "nomic-embed-text-v1.5");
    assert_eq!(a.dim(), 768);
}

/// Different model ids resolve to DISTINCT functional embedders (different dim).
#[tokio::test]
async fn embedder_for_distinct_models_coexist() {
    let engine = LensEngine::for_test().await;

    // Inject nomic (768) under its key and mxbai (1024) under its key.
    let nomic: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new(
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    ));
    engine.set_embedder_for_test(nomic).expect("inject nomic");

    let mxbai_spec = resolve("mxbai-embed-large");
    let mxbai: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new_with_dim(
        mxbai_spec.dim,
        mxbai_spec.id,
        mxbai_spec.prefix_doc,
        mxbai_spec.prefix_query,
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    ));
    engine.set_embedder_for_test(mxbai).expect("inject mxbai");

    let n = engine
        .embedder_for_test_get("nomic-embed-text-v1.5")
        .await
        .expect("nomic");
    let m = engine
        .embedder_for_test_get("mxbai-embed-large")
        .await
        .expect("mxbai");

    assert_eq!(n.dim(), 768);
    assert_eq!(m.dim(), 1024);
    assert_eq!(n.model_id(), "nomic-embed-text-v1.5");
    assert_eq!(m.model_id(), "mxbai-embed-large");
    assert!(
        !Arc::ptr_eq(&n, &m),
        "distinct model ids must be distinct embedders"
    );
}

// ---------------------------------------------------------------------------
// Step 6 — read-path resolver (R1)
// ---------------------------------------------------------------------------

/// A notebook configured with mxbai resolves to ("mxbai-embed-large", 1024).
#[tokio::test]
async fn resolve_notebook_embedding_mxbai() {
    let engine = LensEngine::for_test().await;
    let nb = engine
        .create_notebook("Mxbai NB", None, None)
        .await
        .expect("create notebook");

    sqlx::query("UPDATE notebooks SET embedding_model = ? WHERE id = ?")
        .bind("mxbai-embed-large")
        .bind(nb.id.as_str())
        .execute(&engine.pool().await)
        .await
        .expect("set mxbai");

    let (model, dim) = engine
        .resolve_notebook_embedding(&nb.id)
        .await
        .expect("resolve");
    assert_eq!(model, "mxbai-embed-large");
    assert_eq!(dim, 1024);
}

/// A NULL embedding_model resolves to the default (nomic 768).
#[tokio::test]
async fn resolve_notebook_embedding_null_is_default() {
    let engine = LensEngine::for_test().await;
    let nb = engine
        .create_notebook("Null NB", None, None)
        .await
        .expect("create notebook");

    sqlx::query("UPDATE notebooks SET embedding_model = NULL WHERE id = ?")
        .bind(nb.id.as_str())
        .execute(&engine.pool().await)
        .await
        .expect("null model");

    let (model, dim) = engine
        .resolve_notebook_embedding(&nb.id)
        .await
        .expect("resolve");
    assert_eq!(model, "nomic-embed-text-v1.5");
    assert_eq!(dim, 768);
}

/// An unknown model string resolves to the default (nomic 768).
#[tokio::test]
async fn resolve_notebook_embedding_unknown_is_default() {
    let engine = LensEngine::for_test().await;
    let nb = engine
        .create_notebook("Unknown NB", None, None)
        .await
        .expect("create notebook");

    sqlx::query("UPDATE notebooks SET embedding_model = ? WHERE id = ?")
        .bind("totally-made-up-model")
        .bind(nb.id.as_str())
        .execute(&engine.pool().await)
        .await
        .expect("set unknown");

    let (model, dim) = engine
        .resolve_notebook_embedding(&nb.id)
        .await
        .expect("resolve");
    assert_eq!(model, "nomic-embed-text-v1.5");
    assert_eq!(dim, 768);
}
