//! Engine-level embedding model resolution + keyed embedder cache (M4 Phase 4b,
//! Steps 5 & 6).
//!
//! Exercises the keyed embedder cache (`embedder_for`) and the read-path
//! resolver (`resolve_notebook_embedding`) through the `test-util` seams, using
//! the model-free `CountingEmbedder` so no ONNX weights are downloaded.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use lens_core::embedder::{CountingEmbedder, Embedder, EmbeddingBackend, resolve};
use lens_core::{LensEngine, NotebookId};

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
        .set_embedder_for_test(injected, EmbeddingBackend::Fastembed)
        .expect("inject default embedder");

    let a = engine
        .embedder_for_test_get("nomic-embed-text-v1.5", EmbeddingBackend::Fastembed)
        .await
        .expect("embedder_for nomic");
    let b = engine
        .embedder_for_test_get("nomic-embed-text-v1.5", EmbeddingBackend::Fastembed)
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
    engine
        .set_embedder_for_test(nomic, EmbeddingBackend::Fastembed)
        .expect("inject nomic");

    let mxbai_spec = resolve("mxbai-embed-large");
    let mxbai: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new_with_dim(
        mxbai_spec.dim,
        mxbai_spec.id,
        mxbai_spec.prefix_doc,
        mxbai_spec.prefix_query,
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    ));
    engine
        .set_embedder_for_test(mxbai, EmbeddingBackend::Fastembed)
        .expect("inject mxbai");

    let n = engine
        .embedder_for_test_get("nomic-embed-text-v1.5", EmbeddingBackend::Fastembed)
        .await
        .expect("nomic");
    let m = engine
        .embedder_for_test_get("mxbai-embed-large", EmbeddingBackend::Fastembed)
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
// Step 4 (4b-B) — backend-keyed embedder cache (R4/M3)
// ---------------------------------------------------------------------------

/// The SAME model id under DIFFERENT backends must occupy DISTINCT cache slots:
/// `(nomic, Fastembed)` and `(nomic, Ollama)` return DIFFERENT `Arc`s. A
/// model-id-only key would alias the two backends and serve the wrong backend's
/// embedder for a notebook (silent cross-backend vector pollution).
#[tokio::test]
async fn embedder_cache_keys_distinct_per_backend() {
    let engine = LensEngine::for_test().await;

    // Two functionally-distinct CountingEmbedders, both reporting the SAME
    // model_id (nomic) but injected under different backends. The cache key is
    // `(model_id, backend)`, so they must not collide.
    let fastembed: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new(
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    ));
    let ollama: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new(
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    ));
    engine
        .set_embedder_for_test(fastembed, EmbeddingBackend::Fastembed)
        .expect("inject nomic/fastembed");
    engine
        .set_embedder_for_test(ollama, EmbeddingBackend::Ollama)
        .expect("inject nomic/ollama");

    let f = engine
        .embedder_for_test_get("nomic-embed-text-v1.5", EmbeddingBackend::Fastembed)
        .await
        .expect("nomic/fastembed");
    let o = engine
        .embedder_for_test_get("nomic-embed-text-v1.5", EmbeddingBackend::Ollama)
        .await
        .expect("nomic/ollama");

    assert_eq!(f.model_id(), o.model_id(), "both report the same model id");
    assert!(
        !Arc::ptr_eq(&f, &o),
        "same model id under different backends must be DISTINCT cached instances"
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

    let (model, dim, _backend) = engine
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

    let (model, dim, _backend) = engine
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

    let (model, dim, _backend) = engine
        .resolve_notebook_embedding(&nb.id)
        .await
        .expect("resolve");
    assert_eq!(model, "nomic-embed-text-v1.5");
    assert_eq!(dim, 768);
}

/// `set_notebook_embedding_model` accepts the frontend's legacy alias
/// `"nomic-embed-text"` and persists the CANONICAL id `"nomic-embed-text-v1.5"`
/// (the TS `EMBEDDING_MODELS` uses the Ollama-facing alias). Guards the API
/// contract for the 4b-B selector UI.
#[tokio::test]
async fn set_embedding_model_accepts_legacy_alias_and_persists_canonical() {
    let engine = LensEngine::for_test().await;
    let nb = engine
        .create_notebook("Alias NB", None, None)
        .await
        .expect("create notebook");

    engine
        .set_notebook_embedding_model(&nb.id, "nomic-embed-text", EmbeddingBackend::Fastembed)
        .await
        .expect("legacy alias accepted");

    let stored: Option<String> =
        sqlx::query_scalar("SELECT embedding_model FROM notebooks WHERE id = ?")
            .bind(nb.id.as_str())
            .fetch_one(&engine.pool().await)
            .await
            .unwrap();
    assert_eq!(stored.as_deref(), Some("nomic-embed-text-v1.5"));
}

/// `set_notebook_embedding_model` rejects a genuinely-unknown id (no silent
/// fallback to nomic).
#[tokio::test]
async fn set_embedding_model_rejects_unknown_id() {
    let engine = LensEngine::for_test().await;
    let nb = engine
        .create_notebook("Reject NB", None, None)
        .await
        .expect("create notebook");

    let err = engine
        .set_notebook_embedding_model(&nb.id, "totally-made-up-model", EmbeddingBackend::Fastembed)
        .await
        .expect_err("unknown id rejected");
    assert!(format!("{err}").contains("unknown embedding model id"));
}

/// AC7 — a NEW notebook adopts the app-wide GLOBAL default coordinate set in
/// Settings (`AppConfig::embedding_model` + `embedding_backend`), NOT the
/// compile-time `nomic`/`fastembed` consts. Setting the global default to a
/// non-default model+backend and then creating a notebook must stamp that pair,
/// so `get_notebook_embedding_info` reports it.
#[tokio::test]
async fn new_notebook_adopts_global_default_model_and_backend() {
    let engine = LensEngine::for_test().await;

    // Set the global default to a NON-default model + backend.
    let mut cfg = engine.config().await;
    cfg.embedding_model = "mxbai-embed-large".to_string();
    cfg.embedding_backend = "ollama".to_string();
    engine.set_config(cfg).await;

    let nb = engine
        .create_notebook("Adopts Default NB", None, None)
        .await
        .expect("create notebook");

    let (model, dim, backend, _status) = engine
        .get_notebook_embedding_info(&nb.id)
        .await
        .expect("embedding info");
    assert_eq!(
        model, "mxbai-embed-large",
        "adopts configured default model"
    );
    assert_eq!(dim, 1024);
    assert_eq!(
        backend,
        EmbeddingBackend::Ollama,
        "adopts configured default backend"
    );
}

/// AC7 (negative) — with an UNSET global default (the fresh-install state), a new
/// notebook still gets the registry/enum default (`nomic-embed-text-v1.5` /
/// `fastembed`), preserving the prior compile-time-const behavior.
#[tokio::test]
async fn new_notebook_with_unset_global_default_uses_registry_default() {
    let engine = LensEngine::for_test().await;
    // `for_test` config has empty embedding_model/backend (fresh-install state).
    let nb = engine
        .create_notebook("Default NB", None, None)
        .await
        .expect("create notebook");

    let (model, dim, backend, _status) = engine
        .get_notebook_embedding_info(&nb.id)
        .await
        .expect("embedding info");
    assert_eq!(model, "nomic-embed-text-v1.5");
    assert_eq!(dim, 768);
    assert_eq!(backend, EmbeddingBackend::Fastembed);
}

/// `resolve_notebook_embedding` fails fast for a non-existent notebook (rather
/// than silently returning the default), so callers get a clear error.
#[tokio::test]
async fn resolve_notebook_embedding_errors_for_missing_notebook() {
    let engine = LensEngine::for_test().await;
    let missing = NotebookId::from("no-such-notebook".to_string());
    let err = engine
        .resolve_notebook_embedding(&missing)
        .await
        .expect_err("missing notebook errors");
    assert!(format!("{err}").contains("no notebook with id"));
}
