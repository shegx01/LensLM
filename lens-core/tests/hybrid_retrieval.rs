//! Integration tests for hybrid retrieval (issue #39): FTS trigger sync on
//! ingest/update/delete, BM25 notebook/source/level scoping, and trashed-source
//! exclusion through BOTH retrieval paths of `hybrid_search`. Offline — hand-built
//! vectors, no model downloads, reranker left disabled.

use lens_core::LensEngine;
use lens_core::config::RetrievalConfig;
use lens_core::embedder::EmbeddingBackend;
use lens_core::retrieval::{Reranker, bm25, hybrid_search};
use lens_core::vector_store::{Coordinate, LanceVectorStore, VectorRow, VectorStore};
use sqlx::SqlitePool;

const DIM: usize = 4;

async fn insert_source(pool: &SqlitePool, notebook_id: &str, source_id: &str) {
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
         content_hash, created_at) \
         VALUES (?, ?, 'text', 'seed', 'indexed', '/tmp/seed.txt', 1, ?, ?)",
    )
    .bind(source_id)
    .bind(notebook_id)
    .bind(format!("hash-{source_id}"))
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(pool)
    .await
    .expect("insert source");
}

async fn insert_chunk(pool: &SqlitePool, source_id: &str, chunk_id: &str, level: i32, text: &str) {
    sqlx::query(
        "INSERT INTO chunks \
         (id, source_id, parent_id, kind, level, section_path, text, \
          token_start, token_end, char_start, char_end, block_type, created_at) \
         VALUES (?, ?, NULL, ?, ?, '[\"Intro\"]', ?, 0, 1, 0, ?, 'paragraph', ?)",
    )
    .bind(chunk_id)
    .bind(source_id)
    .bind(if level == 0 { "parent" } else { "child" })
    .bind(level)
    .bind(text)
    .bind(text.len() as i64)
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(pool)
    .await
    .expect("insert chunk");
}

async fn fts_text_for(pool: &SqlitePool, chunk_id: &str) -> Option<String> {
    sqlx::query_scalar::<_, String>("SELECT text FROM chunks_fts WHERE chunk_id = ?")
        .bind(chunk_id)
        .fetch_optional(pool)
        .await
        .expect("query fts")
}

#[tokio::test]
async fn insert_trigger_populates_fts() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_source(&pool, &nb, "s1").await;
    insert_chunk(&pool, "s1", "c1", 1, "quokka marsupial antarctica").await;

    assert_eq!(
        fts_text_for(&pool, "c1").await.as_deref(),
        Some("quokka marsupial antarctica")
    );
}

#[tokio::test]
async fn update_trigger_syncs_fts() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_source(&pool, &nb, "s1").await;
    insert_chunk(&pool, "s1", "c1", 1, "before text").await;

    sqlx::query("UPDATE chunks SET text = ? WHERE id = ?")
        .bind("after replacement text")
        .bind("c1")
        .execute(&pool)
        .await
        .unwrap();

    assert_eq!(
        fts_text_for(&pool, "c1").await.as_deref(),
        Some("after replacement text")
    );
    // Old term no longer matches; new term does.
    let old = bm25::bm25_search(&pool, &nb, None, None, "before", 10)
        .await
        .unwrap();
    assert!(old.is_empty(), "stale term must not match after update");
    let new = bm25::bm25_search(&pool, &nb, None, None, "replacement", 10)
        .await
        .unwrap();
    assert_eq!(new, vec!["c1".to_string()]);
}

#[tokio::test]
async fn delete_trigger_removes_fts_row() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_source(&pool, &nb, "s1").await;
    insert_chunk(&pool, "s1", "c1", 1, "ephemeral content").await;
    assert!(fts_text_for(&pool, "c1").await.is_some());

    sqlx::query("DELETE FROM chunks WHERE id = ?")
        .bind("c1")
        .execute(&pool)
        .await
        .unwrap();

    assert!(
        fts_text_for(&pool, "c1").await.is_none(),
        "delete trigger must drop the FTS row"
    );
}

#[tokio::test]
async fn bm25_respects_notebook_source_and_level_filters() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;
    let nb1 = engine.create_notebook("nb1", None, None).await.unwrap().id;
    let nb2 = engine.create_notebook("nb2", None, None).await.unwrap().id;
    insert_source(&pool, &nb1, "s1").await;
    insert_source(&pool, &nb1, "s2").await;
    insert_source(&pool, &nb2, "s3").await;
    insert_chunk(&pool, "s1", "c1", 0, "voyager golden record parent").await;
    insert_chunk(&pool, "s1", "c2", 1, "voyager golden record child").await;
    insert_chunk(&pool, "s2", "c3", 1, "voyager golden record other source").await;
    insert_chunk(&pool, "s3", "c4", 1, "voyager golden record other notebook").await;

    // Notebook scope: nb2's chunk never appears in an nb1 search.
    let nb1_hits = bm25::bm25_search(&pool, &nb1, None, None, "voyager record", 10)
        .await
        .unwrap();
    assert_eq!(nb1_hits.len(), 3);
    assert!(!nb1_hits.contains(&"c4".to_string()));

    // Source filter.
    let s1_hits = bm25::bm25_search(&pool, &nb1, Some("s1"), None, "voyager", 10)
        .await
        .unwrap();
    assert_eq!(s1_hits.len(), 2);
    assert!(s1_hits.contains(&"c1".to_string()) && s1_hits.contains(&"c2".to_string()));

    // Level filter (children only).
    let child_hits = bm25::bm25_search(&pool, &nb1, None, Some(1), "voyager", 10)
        .await
        .unwrap();
    assert_eq!(child_hits.len(), 2);
    assert!(child_hits.contains(&"c2".to_string()) && child_hits.contains(&"c3".to_string()));
    assert!(!child_hits.contains(&"c1".to_string()));
}

#[tokio::test]
async fn bm25_excludes_trashed_sources() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_source(&pool, &nb, "s1").await;
    insert_chunk(&pool, "s1", "c1", 1, "acronym NASA JPL rare token").await;

    let before = bm25::bm25_search(&pool, &nb, None, None, "NASA JPL", 10)
        .await
        .unwrap();
    assert_eq!(before, vec!["c1".to_string()]);

    engine.trash_source("s1").await.unwrap();
    let trashed = bm25::bm25_search(&pool, &nb, None, None, "NASA JPL", 10)
        .await
        .unwrap();
    assert!(trashed.is_empty(), "trashed source must not appear in BM25");

    engine.restore_source("s1").await.unwrap();
    let restored = bm25::bm25_search(&pool, &nb, None, None, "NASA JPL", 10)
        .await
        .unwrap();
    assert_eq!(restored, vec!["c1".to_string()]);
}

#[tokio::test]
async fn purge_source_cleans_fts_rows() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_source(&pool, &nb, "s1").await;
    insert_chunk(&pool, "s1", "c1", 1, "purge me").await;
    assert!(fts_text_for(&pool, "c1").await.is_some());

    engine.trash_source("s1").await.unwrap();
    engine.purge_source("s1").await.unwrap();

    assert!(
        fts_text_for(&pool, "c1").await.is_none(),
        "purge_source must clean orphan chunks_fts rows"
    );
}

/// A hand-built unit vector aligned to a one-hot axis so cosine distance is
/// deterministic; the query vector picks out the intended chunk.
fn axis_vec(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[axis] = 1.0;
    v
}

#[tokio::test]
async fn hybrid_search_excludes_trashed_via_both_paths() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let data_dir = dir.path().to_path_buf();
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_source(&pool, &nb, "s1").await;
    // Lexical term "quokka" is rare so BM25 finds it; the vector is axis-0 so the
    // axis-0 query also finds it densely — the chunk is reachable via BOTH paths.
    insert_chunk(&pool, "s1", "c1", 1, "quokka rare lexical token").await;

    let store = LanceVectorStore::new(&data_dir, pool.clone());
    let coord = Coordinate::new(nb.to_string(), EmbeddingBackend::Fastembed, "m", DIM);
    store
        .add(
            &coord,
            vec![VectorRow {
                chunk_id: "c1".to_string(),
                source_id: "s1".to_string(),
                notebook_id: nb.to_string(),
                level: 1,
                vector: axis_vec(0),
            }],
        )
        .await
        .unwrap();

    let reranker = Reranker::new(&data_dir);
    let cfg = RetrievalConfig::default();
    let qvec = axis_vec(0);

    let hits = hybrid_search(
        &pool, &store, &reranker, &coord, "quokka", &qvec, None, None, 10, &cfg,
    )
    .await
    .unwrap();
    assert_eq!(
        hits.iter().map(|h| h.chunk_id.as_str()).collect::<Vec<_>>(),
        vec!["c1"],
        "live source reachable via both dense and bm25"
    );

    // Trash: BOTH paths must now exclude it (dense post-filter + bm25 JOIN).
    engine.trash_source("s1").await.unwrap();
    let trashed = hybrid_search(
        &pool, &store, &reranker, &coord, "quokka", &qvec, None, None, 10, &cfg,
    )
    .await
    .unwrap();
    assert!(
        trashed.is_empty(),
        "trashed source must vanish from hybrid_search via both paths, got {trashed:?}"
    );

    // Restore: reachable again.
    engine.restore_source("s1").await.unwrap();
    let restored = hybrid_search(
        &pool, &store, &reranker, &coord, "quokka", &qvec, None, None, 10, &cfg,
    )
    .await
    .unwrap();
    assert_eq!(
        restored
            .iter()
            .map(|h| h.chunk_id.as_str())
            .collect::<Vec<_>>(),
        vec!["c1"]
    );
}

#[tokio::test]
async fn hybrid_disabled_degrades_to_dense_only() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let data_dir = dir.path().to_path_buf();
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_source(&pool, &nb, "s1").await;
    // "c_lex" is only findable lexically (no vector added); "c_dense" only densely.
    insert_chunk(&pool, "s1", "c_lex", 1, "singularterm lexical only").await;
    insert_chunk(&pool, "s1", "c_dense", 1, "generic body").await;

    let store = LanceVectorStore::new(&data_dir, pool.clone());
    let coord = Coordinate::new(nb.to_string(), EmbeddingBackend::Fastembed, "m", DIM);
    store
        .add(
            &coord,
            vec![VectorRow {
                chunk_id: "c_dense".to_string(),
                source_id: "s1".to_string(),
                notebook_id: nb.to_string(),
                level: 1,
                vector: axis_vec(0),
            }],
        )
        .await
        .unwrap();

    let reranker = Reranker::new(&data_dir);
    let cfg = RetrievalConfig {
        hybrid_enabled: false,
        ..RetrievalConfig::default()
    };
    let qvec = axis_vec(0);

    let hits = hybrid_search(
        &pool,
        &store,
        &reranker,
        &coord,
        "singularterm",
        &qvec,
        None,
        None,
        10,
        &cfg,
    )
    .await
    .unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| h.chunk_id.as_str()).collect();
    assert!(
        ids.contains(&"c_dense"),
        "dense-only must return the dense hit"
    );
    assert!(
        !ids.contains(&"c_lex"),
        "hybrid disabled must NOT run the bm25 path"
    );
}
