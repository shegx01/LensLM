//! Offline integration tests for the per-source `language` column (#194).
//!
//! Verifies that adding `language` to the `Source` mapping doesn't break any
//! `SELECT ... FROM sources` query; exercises `list_sources` and
//! `list_trashed_sources`. Fully offline — no embedder, tokenizer, or network.

use lens_core::LensEngine;

#[tokio::test]
async fn list_sources_deserializes_with_language_column() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("lang", None, None).await.unwrap();
    let src = engine
        .add_source(&nb.id, "doc.txt", "/abs/doc.txt")
        .await
        .unwrap()
        .source;

    // Pre-migration/undetected: language reads NULL.
    let listed = engine.list_sources(&nb.id).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].language, None);

    // Once detection persists a code, it round-trips through the listing path.
    let pool = engine.pool().await;
    sqlx::query("UPDATE sources SET language = 'deu' WHERE id = ?")
        .bind(&src.id)
        .execute(&pool)
        .await
        .unwrap();

    let listed = engine.list_sources(&nb.id).await.unwrap();
    assert_eq!(listed[0].language.as_deref(), Some("deu"));
}

#[tokio::test]
async fn list_trashed_sources_deserializes_with_language_column() {
    let engine = LensEngine::for_test().await;
    let nb = engine
        .create_notebook("lang-trash", None, None)
        .await
        .unwrap();
    let src = engine
        .add_source(&nb.id, "doc.txt", "/abs/doc.txt")
        .await
        .unwrap()
        .source;

    let pool = engine.pool().await;
    sqlx::query("UPDATE sources SET language = 'eng' WHERE id = ?")
        .bind(&src.id)
        .execute(&pool)
        .await
        .unwrap();
    engine.trash_source(&src.id).await.unwrap();

    // The manual `row.try_get("language")` mapping in `list_trashed_sources`
    // must deserialize the new column (the query_as grep misses this site).
    let trashed = engine.list_trashed_sources().await.unwrap();
    let found = trashed
        .iter()
        .find(|t| t.source.id == src.id)
        .expect("trashed source is listed");
    assert_eq!(found.source.language.as_deref(), Some("eng"));
}
