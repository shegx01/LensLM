//! End-to-end: ingesting a document populates the `sections` outline (#279), so a
//! positional query has real structure to resolve against. Offline — fake embedder.

mod support;
use support::{file_engine, inject_fake_embedder};

/// A Markdown source's headings become `sections` rows with the right level, ordinal,
/// and title after a full ingest — exercising the extractor → `build_sections` →
/// `insert_sections` wiring.
#[tokio::test]
async fn ingesting_markdown_populates_the_sections_outline() {
    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);
    let nb = engine.create_notebook("book-nb", None, None).await.unwrap();
    let src = engine
        .add_text_source(
            &nb.id,
            "Book",
            "# Chapter 1\n\nApples are red.\n\n# Chapter 2\n\nOranges are orange.\n\n## Background\n\nCitrus history here.\n",
            "markdown",
        )
        .await
        .unwrap()
        .source;
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();

    let pool = engine.pool().await;
    let rows: Vec<(i64, i64, String)> = sqlx::query_as(
        "SELECT level, ordinal, title FROM sections WHERE source_id = ? ORDER BY char_start",
    )
    .bind(&src.id)
    .fetch_all(&pool)
    .await
    .unwrap();

    assert_eq!(
        rows,
        vec![
            (1, 1, "Chapter 1".to_string()),
            (1, 2, "Chapter 2".to_string()),
            (2, 1, "Background".to_string()),
        ],
        "markdown headings become level/ordinal-correct sections"
    );
}

/// Re-ingesting the same source replaces its outline rather than duplicating it
/// (the `delete_sections_for_source` + `insert_sections` pair in the ingest tx).
#[tokio::test]
async fn re_ingest_replaces_the_outline() {
    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);
    let nb = engine.create_notebook("book-nb", None, None).await.unwrap();
    let src = engine
        .add_text_source(&nb.id, "Book", "# Alpha\n\nbody one.\n", "markdown")
        .await
        .unwrap()
        .source;
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();

    let pool = engine.pool().await;
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM sections WHERE source_id = ?")
        .bind(&src.id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 1, "re-ingest must not duplicate outline rows");
}
