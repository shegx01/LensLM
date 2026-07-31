//! End-to-end: ingesting a document populates the `sections` outline (#279), so a
//! positional query has real structure to resolve against. Offline — fake embedder.

use lens_core::Tier;
use lens_core::config::{ModelConfig, RetrievalConfig, TierThresholds};
use lens_core::embedder::EmbeddingBackend;
use lens_core::retrieval::Reranker;
use lens_core::retrieval::router::tiered_search;
use lens_core::vector_store::{Coordinate, LanceVectorStore};

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

/// A positional query resolves via Tier-0 against REAL ingested chunk offsets (not
/// hand-aligned fixtures): "chapter 2" scopes to chapter 2's content even though the
/// bulky chapters chunk into multiple parents, some straddling a heading boundary.
#[tokio::test]
async fn positional_query_scopes_to_the_real_chapter() {
    let (dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);
    let nb = engine.create_notebook("book", None, None).await.unwrap();

    // Bulky, marker-tagged chapters so real chunking yields multiple parents, including a
    // parent straddling the Ch1/Ch2 boundary (the M1 span-overlap case).
    let ch1 = "APPLE ".repeat(400);
    let ch2 = "ORANGE ".repeat(400);
    let ch3 = "BANANA ".repeat(400);
    let md = format!("# Chapter 1\n\n{ch1}\n\n# Chapter 2\n\n{ch2}\n\n# Chapter 3\n\n{ch3}\n");
    let src = engine
        .add_text_source(&nb.id, "Book", &md, "markdown")
        .await
        .unwrap()
        .source;
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();

    let pool = engine.pool().await;
    let store = LanceVectorStore::new(dir.path(), pool.clone());
    let coord = Coordinate::new(nb.id.to_string(), EmbeddingBackend::Fastembed, "m", 4);
    let reranker = Reranker::new(dir.path());
    // Tier-0 resolves from SQLite before any vector access, so the query vector is unused.
    let out = tiered_search(
        &pool,
        &store,
        &reranker,
        None,
        &coord,
        "what is the summary of chapter 2?",
        &[0.0f32; 4],
        &ModelConfig::default(),
        10,
        &RetrievalConfig::default(),
        None,
        &TierThresholds::default(),
        None,
        0,
    )
    .await
    .unwrap();

    assert_eq!(
        out.tier,
        Tier::Tier0,
        "positional query must resolve via Tier-0"
    );
    let text: String = out.units.iter().map(|u| u.text.as_str()).collect();
    let oranges = text.matches("ORANGE").count();
    let apples = text.matches("APPLE").count();
    let bananas = text.matches("BANANA").count();
    assert!(
        oranges > 0,
        "chapter 2's content (incl. its opening) must be present"
    );
    assert!(
        oranges > apples && oranges > bananas,
        "chapter 2 dominates the scoped result (oranges={oranges} apples={apples} bananas={bananas})"
    );
}
