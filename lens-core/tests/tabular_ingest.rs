//! End-to-end ingest tests for the tabular source family (XLSX/XLS/CSV — issue
//! #76): a `.xlsx` / `.csv` file ingests through to chunks + embeddings, the
//! `.extracted.txt` and `.tables.md` siblings are written, the pipe-delimited
//! `.tables.md` content NEVER leaks into `.extracted.txt` or any embedded chunk
//! text, and purge removes both siblings.
//!
//! Tokenizer-dependent (chunking needs the nomic tokenizer); these tests skip
//! cleanly when offline with no cached tokenizer (mirrors `ingest.rs`).

use std::path::Path;

use lens_core::LensEngine;
use sqlx::Row;

mod support;
use support::{inject_counting_engine, tokenizer_available, vector_row_count};

/// Copies a committed fixture into `dir` under its basename and returns the path
/// (so `add_file_source` ingests a real on-disk file with the right extension).
fn stage_fixture(dir: &Path, name: &str) -> std::path::PathBuf {
    let src = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name);
    let dest = dir.join(name);
    std::fs::copy(&src, &dest).expect("copy fixture");
    dest
}

/// Reads the text of every chunk for a source from SQLite.
async fn chunk_texts(engine: &LensEngine, source_id: &str) -> Vec<String> {
    let pool = engine.pool().await;
    let rows = sqlx::query("SELECT text FROM chunks WHERE source_id = ?")
        .bind(source_id)
        .fetch_all(&pool)
        .await
        .expect("query chunk texts");
    rows.iter().map(|r| r.get::<String, _>("text")).collect()
}

fn extracted_sibling(data_dir: &Path, id: &str) -> std::path::PathBuf {
    data_dir.join("sources").join(format!("{id}.extracted.txt"))
}

fn tables_sibling(data_dir: &Path, id: &str) -> std::path::PathBuf {
    data_dir.join("sources").join(format!("{id}.tables.md"))
}

#[tokio::test]
async fn ingest_xlsx_end_to_end() {
    if !tokenizer_available().await {
        eprintln!("skipping ingest_xlsx_end_to_end: no tokenizer (offline)");
        return;
    }
    let (dir, engine) = inject_counting_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine.create_notebook("xlsx-nb", None, None).await.unwrap();

    let path = stage_fixture(dir.path(), "sample.xlsx");
    let src = engine
        .add_file_source(&nb.id, &path, Some("People & Cities"))
        .await
        .expect("add xlsx source");

    engine
        .ingest_source(&src.id, |_p| {})
        .await
        .expect("ingest");

    // Chunks created in the vector store.
    let vecs = vector_row_count(&data_dir, &nb.id.to_string(), &src.id).await;
    assert!(vecs > 0, "xlsx ingest must create vectors");

    // `.extracted.txt` sibling exists with verbalized rows.
    let extracted = std::fs::read_to_string(extracted_sibling(&data_dir, &src.id))
        .expect(".extracted.txt sibling must exist");
    assert!(
        extracted.contains("Name: Alice; Age: 30"),
        "verbalized row missing: {extracted:?}"
    );
    assert!(
        extracted.contains("City: NYC; Pop: 8000000"),
        "Cities-sheet row missing: {extracted:?}"
    );

    // `.tables.md` sibling exists with pipe-delimited markdown + per-sheet sections.
    let tables = std::fs::read_to_string(tables_sibling(&data_dir, &src.id))
        .expect(".tables.md sibling must exist");
    assert!(tables.contains("## People"), "People section: {tables:?}");
    assert!(tables.contains("## Cities"), "Cities section: {tables:?}");
    assert!(tables.contains("| Name | Age |"), "header row: {tables:?}");
}

#[tokio::test]
async fn ingest_csv_end_to_end() {
    if !tokenizer_available().await {
        eprintln!("skipping ingest_csv_end_to_end: no tokenizer (offline)");
        return;
    }
    let (dir, engine) = inject_counting_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine.create_notebook("csv-nb", None, None).await.unwrap();

    let path = stage_fixture(dir.path(), "sample.csv");
    let src = engine
        .add_file_source(&nb.id, &path, Some("People CSV"))
        .await
        .expect("add csv source");

    engine
        .ingest_source(&src.id, |_p| {})
        .await
        .expect("ingest");

    let vecs = vector_row_count(&data_dir, &nb.id.to_string(), &src.id).await;
    assert!(vecs > 0, "csv ingest must create vectors");

    let extracted = std::fs::read_to_string(extracted_sibling(&data_dir, &src.id))
        .expect(".extracted.txt sibling must exist");
    assert!(
        extracted.contains("Name: Alice; Age: 30; City: NYC"),
        "verbalized CSV row missing: {extracted:?}"
    );

    let tables = std::fs::read_to_string(tables_sibling(&data_dir, &src.id))
        .expect(".tables.md sibling must exist");
    assert!(
        tables.contains("| Name | Age | City |"),
        "CSV markdown header: {tables:?}"
    );
}

/// AC-3 core guarantee: the `.tables.md` pipe-delimited content must NOT appear
/// in `.extracted.txt` NOR in any embedded chunk text.
#[tokio::test]
async fn embed_never_tables_md() {
    if !tokenizer_available().await {
        eprintln!("skipping embed_never_tables_md: no tokenizer (offline)");
        return;
    }
    let (dir, engine) = inject_counting_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine
        .create_notebook("embed-never-nb", None, None)
        .await
        .unwrap();

    let path = stage_fixture(dir.path(), "sample.csv");
    let src = engine
        .add_file_source(&nb.id, &path, None)
        .await
        .expect("add csv source");
    engine
        .ingest_source(&src.id, |_p| {})
        .await
        .expect("ingest");

    // The extracted (canonical/embedded) buffer carries NO pipe-delimited markup.
    let extracted = std::fs::read_to_string(extracted_sibling(&data_dir, &src.id))
        .expect(".extracted.txt sibling must exist");
    assert!(
        !extracted.contains('|'),
        ".extracted.txt must not contain pipe-delimited table markup: {extracted:?}"
    );

    // No chunk's embedded text contains a pipe-delimited markdown table row.
    let texts = chunk_texts(&engine, &src.id).await;
    assert!(!texts.is_empty(), "chunks must exist");
    for t in &texts {
        assert!(
            !t.contains("| Name | Age | City |") && !t.contains("| --- |"),
            "chunk text leaked .tables.md markup: {t:?}"
        );
    }

    // Sanity: the `.tables.md` sibling really does contain that markup (so the
    // negative assertions above are meaningful).
    let tables = std::fs::read_to_string(tables_sibling(&data_dir, &src.id))
        .expect(".tables.md sibling must exist");
    assert!(tables.contains("| Name | Age | City |"));
}

/// Purge removes BOTH the `.extracted.txt` and `.tables.md` siblings (AC-4).
#[tokio::test]
async fn purge_tabular_source() {
    if !tokenizer_available().await {
        eprintln!("skipping purge_tabular_source: no tokenizer (offline)");
        return;
    }
    let (dir, engine) = inject_counting_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine
        .create_notebook("purge-tab-nb", None, None)
        .await
        .unwrap();

    let path = stage_fixture(dir.path(), "sample.csv");
    let src = engine
        .add_file_source(&nb.id, &path, None)
        .await
        .expect("add csv source");
    engine
        .ingest_source(&src.id, |_p| {})
        .await
        .expect("ingest");

    let extracted = extracted_sibling(&data_dir, &src.id);
    let tables = tables_sibling(&data_dir, &src.id);
    assert!(
        extracted.is_file(),
        ".extracted.txt must exist before purge"
    );
    assert!(tables.is_file(), ".tables.md must exist before purge");

    engine.trash_source(&src.id).await.expect("trash");
    engine.purge_source(&src.id).await.expect("purge");

    assert!(
        !extracted.exists(),
        ".extracted.txt sibling must be removed on purge"
    );
    assert!(
        !tables.exists(),
        ".tables.md sibling must be removed on purge"
    );
}

/// Notebook-cascade purge (AC-4 via the whole-notebook path): trashing + purging
/// the NOTEBOOK that holds a tabular source must reclaim BOTH the
/// `.extracted.txt` AND `.tables.md` siblings, mirroring the URL cascade test in
/// `tests/url_ingest.rs::url_extracted_sibling_removed_on_purge_notebook`.
#[tokio::test]
async fn purge_notebook_removes_tables_md() {
    if !tokenizer_available().await {
        eprintln!("skipping purge_notebook_removes_tables_md: no tokenizer (offline)");
        return;
    }
    let (dir, engine) = inject_counting_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine
        .create_notebook("purge-nb-cascade", None, None)
        .await
        .unwrap();

    let path = stage_fixture(dir.path(), "sample.csv");
    let src = engine
        .add_file_source(&nb.id, &path, None)
        .await
        .expect("add csv source");
    engine
        .ingest_source(&src.id, |_p| {})
        .await
        .expect("ingest");

    let extracted = extracted_sibling(&data_dir, &src.id);
    let tables = tables_sibling(&data_dir, &src.id);
    assert!(
        extracted.is_file(),
        ".extracted.txt must exist before purge"
    );
    assert!(tables.is_file(), ".tables.md must exist before purge");

    engine.trash_notebook(&nb.id).await.expect("trash_notebook");
    engine.purge_notebook(&nb.id).await.expect("purge_notebook");

    assert!(
        !extracted.exists(),
        ".extracted.txt sibling must be removed on notebook purge"
    );
    assert!(
        !tables.exists(),
        ".tables.md sibling must be removed on notebook purge"
    );
}
