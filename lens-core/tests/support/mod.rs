//! Shared test support for the ingest integration suites (`ingest.rs`,
//! `url_ingest.rs`).
//!
//! These helpers were byte-duplicated across the two integration-test binaries;
//! they are consolidated here and pulled in via `mod support;` in each test file.
//! Only the genuinely shared pieces live here — format-specific fixture builders
//! (PDF/DOCX writers, the `test-seam` fake extractors) stay in the test files that
//! own them, where their dev-dependency/feature coupling belongs.
//!
//! As an included module (not its own test binary) some helpers are used by only
//! one of the two suites; `#[allow(dead_code)]` keeps the module warning-clean
//! regardless of which binary compiles it.
#![allow(dead_code)]

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use lens_core::LensEngine;
use lens_core::embedder::{CountingEmbedder, Embedder};
use tempfile::TempDir;
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// Engine construction
// ---------------------------------------------------------------------------

/// Builds a file-backed engine over a fresh temp dir. Ingest tests need a
/// file-backed engine (text sources are written under `{data_dir}/sources/`),
/// not the in-memory `for_test()`.
pub async fn file_engine() -> (TempDir, LensEngine) {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = LensEngine::init(dir.path()).await.expect("engine init");
    (dir, engine)
}

/// Injects a `CountingEmbedder` into an existing engine so the embedder never
/// downloads the ~130 MB model. The engine's `OnceCell` is pre-filled, so every
/// ingest reuses this one embedder.
pub fn inject_fake_embedder(engine: &LensEngine) {
    let load_count = Arc::new(AtomicUsize::new(0));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let e: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new(load_count, in_flight));
    engine
        .set_embedder_for_test(e)
        .expect("embedder not yet initialized");
}

/// Builds a file-backed engine with an injected `CountingEmbedder` so ingest
/// tests avoid the 130 MB model (they still need the tokenizer for chunking).
pub async fn inject_counting_engine() -> (TempDir, LensEngine) {
    let (dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);
    (dir, engine)
}

// ---------------------------------------------------------------------------
// Tokenizer seeding / availability
// ---------------------------------------------------------------------------

/// Attempts to load the nomic tokenizer the ingest pipeline would use: from the
/// `NOMIC_TOKENIZER_PATH` env var (fast offline path) or by performing the
/// pipeline's own download into `data_dir`. Returns `None` if neither works
/// (offline + no cached tokenizer) so tokenizer-dependent tests skip cleanly.
pub async fn tokenizer_for(data_dir: &Path) -> Option<Tokenizer> {
    if let Ok(path) = std::env::var("NOMIC_TOKENIZER_PATH")
        && let Ok(t) = Tokenizer::from_file(&path)
    {
        // Seed the engine's expected location too, so a subsequent ingest in
        // the same data dir does not re-download.
        let dest = data_dir
            .join("models")
            .join("fastembed")
            .join("tokenizer.json");
        if let Some(parent) = dest.parent() {
            let _ = std::fs::create_dir_all(parent);
            let _ = std::fs::copy(&path, &dest);
        }
        return Some(t);
    }
    download_tokenizer_into(data_dir).await
}

/// Best-effort: download the nomic `tokenizer.json` into the engine's fastembed
/// cache so the ingest pipeline finds it without a second fetch. Returns the
/// loaded tokenizer, or `None` on any network failure.
pub async fn download_tokenizer_into(data_dir: &Path) -> Option<Tokenizer> {
    let url = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json";
    let dest = data_dir
        .join("models")
        .join("fastembed")
        .join("tokenizer.json");
    if dest.is_file() {
        return Tokenizer::from_file(&dest).ok();
    }
    std::fs::create_dir_all(dest.parent()?).ok()?;
    let bytes = reqwest::get(url).await.ok()?.bytes().await.ok()?;
    std::fs::write(&dest, &bytes).ok()?;
    Tokenizer::from_file(&dest).ok()
}

/// True if a tokenizer is reachable (env path or network). Used to skip
/// tokenizer-dependent tests cleanly when offline with no cached tokenizer.
pub async fn tokenizer_available() -> bool {
    let dir = tempfile::tempdir().expect("tempdir");
    tokenizer_for(dir.path()).await.is_some()
}

/// Seeds the engine's fastembed tokenizer cache from `NOMIC_TOKENIZER_PATH` (if
/// set) so an ingest in `data_dir` does not attempt a network download. A no-op
/// when the env var is unset or the copy fails (the test then relies on the
/// pipeline's own best-effort download / skips offline).
pub fn seed_tokenizer_from_env(data_dir: &Path) {
    if let Ok(path_str) = std::env::var("NOMIC_TOKENIZER_PATH") {
        let dest = data_dir
            .join("models")
            .join("fastembed")
            .join("tokenizer.json");
        if let Some(parent) = dest.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::copy(&path_str, &dest);
    }
}

// ---------------------------------------------------------------------------
// Lance vector-store readers
// ---------------------------------------------------------------------------

/// Returns the set of chunk ids stored in Lance for `source_id`. Reads the
/// physical table directly via a fresh lancedb connection to avoid coupling to
/// the (private) store internals. Returns an empty set when the store / table
/// does not exist yet (a never-ingested source).
pub async fn vector_chunk_ids(
    data_dir: &Path,
    notebook: &str,
    source_id: &str,
) -> std::collections::HashSet<String> {
    use arrow_array::StringArray;
    use futures_util::TryStreamExt;
    use lancedb::query::{ExecutableQuery, QueryBase};

    let root = data_dir.join("lancedb");
    let conn = match lancedb::connect(root.to_string_lossy().as_ref())
        .execute()
        .await
    {
        Ok(c) => c,
        Err(_) => return std::collections::HashSet::new(),
    };
    let table_name = format!("vec__{notebook}__nomic_v15__d{}", lens_core::EMBED_DIM);
    let names = conn.table_names().execute().await.unwrap_or_default();
    if !names.iter().any(|n| n == &table_name) {
        return std::collections::HashSet::new();
    }
    let table = conn.open_table(&table_name).execute().await.unwrap();
    let stream = table
        .query()
        .only_if(format!("source_id = '{source_id}'"))
        .execute()
        .await
        .unwrap();
    let batches: Vec<_> = stream.try_collect().await.unwrap();
    let mut ids = std::collections::HashSet::new();
    for batch in &batches {
        let col = batch
            .column_by_name("chunk_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..batch.num_rows() {
            ids.insert(col.value(i).to_string());
        }
    }
    ids
}

/// Counts the Lance vector rows for a given source. Reads the physical table
/// directly (search-by-source is not a trait method) and is robust to a missing
/// store/table (returns 0), which the never-ingested-source tests rely on.
pub async fn vector_row_count(data_dir: &Path, notebook: &str, source_id: &str) -> usize {
    vector_chunk_ids(data_dir, notebook, source_id).await.len()
}
