// issue #71: a spawned instrumented ingest future overflows the default 128-frame
// `Send` auto-trait evaluation (E0275) in this integration-test crate; raise it here too.
#![recursion_limit = "256"]
//! Integration tests for the #237 citation read-back path (`citation_snippet` /
//! `source_view`). Deterministic and offline by default: sources are set up via the
//! real managed-file write (`add_text_source`) or a fabricated `.extracted.txt`
//! sibling, with chunk offsets seeded directly, so AC2 is proven without a
//! tokenizer/model download. One end-to-end test drives the REAL ingest pipeline and
//! is skipped when no nomic tokenizer is reachable (see `tokenizer_available`).

mod support;

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use lens_core::embedder::{CountingEmbedder, Embedder};
use lens_core::{EmbeddingBackend, LensEngine, load_chunk_locators};
use tempfile::TempDir;

/// A file-backed engine with an injected model-free embedder. The tokenizer is
/// left ENABLED (ingest requires it); callers that never ingest are unaffected.
async fn base_engine() -> (TempDir, LensEngine) {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = LensEngine::init(dir.path()).await.expect("engine init");
    let e: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new(
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    ));
    engine
        .set_embedder_for_test(e, EmbeddingBackend::Fastembed)
        .expect("inject embedder");
    (dir, engine)
}

/// Reads the persisted `(chunk_id, char_start, char_end, text)` rows for a source
/// that carry non-null offsets, ordered for determinism.
async fn chunk_offsets(engine: &LensEngine, source_id: &str) -> Vec<(String, i64, i64, String)> {
    let pool = engine.pool().await;
    sqlx::query_as::<_, (String, i64, i64, String)>(
        "SELECT id, char_start, char_end, text FROM chunks \
         WHERE source_id = ? AND char_start IS NOT NULL AND char_end IS NOT NULL \
         ORDER BY char_start, id",
    )
    .bind(source_id)
    .fetch_all(&pool)
    .await
    .expect("read chunk offsets")
}

/// Inserts a derived source row (kind not text-like) whose canonical buffer is the
/// `.extracted.txt` sibling written under `{data_dir}/sources/`.
async fn seed_derived_source(
    engine: &LensEngine,
    data_dir: &Path,
    notebook_id: &str,
    source_id: &str,
    buffer: &str,
) {
    let sources = data_dir.join("sources");
    std::fs::create_dir_all(&sources).expect("sources dir");
    std::fs::write(sources.join(format!("{source_id}.extracted.txt")), buffer)
        .expect("write sibling");
    let pool = engine.pool().await;
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
         content_hash, created_at) \
         VALUES (?, ?, 'pdf', 'Derived Doc', 'indexed', ?, 1, 'h', ?)",
    )
    .bind(source_id)
    .bind(notebook_id)
    // Derived locator is the ORIGINAL upload path, never read back for citations.
    .bind(format!("{source_id}.pdf"))
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(&pool)
    .await
    .expect("insert derived source");
}

/// Inserts one chunk row with real byte offsets satisfying the byte-identity
/// invariant `buffer[char_start..char_end] == text`.
async fn seed_chunk(
    engine: &LensEngine,
    source_id: &str,
    chunk_id: &str,
    buffer: &str,
    start: usize,
    end: usize,
) {
    let pool = engine.pool().await;
    sqlx::query(
        "INSERT INTO chunks \
         (id, source_id, parent_id, kind, level, section_path, text, \
          token_start, token_end, char_start, char_end, block_type, created_at) \
         VALUES (?, ?, NULL, 'child', 1, 'Intro', ?, 0, 1, ?, ?, 'paragraph', ?)",
    )
    .bind(chunk_id)
    .bind(source_id)
    .bind(&buffer[start..end])
    .bind(start as i64)
    .bind(end as i64)
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(&pool)
    .await
    .expect("insert chunk");
}

// ---------------------------------------------------------------------------
// AC2: a citation resolves to the exact original source substring (R6)
// ---------------------------------------------------------------------------

/// Text-like source: `add_text_source` writes the canonical locator buffer, and a
/// citation resolved from persisted (seeded) offsets re-slices it to exactly the
/// original substring.
#[tokio::test]
async fn ac2_text_like_snippet_matches_persisted_offsets() {
    let (dir, engine) = base_engine().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    let body = "The first sentence introduces the topic. \
        A second sentence adds a supporting detail. The third sentence concludes.";
    let src = engine
        .add_text_source(&nb, "doc", body, "text")
        .await
        .unwrap()
        .source;

    // The managed locator file IS the canonical buffer for text-like kinds.
    let buffer = std::fs::read_to_string(&src.locator).expect("read locator buffer");
    assert_eq!(buffer, body);

    let marker = "A second sentence adds a supporting detail.";
    let start = body.find(marker).unwrap();
    let end = start + marker.len();
    let chunk_id = format!("{}-c0", src.id);
    seed_chunk(&engine, &src.id, &chunk_id, body, start, end).await;

    // Resolve through the real persisted-offset read-back path.
    let pool = engine.pool().await;
    let rows = load_chunk_locators(&pool, std::slice::from_ref(&chunk_id))
        .await
        .expect("load locators");
    let row = rows.get(&chunk_id).expect("locator row present");
    let (cs, ce) = (row.char_start.unwrap(), row.char_end.unwrap());

    assert_eq!(&buffer[cs..ce], marker, "byte-identity invariant on disk");
    let seg = engine
        .citation_snippet(&src.id, cs, ce)
        .await
        .expect("snippet");
    assert_eq!(seg.marked, marker, "marked == original substring");
    drop(dir);
}

/// End-to-end through the REAL ingest pipeline (R6): ingest a text source, then for
/// EVERY persisted chunk assert the citation snippet's `marked` equals the chunk's
/// own text (the on-disk byte-identity invariant, surfaced via the read-back API).
/// Skipped offline when no nomic tokenizer is reachable (the chunker needs it).
#[tokio::test]
async fn ac2_real_ingest_end_to_end_pipeline() {
    let (dir, engine) = base_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    if !support::tokenizer_available().await {
        eprintln!("skipping ac2_real_ingest_end_to_end_pipeline: no tokenizer (offline)");
        return;
    }
    support::seed_tokenizer_from_env(&data_dir);

    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    let body = "# Heading\n\nThe first sentence introduces the topic. \
        A second sentence adds a supporting detail. The third sentence concludes.\n";
    let src = engine
        .add_text_source(&nb, "doc", body, "text")
        .await
        .unwrap()
        .source;
    engine
        .ingest_source(&src.id, |_p| {})
        .await
        .expect("ingest");

    let buffer = std::fs::read_to_string(&src.locator).expect("read locator buffer");
    let rows = chunk_offsets(&engine, &src.id).await;
    assert!(!rows.is_empty(), "ingest persisted chunks with offsets");
    for (_id, cs, ce, text) in &rows {
        let (cs, ce) = (*cs as usize, *ce as usize);
        assert_eq!(&buffer[cs..ce], text, "byte-identity invariant on disk");
        let seg = engine
            .citation_snippet(&src.id, cs, ce)
            .await
            .expect("snippet");
        assert_eq!(seg.marked, *text, "marked == original chunk substring");
    }
    drop(dir);
}

/// Derived source (`kind='pdf'`): the citation resolves against the `.extracted.txt`
/// sibling. Exercises the read-back path AND `load_chunk_locators` (persisted offsets).
#[tokio::test]
async fn ac2_derived_snippet_matches_extracted_sibling() {
    let (dir, engine) = base_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    let source_id = uuid::Uuid::now_v7().to_string();
    let buffer = "Extracted body. The claim under citation lives right here. Tail text.";
    seed_derived_source(&engine, &data_dir, &nb, &source_id, buffer).await;

    let marker = "The claim under citation lives right here.";
    let start = buffer.find(marker).unwrap();
    let end = start + marker.len();
    let chunk_id = format!("{source_id}-c0");
    seed_chunk(&engine, &source_id, &chunk_id, buffer, start, end).await;

    // Round-trip through the real hydrate read-back path.
    let pool = engine.pool().await;
    let rows = load_chunk_locators(&pool, std::slice::from_ref(&chunk_id))
        .await
        .expect("load locators");
    let row = rows.get(&chunk_id).expect("locator row present");
    let (cs, ce) = (row.char_start.unwrap(), row.char_end.unwrap());

    let seg = engine
        .citation_snippet(&source_id, cs, ce)
        .await
        .expect("snippet");
    assert_eq!(seg.marked, marker);
    assert_eq!(&buffer[cs..ce], marker);
    drop(dir);
}

// ---------------------------------------------------------------------------
// Retention across enrichment + re-ingest (AC1 / R2)
// ---------------------------------------------------------------------------

/// The text-like locator buffer written at add time is retained and byte-identical
/// after a re-ingest, and `source_view` round-trips it. When a tokenizer is
/// reachable this drives the REAL ingest+re-ingest pipeline; otherwise it asserts
/// the managed-file retention that `add_text_source` establishes (enrichment/ingest
/// never mutate the original — only purge removes it).
#[tokio::test]
async fn retention_survives_reingest_byte_identical() {
    let (dir, engine) = base_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    let body = "Stable content that does not change across a re-ingest pass.\n";
    let src = engine
        .add_text_source(&nb, "doc", body, "text")
        .await
        .unwrap()
        .source;
    let before = std::fs::read_to_string(&src.locator).expect("buffer after add");
    assert_eq!(before, body);

    if support::tokenizer_available().await {
        support::seed_tokenizer_from_env(&data_dir);
        engine
            .ingest_source(&src.id, |_p| {})
            .await
            .expect("ingest 1");
        engine
            .ingest_source(&src.id, |_p| {})
            .await
            .expect("ingest 2");
    }
    let after = std::fs::read_to_string(&src.locator).expect("buffer retained");
    assert_eq!(before, after, "locator buffer byte-identical and retained");

    let view = engine.source_view(&src.id, None).await.expect("view");
    assert_eq!(
        view.before, after,
        "source_view round-trips the whole buffer"
    );
    drop(dir);
}

// ---------------------------------------------------------------------------
// Degradation / null paths (R4 / R9)
// ---------------------------------------------------------------------------

/// `source_view(_, None)` returns the whole retained text; an out-of-range span
/// clamps without panic.
#[tokio::test]
async fn source_view_none_and_out_of_range_span() {
    let (dir, engine) = base_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    let source_id = uuid::Uuid::now_v7().to_string();
    let buffer = "Some derived body text for the viewer.";
    seed_derived_source(&engine, &data_dir, &nb, &source_id, buffer).await;

    let whole = engine.source_view(&source_id, None).await.expect("whole");
    assert_eq!(whole.before, buffer);
    assert!(whole.marked.is_empty() && whole.after.is_empty());
    assert_eq!(whole.kind, "pdf");
    assert_eq!(whole.title, "Derived Doc");

    let clamped = engine
        .source_view(&source_id, Some((10_000, 20_000)))
        .await
        .expect("clamped span");
    assert_eq!(
        clamped.marked, "",
        "out-of-range span clamps to empty marked"
    );
    drop(dir);
}

/// Trashed-but-not-purged sources still resolve (files survive until purge);
/// a purged source (sibling removed) degrades to `Err(Io)`.
#[tokio::test]
async fn trashed_resolves_but_purged_degrades() {
    let (dir, engine) = base_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    let source_id = uuid::Uuid::now_v7().to_string();
    let buffer = "Body of a source that will be trashed then purged.";
    seed_derived_source(&engine, &data_dir, &nb, &source_id, buffer).await;

    // Mark trashed (files intact) — must still resolve.
    let pool = engine.pool().await;
    sqlx::query("UPDATE sources SET trashed_at = ? WHERE id = ?")
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(&source_id)
        .execute(&pool)
        .await
        .unwrap();
    let view = engine
        .source_view(&source_id, None)
        .await
        .expect("trashed resolves");
    assert_eq!(view.before, buffer);

    // Purge the sibling → generic Io (never a path leak).
    std::fs::remove_file(
        data_dir
            .join("sources")
            .join(format!("{source_id}.extracted.txt")),
    )
    .unwrap();
    let err = engine
        .source_view(&source_id, None)
        .await
        .expect_err("purged degrades");
    assert!(matches!(err, lens_core::LensError::Io(_)));
    assert!(
        !err.message().contains(source_id.as_str()),
        "no id/path leak in message"
    );

    // An absent source id → Validation (not Io).
    let missing = engine
        .source_view("does-not-exist", None)
        .await
        .expect_err("absent");
    assert!(matches!(missing, lens_core::LensError::Validation(_)));
    drop(dir);
}

// ---------------------------------------------------------------------------
// Content-changed re-ingest drift = documented limitation (R7)
// ---------------------------------------------------------------------------

/// Offsets are frozen at citation time; if the source buffer changes underneath
/// them (content-changed re-ingest at the same id), resolution clamps and returns
/// valid segments rather than panicking — it may simply not match the old span.
/// This is the accepted limitation (no citation hash-stamping — would force a
/// migration).
#[tokio::test]
async fn content_change_drift_clamps_without_panic() {
    let (dir, engine) = base_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    let source_id = uuid::Uuid::now_v7().to_string();
    let original = "The original long body with a cited span near the very end here.";
    seed_derived_source(&engine, &data_dir, &nb, &source_id, original).await;
    let marker = "cited span near the very end here.";
    let start = original.find(marker).unwrap();
    let end = start + marker.len();

    // Overwrite with a much shorter buffer (simulate content-changed re-ingest).
    std::fs::write(
        data_dir
            .join("sources")
            .join(format!("{source_id}.extracted.txt")),
        "tiny",
    )
    .unwrap();

    // Frozen offsets now index past the new buffer — must clamp, never panic.
    let seg = engine
        .citation_snippet(&source_id, start, end)
        .await
        .expect("no panic");
    assert_eq!(seg.marked, "", "offsets past shrunk buffer clamp to empty");
    drop(dir);
}

// ---------------------------------------------------------------------------
// Concurrency: atomic sibling write (R2)
// ---------------------------------------------------------------------------

/// A read concurrent with a re-ingest that rewrites the `.extracted.txt` sibling
/// sees full-old or full-new content, never an empty/partial buffer, thanks to the
/// tmp + rename write.
///
/// NOTE: the writer here reimplements tmp+rename rather than driving `ingest.rs`'s
/// production write, so this proves reader tolerance only — see
/// `concurrent_read_during_real_ingest_rewrite_never_partial` for the production-path lock.
#[tokio::test]
async fn concurrent_read_during_rewrite_never_partial() {
    let (dir, engine) = base_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    let source_id = uuid::Uuid::now_v7().to_string();
    let old = "OLD ".repeat(5000);
    let new = "NEW ".repeat(6000);
    seed_derived_source(&engine, &data_dir, &nb, &source_id, &old).await;

    let sibling = data_dir
        .join("sources")
        .join(format!("{source_id}.extracted.txt"));

    // Writer: atomically replace via tmp + rename in a loop.
    let writer_path = sibling.clone();
    let new_c = new.clone();
    let old_c = old.clone();
    let writer = tokio::task::spawn_blocking(move || {
        for i in 0..50 {
            let payload = if i % 2 == 0 { &new_c } else { &old_c };
            let tmp = writer_path.with_extension("txt.part");
            std::fs::write(&tmp, payload).unwrap();
            std::fs::rename(&tmp, &writer_path).unwrap();
        }
    });

    // Reader: every observed buffer is one of the two full payloads.
    for _ in 0..200 {
        let view = engine.source_view(&source_id, None).await.expect("read");
        assert!(
            view.before == old || view.before == new,
            "reader saw a full buffer, never a partial one"
        );
    }
    writer.await.unwrap();
    drop(dir);
}

/// Same invariant, but the writer drives the REAL production path: repeated real
/// ingestion of a derived JSON source via `add_file_source`/`ingest_source` (the
/// actual `index_extract_output` tmp+rename), not a reimplementation. Reverting
/// that write to a plain truncating write would let a concurrent reader observe a
/// torn/empty sibling; this must never happen. Skipped offline when no nomic
/// tokenizer is reachable (chunking needs it).
#[tokio::test]
async fn concurrent_read_during_real_ingest_rewrite_never_partial() {
    let (dir, engine) = base_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    if !support::tokenizer_available().await {
        eprintln!(
            "skipping concurrent_read_during_real_ingest_rewrite_never_partial: no tokenizer (offline)"
        );
        return;
    }
    support::seed_tokenizer_from_env(&data_dir);

    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;

    let old_value = "The old filler sentence repeats here. ".repeat(50);
    let new_value = "The new filler sentence repeats here. ".repeat(50);
    let old_json = serde_json::json!({ "body": old_value }).to_string();
    let new_json = serde_json::json!({ "body": new_value }).to_string();
    let expected_old = format!("/body: {old_value}\n");
    let expected_new = format!("/body: {new_value}\n");

    let staged = dir.path().join("staged.json");
    std::fs::write(&staged, &old_json).expect("stage source file");
    let src = engine
        .add_file_source(&nb, &staged, Some("Derived JSON"))
        .await
        .unwrap()
        .source;
    engine
        .ingest_source(&src.id, |_p| {})
        .await
        .expect("initial ingest");

    let sibling = data_dir
        .join("sources")
        .join(format!("{}.extracted.txt", src.id));
    let initial = std::fs::read_to_string(&sibling).expect("sibling after initial ingest");
    assert_eq!(
        initial, expected_old,
        "sibling matches the real extractor output"
    );

    // Writer: repeatedly overwrites the managed locator and re-ingests through the
    // REAL production write path (index_extract_output's tmp+rename).
    let locator = std::path::PathBuf::from(&src.locator);
    let stop = Arc::new(AtomicBool::new(false));
    let engine_w = engine.clone();
    let src_id = src.id.clone();
    let (old_json_c, new_json_c) = (old_json.clone(), new_json.clone());
    let stop_w = Arc::clone(&stop);
    let writer = tokio::spawn(async move {
        for i in 0..10 {
            let payload = if i % 2 == 0 { &new_json_c } else { &old_json_c };
            tokio::fs::write(&locator, payload).await.unwrap();
            engine_w
                .ingest_source(&src_id, |_p| {})
                .await
                .expect("re-ingest");
        }
        stop_w.store(true, Ordering::Relaxed);
    });

    // Reader: every observed sibling read is one of the two full extracted
    // outputs — never empty/partial — for as long as the writer is active.
    let sibling_r = sibling.clone();
    let (expected_old_r, expected_new_r) = (expected_old.clone(), expected_new.clone());
    let stop_r = Arc::clone(&stop);
    let reader = tokio::spawn(async move {
        while !stop_r.load(Ordering::Relaxed) {
            if let Ok(seen) = tokio::fs::read_to_string(&sibling_r).await {
                assert!(
                    seen == expected_old_r || seen == expected_new_r,
                    "reader saw a full extracted buffer, never a partial one"
                );
            }
        }
    });

    writer.await.unwrap();
    reader.await.unwrap();
    drop(dir);
}
