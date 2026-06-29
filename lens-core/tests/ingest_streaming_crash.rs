// issue #71: raise the limit so deep `Send` auto-trait evaluation on the
// streamed-ingest future doesn't overflow (E0275) under stricter toolchains.
#![recursion_limit = "256"]
//! issue #71 Step 5 — crash-injection tests for the bounded-memory streaming PDF
//! ingest (building-table lifecycle).
//!
//! These tests arm process-global crash seams
//! (`CRASH_AFTER_STREAMING_ADD_BEFORE_FLIP`, and the existing
//! `CRASH_AFTER_FLIP_TXN_BEFORE_LANCE_DROP`) and so MUST live in their OWN test
//! binary (separate process) — otherwise the flags would leak into the parallel
//! ingest tests in `ingest.rs` under CI parallelism. This mirrors the isolation
//! pattern of `reembed_backend_switch_crash.rs`.
//!
//! Each test:
//!   1. crashes mid-stream → asserts NO active rows for the source + an orphan
//!      building table left behind + source status `error`;
//!   2. reopens the engine → asserts the startup-GC swept the orphan building
//!      table + its registry row;
//!   3. crashes AFTER the flip txn but BEFORE the stale Lance drop → asserts the
//!      new active table serves search and the stale table is reclaimed by GC.

use std::sync::atomic::Ordering;

use lens_core::LensEngine;
use lens_core::vector_store::{
    CRASH_AFTER_FLIP_TXN_BEFORE_LANCE_DROP, CRASH_AFTER_STREAMING_ADD_BEFORE_FLIP,
};

mod support;
use support::{inject_fake_embedder, tokenizer_available};

/// Serializes the crash tests WITHIN this binary. They share the process-global
/// `CRASH_*` flags, so running them in parallel (the default for `#[tokio::test]`)
/// would let one test consume another's armed flag. A binary-local async mutex
/// makes them run one-at-a-time without needing a `serial_test` dev-dependency.
static CRASH_SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

// ---------------------------------------------------------------------------
// Local helpers (the `ingest.rs` helpers live in that binary; keep these
// self-contained so this isolated binary owns its fixtures).
// ---------------------------------------------------------------------------

/// Builds a multi-page text-layer PDF so the ingest produces multiple chunks and
/// at least one EMBED_BATCH lands in the building table before the crash seam.
fn build_multipage_text_pdf_bytes(pages: usize, prefix: &str) -> Vec<u8> {
    use printpdf::{BuiltinFont, Mm, PdfDocument};
    use std::io::BufWriter;
    let (doc, page1, layer1) = PdfDocument::new("crash-fixture", Mm(210.0), Mm(297.0), "Layer 1");
    let font = doc.add_builtin_font(BuiltinFont::Helvetica).unwrap();
    {
        let layer = doc.get_page(page1).get_layer(layer1);
        for line in 0..8 {
            layer.use_text(
                format!("{prefix} page 1 line {line}: the quick brown fox jumps."),
                12.0,
                Mm(20.0),
                Mm(270.0 - line as f32 * 8.0),
                &font,
            );
        }
    }
    for p in 2..=pages {
        let (page, layer_idx) = doc.add_page(Mm(210.0), Mm(297.0), "Layer 1");
        let layer = doc.get_page(page).get_layer(layer_idx);
        for line in 0..8 {
            layer.use_text(
                format!("{prefix} page {p} line {line}: the quick brown fox jumps."),
                12.0,
                Mm(20.0),
                Mm(270.0 - line as f32 * 8.0),
                &font,
            );
        }
    }
    let mut buf = Vec::new();
    doc.save(&mut BufWriter::new(&mut buf)).unwrap();
    buf
}

/// Writes a multi-page PDF to disk and inserts a queued `pdf` source row.
/// Returns `(source_id, raw_bytes)`, or None if libpdfium cannot bind here.
async fn write_pdf_source(
    engine: &LensEngine,
    notebook: &str,
    pages: usize,
    prefix: &str,
) -> Option<String> {
    use lens_core::extract::extractor_for;
    let raw = build_multipage_text_pdf_bytes(pages, prefix);
    if extractor_for("pdf").unwrap().extract(&raw).is_err() {
        return None;
    }
    let data_dir = engine.data_dir_for_test().await;
    let id = uuid::Uuid::now_v7().to_string();
    let sources_dir = data_dir.join("sources");
    std::fs::create_dir_all(&sources_dir).unwrap();
    let path = sources_dir.join(format!("{id}.pdf"));
    std::fs::write(&path, &raw).unwrap();
    let locator = path.display().to_string();

    let pool = engine.pool().await;
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, created_at) \
         VALUES (?, ?, 'pdf', 'crash', 'queued', ?, 1, ?)",
    )
    .bind(&id)
    .bind(notebook)
    .bind(&locator)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();
    Some(id)
}

async fn embidx_count(engine: &LensEngine, notebook: &str, status: &str) -> i64 {
    sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM embedding_index \
         WHERE notebook_id = ? AND backend = 'fastembed' AND model = ? AND dim = ? AND status = ?",
    )
    .bind(notebook)
    .bind(lens_core::DEFAULT_EMBED_MODEL_ID)
    .bind(lens_core::DEFAULT_EMBED_DIM as i64)
    .bind(status)
    .fetch_one(&engine.pool().await)
    .await
    .unwrap()
}

async fn source_status(engine: &LensEngine, source_id: &str) -> String {
    sqlx::query_scalar::<_, String>("SELECT status FROM sources WHERE id = ?")
        .bind(source_id)
        .fetch_one(&engine.pool().await)
        .await
        .unwrap()
}

/// Counts vectors in the notebook's ACTIVE Lance table (registry-driven, so it
/// follows the gen-suffixed name after a flip). Returns 0 when there is no active
/// table (the first-source-is-PDF crash leaves none).
async fn active_table_total_rows(engine: &LensEngine, notebook: &str) -> usize {
    use futures_util::TryStreamExt;
    use lancedb::query::ExecutableQuery;

    let name: Option<String> = sqlx::query_scalar(
        "SELECT lance_table_name FROM embedding_index \
         WHERE notebook_id = ? AND backend = 'fastembed' AND model = ? AND dim = ? AND status = 'active'",
    )
    .bind(notebook)
    .bind(lens_core::DEFAULT_EMBED_MODEL_ID)
    .bind(lens_core::DEFAULT_EMBED_DIM as i64)
    .fetch_optional(&engine.pool().await)
    .await
    .unwrap();
    let Some(name) = name else { return 0 };
    let root = engine.data_dir_for_test().await.join("lancedb");
    let conn = lancedb::connect(root.to_string_lossy().as_ref())
        .execute()
        .await
        .unwrap();
    let names = conn.table_names().execute().await.unwrap_or_default();
    if !names.iter().any(|n| n == &name) {
        return 0;
    }
    let table = conn.open_table(&name).execute().await.unwrap();
    let stream = table.query().execute().await.unwrap();
    let batches: Vec<_> = stream.try_collect().await.unwrap();
    batches.iter().map(|b| b.num_rows()).sum()
}

/// Counts physical Lance tables present on disk for a notebook prefix (any gen).
async fn physical_tables_for_notebook(engine: &LensEngine, notebook: &str) -> usize {
    let root = engine.data_dir_for_test().await.join("lancedb");
    let conn = match lancedb::connect(root.to_string_lossy().as_ref())
        .execute()
        .await
    {
        Ok(c) => c,
        Err(_) => return 0,
    };
    let names = conn.table_names().execute().await.unwrap_or_default();
    let prefix = format!("vec__{notebook}__");
    names.iter().filter(|n| n.starts_with(&prefix)).count()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Crash between a batch's `add_to_table_no_index` and the flip: NO active rows
/// for the source, the building table is left as an orphan, and the source flips
/// to `error`.
#[tokio::test]
async fn test_crash_mid_stream_no_active_rows() {
    if !tokenizer_available().await {
        eprintln!("skipping test_crash_mid_stream_no_active_rows: no tokenizer (offline)");
        return;
    }
    let _guard = CRASH_SERIAL.lock().await;
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    inject_fake_embedder(&engine);
    let nb = engine
        .create_notebook("crash-mid", None, None)
        .await
        .unwrap()
        .id
        .to_string();

    let Some(src) = write_pdf_source(&engine, &nb, 4, "crashmid").await else {
        eprintln!("skipping test_crash_mid_stream_no_active_rows: libpdfium not bindable");
        return;
    };

    // Arm the crash seam: fire after batch 1 lands in the building table.
    CRASH_AFTER_STREAMING_ADD_BEFORE_FLIP.store(true, Ordering::SeqCst);
    let result = engine.ingest_source(&src, |_p| {}).await;
    CRASH_AFTER_STREAMING_ADD_BEFORE_FLIP.store(false, Ordering::SeqCst);
    assert!(result.is_err(), "the armed crash seam must fail the ingest");

    // The Err→error flip in `ingest_source` set the source to `error`.
    assert_eq!(source_status(&engine, &src).await, "error");

    // No active rows: the flip never ran (first-source-is-PDF, so there is no
    // active table at all).
    assert_eq!(
        embidx_count(&engine, &nb, "active").await,
        0,
        "no active embedding_index row after a mid-stream crash"
    );
    assert_eq!(
        active_table_total_rows(&engine, &nb).await,
        0,
        "no active vectors for the crashed source"
    );

    // The building table + its registry row are left as an orphan with partial
    // rows (at least one batch was written before the crash).
    assert_eq!(
        embidx_count(&engine, &nb, "building").await,
        1,
        "the orphan building registry row is left behind"
    );
    assert!(
        physical_tables_for_notebook(&engine, &nb).await >= 1,
        "the orphan building Lance table is left on disk"
    );
}

/// After the mid-stream crash above, reopening the engine runs the startup-GC,
/// which sweeps the orphan building table + its registry row.
#[tokio::test]
async fn test_crash_mid_stream_gc_on_restart() {
    if !tokenizer_available().await {
        eprintln!("skipping test_crash_mid_stream_gc_on_restart: no tokenizer (offline)");
        return;
    }
    let _guard = CRASH_SERIAL.lock().await;
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    inject_fake_embedder(&engine);
    let nb = engine
        .create_notebook("crash-gc", None, None)
        .await
        .unwrap()
        .id
        .to_string();

    let Some(src) = write_pdf_source(&engine, &nb, 4, "crashgc").await else {
        eprintln!("skipping test_crash_mid_stream_gc_on_restart: libpdfium not bindable");
        return;
    };

    CRASH_AFTER_STREAMING_ADD_BEFORE_FLIP.store(true, Ordering::SeqCst);
    let _ = engine.ingest_source(&src, |_p| {}).await;
    CRASH_AFTER_STREAMING_ADD_BEFORE_FLIP.store(false, Ordering::SeqCst);
    assert_eq!(
        embidx_count(&engine, &nb, "building").await,
        1,
        "precondition: an orphan building row exists before restart"
    );

    // Reopen → startup-GC sweeps the orphan building table + registry row.
    drop(engine);
    let engine2 = LensEngine::init(dir.path()).await.unwrap();
    assert_eq!(
        embidx_count(&engine2, &nb, "building").await,
        0,
        "startup-GC must sweep the orphan building registry row"
    );
    assert_eq!(
        physical_tables_for_notebook(&engine2, &nb).await,
        0,
        "startup-GC must drop the orphan building Lance table"
    );
}

/// Crash AFTER the flip txn commits but BEFORE the stale Lance drop (reusing the
/// existing `CRASH_AFTER_FLIP_TXN_BEFORE_LANCE_DROP` seam, which fires inside
/// `flip_active`): the new active table serves search, and the stale table left
/// behind is reclaimed by the startup-GC on reopen.
///
/// Exercised on the SECOND source so there is a prior active table to demote to
/// stale (the flip's stale-drop window only exists when an active table exists).
#[tokio::test]
async fn test_crash_after_flip_before_stale_drop() {
    if !tokenizer_available().await {
        eprintln!("skipping test_crash_after_flip_before_stale_drop: no tokenizer (offline)");
        return;
    }
    let _guard = CRASH_SERIAL.lock().await;
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    inject_fake_embedder(&engine);
    let nb = engine
        .create_notebook("crash-flip", None, None)
        .await
        .unwrap()
        .id
        .to_string();

    // First source ingests cleanly → an active (gen-1) table exists.
    let Some(_src1) = write_pdf_source(&engine, &nb, 3, "flipfirst").await else {
        eprintln!("skipping test_crash_after_flip_before_stale_drop: libpdfium not bindable");
        return;
    };
    engine.ingest_source(&_src1, |_p| {}).await.unwrap();
    let active_after_first = active_table_total_rows(&engine, &nb).await;
    assert!(active_after_first > 0, "first source must index vectors");

    // Second source: arm the flip-stale-drop crash seam. The flip txn commits
    // (building→active, old active→stale) but returns before dropping the stale
    // Lance table — leaving a `stale` row + orphan table.
    let src2 = write_pdf_source(&engine, &nb, 3, "flipsecond")
        .await
        .expect("second pdf");
    CRASH_AFTER_FLIP_TXN_BEFORE_LANCE_DROP.store(true, Ordering::SeqCst);
    engine.ingest_source(&src2, |_p| {}).await.unwrap();
    CRASH_AFTER_FLIP_TXN_BEFORE_LANCE_DROP.store(false, Ordering::SeqCst);

    // The new active table holds BOTH sources' vectors (the seed preserved the
    // first source), and a stale row lingers from the simulated crash.
    assert_eq!(
        embidx_count(&engine, &nb, "active").await,
        1,
        "exactly one active row after the flip"
    );
    assert_eq!(
        embidx_count(&engine, &nb, "stale").await,
        1,
        "a stale row lingers after the simulated crash before the stale-drop"
    );
    let active_after_second = active_table_total_rows(&engine, &nb).await;
    assert!(
        active_after_second > active_after_first,
        "the new active table serves both sources' vectors"
    );

    // Reopen → startup-GC reclaims the stale row + its orphan table.
    drop(engine);
    let engine2 = LensEngine::init(dir.path()).await.unwrap();
    assert_eq!(
        embidx_count(&engine2, &nb, "stale").await,
        0,
        "startup-GC must reclaim the stale row after the crash"
    );
    assert_eq!(
        embidx_count(&engine2, &nb, "active").await,
        1,
        "the new active coordinate keeps serving after recovery"
    );
}
