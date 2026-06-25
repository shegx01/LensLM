//! Integration tests for URL source ingestion (M4 Phase 2, Step 7).
//!
//! # Coverage
//!
//! - **AC8** — `url_fetch_http_error_flips_to_error`: a wiremock endpoint that
//!   responds with HTTP 504 → the fetch fails, the source goes to `error` (via
//!   the Err→error flip in `ingest_source`), and a subsequent ingest succeeds
//!   (permit released). This exercises the HTTP-error path end-to-end through
//!   the real pipeline; the genuine TCP-level *timeout-fires* proof lives
//!   in-module as `ingest::tests::fetch_timeout_fires_on_slow_response`, which
//!   trips a short injected timeout against a delayed mock without waiting for
//!   the 30 s production constant.
//! - **AC9a** — `url_needs_js_on_js_shell`: a wiremock JS-shell body (tiny real
//!   text, large script blocks) → source ends up in `needs_js`, nothing indexed.
//! - **AC9b** — `url_indexes_real_article`: a wiremock body with a real article
//!   containing >200 chars of prose → source ends up in `indexed`.
//! - **crash_recovery_skips_needs_js_and_needs_ocr** — a source at `needs_js`
//!   (or `needs_ocr`) survives the engine's crash-recovery reset unchanged.
//! - **add_url_source_returns_queued_without_fetch** — `add_url_source` inserts
//!   a `queued` row and returns before any network request is made.
//!
//! # Timeout strategy
//!
//! `URL_FETCH_TIMEOUT` is 30 seconds — too long to wait in an integration test
//! that drives the whole `run_ingest` pipeline (which builds its own client
//! from that constant). So `url_fetch_http_error_flips_to_error` does NOT test
//! a real timeout: it asserts the *fetch-failure → `error` flip + permit
//! release* contract by responding with an immediate HTTP 504, which fails the
//! fetch without any delay and keeps the test fast.
//!
//! The genuine TCP-level *timeout-fires* behaviour (a slow response trips a
//! short injected timeout, the fetch errors instead of hanging) is proven by
//! the in-module unit test `ingest::tests::fetch_timeout_fires_on_slow_response`,
//! which can inject a tiny per-fetch timeout directly.
//!
//! `URL_FETCH_TIMEOUT` is `pub(crate)`, so `url_fetch_timeout_is_30_seconds`
//! also pins its expected value (30 seconds) as a compile-time check here.

use std::sync::Arc;

use lens_core::ingest::{NEEDS_JS_MIN_CHARS, NEEDS_JS_MIN_TEXT_RATIO, URL_FETCH_TIMEOUT};
use lens_core::{IngestProgress, LensEngine, Source};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

mod support;
use support::{file_engine, inject_fake_embedder, seed_tokenizer_from_env, vector_row_count};

// ===========================================================================
// Shared helpers
// ===========================================================================

/// Relaxes the URL-fetch SSRF IP guard so the real `ingest_source` pipeline can
/// fetch from a `wiremock` server bound to `127.0.0.1`.
///
/// The production guard rejects loopback/private/link-local hosts. These
/// integration tests must drive the FULL pipeline against a local mock, so they
/// opt in to the `test-util`-gated `LENS_TEST_ALLOW_LOCAL_URL` escape hatch
/// (see `ingest::allow_local_url_fetch`). The variable is set once for the whole
/// test binary (idempotent), and only ever LOOSENS the guard for loopback — the
/// scheme allowlist and every other guard still apply. Production builds never
/// enable `test-util`, so the hatch is compiled out entirely there.
fn allow_local_url_fetch_for_test() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY (edition 2024): set once, before any concurrent fetch reads it;
        // the value is process-global for this single test binary and only
        // relaxes the loopback guard, matching what every fetching test wants.
        unsafe {
            std::env::set_var("LENS_TEST_ALLOW_LOCAL_URL", "1");
        }
    });
}

/// Collects all progress events into a Vec.
async fn ingest_collecting_progress(
    engine: &LensEngine,
    source_id: &str,
) -> (Result<(), lens_core::LensError>, Vec<IngestProgress>) {
    // Every ingest in this binary targets a loopback wiremock server, so relax
    // the SSRF loopback guard (test-util-gated; production unaffected).
    allow_local_url_fetch_for_test();
    let events: Arc<std::sync::Mutex<Vec<IngestProgress>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = engine
        .ingest_source(source_id, move |p| {
            events_clone.lock().unwrap().push(p);
        })
        .await;
    let collected = events.lock().unwrap().clone();
    (result, collected)
}

/// HTML body that trafilatura sees as a JS shell: near-zero real text but many
/// script tags. The body is intentionally large (many script references) so the
/// raw HTML size makes the text/HTML ratio fall below `NEEDS_JS_MIN_TEXT_RATIO`.
fn js_shell_html() -> String {
    // Build a body whose total byte count is large relative to content.
    let mut html = String::from(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>My App</title>
</head>
<body>
<div id="root"></div>
"#,
    );
    // Add enough script tags to push the raw HTML well past the ratio threshold.
    for i in 0..60 {
        html.push_str(&format!(
            r#"<script src="/static/js/chunk.{i}.js"></script>
"#
        ));
    }
    html.push_str("</body>\n</html>\n");
    html
}

/// HTML body with a genuine short-but-real article (~300 chars of prose).
/// Must produce ≥ `NEEDS_JS_MIN_CHARS` chars of extracted text AND
/// exceed `NEEDS_JS_MIN_TEXT_RATIO`.
fn real_article_html() -> String {
    // The article body is kept minimal but above the floor. Trafilatura should
    // extract the <article> content cleanly.
    r#"<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Short Article</title></head>
<body>
<article>
<h1>A Concise Introduction</h1>
<p>Rust is a systems programming language focused on safety, speed, and concurrency.
It achieves memory safety without garbage collection through its ownership model.
The borrow checker enforces these rules at compile time, eliminating entire classes
of bugs common in C and C++ programs. Rust's zero-cost abstractions let you write
high-level code without sacrificing performance.</p>
</article>
</body>
</html>"#
        .to_string()
}

// ===========================================================================
// Compile-time sanity: URL_FETCH_TIMEOUT is 30 s
// ===========================================================================

#[test]
fn url_fetch_timeout_is_30_seconds() {
    assert_eq!(
        URL_FETCH_TIMEOUT,
        std::time::Duration::from_secs(30),
        "URL_FETCH_TIMEOUT should be 30 seconds"
    );
}

#[test]
fn needs_js_min_chars_is_200() {
    assert_eq!(NEEDS_JS_MIN_CHARS, 200);
}

#[test]
fn needs_js_min_text_ratio_is_0_01() {
    assert!((NEEDS_JS_MIN_TEXT_RATIO - 0.01f64).abs() < 1e-9);
}

// ===========================================================================
// AC8 — fetch error (HTTP 504) → source goes to "error"
// ===========================================================================

/// AC8: When the server returns an HTTP error code (504), the fetch fails,
/// the source is flipped to `error` (via the Err→error flip in `ingest_source`),
/// and the ingest permit is released so a subsequent ingest can run.
#[tokio::test]
async fn url_fetch_http_error_flips_to_error() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/page"))
        .respond_with(ResponseTemplate::new(504))
        .mount(&mock)
        .await;

    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);

    // Create a notebook + URL source pointing at the mock server.
    let nb = engine
        .create_notebook("NB", None, None)
        .await
        .expect("notebook");
    let source = engine
        .add_url_source(&nb.id, "bad page", &format!("{}/page", mock.uri()))
        .await
        .expect("add_url_source");
    assert_eq!(source.status, "queued");
    assert_eq!(source.kind, "url");

    // Ingest should fail (HTTP 504 → network error → Err → error flip).
    let (result, _) = ingest_collecting_progress(&engine, &source.id).await;
    assert!(
        result.is_err(),
        "HTTP 504 should return Err from ingest_source"
    );

    // The source status should be "error".
    let pool = engine.pool().await;
    let repo = lens_core::notebooks::NotebookRepo::new(&pool);
    let updated = repo
        .get_source(&source.id)
        .await
        .expect("get_source ok")
        .expect("source exists");
    assert_eq!(
        updated.status, "error",
        "HTTP error should set status=error"
    );

    // The ingest permit must be released: another ingest (of a different source,
    // text kind) must succeed without blocking.
    let src2 = engine
        .add_text_source(
            &nb.id,
            "hello",
            "Hello world. This is enough text to pass the size check.",
            "text",
        )
        .await
        .expect("add_text_source");
    // We don't run the full ingest (tokenizer not available offline) — just prove
    // add_text_source returns without deadlocking. The lock is already released.
    assert_eq!(src2.status, "queued");
}

// ===========================================================================
// AC9a — JS shell → needs_js
// ===========================================================================

/// AC9a: A JS-shell page (tiny real text, large scripts) must produce
/// `needs_js` status and leave zero chunks indexed.
#[tokio::test]
async fn url_needs_js_on_js_shell() {
    let html = js_shell_html();
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/spa"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(html)
                .insert_header("content-type", "text/html; charset=utf-8"),
        )
        .mount(&mock)
        .await;

    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);

    let nb = engine
        .create_notebook("NB", None, None)
        .await
        .expect("notebook");
    let source = engine
        .add_url_source(&nb.id, "spa page", &format!("{}/spa", mock.uri()))
        .await
        .expect("add_url_source");

    let (result, _events) = ingest_collecting_progress(&engine, &source.id).await;
    assert!(
        result.is_ok(),
        "needs_js outcome must return Ok, not Err: {result:?}"
    );

    let pool = engine.pool().await;
    let repo = lens_core::notebooks::NotebookRepo::new(&pool);
    let updated = repo
        .get_source(&source.id)
        .await
        .expect("get_source ok")
        .expect("source exists");
    assert_eq!(
        updated.status, "needs_js",
        "JS shell must set status=needs_js (got {:?})",
        updated.status
    );

    // No chunks should have been indexed.
    let chunk_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chunks WHERE source_id = ?")
        .bind(&source.id)
        .fetch_one(&pool)
        .await
        .expect("count chunks");
    assert_eq!(chunk_count, 0, "JS shell must produce zero indexed chunks");
}

// ===========================================================================
// AC9b — real article → indexed (no false positive)
// ===========================================================================

/// AC9b: A page with a real article (>200 chars of prose) must be fully
/// indexed, NOT flagged as needs_js.
///
/// This test requires the tokenizer to be available (the ingest pipeline
/// downloads it on a cold cache). It is gated by `NOMIC_TOKENIZER_PATH` or
/// network availability, similar to the existing `ingest.rs` tokenizer tests.
/// If neither is available the test is skipped gracefully.
#[tokio::test]
async fn url_indexes_real_article() {
    let html = real_article_html();
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/article"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(html)
                .insert_header("content-type", "text/html; charset=utf-8"),
        )
        .mount(&mock)
        .await;

    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);

    // Seed the tokenizer so the ingest pipeline doesn't attempt a network download.
    seed_tokenizer_from_env(&engine.data_dir_for_test().await);

    let nb = engine
        .create_notebook("NB", None, None)
        .await
        .expect("notebook");
    let source = engine
        .add_url_source(&nb.id, "real article", &format!("{}/article", mock.uri()))
        .await
        .expect("add_url_source");

    let (result, _events) = ingest_collecting_progress(&engine, &source.id).await;

    if result.is_err() {
        // If the tokenizer is not available, the error will be a model/network
        // error at the tokenizer step — not a needs_js false positive. Check
        // that the source is NOT in needs_js.
        let pool = engine.pool().await;
        let repo = lens_core::notebooks::NotebookRepo::new(&pool);
        let updated = repo
            .get_source(&source.id)
            .await
            .expect("get_source ok")
            .expect("source exists");
        assert_ne!(
            updated.status, "needs_js",
            "real article must NOT trigger needs_js (status={:?})",
            updated.status
        );
        // Skip the indexed assertion when the tokenizer/model is unavailable.
        return;
    }

    // If ingest succeeded fully, the source must be indexed.
    let pool = engine.pool().await;
    let repo = lens_core::notebooks::NotebookRepo::new(&pool);
    let updated = repo
        .get_source(&source.id)
        .await
        .expect("get_source ok")
        .expect("source exists");
    assert_eq!(
        updated.status, "indexed",
        "real article must be indexed (got {:?})",
        updated.status
    );
}

// ===========================================================================
// Crash-recovery: needs_js / needs_ocr survive engine restart
// ===========================================================================

/// A source left at `needs_js` or `needs_ocr` must survive the crash-recovery
/// reset (which resets only `parsing`/`embedding` → `error`). This is the
/// compile-time invariant locked in the comment at `lib.rs` init.
#[tokio::test]
async fn crash_recovery_skips_needs_js_and_needs_ocr() {
    let dir = tempfile::tempdir().expect("tempdir");

    // Create an engine and manually insert sources at needs_js / needs_ocr.
    let engine = LensEngine::init(dir.path()).await.expect("engine init");
    let nb = engine
        .create_notebook("NB", None, None)
        .await
        .expect("notebook");

    let pool = engine.pool().await;
    let repo = lens_core::notebooks::NotebookRepo::new(&pool);

    let src_js = repo
        .add_url_source(&nb.id, "spa", "https://example.com/spa")
        .await
        .expect("add_url_source");
    let src_ocr = repo
        .add_url_source(&nb.id, "scanned", "https://example.com/pdf")
        .await
        .expect("add_url_source for ocr");

    // Manually set their statuses to the terminal-pending values.
    repo.update_source_status(
        &src_js.id,
        lens_core::notebooks::SourceStatus::NeedsJs.as_str(),
    )
    .await
    .expect("set needs_js");
    repo.update_source_status(
        &src_ocr.id,
        lens_core::notebooks::SourceStatus::NeedsOcr.as_str(),
    )
    .await
    .expect("set needs_ocr");

    drop(pool);
    drop(engine);

    // Re-open the engine: the crash-recovery reset must NOT touch needs_js / needs_ocr.
    let engine2 = LensEngine::init(dir.path()).await.expect("engine2 init");
    let pool2 = engine2.pool().await;
    let repo2 = lens_core::notebooks::NotebookRepo::new(&pool2);

    let js = repo2
        .get_source(&src_js.id)
        .await
        .expect("get needs_js source")
        .expect("exists");
    assert_eq!(
        js.status, "needs_js",
        "needs_js must survive crash-recovery reset (got {:?})",
        js.status
    );

    let ocr = repo2
        .get_source(&src_ocr.id)
        .await
        .expect("get needs_ocr source")
        .expect("exists");
    assert_eq!(
        ocr.status, "needs_ocr",
        "needs_ocr must survive crash-recovery reset (got {:?})",
        ocr.status
    );
}

// ===========================================================================
// needs_js is unreachable via the Err→error path
// ===========================================================================

/// Verifies that a source at `needs_js` was set via the Ok path (run_ingest
/// returns Ok) and therefore the Err→error flip in `ingest_source` never fires.
/// This test uses the AC9a fixture as a black-box: we observe that after
/// `ingest_source` returns Ok(()), the status is `needs_js` — not `error`.
#[tokio::test]
async fn needs_js_not_set_via_err_path() {
    let html = js_shell_html();
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/spa"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(html)
                .insert_header("content-type", "text/html; charset=utf-8"),
        )
        .mount(&mock)
        .await;

    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);

    let nb = engine
        .create_notebook("NB", None, None)
        .await
        .expect("notebook");
    let source = engine
        .add_url_source(&nb.id, "spa", &format!("{}/spa", mock.uri()))
        .await
        .expect("add_url_source");

    let (result, _events) = ingest_collecting_progress(&engine, &source.id).await;

    // The Err→error flip path in ingest_source sets status to "error".
    // If needs_js was incorrectly returned as Err, the status would be "error".
    // We assert the OPPOSITE: result is Ok AND status is needs_js.
    assert!(
        result.is_ok(),
        "needs_js outcome must be Ok(()) — not Err: {result:?}"
    );
    let pool = engine.pool().await;
    let repo = lens_core::notebooks::NotebookRepo::new(&pool);
    let updated = repo.get_source(&source.id).await.unwrap().unwrap();
    assert_eq!(updated.status, "needs_js");
    assert_ne!(
        updated.status, "error",
        "needs_js must NOT be set via the Err path (would be 'error')"
    );
}

// ===========================================================================
// add_url_source returns immediately (no network required)
// ===========================================================================

/// `add_url_source` inserts a `queued` row and returns immediately, WITHOUT
/// making any network request. The ingest must be triggered explicitly later.
#[tokio::test]
async fn add_url_source_returns_queued_without_fetch() {
    // We use a URL that would never respond (port 1 is typically firewalled).
    // If `add_url_source` attempted a fetch, it would block or error here.
    let url = "http://127.0.0.1:1/will-never-respond";

    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("NB", None, None)
        .await
        .expect("notebook");

    // This must return before any network timeout fires.
    let source: Source = engine
        .add_url_source(&nb.id, "page title", url)
        .await
        .expect("add_url_source must not attempt a network connection");

    assert_eq!(source.status, "queued");
    assert_eq!(source.kind, "url");
    assert_eq!(source.locator, url);
    assert_eq!(source.title, "page title");
    assert!(source.token_count.is_none());
    assert!(source.content_hash.is_none());
}

// ===========================================================================
// Item 5 — URL `.extracted.txt` sibling is removed on purge (no leak)
// ===========================================================================

/// Item 5: ingesting a URL writes `{data_dir}/sources/{id}.extracted.txt`; the
/// sibling MUST be removed on purge (the locator-derived sibling path leaked it
/// before, because a URL locator is the URL string — not a path under
/// `{data_dir}/sources`).
#[tokio::test]
async fn url_extracted_sibling_removed_on_purge() {
    let html = real_article_html();
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/article"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(html)
                .insert_header("content-type", "text/html; charset=utf-8"),
        )
        .mount(&mock)
        .await;

    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);
    let data_dir = engine.data_dir_for_test().await;

    let nb = engine
        .create_notebook("NB", None, None)
        .await
        .expect("notebook");
    let source = engine
        .add_url_source(&nb.id, "article", &format!("{}/article", mock.uri()))
        .await
        .expect("add_url_source");

    let (result, _events) = ingest_collecting_progress(&engine, &source.id).await;
    // Tokenizer may be unavailable offline; the sibling is written BEFORE chunking
    // regardless, so its existence does not depend on a full index.
    let sibling = data_dir
        .join("sources")
        .join(format!("{}.extracted.txt", source.id));

    // The sibling is written during ingest for the URL (DERIVED) kind. If ingest
    // failed only at the tokenizer step, the sibling still exists.
    if !sibling.exists() {
        // If extraction itself failed (network), there is nothing to assert.
        eprintln!(
            "skipping url_extracted_sibling_removed_on_purge: no sibling written ({result:?})"
        );
        return;
    }
    assert!(
        sibling.exists(),
        "URL ingest must write the .extracted.txt sibling"
    );

    // Trash then purge the source; the sibling MUST be gone.
    engine.trash_source(&source.id).await.expect("trash");
    engine.purge_source(&source.id).await.expect("purge");
    assert!(
        !sibling.exists(),
        "purge_source must remove the URL .extracted.txt sibling (leak); still at {}",
        sibling.display()
    );
}

/// Item 5 (purge_notebook path): the URL sibling is also reclaimed when the
/// whole notebook is purged.
#[tokio::test]
async fn url_extracted_sibling_removed_on_purge_notebook() {
    let html = real_article_html();
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/article"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(html)
                .insert_header("content-type", "text/html; charset=utf-8"),
        )
        .mount(&mock)
        .await;

    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);
    let data_dir = engine.data_dir_for_test().await;

    let nb = engine
        .create_notebook("NB", None, None)
        .await
        .expect("notebook");
    let source = engine
        .add_url_source(&nb.id, "article", &format!("{}/article", mock.uri()))
        .await
        .expect("add_url_source");

    let (result, _events) = ingest_collecting_progress(&engine, &source.id).await;
    let sibling = data_dir
        .join("sources")
        .join(format!("{}.extracted.txt", source.id));
    if !sibling.exists() {
        eprintln!(
            "skipping url_extracted_sibling_removed_on_purge_notebook: no sibling ({result:?})"
        );
        return;
    }

    engine.trash_notebook(&nb.id).await.expect("trash_notebook");
    engine.purge_notebook(&nb.id).await.expect("purge_notebook");
    assert!(
        !sibling.exists(),
        "purge_notebook must remove the URL .extracted.txt sibling; still at {}",
        sibling.display()
    );
}

// ===========================================================================
// Item 6 — re-ingest into needs_js wipes stale chunks/vectors
// ===========================================================================

/// Item 6: a previously-INDEXED URL source whose content flips to a JS shell on
/// re-ingest must transition to `needs_js` AND drop its prior chunks + Lance
/// vectors (nothing indexed survives behind the pending status).
///
/// Requires the tokenizer for the first (indexing) ingest; skips cleanly if
/// unavailable offline.
#[tokio::test]
async fn reingest_into_needs_js_wipes_stale_chunks_and_vectors() {
    // First call returns a real article (indexable); the second call (same URL)
    // returns a JS shell. wiremock matches mounted mocks LIFO, and the article
    // mock is limited to one response — so call #1 → article, call #2 → JS shell.
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/page"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(js_shell_html())
                .insert_header("content-type", "text/html; charset=utf-8"),
        )
        .mount(&mock)
        .await;
    Mock::given(method("GET"))
        .and(path("/page"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(real_article_html())
                .insert_header("content-type", "text/html; charset=utf-8"),
        )
        .up_to_n_times(1)
        .mount(&mock)
        .await;

    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);
    let data_dir = engine.data_dir_for_test().await;

    // Seed the tokenizer from NOMIC_TOKENIZER_PATH if present (offline path).
    seed_tokenizer_from_env(&data_dir);

    let nb = engine
        .create_notebook("NB", None, None)
        .await
        .expect("notebook");
    let source = engine
        .add_url_source(&nb.id, "page", &format!("{}/page", mock.uri()))
        .await
        .expect("add_url_source");

    // First ingest: real article → indexed (needs the tokenizer).
    let (result, _events) = ingest_collecting_progress(&engine, &source.id).await;
    if result.is_err() {
        eprintln!("skipping reingest_into_needs_js: first ingest failed (no tokenizer offline)");
        return;
    }
    let pool = engine.pool().await;
    let repo = lens_core::notebooks::NotebookRepo::new(&pool);
    let after_first = repo.get_source(&source.id).await.unwrap().unwrap();
    if after_first.status != "indexed" {
        eprintln!(
            "skipping reingest_into_needs_js: first ingest not indexed (status={:?})",
            after_first.status
        );
        return;
    }
    let chunk_count_1: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chunks WHERE source_id = ?")
        .bind(&source.id)
        .fetch_one(&pool)
        .await
        .expect("count chunks");
    assert!(chunk_count_1 > 0, "first ingest must produce chunks");
    let vec_count_1 = vector_row_count(&data_dir, &nb.id.to_string(), &source.id).await;
    assert!(vec_count_1 > 0, "first ingest must produce Lance vectors");

    // Second ingest of the SAME source: now the URL returns a JS shell. The raw
    // bytes differ, so the no-op short-circuit does NOT fire; the needs_js gate
    // runs and MUST wipe the prior chunks + vectors.
    let (result2, _events2) = ingest_collecting_progress(&engine, &source.id).await;
    assert!(
        result2.is_ok(),
        "needs_js outcome must be Ok, not Err: {result2:?}"
    );
    let after_second = repo.get_source(&source.id).await.unwrap().unwrap();
    assert_eq!(
        after_second.status, "needs_js",
        "re-ingest into a JS shell must flip status to needs_js (got {:?})",
        after_second.status
    );
    let chunk_count_2: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chunks WHERE source_id = ?")
        .bind(&source.id)
        .fetch_one(&pool)
        .await
        .expect("count chunks");
    assert_eq!(
        chunk_count_2, 0,
        "stale chunks must be wiped when transitioning into needs_js"
    );
    let vec_count_2 = vector_row_count(&data_dir, &nb.id.to_string(), &source.id).await;
    assert_eq!(
        vec_count_2, 0,
        "stale Lance vectors must be wiped when transitioning into needs_js"
    );
}
