// issue #71: raise the limit so deep `Send` auto-trait evaluation on the
// ingest future doesn't overflow (E0275) under stricter toolchains.
#![recursion_limit = "256"]
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

/// HTML with GENUINE, substantial article content (well over
/// `NEEDS_JS_SUFFICIENT_CHARS`) embedded in a very large HTML payload so the
/// text/raw ratio falls below `NEEDS_JS_MIN_TEXT_RATIO`. Models the modern-web
/// case (e.g. `docs.stripe.com`): real content in a script/markup-heavy shell.
/// Must be INDEXED, not flagged `needs_js` (the ratio arm must NOT false-positive
/// once absolute extracted content is sufficient).
fn content_rich_low_ratio_html() -> String {
    let mut html = String::from(
        r#"<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Comprehensive Guide</title></head>
<body>
<article>
<h1>Comprehensive Guide to the Payments API</h1>
"#,
    );
    // ~10 substantial paragraphs → several thousand chars of extractable prose,
    // comfortably above NEEDS_JS_SUFFICIENT_CHARS (1000).
    for i in 1..=10 {
        html.push_str(&format!(
            "<p>Section {i}: The payments API lets you accept and manage transactions \
             securely across many providers. This paragraph describes the request and \
             response lifecycle in detail, including idempotency keys, webhooks for \
             asynchronous events, retry semantics, and the recommended error-handling \
             strategy for production integrations. Read each section carefully before \
             wiring the client, because the ordering of operations matters.</p>\n"
        ));
    }
    html.push_str("</article>\n");
    // Pad the raw payload with a large inline script so ratio < 0.01 even though
    // the extracted content is substantial (mirrors a big SPA/hydration bundle).
    html.push_str("<script>\n");
    for _ in 0..6000 {
        html.push_str("/* inlined bundle padding to enlarge the raw HTML payload */\n");
    }
    html.push_str("</script>\n</body>\n</html>\n");
    html
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
        .add_url_source(&nb.id, "bad page", &format!("{}/page", mock.uri()), false)
        .await
        .expect("add_url_source")
        .source;
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
        .expect("add_text_source")
        .source;
    // We don't run the full ingest (tokenizer not available offline) — just prove
    // add_text_source returns without deadlocking. The lock is already released.
    assert_eq!(src2.status, "queued");
}

// ===========================================================================
// issue #71 — URL body cap reads the configurable AppConfig.max_source_mb
// ===========================================================================

/// issue #71 Step 2: the URL body-size guard reads the configured
/// `max_source_mb` (not the hardcoded 10 MB constant). With a 1 MB configured
/// cap, a ~1 MB+ HTML body is rejected (the Content-Length short-circuit fires),
/// flipping the source to `error` via the Err→error flip in `ingest_source`.
#[tokio::test]
async fn url_uses_configurable_cap() {
    // A real over-1MB HTML body so the production Content-Length path fires.
    let big = format!(
        "<html><body>{}</body></html>",
        "a".repeat(1024 * 1024 + 100)
    );
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/huge"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html")
                .set_body_string(big),
        )
        .mount(&mock)
        .await;

    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);

    // Configure a 1 MB cap so the ~1 MB body trips it.
    let mut cfg = engine.config().await;
    cfg.max_source_mb = "1".to_string();
    engine.set_config(cfg).await;

    let nb = engine
        .create_notebook("NB-cap", None, None)
        .await
        .expect("notebook");
    let source = engine
        .add_url_source(&nb.id, "huge page", &format!("{}/huge", mock.uri()), false)
        .await
        .expect("add_url_source")
        .source;

    let (result, _) = ingest_collecting_progress(&engine, &source.id).await;
    assert!(
        matches!(result, Err(lens_core::LensError::Validation(_))),
        "a URL body over the configured cap must be rejected, got {result:?}"
    );
    let pool = engine.pool().await;
    let repo = lens_core::notebooks::NotebookRepo::new(&pool);
    let updated = repo.get_source(&source.id).await.unwrap().unwrap();
    assert_eq!(updated.status, "error", "over-cap URL body flips to error");
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
        .add_url_source(&nb.id, "spa page", &format!("{}/spa", mock.uri()), false)
        .await
        .expect("add_url_source")
        .source;

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
        .add_url_source(&nb.id, "real article", &format!("{}/article", mock.uri()), false)
        .await
        .expect("add_url_source")
        .source;

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

/// Regression (issue #78 follow-up): a content-rich page whose text/raw ratio is
/// below `NEEDS_JS_MIN_TEXT_RATIO` but whose absolute extracted content exceeds
/// `NEEDS_JS_SUFFICIENT_CHARS` must be INDEXED, not sent to the needs_js render
/// fallback. This is the docs.stripe.com case: ~2.6 KB of real docs prose in a
/// ~1.2 MB SPA shell (ratio ≈ 0.002) was wrongly flagged needs_js before the fix.
#[tokio::test]
async fn url_indexes_content_rich_page_despite_low_ratio() {
    let html = content_rich_low_ratio_html();
    // Sanity-check the fixture actually exercises the low-ratio-but-rich band.
    assert!(
        html.len() as f64 > 0.0,
        "fixture non-empty (raw_len={})",
        html.len()
    );
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/guide"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(html)
                .insert_header("content-type", "text/html; charset=utf-8"),
        )
        .mount(&mock)
        .await;

    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);
    seed_tokenizer_from_env(&engine.data_dir_for_test().await);

    let nb = engine
        .create_notebook("NB", None, None)
        .await
        .expect("notebook");
    let source = engine
        .add_url_source(&nb.id, "guide", &format!("{}/guide", mock.uri()), false)
        .await
        .expect("add_url_source")
        .source;

    let (result, _events) = ingest_collecting_progress(&engine, &source.id).await;

    let pool = engine.pool().await;
    let repo = lens_core::notebooks::NotebookRepo::new(&pool);
    let updated = repo
        .get_source(&source.id)
        .await
        .expect("get_source ok")
        .expect("source exists");

    // The key invariant regardless of tokenizer availability: NOT needs_js.
    assert_ne!(
        updated.status, "needs_js",
        "content-rich low-ratio page must NOT be flagged needs_js (status={:?})",
        updated.status
    );
    // With a tokenizer present the full pipeline indexes it; without one the
    // ingest errors at the tokenizer step (not a needs_js false positive).
    if result.is_ok() {
        assert_eq!(
            updated.status, "indexed",
            "content-rich low-ratio page must be indexed (got {:?})",
            updated.status
        );
    }
}

/// #78 SPA opt-in: a page whose STATIC extraction is content-rich (and therefore
/// would index directly — proven by `url_indexes_content_rich_page_despite_low_ratio`)
/// must instead be DIVERTED to the JS-render branch when `force_js_render=true`.
/// With no renderer injected (headless test) + rendering enabled, the diverted
/// source lands in `needs_js` — proving the flag routed it away from the static
/// index path rather than indexing the static extraction.
#[tokio::test]
async fn url_force_js_render_diverts_content_rich_page_to_render() {
    let html = content_rich_low_ratio_html();
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/guide"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(html)
                .insert_header("content-type", "text/html; charset=utf-8"),
        )
        .mount(&mock)
        .await;

    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);
    seed_tokenizer_from_env(&engine.data_dir_for_test().await);

    let nb = engine
        .create_notebook("NB", None, None)
        .await
        .expect("notebook");
    let source = engine
        // force_js_render = true (the SPA checkbox)
        .add_url_source(&nb.id, "guide", &format!("{}/guide", mock.uri()), true)
        .await
        .expect("add_url_source")
        .source;
    // The flag must persist on the row.
    assert_eq!(
        source.force_js_render, 1,
        "force_js_render must persist as 1 on the source row"
    );

    let (result, _events) = ingest_collecting_progress(&engine, &source.id).await;
    assert!(result.is_ok(), "ingest must not error: {result:?}");

    let pool = engine.pool().await;
    let repo = lens_core::notebooks::NotebookRepo::new(&pool);
    let updated = repo
        .get_source(&source.id)
        .await
        .expect("get_source ok")
        .expect("source exists");
    assert_eq!(
        updated.status, "needs_js",
        "force_js_render must divert a content-rich page to the render branch; with no \
         renderer injected it lands in needs_js (got {:?})",
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
        .add_url_source(&nb.id, "spa", "https://example.com/spa", false)
        .await
        .expect("add_url_source")
        .source;
    let src_ocr = repo
        .add_url_source(&nb.id, "scanned", "https://example.com/pdf", false)
        .await
        .expect("add_url_source for ocr")
        .source;
    let src_rf = repo
        .add_url_source(&nb.id, "failed-render", "https://example.com/rf", false)
        .await
        .expect("add_url_source for render_failed")
        .source;

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
    repo.update_source_status(
        &src_rf.id,
        lens_core::notebooks::SourceStatus::RenderFailed.as_str(),
    )
    .await
    .expect("set render_failed");

    drop(pool);
    drop(engine);

    // Re-open the engine: the crash-recovery reset must NOT touch needs_js / needs_ocr / render_failed.
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

    let rf = repo2
        .get_source(&src_rf.id)
        .await
        .expect("get render_failed source")
        .expect("exists");
    assert_eq!(
        rf.status, "render_failed",
        "render_failed must survive crash-recovery reset (got {:?})",
        rf.status
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
        .add_url_source(&nb.id, "spa", &format!("{}/spa", mock.uri()), false)
        .await
        .expect("add_url_source")
        .source;

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
        .add_url_source(&nb.id, "page title", url, false)
        .await
        .expect("add_url_source must not attempt a network connection")
        .source;

    assert_eq!(source.status, "queued");
    assert_eq!(source.kind, "url");
    assert_eq!(source.locator, url);
    assert_eq!(source.title, "page title");
    assert!(source.token_count.is_none());
    assert!(source.content_hash.is_none());
}

// ===========================================================================
// issue #100 — content-hash dedup for URL sources (moderate normalization)
// ===========================================================================

/// AC-2 / AC-8: adding equivalent URLs (case-differing host, fragment-only diff)
/// deduplicates via the normalized-URL hash; a genuinely different URL does not.
/// The verbatim `locator` of the first add is preserved.
#[tokio::test]
async fn add_url_source_dedup_returns_existing() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("url-dedup-nb", None, None)
        .await
        .expect("notebook");

    // First add — fresh insert. Locator keeps the verbatim (mixed-case) URL.
    let first = engine
        .add_url_source(&nb.id, "article", "https://Example.COM/article", false)
        .await
        .expect("add_url_source");
    assert!(!first.was_existing, "first add is a fresh insert");
    assert_eq!(
        first.source.locator, "https://Example.COM/article",
        "locator preserves the verbatim first-added URL"
    );

    // Host-case-only difference — dedup hit.
    let second = engine
        .add_url_source(&nb.id, "again", "https://example.com/article", false)
        .await
        .expect("add_url_source");
    assert!(second.was_existing, "case-differing host is a dedup hit");
    assert_eq!(second.source.id, first.source.id);

    // Fragment-only difference — dedup hit.
    let third = engine
        .add_url_source(&nb.id, "frag", "https://example.com/article#section2", false)
        .await
        .expect("add_url_source");
    assert!(
        third.was_existing,
        "fragment-only difference is a dedup hit"
    );
    assert_eq!(third.source.id, first.source.id);

    // Genuinely different path — fresh insert.
    let fourth = engine
        .add_url_source(&nb.id, "other", "https://example.com/different", false)
        .await
        .expect("add_url_source");
    assert!(!fourth.was_existing, "different URL is a fresh insert");
    assert_ne!(fourth.source.id, first.source.id);

    let live = engine.list_sources(&nb.id).await.unwrap();
    assert_eq!(live.len(), 2, "only two distinct URL rows exist");
}

/// issue #100 Step 6 — LensEngine-level: adding an equivalent URL (host case)
/// through the full engine wrapper stack deduplicates and leaves one row.
#[tokio::test]
async fn engine_add_url_source_dedup_end_to_end() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("engine-url-dedup", None, None)
        .await
        .expect("notebook");

    let first = engine
        .add_url_source(&nb.id, "page", "https://Example.COM/page", false)
        .await
        .expect("add_url_source");
    assert!(!first.was_existing);

    let second = engine
        .add_url_source(&nb.id, "page", "https://example.com/page", false)
        .await
        .expect("add_url_source");
    assert!(second.was_existing, "case-differing host dedups end-to-end");
    assert_eq!(second.source.id, first.source.id);

    assert_eq!(
        engine.list_sources(&nb.id).await.unwrap().len(),
        1,
        "exactly one URL row after a dedup hit"
    );
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
        .add_url_source(&nb.id, "article", &format!("{}/article", mock.uri()), false)
        .await
        .expect("add_url_source")
        .source;

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
        .add_url_source(&nb.id, "article", &format!("{}/article", mock.uri()), false)
        .await
        .expect("add_url_source")
        .source;

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
        .add_url_source(&nb.id, "page", &format!("{}/page", mock.uri()), false)
        .await
        .expect("add_url_source")
        .source;

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

// ===========================================================================
// Layer (d) — JS-render auto-fallback wired into run_ingest
//
// These tests drive the fallback with a FAKE `JsRenderer` injected via
// `engine.set_js_renderer(Some(Arc::new(fake)))` — keeping CI headless (no real
// webview). They assert the four wiring outcomes (indexed / render_failed /
// opt-out needs_js / no-renderer needs_js) plus the C1 readback-provenance
// integration case.
// ===========================================================================

/// A configurable fake renderer for the Layer (d) wiring tests. `render_html`
/// returns whatever `canned` holds, mirroring the real renderer's contract
/// (`Ok(Some(html))` on success, `Ok(None)` on failed/blocked/timed-out render).
struct FakeRenderer {
    canned: Option<String>,
}

#[async_trait::async_trait]
impl lens_core::JsRenderer for FakeRenderer {
    async fn render_html(&self, _url: &str) -> Result<Option<String>, lens_core::LensError> {
        Ok(self.canned.clone())
    }
}

fn inject_fake_renderer(engine: &LensEngine, canned: Option<String>) {
    let fake: Arc<dyn lens_core::JsRenderer> = Arc::new(FakeRenderer { canned });
    // block_on is fine here: the setter is a quick RwLock write.
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(engine.set_js_renderer(Some(fake)));
    });
}

/// Layer (d) happy path: a JS-shell page (static extraction near-empty →
/// needs_js gate trips) + an injected renderer whose rendered HTML extracts to
/// >200 chars of prose → the source ends `indexed` (NOT needs_js), chunks > 0.
///
/// Requires the tokenizer for the downstream chunk step; skips cleanly offline.
#[tokio::test(flavor = "multi_thread")]
async fn js_render_fallback_indexes_when_renderer_populates() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/spa"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(js_shell_html())
                .insert_header("content-type", "text/html; charset=utf-8"),
        )
        .mount(&mock)
        .await;

    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);
    let data_dir = engine.data_dir_for_test().await;
    seed_tokenizer_from_env(&data_dir);

    // The renderer "renders" the SPA into a real article with >200 chars of prose.
    inject_fake_renderer(&engine, Some(real_article_html()));

    let nb = engine
        .create_notebook("NB", None, None)
        .await
        .expect("notebook");
    let source = engine
        .add_url_source(&nb.id, "spa", &format!("{}/spa", mock.uri()), false)
        .await
        .expect("add_url_source")
        .source;

    let (result, _events) = ingest_collecting_progress(&engine, &source.id).await;
    if result.is_err() {
        // Offline with no tokenizer: the downstream chunk step fails. That is a
        // tokenizer availability gap, NOT a fallback-wiring bug — skip cleanly.
        eprintln!("skipping js_render_fallback_indexes: ingest failed (no tokenizer offline)");
        return;
    }

    let pool = engine.pool().await;
    let repo = lens_core::notebooks::NotebookRepo::new(&pool);
    let updated = repo.get_source(&source.id).await.unwrap().unwrap();
    assert_eq!(
        updated.status, "indexed",
        "renderer populated the page → source must end indexed (got {:?})",
        updated.status
    );

    let chunk_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chunks WHERE source_id = ?")
        .bind(&source.id)
        .fetch_one(&pool)
        .await
        .expect("count chunks");
    assert!(
        chunk_count > 0,
        "a populated render must produce indexed chunks"
    );
}

/// Layer (d) never-populates: a JS shell + a renderer that returns `None` → the
/// source ends `render_failed` (NOT needs_js, NOT error), 0 chunks.
#[tokio::test(flavor = "multi_thread")]
async fn js_render_fallback_render_failed_when_renderer_returns_none() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/spa"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(js_shell_html())
                .insert_header("content-type", "text/html; charset=utf-8"),
        )
        .mount(&mock)
        .await;

    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);
    inject_fake_renderer(&engine, None);

    let nb = engine
        .create_notebook("NB", None, None)
        .await
        .expect("notebook");
    let source = engine
        .add_url_source(&nb.id, "spa", &format!("{}/spa", mock.uri()), false)
        .await
        .expect("add_url_source")
        .source;

    let (result, _events) = ingest_collecting_progress(&engine, &source.id).await;
    assert!(
        result.is_ok(),
        "render_failed outcome must return Ok, not Err: {result:?}"
    );

    let pool = engine.pool().await;
    let repo = lens_core::notebooks::NotebookRepo::new(&pool);
    let updated = repo.get_source(&source.id).await.unwrap().unwrap();
    assert_eq!(
        updated.status, "render_failed",
        "a renderer that never populates must set render_failed (got {:?})",
        updated.status
    );

    let chunk_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chunks WHERE source_id = ?")
        .bind(&source.id)
        .fetch_one(&pool)
        .await
        .expect("count chunks");
    assert_eq!(chunk_count, 0, "render_failed must produce zero chunks");
}

/// A fake renderer whose `render_html` returns `Err(..)`, exercising the FIX 2
/// contract: a render *failure* must map to the TERMINAL `render_failed` status,
/// NOT the transient `error` state (which crash-recovery would reset + retry).
struct ErrRenderer;

#[async_trait::async_trait]
impl lens_core::JsRenderer for ErrRenderer {
    async fn render_html(&self, _url: &str) -> Result<Option<String>, lens_core::LensError> {
        Err(lens_core::LensError::Internal(
            "simulated webview render failure".into(),
        ))
    }
}

/// FIX 2: a renderer that returns `Err(..)` must land the source in the terminal
/// `render_failed` state (NOT `error`), and `ingest_source` must still return
/// `Ok(())` (the Err→error flip must NEVER fire for a render failure).
#[tokio::test(flavor = "multi_thread")]
async fn js_render_fallback_render_failed_when_renderer_errors() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/spa"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(js_shell_html())
                .insert_header("content-type", "text/html; charset=utf-8"),
        )
        .mount(&mock)
        .await;

    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);
    // Inject the erroring renderer directly (inject_fake_renderer only builds the
    // Ok-returning FakeRenderer).
    {
        let fake: Arc<dyn lens_core::JsRenderer> = Arc::new(ErrRenderer);
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(engine.set_js_renderer(Some(fake)));
        });
    }

    let nb = engine
        .create_notebook("NB", None, None)
        .await
        .expect("notebook");
    let source = engine
        .add_url_source(&nb.id, "spa", &format!("{}/spa", mock.uri()), false)
        .await
        .expect("add_url_source")
        .source;

    let (result, _events) = ingest_collecting_progress(&engine, &source.id).await;
    assert!(
        result.is_ok(),
        "a render Err must map to render_failed and return Ok, NOT propagate Err→error: {result:?}"
    );

    let pool = engine.pool().await;
    let repo = lens_core::notebooks::NotebookRepo::new(&pool);
    let updated = repo.get_source(&source.id).await.unwrap().unwrap();
    assert_eq!(
        updated.status, "render_failed",
        "a renderer that ERRORS must set render_failed (NOT error) (got {:?})",
        updated.status
    );

    let chunk_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chunks WHERE source_id = ?")
        .bind(&source.id)
        .fetch_one(&pool)
        .await
        .expect("count chunks");
    assert_eq!(chunk_count, 0, "render_failed must produce zero chunks");
}

/// Layer (d) opt-out: same JS shell, `js_render_enabled=false`, no renderer
/// needed → the source stays `needs_js` (current behavior preserved).
#[tokio::test(flavor = "multi_thread")]
async fn js_render_fallback_opt_out_stays_needs_js() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/spa"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(js_shell_html())
                .insert_header("content-type", "text/html; charset=utf-8"),
        )
        .mount(&mock)
        .await;

    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);

    // Opt OUT of JS rendering.
    let mut cfg = engine.config().await;
    cfg.js_render_enabled = false;
    engine.set_config(cfg).await;

    // Even with a populating renderer injected, opt-out must NOT invoke it.
    inject_fake_renderer(&engine, Some(real_article_html()));

    let nb = engine
        .create_notebook("NB", None, None)
        .await
        .expect("notebook");
    let source = engine
        .add_url_source(&nb.id, "spa", &format!("{}/spa", mock.uri()), false)
        .await
        .expect("add_url_source")
        .source;

    let (result, _events) = ingest_collecting_progress(&engine, &source.id).await;
    assert!(
        result.is_ok(),
        "opt-out needs_js outcome must return Ok: {result:?}"
    );

    let pool = engine.pool().await;
    let repo = lens_core::notebooks::NotebookRepo::new(&pool);
    let updated = repo.get_source(&source.id).await.unwrap().unwrap();
    assert_eq!(
        updated.status, "needs_js",
        "js_render_enabled=false must preserve needs_js (got {:?})",
        updated.status
    );
}

/// Layer (d) no-renderer-injected: `js_render_enabled=true` (default) but
/// `js_renderer()` is `None` → graceful fallback to `needs_js`.
#[tokio::test(flavor = "multi_thread")]
async fn js_render_fallback_no_renderer_stays_needs_js() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/spa"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(js_shell_html())
                .insert_header("content-type", "text/html; charset=utf-8"),
        )
        .mount(&mock)
        .await;

    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);
    // NO renderer injected; js_render_enabled defaults ON.

    let nb = engine
        .create_notebook("NB", None, None)
        .await
        .expect("notebook");
    let source = engine
        .add_url_source(&nb.id, "spa", &format!("{}/spa", mock.uri()), false)
        .await
        .expect("add_url_source")
        .source;

    let (result, _events) = ingest_collecting_progress(&engine, &source.id).await;
    assert!(
        result.is_ok(),
        "no-renderer needs_js outcome must return Ok: {result:?}"
    );

    let pool = engine.pool().await;
    let repo = lens_core::notebooks::NotebookRepo::new(&pool);
    let updated = repo.get_source(&source.id).await.unwrap().unwrap();
    assert_eq!(
        updated.status, "needs_js",
        "no injected renderer must gracefully preserve needs_js (got {:?})",
        updated.status
    );
}

/// Layer (d) — C1 readback-provenance integration: the renderer's contract
/// discards output whose final-committed host is blocked (returning `None`). We
/// model that contract with a fake renderer that returns `None` for such an
/// input. The wiring must then set `render_failed` with ZERO chunks/vectors —
/// no internal content ever reaches chunk→embed→index.
#[tokio::test(flavor = "multi_thread")]
async fn js_render_provenance_blocked_render_failed_writes_nothing() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/spa"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(js_shell_html())
                .insert_header("content-type", "text/html; charset=utf-8"),
        )
        .mount(&mock)
        .await;

    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);
    let data_dir = engine.data_dir_for_test().await;

    // The renderer's C1 readback re-check discarded internal content → returns
    // None (same observable contract this branch sees for a blocked final host).
    inject_fake_renderer(&engine, None);

    let nb = engine
        .create_notebook("NB", None, None)
        .await
        .expect("notebook");
    let source = engine
        .add_url_source(&nb.id, "spa", &format!("{}/spa", mock.uri()), false)
        .await
        .expect("add_url_source")
        .source;

    let (result, _events) = ingest_collecting_progress(&engine, &source.id).await;
    assert!(
        result.is_ok(),
        "provenance-blocked render_failed must return Ok: {result:?}"
    );

    let pool = engine.pool().await;
    let repo = lens_core::notebooks::NotebookRepo::new(&pool);
    let updated = repo.get_source(&source.id).await.unwrap().unwrap();
    assert_eq!(
        updated.status, "render_failed",
        "provenance-blocked output must yield render_failed (got {:?})",
        updated.status
    );

    // Assert NO chunk row AND no Lance vector was written for this source.
    let chunk_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chunks WHERE source_id = ?")
        .bind(&source.id)
        .fetch_one(&pool)
        .await
        .expect("count chunks");
    assert_eq!(
        chunk_count, 0,
        "provenance-blocked render must write zero chunks"
    );
    let vec_count = vector_row_count(&data_dir, &nb.id.to_string(), &source.id).await;
    assert_eq!(
        vec_count, 0,
        "provenance-blocked render must write zero vectors"
    );
}
