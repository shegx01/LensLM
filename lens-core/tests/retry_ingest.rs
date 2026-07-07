// issue #71: the streamed-ingest future is deep enough to overflow the default
// 128-frame `Send` auto-trait evaluation (E0275) in the integration-test crate.
#![recursion_limit = "256"]
//! Issue #73 — structured ingest-failure reason (`error_meta`) + in-place retry.
//!
//! # Failure seam (offline)
//!
//! Most tests here force a DETERMINISTIC, OFFLINE ingest failure: add a managed
//! text source (which writes `{data_dir}/sources/{id}.{ext}`), then DELETE that
//! locator file before ingesting. `run_ingest` fails at its pre-read
//! `tokio::fs::metadata` guard with a `LensError::Io`, BEFORE any tokenizer /
//! embedder load — so these tests never touch the network.
//!
//! The single success-path test (`retry_success_clears_error_meta_and_indexes`)
//! needs a real ingest, so it is gated behind `tokenizer_available()` and skips
//! cleanly offline, mirroring the rest of the ingest suite.

use lens_core::{IngestProgress, LensEngine, NotebookId};

mod support;
use support::{inject_counting_engine, tokenizer_available};

/// Adds a managed text source, then deletes its on-disk locator so the next
/// ingest fails deterministically at the pre-read metadata guard. Returns the
/// source id.
async fn add_source_then_break_it(engine: &LensEngine, nb: &NotebookId, title: &str) -> String {
    let src = engine
        .add_text_source(nb, title, "some body text for the source", "text")
        .await
        .expect("add text source")
        .source;
    // Remove the managed file so `run_ingest`'s `tokio::fs::metadata` fails.
    std::fs::remove_file(&src.locator).expect("delete managed source file");
    src.id
}

/// Runs an ingest expecting it to FAIL (the locator was deleted).
async fn expect_ingest_fail(engine: &LensEngine, id: &str) {
    let result = engine.ingest_source(id, |_p: IngestProgress| {}).await;
    assert!(result.is_err(), "ingest should fail for a deleted locator");
}

/// Fetches a single source out of `list_sources` by id.
async fn source_by_id(engine: &LensEngine, nb: &NotebookId, id: &str) -> lens_core::Source {
    engine
        .list_sources(nb)
        .await
        .expect("list_sources")
        .into_iter()
        .find(|s| s.id == id)
        .expect("source present in list")
}

#[tokio::test]
async fn ingest_fail_persists_error_meta_survives_list_sources() {
    let (_dir, engine) = inject_counting_engine().await;
    let nb = engine.create_notebook("err-nb", None, None).await.unwrap();
    let id = add_source_then_break_it(&engine, &nb.id, "broken").await;

    expect_ingest_fail(&engine, &id).await;

    // A FRESH read (list_sources) returns the persisted error_meta — proving it
    // survives past the in-memory ingest call (would survive an app restart too).
    let src = source_by_id(&engine, &nb.id, &id).await;
    assert_eq!(src.status, lens_core::notebooks::SourceStatus::Error);
    let json = src.error_meta.expect("error_meta persisted on failure");
    let meta: lens_core::ErrorMeta = serde_json::from_str(&json).expect("error_meta parses");
    assert_eq!(meta.kind, "Io", "deleted-locator failure maps to Io");
    assert!(!meta.message.is_empty());
    assert_eq!(meta.attempt_count, 1, "first failure ⇒ attempt_count 1");
    assert!(!meta.timestamp.is_empty());
}

#[tokio::test]
async fn retry_guard_rejects_non_error_source() {
    let (_dir, engine) = inject_counting_engine().await;
    let nb = engine
        .create_notebook("guard-nb", None, None)
        .await
        .unwrap();
    // A freshly-added text source is `queued`, NOT `error`.
    let src = engine
        .add_text_source(&nb.id, "fresh", "body", "text")
        .await
        .unwrap()
        .source;
    assert_ne!(src.status, lens_core::notebooks::SourceStatus::Error);

    let result = engine.retry_source(&src.id, |_p: IngestProgress| {}).await;
    assert!(
        result.is_err(),
        "retrying a non-error source must be rejected"
    );
    // The source was NOT transitioned.
    let after = source_by_id(&engine, &nb.id, &src.id).await;
    assert_eq!(
        after.status, src.status,
        "status unchanged after guard reject"
    );
}

#[tokio::test]
async fn retry_guard_rejects_trashed_source() {
    let (_dir, engine) = inject_counting_engine().await;
    let nb = engine
        .create_notebook("trash-nb", None, None)
        .await
        .unwrap();
    let id = add_source_then_break_it(&engine, &nb.id, "to-trash").await;
    expect_ingest_fail(&engine, &id).await; // now `error` with error_meta

    engine.trash_source(&id).await.expect("trash source");

    let result = engine.retry_source(&id, |_p: IngestProgress| {}).await;
    assert!(
        result.is_err(),
        "retrying a trashed (errored) source must be rejected"
    );
}

#[tokio::test]
async fn fail_then_retry_fail_increments_attempt_count_to_2() {
    let (_dir, engine) = inject_counting_engine().await;
    let nb = engine
        .create_notebook("count-nb", None, None)
        .await
        .unwrap();
    let id = add_source_then_break_it(&engine, &nb.id, "twice").await;

    // First failure ⇒ attempt_count 1.
    expect_ingest_fail(&engine, &id).await;
    let meta1 = parse_meta(&source_by_id(&engine, &nb.id, &id).await);
    assert_eq!(meta1.attempt_count, 1);

    // Retry — the locator is still missing, so it fails again ⇒ attempt_count 2.
    let retry = engine.retry_source(&id, |_p: IngestProgress| {}).await;
    assert!(
        retry.is_err(),
        "retry should fail again (locator still gone)"
    );

    let after = source_by_id(&engine, &nb.id, &id).await;
    assert_eq!(
        after.status,
        lens_core::notebooks::SourceStatus::Error,
        "still errored after failed retry"
    );
    let meta2 = parse_meta(&after);
    assert_eq!(meta2.attempt_count, 2, "failed retry increments to 2");
}

#[tokio::test]
async fn retry_preserves_row_identity_no_new_row() {
    let (_dir, engine) = inject_counting_engine().await;
    let nb = engine.create_notebook("id-nb", None, None).await.unwrap();
    let id = add_source_then_break_it(&engine, &nb.id, "identity").await;
    expect_ingest_fail(&engine, &id).await;

    let before = source_by_id(&engine, &nb.id, &id).await;
    let count_before = engine.list_sources(&nb.id).await.unwrap().len();

    // Retry (fails again — locator still gone), but the row must be reused.
    let _ = engine.retry_source(&id, |_p: IngestProgress| {}).await;

    let sources = engine.list_sources(&nb.id).await.unwrap();
    assert_eq!(
        sources.len(),
        count_before,
        "retry must NOT create a new row (dedup regression guard)"
    );
    let after = sources.into_iter().find(|s| s.id == id).unwrap();
    assert_eq!(after.id, before.id, "same id");
    assert_eq!(after.created_at, before.created_at, "same created_at/order");
    assert_eq!(after.selected, before.selected, "selected flag preserved");
}

#[tokio::test]
async fn retry_success_clears_error_meta_and_indexes() {
    if !tokenizer_available().await {
        eprintln!("skipping retry_success_clears_error_meta_and_indexes: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;
    let nb = engine
        .create_notebook("success-nb", None, None)
        .await
        .unwrap();

    // Add a text source, capture its managed locator + body, then DELETE the file
    // so the first ingest fails at the metadata guard.
    let src = engine
        .add_text_source(&nb.id, "recoverable", "recoverable body text", "text")
        .await
        .unwrap()
        .source;
    let locator = src.locator.clone();
    std::fs::remove_file(&locator).unwrap();

    expect_ingest_fail(&engine, &src.id).await;
    let errored = source_by_id(&engine, &nb.id, &src.id).await;
    assert_eq!(errored.status, lens_core::notebooks::SourceStatus::Error);
    assert!(errored.error_meta.is_some());

    // Restore the locator file so the retry can succeed.
    std::fs::write(&locator, "recoverable body text").unwrap();

    engine
        .retry_source(&src.id, |_p: IngestProgress| {})
        .await
        .expect("retry should succeed once the locator is restored");

    let recovered = source_by_id(&engine, &nb.id, &src.id).await;
    assert_eq!(
        recovered.status,
        lens_core::notebooks::SourceStatus::Indexed,
        "successful retry ⇒ indexed"
    );
    assert!(
        recovered.error_meta.is_none(),
        "successful retry clears error_meta"
    );
    assert_eq!(recovered.id, src.id, "same row id preserved through retry");
}

// ---------------------------------------------------------------------------
// small local helpers
// ---------------------------------------------------------------------------

fn parse_meta(src: &lens_core::Source) -> lens_core::ErrorMeta {
    serde_json::from_str(src.error_meta.as_ref().expect("error_meta present")).expect("meta parses")
}
