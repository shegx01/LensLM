//! AC-4.9: a failure after the target exists must strand no corpus and no copied
//! `config.json` holding a plaintext key. The failure has to land AFTER the copy —
//! an early one copies nothing, making every assertion here vacuous.

#![cfg(unix)]

mod common;

use common::pin_catalog_offline;
use lens_core::LensEngine;
use std::path::Path;

/// Fails verification with a fully populated target: the copy runs, then the
/// AC-4.2 `ready`-audio check rejects it because `notebooks` was skipped. This is
/// the only lever available that fails late enough to exercise the rollback.
const SKIP_TO_FAIL_LATE: &[&str] = &["notebooks"];

async fn engine_with_corpus_and_secret(dir: &Path) -> LensEngine {
    pin_catalog_offline(dir);
    let engine = LensEngine::init(dir).await.expect("init engine");
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();

    std::fs::write(
        dir.join("config.json"),
        br#"{"schema_version":1,"api_key":"sk-secret"}"#,
    )
    .expect("seed config");
    std::fs::write(dir.join("config.json.bak"), br#"{"api_key":"sk-secret"}"#)
        .expect("seed config backup");
    std::fs::create_dir_all(dir.join("sources")).expect("sources dir");
    std::fs::write(dir.join("sources").join("a.txt"), b"corpus").expect("seed corpus");

    // A `ready` overview whose file EXISTS: once the copy skips `notebooks`, the
    // AC-4.2 check sees source-present / copy-missing and cancels the move.
    let wav = dir.join("notebooks").join(&nb).join("ok.wav");
    std::fs::create_dir_all(wav.parent().expect("parent")).expect("mkdir");
    std::fs::write(&wav, b"audio").expect("seed wav");
    sqlx::query(
        "INSERT INTO audio_overviews (notebook_id, path, generated_at, status, source_set_hash) \
         VALUES (?, ?, 't', 'ready', 'h')",
    )
    .bind(&nb)
    .bind(wav.display().to_string())
    .execute(&engine.pool().await)
    .await
    .expect("seed audio row");

    engine
}

#[tokio::test]
async fn a_failed_relocation_strands_no_corpus_and_no_secrets() {
    let from = tempfile::tempdir().expect("from");
    let to_parent = tempfile::tempdir().expect("to_parent");
    let to = to_parent.path().join("moved");
    let engine = engine_with_corpus_and_secret(from.path()).await;
    let pool = engine.pool().await;

    let err = lens_core::relocate::relocate_data_dir(&pool, from.path(), &to, SKIP_TO_FAIL_LATE)
        .await
        .expect_err("the fixture must fail the relocation");
    assert!(
        matches!(err, lens_core::LensError::Io(_)),
        "must fail at verification, not earlier — an early failure copies nothing \
         and makes every assertion below vacuous: {err:?}"
    );

    for rel in [
        "config.json",
        "config.json.bak",
        "sources",
        "lens.db",
        "lancedb",
    ] {
        assert!(
            !to.join(rel).exists(),
            "{rel} stranded in the target after a failed relocation"
        );
    }
    pool.close().await;
}

/// AC-4.7: the folder the user chose survives; only its contents go.
#[tokio::test]
async fn a_failed_relocation_preserves_the_target_directory() {
    let from = tempfile::tempdir().expect("from");
    let to_parent = tempfile::tempdir().expect("to_parent");
    let to = to_parent.path().join("moved");
    let engine = engine_with_corpus_and_secret(from.path()).await;
    let pool = engine.pool().await;

    lens_core::relocate::relocate_data_dir(&pool, from.path(), &to, SKIP_TO_FAIL_LATE)
        .await
        .expect_err("the fixture must fail the relocation");

    assert!(
        to.exists(),
        "the folder the user chose must survive a failed relocation"
    );
    let leftovers: Vec<_> = std::fs::read_dir(&to)
        .expect("read target")
        .flatten()
        .map(|e| e.file_name())
        .collect();
    assert!(
        leftovers.is_empty(),
        "the target must be left empty, found {leftovers:?}"
    );
    pool.close().await;
}
