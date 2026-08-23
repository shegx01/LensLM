//! AC-4.9: `relocate_data_dir` promises that any failure after the target exists
//! leaves nothing behind — a partial corpus plus a copied `config.json` holding a
//! plaintext API key would otherwise strand in a folder the user picked.
//!
//! Nothing pinned that promise. Written before Unit 4 changes the rollback, so the
//! change is a behaviour edit against a green test rather than an unverified one.

#![cfg(unix)]

mod common;

use common::pin_catalog_offline;
use lens_core::LensEngine;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

/// Fails the relocation *after* `create_dir_all(to)` succeeds: `validate_target`
/// accepts an existing empty dir, `create_dir_all` no-ops on it, and `VACUUM INTO`
/// then hits EACCES. That is exactly the window the rollback promise covers.
fn unwritable_target(parent: &Path) -> std::path::PathBuf {
    let to = parent.join("moved");
    std::fs::create_dir_all(&to).expect("create target");
    std::fs::set_permissions(&to, std::fs::Permissions::from_mode(0o500)).expect("chmod target");
    to
}

async fn engine_with_corpus(dir: &Path) -> lens_core::LensEngine {
    pin_catalog_offline(dir);
    let engine = LensEngine::init(dir).await.expect("init engine");
    std::fs::write(
        dir.join("config.json"),
        br#"{"schema_version":1,"api_key":"sk-secret"}"#,
    )
    .expect("seed config");
    std::fs::create_dir_all(dir.join("sources")).expect("sources dir");
    std::fs::write(dir.join("sources").join("a.txt"), b"corpus").expect("seed corpus");
    engine
}

#[tokio::test]
async fn a_failed_relocation_strands_no_corpus_and_no_secrets() {
    let from = tempfile::tempdir().expect("from");
    let to_parent = tempfile::tempdir().expect("to_parent");
    let engine = engine_with_corpus(from.path()).await;
    let pool = engine.pool().await;

    let to = unwritable_target(to_parent.path());
    let err = lens_core::relocate::relocate_data_dir(&pool, from.path(), &to, &[]).await;
    assert!(
        err.is_err(),
        "the fixture must actually fail the relocation"
    );

    // Survives the AC-4.7 change: whether or not the directory itself is kept, none
    // of these may remain in a folder the user chose.
    for rel in ["config.json", "sources", "lens.db", "lancedb"] {
        assert!(
            !to.join(rel).exists(),
            "{rel} stranded in the target after a failed relocation"
        );
    }
    pool.close().await;
}

/// AC-4.7: the user's chosen folder survives a failed relocation. It previously
/// did not — the whole directory was removed, which both surprises the user and,
/// combined with Unit 4 raising the failure rate, would have become common.
#[tokio::test]
async fn a_failed_relocation_preserves_the_target_directory() {
    let from = tempfile::tempdir().expect("from");
    let to_parent = tempfile::tempdir().expect("to_parent");
    let engine = engine_with_corpus(from.path()).await;
    let pool = engine.pool().await;

    let to = unwritable_target(to_parent.path());
    assert!(to.exists(), "control: the target exists before the attempt");

    let err = lens_core::relocate::relocate_data_dir(&pool, from.path(), &to, &[]).await;
    assert!(
        err.is_err(),
        "the fixture must actually fail the relocation"
    );

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
