//! AC-0.1 — copy completeness for data-dir relocation.
//!
//! `copy_tree` is a DENY-list copier driven by `COPY_SKIP`, not an allow-list over
//! any enumeration of known entries. Re-expressing it as an allow-list would stop
//! copying `models`/`hf-cache` and every future unknown top-level dir, silently.
//! This test exists so that behaviour has a failing test before the #248 layout
//! descriptor refactor touches it.

use lens_core::LensEngine;
use std::path::Path;

fn write_at(root: &Path, rel: &str, body: &[u8]) {
    let p = root.join(rel);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).expect("create parent");
    }
    std::fs::write(&p, body).expect("write seed");
}

/// Everything a data dir can hold, including an entry no enumeration knows about.
/// `location.json` / `location.json.pending` are anchor pointer files: before any
/// relocation the anchor *is* the data dir, so they are legitimately present here
/// and must NOT be carried into the new dir.
fn seed_data_dir(root: &Path) {
    write_at(root, "sources/a.pdf", b"pdf");
    write_at(root, "sources/nested/b.txt", b"nested");
    write_at(root, "notebooks/nb1/overview.wav", b"wav");
    write_at(root, "models/orpheus/model.gguf", b"gguf");
    write_at(root, "models/fastembed/models--x--y/blob", b"blob");
    write_at(root, "hf-cache/models--a--b/snapshot", b"snap");
    write_at(root, "config.json", br#"{"schema_version":1}"#);
    write_at(root, "future-subsystem/keep.bin", b"unknown");
    write_at(root, "location.json", br#"{"data_dir":"/old","cleanup":null}"#);
    write_at(root, "location.json.pending", b"{}");
}

const COPIED: &[&str] = &[
    "sources/a.pdf",
    "sources/nested/b.txt",
    "notebooks/nb1/overview.wav",
    "models/orpheus/model.gguf",
    "models/fastembed/models--x--y/blob",
    "hf-cache/models--a--b/snapshot",
    "config.json",
    "future-subsystem/keep.bin",
];

#[tokio::test]
async fn relocate_copies_every_data_dir_entry_including_unknown_ones() {
    let from = tempfile::tempdir().expect("from");
    let to_parent = tempfile::tempdir().expect("to_parent");
    let to = to_parent.path().join("moved");

    let engine = LensEngine::init(from.path()).await.expect("init engine");
    let pool = engine.pool().await;
    seed_data_dir(from.path());

    lens_core::relocate::relocate_data_dir(&pool, from.path(), &to, &[])
        .await
        .expect("relocate");

    for rel in COPIED {
        let dest = to.join(rel);
        assert!(
            dest.exists(),
            "{rel} must be copied into the new data dir (allow-list regression?)"
        );
        assert_eq!(
            std::fs::read(&dest).expect("read copied"),
            std::fs::read(from.path().join(rel)).expect("read source"),
            "{rel} copied with different content"
        );
    }

    assert!(
        to.join("lens.db").exists(),
        "lens.db arrives via VACUUM INTO, not the bulk copy"
    );

    for rel in ["location.json", "location.json.pending"] {
        assert!(
            !to.join(rel).exists(),
            "{rel} is an anchor pointer and must never be copied into the new dir"
        );
    }
}

/// The `extra_skip` passed by the desktop layer (Qwen sidecar runtime dirs) is
/// re-provisioned at the destination rather than copied — the one case where an
/// entry present in the source is legitimately absent in the target.
#[tokio::test]
async fn relocate_honours_caller_supplied_skips() {
    let from = tempfile::tempdir().expect("from");
    let to_parent = tempfile::tempdir().expect("to_parent");
    let to = to_parent.path().join("moved");

    let engine = LensEngine::init(from.path()).await.expect("init engine");
    let pool = engine.pool().await;
    seed_data_dir(from.path());
    write_at(from.path(), "qwen-venv/bin/python", b"venv");
    write_at(from.path(), "uv-cache/x", b"cache");

    lens_core::relocate::relocate_data_dir(&pool, from.path(), &to, &["qwen-venv", "uv-cache"])
        .await
        .expect("relocate");

    assert!(!to.join("qwen-venv").exists(), "skipped dir must not copy");
    assert!(!to.join("uv-cache").exists(), "skipped dir must not copy");
    assert!(
        to.join("future-subsystem/keep.bin").exists(),
        "an explicit skip must not suppress unrelated entries"
    );
}
