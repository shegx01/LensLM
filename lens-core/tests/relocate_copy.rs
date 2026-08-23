//! Copy completeness for data-dir relocation. `copy_tree` is a deny-list copier:
//! every entry it does not explicitly skip must reach the new dir, including ones
//! no static enumeration names. Converting it to an allow-list is the regression.

mod common;

use common::pin_catalog_offline;
use lens_core::LensEngine;
use std::collections::BTreeSet;
use std::path::Path;

/// Every shape a data dir can hold, including one no enumeration in `relocate.rs`
/// names. `lancedb` matters most: it is in `DATA_ENTRIES`, so boot cleanup deletes
/// the old copy once the pointer flips.
const SEED: &[(&str, &[u8])] = &[
    ("sources/a.pdf", b"pdf"),
    ("sources/nested/b.txt", b"nested"),
    ("notebooks/nb1/overview.wav", b"wav"),
    ("lancedb/vec__nb1__x__d768.lance/data/part-0.lance", b"vec"),
    ("lancedb/ent__nb1__x.lance/data/part-0.lance", b"ent"),
    ("models/orpheus/model.gguf", b"gguf"),
    ("models/fastembed/models--x--y/blob", b"blob"),
    ("hf-cache/models--a--b/snapshot", b"snap"),
    ("prompts/dialogue.md", b"prompt"),
    ("future-subsystem/keep.bin", b"unknown"),
];

/// What production never bulk-copies, so the completeness diff must not expect it.
const NOT_BULK_COPIED: &[&str] = &[
    "lens.db",
    "lens.db-wal",
    "lens.db-shm",
    "location.json",
    "location.json.pending",
];

/// The subset whose absence is observable. `-wal`/`-shm` are unlinked by the target
/// pool's clean close whatever was copied, so asserting those passes for the wrong
/// reason; DB lineage is covered by the committed-row check instead.
const ASSERTED_ABSENT: &[&str] = &["location.json", "location.json.pending"];

fn write_at(root: &Path, rel: &str, body: &[u8]) {
    let p = root.join(rel);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).expect("create parent");
    }
    std::fs::write(&p, body).expect("write seed");
}

fn seed(root: &Path) {
    for (rel, body) in SEED {
        write_at(root, rel, body);
    }
    write_at(
        root,
        "location.json",
        br#"{"data_dir":"/old","cleanup":null}"#,
    );
    write_at(root, "location.json.pending", b"{}");
}

/// Relative paths of every file under `root`, minus in-flight scratch files that
/// concurrent writers may leave (they are not part of any copy contract).
fn walk(root: &Path) -> BTreeSet<String> {
    fn rec(dir: &Path, base: &Path, out: &mut BTreeSet<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            let Ok(meta) = e.metadata() else { continue };
            if meta.is_dir() {
                rec(&p, base, out);
            } else if let Ok(rel) = p.strip_prefix(base) {
                let rel = rel.to_string_lossy().replace('\\', "/");
                let transient = rel.ends_with(".tmp")
                    || rel.ends_with(".part")
                    || rel.ends_with("-journal")
                    || rel.contains("/.");
                if !transient {
                    out.insert(rel);
                }
            }
        }
    }
    let mut out = BTreeSet::new();
    rec(root, root, &mut out);
    out
}

/// Everything in `from` that the copy contract says must land in `to`.
fn expected_copies(from: &Path, extra_skip: &[&str]) -> BTreeSet<String> {
    walk(from)
        .into_iter()
        .filter(|rel| !NOT_BULK_COPIED.contains(&rel.as_str()))
        .filter(|rel| {
            let top = rel.split('/').next().unwrap_or(rel);
            !extra_skip.contains(&top)
        })
        .collect()
}

fn assert_copy_complete(from: &Path, to: &Path, extra_skip: &[&str]) {
    let expected = expected_copies(from, extra_skip);
    assert!(
        expected.len() >= SEED.len(),
        "fixture did not survive to the assertion: only {} candidates",
        expected.len()
    );
    let actual = walk(to);
    let missing: Vec<_> = expected.difference(&actual).cloned().collect();
    assert!(
        missing.is_empty(),
        "entries present in the old data dir but absent from the new one: {missing:?}"
    );
    for rel in &expected {
        assert_eq!(
            std::fs::read(to.join(rel)).expect("read copied"),
            std::fs::read(from.join(rel)).expect("read source"),
            "{rel} copied with different content"
        );
    }
    for rel in ASSERTED_ABSENT {
        assert!(
            from.join(rel).exists(),
            "{rel} absent from the source — the negative assertion below would be vacuous"
        );
        assert!(
            !to.join(rel).exists(),
            "{rel} must never be bulk-copied into the new data dir"
        );
    }
}

#[tokio::test]
async fn relocate_copies_every_data_dir_entry_including_unknown_ones() {
    let from = tempfile::tempdir().expect("from");
    let to_parent = tempfile::tempdir().expect("to_parent");
    let to = to_parent.path().join("moved");

    // `init` must precede seeding: it writes the real config.json, and a stub here
    // would fail AppConfig deserialization inside init.
    pin_catalog_offline(from.path());
    let engine = LensEngine::init(from.path()).await.expect("init engine");
    let pool = engine.pool().await;
    seed(from.path());

    sqlx::query(
        "INSERT INTO notebooks (id, title, created_at, updated_at) VALUES ('nb-vac','V','t','t')",
    )
    .execute(&pool)
    .await
    .expect("seed row");

    lens_core::relocate::relocate_data_dir(&pool, from.path(), &to, &[])
        .await
        .expect("relocate");

    assert_copy_complete(from.path(), &to, &[]);

    // Committed-but-uncheckpointed data must survive: written through the live pool,
    // this row is in the WAL, not lens.db. A snapshot taken from a stale or
    // pre-insert lens.db would come up empty here.
    let moved = sqlx::SqlitePool::connect(&format!(
        "sqlite://{}?mode=ro",
        to.join("lens.db").display()
    ))
    .await
    .expect("open copied db");
    let seen: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM notebooks WHERE id='nb-vac'")
        .fetch_one(&moved)
        .await
        .expect("query copied db");
    assert_eq!(
        seen, 1,
        "a row committed before the move must reach the new db"
    );
    moved.close().await;
    pool.close().await;
}

/// `config.json` holds a plaintext cloud API key and is written `0600`. Relocation
/// preserves that only because `fs::copy` carries permission bits.
#[cfg(unix)]
#[tokio::test]
async fn relocate_preserves_config_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let from = tempfile::tempdir().expect("from");
    let to_parent = tempfile::tempdir().expect("to_parent");
    let to = to_parent.path().join("moved");

    pin_catalog_offline(from.path());
    let engine = LensEngine::init(from.path()).await.expect("init engine");
    let pool = engine.pool().await;
    seed(from.path());

    std::fs::set_permissions(
        from.path().join("config.json"),
        std::fs::Permissions::from_mode(0o600),
    )
    .expect("chmod source");

    lens_core::relocate::relocate_data_dir(&pool, from.path(), &to, &[])
        .await
        .expect("relocate");

    let mode = std::fs::metadata(to.join("config.json"))
        .expect("stat copied config")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o600, "the copied config must stay owner-only");
    pool.close().await;
}

/// The desktop layer passes the Qwen sidecar runtime dirs, which are re-provisioned
/// at the destination rather than copied.
#[tokio::test]
async fn relocate_honours_caller_supplied_skips() {
    let from = tempfile::tempdir().expect("from");
    let to_parent = tempfile::tempdir().expect("to_parent");
    let to = to_parent.path().join("moved");

    pin_catalog_offline(from.path());
    let engine = LensEngine::init(from.path()).await.expect("init engine");
    let pool = engine.pool().await;
    seed(from.path());
    let skip = ["qwen-venv", "uv-cache", "bin"];
    for dir in skip {
        write_at(from.path(), &format!("{dir}/payload"), b"regenerable");
    }

    lens_core::relocate::relocate_data_dir(&pool, from.path(), &to, &skip)
        .await
        .expect("relocate");

    for dir in skip {
        assert!(!to.join(dir).exists(), "{dir} must not be copied");
    }
    assert_copy_complete(from.path(), &to, &skip);
    pool.close().await;
}

/// AC-4.8: a real move runs against a live data dir (catalog refresh, config saves,
/// SQLite journals), so an entry can vanish between `read_dir` and the copy. That
/// skips the entry rather than aborting the relocation and clearing the target.
#[tokio::test]
async fn an_entry_vanishing_mid_copy_does_not_abort_the_move() {
    let from = tempfile::tempdir().expect("from");
    let to_parent = tempfile::tempdir().expect("to_parent");
    let to = to_parent.path().join("moved");

    pin_catalog_offline(from.path());
    let engine = LensEngine::init(from.path()).await.expect("init engine");
    let pool = engine.pool().await;
    seed(from.path());

    let doomed = from.path().join("vanishing.tmp");
    std::fs::write(&doomed, b"about to disappear").expect("seed doomed");
    *lens_core::relocate::VANISH_DURING_COPY
        .lock()
        .expect("seam lock") = Some(doomed.clone());

    lens_core::relocate::relocate_data_dir(&pool, from.path(), &to, &[])
        .await
        .expect("a vanishing entry must not cancel the move");

    assert!(!doomed.exists(), "control: the seam actually removed it");
    assert!(
        !to.join("vanishing.tmp").exists(),
        "a file gone from the source must not appear in the copy"
    );
    // Everything else still made it.
    assert_copy_complete(from.path(), &to, &[]);
    pool.close().await;
}
