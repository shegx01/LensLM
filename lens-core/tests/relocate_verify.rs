//! AC-4.1/4.2: verify the copy beyond a row count, without refusing moves that were
//! fine. Both are bound narrowly — all Lance tables would abort on a mid-write
//! `building` one; all `ready` audio would refuse anyone with a failed overview.

mod common;

use common::{capture_logs, pin_catalog_offline};
use lens_core::LensEngine;
use std::path::Path;
use std::sync::Arc;

async fn engine(dir: &Path) -> LensEngine {
    pin_catalog_offline(dir);
    LensEngine::init(dir).await.expect("init engine")
}

fn lance_root(data_dir: &Path) -> String {
    data_dir.join("lancedb").to_string_lossy().into_owned()
}

async fn make_table(data_dir: &Path, name: &str) {
    let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "id",
        arrow_schema::DataType::Utf8,
        false,
    )]));
    lancedb::connect(&lance_root(data_dir))
        .execute()
        .await
        .expect("connect")
        .create_empty_table(name, schema)
        .execute()
        .await
        .expect("create table");
}

/// Removes the manifest, keeping the `.lance` directory — the shape a mid-write
/// copy captures, and what makes a table refuse to open.
fn tear_table(data_dir: &Path, name: &str) {
    let versions = Path::new(&lance_root(data_dir))
        .join(format!("{name}.lance"))
        .join("_versions");
    for entry in std::fs::read_dir(&versions).expect("read _versions") {
        let p = entry.expect("entry").path();
        if p.is_file() {
            std::fs::remove_file(&p).expect("remove manifest");
        }
    }
}

async fn register(pool: &sqlx::SqlitePool, nb: &str, name: &str, status: &str) {
    sqlx::query(
        "INSERT INTO embedding_index \
         (id, notebook_id, model, dim, prefix_convention, lance_table_name, status, backend, created_at) \
         VALUES (?, ?, 'nomic-embed-text-v1.5', 768, 'nomic', ?, ?, 'fastembed', ?)",
    )
    .bind(uuid::Uuid::now_v7().to_string())
    .bind(nb)
    .bind(name)
    .bind(status)
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(pool)
    .await
    .expect("register");
}

/// Sets an mtime so the old-vs-active freshness guard is deterministic (no sleeps).
fn stamp(path: &Path, secs: u64) {
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("open for stamp")
        .set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs))
        .expect("set mtime");
}

async fn seed_notebook(engine: &LensEngine) -> String {
    engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string()
}

/// AC-4.1: a torn `building` table must NOT abort the move. It is GC-reclaimed and
/// not load-bearing, and a copy can legitimately capture one mid-write.
#[tokio::test]
async fn a_broken_building_table_does_not_block_relocation() {
    let from = tempfile::tempdir().expect("from");
    let to_parent = tempfile::tempdir().expect("to_parent");
    let to = to_parent.path().join("moved");
    let engine = engine(from.path()).await;
    let pool = engine.pool().await;
    let nb = seed_notebook(&engine).await;

    let name = format!("vec__{nb}__fastembed__nomic_v15__d768__2");
    make_table(from.path(), &name).await;
    register(&pool, &nb, &name, "building").await;
    tear_table(from.path(), &name);

    lens_core::relocate::relocate_data_dir(&pool, from.path(), &to, &[])
        .await
        .expect("a torn building table must not block the move");
    pool.close().await;
}

/// AC-4.1: a torn `active` table MUST abort — that is the live index, and letting
/// the move complete would flip the pointer to a corpus whose search is broken.
#[tokio::test]
async fn a_broken_active_table_blocks_relocation() {
    let from = tempfile::tempdir().expect("from");
    let to_parent = tempfile::tempdir().expect("to_parent");
    let to = to_parent.path().join("moved");
    let engine = engine(from.path()).await;
    let pool = engine.pool().await;
    let nb = seed_notebook(&engine).await;

    let name = format!("vec__{nb}__fastembed__nomic_v15__d768__1");
    make_table(from.path(), &name).await;
    register(&pool, &nb, &name, "active").await;
    tear_table(from.path(), &name);

    let err = lens_core::relocate::relocate_data_dir(&pool, from.path(), &to, &[]).await;
    assert!(err.is_err(), "a broken active table must cancel the move");
    pool.close().await;
}

async fn seed_audio(pool: &sqlx::SqlitePool, nb: &str, path: &Path, status: &str) {
    sqlx::query(
        "INSERT INTO audio_overviews (notebook_id, path, generated_at, status, source_set_hash) \
         VALUES (?, ?, 't', ?, 'h')",
    )
    .bind(nb)
    .bind(path.display().to_string())
    .bind(status)
    .execute(pool)
    .await
    .expect("seed audio");
}

/// AC-4.2: `ready` legitimately outlives its file (`Missing` exists for exactly
/// that), so a row already dangling before the move warns and does not block. A
/// blanket presence check would refuse every user in this state.
#[tokio::test]
async fn a_pre_dangling_ready_row_warns_but_does_not_block() {
    let cap = capture_logs();
    let from = tempfile::tempdir().expect("from");
    let to_parent = tempfile::tempdir().expect("to_parent");
    let to = to_parent.path().join("moved");
    let engine = engine(from.path()).await;
    let pool = engine.pool().await;
    let nb = seed_notebook(&engine).await;

    let missing = from.path().join("notebooks").join(&nb).join("gone.wav");
    seed_audio(&pool, &nb, &missing, "ready").await;

    lens_core::relocate::relocate_data_dir(&pool, from.path(), &to, &[])
        .await
        .expect("a pre-dangling ready row must not block the move");

    assert!(
        cap.any_with(&["audio_pre_dangling"]),
        "the skipped row must be reported, not silently tolerated"
    );
    pool.close().await;
}

/// AC-4.2: a `failed` row's file is expected to be absent and must never be checked.
#[tokio::test]
async fn a_failed_status_row_is_ignored() {
    let from = tempfile::tempdir().expect("from");
    let to_parent = tempfile::tempdir().expect("to_parent");
    let to = to_parent.path().join("moved");
    let engine = engine(from.path()).await;
    let pool = engine.pool().await;
    let nb = seed_notebook(&engine).await;

    let missing = from.path().join("notebooks").join(&nb).join("failed.wav");
    seed_audio(&pool, &nb, &missing, "failed").await;

    lens_core::relocate::relocate_data_dir(&pool, from.path(), &to, &[])
        .await
        .expect("a failed-status row must not block the move");
    pool.close().await;
}

/// AC-4.5: a `ready` file that exists in the source and is missing from the copy is
/// a genuine copy failure and must cancel the move.
#[tokio::test]
async fn a_ready_file_lost_in_the_copy_blocks_relocation() {
    let from = tempfile::tempdir().expect("from");
    let to_parent = tempfile::tempdir().expect("to_parent");
    let to = to_parent.path().join("moved");
    let engine = engine(from.path()).await;
    let pool = engine.pool().await;
    let nb = seed_notebook(&engine).await;

    let wav = from.path().join("notebooks").join(&nb).join("ok.wav");
    std::fs::create_dir_all(wav.parent().expect("parent")).expect("mkdir");
    std::fs::write(&wav, b"audio").expect("seed wav");
    seed_audio(&pool, &nb, &wav, "ready").await;

    // `notebooks` skipped: the source file exists, the copy will not have it.
    let err = lens_core::relocate::relocate_data_dir(&pool, from.path(), &to, &["notebooks"]).await;
    assert!(
        err.is_err(),
        "a ready overview present in the source but absent from the copy must cancel"
    );
    pool.close().await;
}

/// AC-4.6: the deletion gate is the only moment both directories exist, so it
/// re-verifies before deleting the source of truth. On refusal the old dir and the
/// cleanup marker are both kept, and the log names the failing check.
#[tokio::test]
async fn the_deletion_gate_refuses_when_the_active_index_is_unreadable() {
    let cap = capture_logs();
    let anchor = tempfile::tempdir().expect("anchor");
    let old = tempfile::tempdir().expect("old");
    let active = tempfile::tempdir().expect("active");

    let engine = engine(active.path()).await;
    let pool = engine.pool().await;
    let nb = seed_notebook(&engine).await;

    // An active row whose table will not open: exactly what a bad copy leaves.
    let name = format!("vec__{nb}__fastembed__nomic_v15__d768__1");
    make_table(active.path(), &name).await;
    register(&pool, &nb, &name, "active").await;
    tear_table(active.path(), &name);

    // Stamp mtimes so the mtime guard PASSES: otherwise it short-circuits first and
    // the refusal under test is never reached.
    std::fs::write(old.path().join("lens.db"), b"db").expect("seed old db");
    stamp(&old.path().join("lens.db"), 100);
    stamp(&active.path().join("lens.db"), 900);
    std::fs::create_dir_all(old.path().join("sources")).expect("seed old corpus");
    lens_core::relocate::write_location(
        anchor.path(),
        &lens_core::relocate::DataLocation {
            data_dir: active.path().display().to_string(),
            cleanup: Some(old.path().display().to_string()),
        },
    )
    .expect("write pointer");

    lens_core::relocate::run_boot_cleanup(anchor.path(), active.path(), &[], &pool).await;

    assert!(
        old.path().join("sources").exists(),
        "the old dir must survive when the new one does not verify"
    );
    assert!(
        cap.any_with(&["active_vector_table_unreadable"]),
        "the refusal must name which check failed"
    );
    pool.close().await;
}

/// AC-4.10: `config.json` holds a plaintext cloud API key. A refusal keeps the old
/// dir indefinitely — the mtime guard never self-heals — so the secret is erased
/// even then. It is settings, already copied, and not corpus.
#[tokio::test]
async fn a_refused_cleanup_still_erases_the_old_config() {
    let anchor = tempfile::tempdir().expect("anchor");
    let old = tempfile::tempdir().expect("old");
    let active = tempfile::tempdir().expect("active");

    let engine = engine(active.path()).await;
    let pool = engine.pool().await;

    // Old dir newer than the active snapshot → the mtime guard refuses.
    for (dir, secs) in [(old.path(), 900u64), (active.path(), 100u64)] {
        let db = dir.join("lens.db");
        std::fs::write(&db, b"db").expect("seed db");
        stamp(&db, secs);
    }
    let secret = old.path().join("config.json");
    std::fs::write(&secret, br#"{"api_key":"sk-secret"}"#).expect("seed secret");
    std::fs::create_dir_all(old.path().join("sources")).expect("seed corpus");

    lens_core::relocate::write_location(
        anchor.path(),
        &lens_core::relocate::DataLocation {
            data_dir: active.path().display().to_string(),
            cleanup: Some(old.path().display().to_string()),
        },
    )
    .expect("write pointer");

    lens_core::relocate::run_boot_cleanup(anchor.path(), active.path(), &[], &pool).await;

    assert!(
        !secret.exists(),
        "the plaintext key must not outlive a refusal"
    );
    assert!(
        old.path().join("sources").exists(),
        "corpus the refusal is protecting must be untouched"
    );
    assert!(
        old.path().join("lens.db").exists(),
        "the newer old DB the guard protects must be untouched"
    );
    pool.close().await;
}

/// The FIRST relocation always leaves the anchor as the old dir. Refusing there
/// stranded the whole previous corpus in the OS app-data dir AND cleared the marker,
/// so nothing could surface it. Cleaning is safe because `data_entries()` omits the
/// pointer files — the corpus goes, the pointer that finds the new dir stays.
#[tokio::test]
async fn the_first_relocation_cleans_the_anchor_but_keeps_its_pointer() {
    let anchor = tempfile::tempdir().expect("anchor");
    let active = tempfile::tempdir().expect("active");
    let engine = engine(active.path()).await;
    let pool = engine.pool().await;

    // Anchor is the OLD dir: it holds the previous corpus and the pointer.
    std::fs::write(anchor.path().join("lens.db"), b"db").expect("seed old db");
    stamp(&anchor.path().join("lens.db"), 100);
    stamp(&active.path().join("lens.db"), 900);
    std::fs::create_dir_all(anchor.path().join("sources")).expect("seed corpus");
    std::fs::write(anchor.path().join("sources").join("a.txt"), b"corpus").expect("seed file");

    lens_core::relocate::write_location(
        anchor.path(),
        &lens_core::relocate::DataLocation {
            data_dir: active.path().display().to_string(),
            cleanup: Some(anchor.path().display().to_string()),
        },
    )
    .expect("write pointer");

    lens_core::relocate::run_boot_cleanup(anchor.path(), active.path(), &[], &pool).await;

    assert!(
        !anchor.path().join("sources").exists(),
        "the previous corpus must actually be reclaimed"
    );
    assert!(
        anchor.path().join("location.json").exists(),
        "the pointer MUST survive — without it the app cannot find its data"
    );
    assert_eq!(
        lens_core::relocate::resolve_data_dir(anchor.path()),
        active.path(),
        "the anchor must still resolve to the relocated data dir"
    );
    pool.close().await;
}
