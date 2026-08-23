//! AC-4.1/4.2: the relocation copy is verified beyond a `sources` row count, without
//! the tightening refusing moves that were previously fine.
//!
//! Both ACs are bound narrowly on purpose. Checking every Lance table would abort a
//! valid multi-minute move over a mid-write `building` table; checking every `ready`
//! audio row would refuse every user who has one failed overview on disk.

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
