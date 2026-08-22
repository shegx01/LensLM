//! #248 AC-1.12: the startup sweep that reclaims physical `vec__` tables no
//! `embedding_index` row names.
//!
//! Every test re-inits an engine on the same dir, because the sweep runs only in
//! `LensEngine::init` — `for_test()` runs no GC at all and would pass vacuously.

mod common;

use common::pin_catalog_offline;
use lens_core::LensEngine;
use std::path::Path;
use std::sync::Arc;

fn lance_root(data_dir: &Path) -> String {
    data_dir.join("lancedb").to_string_lossy().into_owned()
}

async fn connect(data_dir: &Path) -> lancedb::Connection {
    lancedb::connect(&lance_root(data_dir))
        .execute()
        .await
        .expect("lancedb connect")
}

/// Creates a physical table with no `embedding_index` row — what a crash between
/// `create_empty_table` and the registry insert leaves behind.
async fn make_unregistered_table(data_dir: &Path, name: &str) {
    let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "id",
        arrow_schema::DataType::Utf8,
        false,
    )]));
    connect(data_dir)
        .await
        .create_empty_table(name, schema)
        .execute()
        .await
        .expect("create physical table");
}

async fn table_names(data_dir: &Path) -> Vec<String> {
    connect(data_dir)
        .await
        .table_names()
        .execute()
        .await
        .expect("table_names")
}

/// Inits once on a fresh dir and returns a persisted notebook id.
async fn new_notebook(data_dir: &Path) -> String {
    let engine = LensEngine::init(data_dir).await.expect("init engine");
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    engine.pool().await.close().await;
    nb
}

async fn init_and_drop(data_dir: &Path) {
    let engine = LensEngine::init(data_dir).await.expect("init engine");
    engine.pool().await.close().await;
}

/// (a) an unregistered `vec__` table is reclaimed at startup.
#[tokio::test]
async fn unregistered_vec_table_is_reclaimed() {
    let dir = tempfile::tempdir().expect("tempdir");
    pin_catalog_offline(dir.path());
    init_and_drop(dir.path()).await;

    make_unregistered_table(dir.path(), "vec__nb1__fastembed__nomic_v15__d768__3").await;
    assert!(
        table_names(dir.path())
            .await
            .iter()
            .any(|t| t.starts_with("vec__")),
        "fixture must exist before the sweep, or this test is vacuous"
    );

    init_and_drop(dir.path()).await;

    assert!(
        !table_names(dir.path())
            .await
            .iter()
            .any(|t| t.starts_with("vec__")),
        "an unregistered vec table must be reclaimed by the startup sweep"
    );
}

/// (b) a legacy-format orphan is reclaimed too — the sweep matches the `vec__`
/// prefix, not today's `table_name` formula, which pre-4b-B names do not satisfy.
#[tokio::test]
async fn legacy_format_orphan_is_reclaimed() {
    let dir = tempfile::tempdir().expect("tempdir");
    pin_catalog_offline(dir.path());
    init_and_drop(dir.path()).await;

    let legacy = "vec__nb1__nomic_v15__d768";
    make_unregistered_table(dir.path(), legacy).await;
    assert!(table_names(dir.path()).await.iter().any(|t| t == legacy));

    init_and_drop(dir.path()).await;

    assert!(
        !table_names(dir.path()).await.iter().any(|t| t == legacy),
        "a pre-4b-B name must still be reclaimed (prefix match, not format match)"
    );
}

/// (c) NON-OPTIONAL. `ent__` tables are coordinate-derived and never registered, so
/// a sweep defined by subtraction rather than a positive `vec__` filter would
/// delete the entire entity graph on every startup, with no automatic recovery.
#[tokio::test]
async fn entity_tables_survive_the_sweep() {
    let dir = tempfile::tempdir().expect("tempdir");
    pin_catalog_offline(dir.path());
    // A REAL notebook: `gc_orphan_entity_tables` reclaims ent__ tables whose
    // notebook row is gone, which would delete the fixture for the wrong reason.
    let nb = new_notebook(dir.path()).await;

    let ent = format!("ent__{nb}__fastembed__nomic_v15__d768");
    make_unregistered_table(dir.path(), &ent).await;
    assert!(table_names(dir.path()).await.contains(&ent));

    init_and_drop(dir.path()).await;

    assert!(
        table_names(dir.path()).await.contains(&ent),
        "the sweep must never touch ent__ tables; they carry no registry by design"
    );
}

/// (d) a torn table is reclaimed rather than erroring the sweep. The whole GC is
/// best-effort, so a sweep that aborted here would leave the orphan AND the
/// generation collision that bricks re-embed for that notebook.
#[tokio::test]
async fn torn_vec_table_is_reclaimed_without_erroring_the_sweep() {
    let dir = tempfile::tempdir().expect("tempdir");
    pin_catalog_offline(dir.path());
    init_and_drop(dir.path()).await;

    let torn = "vec__nb1__fastembed__nomic_v15__d768__7";
    make_unregistered_table(dir.path(), torn).await;

    // Delete the manifest but keep the `.lance` directory — the shape a crash
    // mid-write leaves. `table_names()` is a directory listing, so it still sees it.
    let versions = Path::new(&lance_root(dir.path()))
        .join(format!("{torn}.lance"))
        .join("_versions");
    for entry in std::fs::read_dir(&versions).expect("read _versions") {
        let path = entry.expect("entry").path();
        if path.is_file() {
            std::fs::remove_file(&path).expect("remove manifest");
        }
    }
    assert!(
        table_names(dir.path()).await.iter().any(|t| t == torn),
        "a manifest-less .lance dir must still be listed, or (d) is vacuous"
    );

    init_and_drop(dir.path()).await;

    assert!(
        !table_names(dir.path()).await.iter().any(|t| t == torn),
        "a torn vec table must be reclaimed, not skipped"
    );
}

/// The union in `create_building_table`: with an unregistered orphan still on disk
/// and NO restart, a re-embed must pick a different generation instead of erroring.
/// This is the half the startup sweep cannot fix — it makes the brick recoverable
/// at next launch, not impossible within the session.
#[tokio::test]
async fn building_table_skips_a_live_unregistered_orphan() {
    use lens_core::vector_store::{Coordinate, LanceVectorStore, VectorStore};

    let dir = tempfile::tempdir().expect("tempdir");
    pin_catalog_offline(dir.path());
    let engine = LensEngine::init(dir.path()).await.expect("init engine");
    let pool = engine.pool().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();

    let coord = Coordinate::new(
        nb.clone(),
        lens_core::EmbeddingBackend::Fastembed,
        "nomic-embed-text-v1.5".to_string(),
        768,
    );
    // Occupies exactly the name the registry-only search would return first.
    let squatter = format!("vec__{nb}__fastembed__nomic_v15__d768__1");
    make_unregistered_table(dir.path(), &squatter).await;

    let store = LanceVectorStore::new(dir.path(), pool.clone());
    let picked = VectorStore::create_building_table(&store, &coord)
        .await
        .expect("create_building_table must skip the orphan, not error on it");
    assert_ne!(
        picked, squatter,
        "the generation search must not hand back an occupied physical name"
    );
    pool.close().await;
}

/// The `active` table survives; `building`/`stale` are reclaimed — but by the
/// PRE-EXISTING registry-driven pass, which drops those tables and deletes their
/// rows before the `vec__` sweep runs. The sweep therefore only ever sees a
/// leftover the earlier pass could not name, which is the whole point of it.
#[tokio::test]
async fn active_survives_and_building_stale_are_reclaimed_by_the_earlier_pass() {
    let dir = tempfile::tempdir().expect("tempdir");
    pin_catalog_offline(dir.path());
    let engine = LensEngine::init(dir.path()).await.expect("init engine");
    let pool = engine.pool().await;

    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();

    let mut names = Vec::new();
    for (status, generation) in [("active", 1), ("building", 2), ("stale", 3)] {
        let name = format!("vec__{nb}__fastembed__nomic_v15__d768__{generation}");
        make_unregistered_table(dir.path(), &name).await;
        sqlx::query(
            "INSERT INTO embedding_index \
             (id, notebook_id, model, dim, prefix_convention, lance_table_name, status, backend, created_at) \
             VALUES (?, ?, 'nomic-embed-text-v1.5', 768, 'nomic', ?, ?, 'fastembed', ?)",
        )
        .bind(uuid::Uuid::now_v7().to_string())
        .bind(&nb)
        .bind(&name)
        .bind(status)
        .bind(chrono::Utc::now().to_rfc3339())
        .execute(&pool)
        .await
        .expect("register");
        names.push(name);
    }
    pool.close().await;
    drop(engine);

    init_and_drop(dir.path()).await;

    let live = table_names(dir.path()).await;
    assert!(
        live.iter().any(|t| t == &names[0]),
        "the active registered table must survive: {}",
        names[0]
    );
    for name in &names[1..] {
        assert!(
            !live.iter().any(|t| t == name),
            "{name} is building/stale and is reclaimed by the registry-driven pass"
        );
    }
}

/// Floor: an empty `embedding_index` beside notebooks that exist means the DB lost
/// its index, not that every vector table is garbage. Leaking bytes is recoverable;
/// deleting every vector is not.
#[tokio::test]
async fn empty_registry_with_live_notebooks_refuses_to_sweep() {
    let dir = tempfile::tempdir().expect("tempdir");
    pin_catalog_offline(dir.path());
    let nb = new_notebook(dir.path()).await;

    let orphan = format!("vec__{nb}__fastembed__nomic_v15__d768__1");
    make_unregistered_table(dir.path(), &orphan).await;

    init_and_drop(dir.path()).await;

    assert!(
        table_names(dir.path()).await.contains(&orphan),
        "with notebooks present and a wiped registry, the sweep must refuse rather \
         than delete every vec table"
    );
}
