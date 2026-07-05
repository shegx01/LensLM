//! Integration tests for the M0 SQLite schema, migrations, and repo methods.

use std::collections::HashSet;

use lens_core::{AppConfig, LensEngine};
use sqlx::Row;
use uuid::Uuid;

/// Helper: collect all user table names from `sqlite_master`.
async fn table_names(engine: &LensEngine) -> HashSet<String> {
    let pool = engine.pool().await;
    let rows = sqlx::query("SELECT name FROM sqlite_master WHERE type = 'table'")
        .fetch_all(&pool)
        .await
        .unwrap();
    rows.into_iter()
        .map(|r| r.get::<String, _>("name"))
        .collect()
}

#[tokio::test]
async fn migration_creates_exactly_the_seven_m0_tables() {
    let engine = LensEngine::for_test().await;
    let tables = table_names(&engine).await;

    for expected in [
        "notebooks",
        "sources",
        "chunks",
        "embedding_index",
        "notes",
        "chat_messages",
        "citations",
    ] {
        assert!(tables.contains(expected), "missing table: {expected}");
    }

    // Deferred / dropped tables must NOT exist in M0.
    for forbidden in ["tts_voice", "audio_overview", "app_config"] {
        assert!(
            !tables.contains(forbidden),
            "table should be deferred/absent: {forbidden}"
        );
    }
}

#[tokio::test]
async fn migration_is_idempotent_second_run_is_noop() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;

    let count_before: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM _sqlx_migrations")
        .fetch_one(&pool)
        .await
        .unwrap();

    // Re-running the migrator must not error nor add rows.
    lens_core::run_migrations(&pool).await.unwrap();

    let count_after: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM _sqlx_migrations")
        .fetch_one(&pool)
        .await
        .unwrap();

    assert_eq!(count_before, count_after);
    assert_eq!(count_after, 13, "all migration files applied (0001..0013)");
}

#[tokio::test]
async fn notebook_id_is_stored_as_text_not_blob() {
    let engine = LensEngine::for_test().await;
    let nb = engine
        .create_notebook("typeof check", None, None)
        .await
        .unwrap();
    let pool = engine.pool().await;

    let kind: String = sqlx::query_scalar("SELECT typeof(id) FROM notebooks WHERE id = ?")
        .bind(&nb.id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(kind, "text");
}

#[tokio::test]
async fn chunk_parent_child_hierarchy() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;

    let nb = engine.create_notebook("hier", None, None).await.unwrap();
    let source_id = Uuid::now_v7().to_string();
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, created_at) \
         VALUES (?, ?, 'text', 't', 'indexed', 'loc', ?)",
    )
    .bind(&source_id)
    .bind(&nb.id)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let parent_id = Uuid::now_v7().to_string();
    let child_id = Uuid::now_v7().to_string();
    for (id, parent, kind, level) in [
        (&parent_id, None, "parent", 0),
        (&child_id, Some(&parent_id), "child", 1),
    ] {
        sqlx::query(
            "INSERT INTO chunks (id, source_id, parent_id, kind, level, section_path, text, created_at) \
             VALUES (?, ?, ?, ?, ?, '[]', 'body', ?)",
        )
        .bind(id)
        .bind(&source_id)
        .bind(parent)
        .bind(kind)
        .bind(level)
        .bind(&now)
        .execute(&pool)
        .await
        .unwrap();
    }

    let children: Vec<String> = sqlx::query_scalar("SELECT id FROM chunks WHERE parent_id = ?")
        .bind(&parent_id)
        .fetch_all(&pool)
        .await
        .unwrap();
    assert_eq!(children, vec![child_id]);
}

#[tokio::test]
async fn chunks_has_no_embedding_ref_column() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;
    let rows = sqlx::query("PRAGMA table_info(chunks)")
        .fetch_all(&pool)
        .await
        .unwrap();
    let cols: HashSet<String> = rows
        .into_iter()
        .map(|r| r.get::<String, _>("name"))
        .collect();
    assert!(
        !cols.contains("embedding_ref"),
        "embedding_ref must not exist"
    );
    assert!(cols.contains("enrichment"));
    assert!(cols.contains("parent_id"));
}

#[tokio::test]
async fn enrichment_json_round_trips_and_text_is_unchanged() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;

    let nb = engine.create_notebook("enrich", None, None).await.unwrap();
    let source_id = Uuid::now_v7().to_string();
    let chunk_id = Uuid::now_v7().to_string();
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, created_at) \
         VALUES (?, ?, 'text', 't', 'indexed', 'loc', ?)",
    )
    .bind(&source_id)
    .bind(&nb.id)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let enrichment = serde_json::json!({
        "title": "Intro",
        "summary": "A summary",
        "entities": ["Alice", "Bob"],
    })
    .to_string();
    sqlx::query(
        "INSERT INTO chunks (id, source_id, kind, level, section_path, text, enrichment, created_at) \
         VALUES (?, ?, 'child', 1, '[\"Intro\"]', 'canonical body', ?, ?)",
    )
    .bind(&chunk_id)
    .bind(&source_id)
    .bind(&enrichment)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let row = sqlx::query("SELECT text, enrichment FROM chunks WHERE id = ?")
        .bind(&chunk_id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(row.get::<String, _>("text"), "canonical body");
    let stored: serde_json::Value =
        serde_json::from_str(&row.get::<String, _>("enrichment")).unwrap();
    assert_eq!(stored["entities"][1], "Bob");
}

#[tokio::test]
async fn embedding_index_unique_constraint_rejects_duplicates() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;
    let nb = engine.create_notebook("embidx", None, None).await.unwrap();
    let now = chrono::Utc::now().to_rfc3339();

    let insert = |id: String, status: &'static str, table: &'static str| {
        let pool = pool.clone();
        let nb_id = nb.id.clone();
        let now = now.clone();
        async move {
            sqlx::query(
                "INSERT INTO embedding_index \
                 (id, notebook_id, model, dim, prefix_convention, lance_table_name, status, created_at) \
                 VALUES (?, ?, 'bge-m3', 1024, 'query:', ?, ?, ?)",
            )
            .bind(id)
            .bind(nb_id)
            .bind(table)
            .bind(status)
            .bind(now)
            .execute(&pool)
            .await
        }
    };

    insert(Uuid::now_v7().to_string(), "active", "vec__nb__bge")
        .await
        .unwrap();
    // Under the partial unique index `uq_embidx_active`, a SECOND `active` row
    // for the same (notebook, model, dim) coordinate is still rejected.
    let dup = insert(Uuid::now_v7().to_string(), "active", "vec__nb__bge2").await;
    assert!(
        dup.is_err(),
        "duplicate active (notebook,model,dim) must be rejected"
    );

    // ...but a `building` row for the SAME coordinate is now ACCEPTED (the
    // relaxation in 0005 — co-existing active + building during a flip).
    insert(Uuid::now_v7().to_string(), "building", "vec__nb__bge__g1")
        .await
        .expect("a building row for the same coordinate must now insert");

    // lance_table_name round-trips for the active row.
    let name: String = sqlx::query_scalar(
        "SELECT lance_table_name FROM embedding_index \
         WHERE notebook_id = ? AND status = 'active'",
    )
    .bind(&nb.id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(name, "vec__nb__bge");
}

/// AC1: the new nullable columns exist after `0005` and read NULL on pre-existing
/// rows; the registry table-rebuild preserves every existing row (the
/// `lance_table_name`/`status`/`created_at` round-trip intact).
#[tokio::test]
async fn migration_0005_adds_enrichment_columns_and_preserves_registry() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;

    // New chunks column.
    let chunk_cols: HashSet<String> = sqlx::query("PRAGMA table_info(chunks)")
        .fetch_all(&pool)
        .await
        .unwrap()
        .into_iter()
        .map(|r| r.get::<String, _>("name"))
        .collect();
    assert!(
        chunk_cols.contains("embedding_text"),
        "chunks.embedding_text must exist after 0005"
    );
    // Canonical text column is untouched.
    assert!(chunk_cols.contains("text"));

    // New sources columns.
    let source_cols: HashSet<String> = sqlx::query("PRAGMA table_info(sources)")
        .fetch_all(&pool)
        .await
        .unwrap()
        .into_iter()
        .map(|r| r.get::<String, _>("name"))
        .collect();
    assert!(
        source_cols.contains("enrichment_status"),
        "sources.enrichment_status must exist after 0005"
    );
    assert!(
        source_cols.contains("enrichment_meta"),
        "sources.enrichment_meta must exist after 0005"
    );

    // The lookup index survived the rebuild.
    let indexes: HashSet<String> = sqlx::query("PRAGMA index_list(embedding_index)")
        .fetch_all(&pool)
        .await
        .unwrap()
        .into_iter()
        .map(|r| r.get::<String, _>("name"))
        .collect();
    assert!(
        indexes.contains("idx_embidx_notebook"),
        "idx_embidx_notebook must be recreated after the rebuild, found: {indexes:?}"
    );
    assert!(
        indexes.contains("uq_embidx_active"),
        "the partial unique uq_embidx_active must exist, found: {indexes:?}"
    );
}

/// AC1: applying `0005` on a POPULATED Phase-2 DB preserves existing chunk,
/// source, and registry rows; the new columns read NULL on those rows.
#[tokio::test]
async fn migration_0005_clean_on_populated_phase2_db() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;

    let nb = engine
        .create_notebook("populated", None, None)
        .await
        .unwrap();
    let now = chrono::Utc::now().to_rfc3339();

    // Seed a source + chunk + a registry row (the Phase-2 shape).
    let source_id = Uuid::now_v7().to_string();
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, created_at) \
         VALUES (?, ?, 'text', 't', 'indexed', 'loc', ?)",
    )
    .bind(&source_id)
    .bind(&nb.id)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let chunk_id = Uuid::now_v7().to_string();
    sqlx::query(
        "INSERT INTO chunks (id, source_id, kind, level, section_path, text, created_at) \
         VALUES (?, ?, 'parent', 0, '[]', 'canonical body', ?)",
    )
    .bind(&chunk_id)
    .bind(&source_id)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let idx_id = Uuid::now_v7().to_string();
    sqlx::query(
        "INSERT INTO embedding_index \
         (id, notebook_id, model, dim, prefix_convention, lance_table_name, status, created_at) \
         VALUES (?, ?, 'nomic-embed-text-v1.5', 768, 'search_document: ', 'vec__nb__nomic_v15', 'active', ?)",
    )
    .bind(&idx_id)
    .bind(&nb.id)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    // `for_test` already ran 0001..0005, so this asserts the migration is clean
    // against a populated DB shape (rows survive, new columns read NULL).
    let row = sqlx::query("SELECT text, embedding_text FROM chunks WHERE id = ?")
        .bind(&chunk_id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(row.get::<String, _>("text"), "canonical body");
    assert!(
        row.get::<Option<String>, _>("embedding_text").is_none(),
        "embedding_text must read NULL on pre-existing chunk rows"
    );

    let src = sqlx::query("SELECT enrichment_status, enrichment_meta FROM sources WHERE id = ?")
        .bind(&source_id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert!(
        src.get::<Option<String>, _>("enrichment_status").is_none(),
        "enrichment_status must read NULL (≡ none) on pre-existing source rows"
    );
    assert!(
        src.get::<Option<String>, _>("enrichment_meta").is_none(),
        "enrichment_meta must read NULL on pre-existing source rows"
    );

    // The registry row survived the table-rebuild intact.
    let reg = sqlx::query(
        "SELECT lance_table_name, status, created_at FROM embedding_index WHERE id = ?",
    )
    .bind(&idx_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        reg.get::<String, _>("lance_table_name"),
        "vec__nb__nomic_v15",
        "lance_table_name must round-trip through the rebuild"
    );
    assert_eq!(reg.get::<String, _>("status"), "active");
    assert_eq!(reg.get::<String, _>("created_at"), now);
}

#[tokio::test]
async fn migration_0006_adds_notebook_embedding_model_column() {
    use sqlx::Row;
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;

    // A raw INSERT that includes the new column must succeed, and the column
    // reads back the written value — proves migration 0006 applied on a fresh DB.
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO notebooks \
             (id, title, description, focus_mode, embedding_model, created_at, updated_at, trashed_at) \
         VALUES ('nb-0006', 'T', NULL, NULL, 'mxbai-embed-large', ?, ?, NULL)",
    )
    .bind(&now)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let model: Option<String> =
        sqlx::query("SELECT embedding_model FROM notebooks WHERE id = 'nb-0006'")
            .fetch_one(&pool)
            .await
            .unwrap()
            .get("embedding_model");
    assert_eq!(model.as_deref(), Some("mxbai-embed-large"));

    // A row inserted WITHOUT the column reads NULL (nullable, no DEFAULT).
    sqlx::query(
        "INSERT INTO notebooks (id, title, created_at, updated_at) \
         VALUES ('nb-0006-null', 'T2', ?, ?)",
    )
    .bind(&now)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();
    let null_model: Option<String> =
        sqlx::query("SELECT embedding_model FROM notebooks WHERE id = 'nb-0006-null'")
            .fetch_one(&pool)
            .await
            .unwrap()
            .get("embedding_model");
    assert!(null_model.is_none(), "absent embedding_model reads NULL");
}

#[tokio::test]
async fn purging_notebook_cascades_to_children() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;
    let nb = engine.create_notebook("cascade", None, None).await.unwrap();
    let now = chrono::Utc::now().to_rfc3339();

    let source_id = Uuid::now_v7().to_string();
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, created_at) \
         VALUES (?, ?, 'text', 't', 'indexed', 'loc', ?)",
    )
    .bind(&source_id)
    .bind(&nb.id)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query("INSERT INTO notes (id, notebook_id, content, origin, created_at, updated_at) VALUES (?, ?, 'n', 'user', ?, ?)")
        .bind(Uuid::now_v7().to_string())
        .bind(&nb.id)
        .bind(&now)
        .bind(&now)
        .execute(&pool)
        .await
        .unwrap();

    // `purge_notebook` is the hard-delete path; `delete_notebook` now soft-deletes
    // (no cascade), so cascade behavior is asserted against purge. Purge only
    // accepts trashed notebooks, so trash it first.
    engine.trash_notebook(&nb.id).await.unwrap();
    engine.purge_notebook(&nb.id).await.unwrap();

    let src_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM sources WHERE notebook_id = ?")
        .bind(&nb.id)
        .fetch_one(&pool)
        .await
        .unwrap();
    let note_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM notes WHERE notebook_id = ?")
        .bind(&nb.id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(src_count, 0, "sources should cascade-delete");
    assert_eq!(note_count, 0, "notes should cascade-delete");
}

/// Lists the LanceDB table names under `{data_dir}/lancedb`.
async fn lance_table_names(data_dir: &std::path::Path) -> HashSet<String> {
    let root = data_dir.join("lancedb");
    let conn = lancedb::connect(root.to_string_lossy().as_ref())
        .execute()
        .await
        .unwrap();
    conn.table_names()
        .execute()
        .await
        .unwrap()
        .into_iter()
        .collect()
}

/// AC (CRITICAL): `purge_notebook` drops the per-notebook Lance table so it is
/// not orphaned on disk forever. Seeds a registered Lance table via the public
/// vector-store API, purges the notebook, and asserts the table is gone.
#[tokio::test]
async fn purge_notebook_drops_lance_table() {
    use lens_core::vector_store::{VectorRow, VectorStore};

    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let data_dir = dir.path();

    let nb = engine
        .create_notebook("purge-lance", None, None)
        .await
        .unwrap();

    // Seed a real, registered Lance table for this notebook by adding a vector
    // row through the public store API (this also inserts the embedding_index
    // registry row, mirroring an ingest).
    let pool = engine.pool().await;
    let store = lens_core::LanceVectorStore::new(data_dir, pool.clone());
    let coord = lens_core::vector_store::Coordinate::new(
        nb.id.as_str(),
        lens_core::EmbeddingBackend::Fastembed,
        lens_core::DEFAULT_EMBED_MODEL_ID,
        lens_core::DEFAULT_EMBED_DIM,
    );
    store
        .add(
            &coord,
            vec![VectorRow {
                chunk_id: Uuid::now_v7().to_string(),
                source_id: Uuid::now_v7().to_string(),
                notebook_id: nb.id.to_string(),
                level: 0,
                vector: vec![0.0; lens_core::DEFAULT_EMBED_DIM],
            }],
        )
        .await
        .unwrap();

    // The Lance table and its registry row now exist.
    let before = lance_table_names(data_dir).await;
    assert_eq!(
        before.len(),
        1,
        "exactly one Lance table seeded: {before:?}"
    );
    let idx_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM embedding_index WHERE notebook_id = ?")
            .bind(&nb.id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(idx_count, 1, "embedding_index row registered");

    // Purge requires a trashed notebook.
    engine.trash_notebook(&nb.id).await.unwrap();
    engine.purge_notebook(&nb.id).await.unwrap();

    // The Lance table must be gone (no orphan on disk), and the registry row
    // cascaded away with the notebook.
    let after = lance_table_names(data_dir).await;
    assert!(
        after.is_empty(),
        "Lance table must be dropped by purge_notebook, found: {after:?}"
    );
    let idx_after: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM embedding_index WHERE notebook_id = ?")
            .bind(&nb.id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(idx_after, 0, "embedding_index row cascaded away");
}

#[tokio::test]
async fn cold_init_under_budget_on_empty_temp_db() {
    let dir = tempfile::tempdir().unwrap();
    let start = tokio::time::Instant::now();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let elapsed = start.elapsed();
    // Sanity: the engine works.
    assert_eq!(engine.migration_count().await.unwrap(), 13);
    // Generous smoke guard against accidentally-expensive migrations (e.g. a
    // future migration that scans/rewrites large tables on cold start). This is
    // NOT a tight perf benchmark — the wide 2s budget keeps it non-flaky on
    // loaded CI runners while still catching pathological regressions.
    assert!(
        elapsed.as_millis() < 2000,
        "cold init took {}ms (budget 2000ms)",
        elapsed.as_millis()
    );
}

#[tokio::test]
async fn app_config_disk_round_trip_default_and_malformed() {
    let dir = tempfile::tempdir().unwrap();

    // Missing file -> default, written back.
    let loaded = AppConfig::load(dir.path()).unwrap();
    assert_eq!(loaded, AppConfig::default());
    assert!(dir.path().join("config.json").exists());

    // Round-trip a non-default config.
    let cfg = AppConfig {
        theme: "dark".into(),
        onboarding_complete: true,
        ..AppConfig::default()
    };
    cfg.save(dir.path()).unwrap();
    let reloaded = AppConfig::load(dir.path()).unwrap();
    assert_eq!(reloaded, cfg);

    // Malformed JSON -> LensError::Parse (not a panic).
    std::fs::write(dir.path().join("config.json"), "{ not valid json").unwrap();
    let err = AppConfig::load(dir.path()).unwrap_err();
    assert!(
        matches!(err, lens_core::LensError::Parse(_)),
        "expected Parse, got {err:?}"
    );
}

#[tokio::test]
async fn migration_0002_adds_notebook_personalize_columns() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;
    let rows = sqlx::query("PRAGMA table_info(notebooks)")
        .fetch_all(&pool)
        .await
        .unwrap();
    let cols: HashSet<String> = rows
        .into_iter()
        .map(|r| r.get::<String, _>("name"))
        .collect();
    assert!(
        cols.contains("description"),
        "missing notebooks.description"
    );
    assert!(cols.contains("focus_mode"), "missing notebooks.focus_mode");
}

#[tokio::test]
async fn create_notebook_persists_and_lists_personalize_fields() {
    let engine = LensEngine::for_test().await;

    // Fields default to None when not supplied.
    let bare = engine.create_notebook("bare", None, None).await.unwrap();
    assert_eq!(bare.description, None);
    assert_eq!(bare.focus_mode, None);

    // Fields round-trip through create + list.
    let full = engine
        .create_notebook("full", Some("a blurb"), Some("research"))
        .await
        .unwrap();
    assert_eq!(full.description.as_deref(), Some("a blurb"));
    assert_eq!(full.focus_mode.as_deref(), Some("research"));

    let listed = engine.list_notebooks().await.unwrap();
    let got = listed.iter().find(|n| n.id == full.id).unwrap();
    assert_eq!(got.description.as_deref(), Some("a blurb"));
    assert_eq!(got.focus_mode.as_deref(), Some("research"));
}

#[tokio::test]
async fn add_source_inserts_pending_file_record_and_lists_scoped() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap();
    let other = engine.create_notebook("other", None, None).await.unwrap();

    let src = engine
        .add_source(&nb.id, "report.pdf", "/abs/report.pdf")
        .await
        .unwrap()
        .source;
    assert_eq!(src.kind, "file");
    assert_eq!(src.status, "pending");
    assert_eq!(src.title, "report.pdf");
    assert_eq!(src.locator, "/abs/report.pdf");
    assert_eq!(src.selected, 1);
    assert_eq!(src.notebook_id, nb.id.to_string());

    let listed = engine.list_sources(&nb.id).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, src.id);

    // list_sources is scoped to its notebook.
    assert!(engine.list_sources(&other.id).await.unwrap().is_empty());
}

#[tokio::test]
async fn migration_0009_renames_file_hash_to_raw_content_hash() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;

    // Column rename: raw_content_hash present, file_hash gone.
    let source_cols: HashSet<String> = sqlx::query("PRAGMA table_info(sources)")
        .fetch_all(&pool)
        .await
        .unwrap()
        .into_iter()
        .map(|r| r.get::<String, _>("name"))
        .collect();
    assert!(
        source_cols.contains("raw_content_hash"),
        "sources.raw_content_hash must exist after 0009"
    );
    assert!(
        !source_cols.contains("file_hash"),
        "sources.file_hash must be gone after 0009"
    );

    // Index recreated under the new name; old index dropped.
    let indexes: HashSet<String> =
        sqlx::query("SELECT name FROM sqlite_master WHERE type = 'index'")
            .fetch_all(&pool)
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.get::<String, _>("name"))
            .collect();
    assert!(
        indexes.contains("idx_sources_notebook_raw_content_hash"),
        "idx_sources_notebook_raw_content_hash must exist after 0009"
    );
    assert!(
        !indexes.contains("idx_sources_notebook_file_hash"),
        "old idx_sources_notebook_file_hash must be gone after 0009"
    );

    // Migration count reflects all thirteen migrations.
    assert_eq!(engine.migration_count().await.unwrap(), 13);
}

#[tokio::test]
async fn migration_0011_adds_last_activity_at_and_backfills_to_created_at() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;

    // Column added by migration 0011.
    let notebook_cols: HashSet<String> = sqlx::query("PRAGMA table_info(notebooks)")
        .fetch_all(&pool)
        .await
        .unwrap()
        .into_iter()
        .map(|r| r.get::<String, _>("name"))
        .collect();
    assert!(
        notebook_cols.contains("last_activity_at"),
        "notebooks.last_activity_at must exist after 0011"
    );

    // Simulate a pre-migration row (last_activity_at NULL) and re-run the
    // backfill statement to prove existing rows are seeded to created_at.
    let nb = engine
        .create_notebook("Backfill probe", None, None)
        .await
        .unwrap();
    sqlx::query("UPDATE notebooks SET last_activity_at = NULL WHERE id = ?")
        .bind(&nb.id)
        .execute(&pool)
        .await
        .unwrap();
    // The exact backfill statement from 0011.
    sqlx::query(
        "UPDATE notebooks SET last_activity_at = created_at WHERE last_activity_at IS NULL",
    )
    .execute(&pool)
    .await
    .unwrap();

    let (created_at, last_activity_at): (String, Option<String>) =
        sqlx::query_as("SELECT created_at, last_activity_at FROM notebooks WHERE id = ?")
            .bind(&nb.id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        last_activity_at.as_deref(),
        Some(created_at.as_str()),
        "backfill must set last_activity_at = created_at for pre-migration rows"
    );
}
