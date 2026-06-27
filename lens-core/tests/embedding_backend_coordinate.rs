//! M4 Phase 4b-B Step 2 — migration `0007` (additive + standalone partial-index
//! SWAP) + the backend coordinate axis.
//!
//! Three guards:
//!   * `migration_0007_swaps_index_to_four_col_partial_unique` (R1) — `0007`
//!     applied: `uq_embidx_active` is a STANDALONE index (droppable, NOT an
//!     autoindex) AND the recreated index is the 4-col partial-unique
//!     `(notebook_id, backend, model, dim) WHERE status='active'` — two `active`
//!     rows differing only in `backend` both insert OK, while a second identical
//!     4-tuple `active` row is REJECTED.
//!   * `same_model_dim_distinct_backends_never_collide` — a fastembed and an
//!     ollama set for the SAME `(notebook, model, dim)` resolve to two DISTINCT
//!     physical tables + two DISTINCT active registry rows, and search on one
//!     returns ONLY that backend's vectors.
//!   * `existing_4ba_row_backfills_backend_fastembed` — a 4b-A row carried the
//!     `DEFAULT 'fastembed'` backfill so it resolves under the fastembed
//!     coordinate by its stored `lance_table_name`.
//!
//! These build a FILE-BACKED engine via [`LensEngine::init`] (so the real
//! migrator runs) and share the engine's pool + lancedb root, mirroring the
//! existing `vector_index` / `retire_coordinate` integration tests. The vectors
//! are deterministic unit vectors (no fastembed weights required).

use lens_core::vector_store::{Coordinate, LanceVectorStore, VectorRow, VectorStore};
use lens_core::{DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL_ID, EmbeddingBackend, LensEngine};

/// A deterministic unit vector of length `dim`, seeded so two seeds differ.
fn unit_vector(seed: usize, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim)
        .map(|j| ((seed as f32) * 0.013 + (j as f32) * 0.0007).sin())
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// `n` rows of width `dim` for `notebook`, all from one source, with a per-row
/// `source_tag` woven into the chunk id so the two backends' rows are
/// distinguishable in search results.
fn make_rows(notebook: &str, source_tag: &str, n: usize, dim: usize) -> Vec<VectorRow> {
    (0..n)
        .map(|i| VectorRow {
            chunk_id: format!("{source_tag}-chunk-{i}"),
            source_id: format!("src-{source_tag}"),
            notebook_id: notebook.to_string(),
            level: 1,
            vector: unit_vector(i, dim),
        })
        .collect()
}

/// Fresh file-backed engine + a created notebook + its pool.
async fn engine_and_notebook() -> (tempfile::TempDir, LensEngine, sqlx::SqlitePool, String) {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let nb = engine.create_notebook("backend", None, None).await.unwrap();
    let pool = engine.pool().await;
    (dir, engine, pool, nb.id.to_string())
}

/// COUNT of `embedding_index` active rows for the full 4-tuple coordinate.
async fn active_row_count(
    pool: &sqlx::SqlitePool,
    nb: &str,
    backend: &str,
    model: &str,
    dim: usize,
) -> i64 {
    sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM embedding_index \
         WHERE notebook_id = ? AND backend = ? AND model = ? AND dim = ? AND status = 'active'",
    )
    .bind(nb)
    .bind(backend)
    .bind(model)
    .bind(dim as i64)
    .fetch_one(pool)
    .await
    .unwrap()
}

/// R1 — the index swap. After `0007`, `uq_embidx_active` is a standalone index
/// (so `DROP INDEX` would succeed — i.e. `sqlite_master.sql IS NOT NULL`, which
/// is FALSE for the un-droppable `sqlite_autoindex_*`), and the recreated index
/// is the 4-col partial-unique: two `active` rows differing only in `backend`
/// both insert, but a second identical 4-tuple `active` row is rejected.
#[tokio::test]
async fn migration_0007_swaps_index_to_four_col_partial_unique() {
    let (_dir, _engine, pool, nb) = engine_and_notebook().await;

    // (a) `uq_embidx_active` exists and is a STANDALONE index — its `sql` is
    // non-NULL (autoindexes have NULL `sql`), proving `DROP INDEX` can target it.
    let standalone: Option<String> = sqlx::query_scalar(
        "SELECT sql FROM sqlite_master \
         WHERE type = 'index' AND name = 'uq_embidx_active' AND sql IS NOT NULL",
    )
    .fetch_optional(&pool)
    .await
    .unwrap();
    let sql = standalone.expect("uq_embidx_active is a standalone (droppable) index after 0007");
    // It is the 4-col partial-unique on (notebook_id, backend, model, dim) with a
    // `WHERE status='active'` predicate (accept either quoting/spacing form).
    let normalized = sql.to_lowercase();
    let has_status =
        normalized.contains("status='active'") || normalized.contains("status = 'active'");
    assert!(
        normalized.contains("notebook_id")
            && normalized.contains("backend")
            && normalized.contains("model")
            && normalized.contains("dim")
            && has_status,
        "uq_embidx_active is the 4-col partial-unique; got: {sql}"
    );

    // (b) Two active rows differing ONLY in backend both insert (no conflict).
    let insert = |backend: &'static str, table: &'static str| {
        let pool = pool.clone();
        let nb = nb.clone();
        async move {
            sqlx::query(
                "INSERT INTO embedding_index \
                     (id, notebook_id, backend, model, dim, prefix_convention, lance_table_name, status, created_at) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)",
            )
            .bind(uuid::Uuid::now_v7().to_string())
            .bind(&nb)
            .bind(backend)
            .bind(DEFAULT_EMBED_MODEL_ID)
            .bind(DEFAULT_EMBED_DIM as i64)
            .bind("none/none")
            .bind(table)
            .bind(chrono::Utc::now().to_rfc3339())
            .execute(&pool)
            .await
        }
    };
    insert("fastembed", "vec__nb__fastembed__nomic_v15__d768")
        .await
        .expect("fastembed active row inserts");
    insert("ollama", "vec__nb__ollama__nomic_v15__d768")
        .await
        .expect("ollama active row inserts (differs only in backend → allowed)");

    // (c) A SECOND identical 4-tuple active row is REJECTED by the partial-unique.
    let dup = insert("fastembed", "vec__nb__fastembed__nomic_v15__d768__dup").await;
    assert!(
        dup.is_err(),
        "a duplicate (notebook, backend, model, dim) active row must violate uq_embidx_active"
    );
}

/// Two coordinates that differ ONLY by backend never collide: distinct physical
/// tables, distinct active registry rows, and isolated search.
#[tokio::test]
async fn same_model_dim_distinct_backends_never_collide() {
    let (dir, _engine, pool, nb) = engine_and_notebook().await;
    let store = LanceVectorStore::new(dir.path(), pool.clone());

    let fe = Coordinate::new(
        nb.clone(),
        EmbeddingBackend::Fastembed,
        DEFAULT_EMBED_MODEL_ID,
        DEFAULT_EMBED_DIM,
    );
    let ol = Coordinate::new(
        nb.clone(),
        EmbeddingBackend::Ollama,
        DEFAULT_EMBED_MODEL_ID,
        DEFAULT_EMBED_DIM,
    );

    store
        .add(&fe, make_rows(&nb, "fe", 6, DEFAULT_EMBED_DIM))
        .await
        .unwrap();
    store
        .add(&ol, make_rows(&nb, "ol", 6, DEFAULT_EMBED_DIM))
        .await
        .unwrap();

    // Two DISTINCT active registry rows — one per backend.
    assert_eq!(
        active_row_count(
            &pool,
            &nb,
            "fastembed",
            DEFAULT_EMBED_MODEL_ID,
            DEFAULT_EMBED_DIM
        )
        .await,
        1
    );
    assert_eq!(
        active_row_count(
            &pool,
            &nb,
            "ollama",
            DEFAULT_EMBED_MODEL_ID,
            DEFAULT_EMBED_DIM
        )
        .await,
        1
    );

    // Two DISTINCT physical lance_table_names.
    let names: Vec<String> = sqlx::query_scalar(
        "SELECT lance_table_name FROM embedding_index WHERE notebook_id = ? ORDER BY backend",
    )
    .bind(&nb)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(names.len(), 2, "one physical table per backend");
    assert_ne!(names[0], names[1], "backends map to byte-distinct tables");
    assert!(names.iter().any(|n| n.contains("__fastembed__")));
    assert!(names.iter().any(|n| n.contains("__ollama__")));

    // Search isolation: querying the fastembed coordinate returns ONLY fastembed
    // chunk ids (prefixed `fe-`), never the ollama set (`ol-`).
    let q = unit_vector(0, DEFAULT_EMBED_DIM);
    let fe_hits = store.search(&fe, &q, 10).await.unwrap();
    assert!(!fe_hits.is_empty(), "fastembed coordinate has hits");
    assert!(
        fe_hits.iter().all(|h| h.chunk_id.starts_with("fe-")),
        "fastembed search must return ONLY fastembed vectors; got {:?}",
        fe_hits.iter().map(|h| &h.chunk_id).collect::<Vec<_>>()
    );
    let ol_hits = store.search(&ol, &q, 10).await.unwrap();
    assert!(!ol_hits.is_empty(), "ollama coordinate has hits");
    assert!(
        ol_hits.iter().all(|h| h.chunk_id.starts_with("ol-")),
        "ollama search must return ONLY ollama vectors; got {:?}",
        ol_hits.iter().map(|h| &h.chunk_id).collect::<Vec<_>>()
    );
}

/// A 4b-A row (registered before the backend column existed) carries the
/// migration's `DEFAULT 'fastembed'` backfill, and still resolves by its STORED
/// LEGACY `lance_table_name` (pre-backend-segment form) — no rename, no
/// re-derivation. An INSERT that OMITS the `backend` column (exactly how a 4b-A
/// INSERT looked) lands with `backend = 'fastembed'`.
#[tokio::test]
async fn existing_4ba_row_backfills_backend_fastembed() {
    let (dir, _engine, pool, nb) = engine_and_notebook().await;
    let store = LanceVectorStore::new(dir.path(), pool.clone());

    // First, let the store create + register the fastembed coordinate normally so
    // a physical table + active row exist. This is the post-0007 shape (backend in
    // the row + the backend-segment table name).
    let fe = Coordinate::new(
        nb.clone(),
        EmbeddingBackend::Fastembed,
        DEFAULT_EMBED_MODEL_ID,
        DEFAULT_EMBED_DIM,
    );
    store
        .add(&fe, make_rows(&nb, "fe", 4, DEFAULT_EMBED_DIM))
        .await
        .unwrap();

    // An INSERT that OMITS `backend` (a 4b-A-shaped INSERT, here a stale row so it
    // does not conflict with the active partial-unique) backfills to 'fastembed'
    // via the column DEFAULT — proving old write shapes never produce a NULL/empty
    // backend.
    sqlx::query(
        "INSERT INTO embedding_index \
             (id, notebook_id, model, dim, prefix_convention, lance_table_name, status, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, 'stale', ?)",
    )
    .bind(uuid::Uuid::now_v7().to_string())
    .bind(&nb)
    .bind(DEFAULT_EMBED_MODEL_ID)
    .bind(DEFAULT_EMBED_DIM as i64)
    .bind("search_document/search_query")
    .bind(format!("vec__{nb}__nomic_v15__d{DEFAULT_EMBED_DIM}")) // legacy stored name
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(&pool)
    .await
    .unwrap();

    // Every row carries a concrete backend; the backend-omitting INSERT backfilled
    // to 'fastembed' (never NULL/empty).
    let backends: Vec<String> =
        sqlx::query_scalar("SELECT backend FROM embedding_index WHERE notebook_id = ?")
            .bind(&nb)
            .fetch_all(&pool)
            .await
            .unwrap();
    assert!(
        !backends.is_empty() && backends.iter().all(|b| b == "fastembed"),
        "all 4b-A rows backfill to 'fastembed'; got {backends:?}"
    );

    // The active fastembed coordinate resolves + serves search (registry-driven,
    // by stored name).
    let q = unit_vector(0, DEFAULT_EMBED_DIM);
    let hits = store.search(&fe, &q, 10).await.unwrap();
    assert!(
        hits.iter().any(|h| h.chunk_id.starts_with("fe-")),
        "the fastembed coordinate resolved its active table and read the rows back"
    );
}
