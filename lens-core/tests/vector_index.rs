//! IVF_PQ ANN index lifecycle (M4 Phase 4a).
//!
//! Exercises [`LanceVectorStore`]'s automatic index maintenance through the
//! public [`VectorStore`] API plus the test-only `with_ann_index_min_rows` seam
//! (so the 100k-row threshold is reached at a few hundred rows instead — building
//! a real 100k-vector table per test would be far too slow). Indices are
//! inspected by connecting to the physical LanceDB table directly: `list_indices`
//! is a public LanceDB API, so the test never reaches into store internals.
//!
//! ## Why the index must actually build at this scale
//!
//! IVF_PQ with `num_bits = 8` trains a 256-entry PQ codebook per sub-vector, so
//! the corpus must hold at least a few hundred *varied* vectors or k-means
//! collapses. [`unit_vector`] spreads each component over a sine of `(row, dim)`
//! to give genuine variety across all 768 dimensions; the at-threshold tests use
//! 400 rows so the build is deterministic, not flaky.

use std::collections::HashSet;
use std::path::Path;

use lens_core::vector_store::{VectorRow, VectorStore};
use lens_core::{DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL_ID, LanceVectorStore, LensEngine};

/// A deterministic, well-spread unit vector of length [`DEFAULT_EMBED_DIM`]. Each
/// component is a sine of `(seed, dim)` so PQ/IVF k-means training sees real
/// variety across every dimension (a near-constant corpus would collapse it).
fn unit_vector(seed: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..DEFAULT_EMBED_DIM)
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

/// `n` deterministic vector rows for `notebook`, ids `chunk-0..n`.
fn make_rows(notebook: &str, n: usize) -> Vec<VectorRow> {
    (0..n)
        .map(|i| VectorRow {
            chunk_id: format!("chunk-{i}"),
            source_id: "src-1".to_string(),
            notebook_id: notebook.to_string(),
            level: 1,
            vector: unit_vector(i),
        })
        .collect()
}

/// The physical table name the store resolves for the default nomic coordinate.
/// Step 4 generalizes the scheme to embed the dim: `vec__{nb}__{slug}__d{dim}`.
fn nomic_table(notebook: &str) -> String {
    format!("vec__{notebook}__nomic_v15__d{DEFAULT_EMBED_DIM}")
}

/// Connects to the physical Lance table and returns its index configs (public
/// LanceDB API — no coupling to store internals).
async fn vector_indices(data_dir: &Path, table_name: &str) -> Vec<lancedb::index::IndexConfig> {
    let root = data_dir.join("lancedb");
    let conn = lancedb::connect(root.to_string_lossy().as_ref())
        .execute()
        .await
        .unwrap();
    let table = conn.open_table(table_name).execute().await.unwrap();
    table.list_indices().await.unwrap()
}

/// Count of DISTINCT logical indices covering the `vector` column.
///
/// `list_indices` returns one entry per index *delta* (a `create` then an
/// `optimize(append)` leave two same-named segments of ONE logical index), so the
/// idempotency invariant is "one distinct index name", not "one entry".
fn vector_index_count(indices: &[lancedb::index::IndexConfig]) -> usize {
    indices
        .iter()
        .filter(|i| i.columns.iter().any(|c| c == "vector"))
        .map(|i| i.name.clone())
        .collect::<HashSet<_>>()
        .len()
}

/// Returns the single logical vector-index name, or `None` if none exists.
fn vector_index_name(indices: &[lancedb::index::IndexConfig]) -> Option<String> {
    indices
        .iter()
        .find(|i| i.columns.iter().any(|c| c == "vector"))
        .map(|i| i.name.clone())
}

/// `num_unindexed_rows` for the named index on the physical table — `0` once
/// every appended row has been folded into the index by `optimize(append)`.
async fn unindexed_rows(data_dir: &Path, table_name: &str, index_name: &str) -> usize {
    let root = data_dir.join("lancedb");
    let conn = lancedb::connect(root.to_string_lossy().as_ref())
        .execute()
        .await
        .unwrap();
    let table = conn.open_table(table_name).execute().await.unwrap();
    table
        .index_stats(index_name)
        .await
        .unwrap()
        .expect("index stats present")
        .num_unindexed_rows
}

/// Fresh engine + a created notebook + its pool (registry FK satisfied).
async fn engine_and_notebook() -> (tempfile::TempDir, LensEngine, sqlx::SqlitePool, String) {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let nb = engine.create_notebook("idx", None, None).await.unwrap();
    let pool = engine.pool().await;
    (dir, engine, pool, nb.id.to_string())
}

#[tokio::test]
async fn below_threshold_creates_no_index() {
    let (dir, _engine, pool, nb) = engine_and_notebook().await;
    let store = LanceVectorStore::new(dir.path(), pool).with_ann_index_min_rows(256);

    store
        .add(
            &nb,
            DEFAULT_EMBED_MODEL_ID,
            DEFAULT_EMBED_DIM,
            make_rows(&nb, 100),
        )
        .await
        .unwrap();

    let idx = vector_indices(dir.path(), &nomic_table(&nb)).await;
    assert_eq!(
        vector_index_count(&idx),
        0,
        "no ANN index below threshold, found: {idx:?}"
    );
}

#[tokio::test]
async fn crossing_threshold_builds_one_ivfpq_index() {
    let (dir, _engine, pool, nb) = engine_and_notebook().await;
    let store = LanceVectorStore::new(dir.path(), pool).with_ann_index_min_rows(256);

    store
        .add(
            &nb,
            DEFAULT_EMBED_MODEL_ID,
            DEFAULT_EMBED_DIM,
            make_rows(&nb, 400),
        )
        .await
        .unwrap();

    let idx = vector_indices(dir.path(), &nomic_table(&nb)).await;
    assert_eq!(
        vector_index_count(&idx),
        1,
        "exactly one vector index past threshold, found: {idx:?}"
    );
    let vector_idx = idx
        .iter()
        .find(|i| i.columns.iter().any(|c| c == "vector"))
        .expect("a vector index");
    assert_eq!(vector_idx.index_type, lancedb::index::IndexType::IvfPq);
    assert_eq!(vector_idx.columns, vec!["vector".to_string()]);
}

#[tokio::test]
async fn second_add_past_threshold_does_not_duplicate_index() {
    let (dir, _engine, pool, nb) = engine_and_notebook().await;
    let store = LanceVectorStore::new(dir.path(), pool).with_ann_index_min_rows(256);

    store
        .add(
            &nb,
            DEFAULT_EMBED_MODEL_ID,
            DEFAULT_EMBED_DIM,
            make_rows(&nb, 400),
        )
        .await
        .unwrap();

    // A second append above threshold must refresh (optimize) the existing index,
    // never create a duplicate.
    let more: Vec<VectorRow> = (400..460)
        .map(|i| VectorRow {
            chunk_id: format!("chunk-{i}"),
            source_id: "src-2".to_string(),
            notebook_id: nb.clone(),
            level: 1,
            vector: unit_vector(i),
        })
        .collect();
    store
        .add(&nb, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM, more)
        .await
        .unwrap();

    let table = nomic_table(&nb);
    let idx = vector_indices(dir.path(), &table).await;
    assert_eq!(
        vector_index_count(&idx),
        1,
        "still exactly ONE logical vector index after second add (no duplicate): {idx:?}"
    );

    // The second add must have REFRESHED the index (optimize(append)), not skipped
    // maintenance: every appended row is folded in, so nothing is left unindexed.
    let name = vector_index_name(&idx).expect("a vector index");
    assert_eq!(
        unindexed_rows(dir.path(), &table, &name).await,
        0,
        "optimize(append) folded the newly-appended rows into the index"
    );

    // Rows stay searchable (the index serves without error).
    let hits = store
        .search(
            &nb,
            DEFAULT_EMBED_MODEL_ID,
            DEFAULT_EMBED_DIM,
            &unit_vector(450),
            5,
        )
        .await
        .unwrap();
    assert!(!hits.is_empty(), "indexed table still returns hits");
}

/// Recall canary (the deterministic stand-in the plan/reviewers asked for).
///
/// The `eval.rs` corpus (~12 chunks) is far too small to TRAIN an IVF_PQ index
/// (an 8-bit PQ codebook needs ≥256 training vectors), so it can never measure ANN
/// recall — it would always fall back to brute force. This test instead builds a
/// genuinely-trained IVF_PQ index over 500 vectors and measures recall@10 against
/// the EXACT ground truth: every planted query equals a stored vector, so its true
/// nearest neighbour is itself. Querying the exact vector routes the query to the
/// SAME IVF partition the row landed in, so a correct index returns each planted
/// id in the top-k. We assert recall@10 == 1.00 over 25 distinct planted points —
/// a hard floor that fails loudly if the index metric, partitioning, or query path
/// regresses.
#[tokio::test]
async fn recall_canary_ann_returns_exact_matches_in_top_k() {
    let (dir, _engine, pool, nb) = engine_and_notebook().await;
    let store = LanceVectorStore::new(dir.path(), pool).with_ann_index_min_rows(256);

    // 500 background vectors + 25 uniquely-identifiable planted points.
    let mut rows = make_rows(&nb, 500);
    let planted_ids: Vec<String> = (0..25).map(|i| format!("planted-{i}")).collect();
    let planted_vecs: Vec<Vec<f32>> = (0..25).map(|i| unit_vector(900_000 + i)).collect();
    for (id, v) in planted_ids.iter().zip(&planted_vecs) {
        rows.push(VectorRow {
            chunk_id: id.clone(),
            source_id: "planted".to_string(),
            notebook_id: nb.clone(),
            level: 1,
            vector: v.clone(),
        });
    }
    store
        .add(&nb, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM, rows)
        .await
        .unwrap();

    // Guard: this measures the ANN path only if the index actually built.
    assert_eq!(
        vector_index_count(&vector_indices(dir.path(), &nomic_table(&nb)).await),
        1,
        "index must be built for the recall canary to exercise the ANN path"
    );

    let mut found = 0usize;
    for (id, qvec) in planted_ids.iter().zip(&planted_vecs) {
        let hits = store
            .search(&nb, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM, qvec, 10)
            .await
            .unwrap();
        if hits.iter().any(|h| &h.chunk_id == id) {
            found += 1;
        }
    }
    let recall = found as f32 / planted_ids.len() as f32;
    assert!(
        (recall - 1.0).abs() < f32::EPSILON,
        "ANN recall@10 over exact-match queries must be 1.00, got {recall:.4} ({found}/{})",
        planted_ids.len()
    );
}

#[tokio::test]
async fn building_table_indexed_then_serves_after_flip() {
    let (dir, _engine, pool, nb) = engine_and_notebook().await;
    let store = LanceVectorStore::new(dir.path(), pool).with_ann_index_min_rows(256);

    // Populate a building table past threshold through the public flip APIs.
    let building = store
        .create_building_table(&nb, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM)
        .await
        .unwrap();
    let mut rows = make_rows(&nb, 400);
    let planted = unit_vector(888_888);
    rows.push(VectorRow {
        chunk_id: "planted".to_string(),
        source_id: "src-1".to_string(),
        notebook_id: nb.clone(),
        level: 1,
        vector: planted.clone(),
    });
    store
        .add_to_table(&building, rows, DEFAULT_EMBED_DIM)
        .await
        .unwrap();

    // The building table is indexed BEFORE the flip, so it serves ANN the moment
    // it becomes active (the reason 4a lands before the 4b reindex flow).
    assert_eq!(
        vector_index_count(&vector_indices(dir.path(), &building).await),
        1,
        "building table indexed pre-flip"
    );

    store
        .flip_active(&nb, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM, &building)
        .await
        .unwrap();

    let hits = store
        .search(&nb, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM, &planted, 10)
        .await
        .unwrap();
    let ids: HashSet<String> = hits.into_iter().map(|h| h.chunk_id).collect();
    assert!(
        ids.contains("planted"),
        "post-flip ANN search returns the planted hit: {ids:?}"
    );
}

/// A deterministic unit vector of an arbitrary dimension (Step 4: variable-dim
/// coordinates). Mirrors [`unit_vector`] but parameterized on `dim`.
fn unit_vector_dim(seed: usize, dim: usize) -> Vec<f32> {
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

#[tokio::test]
async fn add_to_table_accepts_1024_dim_rows() {
    // Step 4: add_to_table threads the coordinate dim (no hard-coded VECTOR_DIM /
    // debug_assert), so a non-768 coordinate populates without an assertion panic.
    let (dir, _engine, pool, nb) = engine_and_notebook().await;
    let store = LanceVectorStore::new(dir.path(), pool);

    let model = "mxbai-embed-large";
    let dim = 1024usize;
    let building = store.create_building_table(&nb, model, dim).await.unwrap();

    let rows: Vec<VectorRow> = (0..5)
        .map(|i| VectorRow {
            chunk_id: format!("chunk-{i}"),
            source_id: "src-1".to_string(),
            notebook_id: nb.clone(),
            level: 1,
            vector: unit_vector_dim(i, dim),
        })
        .collect();

    store
        .add_to_table(&building, rows, dim)
        .await
        .expect("1024-dim rows append without an assertion failure");

    store.flip_active(&nb, model, dim, &building).await.unwrap();
    let hits = store
        .search(&nb, model, dim, &unit_vector_dim(0, dim), 5)
        .await
        .unwrap();
    assert!(
        !hits.is_empty(),
        "1024-dim coordinate is searchable after flip"
    );
}

#[tokio::test]
async fn existing_gen0_table_resolved_by_stored_name_not_regenerated() {
    // CRITICAL invariant: a coordinate is resolved by its STORED lance_table_name
    // in the registry, never by re-deriving a name. We register a coordinate
    // whose stored physical name is the LEGACY dim-less form (`vec__{nb}__nomic_v15`,
    // what shipped before Step 4) and create exactly that physical table; search
    // must open the stored name, not the new dim-suffixed name the generator now
    // produces (`vec__{nb}__nomic_v15__d768`).
    let (dir, _engine, pool, nb) = engine_and_notebook().await;
    let store = LanceVectorStore::new(dir.path(), pool.clone());

    let legacy_name = format!("vec__{nb}__nomic_v15");

    // Build the legacy physical table directly under the store's lance root.
    {
        use arrow_array::types::Float32Type;
        use arrow_array::{FixedSizeListArray, Int32Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        let conn = lancedb::connect(dir.path().join("lancedb").to_str().unwrap())
            .execute()
            .await
            .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("chunk_id", DataType::Utf8, false),
            Field::new("source_id", DataType::Utf8, false),
            Field::new("notebook_id", DataType::Utf8, false),
            Field::new("level", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DEFAULT_EMBED_DIM as i32,
                ),
                false,
            ),
        ]));
        let v = unit_vector(7);
        let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![Some(v.into_iter().map(Some).collect::<Vec<_>>())],
            DEFAULT_EMBED_DIM as i32,
        );
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["legacy-chunk"])),
                Arc::new(StringArray::from(vec!["src-1"])),
                Arc::new(StringArray::from(vec![nb.as_str()])),
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(vectors),
            ],
        )
        .unwrap();
        // Mirror the store's own pattern: create the empty schema-only table, then
        // append the batch (avoids the create_table Scannable bound on a reader).
        let table = conn
            .create_empty_table(&legacy_name, schema.clone())
            .execute()
            .await
            .unwrap();
        table.add(batch).execute().await.unwrap();
    }

    // Register the legacy physical name as the active coordinate.
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO embedding_index \
         (id, notebook_id, model, dim, prefix_convention, lance_table_name, status, created_at) \
         VALUES (?, ?, 'nomic-embed-text-v1.5', 768, 'search_document/search_query', ?, 'active', ?)",
    )
    .bind(uuid::Uuid::now_v7().to_string())
    .bind(&nb)
    .bind(&legacy_name)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    // Resolution must open the STORED legacy table — the planted hit proves it.
    let hits = store
        .search(
            &nb,
            DEFAULT_EMBED_MODEL_ID,
            DEFAULT_EMBED_DIM,
            &unit_vector(7),
            5,
        )
        .await
        .unwrap();
    let ids: HashSet<String> = hits.into_iter().map(|h| h.chunk_id).collect();
    assert!(
        ids.contains("legacy-chunk"),
        "search resolved the stored legacy table name (not a regenerated one): {ids:?}"
    );
}
