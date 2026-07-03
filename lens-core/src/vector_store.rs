//! Vector store seam (M4 Phase 1, Group c).
//!
//! Defines [`VectorStore`] — the stable contract addressed by logical coordinate
//! `(notebook, backend, model, dim)` only; physical table names are resolved inside
//! [`LanceVectorStore`] (Decision B1). [`LanceVectorStore`] owns the
//! `embedding_index` registry (Decision M1); the ingest pipeline never sees a table
//! name directly. One LanceDB table per coordinate; searches pin cosine distance and
//! add `.only_if("notebook_id = '…'")` for defense-in-depth isolation.
//! Brute-force kNN below [`ANN_INDEX_MIN_ROWS`]; IVF_PQ above it (M4 Phase 4a).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, StringArray,
    types::Float32Type,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures_util::TryStreamExt;
use lancedb::index::Index;
use lancedb::index::vector::IvfPqIndexBuilder;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::{OptimizeAction, OptimizeOptions};
use lancedb::{Connection, DistanceType, Table};
use sqlx::SqlitePool;
use uuid::Uuid;

use crate::LensError;
use crate::embedder::EmbeddingBackend;

/// Default embedding dimension (nomic-embed-text-v1.5 = 768). Retained for tests;
/// prefer [`crate::DEFAULT_EMBED_DIM`] in new code (each coordinate carries its own dim).
pub const VECTOR_DIM: usize = 768;

/// Row count at/above which an IVF_PQ ANN index is built (M4 Phase 4a). Below this
/// brute-force kNN is exact and faster. Tests lower it via
/// `with_ann_index_min_rows`; there is no production runtime override.
const ANN_INDEX_MIN_ROWS: usize = 100_000;

/// `embedding_index.status` for a live, usable index. The only status written
/// in Phase 1 (`building`/`stale` are reserved for the Phase-4 model-switch
/// flow). Single source of truth for the registry status literal.
const REGISTRY_STATUS_ACTIVE: &str = "active";

/// AC7 crash-injection: when `true`, `flip_active` returns after the SQLite flip
/// txn but before the Lance drop, simulating the startup-GC's recovery window.
/// Consumed (reset to `false`) on use.
#[cfg(feature = "test-util")]
pub static CRASH_AFTER_FLIP_TXN_BEFORE_LANCE_DROP: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// R3 crash-injection: when `true`, `retire_coordinate` returns after demoting the
/// old active row to `stale` but before the Lance drop. Consumed on use.
#[cfg(feature = "test-util")]
pub static CRASH_AFTER_RETIRE_STALE_BEFORE_LANCE_DROP: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Issue #71 crash-injection: when `true`, the streaming-ingest loop returns an
/// error after writing at least one EMBED_BATCH but before `flip_active`, leaving
/// an orphan building table for the startup-GC to reclaim. Consumed on use.
/// Lives in its own test binary so this process-global never races parallel ingests.
#[cfg(feature = "test-util")]
pub static CRASH_AFTER_STREAMING_ADD_BEFORE_FLIP: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// The four-axis logical embedding coordinate (M4 Phase 4b-B):
/// `(notebook, backend, model, dim)`. Using a struct rather than four scalars
/// makes transposing or omitting an axis a compile error instead of a silent
/// cross-backend vector-pollution bug (Decision A / ADR).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coordinate {
    /// Owning notebook id.
    pub notebook: String,
    /// The embedding backend (`fastembed` | `ollama`).
    pub backend: EmbeddingBackend,
    /// Canonical registry model id (e.g. `nomic-embed-text-v1.5`).
    pub model: String,
    /// Output vector dimension.
    pub dim: usize,
}

impl Coordinate {
    pub fn new(
        notebook: impl Into<String>,
        backend: EmbeddingBackend,
        model: impl Into<String>,
        dim: usize,
    ) -> Self {
        Self {
            notebook: notebook.into(),
            backend,
            model: model.into(),
            dim,
        }
    }
}

/// A single vector row to insert. `chunk_id` is the SQLite `chunks.id` —
/// the LanceDB→SQLite link; there is no `embedding_ref` column in SQLite.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorRow {
    /// The `chunks.id` this vector embeds.
    pub chunk_id: String,
    /// The owning `sources.id`.
    pub source_id: String,
    /// The owning notebook id (stored on the row for `.only_if` scoping).
    pub notebook_id: String,
    /// Chunk level: `0` = parent, `1` = child.
    pub level: i32,
    /// The embedding vector. MUST match the coordinate's dim.
    pub vector: Vec<f32>,
}

/// A single search hit: a chunk id and its distance from the query vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    /// The matched `chunks.id`.
    pub chunk_id: String,
    /// Cosine distance from the query vector (lower = nearer).
    pub distance: f32,
}

/// The vector-store seam. Implementations resolve `(notebook, model, dim)` to a
/// physical store internally; callers never pass a physical table name.
#[async_trait::async_trait]
pub trait VectorStore: Send + Sync {
    /// Adds `rows` to the logical store for `coord`, creating it (and registering
    /// it) on first use.
    async fn add(&self, coord: &Coordinate, rows: Vec<VectorRow>) -> Result<(), LensError>;

    /// Returns the `k` nearest chunk ids for `query` within `coord`, ordered by
    /// ascending cosine distance. Pins `DistanceType::Cosine` explicitly.
    async fn search(
        &self,
        coord: &Coordinate,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<Hit>, LensError>;

    /// Drops all vectors for `source_id` from the active table. Runs BEFORE the
    /// SQLite `chunks` delete (cross-store ordering guarantee).
    async fn drop_source(&self, coord: &Coordinate, source_id: &str) -> Result<(), LensError>;

    /// Drops the named physical Lance tables; missing tables are a no-op (idempotent).
    async fn drop_tables(&self, table_names: &[String]) -> Result<(), LensError>;

    /// Creates an empty gen-suffixed `building` table and registers a `building`
    /// registry row (M4 Phase 3, Step 5 / AC7). Building tables are invisible to
    /// `search` (which resolves `status='active'` only), so no `ingest_lock` needed.
    async fn create_building_table(&self, coord: &Coordinate) -> Result<String, LensError>;

    /// Appends `rows` to the named physical table (M4 Phase 3, Step 5). Used to
    /// populate a `building` table by explicit name. `dim` must match every row's
    /// vector length. Empty `rows` is a no-op.
    async fn add_to_table(
        &self,
        table_name: &str,
        rows: Vec<VectorRow>,
        dim: usize,
    ) -> Result<(), LensError>;

    /// Like `add_to_table` but skips ANN index maintenance (issue #71). Used by the
    /// streaming PDF populate loop to avoid an O(n²) per-batch rebuild storm; the
    /// caller builds the index once after the full populate via `build_index_on_table`.
    async fn add_to_table_no_index(
        &self,
        table_name: &str,
        rows: Vec<VectorRow>,
        dim: usize,
    ) -> Result<(), LensError>;

    /// Builds the ANN index once over the complete row set after streaming populate
    /// (issue #71). Non-fatal — a failure degrades search to exact kNN. Missing
    /// table is a no-op.
    async fn build_index_on_table(&self, table_name: &str, dim: usize) -> Result<(), LensError>;

    /// Atomically flips `building`→`active` and the current `active`→`stale` in one
    /// SQLite txn (M4 Phase 3, Step 5 / AC7), then drops the stale Lance table.
    /// `uq_embidx_active` guarantees at most one `active` row per coordinate at every
    /// commit boundary. A crash between the txn and the Lance-drop is recovered by GC.
    async fn flip_active(&self, coord: &Coordinate, building_name: &str) -> Result<(), LensError>;

    /// Retires an OLD coordinate after a re-embed flip (M4 Phase 4b, Step 9 / R3):
    /// demotes its `active` row to `stale`, drops the Lance table, deletes the row.
    /// A crash between any step is recovered by GC. No-op if already retired.
    async fn retire_coordinate(&self, coord: &Coordinate) -> Result<(), LensError>;
}

/// Slugifies a model id into a table-safe token: `nomic-embed-text-v1.5` →
/// `nomic_v15`; other ids are lowercased with non-alphanumerics collapsed to `_`.
fn model_slug(model: &str) -> String {
    match model {
        "nomic-embed-text-v1.5" => "nomic_v15".to_string(),
        other => {
            let mut slug = String::with_capacity(other.len());
            let mut prev_us = false;
            for ch in other.chars() {
                if ch.is_ascii_alphanumeric() {
                    slug.push(ch.to_ascii_lowercase());
                    prev_us = false;
                } else if !prev_us {
                    slug.push('_');
                    prev_us = true;
                }
            }
            slug.trim_matches('_').to_string()
        }
    }
}

/// Resolves the physical LanceDB table name for a new coordinate (slug A1, M4
/// Phase 4b-B): `vec__{notebook}__{backend}__{model_slug}__d{dim}`.
///
/// CRITICAL: for existing tables, always use the name stored in the registry —
/// never re-derive it. A pre-4b-B table (`vec__{nb}__nomic_v15__d768`) keeps
/// its stored name; only new coordinates flow through this function.
fn table_name(notebook: &str, backend: EmbeddingBackend, model: &str, dim: usize) -> String {
    format!(
        "vec__{notebook}__{}__{}__d{dim}",
        backend.as_str(),
        model_slug(model)
    )
}

/// Gen-suffixed table name (Decision A). gen-0 == `table_name` (byte-identical).
/// Non-zero appends `__{gen}` after the dim segment (R7: dim before gen).
fn gen_table_name(
    notebook: &str,
    backend: EmbeddingBackend,
    model: &str,
    dim: usize,
    generation: u32,
) -> String {
    if generation == 0 {
        table_name(notebook, backend, model, dim)
    } else {
        format!(
            "{}__{generation}",
            table_name(notebook, backend, model, dim)
        )
    }
}

/// Escapes a string for a LanceDB SQL filter literal by doubling single quotes
/// (`'` → `''`). Production ids are UUIDs (safe); eval harness ids may contain apostrophes.
fn escape_lance_literal(s: &str) -> String {
    s.replace('\'', "''")
}

/// Builds the Arrow schema for a vector table: the fixed five-column row shape
/// with a `FixedSizeList<Float32, dim>` vector column.
fn vector_schema(dim: usize) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("chunk_id", DataType::Utf8, false),
        Field::new("source_id", DataType::Utf8, false),
        Field::new("notebook_id", DataType::Utf8, false),
        Field::new("level", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]))
}

/// Builds a single `RecordBatch` from `rows`, validating each vector's length.
fn rows_to_batch(rows: &[VectorRow], dim: usize) -> Result<RecordBatch, LensError> {
    let chunk_ids = StringArray::from_iter_values(rows.iter().map(|r| r.chunk_id.as_str()));
    let source_ids = StringArray::from_iter_values(rows.iter().map(|r| r.source_id.as_str()));
    let notebook_ids = StringArray::from_iter_values(rows.iter().map(|r| r.notebook_id.as_str()));
    let levels = Int32Array::from_iter_values(rows.iter().map(|r| r.level));

    // FixedSizeListArray of Float32: every row must be exactly `dim` long.
    let mut vector_values: Vec<Option<Vec<Option<f32>>>> = Vec::with_capacity(rows.len());
    for r in rows {
        if r.vector.len() != dim {
            return Err(LensError::Vector(format!(
                "vector length {} != expected dim {dim} for chunk {}",
                r.vector.len(),
                r.chunk_id
            )));
        }
        vector_values.push(Some(r.vector.iter().map(|v| Some(*v)).collect()));
    }
    let vectors =
        FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vector_values, dim as i32);

    RecordBatch::try_new(
        vector_schema(dim),
        vec![
            Arc::new(chunk_ids),
            Arc::new(source_ids),
            Arc::new(notebook_ids),
            Arc::new(levels),
            Arc::new(vectors),
        ],
    )
    .map_err(|e| LensError::Vector(format!("failed to build record batch: {e}")))
}

/// Private registry collaborator (Decision M1). Maps each
/// `(notebook, backend, model, dim)` coordinate to its physical `lance_table_name`
/// with a status lifecycle (`active`; `building`/`stale` during flips).
struct EmbeddingIndexRepo {
    pool: SqlitePool,
}

impl EmbeddingIndexRepo {
    /// Wraps an owned pool handle (cheap clone of the `SqlitePool` Arc).
    fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Registers a coordinate mapping if absent. The upsert conflict target uses
    /// `WHERE status='active'` to match migration 0005's partial-unique index
    /// `uq_embidx_active`, so `building`/`stale` rows freely co-exist with the
    /// live `active` row during a flip.
    async fn register(
        &self,
        coord: &Coordinate,
        prefix: &str,
        table: &str,
        status: &str,
    ) -> Result<(), LensError> {
        let id = Uuid::now_v7().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO embedding_index \
                 (id, notebook_id, backend, model, dim, prefix_convention, lance_table_name, status, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(notebook_id, backend, model, dim) WHERE status='active' DO NOTHING",
        )
        .bind(&id)
        .bind(&coord.notebook)
        .bind(coord.backend.as_str())
        .bind(&coord.model)
        .bind(coord.dim as i64)
        .bind(prefix)
        .bind(table)
        .bind(status)
        .bind(&now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Fetches the registry row for a coordinate, if registered. Unused since
    /// Phase-4b (superseded by `active_lance_table_name`); kept as a complete API.
    #[allow(dead_code)]
    async fn get(&self, coord: &Coordinate) -> Result<Option<EmbeddingIndexRow>, LensError> {
        let row = sqlx::query_as::<_, EmbeddingIndexRow>(
            "SELECT id, notebook_id, model, dim, prefix_convention, lance_table_name, status, created_at \
             FROM embedding_index WHERE notebook_id = ? AND backend = ? AND model = ? AND dim = ?",
        )
        .bind(&coord.notebook)
        .bind(coord.backend.as_str())
        .bind(&coord.model)
        .bind(coord.dim as i64)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row)
    }

    /// Returns every physical `lance_table_name` registered for `notebook`.
    ///
    /// Used by the notebook hard-delete path to discover which Lance tables to
    /// drop before the SQLite delete cascades the registry rows away.
    async fn lance_table_names_for_notebook(
        &self,
        notebook: &str,
    ) -> Result<Vec<String>, LensError> {
        let names = sqlx::query_scalar::<_, String>(
            "SELECT lance_table_name FROM embedding_index WHERE notebook_id = ?",
        )
        .bind(notebook)
        .fetch_all(&self.pool)
        .await?;
        Ok(names)
    }

    /// Updates the `status` of every row for a coordinate. Superseded by
    /// `demote_active_to_stale` for the retire path; kept as a complete registry API.
    #[allow(dead_code)]
    async fn set_status(&self, coord: &Coordinate, status: &str) -> Result<(), LensError> {
        let result = sqlx::query(
            "UPDATE embedding_index SET status = ? \
             WHERE notebook_id = ? AND backend = ? AND model = ? AND dim = ?",
        )
        .bind(status)
        .bind(&coord.notebook)
        .bind(coord.backend.as_str())
        .bind(&coord.model)
        .bind(coord.dim as i64)
        .execute(&self.pool)
        .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!(
                "no embedding_index row for ({}, {}, {}, {})",
                coord.notebook,
                coord.backend.as_str(),
                coord.model,
                coord.dim
            )));
        }
        Ok(())
    }

    /// Returns every `lance_table_name` registered for a coordinate, regardless of
    /// status (used to compute the next free generation for a building table).
    async fn lance_table_names_for_coordinate(
        &self,
        coord: &Coordinate,
    ) -> Result<Vec<String>, LensError> {
        let names = sqlx::query_scalar::<_, String>(
            "SELECT lance_table_name FROM embedding_index \
             WHERE notebook_id = ? AND backend = ? AND model = ? AND dim = ?",
        )
        .bind(&coord.notebook)
        .bind(coord.backend.as_str())
        .bind(&coord.model)
        .bind(coord.dim as i64)
        .fetch_all(&self.pool)
        .await?;
        Ok(names)
    }

    /// Resolves the physical `lance_table_name` of the live `status='active'` row
    /// for a coordinate, or `None` when none is registered (AC8 search read-path).
    async fn active_lance_table_name(
        &self,
        coord: &Coordinate,
    ) -> Result<Option<String>, LensError> {
        let name = sqlx::query_scalar::<_, String>(
            "SELECT lance_table_name FROM embedding_index \
             WHERE notebook_id = ? AND backend = ? AND model = ? AND dim = ? AND status = 'active'",
        )
        .bind(&coord.notebook)
        .bind(coord.backend.as_str())
        .bind(&coord.model)
        .bind(coord.dim as i64)
        .fetch_optional(&self.pool)
        .await?;
        Ok(name)
    }

    /// Returns `lance_table_name`s of every `building` row for a coordinate (issue
    /// #71). Used by the in-process orphan sweep before creating a new building table.
    async fn building_lance_table_names(
        &self,
        coord: &Coordinate,
    ) -> Result<Vec<String>, LensError> {
        let names = sqlx::query_scalar::<_, String>(
            "SELECT lance_table_name FROM embedding_index \
             WHERE notebook_id = ? AND backend = ? AND model = ? AND dim = ? AND status = 'building'",
        )
        .bind(&coord.notebook)
        .bind(coord.backend.as_str())
        .bind(&coord.model)
        .bind(coord.dim as i64)
        .fetch_all(&self.pool)
        .await?;
        Ok(names)
    }

    /// Deletes every `status='building'` registry row for a coordinate (issue #71).
    /// The orphan-sweep counterpart: after the physical building tables are dropped,
    /// their registry rows are removed so a fresh build starts from a clean slate.
    async fn delete_building_rows(&self, coord: &Coordinate) -> Result<(), LensError> {
        sqlx::query(
            "DELETE FROM embedding_index \
             WHERE notebook_id = ? AND backend = ? AND model = ? AND dim = ? AND status = 'building'",
        )
        .bind(&coord.notebook)
        .bind(coord.backend.as_str())
        .bind(&coord.model)
        .bind(coord.dim as i64)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// One SQLite txn: demote `active`→`stale`, promote `building`→`active` (AC7).
    /// `uq_embidx_active` ensures at most one `active` row per coordinate per commit.
    /// Returns the stale physical table name for the caller to drop.
    async fn flip_active_txn(
        &self,
        coord: &Coordinate,
        building_name: &str,
    ) -> Result<Option<String>, LensError> {
        let mut tx = self.pool.begin().await?;

        let stale_name = sqlx::query_scalar::<_, String>(
            "SELECT lance_table_name FROM embedding_index \
             WHERE notebook_id = ? AND backend = ? AND model = ? AND dim = ? AND status = 'active'",
        )
        .bind(&coord.notebook)
        .bind(coord.backend.as_str())
        .bind(&coord.model)
        .bind(coord.dim as i64)
        .fetch_optional(&mut *tx)
        .await?;

        // Demote first so the partial-unique index permits the promote below.
        sqlx::query(
            "UPDATE embedding_index SET status = 'stale' \
             WHERE notebook_id = ? AND backend = ? AND model = ? AND dim = ? AND status = 'active'",
        )
        .bind(&coord.notebook)
        .bind(coord.backend.as_str())
        .bind(&coord.model)
        .bind(coord.dim as i64)
        .execute(&mut *tx)
        .await?;

        let promoted = sqlx::query(
            "UPDATE embedding_index SET status = 'active' \
             WHERE notebook_id = ? AND backend = ? AND model = ? AND dim = ? \
               AND lance_table_name = ? AND status = 'building'",
        )
        .bind(&coord.notebook)
        .bind(coord.backend.as_str())
        .bind(&coord.model)
        .bind(coord.dim as i64)
        .bind(building_name)
        .execute(&mut *tx)
        .await?;
        if promoted.rows_affected() == 0 {
            // Abort so the coordinate is never left with zero active rows.
            tx.rollback().await?;
            return Err(LensError::Validation(format!(
                "no building embedding_index row named {building_name} for \
                 ({}, {}, {}, {}) to promote",
                coord.notebook,
                coord.backend.as_str(),
                coord.model,
                coord.dim
            )));
        }

        tx.commit().await?;
        Ok(stale_name)
    }

    /// Demotes ONLY the `active` row to `stale`, returning its physical table name
    /// (or `None` — idempotent no-op). Distinct from `set_status` which rewrites
    /// every row; retire must never touch a transient `building`/`stale` row (R3).
    async fn demote_active_to_stale(
        &self,
        coord: &Coordinate,
    ) -> Result<Option<String>, LensError> {
        let mut tx = self.pool.begin().await?;
        let active = sqlx::query_scalar::<_, String>(
            "SELECT lance_table_name FROM embedding_index \
             WHERE notebook_id = ? AND backend = ? AND model = ? AND dim = ? AND status = 'active'",
        )
        .bind(&coord.notebook)
        .bind(coord.backend.as_str())
        .bind(&coord.model)
        .bind(coord.dim as i64)
        .fetch_optional(&mut *tx)
        .await?;
        if active.is_some() {
            sqlx::query(
                "UPDATE embedding_index SET status = 'stale' \
                 WHERE notebook_id = ? AND backend = ? AND model = ? AND dim = ? AND status = 'active'",
            )
            .bind(&coord.notebook)
            .bind(coord.backend.as_str())
            .bind(&coord.model)
            .bind(coord.dim as i64)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(active)
    }

    /// Deletes the registry row naming `table` for a coordinate (the stale row
    /// removed after its Lance table is dropped post-flip). Idempotent: a missing
    /// row is not an error (the startup-GC may have already reclaimed it).
    async fn delete_row_by_table(&self, coord: &Coordinate, table: &str) -> Result<(), LensError> {
        sqlx::query(
            "DELETE FROM embedding_index \
             WHERE notebook_id = ? AND backend = ? AND model = ? AND dim = ? AND lance_table_name = ?",
        )
        .bind(&coord.notebook)
        .bind(coord.backend.as_str())
        .bind(&coord.model)
        .bind(coord.dim as i64)
        .bind(table)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

/// A row of the `embedding_index` registry table. Returned by
/// `EmbeddingIndexRepo::get`; fields are unread in Phase 1.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, sqlx::FromRow)]
struct EmbeddingIndexRow {
    id: String,
    notebook_id: String,
    model: String,
    dim: i64,
    prefix_convention: String,
    lance_table_name: String,
    status: String,
    created_at: String,
}

/// Default [`VectorStore`] backed by an embedded LanceDB at `{data_dir}/lancedb`.
/// Owns the [`EmbeddingIndexRepo`] collaborator (Decision M1).
pub struct LanceVectorStore {
    root: PathBuf,
    registry: EmbeddingIndexRepo,
    ann_index_min_rows: usize,
}

impl LanceVectorStore {
    pub fn new(data_dir: &Path, pool: SqlitePool) -> Self {
        Self {
            root: data_dir.join("lancedb"),
            registry: EmbeddingIndexRepo::new(pool),
            ann_index_min_rows: ANN_INDEX_MIN_ROWS,
        }
    }

    /// Overrides the ANN-index row threshold for tests (production default is 100k).
    #[cfg(feature = "test-util")]
    pub fn with_ann_index_min_rows(mut self, min_rows: usize) -> Self {
        self.ann_index_min_rows = min_rows;
        self
    }

    /// Drops every Lance table registered for `notebook` via the owned registry.
    /// Called by `purge_notebook` BEFORE the SQLite delete to prevent orphaned tables.
    pub async fn drop_notebook_tables(&self, notebook: &str) -> Result<(), LensError> {
        let names = self
            .registry
            .lance_table_names_for_notebook(notebook)
            .await?;
        self.drop_tables(&names).await
    }

    /// Opens a LanceDB connection with `read_consistency_interval = 0` so every
    /// table open re-reads the latest committed version. The default (`None`) caches
    /// the version on open and would miss commits made by a prior handle (e.g.
    /// a post-flip search would read a stale version).
    async fn connect(&self) -> Result<Connection, LensError> {
        let root = self.root.to_string_lossy();
        lancedb::connect(&root)
            .read_consistency_interval(std::time::Duration::ZERO)
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb connect failed: {e}")))
    }

    /// Opens a physical table by exact name, or returns `None`. Never creates anything.
    async fn open_table_by_name(&self, name: &str) -> Result<Option<Table>, LensError> {
        let conn = self.connect().await?;
        let existing = conn
            .table_names()
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb table_names failed: {e}")))?;
        if !existing.iter().any(|t| t == name) {
            return Ok(None);
        }
        let table = conn
            .open_table(name)
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb open_table failed: {e}")))?;
        Ok(Some(table))
    }

    /// Resolves the `active` registry row and opens its table (AC8). Returns `None`
    /// when there is no active row or when the table is momentarily absent (flip
    /// TOCTOU window) — search degrades to empty, never errors.
    async fn open_active_table(&self, coord: &Coordinate) -> Result<Option<Table>, LensError> {
        let name = match self.registry.active_lance_table_name(coord).await? {
            Some(n) => n,
            None => return Ok(None),
        };
        self.open_table_by_name(&name).await
    }

    /// Resolves a coordinate to its physical table, creating and registering it on
    /// first use. Private — driven by `add` only; read paths use `open_active_table`
    /// so they never create a table as a side effect.
    async fn ensure_table(&self, coord: &Coordinate) -> Result<Table, LensError> {
        let conn = self.connect().await?;
        let dim = coord.dim;

        // Registry-driven resolution (write-path twin of AC8): after a re-embed flip
        // the active table is gen-suffixed; using the formula would create a gen-0
        // orphan and silently lose the appended vectors.
        if let Some(active_name) = self.registry.active_lance_table_name(coord).await? {
            if let Some(table) = self.open_table_by_name(&active_name).await? {
                return Ok(table);
            }
            // Active registry row present but the physical table is absent. Should
            // not occur (add + flip_active serialize under ingest_lock); recreate
            // under the REGISTERED name. Warn — this signals a registry/Lance drift.
            tracing::warn!(
                notebook = coord.notebook,
                backend = coord.backend.as_str(),
                model = coord.model,
                active_table = active_name,
                "ensure_table: active registry row present but its Lance table was \
                 absent; recreating it (unexpected — investigate registry/Lance drift)"
            );
            return conn
                .create_empty_table(active_name.as_str(), vector_schema(dim))
                .execute()
                .await
                .map_err(|e| LensError::Vector(format!("lancedb create_empty_table failed: {e}")));
        }

        // No active row — first CREATE. gen-0 == formula name. Register is a no-op
        // on conflict, so a concurrent create is self-healing.
        let name = table_name(&coord.notebook, coord.backend, &coord.model, dim);
        let existing = conn
            .table_names()
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb table_names failed: {e}")))?;

        let table = if existing.iter().any(|t| t == &name) {
            conn.open_table(name.as_str())
                .execute()
                .await
                .map_err(|e| LensError::Vector(format!("lancedb open_table failed: {e}")))?
        } else {
            conn.create_empty_table(name.as_str(), vector_schema(dim))
                .execute()
                .await
                .map_err(|e| LensError::Vector(format!("lancedb create_empty_table failed: {e}")))?
        };

        self.registry
            .register(
                coord,
                &crate::embedder::resolve(&coord.model).prefix_convention(),
                &name,
                REGISTRY_STATUS_ACTIVE,
            )
            .await?;

        Ok(table)
    }

    /// In-process orphan sweep (issue #71): drops every lingering `building` table
    /// for a coordinate so a crash-retry loop within one process doesn't accumulate
    /// orphans. Called BEFORE `create_building_table`. Drop tables first (idempotent),
    /// then delete registry rows — mirrors startup-GC ordering.
    pub(crate) async fn sweep_orphan_building_tables(
        &self,
        coord: &Coordinate,
    ) -> Result<(), LensError> {
        let names = self.registry.building_lance_table_names(coord).await?;
        if names.is_empty() {
            return Ok(());
        }
        tracing::warn!(
            notebook = coord.notebook,
            backend = coord.backend.as_str(),
            model = coord.model,
            count = names.len(),
            "streaming ingest: sweeping orphan building table(s) from a prior crashed ingest"
        );
        self.drop_tables(&names).await?;
        self.registry.delete_building_rows(coord).await?;
        Ok(())
    }

    pub(crate) async fn seed_building_from_active(
        &self,
        coord: &Coordinate,
        building_name: &str,
        exclude_source_id: &str,
    ) -> Result<(), LensError> {
        let active = match self.open_active_table(coord).await? {
            Some(t) => t,
            None => return Ok(()),
        };
        let notebook = coord.notebook.as_str();
        let building = self
            .open_table_by_name(building_name)
            .await?
            .ok_or_else(|| {
                LensError::Vector(format!("no building table {building_name} to seed"))
            })?;

        // Stream one batch at a time to avoid materializing the whole table in RAM.
        let stream = active
            .query()
            .only_if(format!(
                "source_id != '{}'",
                escape_lance_literal(exclude_source_id)
            ))
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb seed scan failed: {e}")))?;
        let mut stream = std::pin::pin!(stream);
        let mut seeded_rows = 0usize;
        while let Some(batch) = stream
            .try_next()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb seed scan failed: {e}")))?
        {
            if batch.num_rows() == 0 {
                continue;
            }
            seeded_rows += batch.num_rows();
            building
                .add(batch)
                .execute()
                .await
                .map_err(|e| LensError::Vector(format!("lancedb seed add failed: {e}")))?;
        }
        tracing::debug!(
            notebook,
            building = building_name,
            exclude_source_id,
            seeded_rows,
            "seed_building_from_active: copied other sources into building table"
        );
        Ok(())
    }

    /// Builds or refreshes the IVF_PQ ANN index once row count crosses the threshold
    /// (M4 Phase 4a). Non-fatal: any failure degrades search to exact brute-force kNN.
    /// `dim` MUST match the table's vector width. Reopens a fresh handle at/above
    /// threshold so `list_indices`/`create_index` read committed state rather than
    /// stale pre-create metadata from the just-written handle.
    async fn maybe_build_or_refresh_index(&self, table: &Table, dim: usize) {
        let n = match table.count_rows(None).await {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!(error = %e, "count_rows failed; skipping ANN index maintenance");
                return;
            }
        };
        if n < self.ann_index_min_rows {
            return;
        }

        let table = match self.open_table_by_name(table.name()).await {
            Ok(Some(t)) => t,
            Ok(None) => {
                // The caller just appended, so the table must exist; a miss is an
                // unexpected registry/Lance divergence — skip, non-fatal.
                tracing::warn!(
                    table = table.name(),
                    "ANN index maintenance: table not found on reopen; skipping"
                );
                return;
            }
            Err(e) => {
                tracing::warn!(error = %e, "ANN index maintenance: reopen failed; skipping");
                return;
            }
        };

        let has_vector_index = match table.list_indices().await {
            Ok(indices) => indices
                .iter()
                .any(|i| i.columns.iter().any(|c| c == "vector")),
            Err(e) => {
                tracing::warn!(error = %e, "list_indices failed; skipping ANN index maintenance");
                return;
            }
        };

        if has_vector_index {
            if let Err(e) = table
                .optimize(OptimizeAction::Index(OptimizeOptions::append()))
                .await
            {
                tracing::warn!(
                    error = %e,
                    "optimize(append) failed; newly-appended rows stay on brute-force until the next refresh"
                );
            }
            return;
        }

        // `num_partitions ≈ √rows` (Lance guidance). `as u32` truncation is fine.
        let num_partitions = ((n as f64).sqrt() as u32).max(1);
        // `num_sub_vectors` MUST divide `dim`; for 768 this is 48.
        let num_sub_vectors = if dim.is_multiple_of(16) {
            (dim / 16) as u32
        } else if dim.is_multiple_of(8) {
            (dim / 8) as u32
        } else {
            1
        };

        let index = Index::IvfPq(
            IvfPqIndexBuilder::default()
                // Pin cosine to match the query metric; builder defaults to L2.
                .distance_type(DistanceType::Cosine)
                .num_partitions(num_partitions)
                .num_sub_vectors(num_sub_vectors)
                .num_bits(8),
        );
        match table
            .create_index(&["vector"], index)
            .replace(true)
            .execute()
            .await
        {
            Ok(()) => tracing::info!(
                rows = n,
                num_partitions,
                num_sub_vectors,
                "built IVF_PQ cosine index on the vector column"
            ),
            Err(e) => tracing::warn!(
                error = %e,
                rows = n,
                num_partitions,
                num_sub_vectors,
                "IVF_PQ create_index failed; search stays on brute-force kNN"
            ),
        }
    }
}

#[async_trait::async_trait]
impl VectorStore for LanceVectorStore {
    async fn add(&self, coord: &Coordinate, rows: Vec<VectorRow>) -> Result<(), LensError> {
        if rows.is_empty() {
            return Ok(());
        }
        let dim = coord.dim;
        let table = self.ensure_table(coord).await?;
        let batch = rows_to_batch(&rows, dim)?;
        table
            .add(batch)
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb add failed: {e}")))?;
        self.maybe_build_or_refresh_index(&table, dim).await;
        Ok(())
    }

    async fn search(
        &self,
        coord: &Coordinate,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<Hit>, LensError> {
        let dim = coord.dim;
        let notebook = coord.notebook.as_str();
        if query.len() != dim {
            return Err(LensError::Vector(format!(
                "query vector length {} != expected dim {dim}",
                query.len()
            )));
        }
        // AC8: resolve via the registry's `active` row, not the formula. gen-0 ==
        // formula, so pre-flip notebooks are byte-identical. Never create as a side
        // effect: no active row or TOCTOU-absent table returns empty, not an error.
        let table = match self.open_active_table(coord).await? {
            Some(t) => t,
            None => return Ok(Vec::new()),
        };

        // Pin cosine (never rely on the L2 default); `.only_if` is defense-in-depth
        // notebook isolation on top of the per-notebook table.
        let stream = table
            .query()
            .nearest_to(query)
            .map_err(|e| LensError::Vector(format!("lancedb nearest_to failed: {e}")))?
            .distance_type(DistanceType::Cosine)
            .only_if(format!(
                "notebook_id = '{}'",
                escape_lance_literal(notebook)
            ))
            .limit(k)
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb search execute failed: {e}")))?;

        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb result stream failed: {e}")))?;

        let mut hits = Vec::new();
        for batch in &batches {
            let chunk_ids = batch
                .column_by_name("chunk_id")
                .ok_or_else(|| LensError::Vector("result missing chunk_id column".into()))?
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| LensError::Vector("chunk_id column is not Utf8".into()))?;
            // LanceDB appends the distance as a Float32 `_distance` column.
            let distances = batch
                .column_by_name("_distance")
                .ok_or_else(|| LensError::Vector("result missing _distance column".into()))?
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| LensError::Vector("_distance column is not Float32".into()))?;

            for i in 0..batch.num_rows() {
                hits.push(Hit {
                    chunk_id: chunk_ids.value(i).to_string(),
                    distance: distances.value(i),
                });
            }
        }

        // Sort ascending: collecting across batches can interleave distances.
        hits.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        Ok(hits)
    }

    /// Removes a source's rows from the active embedding table. Building tables
    /// (mid-flip) are safe to ignore: the flip holds `ingest_lock`, serializing
    /// against `purge_source`.
    async fn drop_source(&self, coord: &Coordinate, source_id: &str) -> Result<(), LensError> {
        let table = match self.open_active_table(coord).await? {
            Some(t) => t,
            None => return Ok(()),
        };
        table
            .delete(format!("source_id = '{}'", escape_lance_literal(source_id)).as_str())
            .await
            .map_err(|e| LensError::Vector(format!("lancedb delete failed: {e}")))?;
        Ok(())
    }

    async fn drop_tables(&self, table_names: &[String]) -> Result<(), LensError> {
        if table_names.is_empty() {
            return Ok(());
        }
        let conn = self.connect().await?;
        // Snapshot the live table set and guard each drop: lancedb 0.30 errors on a
        // missing table, but purge must be idempotent (a re-purge is a no-op).
        let existing = conn
            .table_names()
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb table_names failed: {e}")))?;
        for name in table_names {
            if !existing.iter().any(|t| t == name) {
                continue;
            }
            conn.drop_table(name.as_str(), &[])
                .await
                .map_err(|e| LensError::Vector(format!("lancedb drop_table failed: {e}")))?;
        }
        Ok(())
    }

    async fn create_building_table(&self, coord: &Coordinate) -> Result<String, LensError> {
        let dim = coord.dim;
        let existing = self
            .registry
            .lance_table_names_for_coordinate(coord)
            .await?;
        let mut generation = 1u32;
        let building_name = loop {
            let candidate = gen_table_name(
                &coord.notebook,
                coord.backend,
                &coord.model,
                dim,
                generation,
            );
            if !existing.iter().any(|n| n == &candidate) {
                break candidate;
            }
            generation += 1;
        };

        // Create the physical table first: a crash before the registry insert
        // leaves a harmless unregistered orphan; a crash after leaves a `building`
        // row the startup-GC reclaims.
        let conn = self.connect().await?;
        conn.create_empty_table(building_name.as_str(), vector_schema(dim))
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb create_empty_table failed: {e}")))?;
        self.registry
            .register(
                coord,
                &crate::embedder::resolve(&coord.model).prefix_convention(),
                &building_name,
                "building",
            )
            .await?;
        Ok(building_name)
    }

    async fn add_to_table(
        &self,
        table_name: &str,
        rows: Vec<VectorRow>,
        dim: usize,
    ) -> Result<(), LensError> {
        if rows.is_empty() {
            return Ok(());
        }
        let table = self
            .open_table_by_name(table_name)
            .await?
            .ok_or_else(|| LensError::Vector(format!("no table named {table_name} to add to")))?;
        let batch = rows_to_batch(&rows, dim)?;
        table
            .add(batch)
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb add failed: {e}")))?;
        // Same ANN maintenance as `add`: a building table indexed before the flip
        // serves ANN immediately on becoming active.
        self.maybe_build_or_refresh_index(&table, dim).await;
        Ok(())
    }

    async fn add_to_table_no_index(
        &self,
        table_name: &str,
        rows: Vec<VectorRow>,
        dim: usize,
    ) -> Result<(), LensError> {
        if rows.is_empty() {
            return Ok(());
        }
        let table = self
            .open_table_by_name(table_name)
            .await?
            .ok_or_else(|| LensError::Vector(format!("no table named {table_name} to add to")))?;
        let batch = rows_to_batch(&rows, dim)?;
        table
            .add(batch)
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb add failed: {e}")))?;
        // DELIBERATELY no `maybe_build_or_refresh_index` here (issue #71): the
        // streaming PDF populate loop calls this per EMBED_BATCH and builds the
        // index ONCE after the full populate via `build_index_on_table`, avoiding
        // the per-batch IVF_PQ rebuild storm once the table crosses the threshold.
        Ok(())
    }

    async fn build_index_on_table(&self, table_name: &str, dim: usize) -> Result<(), LensError> {
        // A missing table is a no-op (empty source produced no rows).
        if let Some(table) = self.open_table_by_name(table_name).await? {
            self.maybe_build_or_refresh_index(&table, dim).await;
        }
        Ok(())
    }

    async fn flip_active(&self, coord: &Coordinate, building_name: &str) -> Result<(), LensError> {
        let stale_name = self.registry.flip_active_txn(coord, building_name).await?;

        // CRASH WINDOW (AC7): the stale Lance drop + registry-row delete happen after
        // the flip txn commits. A crash here leaves a stale row + orphan table for GC.
        #[cfg(feature = "test-util")]
        if CRASH_AFTER_FLIP_TXN_BEFORE_LANCE_DROP.swap(false, std::sync::atomic::Ordering::SeqCst) {
            return Ok(());
        }

        // Drop table first so a crash between the two leaves only a dangling stale row.
        if let Some(stale) = stale_name {
            self.drop_tables(std::slice::from_ref(&stale)).await?;
            self.registry.delete_row_by_table(coord, &stale).await?;
        }
        Ok(())
    }

    async fn retire_coordinate(&self, coord: &Coordinate) -> Result<(), LensError> {
        let Some(stale) = self.registry.demote_active_to_stale(coord).await? else {
            return Ok(());
        };

        // CRASH WINDOW (R3): stale drop + row delete happen after the demote commits.
        // A crash here leaves a stale row + orphan table for GC; the new coordinate's
        // active row (already flipped) keeps serving search.
        #[cfg(feature = "test-util")]
        if CRASH_AFTER_RETIRE_STALE_BEFORE_LANCE_DROP
            .swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            return Ok(());
        }

        // Drop table first so a crash between the two leaves only a dangling stale row.
        self.drop_tables(std::slice::from_ref(&stale)).await?;
        self.registry.delete_row_by_table(coord, &stale).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_slug_maps_nomic() {
        assert_eq!(model_slug("nomic-embed-text-v1.5"), "nomic_v15");
    }

    #[test]
    fn model_slug_sanitizes_unknown() {
        assert_eq!(model_slug("Some/Weird Model@2"), "some_weird_model_2");
    }

    #[test]
    fn escape_lance_literal_doubles_quotes() {
        assert_eq!(escape_lance_literal("plain"), "plain");
        assert_eq!(escape_lance_literal("o'brien"), "o''brien");
        assert_eq!(escape_lance_literal("a'b'c"), "a''b''c");
    }

    #[test]
    fn table_name_includes_backend_and_dim_segments() {
        assert_eq!(
            table_name(
                "nb1",
                EmbeddingBackend::Fastembed,
                "nomic-embed-text-v1.5",
                768
            ),
            "vec__nb1__fastembed__nomic_v15__d768"
        );
        assert_eq!(
            table_name("nb1", EmbeddingBackend::Ollama, "all-minilm", 384),
            "vec__nb1__ollama__all_minilm__d384"
        );
    }

    #[test]
    fn same_model_dim_distinct_backends_never_collide_in_slug() {
        let fe = table_name(
            "nb1",
            EmbeddingBackend::Fastembed,
            "nomic-embed-text-v1.5",
            768,
        );
        let ol = table_name(
            "nb1",
            EmbeddingBackend::Ollama,
            "nomic-embed-text-v1.5",
            768,
        );
        assert_ne!(fe, ol);
        assert_eq!(fe, "vec__nb1__fastembed__nomic_v15__d768");
        assert_eq!(ol, "vec__nb1__ollama__nomic_v15__d768");
        assert_ne!(
            gen_table_name(
                "nb1",
                EmbeddingBackend::Fastembed,
                "nomic-embed-text-v1.5",
                768,
                1
            ),
            gen_table_name(
                "nb1",
                EmbeddingBackend::Ollama,
                "nomic-embed-text-v1.5",
                768,
                1
            )
        );
    }

    #[test]
    fn gen_table_name_dim_before_gen_ordering() {
        assert_eq!(
            gen_table_name(
                "nb1",
                EmbeddingBackend::Fastembed,
                "mxbai-embed-large",
                1024,
                0
            ),
            "vec__nb1__fastembed__mxbai_embed_large__d1024"
        );
        assert_eq!(
            gen_table_name(
                "nb1",
                EmbeddingBackend::Fastembed,
                "mxbai-embed-large",
                1024,
                2
            ),
            "vec__nb1__fastembed__mxbai_embed_large__d1024__2"
        );
    }

    #[test]
    fn different_dims_never_collide_for_same_notebook() {
        let n384 = table_name("nb1", EmbeddingBackend::Fastembed, "all-minilm", 384);
        let n768 = table_name(
            "nb1",
            EmbeddingBackend::Fastembed,
            "nomic-embed-text-v1.5",
            768,
        );
        assert_ne!(n384, n768);
        assert_ne!(
            gen_table_name("nb1", EmbeddingBackend::Fastembed, "all-minilm", 384, 1),
            gen_table_name(
                "nb1",
                EmbeddingBackend::Fastembed,
                "nomic-embed-text-v1.5",
                768,
                1
            )
        );
    }

    #[test]
    fn vector_schema_shape() {
        let schema = vector_schema(VECTOR_DIM);
        assert_eq!(schema.fields().len(), 5);
        assert_eq!(schema.field(0).name(), "chunk_id");
        assert_eq!(schema.field(0).data_type(), &DataType::Utf8);
        assert_eq!(schema.field(3).name(), "level");
        assert_eq!(schema.field(3).data_type(), &DataType::Int32);
        match schema.field(4).data_type() {
            DataType::FixedSizeList(item, len) => {
                assert_eq!(item.data_type(), &DataType::Float32);
                assert_eq!(*len, VECTOR_DIM as i32);
            }
            other => panic!("vector field is not FixedSizeList: {other:?}"),
        }
    }

    #[test]
    fn rows_to_batch_rejects_wrong_dim() {
        let rows = vec![VectorRow {
            chunk_id: "c1".into(),
            source_id: "s1".into(),
            notebook_id: "n1".into(),
            level: 1,
            vector: vec![0.0; 3],
        }];
        assert!(matches!(
            rows_to_batch(&rows, VECTOR_DIM),
            Err(LensError::Vector(_))
        ));
    }

    #[test]
    fn rows_to_batch_builds_expected_shape() {
        let rows = vec![
            VectorRow {
                chunk_id: "c1".into(),
                source_id: "s1".into(),
                notebook_id: "n1".into(),
                level: 0,
                vector: vec![0.1; 4],
            },
            VectorRow {
                chunk_id: "c2".into(),
                source_id: "s1".into(),
                notebook_id: "n1".into(),
                level: 1,
                vector: vec![0.2; 4],
            },
        ];
        let batch = rows_to_batch(&rows, 4).expect("batch builds");
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 5);
    }

    #[test]
    fn rows_to_batch_builds_correct_384_dim_schema() {
        let rows = vec![VectorRow {
            chunk_id: "c1".into(),
            source_id: "s1".into(),
            notebook_id: "n1".into(),
            level: 0,
            vector: vec![0.1; 384],
        }];
        let batch = rows_to_batch(&rows, 384).expect("384-dim batch builds");
        assert_eq!(batch.num_rows(), 1);
        match batch.schema().field(4).data_type() {
            DataType::FixedSizeList(item, len) => {
                assert_eq!(item.data_type(), &DataType::Float32);
                assert_eq!(*len, 384);
            }
            other => panic!("vector field is not a 384-wide FixedSizeList: {other:?}"),
        }
    }
}
