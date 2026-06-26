//! Vector store seam (M4 Phase 1, Group c).
//!
//! Defines the [`VectorStore`] trait — the stable contract the ingest pipeline
//! and (later) the retrieval router address — plus its default LanceDB-backed
//! implementation [`LanceVectorStore`].
//!
//! # Altitude / logical coordinates
//!
//! Callers address vectors **only by logical coordinates** `(notebook, model,
//! dim)` — never by a physical `table: &str`. The physical table name
//! `vec__{notebook}__{model_slug}` is resolved *inside* [`LanceVectorStore`]
//! (see [`LanceVectorStore::ensure_table`]). This keeps the table-per-(notebook,
//! model, dim) layout (Decision B1) fully behind the seam, so a future
//! single-table collapse (Decision B2) would be a zero-caller-change refactor.
//!
//! # Owned registry collaborator (Decision M1)
//!
//! [`LanceVectorStore`] owns the `embedding_index` registry repository as a
//! field collaborator ([`EmbeddingIndexRepo`]). On the first `CREATE` of a
//! logical table it registers the `(notebook, model, dim)` → `lance_table_name`
//! mapping in SQLite. The registry lookup/register is entirely internal — the
//! ingest pipeline never sees a physical table name and never touches the
//! registry directly.
//!
//! # Storage layout (Decision B1)
//!
//! One LanceDB table per `(notebook, model, dim)`. Each row is
//! `{chunk_id: Utf8, source_id: Utf8, notebook_id: Utf8, level: Int32,
//! vector: FixedSizeList<Float32, dim>}` where `dim` is the coordinate's
//! embedding dimension (384/768/1024 per model). Searches pin the distance metric to
//! cosine **explicitly** and additionally run `.only_if("notebook_id = '…'")`
//! as cheap defense-in-depth (the AC asserts notebook isolation directly).
//! Brute-force kNN below [`ANN_INDEX_MIN_ROWS`]; above it an IVF_PQ cosine index is
//! built and refreshed automatically on the write paths (M4 Phase 4a).

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

/// The default embedding dimension (nomic-embed-text-v1.5 = 768). NOT a global
/// constraint since M4 Phase 4b: each coordinate carries its own `dim` (384/768/
/// 1024), threaded through the write/read paths. Retained for the default-model
/// schema in unit tests; prefer [`crate::DEFAULT_EMBED_DIM`] in new code.
pub const VECTOR_DIM: usize = 768;

/// Row count at/above which an active (or building) table gets an IVF_PQ ANN
/// index on its `vector` column (M4 Phase 4a).
///
/// Below this, brute-force kNN is both exact and faster (an ANN index over a few
/// thousand vectors only adds quantization error and build cost); above it, the
/// index keeps search sub-linear as a notebook scales. The threshold is a
/// per-store field defaulting to this constant; tests lower it via
/// [`LanceVectorStore::with_ann_index_min_rows`] to exercise the path at a few
/// hundred rows. There is intentionally NO env-var/runtime override on the
/// production path — the only seam is the test-only setter.
const ANN_INDEX_MIN_ROWS: usize = 100_000;

/// `embedding_index.status` for a live, usable index. The only status written
/// in Phase 1 (`building`/`stale` are reserved for the Phase-4 model-switch
/// flow). Single source of truth for the registry status literal.
const REGISTRY_STATUS_ACTIVE: &str = "active";

/// Test-only crash-injection point for the re-embed flip (AC7).
///
/// When set to `true`, [`VectorStore::flip_active`] commits the SQLite flip txn
/// (so the registry shows `active`→`stale`, `building`→`active`) and then returns
/// EARLY — BEFORE dropping the stale Lance table or deleting its row — simulating
/// a process crash in exactly the crash window the startup-GC must recover. The
/// flag is consumed (reset to `false`) on use so a single test can arm it once.
/// Compiled out of production builds.
#[cfg(feature = "test-util")]
pub static CRASH_AFTER_FLIP_TXN_BEFORE_LANCE_DROP: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// When set to `true`, [`VectorStore::retire_coordinate`] commits the demotion of
/// the OLD coordinate's `active` row to `stale` and then returns EARLY — BEFORE
/// dropping the stale Lance table or deleting its row — simulating a process crash
/// in exactly the crash window the startup-GC must recover (R3). The flag is
/// consumed (reset to `false`) on use so a single test can arm it once. Compiled
/// out of production builds.
#[cfg(feature = "test-util")]
pub static CRASH_AFTER_RETIRE_STALE_BEFORE_LANCE_DROP: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// A single vector row to insert, addressed by its owning logical coordinates.
///
/// `chunk_id` is the SQLite `chunks.id` — the LanceDB→SQLite link is keyed by
/// chunk id on the LanceDB side (there is no `embedding_ref` column in SQLite).
#[derive(Debug, Clone, PartialEq)]
pub struct VectorRow {
    /// The `chunks.id` this vector embeds.
    pub chunk_id: String,
    /// The owning `sources.id` (used by [`VectorStore::drop_source`]).
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
    /// Adds `rows` to the logical store for `(notebook, model, dim)`, creating
    /// it (and registering it) on first use.
    async fn add(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
        rows: Vec<VectorRow>,
    ) -> Result<(), LensError>;

    /// Returns the `k` nearest chunk ids for `query` within `notebook`, ordered
    /// by ascending cosine distance. Pins `DistanceType::Cosine` explicitly.
    async fn search(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<Hit>, LensError>;

    /// Drops every vector belonging to `source_id` (the re-ingest wipe). Per the
    /// cross-store ordering guarantee, this runs BEFORE the SQLite `chunks`
    /// delete so a completed wipe leaves no orphan Lance rows.
    async fn drop_source(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
        source_id: &str,
    ) -> Result<(), LensError>;

    /// Drops the physical Lance tables named in `table_names`, ignoring any that
    /// do not exist (idempotent — a re-purge re-drops nothing).
    ///
    /// Used by the notebook hard-delete path: a purge looks up the notebook's
    /// `embedding_index` rows (which carry the physical `lance_table_name`) and
    /// drops each table BEFORE the SQLite delete cascades those registry rows
    /// away, so the per-notebook Lance tables are never orphaned on disk.
    async fn drop_tables(&self, table_names: &[String]) -> Result<(), LensError>;

    /// Creates an empty, gen-suffixed `building` table for the re-embed flip and
    /// registers a `status='building'` registry row pointing at it (M4 Phase 3,
    /// Step 5 / AC7).
    ///
    /// Picks the next generation above every name already registered for the
    /// coordinate (gen-0 is the live `active` table), so the new physical table
    /// co-exists beside the active one. Returns the physical building table name —
    /// the caller threads it into [`add_to_table`](VectorStore::add_to_table) and
    /// [`flip_active`](VectorStore::flip_active). The building table is invisible
    /// to [`search`](VectorStore::search) (which resolves `status='active'` only),
    /// so populating it needs NO `ingest_lock`.
    async fn create_building_table(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
    ) -> Result<String, LensError>;

    /// Appends `rows` to the physical table named `table_name` (M4 Phase 3,
    /// Step 5). Used to populate a `building` table created by
    /// [`create_building_table`](VectorStore::create_building_table) by its
    /// EXPLICIT name (the active read path resolves names from the registry, so a
    /// half-populated building table is unobservable). Errors if the table does
    /// not exist. An empty `rows` is a no-op. `dim` is the coordinate's vector
    /// dimension (used to build the Arrow batch / index — must match every row).
    async fn add_to_table(
        &self,
        table_name: &str,
        rows: Vec<VectorRow>,
        dim: usize,
    ) -> Result<(), LensError>;

    /// Atomically flips the `building` table to `active` for a coordinate, then
    /// drops the now-`stale` Lance table (M4 Phase 3, Step 5 / AC7).
    ///
    /// ONE SQLite transaction performs the swap: the current `active` row →
    /// `stale`, and the `building` row whose `lance_table_name == building_name` →
    /// `active`. The partial-unique `uq_embidx_active` guarantees at most one
    /// `active` row per coordinate at every commit boundary, so search never sees
    /// mixed raw/enriched or an empty index. AFTER the txn commits, the stale
    /// physical Lance table is dropped and its registry row deleted. A crash
    /// between the commit and the Lance-drop leaves a `stale` row + orphan table
    /// that the startup-GC reclaims (idempotently — a missing table is a no-op).
    async fn flip_active(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
        building_name: &str,
    ) -> Result<(), LensError>;

    /// Retires an OLD-model coordinate after a model/dim-change re-embed has
    /// flipped the NEW coordinate `active` (M4 Phase 4b, Step 9 / R3).
    ///
    /// The partial-unique `uq_embidx_active` constrains `active` rows per
    /// coordinate, and the OLD and NEW coordinates differ in `(model, dim)`, so
    /// both can be `active` simultaneously during the swap; this call removes the
    /// OLD one afterwards. Three idempotent steps — demote the OLD `active` row to
    /// `stale` (committed), drop its physical Lance table, delete its registry row.
    /// A crash between any two leaves a `stale` row + (possibly already-dropped)
    /// table that the startup-GC `gc_orphan_embedding_tables` reclaims. A no-op
    /// when the coordinate has no `active` row (already retired).
    async fn retire_coordinate(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
    ) -> Result<(), LensError>;
}

/// Slugifies a model id into a filesystem/table-safe token.
///
/// `nomic-embed-text-v1.5` → `nomic_v15` (the registered convention); any other
/// id is conservatively lowercased with non-alphanumerics collapsed to `_` so
/// the resulting table name is always a valid identifier.
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

/// Resolves the physical LanceDB table name for a logical coordinate
/// `(notebook, model, dim)`.
///
/// The `dim` is encoded as a `__d{dim}` segment so two coordinates that share a
/// `(notebook, model)` but differ in `dim` (M4 Phase 4b: the same notebook may be
/// re-embedded with a different model/dim) never collide on one physical table:
/// `vec__{notebook}__{model_slug}__d{dim}`.
///
/// CRITICAL: this generator names NEW coordinates only. EXISTING tables keep their
/// stored physical name — every read/write path resolves a coordinate to a table
/// via the registry's `lance_table_name` column, never by re-deriving it here. A
/// nomic-768 table created before this segment existed (`vec__{nb}__nomic_v15`)
/// stays valid because its name lives in the registry; only freshly-created
/// coordinates flow through this function.
fn table_name(notebook: &str, model: &str, dim: usize) -> String {
    format!("vec__{notebook}__{}__d{dim}", model_slug(model))
}

/// Resolves the gen-suffixed physical table name for a coordinate + generation
/// (Decision A: the re-embed new-table-flip).
///
/// **gen-0 == [`table_name`] (byte-identical).** A non-zero generation appends
/// `__{gen}` AFTER the dim segment (`vec__{nb}__{slug}__d{dim}__{gen}`, R7
/// ordering: dim before gen) so a freshly-built `building` table co-exists beside
/// the live `active` table for the SAME coordinate (the partial-unique registry
/// from migration `0005` allows it).
fn gen_table_name(notebook: &str, model: &str, dim: usize, generation: u32) -> String {
    if generation == 0 {
        table_name(notebook, model, dim)
    } else {
        format!("{}__{generation}", table_name(notebook, model, dim))
    }
}

/// Escapes a string for safe interpolation into a LanceDB SQL filter literal.
///
/// LanceDB filter strings are SQL-like; a single quote inside an interpolated id
/// would otherwise terminate the literal early. Production ids are UUIDs (no
/// quotes), but the eval harness uses file-stem `source_id`s, so a stem with an
/// apostrophe must be doubled (`'` → `''`) per SQL string-literal escaping.
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

/// Registry repository over the `embedding_index` table.
///
/// Owned by [`LanceVectorStore`] as a private collaborator (Decision M1) — the
/// ingest pipeline never touches it directly. Maps each logical
/// `(notebook, model, dim)` to its physical `lance_table_name`, plus a status
/// lifecycle (`active` in Phase 1; `building`/`stale` reserved for the Phase-4
/// model-switch flow). Mirrors the `notebooks.rs` repo conventions: borrows a
/// pool, holds no other state, UUIDv7 ids, RFC3339 `created_at`.
struct EmbeddingIndexRepo {
    pool: SqlitePool,
}

impl EmbeddingIndexRepo {
    /// Wraps an owned pool handle (cheap clone of the `SqlitePool` Arc).
    fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Registers a `(notebook, model, dim)` mapping if absent.
    ///
    /// Uses `ON CONFLICT(notebook_id, model, dim) WHERE status='active' DO
    /// NOTHING` so re-registering an existing logical coordinate (e.g. on table
    /// re-open) keeps the existing row untouched and is a no-op. The conflict
    /// target carries the `WHERE status='active'` predicate because the registry
    /// constraint was relaxed in migration `0005` from a table-level
    /// `UNIQUE(notebook_id, model, dim)` to the PARTIAL unique index
    /// `uq_embidx_active … WHERE status='active'` (so a transient `building`/
    /// `stale` row can co-exist with the live `active` row during a re-embed
    /// flip). SQLite requires the upsert conflict target to match the partial
    /// index's predicate. `register` inserts a row with the given `status`
    /// (e.g. `create_building_table` registers a `building` row during a re-embed
    /// flip); only the partial-unique `uq_embidx_active` constrains uniqueness,
    /// and it constrains `active` rows ONLY, so transient `building`/`stale` rows
    /// freely co-exist with the live `active` row for the same coordinate.
    async fn register(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
        prefix: &str,
        table: &str,
        status: &str,
    ) -> Result<(), LensError> {
        let id = Uuid::now_v7().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO embedding_index \
                 (id, notebook_id, model, dim, prefix_convention, lance_table_name, status, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(notebook_id, model, dim) WHERE status='active' DO NOTHING",
        )
        .bind(&id)
        .bind(notebook)
        .bind(model)
        .bind(dim as i64)
        .bind(prefix)
        .bind(table)
        .bind(status)
        .bind(&now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Fetches the registry row for a logical coordinate, if registered.
    ///
    /// Part of the registry surface mandated by the plan (Step c.3); consumed by
    /// the Phase-4 model-switch flow, hence `allow(dead_code)` in Phase 1.
    #[allow(dead_code)]
    async fn get(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
    ) -> Result<Option<EmbeddingIndexRow>, LensError> {
        let row = sqlx::query_as::<_, EmbeddingIndexRow>(
            "SELECT id, notebook_id, model, dim, prefix_convention, lance_table_name, status, created_at \
             FROM embedding_index WHERE notebook_id = ? AND model = ? AND dim = ?",
        )
        .bind(notebook)
        .bind(model)
        .bind(dim as i64)
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

    /// Updates the `status` of an existing registry row (Phase-4 model-switch
    /// lifecycle: `active`/`building`/`stale`). Errors if no row matches.
    ///
    /// Reserved for the Phase-4 model-switch flow; hence `allow(dead_code)` in
    /// Phase 1, where status is always `active`.
    #[allow(dead_code)]
    async fn set_status(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
        status: &str,
    ) -> Result<(), LensError> {
        let result = sqlx::query(
            "UPDATE embedding_index SET status = ? \
             WHERE notebook_id = ? AND model = ? AND dim = ?",
        )
        .bind(status)
        .bind(notebook)
        .bind(model)
        .bind(dim as i64)
        .execute(&self.pool)
        .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!(
                "no embedding_index row for ({notebook}, {model}, {dim})"
            )));
        }
        Ok(())
    }

    /// Returns every `lance_table_name` registered for a coordinate, regardless of
    /// status (used to compute the next free generation for a building table).
    async fn lance_table_names_for_coordinate(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
    ) -> Result<Vec<String>, LensError> {
        let names = sqlx::query_scalar::<_, String>(
            "SELECT lance_table_name FROM embedding_index \
             WHERE notebook_id = ? AND model = ? AND dim = ?",
        )
        .bind(notebook)
        .bind(model)
        .bind(dim as i64)
        .fetch_all(&self.pool)
        .await?;
        Ok(names)
    }

    /// Resolves the physical `lance_table_name` of the live `status='active'` row
    /// for a coordinate, or `None` when none is registered (AC8 search read-path).
    async fn active_lance_table_name(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
    ) -> Result<Option<String>, LensError> {
        let name = sqlx::query_scalar::<_, String>(
            "SELECT lance_table_name FROM embedding_index \
             WHERE notebook_id = ? AND model = ? AND dim = ? AND status = 'active'",
        )
        .bind(notebook)
        .bind(model)
        .bind(dim as i64)
        .fetch_optional(&self.pool)
        .await?;
        Ok(name)
    }

    /// The ONE SQLite transaction of the re-embed flip (AC7): demote the current
    /// `active` row to `stale` and promote the `building` row named `building_name`
    /// to `active`, atomically. Returns the `lance_table_name` of the row demoted
    /// to `stale` (the caller drops that physical table + deletes its row after the
    /// txn commits).
    ///
    /// The partial-unique `uq_embidx_active` (migration `0005`) means at most one
    /// `active` row per coordinate survives each commit; doing both UPDATEs in one
    /// txn means a reader between the two statements is impossible (SQLite is
    /// serialized), so search never observes zero or two active tables.
    async fn flip_active_txn(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
        building_name: &str,
    ) -> Result<Option<String>, LensError> {
        let mut tx = self.pool.begin().await?;

        // Snapshot the current active table name (the one we are about to stale)
        // inside the txn so it is consistent with the UPDATEs below.
        let stale_name = sqlx::query_scalar::<_, String>(
            "SELECT lance_table_name FROM embedding_index \
             WHERE notebook_id = ? AND model = ? AND dim = ? AND status = 'active'",
        )
        .bind(notebook)
        .bind(model)
        .bind(dim as i64)
        .fetch_optional(&mut *tx)
        .await?;

        // Demote the current active row to stale FIRST so the partial-unique index
        // permits the promote below (at most one `active` per coordinate).
        sqlx::query(
            "UPDATE embedding_index SET status = 'stale' \
             WHERE notebook_id = ? AND model = ? AND dim = ? AND status = 'active'",
        )
        .bind(notebook)
        .bind(model)
        .bind(dim as i64)
        .execute(&mut *tx)
        .await?;

        // Promote the building row (matched by its physical table name) to active.
        let promoted = sqlx::query(
            "UPDATE embedding_index SET status = 'active' \
             WHERE notebook_id = ? AND model = ? AND dim = ? \
               AND lance_table_name = ? AND status = 'building'",
        )
        .bind(notebook)
        .bind(model)
        .bind(dim as i64)
        .bind(building_name)
        .execute(&mut *tx)
        .await?;
        if promoted.rows_affected() == 0 {
            // No building row to promote — abort so we never leave the coordinate
            // with zero active rows.
            tx.rollback().await?;
            return Err(LensError::Validation(format!(
                "no building embedding_index row named {building_name} for \
                 ({notebook}, {model}, {dim}) to promote"
            )));
        }

        tx.commit().await?;
        Ok(stale_name)
    }

    /// Demotes ONLY the `status='active'` row of a coordinate to `stale`,
    /// returning its physical `lance_table_name` (or `None` when the coordinate
    /// has no active row — an idempotent no-op). The retirement counterpart to the
    /// flip's in-txn demote.
    ///
    /// Deliberately distinct from [`set_status`](Self::set_status): `set_status`
    /// rewrites EVERY row for a coordinate regardless of status (it is used for
    /// arbitrary lifecycle transitions), whereas retiring an OLD-model coordinate
    /// must touch the live `active` row only — never a transient `building`/`stale`
    /// row that may coexist for the same coordinate during a concurrent flip (R3).
    async fn demote_active_to_stale(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
    ) -> Result<Option<String>, LensError> {
        let mut tx = self.pool.begin().await?;
        let active = sqlx::query_scalar::<_, String>(
            "SELECT lance_table_name FROM embedding_index \
             WHERE notebook_id = ? AND model = ? AND dim = ? AND status = 'active'",
        )
        .bind(notebook)
        .bind(model)
        .bind(dim as i64)
        .fetch_optional(&mut *tx)
        .await?;
        if active.is_some() {
            sqlx::query(
                "UPDATE embedding_index SET status = 'stale' \
                 WHERE notebook_id = ? AND model = ? AND dim = ? AND status = 'active'",
            )
            .bind(notebook)
            .bind(model)
            .bind(dim as i64)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(active)
    }

    /// Deletes the registry row naming `table` for a coordinate (the stale row
    /// removed after its Lance table is dropped post-flip). Idempotent: a missing
    /// row is not an error (the startup-GC may have already reclaimed it).
    async fn delete_row_by_table(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
        table: &str,
    ) -> Result<(), LensError> {
        sqlx::query(
            "DELETE FROM embedding_index \
             WHERE notebook_id = ? AND model = ? AND dim = ? AND lance_table_name = ?",
        )
        .bind(notebook)
        .bind(model)
        .bind(dim as i64)
        .bind(table)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

/// A row of the `embedding_index` registry table.
///
/// Returned only by [`EmbeddingIndexRepo::get`] (the Phase-4 model-switch
/// surface); its fields are unread in Phase 1, hence `allow(dead_code)`.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, sqlx::FromRow)]
struct EmbeddingIndexRow {
    /// UUIDv7 primary key.
    id: String,
    /// Owning notebook id.
    notebook_id: String,
    /// Embedding model id (e.g. `nomic-embed-text-v1.5`).
    model: String,
    /// Embedding dimension.
    dim: i64,
    /// Prefix convention applied at embed time (`search_document/search_query`).
    prefix_convention: String,
    /// Physical LanceDB table name (`vec__{notebook}__{model_slug}`).
    lance_table_name: String,
    /// Lifecycle status (`active` in Phase 1).
    status: String,
    /// RFC3339 creation timestamp.
    created_at: String,
}

/// Default [`VectorStore`] backed by an embedded LanceDB at `{data_dir}/lancedb`.
///
/// Holds the connection root and owns the [`EmbeddingIndexRepo`] collaborator
/// (Decision M1). Construct with [`LanceVectorStore::new`].
pub struct LanceVectorStore {
    /// `{data_dir}/lancedb` — the LanceDB connection root.
    root: PathBuf,
    /// The owned registry repository (private collaborator, never exposed).
    registry: EmbeddingIndexRepo,
    /// Row count at/above which `add`/`add_to_table` build (or refresh) the
    /// IVF_PQ index on the `vector` column. Defaults to [`ANN_INDEX_MIN_ROWS`];
    /// tests lower it via [`with_ann_index_min_rows`](Self::with_ann_index_min_rows).
    ann_index_min_rows: usize,
}

impl LanceVectorStore {
    /// Creates a store rooted at `{data_dir}/lancedb`, owning a registry repo
    /// over `pool`. The directory is created lazily by LanceDB on first connect.
    pub fn new(data_dir: &Path, pool: SqlitePool) -> Self {
        Self {
            root: data_dir.join("lancedb"),
            registry: EmbeddingIndexRepo::new(pool),
            ann_index_min_rows: ANN_INDEX_MIN_ROWS,
        }
    }

    /// Overrides the ANN-index row threshold (test-only seam).
    ///
    /// Lets integration tests trigger IVF_PQ creation at a few hundred rows
    /// instead of the production [`ANN_INDEX_MIN_ROWS`] (100k) — building a real
    /// 100k-vector table per test would be prohibitively slow. Compiled out of
    /// production builds; there is no runtime/env override on the production path.
    #[cfg(feature = "test-util")]
    pub fn with_ann_index_min_rows(mut self, min_rows: usize) -> Self {
        self.ann_index_min_rows = min_rows;
        self
    }

    /// Drops every per-notebook Lance table registered for `notebook`.
    ///
    /// Resolves the notebook's physical `lance_table_name`s via the owned
    /// registry, then drops each table (missing tables are a no-op). Used by the
    /// notebook hard-delete path (`LensEngine::purge_notebook`) BEFORE the SQLite
    /// delete so the cascade that removes the `embedding_index` rows can never
    /// orphan a Lance table on disk.
    pub async fn drop_notebook_tables(&self, notebook: &str) -> Result<(), LensError> {
        let names = self
            .registry
            .lance_table_names_for_notebook(notebook)
            .await?;
        self.drop_tables(&names).await
    }

    /// Opens a LanceDB connection at the store root.
    ///
    /// Pins `read_consistency_interval = 0` so EVERY table open re-reads the latest
    /// committed dataset version. The default (`None`) caches a table's version on
    /// open and never refreshes, which means a second connection (or a reopened
    /// handle) can miss a commit another just made — e.g. an index built by the
    /// previous `add` would be invisible to the next, producing a duplicate index;
    /// likewise a post-flip search could read a stale version. Strong read
    /// consistency costs only a cheap version check per op on this local embedded
    /// store, and it keeps the registry (SQLite, always current) and Lance in step.
    async fn connect(&self) -> Result<Connection, LensError> {
        let root = self.root.to_string_lossy();
        lancedb::connect(&root)
            .read_consistency_interval(std::time::Duration::ZERO)
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb connect failed: {e}")))
    }

    /// Opens a physical table by its EXACT name IF it exists, else `None`.
    ///
    /// Shared by the registry-driven read path ([`open_active_table`]) and the
    /// explicit-name write path ([`add_to_table`](VectorStore::add_to_table)). It
    /// never creates or registers anything.
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

    /// Resolves the `status='active'` table from the registry and opens it (AC8).
    ///
    /// Returns `None` when there is no `active` registry row (never ingested) OR
    /// when the named table is momentarily absent (the flip TOCTOU window) — search
    /// degrades to empty in both cases, never errors. Because gen-0 == the formula
    /// name, a pre-flip coordinate (whose only registered row was written by
    /// `ensure_table` as `active`) resolves to the identical physical table the
    /// old formula-based path opened.
    async fn open_active_table(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
    ) -> Result<Option<Table>, LensError> {
        let name = match self
            .registry
            .active_lance_table_name(notebook, model, dim)
            .await?
        {
            Some(n) => n,
            None => return Ok(None),
        };
        self.open_table_by_name(&name).await
    }

    /// Resolves `(notebook, model, dim)` to its physical table, creating it on
    /// first use.
    ///
    /// PRIVATE — driven by [`add`](VectorStore::add) only; the read/no-op paths
    /// use [`open_active_table`](Self::open_active_table) (registry-driven) so they
    /// never create a table as a side effect. Callers never see the physical table
    /// name. Idempotent: opens the table if it already exists, otherwise creates
    /// it (empty, schema-only) and registers the logical coordinate in the owned
    /// `embedding_index` registry.
    async fn ensure_table(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
    ) -> Result<Table, LensError> {
        let conn = self.connect().await?;

        // REGISTRY-DRIVEN resolution (mirror of `open_active_table`): if a
        // `status='active'` row already exists for this coordinate, append to THAT
        // physical table. After a re-embed flip the active table is gen-suffixed and
        // the bare `table_name()` formula points at a DROPPED stale table; resolving
        // via the formula here would create a fresh gen-0 orphan and silently lose
        // the appended vectors (search reads only the registered active table). This
        // is the write-path twin of the read-path AC8 resolution.
        if let Some(active_name) = self
            .registry
            .active_lance_table_name(notebook, model, dim)
            .await?
        {
            if let Some(table) = self.open_table_by_name(&active_name).await? {
                return Ok(table);
            }
            // Registered active row but the physical table is momentarily absent.
            // `add` and `flip_active` are both serialized under `ingest_lock`, so
            // this should not occur in practice; recreate the REGISTERED-named table
            // (not a gen-0 formula table) so the append lands where search reads.
            // Warn so an operator notices if this defensive path ever fires — it
            // signals an unexpected registry/Lance divergence.
            tracing::warn!(
                notebook,
                model,
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

        // No active row ⇒ first CREATE for this coordinate. gen-0 == the formula
        // name. Open a pre-existing (unregistered, e.g. crash-orphaned) table or
        // create a fresh one, then register the mapping `active` (the register is a
        // no-op on conflict, so a concurrent create is self-healing).
        let name = table_name(notebook, model, dim);
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
                notebook,
                model,
                dim,
                crate::embedder::PREFIX_CONVENTION,
                &name,
                REGISTRY_STATUS_ACTIVE,
            )
            .await?;

        Ok(table)
    }

    /// Copies every row of the current active table whose `source_id` differs from
    /// `exclude_source_id` into the named building table.
    ///
    /// Used by the per-source re-embed flip so the rebuilt table PRESERVES the
    /// notebook's other sources instead of replacing the whole table with just the
    /// re-embedded source. The caller then appends `exclude_source_id`'s freshly
    /// re-embedded vectors, and the flip promotes a table holding every source.
    ///
    /// A no-op when the coordinate has no active table yet (the notebook's first
    /// source) — the caller then populates the building table with that source
    /// alone. Vectors are copied as Arrow batches directly (both tables share
    /// [`vector_schema`]), so no float round-trip or re-embed of unchanged sources.
    pub(crate) async fn seed_building_from_active(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
        building_name: &str,
        exclude_source_id: &str,
    ) -> Result<(), LensError> {
        let active = match self.open_active_table(notebook, model, dim).await? {
            Some(t) => t,
            None => return Ok(()),
        };
        let building = self
            .open_table_by_name(building_name)
            .await?
            .ok_or_else(|| {
                LensError::Vector(format!("no building table {building_name} to seed"))
            })?;

        // Stream the scan one batch at a time (read → write → drop), capping memory
        // at a single batch rather than materializing the whole active-minus-source
        // table in RAM (the notebook can hold many sources' vectors).
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

    /// Builds or refreshes the IVF_PQ ANN index on `table`'s `vector` column once
    /// the row count crosses [`ann_index_min_rows`](Self::ann_index_min_rows);
    /// below threshold (and unindexed) it is a no-op (M4 Phase 4a).
    ///
    /// Invoked at the END of every row-landing path ([`add`](VectorStore::add) and
    /// [`add_to_table`](VectorStore::add_to_table)), never on the search hot path:
    ///
    /// - no index yet + rows ≥ threshold ⇒ `create_index` (cosine, `replace(true)`);
    /// - index already present ⇒ `optimize(append)` to fold the just-appended
    ///   fragments into it;
    /// - no index + below threshold ⇒ nothing.
    ///
    /// NON-FATAL by design: any `list_indices`/`count_rows`/build/refresh failure is
    /// logged and swallowed — search degrades to exact brute-force kNN (correct,
    /// only slower), so a transient index error never fails an ingest or re-embed.
    /// `dim` selects the PQ sub-vector count; it MUST match the table's vector
    /// width (callers pass the coordinate's `dim`, resolved per notebook model).
    ///
    /// Takes the JUST-WRITTEN handle so the threshold gate (`count_rows`, which DOES
    /// reflect the rows this call appended) costs nothing extra in the common case —
    /// a notebook far below the threshold returns immediately, with no table reopen
    /// and no index-metadata read. Only once the threshold is crossed does it reopen
    /// a FRESH handle for the index work: `list_indices`/`create_index` on the same
    /// handle just mutated by `add()` return stale (pre-create) metadata, so a prior
    /// add's index would be invisible and we'd build a duplicate; the reopen reads
    /// the committed on-disk index state.
    async fn maybe_build_or_refresh_index(&self, table: &Table, dim: usize) {
        // Cheap gate on the just-written handle. The overwhelmingly common path — a
        // corpus below the threshold — stops HERE, adding only a row count per add.
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

        // At/above threshold: reopen so the index metadata is the committed state.
        let table = match self.open_table_by_name(table.name()).await {
            Ok(Some(t)) => t,
            Ok(None) => {
                // The caller just appended to this table, so it must exist; a miss
                // here is an unexpected registry/Lance divergence — skip, non-fatal.
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
            // Fold the fragments appended since the last optimize into the index.
            // A failure here just leaves the newest rows on the brute-force path
            // until the next refresh — never fatal (mirrors the plan's contract).
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

        // `num_partitions ≈ √rows` (Lance guidance). `as u32` truncation is fine —
        // partition count is a tuning knob, not a correctness input. `f64::sqrt`
        // keeps the MSRV (no `usize::isqrt`, stabilized later).
        let num_partitions = ((n as f64).sqrt() as u32).max(1);
        // PQ sub-vectors: `dim/16` (each spans 16 dims) when divisible, else `dim/8`,
        // else 1. `num_sub_vectors` MUST divide `dim`; for 768 this is 48.
        let num_sub_vectors = if dim.is_multiple_of(16) {
            (dim / 16) as u32
        } else if dim.is_multiple_of(8) {
            (dim / 8) as u32
        } else {
            1
        };

        let index = Index::IvfPq(
            IvfPqIndexBuilder::default()
                // Pin cosine to MATCH the query metric (`search` pins it too); the
                // builder defaults to L2, which would silently mis-rank.
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
    async fn add(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
        rows: Vec<VectorRow>,
    ) -> Result<(), LensError> {
        if rows.is_empty() {
            return Ok(());
        }
        let table = self.ensure_table(notebook, model, dim).await?;
        let batch = rows_to_batch(&rows, dim)?;
        // `Scannable` is implemented for `RecordBatch` directly in lancedb 0.30.0
        // (no `RecordBatchIterator` impl), so pass the batch as-is.
        table
            .add(batch)
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb add failed: {e}")))?;
        // Maintain the ANN index once this table crosses the row threshold (a
        // no-op for the small corpora that never approach it). Non-fatal — a
        // failure leaves search on exact brute-force kNN.
        self.maybe_build_or_refresh_index(&table, dim).await;
        Ok(())
    }

    async fn search(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<Hit>, LensError> {
        if query.len() != dim {
            return Err(LensError::Vector(format!(
                "query vector length {} != expected dim {dim}",
                query.len()
            )));
        }
        // AC8 — REGISTRY-DRIVEN resolution: the physical table is the one named by
        // the `status='active'` registry row, NOT the bare `table_name()` formula.
        // Because gen-0 == the formula name (Decision A), a notebook that was only
        // ever ingested pre-flip resolves to the SAME physical table as before, so
        // existing Phase-1/2 search is byte-identical. After a re-embed flip the
        // active row names the gen-suffixed enriched table and search follows it.
        // A search must NEVER create a table as a side effect: a never-ingested
        // coordinate (no active row) OR a row whose Lance table is momentarily
        // gone (the flip TOCTOU window) returns an empty result, never an error.
        let table = match self.open_active_table(notebook, model, dim).await? {
            Some(t) => t,
            None => return Ok(Vec::new()),
        };

        // PIN cosine explicitly (the ACs assume ascending cosine distance —
        // never rely on the crate's L2 default). `.only_if` is cheap
        // defense-in-depth notebook isolation on top of the per-notebook table.
        // The notebook id is escaped for the SQL-like filter literal.
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

        // The query orders by distance, but collecting across batches can
        // interleave; sort ascending to honor the AC contract unconditionally.
        hits.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        Ok(hits)
    }

    /// Removes a source's rows from the ACTIVE embedding table. Concurrent
    /// `building` tables (mid re-embed flip) are safe to ignore: the flip acquires
    /// `ingest_lock`, serializing it against `purge_source`, so a drop never races
    /// a half-built table into an inconsistent active set.
    async fn drop_source(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
        source_id: &str,
    ) -> Result<(), LensError> {
        // Resolve the ACTIVE physical table from the registry (AC8/AC6): after a
        // re-embed flip the active table is gen-suffixed, and the summary RAPTOR
        // node (which carries this `source_id`) lives in it — so a drop must target
        // the active table, not the bare formula name (gen-0), to reclaim it. A
        // wipe of a never-ingested coordinate (no active row) is a no-op.
        let table = match self.open_active_table(notebook, model, dim).await? {
            Some(t) => t,
            None => return Ok(()),
        };
        // Escape the source_id for the SQL-like filter literal (eval harness uses
        // non-UUID file-stem ids that may contain an apostrophe).
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
        // Snapshot the live table set once and guard each drop with an existence
        // check: lancedb 0.30's `drop_table` errors on a missing table, but the
        // purge path must be idempotent (a crash between Lance-drop and the
        // SQLite commit means a re-purge re-drops the same names), so a missing
        // table is treated as a no-op.
        let existing = conn
            .table_names()
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb table_names failed: {e}")))?;
        for name in table_names {
            if !existing.iter().any(|t| t == name) {
                continue;
            }
            // Empty namespace path = root namespace, matching how
            // `create_empty_table(...).execute()` registers these tables.
            conn.drop_table(name.as_str(), &[])
                .await
                .map_err(|e| LensError::Vector(format!("lancedb drop_table failed: {e}")))?;
        }
        Ok(())
    }

    async fn create_building_table(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
    ) -> Result<String, LensError> {
        // Compute the next free generation above every name registered for the
        // coordinate (gen-0 is the live active table; prior building/stale rows may
        // linger until the startup-GC sweeps them).
        let existing = self
            .registry
            .lance_table_names_for_coordinate(notebook, model, dim)
            .await?;
        let mut generation = 1u32;
        loop {
            let candidate = gen_table_name(notebook, model, dim, generation);
            if !existing.iter().any(|n| n == &candidate) {
                break;
            }
            generation += 1;
        }
        let building_name = gen_table_name(notebook, model, dim, generation);

        // Create the empty physical table, then register the `building` row. The
        // physical table is created FIRST so a crash before the registry insert
        // leaves an unregistered orphan table (harmless — invisible to the read
        // path and reclaimed lazily); a crash after leaves a `building` row the
        // startup-GC reclaims (idempotent drop of a possibly-missing table).
        let conn = self.connect().await?;
        conn.create_empty_table(building_name.as_str(), vector_schema(dim))
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb create_empty_table failed: {e}")))?;
        self.registry
            .register(
                notebook,
                model,
                dim,
                crate::embedder::PREFIX_CONVENTION,
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
        // Same ANN maintenance as `add`: a `building` table populated past the
        // threshold is indexed BEFORE the flip promotes it to active, so it serves
        // ANN immediately on becoming active (why 4a lands before the 4b reindex).
        // The bulk seed copy in `seed_building_from_active` goes through the raw
        // table handle (not this path), so the index is built once the per-source
        // populate begins — over the complete row set, not the half-seeded one.
        self.maybe_build_or_refresh_index(&table, dim).await;
        Ok(())
    }

    async fn flip_active(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
        building_name: &str,
    ) -> Result<(), LensError> {
        // (1) The ONE atomic SQLite txn: active→stale, building→active. Returns the
        // physical name of the row demoted to stale (if any) to drop afterwards.
        let stale_name = self
            .registry
            .flip_active_txn(notebook, model, dim, building_name)
            .await?;

        // ── CRASH WINDOW (AC7): everything below — the stale Lance drop + the
        // stale registry-row delete — happens AFTER the flip txn has committed. A
        // crash here leaves a `stale` row + orphan Lance table that the startup-GC
        // reclaims. The active row already points at the COMPLETE enriched table.
        #[cfg(feature = "test-util")]
        if CRASH_AFTER_FLIP_TXN_BEFORE_LANCE_DROP.swap(false, std::sync::atomic::Ordering::SeqCst) {
            return Ok(());
        }

        // (2) Drop the stale physical table (idempotent) and delete its registry
        // row. Order: drop the table FIRST so a crash between the two leaves only a
        // dangling `stale` row whose (already-gone) table the GC drops as a no-op.
        if let Some(stale) = stale_name {
            self.drop_tables(std::slice::from_ref(&stale)).await?;
            self.registry
                .delete_row_by_table(notebook, model, dim, &stale)
                .await?;
        }
        Ok(())
    }

    async fn retire_coordinate(
        &self,
        notebook: &str,
        model: &str,
        dim: usize,
    ) -> Result<(), LensError> {
        // (1) Demote ONLY the OLD coordinate's active row to stale (committed).
        // Returns its physical table name, or None when there is nothing to retire
        // (idempotent no-op — e.g. a retried retire after the GC already swept it).
        let Some(stale) = self
            .registry
            .demote_active_to_stale(notebook, model, dim)
            .await?
        else {
            return Ok(());
        };

        // ── CRASH WINDOW (R3): the stale Lance drop + the stale registry-row
        // delete happen AFTER the demote has committed. A crash here leaves a
        // `stale` row + orphan Lance table that the startup-GC reclaims; the NEW
        // coordinate's active row (already flipped) keeps serving search.
        #[cfg(feature = "test-util")]
        if CRASH_AFTER_RETIRE_STALE_BEFORE_LANCE_DROP
            .swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            return Ok(());
        }

        // (2) Drop the stale physical table FIRST (idempotent), then delete its
        // registry row — so a crash between the two leaves only a dangling `stale`
        // row whose (already-gone) table the GC drops as a no-op.
        self.drop_tables(std::slice::from_ref(&stale)).await?;
        self.registry
            .delete_row_by_table(notebook, model, dim, &stale)
            .await?;
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
    fn table_name_includes_dim_segment() {
        assert_eq!(
            table_name("nb1", "nomic-embed-text-v1.5", 768),
            "vec__nb1__nomic_v15__d768"
        );
        assert_eq!(
            table_name("nb1", "all-minilm", 384),
            "vec__nb1__all_minilm__d384"
        );
    }

    #[test]
    fn gen_table_name_dim_before_gen_ordering() {
        // gen-0 == table_name (no gen suffix).
        assert_eq!(
            gen_table_name("nb1", "mxbai-embed-large", 1024, 0),
            "vec__nb1__mxbai_embed_large__d1024"
        );
        // Non-zero generation appends `__{gen}` AFTER the dim segment (R7).
        assert_eq!(
            gen_table_name("nb1", "mxbai-embed-large", 1024, 2),
            "vec__nb1__mxbai_embed_large__d1024__2"
        );
    }

    #[test]
    fn different_dims_never_collide_for_same_notebook() {
        // A 384 and a 768 coordinate for the same notebook must produce distinct
        // physical names so two active coordinates never share a Lance table.
        let n384 = table_name("nb1", "all-minilm", 384);
        let n768 = table_name("nb1", "nomic-embed-text-v1.5", 768);
        assert_ne!(n384, n768);
        // Same property holds across generations.
        assert_ne!(
            gen_table_name("nb1", "all-minilm", 384, 1),
            gen_table_name("nb1", "nomic-embed-text-v1.5", 768, 1)
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
