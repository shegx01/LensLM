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
//! vector: FixedSizeList<Float32, 768>}`. Searches pin the distance metric to
//! cosine **explicitly** and additionally run `.only_if("notebook_id = '…'")`
//! as cheap defense-in-depth (the AC asserts notebook isolation directly).
//! Brute-force kNN only — no IvfPq in Phase 1 (corpora < ~100k vectors).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, StringArray,
    types::Float32Type,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures_util::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, DistanceType, Table};
use sqlx::SqlitePool;
use uuid::Uuid;

use crate::LensError;

/// Embedding dimension for nomic-embed-text-v1.5. Phase 1 is fixed at 768.
pub const VECTOR_DIM: usize = 768;

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
    /// The embedding vector. MUST be length [`VECTOR_DIM`].
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
    /// not exist. An empty `rows` is a no-op.
    async fn add_to_table(&self, table_name: &str, rows: Vec<VectorRow>) -> Result<(), LensError>;

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

/// Resolves the physical LanceDB table name for a logical coordinate.
///
/// **`dim` is intentionally absent from the physical name** even though it is part
/// of the *logical* key `(notebook, model, dim)`. Phase 1 relies on the invariant
/// `model ⇒ dim` (the only registered model, `nomic-embed-text-v1.5`, is fixed at
/// [`VECTOR_DIM`] = 768), so `vec__{notebook}__{model_slug}` uniquely identifies a
/// table. The table-resolution path ([`LanceVectorStore::ensure_table`]) carries a
/// `debug_assert_eq!(dim, VECTOR_DIM, …)` guard documenting this assumption.
///
/// If a future model is ever registered at *two* dims, two logical registry rows
/// would collide on this one physical table name — at that point the scheme MUST
/// be extended to include `dim` (e.g. `vec__{notebook}__{model_slug}__{dim}`).
fn table_name(notebook: &str, model: &str) -> String {
    format!("vec__{notebook}__{}", model_slug(model))
}

/// Resolves the gen-suffixed physical table name for a coordinate + generation
/// (Decision A: the re-embed new-table-flip).
///
/// **gen-0 == [`table_name`] (byte-identical).** The Phase-1/2 tables shipped
/// before the flip existed; they are gen-0 *by convention*, so resolving gen-0
/// must reproduce the exact formula name or pre-flip search would point at a
/// table that does not exist. A non-zero generation appends `__{gen}` so a
/// freshly-built `building` table co-exists beside the live `active` table for
/// the SAME coordinate (the partial-unique registry from migration `0005` allows
/// it). Enrichment never changes model/dim, so the `dim`-absent caveat on
/// [`table_name`] is irrelevant here — only the generation differs.
fn gen_table_name(notebook: &str, model: &str, generation: u32) -> String {
    if generation == 0 {
        table_name(notebook, model)
    } else {
        format!("{}__{generation}", table_name(notebook, model))
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
}

impl LanceVectorStore {
    /// Creates a store rooted at `{data_dir}/lancedb`, owning a registry repo
    /// over `pool`. The directory is created lazily by LanceDB on first connect.
    pub fn new(data_dir: &Path, pool: SqlitePool) -> Self {
        Self {
            root: data_dir.join("lancedb"),
            registry: EmbeddingIndexRepo::new(pool),
        }
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
    async fn connect(&self) -> Result<Connection, LensError> {
        let root = self.root.to_string_lossy();
        lancedb::connect(&root)
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
        // PHASE-1 INVARIANT (see `table_name`): the physical table name omits
        // `dim` and relies on `model ⇒ dim`. The only registered model is fixed
        // at `VECTOR_DIM`, so a `dim` other than that would silently resolve to a
        // table built for a different dimension. Guard it here at the resolution
        // path; registering a second dim for the same model must first extend the
        // table-name scheme to include `dim`.
        debug_assert_eq!(
            dim, VECTOR_DIM,
            "table_name omits dim and relies on model⇒dim (Phase 1): a second dim \
             for the same model must extend the table-name scheme to include dim"
        );

        let conn = self.connect().await?;
        let name = table_name(notebook, model);

        let existing = conn
            .table_names()
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb table_names failed: {e}")))?;

        if existing.iter().any(|t| t == &name) {
            return conn
                .open_table(name.as_str())
                .execute()
                .await
                .map_err(|e| LensError::Vector(format!("lancedb open_table failed: {e}")));
        }

        // First CREATE for this logical coordinate: create an empty table with
        // our schema, then register the mapping (registry insert is a no-op on
        // conflict, so a concurrent create is self-healing).
        let table = conn
            .create_empty_table(name.as_str(), vector_schema(dim))
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb create_empty_table failed: {e}")))?;

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
        // PHASE-1 INVARIANT (see `table_name`): the physical name omits `dim`. The
        // re-embed never changes model/dim, so the same guard applies.
        debug_assert_eq!(
            dim, VECTOR_DIM,
            "gen_table_name omits dim and relies on model⇒dim (Phase 1)"
        );

        // Compute the next free generation above every name registered for the
        // coordinate (gen-0 is the live active table; prior building/stale rows may
        // linger until the startup-GC sweeps them).
        let existing = self
            .registry
            .lance_table_names_for_coordinate(notebook, model, dim)
            .await?;
        let mut generation = 1u32;
        loop {
            let candidate = gen_table_name(notebook, model, generation);
            if !existing.iter().any(|n| n == &candidate) {
                break;
            }
            generation += 1;
        }
        let building_name = gen_table_name(notebook, model, generation);

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

    async fn add_to_table(&self, table_name: &str, rows: Vec<VectorRow>) -> Result<(), LensError> {
        if rows.is_empty() {
            return Ok(());
        }
        let table = self
            .open_table_by_name(table_name)
            .await?
            .ok_or_else(|| LensError::Vector(format!("no table named {table_name} to add to")))?;
        let batch = rows_to_batch(&rows, VECTOR_DIM)?;
        table
            .add(batch)
            .execute()
            .await
            .map_err(|e| LensError::Vector(format!("lancedb add failed: {e}")))?;
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
    fn table_name_format() {
        assert_eq!(
            table_name("nb1", "nomic-embed-text-v1.5"),
            "vec__nb1__nomic_v15"
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
}
