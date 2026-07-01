//! Notebook domain: the `Notebook` entity, its strongly-typed id, and the
//! repository implementing CRUD over the `notebooks` table.
//!
//! This module establishes the per-domain repository pattern that M1+ entities
//! (sources, chunks, notes, …) follow: the engine (`lib.rs`) stays thin and owns
//! no domain entities; each domain owns its struct, id newtype, and a repo that
//! takes a `&SqlitePool`. `LensEngine` exposes a `pool()` accessor and delegates.

use std::fmt;
use std::io::Read as _;
use std::ops::Deref;
use std::path::Path;
use std::str::FromStr;

use sha2::{Digest, Sha256};

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use uuid::Uuid;

use crate::LensError;
use crate::parse::SourceKind;
use crate::url_normalize::normalize_url;

/// Maximum accepted notebook title length, in characters. Titles longer than
/// this are rejected with [`LensError::Validation`] rather than silently stored.
const MAX_TITLE_LEN: usize = 500;

/// Strongly-typed notebook identifier (a UUIDv7 stored as TEXT).
///
/// A newtype over `String` so notebook ids can't be silently mixed with the ids
/// of other entities (sources, chunks, …) introduced in later milestones. It
/// `Deref`s to `str` and is `From<String>`/`Display`, so it stays ergonomic at
/// call sites and binds directly into sqlx queries.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type)]
#[serde(transparent)]
#[sqlx(transparent)]
pub struct NotebookId(pub(crate) String);

impl NotebookId {
    /// Mints a fresh time-ordered (UUIDv7) notebook id.
    pub fn new() -> Self {
        Self(Uuid::now_v7().to_string())
    }

    /// Borrows the inner id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for NotebookId {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for NotebookId {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<String> for NotebookId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for NotebookId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl fmt::Display for NotebookId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// Source ingestion status values (the `sources.status` column).
///
/// Single source of truth for the status string literals used across the ingest
/// pipeline, the engine, and the crash-recovery path. The lifecycle is
/// `queued → parsing → embedding → indexed` (or `error` on failure). `pending`
/// is the legacy status [`NotebookRepo::add_source`] writes for inert M1 file
/// records (awaiting M4 ingestion).
///
/// # Terminal-pending statuses
///
/// `needs_ocr` and `needs_js` are TERMINAL-PENDING: the source has been
/// processed but could not be fully indexed without additional capability (OCR
/// or a JS-rendering browser). They are NOT transient — a restart must NOT reset
/// them to `error`. The crash-recovery reset in [`crate::LensEngine::init`]
/// (`WHERE status IN (parsing, embedding)`) explicitly excludes both. This
/// exclusion is locked by the `crash_recovery_skips_needs_js_and_needs_ocr` test.
///
/// Serialized to / parsed from the EXACT legacy `sources.status` strings via
/// [`as_str`](Self::as_str) and [`FromStr`] — the persisted column values are
/// unchanged (no DB migration). The [`Source`] FromRow struct keeps `status:
/// String` at the DB-row / IPC boundary; write sites bind `SourceStatus::X.as_str()`
/// and gates/transitions match on the parsed enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceStatus {
    /// Inert M1 file record awaiting M4 ingestion.
    Pending,
    /// Queued for ingestion (the M4 managed-text entry state).
    Queued,
    /// Parse phase in progress (transient — reset to `error` on crash recovery).
    Parsing,
    /// Embed phase in progress (transient — reset to `error` on crash recovery).
    Embedding,
    /// Fully ingested and indexed.
    Indexed,
    /// Ingestion failed (terminal until re-ingest).
    Error,
    /// Terminal-pending: PDF/image source requires OCR to extract text.
    /// Must NOT be reset to `error` on crash recovery (it is not transient).
    NeedsOcr,
    /// Terminal-pending: URL source returned near-empty text — likely a JS-rendered
    /// SPA. Must NOT be reset to `error` on crash recovery (it is not transient).
    NeedsJs,
}

impl SourceStatus {
    /// The EXACT `sources.status` string stored in SQLite for this variant.
    /// Inverse of [`FromStr`]. These strings are the persisted wire format and
    /// MUST NOT change (no DB migration).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Queued => "queued",
            Self::Parsing => "parsing",
            Self::Embedding => "embedding",
            Self::Indexed => "indexed",
            Self::Error => "error",
            Self::NeedsOcr => "needs_ocr",
            Self::NeedsJs => "needs_js",
        }
    }

    /// Whether this status is a TRANSIENT in-progress state that the startup
    /// crash-recovery reset must flip back to [`Error`](Self::Error) (a process
    /// that died mid-ingest left no task to advance it).
    ///
    /// Exhaustive by construction: every status is classified here, so adding a
    /// variant is a compile error that forces a recovery decision. The
    /// terminal-pending `NeedsOcr`/`NeedsJs` (and every terminal state) are
    /// `false` — they must NOT be reset (locked by
    /// `crash_recovery_skips_needs_js_and_needs_ocr`).
    pub fn is_transient(&self) -> bool {
        match self {
            Self::Parsing | Self::Embedding => true,
            Self::Pending
            | Self::Queued
            | Self::Indexed
            | Self::Error
            | Self::NeedsOcr
            | Self::NeedsJs => false,
        }
    }
}

impl FromStr for SourceStatus {
    type Err = LensError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pending" => Ok(Self::Pending),
            "queued" => Ok(Self::Queued),
            "parsing" => Ok(Self::Parsing),
            "embedding" => Ok(Self::Embedding),
            "indexed" => Ok(Self::Indexed),
            "error" => Ok(Self::Error),
            "needs_ocr" => Ok(Self::NeedsOcr),
            "needs_js" => Ok(Self::NeedsJs),
            other => Err(LensError::Validation(format!(
                "unknown source status: {other:?}; expected one of \"pending\", \"queued\", \
                 \"parsing\", \"embedding\", \"indexed\", \"error\", \"needs_ocr\", \"needs_js\""
            ))),
        }
    }
}

/// The per-source enrichment lifecycle, persisted in `sources.enrichment_status`
/// as one of the literal strings below. SEPARATE from [`SourceStatus`] (lock #2
/// of the M4 Phase-3 plan): "searchable" (the `SourceStatus`) stays orthogonal to
/// "enriched", so adding enrichment never touches the compile-locked SourceStatus
/// crash-recovery invariant.
///
/// `NULL` in the column (pre-Phase-3 rows, or a freshly-ingested source before the
/// worker touches it) is read as [`None`](EnrichmentStatus::None) by convention.
/// The string literals are the persisted wire format and MUST NOT change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnrichmentStatus {
    /// Not enriched (and not queued). Equivalent to a NULL column value.
    None,
    /// Queued for the background enrichment worker (or awaiting a reachable
    /// provider). Eligible for the startup/rescan queue-rebuild.
    Pending,
    /// The worker is actively enriching this source. TRANSIENT — reset to
    /// [`Pending`](EnrichmentStatus::Pending) on crash recovery (a process that
    /// died mid-enrichment left no task to advance it). Distinct from the
    /// `SourceStatus` transient reset, which it never collides with.
    Enriching,
    /// Enrichment completed (structural map + contextual `embedding_text` +
    /// summary node + re-embed flip all applied).
    Enriched,
    /// Enrichment failed (LLM unreachable mid-run, budget overspend, malformed
    /// output). Raw vectors are untouched; eligible for re-enqueue.
    Failed,
    /// The structural-map pass was deliberately skipped (non-prose kind); a
    /// lightweight context-prefix `embedding_text` may still be applied.
    Skipped,
}

impl EnrichmentStatus {
    /// The EXACT `sources.enrichment_status` string stored in SQLite. Inverse of
    /// [`from_db`](Self::from_db). These strings are the persisted wire format.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Pending => "pending",
            Self::Enriching => "enriching",
            Self::Enriched => "enriched",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
        }
    }

    /// Parses a stored column value into the enum. A NULL column (`None`) or the
    /// literal `"none"` both map to [`None`](EnrichmentStatus::None); an
    /// out-of-vocabulary value is a fail-loud invariant breach.
    pub fn from_db(value: Option<&str>) -> Result<Self, LensError> {
        match value {
            None | Some("none") => Ok(Self::None),
            Some("pending") => Ok(Self::Pending),
            Some("enriching") => Ok(Self::Enriching),
            Some("enriched") => Ok(Self::Enriched),
            Some("failed") => Ok(Self::Failed),
            Some("skipped") => Ok(Self::Skipped),
            Some(other) => Err(LensError::Validation(format!(
                "unknown enrichment status: {other:?}; expected one of \"none\", \"pending\", \
                 \"enriching\", \"enriched\", \"failed\", \"skipped\""
            ))),
        }
    }
}

/// A source row, returned across the IPC boundary.
///
/// In M1 sources are inert *records* only: the onboarding "Add sources" step
/// inserts file rows with `status = "pending"`, and M4 ingestion later picks up
/// the pending rows to parse/enrich/embed. No parsing/embedding happens here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, sqlx::FromRow)]
pub struct Source {
    /// UUIDv7 primary key, stored as TEXT.
    pub id: String,
    /// Owning notebook id.
    pub notebook_id: String,
    /// Source kind. Always `"file"` in M1.
    pub kind: String,
    /// Display title (the file name).
    pub title: String,
    /// Ingestion status. Always `"pending"` in M1 (awaiting M4 ingestion).
    pub status: String,
    /// Absolute file path.
    pub locator: String,
    /// Whether the source is selected for retrieval (`1` = selected).
    pub selected: i64,
    /// Total token count of the source text, populated by M4 ingestion.
    /// `None` until the source has been ingested.
    pub token_count: Option<i64>,
    /// SHA-256 of the canonical source text, populated by M4 ingestion.
    /// Used to short-circuit re-ingest when the content is unchanged. `None`
    /// until the source has been ingested.
    pub content_hash: Option<String>,
    /// SHA-256 of the RAW file bytes, computed at *add* time (issue #96), used as
    /// the add-time dedup key. SEPARATE from `content_hash` (extracted-text hash,
    /// set at ingest time). `None` for text/paste/url sources and pre-migration
    /// rows. A partial unique index on `(notebook_id, raw_content_hash) WHERE
    /// trashed_at IS NULL AND raw_content_hash IS NOT NULL` enforces
    /// at-most-one live source per (notebook, raw-content) pair.
    pub raw_content_hash: Option<String>,
    /// RFC3339 creation timestamp.
    pub created_at: String,
    /// RFC3339 soft-delete timestamp, or `None` if live.
    pub trashed_at: Option<String>,
    /// Enrichment lifecycle, SEPARATE from `status` (SourceStatus):
    /// `none|pending|enriching|enriched|failed|skipped`. `None` (NULL) ≡ `none`
    /// for pre-Phase-3 rows; populated by the M4 Phase-3 enrichment worker.
    pub enrichment_status: Option<String>,
    /// JSON enrichment metadata (composite cache key + budget/skip reason),
    /// written by the enrichment worker. `None` until the source is enriched.
    pub enrichment_meta: Option<String>,
}

/// The result of [`NotebookRepo::add_file_source`] (issue #96).
///
/// Carries the [`Source`] row plus a `was_existing` flag distinguishing a
/// freshly-inserted source (`false`) from a dedup hit — an already-present live
/// source in the same notebook with identical raw-file content (`true`). Serde
/// `camelCase` renaming yields the wire shape `{ source, wasExisting }` for the
/// Tauri command and the frontend `addFileSource` IPC wrapper.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AddSourceOutcome {
    /// The inserted (or pre-existing, on a dedup hit) source row.
    pub source: Source,
    /// `true` when the content already existed in the notebook (dedup hit — no
    /// new row was written and the file was NOT copied to managed storage);
    /// `false` when a new source row was inserted.
    pub was_existing: bool,
}

/// A chunk row projected for the enrichment pass (M4 Phase-3 Step 4).
///
/// A read-only view of the columns the structural-map + `embedding_text`
/// composition need; the canonical `text` here is the IMMUTABLE citation text
/// (enrichment writes only `embedding_text` / `enrichment`, never `text`).
#[derive(Debug, Clone, PartialEq, Eq, sqlx::FromRow)]
pub struct EnrichmentChunk {
    /// Chunk primary key.
    pub id: String,
    /// `None` for level-0 parents; `Some(parent.id)` for children.
    pub parent_id: Option<String>,
    /// `"parent"` (level 0) / `"child"` (level 1).
    pub kind: String,
    /// `0` parent, `1` child.
    pub level: i32,
    /// Heading-trail context for the prefix.
    pub section_path: String,
    /// Canonical, immutable citation text (the body of `embedding_text`).
    pub text: String,
    /// Leading block type (`paragraph`/`heading`/`code`/`table`/…) for the
    /// kind-awareness gate (sub-decision b). `None` when unknown.
    pub block_type: Option<String>,
}

/// A single chunk's enrichment write (M4 Phase-3 Step 4): the composed
/// `embedding_text` and, for a representative parent row, the structural-map
/// `enrichment` JSON. Applied via [`NotebookRepo::write_chunk_enrichment`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkEnrichmentUpdate {
    /// The chunk to update.
    pub chunk_id: String,
    /// The composed contextual `embedding_text` (prefix + canonical body).
    pub embedding_text: String,
    /// `Some(map_json)` to also write the per-doc structural map onto this row
    /// (parent rows only); `None` leaves `chunks.enrichment` unchanged.
    pub enrichment_json: Option<String>,
}

/// A chunk row projected for the Step-5 re-embed flip: the id, level, and the
/// text the re-embed embeds — `COALESCE(embedding_text, text)` so enriched chunks
/// embed their contextual text and any chunk the worker did not touch (NULL
/// `embedding_text`) falls back to the canonical body.
#[derive(Debug, Clone, PartialEq, Eq, sqlx::FromRow)]
pub struct ReembedChunk {
    /// Chunk primary key (becomes the Lance `chunk_id`).
    pub id: String,
    /// Chunk level (`0` parent, `1` child, `2` summary).
    pub level: i32,
    /// `COALESCE(embedding_text, text)` — the text to embed.
    pub embed_text: String,
}

/// A chunk row projected for the dev/QA Embeddings Inspector (M4).
///
/// A read-only, IPC-serializable view of a chunk's identity, position in the
/// parent/child hierarchy, and the metadata the inspector renders: the canonical
/// citation `text`, the block type, the character span, the `source_anchor` JSON,
/// and the contextual `embedding_text` (NULL until the enrichment pass runs).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, sqlx::FromRow)]
pub struct InspectorChunk {
    /// Chunk primary key.
    pub id: String,
    /// `None` for level-0 parents (and the level-2 summary); `Some(parent.id)`
    /// for child rows.
    pub parent_id: Option<String>,
    /// `"parent"` (level 0) / `"child"` (level 1) / `"summary"` (level 2).
    pub kind: String,
    /// `0` parent, `1` child, `2` summary.
    pub level: i32,
    /// Heading-trail context for the chunk.
    pub section_path: String,
    /// Canonical, immutable citation text.
    pub text: String,
    /// Leading block type (`paragraph`/`heading`/`code`/`table`/…). `None` when
    /// unknown.
    pub block_type: Option<String>,
    /// Character offset of the chunk's start in the source. `None` when unknown.
    pub char_start: Option<i64>,
    /// Character offset of the chunk's end in the source. `None` when unknown.
    pub char_end: Option<i64>,
    /// JSON source-anchor payload (page/coords for click-to-open). `None` until
    /// extraction records one.
    pub source_anchor: Option<String>,
    /// The contextual text the embedder embeds (context-prefix + canonical body).
    /// `None` until the enrichment pass populates it.
    pub embedding_text: Option<String>,
}

/// Per-(model, dim) embedding-index stats for a notebook, returned to the dev/QA
/// Embeddings Inspector header (M4).
///
/// One row per ACTIVE `embedding_index` registry entry. A notebook may have
/// MULTIPLE active rows (the partial-unique `uq_embidx_active` keys on
/// `(notebook_id, model, dim)`), so the inspector header renders one badge per
/// row rather than assuming a single active index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, sqlx::FromRow)]
pub struct EmbeddingStats {
    /// The embedding model id (e.g. the nomic model).
    pub model: String,
    /// The embedding dimensionality.
    pub dim: i64,
    /// The registry status (always `"active"` for inspector rows).
    pub status: String,
}

/// A notebook row, returned across the IPC boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, sqlx::FromRow)]
pub struct Notebook {
    /// UUIDv7 primary key, stored as TEXT.
    pub id: NotebookId,
    /// Display title.
    pub title: String,
    /// Optional free-text description captured during onboarding. Write-only in
    /// M1 (no read/edit surface yet); M3 extends it. `None` when unset.
    pub description: Option<String>,
    /// Optional focus mode (`"research"` | `"coding"` | `"notes"`) captured
    /// during onboarding. Write-only in M1; M3 extends it. `None` when unset.
    pub focus_mode: Option<String>,
    /// Embedding model id this notebook is indexed with (M4 Phase 4b). A stable
    /// registry id (`embedder::registry`). `None` on pre-migration rows; the
    /// read path resolves `None` to [`DEFAULT_EMBED_MODEL_ID`] via the registry.
    /// New notebooks are stamped with the current global default at create time.
    pub embedding_model: Option<String>,
    /// Embedding *backend* this notebook is indexed with (M4 Phase 4b-B):
    /// `"fastembed"` | `"ollama"`. `None` on pre-migration rows; the read path
    /// resolves `None`/empty/unknown to the global default backend via
    /// [`crate::embedder::EmbeddingBackend::from_opt_str`]. New notebooks are
    /// stamped with the current global default at create time.
    pub embedding_backend: Option<String>,
    /// RFC3339 creation timestamp.
    pub created_at: String,
    /// RFC3339 last-update timestamp.
    pub updated_at: String,
    /// RFC3339 soft-delete timestamp, or `None` if live.
    pub trashed_at: Option<String>,
}

/// A notebook list response with its maintained source count.
///
/// This is the API/response shape (distinct from the pure [`Notebook`] row
/// struct), used by `list_with_counts` / `list_trashed_with_counts`. It does NOT
/// derive `sqlx::FromRow`: `source_count` is a `COUNT(...)` aggregate that has no
/// column on the `notebooks` table, so the list queries map rows manually.
///
/// `#[serde(flatten)]` hoists the inner `Notebook`'s fields to the top level, so
/// the wire shape is `{id, title, description, focus_mode, created_at,
/// updated_at, trashed_at, source_count}` — the TS `NotebookSummary` mirror.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NotebookSummary {
    /// The underlying notebook row (its fields are flattened into the response).
    #[serde(flatten)]
    pub notebook: Notebook,
    /// Number of sources belonging to this notebook (`COUNT` of `sources`).
    pub source_count: i64,
}

/// SHA-256 of the file at `path`, computed by streaming 64 KiB chunks so the
/// entire file is never held in memory at once. Returns the lowercase hex digest.
fn sha256_file(path: &Path) -> std::io::Result<String> {
    const CHUNK: usize = 64 * 1024;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(crate::hex_encode(&hasher.finalize()))
}

/// SHA-256 of an in-memory byte slice. Returns the lowercase hex digest.
///
/// The in-memory counterpart to [`sha256_file`] (issue #100): used by the
/// text/URL/onboarding add paths whose content is already resident in memory
/// (pasted text bytes, a normalized URL string, or a file-path string) so no
/// file I/O is needed to compute the dedup key.
fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    crate::hex_encode(&hasher.finalize())
}

/// Validates and normalizes a user-supplied notebook title.
///
/// Trims surrounding whitespace, rejects empty/whitespace-only input, and caps
/// length at [`MAX_TITLE_LEN`] characters. Returns the trimmed, owned title.
fn validate_title(title: &str) -> Result<String, LensError> {
    let trimmed = title.trim();
    if trimmed.is_empty() {
        return Err(LensError::Validation(
            "notebook title must not be empty".into(),
        ));
    }
    if trimmed.chars().count() > MAX_TITLE_LEN {
        return Err(LensError::Validation(format!(
            "notebook title must be at most {MAX_TITLE_LEN} characters"
        )));
    }
    Ok(trimmed.to_string())
}

/// Repository over the `notebooks` table. Borrows a pool; holds no state.
///
/// Construct one per call via [`NotebookRepo::new`]; it's a zero-cost handle.
pub struct NotebookRepo<'a> {
    pool: &'a SqlitePool,
}

impl<'a> NotebookRepo<'a> {
    /// Wraps a borrowed connection pool.
    pub fn new(pool: &'a SqlitePool) -> Self {
        Self { pool }
    }

    /// Lists all live (non-trashed) notebooks, newest first.
    pub async fn list(&self) -> Result<Vec<Notebook>, LensError> {
        let rows = sqlx::query_as::<_, Notebook>(
            "SELECT id, title, description, focus_mode, embedding_model, embedding_backend, \
                    created_at, updated_at, trashed_at \
             FROM notebooks WHERE trashed_at IS NULL ORDER BY created_at DESC",
        )
        .fetch_all(self.pool)
        .await?;
        Ok(rows)
    }

    /// Lists all live (non-trashed) notebooks with their source counts, newest
    /// `created_at` first.
    ///
    /// Uses a `LEFT JOIN` + `GROUP BY` so notebooks with zero sources still
    /// appear (with `source_count = 0`). Maps each row manually because
    /// `NotebookSummary::source_count` is a `COUNT(...)` aggregate with no
    /// backing column, so `query_as::<_, Notebook>` cannot populate it.
    pub async fn list_with_counts(&self) -> Result<Vec<NotebookSummary>, LensError> {
        self.list_summaries(
            "SELECT n.id, n.title, n.description, n.focus_mode, n.embedding_model, \
                    n.embedding_backend, n.created_at, n.updated_at, n.trashed_at, \
                    COALESCE(COUNT(s.id), 0) AS source_count \
             FROM notebooks n \
             LEFT JOIN sources s ON s.notebook_id = n.id \
             WHERE n.trashed_at IS NULL \
             GROUP BY n.id \
             ORDER BY n.created_at DESC",
        )
        .await
    }

    /// Lists all trashed notebooks with their source counts, newest
    /// `trashed_at` first.
    pub async fn list_trashed_with_counts(&self) -> Result<Vec<NotebookSummary>, LensError> {
        self.list_summaries(
            "SELECT n.id, n.title, n.description, n.focus_mode, n.embedding_model, \
                    n.embedding_backend, n.created_at, n.updated_at, n.trashed_at, \
                    COALESCE(COUNT(s.id), 0) AS source_count \
             FROM notebooks n \
             LEFT JOIN sources s ON s.notebook_id = n.id \
             WHERE n.trashed_at IS NOT NULL \
             GROUP BY n.id \
             ORDER BY n.trashed_at DESC",
        )
        .await
    }

    /// Runs a `NotebookSummary` list query and maps each row by column name.
    ///
    /// Shared by [`list_with_counts`](Self::list_with_counts) and
    /// [`list_trashed_with_counts`](Self::list_trashed_with_counts), which differ
    /// only in their `WHERE`/`ORDER BY`. The `SELECT` projection must expose the
    /// columns `id, title, description, focus_mode, embedding_model,
    /// embedding_backend, created_at, updated_at, trashed_at, source_count` in any
    /// order.
    async fn list_summaries(&self, query: &str) -> Result<Vec<NotebookSummary>, LensError> {
        use sqlx::Row;
        let rows = sqlx::query(query).fetch_all(self.pool).await?;
        let summaries = rows
            .into_iter()
            .map(|row| {
                Ok(NotebookSummary {
                    notebook: Notebook {
                        id: NotebookId::from(row.try_get::<String, _>("id")?),
                        title: row.try_get("title")?,
                        description: row.try_get("description")?,
                        focus_mode: row.try_get("focus_mode")?,
                        embedding_model: row.try_get("embedding_model")?,
                        embedding_backend: row.try_get("embedding_backend")?,
                        created_at: row.try_get("created_at")?,
                        updated_at: row.try_get("updated_at")?,
                        trashed_at: row.try_get("trashed_at")?,
                    },
                    source_count: row.try_get("source_count")?,
                })
            })
            .collect::<Result<Vec<_>, LensError>>()?;
        Ok(summaries)
    }

    /// Creates a notebook with a freshly-minted UUIDv7 id and returns it.
    ///
    /// The title is trimmed and validated (non-empty, length-capped).
    /// `description` and `focus_mode` are optional onboarding fields persisted
    /// verbatim (write-only in M1); pass `None` to leave them unset.
    ///
    /// `embedding_model` / `embedding_backend` are the canonical coordinate the
    /// notebook is pinned to from birth — the caller resolves these from the
    /// app-wide global default ([`crate::config::AppConfig::embedding_model`] /
    /// [`crate::config::AppConfig::embedding_backend`]) so that setting a new
    /// default in Settings is adopted by NEW notebooks (M4 Phase 4b-B, AC7). They
    /// are stamped verbatim; the resolver upstream already collapses an empty /
    /// unset config to the registry/enum default, so passing the resolved values
    /// preserves the prior behavior when config is unset. The read path still
    /// tolerates a NULL on pre-migration rows by resolving to the default.
    pub async fn create(
        &self,
        title: &str,
        description: Option<&str>,
        focus_mode: Option<&str>,
        embedding_model: &str,
        embedding_backend: &str,
    ) -> Result<Notebook, LensError> {
        let title = validate_title(title)?;
        let description = description.map(str::to_string);
        let focus_mode = focus_mode.map(str::to_string);
        let embedding_model = embedding_model.to_string();
        let embedding_backend = embedding_backend.to_string();
        let id = NotebookId::new();
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO notebooks \
                 (id, title, description, focus_mode, embedding_model, embedding_backend, \
                  created_at, updated_at, trashed_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
        )
        .bind(&id)
        .bind(&title)
        .bind(&description)
        .bind(&focus_mode)
        .bind(&embedding_model)
        .bind(&embedding_backend)
        .bind(&now)
        .bind(&now)
        .execute(self.pool)
        .await?;
        Ok(Notebook {
            id,
            title,
            description,
            focus_mode,
            embedding_model: Some(embedding_model),
            embedding_backend: Some(embedding_backend),
            created_at: now.clone(),
            updated_at: now,
            trashed_at: None,
        })
    }

    /// Renames a notebook, bumping `updated_at`. The title is validated.
    ///
    /// The `AND trashed_at IS NULL` guard is defense-in-depth: the UI never
    /// exposes renaming a trashed notebook, but the clause prevents misuse via a
    /// direct IPC call.
    pub async fn rename(&self, id: &NotebookId, title: &str) -> Result<(), LensError> {
        let title = validate_title(title)?;
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE notebooks SET title = ?, updated_at = ? WHERE id = ? AND trashed_at IS NULL",
        )
        .bind(&title)
        .bind(&now)
        .bind(id)
        .execute(self.pool)
        .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!("no notebook with id {id}")));
        }
        Ok(())
    }

    /// Soft-deletes a notebook: an alias for [`trash`](Self::trash).
    ///
    /// Historically this was a hard `DELETE`; M3 reframes deletion as a recoverable
    /// soft-delete via `trashed_at`. [`purge`](Self::purge) is now the sole
    /// hard-delete path.
    #[deprecated(note = "Use trash() directly; kept for backward compat")]
    pub async fn delete(&self, id: &NotebookId) -> Result<(), LensError> {
        self.trash(id).await
    }

    /// Soft-deletes a notebook: sets `trashed_at` to now and bumps `updated_at`.
    ///
    /// Only affects live notebooks (`trashed_at IS NULL`); trashing an already
    /// trashed or unknown notebook affects 0 rows and returns a validation error.
    pub async fn trash(&self, id: &NotebookId) -> Result<(), LensError> {
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE notebooks SET trashed_at = ?, updated_at = ? \
             WHERE id = ? AND trashed_at IS NULL",
        )
        .bind(&now)
        .bind(&now)
        .bind(id)
        .execute(self.pool)
        .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!(
                "no live notebook with id {id}"
            )));
        }
        Ok(())
    }

    /// Restores a trashed notebook: clears `trashed_at` and bumps `updated_at`.
    ///
    /// Only affects trashed notebooks (`trashed_at IS NOT NULL`); restoring a live
    /// or unknown notebook affects 0 rows and returns a validation error.
    pub async fn restore(&self, id: &NotebookId) -> Result<(), LensError> {
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE notebooks SET trashed_at = NULL, updated_at = ? \
             WHERE id = ? AND trashed_at IS NOT NULL",
        )
        .bind(&now)
        .bind(id)
        .execute(self.pool)
        .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!(
                "no trashed notebook with id {id}"
            )));
        }
        Ok(())
    }

    /// Permanently deletes a notebook. Child rows cascade via `ON DELETE CASCADE`.
    ///
    /// This is the only hard-delete path (used by "Delete forever"). Only affects
    /// trashed notebooks (`trashed_at IS NOT NULL`); purging a live or unknown
    /// notebook affects 0 rows and returns a validation error, so a live notebook
    /// can never be hard-deleted without first being trashed.
    pub async fn purge(&self, id: &NotebookId) -> Result<(), LensError> {
        let result = sqlx::query("DELETE FROM notebooks WHERE id = ? AND trashed_at IS NOT NULL")
            .bind(id)
            .execute(self.pool)
            .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!(
                "no trashed notebook with id {id}"
            )));
        }
        Ok(())
    }

    /// Inserts a file source *record* (M1 onboarding "Add sources").
    ///
    /// The row is inert: `kind = "file"`, `status = "pending"`, `selected = 1`,
    /// `locator` = the absolute file path. NO parsing/embedding/chunking happens
    /// here — M4 ingestion picks up the `pending` row later. Returns the inserted
    /// [`Source`].
    pub async fn add_source(
        &self,
        notebook_id: &NotebookId,
        title: &str,
        locator: &str,
    ) -> Result<Source, LensError> {
        let id = Uuid::now_v7().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, created_at) \
             VALUES (?, ?, 'file', ?, ?, ?, 1, ?)",
        )
        .bind(&id)
        .bind(notebook_id)
        .bind(title)
        .bind(SourceStatus::Pending.as_str())
        .bind(locator)
        .bind(&now)
        .execute(self.pool)
        .await?;
        Ok(Source {
            id,
            notebook_id: notebook_id.to_string(),
            kind: "file".to_string(),
            title: title.to_string(),
            status: SourceStatus::Pending.as_str().to_string(),
            locator: locator.to_string(),
            selected: 1,
            token_count: None,
            content_hash: None,
            raw_content_hash: None,
            created_at: now,
            trashed_at: None,
            enrichment_status: None,
            enrichment_meta: None,
        })
    }

    /// Inserts a managed text/markdown source for M4 ingestion.
    ///
    /// Writes the verbatim `text` to a managed file
    /// `{data_dir}/sources/{id}.{ext}` (ext from `kind`: `text` → `txt`,
    /// `markdown` → `md`), then inserts a `sources` row with `kind ∈
    /// {"text","markdown"}`, `status = "queued"`, `selected = 1`, and `locator`
    /// = that managed file path. Returns an [`AddSourceOutcome`] (`token_count`
    /// and `content_hash` are `NULL` until ingestion populates them).
    ///
    /// **Content dedup (issue #100):** the raw text bytes are hashed (SHA-256)
    /// and stored as `raw_content_hash`. If a live (non-trashed) source in the
    /// same notebook already carries that `raw_content_hash`, this returns the
    /// existing row (`was_existing = true`) WITHOUT writing the managed file or
    /// inserting a new row. The partial unique index on `(notebook_id,
    /// raw_content_hash) WHERE trashed_at IS NULL AND raw_content_hash IS NOT
    /// NULL` is the authoritative guard: the upfront `SELECT` is a fast-path
    /// optimisation and an `INSERT … ON CONFLICT DO NOTHING` resolves any race
    /// (the loser reclaims its managed file and re-queries the winner).
    pub async fn add_text_source(
        &self,
        data_dir: &Path,
        notebook_id: &NotebookId,
        title: &str,
        text: &str,
        kind: &str,
        max_source_bytes: usize,
    ) -> Result<AddSourceOutcome, LensError> {
        // OOM guard: reject an oversized paste before writing it to disk and
        // queueing it for ingest (the ingest pipeline enforces the same cap after
        // reading any file path). `max_source_bytes` is the configured cap
        // (issue #71), resolved by the engine wrapper from `AppConfig.max_source_mb`
        // via [`crate::ingest::resolve_max_source_bytes`].
        if text.len() > max_source_bytes {
            return Err(LensError::Validation(format!(
                "source text is {} bytes, exceeding the {max_source_bytes}-byte limit",
                text.len()
            )));
        }
        // Parse the boundary string into the enum and dispatch the managed
        // extension via an exhaustive match. Only the text-like kinds are valid
        // for a pasted-text source; the derived kinds are rejected here.
        let ext = match SourceKind::from_kind_str(kind)? {
            SourceKind::Text => "txt",
            SourceKind::Markdown => "md",
            other => {
                return Err(LensError::Validation(format!(
                    "unknown text source kind: {:?}; expected \"text\" or \"markdown\"",
                    other.as_str()
                )));
            }
        };
        // Hash the raw text bytes as the dedup key (issue #100). A dedup hit
        // returns immediately (no managed file written); a miss writes the file.
        let raw_content_hash = sha256_bytes(text.as_bytes());

        // SELECT projection reused by the fast-path check and the race re-query.
        const SOURCE_SELECT: &str = "SELECT id, notebook_id, kind, title, status, locator, \
             selected, token_count, content_hash, raw_content_hash, created_at, trashed_at, \
             enrichment_status, enrichment_meta \
             FROM sources WHERE notebook_id = ? AND raw_content_hash = ? AND trashed_at IS NULL LIMIT 1";

        // Fast-path: a live source with identical raw content already exists.
        // Return it WITHOUT writing the managed file or inserting a row.
        if let Some(dup) = sqlx::query_as::<_, Source>(SOURCE_SELECT)
            .bind(notebook_id)
            .bind(&raw_content_hash)
            .fetch_optional(self.pool)
            .await?
        {
            tracing::info!(
                notebook_id = %notebook_id,
                raw_content_hash = %raw_content_hash,
                source_id = %dup.id,
                "duplicate text detected at add time — returning existing source"
            );
            return Ok(AddSourceOutcome {
                source: dup,
                was_existing: true,
            });
        }

        let id = Uuid::now_v7().to_string();
        let now = chrono::Utc::now().to_rfc3339();

        // Write the canonical text to the managed sources dir so `locator` stays
        // a path (no new migration / no inline content column).
        let sources_dir = data_dir.join("sources");
        std::fs::create_dir_all(&sources_dir)
            .map_err(|e| LensError::Io(format!("{}: {e}", sources_dir.display())))?;
        let path = sources_dir.join(format!("{id}.{ext}"));
        std::fs::write(&path, text)
            .map_err(|e| LensError::Io(format!("{}: {e}", path.display())))?;
        let locator = path.display().to_string();

        // Authoritative dedup: the partial unique index makes a racing duplicate
        // insert a no-op. `ON CONFLICT DO NOTHING` (untargeted) lets SQLite pick
        // the matching partial index, so no index-target ambiguity arises.
        let result = sqlx::query(
            "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
             created_at, raw_content_hash) \
             VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?) \
             ON CONFLICT DO NOTHING",
        )
        .bind(&id)
        .bind(notebook_id)
        .bind(kind)
        .bind(title)
        .bind(SourceStatus::Queued.as_str())
        .bind(&locator)
        .bind(&now)
        .bind(&raw_content_hash)
        .execute(self.pool)
        .await?;

        if result.rows_affected() == 0 {
            // Lost the race: another add inserted the winning row first. Reclaim
            // the managed file we just wrote and return the winner as existing.
            let _ = std::fs::remove_file(&path);
            let winner = sqlx::query_as::<_, Source>(SOURCE_SELECT)
                .bind(notebook_id)
                .bind(&raw_content_hash)
                .fetch_one(self.pool)
                .await?;
            tracing::info!(
                notebook_id = %notebook_id,
                raw_content_hash = %raw_content_hash,
                source_id = %winner.id,
                "duplicate text detected via ON CONFLICT race — returning existing source"
            );
            return Ok(AddSourceOutcome {
                source: winner,
                was_existing: true,
            });
        }

        Ok(AddSourceOutcome {
            source: Source {
                id,
                notebook_id: notebook_id.to_string(),
                kind: kind.to_string(),
                title: title.to_string(),
                status: SourceStatus::Queued.as_str().to_string(),
                locator,
                selected: 1,
                token_count: None,
                content_hash: None,
                raw_content_hash: Some(raw_content_hash),
                created_at: now,
                trashed_at: None,
                enrichment_status: None,
                enrichment_meta: None,
            },
            was_existing: false,
        })
    }

    /// Inserts a managed local-file source (PDF/DOCX/text/markdown) for M4
    /// ingestion.
    ///
    /// The product entry point for ingesting a local file (the M1
    /// [`add_source`](Self::add_source) path stays inert; this one queues for
    /// ingest). Detects `kind` from the file EXTENSION — `.pdf` → `"pdf"`,
    /// `.docx` → `"docx"`, `.txt` → `"text"`, `.md`/`.markdown` → `"markdown"`;
    /// any other (or missing) extension is rejected with [`LensError::Validation`].
    /// COPIES the file into managed storage `{data_dir}/sources/{id}.{ext}` (so
    /// the `locator` is a managed path — consistent with
    /// [`add_text_source`](Self::add_text_source) — and a purge of the source
    /// reclaims the copy via `remove_managed_source_file`). Inserts a `sources`
    /// row with `status = "queued"`, `selected = 1`. `title` defaults to the
    /// source file's name when not supplied.
    ///
    /// **Content dedup (issue #96):** the file is hashed (SHA-256) by streaming
    /// 64 KiB chunks — the entire file is never buffered in memory — and the
    /// digest stored as `raw_content_hash`. If a live (non-trashed) source in
    /// the same notebook already carries that `raw_content_hash`, this returns
    /// the existing row (`was_existing = true`) WITHOUT copying the file or
    /// inserting a new row. A partial unique index on `(notebook_id,
    /// raw_content_hash) WHERE trashed_at IS NULL AND raw_content_hash IS NOT
    /// NULL` is the authoritative guard: the upfront `SELECT`
    /// is a fast-path optimisation, and an `INSERT … ON CONFLICT DO NOTHING`
    /// resolves any race (the loser cleans up its copy and re-queries the winner).
    /// `content_hash` stays `NULL` at add time (populated later by ingestion) so
    /// the re-ingest no-op is unaffected. Returns an [`AddSourceOutcome`].
    pub async fn add_file_source(
        &self,
        data_dir: &Path,
        notebook_id: &NotebookId,
        src_path: &Path,
        title: Option<&str>,
    ) -> Result<AddSourceOutcome, LensError> {
        // Detect kind + canonical extension from the source file extension
        // (case-insensitive). An unknown / missing extension is a clear
        // validation error rather than a silently-mis-ingested source.
        let ext_lower = src_path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        let (kind, ext) = match ext_lower.as_deref() {
            Some("pdf") => (SourceKind::Pdf.as_str(), "pdf"),
            Some("docx") => (SourceKind::Docx.as_str(), "docx"),
            Some("txt") => (SourceKind::Text.as_str(), "txt"),
            Some("md") | Some("markdown") | Some("mdx") => (SourceKind::Markdown.as_str(), "md"),
            // Structured formats (M4 Phase 2.5c). The canonical managed-file
            // extension preserves the original so the on-disk locator round-trips.
            Some("json") => (SourceKind::Json.as_str(), "json"),
            Some("jsonl") => (SourceKind::Jsonl.as_str(), "jsonl"),
            Some("ndjson") => (SourceKind::Jsonl.as_str(), "ndjson"),
            Some("yaml") => (SourceKind::Yaml.as_str(), "yaml"),
            Some("yml") => (SourceKind::Yaml.as_str(), "yml"),
            Some("xml") => (SourceKind::Xml.as_str(), "xml"),
            // Office/binary formats (M4 issue #77). The canonical managed-file
            // extension preserves the original so the on-disk locator round-trips.
            Some("rtf") => (SourceKind::Rtf.as_str(), "rtf"),
            Some("odt") => (SourceKind::Odt.as_str(), "odt"),
            Some("epub") => (SourceKind::Epub.as_str(), "epub"),
            // Tabular formats (M4 issue #76). The canonical managed-file extension
            // preserves the original so the on-disk locator round-trips.
            Some("xlsx") => (SourceKind::Xlsx.as_str(), "xlsx"),
            Some("xls") => (SourceKind::Xls.as_str(), "xls"),
            Some("csv") => (SourceKind::Csv.as_str(), "csv"),
            other => {
                return Err(LensError::Validation(format!(
                    "unsupported file extension {other:?} for {}; expected one of \
                     \".pdf\", \".docx\", \".txt\", \".md\", \".markdown\", \".mdx\", \".json\", \
                     \".jsonl\", \".ndjson\", \".yaml\", \".yml\", \".xml\", \".rtf\", \".odt\", \
                     \".epub\", \".xlsx\", \".xls\", \".csv\"",
                    src_path.display()
                )));
            }
        };

        // Derive a default title from the file name when none is supplied.
        let title = match title {
            Some(t) => t.to_string(),
            None => src_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("Untitled")
                .to_string(),
        };

        // Hash the raw file bytes by streaming 64 KiB chunks — the entire file
        // is never held in memory. A dedup hit returns immediately (no copy);
        // a miss copies via std::fs::copy (kernel-streamed, no user-space buffer).
        let file_hash = sha256_file(src_path)
            .map_err(|e| LensError::Io(format!("hash {}: {e}", src_path.display())))?;

        // SELECT projection reused by the fast-path check and the race re-query.
        const SOURCE_SELECT: &str = "SELECT id, notebook_id, kind, title, status, locator, \
             selected, token_count, content_hash, raw_content_hash, created_at, trashed_at, \
             enrichment_status, enrichment_meta \
             FROM sources WHERE notebook_id = ? AND raw_content_hash = ? AND trashed_at IS NULL LIMIT 1";

        // Fast-path: a live source with identical raw content already exists.
        // Return it WITHOUT copying the file or inserting a row.
        if let Some(dup) = sqlx::query_as::<_, Source>(SOURCE_SELECT)
            .bind(notebook_id)
            .bind(&file_hash)
            .fetch_optional(self.pool)
            .await?
        {
            tracing::info!(
                notebook_id = %notebook_id,
                raw_content_hash = %file_hash,
                source_id = %dup.id,
                "duplicate file detected at add time — returning existing source"
            );
            return Ok(AddSourceOutcome {
                source: dup,
                was_existing: true,
            });
        }

        let id = Uuid::now_v7().to_string();
        let now = chrono::Utc::now().to_rfc3339();

        // Copy the source file into managed storage so `locator` is a managed
        // path (purge reclaims it; the ingest read path matches the paste/text
        // path). std::fs::copy uses kernel-level transfer — no user-space buffer.
        let sources_dir = data_dir.join("sources");
        std::fs::create_dir_all(&sources_dir)
            .map_err(|e| LensError::Io(format!("{}: {e}", sources_dir.display())))?;
        let dest = sources_dir.join(format!("{id}.{ext}"));
        std::fs::copy(src_path, &dest)
            .map_err(|e| LensError::Io(format!("copy {}: {e}", dest.display())))?;
        let locator = dest.display().to_string();

        // Authoritative dedup: the partial unique index makes a racing duplicate
        // insert a no-op. `ON CONFLICT DO NOTHING` (untargeted) lets SQLite pick
        // the matching partial index, so no index-target ambiguity arises.
        let result = sqlx::query(
            "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
             created_at, raw_content_hash) \
             VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?) \
             ON CONFLICT DO NOTHING",
        )
        .bind(&id)
        .bind(notebook_id)
        .bind(kind)
        .bind(&title)
        .bind(SourceStatus::Queued.as_str())
        .bind(&locator)
        .bind(&now)
        .bind(&file_hash)
        .execute(self.pool)
        .await?;

        if result.rows_affected() == 0 {
            // Lost the race: another add inserted the winning row first. Reclaim
            // the copy we just wrote and return the winner as an existing source.
            let _ = std::fs::remove_file(&dest);
            let winner = sqlx::query_as::<_, Source>(SOURCE_SELECT)
                .bind(notebook_id)
                .bind(&file_hash)
                .fetch_one(self.pool)
                .await?;
            tracing::info!(
                notebook_id = %notebook_id,
                raw_content_hash = %file_hash,
                source_id = %winner.id,
                "duplicate file detected via ON CONFLICT race — returning existing source"
            );
            return Ok(AddSourceOutcome {
                source: winner,
                was_existing: true,
            });
        }

        Ok(AddSourceOutcome {
            source: Source {
                id,
                notebook_id: notebook_id.to_string(),
                kind: kind.to_string(),
                title,
                status: SourceStatus::Queued.as_str().to_string(),
                locator,
                selected: 1,
                token_count: None,
                content_hash: None,
                raw_content_hash: Some(file_hash),
                created_at: now,
                trashed_at: None,
                enrichment_status: None,
                enrichment_meta: None,
            },
            was_existing: false,
        })
    }

    /// Inserts a URL source for M4 ingestion.
    ///
    /// Inserts a `sources` row with `kind = "url"`, `status = "queued"`,
    /// `selected = 1`, and `locator` = the verbatim URL string. No file is written
    /// to disk — the locator IS the URL, and the ingest pipeline fetches the HTML
    /// at ingest time. Returns an [`AddSourceOutcome`] (`token_count` and
    /// `content_hash` are `NULL` until ingestion populates them).
    ///
    /// **Content dedup (issue #100):** the URL is moderately normalized via
    /// [`normalize_url`] (lowercase scheme+host, strip default ports, drop
    /// fragment, strip one trailing slash — query order/content preserved) and
    /// the SHA-256 of that normalized form stored as `raw_content_hash`. The
    /// `locator` keeps the verbatim first-added URL for display; the hash catches
    /// equivalent URLs. A live source in the same notebook with a matching
    /// `raw_content_hash` short-circuits to a dedup hit (`was_existing = true`).
    /// The partial unique index is the authoritative guard; `SELECT` is the
    /// fast path and `INSERT … ON CONFLICT DO NOTHING` resolves any race.
    pub async fn add_url_source(
        &self,
        notebook_id: &NotebookId,
        title: &str,
        url: &str,
    ) -> Result<AddSourceOutcome, LensError> {
        // Hash the NORMALIZED URL (not the verbatim string) as the dedup key so
        // case/port/fragment/trailing-slash variants of the same URL collide.
        let raw_content_hash = sha256_bytes(normalize_url(url)?.as_bytes());

        // SELECT projection reused by the fast-path check and the race re-query.
        const SOURCE_SELECT: &str = "SELECT id, notebook_id, kind, title, status, locator, \
             selected, token_count, content_hash, raw_content_hash, created_at, trashed_at, \
             enrichment_status, enrichment_meta \
             FROM sources WHERE notebook_id = ? AND raw_content_hash = ? AND trashed_at IS NULL LIMIT 1";

        // Fast-path: a live source with an equivalent URL already exists.
        if let Some(dup) = sqlx::query_as::<_, Source>(SOURCE_SELECT)
            .bind(notebook_id)
            .bind(&raw_content_hash)
            .fetch_optional(self.pool)
            .await?
        {
            tracing::info!(
                notebook_id = %notebook_id,
                raw_content_hash = %raw_content_hash,
                source_id = %dup.id,
                "duplicate URL detected at add time — returning existing source"
            );
            return Ok(AddSourceOutcome {
                source: dup,
                was_existing: true,
            });
        }

        let id = Uuid::now_v7().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        // Authoritative dedup: the partial unique index makes a racing duplicate
        // insert a no-op via `ON CONFLICT DO NOTHING`.
        let result = sqlx::query(
            "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
             created_at, raw_content_hash) \
             VALUES (?, ?, 'url', ?, ?, ?, 1, ?, ?) \
             ON CONFLICT DO NOTHING",
        )
        .bind(&id)
        .bind(notebook_id)
        .bind(title)
        .bind(SourceStatus::Queued.as_str())
        .bind(url)
        .bind(&now)
        .bind(&raw_content_hash)
        .execute(self.pool)
        .await?;

        if result.rows_affected() == 0 {
            // Lost the race: return the winner as an existing source.
            let winner = sqlx::query_as::<_, Source>(SOURCE_SELECT)
                .bind(notebook_id)
                .bind(&raw_content_hash)
                .fetch_one(self.pool)
                .await?;
            tracing::info!(
                notebook_id = %notebook_id,
                raw_content_hash = %raw_content_hash,
                source_id = %winner.id,
                "duplicate URL detected via ON CONFLICT race — returning existing source"
            );
            return Ok(AddSourceOutcome {
                source: winner,
                was_existing: true,
            });
        }

        Ok(AddSourceOutcome {
            source: Source {
                id,
                notebook_id: notebook_id.to_string(),
                kind: SourceKind::Url.as_str().to_string(),
                title: title.to_string(),
                status: SourceStatus::Queued.as_str().to_string(),
                locator: url.to_string(),
                selected: 1,
                token_count: None,
                content_hash: None,
                raw_content_hash: Some(raw_content_hash),
                created_at: now,
                trashed_at: None,
                enrichment_status: None,
                enrichment_meta: None,
            },
            was_existing: false,
        })
    }

    /// Soft-deletes a source: sets `trashed_at` to now.
    ///
    /// Only affects live sources (`trashed_at IS NULL`); trashing an already
    /// trashed or unknown source affects 0 rows and returns a validation error.
    pub async fn trash_source(&self, id: &str) -> Result<(), LensError> {
        let now = chrono::Utc::now().to_rfc3339();
        let result =
            sqlx::query("UPDATE sources SET trashed_at = ? WHERE id = ? AND trashed_at IS NULL")
                .bind(&now)
                .bind(id)
                .execute(self.pool)
                .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!(
                "no live source with id {id}"
            )));
        }
        Ok(())
    }

    /// Restores a trashed source: clears `trashed_at`.
    ///
    /// Only affects trashed sources (`trashed_at IS NOT NULL`); restoring a live
    /// or unknown source affects 0 rows and returns a validation error.
    pub async fn restore_source(&self, id: &str) -> Result<(), LensError> {
        let result = sqlx::query(
            "UPDATE sources SET trashed_at = NULL WHERE id = ? AND trashed_at IS NOT NULL",
        )
        .bind(id)
        .execute(self.pool)
        .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!(
                "no trashed source with id {id}"
            )));
        }
        Ok(())
    }

    /// Permanently deletes a source row by id. Child `chunks` rows cascade via
    /// `ON DELETE CASCADE`.
    ///
    /// Callers are responsible for removing any associated Lance vectors before
    /// calling this (Lance before SQLite ordering). Only affects trashed sources
    /// (`trashed_at IS NOT NULL`); purging a live or unknown source affects 0 rows
    /// and returns a validation error, so a live source can never be hard-deleted
    /// without first being trashed (mirroring [`purge`](Self::purge)).
    pub async fn purge_source(&self, id: &str) -> Result<(), LensError> {
        let result = sqlx::query("DELETE FROM sources WHERE id = ? AND trashed_at IS NOT NULL")
            .bind(id)
            .execute(self.pool)
            .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!(
                "no trashed source with id {id}"
            )));
        }
        Ok(())
    }

    /// Toggles a source's `selected` flag (persisted). Errors if no row matches.
    pub async fn set_source_selected(&self, id: &str, selected: bool) -> Result<(), LensError> {
        let result = sqlx::query("UPDATE sources SET selected = ? WHERE id = ?")
            .bind(selected as i64)
            .bind(id)
            .execute(self.pool)
            .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!("no source with id {id}")));
        }
        Ok(())
    }

    /// Sets a source's ingestion `status` (e.g. `queued`/`parsing`/`embedding`/
    /// `indexed`/`error`). Errors if no row matches.
    pub async fn update_source_status(&self, id: &str, status: &str) -> Result<(), LensError> {
        let result = sqlx::query("UPDATE sources SET status = ? WHERE id = ?")
            .bind(status)
            .bind(id)
            .execute(self.pool)
            .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!("no source with id {id}")));
        }
        Ok(())
    }

    /// Sets a source's enrichment lifecycle status (`sources.enrichment_status`),
    /// SEPARATE from [`update_source_status`](Self::update_source_status) which
    /// writes the orthogonal `sources.status` (`SourceStatus`). Bind via
    /// [`EnrichmentStatus::as_str`].
    ///
    /// A missing row is a benign no-op (`Ok(())`), NOT an error: the enrichment
    /// worker updates status at points where a concurrent `purge_source` may have
    /// already deleted the row, so the source being gone is the correct outcome.
    pub async fn update_enrichment_status(
        &self,
        id: &str,
        status: EnrichmentStatus,
    ) -> Result<(), LensError> {
        // rows_affected() == 0 ⇒ the source was purged mid-enrichment; treat as a
        // successful no-op rather than a misleading validation error.
        sqlx::query("UPDATE sources SET enrichment_status = ? WHERE id = ?")
            .bind(status.as_str())
            .bind(id)
            .execute(self.pool)
            .await?;
        Ok(())
    }

    /// Sets a source's `enrichment_status` AND `enrichment_meta` JSON in one
    /// statement (M4 Phase-3 Step 4). The meta JSON carries the composite cache
    /// key + budget/skip reason (AC9/AC11).
    ///
    /// A missing row is a benign no-op (`Ok(())`), NOT an error: the enrichment
    /// worker updates meta at points where a concurrent `purge_source` may have
    /// already deleted the row, so the source being gone is the correct outcome.
    pub async fn update_enrichment_status_and_meta(
        &self,
        id: &str,
        status: EnrichmentStatus,
        meta_json: &str,
    ) -> Result<(), LensError> {
        // rows_affected() == 0 ⇒ the source was purged mid-enrichment; treat as a
        // successful no-op rather than a misleading validation error.
        sqlx::query("UPDATE sources SET enrichment_status = ?, enrichment_meta = ? WHERE id = ?")
            .bind(status.as_str())
            .bind(meta_json)
            .bind(id)
            .execute(self.pool)
            .await?;
        Ok(())
    }

    /// Reads the chunks of a source needed for the enrichment pass (M4 Phase-3
    /// Step 4): id, parent_id, kind, level, section_path, the canonical text, and
    /// block_type. Ordered parents-first then by `token_start` so the worker maps
    /// over level-0 parents in document order. The canonical `text` is read but
    /// NEVER mutated by enrichment (only `enrichment`/`embedding_text` are written).
    pub async fn list_chunks_for_enrichment(
        &self,
        source_id: &str,
    ) -> Result<Vec<EnrichmentChunk>, LensError> {
        let rows = sqlx::query_as::<_, EnrichmentChunk>(
            "SELECT id, parent_id, kind, level, section_path, text, block_type \
             FROM chunks WHERE source_id = ? ORDER BY level ASC, token_start ASC",
        )
        .bind(source_id)
        .fetch_all(self.pool)
        .await?;
        Ok(rows)
    }

    /// Writes the per-chunk `embedding_text` (and, for the representative parent
    /// row, the structural-map `enrichment` JSON) for a source (M4 Phase-3 Step 4).
    ///
    /// Each `(chunk_id, embedding_text, enrichment_json)` is applied as a single
    /// `UPDATE … WHERE id = ?` inside ONE transaction (Decision D); the canonical
    /// `chunks.text` column is untouched. `enrichment_json = None` leaves that
    /// column as-is for child rows (the map attaches to parent rows only, AC4).
    pub async fn write_chunk_enrichment(
        &self,
        updates: &[ChunkEnrichmentUpdate],
    ) -> Result<(), LensError> {
        let mut tx = self.pool.begin().await?;
        for u in updates {
            match &u.enrichment_json {
                Some(json) => {
                    sqlx::query(
                        "UPDATE chunks SET embedding_text = ?, enrichment = ? WHERE id = ?",
                    )
                    .bind(&u.embedding_text)
                    .bind(json)
                    .bind(&u.chunk_id)
                    .execute(&mut *tx)
                    .await?;
                }
                None => {
                    sqlx::query("UPDATE chunks SET embedding_text = ? WHERE id = ?")
                        .bind(&u.embedding_text)
                        .bind(&u.chunk_id)
                        .execute(&mut *tx)
                        .await?;
                }
            }
        }
        tx.commit().await?;
        Ok(())
    }

    /// Reads every chunk of a source projected for the Step-5 re-embed flip:
    /// `(id, level, COALESCE(embedding_text, text))`. Includes the level-2 summary
    /// RAPTOR node if one was inserted. Ordered by level then `token_start` so the
    /// re-embed walks a stable order (summary nodes — `token_start IS NULL` — sort
    /// last within their level, which is harmless).
    pub async fn list_chunks_for_reembed(
        &self,
        source_id: &str,
    ) -> Result<Vec<ReembedChunk>, LensError> {
        let rows = sqlx::query_as::<_, ReembedChunk>(
            "SELECT id, level, COALESCE(embedding_text, text) AS embed_text \
             FROM chunks WHERE source_id = ? ORDER BY level ASC, token_start ASC",
        )
        .bind(source_id)
        .fetch_all(self.pool)
        .await?;
        Ok(rows)
    }

    /// Inserts the doc-summary RAPTOR node (AC6): a `chunks` row with
    /// `kind="summary"`, `level=2`, `parent_id=NULL`, `source_id` SET (so
    /// `drop_source`/purge reclaims its vector), `section_path=""`,
    /// `char_start=char_end=0`, `block_type=NULL`, `enrichment=NULL`,
    /// `embedding_text=NULL` (the summary embeds its `text` directly). A SINGLE
    /// INSERT — NOT the parent-first `insert_chunk_batch` path. Returns the new
    /// chunk id. If a summary node already exists for the source it is deleted
    /// first so a re-run does not accumulate duplicates.
    pub async fn insert_summary_chunk(
        &self,
        source_id: &str,
        text: &str,
    ) -> Result<String, LensError> {
        let mut tx = self.pool.begin().await?;
        // Idempotency: a re-run (e.g. after an `enriching→pending` reset) must not
        // stack a second summary row. Drop any prior summary node for the source.
        sqlx::query("DELETE FROM chunks WHERE source_id = ? AND kind = ?")
            .bind(source_id)
            .bind(crate::chunk::kind::SUMMARY)
            .execute(&mut *tx)
            .await?;
        let id = Uuid::now_v7().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO chunks \
                 (id, source_id, parent_id, kind, level, section_path, text, \
                  token_start, token_end, page, char_start, char_end, block_type, \
                  enrichment, embedding_text, source_anchor, created_at) \
             VALUES (?, ?, NULL, ?, 2, '', ?, NULL, NULL, NULL, 0, 0, NULL, NULL, NULL, NULL, ?)",
        )
        .bind(&id)
        .bind(source_id)
        .bind(crate::chunk::kind::SUMMARY)
        .bind(text)
        .bind(&now)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(id)
    }

    /// Populates a source's post-ingest metadata (`token_count`,
    /// `content_hash`). Errors if no row matches.
    pub async fn update_source_metadata(
        &self,
        id: &str,
        token_count: i64,
        content_hash: &str,
    ) -> Result<(), LensError> {
        let result =
            sqlx::query("UPDATE sources SET token_count = ?, content_hash = ? WHERE id = ?")
                .bind(token_count)
                .bind(content_hash)
                .bind(id)
                .execute(self.pool)
                .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!("no source with id {id}")));
        }
        Ok(())
    }

    /// Fetches a single source row by id, if it exists (including trashed).
    pub async fn get_source(&self, id: &str) -> Result<Option<Source>, LensError> {
        let row = sqlx::query_as::<_, Source>(
            "SELECT id, notebook_id, kind, title, status, locator, selected, token_count, \
             content_hash, raw_content_hash, created_at, trashed_at, enrichment_status, enrichment_meta \
             FROM sources WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(self.pool)
        .await?;
        Ok(row)
    }

    /// Lists all live (non-trashed) sources for a notebook, newest first.
    pub async fn list_sources(&self, notebook_id: &NotebookId) -> Result<Vec<Source>, LensError> {
        let rows = sqlx::query_as::<_, Source>(
            "SELECT id, notebook_id, kind, title, status, locator, selected, token_count, \
             content_hash, raw_content_hash, created_at, trashed_at, enrichment_status, enrichment_meta \
             FROM sources WHERE notebook_id = ? AND trashed_at IS NULL ORDER BY created_at DESC",
        )
        .bind(notebook_id)
        .fetch_all(self.pool)
        .await?;
        Ok(rows)
    }

    /// Reads every chunk of a source for the dev/QA Embeddings Inspector (M4):
    /// the full per-chunk metadata the inspector renders. Ordered `level ASC,
    /// token_start ASC` (matching `list_chunks_for_enrichment`/`_for_reembed`) so
    /// parents precede their children in document order; the summary node
    /// (`token_start IS NULL`) sorts last within its level, which is harmless.
    /// Read-only (SELECT) — never mutates `chunks`.
    pub async fn list_source_chunks(
        &self,
        source_id: &str,
    ) -> Result<Vec<InspectorChunk>, LensError> {
        let rows = sqlx::query_as::<_, InspectorChunk>(
            "SELECT id, parent_id, kind, level, section_path, text, block_type, \
             char_start, char_end, source_anchor, embedding_text \
             FROM chunks WHERE source_id = ? ORDER BY level ASC, token_start ASC",
        )
        .bind(source_id)
        .fetch_all(self.pool)
        .await?;
        Ok(rows)
    }

    /// Reads the ACTIVE embedding-index stats for a notebook for the dev/QA
    /// Embeddings Inspector header (M4): one `(model, dim, status)` row per active
    /// registry entry, ordered by `model ASC`. Returns a `Vec` (NOT an `Option`):
    /// the partial-unique `uq_embidx_active(notebook_id, model, dim)` permits
    /// multiple active rows, so a `LIMIT 1` would be non-deterministic. An empty
    /// `Vec` means the notebook has not been embedded yet. Read-only (SELECT).
    pub async fn get_embedding_stats(
        &self,
        notebook_id: &str,
    ) -> Result<Vec<EmbeddingStats>, LensError> {
        let rows = sqlx::query_as::<_, EmbeddingStats>(
            "SELECT model, dim, status FROM embedding_index \
             WHERE notebook_id = ? AND status = 'active' ORDER BY model ASC",
        )
        .bind(notebook_id)
        .fetch_all(self.pool)
        .await?;
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_title_trims_and_accepts() {
        assert_eq!(validate_title("  hello  ").unwrap(), "hello");
    }

    #[test]
    fn validate_title_rejects_empty_and_whitespace() {
        assert!(matches!(validate_title(""), Err(LensError::Validation(_))));
        assert!(matches!(
            validate_title("   \t\n "),
            Err(LensError::Validation(_))
        ));
    }

    #[test]
    fn validate_title_rejects_too_long() {
        let long = "x".repeat(MAX_TITLE_LEN + 1);
        assert!(matches!(
            validate_title(&long),
            Err(LensError::Validation(_))
        ));
        // Exactly at the cap is fine.
        let ok = "y".repeat(MAX_TITLE_LEN);
        assert_eq!(validate_title(&ok).unwrap().chars().count(), MAX_TITLE_LEN);
    }

    #[test]
    fn notebook_id_is_ergonomic() {
        let id: NotebookId = "abc".to_string().into();
        assert_eq!(&*id, "abc");
        assert_eq!(id.to_string(), "abc");
        assert_eq!(id.as_str(), "abc");
    }

    #[test]
    fn source_status_roundtrip_and_wire_strings() {
        // Lock the EXACT persisted `sources.status` strings (no DB migration) —
        // this is the highest-stakes enum: its wire values gate crash-recovery.
        let cases = [
            (SourceStatus::Pending, "pending"),
            (SourceStatus::Queued, "queued"),
            (SourceStatus::Parsing, "parsing"),
            (SourceStatus::Embedding, "embedding"),
            (SourceStatus::Indexed, "indexed"),
            (SourceStatus::Error, "error"),
            (SourceStatus::NeedsOcr, "needs_ocr"),
            (SourceStatus::NeedsJs, "needs_js"),
        ];
        for (status, s) in cases {
            assert_eq!(status.as_str(), s, "as_str must equal legacy wire string");
            assert_eq!(
                SourceStatus::from_str(s).unwrap(),
                status,
                "FromStr round-trips as_str"
            );
        }
    }

    #[test]
    fn source_status_is_transient_table() {
        // Exactly {Parsing, Embedding} are transient — locked in release (not
        // only via the exhaustive-match debug guarantee). Crash-recovery resets
        // these back to Error and MUST skip every other status.
        let cases = [
            (SourceStatus::Pending, false),
            (SourceStatus::Queued, false),
            (SourceStatus::Parsing, true),
            (SourceStatus::Embedding, true),
            (SourceStatus::Indexed, false),
            (SourceStatus::Error, false),
            (SourceStatus::NeedsOcr, false),
            (SourceStatus::NeedsJs, false),
        ];
        for (status, transient) in cases {
            assert_eq!(
                status.is_transient(),
                transient,
                "{status:?} transient classification must be locked"
            );
        }
    }

    /// Spins up a fully-migrated in-memory pool for repo tests.
    async fn test_pool() -> SqlitePool {
        let pool = crate::db::open_in_memory_pool()
            .await
            .expect("in-memory pool should open");
        crate::db::run_migrations(&pool)
            .await
            .expect("migrations should apply to a fresh in-memory db");
        pool
    }

    #[tokio::test]
    async fn list_with_counts_empty() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        assert!(repo.list_with_counts().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn source_count_correct_after_add() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Notebook",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();

        // No sources yet -> count is 0.
        let summaries = repo.list_with_counts().await.unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].source_count, 0);

        // Add N sources -> count == N.
        for i in 0..3 {
            repo.add_source(
                &nb.id,
                &format!("file{i}.pdf"),
                &format!("/abs/file{i}.pdf"),
            )
            .await
            .unwrap();
        }
        let summaries = repo.list_with_counts().await.unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].source_count, 3);
        assert_eq!(summaries[0].notebook.id, nb.id);
    }

    #[tokio::test]
    async fn create_stamps_global_default_embedding_model() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Notebook",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();
        // The created notebook carries the current global default model id.
        assert_eq!(
            nb.embedding_model.as_deref(),
            Some(crate::embedder::registry::DEFAULT_EMBED_MODEL_ID)
        );
        assert_eq!(nb.embedding_model.as_deref(), Some("nomic-embed-text-v1.5"));
        // ...and the current global default backend (M4 Phase 4b-B): unset config
        // resolves to the enum default `fastembed`.
        assert_eq!(nb.embedding_backend.as_deref(), Some("fastembed"));

        // It is persisted, not just returned in-memory: re-list reads it back.
        let listed = repo.list().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(
            listed[0].embedding_model.as_deref(),
            Some(crate::embedder::registry::DEFAULT_EMBED_MODEL_ID)
        );
        assert_eq!(listed[0].embedding_backend.as_deref(), Some("fastembed"));
    }

    #[tokio::test]
    async fn notebook_stores_and_reads_embedding_model() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Notebook",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();

        // Write a non-default model id directly, then read it back through the
        // repo's SELECT projection.
        sqlx::query("UPDATE notebooks SET embedding_model = ? WHERE id = ?")
            .bind("mxbai-embed-large")
            .bind(&nb.id)
            .execute(&pool)
            .await
            .unwrap();

        let listed = repo.list().await.unwrap();
        assert_eq!(
            listed[0].embedding_model.as_deref(),
            Some("mxbai-embed-large")
        );
    }

    #[tokio::test]
    async fn null_embedding_model_reads_back_as_none_and_resolves_to_default() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Notebook",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();

        // Simulate a pre-migration row: NULL embedding_model.
        sqlx::query("UPDATE notebooks SET embedding_model = NULL WHERE id = ?")
            .bind(&nb.id)
            .execute(&pool)
            .await
            .unwrap();

        let listed = repo.list().await.unwrap();
        assert_eq!(listed[0].embedding_model, None);

        // The registry resolves a None/absent id to the default model.
        let resolved =
            crate::embedder::registry::resolve(listed[0].embedding_model.as_deref().unwrap_or(""));
        assert_eq!(
            resolved.id,
            crate::embedder::registry::DEFAULT_EMBED_MODEL_ID
        );
        assert_eq!(resolved.dim, crate::embedder::registry::DEFAULT_EMBED_DIM);
    }

    #[test]
    fn notebook_ipc_json_includes_embedding_model() {
        let nb = Notebook {
            id: NotebookId::from("nb-1".to_string()),
            title: "T".into(),
            description: None,
            focus_mode: None,
            embedding_model: Some("bge-m3".into()),
            embedding_backend: Some("ollama".into()),
            created_at: "2026-01-01T00:00:00Z".into(),
            updated_at: "2026-01-01T00:00:00Z".into(),
            trashed_at: None,
        };
        let json = serde_json::to_value(&nb).unwrap();
        assert_eq!(json["embedding_model"], "bge-m3");
        assert_eq!(json["embedding_backend"], "ollama");

        // A None embedding_model/backend serializes as JSON null (still present on
        // the wire — the TS mirror reads them as `string | null`).
        let nb_none = Notebook {
            embedding_model: None,
            embedding_backend: None,
            ..nb
        };
        let json_none = serde_json::to_value(&nb_none).unwrap();
        assert!(json_none.get("embedding_model").is_some());
        assert!(json_none["embedding_model"].is_null());
        assert!(json_none.get("embedding_backend").is_some());
        assert!(json_none["embedding_backend"].is_null());
    }

    #[tokio::test]
    async fn update_enrichment_status_missing_source_is_noop() {
        // A concurrent purge_source can delete the row between the worker's
        // enrichment steps; updating a gone source must be a benign Ok no-op, NOT
        // a misleading validation error.
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);

        repo.update_enrichment_status("does-not-exist", EnrichmentStatus::Enriching)
            .await
            .expect("status update on a missing source is a no-op");
        repo.update_enrichment_status_and_meta("does-not-exist", EnrichmentStatus::Enriched, "{}")
            .await
            .expect("status+meta update on a missing source is a no-op");

        // No row was created by the no-op updates.
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM sources")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 0, "no-op update must not insert a row");
    }

    #[tokio::test]
    async fn update_enrichment_status_persists_for_real_source() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Notebook",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();
        let src = repo
            .add_source(&nb.id, "file.pdf", "/abs/file.pdf")
            .await
            .unwrap();

        repo.update_enrichment_status(&src.id, EnrichmentStatus::Skipped)
            .await
            .unwrap();
        let status: Option<String> =
            sqlx::query_scalar("SELECT enrichment_status FROM sources WHERE id = ?")
                .bind(&src.id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(status.as_deref(), Some("skipped"));
    }

    #[tokio::test]
    async fn list_with_counts_only_live_newest_first() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let first = repo
            .create(
                "First",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();
        let second = repo
            .create(
                "Second",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();

        let summaries = repo.list_with_counts().await.unwrap();
        // Newest created_at first.
        assert_eq!(summaries[0].notebook.id, second.id);
        assert_eq!(summaries[1].notebook.id, first.id);
    }

    #[tokio::test]
    async fn create_rename_roundtrip() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Original",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();
        repo.rename(&nb.id, "Renamed").await.unwrap();
        let summaries = repo.list_with_counts().await.unwrap();
        assert_eq!(summaries[0].notebook.title, "Renamed");
    }

    #[tokio::test]
    async fn trash_and_restore_roundtrip() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Notebook",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();

        repo.trash(&nb.id).await.unwrap();
        // Disappears from live list, appears in trashed list.
        assert!(repo.list_with_counts().await.unwrap().is_empty());
        let trashed = repo.list_trashed_with_counts().await.unwrap();
        assert_eq!(trashed.len(), 1);
        assert_eq!(trashed[0].notebook.id, nb.id);
        assert!(trashed[0].notebook.trashed_at.is_some());

        repo.restore(&nb.id).await.unwrap();
        // Returns to live list, gone from trashed list.
        let live = repo.list_with_counts().await.unwrap();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].notebook.id, nb.id);
        assert!(live[0].notebook.trashed_at.is_none());
        assert!(repo.list_trashed_with_counts().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn trash_already_trashed_errors() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Notebook",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();
        repo.trash(&nb.id).await.unwrap();
        assert!(matches!(
            repo.trash(&nb.id).await,
            Err(LensError::Validation(_))
        ));
    }

    #[tokio::test]
    async fn restore_non_trashed_errors() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Notebook",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();
        assert!(matches!(
            repo.restore(&nb.id).await,
            Err(LensError::Validation(_))
        ));
    }

    #[tokio::test]
    async fn list_trashed_with_counts_carries_source_count() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Notebook",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();
        repo.add_source(&nb.id, "a.pdf", "/abs/a.pdf")
            .await
            .unwrap();
        repo.add_source(&nb.id, "b.pdf", "/abs/b.pdf")
            .await
            .unwrap();
        repo.trash(&nb.id).await.unwrap();

        let trashed = repo.list_trashed_with_counts().await.unwrap();
        assert_eq!(trashed.len(), 1);
        assert_eq!(trashed[0].source_count, 2);
    }

    #[tokio::test]
    async fn purge_removes_permanently() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Notebook",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();
        repo.add_source(&nb.id, "a.pdf", "/abs/a.pdf")
            .await
            .unwrap();
        repo.trash(&nb.id).await.unwrap();

        repo.purge(&nb.id).await.unwrap();
        // Gone from both lists.
        assert!(repo.list_with_counts().await.unwrap().is_empty());
        assert!(repo.list_trashed_with_counts().await.unwrap().is_empty());
        // Child sources cascaded.
        assert!(repo.list_sources(&nb.id).await.unwrap().is_empty());
        // Purging again errors (no rows).
        assert!(matches!(
            repo.purge(&nb.id).await,
            Err(LensError::Validation(_))
        ));
    }

    #[tokio::test]
    async fn purge_live_notebook_errors() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Notebook",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();

        // Purging a LIVE (non-trashed) notebook must be rejected and must NOT
        // hard-delete the row.
        assert!(matches!(
            repo.purge(&nb.id).await,
            Err(LensError::Validation(_))
        ));
        // The notebook still exists in the live list.
        let live = repo.list_with_counts().await.unwrap();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].notebook.id, nb.id);
    }

    #[tokio::test]
    async fn delete_is_now_soft() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Notebook",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();

        // The deprecated `delete` now soft-deletes (sets trashed_at).
        #[allow(deprecated)]
        repo.delete(&nb.id).await.unwrap();
        assert!(repo.list_with_counts().await.unwrap().is_empty());
        let trashed = repo.list_trashed_with_counts().await.unwrap();
        assert_eq!(trashed.len(), 1);
        assert!(trashed[0].notebook.trashed_at.is_some());
    }

    #[tokio::test]
    async fn purge_source_requires_trashed() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Notebook",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();
        let src = repo
            .add_source(&nb.id, "a.pdf", "/abs/a.pdf")
            .await
            .unwrap();

        // A LIVE (non-trashed) source must not be hard-purgeable.
        assert!(matches!(
            repo.purge_source(&src.id).await,
            Err(LensError::Validation(_))
        ));
        // The source still exists.
        assert!(repo.get_source(&src.id).await.unwrap().is_some());

        // After trashing, purge succeeds.
        repo.trash_source(&src.id).await.unwrap();
        repo.purge_source(&src.id).await.unwrap();
        assert!(repo.get_source(&src.id).await.unwrap().is_none());
        // Purging again errors (no rows).
        assert!(matches!(
            repo.purge_source(&src.id).await,
            Err(LensError::Validation(_))
        ));
    }

    #[tokio::test]
    async fn notebook_summary_serde_round_trip() {
        let summary = NotebookSummary {
            notebook: Notebook {
                id: NotebookId::from("nb-1"),
                title: "Title".to_string(),
                description: Some("desc".to_string()),
                focus_mode: Some("research".to_string()),
                embedding_model: Some("nomic-embed-text-v1.5".to_string()),
                embedding_backend: Some("fastembed".to_string()),
                created_at: "2026-06-23T00:00:00+00:00".to_string(),
                updated_at: "2026-06-23T00:00:00+00:00".to_string(),
                trashed_at: None,
            },
            source_count: 5,
        };

        let value = serde_json::to_value(&summary).unwrap();
        let obj = value.as_object().expect("serializes to a JSON object");

        // The top-level key set must be EXACTLY these — `serde(flatten)` hoists
        // the Notebook fields to the top level. Guards the TS contract against
        // accidental field additions or flatten collisions.
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "created_at",
                "description",
                "embedding_backend",
                "embedding_model",
                "focus_mode",
                "id",
                "source_count",
                "title",
                "trashed_at",
                "updated_at",
            ]
        );

        // Round-trips back to an equal value.
        let back: NotebookSummary = serde_json::from_value(value).unwrap();
        assert_eq!(back, summary);
    }

    // -----------------------------------------------------------------------
    // M4 Phase 2.5c — add_file_source extension → SourceKind detection
    // -----------------------------------------------------------------------

    /// Writes a tiny source file with the given extension into `dir` and runs
    /// `add_file_source`, returning the inserted source's persisted `kind`.
    async fn kind_for_extension(ext: &str) -> String {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Notebook",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();

        let tmp = tempfile::tempdir().expect("tempdir");
        let data_dir = tmp.path();
        let src_path = data_dir.join(format!("input.{ext}"));
        std::fs::write(&src_path, b"placeholder content").expect("write source file");

        let source = repo
            .add_file_source(data_dir, &nb.id, &src_path, None)
            .await
            .expect("add_file_source must accept the extension")
            .source;
        source.kind
    }

    #[tokio::test]
    async fn add_file_source_json_extension() {
        assert_eq!(kind_for_extension("json").await, SourceKind::Json.as_str());
    }

    #[tokio::test]
    async fn add_file_source_jsonl_extension() {
        assert_eq!(
            kind_for_extension("jsonl").await,
            SourceKind::Jsonl.as_str()
        );
    }

    #[tokio::test]
    async fn add_file_source_ndjson_extension() {
        assert_eq!(
            kind_for_extension("ndjson").await,
            SourceKind::Jsonl.as_str()
        );
    }

    #[tokio::test]
    async fn add_file_source_yaml_extension() {
        assert_eq!(kind_for_extension("yaml").await, SourceKind::Yaml.as_str());
    }

    #[tokio::test]
    async fn add_file_source_yml_extension() {
        assert_eq!(kind_for_extension("yml").await, SourceKind::Yaml.as_str());
    }

    #[tokio::test]
    async fn add_file_source_xml_extension() {
        assert_eq!(kind_for_extension("xml").await, SourceKind::Xml.as_str());
    }

    #[tokio::test]
    async fn add_file_source_mdx_extension() {
        // MDX is a Markdown superset — treated as Markdown for extraction.
        assert_eq!(
            kind_for_extension("mdx").await,
            SourceKind::Markdown.as_str()
        );
    }

    #[tokio::test]
    async fn add_file_source_rtf_extension() {
        assert_eq!(kind_for_extension("rtf").await, SourceKind::Rtf.as_str());
    }

    #[tokio::test]
    async fn add_file_source_odt_extension() {
        assert_eq!(kind_for_extension("odt").await, SourceKind::Odt.as_str());
    }

    #[tokio::test]
    async fn add_file_source_epub_extension() {
        assert_eq!(kind_for_extension("epub").await, SourceKind::Epub.as_str());
    }

    // -----------------------------------------------------------------------
    // M4 Embeddings Inspector (dev/QA) — list_source_chunks + get_embedding_stats
    // -----------------------------------------------------------------------

    /// Inserts a `chunks` row directly via raw SQL. Bypasses full ingest (which
    /// needs the tokenizer, skipped offline) so the inspector reads are tested
    /// deterministically with no model download. `token_start` drives the
    /// secondary sort; `None` (a summary node) sorts last within its level.
    #[allow(clippy::too_many_arguments)]
    async fn insert_chunk_row(
        pool: &SqlitePool,
        source_id: &str,
        id: &str,
        parent_id: Option<&str>,
        kind: &str,
        level: i32,
        section_path: &str,
        text: &str,
        token_start: Option<i64>,
        char_start: Option<i64>,
        char_end: Option<i64>,
        block_type: Option<&str>,
        source_anchor: Option<&str>,
        embedding_text: Option<&str>,
    ) {
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO chunks \
                 (id, source_id, parent_id, kind, level, section_path, text, \
                  token_start, token_end, page, char_start, char_end, block_type, \
                  source_anchor, embedding_text, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?)",
        )
        .bind(id)
        .bind(source_id)
        .bind(parent_id)
        .bind(kind)
        .bind(level)
        .bind(section_path)
        .bind(text)
        .bind(token_start)
        .bind(char_start)
        .bind(char_end)
        .bind(block_type)
        .bind(source_anchor)
        .bind(embedding_text)
        .bind(&now)
        .execute(pool)
        .await
        .expect("insert chunk row");
    }

    /// Inserts an `embedding_index` registry row directly via raw SQL.
    /// `for_test()` stubs the embed worker, so ingest yields no registry rows —
    /// the inspector stats read must be seeded directly.
    async fn insert_embedding_index_row(
        pool: &SqlitePool,
        notebook_id: &str,
        model: &str,
        dim: i64,
        status: &str,
    ) {
        let id = Uuid::now_v7().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO embedding_index \
                 (id, notebook_id, model, dim, prefix_convention, lance_table_name, status, created_at) \
             VALUES (?, ?, ?, ?, 'nomic', ?, ?, ?)",
        )
        .bind(&id)
        .bind(notebook_id)
        .bind(model)
        .bind(dim)
        .bind(format!("lance_{model}_{dim}"))
        .bind(status)
        .bind(&now)
        .execute(pool)
        .await
        .expect("insert embedding_index row");
    }

    #[tokio::test]
    async fn test_list_source_chunks_returns_ordered_chunks() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Notebook",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();
        let src = repo
            .add_source(&nb.id, "doc.md", "/abs/doc.md")
            .await
            .unwrap();

        // Insert ≥3 chunks across levels with varied token_start, deliberately
        // out of final order so ORDER BY (level ASC, token_start ASC) is exercised.
        // The parent is inserted first because the children carry a self-FK to it.
        // parent 0: level 0, token_start 0
        insert_chunk_row(
            &pool,
            &src.id,
            "parent-0",
            None,
            crate::chunk::kind::PARENT,
            0,
            "Intro",
            "parent text",
            Some(0),
            Some(0),
            Some(40),
            Some("heading"),
            Some("{\"page\":1}"),
            Some("Intro: parent text"),
        )
        .await;
        // child b: level 1, token_start 5 (inserted BEFORE child-a so the
        // token_start sort, not insertion order, decides the result order)
        insert_chunk_row(
            &pool,
            &src.id,
            "child-b",
            Some("parent-0"),
            crate::chunk::kind::CHILD,
            1,
            "Intro > B",
            "child b text",
            Some(5),
            Some(50),
            Some(60),
            Some("paragraph"),
            None,
            None,
        )
        .await;
        // child a: level 1, token_start 1 (must precede child b at the same level)
        insert_chunk_row(
            &pool,
            &src.id,
            "child-a",
            Some("parent-0"),
            crate::chunk::kind::CHILD,
            1,
            "Intro > A",
            "child a text",
            Some(1),
            Some(10),
            Some(20),
            None,
            None,
            Some("Intro: child a text"),
        )
        .await;

        let chunks = repo.list_source_chunks(&src.id).await.unwrap();
        let ids: Vec<&str> = chunks.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["parent-0", "child-a", "child-b"],
            "ordered by level ASC then token_start ASC"
        );

        // Field projection is faithful for the parent row.
        let parent = &chunks[0];
        assert_eq!(parent.parent_id, None);
        assert_eq!(parent.kind, crate::chunk::kind::PARENT);
        assert_eq!(parent.level, 0);
        assert_eq!(parent.section_path, "Intro");
        assert_eq!(parent.text, "parent text");
        assert_eq!(parent.block_type.as_deref(), Some("heading"));
        assert_eq!(parent.char_start, Some(0));
        assert_eq!(parent.char_end, Some(40));
        assert_eq!(parent.source_anchor.as_deref(), Some("{\"page\":1}"));
        assert_eq!(parent.embedding_text.as_deref(), Some("Intro: parent text"));

        // Nullable columns surface as None on a child with no anchor.
        let child_a = &chunks[1];
        assert_eq!(child_a.parent_id.as_deref(), Some("parent-0"));
        assert_eq!(child_a.block_type, None);
        assert_eq!(child_a.source_anchor, None);

        // Unknown source_id → empty vec.
        let empty = repo.list_source_chunks("no-such-source").await.unwrap();
        assert!(empty.is_empty());
    }

    #[tokio::test]
    async fn test_get_embedding_stats() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Notebook",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();

        // (a) No embedding_index rows → empty Vec.
        let empty = repo.get_embedding_stats(nb.id.as_str()).await.unwrap();
        assert!(empty.is_empty(), "no rows → empty Vec");

        // (b) Two ACTIVE rows for the SAME notebook with different (model, dim).
        // Insert out of model order so ORDER BY model ASC is exercised.
        insert_embedding_index_row(&pool, nb.id.as_str(), "model-z", 768, "active").await;
        insert_embedding_index_row(&pool, nb.id.as_str(), "model-a", 384, "active").await;
        // A non-active row for the same notebook must be EXCLUDED.
        insert_embedding_index_row(&pool, nb.id.as_str(), "model-m", 512, "building").await;

        let stats = repo.get_embedding_stats(nb.id.as_str()).await.unwrap();
        assert_eq!(stats.len(), 2, "only the 2 active rows, building excluded");
        assert_eq!(stats[0].model, "model-a", "ordered by model ASC");
        assert_eq!(stats[0].dim, 384);
        assert_eq!(stats[0].status, "active");
        assert_eq!(stats[1].model, "model-z");
        assert_eq!(stats[1].dim, 768);
    }

    // -----------------------------------------------------------------------
    // #96 — add_file_source content-hash dedup (raw_content_hash column)
    // -----------------------------------------------------------------------

    /// Creates a fresh notebook in `repo` and returns it.
    async fn make_notebook(repo: &NotebookRepo<'_>, title: &str) -> Notebook {
        repo.create(
            title,
            None,
            None,
            crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
            "fastembed",
        )
        .await
        .expect("create notebook")
    }

    /// Writes `bytes` to `dir/<name>` and returns the path.
    fn write_source_file(dir: &Path, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, bytes).expect("write source file");
        path
    }

    #[tokio::test]
    async fn add_file_source_computes_raw_content_hash() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = make_notebook(&repo, "Notebook").await;
        let tmp = tempfile::tempdir().expect("tempdir");
        let data_dir = tmp.path();

        let bytes = b"hello world content #96";
        let src = write_source_file(data_dir, "a.txt", bytes);

        let outcome = repo
            .add_file_source(data_dir, &nb.id, &src, None)
            .await
            .expect("add_file_source");

        assert!(!outcome.was_existing, "a fresh add is not an existing dup");
        assert_eq!(
            outcome.source.raw_content_hash,
            Some(crate::ingest::sha256_hex(bytes)),
            "raw_content_hash must be the SHA-256 of the raw file bytes"
        );
        // content_hash stays NULL at add time (ingest populates it later).
        assert_eq!(outcome.source.content_hash, None);
    }

    #[tokio::test]
    async fn add_file_source_dedup_returns_existing() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = make_notebook(&repo, "Notebook").await;
        let tmp = tempfile::tempdir().expect("tempdir");
        let data_dir = tmp.path();

        let bytes = b"duplicate content bytes";
        let first_path = write_source_file(data_dir, "first.txt", bytes);
        let first = repo
            .add_file_source(data_dir, &nb.id, &first_path, None)
            .await
            .expect("first add");
        assert!(!first.was_existing);

        // A DIFFERENT file path but IDENTICAL content → dedup hit, same id.
        let second_path = write_source_file(data_dir, "second.txt", bytes);
        let second = repo
            .add_file_source(data_dir, &nb.id, &second_path, None)
            .await
            .expect("second add (dup)");
        assert!(second.was_existing, "identical content is a dedup hit");
        assert_eq!(
            second.source.id, first.source.id,
            "dedup returns the pre-existing source row"
        );

        // Exactly one live source row exists.
        assert_eq!(
            repo.list_sources(&nb.id).await.unwrap().len(),
            1,
            "no duplicate row inserted for identical content"
        );
    }

    #[tokio::test]
    async fn add_file_source_dedup_different_content() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = make_notebook(&repo, "Notebook").await;
        let tmp = tempfile::tempdir().expect("tempdir");
        let data_dir = tmp.path();

        let a = write_source_file(data_dir, "a.txt", b"content A");
        let b = write_source_file(data_dir, "b.txt", b"content B - different");
        let first = repo
            .add_file_source(data_dir, &nb.id, &a, None)
            .await
            .unwrap();
        let second = repo
            .add_file_source(data_dir, &nb.id, &b, None)
            .await
            .unwrap();

        assert!(!first.was_existing);
        assert!(!second.was_existing, "different content is a new row");
        assert_ne!(first.source.id, second.source.id);
        assert_eq!(
            repo.list_sources(&nb.id).await.unwrap().len(),
            2,
            "distinct content yields two rows"
        );
    }

    #[tokio::test]
    async fn add_file_source_dedup_allows_trashed() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = make_notebook(&repo, "Notebook").await;
        let tmp = tempfile::tempdir().expect("tempdir");
        let data_dir = tmp.path();

        let bytes = b"content that gets trashed then re-added";
        let path = write_source_file(data_dir, "doc.txt", bytes);
        let first = repo
            .add_file_source(data_dir, &nb.id, &path, None)
            .await
            .unwrap();

        // Trash it — the partial unique index excludes trashed rows.
        repo.trash_source(&first.source.id).await.unwrap();

        // Re-adding the same content now creates a NEW live row (dedup does not
        // consider trashed sources).
        let second = repo
            .add_file_source(data_dir, &nb.id, &path, None)
            .await
            .unwrap();
        assert!(
            !second.was_existing,
            "a trashed source must not block re-adding the same content"
        );
        assert_ne!(second.source.id, first.source.id);
        // One live source (the trashed one is excluded).
        assert_eq!(repo.list_sources(&nb.id).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn add_file_source_dedup_cross_notebook() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb_a = make_notebook(&repo, "Notebook A").await;
        let nb_b = make_notebook(&repo, "Notebook B").await;
        let tmp = tempfile::tempdir().expect("tempdir");
        let data_dir = tmp.path();

        let bytes = b"shared content across notebooks";
        let path = write_source_file(data_dir, "shared.txt", bytes);

        let in_a = repo
            .add_file_source(data_dir, &nb_a.id, &path, None)
            .await
            .unwrap();
        let in_b = repo
            .add_file_source(data_dir, &nb_b.id, &path, None)
            .await
            .unwrap();

        assert!(!in_a.was_existing);
        assert!(
            !in_b.was_existing,
            "dedup is scoped per-notebook — same content in a different notebook is a new row"
        );
        assert_ne!(in_a.source.id, in_b.source.id);
        assert_eq!(repo.list_sources(&nb_a.id).await.unwrap().len(), 1);
        assert_eq!(repo.list_sources(&nb_b.id).await.unwrap().len(), 1);
    }
}
