//! Notebook domain: the `Notebook` entity, its strongly-typed id, and the
//! repository implementing CRUD over the `notebooks` table.

use std::fmt;
use std::io::Read as _;
use std::ops::Deref;
use std::path::Path;
use std::str::FromStr;

use sha2::{Digest, Sha256};

use serde::{Deserialize, Serialize};
use sqlx::{SqliteConnection, SqlitePool};
use uuid::Uuid;

use crate::LensError;
use crate::graph::EntityGraphRows;
use crate::parse::SourceKind;
use crate::url_normalize::normalize_url;

/// Maximum accepted notebook title length, in characters. Titles longer than
/// this are rejected with [`LensError::Validation`] rather than silently stored.
const MAX_TITLE_LEN: usize = 500;

/// Strongly-typed notebook identifier (UUIDv7 stored as TEXT).
///
/// Newtype prevents silent cross-entity id confusion; `Deref`s to `str` and
/// binds directly into sqlx queries.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type)]
#[serde(transparent)]
#[sqlx(transparent)]
pub struct NotebookId(pub(crate) String);

impl NotebookId {
    pub fn new() -> Self {
        Self(Uuid::now_v7().to_string())
    }

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
/// Lifecycle: `queued → parsing → embedding → indexed` (or `error`). `pending`
/// is the M1 inert-record status (pre-M4). `needs_ocr`/`needs_js`/`render_failed`
/// are TERMINAL-PENDING — crash-recovery must NOT reset them to `error`; this
/// invariant is locked by `crash_recovery_skips_needs_js_and_needs_ocr`.
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
    /// Terminal: URL source was attempted via JS renderer but the render failed
    /// (timeout, blocked host, or empty DOM). Must NOT be reset to `error` on
    /// crash recovery (it is not transient).
    RenderFailed,
}

impl SourceStatus {
    /// Persisted wire format string. MUST NOT change (no DB migration).
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
            Self::RenderFailed => "render_failed",
        }
    }

    /// `true` for in-progress states that crash-recovery resets to `Error`.
    /// Exhaustive by construction — adding a variant forces a recovery decision.
    pub fn is_transient(&self) -> bool {
        match self {
            Self::Parsing | Self::Embedding => true,
            Self::Pending
            | Self::Queued
            | Self::Indexed
            | Self::Error
            | Self::NeedsOcr
            | Self::NeedsJs
            | Self::RenderFailed => false,
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
            "render_failed" => Ok(Self::RenderFailed),
            other => Err(LensError::Validation(format!(
                "unknown source status: {other:?}; expected one of \"pending\", \"queued\", \
                 \"parsing\", \"embedding\", \"indexed\", \"error\", \"needs_ocr\", \"needs_js\", \
                 \"render_failed\""
            ))),
        }
    }
}

impl TryFrom<String> for SourceStatus {
    type Error = LensError;

    fn try_from(s: String) -> Result<Self, LensError> {
        s.parse()
    }
}

impl Serialize for SourceStatus {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for SourceStatus {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// Enrichment lifecycle, persisted in `sources.enrichment_status`. SEPARATE from
/// [`SourceStatus`] so enrichment never touches the crash-recovery invariant.
/// NULL in the column is read as [`None`](EnrichmentStatus::None). Wire strings
/// MUST NOT change.
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
    /// Persisted wire format string. MUST NOT change.
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

    /// Parses a stored column value. NULL and `"none"` both map to
    /// [`None`](EnrichmentStatus::None); unknown values fail loudly.
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, sqlx::FromRow)]
pub struct Source {
    pub id: String,
    pub notebook_id: String,
    #[sqlx(try_from = "String")]
    pub kind: crate::parse::SourceKind,
    pub title: String,
    #[sqlx(try_from = "String")]
    pub status: SourceStatus,
    pub locator: String,
    /// `1` = selected for retrieval.
    pub selected: i64,
    /// Populated by M4 ingestion; `None` until then.
    pub token_count: Option<i64>,
    /// SHA-256 of the extracted text (set at ingest). Used to skip re-ingest
    /// when content is unchanged. `None` until ingested.
    pub content_hash: Option<String>,
    /// SHA-256 dedup key computed at *add* time across all add paths (#96 + #100).
    /// SEPARATE from `content_hash` (which is set at ingest). `None` only for
    /// pre-migration rows. A partial unique index enforces at-most-one live source
    /// per (notebook, raw-content) pair.
    pub raw_content_hash: Option<String>,
    pub created_at: String,
    /// `None` when live.
    pub trashed_at: Option<String>,
    /// Enrichment lifecycle column; `None` (NULL) ≡ `none` for pre-Phase-3 rows.
    pub enrichment_status: Option<String>,
    pub enrichment_meta: Option<String>,
    /// Per-source JS-render opt-in (#78): `1` forces the offscreen-webview path
    /// at ingest regardless of static-extraction result. URL sources only.
    pub force_js_render: i64,
    /// Serialized [`ErrorMeta`](crate::error::ErrorMeta) from the last ingest
    /// failure (#73). `None` unless currently errored; cleared on success.
    pub error_meta: Option<String>,
}

/// A trashed source with its parent notebook's title, used by the Trash modal.
///
/// `notebook_title` is a JOIN alias, so this must NOT derive `sqlx::FromRow`.
/// Only covers sources whose parent notebook is still live; sources under a
/// trashed notebook recover/purge with it and are excluded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrashedSource {
    #[serde(flatten)]
    pub source: Source,
    pub notebook_title: String,
}

/// Return type of all add-source paths (#96 + #100). `was_existing` is `true`
/// on a dedup hit (identical raw content already in the notebook) and `false`
/// on a fresh insert. Wire shape: `{ source, wasExisting }` (camelCase).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AddSourceOutcome {
    pub source: Source,
    pub was_existing: bool,
}

/// Chunk columns needed for the enrichment pass (M4 Phase-3 Step 4).
/// `text` is the IMMUTABLE citation text — enrichment writes only `embedding_text`
/// and `enrichment`, never `text`.
#[derive(Debug, Clone, PartialEq, Eq, sqlx::FromRow)]
pub struct EnrichmentChunk {
    pub id: String,
    pub parent_id: Option<String>,
    pub kind: String,
    pub level: i32,
    pub section_path: String,
    pub text: String,
    pub block_type: Option<String>,
}

/// Per-chunk enrichment write (M4 Phase-3 Step 4): composed `embedding_text`
/// and, for parent rows, the structural-map `enrichment` JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkEnrichmentUpdate {
    pub chunk_id: String,
    pub embedding_text: String,
    /// `Some` writes the structural map (parent rows only); `None` leaves
    /// `chunks.enrichment` unchanged.
    pub enrichment_json: Option<String>,
}

/// Chunk projected for the Step-5 re-embed flip. `embed_text` is
/// `COALESCE(embedding_text, text)` — enriched chunks embed contextual text;
/// unenriched chunks fall back to the canonical body.
#[derive(Debug, Clone, PartialEq, Eq, sqlx::FromRow)]
pub struct ReembedChunk {
    pub id: String,
    pub level: i32,
    pub embed_text: String,
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct EntityNodeRow {
    id: String,
    notebook_id: String,
    source_id: String,
    kind: String,
    name: String,
    canonical_name: Option<String>,
    definition: Option<String>,
    resolution_conf: Option<f64>,
    resolution_prompt_version: Option<String>,
    created_at: String,
}

/// Chunk projected for the dev/QA Embeddings Inspector (M4). Read-only view
/// of identity, hierarchy, citation text, and enrichment metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, sqlx::FromRow)]
pub struct InspectorChunk {
    pub id: String,
    pub parent_id: Option<String>,
    pub kind: String,
    pub level: i32,
    pub section_path: String,
    pub text: String,
    pub block_type: Option<String>,
    pub char_start: Option<i64>,
    pub char_end: Option<i64>,
    pub source_anchor: Option<String>,
    /// `None` until the enrichment pass populates it.
    pub embedding_text: Option<String>,
}

/// Per-(model, dim) embedding-index stats for the Inspector header (M4).
/// One row per ACTIVE registry entry; a notebook may have multiple active rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, sqlx::FromRow)]
pub struct EmbeddingStats {
    pub model: String,
    pub dim: i64,
    pub status: String,
}

/// A notebook row, returned across the IPC boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, sqlx::FromRow)]
pub struct Notebook {
    pub id: NotebookId,
    pub title: String,
    pub description: Option<String>,
    pub focus_mode: Option<String>,
    /// `None` on pre-migration rows; resolved to the registry default at read time.
    pub embedding_model: Option<String>,
    /// `"fastembed"` | `"ollama"`. `None` on pre-migration rows.
    pub embedding_backend: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    /// `None` when live.
    pub trashed_at: Option<String>,
    /// MRU ordering key. Bumped on open, rename, and non-dedup source-add.
    /// `None` on pre-migration rows; read paths use `COALESCE(last_activity_at, created_at)`.
    pub last_activity_at: Option<String>,
}

/// Notebook list response with source count. Does NOT derive `sqlx::FromRow`:
/// `source_count` is a `COUNT(...)` aggregate with no backing column; rows are
/// mapped manually. `#[serde(flatten)]` hoists `Notebook` fields to the top level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NotebookSummary {
    #[serde(flatten)]
    pub notebook: Notebook,
    pub source_count: i64,
}

/// SHA-256 of a file, streamed in 64 KiB chunks. Returns the lowercase hex digest.
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
fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    crate::hex_encode(&hasher.finalize())
}

/// Trims, validates (non-empty, ≤ [`MAX_TITLE_LEN`] chars), and returns the title.
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

/// SELECT projection used by all `add_*` dedup checks. Keyed on
/// `(notebook_id, raw_content_hash)` over live rows; returns at most one row.
const SOURCE_DEDUP_SELECT: &str = "SELECT id, notebook_id, kind, title, status, locator, \
     selected, token_count, content_hash, raw_content_hash, created_at, trashed_at, \
     enrichment_status, enrichment_meta, force_js_render, error_meta \
     FROM sources WHERE notebook_id = ? AND raw_content_hash = ? AND trashed_at IS NULL LIMIT 1";

/// Applies `updates` (composed `embedding_text`, plus `enrichment` JSON on parent
/// rows) on `conn` inside an in-progress transaction. `chunks.text` is never modified.
async fn write_chunk_enrichment_tx(
    conn: &mut SqliteConnection,
    updates: &[ChunkEnrichmentUpdate],
) -> Result<(), LensError> {
    for u in updates {
        match &u.enrichment_json {
            Some(json) => {
                sqlx::query("UPDATE chunks SET embedding_text = ?, enrichment = ? WHERE id = ?")
                    .bind(&u.embedding_text)
                    .bind(json)
                    .bind(&u.chunk_id)
                    .execute(&mut *conn)
                    .await?;
            }
            None => {
                sqlx::query("UPDATE chunks SET embedding_text = ? WHERE id = ?")
                    .bind(&u.embedding_text)
                    .bind(&u.chunk_id)
                    .execute(&mut *conn)
                    .await?;
            }
        }
    }
    Ok(())
}

/// Deletes every `entity_nodes` row for `source_id`; edges and mentions cascade via
/// their `ON DELETE CASCADE` FKs (`from_node`/`to_node`/`entity_node_id`). `entity_nodes`
/// is source-keyed (no chunk FK), so it is the only graph table that survives a
/// chunk-only wipe — hence the explicit delete (#157). Returns the row count for tracing.
pub(crate) async fn delete_entity_nodes_for_source(
    conn: &mut SqliteConnection,
    source_id: &str,
) -> Result<u64, LensError> {
    let result = sqlx::query("DELETE FROM entity_nodes WHERE source_id = ?")
        .bind(source_id)
        .execute(&mut *conn)
        .await?;
    Ok(result.rows_affected())
}

/// Persists the M13 entity-graph rows (nodes → edges → mentions) on `conn` inside an
/// in-progress transaction. The write is **self-replacing**: it first deletes the
/// source's existing nodes (edges/mentions cascade), so a re-enrichment whose entity
/// set shrank leaves no stale nodes (#157). Inserts are `INSERT OR IGNORE` against each
/// UNIQUE constraint (idempotent for identical re-runs). Insert order is
/// nodes-before-edges/mentions so the FK references resolve.
async fn write_entity_graph_tx(
    conn: &mut SqliteConnection,
    graph: &EntityGraphRows,
) -> Result<(), LensError> {
    let deleted = delete_entity_nodes_for_source(conn, &graph.source_id).await?;
    tracing::debug!(
        source_id = %graph.source_id,
        deleted_nodes = deleted,
        "entity-graph write: replaced prior nodes"
    );
    for node in &graph.nodes {
        sqlx::query(
            "INSERT OR IGNORE INTO entity_nodes \
                 (id, notebook_id, source_id, kind, name, canonical_name, definition, \
                  resolution_conf, resolution_prompt_version, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&node.id)
        .bind(&node.notebook_id)
        .bind(&node.source_id)
        .bind(node.kind.as_str())
        .bind(&node.name)
        .bind(&node.canonical_name)
        .bind(&node.definition)
        .bind(node.resolution_conf)
        .bind(&node.resolution_prompt_version)
        .bind(&node.created_at)
        .execute(&mut *conn)
        .await?;
    }
    for edge in &graph.edges {
        sqlx::query(
            "INSERT OR IGNORE INTO entity_edges \
                 (id, notebook_id, source_id, chunk_id, from_node, to_node, relation, \
                  weight, confidence, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&edge.id)
        .bind(&edge.notebook_id)
        .bind(&edge.source_id)
        .bind(&edge.chunk_id)
        .bind(&edge.from_node)
        .bind(&edge.to_node)
        .bind(edge.relation.as_str())
        .bind(edge.weight)
        .bind(edge.confidence)
        .bind(&edge.created_at)
        .execute(&mut *conn)
        .await?;
    }
    for mention in &graph.mentions {
        sqlx::query(
            "INSERT OR IGNORE INTO entity_mentions \
                 (id, notebook_id, entity_node_id, chunk_id, char_start, char_end, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&mention.id)
        .bind(&mention.notebook_id)
        .bind(&mention.entity_node_id)
        .bind(&mention.chunk_id)
        .bind(mention.char_start)
        .bind(mention.char_end)
        .bind(&mention.created_at)
        .execute(&mut *conn)
        .await?;
    }
    Ok(())
}

/// Repository over the `notebooks` table. Zero-cost borrowed handle; holds no state.
pub struct NotebookRepo<'a> {
    pool: &'a SqlitePool,
}

impl<'a> NotebookRepo<'a> {
    pub fn new(pool: &'a SqlitePool) -> Self {
        Self { pool }
    }

    pub async fn list(&self) -> Result<Vec<Notebook>, LensError> {
        let rows = sqlx::query_as::<_, Notebook>(
            "SELECT id, title, description, focus_mode, embedding_model, embedding_backend, \
                    created_at, updated_at, trashed_at, last_activity_at \
             FROM notebooks WHERE trashed_at IS NULL \
             ORDER BY COALESCE(last_activity_at, created_at) DESC",
        )
        .fetch_all(self.pool)
        .await?;
        Ok(rows)
    }

    /// Lists live notebooks with source counts. `LEFT JOIN` so notebooks with
    /// zero sources appear. `s.trashed_at IS NULL` is in the `ON` clause (not
    /// `WHERE`) so individually-trashed sources are excluded without dropping
    /// notebooks that have zero live sources.
    pub async fn list_with_counts(&self) -> Result<Vec<NotebookSummary>, LensError> {
        self.list_summaries(
            "SELECT n.id, n.title, n.description, n.focus_mode, n.embedding_model, \
                    n.embedding_backend, n.created_at, n.updated_at, n.trashed_at, \
                    n.last_activity_at, \
                    COALESCE(COUNT(s.id), 0) AS source_count \
             FROM notebooks n \
             LEFT JOIN sources s ON s.notebook_id = n.id AND s.trashed_at IS NULL \
             WHERE n.trashed_at IS NULL \
             GROUP BY n.id \
             ORDER BY COALESCE(n.last_activity_at, n.created_at) DESC",
        )
        .await
    }

    /// Lists trashed notebooks with source counts. Unlike [`list_with_counts`],
    /// the JOIN has NO `s.trashed_at IS NULL` predicate — a trashed notebook's
    /// count includes all sources (they purge/restore together with the notebook).
    pub async fn list_trashed_with_counts(&self) -> Result<Vec<NotebookSummary>, LensError> {
        self.list_summaries(
            "SELECT n.id, n.title, n.description, n.focus_mode, n.embedding_model, \
                    n.embedding_backend, n.created_at, n.updated_at, n.trashed_at, \
                    n.last_activity_at, \
                    COALESCE(COUNT(s.id), 0) AS source_count \
             FROM notebooks n \
             LEFT JOIN sources s ON s.notebook_id = n.id \
             WHERE n.trashed_at IS NOT NULL \
             GROUP BY n.id \
             ORDER BY n.trashed_at DESC",
        )
        .await
    }

    /// Lists individually-trashed sources whose parent notebook is still live,
    /// ordered by `trashed_at DESC`. Sources under a trashed notebook are excluded.
    /// Uses manual row mapping because `notebook_title` is a JOIN alias.
    pub async fn list_trashed_sources(&self) -> Result<Vec<TrashedSource>, LensError> {
        use sqlx::Row;
        let rows = sqlx::query(
            "SELECT s.id, s.notebook_id, s.kind, s.title, s.status, s.locator, s.selected, \
                    s.token_count, s.content_hash, s.raw_content_hash, s.created_at, s.trashed_at, \
                    s.enrichment_status, s.enrichment_meta, s.force_js_render, s.error_meta, \
                    n.title AS notebook_title \
             FROM sources s \
             JOIN notebooks n ON n.id = s.notebook_id \
             WHERE s.trashed_at IS NOT NULL AND n.trashed_at IS NULL \
             ORDER BY s.trashed_at DESC",
        )
        .fetch_all(self.pool)
        .await?;
        let trashed = rows
            .into_iter()
            .map(|row| {
                let kind_str: String = row.try_get("kind").map_err(LensError::from)?;
                let status_str: String = row.try_get("status").map_err(LensError::from)?;
                Ok(TrashedSource {
                    source: Source {
                        id: row.try_get("id").map_err(LensError::from)?,
                        notebook_id: row.try_get("notebook_id").map_err(LensError::from)?,
                        kind: crate::parse::SourceKind::from_kind_str(&kind_str)?,
                        title: row.try_get("title").map_err(LensError::from)?,
                        status: status_str.parse::<SourceStatus>()?,
                        locator: row.try_get("locator").map_err(LensError::from)?,
                        selected: row.try_get("selected").map_err(LensError::from)?,
                        token_count: row.try_get("token_count").map_err(LensError::from)?,
                        content_hash: row.try_get("content_hash").map_err(LensError::from)?,
                        raw_content_hash: row
                            .try_get("raw_content_hash")
                            .map_err(LensError::from)?,
                        created_at: row.try_get("created_at").map_err(LensError::from)?,
                        trashed_at: row.try_get("trashed_at").map_err(LensError::from)?,
                        enrichment_status: row
                            .try_get("enrichment_status")
                            .map_err(LensError::from)?,
                        enrichment_meta: row.try_get("enrichment_meta").map_err(LensError::from)?,
                        force_js_render: row.try_get("force_js_render").map_err(LensError::from)?,
                        error_meta: row.try_get("error_meta").map_err(LensError::from)?,
                    },
                    notebook_title: row.try_get("notebook_title").map_err(LensError::from)?,
                })
            })
            .collect::<Result<Vec<_>, LensError>>()?;
        Ok(trashed)
    }

    /// Shared row-mapping helper for `list_with_counts` and `list_trashed_with_counts`.
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
                        last_activity_at: row.try_get("last_activity_at")?,
                    },
                    source_count: row.try_get("source_count")?,
                })
            })
            .collect::<Result<Vec<_>, LensError>>()?;
        Ok(summaries)
    }

    /// Creates a notebook. Title is validated; `embedding_model`/`embedding_backend`
    /// are stamped verbatim (caller resolves the app-wide default before calling).
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
                  created_at, updated_at, trashed_at, last_activity_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
        )
        .bind(&id)
        .bind(&title)
        .bind(&description)
        .bind(&focus_mode)
        .bind(&embedding_model)
        .bind(&embedding_backend)
        .bind(&now)
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
            updated_at: now.clone(),
            trashed_at: None,
            last_activity_at: Some(now),
        })
    }

    /// Renames a notebook, bumping `updated_at` and `last_activity_at`. Only
    /// affects live notebooks (`AND trashed_at IS NULL` guards against direct IPC
    /// misuse).
    pub async fn rename(&self, id: &NotebookId, title: &str) -> Result<(), LensError> {
        let title = validate_title(title)?;
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE notebooks SET title = ?, updated_at = ?, last_activity_at = ? \
             WHERE id = ? AND trashed_at IS NULL",
        )
        .bind(&title)
        .bind(&now)
        .bind(&now)
        .bind(id)
        .execute(self.pool)
        .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!("no notebook with id {id}")));
        }
        Ok(())
    }

    /// Bumps `last_activity_at` to now. A trashed or unknown id returns an error
    /// (a trashed notebook must never surface as "most recent").
    pub async fn touch_activity(&self, id: &NotebookId) -> Result<(), LensError> {
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE notebooks SET last_activity_at = ? WHERE id = ? AND trashed_at IS NULL",
        )
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

    /// Bumps `last_activity_at` on the non-dedup success path of all `add_*`
    /// methods. Never errors on 0 rows; `AND trashed_at IS NULL` prevents a
    /// trashed notebook from gaining recency.
    async fn bump_activity(&self, id: &NotebookId, now: &str) -> Result<(), LensError> {
        sqlx::query(
            "UPDATE notebooks SET last_activity_at = ? WHERE id = ? AND trashed_at IS NULL",
        )
        .bind(now)
        .bind(id)
        .execute(self.pool)
        .await?;
        Ok(())
    }

    /// Alias for [`trash`](Self::trash). Historically a hard DELETE; now soft.
    #[deprecated(note = "Use trash() directly; kept for backward compat")]
    pub async fn delete(&self, id: &NotebookId) -> Result<(), LensError> {
        self.trash(id).await
    }

    /// Soft-deletes a notebook (sets `trashed_at`). Only affects live notebooks.
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

    /// Restores a trashed notebook (clears `trashed_at`). Only affects trashed notebooks.
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

    /// Permanently deletes a trashed notebook. Child rows cascade via
    /// `ON DELETE CASCADE`. A live notebook must be trashed first.
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

    /// Inserts an inert M1 file record (`status = "pending"`). Dedup is PATH-based
    /// (sha256 of the locator string) — no file bytes are read here, so two
    /// different paths with identical content will NOT dedup via this path.
    pub async fn add_source(
        &self,
        notebook_id: &NotebookId,
        title: &str,
        locator: &str,
    ) -> Result<AddSourceOutcome, LensError> {
        let raw_content_hash = sha256_bytes(locator.as_bytes());

        if let Some(dup) = sqlx::query_as::<_, Source>(SOURCE_DEDUP_SELECT)
            .bind(notebook_id)
            .bind(&raw_content_hash)
            .fetch_optional(self.pool)
            .await?
        {
            tracing::info!(
                notebook_id = %notebook_id,
                raw_content_hash = %raw_content_hash,
                source_id = %dup.id,
                "duplicate onboarding source (path) detected at add time — returning existing source"
            );
            return Ok(AddSourceOutcome {
                source: dup,
                was_existing: true,
            });
        }

        let id = Uuid::now_v7().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query(
            "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
             created_at, raw_content_hash) \
             VALUES (?, ?, 'file', ?, ?, ?, 1, ?, ?) \
             ON CONFLICT DO NOTHING",
        )
        .bind(&id)
        .bind(notebook_id)
        .bind(title)
        .bind(SourceStatus::Pending.as_str())
        .bind(locator)
        .bind(&now)
        .bind(&raw_content_hash)
        .execute(self.pool)
        .await?;

        if result.rows_affected() == 0 {
            let winner = sqlx::query_as::<_, Source>(SOURCE_DEDUP_SELECT)
                .bind(notebook_id)
                .bind(&raw_content_hash)
                .fetch_one(self.pool)
                .await?;
            tracing::info!(
                notebook_id = %notebook_id,
                raw_content_hash = %raw_content_hash,
                source_id = %winner.id,
                "duplicate onboarding source (path) detected via ON CONFLICT race — returning existing source"
            );
            return Ok(AddSourceOutcome {
                source: winner,
                was_existing: true,
            });
        }

        self.bump_activity(notebook_id, &now).await?;

        Ok(AddSourceOutcome {
            source: Source {
                id,
                notebook_id: notebook_id.to_string(),
                kind: crate::parse::SourceKind::File,
                title: title.to_string(),
                status: SourceStatus::Pending,
                locator: locator.to_string(),
                selected: 1,
                token_count: None,
                content_hash: None,
                raw_content_hash: Some(raw_content_hash),
                created_at: now,
                trashed_at: None,
                enrichment_status: None,
                enrichment_meta: None,
                force_js_render: 0,
                error_meta: None,
            },
            was_existing: false,
        })
    }

    /// Writes pasted text to `{data_dir}/sources/{id}.{ext}` and inserts a
    /// `sources` row with `status = "queued"`. Dedup is content-based (SHA-256 of
    /// the raw text bytes). A dedup hit returns the existing row without writing
    /// the managed file.
    pub async fn add_text_source(
        &self,
        data_dir: &Path,
        notebook_id: &NotebookId,
        title: &str,
        text: &str,
        kind: &str,
        max_source_bytes: usize,
    ) -> Result<AddSourceOutcome, LensError> {
        if text.len() > max_source_bytes {
            return Err(LensError::Validation(format!(
                "source text is {} bytes, exceeding the {max_source_bytes}-byte limit",
                text.len()
            )));
        }
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
        let raw_content_hash = sha256_bytes(text.as_bytes());

        if let Some(dup) = sqlx::query_as::<_, Source>(SOURCE_DEDUP_SELECT)
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

        let sources_dir = data_dir.join("sources");
        std::fs::create_dir_all(&sources_dir)
            .map_err(|e| LensError::Io(format!("{}: {e}", sources_dir.display())))?;
        let path = sources_dir.join(format!("{id}.{ext}"));
        std::fs::write(&path, text)
            .map_err(|e| LensError::Io(format!("{}: {e}", path.display())))?;
        let locator = path.display().to_string();

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
            // Lost the race: reclaim the managed file and return the winner.
            let _ = std::fs::remove_file(&path);
            let winner = sqlx::query_as::<_, Source>(SOURCE_DEDUP_SELECT)
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

        self.bump_activity(notebook_id, &now).await?;

        Ok(AddSourceOutcome {
            source: Source {
                id,
                notebook_id: notebook_id.to_string(),
                kind: SourceKind::from_kind_str(kind)?,
                title: title.to_string(),
                status: SourceStatus::Queued,
                locator,
                selected: 1,
                token_count: None,
                content_hash: None,
                raw_content_hash: Some(raw_content_hash),
                created_at: now,
                trashed_at: None,
                enrichment_status: None,
                enrichment_meta: None,
                force_js_render: 0,
                error_meta: None,
            },
            was_existing: false,
        })
    }

    /// Copies a local file into managed storage and inserts a `sources` row with
    /// `status = "queued"`. Detects `kind` from the file extension; unsupported
    /// extensions are rejected. Dedup is content-based (SHA-256 of the raw file
    /// bytes, streamed in 64 KiB chunks). A dedup hit returns the existing row
    /// without copying the file.
    pub async fn add_file_source(
        &self,
        data_dir: &Path,
        notebook_id: &NotebookId,
        src_path: &Path,
        title: Option<&str>,
    ) -> Result<AddSourceOutcome, LensError> {
        let ext_lower = src_path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        let (kind, ext) = match ext_lower.as_deref() {
            Some("pdf") => (SourceKind::Pdf, "pdf"),
            Some("docx") => (SourceKind::Docx, "docx"),
            Some("txt") => (SourceKind::Text, "txt"),
            Some("md") | Some("markdown") | Some("mdx") => (SourceKind::Markdown, "md"),
            Some("json") => (SourceKind::Json, "json"),
            Some("jsonl") => (SourceKind::Jsonl, "jsonl"),
            Some("ndjson") => (SourceKind::Jsonl, "ndjson"),
            Some("yaml") => (SourceKind::Yaml, "yaml"),
            Some("yml") => (SourceKind::Yaml, "yml"),
            Some("xml") => (SourceKind::Xml, "xml"),
            Some("rtf") => (SourceKind::Rtf, "rtf"),
            Some("odt") => (SourceKind::Odt, "odt"),
            Some("epub") => (SourceKind::Epub, "epub"),
            Some("xlsx") => (SourceKind::Xlsx, "xlsx"),
            Some("xls") => (SourceKind::Xls, "xls"),
            Some("csv") => (SourceKind::Csv, "csv"),
            Some("mp3") => (SourceKind::Audio, "mp3"),
            Some("m4a") => (SourceKind::Audio, "m4a"),
            Some("aac") => (SourceKind::Audio, "aac"),
            Some("wav") => (SourceKind::Audio, "wav"),
            Some("flac") => (SourceKind::Audio, "flac"),
            other => {
                return Err(LensError::Validation(format!(
                    "unsupported file extension {other:?} for {}; expected one of \
                     \".pdf\", \".docx\", \".txt\", \".md\", \".markdown\", \".mdx\", \".json\", \
                     \".jsonl\", \".ndjson\", \".yaml\", \".yml\", \".xml\", \".rtf\", \".odt\", \
                     \".epub\", \".xlsx\", \".xls\", \".csv\", \".mp3\", \".m4a\", \".aac\", \
                     \".wav\", \".flac\"",
                    src_path.display()
                )));
            }
        };

        let title = match title {
            Some(t) => t.to_string(),
            None => src_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("Untitled")
                .to_string(),
        };

        let raw_content_hash = sha256_file(src_path)
            .map_err(|e| LensError::Io(format!("hash {}: {e}", src_path.display())))?;

        if let Some(dup) = sqlx::query_as::<_, Source>(SOURCE_DEDUP_SELECT)
            .bind(notebook_id)
            .bind(&raw_content_hash)
            .fetch_optional(self.pool)
            .await?
        {
            tracing::info!(
                notebook_id = %notebook_id,
                raw_content_hash = %raw_content_hash,
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

        let sources_dir = data_dir.join("sources");
        std::fs::create_dir_all(&sources_dir)
            .map_err(|e| LensError::Io(format!("{}: {e}", sources_dir.display())))?;
        let dest = sources_dir.join(format!("{id}.{ext}"));
        std::fs::copy(src_path, &dest)
            .map_err(|e| LensError::Io(format!("copy {}: {e}", dest.display())))?;
        let locator = dest.display().to_string();

        let result = sqlx::query(
            "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
             created_at, raw_content_hash) \
             VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?) \
             ON CONFLICT DO NOTHING",
        )
        .bind(&id)
        .bind(notebook_id)
        .bind(kind.as_str())
        .bind(&title)
        .bind(SourceStatus::Queued.as_str())
        .bind(&locator)
        .bind(&now)
        .bind(&raw_content_hash)
        .execute(self.pool)
        .await?;

        if result.rows_affected() == 0 {
            // Lost the race: reclaim the copy and return the winner.
            let _ = std::fs::remove_file(&dest);
            let winner = sqlx::query_as::<_, Source>(SOURCE_DEDUP_SELECT)
                .bind(notebook_id)
                .bind(&raw_content_hash)
                .fetch_one(self.pool)
                .await?;
            tracing::info!(
                notebook_id = %notebook_id,
                raw_content_hash = %raw_content_hash,
                source_id = %winner.id,
                "duplicate file detected via ON CONFLICT race — returning existing source"
            );
            return Ok(AddSourceOutcome {
                source: winner,
                was_existing: true,
            });
        }

        self.bump_activity(notebook_id, &now).await?;

        Ok(AddSourceOutcome {
            source: Source {
                id,
                notebook_id: notebook_id.to_string(),
                kind,
                title,
                status: SourceStatus::Queued,
                locator,
                selected: 1,
                token_count: None,
                content_hash: None,
                raw_content_hash: Some(raw_content_hash),
                created_at: now,
                trashed_at: None,
                enrichment_status: None,
                enrichment_meta: None,
                force_js_render: 0,
                error_meta: None,
            },
            was_existing: false,
        })
    }

    /// Inserts a URL source (`kind = "url"`, `status = "queued"`). The locator
    /// is the verbatim URL; no file is written. Dedup is on the SHA-256 of the
    /// normalized URL. `force_js_render` (#78): when `true`, ingest always routes
    /// through the JS-render path; a dedup hit returns the existing row unchanged.
    pub async fn add_url_source(
        &self,
        notebook_id: &NotebookId,
        title: &str,
        url: &str,
        force_js_render: bool,
    ) -> Result<AddSourceOutcome, LensError> {
        let raw_content_hash = sha256_bytes(normalize_url(url)?.as_bytes());

        if let Some(dup) = sqlx::query_as::<_, Source>(SOURCE_DEDUP_SELECT)
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
        let result = sqlx::query(
            "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
             created_at, raw_content_hash, force_js_render) \
             VALUES (?, ?, 'url', ?, ?, ?, 1, ?, ?, ?) \
             ON CONFLICT DO NOTHING",
        )
        .bind(&id)
        .bind(notebook_id)
        .bind(title)
        .bind(SourceStatus::Queued.as_str())
        .bind(url)
        .bind(&now)
        .bind(&raw_content_hash)
        .bind(i64::from(force_js_render))
        .execute(self.pool)
        .await?;

        if result.rows_affected() == 0 {
            let winner = sqlx::query_as::<_, Source>(SOURCE_DEDUP_SELECT)
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

        self.bump_activity(notebook_id, &now).await?;

        Ok(AddSourceOutcome {
            source: Source {
                id,
                notebook_id: notebook_id.to_string(),
                kind: SourceKind::Url,
                title: title.to_string(),
                status: SourceStatus::Queued,
                locator: url.to_string(),
                selected: 1,
                token_count: None,
                content_hash: None,
                raw_content_hash: Some(raw_content_hash),
                created_at: now,
                trashed_at: None,
                enrichment_status: None,
                enrichment_meta: None,
                force_js_render: i64::from(force_js_render),
                error_meta: None,
            },
            was_existing: false,
        })
    }

    /// Soft-deletes a source. Only affects live sources.
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

    pub async fn restore_source(&self, id: &str) -> Result<(), LensError> {
        let result = sqlx::query(
            "UPDATE sources SET trashed_at = NULL \
             WHERE id = ? AND trashed_at IS NOT NULL \
               AND NOT EXISTS ( \
                 SELECT 1 FROM sources live \
                 WHERE live.notebook_id = sources.notebook_id \
                   AND live.raw_content_hash = sources.raw_content_hash \
                   AND live.trashed_at IS NULL \
                   AND live.id != sources.id \
               )",
        )
        .bind(id)
        .execute(self.pool)
        .await?;

        if result.rows_affected() > 0 {
            return Ok(());
        }

        let still_trashed: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM sources WHERE id = ? AND trashed_at IS NOT NULL",
        )
        .bind(id)
        .fetch_one(self.pool)
        .await?;

        if still_trashed > 0 {
            tracing::warn!(
                restoring_id = %id,
                "restore blocked — a live source with identical content already exists"
            );
            return Err(LensError::Validation(
                "a source with this content already exists in the notebook".into(),
            ));
        }

        Err(LensError::Validation(format!(
            "no trashed source with id {id}"
        )))
    }

    /// Permanently deletes a trashed source. Child `chunks` cascade via
    /// `ON DELETE CASCADE`. Callers must remove Lance vectors first.
    ///
    /// FTS hygiene (#39): explicitly delete the source's `chunks_fts` rows in the
    /// SAME txn (FK-cascade won't fire the AFTER DELETE trigger; see migration 0013).
    pub async fn purge_source(&self, id: &str) -> Result<(), LensError> {
        let mut tx = self.pool.begin().await?;
        sqlx::query(
            "DELETE FROM chunks_fts \
             WHERE chunk_id IN (SELECT id FROM chunks WHERE source_id = ?)",
        )
        .bind(id)
        .execute(&mut *tx)
        .await?;
        let result = sqlx::query("DELETE FROM sources WHERE id = ? AND trashed_at IS NOT NULL")
            .bind(id)
            .execute(&mut *tx)
            .await?;
        if result.rows_affected() == 0 {
            tx.rollback().await?;
            return Err(LensError::Validation(format!(
                "no trashed source with id {id}"
            )));
        }
        tx.commit().await?;
        Ok(())
    }

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

    /// Atomically sets `status='error'` and writes `error_meta` JSON (#73).
    /// A missing row is a benign no-op (concurrent `purge_source` may have already
    /// removed the row).
    pub async fn set_source_error(
        &self,
        id: &str,
        meta: &crate::error::ErrorMeta,
    ) -> Result<(), LensError> {
        let meta_json = serde_json::to_string(meta)?;
        sqlx::query("UPDATE sources SET status = 'error', error_meta = ? WHERE id = ?")
            .bind(meta_json)
            .bind(id)
            .execute(self.pool)
            .await?;
        Ok(())
    }

    /// Clears `error_meta` to NULL on successful re-ingest (#73). Missing row is
    /// a benign no-op.
    pub async fn clear_source_error_meta(&self, id: &str) -> Result<(), LensError> {
        sqlx::query("UPDATE sources SET error_meta = NULL WHERE id = ?")
            .bind(id)
            .execute(self.pool)
            .await?;
        Ok(())
    }

    /// Sets `sources.enrichment_status`. SEPARATE from `update_source_status`
    /// (which writes the orthogonal `sources.status`). Missing row is a no-op.
    pub async fn update_enrichment_status(
        &self,
        id: &str,
        status: EnrichmentStatus,
    ) -> Result<(), LensError> {
        sqlx::query("UPDATE sources SET enrichment_status = ? WHERE id = ?")
            .bind(status.as_str())
            .bind(id)
            .execute(self.pool)
            .await?;
        Ok(())
    }

    /// Sets `enrichment_status` and `enrichment_meta` in one statement (Phase-3
    /// Step 4). Missing row is a no-op.
    pub async fn update_enrichment_status_and_meta(
        &self,
        id: &str,
        status: EnrichmentStatus,
        meta_json: &str,
    ) -> Result<(), LensError> {
        sqlx::query("UPDATE sources SET enrichment_status = ?, enrichment_meta = ? WHERE id = ?")
            .bind(status.as_str())
            .bind(meta_json)
            .bind(id)
            .execute(self.pool)
            .await?;
        Ok(())
    }

    /// Lists chunk columns needed for the enrichment pass, ordered parents-first
    /// then by `token_start`.
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

    /// Loads the relation predicate vocabulary (`relation_types.name`) for the
    /// #154 extraction pass. An empty set (e.g. migration not yet run) makes the
    /// pass drop every triple — safe (zero semantic edges).
    pub async fn list_relation_type_names(
        &self,
    ) -> Result<std::collections::HashSet<String>, LensError> {
        let names: Vec<String> = sqlx::query_scalar("SELECT name FROM relation_types")
            .fetch_all(self.pool)
            .await?;
        Ok(names.into_iter().collect())
    }

    /// Loads relation-type aliases (`alias -> canonical`) for query-time / write-time
    /// alias resolution (#154). Never mutates stored predicates.
    pub async fn list_relation_type_aliases(
        &self,
    ) -> Result<std::collections::HashMap<String, String>, LensError> {
        let rows: Vec<(String, String)> =
            sqlx::query_as("SELECT alias, canonical FROM relation_type_aliases")
                .fetch_all(self.pool)
                .await?;
        Ok(rows.into_iter().collect())
    }

    /// Writes `embedding_text` (and, for parent rows, `enrichment` JSON) in one
    /// transaction. `chunks.text` is never modified. Retained for the prefix-only
    /// fallback path (`worker.rs`), which has no graph rows.
    pub async fn write_chunk_enrichment(
        &self,
        updates: &[ChunkEnrichmentUpdate],
    ) -> Result<(), LensError> {
        let mut tx = self.pool.begin().await?;
        write_chunk_enrichment_tx(&mut tx, updates).await?;
        tx.commit().await?;
        Ok(())
    }

    /// Writes chunk enrichment (TEXT columns) and the M13 entity-graph rows in ONE
    /// transaction (AC7): begin → chunk enrichment → graph → commit. Either both land
    /// or neither does, so a mid-write failure never leaves enriched chunks with a
    /// partial graph. The graph write is self-replacing per source (see
    /// [`write_entity_graph_tx`]), so a re-enrichment neither accumulates nor strands rows.
    pub async fn write_enrichment_and_graph(
        &self,
        updates: &[ChunkEnrichmentUpdate],
        graph: &EntityGraphRows,
    ) -> Result<(), LensError> {
        let mut tx = self.pool.begin().await?;
        write_chunk_enrichment_tx(&mut tx, updates).await?;
        write_entity_graph_tx(&mut tx, graph).await?;
        tx.commit().await?;
        Ok(())
    }

    /// All `entity_nodes` for a notebook, ordered by `created_at` for a deterministic pass.
    pub async fn list_entity_nodes(
        &self,
        notebook_id: &str,
    ) -> Result<Vec<crate::graph::EntityNode>, LensError> {
        let rows: Vec<EntityNodeRow> = sqlx::query_as(
            "SELECT id, notebook_id, source_id, kind, name, canonical_name, definition, \
                    resolution_conf, resolution_prompt_version, created_at \
             FROM entity_nodes WHERE notebook_id = ? ORDER BY created_at ASC, id ASC",
        )
        .bind(notebook_id)
        .fetch_all(self.pool)
        .await?;
        rows.into_iter()
            .map(|r| {
                Ok(crate::graph::EntityNode {
                    kind: crate::graph::EntityKind::from_db(&r.kind)?,
                    id: r.id,
                    notebook_id: r.notebook_id,
                    source_id: r.source_id,
                    name: r.name,
                    canonical_name: r.canonical_name,
                    definition: r.definition,
                    resolution_conf: r.resolution_conf,
                    resolution_prompt_version: r.resolution_prompt_version,
                    created_at: r.created_at,
                })
            })
            .collect()
    }

    /// Writes one resolution pass atomically: resets all resolution columns, stamps the version,
    /// then applies canonical assignments (prevents stale aliases from a prior pass).
    pub async fn write_resolution_updates(
        &self,
        notebook_id: &str,
        prompt_version: &str,
        updates: &[crate::resolution::ResolutionUpdate],
        fault_after_stamp: bool,
    ) -> Result<(), LensError> {
        let mut tx = self.pool.begin().await?;
        // Full reset before re-applying: clears prior-pass canonical/conf so a node that
        // is no longer in a merged group does not retain a stale alias, while stamping
        // the current version on the whole notebook.
        sqlx::query(
            "UPDATE entity_nodes \
             SET resolution_prompt_version = ?, canonical_name = NULL, resolution_conf = NULL \
             WHERE notebook_id = ?",
        )
        .bind(prompt_version)
        .bind(notebook_id)
        .execute(&mut *tx)
        .await?;
        // Test seam (always `false` in production): abort AFTER the version stamp but
        // BEFORE the canonical updates. Dropping `tx` unwritten rolls back the stamp
        // too — proving the write is all-or-nothing.
        if fault_after_stamp {
            return Err(LensError::Internal(
                "resolution write fault (test seam)".to_string(),
            ));
        }
        for update in updates {
            sqlx::query(
                "UPDATE entity_nodes SET canonical_name = ?, resolution_conf = ? WHERE id = ?",
            )
            .bind(&update.canonical_name)
            .bind(update.resolution_conf)
            .bind(&update.entity_node_id)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    /// Lists chunks for the Step-5 re-embed flip: `(id, level,
    /// COALESCE(embedding_text, text))`. Summary nodes sort last within their level.
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

    /// Inserts the doc-summary RAPTOR node (`kind="summary"`, `level=2`). Deletes
    /// any prior summary for the source first so re-runs don't accumulate duplicates.
    pub async fn insert_summary_chunk(
        &self,
        source_id: &str,
        text: &str,
    ) -> Result<String, LensError> {
        let mut tx = self.pool.begin().await?;
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

    pub async fn get_source(&self, id: &str) -> Result<Option<Source>, LensError> {
        let row = sqlx::query_as::<_, Source>(
            "SELECT id, notebook_id, kind, title, status, locator, selected, token_count, \
             content_hash, raw_content_hash, created_at, trashed_at, enrichment_status, \
             enrichment_meta, force_js_render, error_meta \
             FROM sources WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(self.pool)
        .await?;
        Ok(row)
    }

    pub async fn list_sources(&self, notebook_id: &NotebookId) -> Result<Vec<Source>, LensError> {
        let rows = sqlx::query_as::<_, Source>(
            "SELECT id, notebook_id, kind, title, status, locator, selected, token_count, \
             content_hash, raw_content_hash, created_at, trashed_at, enrichment_status, \
             enrichment_meta, force_js_render, error_meta \
             FROM sources WHERE notebook_id = ? AND trashed_at IS NULL ORDER BY created_at DESC",
        )
        .bind(notebook_id)
        .fetch_all(self.pool)
        .await?;
        Ok(rows)
    }

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

    /// Returns ACTIVE embedding-index stats for the Inspector header. A notebook
    /// may have multiple active rows; an empty Vec means not yet embedded.
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
            (SourceStatus::RenderFailed, "render_failed"),
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
            (SourceStatus::RenderFailed, false),
        ];
        for (status, transient) in cases {
            assert_eq!(
                status.is_transient(),
                transient,
                "{status:?} transient classification must be locked"
            );
        }
    }

    #[test]
    fn source_status_serde_wire_lock() {
        let variants = [
            SourceStatus::Pending,
            SourceStatus::Queued,
            SourceStatus::Parsing,
            SourceStatus::Embedding,
            SourceStatus::Indexed,
            SourceStatus::Error,
            SourceStatus::NeedsOcr,
            SourceStatus::NeedsJs,
            SourceStatus::RenderFailed,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            assert_eq!(
                json,
                format!("\"{}\"", v.as_str()),
                "serde wire must equal as_str for {:?}",
                v
            );
            let rt: SourceStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(rt, v, "serde round-trip must equal original for {:?}", v);
        }
    }

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

        let summaries = repo.list_with_counts().await.unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].source_count, 0);

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
        assert_eq!(
            nb.embedding_model.as_deref(),
            Some(crate::embedder::registry::DEFAULT_EMBED_MODEL_ID)
        );
        assert_eq!(nb.embedding_model.as_deref(), Some("nomic-embed-text-v1.5"));
        assert_eq!(nb.embedding_backend.as_deref(), Some("fastembed"));

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

        sqlx::query("UPDATE notebooks SET embedding_model = NULL WHERE id = ?")
            .bind(&nb.id)
            .execute(&pool)
            .await
            .unwrap();

        let listed = repo.list().await.unwrap();
        assert_eq!(listed[0].embedding_model, None);

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
            last_activity_at: Some("2026-01-01T00:00:00Z".into()),
        };
        let json = serde_json::to_value(&nb).unwrap();
        assert_eq!(json["embedding_model"], "bge-m3");
        assert_eq!(json["embedding_backend"], "ollama");

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
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);

        repo.update_enrichment_status("does-not-exist", EnrichmentStatus::Enriching)
            .await
            .expect("status update on a missing source is a no-op");
        repo.update_enrichment_status_and_meta("does-not-exist", EnrichmentStatus::Enriched, "{}")
            .await
            .expect("status+meta update on a missing source is a no-op");

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
            .unwrap()
            .source;

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

    /// Small helper: creates a live notebook and returns its id. Sleeps a few ms
    /// first so successive creates get strictly-increasing RFC3339 timestamps
    /// (they carry sub-second precision), making recency ordering deterministic.
    async fn create_after_tick(repo: &NotebookRepo<'_>, title: &str) -> NotebookId {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        repo.create(
            title,
            None,
            None,
            crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
            "fastembed",
        )
        .await
        .unwrap()
        .id
    }

    #[tokio::test]
    async fn touch_activity_surfaces_notebook_to_front_of_mru_list() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);

        let a = create_after_tick(&repo, "A").await;
        let b = create_after_tick(&repo, "B").await;
        let c = create_after_tick(&repo, "C").await;

        let ids: Vec<_> = repo
            .list_with_counts()
            .await
            .unwrap()
            .into_iter()
            .map(|s| s.notebook.id)
            .collect();
        assert_eq!(ids, vec![c.clone(), b.clone(), a.clone()]);

        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        repo.touch_activity(&b).await.unwrap();

        let ids: Vec<_> = repo
            .list_with_counts()
            .await
            .unwrap()
            .into_iter()
            .map(|s| s.notebook.id)
            .collect();
        assert_eq!(ids, vec![b, c, a], "touched notebook must surface first");
    }

    #[tokio::test]
    async fn touch_activity_updates_last_activity_at_timestamp() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "N",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();
        let before = nb.last_activity_at.clone().unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        repo.touch_activity(&nb.id).await.unwrap();

        let after = repo
            .list_with_counts()
            .await
            .unwrap()
            .into_iter()
            .find(|s| s.notebook.id == nb.id)
            .unwrap()
            .notebook
            .last_activity_at
            .unwrap();
        assert!(after > before, "touch must advance last_activity_at");
    }

    #[tokio::test]
    async fn touch_activity_on_trashed_notebook_is_validation_error() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "N",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();
        repo.trash(&nb.id).await.unwrap();
        let err = repo.touch_activity(&nb.id).await.unwrap_err();
        assert!(matches!(err, LensError::Validation(_)));
    }

    #[tokio::test]
    async fn touch_activity_on_missing_notebook_is_validation_error() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let err = repo.touch_activity(&NotebookId::new()).await.unwrap_err();
        assert!(matches!(err, LensError::Validation(_)));
    }

    #[tokio::test]
    async fn add_source_bumps_notebook_last_activity_at() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "N",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();
        let before = nb.last_activity_at.clone().unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        repo.add_source(&nb.id, "file.pdf", "/abs/file.pdf")
            .await
            .unwrap();

        let after = repo
            .list_with_counts()
            .await
            .unwrap()
            .into_iter()
            .find(|s| s.notebook.id == nb.id)
            .unwrap()
            .notebook
            .last_activity_at
            .unwrap();
        assert!(
            after > before,
            "adding a source must advance last_activity_at"
        );
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
        assert!(repo.list_with_counts().await.unwrap().is_empty());
        let trashed = repo.list_trashed_with_counts().await.unwrap();
        assert_eq!(trashed.len(), 1);
        assert_eq!(trashed[0].notebook.id, nb.id);
        assert!(trashed[0].notebook.trashed_at.is_some());

        repo.restore(&nb.id).await.unwrap();
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
    async fn source_count_excludes_individually_trashed_sources_from_live_listing() {
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
        repo.add_source(&nb.id, "live.pdf", "/abs/live.pdf")
            .await
            .unwrap();
        let trashed_src = repo
            .add_source(&nb.id, "gone.pdf", "/abs/gone.pdf")
            .await
            .unwrap();
        repo.trash_source(&trashed_src.source.id).await.unwrap();

        let live = repo.list_with_counts().await.unwrap();
        assert_eq!(live.len(), 1);
        assert_eq!(
            live[0].source_count, 1,
            "individually-trashed source must not inflate the live count"
        );

        repo.trash(&nb.id).await.unwrap();
        let trashed = repo.list_trashed_with_counts().await.unwrap();
        assert_eq!(trashed.len(), 1);
        assert_eq!(
            trashed[0].source_count, 2,
            "trashed notebook must count both live and individually-trashed sources"
        );
    }

    /// Regression guard for the "Showing 0 file" bug (commit 85471b8):
    /// a notebook whose ONLY source was individually-trashed must show source_count 1
    /// in the trashed listing, not 0.
    #[tokio::test]
    async fn list_trashed_with_counts_counts_notebook_with_only_trashed_sources() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "AllTrashed",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();
        let src = repo
            .add_source(&nb.id, "only.pdf", "/abs/only.pdf")
            .await
            .unwrap();
        repo.trash_source(&src.source.id).await.unwrap();

        let live = repo.list_with_counts().await.unwrap();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].source_count, 0);

        repo.trash(&nb.id).await.unwrap();

        let trashed = repo.list_trashed_with_counts().await.unwrap();
        assert_eq!(trashed.len(), 1);
        assert_eq!(
            trashed[0].source_count, 1,
            "notebook with only individually-trashed sources must not show 0 files in trash"
        );
    }

    #[tokio::test]
    async fn source_count_includes_notebooks_with_zero_live_sources() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo
            .create(
                "Empty",
                None,
                None,
                crate::embedder::registry::DEFAULT_EMBED_MODEL_ID,
                "fastembed",
            )
            .await
            .unwrap();
        let src = repo
            .add_source(&nb.id, "gone.pdf", "/abs/gone.pdf")
            .await
            .unwrap();
        repo.trash_source(&src.source.id).await.unwrap();

        let live = repo.list_with_counts().await.unwrap();
        assert_eq!(live.len(), 1, "notebook must not vanish from the list");
        assert_eq!(live[0].source_count, 0);
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
        assert!(repo.list_with_counts().await.unwrap().is_empty());
        assert!(repo.list_trashed_with_counts().await.unwrap().is_empty());
        assert!(repo.list_sources(&nb.id).await.unwrap().is_empty());
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

        assert!(matches!(
            repo.purge(&nb.id).await,
            Err(LensError::Validation(_))
        ));
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
            .unwrap()
            .source;

        assert!(matches!(
            repo.purge_source(&src.id).await,
            Err(LensError::Validation(_))
        ));
        assert!(repo.get_source(&src.id).await.unwrap().is_some());

        repo.trash_source(&src.id).await.unwrap();
        repo.purge_source(&src.id).await.unwrap();
        assert!(repo.get_source(&src.id).await.unwrap().is_none());
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
                last_activity_at: Some("2026-06-23T00:00:00+00:00".to_string()),
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
                "last_activity_at",
                "source_count",
                "title",
                "trashed_at",
                "updated_at",
            ]
        );

        let back: NotebookSummary = serde_json::from_value(value).unwrap();
        assert_eq!(back, summary);
    }

    async fn kind_for_extension(ext: &str) -> SourceKind {
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

        repo.add_file_source(data_dir, &nb.id, &src_path, None)
            .await
            .expect("add_file_source must accept the extension")
            .source
            .kind
    }

    #[tokio::test]
    async fn add_file_source_json_extension() {
        assert_eq!(kind_for_extension("json").await, SourceKind::Json);
    }

    #[tokio::test]
    async fn add_file_source_jsonl_extension() {
        assert_eq!(kind_for_extension("jsonl").await, SourceKind::Jsonl);
    }

    #[tokio::test]
    async fn add_file_source_ndjson_extension() {
        assert_eq!(kind_for_extension("ndjson").await, SourceKind::Jsonl);
    }

    #[tokio::test]
    async fn add_file_source_yaml_extension() {
        assert_eq!(kind_for_extension("yaml").await, SourceKind::Yaml);
    }

    #[tokio::test]
    async fn add_file_source_yml_extension() {
        assert_eq!(kind_for_extension("yml").await, SourceKind::Yaml);
    }

    #[tokio::test]
    async fn add_file_source_xml_extension() {
        assert_eq!(kind_for_extension("xml").await, SourceKind::Xml);
    }

    #[tokio::test]
    async fn add_file_source_mdx_extension() {
        assert_eq!(kind_for_extension("mdx").await, SourceKind::Markdown);
    }

    #[tokio::test]
    async fn add_file_source_rtf_extension() {
        assert_eq!(kind_for_extension("rtf").await, SourceKind::Rtf);
    }

    #[tokio::test]
    async fn add_file_source_odt_extension() {
        assert_eq!(kind_for_extension("odt").await, SourceKind::Odt);
    }

    #[tokio::test]
    async fn add_file_source_epub_extension() {
        assert_eq!(kind_for_extension("epub").await, SourceKind::Epub);
    }

    /// Inserts a `chunks` row directly via raw SQL (bypasses full ingest).
    /// `token_start` drives the secondary sort; `None` sorts last within its level.
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
            .unwrap()
            .source;

        // Insert ≥3 chunks across levels with varied token_start, deliberately
        // out of final order so ORDER BY (level ASC, token_start ASC) is exercised.
        // The parent is inserted first because the children carry a self-FK to it.
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

        let child_a = &chunks[1];
        assert_eq!(child_a.parent_id.as_deref(), Some("parent-0"));
        assert_eq!(child_a.block_type, None);
        assert_eq!(child_a.source_anchor, None);

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

        let empty = repo.get_embedding_stats(nb.id.as_str()).await.unwrap();
        assert!(empty.is_empty(), "no rows → empty Vec");

        // Insert out of model order so ORDER BY model ASC is exercised; building row excluded.
        insert_embedding_index_row(&pool, nb.id.as_str(), "model-z", 768, "active").await;
        insert_embedding_index_row(&pool, nb.id.as_str(), "model-a", 384, "active").await;
        insert_embedding_index_row(&pool, nb.id.as_str(), "model-m", 512, "building").await;

        let stats = repo.get_embedding_stats(nb.id.as_str()).await.unwrap();
        assert_eq!(stats.len(), 2, "only the 2 active rows, building excluded");
        assert_eq!(stats[0].model, "model-a", "ordered by model ASC");
        assert_eq!(stats[0].dim, 384);
        assert_eq!(stats[0].status, "active");
        assert_eq!(stats[1].model, "model-z");
        assert_eq!(stats[1].dim, 768);
    }

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

        repo.trash_source(&first.source.id).await.unwrap();

        let second = repo
            .add_file_source(data_dir, &nb.id, &path, None)
            .await
            .unwrap();
        assert!(
            !second.was_existing,
            "a trashed source must not block re-adding the same content"
        );
        assert_ne!(second.source.id, first.source.id);
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

    #[tokio::test]
    async fn list_trashed_sources_returns_trashed_under_live_notebook() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = make_notebook(&repo, "My Notebook").await;

        let src = repo
            .add_source(&nb.id, "report.pdf", "/abs/report.pdf")
            .await
            .unwrap();

        assert!(repo.list_trashed_sources().await.unwrap().is_empty());

        repo.trash_source(&src.source.id).await.unwrap();

        let trashed = repo.list_trashed_sources().await.unwrap();
        assert_eq!(trashed.len(), 1);
        assert_eq!(trashed[0].source.id, src.source.id);
        assert_eq!(trashed[0].notebook_title, "My Notebook");
        assert!(trashed[0].source.trashed_at.is_some());
    }

    #[tokio::test]
    async fn list_trashed_sources_excludes_live_sources() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = make_notebook(&repo, "Notebook").await;

        repo.add_source(&nb.id, "live.pdf", "/abs/live.pdf")
            .await
            .unwrap();

        assert!(repo.list_trashed_sources().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn list_trashed_sources_excludes_sources_under_trashed_notebook() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = make_notebook(&repo, "Notebook").await;

        let src = repo
            .add_source(&nb.id, "report.pdf", "/abs/report.pdf")
            .await
            .unwrap();

        repo.trash_source(&src.source.id).await.unwrap();
        repo.trash(&nb.id).await.unwrap();

        assert!(repo.list_trashed_sources().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn list_trashed_sources_ordered_by_trashed_at_desc() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = make_notebook(&repo, "Notebook").await;

        let src_a = repo
            .add_source(&nb.id, "a.pdf", "/abs/a.pdf")
            .await
            .unwrap();
        let src_b = repo
            .add_source(&nb.id, "b.pdf", "/abs/b.pdf")
            .await
            .unwrap();

        repo.trash_source(&src_a.source.id).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        repo.trash_source(&src_b.source.id).await.unwrap();

        let trashed = repo.list_trashed_sources().await.unwrap();
        assert_eq!(trashed.len(), 2);
        assert_eq!(
            trashed[0].source.id, src_b.source.id,
            "most recently trashed first"
        );
        assert_eq!(trashed[1].source.id, src_a.source.id);
    }
}
