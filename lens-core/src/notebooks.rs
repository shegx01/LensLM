//! Notebook domain: the `Notebook` entity, its strongly-typed id, and the
//! repository implementing CRUD over the `notebooks` table.
//!
//! This module establishes the per-domain repository pattern that M1+ entities
//! (sources, chunks, notes, …) follow: the engine (`lib.rs`) stays thin and owns
//! no domain entities; each domain owns its struct, id newtype, and a repo that
//! takes a `&SqlitePool`. `LensEngine` exposes a `pool()` accessor and delegates.

use std::fmt;
use std::ops::Deref;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use uuid::Uuid;

use crate::LensError;

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
pub(crate) mod source_status {
    /// Inert M1 file record awaiting M4 ingestion.
    pub const PENDING: &str = "pending";
    /// Queued for ingestion (the M4 managed-text entry state).
    pub const QUEUED: &str = "queued";
    /// Parse phase in progress (transient — reset to `error` on crash recovery).
    pub const PARSING: &str = "parsing";
    /// Embed phase in progress (transient — reset to `error` on crash recovery).
    pub const EMBEDDING: &str = "embedding";
    /// Fully ingested and indexed.
    pub const INDEXED: &str = "indexed";
    /// Ingestion failed (terminal until re-ingest).
    pub const ERROR: &str = "error";
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
    /// RFC3339 creation timestamp.
    pub created_at: String,
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
            "SELECT id, title, description, focus_mode, created_at, updated_at, trashed_at \
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
            "SELECT n.id, n.title, n.description, n.focus_mode, n.created_at, n.updated_at, \
                    n.trashed_at, COALESCE(COUNT(s.id), 0) AS source_count \
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
            "SELECT n.id, n.title, n.description, n.focus_mode, n.created_at, n.updated_at, \
                    n.trashed_at, COALESCE(COUNT(s.id), 0) AS source_count \
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
    /// columns `id, title, description, focus_mode, created_at, updated_at,
    /// trashed_at, source_count` in any order.
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
    pub async fn create(
        &self,
        title: &str,
        description: Option<&str>,
        focus_mode: Option<&str>,
    ) -> Result<Notebook, LensError> {
        let title = validate_title(title)?;
        let description = description.map(str::to_string);
        let focus_mode = focus_mode.map(str::to_string);
        let id = NotebookId::new();
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO notebooks (id, title, description, focus_mode, created_at, updated_at, trashed_at) \
             VALUES (?, ?, ?, ?, ?, ?, NULL)",
        )
        .bind(&id)
        .bind(&title)
        .bind(&description)
        .bind(&focus_mode)
        .bind(&now)
        .bind(&now)
        .execute(self.pool)
        .await?;
        Ok(Notebook {
            id,
            title,
            description,
            focus_mode,
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
        .bind(source_status::PENDING)
        .bind(locator)
        .bind(&now)
        .execute(self.pool)
        .await?;
        Ok(Source {
            id,
            notebook_id: notebook_id.to_string(),
            kind: "file".to_string(),
            title: title.to_string(),
            status: source_status::PENDING.to_string(),
            locator: locator.to_string(),
            selected: 1,
            token_count: None,
            content_hash: None,
            created_at: now,
        })
    }

    /// Inserts a managed text/markdown source for M4 ingestion.
    ///
    /// Writes the verbatim `text` to a managed file
    /// `{data_dir}/sources/{id}.{ext}` (ext from `kind`: `text` → `txt`,
    /// `markdown` → `md`), then inserts a `sources` row with `kind ∈
    /// {"text","markdown"}`, `status = "queued"`, `selected = 1`, and `locator`
    /// = that managed file path. Returns the inserted [`Source`] (`token_count`
    /// and `content_hash` are `NULL` until ingestion populates them).
    pub async fn add_text_source(
        &self,
        data_dir: &Path,
        notebook_id: &NotebookId,
        title: &str,
        text: &str,
        kind: &str,
    ) -> Result<Source, LensError> {
        let ext = match kind {
            "text" => "txt",
            "markdown" => "md",
            other => {
                return Err(LensError::Validation(format!(
                    "unknown text source kind: {other:?}; expected \"text\" or \"markdown\""
                )));
            }
        };
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

        sqlx::query(
            "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
        )
        .bind(&id)
        .bind(notebook_id)
        .bind(kind)
        .bind(title)
        .bind(source_status::QUEUED)
        .bind(&locator)
        .bind(&now)
        .execute(self.pool)
        .await?;

        Ok(Source {
            id,
            notebook_id: notebook_id.to_string(),
            kind: kind.to_string(),
            title: title.to_string(),
            status: source_status::QUEUED.to_string(),
            locator,
            selected: 1,
            token_count: None,
            content_hash: None,
            created_at: now,
        })
    }

    /// Hard-deletes a source row by id. Errors if no row matches.
    ///
    /// Callers are responsible for removing any associated Lance vectors before
    /// calling this (Lance before SQLite ordering).
    pub async fn delete_source(&self, id: &str) -> Result<(), LensError> {
        let result = sqlx::query("DELETE FROM sources WHERE id = ?")
            .bind(id)
            .execute(self.pool)
            .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!("no source with id {id}")));
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

    /// Fetches a single source row by id, if it exists.
    pub async fn get_source(&self, id: &str) -> Result<Option<Source>, LensError> {
        let row = sqlx::query_as::<_, Source>(
            "SELECT id, notebook_id, kind, title, status, locator, selected, token_count, content_hash, created_at \
             FROM sources WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(self.pool)
        .await?;
        Ok(row)
    }

    /// Lists all sources for a notebook, newest first.
    pub async fn list_sources(&self, notebook_id: &NotebookId) -> Result<Vec<Source>, LensError> {
        let rows = sqlx::query_as::<_, Source>(
            "SELECT id, notebook_id, kind, title, status, locator, selected, token_count, content_hash, created_at \
             FROM sources WHERE notebook_id = ? ORDER BY created_at DESC",
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
        let nb = repo.create("Notebook", None, None).await.unwrap();

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
    async fn list_with_counts_only_live_newest_first() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let first = repo.create("First", None, None).await.unwrap();
        let second = repo.create("Second", None, None).await.unwrap();

        let summaries = repo.list_with_counts().await.unwrap();
        // Newest created_at first.
        assert_eq!(summaries[0].notebook.id, second.id);
        assert_eq!(summaries[1].notebook.id, first.id);
    }

    #[tokio::test]
    async fn create_rename_roundtrip() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo.create("Original", None, None).await.unwrap();
        repo.rename(&nb.id, "Renamed").await.unwrap();
        let summaries = repo.list_with_counts().await.unwrap();
        assert_eq!(summaries[0].notebook.title, "Renamed");
    }

    #[tokio::test]
    async fn trash_and_restore_roundtrip() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo.create("Notebook", None, None).await.unwrap();

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
        let nb = repo.create("Notebook", None, None).await.unwrap();
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
        let nb = repo.create("Notebook", None, None).await.unwrap();
        assert!(matches!(
            repo.restore(&nb.id).await,
            Err(LensError::Validation(_))
        ));
    }

    #[tokio::test]
    async fn list_trashed_with_counts_carries_source_count() {
        let pool = test_pool().await;
        let repo = NotebookRepo::new(&pool);
        let nb = repo.create("Notebook", None, None).await.unwrap();
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
        let nb = repo.create("Notebook", None, None).await.unwrap();
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
        let nb = repo.create("Notebook", None, None).await.unwrap();

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
        let nb = repo.create("Notebook", None, None).await.unwrap();

        // The deprecated `delete` now soft-deletes (sets trashed_at).
        #[allow(deprecated)]
        repo.delete(&nb.id).await.unwrap();
        assert!(repo.list_with_counts().await.unwrap().is_empty());
        let trashed = repo.list_trashed_with_counts().await.unwrap();
        assert_eq!(trashed.len(), 1);
        assert!(trashed[0].notebook.trashed_at.is_some());
    }

    #[tokio::test]
    async fn notebook_summary_serde_round_trip() {
        let summary = NotebookSummary {
            notebook: Notebook {
                id: NotebookId::from("nb-1"),
                title: "Title".to_string(),
                description: Some("desc".to_string()),
                focus_mode: Some("research".to_string()),
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
}
