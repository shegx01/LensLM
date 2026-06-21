//! Notebook domain: the `Notebook` entity, its strongly-typed id, and the
//! repository implementing CRUD over the `notebooks` table.
//!
//! This module establishes the per-domain repository pattern that M1+ entities
//! (sources, chunks, notes, …) follow: the engine (`lib.rs`) stays thin and owns
//! no domain entities; each domain owns its struct, id newtype, and a repo that
//! takes a `&SqlitePool`. `LensEngine` exposes a `pool()` accessor and delegates.

use std::fmt;
use std::ops::Deref;

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
pub struct NotebookId(pub String);

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

/// A notebook row, returned across the IPC boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, sqlx::FromRow)]
pub struct Notebook {
    /// UUIDv7 primary key, stored as TEXT.
    pub id: NotebookId,
    /// Display title.
    pub title: String,
    /// RFC3339 creation timestamp.
    pub created_at: String,
    /// RFC3339 last-update timestamp.
    pub updated_at: String,
    /// RFC3339 soft-delete timestamp, or `None` if live.
    pub trashed_at: Option<String>,
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
            "SELECT id, title, created_at, updated_at, trashed_at \
             FROM notebooks WHERE trashed_at IS NULL ORDER BY created_at DESC",
        )
        .fetch_all(self.pool)
        .await?;
        Ok(rows)
    }

    /// Creates a notebook with a freshly-minted UUIDv7 id and returns it.
    ///
    /// The title is trimmed and validated (non-empty, length-capped).
    pub async fn create(&self, title: &str) -> Result<Notebook, LensError> {
        let title = validate_title(title)?;
        let id = NotebookId::new();
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO notebooks (id, title, created_at, updated_at, trashed_at) \
             VALUES (?, ?, ?, ?, NULL)",
        )
        .bind(&id)
        .bind(&title)
        .bind(&now)
        .bind(&now)
        .execute(self.pool)
        .await?;
        Ok(Notebook {
            id,
            title,
            created_at: now.clone(),
            updated_at: now,
            trashed_at: None,
        })
    }

    /// Renames a notebook, bumping `updated_at`. The title is validated.
    pub async fn rename(&self, id: &NotebookId, title: &str) -> Result<(), LensError> {
        let title = validate_title(title)?;
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query("UPDATE notebooks SET title = ?, updated_at = ? WHERE id = ?")
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

    /// Hard-deletes a notebook. Child rows cascade via `ON DELETE CASCADE`.
    pub async fn delete(&self, id: &NotebookId) -> Result<(), LensError> {
        let result = sqlx::query("DELETE FROM notebooks WHERE id = ?")
            .bind(id)
            .execute(self.pool)
            .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!("no notebook with id {id}")));
        }
        Ok(())
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
}
