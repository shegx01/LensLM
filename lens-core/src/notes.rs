//! Notes persistence (issue #24): durable, source-grounded note snapshots.
//!
//! A `origin=chat` [`Note`] freezes a grounded answer at save time: `content`,
//! `citations` JSON, and denormalized `source_title` survive later source
//! deletion/rename. `source_message_id` links back to the originating
//! `chat_messages` row (see 0019 migration header for the no-FK rationale). The
//! `citations` JSON mirrors `chat_messages.citations` (0018) verbatim for a
//! lossless round-trip; [`Note::citations_parsed`] exposes the typed seam.

use std::ops::Deref;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use uuid::Uuid;

use crate::LensError;
use crate::citation::Citation;

/// Strongly-typed note identifier (UUIDv7 stored as TEXT). Mirrors
/// [`NotebookId`](crate::NotebookId).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type)]
#[serde(transparent)]
#[sqlx(transparent)]
pub struct NoteId(pub(crate) String);

impl NoteId {
    pub fn new() -> Self {
        Self(Uuid::now_v7().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for NoteId {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for NoteId {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<String> for NoteId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for NoteId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// How a note originated (the `notes.origin` column). Enum, not a magic string;
/// wire strings MUST NOT change (they are persisted). `Manual` is #25's domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteOrigin {
    Chat,
    Manual,
}

impl NoteOrigin {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Manual => "manual",
        }
    }
}

impl FromStr for NoteOrigin {
    type Err = LensError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "chat" => Ok(Self::Chat),
            "manual" => Ok(Self::Manual),
            other => Err(LensError::Validation(format!(
                "unknown note origin: {other:?}; expected one of \"chat\", \"manual\""
            ))),
        }
    }
}

impl TryFrom<String> for NoteOrigin {
    type Error = LensError;

    fn try_from(s: String) -> Result<Self, LensError> {
        s.parse()
    }
}

impl Serialize for NoteOrigin {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for NoteOrigin {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// A note row, returned across the IPC boundary. `citations` is kept as the raw
/// JSON `TEXT` column for a lossless round-trip; [`citations_parsed`] deserializes
/// it into `Vec<Citation>`.
///
/// `FromRow` is hand-written (not derived): `origin` is a fallible enum column and
/// sqlx's `try_from` derive has no `TryFrom<String>` path for it.
///
/// [`citations_parsed`]: Note::citations_parsed
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Note {
    pub id: String,
    pub notebook_id: String,
    pub origin: NoteOrigin,
    pub content: String,
    /// Raw JSON `Vec<Citation>` snapshot (chat notes); `None` when uncited/manual.
    pub citations: Option<String>,
    /// Frozen ordinal-1 source title.
    pub source_title: Option<String>,
    /// Toggle-linkage key to the originating `chat_messages.id`.
    pub source_message_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

impl<'r> sqlx::FromRow<'r, sqlx::sqlite::SqliteRow> for Note {
    fn from_row(row: &'r sqlx::sqlite::SqliteRow) -> Result<Self, sqlx::Error> {
        use sqlx::Row;
        let origin_str: String = row.try_get("origin")?;
        let origin = origin_str
            .parse::<NoteOrigin>()
            .map_err(|e| sqlx::Error::Decode(Box::new(e)))?;
        Ok(Self {
            id: row.try_get("id")?,
            notebook_id: row.try_get("notebook_id")?,
            origin,
            content: row.try_get("content")?,
            citations: row.try_get("citations")?,
            source_title: row.try_get("source_title")?,
            source_message_id: row.try_get("source_message_id")?,
            created_at: row.try_get("created_at")?,
            updated_at: row.try_get("updated_at")?,
        })
    }
}

impl Note {
    /// Deserializes the stored `citations` JSON into the typed payload. `None` when
    /// the column is NULL (uncited chat notes / manual notes).
    pub fn citations_parsed(&self) -> Result<Option<Vec<Citation>>, LensError> {
        match &self.citations {
            None => Ok(None),
            Some(json) => serde_json::from_str(json)
                .map(Some)
                .map_err(|e| LensError::Internal(format!("citations deserialize failed: {e}"))),
        }
    }
}

/// Notes persistence over the shared pool. Mirrors `ChatRepo`/`NotebookRepo`: a
/// borrowed handle whose methods run one `sqlx` query each and mint UUIDv7 ids.
pub struct NotesRepo<'a> {
    pool: &'a SqlitePool,
}

impl<'a> NotesRepo<'a> {
    pub fn new(pool: &'a SqlitePool) -> Self {
        Self { pool }
    }

    /// Inserts an `origin=chat` note snapshot. `citations_json` is the
    /// pre-serialized `Vec<Citation>` payload (engine owns serialization);
    /// `source_title` is the frozen ordinal-1 source title. `updated_at` equals
    /// `created_at` at insert (the column is `NOT NULL`, 0001:81).
    pub async fn create_chat_note(
        &self,
        notebook_id: &str,
        content: &str,
        citations_json: Option<&str>,
        source_title: Option<&str>,
        source_message_id: &str,
    ) -> Result<Note, LensError> {
        let id = Uuid::now_v7().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        self.insert_note(
            &id,
            notebook_id,
            content,
            NoteOrigin::Chat,
            citations_json,
            source_title,
            Some(source_message_id),
            &now,
        )
        .await?;
        Ok(Note {
            id,
            notebook_id: notebook_id.to_string(),
            origin: NoteOrigin::Chat,
            content: content.to_string(),
            citations: citations_json.map(str::to_string),
            source_title: source_title.map(str::to_string),
            source_message_id: Some(source_message_id.to_string()),
            created_at: now.clone(),
            updated_at: now,
        })
    }

    /// Inserts one `notes` row with the shared 9-column scaffold. `updated_at`
    /// equals `created_at` at insert (the column is `NOT NULL`, 0001:81); the
    /// public `create_*` methods own id/timestamp minting and the returned `Note`.
    #[allow(clippy::too_many_arguments)]
    async fn insert_note(
        &self,
        id: &str,
        notebook_id: &str,
        content: &str,
        origin: NoteOrigin,
        citations: Option<&str>,
        source_title: Option<&str>,
        source_message_id: Option<&str>,
        now: &str,
    ) -> Result<(), LensError> {
        sqlx::query(
            "INSERT INTO notes \
                 (id, notebook_id, content, origin, citations, source_title, \
                  source_message_id, created_at, updated_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(id)
        .bind(notebook_id)
        .bind(content)
        .bind(origin.as_str())
        .bind(citations)
        .bind(source_title)
        .bind(source_message_id)
        .bind(now)
        .bind(now)
        .execute(self.pool)
        .await?;
        Ok(())
    }

    /// Inserts an `origin=manual` note (#25). Citations, `source_title`, and
    /// `source_message_id` are always NULL — a manual note has no grounding.
    /// `updated_at` equals `created_at` at insert.
    pub async fn create_manual_note(
        &self,
        notebook_id: &str,
        content: &str,
    ) -> Result<Note, LensError> {
        let id = Uuid::now_v7().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        self.insert_note(
            &id,
            notebook_id,
            content,
            NoteOrigin::Manual,
            None,
            None,
            None,
            &now,
        )
        .await?;
        Ok(Note {
            id,
            notebook_id: notebook_id.to_string(),
            origin: NoteOrigin::Manual,
            content: content.to_string(),
            citations: None,
            source_title: None,
            source_message_id: None,
            created_at: now.clone(),
            updated_at: now,
        })
    }

    /// Lists a notebook's notes, newest first.
    pub async fn list_notes(&self, notebook_id: &str) -> Result<Vec<Note>, LensError> {
        let rows = sqlx::query_as::<_, Note>(
            "SELECT id, notebook_id, origin, content, citations, source_title, \
                    source_message_id, created_at, updated_at \
             FROM notes WHERE notebook_id = ? \
             ORDER BY created_at DESC, id DESC",
        )
        .bind(notebook_id)
        .fetch_all(self.pool)
        .await?;
        Ok(rows)
    }

    /// Deletes a note by id (idempotent — no error if the row is absent).
    pub async fn delete_note(&self, id: &str) -> Result<(), LensError> {
        sqlx::query("DELETE FROM notes WHERE id = ?")
            .bind(id)
            .execute(self.pool)
            .await?;
        Ok(())
    }
}
