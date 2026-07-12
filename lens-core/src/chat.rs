//! Chat persistence (issue #22): grounded Q&A transcript storage.
//!
//! One [`ChatMessage`] row per message. A "turn" is one user row plus its
//! assistant version(s) sharing `turn_id` (the grouping key, minted by the
//! frontend). Only committed truth is persisted: a user row on send, an assistant
//! row on stream `Done`; cancelled/errored turns write nothing. The `citations`
//! JSON payload is owned by the engine and stored verbatim for a lossless
//! round-trip; [`ChatMessage::citations_parsed`] exposes the typed seam for #23.

use std::str::FromStr;

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use uuid::Uuid;

use crate::LensError;
use crate::citation::Citation;

/// Who authored a chat message (the `chat_messages.role` column). Enum, not a
/// magic string; wire strings MUST NOT change (they are persisted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
}

impl ChatRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

impl FromStr for ChatRole {
    type Err = LensError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            other => Err(LensError::Validation(format!(
                "unknown chat role: {other:?}; expected one of \"user\", \"assistant\""
            ))),
        }
    }
}

impl TryFrom<String> for ChatRole {
    type Error = LensError;

    fn try_from(s: String) -> Result<Self, LensError> {
        s.parse()
    }
}

impl Serialize for ChatRole {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ChatRole {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// User feedback on an assistant message (the nullable `chat_messages.feedback`
/// column). NULL = no feedback. Mutually exclusive and toggleable back to `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFeedback {
    Up,
    Down,
}

impl ChatFeedback {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
        }
    }
}

impl FromStr for ChatFeedback {
    type Err = LensError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "up" => Ok(Self::Up),
            "down" => Ok(Self::Down),
            other => Err(LensError::Validation(format!(
                "unknown chat feedback: {other:?}; expected one of \"up\", \"down\""
            ))),
        }
    }
}

impl TryFrom<String> for ChatFeedback {
    type Error = LensError;

    fn try_from(s: String) -> Result<Self, LensError> {
        s.parse()
    }
}

impl Serialize for ChatFeedback {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ChatFeedback {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// A chat message row, returned across the IPC boundary. `citations` is kept as
/// the raw JSON `TEXT` column for a lossless round-trip; [`citations_parsed`]
/// deserializes it into `Vec<Citation>` for #23.
///
/// `FromRow` is hand-written (not derived): `feedback` is a nullable enum column,
/// and sqlx's `try_from` derive has no `TryFrom<Option<String>>` for the fallible
/// `Option<ChatFeedback>` conversion.
///
/// [`citations_parsed`]: ChatMessage::citations_parsed
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: String,
    pub notebook_id: String,
    pub turn_id: String,
    pub role: ChatRole,
    pub content: String,
    /// Raw JSON `Vec<Citation>` (assistant rows only); `None` for user rows.
    pub citations: Option<String>,
    pub feedback: Option<ChatFeedback>,
    pub tokens_used: Option<i64>,
    pub created_at: String,
}

impl<'r> sqlx::FromRow<'r, sqlx::sqlite::SqliteRow> for ChatMessage {
    fn from_row(row: &'r sqlx::sqlite::SqliteRow) -> Result<Self, sqlx::Error> {
        use sqlx::Row;
        let role_str: String = row.try_get("role")?;
        let feedback_str: Option<String> = row.try_get("feedback")?;
        let role = role_str
            .parse::<ChatRole>()
            .map_err(|e| sqlx::Error::Decode(Box::new(e)))?;
        let feedback = feedback_str
            .map(|s| s.parse::<ChatFeedback>())
            .transpose()
            .map_err(|e| sqlx::Error::Decode(Box::new(e)))?;
        Ok(Self {
            id: row.try_get("id")?,
            notebook_id: row.try_get("notebook_id")?,
            turn_id: row.try_get("turn_id")?,
            role,
            content: row.try_get("content")?,
            citations: row.try_get("citations")?,
            feedback,
            tokens_used: row.try_get("tokens_used")?,
            created_at: row.try_get("created_at")?,
        })
    }
}

impl ChatMessage {
    /// Deserializes the stored `citations` JSON into the typed payload (the #23
    /// seam). `None` when the column is NULL (user rows / no citations).
    pub fn citations_parsed(&self) -> Result<Option<Vec<Citation>>, LensError> {
        match &self.citations {
            None => Ok(None),
            Some(json) => serde_json::from_str(json)
                .map(Some)
                .map_err(|e| LensError::Internal(format!("citations deserialize failed: {e}"))),
        }
    }
}

/// Chat persistence over the shared pool. Mirrors `NotebookRepo`: a borrowed
/// handle whose methods run one `sqlx` query each and mint UUIDv7 row ids.
pub struct ChatRepo<'a> {
    pool: &'a SqlitePool,
}

impl<'a> ChatRepo<'a> {
    pub fn new(pool: &'a SqlitePool) -> Self {
        Self { pool }
    }

    /// Inserts a user message on send (committed truth). `turn_id` is the
    /// frontend-minted grouping key the later assistant row will share.
    pub async fn insert_user(
        &self,
        notebook_id: &str,
        turn_id: &str,
        content: &str,
    ) -> Result<ChatMessage, LensError> {
        let id = Uuid::now_v7().to_string();
        let created_at = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO chat_messages \
                 (id, notebook_id, turn_id, role, content, citations, feedback, tokens_used, \
                  created_at) \
             VALUES (?, ?, ?, 'user', ?, NULL, NULL, NULL, ?)",
        )
        .bind(&id)
        .bind(notebook_id)
        .bind(turn_id)
        .bind(content)
        .bind(&created_at)
        .execute(self.pool)
        .await?;
        Ok(ChatMessage {
            id,
            notebook_id: notebook_id.to_string(),
            turn_id: turn_id.to_string(),
            role: ChatRole::User,
            content: content.to_string(),
            citations: None,
            feedback: None,
            tokens_used: None,
            created_at,
        })
    }

    /// Inserts an assistant message on stream `Done`. `citations_json` is the
    /// pre-serialized `Vec<Citation>` payload (engine owns serialization).
    ///
    /// Guards turn integrity: rejects an insert whose `turn_id` has no `user` row
    /// in this notebook (a bad frontend `turn_id` would silently corrupt grouping).
    pub async fn insert_assistant(
        &self,
        notebook_id: &str,
        turn_id: &str,
        content: &str,
        citations_json: Option<&str>,
        tokens_used: i64,
    ) -> Result<ChatMessage, LensError> {
        let has_user: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM chat_messages \
             WHERE notebook_id = ? AND turn_id = ? AND role = 'user')",
        )
        .bind(notebook_id)
        .bind(turn_id)
        .fetch_one(self.pool)
        .await?;
        if !has_user {
            return Err(LensError::Validation(
                "cannot save assistant message: turn has no user message".into(),
            ));
        }

        let id = Uuid::now_v7().to_string();
        let created_at = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO chat_messages \
                 (id, notebook_id, turn_id, role, content, citations, feedback, tokens_used, \
                  created_at) \
             VALUES (?, ?, ?, 'assistant', ?, ?, NULL, ?, ?)",
        )
        .bind(&id)
        .bind(notebook_id)
        .bind(turn_id)
        .bind(content)
        .bind(citations_json)
        .bind(tokens_used)
        .bind(&created_at)
        .execute(self.pool)
        .await?;
        Ok(ChatMessage {
            id,
            notebook_id: notebook_id.to_string(),
            turn_id: turn_id.to_string(),
            role: ChatRole::Assistant,
            content: content.to_string(),
            citations: citations_json.map(str::to_string),
            feedback: None,
            tokens_used: Some(tokens_used),
            created_at,
        })
    }

    /// Sets (or clears, with `None`) the feedback on a message. Toggleable.
    pub async fn set_feedback(
        &self,
        message_id: &str,
        feedback: Option<ChatFeedback>,
    ) -> Result<(), LensError> {
        sqlx::query("UPDATE chat_messages SET feedback = ? WHERE id = ?")
            .bind(feedback.map(|f| f.as_str()))
            .bind(message_id)
            .execute(self.pool)
            .await?;
        Ok(())
    }

    /// Lists a notebook's messages as flat rows in transcript order. The store
    /// folds these into turns (grouping by `turn_id`).
    pub async fn list(&self, notebook_id: &str) -> Result<Vec<ChatMessage>, LensError> {
        let rows = sqlx::query_as::<_, ChatMessage>(
            "SELECT id, notebook_id, turn_id, role, content, citations, feedback, tokens_used, \
                    created_at \
             FROM chat_messages WHERE notebook_id = ? \
             ORDER BY created_at, id",
        )
        .bind(notebook_id)
        .fetch_all(self.pool)
        .await?;
        Ok(rows)
    }
}
