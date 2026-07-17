//! Chat persistence (issue #22): grounded Q&A transcript storage.
//!
//! One [`ChatMessage`] row per message. A "turn" is one user row plus its
//! assistant version(s) sharing `turn_id` (the grouping key, minted by the
//! frontend). Only committed truth is persisted: a user row on send, an assistant
//! row on stream `Done`; cancelled/errored turns write nothing. The `citations`
//! JSON payload is owned by the engine and stored verbatim for a lossless
//! round-trip; [`ChatMessage::citations_parsed`] exposes the typed seam for #23.

use std::collections::HashMap;
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

/// Terminal state of a turn that did NOT complete normally (the nullable
/// `chat_messages.state` column). NULL on normal `Done` rows and user rows; a
/// marker row carries `Cancelled` (Stop/superseded) or `Errored` (stream failure).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatState {
    Cancelled,
    Errored,
}

impl ChatState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cancelled => "cancelled",
            Self::Errored => "errored",
        }
    }
}

impl FromStr for ChatState {
    type Err = LensError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cancelled" => Ok(Self::Cancelled),
            "errored" => Ok(Self::Errored),
            other => Err(LensError::Validation(format!(
                "unknown chat state: {other:?}; expected one of \"cancelled\", \"errored\""
            ))),
        }
    }
}

impl TryFrom<String> for ChatState {
    type Error = LensError;

    fn try_from(s: String) -> Result<Self, LensError> {
        s.parse()
    }
}

impl Serialize for ChatState {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ChatState {
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
    /// Terminal marker state (Plan 2): `Some` only on cancelled/errored marker rows.
    pub state: Option<ChatState>,
    /// Sanitized `LensError` kind on an errored marker row; `None` otherwise.
    pub error_kind: Option<String>,
    pub created_at: String,
}

impl<'r> sqlx::FromRow<'r, sqlx::sqlite::SqliteRow> for ChatMessage {
    fn from_row(row: &'r sqlx::sqlite::SqliteRow) -> Result<Self, sqlx::Error> {
        use sqlx::Row;
        let role_str: String = row.try_get("role")?;
        let feedback_str: Option<String> = row.try_get("feedback")?;
        let state_str: Option<String> = row.try_get("state")?;
        let role = role_str
            .parse::<ChatRole>()
            .map_err(|e| sqlx::Error::Decode(Box::new(e)))?;
        let feedback = feedback_str
            .map(|s| s.parse::<ChatFeedback>())
            .transpose()
            .map_err(|e| sqlx::Error::Decode(Box::new(e)))?;
        let state = state_str
            .map(|s| s.parse::<ChatState>())
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
            state,
            error_kind: row.try_get("error_kind")?,
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
            state: None,
            error_kind: None,
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
            state: None,
            error_kind: None,
            created_at,
        })
    }

    /// Inserts a terminal-marker assistant row for a turn that ended without a
    /// normal `Done` (Plan 2 / PC-1): `Cancelled` (Stop / superseded) or `Errored`.
    /// `content` may carry the partial answer streamed so far (empty when none);
    /// `error_kind` is the sanitized `LensError` kind on an errored marker. Shares
    /// the turn's `turn_id` and, like `insert_assistant`, requires the user row so a
    /// bad `turn_id` cannot corrupt grouping.
    pub async fn insert_terminal_marker(
        &self,
        notebook_id: &str,
        turn_id: &str,
        content: &str,
        state: ChatState,
        error_kind: Option<&str>,
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
                "cannot save turn marker: turn has no user message".into(),
            ));
        }

        let id = Uuid::now_v7().to_string();
        let created_at = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO chat_messages \
                 (id, notebook_id, turn_id, role, content, citations, feedback, tokens_used, \
                  state, error_kind, created_at) \
             VALUES (?, ?, ?, 'assistant', ?, NULL, NULL, NULL, ?, ?, ?)",
        )
        .bind(&id)
        .bind(notebook_id)
        .bind(turn_id)
        .bind(content)
        .bind(state.as_str())
        .bind(error_kind)
        .bind(&created_at)
        .execute(self.pool)
        .await?;
        Ok(ChatMessage {
            id,
            notebook_id: notebook_id.to_string(),
            turn_id: turn_id.to_string(),
            role: ChatRole::Assistant,
            content: content.to_string(),
            citations: None,
            feedback: None,
            tokens_used: None,
            state: Some(state),
            error_kind: error_kind.map(str::to_string),
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

    /// Loads prior-conversation history for the prompt/retrieval (CX-1): complete
    /// turns before `current_turn_id` as `LlmMessage`s, newest `limit` (see
    /// [`history_messages`]). Excluding by `current_turn_id` is what keeps the
    /// frontend's just-inserted live question out of history.
    pub async fn history(
        &self,
        notebook_id: &str,
        current_turn_id: &str,
        limit: usize,
    ) -> Result<Vec<crate::llm::LlmMessage>, LensError> {
        let rows = self.list(notebook_id).await?;
        Ok(history_messages(&rows, current_turn_id, limit))
    }

    /// Lists a notebook's messages as flat rows in transcript order. The store
    /// folds these into turns (grouping by `turn_id`).
    pub async fn list(&self, notebook_id: &str) -> Result<Vec<ChatMessage>, LensError> {
        let rows = sqlx::query_as::<_, ChatMessage>(
            "SELECT id, notebook_id, turn_id, role, content, citations, feedback, tokens_used, \
                    state, error_kind, created_at \
             FROM chat_messages WHERE notebook_id = ? \
             ORDER BY created_at, id",
        )
        .bind(notebook_id)
        .fetch_all(self.pool)
        .await?;
        Ok(rows)
    }
}

/// Maps flat transcript rows to prior-conversation history (Plan 2 / CX-1): the
/// COMPLETE turns before `current_turn_id`, each emitted as a `[user, assistant]`
/// pair. Incomplete turns (cancelled/errored marker, or no real answer) are dropped
/// WHOLE — never leaving a dangling user message — so the result is strictly
/// alternating and user-first, which alternation-strict providers require. Bounded to
/// the newest `limit` messages, trimmed in whole pairs. Pure — unit-testable sans DB.
pub fn history_messages(
    rows: &[ChatMessage],
    current_turn_id: &str,
    limit: usize,
) -> Vec<crate::llm::LlmMessage> {
    if limit == 0 {
        return Vec::new();
    }
    // Group rows into turns (first-seen order = creation order of the user row). A
    // regenerate appends a later assistant row to an OLDER turn, so grouping — not a
    // contiguous scan — is required; the newest real answer wins per turn.
    let mut order: Vec<&str> = Vec::new();
    let mut user_of: HashMap<&str, &str> = HashMap::new();
    let mut answer_of: HashMap<&str, &str> = HashMap::new();
    for r in rows {
        if r.turn_id == current_turn_id {
            break; // the current turn (and anything after) is not history
        }
        let turn = r.turn_id.as_str();
        if !user_of.contains_key(turn) && !answer_of.contains_key(turn) {
            order.push(turn);
        }
        match r.role {
            ChatRole::User => {
                user_of.insert(turn, r.content.as_str());
            }
            ChatRole::Assistant => {
                if r.state.is_none() && !r.content.trim().is_empty() {
                    answer_of.insert(turn, r.content.as_str());
                }
            }
        }
    }

    let mut msgs: Vec<crate::llm::LlmMessage> = Vec::new();
    for turn in order {
        if let (Some(u), Some(a)) = (user_of.get(turn), answer_of.get(turn)) {
            msgs.push(crate::llm::LlmMessage {
                role: ChatRole::User,
                content: (*u).to_string(),
            });
            msgs.push(crate::llm::LlmMessage {
                role: ChatRole::Assistant,
                content: (*a).to_string(),
            });
        }
    }
    if msgs.len() > limit {
        // Drop whole pairs from the front so the window stays user-first.
        let drop = (msgs.len() - limit).next_multiple_of(2).min(msgs.len());
        msgs.drain(0..drop);
    }
    msgs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(turn: &str, role: ChatRole, content: &str, state: Option<ChatState>) -> ChatMessage {
        ChatMessage {
            id: format!("id-{turn}-{}", role.as_str()),
            notebook_id: "nb".into(),
            turn_id: turn.into(),
            role,
            content: content.into(),
            citations: None,
            feedback: None,
            tokens_used: None,
            state,
            error_kind: None,
            created_at: "2026-07-17T00:00:00Z".into(),
        }
    }

    #[test]
    fn chat_state_serde_round_trips_wire_strings() {
        for s in [ChatState::Cancelled, ChatState::Errored] {
            let json = serde_json::to_string(&s).unwrap();
            assert_eq!(json, format!("\"{}\"", s.as_str()));
            assert_eq!(serde_json::from_str::<ChatState>(&json).unwrap(), s);
        }
        assert!("bogus".parse::<ChatState>().is_err());
    }

    #[test]
    fn history_excludes_current_turn_and_everything_after() {
        let rows = vec![
            row("t1", ChatRole::User, "q1", None),
            row("t1", ChatRole::Assistant, "a1", None),
            row("t2", ChatRole::User, "q2", None), // current turn (just inserted)
        ];
        let h = history_messages(&rows, "t2", 6);
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].content, "q1");
        assert_eq!(h[1].content, "a1");
    }

    #[test]
    fn history_drops_incomplete_turns_wholesale_no_dangling_user() {
        // A turn with only a cancelled marker (t1) or an empty answer (t2) is
        // incomplete → dropped WHOLE, never leaving a dangling user message that
        // would break user/assistant alternation.
        let rows = vec![
            row("t1", ChatRole::User, "q1", None),
            row("t1", ChatRole::Assistant, "", Some(ChatState::Cancelled)), // marker
            row("t2", ChatRole::User, "q2", None),
            row("t2", ChatRole::Assistant, "  ", None), // empty (whitespace) answer
            row("t3", ChatRole::User, "q3", None),      // current
        ];
        assert!(history_messages(&rows, "t3", 6).is_empty());
    }

    #[test]
    fn history_stays_alternating_across_a_cancelled_turn() {
        // Stop-then-follow-up: t2 was cancelled; history must be the complete t1 pair
        // only — strictly [user, assistant], user-first, no dangling t2 question.
        let rows = vec![
            row("t1", ChatRole::User, "q1", None),
            row("t1", ChatRole::Assistant, "a1", None),
            row("t2", ChatRole::User, "q2", None),
            row(
                "t2",
                ChatRole::Assistant,
                "partial",
                Some(ChatState::Cancelled),
            ),
            row("t3", ChatRole::User, "q3", None), // current
        ];
        let h = history_messages(&rows, "t3", 6);
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].role, ChatRole::User);
        assert_eq!(h[0].content, "q1");
        assert_eq!(h[1].role, ChatRole::Assistant);
        assert_eq!(h[1].content, "a1");
    }

    #[test]
    fn history_uses_newest_answer_after_regenerate() {
        // A regenerate appends a later assistant row to an OLDER turn; the newest real
        // answer wins, and grouping (not a contiguous scan) keeps turn order.
        let rows = vec![
            row("t1", ChatRole::User, "q1", None),
            row("t1", ChatRole::Assistant, "a1-old", None),
            row("t2", ChatRole::User, "q2", None),
            row("t2", ChatRole::Assistant, "a2", None),
            row("t1", ChatRole::Assistant, "a1-new", None), // regenerated later
            row("t3", ChatRole::User, "q3", None),          // current
        ];
        let h = history_messages(&rows, "t3", 6);
        assert_eq!(
            h.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(),
            vec!["q1", "a1-new", "q2", "a2"]
        );
    }

    #[test]
    fn history_bounds_to_newest_limit() {
        let rows = vec![
            row("t1", ChatRole::User, "q1", None),
            row("t1", ChatRole::Assistant, "a1", None),
            row("t2", ChatRole::User, "q2", None),
            row("t2", ChatRole::Assistant, "a2", None),
            row("t3", ChatRole::User, "q3", None), // current
        ];
        let h = history_messages(&rows, "t3", 2);
        // Newest two of {q1,a1,q2,a2} → q2,a2.
        assert_eq!(
            h.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(),
            vec!["q2", "a2"]
        );
        assert!(history_messages(&rows, "t3", 0).is_empty());
    }
}
