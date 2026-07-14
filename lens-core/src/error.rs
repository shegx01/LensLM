use serde::{Deserialize, Serialize};

/// Top-level error type for the LensLM engine.
///
/// Derives `Serialize` so it can cross the Tauri IPC boundary as a structured,
/// programmatically-handleable error (tagged by `kind`) instead of an opaque string.
///
/// Every variant carries a `String` payload so the serialized shape is uniformly
/// `{"kind": <Variant>, "message": <String>}`. New variants are purely additive:
/// the adjacent `#[serde(tag = "kind", content = "message")]` tagging means each
/// variant serializes independently, so adding variants cannot change the wire
/// shape of the existing `Validation`/`Internal` variants (locked by an insta
/// snapshot in `tests/lens_core.rs`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
#[serde(tag = "kind", content = "message")]
pub enum LensError {
    /// Caller supplied invalid input.
    #[error("invalid input: {0}")]
    Validation(String),

    /// An unexpected internal failure.
    #[error("internal error: {0}")]
    Internal(String),

    /// A filesystem / I/O operation failed.
    #[error("io error: {0}")]
    Io(String),

    /// Serialization, deserialization, or other parsing failed.
    #[error("parse error: {0}")]
    Parse(String),

    /// An inference model or model-runtime operation failed.
    #[error("model error: {0}")]
    Model(String),

    /// A network operation failed.
    #[error("network error: {0}")]
    Network(String),

    /// A vector-store (LanceDB / Arrow) operation failed.
    #[error("vector store error: {0}")]
    Vector(String),

    /// An audio/media file used a codec/container this build cannot decode
    /// (issue #41). The payload names the offending extension/codec. Distinct
    /// from [`MediaDecodeFailed`](Self::MediaDecodeFailed): the input is valid
    /// but unsupported, not corrupt.
    #[error("unsupported media codec: {0}")]
    UnsupportedMediaCodec(String),

    /// Decoding an audio/media file failed — the container/bitstream is corrupt,
    /// truncated, or otherwise undecodable (issue #41).
    #[error("media decode failed: {0}")]
    MediaDecodeFailed(String),

    /// An audio file decoded successfully but yielded no usable audio (empty or
    /// all-silent PCM) — there is nothing to transcribe (issue #41).
    #[error("empty audio: {0}")]
    EmptyAudio(String),

    /// Speech-to-text (ASR) failed — model load, inference, or the Apple/Swift
    /// bridge (issue #42). Distinct from the #41 media-decode errors.
    #[error("transcription failed: {0}")]
    Transcription(String),

    /// An operation was cooperatively cancelled by the user (e.g. audio ingest
    /// cancel, issue #43). The payload is a generic description — no path or
    /// internal detail leaks across the IPC boundary.
    #[error("cancelled: {0}")]
    Cancelled(String),

    #[error("tts error: {0}")]
    Tts(String),
}

// Manual `From` mappings (NOT `#[from]`): source error types are not `Serialize`
// and `#[from]` would force the variant to hold the source, breaking the locked
// `{kind, message}` wire shape. We flatten each source into its `String` payload.

impl From<std::io::Error> for LensError {
    fn from(err: std::io::Error) -> Self {
        LensError::Io(err.to_string())
    }
}

impl From<serde_json::Error> for LensError {
    fn from(err: serde_json::Error) -> Self {
        LensError::Parse(err.to_string())
    }
}

impl From<sqlx::Error> for LensError {
    fn from(err: sqlx::Error) -> Self {
        // IPC sanitization: never leak raw sqlx/io strings or filesystem paths
        // across the IPC boundary. Constraint violations carry actionable, safe
        // information (the caller can fix the input), so route those to
        // `Validation` with a generic message; everything else is logged in full
        // for operators and surfaced as an opaque `Internal` error.
        if let sqlx::Error::Database(db_err) = &err {
            if db_err.is_unique_violation() {
                tracing::error!(error = %err, "database unique-constraint violation");
                return LensError::Validation("a record with that value already exists".into());
            }
            if db_err.is_foreign_key_violation() {
                tracing::error!(error = %err, "database foreign-key violation");
                return LensError::Validation("referenced record does not exist".into());
            }
        }
        tracing::error!(error = %err, "database operation failed");
        LensError::Internal("database operation failed".into())
    }
}

impl From<sqlx::migrate::MigrateError> for LensError {
    fn from(err: sqlx::migrate::MigrateError) -> Self {
        // Migrations run at startup, not across IPC, but keep the message opaque
        // and log the detail for operators for consistency with the sqlx mapping.
        tracing::error!(error = %err, "database migration failed");
        LensError::Internal("database migration failed".into())
    }
}

impl From<tokio::task::JoinError> for LensError {
    fn from(err: tokio::task::JoinError) -> Self {
        // A cancelled `spawn_blocking` (e.g. the grounded-answer query-embed task,
        // #173) maps to the cooperative `Cancelled` variant; a panicked task is an
        // unexpected `Internal` failure. Keeps the answer path off `.unwrap()`.
        if err.is_cancelled() {
            LensError::Cancelled("task cancelled".into())
        } else {
            LensError::Internal("background task failed".into())
        }
    }
}

impl LensError {
    /// The stable `kind` discriminant string — identical to the `kind` field of
    /// this error's serialized `{kind, message}` wire shape. Used to build the
    /// persisted [`ErrorMeta`] without a serialize round-trip.
    pub fn kind(&self) -> &'static str {
        match self {
            LensError::Validation(_) => "Validation",
            LensError::Internal(_) => "Internal",
            LensError::Io(_) => "Io",
            LensError::Parse(_) => "Parse",
            LensError::Model(_) => "Model",
            LensError::Network(_) => "Network",
            LensError::Vector(_) => "Vector",
            LensError::UnsupportedMediaCodec(_) => "UnsupportedMediaCodec",
            LensError::MediaDecodeFailed(_) => "MediaDecodeFailed",
            LensError::EmptyAudio(_) => "EmptyAudio",
            LensError::Transcription(_) => "Transcription",
            LensError::Cancelled(_) => "Cancelled",
            LensError::Tts(_) => "Tts",
        }
    }

    /// The inner `String` payload — identical to the `message` field of this
    /// error's serialized `{kind, message}` wire shape (NOT the `Display` text,
    /// which prefixes the kind).
    pub fn message(&self) -> &str {
        match self {
            LensError::Validation(m)
            | LensError::Internal(m)
            | LensError::Io(m)
            | LensError::Parse(m)
            | LensError::Model(m)
            | LensError::Network(m)
            | LensError::Vector(m)
            | LensError::UnsupportedMediaCodec(m)
            | LensError::MediaDecodeFailed(m)
            | LensError::EmptyAudio(m)
            | LensError::Transcription(m)
            | LensError::Cancelled(m)
            | LensError::Tts(m) => m,
        }
    }
}

/// A structured, persisted snapshot of the failure that flipped a source to
/// `status="error"` (issue #73). Serialized to JSON in the nullable
/// `sources.error_meta` TEXT column and surfaced in the UI.
///
/// `kind`/`message` mirror the [`LensError`] `{kind, message}` wire shape via
/// [`ErrorMeta::from_error`]. `attempt_count` is the cumulative number of failed
/// ingest attempts (1 on the first failure; incremented on each failed retry).
/// `timestamp` is the RFC3339 instant of THIS failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorMeta {
    /// The `LensError` discriminant — the stable variant name as a string (e.g.
    /// `"Validation"`, `"Internal"`, `"Io"`, `"Parse"`, `"Model"`, `"Network"`,
    /// `"Vector"`, `"UnsupportedMediaCodec"`, `"MediaDecodeFailed"`,
    /// `"EmptyAudio"`, `"Cancelled"`). Matches the `kind` field of the
    /// `{kind, message}` IPC wire shape. New variants are additive and do not
    /// change existing names.
    pub kind: String,
    /// The human-readable failure message (the `LensError` inner payload).
    pub message: String,
    /// RFC3339 timestamp of this failure.
    pub timestamp: String,
    /// Cumulative failed-attempt count (>= 1); increments on each failed retry.
    pub attempt_count: u32,
}

impl ErrorMeta {
    /// Builds an [`ErrorMeta`] from the failing [`LensError`] plus the new
    /// `attempt_count`, stamping the current time. `attempt_count` is supplied by
    /// the caller (prior count, NULL⇒0, plus one).
    pub fn from_error(err: &LensError, attempt_count: u32) -> Self {
        ErrorMeta {
            kind: err.kind().to_string(),
            message: err.message().to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            attempt_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_meta_serde_round_trip() {
        let meta = ErrorMeta {
            kind: "Network".to_string(),
            message: "connection refused".to_string(),
            timestamp: "2026-07-03T00:00:00+00:00".to_string(),
            attempt_count: 2,
        };
        let json = serde_json::to_string(&meta).unwrap();
        let back: ErrorMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(meta, back);
    }

    #[test]
    fn error_meta_from_lens_error_maps_kind_and_message() {
        let cases = [
            (LensError::Validation("bad".into()), "Validation", "bad"),
            (LensError::Internal("oops".into()), "Internal", "oops"),
            (LensError::Io("disk".into()), "Io", "disk"),
            (LensError::Parse("json".into()), "Parse", "json"),
            (LensError::Model("onnx".into()), "Model", "onnx"),
            (LensError::Network("dns".into()), "Network", "dns"),
            (LensError::Vector("lance".into()), "Vector", "lance"),
            (
                LensError::Transcription("asr".into()),
                "Transcription",
                "asr",
            ),
            (
                LensError::Cancelled("audio ingest cancelled".into()),
                "Cancelled",
                "audio ingest cancelled",
            ),
        ];
        for (err, kind, message) in cases {
            let meta = ErrorMeta::from_error(&err, 1);
            assert_eq!(meta.kind, kind);
            assert_eq!(meta.message, message);
            assert_eq!(meta.attempt_count, 1);
            assert!(!meta.timestamp.is_empty());
        }
    }

    #[tokio::test]
    async fn join_error_cancelled_maps_to_cancelled() {
        let handle = tokio::task::spawn(async {
            std::future::pending::<()>().await;
        });
        handle.abort();
        let join_err = handle.await.expect_err("aborted task yields a JoinError");
        assert!(join_err.is_cancelled());
        assert!(matches!(LensError::from(join_err), LensError::Cancelled(_)));
    }

    #[tokio::test]
    async fn join_error_panic_maps_to_internal() {
        let handle = tokio::task::spawn_blocking(|| panic!("boom"));
        let join_err = handle.await.expect_err("panicked task yields a JoinError");
        assert!(join_err.is_panic());
        assert!(matches!(LensError::from(join_err), LensError::Internal(_)));
    }

    #[test]
    fn error_meta_kind_message_match_serialized_wire_shape() {
        // The `kind()`/`message()` helpers must match the serde `{kind, message}`
        // wire shape (locked by the tagged enum), so the persisted ErrorMeta and
        // the IPC error carry identical discriminants.
        let err = LensError::Model("mismatched vector count".into());
        let wire = serde_json::to_value(&err).unwrap();
        assert_eq!(wire["kind"], err.kind());
        assert_eq!(wire["message"], err.message());
    }
}
