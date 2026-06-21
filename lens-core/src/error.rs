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
