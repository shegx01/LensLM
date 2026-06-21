use serde::Serialize;

/// Top-level error type for the LensLM engine.
///
/// Derives `Serialize` so it can cross the Tauri IPC boundary as a structured,
/// programmatically-handleable error (tagged by `kind`) instead of an opaque string.
#[derive(Debug, thiserror::Error, Serialize)]
#[serde(tag = "kind", content = "message")]
pub enum LensError {
    /// Caller supplied invalid input.
    #[error("invalid input: {0}")]
    Validation(String),

    /// An unexpected internal failure.
    #[error("internal error: {0}")]
    Internal(String),
}
