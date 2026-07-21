//! Per-notebook Audio Overview status persistence (#29). The `audio_overviews` table
//! (migration 0023) holds at most one row per notebook; regenerate overwrites it via
//! UPSERT. Only the terminal states `Ready`/`Failed` are ever stored — `Stale` and
//! `Missing` are read-time derivations (source-set drift and a vanished file) computed
//! by the engine, never written to disk.

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

use crate::error::LensError;

/// Overview lifecycle across the IPC boundary (see module header for the
/// persisted-vs-derived split).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioOverviewStatus {
    Ready,
    Failed,
    Stale,
    Missing,
}

impl AudioOverviewStatus {
    /// TEXT written to `audio_overviews.status`.
    pub(crate) fn as_db_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Failed => "failed",
            Self::Stale => "stale",
            Self::Missing => "missing",
        }
    }

    /// Parses a persisted status; a non-terminal value is a corrupt row.
    pub(crate) fn from_db_str(s: &str) -> Result<Self, LensError> {
        match s {
            "ready" => Ok(Self::Ready),
            "failed" => Ok(Self::Failed),
            other => Err(LensError::Parse(format!(
                "unknown audio overview status: {other}"
            ))),
        }
    }
}

/// A persisted (and read-time reconciled) Audio Overview record, returned across IPC.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioOverviewRecord {
    pub path: String,
    pub generated_at: String,
    pub status: AudioOverviewStatus,
    pub source_set_hash: String,
}

/// [M1] Upsert the terminal row for a notebook. `ON CONFLICT DO UPDATE` (not
/// `DO NOTHING`) so a regenerate overwrites the prior row and a later `ready` clears a
/// prior `failed`.
pub(crate) async fn upsert_overview(
    pool: &SqlitePool,
    notebook_id: &str,
    path: &str,
    generated_at: &str,
    status: AudioOverviewStatus,
    source_set_hash: &str,
) -> Result<(), LensError> {
    sqlx::query(
        "INSERT INTO audio_overviews (notebook_id, path, generated_at, status, source_set_hash) \
         VALUES (?, ?, ?, ?, ?) \
         ON CONFLICT(notebook_id) DO UPDATE SET \
         path = excluded.path, generated_at = excluded.generated_at, \
         status = excluded.status, source_set_hash = excluded.source_set_hash",
    )
    .bind(notebook_id)
    .bind(path)
    .bind(generated_at)
    .bind(status.as_db_str())
    .bind(source_set_hash)
    .execute(pool)
    .await?;
    Ok(())
}

/// Raw read of the stored row (no file reconciliation). Returns
/// `(path, generated_at, status_str, source_set_hash)`.
pub(crate) async fn read_overview_row(
    pool: &SqlitePool,
    notebook_id: &str,
) -> Result<Option<(String, String, String, String)>, LensError> {
    let row = sqlx::query_as::<_, (String, String, String, String)>(
        "SELECT path, generated_at, status, source_set_hash \
         FROM audio_overviews WHERE notebook_id = ?",
    )
    .bind(notebook_id)
    .fetch_optional(pool)
    .await?;
    Ok(row)
}
