//! SQLite connection-pool construction and migration application.
//!
//! Uses sqlx with bundled SQLite (no system library required) and the embedded
//! `migrate!` macro so the binary stays self-contained.

use std::path::Path;
use std::str::FromStr;

use sqlx::SqlitePool;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};

use crate::LensError;

/// File name of the on-disk database within the engine data directory.
const DB_FILE_NAME: &str = "lens.db";

/// Embedded migrator over `lens-core/migrations/`. Files are compiled into the
/// binary at build time and applied idempotently; each file runs in its own
/// transaction (one file = one atomic unit).
static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// Opens (creating if necessary) the on-disk pool at `{data_dir}/lens.db` with
/// WAL journaling and foreign-key enforcement enabled.
#[tracing::instrument(skip_all, fields(dir = %data_dir.display()))]
pub async fn open_pool(data_dir: &Path) -> Result<SqlitePool, LensError> {
    let db_path = data_dir.join(DB_FILE_NAME);
    let options = SqliteConnectOptions::new()
        .filename(&db_path)
        .create_if_missing(true)
        .journal_mode(SqliteJournalMode::Wal)
        .foreign_keys(true);
    let pool = SqlitePool::connect_with(options).await?;
    tracing::info!("opened db at {}", db_path.display());
    Ok(pool)
}

/// Opens an in-memory pool limited to a single connection.
///
/// SQLite `:memory:` databases are per-connection, so a multi-connection pool
/// would lose the migrated schema whenever a query checked out a fresh, empty
/// connection. Capping `max_connections(1)` guarantees every query reuses the
/// one connection that ran the migrations. For tests needing concurrency, use a
/// `tempfile`-backed on-disk DB via [`open_pool`] instead.
pub async fn open_in_memory_pool() -> Result<SqlitePool, LensError> {
    let options = SqliteConnectOptions::from_str("sqlite::memory:")
        .map_err(|e| LensError::Internal(e.to_string()))?
        .foreign_keys(true);
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(options)
        .await?;
    Ok(pool)
}

/// Applies all pending embedded migrations. Idempotent: re-running is a no-op.
#[tracing::instrument(skip_all)]
pub async fn run_migrations(pool: &SqlitePool) -> Result<(), LensError> {
    MIGRATOR.run(pool).await?;
    tracing::info!("migrations applied ({} known)", MIGRATOR.iter().count());
    Ok(())
}
