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

/// Embedded migrator over the crate's `./migrations` directory. Files are
/// compiled into the binary at build time and applied idempotently; each file
/// runs in its own transaction (one file = one atomic unit).
static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// Maximum number of pooled connections for the production on-disk database.
const MAX_CONNECTIONS: u32 = 5;

/// Busy-timeout (milliseconds) applied to each connection: how long SQLite waits
/// on a locked database before returning `SQLITE_BUSY` instead of failing fast.
const BUSY_TIMEOUT_MS: u64 = 5_000;

/// Opens (creating if necessary) the on-disk pool at `{data_dir}/lens.db` with
/// WAL journaling, foreign-key enforcement, a bounded connection count, and a
/// busy-timeout so concurrent writers wait rather than erroring on lock.
#[tracing::instrument(skip_all, fields(dir = %data_dir.display()))]
pub async fn open_pool(data_dir: &Path) -> Result<SqlitePool, LensError> {
    let db_path = data_dir.join(DB_FILE_NAME);
    let options = SqliteConnectOptions::new()
        .filename(&db_path)
        .create_if_missing(true)
        .journal_mode(SqliteJournalMode::Wal)
        .busy_timeout(std::time::Duration::from_millis(BUSY_TIMEOUT_MS))
        .foreign_keys(true);
    let pool = SqlitePoolOptions::new()
        .max_connections(MAX_CONNECTIONS)
        .connect_with(options)
        .await?;
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
        .map_err(|e| LensError::Parse(e.to_string()))?
        .foreign_keys(true);
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(options)
        .await?;
    Ok(pool)
}

/// Max ids per batched `IN (…)` query. SQLite's default bind-variable limit is
/// 999; stay well under it so large id lists don't overflow the parameter cap.
pub(crate) const BIND_BATCH: usize = 500;

/// Comma-joined `?` placeholders for a SQLite `IN (…)` clause of length `n`
/// (`n = 3` → `"?,?,?"`). Only `?` characters — never user data — so the result
/// is injection-safe; bind the values separately. Pair with [`BIND_BATCH`].
pub(crate) fn in_placeholders(n: usize) -> String {
    std::iter::repeat_n("?", n).collect::<Vec<_>>().join(",")
}

/// Runs a batched `IN (…)` query over `ids`, binding each id in order. `sql_for`
/// receives the placeholder string and returns the full SQL (the only variable part
/// besides the ids is static text). Owns the `BIND_BATCH` / `in_placeholders` /
/// bind-loop invariant so the seven read-path call sites don't each re-implement it.
pub(crate) async fn fetch_batched<T>(
    pool: &SqlitePool,
    ids: &[String],
    sql_for: impl Fn(&str) -> String,
) -> Result<Vec<T>, LensError>
where
    T: for<'r> sqlx::FromRow<'r, sqlx::sqlite::SqliteRow> + Send + Unpin,
{
    let mut out = Vec::new();
    for batch in ids.chunks(BIND_BATCH) {
        let sql = sql_for(&in_placeholders(batch.len()));
        let mut q = sqlx::query_as::<_, T>(&sql);
        for id in batch {
            q = q.bind(id);
        }
        out.extend(q.fetch_all(pool).await?);
    }
    Ok(out)
}

/// Applies all pending embedded migrations. Idempotent: re-running is a no-op.
#[tracing::instrument(skip_all)]
pub async fn run_migrations(pool: &SqlitePool) -> Result<(), LensError> {
    MIGRATOR.run(pool).await?;
    tracing::info!("migrations applied ({} known)", MIGRATOR.iter().count());
    Ok(())
}
