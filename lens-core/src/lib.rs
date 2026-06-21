//! `lens-core` — the headless engine for LensLM.
//!
//! Pure Rust. Contains no Tauri, windowing, or UI dependencies. All localized
//! file-parsing, database routines, and inference tasks will be implemented here.

pub mod config;
pub mod db;
pub mod error;

pub use config::AppConfig;
pub use error::LensError;

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use uuid::Uuid;

/// A notebook row, returned across the IPC boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, sqlx::FromRow)]
pub struct Notebook {
    /// UUIDv7 primary key, stored as TEXT.
    pub id: String,
    /// Display title.
    pub title: String,
    /// RFC3339 creation timestamp.
    pub created_at: String,
    /// RFC3339 last-update timestamp.
    pub updated_at: String,
    /// RFC3339 soft-delete timestamp, or `None` if live.
    pub trashed_at: Option<String>,
}

/// Mutable engine resources live here: the database connection pool and the
/// loaded application configuration.
pub struct LensEngineInner {
    /// Async SQLite connection pool (WAL, foreign keys on).
    pub db: SqlitePool,
    /// Loaded application configuration (disk-only `config.json`).
    pub config: AppConfig,
}

/// Thread-safe, cheaply-cloneable handle to the LensLM engine state.
///
/// Cloning shares the same underlying state (`Arc`). Mutations go through an
/// async-aware `RwLock` so guards can be safely held across `.await` points —
/// this is the interior mutability Tauri's immutable `State<T>` requires.
#[derive(Clone)]
pub struct LensEngine {
    inner: Arc<RwLock<LensEngineInner>>,
}

impl LensEngine {
    /// Production constructor: ensures the data directory exists, opens the
    /// on-disk pool (WAL + foreign keys), applies migrations, and loads (or
    /// initializes) `config.json`.
    #[tracing::instrument(skip_all, fields(dir = %data_dir.as_ref().display()))]
    pub async fn init(data_dir: impl AsRef<Path>) -> Result<Self, LensError> {
        let data_dir = data_dir.as_ref();
        std::fs::create_dir_all(data_dir)
            .map_err(|e| LensError::Io(format!("{}: {e}", data_dir.display())))?;
        let db = db::open_pool(data_dir).await?;
        db::run_migrations(&db).await?;
        let config = AppConfig::load(data_dir)?;
        tracing::info!("engine initialized");
        Ok(Self {
            inner: Arc::new(RwLock::new(LensEngineInner { db, config })),
        })
    }

    /// Test constructor: a fully-migrated in-memory engine with a default config.
    ///
    /// Uses a single-connection in-memory pool so the migrated schema persists
    /// across all queries (`:memory:` is per-connection in SQLite). See
    /// [`db::open_in_memory_pool`]. Tests needing real concurrency should build
    /// an engine over a `tempfile` directory via [`LensEngine::init`].
    pub async fn for_test() -> Self {
        let db = db::open_in_memory_pool()
            .await
            .expect("in-memory pool should open");
        db::run_migrations(&db)
            .await
            .expect("migrations should apply to a fresh in-memory db");
        Self {
            inner: Arc::new(RwLock::new(LensEngineInner {
                db,
                config: AppConfig::default(),
            })),
        }
    }

    /// Acquires a shared read guard over the engine state.
    pub async fn read(&self) -> RwLockReadGuard<'_, LensEngineInner> {
        self.inner.read().await
    }

    /// Acquires an exclusive write guard over the engine state.
    pub async fn write(&self) -> RwLockWriteGuard<'_, LensEngineInner> {
        self.inner.write().await
    }

    /// Returns a clone of the current application configuration.
    pub async fn config(&self) -> AppConfig {
        self.read().await.config.clone()
    }

    /// Replaces the in-memory configuration. Persistence to disk is the caller's
    /// responsibility (the production command layer saves to `config.json`).
    pub async fn set_config(&self, config: AppConfig) {
        self.write().await.config = config;
    }

    /// Returns the number of migrations applied to the live database.
    #[tracing::instrument(skip_all)]
    pub async fn migration_count(&self) -> Result<i64, LensError> {
        let pool = self.read().await.db.clone();
        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM _sqlx_migrations")
            .fetch_one(&pool)
            .await?;
        Ok(count)
    }

    /// Lists all live (non-trashed) notebooks, newest first.
    #[tracing::instrument(skip_all)]
    pub async fn list_notebooks(&self) -> Result<Vec<Notebook>, LensError> {
        let pool = self.read().await.db.clone();
        let rows = sqlx::query_as::<_, Notebook>(
            "SELECT id, title, created_at, updated_at, trashed_at \
             FROM notebooks WHERE trashed_at IS NULL ORDER BY created_at DESC",
        )
        .fetch_all(&pool)
        .await?;
        Ok(rows)
    }

    /// Creates a notebook with a freshly-minted UUIDv7 id and returns it.
    #[tracing::instrument(skip_all)]
    pub async fn create_notebook(&self, title: &str) -> Result<Notebook, LensError> {
        let pool = self.read().await.db.clone();
        let id = Uuid::now_v7().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO notebooks (id, title, created_at, updated_at, trashed_at) \
             VALUES (?, ?, ?, ?, NULL)",
        )
        .bind(&id)
        .bind(title)
        .bind(&now)
        .bind(&now)
        .execute(&pool)
        .await?;
        Ok(Notebook {
            id,
            title: title.to_string(),
            created_at: now.clone(),
            updated_at: now,
            trashed_at: None,
        })
    }

    /// Renames a notebook, bumping `updated_at`.
    #[tracing::instrument(skip_all)]
    pub async fn rename_notebook(&self, id: &str, title: &str) -> Result<(), LensError> {
        let pool = self.read().await.db.clone();
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query("UPDATE notebooks SET title = ?, updated_at = ? WHERE id = ?")
            .bind(title)
            .bind(&now)
            .bind(id)
            .execute(&pool)
            .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!("no notebook with id {id}")));
        }
        Ok(())
    }

    /// Hard-deletes a notebook. Child rows cascade via `ON DELETE CASCADE`.
    #[tracing::instrument(skip_all)]
    pub async fn delete_notebook(&self, id: &str) -> Result<(), LensError> {
        let pool = self.read().await.db.clone();
        let result = sqlx::query("DELETE FROM notebooks WHERE id = ?")
            .bind(id)
            .execute(&pool)
            .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!("no notebook with id {id}")));
        }
        Ok(())
    }
}
