//! `lens-core` — the headless engine for LensLM.
//!
//! Pure Rust. Contains no Tauri, windowing, or UI dependencies. All localized
//! file-parsing, database routines, and inference tasks will be implemented here.
//!
//! Domain entities live in per-domain modules (e.g. [`notebooks`]), each owning
//! its struct, id newtype, and a repository over the connection pool. `lib.rs`
//! defines no domain entities itself: [`LensEngine`] is a thin handle that
//! exposes the pool via [`LensEngine::pool`] and delegates to the repos.

pub mod config;
pub(crate) mod db;
pub mod embedding;
pub mod error;
pub mod notebooks;
pub mod system_check;
pub mod tts;

pub use config::AppConfig;
pub use embedding::{InstallProgress, pull_embedding_model};
pub use error::LensError;
pub use notebooks::{Notebook, NotebookId, Source};
pub use system_check::{
    ALLOWED_EMBEDDING_MODELS, CheckAction, CheckId, CheckResult, CheckStatus, LlmDetection,
    detect_llm, ollama_base_url,
};
pub use tts::{
    DownloadProgress, Gender, KOKORO_MODEL_FILENAME, KOKORO_MODEL_RELPATH, KOKORO_MODEL_URL,
    TtsVoice, download_kokoro_model, kokoro_model_path, list_tts_voices,
};

/// Re-exported so the integration-test crate can re-run the migrator against a
/// pool obtained via [`LensEngine::pool`] without exposing the rest of the
/// `pub(crate)` `db` module.
pub use db::run_migrations;

use std::path::Path;
use std::sync::Arc;

use sqlx::SqlitePool;
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::notebooks::NotebookRepo;

/// Mutable engine resources live here: the database connection pool and the
/// loaded application configuration.
///
/// Fields are `pub(crate)` so external code (including the integration-test
/// crate) cannot reach past the [`LensEngine`] API into raw state; use
/// [`LensEngine::pool`] / [`LensEngine::config`] instead.
pub struct LensEngineInner {
    /// Async SQLite connection pool (WAL, foreign keys on).
    pub(crate) db: SqlitePool,
    /// Loaded application configuration (disk-only `config.json`).
    pub(crate) config: AppConfig,
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
    ///
    /// The loaded config's `paths.data_dir` is populated with the resolved data
    /// directory so downstream consumers don't have to re-derive it.
    #[tracing::instrument(skip_all, fields(dir = %data_dir.as_ref().display()))]
    pub async fn init(data_dir: impl AsRef<Path>) -> Result<Self, LensError> {
        let data_dir = data_dir.as_ref();
        std::fs::create_dir_all(data_dir)
            .map_err(|e| LensError::Io(format!("{}: {e}", data_dir.display())))?;
        let db = db::open_pool(data_dir).await?;
        db::run_migrations(&db).await?;
        let mut config = AppConfig::load(data_dir)?;
        config.paths.data_dir = data_dir.display().to_string();
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

    /// Returns a clone of the database connection pool.
    ///
    /// Cloning a `SqlitePool` is cheap (it's an `Arc` internally) and shares the
    /// same underlying connections. This is the canonical way to reach the pool
    /// from repos, commands, and tests — no code should touch `inner.db` directly.
    pub async fn pool(&self) -> SqlitePool {
        self.read().await.db.clone()
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
        let pool = self.pool().await;
        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM _sqlx_migrations")
            .fetch_one(&pool)
            .await?;
        Ok(count)
    }

    /// Runs the three first-run system-check probes and returns the ordered
    /// results (LlmRuntime, EmbeddingModel, TextToSpeech). The LLM-runtime probe
    /// runs first, the embedding probe reuses its outcome, then the TTS probe.
    ///
    /// Probes that detect an expected-absent subsystem return a `Fail` status
    /// rather than an `Err`; this method therefore returns `Ok` unless an
    /// unexpected internal failure occurs. (Today all probe paths are infallible,
    /// but the `Result` signature is the frozen contract for future probes.)
    #[tracing::instrument(skip_all)]
    pub async fn run_system_check(&self) -> Result<Vec<CheckResult>, LensError> {
        // Clone config under the read guard, then DROP the guard before running
        // the probes. The probes issue multi-second HTTP requests; doing so while
        // holding the read guard would block any concurrent writer (`set_config`)
        // for the whole probe window. The clone is cheap.
        let config = self.read().await.config.clone();
        Ok(system_check::run_system_check(&config).await)
    }

    /// Lists all live (non-trashed) notebooks, newest first.
    #[tracing::instrument(skip_all)]
    pub async fn list_notebooks(&self) -> Result<Vec<Notebook>, LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool).list().await
    }

    /// Creates a notebook with the given (validated) title and optional
    /// onboarding `description`/`focus_mode`, and returns it.
    #[tracing::instrument(skip_all)]
    pub async fn create_notebook(
        &self,
        title: &str,
        description: Option<&str>,
        focus_mode: Option<&str>,
    ) -> Result<Notebook, LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool)
            .create(title, description, focus_mode)
            .await
    }

    /// Inserts a file source record for a notebook (M1 onboarding). Returns it.
    #[tracing::instrument(skip_all)]
    pub async fn add_source(
        &self,
        notebook_id: &NotebookId,
        title: &str,
        locator: &str,
    ) -> Result<Source, LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool)
            .add_source(notebook_id, title, locator)
            .await
    }

    /// Lists all sources for a notebook, newest first.
    #[tracing::instrument(skip_all)]
    pub async fn list_sources(&self, notebook_id: &NotebookId) -> Result<Vec<Source>, LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool).list_sources(notebook_id).await
    }

    /// Renames a notebook, bumping `updated_at`.
    #[tracing::instrument(skip_all)]
    pub async fn rename_notebook(&self, id: &NotebookId, title: &str) -> Result<(), LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool).rename(id, title).await
    }

    /// Hard-deletes a notebook. Child rows cascade via `ON DELETE CASCADE`.
    #[tracing::instrument(skip_all)]
    pub async fn delete_notebook(&self, id: &NotebookId) -> Result<(), LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool).delete(id).await
    }
}
