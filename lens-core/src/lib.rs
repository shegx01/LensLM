//! `lens-core` — the headless engine for LensLM.
//!
//! Pure Rust. Contains no Tauri, windowing, or UI dependencies. All localized
//! file-parsing, database routines, and inference tasks will be implemented here.
//!
//! Domain entities live in per-domain modules (e.g. [`notebooks`]), each owning
//! its struct, id newtype, and a repository over the connection pool. `lib.rs`
//! defines no domain entities itself: [`LensEngine`] is a thin handle that
//! exposes the pool via [`LensEngine::pool`] and delegates to the repos.

pub mod chunk;
pub mod config;
pub(crate) mod db;
pub mod embedder;
pub mod embedding;
pub mod error;
pub mod extract;
pub mod ingest;
pub mod notebooks;
pub mod parse;
pub mod system_check;
pub mod tts;
pub mod vector_store;

pub use config::AppConfig;
pub use embedder::{CountingEmbedder, EMBED_DIM, EMBED_MODEL_ID, Embedder, FastembedEmbedder};
pub use embedding::{InstallProgress, pull_embedding_model};
pub use error::LensError;
pub use extract::{ExtractOutput, Extractor, SourceAnchor, extractor_for, is_text_like_kind};
pub use ingest::{
    IngestProgress, NEEDS_JS_MIN_CHARS, NEEDS_JS_MIN_TEXT_RATIO, URL_FETCH_TIMEOUT, ingest_source,
    resolve_nomic_tokenizer,
};
pub use notebooks::{Notebook, NotebookId, NotebookSummary, Source};
pub use system_check::{
    ALLOWED_EMBEDDING_MODELS, CheckAction, CheckId, CheckResult, CheckStatus, LlmDetection,
    detect_llm, ollama_base_url,
};
pub use tts::{
    DownloadProgress, Gender, KOKORO_MODEL_FILENAME, KOKORO_MODEL_RELPATH, KOKORO_MODEL_URL,
    TtsVoice, download_kokoro_model, kokoro_model_path, list_tts_voices,
};
pub use vector_store::{LanceVectorStore, VectorStore};

/// Re-exported so the integration-test crate can re-run the migrator against a
/// pool obtained via [`LensEngine::pool`] without exposing the rest of the
/// `pub(crate)` `db` module.
pub use db::run_migrations;

use std::path::Path;
use std::sync::Arc;

use sqlx::SqlitePool;
use tokenizers::Tokenizer;
use tokio::sync::{OnceCell, RwLock, RwLockReadGuard, RwLockWriteGuard, Semaphore};

use crate::notebooks::NotebookRepo;

/// Lowercase-hex encoding of a byte slice.
///
/// Single source of truth for the `write!("{b:02x}")` digest-formatting loop
/// shared by the ingest content-hash and the TTS integrity gate.
pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

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
///
/// # Concurrency invariants (load-bearing)
///
/// * **Single ingest at a time.** Every ingest run holds the single-permit
///   [`ingest_lock`](Self::ingest_lock) semaphore for its whole duration (the
///   ONNX session is single-threaded; concurrent `embed()` must not overlap).
/// * **Destructive deletes take `ingest_lock`.** [`purge_source`](Self::purge_source)
///   and [`purge_notebook`](Self::purge_notebook) acquire the same permit across
///   their cross-store (Lance-then-SQLite) deletes, so a destructive wipe can
///   never interleave a live ingest of the same source/notebook and leave orphan
///   Lance rows. [`trash_source`](Self::trash_source) /
///   [`restore_source`](Self::restore_source) are flag-only (no cross-store
///   mutation) and are intentionally lock-free.
/// * **One app instance per data dir.** There is NO cross-process lock; correct
///   operation assumes a single process owns a given `data_dir` at a time.
/// * **Trashed-source vectors stay in Lance.** Trashing a source leaves its Lance
///   vectors in place so it can be restored; retrieval MUST therefore exclude
///   trashed sources at query time (an M5 obligation).
#[derive(Clone)]
pub struct LensEngine {
    inner: Arc<RwLock<LensEngineInner>>,
    /// Lazily-constructed, shared embedding model (Decision D1 / M2).
    ///
    /// Lives OUTSIDE the `RwLock` so a model load never serializes DB reads.
    /// Built exactly once via [`LensEngine::embedder`]'s `get_or_try_init`.
    embedder: Arc<OnceCell<Arc<dyn Embedder>>>,
    /// Lazily-resolved, shared nomic tokenizer (parallel to `embedder`).
    ///
    /// The nomic `tokenizer.json` is a multi-MB file; resolving it per-ingest
    /// re-reads and re-parses it from disk every time. Cache it once here —
    /// built exactly once via [`LensEngine::tokenizer`]'s `get_or_try_init`
    /// using the shared [`resolve_nomic_tokenizer`] resolver — and reuse the
    /// `Arc` across ingests. Lives OUTSIDE the `RwLock` for the same reason as
    /// `embedder`: a resolve/download must never serialize DB reads.
    tokenizer: Arc<OnceCell<Arc<Tokenizer>>>,
    /// Single-permit gate serializing ingest runs (the ONNX session is
    /// single-threaded; concurrent `embed()` calls must not overlap).
    ingest_lock: Arc<Semaphore>,
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
        // Crash-recovery path: a process that died mid-ingest leaves a source
        // stuck in a transient `parsing`/`embedding` status with no running task
        // to advance it. Reset those to `error` once at startup so the UI can
        // surface them as re-ingestable rather than spinning forever. Terminal
        // states (`queued`/`indexed`/`error`/`pending`) are untouched.
        //
        // INVARIANT (locked by test `crash_recovery_skips_needs_js_and_needs_ocr`):
        // `needs_js` and `needs_ocr` are TERMINAL-PENDING — they must NOT be
        // reset here. They are deliberately absent from the `IN (?, ?)` clause.
        // Run_ingest sets them directly via `update_source_status` and returns
        // `Ok(())` so they are never surfaced via the Err→error flip path.
        sqlx::query("UPDATE sources SET status = ? WHERE status IN (?, ?)")
            .bind(notebooks::source_status::ERROR)
            .bind(notebooks::source_status::PARSING)
            .bind(notebooks::source_status::EMBEDDING)
            .execute(&db)
            .await?;
        let mut config = AppConfig::load(data_dir)?;
        config.paths.data_dir = data_dir.display().to_string();
        tracing::info!("engine initialized");
        Ok(Self {
            inner: Arc::new(RwLock::new(LensEngineInner { db, config })),
            embedder: Arc::new(OnceCell::new()),
            tokenizer: Arc::new(OnceCell::new()),
            ingest_lock: Arc::new(Semaphore::new(1)),
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
            embedder: Arc::new(OnceCell::new()),
            tokenizer: Arc::new(OnceCell::new()),
            ingest_lock: Arc::new(Semaphore::new(1)),
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

    /// Lists all live (non-trashed) notebooks with their source counts, newest
    /// `created_at` first.
    #[tracing::instrument(skip_all)]
    pub async fn list_notebooks_with_counts(&self) -> Result<Vec<NotebookSummary>, LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool).list_with_counts().await
    }

    /// Lists all trashed notebooks with their source counts, newest `trashed_at`
    /// first.
    #[tracing::instrument(skip_all)]
    pub async fn list_trashed_with_counts(&self) -> Result<Vec<NotebookSummary>, LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool).list_trashed_with_counts().await
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

    /// Inserts a URL source: inserts a `queued` `sources` row whose `locator` is
    /// the verbatim URL string. Returns immediately — no fetch happens here.
    /// The caller should invoke [`ingest_source`](Self::ingest_source) separately
    /// to fetch and extract the page in the background.
    #[tracing::instrument(skip(self))]
    pub async fn add_url_source(
        &self,
        notebook_id: &NotebookId,
        title: &str,
        url: &str,
    ) -> Result<Source, LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool)
            .add_url_source(notebook_id, title, url)
            .await
    }

    /// Inserts a managed text/markdown source: writes `text` to a managed file
    /// under `{data_dir}/sources/` and inserts a `queued` `sources` row.
    /// `kind` must be `"text"` or `"markdown"`. Returns the inserted source.
    #[tracing::instrument(skip(self, text))]
    pub async fn add_text_source(
        &self,
        notebook_id: &NotebookId,
        title: &str,
        text: &str,
        kind: &str,
    ) -> Result<Source, LensError> {
        let data_dir = self.data_dir().await;
        let pool = self.pool().await;
        NotebookRepo::new(&pool)
            .add_text_source(&data_dir, notebook_id, title, text, kind)
            .await
    }

    /// Inserts a managed local-file source (PDF/DOCX/text/markdown): copies the
    /// file into managed storage under `{data_dir}/sources/` and inserts a
    /// `queued` `sources` row. `kind` is detected from the file EXTENSION. `title`
    /// defaults to the file name when not supplied. Returns the inserted source.
    /// Call [`ingest_source`](Self::ingest_source) separately to extract + index it.
    #[tracing::instrument(skip(self))]
    pub async fn add_file_source(
        &self,
        notebook_id: &NotebookId,
        src_path: &Path,
        title: Option<&str>,
    ) -> Result<Source, LensError> {
        let data_dir = self.data_dir().await;
        let pool = self.pool().await;
        NotebookRepo::new(&pool)
            .add_file_source(&data_dir, notebook_id, src_path, title)
            .await
    }

    /// Soft-deletes a source: sets `trashed_at` to now. Keeps all chunks and
    /// Lance vectors so the source can be restored. Errors if the source is
    /// missing or already trashed.
    #[tracing::instrument(skip(self))]
    pub async fn trash_source(&self, source_id: &str) -> Result<(), LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool).trash_source(source_id).await
    }

    /// Restores a trashed source: clears `trashed_at`. Errors if the source is
    /// live (not trashed) or does not exist.
    #[tracing::instrument(skip(self))]
    pub async fn restore_source(&self, source_id: &str) -> Result<(), LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool).restore_source(source_id).await
    }

    /// Permanently deletes a source: drops its Lance vectors first (Lance before
    /// SQLite ordering), then removes the `sources` row. Child `chunks` rows
    /// cascade. Errors if the source does not exist or is not trashed.
    ///
    /// Holds the `ingest_lock` permit across the whole cross-store delete so a
    /// destructive wipe cannot interleave a live ingest of the same source (which
    /// would otherwise re-insert vectors after the drop, leaving orphans). See
    /// the module-level concurrency invariants on [`LensEngine`].
    #[tracing::instrument(skip(self))]
    pub async fn purge_source(&self, source_id: &str) -> Result<(), LensError> {
        let _permit = self
            .ingest_lock()
            .acquire()
            .await
            .map_err(|e| LensError::Internal(format!("ingest semaphore closed: {e}")))?;
        let pool = self.pool().await;
        let data_dir = self.data_dir().await;
        let source = NotebookRepo::new(&pool)
            .get_source(source_id)
            .await?
            .ok_or_else(|| LensError::Validation(format!("no source with id {source_id}")))?;
        let store = crate::vector_store::LanceVectorStore::new(&data_dir, pool.clone());
        store
            .drop_source(&source.notebook_id, EMBED_MODEL_ID, EMBED_DIM, source_id)
            .await?;
        NotebookRepo::new(&pool).purge_source(source_id).await?;
        // Best-effort: remove the managed source file AND its `.extracted.txt`
        // sibling so "Delete forever" does not leak either on disk. A missing
        // file (already gone) is not an error.
        remove_managed_source_file(&data_dir, source_id, &source.locator);
        Ok(())
    }

    /// Toggles a source's `selected` flag (persisted). `true` = selected.
    #[tracing::instrument(skip(self))]
    pub async fn set_source_selected(&self, id: &str, selected: bool) -> Result<(), LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool)
            .set_source_selected(id, selected)
            .await
    }

    /// Ingests a queued source end-to-end (parse → chunk → embed → index),
    /// streaming progress through `on_progress`.
    #[tracing::instrument(skip(self, on_progress))]
    pub async fn ingest_source(
        &self,
        source_id: &str,
        on_progress: impl FnMut(crate::ingest::IngestProgress),
    ) -> Result<(), LensError> {
        crate::ingest::ingest_source(self, source_id, on_progress).await
    }

    /// Returns the resolved data directory from the loaded config.
    pub(crate) async fn data_dir(&self) -> std::path::PathBuf {
        std::path::PathBuf::from(self.read().await.config.paths.data_dir.clone())
    }

    /// Test-only accessor for the resolved data directory (the `pub(crate)`
    /// [`data_dir`](Self::data_dir) is not reachable from the integration-test
    /// crate). Gated behind `test-util` so it is absent from production builds.
    #[cfg(feature = "test-util")]
    pub async fn data_dir_for_test(&self) -> std::path::PathBuf {
        self.data_dir().await
    }

    /// Borrows the single-permit ingest semaphore (Decision D1 / M2).
    pub(crate) fn ingest_lock(&self) -> &Arc<Semaphore> {
        &self.ingest_lock
    }

    /// Test-only seam: pre-fills the embedder `OnceCell` with a caller-supplied
    /// [`Embedder`] so integration tests can inject a `CountingEmbedder` (and so
    /// avoid the ~130 MB `FastembedEmbedder` model download).
    ///
    /// Gated behind the `test-util` feature so it is NEVER present in a
    /// production build. Returns `Err` if the embedder was already initialized
    /// (e.g. a prior `ingest_source` already lazily constructed the real model).
    ///
    /// The injected embedder is shared exactly like the lazily-constructed one,
    /// so the cached-once AC (`load_count == 1` across two ingests) and the
    /// concurrency AC (`in_flight` never exceeds `1`) are both observable through
    /// the same `Arc` the pipeline reuses.
    #[cfg(feature = "test-util")]
    pub fn set_embedder_for_test(&self, embedder: Arc<dyn Embedder>) -> Result<(), LensError> {
        self.embedder
            .set(embedder)
            .map_err(|_| LensError::Internal("embedder already initialized".into()))
    }

    /// Lazily constructs (once) and returns the shared embedding model.
    ///
    /// The first caller builds a [`FastembedEmbedder`] over `{data_dir}/models/
    /// fastembed/` (a ~130 MB ONNX session, with a one-time HuggingFace download
    /// on a cold cache); subsequent callers reuse the cached `Arc`. The
    /// construction runs under [`tokio::task::spawn_blocking`] because fastembed
    /// init is synchronous and CPU/IO-heavy.
    pub(crate) async fn embedder(&self) -> Result<Arc<dyn Embedder>, LensError> {
        self.embedder
            .get_or_try_init(|| async {
                let data_dir = self.data_dir().await;
                let embedder =
                    tokio::task::spawn_blocking(move || FastembedEmbedder::new(&data_dir))
                        .await
                        .map_err(|e| {
                            LensError::Model(format!("embedder init task panicked: {e}"))
                        })??;
                Ok::<Arc<dyn Embedder>, LensError>(Arc::new(embedder))
            })
            .await
            .cloned()
    }

    /// Lazily resolves (once) and returns the shared nomic tokenizer.
    ///
    /// The first caller resolves the nomic `tokenizer.json` via the shared
    /// [`resolve_nomic_tokenizer`] resolver (locating a cached copy or
    /// downloading it once); subsequent callers reuse the cached `Arc`. This
    /// mirrors [`LensEngine::embedder`] so the multi-MB tokenizer is parsed
    /// from disk exactly once per engine rather than on every ingest.
    pub(crate) async fn tokenizer(&self) -> Result<Arc<Tokenizer>, LensError> {
        self.tokenizer
            .get_or_try_init(|| async {
                let data_dir = self.data_dir().await;
                let tokenizer = resolve_nomic_tokenizer(&data_dir).await?;
                Ok::<Arc<Tokenizer>, LensError>(Arc::new(tokenizer))
            })
            .await
            .cloned()
    }

    /// Renames a notebook, bumping `updated_at`.
    #[tracing::instrument(skip_all)]
    pub async fn rename_notebook(&self, id: &NotebookId, title: &str) -> Result<(), LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool).rename(id, title).await
    }

    /// Soft-deletes a notebook (backward-compat alias for `trash_notebook`).
    ///
    /// Historically a hard delete; M3 reframes deletion as a recoverable
    /// soft-delete via `trashed_at`. `purge_notebook` is now the sole hard delete.
    #[deprecated(note = "Use trash_notebook() directly; kept for backward compat")]
    #[tracing::instrument(skip_all)]
    pub async fn delete_notebook(&self, id: &NotebookId) -> Result<(), LensError> {
        self.trash_notebook(id).await
    }

    /// Soft-deletes a notebook: sets `trashed_at` and bumps `updated_at`.
    #[tracing::instrument(skip_all)]
    pub async fn trash_notebook(&self, id: &NotebookId) -> Result<(), LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool).trash(id).await
    }

    /// Restores a trashed notebook: clears `trashed_at` and bumps `updated_at`.
    #[tracing::instrument(skip_all)]
    pub async fn restore_notebook(&self, id: &NotebookId) -> Result<(), LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool).restore(id).await
    }

    /// Permanently deletes a notebook. Child rows cascade via `ON DELETE CASCADE`.
    /// This is the sole hard-delete path (used by "Delete forever").
    ///
    /// Drops the notebook's per-notebook Lance tables FIRST (Lance before SQLite,
    /// mirroring [`purge_source`](Self::purge_source)): the SQLite delete cascades
    /// the `embedding_index` rows away, so unless the Lance tables are dropped
    /// beforehand they would be orphaned on disk forever (no registry row left to
    /// find them by). A crash between the Lance drop and the SQLite commit is
    /// benign — a re-purge re-drops the (already-gone) tables idempotently.
    ///
    /// Holds the `ingest_lock` permit across the whole cross-store delete so a
    /// destructive wipe cannot interleave a live ingest into the same notebook.
    /// See the module-level concurrency invariants on [`LensEngine`].
    #[tracing::instrument(skip_all)]
    pub async fn purge_notebook(&self, id: &NotebookId) -> Result<(), LensError> {
        let _permit = self
            .ingest_lock()
            .acquire()
            .await
            .map_err(|e| LensError::Internal(format!("ingest semaphore closed: {e}")))?;
        let pool = self.pool().await;
        let data_dir = self.data_dir().await;
        // Capture every source (id, locator) pair (live AND trashed) BEFORE the
        // cascade deletes the `sources` rows, so the managed source files AND
        // their `.extracted.txt` siblings can be removed afterwards rather than
        // leaked on disk forever. The id is needed to derive the sibling path.
        let sources: Vec<(String, String)> =
            sqlx::query_as("SELECT id, locator FROM sources WHERE notebook_id = ?")
                .bind(id.as_str())
                .fetch_all(&pool)
                .await?;
        // Lance-first: drop the per-notebook tables BEFORE the SQLite delete
        // cascades the `embedding_index` rows that name them.
        let store = crate::vector_store::LanceVectorStore::new(&data_dir, pool.clone());
        store.drop_notebook_tables(id.as_str()).await?;
        NotebookRepo::new(&pool).purge(id).await?;
        // Best-effort: remove the managed source files. A missing file (e.g. an
        // M1 `file` record whose locator points outside the managed dir, or an
        // already-deleted file) is ignored — purge must not fail on it.
        for (source_id, locator) in &sources {
            remove_managed_source_file(&data_dir, source_id, locator);
        }
        Ok(())
    }
}

/// Best-effort removal of a managed source file AND its canonical
/// `.extracted.txt` sibling, ignoring a missing file.
///
/// Used by the purge paths to reclaim `{data_dir}/sources/{id}.{ext}` files
/// written by `add_text_source`, PLUS the canonical
/// `{data_dir}/sources/{id}.extracted.txt` sibling that Phase 2 persists for
/// DERIVED (pdf/docx/url) kinds (see [`ingest`]). The sibling is derived from
/// `(data_dir, source_id)` via the SHARED [`ingest::extracted_sibling_path`] —
/// the SAME builder the ingest write site uses — so the write and purge paths
/// can never diverge. Deriving it from `(data_dir, source_id)` (NOT the
/// locator's parent+stem) is REQUIRED because a URL source's locator is the URL
/// string, whose parent/stem do not point at `{data_dir}/sources/{id}`.
///
/// A `NotFound` is silently ignored (the file may already be gone, or the
/// locator may point at an external file an M1 `file` record references); any
/// other error is logged but never fails the purge.
fn remove_managed_source_file(data_dir: &Path, source_id: &str, locator: &str) {
    remove_file_best_effort(Path::new(locator));
    // The `.extracted.txt` sibling lives at {data_dir}/sources/{id}.extracted.txt
    // regardless of the locator (a URL locator is not a filesystem path).
    let sibling = crate::ingest::extracted_sibling_path(data_dir, source_id);
    remove_file_best_effort(&sibling);
}

/// Removes a single file, ignoring `NotFound` and logging any other error.
fn remove_file_best_effort(path: &Path) {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => tracing::warn!(
            path = %path.display(),
            "failed to remove managed source file: {e}"
        ),
    }
}
