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
pub mod enrichment;
pub mod error;
pub mod extract;
pub(crate) mod http;
pub mod ingest;
pub mod llm;
pub mod model_catalog;
pub mod notebooks;
pub mod parse;
pub mod system_check;
pub mod tts;
pub mod vector_store;

pub use config::{AppConfig, EnrichmentConfig, TaskModel};
pub use embedder::{
    CountingEmbedder, DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL_ID, Embedder, EmbeddingBackend,
    EmbeddingModelSpec, FastembedEmbedder, OllamaEmbedder, REGISTRY, resolve, resolve_opt,
};
pub use embedding::{InstallProgress, pull_embedding_model};
pub use enrichment::{ENRICHMENT_QUEUE_CAPACITY, EnrichmentJob};
pub use error::LensError;
pub use extract::{ExtractOutput, Extractor, SourceAnchor, extractor_for};
pub use ingest::{
    IngestProgress, NEEDS_JS_MIN_CHARS, NEEDS_JS_MIN_TEXT_RATIO, URL_FETCH_TIMEOUT, ingest_source,
    resolve_nomic_tokenizer,
};
pub use llm::{
    GenaiProvider, LlmProvider, LlmRequest, LlmResponse, LlmRouting, ReasoningEffort, StreamChunk,
    provider_from_config,
};
pub use model_catalog::{
    Cost, MODELS_CATALOG_REFRESH_INTERVAL, MODELS_CATALOG_RELPATH, MODELS_CATALOG_URL, Modalities,
    ModelCatalog, ModelInfo, ProviderEntry, ReasoningOption, SupportedProvider, catalog_cache_path,
    load_catalog, refresh_if_stale,
};
pub use notebooks::{
    EmbeddingStats, InspectorChunk, Notebook, NotebookId, NotebookSummary, Source,
};
pub use system_check::{
    ALLOWED_EMBEDDING_MODELS, CheckAction, CheckId, CheckResult, CheckStatus, LlmDetection,
    detect_llm, fastembed_weights_cached, is_allowlisted_embedding_id, list_ollama_models,
    ollama_base_url,
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

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use sqlx::SqlitePool;
use tokenizers::Tokenizer;
use tokio::sync::{Mutex, OnceCell, RwLock, RwLockReadGuard, RwLockWriteGuard, Semaphore, mpsc};

use crate::notebooks::{EnrichmentStatus, NotebookRepo};

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
    /// Lazily-constructed, shared embedding models, keyed by model id
    /// (Decision D1 / M2, generalized for M4 Phase 4b per-notebook models — R8).
    ///
    /// Lives OUTSIDE the `RwLock` so a model load never serializes DB reads.
    /// Each entry is built exactly once via [`LensEngine::embedder_for`]; the
    /// same `Arc<dyn Embedder>` is then shared across every ingest/query that
    /// resolves to that model id.
    ///
    /// ## R8 — RAM cost of holding multiple models
    ///
    /// A notebook now carries its own embedding model, so different notebooks
    /// can resolve to different models in the same session. The cache holds every
    /// model that has been touched, each a live ONNX session (~130 MB for nomic
    /// up to ~1.3 GB for the larger models). There is intentionally NO eviction
    /// cap or LRU here: bounding the cache (and unloading idle sessions) is
    /// deferred to M9. In practice the working set is the handful of models the
    /// open notebooks use.
    ///
    /// Concurrency: this is a single `Mutex` guarding the whole map, held across
    /// the `spawn_blocking` ONNX init in [`LensEngine::embedder_for`], so ALL
    /// concurrent `embedder_for` calls serialize on it — including ones for a
    /// DIFFERENT, already-cached key. This guarantees an expensive ONNX init runs
    /// exactly once per key (no duplicate construction under a race), at the cost
    /// of serializing cold-start inits across models. In practice the single-permit
    /// `ingest_lock` already serializes the embed paths, so this rarely bites; a
    /// per-key `OnceCell` (init outside the map lock) is the M9 follow-up if cold
    /// multi-model startup latency becomes a concern.
    embedders: Arc<Mutex<HashMap<String, Arc<dyn Embedder>>>>,
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
    /// Sender half of the background enrichment queue (M4 Phase 3, Step 3).
    ///
    /// `mpsc::Sender` is `Clone`, so it rides the `#[derive(Clone)]` engine
    /// handle. Enqueue is a non-blocking [`try_send`](mpsc::Sender::try_send)
    /// issued OUTSIDE the `ingest_lock` permit (see [`enqueue_enrichment`]).
    /// Dropping every clone closes the channel and stops the worker.
    enrichment_tx: mpsc::Sender<EnrichmentJob>,
    /// The active enrichment LLM provider, or `None` when none is reachable.
    ///
    /// `Arc<RwLock<Option<Arc<dyn LlmProvider>>>>` rather than `OnceCell` (the
    /// plan's ratified deviation from lock #4): AC10 requires REBINDING the
    /// provider on an unreachable→reachable transition, which `OnceCell` (write-
    /// once) forbids. Write the cell to swap the provider; read it to dispatch.
    llm_provider: Arc<RwLock<Option<Arc<dyn LlmProvider>>>>,
    /// In-memory cache of the loaded model catalog (cached `models-catalog.json`,
    /// else the bundled snapshot), behind an `Arc` so a hit is a cheap pointer
    /// clone — NOT a ~2.6 MB read + parse on every picker open (fix #5). Populated
    /// lazily by [`model_catalog`](Self::model_catalog) (the blocking read+parse
    /// runs once, off the async runtime via `spawn_blocking`) and INVALIDATED by
    /// [`refresh_model_catalog`](Self::refresh_model_catalog) when it actually
    /// rewrites the cache file, so the next load re-reads the fresh catalog while
    /// repeated opens between refreshes hit the cache. Lives OUTSIDE the inner
    /// `RwLock` (like `embedder`) so a catalog load never serializes DB reads.
    catalog_cache: Arc<RwLock<Option<Arc<crate::model_catalog::ModelCatalog>>>>,
    /// Test-only gate awaited inside the worker's stub job body so an AC3 test can
    /// hold a job "in flight" and assert the worker holds no `ingest_lock` permit
    /// during the body. `None` (the default) is a no-op. Compiled out of
    /// production builds.
    #[cfg(feature = "test-util")]
    enrichment_gate: Arc<RwLock<Option<Arc<tokio::sync::Notify>>>>,
    /// Test-only gate awaited inside [`reembed::reembed_and_flip`] AFTER the
    /// lock-free building-table populate but BEFORE the `ingest_lock` flip window,
    /// so a test can deterministically interleave a `purge_source` (which fully
    /// completes + releases the lock) into the sequential race the fix #2 re-check
    /// closes. `None` (the default) is a no-op. Compiled out of production builds.
    #[cfg(feature = "test-util")]
    reembed_preflip_gate: Arc<RwLock<Option<Arc<tokio::sync::Notify>>>>,
    /// Test-only switch: when `true`, [`LensEngine::tokenizer`] fails fast instead
    /// of resolving/downloading the multi-MB nomic tokenizer. Enrichment tolerates
    /// a missing tokenizer (it falls back to a whitespace-word token count), so a
    /// Step-4 integration test can run fully offline. `false` (default) is a no-op.
    /// Compiled out of production builds.
    #[cfg(feature = "test-util")]
    skip_tokenizer: Arc<std::sync::atomic::AtomicBool>,
    /// Test-only override for the per-job LLM-call ceiling (AC11 budget seam). `0`
    /// (the default) means "use [`ENRICHMENT_MAX_CALLS_PER_JOB`]"; a non-zero value
    /// tightens the per-job budget so a test can assert the circuit-break fires
    /// after exactly N admitted calls. Compiled out of production builds.
    #[cfg(feature = "test-util")]
    enrichment_max_calls_override: Arc<std::sync::atomic::AtomicU32>,
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
        // The transient set is derived from `SourceStatus::is_transient` (an
        // exhaustive match), so adding a status variant forces a recovery
        // decision rather than silently leaving a new transient state stranded.
        // For the in-progress states this is exactly `(parsing, embedding)`, the
        // same `IN (?, ?)` clause as before — `needs_ocr`/`needs_js` are NOT
        // transient and stay excluded.
        use notebooks::SourceStatus;
        let transient: Vec<SourceStatus> = [
            SourceStatus::Pending,
            SourceStatus::Queued,
            SourceStatus::Parsing,
            SourceStatus::Embedding,
            SourceStatus::Indexed,
            SourceStatus::Error,
            SourceStatus::NeedsOcr,
            SourceStatus::NeedsJs,
        ]
        .into_iter()
        .filter(SourceStatus::is_transient)
        .collect();
        debug_assert_eq!(
            transient,
            vec![SourceStatus::Parsing, SourceStatus::Embedding],
            "crash-recovery transient set must stay (parsing, embedding)"
        );
        let placeholders = std::iter::repeat_n("?", transient.len())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!("UPDATE sources SET status = ? WHERE status IN ({placeholders})");
        let mut query = sqlx::query(&sql).bind(SourceStatus::Error.as_str());
        for s in &transient {
            query = query.bind(s.as_str());
        }
        query.execute(&db).await?;

        // ── Enrichment crash-recovery (AC12) — SEPARATE from the SourceStatus
        // reset above and deliberately AFTER it. `enrichment_status` is orthogonal
        // to `SourceStatus` (lock #2): a source mid-enrichment when the process
        // died is left `enriching` with no task to advance it. Reset it to
        // `pending` so the queue-rebuild below re-enqueues it; `SourceStatus`
        // (which stays `indexed` — the source is still searchable on raw vectors)
        // is untouched, so the compile-locked SourceStatus invariant above is not
        // affected.
        sqlx::query("UPDATE sources SET enrichment_status = ? WHERE enrichment_status = ?")
            .bind(EnrichmentStatus::Pending.as_str())
            .bind(EnrichmentStatus::Enriching.as_str())
            .execute(&db)
            .await?;

        let mut config = AppConfig::load(data_dir)?;
        config.paths.data_dir = data_dir.display().to_string();

        // ── Startup-GC of orphaned transient re-embed tables (AC7). A crash
        // during the Step-5 re-embed leaves either a `building` row (crash mid-
        // populate, before the flip) or a `stale` row (crash after the flip-txn
        // commit but before the Lance-drop). Reclaim BOTH: drop the physical Lance
        // table (idempotent — a missing table is a no-op) and delete the registry
        // row even when its Lance table is already gone. The `active` row is never
        // touched. Best-effort: a GC failure must NOT prevent the engine from
        // starting (raw vectors still serve search), so it is logged, not fatal.
        let gc_data_dir = std::path::PathBuf::from(&config.paths.data_dir);
        if let Err(e) = Self::gc_orphan_embedding_tables(&db, &gc_data_dir).await {
            tracing::warn!("startup-GC of orphan embedding tables failed (non-fatal): {e}");
        }

        // ── Build the bounded enrichment queue + spawn the worker. The worker
        // owns the receiver; the engine keeps the sender. The provider cell starts
        // empty (the worker / a later detect_llm transition installs a provider).
        let (enrichment_tx, enrichment_rx) =
            mpsc::channel::<EnrichmentJob>(enrichment::ENRICHMENT_QUEUE_CAPACITY);

        let engine = Self {
            inner: Arc::new(RwLock::new(LensEngineInner { db, config })),
            embedders: Arc::new(Mutex::new(HashMap::new())),
            tokenizer: Arc::new(OnceCell::new()),
            ingest_lock: Arc::new(Semaphore::new(1)),
            enrichment_tx,
            llm_provider: Arc::new(RwLock::new(None)),
            catalog_cache: Arc::new(RwLock::new(None)),
            #[cfg(feature = "test-util")]
            enrichment_gate: Arc::new(RwLock::new(None)),
            #[cfg(feature = "test-util")]
            reembed_preflip_gate: Arc::new(RwLock::new(None)),
            #[cfg(feature = "test-util")]
            skip_tokenizer: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            #[cfg(feature = "test-util")]
            enrichment_max_calls_override: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        };

        enrichment::spawn_worker(engine.clone(), enrichment_rx);

        // ── Best-effort model-catalog refresh (Stage 1). Fire-and-forget on a
        // detached task so a slow/failed fetch NEVER blocks init or panics: a
        // network/parse error degrades to the cached/bundled copy (mirrors the
        // startup-GC contract above). The 2-3×/day cadence is achieved by the
        // staleness check here at startup plus on-demand
        // `refresh_model_catalog` calls (e.g. when a picker opens).
        {
            let data_dir = std::path::PathBuf::from(&engine.config().await.paths.data_dir);
            tokio::spawn(async move {
                let client = crate::model_catalog::catalog_client();
                if let Err(e) = crate::model_catalog::refresh_if_stale(
                    &data_dir,
                    crate::model_catalog::MODELS_CATALOG_URL,
                    &client,
                )
                .await
                {
                    tracing::warn!("startup model-catalog refresh failed (non-fatal): {e}");
                }
            });
        }

        // ── Install the enrichment LLM provider from the REAL config (Step 6).
        // When enrichment is enabled, build the provider from `AppConfig.models[]`
        // gating cloud backends on `enrichment.cloud_consent` (the factory rejects
        // a cloud provider without consent). When disabled the cell stays empty
        // and the worker no-ops on each job — sources stay on raw vectors. A later
        // `detect_llm` unreachable→reachable transition can still rebind via
        // [`rescan_enrichment_on_provider_change`].
        {
            let cfg = engine.config().await;
            if cfg.enrichment.enabled {
                let provider = crate::llm::provider_from_config(&cfg, cfg.enrichment.cloud_consent);
                engine.set_llm_provider(provider).await;
            }
        }

        // ── Queue-rebuild (AC10/AC12): enqueue every indexed-but-not-yet-enriched
        // source so a restart (or a back-fill after a provider becomes reachable)
        // resumes enrichment. Best-effort — a full channel or a transient DB error
        // is recovered by the next rescan, so it never blocks startup.
        if let Err(e) = engine.rebuild_enrichment_queue().await {
            tracing::warn!("enrichment queue-rebuild at startup failed (non-fatal): {e}");
        }

        tracing::info!("engine initialized");
        Ok(engine)
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
        let (enrichment_tx, enrichment_rx) =
            mpsc::channel::<EnrichmentJob>(enrichment::ENRICHMENT_QUEUE_CAPACITY);
        let engine = Self {
            inner: Arc::new(RwLock::new(LensEngineInner {
                db,
                config: AppConfig::default(),
            })),
            embedders: Arc::new(Mutex::new(HashMap::new())),
            tokenizer: Arc::new(OnceCell::new()),
            ingest_lock: Arc::new(Semaphore::new(1)),
            enrichment_tx,
            llm_provider: Arc::new(RwLock::new(None)),
            catalog_cache: Arc::new(RwLock::new(None)),
            #[cfg(feature = "test-util")]
            enrichment_gate: Arc::new(RwLock::new(None)),
            #[cfg(feature = "test-util")]
            reembed_preflip_gate: Arc::new(RwLock::new(None)),
            #[cfg(feature = "test-util")]
            skip_tokenizer: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            #[cfg(feature = "test-util")]
            enrichment_max_calls_override: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        };
        // Spawn the (stub) worker so the enrichment surface behaves identically to
        // a production engine; it is inert unless a test explicitly enqueues a job.
        enrichment::spawn_worker(engine.clone(), enrichment_rx);
        engine
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
        let data_dir = self.data_dir().await;
        Ok(system_check::run_system_check(&config, &data_dir).await)
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
        // Resolve the app-wide global default coordinate so a NEW notebook adopts
        // whatever default was set in Settings (M4 Phase 4b-B, AC7). Both fields
        // collapse an empty / unset config value to the registry/enum default, so
        // an unconfigured app stamps the same `nomic-embed-text-v1.5`/`fastembed`
        // pair the previous compile-time consts produced.
        let cfg = self.config().await;
        let embedding_model = crate::embedder::registry::resolve(&cfg.embedding_model).id;
        let embedding_backend =
            crate::embedder::EmbeddingBackend::from_opt_str(Some(&cfg.embedding_backend)).as_str();
        let pool = self.pool().await;
        NotebookRepo::new(&pool)
            .create(
                title,
                description,
                focus_mode,
                embedding_model,
                embedding_backend,
            )
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

    /// Reads a source's chunks (full per-chunk metadata, ordered `level`,
    /// `token_start`) for the dev/QA Embeddings Inspector (M4). Read-only.
    #[tracing::instrument(skip_all)]
    pub async fn list_source_chunks(
        &self,
        source_id: &str,
    ) -> Result<Vec<InspectorChunk>, LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool).list_source_chunks(source_id).await
    }

    /// Reads a notebook's ACTIVE embedding-index stats (one entry per active
    /// `(model, dim)`) for the dev/QA Embeddings Inspector header (M4). Read-only.
    #[tracing::instrument(skip_all)]
    pub async fn get_embedding_stats(
        &self,
        notebook_id: &str,
    ) -> Result<Vec<EmbeddingStats>, LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool)
            .get_embedding_stats(notebook_id)
            .await
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
        // Resolve the configurable cap (issue #71) from `AppConfig.max_source_mb`
        // (empty → 50 MB default) and enforce it at the paste boundary.
        let max_source_bytes =
            crate::ingest::resolve_max_source_bytes(&self.config().await.max_source_mb);
        NotebookRepo::new(&pool)
            .add_text_source(&data_dir, notebook_id, title, text, kind, max_source_bytes)
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
        // R7b: drop the source from EVERY active coordinate the notebook holds, not
        // just the currently-configured one. A same-(model, dim) cross-backend
        // switch (or a re-embed mid-flight) can leave MULTIPLE active coordinates
        // for one notebook (e.g. a lingering fastembed row alongside a new ollama
        // one); resolving only the configured coordinate would leave the other
        // backend's vectors dangling (search would return hits no source backs).
        // Enumerate the active registry rows — each yields its own backend — and
        // drop from each. A coordinate with no active row is a no-op inside
        // `drop_source`.
        let active_coords: Vec<(String, i64, String)> = sqlx::query_as(
            "SELECT DISTINCT model, dim, backend FROM embedding_index \
             WHERE notebook_id = ? AND status = 'active'",
        )
        .bind(&source.notebook_id)
        .fetch_all(&pool)
        .await?;
        for (model, dim, backend) in active_coords {
            let coord = crate::vector_store::Coordinate::new(
                source.notebook_id.clone(),
                crate::embedder::EmbeddingBackend::from_opt_str(Some(&backend)),
                model,
                dim as usize,
            );
            store.drop_source(&coord, source_id).await?;
        }
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

    // ── Enrichment wiring (M4 Phase 3, Step 3) ──────────────────────────────

    /// Installs (or replaces) the active enrichment LLM provider under the
    /// `RwLock` cell. `None` clears it (degrade to raw vectors).
    ///
    /// This is the AC10 rebinding seam: on a `detect_llm` unreachable→reachable
    /// transition (or a config change), call this to swap the provider so the
    /// queue-rebuild can enrich the back-fill. `OnceCell` could not model this —
    /// hence the ratified `RwLock` cell.
    pub async fn set_llm_provider(&self, provider: Option<Arc<dyn LlmProvider>>) {
        *self.llm_provider.write().await = provider;
    }

    /// Returns a clone of the active enrichment provider handle (the worker reads
    /// this to decide whether to dispatch; `None` → degrade to raw vectors).
    pub async fn llm_provider(&self) -> Option<Arc<dyn LlmProvider>> {
        self.llm_provider.read().await.clone()
    }

    /// Non-blocking enqueue of a source for background enrichment (AC3).
    ///
    /// A [`try_send`](mpsc::Sender::try_send) — it NEVER awaits and NEVER blocks,
    /// so the ingest path can call it without holding the `ingest_lock` permit and
    /// a full channel can never deadlock against a held permit. A full/closed
    /// channel logs and drops the job; the startup/rescan
    /// [`rebuild_enrichment_queue`](Self::rebuild_enrichment_queue) recovers any
    /// dropped source. This is intentionally infallible at the call site.
    pub fn enqueue_enrichment(&self, source_id: &str) {
        let job = EnrichmentJob {
            source_id: source_id.to_string(),
        };
        match self.enrichment_tx.try_send(job) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                tracing::warn!(
                    source_id,
                    "enrichment queue full; dropping enqueue (recovered by rescan)"
                );
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                tracing::warn!(source_id, "enrichment queue closed; worker stopped");
            }
        }
    }

    /// Scans for sources eligible for enrichment and enqueues them (AC10/AC12).
    ///
    /// Eligibility: `SourceStatus::Indexed` (searchable on raw vectors) AND
    /// `enrichment_status IN (NULL/none, 'pending', 'failed')` (never enriched, or
    /// reset by crash-recovery, or previously failed). Excludes `enriched`,
    /// `enriching`, and `skipped`. Called at startup and as the AC10 back-fill
    /// rescan hook on a provider unreachable→reachable transition.
    pub async fn rebuild_enrichment_queue(&self) -> Result<(), LensError> {
        let pool = self.pool().await;
        let ids: Vec<String> = sqlx::query_scalar(
            "SELECT id FROM sources \
             WHERE status = ? AND trashed_at IS NULL \
               AND (enrichment_status IS NULL OR enrichment_status IN (?, ?, ?))",
        )
        .bind(notebooks::SourceStatus::Indexed.as_str())
        .bind(EnrichmentStatus::None.as_str())
        .bind(EnrichmentStatus::Pending.as_str())
        .bind(EnrichmentStatus::Failed.as_str())
        .fetch_all(&pool)
        .await?;
        for id in &ids {
            self.enqueue_enrichment(id);
        }
        tracing::debug!(
            count = ids.len(),
            "enrichment queue-rebuild enqueued sources"
        );
        Ok(())
    }

    /// AC10 back-fill hook: re-binds the provider from the current config and, if a
    /// provider is now installed, re-scans the queue. Intended to be called on a
    /// `detect_llm` unreachable→reachable transition. Full `system_check`
    /// integration is deferred (Step 4+); this provides the engine-side seam.
    ///
    /// The cloud-consent flag is now sourced from the REAL config
    /// (`AppConfig.enrichment.cloud_consent`, Step 6) rather than threaded in by
    /// the caller; `enabled=false` clears the provider (enrichment off ⇒ raw
    /// vectors). `cloud_consent` gates cloud providers via
    /// [`crate::llm::provider_from_config`].
    pub async fn rescan_enrichment_on_provider_change(&self) -> Result<(), LensError> {
        let config = self.config().await;
        // Honor the master toggle: disabled enrichment clears any provider.
        let provider = if config.enrichment.enabled {
            crate::llm::provider_from_config(&config, config.enrichment.cloud_consent)
        } else {
            None
        };
        let installed = provider.is_some();
        self.set_llm_provider(provider).await;
        if installed {
            self.rebuild_enrichment_queue().await?;
        }
        Ok(())
    }

    // ── Model catalog (Stage 1) ─────────────────────────────────────────────

    /// Loads the typed model catalog (cached `models-catalog.json`, else the
    /// bundled snapshot), behind an `Arc`. Never fails hard — always returns a
    /// usable catalog.
    ///
    /// Hits an in-memory cache so repeated picker opens are a cheap pointer clone
    /// rather than a ~2.6 MB disk read + JSON parse each time (fix #5). On a cache
    /// MISS the blocking read+parse runs on the blocking pool via
    /// [`tokio::task::spawn_blocking`] — never on the async runtime — then the
    /// result is memoized. [`refresh_model_catalog`](Self::refresh_model_catalog)
    /// invalidates the cache when it rewrites the file, so the next call re-reads
    /// the fresh catalog.
    #[tracing::instrument(skip_all)]
    pub async fn model_catalog(&self) -> Arc<crate::model_catalog::ModelCatalog> {
        // Fast path: a cache hit is a pointer clone under a read lock.
        if let Some(catalog) = self.catalog_cache.read().await.as_ref() {
            return catalog.clone();
        }
        // Miss: load off the async runtime, then memoize. The read+parse is the
        // blocking work `spawn_blocking` keeps off the executor thread.
        let data_dir = self.data_dir().await;
        let loaded =
            tokio::task::spawn_blocking(move || crate::model_catalog::load_catalog(&data_dir))
                .await
                .map(Arc::new)
                // A `JoinError` (the load task panicked) is non-fatal: fall back to the
                // bundled snapshot so the catalog surface never fails hard.
                .unwrap_or_else(|e| {
                    tracing::warn!("model-catalog load task panicked; using bundled snapshot: {e}");
                    Arc::new(crate::model_catalog::ModelCatalog::bundled())
                });
        // Populate the cache. If another task raced us, either copy is equivalent
        // (both read the same on-disk/bundled catalog), so simply overwrite.
        *self.catalog_cache.write().await = Some(loaded.clone());
        loaded
    }

    /// Forces an on-demand model-catalog refresh (e.g. when a model picker
    /// opens). Best-effort: still gated by the staleness check, so a fresh cache
    /// is left untouched. Returns `Ok(true)` when the cache was refreshed.
    ///
    /// When the on-disk cache is actually rewritten (`Ok(true)`), the in-memory
    /// catalog cache is INVALIDATED so the next [`model_catalog`](Self::model_catalog)
    /// re-reads the fresh file (fix #5). A no-op refresh (fresh cache, `Ok(false)`)
    /// leaves the in-memory cache intact.
    #[tracing::instrument(skip_all)]
    pub async fn refresh_model_catalog(&self) -> Result<bool, LensError> {
        let data_dir = self.data_dir().await;
        let client = crate::model_catalog::catalog_client();
        let refreshed = crate::model_catalog::refresh_if_stale(
            &data_dir,
            crate::model_catalog::MODELS_CATALOG_URL,
            &client,
        )
        .await?;
        if refreshed {
            // Invalidate so the next load re-reads the freshly-written cache.
            *self.catalog_cache.write().await = None;
        }
        Ok(refreshed)
    }

    /// Startup-GC (AC7): drop every orphaned `building`/`stale` re-embed table and
    /// delete its registry row, idempotently.
    ///
    /// A static helper (takes the raw pool/data_dir) so `init` can call it BEFORE
    /// the engine handle exists. Sweeps `embedding_index` rows where
    /// `status IN ('building','stale')`: `drop_tables` removes the physical Lance
    /// tables (a missing table is a no-op), then a single DELETE removes the rows —
    /// even when the Lance table was already gone (crash between the flip-txn
    /// commit and the Lance-drop). The `active` row is never selected, so the live
    /// index is untouched.
    async fn gc_orphan_embedding_tables(db: &SqlitePool, data_dir: &Path) -> Result<(), LensError> {
        let table_names: Vec<String> = sqlx::query_scalar(
            "SELECT lance_table_name FROM embedding_index WHERE status IN ('building', 'stale')",
        )
        .fetch_all(db)
        .await?;
        if table_names.is_empty() {
            return Ok(());
        }
        let store = crate::vector_store::LanceVectorStore::new(data_dir, db.clone());
        // Drop the physical tables first (idempotent), THEN delete the registry
        // rows. If the drop fails the rows survive and a later GC retries; if the
        // process dies between the drop and the delete, the rows are reclaimed on
        // the next startup (the table is already gone, so drop_tables no-ops).
        crate::vector_store::VectorStore::drop_tables(&store, &table_names).await?;
        sqlx::query("DELETE FROM embedding_index WHERE status IN ('building', 'stale')")
            .execute(db)
            .await?;
        tracing::info!(
            count = table_names.len(),
            "startup-GC reclaimed orphan building/stale embedding tables"
        );
        Ok(())
    }

    /// Test-only: awaited inside the worker's stub job body so a test can hold a
    /// job "in flight". Returns immediately when no gate is installed. Compiled out
    /// of production builds.
    #[cfg(feature = "test-util")]
    pub(crate) async fn enrichment_job_gate(&self) {
        let gate = self.enrichment_gate.read().await.clone();
        if let Some(notify) = gate {
            notify.notified().await;
        }
    }

    /// Test-only seam: installs (or clears) the worker job gate. While a gate is
    /// installed, the worker blocks in its job body until the returned `Notify` is
    /// `notify_one`'d — letting an AC3 test assert the worker holds no
    /// `ingest_lock` permit during the body (a concurrent `purge_source`
    /// proceeds). Gated behind `test-util`; absent from production builds.
    #[cfg(feature = "test-util")]
    pub async fn set_enrichment_gate_for_test(&self, gate: Option<Arc<tokio::sync::Notify>>) {
        *self.enrichment_gate.write().await = gate;
    }

    /// Test-only: awaited inside [`reembed::reembed_and_flip`] after the lock-free
    /// populate and before the `ingest_lock` flip window. Returns immediately when
    /// no gate is installed. Lets a fix #2 test hold the reembed "just before the
    /// flip" while it runs a `purge_source` to completion (sequential race).
    /// Compiled out of production builds.
    #[cfg(feature = "test-util")]
    pub(crate) async fn reembed_preflip_gate(&self) {
        let gate = self.reembed_preflip_gate.read().await.clone();
        if let Some(notify) = gate {
            notify.notified().await;
        }
    }

    /// Test-only seam: installs (or clears) the reembed pre-flip gate. While
    /// installed, [`reembed::reembed_and_flip`] blocks after populate (before the
    /// flip) until the `Notify` is `notify_one`'d. Gated behind `test-util`; absent
    /// from production builds.
    #[cfg(feature = "test-util")]
    pub async fn set_reembed_preflip_gate_for_test(&self, gate: Option<Arc<tokio::sync::Notify>>) {
        *self.reembed_preflip_gate.write().await = gate;
    }

    /// Test-only seam: directly enqueue a source onto the enrichment queue (the
    /// production enqueue is internal to the ingest path). Gated behind
    /// `test-util`; absent from production builds.
    #[cfg(feature = "test-util")]
    pub fn enqueue_enrichment_for_test(&self, source_id: &str) {
        self.enqueue_enrichment(source_id);
    }

    /// Test-only seam: runs the Step-5 re-embed new-table-flip directly (the
    /// production path is internal to the worker). Lets a test drive the
    /// purge-vs-flip race deterministically — populate the building table via this
    /// call AFTER a `purge_source` has already removed the source — to assert the
    /// in-lock re-check SKIPS the flip (fix #2). Gated behind `test-util`; absent
    /// from production builds.
    #[cfg(feature = "test-util")]
    pub async fn reembed_and_flip_for_test(
        &self,
        source_id: &str,
        notebook: &str,
        doc_summary: &str,
    ) -> Result<(), LensError> {
        crate::enrichment::reembed::reembed_and_flip(self, source_id, notebook, doc_summary).await
    }

    /// Test-only seam: disables [`tokenizer`](Self::tokenizer) resolution so a
    /// Step-4 enrichment test runs fully offline (the worker falls back to a
    /// whitespace-word token count). Gated behind `test-util`; absent from
    /// production builds.
    #[cfg(feature = "test-util")]
    pub fn disable_tokenizer_for_test(&self) {
        self.skip_tokenizer
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Test-only seam (AC11 budget): override the worker's per-job LLM-call ceiling
    /// so a test can assert the circuit-break fires after exactly `max_calls`
    /// admitted calls (`0` restores the default). Gated behind `test-util`; absent
    /// from production builds.
    #[cfg(feature = "test-util")]
    pub fn set_enrichment_max_calls_for_test(&self, max_calls: u32) {
        self.enrichment_max_calls_override
            .store(max_calls, std::sync::atomic::Ordering::Relaxed);
    }

    /// The effective per-job LLM-call ceiling: the test override when non-zero,
    /// else the production default. Read by the worker when it builds the per-job
    /// [`enrichment::Budget`](crate::enrichment::Budget).
    pub(crate) fn enrichment_max_calls_per_job(&self) -> u32 {
        #[cfg(feature = "test-util")]
        {
            let o = self
                .enrichment_max_calls_override
                .load(std::sync::atomic::Ordering::Relaxed);
            if o != 0 {
                return o;
            }
        }
        enrichment::meta::ENRICHMENT_MAX_CALLS_PER_JOB
    }

    /// Test-only seam: the enrichment queue's spare capacity right now. Lets a test
    /// fill the channel and assert a `try_send` overflow does not deadlock. Gated
    /// behind `test-util`; absent from production builds.
    #[cfg(feature = "test-util")]
    pub fn enrichment_queue_capacity(&self) -> usize {
        self.enrichment_tx.capacity()
    }

    /// Test-only seam: pre-fills the keyed embedder cache with a caller-supplied
    /// [`Embedder`] so integration tests can inject a `CountingEmbedder` (and so
    /// avoid the ~130 MB `FastembedEmbedder` model download).
    ///
    /// The embedder is registered under `(model_id, backend)` (its own
    /// [`Embedder::model_id`] plus the caller-supplied `backend`), so a later
    /// [`embedder_for`](Self::embedder_for) for that `(id, backend)` returns this
    /// injected instance instead of building a real model. Inject one per
    /// `(model, backend)` a test exercises (e.g. a default-nomic Fastembed
    /// embedder AND the same nomic id under Ollama, each under its own key).
    ///
    /// Gated behind the `test-util` feature so it is NEVER present in a
    /// production build. Returns `Err` if an embedder is already cached for that
    /// model id (e.g. a prior `ingest_source` already lazily constructed it).
    ///
    /// The injected embedder is shared exactly like the lazily-constructed one,
    /// so the cached-once AC (`load_count == 1` across two ingests) and the
    /// concurrency AC (`in_flight` never exceeds `1`) are both observable through
    /// the same `Arc` the pipeline reuses.
    #[cfg(feature = "test-util")]
    pub fn set_embedder_for_test(
        &self,
        embedder: Arc<dyn Embedder>,
        backend: crate::embedder::EmbeddingBackend,
    ) -> Result<(), LensError> {
        let key = Self::embedder_cache_key(embedder.model_id(), backend);
        // `try_lock` (not `blocking_lock`) keeps this a sync fn that is safe to
        // call from inside a `#[tokio::test]` async context: the cache is
        // uncontended at injection time, so the lock is always immediately
        // available.
        let mut cache = self
            .embedders
            .try_lock()
            .map_err(|e| LensError::Internal(format!("embedder cache busy: {e}")))?;
        if cache.contains_key(&key) {
            return Err(LensError::Internal(format!(
                "embedder already initialized for model {key}"
            )));
        }
        cache.insert(key, embedder);
        Ok(())
    }

    /// Test-only seam: returns the cached/lazily-built embedder for `model_id`
    /// (the `pub(crate)` [`embedder_for`](Self::embedder_for) is not reachable
    /// from the integration-test crate). Gated behind `test-util`.
    #[cfg(feature = "test-util")]
    pub async fn embedder_for_test_get(
        &self,
        model_id: &str,
        backend: crate::embedder::EmbeddingBackend,
    ) -> Result<Arc<dyn Embedder>, LensError> {
        self.embedder_for(model_id, backend).await
    }

    /// The embedder-cache key for a `(resolved-model-id, backend)` pair.
    ///
    /// The backend is part of the key (M4 Phase 4b-B): the SAME registry model
    /// served by `fastembed` vs `ollama` is two physically-distinct embedders
    /// (different numerical vectors), so they MUST occupy separate cache slots —
    /// a model-id-only key would alias them and return the wrong backend's
    /// embedder for a notebook. Format: `"{backend}:{model_id}"`.
    fn embedder_cache_key(model_id: &str, backend: crate::embedder::EmbeddingBackend) -> String {
        format!("{}:{model_id}", backend.as_str())
    }

    /// Lazily constructs (once per `(model_id, backend)`) and returns the shared
    /// embedder for that coordinate, caching it in the keyed embedder cache (R8).
    ///
    /// On a cache hit the cached `Arc` is cloned and returned. On a miss the
    /// model id is resolved through the registry ([`crate::embedder::resolve`],
    /// which falls back to the default for an unknown/empty id) and an embedder is
    /// built for that spec PER BACKEND:
    ///
    /// - [`EmbeddingBackend::Fastembed`](crate::embedder::EmbeddingBackend::Fastembed):
    ///   a [`FastembedEmbedder`] over `{data_dir}/models/fastembed/` (a
    ///   ~130 MB–1.3 GB ONNX session, with a one-time HuggingFace download on a
    ///   cold cache). Construction runs under [`tokio::task::spawn_blocking`]
    ///   because fastembed init is synchronous and CPU/IO-heavy.
    /// - [`EmbeddingBackend::Ollama`](crate::embedder::EmbeddingBackend::Ollama):
    ///   an [`OllamaEmbedder`](crate::embedder::OllamaEmbedder) targeting the
    ///   configured (loopback-only) Ollama base URL.
    ///
    /// The cache is keyed by `(resolved-spec-id, backend)`
    /// ([`embedder_cache_key`](Self::embedder_cache_key)) so the legacy alias and
    /// an unknown id both collapse onto the canonical entry, while the two
    /// backends for the same model NEVER alias. The whole construct-and-insert
    /// runs while holding the cache `Mutex`, so concurrent callers for the same
    /// key serialize: the expensive init runs exactly once (see the field's R8
    /// doc).
    pub(crate) async fn embedder_for(
        &self,
        model_id: &str,
        backend: crate::embedder::EmbeddingBackend,
    ) -> Result<Arc<dyn Embedder>, LensError> {
        let spec = crate::embedder::resolve(model_id);
        let key = Self::embedder_cache_key(spec.id, backend);
        let mut cache = self.embedders.lock().await;
        if let Some(existing) = cache.get(&key) {
            return Ok(Arc::clone(existing));
        }
        let embedder: Arc<dyn Embedder> = match backend {
            crate::embedder::EmbeddingBackend::Fastembed => {
                let data_dir = self.data_dir().await;
                // `spec` is a `&'static EmbeddingModelSpec` (Copy), so the closure
                // can capture it directly without a clone or move of `key`.
                let e = tokio::task::spawn_blocking(move || {
                    FastembedEmbedder::new_with_spec(&data_dir, spec)
                })
                .await
                .map_err(|e| LensError::Model(format!("embedder init task panicked: {e}")))??;
                Arc::new(e)
            }
            crate::embedder::EmbeddingBackend::Ollama => {
                let base_url = ollama_base_url(&self.config().await);
                Arc::new(crate::embedder::OllamaEmbedder::new(&base_url, spec)?)
            }
        };
        cache.insert(key, Arc::clone(&embedder));
        Ok(embedder)
    }

    /// Warms (constructs + caches) the fastembed embedder for `model_id`,
    /// downloading its HuggingFace weights to `{data_dir}/models/fastembed/` on a
    /// cold cache. Idempotent: a warm cache hit returns immediately.
    ///
    /// This is the onboarding/Settings "Install [fastembed model]" path: fastembed
    /// has no separate download step (weights land lazily on first embedder
    /// construction), so warming up-front lets onboarding pass the per-backend
    /// readiness gate ([`crate::fastembed_weights_cached`]) for a fastembed
    /// selection on a fresh, Ollama-less machine. Always `Fastembed` (Ollama is
    /// detect-only — the app never pulls).
    pub async fn warm_fastembed_model(&self, model_id: &str) -> Result<(), LensError> {
        self.embedder_for(model_id, crate::embedder::EmbeddingBackend::Fastembed)
            .await
            .map(|_| ())
    }

    /// Resolves a notebook's configured embedding coordinate components
    /// `(model_id, dim, backend)` (R1, M4 Phase 4b-B widened from `(model, dim)`).
    ///
    /// This is the SINGLE read-path entry point every query / ingest / re-embed
    /// caller resolves through before touching the vector store: it reads the
    /// notebook row's `embedding_model` (NULL/absent for pre-migration rows) and
    /// runs it through the registry ([`crate::embedder::resolve`]), which falls
    /// back to the default ([`DEFAULT_EMBED_MODEL_ID`]) for a NULL, empty, or
    /// unknown value. The returned id is the *canonical* registry id (the legacy
    /// alias and unknown ids collapse onto a canonical entry), so it is safe to
    /// thread straight into [`embedder_for`](Self::embedder_for) and the
    /// `VectorStore` coordinate APIs.
    ///
    /// The backend (M4 Phase 4b-B) is the THIRD coordinate axis: a NULL/empty/
    /// unknown `embedding_backend` column resolves to the global default backend
    /// via [`crate::embedder::EmbeddingBackend::from_opt_str`] (which, with an
    /// unset config, is `fastembed`). The returned `(model, dim, backend)` triple
    /// is everything a caller needs to construct a `VectorStore` `Coordinate`.
    pub async fn resolve_notebook_embedding(
        &self,
        notebook_id: &NotebookId,
    ) -> Result<(String, usize, crate::embedder::EmbeddingBackend), LensError> {
        let pool = self.pool().await;
        // `fetch_optional` → None means NO such notebook row (fail fast); `Some(row)`
        // with NULL columns means the row exists with a NULL model/backend (resolve
        // each to the default).
        let row: (Option<String>, Option<String>) =
            sqlx::query_as("SELECT embedding_model, embedding_backend FROM notebooks WHERE id = ?")
                .bind(notebook_id.as_str())
                .fetch_optional(&pool)
                .await?
                .ok_or_else(|| {
                    LensError::Validation(format!("no notebook with id {notebook_id}"))
                })?;
        let (stored_model, stored_backend) = row;
        let spec = crate::embedder::resolve(stored_model.as_deref().unwrap_or(""));
        let backend = crate::embedder::EmbeddingBackend::from_opt_str(stored_backend.as_deref());
        Ok((spec.id.to_string(), spec.dim, backend))
    }

    /// Persists a new embedding model choice for a notebook.
    ///
    /// Validates that `model_id` is a known registry entry (unknown ids are
    /// rejected with a [`LensError::Validation`] rather than silently falling
    /// through to the nomic default). Writes `notebooks.embedding_model = model_id`
    /// so subsequent [`resolve_notebook_embedding`](Self::resolve_notebook_embedding)
    /// calls return the new coordinate.
    ///
    /// This does NOT kick off re-embedding — the Tauri command layer calls
    /// [`reembed_notebook`](Self::reembed_notebook) after persisting.
    ///
    /// `backend` (M4 Phase 4b-B) is the THIRD coordinate axis: it is persisted to
    /// `notebooks.embedding_backend` alongside the model so a same-(model, dim)
    /// backend switch is a genuine coordinate change that
    /// [`reembed_notebook`](Self::reembed_notebook) re-embeds + retires (R2).
    pub async fn set_notebook_embedding_model(
        &self,
        notebook_id: &NotebookId,
        model_id: &str,
        backend: crate::embedder::EmbeddingBackend,
    ) -> Result<(), LensError> {
        // Reject genuinely-unknown ids via the registry's strict lookup (which
        // accepts the legacy alias `nomic-embed-text` → nomic). Persist the
        // CANONICAL `spec.id` (e.g. the frontend's Ollama-facing `nomic-embed-text`
        // is stored as `nomic-embed-text-v1.5`) so resolution downstream is exact.
        let spec = crate::embedder::resolve_opt(model_id).ok_or_else(|| {
            LensError::Validation(format!(
                "unknown embedding model id: {model_id:?}; known ids: nomic-embed-text-v1.5 \
                 (alias nomic-embed-text), mxbai-embed-large, all-minilm, bge-m3"
            ))
        })?;
        let pool = self.pool().await;
        let result = sqlx::query(
            "UPDATE notebooks SET embedding_model = ?, embedding_backend = ?, updated_at = ? \
             WHERE id = ? AND trashed_at IS NULL",
        )
        .bind(spec.id)
        .bind(backend.as_str())
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(notebook_id.as_str())
        .execute(&pool)
        .await?;
        if result.rows_affected() == 0 {
            return Err(LensError::Validation(format!(
                "no live notebook with id {notebook_id}"
            )));
        }
        Ok(())
    }

    /// Returns `(model_id, dim, backend, status)` for a notebook's current
    /// embedding coordinate, where `status` is `"active"` when a live
    /// `embedding_index` row exists for the FULL `(notebook, backend, model, dim)`
    /// coordinate, or `"none"` otherwise.
    ///
    /// R4/R7a (M4 Phase 4b-B): the status query is backend-scoped (`AND backend =
    /// ?`). After a same-dim cross-backend switch — e.g. fastembed-nomic-768 →
    /// ollama-nomic-768 — the OLD backend's row may still be `stale`/being retired
    /// while the NEW backend's row is `active`. A backend-blind query (matching
    /// only `(model, dim)`) would report the WRONG backend's status; binding the
    /// resolved backend returns the configured coordinate's true status.
    ///
    /// Used by [`get_notebook_embedding_model`] in the Tauri command layer.
    pub async fn get_notebook_embedding_info(
        &self,
        notebook_id: &NotebookId,
    ) -> Result<(String, usize, crate::embedder::EmbeddingBackend, String), LensError> {
        let (model_id, dim, backend) = self.resolve_notebook_embedding(notebook_id).await?;
        let pool = self.pool().await;
        let active: Option<i64> = sqlx::query_scalar(
            "SELECT 1 FROM embedding_index \
             WHERE notebook_id = ? AND model = ? AND dim = ? AND backend = ? AND status = 'active'",
        )
        .bind(notebook_id.as_str())
        .bind(&model_id)
        .bind(dim as i64)
        .bind(backend.as_str())
        .fetch_optional(&pool)
        .await?;
        let status = if active.is_some() { "active" } else { "none" }.to_string();
        Ok((model_id, dim, backend, status))
    }

    /// Re-embeds every chunk of `notebook_id` into the notebook's currently
    /// configured embedding coordinate, flips it active, and retires the previous
    /// coordinate(s) (M4 Phase 4b, Step 9 — the model-switch re-embed).
    ///
    /// Background-safe: the embed + populate runs lock-free; only the brief flip +
    /// retirement take `ingest_lock`, and the OLD index keeps serving search until
    /// the flip. A no-op when the configured model already matches the active
    /// coordinate. `on_progress(done, total)` fires after each populated batch
    /// (pass a no-op closure for headless callers). See
    /// [`enrichment::reembed::reembed_notebook`] for the crash-safety + R2
    /// coordinate-re-check details.
    #[tracing::instrument(skip_all, fields(notebook = %notebook_id.as_str()))]
    pub async fn reembed_notebook(
        &self,
        notebook_id: &NotebookId,
        on_progress: impl FnMut(usize, usize) + Send,
    ) -> Result<crate::enrichment::reembed::ReembedOutcome, LensError> {
        crate::enrichment::reembed::reembed_notebook(self, notebook_id, on_progress).await
    }

    /// Lazily resolves (once) and returns the shared nomic tokenizer.
    ///
    /// The first caller resolves the nomic `tokenizer.json` via the shared
    /// [`resolve_nomic_tokenizer`] resolver (locating a cached copy or
    /// downloading it once); subsequent callers reuse the cached `Arc`. This
    /// mirrors [`LensEngine::embedder_for`] so the multi-MB tokenizer is parsed
    /// from disk exactly once per engine rather than on every ingest.
    pub(crate) async fn tokenizer(&self) -> Result<Arc<Tokenizer>, LensError> {
        #[cfg(feature = "test-util")]
        if self
            .skip_tokenizer
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return Err(LensError::Model(
                "tokenizer disabled for test (skip_tokenizer)".into(),
            ));
        }
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
