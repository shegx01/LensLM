// issue #71: the streamed-ingest future deepened `Send` auto-trait evaluation
// enough to overflow the default 128-frame limit (E0275) on some toolchains.
// Compile-time only; no runtime cost.
#![recursion_limit = "256"]
//! `lens-core` — the headless Rust engine for LensLM.
//!
//! No Tauri, windowing, or UI dependencies. [`LensEngine`] is a thin handle
//! that delegates to per-domain repositories over the shared connection pool.

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
pub mod render;
pub mod system_check;
pub mod transcription;
pub mod tts;
pub mod url_normalize;
pub mod vector_store;

pub use config::{AppConfig, EnrichmentConfig, TaskModel};
pub use embedder::{
    CountingEmbedder, DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL_ID, Embedder, EmbeddingBackend,
    EmbeddingModelSpec, FastembedEmbedder, OllamaEmbedder, REGISTRY, resolve, resolve_opt,
};
pub use embedding::{InstallProgress, pull_embedding_model};
pub use enrichment::{ENRICHMENT_QUEUE_CAPACITY, EnrichmentJob};
pub use error::{ErrorMeta, LensError};
pub use extract::{ExtractOutput, Extractor, SourceAnchor, extractor_for};
pub use ingest::{
    IngestProgress, NEEDS_JS_MIN_CHARS, NEEDS_JS_MIN_TEXT_RATIO, URL_FETCH_TIMEOUT, ingest_source,
    readback_host_allowed, resolve_nomic_tokenizer, ssrf_check_host, ssrf_check_url,
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
    AddSourceOutcome, EmbeddingStats, InspectorChunk, Notebook, NotebookId, NotebookSummary,
    Source, TrashedSource,
};
pub use render::JsRenderer;
pub use system_check::{
    ALLOWED_EMBEDDING_MODELS, CheckAction, CheckId, CheckResult, CheckStatus, LlmDetection,
    ModelValidation, detect_llm, fastembed_weights_cached, is_allowlisted_embedding_id,
    list_ollama_models, ollama_base_url, validate_model_interactive,
};
pub use transcription::{WindowConfig, decode_and_resample_audio, decode_resample_windows};
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

/// Lowercase-hex encoding of a byte slice; shared by the ingest content-hash
/// and TTS integrity gate.
pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Builds the candle embedder for a fastembed-coordinate spec on Apple Silicon
/// (issue #91): Metal for Bulk, CPU for Interactive. Returns `None` — falling back
/// to fastembed — for unsupported models or any candle init failure.
#[cfg(feature = "native-ml-metal")]
async fn build_candle_if_supported(
    compute: crate::embedder::Compute,
    data_dir: &Path,
    spec: &'static crate::embedder::EmbeddingModelSpec,
) -> Option<Arc<dyn Embedder>> {
    // An unsupported model is an expected fastembed fallback — log at debug, not warn.
    if !crate::embedder::candle_supports_model(spec.id) {
        tracing::debug!(
            model = %spec.id,
            "candle backend does not yet implement this model; using fastembed"
        );
        return None;
    }
    let candle_dir = data_dir.join("models").join("candle");
    match tokio::task::spawn_blocking(move || {
        crate::embedder::CandleNomicEmbedder::new_with_spec(&candle_dir, compute, spec)
    })
    .await
    {
        Ok(Ok(e)) => {
            let e: Arc<dyn Embedder> = Arc::new(e);
            Some(e)
        }
        // A supported model that failed init is a real warn; still fall back.
        Ok(Err(err)) => {
            tracing::warn!(
                model = %spec.id,
                device = compute.as_str(),
                error = %err,
                "candle embedder init failed; falling back to fastembed"
            );
            None
        }
        Err(join) => {
            tracing::warn!(
                error = %join,
                "candle init task panicked; falling back to fastembed"
            );
            None
        }
    }
}

/// No-op stub: fastembed handles all embeddings on non-Apple-Silicon targets.
#[cfg(not(feature = "native-ml-metal"))]
async fn build_candle_if_supported(
    _compute: crate::embedder::Compute,
    _data_dir: &Path,
    _spec: &'static crate::embedder::EmbeddingModelSpec,
) -> Option<Arc<dyn Embedder>> {
    None
}

/// Interface stub for CUDA embedding (issue #91). Always returns `None` until a
/// candle-CUDA backend is implemented; CUDA jobs fall back to fastembed-CPU.
#[cfg(feature = "native-ml-cuda")]
async fn build_cuda_if_supported(
    _compute: crate::embedder::Compute,
    _data_dir: &Path,
    spec: &'static crate::embedder::EmbeddingModelSpec,
) -> Option<Arc<dyn Embedder>> {
    tracing::debug!(
        model = %spec.id,
        "candle-CUDA backend not yet implemented (interface only); using fastembed-CPU"
    );
    None
}

/// No-op stub: without `native-ml-cuda` the policy never resolves `Cuda`.
#[cfg(not(feature = "native-ml-cuda"))]
async fn build_cuda_if_supported(
    _compute: crate::embedder::Compute,
    _data_dir: &Path,
    _spec: &'static crate::embedder::EmbeddingModelSpec,
) -> Option<Arc<dyn Embedder>> {
    None
}

/// Inner engine state: the database connection pool and loaded configuration.
/// Accessed only through [`LensEngine::pool`] / [`LensEngine::config`].
pub struct LensEngineInner {
    pub(crate) db: SqlitePool,
    pub(crate) config: AppConfig,
}

/// Thread-safe, cheaply-cloneable handle to the LensLM engine state.
///
/// # Concurrency invariants (load-bearing)
///
/// * **Single ingest at a time.** Every ingest holds `ingest_lock` (single-permit
///   semaphore); concurrent `embed()` calls must not overlap.
/// * **Destructive deletes take `ingest_lock`.** `purge_source`/`purge_notebook`
///   hold the permit across their Lance-then-SQLite deletes; `trash_source`/
///   `restore_source` are flag-only and intentionally lock-free.
/// * **One app instance per data dir.** No cross-process lock exists.
/// * **Trashed-source vectors stay in Lance** for restorability; retrieval MUST
///   exclude trashed sources at query time (M5 obligation).
#[derive(Clone)]
pub struct LensEngine {
    inner: Arc<RwLock<LensEngineInner>>,
    /// Keyed embedder cache (R8). Lives outside the `RwLock` so a model load never
    /// serializes DB reads. Built exactly once per key via [`embedder_for`]; the
    /// single `Mutex` over the whole map ensures no duplicate ONNX init under a
    /// race. No eviction cap — deferred to M9.
    embedders: Arc<Mutex<HashMap<String, Arc<dyn Embedder>>>>,
    /// Native-ML acceleration probe (issue #91). Trait object so tests can inject
    /// a fake and future accelerators (CUDA, MLX) drop in without touching policy.
    accelerator: Arc<dyn crate::embedder::NativeAccelerator>,
    /// Shared nomic tokenizer — parsed once via `OnceCell`, outside the `RwLock`
    /// so a resolve/download never serializes DB reads.
    tokenizer: Arc<OnceCell<Arc<Tokenizer>>>,
    /// Single-permit gate serializing ingest runs (ONNX session is single-threaded).
    ingest_lock: Arc<Semaphore>,
    /// Sender half of the background enrichment queue (M4 Phase 3). `Clone` so it
    /// rides `#[derive(Clone)]`. Dropping every clone closes the channel.
    enrichment_tx: mpsc::Sender<EnrichmentJob>,
    /// Active enrichment LLM provider. `RwLock<Option<...>>` rather than `OnceCell`
    /// because AC10 requires rebinding on an unreachable→reachable transition.
    llm_provider: Arc<RwLock<Option<Arc<dyn LlmProvider>>>>,
    /// Injected JS renderer for SPA URL-render fallback (issue #78). `None` in
    /// headless tests or before `src-tauri` wires `TauriJsRenderer`; degrades to
    /// `needs_js` when absent.
    js_renderer: Arc<RwLock<Option<Arc<dyn render::JsRenderer>>>>,
    /// In-memory catalog cache (fix #5). Populated lazily via `spawn_blocking`;
    /// invalidated by `refresh_model_catalog`. Outside the inner `RwLock` so a
    /// catalog load never serializes DB reads.
    catalog_cache: Arc<RwLock<Option<Arc<crate::model_catalog::ModelCatalog>>>>,
    /// AC3 test seam: blocks the worker in its job body until `notify_one`'d.
    #[cfg(feature = "test-util")]
    enrichment_gate: Arc<RwLock<Option<Arc<tokio::sync::Notify>>>>,
    /// Fix-#2 test seam: blocks reembed after populate, before the flip window.
    #[cfg(feature = "test-util")]
    reembed_preflip_gate: Arc<RwLock<Option<Arc<tokio::sync::Notify>>>>,
    /// When `true`, `tokenizer()` fails fast so Step-4 tests run fully offline.
    #[cfg(feature = "test-util")]
    skip_tokenizer: Arc<std::sync::atomic::AtomicBool>,
    /// AC11 test seam: non-zero overrides the per-job LLM-call ceiling.
    #[cfg(feature = "test-util")]
    enrichment_max_calls_override: Arc<std::sync::atomic::AtomicU32>,
}

impl LensEngine {
    /// Opens the on-disk pool, applies migrations, and loads `config.json`.
    /// Populates `config.paths.data_dir` so callers don't re-derive it.
    #[tracing::instrument(skip_all, fields(dir = %data_dir.as_ref().display()))]
    pub async fn init(data_dir: impl AsRef<Path>) -> Result<Self, LensError> {
        let data_dir = data_dir.as_ref();
        std::fs::create_dir_all(data_dir)
            .map_err(|e| LensError::Io(format!("{}: {e}", data_dir.display())))?;
        let db = db::open_pool(data_dir).await?;
        db::run_migrations(&db).await?;
        // Crash-recovery: a mid-ingest death leaves sources stuck in a transient
        // status. Reset them to `error` so the UI surfaces them as re-ingestable.
        //
        // INVARIANT (locked by test `crash_recovery_skips_needs_js_and_needs_ocr`):
        // `needs_js`/`needs_ocr` are TERMINAL-PENDING — must NOT be reset here.
        // The REAL guard is `is_transient()` (an exhaustive match in notebooks.rs),
        // plus the `debug_assert_eq!` below which pins the derived set to exactly
        // `[Parsing, Embedding]`.
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

        // Enrichment crash-recovery (AC12): reset `enriching` → `pending` so the
        // queue-rebuild re-enqueues it. `SourceStatus` stays `indexed` (untouched).
        sqlx::query("UPDATE sources SET enrichment_status = ? WHERE enrichment_status = ?")
            .bind(EnrichmentStatus::Pending.as_str())
            .bind(EnrichmentStatus::Enriching.as_str())
            .execute(&db)
            .await?;

        let mut config = AppConfig::load(data_dir)?;
        config.paths.data_dir = data_dir.display().to_string();

        // Startup-GC (AC7): reclaim orphaned `building`/`stale` re-embed tables.
        // Best-effort — a GC failure must not prevent startup.
        let gc_data_dir = std::path::PathBuf::from(&config.paths.data_dir);
        if let Err(e) = Self::gc_orphan_embedding_tables(&db, &gc_data_dir).await {
            tracing::warn!("startup-GC of orphan embedding tables failed (non-fatal): {e}");
        }

        let (enrichment_tx, enrichment_rx) =
            mpsc::channel::<EnrichmentJob>(enrichment::ENRICHMENT_QUEUE_CAPACITY);

        let engine = Self {
            inner: Arc::new(RwLock::new(LensEngineInner { db, config })),
            embedders: Arc::new(Mutex::new(HashMap::new())),
            accelerator: crate::embedder::default_accelerator(),
            tokenizer: Arc::new(OnceCell::new()),
            ingest_lock: Arc::new(Semaphore::new(1)),
            enrichment_tx,
            llm_provider: Arc::new(RwLock::new(None)),
            js_renderer: Arc::new(RwLock::new(None)),
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

        // Best-effort model-catalog refresh at startup; a slow/failed fetch degrades
        // to the cached/bundled copy — never blocks init.
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

        // Install the enrichment LLM provider from config (Step 6). When disabled,
        // the cell stays empty and sources remain on raw vectors.
        {
            let cfg = engine.config().await;
            if cfg.enrichment.enabled {
                let provider = crate::llm::provider_from_config(&cfg, cfg.enrichment.cloud_consent);
                engine.set_llm_provider(provider).await;
            }
        }

        // Queue-rebuild (AC10/AC12): enqueue indexed-but-not-yet-enriched sources.
        // Best-effort — never blocks startup.
        if let Err(e) = engine.rebuild_enrichment_queue().await {
            tracing::warn!("enrichment queue-rebuild at startup failed (non-fatal): {e}");
        }

        tracing::info!("engine initialized");
        Ok(engine)
    }

    /// Test constructor: a fully-migrated in-memory engine with a default config.
    /// Uses a single-connection pool so the schema persists across queries.
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
            accelerator: crate::embedder::default_accelerator(),
            tokenizer: Arc::new(OnceCell::new()),
            ingest_lock: Arc::new(Semaphore::new(1)),
            enrichment_tx,
            llm_provider: Arc::new(RwLock::new(None)),
            js_renderer: Arc::new(RwLock::new(None)),
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
        engine
    }

    pub async fn read(&self) -> RwLockReadGuard<'_, LensEngineInner> {
        self.inner.read().await
    }

    pub async fn write(&self) -> RwLockWriteGuard<'_, LensEngineInner> {
        self.inner.write().await
    }

    /// Returns a clone of the connection pool. Cloning is cheap (`Arc` internally).
    pub async fn pool(&self) -> SqlitePool {
        self.read().await.db.clone()
    }

    pub async fn config(&self) -> AppConfig {
        self.read().await.config.clone()
    }

    /// Replaces the in-memory configuration. Persistence to disk is the caller's responsibility.
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

    /// Runs the three first-run system-check probes (LlmRuntime, EmbeddingModel,
    /// TTS) in order. Expected-absent subsystems return `Fail`, not `Err`.
    #[tracing::instrument(skip_all)]
    pub async fn run_system_check(&self) -> Result<Vec<CheckResult>, LensError> {
        // Clone config and drop the guard before probes: each probe issues a
        // multi-second HTTP request that must not hold the read lock.
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

    /// Lists individually-trashed sources whose parent notebook is still live,
    /// newest `trashed_at` first. Used by the Trash modal Sources section (issue
    /// #94). Sources under a trashed notebook are excluded.
    #[tracing::instrument(skip_all)]
    pub async fn list_trashed_sources(&self) -> Result<Vec<TrashedSource>, LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool).list_trashed_sources().await
    }

    /// Creates a notebook with the given title and optional onboarding fields,
    /// stamping the app-wide default embedding coordinate (M4 Phase 4b-B, AC7).
    #[tracing::instrument(skip_all)]
    pub async fn create_notebook(
        &self,
        title: &str,
        description: Option<&str>,
        focus_mode: Option<&str>,
    ) -> Result<Notebook, LensError> {
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

    /// Inserts a file source record for a notebook (M1 onboarding). Returns an
    /// [`AddSourceOutcome`]: on a PATH-based dedup hit (issue #100 — this path
    /// hashes the locator, not file content) the existing live source is returned
    /// (`was_existing = true`).
    #[tracing::instrument(skip_all)]
    pub async fn add_source(
        &self,
        notebook_id: &NotebookId,
        title: &str,
        locator: &str,
    ) -> Result<AddSourceOutcome, LensError> {
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

    /// Inserts a `queued` URL source row. No fetch occurs here; call
    /// `ingest_source` separately. `force_js_render` persists the SPA opt-in (issue
    /// #78). Returns an `AddSourceOutcome`; deduplicates on normalized URL (#100).
    #[tracing::instrument(skip(self))]
    pub async fn add_url_source(
        &self,
        notebook_id: &NotebookId,
        title: &str,
        url: &str,
        force_js_render: bool,
    ) -> Result<AddSourceOutcome, LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool)
            .add_url_source(notebook_id, title, url, force_js_render)
            .await
    }

    /// Inserts a managed text/markdown source. `kind` must be `"text"` or
    /// `"markdown"`. Deduplicates on content hash (#100).
    #[tracing::instrument(skip(self, text))]
    pub async fn add_text_source(
        &self,
        notebook_id: &NotebookId,
        title: &str,
        text: &str,
        kind: &str,
    ) -> Result<AddSourceOutcome, LensError> {
        let data_dir = self.data_dir().await;
        let pool = self.pool().await;
        // Enforce the configurable size cap (issue #71) at the paste boundary.
        let max_source_bytes =
            crate::ingest::resolve_max_source_bytes(&self.config().await.max_source_mb);
        NotebookRepo::new(&pool)
            .add_text_source(&data_dir, notebook_id, title, text, kind, max_source_bytes)
            .await
    }

    /// Copies a local file into managed storage and inserts a `queued` row.
    /// Deduplicates on file content hash (issue #96). Call `ingest_source` separately.
    #[tracing::instrument(skip(self))]
    pub async fn add_file_source(
        &self,
        notebook_id: &NotebookId,
        src_path: &Path,
        title: Option<&str>,
    ) -> Result<AddSourceOutcome, LensError> {
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

    /// Permanently deletes a source: drops Lance vectors first (Lance before SQLite
    /// ordering), then removes the `sources` row. Holds `ingest_lock` across the
    /// whole cross-store delete to prevent orphan Lance rows.
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
        // R7b: drop from EVERY active coordinate, not just the configured one.
        // A cross-backend switch can leave multiple active coordinates; only dropping
        // the configured one would leave the other backend's vectors dangling.
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
        // Best-effort: remove managed source file + siblings; a missing file is not an error.
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

    /// Retries a FAILED source in place (issue #73): guards it is `error` and
    /// live, transitions `error → parsing`, and re-runs the pipeline via the
    /// public [`ingest_source`](Self::ingest_source) entry (streaming through
    /// `on_progress`). See [`crate::ingest::retry_source`].
    #[tracing::instrument(skip(self, on_progress))]
    pub async fn retry_source(
        &self,
        source_id: &str,
        on_progress: impl FnMut(crate::ingest::IngestProgress),
    ) -> Result<(), LensError> {
        crate::ingest::retry_source(self, source_id, on_progress).await
    }

    pub(crate) async fn data_dir(&self) -> std::path::PathBuf {
        std::path::PathBuf::from(self.read().await.config.paths.data_dir.clone())
    }

    /// Test-only accessor: `pub(crate)` `data_dir` is unreachable from the test
    /// crate; this exposes it. Absent from production builds.
    #[cfg(feature = "test-util")]
    pub async fn data_dir_for_test(&self) -> std::path::PathBuf {
        self.data_dir().await
    }

    pub(crate) fn ingest_lock(&self) -> &Arc<Semaphore> {
        &self.ingest_lock
    }

    /// Installs (or replaces) the active enrichment LLM provider. `None` clears it
    /// (degrades to raw vectors). AC10 rebinding seam for unreachable→reachable.
    pub async fn set_llm_provider(&self, provider: Option<Arc<dyn LlmProvider>>) {
        *self.llm_provider.write().await = provider;
    }

    /// Returns a clone of the active enrichment provider handle (the worker reads
    /// this to decide whether to dispatch; `None` → degrade to raw vectors).
    pub async fn llm_provider(&self) -> Option<Arc<dyn LlmProvider>> {
        self.llm_provider.read().await.clone()
    }

    /// Installs (or replaces) the JS renderer for SPA URL-render fallback (issue
    /// #78). `None` degrades to `needs_js`. The concrete `TauriJsRenderer` is
    /// injected here because `lens-core` cannot depend on `tauri`.
    pub async fn set_js_renderer(&self, renderer: Option<Arc<dyn render::JsRenderer>>) {
        *self.js_renderer.write().await = renderer;
    }

    /// Returns a clone of the active JS renderer handle, or `None` when none is
    /// installed (the URL-ingest fallback then keeps the legacy `needs_js`
    /// behavior).
    pub async fn js_renderer(&self) -> Option<Arc<dyn render::JsRenderer>> {
        self.js_renderer.read().await.clone()
    }

    /// Non-blocking enqueue for background enrichment (AC3). Uses `try_send` so
    /// the ingest path never holds `ingest_lock` and a full channel cannot deadlock.
    /// A full/closed channel logs and drops the job; `rebuild_enrichment_queue` recovers it.
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

    /// Enqueues all indexed-but-not-yet-enriched sources (AC10/AC12). Called at
    /// startup and on provider unreachable→reachable transitions.
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

    /// AC10 back-fill hook: re-binds the provider from the current config and
    /// re-scans the queue if a provider is now installed. `enabled=false` clears
    /// the provider; `cloud_consent` gates cloud providers.
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

    /// Returns the model catalog (cached `models-catalog.json`, else the bundled
    /// snapshot). In-memory cache so repeated picker opens are a cheap pointer clone
    /// (fix #5). A cache miss loads via `spawn_blocking` and memoizes.
    #[tracing::instrument(skip_all)]
    pub async fn model_catalog(&self) -> Arc<crate::model_catalog::ModelCatalog> {
        if let Some(catalog) = self.catalog_cache.read().await.as_ref() {
            return catalog.clone();
        }
        let data_dir = self.data_dir().await;
        let loaded =
            tokio::task::spawn_blocking(move || crate::model_catalog::load_catalog(&data_dir))
                .await
                .map(Arc::new)
                // A JoinError (load task panicked) is non-fatal; fall back to the bundled snapshot.
                .unwrap_or_else(|e| {
                    tracing::warn!("model-catalog load task panicked; using bundled snapshot: {e}");
                    Arc::new(crate::model_catalog::ModelCatalog::bundled())
                });
        *self.catalog_cache.write().await = Some(loaded.clone());
        loaded
    }

    /// Forces an on-demand model-catalog refresh. Returns `Ok(true)` when the
    /// on-disk cache was rewritten; invalidates the in-memory cache so the next
    /// `model_catalog()` re-reads the fresh file (fix #5).
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
    /// delete its registry row. Static helper (no engine handle) so `init` can call
    /// it early. The `active` row is never touched.
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
        // Drop tables first (idempotent), then delete rows. A crash between the two
        // leaves only a dangling stale row; the next startup's GC no-ops the drop.
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

    /// Awaited inside the worker job body to hold it "in flight" for AC3 tests.
    #[cfg(feature = "test-util")]
    pub(crate) async fn enrichment_job_gate(&self) {
        let gate = self.enrichment_gate.read().await.clone();
        if let Some(notify) = gate {
            notify.notified().await;
        }
    }

    /// Installs (or clears) the worker job gate for AC3 tests.
    #[cfg(feature = "test-util")]
    pub async fn set_enrichment_gate_for_test(&self, gate: Option<Arc<tokio::sync::Notify>>) {
        *self.enrichment_gate.write().await = gate;
    }

    /// Awaited after populate and before the flip window for fix-#2 race tests.
    #[cfg(feature = "test-util")]
    pub(crate) async fn reembed_preflip_gate(&self) {
        let gate = self.reembed_preflip_gate.read().await.clone();
        if let Some(notify) = gate {
            notify.notified().await;
        }
    }

    /// Installs (or clears) the reembed pre-flip gate for fix-#2 tests.
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

    /// Drives the Step-5 re-embed flip directly so fix-#2 race tests can control
    /// the purge-vs-flip ordering. Absent from production builds.
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

    /// AC11 budget test seam: `0` restores the production default.
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

    /// Pre-fills the embedder cache so tests inject a `CountingEmbedder` instead of
    /// downloading a real model. Returns `Err` if an embedder for that key is already
    /// cached. Absent from production builds.
    #[cfg(feature = "test-util")]
    pub fn set_embedder_for_test(
        &self,
        embedder: Arc<dyn Embedder>,
        backend: crate::embedder::EmbeddingBackend,
    ) -> Result<(), LensError> {
        // Register under the key `embedder_for` resolves for a Bulk workload so the
        // injected double is found on Metal hardware too (issue #91).
        let spec = crate::embedder::resolve(embedder.model_id());
        let compute = crate::embedder::select_compute(
            self.accelerator.probe(),
            spec,
            backend,
            crate::embedder::WorkloadKind::Bulk,
        );
        let key = Self::embedder_cache_key(embedder.model_id(), backend, compute);
        // `try_lock` keeps this a sync fn safe inside `#[tokio::test]`: the cache
        // is uncontended at injection time.
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

    /// Returns the cached/lazily-built embedder for tests (`pub(crate)` `embedder_for`
    /// is unreachable from the test crate). Absent from production builds.
    #[cfg(feature = "test-util")]
    pub async fn embedder_for_test_get(
        &self,
        model_id: &str,
        backend: crate::embedder::EmbeddingBackend,
    ) -> Result<Arc<dyn Embedder>, LensError> {
        self.embedder_for(model_id, backend, crate::embedder::WorkloadKind::Bulk)
            .await
    }

    /// Cache key for a `(backend, model_id, compute)` triple. Backend and compute
    /// are both in the key so a fastembed/Metal pair never aliases a fastembed/CPU
    /// or Ollama entry for the same model. Format: `"{backend}:{model_id}:{compute}"`.
    fn embedder_cache_key(
        model_id: &str,
        backend: crate::embedder::EmbeddingBackend,
        compute: crate::embedder::Compute,
    ) -> String {
        format!("{}:{model_id}:{}", backend.as_str(), compute.as_str())
    }

    /// Lazily constructs and caches the embedder for `(model_id, backend, workload)`
    /// (R8). Cache hit returns the cached `Arc`. Cache miss resolves the spec,
    /// selects the device, and builds: `Fastembed` via `spawn_blocking` (ONNX, may
    /// download weights); `Ollama` via a lightweight client. The whole
    /// construct-and-insert holds the cache `Mutex` so init runs exactly once.
    pub(crate) async fn embedder_for(
        &self,
        model_id: &str,
        backend: crate::embedder::EmbeddingBackend,
        workload: crate::embedder::WorkloadKind,
    ) -> Result<Arc<dyn Embedder>, LensError> {
        let spec = crate::embedder::resolve(model_id);
        // Reject an unsupported (model, backend) pair early (issue #80) so callers
        // get a single clean error; construction-time guards are the backstop.
        if !spec.supports(backend) {
            return Err(LensError::Validation(format!(
                "model {} does not support the {} backend",
                spec.id,
                backend.as_str()
            )));
        }
        // Resolve the execution device (issue #91): Metal only for a GPU-eligible
        // bulk job on Apple Silicon; CPU everywhere else.
        let compute =
            crate::embedder::select_compute(self.accelerator.probe(), spec, backend, workload);
        let key = Self::embedder_cache_key(spec.id, backend, compute);
        let mut cache = self.embedders.lock().await;
        if let Some(existing) = cache.get(&key) {
            return Ok(Arc::clone(existing));
        }
        let embedder: Arc<dyn Embedder> = match backend {
            crate::embedder::EmbeddingBackend::Fastembed => {
                let data_dir = self.data_dir().await;
                // Try candle (Metal/CUDA) first; fall back to fastembed on any failure
                // or unsupported model. Never fail a job over a device choice.
                let gpu = if compute == crate::embedder::Compute::Cuda {
                    build_cuda_if_supported(compute, &data_dir, spec).await
                } else {
                    build_candle_if_supported(compute, &data_dir, spec).await
                };
                match gpu {
                    Some(e) => e,
                    None => {
                        let e = tokio::task::spawn_blocking(move || {
                            FastembedEmbedder::new_with_spec(&data_dir, spec)
                        })
                        .await
                        .map_err(|e| {
                            LensError::Model(format!("embedder init task panicked: {e}"))
                        })??;
                        Arc::new(e)
                    }
                }
            }
            crate::embedder::EmbeddingBackend::Ollama => {
                let base_url = ollama_base_url(&self.config().await);
                Arc::new(crate::embedder::OllamaEmbedder::new(&base_url, spec)?)
            }
        };
        cache.insert(key, Arc::clone(&embedder));
        Ok(embedder)
    }

    /// Warms (constructs + caches) the fastembed embedder for `model_id`.
    ///
    /// Uses `WorkloadKind::Interactive` so it always builds the CPU/ONNX fastembed
    /// engine (issue #91): the readiness gate checks the fastembed ONNX cache, so
    /// `Bulk` on Apple Silicon would download candle-Metal weights instead, leaving
    /// the gate unsatisfied. The candle-Metal weights download lazily on first ingest.
    pub async fn warm_fastembed_model(&self, model_id: &str) -> Result<(), LensError> {
        self.embedder_for(
            model_id,
            crate::embedder::EmbeddingBackend::Fastembed,
            crate::embedder::WorkloadKind::Interactive,
        )
        .await
        .map(|_| ())
    }

    /// Resolves a notebook's `(model_id, dim, backend)` embedding coordinate (R1,
    /// M4 Phase 4b-B). NULL/unknown columns fall back to registry/enum defaults.
    /// The canonical model id is safe to thread into `embedder_for` and `Coordinate`.
    pub async fn resolve_notebook_embedding(
        &self,
        notebook_id: &NotebookId,
    ) -> Result<(String, usize, crate::embedder::EmbeddingBackend), LensError> {
        let pool = self.pool().await;
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

    /// Persists a new embedding model/backend choice for a notebook. Rejects unknown
    /// model ids. Does NOT kick off re-embedding — the Tauri command layer calls
    /// `reembed_notebook` after persisting. Backend is the third coordinate axis (R2).
    pub async fn set_notebook_embedding_model(
        &self,
        notebook_id: &NotebookId,
        model_id: &str,
        backend: crate::embedder::EmbeddingBackend,
    ) -> Result<(), LensError> {
        // Persist the canonical spec.id (e.g. `nomic-embed-text` → `nomic-embed-text-v1.5`)
        // so downstream resolution is exact.
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

    /// Returns `(model_id, dim, backend, status)` for a notebook's embedding
    /// coordinate. Status query is backend-scoped (R4/R7a, M4 Phase 4b-B): a
    /// cross-backend switch can leave the old backend `stale` while the new one
    /// is `active`; a backend-blind query would report the wrong status.
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

    /// Re-embeds every chunk into the notebook's configured coordinate and retires
    /// previous coordinates (M4 Phase 4b, Step 9). Populate runs lock-free; only
    /// the brief flip takes `ingest_lock`. No-op when already at the active coordinate.
    #[tracing::instrument(skip_all, fields(notebook = %notebook_id.as_str()))]
    pub async fn reembed_notebook(
        &self,
        notebook_id: &NotebookId,
        on_progress: impl FnMut(usize, usize) + Send,
    ) -> Result<crate::enrichment::reembed::ReembedOutcome, LensError> {
        crate::enrichment::reembed::reembed_notebook(self, notebook_id, on_progress).await
    }

    /// Lazily resolves (once) and returns the shared nomic tokenizer, caching it
    /// so the multi-MB `tokenizer.json` is parsed exactly once per engine.
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

    /// Renames a notebook, bumping `updated_at` and `last_activity_at`.
    #[tracing::instrument(skip_all)]
    pub async fn rename_notebook(&self, id: &NotebookId, title: &str) -> Result<(), LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool).rename(id, title).await
    }

    /// Bumps a live notebook's `last_activity_at` (records an "open" for
    /// cold-launch MRU auto-open).
    #[tracing::instrument(skip_all)]
    pub async fn touch_notebook_activity(&self, id: &NotebookId) -> Result<(), LensError> {
        let pool = self.pool().await;
        NotebookRepo::new(&pool).touch_activity(id).await
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

    /// Permanently deletes a notebook. Drops Lance tables FIRST (Lance before
    /// SQLite) so the cascade that removes `embedding_index` rows cannot orphan
    /// them on disk. Holds `ingest_lock` across the cross-store delete.
    #[tracing::instrument(skip_all)]
    pub async fn purge_notebook(&self, id: &NotebookId) -> Result<(), LensError> {
        let _permit = self
            .ingest_lock()
            .acquire()
            .await
            .map_err(|e| LensError::Internal(format!("ingest semaphore closed: {e}")))?;
        let pool = self.pool().await;
        let data_dir = self.data_dir().await;
        // Capture (id, locator) pairs BEFORE the cascade deletes `sources` rows,
        // so managed files can be cleaned up afterwards.
        let sources: Vec<(String, String)> =
            sqlx::query_as("SELECT id, locator FROM sources WHERE notebook_id = ?")
                .bind(id.as_str())
                .fetch_all(&pool)
                .await?;
        let store = crate::vector_store::LanceVectorStore::new(&data_dir, pool.clone());
        store.drop_notebook_tables(id.as_str()).await?;
        NotebookRepo::new(&pool).purge(id).await?;
        for (source_id, locator) in &sources {
            remove_managed_source_file(&data_dir, source_id, locator);
        }
        Ok(())
    }
}

/// Best-effort removal of a managed source file and its `.extracted.txt` /
/// `.tables.md` siblings. Siblings are derived from `(data_dir, source_id)` via the
/// shared `ingest::*_sibling_path` builders — NOT from the locator — so URL sources
/// (whose locator is a URL string) are handled correctly. `NotFound` is silently ignored.
fn remove_managed_source_file(data_dir: &Path, source_id: &str, locator: &str) {
    remove_file_best_effort(Path::new(locator));
    let sibling = crate::ingest::extracted_sibling_path(data_dir, source_id);
    remove_file_best_effort(&sibling);
    // Unconditional + best-effort: non-tabular kinds produce no file so NotFound is ignored.
    let tables_sibling = crate::ingest::tables_sibling_path(data_dir, source_id);
    remove_file_best_effort(&tables_sibling);
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
