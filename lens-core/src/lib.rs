// issue #71: the streamed-ingest future deepened `Send` auto-trait evaluation
// enough to overflow the default 128-frame limit (E0275) on some toolchains.
// Compile-time only; no runtime cost.
#![recursion_limit = "256"]
//! `lens-core` — the headless Rust engine for LensLM.
//!
//! No Tauri, windowing, or UI dependencies. [`LensEngine`] is a thin handle
//! that delegates to per-domain repositories over the shared connection pool.

pub mod answer;
pub mod asr;
pub mod audio_overview;
pub mod chat;
pub mod chunk;
pub mod citation;
pub mod citation_source;
pub mod config;
pub(crate) mod db;
pub mod dialogue;
pub(crate) mod download;
pub mod embedder;
pub mod embedding;
pub mod enrichment;
pub mod error;
pub mod eval;
pub mod extract;
pub mod graph;
pub(crate) mod http;
pub mod ingest;
pub mod llm;
pub mod model_catalog;
pub mod notebooks;
pub mod notes;
pub mod parse;
pub mod paths;
pub(crate) mod prompt;
pub mod relocate;
pub mod render;
pub mod resolution;
pub mod retrieval;
pub mod storage;
pub mod system_check;
pub mod transcription;
pub mod tts;
pub mod url_normalize;
pub mod vector_store;

pub use answer::{AnswerEvent, AnswerStage};
// `AnswerCtx`/`answer_stream` are internal orchestrator seams; only the external
// integration-test crate needs them public, so gate behind `test-util`. The
// desktop bridge uses `answer_notebook` + `AnswerEvent`.
#[cfg(feature = "test-util")]
pub use answer::{AnswerCtx, answer_stream};
pub use audio_overview::{AudioOverviewRecord, AudioOverviewStatus};
// Dialogue-script domain types (#26) — unconditional `pub` so the cross-crate
// integration test sees them. `DialogueCtx`/`generate_dialogue` are the internal
// orchestrator seam, gated behind `test-util` like `AnswerCtx`/`answer_stream`.
#[cfg(feature = "test-util")]
pub use asr::MockAsrEngine;
#[cfg(feature = "local-whisper")]
pub use asr::WhisperEngine;
pub use asr::cloud::CloudAsrEngine;
pub use asr::{
    AsrBackend, AsrEngine, DEFAULT_WHISPER_MODEL_ID, Lang, MIN_MACOS_FOR_APPLE_ASR, Platform,
    TranscribeConfig, TranscriptOutput, TranscriptSegment, WHISPER_REGISTRY, WhisperModelSpec,
    download_whisper_model, resolve_whisper, select_asr_backend, whisper_model_downloaded,
    whisper_model_path,
};
pub use chat::{ChatFeedback, ChatMessage, ChatRole, ChatState};
pub use citation::{
    CITATION_PROMPT_INSTRUCTION, ChunkLocatorRow, Citation, Locator, extract_citations,
    hydrate_locators, load_chunk_locators,
};
pub use citation_source::{SnippetSegments, SourceView};
pub use config::{
    AppConfig, ChatConfig, CloudAsrProvider, CloudTtsConfig, EnrichmentConfig, RerankerConfig,
    RerankerModel, RetrievalConfig, TaskModel, TtsConfig, VoiceConfig, VoiceRef,
};
#[cfg(feature = "test-util")]
pub use dialogue::{DialogueCtx, generate_dialogue};
pub use dialogue::{DialoguePhase, DialogueScript, Emotion, Length, Speaker, Turn};
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
    ActiveModelCandidate, GenaiProvider, LlmProvider, LlmRequest, LlmResponse, LlmRouting,
    ReasoningEffort, StreamChunk, active_model_candidates, provider_from_config,
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
pub use notes::{Note, NoteId, NoteOrigin};
pub use paths::StoragePaths;
pub use render::JsRenderer;
pub use retrieval::router::{ContextUnit, Provenance, RouterOutput, Tier, tiered_search};
// Test-only: the integration test asserts `RESERVED_OUTPUT`'s value; production
// code reads it via `crate::retrieval::router`.
#[cfg(feature = "test-util")]
pub use retrieval::router::RESERVED_OUTPUT;
pub use retrieval::{HitSource, Reranker, RetrievalHit, hybrid_search};
pub use storage::StorageStats;
pub use system_check::{
    ALLOWED_EMBEDDING_MODELS, CheckAction, CheckId, CheckResult, CheckStatus, LlmDetection,
    ModelValidation, detect_llm, fastembed_weights_cached, is_allowlisted_embedding_id,
    list_ollama_models, ollama_base_url, validate_model_interactive,
};
pub use transcription::{WindowConfig, decode_and_resample_audio, decode_resample_windows};
// NOTE: `Lang`/`Platform`/`LanguageSupport` are intentionally NOT re-exported at
// the crate root — `asr::{Lang, Platform}` already own those names here. The TTS
// language types stay reachable via `lens_core::tts::{Lang, Platform, ...}`.
pub use tts::{
    AudioBuffer, CloudTtsKind, DownloadProgress, EngineCapability, EngineCatalogEntry, Gender,
    GuardVerdict, OffendingSource, QwenVoice, TTS_REGISTRY, TtsBackend, TtsEngineId, TtsModelSpec,
    TtsPhase, TtsProvider, TtsProviderInfo, TtsSidecar, TtsVoice, code_to_lang, download_tts_model,
    emotion_tag, evaluate_language_guard, lang_to_qwen_name, qwen_voice, read_wav_mono16,
    resolve_tts, resolve_tts_provider, resolve_tts_provider_full, tts_catalog,
    tts_catalog_serialized, tts_model_downloaded, tts_model_file_present, tts_model_path,
    validate_qwen_language,
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

pub(crate) fn f32_sample_to_i16(s: f32) -> i16 {
    (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

/// Builds the candle embedder for a fastembed-coordinate spec on Apple Silicon
/// (issue #91): Metal for Bulk, CPU for Interactive. Returns `None` — falling back
/// to fastembed — for unsupported models or any candle init failure.
#[cfg(feature = "native-ml-metal")]
async fn build_candle_if_supported(
    compute: crate::embedder::Compute,
    cache_root: &Path,
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
    let candle_dir = cache_root.join("models").join("candle");
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
    _cache_root: &Path,
    _spec: &'static crate::embedder::EmbeddingModelSpec,
) -> Option<Arc<dyn Embedder>> {
    None
}

/// Interface stub for CUDA embedding (issue #91). Always returns `None` until a
/// candle-CUDA backend is implemented; CUDA jobs fall back to fastembed-CPU.
#[cfg(feature = "native-ml-cuda")]
async fn build_cuda_if_supported(
    _compute: crate::embedder::Compute,
    _cache_root: &Path,
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
    _cache_root: &Path,
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
    /// Separate channel/worker from enrichment; fired only after a job fully succeeds.
    /// `Clone` rides `#[derive(Clone)]`.
    resolution_tx: mpsc::Sender<crate::resolution::ResolveNotebook>,
    notebook_locks: Arc<std::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
    /// Active enrichment LLM provider. `RwLock<Option<...>>` rather than `OnceCell`
    /// because AC10 requires rebinding on an unreachable→reachable transition.
    llm_provider: Arc<RwLock<Option<Arc<dyn LlmProvider>>>>,
    /// Injected JS renderer for SPA URL-render fallback (issue #78). `None` in
    /// headless tests or before `src-tauri` wires `TauriJsRenderer`; degrades to
    /// `needs_js` when absent.
    js_renderer: Arc<RwLock<Option<Arc<dyn render::JsRenderer>>>>,
    /// Injected Apple-native ASR engine (#42). `None` on non-Apple targets or
    /// before src-tauri wires `AppleSpeechEngine`. Presence is the committed
    /// `apple_available` signal: src-tauri pre-gates platform/version before
    /// injecting, so the router treats mere presence as authoritative.
    asr_engine: Arc<RwLock<Option<Arc<dyn asr::AsrEngine>>>>,
    /// In-flight audio-ingest cancellation tokens, keyed by `source_id` (#43).
    /// A token is inserted before audio decode starts and removed on
    /// completion/error/cancel. `std::sync::Mutex` (not tokio) since operations
    /// are fast HashMap lookups never held across an await point.
    media_cancel_tokens:
        Arc<std::sync::Mutex<HashMap<String, tokio_util::sync::CancellationToken>>>,
    /// In-flight grounded-answer (#173) cancellation tokens, keyed by `notebook_id`
    /// (single-flight per notebook). Diverges from `media_cancel_tokens`: the token
    /// is `Arc`-wrapped so [`AskCancelGuard`] can `Arc::ptr_eq`-gate its Drop and a
    /// superseded old ask's guard never evicts the new ask's token (ABA-safe). A new
    /// ask actively `.cancel()`s the evicted token so the prior stream stops.
    ask_cancel_tokens:
        Arc<std::sync::Mutex<HashMap<String, Arc<tokio_util::sync::CancellationToken>>>>,
    /// In-flight dialogue-script (#26) cancellation tokens, keyed by `notebook_id`
    /// (single-flight per notebook). A DEDICATED registry, separate from
    /// `ask_cancel_tokens`: a dialogue cancel must not stop an in-flight chat on the
    /// same notebook (and vice-versa). Same ABA-safe `Arc::ptr_eq` guard shape as
    /// the ask registry.
    dialogue_cancel_tokens:
        Arc<std::sync::Mutex<HashMap<String, Arc<tokio_util::sync::CancellationToken>>>>,
    tts_cancel_tokens:
        Arc<std::sync::Mutex<HashMap<String, Arc<tokio_util::sync::CancellationToken>>>>,
    tts_sidecar: Arc<RwLock<Option<Arc<dyn tts::TtsSidecar>>>>,
    /// Lazily-built internal LocalWhisper engines, keyed by model id (#42). Mirrors
    /// the embedder cache but lighter — whisper has one active model at a time. The
    /// single `Mutex` over the map ensures the ggml load runs exactly once per key.
    #[cfg(feature = "local-whisper")]
    whisper_engines: Arc<Mutex<HashMap<String, Arc<asr::WhisperEngine>>>>,
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
    /// #155 test seam: counts completed resolution passes (drain-coalesce assertion).
    #[cfg(feature = "test-util")]
    resolution_pass_count: Arc<std::sync::atomic::AtomicU32>,
    /// #155 test seam: when `true`, `write_resolution_updates` aborts its txn AFTER the
    /// version stamp but BEFORE the canonical updates (single-txn atomicity assertion).
    #[cfg(feature = "test-util")]
    resolution_write_fault: Arc<std::sync::atomic::AtomicBool>,
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
        let (resolution_tx, resolution_rx) = mpsc::channel::<crate::resolution::ResolveNotebook>(
            enrichment::ENRICHMENT_QUEUE_CAPACITY,
        );

        let engine = Self {
            inner: Arc::new(RwLock::new(LensEngineInner { db, config })),
            embedders: Arc::new(Mutex::new(HashMap::new())),
            accelerator: crate::embedder::default_accelerator(),
            tokenizer: Arc::new(OnceCell::new()),
            ingest_lock: Arc::new(Semaphore::new(1)),
            enrichment_tx,
            resolution_tx,
            notebook_locks: Arc::new(std::sync::Mutex::new(HashMap::new())),
            llm_provider: Arc::new(RwLock::new(None)),
            js_renderer: Arc::new(RwLock::new(None)),
            asr_engine: Arc::new(RwLock::new(None)),
            media_cancel_tokens: Arc::new(std::sync::Mutex::new(HashMap::new())),
            ask_cancel_tokens: Arc::new(std::sync::Mutex::new(HashMap::new())),
            dialogue_cancel_tokens: Arc::new(std::sync::Mutex::new(HashMap::new())),
            tts_cancel_tokens: Arc::new(std::sync::Mutex::new(HashMap::new())),
            tts_sidecar: Arc::new(RwLock::new(None)),
            #[cfg(feature = "local-whisper")]
            whisper_engines: Arc::new(Mutex::new(HashMap::new())),
            catalog_cache: Arc::new(RwLock::new(None)),
            #[cfg(feature = "test-util")]
            enrichment_gate: Arc::new(RwLock::new(None)),
            #[cfg(feature = "test-util")]
            reembed_preflip_gate: Arc::new(RwLock::new(None)),
            #[cfg(feature = "test-util")]
            skip_tokenizer: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            #[cfg(feature = "test-util")]
            enrichment_max_calls_override: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            #[cfg(feature = "test-util")]
            resolution_pass_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            #[cfg(feature = "test-util")]
            resolution_write_fault: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };

        enrichment::spawn_worker(engine.clone(), enrichment_rx);
        crate::resolution::spawn_resolution_worker(engine.clone(), resolution_rx);

        // Best-effort model-catalog refresh at startup; a slow/failed fetch degrades
        // to the cached/bundled copy — never blocks init.
        {
            let cache_root = engine.cache_root().await;
            tokio::spawn(async move {
                let client = crate::model_catalog::catalog_client();
                if let Err(e) = crate::model_catalog::refresh_if_stale(
                    &cache_root,
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
        let (resolution_tx, resolution_rx) = mpsc::channel::<crate::resolution::ResolveNotebook>(
            enrichment::ENRICHMENT_QUEUE_CAPACITY,
        );
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
            resolution_tx,
            notebook_locks: Arc::new(std::sync::Mutex::new(HashMap::new())),
            llm_provider: Arc::new(RwLock::new(None)),
            js_renderer: Arc::new(RwLock::new(None)),
            asr_engine: Arc::new(RwLock::new(None)),
            media_cancel_tokens: Arc::new(std::sync::Mutex::new(HashMap::new())),
            ask_cancel_tokens: Arc::new(std::sync::Mutex::new(HashMap::new())),
            dialogue_cancel_tokens: Arc::new(std::sync::Mutex::new(HashMap::new())),
            tts_cancel_tokens: Arc::new(std::sync::Mutex::new(HashMap::new())),
            tts_sidecar: Arc::new(RwLock::new(None)),
            #[cfg(feature = "local-whisper")]
            whisper_engines: Arc::new(Mutex::new(HashMap::new())),
            catalog_cache: Arc::new(RwLock::new(None)),
            #[cfg(feature = "test-util")]
            enrichment_gate: Arc::new(RwLock::new(None)),
            #[cfg(feature = "test-util")]
            reembed_preflip_gate: Arc::new(RwLock::new(None)),
            #[cfg(feature = "test-util")]
            skip_tokenizer: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            #[cfg(feature = "test-util")]
            enrichment_max_calls_override: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            #[cfg(feature = "test-util")]
            resolution_pass_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            #[cfg(feature = "test-util")]
            resolution_write_fault: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        enrichment::spawn_worker(engine.clone(), enrichment_rx);
        crate::resolution::spawn_resolution_worker(engine.clone(), resolution_rx);
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

    /// Persists a user chat message on send (#22). `turn_id` is the frontend-minted
    /// grouping key the assistant row will later share.
    #[tracing::instrument(skip_all)]
    pub async fn save_chat_user(
        &self,
        notebook_id: &NotebookId,
        turn_id: &str,
        content: &str,
    ) -> Result<ChatMessage, LensError> {
        let pool = self.pool().await;
        crate::chat::ChatRepo::new(&pool)
            .insert_user(notebook_id.as_str(), turn_id, content)
            .await
    }

    /// Persists an assistant chat message on stream `Done` (#22). Serializes the
    /// citation payload to JSON internally (the engine owns the JSON contract).
    #[tracing::instrument(skip_all)]
    pub async fn save_chat_assistant(
        &self,
        notebook_id: &NotebookId,
        turn_id: &str,
        content: &str,
        citations: Option<&[Citation]>,
        tokens_used: u32,
    ) -> Result<ChatMessage, LensError> {
        let citations_json = match citations {
            None => None,
            Some(c) => Some(
                serde_json::to_string(c)
                    .map_err(|e| LensError::Internal(format!("citations serialize failed: {e}")))?,
            ),
        };
        let pool = self.pool().await;
        crate::chat::ChatRepo::new(&pool)
            .insert_assistant(
                notebook_id.as_str(),
                turn_id,
                content,
                citations_json.as_deref(),
                i64::from(tokens_used),
            )
            .await
    }

    /// Persists a terminal-state marker for a turn that ended without a normal
    /// `Done` (Plan 2 / PC-1); see
    /// [`ChatRepo::insert_terminal_marker`](crate::chat::ChatRepo::insert_terminal_marker).
    #[tracing::instrument(skip_all)]
    pub async fn save_chat_marker(
        &self,
        notebook_id: &NotebookId,
        turn_id: &str,
        content: &str,
        state: crate::chat::ChatState,
        error_kind: Option<&str>,
    ) -> Result<ChatMessage, LensError> {
        let pool = self.pool().await;
        crate::chat::ChatRepo::new(&pool)
            .insert_terminal_marker(notebook_id.as_str(), turn_id, content, state, error_kind)
            .await
    }

    /// Sets or clears (`None`) feedback on a chat message (#22, toggleable thumbs).
    #[tracing::instrument(skip_all)]
    pub async fn set_chat_feedback(
        &self,
        message_id: &str,
        feedback: Option<ChatFeedback>,
    ) -> Result<(), LensError> {
        let pool = self.pool().await;
        crate::chat::ChatRepo::new(&pool)
            .set_feedback(message_id, feedback)
            .await
    }

    /// Lists a notebook's chat messages as flat rows in transcript order (#22).
    #[tracing::instrument(skip_all)]
    pub async fn list_chat_messages(
        &self,
        notebook_id: &NotebookId,
    ) -> Result<Vec<ChatMessage>, LensError> {
        let pool = self.pool().await;
        crate::chat::ChatRepo::new(&pool)
            .list(notebook_id.as_str())
            .await
    }

    /// Saves a completed grounded answer as a durable `origin=chat` note (#24).
    /// Serializes the citation payload to JSON internally (engine owns the JSON
    /// contract) and freezes `source_title` = the title of the ordinal-1 citation's
    /// source, resolved once at save (`None` when there are no citations).
    #[tracing::instrument(skip_all)]
    pub async fn save_chat_note(
        &self,
        notebook_id: &NotebookId,
        content: &str,
        citations: Option<&[Citation]>,
        source_message_id: &str,
    ) -> Result<Note, LensError> {
        let citations_json = match citations {
            None => None,
            Some(c) => Some(
                serde_json::to_string(c)
                    .map_err(|e| LensError::Internal(format!("citations serialize failed: {e}")))?,
            ),
        };
        let pool = self.pool().await;
        let source_title = match citations
            .and_then(|c| c.iter().find(|cit| cit.ordinal == 1))
            .map(|cit| cit.source_id.as_str())
        {
            None => None,
            Some(sid) => crate::citation::source_titles(&pool, &[sid])
                .await?
                .remove(sid),
        };
        crate::notes::NotesRepo::new(&pool)
            .create_chat_note(
                notebook_id.as_str(),
                content,
                citations_json.as_deref(),
                source_title.as_deref(),
                source_message_id,
            )
            .await
    }

    /// Saves a user-authored manual note (#25). Rejects empty/whitespace-only
    /// content; the persisted note has no citations, `source_title`, or
    /// `source_message_id`.
    #[tracing::instrument(skip_all)]
    pub async fn save_manual_note(
        &self,
        notebook_id: &NotebookId,
        content: &str,
    ) -> Result<Note, LensError> {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Err(LensError::Validation("note content is empty".into()));
        }
        let pool = self.pool().await;
        crate::notes::NotesRepo::new(&pool)
            .create_manual_note(notebook_id.as_str(), trimmed)
            .await
    }

    /// Lists a notebook's notes, newest first (#24).
    #[tracing::instrument(skip_all)]
    pub async fn list_notes(&self, notebook_id: &NotebookId) -> Result<Vec<Note>, LensError> {
        let pool = self.pool().await;
        crate::notes::NotesRepo::new(&pool)
            .list_notes(notebook_id.as_str())
            .await
    }

    /// Updates a note's content (#25). Rejects empty/whitespace-only content
    /// (mirrors `save_manual_note`; this wrapper is the authoritative server-side
    /// guard — the IPC command must route through it, never the repo directly);
    /// grounding columns and `created_at` are preserved, `updated_at` bumped.
    #[tracing::instrument(skip_all)]
    pub async fn update_note(&self, note_id: &str, content: &str) -> Result<Note, LensError> {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Err(LensError::Validation("note content is empty".into()));
        }
        let pool = self.pool().await;
        crate::notes::NotesRepo::new(&pool)
            .update_note(note_id, trimmed)
            .await
    }

    /// Sets a note's pinned flag (#25, pin-to-top).
    #[tracing::instrument(skip_all)]
    pub async fn set_note_pinned(&self, note_id: &str, pinned: bool) -> Result<Note, LensError> {
        let pool = self.pool().await;
        crate::notes::NotesRepo::new(&pool)
            .set_pinned(note_id, pinned)
            .await
    }

    /// Deletes a note by id (#24, unsave path).
    #[tracing::instrument(skip_all)]
    pub async fn delete_note(&self, note_id: &str) -> Result<(), LensError> {
        let pool = self.pool().await;
        crate::notes::NotesRepo::new(&pool)
            .delete_note(note_id)
            .await
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
            // #155: entity-vector drop (same ordering as the chunk-vector drop above).
            store.drop_entity_source(&coord, source_id).await?;
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

    /// Offload root for re-downloadable model/cache dirs (#238): `paths.cache_dir`
    /// when set, else `data_dir`. Read config + data_dir under one lock acquisition.
    pub(crate) async fn cache_root(&self) -> std::path::PathBuf {
        let guard = self.read().await;
        let data_dir = std::path::PathBuf::from(&guard.config.paths.data_dir);
        guard.config.cache_root(&data_dir)
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

    /// Byte-usage breakdown of the data directory (corpus / reclaimable cache /
    /// retained). Read-only. The recursive fs walk runs under `spawn_blocking`
    /// so a multi-GB directory never blocks an async executor thread; missing
    /// dirs on a fresh install read as 0.
    pub async fn storage_stats(&self) -> Result<StorageStats, LensError> {
        let data_dir = self.data_dir().await;
        let config = self.config().await;
        let embedding_model = config.embedding_model.clone();
        let paths = crate::paths::StoragePaths::from_config(&config, &data_dir);
        tokio::task::spawn_blocking(move || {
            crate::storage::storage_stats_blocking(&paths, &embedding_model)
        })
        .await
        .map_err(|e| LensError::Internal(format!("storage_stats task join failed: {e}")))?
    }

    /// Invariant: holds `ingest_lock` so it can't race an in-flight ingest/embed; the
    /// Storage panel disables it while an audio overview generates; a concurrent raw ASR
    /// transcription is unguarded but at worst re-downloads its model afterward (no data loss).
    pub async fn clear_model_cache(&self) -> Result<u64, LensError> {
        let _permit = self
            .ingest_lock()
            .acquire()
            .await
            .map_err(|e| LensError::Internal(format!("ingest semaphore closed: {e}")))?;
        let data_dir = self.data_dir().await;
        let config = self.config().await;
        let embedding_model = config.embedding_model.clone();
        let paths = crate::paths::StoragePaths::from_config(&config, &data_dir);
        tokio::task::spawn_blocking(move || {
            crate::storage::clear_model_cache_blocking(&paths, &embedding_model)
        })
        .await
        .map_err(|e| LensError::Internal(format!("clear_model_cache task join failed: {e}")))?
    }

    /// Copies the data dir to `to`, verifies the snapshot, and rewrites absolute-path
    /// DB columns in the copy (#238). Holds `ingest_lock` so no ingest/embed/purge runs
    /// during the copy. The live pool is untouched — the caller writes the anchor
    /// pointer and prompts a restart, after which [`crate::relocate::resolve_data_dir`]
    /// re-points the engine. Returns the current (old) data dir so the caller can record
    /// it for cleanup.
    pub async fn relocate_data_dir(
        &self,
        to: &std::path::Path,
    ) -> Result<std::path::PathBuf, LensError> {
        let _permit = self
            .ingest_lock()
            .acquire()
            .await
            .map_err(|e| LensError::Internal(format!("ingest semaphore closed: {e}")))?;
        let from = self.data_dir().await;
        let pool = self.pool().await;
        crate::relocate::relocate_data_dir(&pool, &from, to).await?;
        Ok(from)
    }

    /// Moves the model cache to `to_cache_root` (offload, #238) or, when `to_cache_root`
    /// is `None`, back under the data dir (reset). Persists `paths.cache_dir` and applies
    /// it in-memory so subsequent model loads resolve under the new root. Holds
    /// `ingest_lock` so no embed runs mid-move. Returns bytes moved.
    pub async fn offload_cache(
        &self,
        to_cache_root: Option<&std::path::Path>,
    ) -> Result<u64, LensError> {
        let _permit = self
            .ingest_lock()
            .acquire()
            .await
            .map_err(|e| LensError::Internal(format!("ingest semaphore closed: {e}")))?;
        let data_dir = self.data_dir().await;
        let mut config = self.config().await;
        let from = config.cache_root(&data_dir);
        let to = to_cache_root
            .map(|p| p.to_path_buf())
            .unwrap_or(data_dir.clone());
        let moved = {
            let to = to.clone();
            tokio::task::spawn_blocking(move || crate::relocate::offload_cache(&from, &to))
                .await
                .map_err(|e| LensError::Internal(format!("offload task join failed: {e}")))??
        };
        config.paths.cache_dir = to_cache_root.map(|p| p.display().to_string());
        config.save(&data_dir)?;
        self.set_config(config).await;
        Ok(moved)
    }

    /// Returns the per-notebook write lock, creating it on first use. The enrichment
    /// writer (around `write_enrichment_and_graph`) and the #155 resolution pass both
    /// acquire this same lock so the two `entity_nodes` writers never interleave. The
    /// inner `std::sync::Mutex` is held only for the fast get-or-insert (never across
    /// an await); the returned `tokio::sync::Mutex` is the actual serialization point.
    pub(crate) fn notebook_lock(&self, notebook_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self
            .notebook_locks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        locks
            .entry(notebook_id.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
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

    /// Resolves the provider for interactive CHAT/dialogue (CX-5), decoupled from
    /// `enrichment.enabled` (that flag gates only the enrichment worker). Variant B: a
    /// purpose-built `enrichment.chat_model` pin is authoritative when set — it does not
    /// fall back to routing or the cached enrichment provider, so an unusable pin reports
    /// "no chat provider". With no pin, reuses the installed enrichment provider else
    /// builds one from routing. `None` when nothing usable is configured.
    pub async fn chat_provider(&self) -> Option<Arc<dyn LlmProvider>> {
        let config = self.config().await;
        if config.enrichment.chat_model.is_some() {
            return crate::llm::chat_provider_from_config(&config, config.enrichment.cloud_consent);
        }
        if let Some(p) = self.llm_provider().await {
            return Some(p);
        }
        crate::llm::chat_provider_from_config(&config, config.enrichment.cloud_consent)
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

    pub async fn set_tts_sidecar(&self, sidecar: Option<Arc<dyn tts::TtsSidecar>>) {
        *self.tts_sidecar.write().await = sidecar;
    }

    pub async fn tts_sidecar(&self) -> Option<Arc<dyn tts::TtsSidecar>> {
        self.tts_sidecar.read().await.clone()
    }

    /// Installs (or replaces) the Apple-native ASR engine (#42). `None` leaves the
    /// engine absent, so [`transcribe`](Self::transcribe) routes to LocalWhisper.
    /// The concrete `AppleSpeechEngine` is injected here because `lens-core` cannot
    /// depend on `tauri`/Speech.framework; src-tauri pre-gates platform/version
    /// before calling this, so presence is the authoritative `apple_available` signal.
    pub async fn set_asr_engine(&self, engine: Option<Arc<dyn asr::AsrEngine>>) {
        *self.asr_engine.write().await = engine;
    }

    /// Returns a clone of the injected Apple-native ASR engine, or `None` when
    /// none is installed (LocalWhisper is then the routed backend).
    pub async fn asr_engine(&self) -> Option<Arc<dyn asr::AsrEngine>> {
        self.asr_engine.read().await.clone()
    }

    /// Registers a fresh cancellation token for an in-flight audio ingest,
    /// keyed by `source_id`, and returns a clone the ingest branch checks at
    /// decode-window boundaries and before transcription (issue #43). Replaces
    /// any prior token for the same id (a retry supersedes the stale one).
    pub fn register_media_cancel(&self, source_id: &str) -> tokio_util::sync::CancellationToken {
        let token = tokio_util::sync::CancellationToken::new();
        let mut map = self
            .media_cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        map.insert(source_id.to_string(), token.clone());
        token
    }

    /// Cancels the in-flight audio ingest for `source_id` by flipping its token.
    /// Returns `true` if a token was found and cancelled, `false` if no audio
    /// ingest is in flight for that id (issue #43).
    pub fn cancel_media_ingest(&self, source_id: &str) -> bool {
        let map = self
            .media_cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match map.get(source_id) {
            Some(token) => {
                token.cancel();
                true
            }
            None => false,
        }
    }

    /// Removes the cancellation token for `source_id` (issue #43). Called on
    /// audio-ingest completion/error/cancel via a drop guard so the registry
    /// never retains a stale entry.
    pub fn remove_media_cancel(&self, source_id: &str) {
        let mut map = self
            .media_cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        map.remove(source_id);
    }

    /// Registers a fresh single-flight cancellation token for a grounded-answer run,
    /// keyed by `notebook_id` (#173). A prior in-flight ask for this notebook is
    /// `.cancel()`d FIRST (so the superseded stream stops), then replaced. Returns the
    /// `Arc`-wrapped token; the caller passes the `Arc` to
    /// [`ask_cancel_guard`](Self::ask_cancel_guard) for the `Arc::ptr_eq` Drop match.
    pub fn register_ask(&self, notebook_id: &str) -> Arc<tokio_util::sync::CancellationToken> {
        let token = Arc::new(tokio_util::sync::CancellationToken::new());
        let mut map = self
            .ask_cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(prev) = map.insert(notebook_id.to_string(), token.clone()) {
            prev.cancel();
        }
        token
    }

    /// Cancels the in-flight grounded-answer run for `notebook_id` by flipping its
    /// token. Returns `true` if one was found, `false` otherwise (#173).
    pub fn cancel_ask_notebook(&self, notebook_id: &str) -> bool {
        let map = self
            .ask_cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match map.get(notebook_id) {
            Some(token) => {
                token.cancel();
                true
            }
            None => false,
        }
    }

    /// Removes the ask token for `notebook_id` only when `owner` is still the stored
    /// token (`Arc::ptr_eq`). See the `ask_cancel_tokens` field doc for the ABA rationale.
    fn remove_ask_if_owner(
        &self,
        notebook_id: &str,
        owner: &Arc<tokio_util::sync::CancellationToken>,
    ) {
        let mut map = self
            .ask_cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(current) = map.get(notebook_id)
            && Arc::ptr_eq(current, owner)
        {
            map.remove(notebook_id);
        }
    }

    /// Builds the RAII [`AskCancelGuard`] for a registered ask; `owner` is the `Arc`
    /// from [`register_ask`] that the guard's Drop `Arc::ptr_eq`-matches on removal.
    pub fn ask_cancel_guard(
        &self,
        notebook_id: &str,
        owner: Arc<tokio_util::sync::CancellationToken>,
    ) -> AskCancelGuard {
        AskCancelGuard {
            engine: self.clone(),
            notebook_id: notebook_id.to_string(),
            owner,
        }
    }

    /// Registers a fresh single-flight cancellation token for a dialogue-script run
    /// (#26), keyed by `notebook_id`. A prior in-flight dialogue generation for this
    /// notebook is `.cancel()`d FIRST, then replaced. Dedicated registry — never the
    /// ask registry (a dialogue cancel must not stop an in-flight chat).
    pub fn register_dialogue(&self, notebook_id: &str) -> Arc<tokio_util::sync::CancellationToken> {
        let token = Arc::new(tokio_util::sync::CancellationToken::new());
        let mut map = self
            .dialogue_cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(prev) = map.insert(notebook_id.to_string(), token.clone()) {
            prev.cancel();
        }
        token
    }

    /// Cancels the in-flight dialogue-script run for `notebook_id` by flipping its
    /// token. Returns `true` if one was found, `false` otherwise (#26).
    pub fn cancel_dialogue_generation(&self, notebook_id: &str) -> bool {
        let map = self
            .dialogue_cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match map.get(notebook_id) {
            Some(token) => {
                token.cancel();
                true
            }
            None => false,
        }
    }

    /// Removes the dialogue token for `notebook_id` only when `owner` is still the
    /// stored token (`Arc::ptr_eq`), so a superseded run's guard never evicts the
    /// live token (ABA-safe, mirroring `remove_ask_if_owner`).
    fn remove_dialogue_if_owner(
        &self,
        notebook_id: &str,
        owner: &Arc<tokio_util::sync::CancellationToken>,
    ) {
        let mut map = self
            .dialogue_cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(current) = map.get(notebook_id)
            && Arc::ptr_eq(current, owner)
        {
            map.remove(notebook_id);
        }
    }

    /// Builds the RAII [`DialogueCancelGuard`] for a registered dialogue run; `owner`
    /// is the `Arc` from [`register_dialogue`] the guard's Drop `Arc::ptr_eq`-matches.
    pub fn dialogue_cancel_guard(
        &self,
        notebook_id: &str,
        owner: Arc<tokio_util::sync::CancellationToken>,
    ) -> DialogueCancelGuard {
        DialogueCancelGuard {
            engine: self.clone(),
            notebook_id: notebook_id.to_string(),
            owner,
        }
    }

    pub fn register_tts(&self, notebook_id: &str) -> Arc<tokio_util::sync::CancellationToken> {
        let token = Arc::new(tokio_util::sync::CancellationToken::new());
        let mut map = self
            .tts_cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(prev) = map.insert(notebook_id.to_string(), token.clone()) {
            prev.cancel();
        }
        token
    }

    pub fn cancel_synthesis(&self, notebook_id: &str) -> bool {
        let map = self
            .tts_cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match map.get(notebook_id) {
            Some(token) => {
                token.cancel();
                true
            }
            None => false,
        }
    }

    // `Arc::ptr_eq` gate: a superseded run's guard must not evict the live token (ABA-safe).
    fn remove_tts_if_owner(
        &self,
        notebook_id: &str,
        owner: &Arc<tokio_util::sync::CancellationToken>,
    ) {
        let mut map = self
            .tts_cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(current) = map.get(notebook_id)
            && Arc::ptr_eq(current, owner)
        {
            map.remove(notebook_id);
        }
    }

    pub fn tts_cancel_guard(
        &self,
        notebook_id: &str,
        owner: Arc<tokio_util::sync::CancellationToken>,
    ) -> TtsCancelGuard {
        TtsCancelGuard {
            engine: self.clone(),
            notebook_id: notebook_id.to_string(),
            owner,
        }
    }

    pub async fn tts_backend_available(&self, cfg: &config::TtsConfig) -> bool {
        // Qwen3Local has no registry-managed model: `mlx-audio` fetches it lazily
        // on first synth, so availability is just "is a sidecar injected" — no
        // spawn/health probe (that would cost a multi-GB load on a UI gate).
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if matches!(cfg.backend, tts::TtsBackend::Qwen3Local) {
            return self.tts_sidecar().await.is_some();
        }
        let cache_root = self.cache_root().await;
        let required = cfg.backend.required_model_ids();
        !required.is_empty()
            && required
                .iter()
                .all(|id| tts::tts_model_downloaded(&cache_root, id))
    }

    pub async fn synthesize_overview(
        &self,
        notebook_id: &str,
        script: &dialogue::DialogueScript,
        voices: &config::VoiceConfig,
        cfg: &config::TtsConfig,
        on_phase: impl Fn(tts::TtsPhase) + Send + Sync,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> Result<std::path::PathBuf, LensError> {
        // Single dispatch path (161e): `_full` returns the sidecar-backed adapter
        // (Qwen3Local) when a sidecar is injected, else the embedded provider
        // (Orpheus). `data_dir` supplies embedded model paths.
        let data_dir = self.data_dir().await;
        let cache_root = self.cache_root().await;
        let provider =
            tts::resolve_tts_provider_full(cfg.backend, cfg, &cache_root, self.tts_sidecar().await)
                .ok_or_else(|| LensError::Tts("no TTS backend available".into()))?;
        let buffer: tts::AudioBuffer = provider
            .synthesize_script(script, voices, &on_phase, cancel)
            .await?;

        on_phase(tts::TtsPhase::Encoding);
        let path = notebook_audio_path(&data_dir, notebook_id)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(LensError::from)?;
        }
        // Atomic overwrite: write a sibling temp then rename, so a crash mid-write never
        // leaves a torn `overview.wav` that a `ready` row would point at. The temp name is
        // per-run unique so concurrent same-notebook runs can't collide on it.
        let tmp = path.with_extension(format!("wav.{}.tmp", uuid::Uuid::now_v7()));
        if let Err(e) = tts::write_wav_16bit(&buffer, &tmp) {
            remove_file_best_effort(&tmp);
            return Err(e);
        }
        // A late cancel (after the buffer was produced) must not leave a Ready overview:
        // drop the temp and let the orchestrator settle to Ok(None).
        if cancel.is_cancelled() {
            remove_file_best_effort(&tmp);
            return Err(LensError::Cancelled("overview synthesis cancelled".into()));
        }
        if let Err(e) = std::fs::rename(&tmp, &path) {
            remove_file_best_effort(&tmp);
            return Err(LensError::from(e));
        }
        Ok(path)
    }

    /// [M3] Order-independent hash over the selected + live source set (the exact
    /// grounding set the dialogue is built from) combined with each source's
    /// `raw_content_hash`. Captured at generation start and stored so a later source
    /// change surfaces as a `Stale` overview. Uses the same `SELECTED_LIVE_WHERE`
    /// predicate as `selected_live_source_ids`.
    async fn source_set_hash(&self, notebook_id: &str) -> Result<String, LensError> {
        use sha2::Digest;
        let pool = self.pool().await;
        let sql = format!(
            "SELECT id, raw_content_hash FROM sources WHERE {}",
            crate::retrieval::router::SELECTED_LIVE_WHERE
        );
        let mut pairs = sqlx::query_as::<_, (String, Option<String>)>(&sql)
            .bind(notebook_id)
            .fetch_all(&pool)
            .await?;
        pairs.sort();
        let mut hasher = sha2::Sha256::new();
        for (id, content_hash) in &pairs {
            hasher.update(id.as_bytes());
            hasher.update([0u8]);
            hasher.update(content_hash.as_deref().unwrap_or("").as_bytes());
            hasher.update([0u8]);
        }
        Ok(crate::hex_encode(&hasher.finalize()))
    }

    /// [#29] Single-owner orchestration: dialogue → synth (atomic write) → persist the
    /// terminal `audio_overviews` row. `Ok(Some(path))` = success (`ready`), `Ok(None)` =
    /// user cancel ([M2], no row written), `Err` = genuine failure (`failed` persisted).
    pub async fn generate_and_persist_overview(
        &self,
        notebook_id: &str,
        length: dialogue::Length,
        on_phase: impl Fn(tts::TtsPhase) + Send + Sync,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<Option<std::path::PathBuf>, LensError> {
        // Cancelled before any work started → idle, nothing persisted.
        if cancel.is_cancelled() {
            return Ok(None);
        }

        let source_set_hash = self.source_set_hash(notebook_id).await?;

        let script = match self
            .generate_dialogue(
                &NotebookId::from(notebook_id.to_string()),
                length,
                cancel.clone(),
                |_phase| {},
            )
            .await
        {
            Ok(script) => script,
            Err(LensError::Cancelled(_)) => return Ok(None),
            Err(err) => {
                self.persist_overview_failed(notebook_id, &source_set_hash)
                    .await?;
                return Err(err);
            }
        };

        let config = self.config().await;
        let path = match self
            .synthesize_overview(
                notebook_id,
                &script,
                &config.voices,
                &config.tts,
                on_phase,
                &cancel,
            )
            .await
        {
            Ok(path) => path,
            Err(LensError::Cancelled(_)) => return Ok(None),
            Err(err) => {
                self.persist_overview_failed(notebook_id, &source_set_hash)
                    .await?;
                return Err(err);
            }
        };

        let generated_at = chrono::Utc::now().to_rfc3339();
        audio_overview::upsert_overview(
            &self.pool().await,
            notebook_id,
            &path.to_string_lossy(),
            &generated_at,
            AudioOverviewStatus::Ready,
            &source_set_hash,
        )
        .await?;
        Ok(Some(path))
    }

    /// Persists a `failed` terminal row for a genuine dialogue/synth failure. `path` is
    /// the canonical (possibly absent) overview path — the frontend never plays a failed
    /// row, and read-path reconciliation covers a missing file.
    async fn persist_overview_failed(
        &self,
        notebook_id: &str,
        source_set_hash: &str,
    ) -> Result<(), LensError> {
        let data_dir = self.data_dir().await;
        let path = notebook_audio_path(&data_dir, notebook_id)?;
        let generated_at = chrono::Utc::now().to_rfc3339();
        audio_overview::upsert_overview(
            &self.pool().await,
            notebook_id,
            &path.to_string_lossy(),
            &generated_at,
            AudioOverviewStatus::Failed,
            source_set_hash,
        )
        .await
    }

    /// Reads the persisted overview row and reconciles it against disk: a `ready` row
    /// whose file is absent or zero-length (manual delete / crash) is downgraded to
    /// `Missing` so the UI offers a regenerate instead of a dead player. Returns `None`
    /// when no overview has ever been generated.
    pub async fn get_audio_overview_status(
        &self,
        notebook_id: &str,
    ) -> Result<Option<AudioOverviewRecord>, LensError> {
        let pool = self.pool().await;
        let Some((path, generated_at, status_str, source_set_hash)) =
            audio_overview::read_overview_row(&pool, notebook_id).await?
        else {
            return Ok(None);
        };
        let stored = AudioOverviewStatus::from_db_str(&status_str)?;
        // Reconcile a `ready` row at read time. Missing (file gone) wins over Stale
        // (sources changed): a vanished file is the more urgent regenerate signal. Only
        // a `ready` row is reconciled — a `failed` row stays `failed`.
        let status = if stored == AudioOverviewStatus::Ready {
            if !audio_file_is_nonempty(&path) {
                AudioOverviewStatus::Missing
            } else if self.source_set_hash(notebook_id).await? != source_set_hash {
                AudioOverviewStatus::Stale
            } else {
                AudioOverviewStatus::Ready
            }
        } else {
            stored
        };
        Ok(Some(AudioOverviewRecord {
            path,
            generated_at,
            status,
            source_set_hash,
        }))
    }

    /// Test-only accessor for the order-independent source-set hash (the private
    /// staleness key). Lets integration tests assert order-independence, content
    /// sensitivity, and the `ready`-when-matching read path.
    #[cfg(feature = "test-util")]
    pub async fn source_set_hash_for_test(&self, notebook_id: &str) -> Result<String, LensError> {
        self.source_set_hash(notebook_id).await
    }

    /// Whether an overview synthesis is currently in flight for `notebook_id`, backed by
    /// the TTS cancel-token registry (which covers both the dialogue and synth phases).
    /// A cancelled-but-not-yet-dropped token reads as NOT generating.
    pub fn is_overview_generating(&self, notebook_id: &str) -> bool {
        let map = self
            .tts_cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match map.get(notebook_id) {
            Some(token) => !token.is_cancelled(),
            None => false,
        }
    }

    /// Grounded-answer entry point (#173, the "rag route"). Gathers ALL fallible
    /// context up front — pool, config-derived `ModelConfig`/`RetrievalConfig`/
    /// `TierThresholds`, the LLM provider (errors when none is configured), the
    /// notebook's embedder + coordinate, the tokenizer, and the graph when enabled —
    /// then returns the pure [`answer_stream`]. Must be `-> Result<impl Stream + Send
    /// + 'static, _>` (NOT `async fn -> impl Stream`), so the returned stream owns
    /// only `Send + 'static` values and never captures `&self`.
    pub async fn answer_notebook(
        &self,
        notebook_id: &NotebookId,
        turn_id: &str,
        question: String,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<
        impl futures_util::Stream<Item = Result<AnswerEvent, LensError>> + Send + 'static,
        LensError,
    > {
        let pool = self.pool().await;
        let config = self.config().await;

        // Resolves independently of enrichment (CX-5); see `chat_provider`.
        let provider = self
            .chat_provider()
            .await
            .ok_or_else(|| LensError::Model("no chat model configured".into()))?;

        // Prior-conversation history (CX-1); see `ChatRepo::history` for the
        // exclusion and ordering contract.
        let history = crate::chat::ChatRepo::new(&pool)
            .history(notebook_id.as_str(), turn_id, config.chat.history_turns)
            .await?;

        // Coordinate + query embedder resolve from ONE tuple (plan §10): the same
        // (model_id, backend, dim) feeds both the embedder and the search Coordinate,
        // dissolving the "pick the matching active coordinate" circularity.
        let (model_id, dim, backend) = self.resolve_notebook_embedding(notebook_id).await?;
        let embedder = self
            .embedder_for(
                &model_id,
                backend,
                crate::embedder::WorkloadKind::Interactive,
            )
            .await?;
        let coord =
            crate::vector_store::Coordinate::new(notebook_id.as_str(), backend, model_id, dim);

        let tokenizer = self.tokenizer().await.ok();

        let data_dir = self.data_dir().await;
        let cache_root = self.cache_root().await;
        let reranker = crate::retrieval::Reranker::new(&cache_root);
        let store: Arc<dyn crate::vector_store::VectorStore> = Arc::new(
            crate::vector_store::LanceVectorStore::new(&data_dir, pool.clone()),
        );

        // Honor the per-notebook graph-retrieval override (falling back to the
        // app-wide default), not the raw app flag — this is the live #21-era
        // consumer of that toggle. The resolved value gates the graph load AND the
        // `tiered_search` flag below, so the two never disagree.
        let graph_enabled = self.notebook_graph_retrieval_enabled(notebook_id).await?;
        let graph = if graph_enabled {
            Some(Arc::new(
                crate::graph::NotebookGraph::load(&pool, notebook_id.as_str()).await?,
            ))
        } else {
            None
        };

        // The chat model's context window drives the router's token budget. Match the
        // `config.models` entry the resolved provider was built from (by model id);
        // fall back to the first configured model, else defaults.
        let model = config
            .models
            .iter()
            .find(|m| m.model == provider.model_id())
            .or_else(|| config.models.first())
            .cloned()
            .unwrap_or_default();

        let mut retrieval = config.retrieval;
        retrieval.graph_retrieval_enabled = graph_enabled;

        let ctx = crate::answer::AnswerCtx {
            provider,
            store,
            embedder,
            reranker,
            graph,
            pool,
            coord,
            model,
            retrieval,
            thresholds: config.tier_thresholds,
            tokenizer,
            question,
            history,
            chat: config.chat,
        };

        Ok(crate::answer::answer_stream(ctx, cancel))
    }

    /// Grounded dialogue-script entry point (#26). Gathers the same fallible context
    /// as [`answer_notebook`](Self::answer_notebook) — pool, config-derived configs,
    /// the LLM provider (errors when none is configured), the notebook's embedder +
    /// coordinate, tokenizer, and graph when enabled — plus the FULL selected+live
    /// source-id set (the validator's grounding allow-list), then awaits the pure
    /// [`dialogue::generate_dialogue`]. One-shot: returns the validated script or a
    /// terminal [`LensError`]; phase markers stream via `on_phase`.
    pub async fn generate_dialogue(
        &self,
        notebook_id: &NotebookId,
        length: dialogue::Length,
        cancel: tokio_util::sync::CancellationToken,
        on_phase: impl Fn(dialogue::DialoguePhase) + Send,
    ) -> Result<dialogue::DialogueScript, LensError> {
        let pool = self.pool().await;
        let config = self.config().await;

        // Same CX-5 decoupling as chat; see `chat_provider`.
        let provider = self
            .chat_provider()
            .await
            .ok_or_else(|| LensError::Model("no chat model configured".into()))?;

        let (model_id, dim, backend) = self.resolve_notebook_embedding(notebook_id).await?;
        let embedder = self
            .embedder_for(
                &model_id,
                backend,
                crate::embedder::WorkloadKind::Interactive,
            )
            .await?;
        let coord =
            crate::vector_store::Coordinate::new(notebook_id.as_str(), backend, model_id, dim);

        let tokenizer = self.tokenizer().await.ok();

        let data_dir = self.data_dir().await;
        let cache_root = self.cache_root().await;
        let reranker = crate::retrieval::Reranker::new(&cache_root);
        let store: Arc<dyn crate::vector_store::VectorStore> = Arc::new(
            crate::vector_store::LanceVectorStore::new(&data_dir, pool.clone()),
        );

        let graph_enabled = self.notebook_graph_retrieval_enabled(notebook_id).await?;
        let graph = if graph_enabled {
            Some(Arc::new(
                crate::graph::NotebookGraph::load(&pool, notebook_id.as_str()).await?,
            ))
        } else {
            None
        };

        let model = config
            .models
            .iter()
            .find(|m| m.model == provider.model_id())
            .or_else(|| config.models.first())
            .cloned()
            .unwrap_or_default();

        let mut retrieval = config.retrieval;
        retrieval.graph_retrieval_enabled = graph_enabled;

        // The validator's grounding allow-list is the FULL selected+live set, using
        // the SAME predicate `router::resolve_selected_sources` uses — NOT the
        // retrieval subset — so a model citing a selected-but-not-retrieved source is
        // not wrongly rejected.
        let selected_live_ids = self.selected_live_source_ids(notebook_id.as_str()).await?;

        let ctx = crate::dialogue::DialogueCtx {
            provider,
            store,
            embedder,
            reranker,
            graph,
            pool,
            coord,
            model,
            retrieval,
            thresholds: config.tier_thresholds,
            tokenizer,
            length,
            selected_live_ids,
        };

        crate::dialogue::generate_dialogue(ctx, cancel, on_phase).await
    }

    /// The selected + live (not-trashed) source ids for a notebook, over the shared
    /// `router::SELECTED_LIVE_WHERE` predicate `resolve_selected_sources` also uses.
    /// Backs the dialogue validator's grounding set (#26).
    async fn selected_live_source_ids(
        &self,
        notebook_id: &str,
    ) -> Result<std::collections::HashSet<String>, LensError> {
        let pool = self.pool().await;
        let sql = format!(
            "SELECT id FROM sources WHERE {}",
            crate::retrieval::router::SELECTED_LIVE_WHERE
        );
        let rows = sqlx::query_scalar::<_, String>(&sql)
            .bind(notebook_id)
            .fetch_all(&pool)
            .await?;
        Ok(rows.into_iter().collect())
    }

    /// Transcribes 16 kHz mono f32 PCM (#41 output), selecting the backend via
    /// [`select_asr_backend`](asr::select_asr_backend): an explicit `AsrConfig`
    /// override wins, else the injected Apple engine when present, else LocalWhisper.
    ///
    /// Returns the segments plus a `&'static str` label of the backend actually
    /// used (`"cloud"`, `"apple_native"`, `"local_whisper"`, a `"…(fallback)"`
    /// variant, or — when a low-confidence Apple result is re-transcribed on
    /// Whisper — `"local_whisper (degraded)"` on success, or `"apple_native
    /// (degraded)"` if the re-run fails and the Apple result is kept) so ingest
    /// can surface it to the UI (#45).
    pub async fn transcribe(
        &self,
        pcm: &[f32],
        config: &TranscribeConfig,
        progress_tx: Option<mpsc::UnboundedSender<f32>>,
        cancel: Option<tokio_util::sync::CancellationToken>,
    ) -> Result<(Vec<TranscriptSegment>, &'static str), LensError> {
        let asr_cfg = {
            let guard = self.read().await;
            guard.config.asr.clone()
        };
        let config_backend = asr::AsrBackend::from_opt_str(Some(asr_cfg.backend.as_str()));

        let injected = self.asr_engine().await;
        let apple_available = injected.is_some();

        // lens-core stays OS-probe-free: authoritative platform/version facts are
        // enforced in src-tauri BEFORE injecting the engine, so a present engine
        // implies a capable Apple platform. Derive Platform consistently with that
        // seam rather than probing the OS here (no OS-probe crate in lens-core).
        let platform = asr::Platform {
            is_apple_silicon_macos: apple_available,
            macos_major: apple_available.then_some(asr::MIN_MACOS_FOR_APPLE_ASR),
        };
        // Probe the injected Apple engine for locale support, only for an explicit
        // language, off the async runtime (the probe crosses blocking FFI in prod);
        // auto-detect or no engine trusts Apple, with the downstream confidence
        // check as the runtime backstop.
        let apple_supports_locale = match (&config.language, &injected) {
            (Some(lang), Some(engine)) => {
                let (engine, lang) = (engine.clone(), lang.clone());
                tokio::task::spawn_blocking(move || engine.supports_locale(&lang))
                    .await
                    .map_err(|e| {
                        LensError::Transcription(format!("locale probe task failed: {e}"))
                    })?
            }
            _ => true,
        };

        let mut backend = asr::select_asr_backend(
            config_backend,
            platform,
            apple_available,
            apple_supports_locale,
        );

        // SpeechTranscriber has no translate task (translation is Whisper-only).
        // If the caller requests translation and the router picked Apple, fall back
        // to LocalWhisper so the translate request can be fulfilled.
        if config.translate && backend == asr::AsrBackend::AppleNative {
            backend = asr::AsrBackend::LocalWhisper;
        }

        // The injected engine is the Apple-native seam (Apple in prod, a mock in
        // tests); it is used ONLY when the router selects AppleNative. LocalWhisper
        // always uses the internal WhisperEngine, never the injected engine.
        match (backend, injected) {
            (asr::AsrBackend::AppleNative, Some(engine)) => {
                match engine.transcribe_pcm(pcm, config, progress_tx).await {
                    // Apple succeeded but the aggregate confidence is below the
                    // configured floor: re-transcribe the whole clip on Whisper when
                    // a local model is available, else keep the low-confidence result.
                    Ok(TranscriptOutput {
                        segments,
                        confidence,
                    }) => {
                        // `confidence` is the worst-span MIN for the clip (see the
                        // `apple_min_confidence` field doc); clamp the floor so a
                        // corrupt config value can neither force-degrade nor disable.
                        let floor = asr_cfg.apple_min_confidence.clamp(0.0, 1.0);
                        let degraded = confidence.is_some_and(|c| c < floor);
                        if degraded && self.local_whisper_available(&asr_cfg).await {
                            match self
                                .transcribe_local_whisper(pcm, config, None, &asr_cfg, cancel)
                                .await
                            {
                                Ok(segs) => Ok((segs, "local_whisper (degraded)")),
                                Err(e) => {
                                    tracing::warn!(error = %e, "degraded whisper re-run failed; keeping low-confidence Apple result");
                                    Ok((segments, "apple_native (degraded)"))
                                }
                            }
                        } else {
                            Ok((segments, "apple_native"))
                        }
                    }
                    // Apple runtime failure (e.g. a missing on-device asset) must not
                    // leave the user with no transcription — fall back to Whisper when
                    // a local model is available. Skip the fallback on a genuine
                    // user-cancel or when whisper is unavailable, returning the
                    // original Apple error in those cases.
                    Err(apple_err) => {
                        if !is_user_cancel(&apple_err)
                            && self.local_whisper_available(&asr_cfg).await
                        {
                            let segs = self
                                .transcribe_local_whisper(pcm, config, None, &asr_cfg, cancel)
                                .await?;
                            Ok((segs, "local_whisper (fallback)"))
                        } else {
                            Err(apple_err)
                        }
                    }
                }
            }
            (asr::AsrBackend::LocalWhisper, Some(_)) | (asr::AsrBackend::LocalWhisper, None) => {
                let segs = self
                    .transcribe_local_whisper(pcm, config, progress_tx, &asr_cfg, cancel)
                    .await?;
                Ok((segs, "local_whisper"))
            }
            (asr::AsrBackend::AppleNative, None) => Err(LensError::Transcription(
                "apple-native backend selected but no engine is injected".into(),
            )),
            // Cloud (#45): pre-flight (consent/key/provider) + request; any failure
            // that is not a user-cancel transparently degrades to the local cascade
            // (Apple-if-injected → Whisper), mirroring the Apple→Whisper symmetry.
            (asr::AsrBackend::Cloud, injected_local) => {
                let app_config = self.read().await.config.clone();
                match self
                    .cloud_transcribe(pcm, config, &app_config, progress_tx.clone())
                    .await
                {
                    Ok(segments) => Ok((segments, "cloud")),
                    Err(cloud_err) => {
                        if is_user_cancel(&cloud_err) {
                            return Err(cloud_err);
                        }
                        tracing::warn!(error = %cloud_err, "cloud ASR failed, falling back to local");
                        self.cloud_fallback_to_local(
                            pcm,
                            config,
                            &asr_cfg,
                            injected_local,
                            progress_tx,
                            cloud_err,
                            cancel,
                        )
                        .await
                    }
                }
            }
        }
    }

    /// Runs the cloud pre-flight then a [`CloudAsrEngine`] transcription. Pre-flight
    /// failure returns before any request, so the fallback path issues zero cloud
    /// requests when consent/key/provider are missing (#45).
    async fn cloud_transcribe(
        &self,
        pcm: &[f32],
        config: &TranscribeConfig,
        app_config: &config::AppConfig,
        progress_tx: Option<mpsc::UnboundedSender<f32>>,
    ) -> Result<Vec<TranscriptSegment>, LensError> {
        asr::cloud::preflight_check(app_config)?;
        let asr_cfg = &app_config.asr;
        let provider = asr_cfg
            .cloud_provider
            .ok_or_else(|| LensError::Validation("no cloud ASR provider configured".into()))?;
        let engine = asr::cloud::CloudAsrEngine::new(
            provider,
            asr_cfg.cloud_base_url.clone(),
            asr_cfg.cloud_model.clone(),
            asr_cfg.cloud_api_key.clone(),
        );
        let TranscriptOutput { segments, .. } =
            engine.transcribe_pcm(pcm, config, progress_tx).await?;
        Ok(segments)
    }

    /// Cloud→local degradation cascade: Apple-if-injected, else Whisper. Returns
    /// the original `cloud_err` only when no local path can produce segments.
    #[allow(clippy::too_many_arguments)]
    async fn cloud_fallback_to_local(
        &self,
        pcm: &[f32],
        config: &TranscribeConfig,
        asr_cfg: &config::AsrConfig,
        injected: Option<Arc<dyn asr::AsrEngine>>,
        progress_tx: Option<mpsc::UnboundedSender<f32>>,
        cloud_err: LensError,
        cancel: Option<tokio_util::sync::CancellationToken>,
    ) -> Result<(Vec<TranscriptSegment>, &'static str), LensError> {
        if let Some(engine) = injected {
            match engine.transcribe_pcm(pcm, config, progress_tx).await {
                Ok(TranscriptOutput { segments: segs, .. }) => {
                    Ok((segs, "apple_native (fallback)"))
                }
                Err(apple_err) => {
                    if !is_user_cancel(&apple_err) && self.local_whisper_available(asr_cfg).await {
                        let segs = self
                            .transcribe_local_whisper(pcm, config, None, asr_cfg, cancel)
                            .await?;
                        Ok((segs, "local_whisper (fallback)"))
                    } else {
                        Err(apple_err)
                    }
                }
            }
        } else if self.local_whisper_available(asr_cfg).await {
            let segs = self
                .transcribe_local_whisper(pcm, config, progress_tx, asr_cfg, cancel)
                .await?;
            Ok((segs, "local_whisper (fallback)"))
        } else {
            Err(cloud_err)
        }
    }

    /// Whether a local Whisper model is present on disk for the configured id, so
    /// the Apple→Whisper fallback has something to route to. Behind `local-whisper`;
    /// feature-off is always `false` (no Whisper path compiled in).
    #[cfg(feature = "local-whisper")]
    async fn local_whisper_available(&self, asr_cfg: &config::AsrConfig) -> bool {
        let Some(spec) = asr::resolve_whisper(&asr_cfg.whisper_model)
            .or_else(|| asr::resolve_whisper(DEFAULT_WHISPER_MODEL_ID))
        else {
            return false;
        };
        let cache_root = self.cache_root().await;
        asr::whisper_model_path(&cache_root, spec.id).is_file()
    }

    #[cfg(not(feature = "local-whisper"))]
    async fn local_whisper_available(&self, _asr_cfg: &config::AsrConfig) -> bool {
        false
    }

    /// Transcribes via the internal LocalWhisper engine. Resolves the model id
    /// from [`AsrConfig`] (fallback [`DEFAULT_WHISPER_MODEL_ID`]); the model must
    /// already be downloaded (transcribe never auto-downloads — that is the
    /// onboarding command's job). Behind `local-whisper`; feature-off returns a
    /// typed error so lens-core still compiles without whisper.cpp.
    #[cfg(feature = "local-whisper")]
    async fn transcribe_local_whisper(
        &self,
        pcm: &[f32],
        config: &TranscribeConfig,
        progress_tx: Option<mpsc::UnboundedSender<f32>>,
        asr_cfg: &config::AsrConfig,
        cancel: Option<tokio_util::sync::CancellationToken>,
    ) -> Result<Vec<TranscriptSegment>, LensError> {
        let engine = self.whisper_engine_for(asr_cfg).await?;
        engine
            .transcribe_pcm_cancellable(pcm, config, progress_tx, cancel)
            .await
    }

    #[cfg(not(feature = "local-whisper"))]
    async fn transcribe_local_whisper(
        &self,
        _pcm: &[f32],
        _config: &TranscribeConfig,
        _progress_tx: Option<mpsc::UnboundedSender<f32>>,
        _asr_cfg: &config::AsrConfig,
        _cancel: Option<tokio_util::sync::CancellationToken>,
    ) -> Result<Vec<TranscriptSegment>, LensError> {
        Err(LensError::Transcription(
            "local whisper feature not built (build lens-core with the `local-whisper` feature)"
                .into(),
        ))
    }

    /// Lazily builds and caches the [`asr::WhisperEngine`] for the configured
    /// model id (R8-style cache keyed by model id). Errors with a clear typed
    /// [`LensError::Transcription`] when the model is not yet downloaded.
    #[cfg(feature = "local-whisper")]
    async fn whisper_engine_for(
        &self,
        asr_cfg: &config::AsrConfig,
    ) -> Result<Arc<asr::WhisperEngine>, LensError> {
        // Validate the config-supplied id through the registry allowlist BEFORE it
        // touches the filesystem: build the path from the resolved `spec.id` (a
        // &'static allowlisted token), never the raw config string, so a crafted id
        // (e.g. `../..`) cannot escape `models/whisper/`. Unknown/empty → default.
        let spec = asr::resolve_whisper(&asr_cfg.whisper_model)
            .or_else(|| asr::resolve_whisper(DEFAULT_WHISPER_MODEL_ID))
            .ok_or_else(|| LensError::Validation("no resolvable whisper model id".to_string()))?;
        let model_id = spec.id.to_string();
        let cache_root = self.cache_root().await;
        let model_path = asr::whisper_model_path(&cache_root, spec.id);

        // Intentionally hold this `tokio::sync::Mutex` across the spawn_blocking load
        // so init runs exactly once per model id (mirrors the embedder cache). An
        // async Mutex is safe to hold across `.await` — no clippy await_holding_lock.
        let mut cache = self.whisper_engines.lock().await;
        if let Some(existing) = cache.get(&model_id) {
            return Ok(Arc::clone(existing));
        }
        if !model_path.is_file() {
            return Err(LensError::Transcription(format!(
                "whisper model {model_id:?} is not downloaded; \
                 fetch it via the onboarding download step first"
            )));
        }
        // Model load is CPU-blocking (mmaps + parses the ggml weights).
        let engine = tokio::task::spawn_blocking(move || asr::WhisperEngine::load(&model_path))
            .await
            .map_err(|e| {
                LensError::Transcription(format!("whisper model load task failed: {e}"))
            })??;
        let engine = Arc::new(engine);
        cache.insert(model_id, Arc::clone(&engine));
        Ok(engine)
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
        let cache_root = self.cache_root().await;
        let loaded =
            tokio::task::spawn_blocking(move || crate::model_catalog::load_catalog(&cache_root))
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
        let cache_root = self.cache_root().await;
        let client = crate::model_catalog::catalog_client();
        let refreshed = crate::model_catalog::refresh_if_stale(
            &cache_root,
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
        if !table_names.is_empty() {
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
        }
        // ent__ tables carry no registry, so their GC runs regardless of the
        // building/stale scan above (a crashed purge_notebook orphans them alone).
        Self::gc_orphan_entity_tables(db, data_dir).await
    }

    /// Startup-GC for `ent__` entity-vector tables (#155): drops Lance dirs for
    /// notebooks that no longer exist in the `notebooks` table.
    async fn gc_orphan_entity_tables(db: &SqlitePool, data_dir: &Path) -> Result<(), LensError> {
        let store = crate::vector_store::LanceVectorStore::new(data_dir, db.clone());
        let tables = store.entity_tables_with_notebook().await?;
        if tables.is_empty() {
            return Ok(());
        }
        let mut orphans: Vec<String> = Vec::new();
        for (table, notebook_id) in tables {
            let live: Option<i64> = sqlx::query_scalar("SELECT 1 FROM notebooks WHERE id = ?")
                .bind(&notebook_id)
                .fetch_optional(db)
                .await?;
            if live.is_none() {
                orphans.push(table);
            }
        }
        if orphans.is_empty() {
            return Ok(());
        }
        crate::vector_store::VectorStore::drop_tables(&store, &orphans).await?;
        tracing::info!(
            count = orphans.len(),
            "startup-GC reclaimed orphan entity-vector tables"
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

    /// Enqueues a `ResolveNotebook` for testing; the production trigger is in the enrichment worker.
    #[cfg(feature = "test-util")]
    pub fn enqueue_resolution_for_test(&self, notebook_id: &str) {
        let _ = self
            .resolution_tx
            .try_send(crate::resolution::ResolveNotebook {
                notebook_id: notebook_id.to_string(),
            });
    }

    /// Runs one resolution pass synchronously, bypassing the debounced worker.
    #[cfg(feature = "test-util")]
    pub async fn resolve_notebook_for_test(&self, notebook_id: &str) -> Result<(), LensError> {
        crate::resolution::worker::resolve_one(self, notebook_id).await
    }

    /// Exposes `notebook_lock` (pub(crate)) to the test crate; lets tests simulate the enrichment
    /// writer holding the lock to assert the resolution pass serializes behind it.
    #[cfg(feature = "test-util")]
    pub fn notebook_lock_for_test(&self, notebook_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        self.notebook_lock(notebook_id)
    }

    /// Count of resolution passes completed (drain-coalesce assertion).
    #[cfg(feature = "test-util")]
    pub fn resolution_pass_count_for_test(&self) -> u32 {
        self.resolution_pass_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(feature = "test-util")]
    pub(crate) fn note_resolution_pass(&self) {
        self.resolution_pass_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Arms/disarms the write-fault in `write_resolution_updates` (single-txn atomicity seam).
    #[cfg(feature = "test-util")]
    pub fn set_resolution_write_fault_for_test(&self, armed: bool) {
        self.resolution_write_fault
            .store(armed, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(feature = "test-util")]
    pub(crate) fn resolution_write_fault_armed(&self) -> bool {
        self.resolution_write_fault
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(not(feature = "test-util"))]
    pub(crate) fn resolution_write_fault_armed(&self) -> bool {
        false
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
                let cache_root = self.cache_root().await;
                // Try candle (Metal/CUDA) first; fall back to fastembed on any failure
                // or unsupported model. Never fail a job over a device choice.
                let gpu = if compute == crate::embedder::Compute::Cuda {
                    build_cuda_if_supported(compute, &cache_root, spec).await
                } else {
                    build_candle_if_supported(compute, &cache_root, spec).await
                };
                match gpu {
                    Some(e) => e,
                    None => {
                        let e = tokio::task::spawn_blocking(move || {
                            FastembedEmbedder::new_with_spec(&cache_root, spec)
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

    /// Sets (or, with `None`, clears) the per-notebook graph-retrieval override
    /// (#158a). `None` reverts the notebook to inheriting the app-wide
    /// `RetrievalConfig::graph_retrieval_enabled`. Does NOT enable/disable any live
    /// retrieval path — that wiring is #21; this only persists the user's choice.
    pub async fn set_notebook_graph_retrieval_enabled(
        &self,
        notebook_id: &NotebookId,
        enabled: Option<bool>,
    ) -> Result<(), LensError> {
        let pool = self.pool().await;
        let result = sqlx::query(
            "UPDATE notebooks SET graph_retrieval_enabled = ?, updated_at = ? \
             WHERE id = ? AND trashed_at IS NULL",
        )
        .bind(enabled)
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

    /// Effective graph-retrieval opt-in for a notebook: the per-notebook override
    /// if set, else the app-wide `RetrievalConfig` default (#158a). Read by the
    /// eval to record the `graph_enabled` snapshot; no live path consumes it yet.
    pub async fn notebook_graph_retrieval_enabled(
        &self,
        notebook_id: &NotebookId,
    ) -> Result<bool, LensError> {
        let pool = self.pool().await;
        let row: Option<(Option<bool>,)> =
            sqlx::query_as("SELECT graph_retrieval_enabled FROM notebooks WHERE id = ?")
                .bind(notebook_id.as_str())
                .fetch_optional(&pool)
                .await?;
        let override_val = row
            .ok_or_else(|| LensError::Validation(format!("no notebook with id {notebook_id}")))?
            .0;
        match override_val {
            Some(v) => Ok(v),
            None => Ok(self.config().await.retrieval.graph_retrieval_enabled),
        }
    }

    /// Returns the most recent `notebook_eval_log` row for a notebook (#158b), or
    /// `None` if the eval has never run. Read-only observational data for the
    /// Retrieval settings verdict; never touches the flag.
    pub async fn latest_notebook_eval(
        &self,
        notebook_id: &NotebookId,
    ) -> Result<Option<crate::eval::LatestEval>, LensError> {
        use sqlx::Row;
        let pool = self.pool().await;
        let row = sqlx::query(
            "SELECT graph_recall, hybrid_recall, delta_pp, p95_ms, passed, sample_n, dropped_n, \
             graph_enabled, prompt_version, ran_at \
             FROM notebook_eval_log WHERE notebook_id = ? ORDER BY ran_at DESC LIMIT 1",
        )
        .bind(notebook_id.as_str())
        .fetch_optional(&pool)
        .await?;
        // booleans stored as INTEGER, recall/delta/p95 as REAL (see `persist_log`).
        Ok(row.map(|r| crate::eval::LatestEval {
            report: crate::eval::EvalReport {
                graph_recall: r.get::<f64, _>("graph_recall") as f32,
                hybrid_recall: r.get::<f64, _>("hybrid_recall") as f32,
                delta_pp: r.get::<f64, _>("delta_pp") as f32,
                p95_ms: r.get::<f64, _>("p95_ms") as f32,
                passed: r.get::<i64, _>("passed") != 0,
                sample_n: r.get::<i64, _>("sample_n") as usize,
                dropped_n: r.get::<i64, _>("dropped_n") as usize,
                graph_enabled: r.get::<i64, _>("graph_enabled") != 0,
                prompt_version: r.get::<String, _>("prompt_version"),
            },
            ran_at: r.get::<String, _>("ran_at"),
        }))
    }

    /// Runs the per-notebook graph-retrieval eval on demand (#158b), reusing the
    /// configured chat provider (`llm_provider`). Pre-checks the provider is present
    /// AND reachable — returning a typed `LensError::Model` the UI branches on — then
    /// assembles `RunEvalDeps` from engine state and delegates to `run_notebook_eval`.
    /// `on_progress` fires only around the run (`GeneratingQa` before, `Done` after);
    /// it is NOT threaded inside `run_notebook_eval`. Advisory only: never mutates the
    /// flag; below the sample floor the outcome is `Skipped { reason }`.
    pub async fn run_graph_eval(
        &self,
        notebook_id: &NotebookId,
        mut on_progress: impl FnMut(crate::eval::EvalPhase) + Send,
    ) -> Result<crate::eval::EvalOutcome, LensError> {
        let provider = self
            .llm_provider()
            .await
            .ok_or_else(|| LensError::Model("no chat model configured".into()))?;
        if !provider.reachable().await {
            return Err(LensError::Model("chat model unreachable".into()));
        }

        on_progress(crate::eval::EvalPhase::GeneratingQa);

        // `RunEvalDeps` holds borrows, so build owned locals and lend `&`/`.as_ref()`
        // references that live across the `run_notebook_eval(...).await`.
        let pool = self.pool().await;
        let config = self.config().await;
        let (model_id, dim, backend) = self.resolve_notebook_embedding(notebook_id).await?;
        let embedder = self
            .embedder_for(
                &model_id,
                backend,
                crate::embedder::WorkloadKind::Interactive,
            )
            .await?;
        let coord =
            crate::vector_store::Coordinate::new(notebook_id.as_str(), backend, model_id, dim);
        let data_dir = self.data_dir().await;
        let cache_root = self.cache_root().await;
        let reranker = crate::retrieval::Reranker::new(&cache_root);
        let store: Arc<dyn crate::vector_store::VectorStore> = Arc::new(
            crate::vector_store::LanceVectorStore::new(&data_dir, pool.clone()),
        );
        let graph_enabled = self.notebook_graph_retrieval_enabled(notebook_id).await?;

        let deps = crate::eval::RunEvalDeps {
            pool: &pool,
            store: store.as_ref(),
            reranker: &reranker,
            coord: &coord,
            embedder: embedder.as_ref(),
            llm: provider.as_ref(),
            config: &config.retrieval,
            graph_enabled,
        };
        let outcome = crate::eval::run_notebook_eval(&deps, notebook_id.as_str()).await?;
        on_progress(crate::eval::EvalPhase::Done);
        Ok(outcome)
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
                let cache_root = self.cache_root().await;
                let tokenizer = resolve_nomic_tokenizer(&cache_root).await?;
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
        // #155: entity-vector drop (same ordering as the chunk-vector drop above).
        store.drop_entity_tables_for_notebook(id.as_str()).await?;
        NotebookRepo::new(&pool).purge(id).await?;
        for (source_id, locator) in &sources {
            remove_managed_source_file(&data_dir, source_id, locator);
        }
        // #155: drop the per-notebook write lock so the map does not grow unbounded.
        self.notebook_locks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(id.as_str());
        Ok(())
    }
}

/// RAII guard that clears a grounded-answer (#173) cancellation-registry entry on
/// drop via [`remove_ask_if_owner`](LensEngine::remove_ask_if_owner). See the
/// `ask_cancel_tokens` field doc for the ABA rationale this guard exists to satisfy.
pub struct AskCancelGuard {
    engine: LensEngine,
    notebook_id: String,
    owner: Arc<tokio_util::sync::CancellationToken>,
}

impl Drop for AskCancelGuard {
    fn drop(&mut self) {
        self.engine
            .remove_ask_if_owner(&self.notebook_id, &self.owner);
    }
}

/// RAII guard that clears a dialogue-script (#26) cancellation-registry entry on
/// drop via [`remove_dialogue_if_owner`](LensEngine::remove_dialogue_if_owner). A
/// dedicated guard over the dialogue registry, parallel to [`AskCancelGuard`].
pub struct DialogueCancelGuard {
    engine: LensEngine,
    notebook_id: String,
    owner: Arc<tokio_util::sync::CancellationToken>,
}

impl Drop for DialogueCancelGuard {
    fn drop(&mut self) {
        self.engine
            .remove_dialogue_if_owner(&self.notebook_id, &self.owner);
    }
}

pub struct TtsCancelGuard {
    engine: LensEngine,
    notebook_id: String,
    owner: Arc<tokio_util::sync::CancellationToken>,
}

impl Drop for TtsCancelGuard {
    fn drop(&mut self) {
        self.engine
            .remove_tts_if_owner(&self.notebook_id, &self.owner);
    }
}

pub fn notebook_audio_path(
    data_dir: &Path,
    notebook_id: &str,
) -> Result<std::path::PathBuf, LensError> {
    let looks_like_traversal = notebook_id.is_empty()
        || notebook_id.contains('/')
        || notebook_id.contains('\\')
        || notebook_id.contains("..")
        || Path::new(notebook_id).is_absolute();
    if looks_like_traversal {
        return Err(LensError::Validation(format!(
            "invalid notebook id: {notebook_id:?}"
        )));
    }
    Ok(data_dir
        .join("notebooks")
        .join(notebook_id)
        .join("overview.wav"))
}

/// True when `path` is a regular file with non-zero length. Backs the read-path
/// reconciliation that downgrades a `ready` overview whose file vanished to `Missing`.
fn audio_file_is_nonempty(path: &str) -> bool {
    std::fs::metadata(path)
        .map(|m| m.is_file() && m.len() > 0)
        .unwrap_or(false)
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

/// Whether an Apple transcription error represents a genuine user-cancel (which
/// must NOT trigger the Whisper fallback — the user chose to stop). Apple errors
/// cross the bridge as [`LensError::Transcription`] strings, so this matches on
/// the cancel wording rather than a distinct variant.
fn is_user_cancel(err: &LensError) -> bool {
    match err {
        // The cloud/chunking path emits a typed cancel (#45); the Apple bridge
        // crosses as a Transcription string containing "cancel".
        LensError::Cancelled(_) => true,
        LensError::Transcription(msg) => msg.to_ascii_lowercase().contains("cancel"),
        _ => false,
    }
}

#[cfg(test)]
mod transcribe_tests {
    use super::*;

    /// A mock Apple engine whose `transcribe_pcm` always errors — used to prove the
    /// Apple→Whisper fallback path runs (the router picks AppleNative, the error
    /// triggers the fallback, and the whisper branch surfaces its own typed error).
    struct ErroringAppleEngine {
        message: String,
    }

    #[async_trait::async_trait]
    impl asr::AsrEngine for ErroringAppleEngine {
        async fn transcribe_pcm(
            &self,
            _pcm: &[f32],
            _config: &TranscribeConfig,
            _progress_tx: Option<mpsc::UnboundedSender<f32>>,
        ) -> Result<TranscriptOutput, LensError> {
            Err(LensError::Transcription(self.message.clone()))
        }
    }

    /// Points the engine's `data_dir` at `dir` and drops a stub whisper model file
    /// there so [`local_whisper_available`](LensEngine::local_whisper_available)
    /// reports the fallback is possible (the stub is not a valid ggml file, so the
    /// subsequent load fails — which is exactly what proves the whisper branch ran).
    #[cfg(feature = "local-whisper")]
    async fn seed_stub_whisper_model(engine: &LensEngine, dir: &std::path::Path) {
        engine.inner.write().await.config.paths.data_dir = dir.to_string_lossy().into_owned();
        let spec = asr::resolve_whisper(DEFAULT_WHISPER_MODEL_ID).expect("default whisper spec");
        let model_path = asr::whisper_model_path(dir, spec.id);
        std::fs::create_dir_all(model_path.parent().expect("model parent"))
            .expect("create model dir");
        std::fs::write(&model_path, b"not a real ggml model").expect("write stub model");
    }

    /// A runtime Apple failure (missing on-device asset) must fall back to Whisper
    /// when a local model is present. A stub model is seeded so the fallback path
    /// runs; the stub is not a valid ggml file, so the whisper load surfaces its own
    /// typed error — proving the whisper branch ran rather than the Apple error
    /// propagating unchanged.
    #[cfg(feature = "local-whisper")]
    #[tokio::test]
    async fn apple_runtime_error_falls_back_to_whisper() {
        let engine = LensEngine::for_test().await;
        let dir = tempfile::tempdir().expect("tempdir");
        seed_stub_whisper_model(&engine, dir.path()).await;
        engine
            .set_asr_engine(Some(Arc::new(ErroringAppleEngine {
                message: "on-device speech model for locale en-US is not installed".into(),
            })))
            .await;

        let config = TranscribeConfig {
            language: None,
            translate: false,
        };
        let pcm = vec![0.0_f32; 16];
        let result = engine.transcribe(&pcm, &config, None, None).await;

        // The whisper branch ran: the stub is not a valid model, so we get a
        // whisper-path error, NOT the original "not installed" Apple message.
        match result {
            Err(LensError::Transcription(msg)) => {
                assert!(
                    !msg.contains("not installed"),
                    "fallback did not run: got the Apple error unchanged: {msg}"
                );
            }
            other => panic!("expected a whisper-path Transcription error, got: {other:?}"),
        }
    }

    /// When whisper is NOT available, an Apple runtime error propagates unchanged
    /// (no fallback target exists). The data_dir points at an empty tempdir so the
    /// probe deterministically finds no model regardless of the test cwd.
    #[tokio::test]
    async fn apple_runtime_error_without_whisper_returns_apple_error() {
        let engine = LensEngine::for_test().await;
        let dir = tempfile::tempdir().expect("tempdir");
        engine.inner.write().await.config.paths.data_dir =
            dir.path().to_string_lossy().into_owned();
        engine
            .set_asr_engine(Some(Arc::new(ErroringAppleEngine {
                message: "on-device speech model for locale en-US is not installed".into(),
            })))
            .await;

        let config = TranscribeConfig {
            language: None,
            translate: false,
        };
        let pcm = vec![0.0_f32; 16];
        let result = engine.transcribe(&pcm, &config, None, None).await;

        match result {
            Err(LensError::Transcription(msg)) => {
                assert!(
                    msg.contains("not installed"),
                    "expected the original Apple error, got: {msg}"
                );
            }
            other => panic!("expected the original Apple error, got: {other:?}"),
        }
    }

    /// A genuine user-cancel must NOT trigger the whisper fallback: the original
    /// Apple cancel error propagates unchanged.
    #[tokio::test]
    async fn apple_user_cancel_does_not_fall_back() {
        let engine = LensEngine::for_test().await;
        engine
            .set_asr_engine(Some(Arc::new(ErroringAppleEngine {
                message: "transcription cancelled by user".into(),
            })))
            .await;

        let config = TranscribeConfig {
            language: None,
            translate: false,
        };
        let pcm = vec![0.0_f32; 16];
        let result = engine.transcribe(&pcm, &config, None, None).await;

        match result {
            Err(LensError::Transcription(msg)) => {
                assert!(
                    msg.contains("cancelled"),
                    "expected the original Apple cancel error, got: {msg}"
                );
            }
            other => panic!("expected the original Apple cancel error, got: {other:?}"),
        }
    }

    /// When translate=true the Apple engine cannot fulfil the request. The router
    /// normally picks Apple (engine injected + capable platform assumed), but the
    /// translate guard must redirect to LocalWhisper, which surfaces as a typed
    /// Transcription error (model not downloaded) rather than an Apple dispatch.
    #[tokio::test]
    async fn translate_forces_whisper_over_apple() {
        let engine = LensEngine::for_test().await;

        // Inject a mock Apple engine so the router would otherwise pick AppleNative.
        let canned = vec![asr::TranscriptSegment {
            text: "apple".to_string(),
            start_second: 0.0,
            end_second: 1.0,
        }];
        engine
            .set_asr_engine(Some(Arc::new(asr::MockAsrEngine::new(canned))))
            .await;

        // Request translation. The guard must redirect to LocalWhisper.
        let config = TranscribeConfig {
            language: None,
            translate: true,
        };
        let pcm = vec![0.0_f32; 16];
        let result = engine.transcribe(&pcm, &config, None, None).await;

        // LocalWhisper is routed but the model is not downloaded, so we get a
        // typed Transcription error — confirming the Apple mock was NOT called.
        match result {
            Err(LensError::Transcription(msg)) => {
                assert!(
                    msg.contains("local whisper")
                        || msg.contains("not downloaded")
                        || msg.contains("local-whisper"),
                    "expected a LocalWhisper error, got: {msg}"
                );
            }
            other => panic!("expected Transcription error from LocalWhisper path, got: {other:?}"),
        }
    }

    fn canned_apple_segment() -> Vec<TranscriptSegment> {
        vec![asr::TranscriptSegment {
            text: "apple".to_string(),
            start_second: 0.0,
            end_second: 1.0,
        }]
    }

    /// An explicit language the Apple engine does NOT support routes to
    /// LocalWhisper (router gate 4). The Apple mock would return Ok segments, so an
    /// error from the (undownloaded) whisper path proves Apple was not used.
    #[tokio::test]
    async fn explicit_unsupported_locale_routes_to_whisper() {
        let engine = LensEngine::for_test().await;
        engine
            .set_asr_engine(Some(Arc::new(
                asr::MockAsrEngine::new(canned_apple_segment()).with_locale_support(false),
            )))
            .await;

        let config = TranscribeConfig {
            language: Some(asr::Lang::De),
            translate: false,
        };
        let pcm = vec![0.0_f32; 16];
        let result = engine.transcribe(&pcm, &config, None, None).await;

        match result {
            Err(LensError::Transcription(msg)) => assert!(
                msg.contains("local whisper")
                    || msg.contains("not downloaded")
                    || msg.contains("local-whisper"),
                "expected a LocalWhisper error, got: {msg}"
            ),
            other => panic!("expected LocalWhisper Transcription error, got: {other:?}"),
        }
    }

    /// An explicit language the Apple engine supports routes to AppleNative.
    #[tokio::test]
    async fn explicit_supported_locale_uses_apple() {
        let engine = LensEngine::for_test().await;
        let canned = canned_apple_segment();
        engine
            .set_asr_engine(Some(Arc::new(
                asr::MockAsrEngine::new(canned.clone()).with_locale_support(true),
            )))
            .await;

        let config = TranscribeConfig {
            language: Some(asr::Lang::En),
            translate: false,
        };
        let pcm = vec![0.0_f32; 16];
        let (segs, label) = engine
            .transcribe(&pcm, &config, None, None)
            .await
            .expect("apple transcription");
        assert_eq!(label, "apple_native");
        assert_eq!(segs, canned);
    }

    /// Auto-detect (language == None) trusts Apple and skips the locale probe
    /// entirely — even a mock reporting no locale support still routes to Apple.
    #[tokio::test]
    async fn auto_detect_uses_apple_without_probing_locale() {
        let engine = LensEngine::for_test().await;
        let canned = canned_apple_segment();
        engine
            .set_asr_engine(Some(Arc::new(
                asr::MockAsrEngine::new(canned.clone()).with_locale_support(false),
            )))
            .await;

        let config = TranscribeConfig {
            language: None,
            translate: false,
        };
        let pcm = vec![0.0_f32; 16];
        let (segs, label) = engine
            .transcribe(&pcm, &config, None, None)
            .await
            .expect("apple transcription");
        assert_eq!(label, "apple_native");
        assert_eq!(segs, canned);
    }

    /// A low-confidence Apple result triggers a degraded Whisper re-run; a stub
    /// (unloadable) model is seeded so the re-run is ATTEMPTED but fails, and the
    /// fail-safe keeps the Apple segments labelled `"apple_native (degraded)"`.
    /// (A successful `"local_whisper (degraded)"` path needs the non-injectable
    /// internal WhisperEngine and a real model — left to the gated model track.)
    #[cfg(feature = "local-whisper")]
    #[tokio::test]
    async fn low_confidence_apple_result_is_re_run_on_whisper() {
        let engine = LensEngine::for_test().await;
        let dir = tempfile::tempdir().expect("tempdir");
        seed_stub_whisper_model(&engine, dir.path()).await;
        let canned = canned_apple_segment();
        engine
            .set_asr_engine(Some(Arc::new(
                asr::MockAsrEngine::new(canned.clone()).with_confidence(0.1),
            )))
            .await;

        let config = TranscribeConfig {
            language: None,
            translate: false,
        };
        let pcm = vec![0.0_f32; 16];
        let (segs, label) = engine
            .transcribe(&pcm, &config, None, None)
            .await
            .expect("degraded re-run fails safe and keeps the Apple result");

        // The seeded stub is not a valid ggml model, so the re-run errors and the
        // fail-safe keeps the Apple segments — proving the re-run was attempted.
        assert_eq!(label, "apple_native (degraded)");
        assert_eq!(
            segs, canned,
            "the failed re-run must keep the Apple segments"
        );
    }

    /// Confidence exactly at the threshold (0.5) is NOT degraded — the gate is a
    /// strict `<` — so the result is kept as `"apple_native"`.
    #[tokio::test]
    async fn at_threshold_confidence_apple_result_is_kept() {
        let engine = LensEngine::for_test().await;
        let canned = canned_apple_segment();
        engine
            .set_asr_engine(Some(Arc::new(
                asr::MockAsrEngine::new(canned.clone()).with_confidence(0.5),
            )))
            .await;

        let config = TranscribeConfig {
            language: None,
            translate: false,
        };
        let pcm = vec![0.0_f32; 16];
        let (segs, label) = engine
            .transcribe(&pcm, &config, None, None)
            .await
            .expect("apple transcription");
        assert_eq!(label, "apple_native");
        assert_eq!(segs, canned);
    }

    /// A low-confidence Apple result with NO local Whisper model available keeps
    /// the Apple result (the else branch): degraded, but nothing to re-run on. The
    /// data_dir points at an empty tempdir so the probe deterministically finds
    /// no model regardless of the test cwd.
    #[tokio::test]
    async fn low_confidence_without_whisper_keeps_apple() {
        let engine = LensEngine::for_test().await;
        let dir = tempfile::tempdir().expect("tempdir");
        engine.inner.write().await.config.paths.data_dir =
            dir.path().to_string_lossy().into_owned();
        let canned = canned_apple_segment();
        engine
            .set_asr_engine(Some(Arc::new(
                asr::MockAsrEngine::new(canned.clone()).with_confidence(0.1),
            )))
            .await;

        let config = TranscribeConfig {
            language: None,
            translate: false,
        };
        let pcm = vec![0.0_f32; 16];
        let (segs, label) = engine
            .transcribe(&pcm, &config, None, None)
            .await
            .expect("apple transcription kept when whisper unavailable");
        assert_eq!(label, "apple_native");
        assert_eq!(segs, canned);
    }

    /// A successful Apple result whose confidence is at/above the threshold is
    /// kept as `"apple_native"` — no degraded re-run.
    #[tokio::test]
    async fn high_confidence_apple_result_is_kept() {
        let engine = LensEngine::for_test().await;
        let canned = canned_apple_segment();
        engine
            .set_asr_engine(Some(Arc::new(
                asr::MockAsrEngine::new(canned.clone()).with_confidence(0.9),
            )))
            .await;

        let config = TranscribeConfig {
            language: None,
            translate: false,
        };
        let pcm = vec![0.0_f32; 16];
        let (segs, label) = engine
            .transcribe(&pcm, &config, None, None)
            .await
            .expect("apple transcription");
        assert_eq!(label, "apple_native");
        assert_eq!(segs, canned);
    }
}

#[cfg(test)]
mod media_cancel_tests {
    use super::*;

    /// Registry lifecycle (#43): register yields a live (non-cancelled) token;
    /// `cancel_media_ingest` flips it and reports found; `remove_media_cancel`
    /// clears the entry so a later cancel reports not-found.
    #[tokio::test]
    async fn register_cancel_remove_lifecycle() {
        let engine = LensEngine::for_test().await;
        let sid = "src-cancel-1";

        let token = engine.register_media_cancel(sid);
        assert!(!token.is_cancelled(), "freshly registered token is live");

        assert!(
            engine.cancel_media_ingest(sid),
            "cancel finds the registered token"
        );
        assert!(token.is_cancelled(), "the returned token is now cancelled");

        engine.remove_media_cancel(sid);
        assert!(
            !engine.cancel_media_ingest(sid),
            "cancel after remove finds nothing"
        );
    }

    #[tokio::test]
    async fn cancel_unknown_source_is_false() {
        let engine = LensEngine::for_test().await;
        assert!(!engine.cancel_media_ingest("never-registered"));
    }
}

#[cfg(test)]
mod ask_cancel_tests {
    use super::*;

    #[tokio::test]
    async fn register_then_present_and_live() {
        let engine = LensEngine::for_test().await;
        let token = engine.register_ask("nb-1");
        assert!(
            !token.is_cancelled(),
            "freshly registered ask token is live"
        );
        assert!(
            engine.cancel_ask_notebook("nb-1"),
            "registered ask is found"
        );
        assert!(token.is_cancelled(), "cancel flips the registered token");
    }

    #[tokio::test]
    async fn cancel_unknown_notebook_is_false() {
        let engine = LensEngine::for_test().await;
        assert!(!engine.cancel_ask_notebook("never-registered"));
    }

    #[tokio::test]
    async fn supersede_cancels_old_and_registers_new() {
        let engine = LensEngine::for_test().await;
        let old = engine.register_ask("nb-1");
        let new = engine.register_ask("nb-1");
        assert!(
            old.is_cancelled(),
            "registering a new ask cancels the superseded one"
        );
        assert!(!new.is_cancelled(), "the new ask token is live");
        assert!(
            !Arc::ptr_eq(&old, &new),
            "supersession installs a distinct token"
        );
        // The map holds the NEW token: cancelling flips `new`, not `old`.
        assert!(engine.cancel_ask_notebook("nb-1"));
        assert!(new.is_cancelled());
    }

    #[tokio::test]
    async fn old_guard_drop_does_not_evict_new_token() {
        let engine = LensEngine::for_test().await;
        let old = engine.register_ask("nb-1");
        let old_guard = engine.ask_cancel_guard("nb-1", old);
        // A new ask supersedes; the OLD guard must not remove the NEW token on drop.
        let new = engine.register_ask("nb-1");
        drop(old_guard);
        // The registry still holds `new`, so cancel finds it and flips it.
        assert!(
            engine.cancel_ask_notebook("nb-1"),
            "old guard drop must NOT evict the superseding token"
        );
        assert!(new.is_cancelled());
    }

    #[tokio::test]
    async fn own_guard_drop_removes_token() {
        let engine = LensEngine::for_test().await;
        let token = engine.register_ask("nb-1");
        let guard = engine.ask_cancel_guard("nb-1", token);
        drop(guard);
        assert!(
            !engine.cancel_ask_notebook("nb-1"),
            "own-guard drop removes the registry entry"
        );
    }
}

#[cfg(test)]
mod tts_cancel_tests {
    use super::*;

    #[tokio::test]
    async fn register_then_present_and_live() {
        let engine = LensEngine::for_test().await;
        let token = engine.register_tts("nb-1");
        assert!(
            !token.is_cancelled(),
            "freshly registered tts token is live"
        );
        assert!(
            engine.cancel_synthesis("nb-1"),
            "registered synthesis is found"
        );
        assert!(token.is_cancelled(), "cancel flips the registered token");
    }

    #[tokio::test]
    async fn cancel_unknown_notebook_is_false() {
        let engine = LensEngine::for_test().await;
        assert!(!engine.cancel_synthesis("never-registered"));
    }

    #[tokio::test]
    async fn old_guard_drop_does_not_evict_new_token() {
        let engine = LensEngine::for_test().await;
        let old = engine.register_tts("nb-1");
        let old_guard = engine.tts_cancel_guard("nb-1", old);
        let new = engine.register_tts("nb-1");
        drop(old_guard);
        assert!(
            engine.cancel_synthesis("nb-1"),
            "old guard drop must NOT evict the superseding token"
        );
        assert!(new.is_cancelled());
    }

    #[tokio::test]
    async fn own_guard_drop_removes_token() {
        let engine = LensEngine::for_test().await;
        let token = engine.register_tts("nb-1");
        let guard = engine.tts_cancel_guard("nb-1", token);
        drop(guard);
        assert!(!engine.cancel_synthesis("nb-1"));
    }

    #[test]
    fn notebook_audio_path_joins_under_data_dir() {
        let p = notebook_audio_path(Path::new("/data"), "nb-xyz").expect("valid id resolves");
        assert!(
            p.ends_with("notebooks/nb-xyz/overview.wav"),
            "unexpected path: {p:?}"
        );
    }

    #[test]
    fn notebook_audio_path_rejects_traversal_id() {
        for bad in [
            "../../etc/passwd",
            "..",
            "nb/../../secret",
            "",
            "/etc/passwd",
            "a\\b",
        ] {
            let err = notebook_audio_path(Path::new("/data"), bad)
                .expect_err(&format!("traversal id {bad:?} must be rejected"));
            assert!(matches!(err, LensError::Validation(_)), "got {err:?}");
        }
    }

    #[tokio::test]
    async fn synthesize_overview_no_backend_is_tts_error() {
        let engine = LensEngine::for_test().await;
        let script = DialogueScript {
            turns: vec![
                Turn {
                    speaker: Speaker::Host,
                    text: "hi".into(),
                    emotion: None,
                    source_ids: Vec::new(),
                },
                Turn {
                    speaker: Speaker::Guest,
                    text: "yo".into(),
                    emotion: None,
                    source_ids: Vec::new(),
                },
            ],
        };
        let voices = VoiceConfig::default();
        let cfg = TtsConfig::default();
        let cancel = tokio_util::sync::CancellationToken::new();
        let err = engine
            .synthesize_overview("nb-1", &script, &voices, &cfg, |_p| {}, &cancel)
            .await
            .unwrap_err();
        assert_eq!(err.kind(), "Tts");
    }
}
