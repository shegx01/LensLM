//! Ingest pipeline: parse → chunk → embed → index, transitioning `sources.status`
//! `queued → parsing → embedding → indexed` (or `error`) and streaming [`IngestProgress`].
//!
//! The whole pipeline holds a single permit of [`ingest_lock`](crate::LensEngine::ingest_lock)
//! (serializes ONNX). Re-ingest wipes Lance vectors FIRST, then SQLite chunks (G5 ordering):
//! a completed wipe never leaves orphan rows; a mid-wipe crash is recovered by startup
//! crash-recovery + idempotent re-ingest. The nomic tokenizer is resolved once and cached on
//! the engine; the Lance store is constructed per-ingest (cheap embedded connection).

use std::net::{IpAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;
use tracing::Instrument;

use crate::chunk::{Chunk, chunk_blocks};
use crate::extract::Extractor;
use crate::vector_store::{LanceVectorStore, VectorRow, VectorStore};
use crate::{LensEngine, LensError};

/// Canonical HuggingFace URL for the nomic-embed-text-v1.5 tokenizer.
const NOMIC_TOKENIZER_URL: &str =
    "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json";

/// Connect timeout for the tokenizer download (the file is small).
const TOKENIZER_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

const TOKENIZER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Connect + read timeout for URL source fetches.
///
/// The ingest permit is held for the fetch duration; an unbounded hung fetch stalls all
/// subsequent ingests. 30 s is generous for real pages while keeping the pipeline responsive.
pub const URL_FETCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// `User-Agent` sent on the URL-fetch HTTP GET.
///
/// Mimic a current desktop Chrome so bot-blocking CDNs/WAFs and SPA shells serve the
/// same HTML a browser would. Matched per-OS so the platform token is not inconsistent.
#[cfg(target_os = "macos")]
pub const URL_FETCH_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
     AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36";
#[cfg(target_os = "windows")]
pub const URL_FETCH_USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
     AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36";
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub const URL_FETCH_USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) \
     AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36";

/// Embed batch size — bounds peak memory while keeping the ONNX session warm.
const EMBED_BATCH: usize = 32;

/// Minimum extracted-text length (chars) for a URL source to be considered successfully
/// extracted. Below this floor the page is likely a JS-rendered SPA; source is left `needs_js`.
pub const NEEDS_JS_MIN_CHARS: usize = 200;

/// Secondary SPA hint: `extracted_text.len() / raw_html.len()` below this threshold
/// suggests script/style dominates over readable text. Only applied in the marginal band
/// below [`NEEDS_JS_SUFFICIENT_CHARS`]; alone it false-positives on content-rich pages.
pub const NEEDS_JS_MIN_TEXT_RATIO: f64 = 0.01;

/// At/above this extracted-text length a URL source is indexed directly regardless of
/// [`NEEDS_JS_MIN_TEXT_RATIO`], preventing false-positives on large-but-content-rich pages.
pub const NEEDS_JS_SUFFICIENT_CHARS: usize = 1000;

/// Hard cap for the legacy paste path (`add_text_source`), in bytes. Non-PDF sources use
/// the configurable [`DEFAULT_MAX_SOURCE_BYTES`] cap instead.
pub const MAX_SOURCE_BYTES: usize = 10 * 1024 * 1024;

/// Default non-PDF source cap (50 MB) when [`AppConfig::max_source_mb`] is empty/unset.
/// PDF is exempt (it streams into a building table); see [`resolve_max_source_bytes`].
pub const DEFAULT_MAX_SOURCE_BYTES: usize = 50 * 1024 * 1024;

/// Hard pre-read ceiling for PDF sources (500 MB). PDF is exempt from the configurable cap
/// but still does a whole-file read, so this prevents a multi-GB allocation on absurd inputs.
pub const PDF_PREREAD_HARD_CEILING_BYTES: u64 = 500 * 1024 * 1024;

/// Resolves [`AppConfig::max_source_mb`] to bytes. Empty/unparseable/non-positive
/// values fall back to [`DEFAULT_MAX_SOURCE_BYTES`] to avoid a 0-byte cap.
pub fn resolve_max_source_bytes(cfg_value: &str) -> usize {
    match cfg_value.trim().parse::<usize>() {
        Ok(mb) if mb > 0 => mb.saturating_mul(1024 * 1024),
        _ => DEFAULT_MAX_SOURCE_BYTES,
    }
}

/// Phase literals streamed to the progress sink. URL lifecycle:
/// `fetching → parsing → chunking → [model_download] → embedding → indexing → done`.
pub(crate) mod ingest_phase {
    pub const FETCHING: &str = "fetching";
    pub const DECODING: &str = "decoding";
    pub const TRANSCRIBING: &str = "transcribing";
    pub const PARSING: &str = "parsing";
    pub const CHUNKING: &str = "chunking";
    pub const MODEL_DOWNLOAD: &str = "model_download";
    pub const EMBEDDING: &str = "embedding";
    pub const INDEXING: &str = "indexing";
    pub const DONE: &str = "done";
}

/// One ingestion progress event streamed to the caller-supplied sink.
///
/// `phase` is a fine-grained UX signal; it intentionally does not map 1:1 to the coarse
/// `sources.status` (which folds chunking under parsing and model_download/indexing under
/// embedding to keep crash-recovery simple).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestProgress {
    pub phase: String,
    pub done: u64,
    /// `None` when the upper bound is unknown.
    pub total: Option<u64>,
    /// The ASR backend actually used, surfaced on the terminal transcription event
    /// only (#45): `"cloud"`, `"local_whisper"`, a `"…(fallback)"` variant, etc.
    /// Omitted from the wire when absent (backward-compatible).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub effective_backend: Option<String>,
}

impl IngestProgress {
    fn new(phase: &str, done: u64, total: Option<u64>) -> Self {
        Self {
            phase: phase.to_string(),
            done,
            total,
            effective_backend: None,
        }
    }
}

/// Ingests a queued source end-to-end, streaming [`IngestProgress`].
/// See the module docs for the full contract (status transitions, wipe ordering, serialization).
#[tracing::instrument(skip(engine, on_progress))]
pub async fn ingest_source(
    engine: &LensEngine,
    source_id: &str,
    on_progress: impl FnMut(IngestProgress),
) -> Result<(), LensError> {
    // Permit released before the enrichment enqueue below: the enqueue must happen
    // OUTSIDE the held permit so a full channel can never deadlock against it.
    let result = {
        let _permit = engine
            .ingest_lock()
            .acquire()
            .await
            .map_err(|e| LensError::Internal(format!("ingest semaphore closed: {e}")))?;

        let result = run_ingest(engine, source_id, on_progress).await;

        if let Err(err) = &result {
            let pool = engine.pool().await;
            let repo = crate::notebooks::NotebookRepo::new(&pool);
            if matches!(err, LensError::Cancelled(_)) {
                if let Err(e) = repo
                    .update_source_status(
                        source_id,
                        crate::notebooks::SourceStatus::Queued.as_str(),
                    )
                    .await
                {
                    tracing::warn!(
                        source_id,
                        "failed to reset source to queued after cancel: {e}"
                    );
                }
                if let Err(e) = repo.clear_source_error_meta(source_id).await {
                    tracing::warn!(source_id, "failed to clear error_meta after cancel: {e}");
                }
            } else {
                let prior_attempts = repo
                    .get_source(source_id)
                    .await
                    .ok()
                    .flatten()
                    .and_then(|s| s.error_meta)
                    .and_then(|json| serde_json::from_str::<crate::error::ErrorMeta>(&json).ok())
                    .map(|m| m.attempt_count)
                    .unwrap_or(0);
                let meta =
                    crate::error::ErrorMeta::from_error(err, prior_attempts.saturating_add(1));
                tracing::error!(
                    source_id,
                    kind = %meta.kind,
                    attempt_count = meta.attempt_count,
                    "ingest failed: {}",
                    meta.message
                );
                if let Err(e) = repo.set_source_error(source_id, &meta).await {
                    tracing::warn!(
                        source_id,
                        "failed to mark source as error after ingest failure: {e}"
                    );
                }
            }
        }

        result
    }; // `_permit` dropped HERE — ingest_lock released.

    // Enqueue enrichment AFTER the permit drop; a full channel is recovered by startup rescan.
    if result.is_ok() {
        engine.enqueue_enrichment(source_id);
    }

    result
}

/// Retries a failed source in place: guards that it is `status="error"` and not trashed,
/// transitions it `error → parsing` (crash-recoverable), then delegates to [`ingest_source`]
/// which rewrites `error_meta` with an incremented `attempt_count` on failure.
#[tracing::instrument(skip(engine, on_progress))]
pub async fn retry_source(
    engine: &LensEngine,
    source_id: &str,
    on_progress: impl FnMut(IngestProgress),
) -> Result<(), LensError> {
    {
        let pool = engine.pool().await;
        let repo = crate::notebooks::NotebookRepo::new(&pool);
        let source = repo
            .get_source(source_id)
            .await?
            .ok_or_else(|| LensError::Validation(format!("no source with id {source_id}")))?;
        if source.trashed_at.is_some() {
            return Err(LensError::Validation(
                "cannot retry a trashed source; restore it first".into(),
            ));
        }
        if source.status != crate::notebooks::SourceStatus::Error {
            return Err(LensError::Validation(format!(
                "cannot retry source in status {:?}; only errored sources can be retried",
                source.status.as_str()
            )));
        }
        // Transition to `parsing` (transient/crash-recoverable) before the permit window.
        tracing::info!(source_id, "retrying errored source");
        repo.update_source_status(source_id, crate::notebooks::SourceStatus::Parsing.as_str())
            .await?;
    }

    // Reuse the public entry point: same permit lock, Err handler, and enrichment enqueue.
    ingest_source(engine, source_id, on_progress).await
}

/// Embeds one batch of chunks under `spawn_blocking` and returns their [`VectorRow`]s in order.
/// Shared by the streaming PDF path and the non-PDF accumulate path.
async fn embed_batch_to_rows(
    batch: &[Chunk],
    embedder: &std::sync::Arc<dyn crate::embedder::Embedder>,
    source_id: &str,
    notebook: &str,
) -> Result<Vec<VectorRow>, LensError> {
    let texts: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
    let embedder = embedder.clone();
    let vectors = tokio::task::spawn_blocking(move || embedder.embed_documents_owned(texts))
        .await
        .map_err(|e| LensError::Model(format!("embed task panicked: {e}")))??;

    if vectors.len() != batch.len() {
        return Err(LensError::Model(format!(
            "embedder returned {} vectors for {} inputs",
            vectors.len(),
            batch.len()
        )));
    }

    Ok(batch
        .iter()
        .zip(vectors)
        .map(|(chunk, vector)| VectorRow {
            chunk_id: chunk.id.clone(),
            source_id: source_id.to_string(),
            notebook_id: notebook.to_string(),
            level: chunk.level,
            vector,
        })
        .collect())
}

/// The inner pipeline (without the error-status side effect / semaphore).
async fn run_ingest(
    engine: &LensEngine,
    source_id: &str,
    mut on_progress: impl FnMut(IngestProgress),
) -> Result<(), LensError> {
    let pool = engine.pool().await;
    let data_dir = engine.data_dir().await;
    let repo = crate::notebooks::NotebookRepo::new(&pool);

    let source = repo
        .get_source(source_id)
        .await?
        .ok_or_else(|| LensError::Validation(format!("no source with id {source_id}")))?;

    let kind = source.kind;
    let status = source.status;
    let text_like = kind.is_text_like();
    // PDF: streams into a building table, exempt from raw-bytes / extracted-text caps.
    let is_pdf = kind == crate::parse::SourceKind::Pdf;

    let max_source_bytes = resolve_max_source_bytes(&engine.config().await.max_source_mb);

    let raw: Vec<u8> = if kind == crate::parse::SourceKind::Url {
        on_progress(IngestProgress::new(ingest_phase::FETCHING, 0, Some(1)));

        let url = source.locator.clone();
        let bytes = fetch_url_guarded(&url, URL_FETCH_TIMEOUT, max_source_bytes).await?;

        on_progress(IngestProgress::new(ingest_phase::FETCHING, 1, Some(1)));
        bytes
    } else {
        // Check size via metadata BEFORE reading to prevent a multi-GB allocation.
        // Non-PDF: configured cap; PDF: hard 500 MB ceiling (still does a whole-file read).
        let meta = tokio::fs::metadata(&source.locator)
            .await
            .map_err(|e| LensError::Io(format!("stat source {}: {e}", source.locator)))?;
        let file_size = meta.len();
        if is_pdf {
            if file_size > PDF_PREREAD_HARD_CEILING_BYTES {
                return Err(LensError::Validation(format!(
                    "PDF source is {file_size} bytes, exceeding the \
                     {PDF_PREREAD_HARD_CEILING_BYTES}-byte pre-read ceiling"
                )));
            }
        } else if !text_like && file_size > max_source_bytes as u64 {
            return Err(LensError::Validation(format!(
                "source is {file_size} bytes, exceeding the {max_source_bytes}-byte ingest limit"
            )));
        }

        tokio::fs::read(&source.locator)
            .await
            .map_err(|e| LensError::Io(format!("read source {}: {e}", source.locator)))?
    };

    // Stage-1 size guard: cap RAW bytes before invoking the extractor (text/MD and PDF exempt).
    if !text_like && !is_pdf && raw.len() > max_source_bytes {
        return Err(LensError::Validation(format!(
            "source is {} raw bytes, exceeding the {max_source_bytes}-byte ingest limit",
            raw.len()
        )));
    }

    // DERIVED no-op short-circuit: hash over raw bytes lets an unchanged binary skip extraction.
    if !text_like {
        let raw_hash = sha256_hex(&raw);
        if status == crate::notebooks::SourceStatus::Indexed
            && source.content_hash.as_deref() == Some(raw_hash.as_str())
        {
            tracing::info!(
                source_id,
                "binary source already indexed with unchanged raw bytes; no-op (no re-extract)"
            );
            on_progress(IngestProgress::new(ingest_phase::DONE, 1, Some(1)));
            return Ok(());
        }
    }

    // Audio takes the decode+transcribe path instead of an `Extractor` (issue #43).
    // Placed AFTER the Stage-1 size guard and the DERIVED no-op short-circuit (which
    // both run above) and BEFORE `extractor_for`, which has no audio extractor.
    if kind == crate::parse::SourceKind::Audio {
        let pool = pool.clone();
        return run_audio_ingest(
            engine,
            &pool,
            &data_dir,
            source_id,
            &source,
            status,
            max_source_bytes,
            &raw,
            on_progress,
        )
        .await;
    }

    // Sniff `.json` files for JSON Lines: if ≥2 newline-delimited JSON values, use the jsonl extractor.
    let effective_kind: String = if kind == crate::parse::SourceKind::Json && sniff_is_jsonl(&raw) {
        tracing::info!(
            source_id,
            "`.json` source sniffed as JSON Lines (>=2 newline-delimited JSON values); using jsonl extractor"
        );
        crate::parse::SourceKind::Jsonl.as_str().to_string()
    } else {
        kind.as_str().to_string()
    };

    let extractor = crate::extract::extractor_for(&effective_kind)?;
    // DERIVED extraction is CPU-bound; run under spawn_blocking. Text/MD is cheap and stays inline.
    // `raw` is handed back from the closure — reused for the content hash and the needs_js ratio.
    let (raw, out) = if text_like {
        let out = extractor.extract(&raw)?;
        (raw, out)
    } else {
        tokio::task::spawn_blocking(move || {
            let out = extractor.extract(&raw);
            (raw, out)
        })
        .await
        .map_err(|e| LensError::Internal(format!("extractor task panicked: {e}")))
        .and_then(|(raw, out)| out.map(|out| (raw, out)))?
    };

    let ctx = IngestContext {
        engine,
        pool: &pool,
        data_dir: &data_dir,
        source_id,
        source: &source,
        status,
        is_pdf,
        text_like,
        max_source_bytes,
    };

    // INVARIANT: needs_ocr/needs_js are terminal-pending statuses returned as Ok(()) — NOT Err —
    // so the error-flip in `ingest_source` never fires. PdfExtractor signals no text layer
    // by returning empty output (no Err); we set needs_ocr and return Ok early.
    if kind == crate::parse::SourceKind::Pdf && out.extracted_text.trim().is_empty() {
        tracing::info!(
            source_id,
            "PDF source produced no text layer — likely scanned/image-only; setting needs_ocr"
        );
        ctx.set_terminal_pending(crate::notebooks::SourceStatus::NeedsOcr.as_str())
            .await?;
        return Ok(()); // Ok so the Err→error flip in ingest_source never fires.
    }

    if kind == crate::parse::SourceKind::Url {
        match ctx.try_js_render(&raw, &out, &mut on_progress).await? {
            JsRenderOutcome::Handled => return Ok(()),
            JsRenderOutcome::NotNeeded => {}
        }
    }

    ctx.index_extract_output(&raw, out, &mut on_progress).await
}

/// Removes an audio-ingest cancellation token from the engine registry on drop,
/// so completion/error/cancel all clean up exactly once (issue #43).
struct MediaCancelGuard<'a> {
    engine: &'a LensEngine,
    source_id: &'a str,
}

impl Drop for MediaCancelGuard<'_> {
    fn drop(&mut self) {
        self.engine.remove_media_cancel(self.source_id);
    }
}

/// Audio ingest branch (issue #43): decode+resample (#41) → transcribe (#42) →
/// concatenate transcript → chunk/embed/index via the shared tail. Cancellation
/// is cooperative (checked at every decode-window boundary and before transcription).
/// Never sets error status here — the outer `ingest_source` handler does.
#[allow(clippy::too_many_arguments)]
async fn run_audio_ingest(
    engine: &LensEngine,
    pool: &sqlx::SqlitePool,
    data_dir: &Path,
    source_id: &str,
    source: &crate::notebooks::Source,
    status: crate::notebooks::SourceStatus,
    max_source_bytes: usize,
    raw: &[u8],
    mut on_progress: impl FnMut(IngestProgress),
) -> Result<(), LensError> {
    let token = engine.register_media_cancel(source_id);
    let _cancel_guard = MediaCancelGuard { engine, source_id };

    // Decode runs in `spawn_blocking`; progress rides a BOUNDED channel (capacity 1)
    // so `blocking_send` back-pressures decode to the async drain rate — the rendezvous
    // the cancel test relies on. `on_progress` is not `Send`, so it is called only on the
    // async side, never inside the closure.
    let (decode_tx, mut decode_rx) = tokio::sync::mpsc::channel::<IngestProgress>(1);
    let path = PathBuf::from(&source.locator);
    let decode_token = token.clone();
    let decode_handle = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, LensError> {
        // 16 kHz mono; cap decoded audio at ~4h so a pathological compressed file cannot exhaust memory.
        const MAX_PCM_SAMPLES: usize = 16_000 * 60 * 60 * 4;
        let mut pcm: Vec<f32> = Vec::new();
        let mut window_index: u64 = 0;
        for window in crate::transcription::decode_resample_windows(&path, Default::default())? {
            if decode_token.is_cancelled() {
                return Err(LensError::Cancelled(
                    "audio ingest cancelled during decode".into(),
                ));
            }
            let window = window?;
            if pcm.len() + window.len() > MAX_PCM_SAMPLES {
                return Err(LensError::Validation(
                    "audio exceeds the maximum supported duration (~4 hours)".into(),
                ));
            }
            pcm.extend(window);
            window_index += 1;
            if decode_tx
                .blocking_send(IngestProgress::new(
                    ingest_phase::DECODING,
                    window_index,
                    None,
                ))
                .is_err()
            {
                return Err(LensError::Cancelled(
                    "audio ingest cancelled during decode".into(),
                ));
            }
        }
        // Replicate the empty/silence guards that `decode_and_resample_audio`
        // enforces but the streaming windows path does not (transcription.rs).
        if pcm.is_empty() {
            return Err(LensError::EmptyAudio("audio decoded to no samples".into()));
        }
        if pcm.iter().all(|s| *s == 0.0) {
            return Err(LensError::EmptyAudio(
                "audio is empty or entirely silent".into(),
            ));
        }
        Ok(pcm)
    });

    while let Some(ev) = decode_rx.recv().await {
        on_progress(ev);
    }
    let pcm = decode_handle
        .await
        .map_err(|e| LensError::Internal(format!("audio decode task panicked: {e}")))??;

    if token.is_cancelled() {
        return Err(LensError::Cancelled(
            "audio ingest cancelled before transcription".into(),
        ));
    }

    // Separate unbounded f32 channel; a forwarder maps ASR progress into the transcribing phase.
    on_progress(IngestProgress::new(
        ingest_phase::TRANSCRIBING,
        0,
        Some(100),
    ));
    let (asr_tx, mut asr_rx) = tokio::sync::mpsc::unbounded_channel::<f32>();
    let asr_config = crate::asr::TranscribeConfig::default();
    let mut transcribe_fut =
        std::pin::pin!(engine.transcribe(&pcm, &asr_config, Some(asr_tx), Some(token.clone())));
    let (segments, effective_backend) = loop {
        tokio::select! {
            biased;
            Some(p) = asr_rx.recv() => {
                on_progress(IngestProgress::new(
                    ingest_phase::TRANSCRIBING,
                    (p * 100.0) as u64,
                    Some(100),
                ));
            }
            result = &mut transcribe_fut => break result?,
        }
    };
    while let Ok(p) = asr_rx.try_recv() {
        on_progress(IngestProgress::new(
            ingest_phase::TRANSCRIBING,
            (p * 100.0) as u64,
            Some(100),
        ));
    }
    // Terminal transcription event carries the effective backend for UI transparency (#45).
    let mut done_event = IngestProgress::new(ingest_phase::TRANSCRIBING, 100, Some(100));
    done_event.effective_backend = Some(effective_backend.to_string());
    on_progress(done_event);

    let out = transcript_extract_output(&segments)?;

    let ctx = IngestContext {
        engine,
        pool,
        data_dir,
        source_id,
        source,
        status,
        is_pdf: false,
        text_like: false,
        max_source_bytes,
    };
    ctx.index_extract_output(raw, out, &mut on_progress).await
}

/// Builds a canonical [`ExtractOutput`] directly from transcript segments (issue
/// #44): one [`Block`] per [`TranscriptSegment`] with an index-aligned
/// [`SourceAnchor::Audio`], never routed through `parse_blocks`.
///
/// The canonical buffer is the segment texts joined with `SEG_SEP`; each block's
/// `char_start..char_end` indexes into it byte-identically. Empty input yields an
/// empty output (zero blocks/chunks). Non-finite timestamps are rejected so a
/// poisoned value never reaches the `[min,max]` aggregation in
/// `attach_anchors_to_chunks`.
fn transcript_extract_output(
    segments: &[crate::asr::TranscriptSegment],
) -> Result<crate::extract::ExtractOutput, LensError> {
    const SEG_SEP: &str = "\n\n";

    let mut extracted_text = String::new();
    let mut blocks: Vec<crate::parse::Block> = Vec::with_capacity(segments.len());
    let mut anchors: Vec<crate::extract::SourceAnchor> = Vec::with_capacity(segments.len());

    for (i, seg) in segments.iter().enumerate() {
        if !seg.start_second.is_finite() || !seg.end_second.is_finite() {
            return Err(LensError::Parse(format!(
                "transcript segment {i} has a non-finite timestamp"
            )));
        }
        if seg.start_second < 0.0 || seg.start_second > seg.end_second {
            return Err(LensError::Parse(format!(
                "transcript segment {i} has an invalid timestamp range \
                 [{}, {}]: start must be >= 0 and <= end",
                seg.start_second, seg.end_second
            )));
        }
        if i > 0 {
            extracted_text.push_str(SEG_SEP);
        }
        let char_start = extracted_text.len();
        extracted_text.push_str(&seg.text);
        let char_end = extracted_text.len();

        blocks.push(crate::parse::Block {
            block_type: crate::parse::BlockType::Paragraph.as_str().to_string(),
            section_path: String::new(),
            text: seg.text.clone(),
            char_start,
            char_end,
        });
        anchors.push(crate::extract::SourceAnchor::Audio {
            start_second: seg.start_second,
            end_second: seg.end_second,
        });
    }

    Ok(crate::extract::ExtractOutput {
        extracted_text,
        blocks,
        anchors,
        table_markdown: None,
    })
}

/// Outcome of the URL JS-render gate. `Handled` means a terminal status was set and the
/// caller must return `Ok(())`. `NotNeeded` means fall through to the shared static index tail.
enum JsRenderOutcome {
    Handled,
    NotNeeded,
}

/// Compute-once context for a single [`run_ingest`] invocation, passed to downstream helpers.
/// [`NotebookRepo`](crate::notebooks::NotebookRepo) is NOT a field — it borrows the pool and
/// would make this self-referential; reconstruct it on demand via [`IngestContext::repo`].
struct IngestContext<'a> {
    engine: &'a LensEngine,
    pool: &'a sqlx::SqlitePool,
    data_dir: &'a Path,
    source_id: &'a str,
    source: &'a crate::notebooks::Source,
    status: crate::notebooks::SourceStatus,
    is_pdf: bool,
    text_like: bool,
    max_source_bytes: usize,
}

impl IngestContext<'_> {
    fn repo(&self) -> crate::notebooks::NotebookRepo<'_> {
        crate::notebooks::NotebookRepo::new(self.pool)
    }

    /// Wipes prior indexed content and sets a terminal-pending status (`needs_ocr`/`needs_js`/`render_failed`).
    async fn set_terminal_pending(&self, status: &str) -> Result<(), LensError> {
        set_terminal_pending(
            &self.repo(),
            self.pool,
            self.data_dir,
            &self.source.notebook_id,
            self.source_id,
            status,
        )
        .await
    }

    /// JS-render gate for URL sources: if static extraction is too thin, renders in the
    /// offscreen webview and re-extracts. INVARIANT: every terminal outcome is `Ok(..)`, NOT
    /// `Err` — render failure maps to `render_failed`, never the crash-recoverable `error`.
    async fn try_js_render(
        &self,
        raw: &[u8],
        out: &crate::extract::ExtractOutput,
        on_progress: &mut impl FnMut(IngestProgress),
    ) -> Result<JsRenderOutcome, LensError> {
        let engine = self.engine;
        let source = self.source;
        let source_id = self.source_id;

        let text_len = out.extracted_text.len();
        let raw_len = raw.len();
        let ratio = if raw_len == 0 {
            0.0f64
        } else {
            text_len as f64 / raw_len as f64
        };
        // Per-source force_js_render flag OR-es in ahead of the heuristics; the global
        // js_render_enabled opt-out still wins (opt-out beats per-source force).
        let force_js_render = source.force_js_render != 0;
        // Two-arm oracle: (1) absolute floor below NEEDS_JS_MIN_CHARS; or (2) marginal band
        // below NEEDS_JS_SUFFICIENT_CHARS AND ratio below NEEDS_JS_MIN_TEXT_RATIO.
        let needs_js = force_js_render
            || text_len < NEEDS_JS_MIN_CHARS
            || (text_len < NEEDS_JS_SUFFICIENT_CHARS && ratio < NEEDS_JS_MIN_TEXT_RATIO);
        if !needs_js {
            return Ok(JsRenderOutcome::NotNeeded);
        }

        tracing::info!(
            source_id,
            text_len,
            raw_len,
            ratio,
            force_js_render,
            "URL source needs JS render (forced or near-empty extraction); attempting JS-render fallback"
        );

        let js_render_enabled = engine.config().await.js_render_enabled;
        let renderer = if js_render_enabled {
            engine.js_renderer().await
        } else {
            None
        };

        if let Some(renderer) = renderer {
            let span = tracing::info_span!("js_render", source_id, url = %source.locator);
            let started = std::time::Instant::now();
            let rendered = match renderer
                .render_html(&source.locator)
                .instrument(span.clone())
                .await
            {
                Ok(rendered) => rendered,
                Err(e) => {
                    let _guard = span.enter();
                    tracing::warn!(
                        elapsed_ms = started.elapsed().as_millis() as u64,
                        outcome = "render_failed",
                        error = %e,
                        "js_render fallback errored; setting render_failed"
                    );
                    drop(_guard);
                    self.set_terminal_pending(
                        crate::notebooks::SourceStatus::RenderFailed.as_str(),
                    )
                    .await?;
                    return Ok(JsRenderOutcome::Handled);
                }
            };

            // Feed rendered HTML through the same extractor. The ratio arm is NOT reused
            // (no faithful raw-bytes denominator for a JS-rendered SPA outerHTML).
            if let Some(html) = rendered {
                if html.len() > self.max_source_bytes {
                    let _guard = span.enter();
                    tracing::warn!(
                        elapsed_ms = started.elapsed().as_millis() as u64,
                        outcome = "render_failed",
                        rendered_len = html.len(),
                        max_source_bytes = self.max_source_bytes,
                        "rendered HTML exceeds byte cap; setting render_failed"
                    );
                    drop(_guard);
                    self.set_terminal_pending(
                        crate::notebooks::SourceStatus::RenderFailed.as_str(),
                    )
                    .await?;
                    return Ok(JsRenderOutcome::Handled);
                }
                let rendered_out = match crate::extract::url::UrlExtractor.extract(html.as_bytes())
                {
                    Ok(o) => o,
                    Err(e) => {
                        let _guard = span.enter();
                        tracing::warn!(
                            elapsed_ms = started.elapsed().as_millis() as u64,
                            outcome = "render_failed",
                            error = %e,
                            "render-branch extract errored; setting render_failed"
                        );
                        drop(_guard);
                        self.set_terminal_pending(
                            crate::notebooks::SourceStatus::RenderFailed.as_str(),
                        )
                        .await?;
                        return Ok(JsRenderOutcome::Handled);
                    }
                };
                if rendered_out.extracted_text.len() >= NEEDS_JS_MIN_CHARS {
                    let _guard = span.enter();
                    tracing::info!(
                        elapsed_ms = started.elapsed().as_millis() as u64,
                        outcome = "indexed",
                        "js_render fallback cleared the content oracle; indexing rendered DOM"
                    );
                    drop(_guard);
                    // Pass the ORIGINAL fetched `raw` (JS-shell bytes) as the content-hash
                    // identity so re-ingest determinism is unchanged.
                    self.index_extract_output(raw, rendered_out, on_progress)
                        .await?;
                    return Ok(JsRenderOutcome::Handled);
                }
            }

            let _guard = span.enter();
            tracing::info!(
                elapsed_ms = started.elapsed().as_millis() as u64,
                outcome = "render_failed",
                "js_render fallback did not produce indexable content; setting render_failed"
            );
            drop(_guard);
            self.set_terminal_pending(crate::notebooks::SourceStatus::RenderFailed.as_str())
                .await?;
            return Ok(JsRenderOutcome::Handled);
        }

        tracing::info!(
            source_id,
            outcome = if js_render_enabled {
                "needs_js"
            } else {
                "opt_out"
            },
            "no JS-render fallback available; setting needs_js"
        );
        self.set_terminal_pending(crate::notebooks::SourceStatus::NeedsJs.as_str())
            .await?;
        Ok(JsRenderOutcome::Handled)
    }

    /// Shared downstream tail for both the static and JS-render branches: Stage-2 guard →
    /// canonical-buffer persist → content hash → chunk → embed → index → `Indexed`.
    /// The render fallback passes the original fetched `raw` as the content-hash identity.
    async fn index_extract_output(
        &self,
        raw: &[u8],
        out: crate::extract::ExtractOutput,
        on_progress: &mut impl FnMut(IngestProgress),
    ) -> Result<(), LensError> {
        let &IngestContext {
            engine,
            pool,
            data_dir,
            source_id,
            source,
            status,
            is_pdf,
            text_like,
            max_source_bytes,
        } = self;
        let repo = self.repo();
        // Stage-2 size guard on the canonical buffer (PDF exempt — streams into a building table).
        if !is_pdf && out.extracted_text.len() > max_source_bytes {
            return Err(LensError::Validation(format!(
                "source is {} bytes, exceeding the {max_source_bytes}-byte ingest limit",
                out.extracted_text.len()
            )));
        }

        // For DERIVED kinds, persist `extracted_text` to the `.extracted.txt` sibling.
        // For text/MD the locator IS the canonical buffer — no sibling needed.
        if !text_like {
            let sources_dir = data_dir.join("sources");
            tokio::fs::create_dir_all(&sources_dir)
                .await
                .map_err(|e| LensError::Io(format!("{}: {e}", sources_dir.display())))?;
            let sibling = extracted_sibling_path(data_dir, source_id);
            tokio::fs::write(&sibling, &out.extracted_text)
                .await
                .map_err(|e| LensError::Io(format!("{}: {e}", sibling.display())))?;
        }

        // Persist the `table_markdown` sibling for tabular kinds (never embedded; display only).
        if let Some(ref md) = out.table_markdown {
            let tables_path = tables_sibling_path(data_dir, source_id);
            tokio::fs::write(&tables_path, md)
                .await
                .map_err(|e| LensError::Io(format!("{}: {e}", tables_path.display())))?;
        }

        let canonical: &str = &out.extracted_text;

        // DERIVED kinds reuse the raw-bytes hash; text/MD hash the canonical text here.
        let content_hash = if text_like {
            let h = sha256_hex(canonical.as_bytes());
            if status == crate::notebooks::SourceStatus::Indexed
                && source.content_hash.as_deref() == Some(h.as_str())
            {
                tracing::info!(
                    source_id,
                    "source already indexed with unchanged content; no-op"
                );
                on_progress(IngestProgress::new(ingest_phase::DONE, 1, Some(1)));
                return Ok(());
            }
            h
        } else {
            sha256_hex(raw)
        };

        let store = LanceVectorStore::new(data_dir, pool.clone());
        let notebook = source.notebook_id.clone();
        // Resolve the notebook's embedding coordinate once; threaded into every drop/add below.
        let (embed_model, embed_dim, embed_backend) = engine
            .resolve_notebook_embedding(&crate::NotebookId::from(notebook.clone()))
            .await?;
        let coord = crate::vector_store::Coordinate::new(
            notebook.clone(),
            embed_backend,
            embed_model.clone(),
            embed_dim,
        );

        // INVARIANT: status must be `parsing` (transient) BEFORE the cross-store wipe so a
        // mid-wipe crash is reclaimed by startup crash-recovery on next launch.
        {
            repo.update_source_status(source_id, crate::notebooks::SourceStatus::Parsing.as_str())
                .await?;
        }
        on_progress(IngestProgress::new(ingest_phase::PARSING, 0, Some(1)));
        let blocks = &out.blocks;
        on_progress(IngestProgress::new(ingest_phase::PARSING, 1, Some(1)));

        on_progress(IngestProgress::new(ingest_phase::CHUNKING, 0, None));
        maybe_emit_tokenizer_download(data_dir, on_progress);
        let tokenizer = engine.tokenizer().await?;
        let mut chunks = chunk_blocks(canonical, blocks, &tokenizer)?;
        let total_tokens: i64 = chunks
            .iter()
            .filter(|c| c.level == 0)
            .map(|c| c.token_end - c.token_start)
            .sum();

        attach_anchors_to_chunks(&mut chunks, blocks, &out.anchors)?;

        on_progress(IngestProgress::new(
            ingest_phase::CHUNKING,
            chunks.len() as u64,
            Some(chunks.len() as u64),
        ));

        // G5 ordering: Lance vectors dropped FIRST, then SQLite chunks — a completed wipe
        // never leaves orphan Lance rows; chunk delete+insert run inside ONE transaction.
        store.drop_source(&coord, source_id).await?;

        let mut tx = pool.begin().await?;
        delete_chunks_for_source(&mut tx, source_id).await?;
        insert_chunks(&mut tx, source_id, &chunks).await?;
        tx.commit().await?;

        // Content changed (unchanged-content paths returned early): stale enrichment reset
        // so the post-Indexed enqueue re-runs the pass.
        repo.update_enrichment_status(source_id, crate::notebooks::EnrichmentStatus::None)
            .await?;

        // Empty source: zero chunks to embed; skip embedder load to avoid a spurious
        // ~130 MB model download. Finalize as an empty-but-indexed row.
        if chunks.is_empty() {
            repo.update_source_metadata(source_id, 0, &content_hash)
                .await?;
            repo.update_source_status(source_id, crate::notebooks::SourceStatus::Indexed.as_str())
                .await?;
            // A successful (re-)ingest wipes any stale failure reason (#73).
            repo.clear_source_error_meta(source_id).await?;
            on_progress(IngestProgress::new(ingest_phase::DONE, 1, Some(1)));
            return Ok(());
        }

        repo.update_source_status(
            source_id,
            crate::notebooks::SourceStatus::Embedding.as_str(),
        )
        .await?;

        on_progress(IngestProgress::new(ingest_phase::MODEL_DOWNLOAD, 0, None));
        // Bulk workload: GPU-eligible on Apple Silicon for a GPU-hinted model, else CPU.
        let embedder = engine
            .embedder_for(
                &embed_model,
                embed_backend,
                crate::embedder::WorkloadKind::Bulk,
            )
            .await?;
        on_progress(IngestProgress::new(
            ingest_phase::MODEL_DOWNLOAD,
            1,
            Some(1),
        ));

        let total = chunks.len() as u64;
        on_progress(IngestProgress::new(ingest_phase::EMBEDDING, 0, Some(total)));

        if is_pdf {
            // PDF streaming: embed each EMBED_BATCH into a gen-suffixed building table and free it,
            // then flip building → active on completion (atomicity: source never half-visible).
            // ORDERING: wipe → SWEEP → CREATE → SEED → POPULATE → FLIP must be preserved;
            // seed copies from the already-wiped active table, so no old vectors are reintroduced.
            // The ingest_lock permit is held for the full streaming duration (wipe-before-seed).
            let lock_start = std::time::Instant::now();

            store.sweep_orphan_building_tables(&coord).await?;

            let building_name = store.create_building_table(&coord).await?;
            tracing::info!(
                source_id,
                notebook = %notebook,
                building = %building_name,
                "streaming PDF ingest: created building table"
            );

            store
                .seed_building_from_active(&coord, &building_name, source_id)
                .await?;

            let mut embedded: u64 = 0;
            for batch in chunks.chunks(EMBED_BATCH) {
                let rows = embed_batch_to_rows(batch, &embedder, source_id, &notebook).await?;
                let inserted = rows.len();
                store
                    .add_to_table_no_index(&building_name, rows, embed_dim)
                    .await?;
                tracing::info!(
                    source_id,
                    building = %building_name,
                    inserted,
                    "streaming PDF ingest: inserted batch into building table"
                );

                // Crash seam (issue #71, Step 5): after at least one batch has landed in
                // the building table but BEFORE the flip, simulate a process crash.
                #[cfg(feature = "test-util")]
                if crate::vector_store::CRASH_AFTER_STREAMING_ADD_BEFORE_FLIP
                    .swap(false, std::sync::atomic::Ordering::SeqCst)
                {
                    return Err(LensError::Internal(
                        "CRASH_AFTER_STREAMING_ADD_BEFORE_FLIP (test-only crash injection)"
                            .to_string(),
                    ));
                }

                embedded += batch.len() as u64;
                on_progress(IngestProgress::new(
                    ingest_phase::EMBEDDING,
                    embedded,
                    Some(total),
                ));
            }

            on_progress(IngestProgress::new(ingest_phase::INDEXING, 0, Some(1)));
            store
                .build_index_on_table(&building_name, embed_dim)
                .await?;
            store.flip_active(&coord, &building_name).await?;
            tracing::info!(
                source_id,
                notebook = %notebook,
                building = %building_name,
                lock_hold_ms = lock_start.elapsed().as_millis() as u64,
                "streaming PDF ingest: built index + flipped building → active"
            );
            on_progress(IngestProgress::new(ingest_phase::INDEXING, 1, Some(1)));
        } else {
            let mut rows: Vec<VectorRow> = Vec::with_capacity(chunks.len());
            let mut embedded: u64 = 0;
            for batch in chunks.chunks(EMBED_BATCH) {
                let mut batch_rows =
                    embed_batch_to_rows(batch, &embedder, source_id, &notebook).await?;
                rows.append(&mut batch_rows);
                embedded += batch.len() as u64;
                on_progress(IngestProgress::new(
                    ingest_phase::EMBEDDING,
                    embedded,
                    Some(total),
                ));
            }

            on_progress(IngestProgress::new(ingest_phase::INDEXING, 0, Some(1)));
            store.add(&coord, rows).await?;
            on_progress(IngestProgress::new(ingest_phase::INDEXING, 1, Some(1)));
        }

        repo.update_source_metadata(source_id, total_tokens, &content_hash)
            .await?;
        repo.update_source_status(source_id, crate::notebooks::SourceStatus::Indexed.as_str())
            .await?;
        // A successful (re-)ingest wipes any stale failure reason (#73).
        repo.clear_source_error_meta(source_id).await?;
        on_progress(IngestProgress::new(ingest_phase::DONE, 1, Some(1)));
        Ok(())
    }
}

/// SHA-256 of `bytes`, lowercase hex.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    crate::hex_encode(&Sha256::digest(bytes))
}

/// Number of leading bytes inspected by the JSON-vs-JSONL content sniff.
const JSONL_SNIFF_WINDOW: usize = 64 * 1024;

/// Returns `true` if `raw` looks like JSON Lines: ≥2 newline-delimited lines each parsing
/// as a standalone JSON value. Inspects the first [`JSONL_SNIFF_WINDOW`] bytes.
fn sniff_is_jsonl(raw: &[u8]) -> bool {
    let window = &raw[..raw.len().min(JSONL_SNIFF_WINDOW)];
    let Ok(s) = std::str::from_utf8(window) else {
        return false;
    };
    let s = s.strip_prefix('\u{FEFF}').unwrap_or(s);
    let mut parsed_lines = 0usize;
    for line in s.split('\n') {
        let line = line.trim_end_matches('\r').trim();
        if line.is_empty() {
            continue;
        }
        if serde_json::from_str::<serde_json::Value>(line).is_ok() {
            parsed_lines += 1;
            if parsed_lines >= 2 {
                return true;
            }
        } else {
            return false;
        }
    }
    false
}

/// Returns `{data_dir}/sources/{source_id}.extracted.txt` — shared by the ingest write site
/// and the purge cleanup site so they can never derive a different path.
pub(crate) fn extracted_sibling_path(data_dir: &Path, source_id: &str) -> PathBuf {
    data_dir
        .join("sources")
        .join(format!("{source_id}.extracted.txt"))
}

/// Returns `{data_dir}/sources/{source_id}.tables.md` — shared by ingest write and purge
/// cleanup so they can never diverge. Never embedded; exists for future display only.
pub(crate) fn tables_sibling_path(data_dir: &Path, source_id: &str) -> PathBuf {
    data_dir
        .join("sources")
        .join(format!("{source_id}.tables.md"))
}

/// Accepted `Content-Type` prefixes for URL source bodies. Rejects binaries before reading.
const URL_ALLOWED_CONTENT_TYPES: &[&str] = &["text/html", "application/xhtml+xml", "text/"];

/// Cloud-metadata IMDS IP — blocked explicitly as defense-in-depth (also covered by link-local).
const CLOUD_METADATA_IP: IpAddr = IpAddr::V4(std::net::Ipv4Addr::new(169, 254, 169, 254));

/// SSRF guard: `true` if `ip` must NOT be fetched (loopback, link-local, RFC1918, ULA, unspecified).
fn is_blocked_ip(ip: IpAddr) -> bool {
    if ip == CLOUD_METADATA_IP {
        return true;
    }
    match ip {
        IpAddr::V4(v4) => {
            is_loopback_ip(ip)
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                // 100.64.0.0/10 (CGNAT) — not covered by is_private().
                || (v4.octets()[0] == 100 && (v4.octets()[1] & 0xc0) == 0x40)
        }
        IpAddr::V6(v6) => {
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_blocked_ip(IpAddr::V4(mapped));
            }
            is_loopback_ip(ip)
                || v6.is_unspecified()
                // Link-local fe80::/10.
                || (v6.segments()[0] & 0xffc0) == 0xfe80
                // Unique-local fc00::/7.
                || (v6.segments()[0] & 0xfe00) == 0xfc00
        }
    }
}

/// `true` if `ip` is loopback (`127.0.0.0/8`, `::1`, or IPv4-mapped loopback).
/// Single definition reused by the SSRF reject gate and the Ollama require-loopback gate.
pub(crate) fn is_loopback_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback(),
        IpAddr::V6(v6) => {
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return mapped.is_loopback();
            }
            v6.is_loopback()
        }
    }
}

/// Validated loopback base URL: host string + guard-approved loopback addrs for DNS-pin.
/// `pinned_addrs` is empty for IP-literal hosts (no DNS step to rebind).
#[derive(Debug)]
pub(crate) struct LoopbackTarget {
    /// The lowercased host string used as the `resolve_to_addrs` key.
    pub host: String,
    pub pinned_addrs: Vec<std::net::SocketAddr>,
}

/// Loopback-only gate for a local-service base URL (Ollama embedder safety contract).
/// Rejects unless EVERY resolved address is loopback; returns pinned addrs to close DNS-rebind TOCTOU.
pub(crate) fn require_loopback(base_url: &str) -> Result<LoopbackTarget, LensError> {
    let parsed = url::Url::parse(base_url)
        .map_err(|e| LensError::Validation(format!("invalid base URL {base_url:?}: {e}")))?;
    let scheme = parsed.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(LensError::Validation(format!(
            "base URL scheme must be http or https, got {scheme:?}"
        )));
    }
    let host = parsed
        .host()
        .ok_or_else(|| LensError::Validation(format!("base URL {base_url:?} has no host")))?;
    let port = parsed.port_or_known_default().unwrap_or(80);

    let domain = match host {
        url::Host::Ipv4(v4) => {
            if !is_loopback_ip(IpAddr::V4(v4)) {
                return Err(LensError::Validation(format!(
                    "embedding base URL host {v4} is not loopback; the Ollama embedder \
                     accepts loopback-only addresses"
                )));
            }
            return Ok(LoopbackTarget {
                host: v4.to_string(),
                pinned_addrs: Vec::new(),
            });
        }
        url::Host::Ipv6(v6) => {
            if !is_loopback_ip(IpAddr::V6(v6)) {
                return Err(LensError::Validation(format!(
                    "embedding base URL host {v6} is not loopback; the Ollama embedder \
                     accepts loopback-only addresses"
                )));
            }
            return Ok(LoopbackTarget {
                host: v6.to_string(),
                pinned_addrs: Vec::new(),
            });
        }
        url::Host::Domain(d) => d.to_string(),
    };

    let addrs = (domain.as_str(), port).to_socket_addrs().map_err(|e| {
        LensError::Network(format!("failed to resolve base URL host {domain}: {e}"))
    })?;
    let mut pinned_addrs = Vec::new();
    for addr in addrs {
        if !is_loopback_ip(addr.ip()) {
            return Err(LensError::Validation(format!(
                "embedding base URL host {domain} resolves to a non-loopback address ({}); \
                 the Ollama embedder accepts loopback-only addresses",
                addr.ip()
            )));
        }
        pinned_addrs.push(addr);
    }
    if pinned_addrs.is_empty() {
        return Err(LensError::Network(format!(
            "embedding base URL host {domain} did not resolve to any address"
        )));
    }
    Ok(LoopbackTarget {
        host: domain.to_ascii_lowercase(),
        pinned_addrs,
    })
}

/// Validated URL-source locator: parsed URL + guard-approved resolved addrs for DNS-pin.
/// `pinned_addrs` is empty for IP-literal hosts and when `allow_local` is set.
#[derive(Debug)]
struct ValidatedFetchUrl {
    url: url::Url,
    /// Lowercased host string used as the `resolve_to_addrs` key.
    host: String,
    /// Guard-approved resolved addresses to pin reqwest to (empty ⇒ no pinning).
    pinned_addrs: Vec<std::net::SocketAddr>,
}

/// Parses + validates a URL-source locator for SSRF safety.
/// Scheme allowlist → IP guard (every resolved addr checked) → returns pinned addrs to
/// close the DNS-rebind TOCTOU (reqwest connects only to already-validated IPs).
/// `allow_local=true` is a test-only escape hatch for wiremock; the scheme allowlist still applies.
fn validate_fetch_url(locator: &str, allow_local: bool) -> Result<ValidatedFetchUrl, LensError> {
    let parsed = url::Url::parse(locator)
        .map_err(|e| LensError::Validation(format!("invalid URL {locator:?}: {e}")))?;

    let scheme = parsed.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(LensError::Validation(format!(
            "URL source scheme must be http or https, got {scheme:?}"
        )));
    }

    // Use Host enum (not host_str) so IPv6 literals are matched directly without brackets.
    let host = parsed
        .host()
        .ok_or_else(|| LensError::Validation(format!("URL {locator:?} has no host to fetch")))?;
    let host_key = host.to_string().to_ascii_lowercase();
    let port = parsed.port_or_known_default().unwrap_or(80);

    if allow_local {
        return Ok(ValidatedFetchUrl {
            url: parsed,
            host: host_key,
            pinned_addrs: Vec::new(),
        });
    }

    let domain = match host {
        url::Host::Ipv4(v4) => {
            let ip = IpAddr::V4(v4);
            if is_blocked_ip(ip) {
                return Err(LensError::Validation(format!(
                    "URL host {ip} resolves to a blocked address; refusing to fetch"
                )));
            }
            return Ok(ValidatedFetchUrl {
                url: parsed,
                host: host_key,
                pinned_addrs: Vec::new(),
            });
        }
        url::Host::Ipv6(v6) => {
            let ip = IpAddr::V6(v6);
            if is_blocked_ip(ip) {
                return Err(LensError::Validation(format!(
                    "URL host {ip} resolves to a blocked address; refusing to fetch"
                )));
            }
            return Ok(ValidatedFetchUrl {
                url: parsed,
                host: host_key,
                pinned_addrs: Vec::new(),
            });
        }
        url::Host::Domain(d) => d.to_string(),
    };

    let addrs = (domain.as_str(), port)
        .to_socket_addrs()
        .map_err(|e| LensError::Network(format!("failed to resolve URL host {domain}: {e}")))?;
    let mut pinned_addrs = Vec::new();
    for addr in addrs {
        if is_blocked_ip(addr.ip()) {
            return Err(LensError::Validation(format!(
                "URL host {domain} resolves to a blocked address ({}); refusing to fetch",
                addr.ip()
            )));
        }
        pinned_addrs.push(addr);
    }
    if pinned_addrs.is_empty() {
        return Err(LensError::Network(format!(
            "URL host {domain} did not resolve to any address"
        )));
    }
    Ok(ValidatedFetchUrl {
        url: parsed,
        host: host_key,
        pinned_addrs,
    })
}

/// Blocking SSRF gate (scheme allowlist + DNS resolve + [`is_blocked_ip`]) reused by the
/// JS renderer. Discards pinned_addrs — the OS webview performs its own DNS at navigation
/// and has no `resolve_to_addrs` hook; protection is host allow/deny + readback re-check.
pub fn ssrf_check_url(url: &str) -> Result<(), LensError> {
    validate_fetch_url(url, false).map(|_| ())
}

/// Non-blocking SSRF gate for the `on_navigation` event-loop closure: checks IP literals
/// via [`is_blocked_ip`] but performs NO DNS for hostnames (blocking resolve happened at
/// pre-flight; DNS-rebind is caught at readback). Missing host fails closed.
pub fn ssrf_check_host(host: Option<&str>) -> Result<(), LensError> {
    let host =
        host.ok_or_else(|| LensError::Validation("navigation URL has no host".to_string()))?;
    // Strip brackets around an IPv6 literal so it parses as IpAddr ("[::1]" → "::1").
    let bare = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    if let Ok(ip) = bare.parse::<IpAddr>()
        && is_blocked_ip(ip)
    {
        return Err(LensError::Validation(format!(
            "navigation host {host} is a blocked address; refusing to navigate"
        )));
    }
    Ok(())
}

/// Returns `true` iff the final-committed URL's host passes the SSRF policy.
/// Fail-closed: malformed/blocked URLs return `false` so rendered output is discarded.
pub fn readback_host_allowed(final_url: &str) -> bool {
    ssrf_check_url(final_url).is_ok()
}

/// SSRF-guarded, size-bounded, no-redirect HTTP GET.
/// Scheme allowlist + IP guard + DNS pinning (TOCTOU) + Content-Type allowlist + streaming cap.
async fn fetch_url_guarded(
    locator: &str,
    timeout: std::time::Duration,
    max_source_bytes: usize,
) -> Result<Vec<u8>, LensError> {
    fetch_url_guarded_inner(locator, timeout, allow_local_url_fetch(), max_source_bytes).await
}

/// Always `false` in production. The `test-util` feature + `LENS_TEST_ALLOW_LOCAL_URL` env var
/// enable loopback for integration tests that drive the real pipeline against wiremock.
fn allow_local_url_fetch() -> bool {
    #[cfg(feature = "test-util")]
    {
        std::env::var_os("LENS_TEST_ALLOW_LOCAL_URL").is_some()
    }
    #[cfg(not(feature = "test-util"))]
    {
        false
    }
}

/// Inner fetch with the `allow_local` escape hatch. Production always passes `false`.
async fn fetch_url_guarded_inner(
    locator: &str,
    timeout: std::time::Duration,
    allow_local: bool,
    max_source_bytes: usize,
) -> Result<Vec<u8>, LensError> {
    let ValidatedFetchUrl {
        url,
        host,
        pinned_addrs,
    } = validate_fetch_url(locator, allow_local)?;

    let mut builder = reqwest::Client::builder()
        .user_agent(URL_FETCH_USER_AGENT)
        .connect_timeout(timeout)
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::none());

    // Pin reqwest to already-validated IPs (TOCTOU: no second DNS lookup at connect time).
    if !pinned_addrs.is_empty() {
        builder = builder.resolve_to_addrs(&host, &pinned_addrs);
    }

    let client = builder
        .build()
        .map_err(|e| LensError::Network(format!("HTTP client build failed: {e}")))?;

    let resp = client
        .get(url.clone())
        .send()
        .await
        .map_err(|e| LensError::Network(format!("URL fetch failed for {locator}: {e}")))?;

    if resp.status().is_redirection() {
        return Err(LensError::Validation(format!(
            "URL fetch for {locator} returned a redirect (HTTP {}); redirects are not followed",
            resp.status()
        )));
    }
    if !resp.status().is_success() {
        return Err(LensError::Network(format!(
            "URL fetch returned HTTP {} for {locator}",
            resp.status()
        )));
    }

    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_default();
    let ct_ok = URL_ALLOWED_CONTENT_TYPES
        .iter()
        .any(|allowed| content_type.starts_with(allowed));
    if !ct_ok {
        return Err(LensError::Validation(format!(
            "URL source {locator} has unsupported Content-Type {content_type:?}; \
             only HTML/text documents are accepted"
        )));
    }

    if let Some(len) = resp.content_length()
        && len > max_source_bytes as u64
    {
        return Err(LensError::Validation(format!(
            "URL source {locator} declares {len} bytes, exceeding the \
             {max_source_bytes}-byte ingest limit"
        )));
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| {
            LensError::Network(format!("URL fetch body read failed for {locator}: {e}"))
        })?;
        if buf.len() + chunk.len() > max_source_bytes {
            return Err(LensError::Validation(format!(
                "URL source {locator} body exceeds the {max_source_bytes}-byte ingest limit"
            )));
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf)
}

/// Sets a terminal-pending status (`needs_ocr`/`needs_js`/`render_failed`), wiping prior
/// indexed content first (Lance FIRST then SQLite — G5 ordering). Caller returns `Ok(())`
/// so the `Err→error` flip in [`ingest_source`] never fires.
async fn set_terminal_pending(
    repo: &crate::notebooks::NotebookRepo<'_>,
    pool: &sqlx::SqlitePool,
    data_dir: &Path,
    notebook_id: &str,
    source_id: &str,
    status: &str,
) -> Result<(), LensError> {
    wipe_source_content(pool, data_dir, notebook_id, source_id).await?;
    repo.update_source_status(source_id, status).await?;
    Ok(())
}

async fn wipe_source_content(
    pool: &sqlx::SqlitePool,
    data_dir: &Path,
    notebook_id: &str,
    source_id: &str,
) -> Result<(), LensError> {
    let store = LanceVectorStore::new(data_dir, pool.clone());
    let (embed_model, embed_dim, embed_backend) =
        resolve_notebook_embedding_from_pool(pool, notebook_id).await?;
    let coord = crate::vector_store::Coordinate::new(
        notebook_id.to_string(),
        embed_backend,
        embed_model,
        embed_dim,
    );
    store.drop_source(&coord, source_id).await?;
    let mut tx = pool.begin().await?;
    delete_chunks_for_source(&mut tx, source_id).await?;
    tx.commit().await?;
    Ok(())
}

/// Pool-only twin of [`crate::LensEngine::resolve_notebook_embedding`]. NULL columns fall
/// back to the registry/backend default; a missing notebook row fails fast.
async fn resolve_notebook_embedding_from_pool(
    pool: &sqlx::SqlitePool,
    notebook_id: &str,
) -> Result<(String, usize, crate::embedder::EmbeddingBackend), LensError> {
    let row: (Option<String>, Option<String>) =
        sqlx::query_as("SELECT embedding_model, embedding_backend FROM notebooks WHERE id = ?")
            .bind(notebook_id)
            .fetch_optional(pool)
            .await?
            .ok_or_else(|| LensError::Validation(format!("no notebook with id {notebook_id}")))?;
    let (stored_model, stored_backend) = row;
    let spec = crate::embedder::resolve(stored_model.as_deref().unwrap_or(""));
    let backend = crate::embedder::EmbeddingBackend::from_opt_str(stored_backend.as_deref());
    Ok((spec.id.to_string(), spec.dim, backend))
}

/// Deletes all `chunks` rows for `source_id`. Children first (FK defense-in-depth;
/// the schema cascades but ordering keeps the delete safe regardless of enforcement mode).
async fn delete_chunks_for_source(
    conn: &mut sqlx::SqliteConnection,
    source_id: &str,
) -> Result<(), LensError> {
    sqlx::query("DELETE FROM chunks WHERE source_id = ? AND parent_id IS NOT NULL")
        .bind(source_id)
        .execute(&mut *conn)
        .await?;
    sqlx::query("DELETE FROM chunks WHERE source_id = ?")
        .bind(source_id)
        .execute(&mut *conn)
        .await?;
    Ok(())
}

/// Batch size for multi-row `INSERT`. Each row binds 15 variables; 60 × 15 = 900,
/// comfortably under SQLite's 999-variable limit.
const CHUNK_INSERT_BATCH: usize = 60;

/// Inserts chunk rows for `source_id`. Parents first (FK ordering), then children,
/// in multi-row INSERT batches of [`CHUNK_INSERT_BATCH`] inside the caller's transaction.
async fn insert_chunks(
    conn: &mut sqlx::SqliteConnection,
    source_id: &str,
    chunks: &[Chunk],
) -> Result<(), LensError> {
    let now = chrono::Utc::now().to_rfc3339();
    let parents = chunks.iter().filter(|c| c.parent_id.is_none());
    let children = chunks.iter().filter(|c| c.parent_id.is_some());
    let ordered: Vec<&Chunk> = parents.chain(children).collect();

    for batch in ordered.chunks(CHUNK_INSERT_BATCH) {
        insert_chunk_batch(&mut *conn, source_id, batch, &now).await?;
    }
    Ok(())
}

/// Inserts one batch of chunk rows. `enrichment`/`embedding_text` are literal NULLs
/// (populated later by Phase-3). `page` is derived from `SourceAnchor::Pdf` when present.
async fn insert_chunk_batch(
    conn: &mut sqlx::SqliteConnection,
    source_id: &str,
    chunks: &[&Chunk],
    now: &str,
) -> Result<(), LensError> {
    if chunks.is_empty() {
        return Ok(());
    }
    let mut qb = sqlx::QueryBuilder::new(
        "INSERT INTO chunks \
             (id, source_id, parent_id, kind, level, section_path, text, \
              token_start, token_end, page, char_start, char_end, block_type, \
              enrichment, embedding_text, source_anchor, created_at) ",
    );
    qb.push_values(chunks, |mut b, chunk| {
        let page: Option<i64> = chunk
            .source_anchor
            .as_deref()
            .and_then(|json| serde_json::from_str::<crate::extract::SourceAnchor>(json).ok())
            .and_then(|anchor| {
                if let crate::extract::SourceAnchor::Pdf { page, .. } = anchor {
                    Some(page as i64)
                } else {
                    None
                }
            });

        b.push_bind(&chunk.id)
            .push_bind(source_id)
            .push_bind(&chunk.parent_id)
            .push_bind(&chunk.kind)
            .push_bind(chunk.level)
            .push_bind(&chunk.section_path)
            .push_bind(&chunk.text)
            .push_bind(chunk.token_start)
            .push_bind(chunk.token_end)
            .push_bind(page)
            .push_bind(chunk.char_start)
            .push_bind(chunk.char_end)
            .push_bind(&chunk.block_type)
            .push("NULL") // enrichment — reserved for Phase-3
            .push("NULL") // embedding_text — populated by Phase-3 worker
            .push_bind(&chunk.source_anchor)
            .push_bind(now);
    });
    qb.build().execute(&mut *conn).await?;
    Ok(())
}

/// Assigns a JSON-serialized [`SourceAnchor`] to each chunk. `blocks` and `anchors`
/// are index-aligned; unmatched chunks keep `source_anchor = None`.
///
/// - **Audio** (issue #44): a chunk collects every block whose `char_start` falls in
///   `[chunk.char_start, chunk.char_end)` and derives `Audio { min(start), max(end) }`
///   over them, so a multi-segment chunk covers all its segments.
/// - **All other kinds:** the anchor of the last block whose `char_start ≤
///   chunk.char_start` (same dominance rule as `block_type`/`section_path`).
///
/// A `blocks.len() != anchors.len()` mismatch is silent data loss (dropped audio
/// timestamps), so it is a hard error rather than a skip.
fn attach_anchors_to_chunks(
    chunks: &mut [crate::chunk::Chunk],
    blocks: &[crate::parse::Block],
    anchors: &[crate::extract::SourceAnchor],
) -> Result<(), LensError> {
    if anchors.is_empty() || blocks.is_empty() {
        return Ok(());
    }
    if anchors.len() != blocks.len() {
        return Err(LensError::Internal(format!(
            "attach_anchors_to_chunks: anchors.len()={} != blocks.len()={}",
            anchors.len(),
            blocks.len()
        )));
    }
    for chunk in chunks.iter_mut() {
        let cs = chunk.char_start as usize;
        let ce = chunk.char_end as usize;

        // Aggregate the Audio timestamps of every block the chunk textually covers.
        let mut audio_range: Option<(f32, f32)> = None;
        for (b, a) in blocks.iter().zip(anchors.iter()) {
            if b.char_start >= cs
                && b.char_start < ce
                && let crate::extract::SourceAnchor::Audio {
                    start_second,
                    end_second,
                } = a
            {
                audio_range = Some(match audio_range {
                    Some((lo, hi)) => (lo.min(*start_second), hi.max(*end_second)),
                    None => (*start_second, *end_second),
                });
            }
        }

        let anchor = if let Some((start_second, end_second)) = audio_range {
            Some(crate::extract::SourceAnchor::Audio {
                start_second,
                end_second,
            })
        } else {
            // Non-Audio: last block whose char_start ≤ chunk.char_start wins.
            blocks
                .iter()
                .zip(anchors.iter())
                .rev()
                .find(|(b, _)| b.char_start <= cs)
                .map(|(_, a)| a.clone())
        };

        if let Some(a) = anchor {
            chunk.source_anchor = serde_json::to_string(&a).ok();
        }
    }
    Ok(())
}

/// Emits a `model_download` progress event when the tokenizer is absent from disk
/// (a network fetch is about to happen on the next [`LensEngine::tokenizer`] call).
fn maybe_emit_tokenizer_download(data_dir: &Path, on_progress: &mut impl FnMut(IngestProgress)) {
    let fastembed_dir = data_dir.join("models").join("fastembed");
    let canonical = fastembed_dir.join("tokenizer.json");
    if !canonical.is_file() && find_tokenizer_json(&fastembed_dir).is_none() {
        on_progress(IngestProgress::new(ingest_phase::MODEL_DOWNLOAD, 0, None));
    }
}

/// Resolves the nomic `tokenizer.json`: (1) canonical cache path, (2) fastembed subtree,
/// (3) atomic HuggingFace download. Shared by ingest and the eval harness.
pub async fn resolve_nomic_tokenizer(data_dir: &Path) -> Result<Tokenizer, LensError> {
    let fastembed_dir = data_dir.join("models").join("fastembed");
    let canonical = fastembed_dir.join("tokenizer.json");

    if canonical.is_file() {
        return Tokenizer::from_file(&canonical)
            .map_err(|e| LensError::Model(format!("load tokenizer {}: {e}", canonical.display())));
    }

    if let Some(found) = find_tokenizer_json(&fastembed_dir) {
        return Tokenizer::from_file(&found)
            .map_err(|e| LensError::Model(format!("load tokenizer {}: {e}", found.display())));
    }

    download_tokenizer(NOMIC_TOKENIZER_URL, &canonical).await?;
    Tokenizer::from_file(&canonical)
        .map_err(|e| LensError::Model(format!("load tokenizer {}: {e}", canonical.display())))
}

/// Searches `dir` up to 3 levels deep for `tokenizer.json` (fastembed model layout varies).
fn find_tokenizer_json(dir: &Path) -> Option<PathBuf> {
    fn search(dir: &Path, depth: usize) -> Option<PathBuf> {
        if depth == 0 {
            return None;
        }
        let entries = std::fs::read_dir(dir).ok()?;
        let mut subdirs = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.file_name().is_some_and(|n| n == "tokenizer.json") {
                return Some(path);
            }
            if path.is_dir() {
                subdirs.push(path);
            }
        }
        for sub in subdirs {
            if let Some(found) = search(&sub, depth - 1) {
                return Some(found);
            }
        }
        None
    }
    search(dir, 3)
}

/// Downloads `url` to `dest` atomically via a `.part` temp file.
async fn download_tokenizer(url: &str, dest: &Path) -> Result<(), LensError> {
    download_tokenizer_inner(url, dest, TOKENIZER_CONNECT_TIMEOUT, TOKENIZER_TIMEOUT).await
}

async fn download_tokenizer_inner(
    url: &str,
    dest: &Path,
    connect_timeout: std::time::Duration,
    timeout: std::time::Duration,
) -> Result<(), LensError> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| LensError::Io(format!("create {}: {e}", parent.display())))?;
    }
    let client = reqwest::Client::builder()
        .connect_timeout(connect_timeout)
        .timeout(timeout)
        .build()
        .map_err(|e| LensError::Network(format!("tokenizer download client init failed: {e}")))?;
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| LensError::Network(format!("tokenizer download request failed: {e}")))?;
    if !resp.status().is_success() {
        return Err(LensError::Network(format!(
            "tokenizer download failed with status {}",
            resp.status()
        )));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| LensError::Network(format!("tokenizer download stream error: {e}")))?;
    let tmp = dest.with_extension("json.part");
    std::fs::write(&tmp, &bytes)
        .map_err(|e| LensError::Io(format!("write {}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, dest).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        LensError::Io(format!("finalize {}: {e}", dest.display()))
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};

    #[test]
    fn kind_detection_json_vs_jsonl_sniff() {
        assert!(!sniff_is_jsonl(b"{\"a\":1}"));
        assert!(!sniff_is_jsonl(b"{\n  \"a\": 1,\n  \"b\": 2\n}"));
        assert!(!sniff_is_jsonl(b"[1, 2, 3]"));
        assert!(sniff_is_jsonl(b"{\"a\":1}\n{\"b\":2}\n"));
        assert!(sniff_is_jsonl(b"{\"a\":1}\r\n{\"b\":2}\r\n"));
        assert!(!sniff_is_jsonl(&[0xFF, 0xFE]));
    }

    #[test]
    fn structured_kinds_follow_derived_path() {
        use crate::parse::SourceKind;
        for k in [
            SourceKind::Json,
            SourceKind::Jsonl,
            SourceKind::Yaml,
            SourceKind::Xml,
        ] {
            assert!(!k.is_text_like(), "{k:?} must be a derived (non-text) kind");
        }
    }

    #[test]
    fn tabular_kinds_follow_derived_path() {
        use crate::parse::SourceKind;
        for k in [SourceKind::Xlsx, SourceKind::Xls, SourceKind::Csv] {
            assert!(!k.is_text_like(), "{k:?} must be a derived (non-text) kind");
        }
    }

    #[test]
    fn tables_sibling_path_builder() {
        let data_dir = Path::new("/data");
        let p = tables_sibling_path(data_dir, "abc-123");
        assert_eq!(p, Path::new("/data/sources/abc-123.tables.md"));
    }

    #[test]
    fn test_max_source_mb_resolver() {
        assert_eq!(resolve_max_source_bytes(""), 50 * 1024 * 1024);
        assert_eq!(resolve_max_source_bytes("  "), 50 * 1024 * 1024);
        assert_eq!(resolve_max_source_bytes("100"), 100 * 1024 * 1024);
        assert_eq!(resolve_max_source_bytes("1"), 1024 * 1024);
        assert_eq!(resolve_max_source_bytes(" 25 "), 25 * 1024 * 1024);
        assert_eq!(resolve_max_source_bytes("0"), 50 * 1024 * 1024);
        assert_eq!(resolve_max_source_bytes("garbage"), 50 * 1024 * 1024);
        assert_eq!(resolve_max_source_bytes("-5"), 50 * 1024 * 1024);
    }

    #[test]
    fn is_blocked_ip_rejects_loopback_link_local_private_and_metadata() {
        let blocked: &[IpAddr] = &[
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            IpAddr::V4(Ipv4Addr::new(127, 99, 99, 99)),
            IpAddr::V4(Ipv4Addr::UNSPECIFIED),         // 0.0.0.0
            IpAddr::V4(Ipv4Addr::new(169, 254, 0, 1)), // link-local
            IpAddr::V4(Ipv4Addr::new(169, 254, 169, 254)), // cloud metadata
            IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5)),    // RFC1918
            IpAddr::V4(Ipv4Addr::new(172, 16, 0, 1)),
            IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            IpAddr::V4(Ipv4Addr::new(100, 64, 0, 1)), // CGNAT
            IpAddr::V6(Ipv6Addr::LOCALHOST),          // ::1
            IpAddr::V6(Ipv6Addr::UNSPECIFIED),        // ::
            IpAddr::V6(Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 1)), // link-local
            IpAddr::V6(Ipv6Addr::new(0xfc00, 0, 0, 0, 0, 0, 0, 1)), // ULA
            // IPv4-mapped loopback ::ffff:127.0.0.1
            IpAddr::V6(Ipv4Addr::new(127, 0, 0, 1).to_ipv6_mapped()),
            // IPv4-mapped metadata ::ffff:169.254.169.254
            IpAddr::V6(Ipv4Addr::new(169, 254, 169, 254).to_ipv6_mapped()),
        ];
        for ip in blocked {
            assert!(is_blocked_ip(*ip), "expected {ip} to be blocked");
        }
    }

    #[test]
    fn is_blocked_ip_allows_public_addresses() {
        let allowed: &[IpAddr] = &[
            IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
            IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1)),
            IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34)),
            IpAddr::V6(Ipv6Addr::new(
                0x2606, 0x2800, 0x220, 1, 0x248, 0x1893, 0x25c8, 0x1946,
            )),
        ];
        for ip in allowed {
            assert!(!is_blocked_ip(*ip), "expected {ip} to be allowed");
        }
    }

    #[test]
    fn validate_fetch_url_rejects_non_http_scheme() {
        for bad in [
            "file:///etc/passwd",
            "gopher://example.com/",
            "ftp://example.com/x",
            "data:text/plain,hi",
        ] {
            let err =
                validate_fetch_url(bad, false).expect_err("non-http(s) scheme must be rejected");
            assert!(
                matches!(err, LensError::Validation(_)),
                "expected Validation for {bad:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn validate_fetch_url_rejects_loopback_and_private_literals() {
        for bad in [
            "http://127.0.0.1/secret",
            "http://127.0.0.1:8080/secret",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.1/",
            "http://192.168.0.1/admin",
            "http://[::1]/",
            "http://0.0.0.0/",
        ] {
            let err = validate_fetch_url(bad, false)
                .expect_err("loopback/private literal must be rejected");
            assert!(
                matches!(err, LensError::Validation(_)),
                "expected Validation for {bad:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn validate_fetch_url_accepts_public_literal() {
        let ok = validate_fetch_url("http://8.8.8.8/", false).expect("public literal must pass");
        assert_eq!(ok.url.scheme(), "http");
        assert!(
            ok.pinned_addrs.is_empty(),
            "an IP-literal host must not produce pinned addrs (no DNS to rebind)"
        );
        assert_eq!(ok.host, "8.8.8.8");
    }

    #[test]
    fn validate_fetch_url_pins_resolved_addrs_for_hostname() {
        // `localhost` resolves to loopback (BLOCKED); proves every resolved addr is checked.
        let err = validate_fetch_url("http://localhost/", false)
            .expect_err("localhost resolves to loopback and must be rejected");
        assert!(
            matches!(err, LensError::Validation(_)),
            "a hostname resolving to a blocked addr must be a Validation error, got {err:?}"
        );

        let local = validate_fetch_url("http://localhost:8080/x", true)
            .expect("allow_local must bypass the IP guard");
        assert!(
            local.pinned_addrs.is_empty(),
            "allow_local must not pin (reqwest resolves loopback itself)"
        );
        assert_eq!(local.host, "localhost");
    }

    #[test]
    fn ssrf_check_url_rejects_blocked_and_scheme() {
        for bad in [
            "http://169.254.169.254/latest/meta-data/",
            "http://127.0.0.1/secret",
            "http://[::1]/",
            "http://10.0.0.1/",
            "http://192.168.1.1/admin",
            "http://[fe80::1]/",
            "file:///etc/passwd",
            "gopher://example.com/",
            "data:text/plain,hi",
        ] {
            let err = ssrf_check_url(bad).expect_err("blocked/non-http URL must be rejected");
            assert!(
                matches!(err, LensError::Validation(_) | LensError::Network(_)),
                "expected Validation/Network for {bad:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn ssrf_check_url_accepts_public_literal() {
        ssrf_check_url("http://8.8.8.8/").expect("public literal must pass the SSRF gate");
    }

    #[test]
    fn ssrf_check_host_rejects_blocked_ip_literals() {
        for bad in [
            "169.254.169.254",
            "127.0.0.1",
            "::1",
            "10.0.0.1",
            "192.168.1.1",
            "fe80::1",
            "0.0.0.0",
        ] {
            let err =
                ssrf_check_host(Some(bad)).expect_err("a blocked IP-literal host must be rejected");
            assert!(
                matches!(err, LensError::Validation(_)),
                "expected Validation for host {bad:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn ssrf_check_host_allows_public_host_string_without_dns() {
        ssrf_check_host(Some("example.com")).expect("a public hostname must be allowed");
        // `localhost` resolves to loopback but is allowed here (no DNS performed).
        ssrf_check_host(Some("localhost"))
            .expect("a hostname is allowed WITHOUT resolving (proving no DNS is performed)");
    }

    #[test]
    fn ssrf_check_host_rejects_missing_host() {
        let err = ssrf_check_host(None).expect_err("a missing host must fail closed");
        assert!(matches!(err, LensError::Validation(_)), "got {err:?}");
    }

    #[test]
    fn readback_host_allowed_discards_blocked_and_malformed() {
        for blocked in [
            "http://169.254.169.254/latest/meta-data/",
            "http://127.0.0.1/x",
            "http://[::1]/",
            "http://10.0.0.5/",
            "not a url",
            "http:///no-host",
            "file:///etc/passwd",
        ] {
            assert!(
                !readback_host_allowed(blocked),
                "expected {blocked:?} to be discarded (false)"
            );
        }
    }

    #[test]
    fn readback_host_allowed_keeps_public() {
        assert!(
            readback_host_allowed("http://8.8.8.8/page"),
            "a public final host must be kept"
        );
    }

    #[test]
    fn validate_fetch_url_pinning_untouched_by_ssrf_gates() {
        let ok = validate_fetch_url("http://8.8.8.8/", false).expect("public literal passes");
        assert!(
            ok.pinned_addrs.is_empty(),
            "IP-literal must still produce no pins (pinning machinery untouched)"
        );
        assert_eq!(ok.host, "8.8.8.8");
    }

    use wiremock::matchers::{method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // wiremock binds to 127.0.0.1 (IP-blocked); use allow_local=true to test post-guard logic.
    async fn fetch_against_mock(
        url: &str,
        timeout: std::time::Duration,
    ) -> Result<Vec<u8>, LensError> {
        fetch_url_guarded_inner(url, timeout, true, MAX_SOURCE_BYTES).await
    }

    #[tokio::test]
    async fn fetch_rejects_redirect_to_blocked_host() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/redir"))
            .respond_with(
                ResponseTemplate::new(302).insert_header("location", "http://127.0.0.1/secret"),
            )
            .mount(&mock)
            .await;
        let err = fetch_against_mock(
            &format!("{}/redir", mock.uri()),
            std::time::Duration::from_secs(5),
        )
        .await
        .expect_err("a 302 must not be followed");
        assert!(
            matches!(err, LensError::Validation(_)),
            "redirect must surface a Validation error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn fetch_rejects_non_text_content_type() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/binary"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/octet-stream")
                    .set_body_bytes(vec![0u8; 16]),
            )
            .mount(&mock)
            .await;
        let err = fetch_against_mock(
            &format!("{}/binary", mock.uri()),
            std::time::Duration::from_secs(5),
        )
        .await
        .expect_err("octet-stream must be rejected");
        assert!(
            matches!(err, LensError::Validation(_)),
            "non-text content-type must be a Validation error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn fetch_accepts_text_html_content_type() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/page"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/html; charset=utf-8")
                    .set_body_string("<html><body>hi</body></html>"),
            )
            .mount(&mock)
            .await;
        let body = fetch_against_mock(
            &format!("{}/page", mock.uri()),
            std::time::Duration::from_secs(5),
        )
        .await
        .expect("text/html must be accepted");
        assert!(!body.is_empty());
    }

    // wiremock's header() matcher splits on commas; check received_requests directly
    // to assert the full Chrome UA (which contains "(KHTML, like Gecko)").
    #[tokio::test]
    async fn fetch_sends_user_agent_header() {
        assert!(!URL_FETCH_USER_AGENT.is_empty());
        assert!(
            URL_FETCH_USER_AGENT.contains("Chrome/")
                && URL_FETCH_USER_AGENT.starts_with("Mozilla/5.0"),
            "URL_FETCH_USER_AGENT must be a browser-mimicking Chrome UA, got: {URL_FETCH_USER_AGENT}"
        );
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/ua"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/html; charset=utf-8")
                    .set_body_string("<html><body>hi</body></html>"),
            )
            .mount(&mock)
            .await;
        let body = fetch_against_mock(
            &format!("{}/ua", mock.uri()),
            std::time::Duration::from_secs(5),
        )
        .await
        .expect("fetch must succeed");
        assert!(!body.is_empty());

        let reqs = mock
            .received_requests()
            .await
            .expect("mock records requests");
        let sent_ua = reqs
            .iter()
            .find_map(|r| r.headers.get("user-agent"))
            .expect("a User-Agent header was sent");
        assert_eq!(
            sent_ua.to_str().unwrap(),
            URL_FETCH_USER_AGENT,
            "the full browser UA must be sent as a single header value"
        );
    }

    #[tokio::test]
    async fn fetch_rejects_over_cap_body() {
        let big = vec![b'a'; MAX_SOURCE_BYTES + 100];
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/huge"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/html")
                    .set_body_bytes(big),
            )
            .mount(&mock)
            .await;
        let err = fetch_against_mock(
            &format!("{}/huge", mock.uri()),
            std::time::Duration::from_secs(10),
        )
        .await
        .expect_err("over-cap body must be rejected");
        assert!(
            matches!(err, LensError::Validation(_)),
            "over-cap body must be a Validation error, got {err:?}"
        );
    }

    #[test]
    fn streaming_cap_logic_bails_over_threshold() {
        const TINY_CAP: usize = 8;
        let chunks: Vec<Vec<u8>> = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8, 9]];
        let mut buf: Vec<u8> = Vec::new();
        let mut bailed = false;
        for chunk in chunks {
            if buf.len() + chunk.len() > TINY_CAP {
                bailed = true;
                break;
            }
            buf.extend_from_slice(&chunk);
        }
        assert!(
            bailed,
            "accumulator must bail once the running total exceeds the cap"
        );
        assert!(buf.len() <= TINY_CAP);
    }

    #[tokio::test]
    async fn fetch_timeout_fires_on_slow_response() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/slow"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/html")
                    .set_body_string("<html></html>")
                    .set_delay(std::time::Duration::from_secs(10)),
            )
            .mount(&mock)
            .await;
        let started = std::time::Instant::now();
        let err = fetch_against_mock(
            &format!("{}/slow", mock.uri()),
            std::time::Duration::from_millis(300),
        )
        .await
        .expect_err("a slow response must trip the short timeout");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "timeout must fire fast (took {:?})",
            started.elapsed()
        );
        assert!(
            matches!(err, LensError::Network(_)),
            "a timeout is surfaced as a Network error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn tokenizer_download_timeout_fires_on_slow_response() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/tokenizer.json"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string("{}")
                    .set_delay(std::time::Duration::from_secs(10)),
            )
            .mount(&mock)
            .await;
        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("tokenizer.json");
        let started = std::time::Instant::now();
        let err = download_tokenizer_inner(
            &format!("{}/tokenizer.json", mock.uri()),
            &dest,
            std::time::Duration::from_secs(30),
            std::time::Duration::from_millis(300),
        )
        .await
        .expect_err("a slow body must trip the short total timeout");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "timeout must fire fast (took {:?})",
            started.elapsed()
        );
        assert!(
            matches!(err, LensError::Network(_)),
            "a timeout is surfaced as a Network error, got {err:?}"
        );
        assert!(
            !dest.exists(),
            "a timed-out download must not leave a finalized tokenizer file"
        );
        assert!(
            !dest.with_extension("json.part").exists(),
            "a timed-out download must not leave a .part temp file"
        );
    }

    // Pin a bogus hostname to the mock's loopback addr via resolve_to_addrs; proves reqwest
    // uses the pinned addr rather than re-resolving (TOCTOU: connect-time IP = guard-validated IP).
    #[tokio::test]
    async fn resolve_to_addrs_pins_connection_to_validated_addr() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/pinned"))
            .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
            .mount(&mock)
            .await;

        let mock_url = url::Url::parse(&mock.uri()).expect("mock uri parses");
        let mock_addrs: Vec<std::net::SocketAddr> = (
            mock_url.host_str().expect("mock host"),
            mock_url.port().expect("mock port"),
        )
            .to_socket_addrs()
            .expect("resolve mock addr")
            .collect();
        assert!(!mock_addrs.is_empty());

        let bogus_host = "pinned.invalid";
        let client = reqwest::Client::builder()
            .resolve_to_addrs(bogus_host, &mock_addrs)
            .build()
            .expect("client builds");
        let target = format!("http://{bogus_host}:{}/pinned", mock_url.port().unwrap());
        let resp = client
            .get(&target)
            .send()
            .await
            .expect("pinned connection must reach the mock");
        assert!(resp.status().is_success(), "got {}", resp.status());
        let body = resp.text().await.expect("body");
        assert_eq!(body, "ok", "must have hit the pinned mock server");
    }
}
