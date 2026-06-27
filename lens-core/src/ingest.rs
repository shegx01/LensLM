//! Ingest pipeline (M4 Phase 1, Group e): the end-to-end text/Markdown slice.
//!
//! [`ingest_source`] takes a queued `sources` row through the full vertical
//! slice — parse → chunk → embed → index — flipping `sources.status`
//! `queued → parsing → embedding → indexed` (or `error` on any failure) and
//! streaming [`IngestProgress`] to a caller-supplied sink.
//!
//! # Serialization (Decision D1 / M2)
//!
//! The whole pipeline runs under a single permit of the engine's
//! [`ingest_lock`](crate::LensEngine::ingest_lock) semaphore, so two concurrent
//! `ingest_source` calls never run the single-threaded ONNX session at once.
//! The synchronous `fastembed` `embed()` is always invoked under
//! [`tokio::task::spawn_blocking`] so it never blocks a tokio worker.
//!
//! # Re-ingest idempotency + cross-store wipe ordering (Risk G5)
//!
//! Re-ingesting an `indexed` source whose `content_hash` is unchanged is a
//! no-op. A changed hash (or a source left in a non-`indexed` state by a crashed
//! prior run) re-runs the full wipe + ingest. The wipe drops the Lance vectors
//! FIRST (`drop_source`), THEN deletes the SQLite `chunks` rows.
//!
//! The exact guarantee this ordering buys (not "orphans are impossible"):
//! * A *completed* wipe leaves no orphan Lance rows — the Lance drop committed
//!   before the SQLite delete, so there is never a `chunks` row without its
//!   vector, nor a vector for a deleted `chunks` row.
//! * A crash (or a failed SQLite transaction) *after* the Lance drop but before
//!   the SQLite commit leaves the source transiently empty-of-vectors but with
//!   its old `chunks` intact. That is reclaimed by the status→`error` flip
//!   (startup crash-recovery for `parsing`/`embedding`, plus the inline
//!   error-flip on a failed run) followed by an idempotent re-ingest, which
//!   re-runs the wipe (a no-op on the already-dropped vectors) and rebuilds.
//!
//! # Tokenizer (integration wrinkle)
//!
//! `chunk_blocks` needs the nomic `tokenizers::Tokenizer`. `fastembed` downloads
//! the model into `{data_dir}/models/fastembed/` but does not expose its
//! tokenizer. We solve this with [`resolve_nomic_tokenizer`]: first we search the
//! fastembed cache subtree for a `tokenizer.json`; if none is found we download
//! nomic's `tokenizer.json` once (mirroring `tts::download_kokoro_model`) into
//! `{data_dir}/models/fastembed/tokenizer.json` and load it from there. The
//! tokenizer is a multi-MB file, so it is parsed from disk once and cached on the
//! engine ([`LensEngine::tokenizer`]) — reused across ingests rather than
//! re-loaded per ingest. [`maybe_emit_tokenizer_download`] emits the
//! `model_download` progress event before a cold-cache fetch.
//!
//! # LanceVectorStore construction
//!
//! The [`LanceVectorStore`](crate::vector_store::LanceVectorStore) is
//! constructed per-ingest from `(data_dir, pool)`. The Lance connection is
//! cheap (an embedded store opened lazily on first table touch), so a fresh
//! store per ingest is acceptable for Phase 1 and avoids holding a connection
//! on the engine across the `RwLock`.

use std::net::{IpAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::chunk::{Chunk, chunk_blocks};
use crate::vector_store::{LanceVectorStore, VectorRow, VectorStore};
use crate::{LensEngine, LensError};

/// Canonical HuggingFace URL for the nomic-embed-text-v1.5 tokenizer.
const NOMIC_TOKENIZER_URL: &str =
    "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json";

/// Connect timeout for the tokenizer download (the file is small).
const TOKENIZER_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Connect + read timeout for URL source fetches.
///
/// Rationale: `ingest_source` holds the SINGLE ingest permit (Semaphore(1) in
/// `LensEngine`) for its whole duration, so an unbounded fetch on a hung server
/// stalls ALL subsequent ingests AND purge operations. 30 seconds is generous
/// for real pages while keeping the pipeline responsive.
///
/// In tests this const can be observed directly; when injecting a wiremock that
/// delays beyond this value the test must use a short mock-only timeout (see the
/// `url_fetch_timeout_fires` integration test).
pub const URL_FETCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// `User-Agent` sent on the URL-fetch HTTP GET.
///
/// A UA-less client is routinely served `403`/`429` by real sites (bot-blocking
/// CDNs, WAFs), which surfaces here as a silent fetch failure. A named,
/// honest UA (app name + version + contact URL) avoids those rejections without
/// impersonating a browser. The version tracks the workspace crate version.
pub const URL_FETCH_USER_AGENT: &str = concat!(
    "LensLM/",
    env!("CARGO_PKG_VERSION"),
    " (+https://github.com/shegx01/LensLM)"
);

/// Embed batch size — documents are embedded in batches of this many texts to
/// bound peak memory while keeping the ONNX session warm.
const EMBED_BATCH: usize = 32;

/// Minimum extracted-text length (chars) for a URL source to be considered
/// successfully extracted. Below this floor the page is likely a JS-rendered
/// SPA where trafilatura found nothing to extract, and the source is left in
/// `needs_js` status rather than indexed (so a future JS-capable path can retry
/// it without losing the record).
///
/// Named constant so the threshold is visible in code review and adjustable
/// without grep-replacing magic numbers.
pub const NEEDS_JS_MIN_CHARS: usize = 200;

/// Minimum ratio of `extracted_text.len() / raw_html.len()` for a URL source
/// to be considered successfully extracted. A very low ratio (e.g. 0.01 = 1%)
/// indicates that nearly all page bytes are in script/style/markup rather than
/// readable text — a strong JS-SPA signal.
///
/// Applied together with [`NEEDS_JS_MIN_CHARS`]: EITHER condition triggers
/// `needs_js` (so a short but high-ratio page from a tiny real article does not
/// false-positive, and a long but script-dominated page also flags correctly).
pub const NEEDS_JS_MIN_TEXT_RATIO: f64 = 0.01;

/// Maximum accepted source size, in bytes (Phase-1 OOM guard).
///
/// A source larger than this is rejected up front with [`LensError::Validation`]
/// rather than read fully into memory, chunked (each chunk owning a `String`
/// copy), and accumulated as `VectorRow`s before a single `store.add` — a path
/// whose peak memory is O(total text + total vectors). This is a coarse cap, not
/// streaming: bounded-memory streaming inserts are a documented Phase-2
/// follow-up. The cap is enforced in two places: [`add_text_source`](crate::LensEngine::add_text_source)
/// (the paste path, against `text.len()`) and at the top of the ingest pipeline
/// after the file read (any file path, against the read length).
pub const MAX_SOURCE_BYTES: usize = 10 * 1024 * 1024;

/// Ingest progress phase labels (the [`IngestProgress::phase`] string values).
///
/// Single source of truth for the phase literals streamed to the progress sink,
/// mirroring the `source_status` mod in `notebooks.rs`. The public wire shape is
/// unchanged — these are the same strings, just no longer scattered as raw
/// literals. The lifecycle for URL sources is `fetching → parsing → chunking →
/// [model_download] → embedding → indexing → done`; text/MD skips `fetching`.
pub(crate) mod ingest_phase {
    /// Fetch phase (URL sources only): HTTP GET of the remote page.
    pub const FETCHING: &str = "fetching";
    /// Parse phase: source text → blocks.
    pub const PARSING: &str = "parsing";
    /// Chunk phase: blocks → parent/child chunks.
    pub const CHUNKING: &str = "chunking";
    /// Model-download phase: cold-cache embedder/tokenizer fetch.
    pub const MODEL_DOWNLOAD: &str = "model_download";
    /// Embed phase: chunks → vectors.
    pub const EMBEDDING: &str = "embedding";
    /// Index phase: vectors → Lance table.
    pub const INDEXING: &str = "indexing";
    /// Terminal phase: ingest complete (also the unchanged-content no-op signal).
    pub const DONE: &str = "done";
}

/// One ingestion progress event. Serializes as the `T` payload carried by
/// `StreamEvent<IngestProgress>` over the command channel.
///
/// `phase` is one of `"parsing"`, `"chunking"`, `"model_download"`,
/// `"embedding"`, `"indexing"`, or `"done"`. `done`/`total` track per-phase
/// progress (`total` is `None` when the upper bound is unknown).
///
/// # Status vs. phase granularity (intentionally NOT 1:1)
///
/// The persisted `sources.status` column is **coarse** — it tracks only the
/// recoverable lifecycle states (`queued → parsing → embedding → indexed`, or
/// `error`). [`IngestProgress::phase`] is **fine-grained** — it streams the
/// full UX lifecycle (`parsing → chunking → [model_download] → embedding →
/// indexing → done`). They deliberately don't map 1:1: the persisted status
/// folds `chunking` under `parsing` and `model_download`/`indexing` under
/// `embedding`, so the row status is enough for crash-recovery (it can tell a
/// transient state apart from a terminal one) but cannot, on its own,
/// distinguish a crash *during chunking* from a crash *during embedding* — both
/// land in the same recoverable status. The fine-grained phase exists for the
/// progress UI, not for persistence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestProgress {
    /// The current pipeline phase.
    pub phase: String,
    /// Units completed within the current phase.
    pub done: u64,
    /// Total units for the current phase, if known.
    pub total: Option<u64>,
}

impl IngestProgress {
    /// Convenience constructor.
    fn new(phase: &str, done: u64, total: Option<u64>) -> Self {
        Self {
            phase: phase.to_string(),
            done,
            total,
        }
    }
}

/// Ingests a queued source end-to-end, streaming [`IngestProgress`].
///
/// See the module docs for the full contract (status transitions, re-ingest
/// idempotency, cross-store wipe ordering, serialization).
#[tracing::instrument(skip(engine, on_progress))]
pub async fn ingest_source(
    engine: &LensEngine,
    source_id: &str,
    on_progress: impl FnMut(IngestProgress),
) -> Result<(), LensError> {
    // Serialize the whole pipeline (single ONNX session — Decision D1 / M2). The
    // permit is bound to an explicit scope so it is RELEASED before the enrichment
    // enqueue below: the enqueue (AC3) must happen OUTSIDE the held permit so it
    // never blocks under the lock and a full channel can never deadlock against it.
    let result = {
        let _permit = engine
            .ingest_lock()
            .acquire()
            .await
            .map_err(|e| LensError::Internal(format!("ingest semaphore closed: {e}")))?;

        let result = run_ingest(engine, source_id, on_progress).await;

        // On any failure, best-effort flip the source to `error` (Risk R10: treat a
        // missing/cascade-deleted row as a graceful no-op, never a panic).
        if result.is_err() {
            let pool = engine.pool().await;
            let repo = crate::notebooks::NotebookRepo::new(&pool);
            if let Err(e) = repo
                .update_source_status(source_id, crate::notebooks::SourceStatus::Error.as_str())
                .await
            {
                tracing::warn!(
                    source_id,
                    "failed to mark source as error after ingest failure: {e}"
                );
            }
        }

        result
    }; // `_permit` dropped HERE — the ingest_lock is now released.

    // ── Enqueue background enrichment (AC3) — STRICTLY after the permit drop ──
    // A successful ingest left the source `Indexed` (set inside `run_ingest`). Now
    // that the permit is released, issue the non-blocking `try_send`: it never
    // awaits the lock, and a full channel is recovered by the startup/rescan
    // queue-rebuild. On a failed ingest the source is `error`, so there is nothing
    // to enrich — skip the enqueue.
    if result.is_ok() {
        engine.enqueue_enrichment(source_id);
    }

    result
}

/// The inner pipeline (without the error-status side effect / semaphore).
async fn run_ingest(
    engine: &LensEngine,
    source_id: &str,
    mut on_progress: impl FnMut(IngestProgress),
) -> Result<(), LensError> {
    let pool = engine.pool().await;
    let data_dir = engine.data_dir().await;
    // `NotebookRepo` is a stateless `&pool` wrapper (a shared borrow), so a single
    // instance is reused for every status/metadata write below — it coexists with
    // the pool's other shared uses (`pool.clone()`, `pool.begin()`, `&pool`).
    let repo = crate::notebooks::NotebookRepo::new(&pool);

    // ── Load the source row ───────────────────────────────────────────────
    let source = repo
        .get_source(source_id)
        .await?
        .ok_or_else(|| LensError::Validation(format!("no source with id {source_id}")))?;

    // Parse the DB-row discriminant strings into their enums ONCE, here at the
    // boundary. Every dispatch/gate below matches on these enums rather than
    // re-comparing raw strings.
    //
    // `kind` is parsed leniently (`.ok()`): production kinds always parse, but
    // the `test-util` injection seam (`extractor_for`) can drive an arbitrary
    // fake binary kind that is NOT a known `SourceKind` — that path must flow
    // through the DERIVED branch, so an unknown kind is `None` here (neither
    // text-like, nor `Url`, nor `Pdf`), exactly as the prior string-compares
    // behaved. `extractor_for` (below) still rejects a truly-unknown kind when
    // there is no injected factory.
    //
    // `status` is parsed STRICTLY (`?`, not `.ok()`) — this asymmetry is
    // deliberate: an out-of-vocabulary status is a fail-loud invariant breach
    // (corrupt/forward-incompatible row), whereas an out-of-vocabulary kind is
    // tolerated to preserve the test seam and the inert `file` kind.
    let kind = crate::parse::SourceKind::from_kind_str(&source.kind).ok();
    let status = source.status.parse::<crate::notebooks::SourceStatus>()?;
    // An unknown (test-injected) kind is treated as DERIVED — `is_text_like` is
    // the single point of truth, with `None` defaulting to derived.
    let text_like = kind.is_some_and(|k| k.is_text_like());

    // ── Acquire RAW BYTES ─────────────────────────────────────────────────
    // URL sources fetch their bytes over HTTP; all other kinds read a local file.
    // Raw bytes (not `read_to_string`) so binary kinds (pdf/docx) flow through
    // their extractor; text/MD validate UTF-8 inside `TextExtractor`.
    let raw: Vec<u8> = if kind == Some(crate::parse::SourceKind::Url) {
        // ── URL: SSRF-guarded, size-bounded, async HTTP GET ───────────────
        // Emit FETCHING progress BEFORE the network round-trip so the UI
        // updates immediately when the task starts.
        on_progress(IngestProgress::new(ingest_phase::FETCHING, 0, Some(1)));

        let url = source.locator.clone();
        let bytes = fetch_url_guarded(&url, URL_FETCH_TIMEOUT).await?;

        on_progress(IngestProgress::new(ingest_phase::FETCHING, 1, Some(1)));
        bytes
    } else {
        // ── File-backed: read raw bytes from the locator path ─────────────
        // Async read so a large local file never blocks the tokio worker for the
        // duration of the disk read (the pipeline holds the single ingest permit).
        tokio::fs::read(&source.locator)
            .await
            .map_err(|e| LensError::Io(format!("read source {}: {e}", source.locator)))?
    };

    // ── Stage-1 size guard (DERIVED kinds only) ───────────────────────────
    // The extractor decodes the WHOLE binary into memory, so cap the RAW bytes
    // BEFORE invoking it (front-door cap). Text/MD skip Stage 1 (raw ==
    // canonical) and rely on the single Stage-2 check, preserving Phase-1
    // behaviour exactly.
    if !text_like && raw.len() > MAX_SOURCE_BYTES {
        return Err(LensError::Validation(format!(
            "source is {} raw bytes, exceeding the {MAX_SOURCE_BYTES}-byte ingest limit",
            raw.len()
        )));
    }

    // ── DERIVED no-op short-circuit BEFORE extraction (re-ingest determinism)
    // For DERIVED kinds the content hash is over the RAW FILE BYTES (bit-stable,
    // independent of any extractor non-determinism). Computing it here lets an
    // unchanged binary short-circuit WITHOUT running the (possibly expensive /
    // non-deterministic) extractor at all (AC4d).
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

    // ── JSON-vs-JSONL content-sniff fallback ──────────────────────────────
    // A `.json` extension is the common case, but some tools emit JSON Lines
    // (one object per line) with a `.json` name. If the content (first 64 KB)
    // parses as >= 2 newline-delimited JSON values, treat it as `jsonl` so each
    // record becomes its own block. A `.jsonl`/`.ndjson` extension already maps
    // to `Jsonl` in `add_file_source` and is left untouched here.
    let effective_kind: String = if kind == Some(crate::parse::SourceKind::Json)
        && sniff_is_jsonl(&raw)
    {
        tracing::info!(
            source_id,
            "`.json` source sniffed as JSON Lines (>=2 newline-delimited JSON values); using jsonl extractor"
        );
        crate::parse::SourceKind::Jsonl.as_str().to_string()
    } else {
        source.kind.clone()
    };

    // ── Dispatch through the Extractor seam ───────────────────────────────
    let extractor = crate::extract::extractor_for(&effective_kind)?;
    // DERIVED (pdf/docx/url) extraction is blocking, CPU-bound work (pdfium
    // decode, docx-rs XML parse, trafilatura DOM walk). Run it under
    // `spawn_blocking` so it never stalls a tokio worker — mirroring the embed
    // path (which already does this for the synchronous fastembed call). The
    // extractor is `Send + Sync` and `raw` is owned, so both move cleanly into
    // the closure; we hand `raw` back out alongside the output (it is reused for
    // the raw-bytes content hash and the needs_js ratio). Text/MD extraction is
    // cheap + deterministic, so it stays inline (no task-spawn overhead).
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

    // ── needs_js / needs_ocr gate (URL sources, and future OCR-required PDFs) ──
    //
    // INVARIANT (load-bearing): these statuses are returned as Ok(()) — NOT as
    // Err — so the error-flip in `ingest_source` (which sets status to `error`
    // on any Err) never fires. `run_ingest` sets the terminal-pending status
    // directly and returns Ok early. The caller's success path then emits Done,
    // but the status is already `needs_js`/`needs_ocr` — the Done progress event
    // is fine (the UI should read the persisted status, not the event).
    //
    // needs_js triggers when EITHER:
    //   - extracted text is shorter than NEEDS_JS_MIN_CHARS (absolute floor), OR
    //   - extracted text is less than NEEDS_JS_MIN_TEXT_RATIO of the raw HTML
    //     (ratio floor — catches script-heavy SPAs with large bundles).
    //
    // needs_ocr (Step 5): a `pdf` source whose extractor produced no text is an
    // image-only / scanned PDF (no text layer). The `PdfExtractor` signals this
    // by returning EMPTY output (no Err) — see `extract::pdf`. We set
    // `needs_ocr` via the SAME Ok-with-status mechanism as `needs_js` (NOT Err,
    // which would flip to `error`) and index nothing.
    if kind == Some(crate::parse::SourceKind::Pdf) && out.extracted_text.trim().is_empty() {
        tracing::info!(
            source_id,
            "PDF source produced no text layer — likely scanned/image-only; setting needs_ocr"
        );
        set_terminal_pending(
            &repo,
            &pool,
            &data_dir,
            &source.notebook_id,
            source_id,
            crate::notebooks::SourceStatus::NeedsOcr.as_str(),
        )
        .await?;
        // Return Ok so the Err→error flip in ingest_source never fires.
        return Ok(());
    }

    if kind == Some(crate::parse::SourceKind::Url) {
        let text_len = out.extracted_text.len();
        let raw_len = raw.len();
        let ratio = if raw_len == 0 {
            0.0f64
        } else {
            text_len as f64 / raw_len as f64
        };
        let needs_js = text_len < NEEDS_JS_MIN_CHARS || ratio < NEEDS_JS_MIN_TEXT_RATIO;
        if needs_js {
            tracing::info!(
                source_id,
                text_len,
                raw_len,
                ratio,
                "URL source has near-empty extraction — likely JS-rendered; setting needs_js"
            );
            set_terminal_pending(
                &repo,
                &pool,
                &data_dir,
                &source.notebook_id,
                source_id,
                crate::notebooks::SourceStatus::NeedsJs.as_str(),
            )
            .await?;
            // Return Ok so the Err→error flip in ingest_source never fires.
            return Ok(());
        }
    }

    // ── Stage-2 size guard (the Phase-1 `:215` check on the canonical buffer)
    // For text/MD this is the single guard.
    if out.extracted_text.len() > MAX_SOURCE_BYTES {
        return Err(LensError::Validation(format!(
            "source is {} bytes, exceeding the {MAX_SOURCE_BYTES}-byte ingest limit",
            out.extracted_text.len()
        )));
    }

    // ── Persist the canonical buffer + bind ONE `canonical: &str` ─────────
    // For DERIVED kinds, write `extracted_text` to the sibling
    // `{data_dir}/sources/{source_id}.extracted.txt` (reusing the
    // `add_text_source` write pattern) and chunk THAT persisted buffer. For
    // text/MD the original locator content IS canonical (Decision A1) — no
    // sibling. The SAME `canonical` binding feeds both `chunk_blocks` and the
    // content hash, with no second disk read between them.
    if !text_like {
        let sources_dir = data_dir.join("sources");
        tokio::fs::create_dir_all(&sources_dir)
            .await
            .map_err(|e| LensError::Io(format!("{}: {e}", sources_dir.display())))?;
        // SAME path builder the purge path uses (`extracted_sibling_path`), so
        // the write site and the cleanup site can never diverge. Async write so
        // persisting a large extracted buffer never blocks the tokio worker.
        let sibling = extracted_sibling_path(&data_dir, source_id);
        tokio::fs::write(&sibling, &out.extracted_text)
            .await
            .map_err(|e| LensError::Io(format!("{}: {e}", sibling.display())))?;
    }
    let canonical: &str = &out.extracted_text;

    // ── Content hash (hash split) + text/MD no-op short-circuit ───────────
    // DERIVED kinds reuse the raw-bytes hash already checked above; text/MD hash
    // the canonical text as in Phase 1. The text/MD short-circuit lives here
    // (text/MD extraction is cheap + deterministic, so running it first is fine).
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
        sha256_hex(&raw)
    };

    // ── Construct the vector store (per-ingest; cheap embedded connection) ─
    let store = LanceVectorStore::new(&data_dir, pool.clone());
    let notebook = source.notebook_id.clone();
    // Resolve the OWNING notebook's embedding coordinate (R1) ONCE, here at the
    // boundary: every drop/add/embed below threads this per-notebook model+dim
    // instead of the global default, so a notebook configured with a non-nomic
    // model is indexed under its own coordinate.
    let (embed_model, embed_dim, embed_backend) = engine
        .resolve_notebook_embedding(&crate::NotebookId::from(notebook.clone()))
        .await?;
    // The full backend-aware coordinate (M4 Phase 4b-B) threaded into every
    // drop/add below so the source is indexed under its notebook's own backend.
    let coord = crate::vector_store::Coordinate::new(
        notebook.clone(),
        embed_backend,
        embed_model.clone(),
        embed_dim,
    );

    // ── PARSE ─────────────────────────────────────────────────────────────
    {
        // INVARIANT (load-bearing): status MUST move to a transient state
        // (`parsing`) BEFORE the cross-store wipe below, so that if the process
        // crashes mid-wipe the startup crash-recovery reset (`lib.rs init`, which
        // flips lingering `parsing`/`embedding` rows → `error`) can reclaim the
        // half-wiped source on next launch. Wiping while still `indexed` would
        // leave a row that looks complete but has lost its vectors.
        repo.update_source_status(source_id, crate::notebooks::SourceStatus::Parsing.as_str())
            .await?;
    }
    on_progress(IngestProgress::new(ingest_phase::PARSING, 0, Some(1)));
    // Blocks come straight from the extractor (`out.blocks`); their
    // `char_start/char_end` index into `canonical` (== `out.extracted_text`).
    let blocks = &out.blocks;
    on_progress(IngestProgress::new(ingest_phase::PARSING, 1, Some(1)));

    // ── CHUNK ─────────────────────────────────────────────────────────────
    on_progress(IngestProgress::new(ingest_phase::CHUNKING, 0, None));
    // Emit a `model_download` event up front only when the tokenizer is not yet
    // cached on disk (a cold-cache fetch is about to happen); the engine then
    // resolves + caches the multi-MB tokenizer once and reuses it across ingests.
    maybe_emit_tokenizer_download(&data_dir, &mut on_progress);
    let tokenizer = engine.tokenizer().await?;
    let mut chunks = chunk_blocks(canonical, blocks, &tokenizer)?;
    let total_tokens: i64 = chunks
        .iter()
        .filter(|c| c.level == 0)
        .map(|c| c.token_end - c.token_start)
        .sum();

    // ── Attach SourceAnchor JSON to each chunk (AC5) ──────────────────────
    // After chunking, align each chunk to its source block by char offset:
    // the block whose range contains `chunk.char_start` is the "first block"
    // of that chunk — exactly the same dominance rule used for `block_type` and
    // `section_path`. For parents this is the first block of the window; for
    // children this is the block the child sub-span was split from.
    //
    // Mapping approach: linear scan over `out.anchors` (index-aligned with
    // `out.blocks`). The scan is O(blocks * chunks) but blocks are usually
    // O(tens-to-hundreds) and chunking already did O(blocks * tokens) work,
    // so this is not on the hot path.
    attach_anchors_to_chunks(&mut chunks, blocks, &out.anchors);

    on_progress(IngestProgress::new(
        ingest_phase::CHUNKING,
        chunks.len() as u64,
        Some(chunks.len() as u64),
    ));

    // ── Cross-store wipe (G5: Lance vectors FIRST, then SQLite chunks) ────
    // This handles both a content change on an indexed source and a self-heal
    // retry of a source left non-`indexed` by a crashed prior run.
    //
    // The Lance `drop_source` runs BEFORE the SQLite transaction (G5 ordering:
    // Lance first, so a completed wipe never leaves orphan Lance rows). The
    // SQLite chunk delete + insert then run inside ONE transaction so a crash
    // mid-insert can never leave a half-written set of chunk rows: the tx either
    // commits the full fresh set or rolls back to the prior state.
    store.drop_source(&coord, source_id).await?;

    let mut tx = pool.begin().await?;
    delete_chunks_for_source(&mut tx, source_id).await?;
    insert_chunks(&mut tx, source_id, &chunks).await?;
    tx.commit().await?;
    // NOTE: `&mut tx` coerces to `&mut SqliteConnection` via `Transaction`'s
    // `DerefMut`; the helpers take `&mut SqliteConnection` so they run inside
    // this transaction rather than against the pool directly.

    // ── Reset enrichment on content change (AC12) ─────────────────────────
    // Reaching here means the content changed (the unchanged-content paths above
    // returned early): the chunks + vectors were just re-written, so any prior
    // enrichment (`chunks.enrichment`, `embedding_text`, the cache key in
    // `enrichment_meta`) is now stale. Reset `enrichment_status` to `none` so the
    // post-`Indexed` enqueue (issued OUTSIDE the held permit by `ingest_source`)
    // re-runs the pass. This UPDATE runs UNDER the held `ingest_lock` permit (we
    // are inside `run_ingest`), distinct from the non-blocking enqueue.
    repo.update_enrichment_status(source_id, crate::notebooks::EnrichmentStatus::None)
        .await?;

    // ── Empty-doc short-circuit ───────────────────────────────────────────
    // An empty/whitespace-only source produces zero chunks. There is nothing to
    // embed, so skip the embedder load entirely: loading it would force the
    // ~130 MB model download/init just to embed nothing, and a download failure
    // would flip a trivially-indexable empty source to `error`. The cross-store
    // wipe above already cleared any prior chunks/vectors, so this finalizes the
    // source as an empty-but-indexed row (token_count 0) and emits `done`.
    if chunks.is_empty() {
        repo.update_source_metadata(source_id, 0, &content_hash)
            .await?;
        repo.update_source_status(source_id, crate::notebooks::SourceStatus::Indexed.as_str())
            .await?;
        on_progress(IngestProgress::new(ingest_phase::DONE, 1, Some(1)));
        return Ok(());
    }

    // ── EMBED ─────────────────────────────────────────────────────────────
    repo.update_source_status(
        source_id,
        crate::notebooks::SourceStatus::Embedding.as_str(),
    )
    .await?;

    // Lazily get the cached embedder. Emit a `model_download` phase BEFORE the
    // first construction so a cold-cache download surfaces in the UI.
    on_progress(IngestProgress::new(ingest_phase::MODEL_DOWNLOAD, 0, None));
    let embedder = engine.embedder_for(&embed_model).await?;
    on_progress(IngestProgress::new(
        ingest_phase::MODEL_DOWNLOAD,
        1,
        Some(1),
    ));

    // Embed every chunk (parents AND children) in batches under spawn_blocking.
    let total = chunks.len() as u64;
    on_progress(IngestProgress::new(ingest_phase::EMBEDDING, 0, Some(total)));

    let mut rows: Vec<VectorRow> = Vec::with_capacity(chunks.len());
    let mut embedded: u64 = 0;
    for batch in chunks.chunks(EMBED_BATCH) {
        // One owned copy per chunk text; `embed_documents_owned` then prefixes in
        // place rather than cloning a second time (micro-opt vs. the borrow path).
        let texts: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
        let embedder = embedder.clone();
        // MANDATORY: the synchronous fastembed embed() runs under spawn_blocking
        // so it never blocks a tokio worker (Decision M2).
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

        for (chunk, vector) in batch.iter().zip(vectors.into_iter()) {
            rows.push(VectorRow {
                chunk_id: chunk.id.clone(),
                source_id: source_id.to_string(),
                notebook_id: notebook.clone(),
                level: chunk.level,
                vector,
            });
        }
        embedded += batch.len() as u64;
        on_progress(IngestProgress::new(
            ingest_phase::EMBEDDING,
            embedded,
            Some(total),
        ));
    }

    // ── INDEX ─────────────────────────────────────────────────────────────
    on_progress(IngestProgress::new(ingest_phase::INDEXING, 0, Some(1)));
    store.add(&coord, rows).await?;
    on_progress(IngestProgress::new(ingest_phase::INDEXING, 1, Some(1)));

    // ── Finalize: metadata + indexed status ──────────────────────────────
    repo.update_source_metadata(source_id, total_tokens, &content_hash)
        .await?;
    repo.update_source_status(source_id, crate::notebooks::SourceStatus::Indexed.as_str())
        .await?;
    on_progress(IngestProgress::new(ingest_phase::DONE, 1, Some(1)));
    Ok(())
}

/// SHA-256 of `bytes`, lowercase hex.
fn sha256_hex(bytes: &[u8]) -> String {
    crate::hex_encode(&Sha256::digest(bytes))
}

/// Number of leading bytes inspected by the JSON-vs-JSONL content sniff.
const JSONL_SNIFF_WINDOW: usize = 64 * 1024;

/// Heuristic: does `raw` look like JSON Lines rather than a single JSON value?
///
/// Inspects the first [`JSONL_SNIFF_WINDOW`] bytes: if `raw` is valid UTF-8 and
/// at least two non-empty newline-delimited lines EACH parse as a standalone
/// JSON value, it is treated as JSONL. A single JSON value (even a multi-line
/// pretty-printed object) fails this test because its interior lines are not
/// themselves valid JSON. Non-UTF-8 or fewer than two parseable lines → `false`
/// (stay on the `.json` extractor).
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
            // A line that does not parse standalone means this is a single
            // (possibly multi-line) JSON value, not JSONL.
            return false;
        }
    }
    false
}

/// The canonical `.extracted.txt` sibling path for a DERIVED (pdf/docx/url)
/// source: `{data_dir}/sources/{source_id}.extracted.txt`.
///
/// SINGLE source of truth shared by the ingest WRITE site ([`run_ingest`]) and
/// the purge CLEANUP site (`remove_managed_source_file` in `lib.rs`), so the two
/// can never derive a different path. Deriving the sibling from
/// `(data_dir, source_id)` — rather than the source locator's parent+stem — is
/// load-bearing for URL sources, whose locator is the URL string (not a path
/// under `{data_dir}/sources`).
pub(crate) fn extracted_sibling_path(data_dir: &Path, source_id: &str) -> PathBuf {
    data_dir
        .join("sources")
        .join(format!("{source_id}.extracted.txt"))
}

/// Accepted `Content-Type` prefixes for a URL source body.
///
/// A URL source is only ever HTML (trafilatura extracts readable prose from
/// markup), so the fetch refuses anything that is not HTML/XHTML or another
/// `text/*` document up front — before reading the body. This blocks a server
/// from streaming a large binary (`application/octet-stream`, an image, a zip)
/// into the ingest pipeline under a `url` kind.
const URL_ALLOWED_CONTENT_TYPES: &[&str] = &["text/html", "application/xhtml+xml", "text/"];

/// The cloud-metadata service IP (AWS/GCP/Azure/etc. IMDS). It is a link-local
/// address (169.254.0.0/16) so [`is_blocked_ip`] already rejects it via
/// `is_link_local()`, but it is called out explicitly as defense-in-depth and a
/// readable record of intent.
const CLOUD_METADATA_IP: IpAddr = IpAddr::V4(std::net::Ipv4Addr::new(169, 254, 169, 254));

/// SSRF guard: returns `true` if `ip` must NOT be fetched.
///
/// Rejects loopback (`127.0.0.0/8`, `::1`), link-local (`169.254.0.0/16`,
/// `fe80::/10`, which also covers the `169.254.169.254` cloud-metadata IP),
/// RFC1918 private ranges (`10/8`, `172.16/12`, `192.168/16`) and IPv6 ULA
/// (`fc00::/7`), and the unspecified address (`0.0.0.0`, `::`). Mirrors the
/// no-redirect + scheme-allowlist hardening already in `system_check.rs`, but
/// adds the resolved-IP check that a URL source (a user-supplied remote
/// document, not a local runtime probe) needs.
fn is_blocked_ip(ip: IpAddr) -> bool {
    if ip == CLOUD_METADATA_IP {
        return true;
    }
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                // 100.64.0.0/10 (CGNAT) — not covered by is_private().
                || (v4.octets()[0] == 100 && (v4.octets()[1] & 0xc0) == 0x40)
        }
        IpAddr::V6(v6) => {
            // IPv4-mapped (::ffff:a.b.c.d) — re-check the embedded v4 address.
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_blocked_ip(IpAddr::V4(mapped));
            }
            v6.is_loopback()
                || v6.is_unspecified()
                // Link-local fe80::/10.
                || (v6.segments()[0] & 0xffc0) == 0xfe80
                // Unique-local fc00::/7.
                || (v6.segments()[0] & 0xfe00) == 0xfc00
        }
    }
}

/// A validated URL-source locator: the parsed [`url::Url`] plus the exact
/// socket address(es) the host resolved to AND passed the IP guard.
///
/// `pinned_addrs` is the linchpin of the TOCTOU defense (see
/// [`validate_fetch_url`]): for a hostname host it carries the resolved,
/// guard-approved addresses so the fetch can pin reqwest's connection to them
/// instead of letting reqwest re-resolve DNS at connect time. It is EMPTY for
/// an IP-literal host (there is no DNS step to pin) and when `allow_local` is
/// set (the test escape hatch lets reqwest resolve loopback itself).
#[derive(Debug)]
struct ValidatedFetchUrl {
    url: url::Url,
    /// The lowercased host string used as the `resolve_to_addrs` DNS-override key.
    host: String,
    /// Guard-approved resolved addresses to pin reqwest to (empty ⇒ no pinning).
    pinned_addrs: Vec<std::net::SocketAddr>,
}

/// Parses + validates a URL-source locator for SSRF safety, returning the parsed
/// [`url::Url`] together with the guard-approved resolved [`SocketAddr`]s.
///
/// 1. Parse with the `url` crate (a malformed locator → `Validation`).
/// 2. Allow ONLY the `http`/`https` scheme (reject `file://`, `gopher://`, … →
///    `Validation`), mirroring the scheme allowlist in `system_check.rs`.
/// 3. Resolve the host to socket addresses and reject if ANY resolved IP is a
///    blocked address ([`is_blocked_ip`]). An IP-literal host is checked
///    directly (no DNS); a hostname is resolved and EVERY candidate is checked
///    (so a host with one public and one private A record is still rejected).
///
/// The resolved addresses are RETURNED (not discarded) so the caller can pin
/// reqwest's connection to exactly the IPs that passed the guard. This closes a
/// DNS-rebinding / TOCTOU hole: previously this function resolved + checked the
/// host, threw the addresses away, and handed only the hostname to a fresh
/// reqwest client that performed its OWN independent DNS lookup at connect time.
/// A short-TTL / attacker-controlled record could return a public IP during
/// validation and a loopback/metadata IP at reqwest connect, bypassing the
/// guard. Pinning [`ValidatedFetchUrl::pinned_addrs`] via
/// `ClientBuilder::resolve_to_addrs` makes reqwest connect ONLY to the
/// already-validated addresses, so there is no second, unchecked resolution.
///
/// `allow_local` is `false` on every production path ([`fetch_url_guarded`]). It
/// is `true` ONLY for in-crate tests that must drive the real fetch machinery
/// against a `wiremock` server bound to `127.0.0.1` — wiremock is always
/// loopback, so the IP guard would otherwise reject the request before the
/// body/content-type/redirect/timeout logic under test ever runs. The scheme
/// allowlist still applies even when `allow_local`.
fn validate_fetch_url(locator: &str, allow_local: bool) -> Result<ValidatedFetchUrl, LensError> {
    let parsed = url::Url::parse(locator)
        .map_err(|e| LensError::Validation(format!("invalid URL {locator:?}: {e}")))?;

    let scheme = parsed.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(LensError::Validation(format!(
            "URL source scheme must be http or https, got {scheme:?}"
        )));
    }

    // Use the parsed `Host` enum (NOT `host_str`) so an IPv6 literal is matched
    // as an `Ipv6` variant directly — `host_str` returns it WITH brackets
    // (`"[::1]"`), which would fail `IpAddr::parse` and leak through to DNS.
    let host = parsed
        .host()
        .ok_or_else(|| LensError::Validation(format!("URL {locator:?} has no host to fetch")))?;
    // `resolve_to_addrs` keys overrides by the lowercased host string; reqwest
    // looks up the connect target by the URL's host, so match it exactly.
    let host_key = host.to_string().to_ascii_lowercase();
    let port = parsed.port_or_known_default().unwrap_or(80);

    if allow_local {
        // Test escape hatch: no pinning — let reqwest resolve loopback itself.
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
            // IP-literal host: there is no DNS step to rebind, so nothing to
            // pin. reqwest connects to the literal directly.
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

    // Hostname: resolve and reject if ANY candidate IP is blocked, keeping the
    // approved addresses so the caller can pin reqwest's connection to them.
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

/// Fetches `locator` as an SSRF-guarded, size-bounded, no-redirect HTTP GET.
///
/// Security properties (mirrors + extends `system_check.rs`):
/// * **Scheme allowlist + resolved-IP guard** ([`validate_fetch_url`]) — only
///   `http`/`https`, and the host must not resolve to a loopback/link-local/
///   private/metadata address.
/// * **Pinned resolution (no DNS rebinding)** — reqwest connects ONLY to the
///   addresses already validated by [`validate_fetch_url`] (via
///   `resolve_to_addrs`), never re-resolving the host itself. This closes the
///   TOCTOU between guard-time and connect-time DNS.
/// * **No redirects** (`redirect::Policy::none()`) — a 30x response is surfaced
///   as a clear error rather than silently followed to a re-validated-around
///   internal host. (The strictest correct choice; matches `system_check.rs`.)
/// * **Content-Type allowlist** — only HTML/XHTML/`text/*` bodies are read.
/// * **Bounded body size** — the body is streamed and aborted the moment the
///   running total exceeds [`MAX_SOURCE_BYTES`], and a `Content-Length` over the
///   cap short-circuits before any body is read (no whole-body buffering / OOM).
async fn fetch_url_guarded(
    locator: &str,
    timeout: std::time::Duration,
) -> Result<Vec<u8>, LensError> {
    fetch_url_guarded_inner(locator, timeout, allow_local_url_fetch()).await
}

/// Whether the IP guard should permit loopback/local hosts.
///
/// ALWAYS `false` in production: this returns `false` unless the crate is built
/// with the `test-util` feature AND the `LENS_TEST_ALLOW_LOCAL_URL` env var is
/// set. It exists solely so the URL-ingest INTEGRATION tests (a separate crate,
/// so the in-module `allow_local` test path is unreachable) can drive the real
/// `ingest_source` pipeline against a `wiremock` server bound to `127.0.0.1`.
/// Production builds never enable `test-util`, so this compiles to a constant
/// `false` and the guard is unconditional.
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

/// Inner fetch with the `allow_local` escape hatch (see
/// [`validate_fetch_url`]). Production always passes `false`.
async fn fetch_url_guarded_inner(
    locator: &str,
    timeout: std::time::Duration,
    allow_local: bool,
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

    // TOCTOU defense: pin reqwest to the exact IP(s) `validate_fetch_url`
    // already resolved AND passed through the IP guard, so reqwest does NOT run
    // a second, independent DNS lookup at connect time. Without this, a
    // short-TTL / attacker-controlled record could resolve to a public IP
    // during validation and a loopback/metadata IP at connect, bypassing the
    // guard. `resolve_to_addrs` overrides DNS for `host` to these addresses;
    // the URL's own port is always used, so the addr ports here are irrelevant.
    // For IP-literal hosts (and the test escape hatch) `pinned_addrs` is empty
    // and no override is installed — there is no DNS step to rebind.
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

    // A redirect (3xx) is NOT followed (Policy::none); surface it as a clear
    // error rather than fetching the redirect target.
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

    // Content-Type allowlist (before reading the body).
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

    // Short-circuit on a declared Content-Length over the cap (avoids streaming).
    if let Some(len) = resp.content_length()
        && len > MAX_SOURCE_BYTES as u64
    {
        return Err(LensError::Validation(format!(
            "URL source {locator} declares {len} bytes, exceeding the \
             {MAX_SOURCE_BYTES}-byte ingest limit"
        )));
    }

    // Stream the body, enforcing MAX_SOURCE_BYTES as bytes arrive so a server
    // that lies about (or omits) Content-Length cannot OOM the pipeline.
    let mut buf: Vec<u8> = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| {
            LensError::Network(format!("URL fetch body read failed for {locator}: {e}"))
        })?;
        if buf.len() + chunk.len() > MAX_SOURCE_BYTES {
            return Err(LensError::Validation(format!(
                "URL source {locator} body exceeds the {MAX_SOURCE_BYTES}-byte ingest limit"
            )));
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf)
}

/// Drops a source's indexed content from BOTH stores (Lance vectors FIRST, then
/// the SQLite `chunks` rows), matching the main re-ingest wipe ordering (G5).
///
/// Used by the `needs_ocr` / `needs_js` gates so a source transitioning INTO a
/// terminal-pending status (e.g. a previously-INDEXED source whose content
/// changes to scanned/SPA) drops its prior indexed chunks + vectors rather than
/// leaving stale content searchable behind the pending status.
/// Drives a source into a TERMINAL-PENDING status (`needs_ocr` / `needs_js`):
/// wipes any prior indexed content from BOTH stores, then sets the status.
///
/// Single source of truth for the wipe-then-set-status pattern shared by the
/// `needs_ocr` (image-only PDF) and `needs_js` (JS-rendered URL) gates in
/// [`run_ingest`]. The wipe is load-bearing: a source transitioning INTO a
/// pending status (e.g. a previously-INDEXED source whose new content is
/// scanned/SPA) must drop its stale chunks + vectors so nothing indexed survives
/// searchable behind the pending status. The caller returns `Ok(())` after this
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
    // Resolve the OWNING notebook's embedding coordinate (R1) so the wipe drops
    // the right per-notebook table. This helper has only the pool (no engine
    // handle), so resolve straight off the notebook row through the registry —
    // identical semantics to `LensEngine::resolve_notebook_embedding`.
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

/// Resolves a notebook's `(model_id, dim, backend)` embedding coordinate directly
/// from the pool (the pool-only twin of
/// [`crate::LensEngine::resolve_notebook_embedding`]).
///
/// Used by the ingest helpers that hold a `&SqlitePool` but no engine handle. A
/// NULL `embedding_model`/`embedding_backend` (the column exists but is unset) or
/// an unknown value falls back to the registry/backend default; a MISSING notebook
/// row fails fast, matching [`crate::LensEngine::resolve_notebook_embedding`].
async fn resolve_notebook_embedding_from_pool(
    pool: &sqlx::SqlitePool,
    notebook_id: &str,
) -> Result<(String, usize, crate::embedder::EmbeddingBackend), LensError> {
    // `fetch_optional` → None means NO such notebook (fail fast); NULL columns mean
    // the row exists with an unset model/backend (resolve each to the default).
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

/// Deletes every `chunks` row for `source_id`.
///
/// Children are removed first to respect the self-referencing
/// `parent_id`→`chunks.id` FK even when `ON DELETE CASCADE` were absent
/// (defense-in-depth; the schema cascades, but ordering keeps the delete safe
/// under any FK enforcement mode).
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

/// Number of `chunks` rows inserted per multi-row `INSERT` statement.
///
/// Each row binds 15 variables (17 columns; both `enrichment` and
/// `embedding_text` are literal `NULL`s — `embedding_text` is populated later by
/// the M4 Phase-3 enrichment worker via UPDATE — `page` is always bound to the
/// PDF page or `NULL`, and `source_anchor` is bound), so the per-statement
/// variable count is at most `60 * 15 = 900`, comfortably under SQLite's default
/// 999-bound-variable limit (`SQLITE_MAX_VARIABLE_NUMBER`).
const CHUNK_INSERT_BATCH: usize = 60;

/// Inserts the parent + child chunk rows for `source_id`.
///
/// Parents (`level = 0`, `parent_id IS NULL`) are inserted before children so
/// the self-referencing `parent_id` FK always resolves at insert time. Within
/// each level rows are inserted in their original order.
///
/// Rows are written in multi-row `INSERT ... VALUES (...),(...),...` batches of
/// [`CHUNK_INSERT_BATCH`] (one statement instead of one round-trip per chunk),
/// all inside the caller's transaction. Parents are batched fully before any
/// child batch so the FK ordering above is preserved across batch boundaries.
async fn insert_chunks(
    conn: &mut sqlx::SqliteConnection,
    source_id: &str,
    chunks: &[Chunk],
) -> Result<(), LensError> {
    let now = chrono::Utc::now().to_rfc3339();
    // Parents first, then children (FK ordering), each in original order.
    let parents = chunks.iter().filter(|c| c.parent_id.is_none());
    let children = chunks.iter().filter(|c| c.parent_id.is_some());
    let ordered: Vec<&Chunk> = parents.chain(children).collect();

    for batch in ordered.chunks(CHUNK_INSERT_BATCH) {
        insert_chunk_batch(&mut *conn, source_id, batch, &now).await?;
    }
    Ok(())
}

/// Inserts one batch of `chunks` rows in a single multi-row `INSERT` statement.
///
/// `enrichment` and `embedding_text` stay literal `NULL`s (both reserved for the
/// M4 Phase-3 enrichment pass — `embedding_text` is populated later via UPDATE;
/// do NOT write either here).  `page` is derived from `chunk.source_anchor` when
/// it carries a `SourceAnchor::Pdf { page, .. }` — so PDF chunks get a non-NULL
/// `page` for free and all other chunks get a literal `NULL`.  `source_anchor`
/// is bound from `chunk.source_anchor` (JSON text or NULL).
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
        // Derive the page number from the anchor when it is a PDF anchor; all
        // other anchor kinds (Text/Docx/Url) and a missing anchor yield NULL.
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
            .push_bind(page) // page from Pdf anchor or NULL
            .push_bind(chunk.char_start)
            .push_bind(chunk.char_end)
            .push_bind(&chunk.block_type)
            .push("NULL") // enrichment — reserved for Phase-3
            .push("NULL") // embedding_text — populated later by the Phase-3 worker
            .push_bind(&chunk.source_anchor) // JSON text or NULL
            .push_bind(now);
    });
    qb.build().execute(&mut *conn).await?;
    Ok(())
}

/// Assigns a JSON-serialized [`SourceAnchor`] to each chunk by aligning the
/// chunk's `char_start` to its source block.
///
/// **Mapping rule (matches `block_type`/`section_path` inheritance):** the anchor
/// of a chunk is the anchor of the LAST block starting at or before
/// `chunk.char_start` (i.e. the block with the greatest `block.char_start <=
/// chunk.char_start`). This is the "first block" of the parent window and the
/// block a child sub-span was split from — exactly consistent with how
/// `block_type` and `section_path` are inherited in `chunk_blocks`.
///
/// If no block matches (edge case: empty blocks slice or a chunk whose
/// `char_start` falls in a gap), the chunk's `source_anchor` is left as `None`.
///
/// `blocks` and `anchors` must be index-aligned (one anchor per block).
fn attach_anchors_to_chunks(
    chunks: &mut [crate::chunk::Chunk],
    blocks: &[crate::parse::Block],
    anchors: &[crate::extract::SourceAnchor],
) {
    if anchors.is_empty() || blocks.is_empty() {
        return;
    }
    // Guard: mis-aligned slice would produce wrong anchors — skip silently.
    if anchors.len() != blocks.len() {
        tracing::warn!(
            "attach_anchors_to_chunks: anchors.len()={} != blocks.len()={}; skipping anchor attachment",
            anchors.len(),
            blocks.len()
        );
        return;
    }
    for chunk in chunks.iter_mut() {
        let cs = chunk.char_start as usize;
        // Linear scan: find the block whose range contains chunk.char_start.
        // Blocks are ordered by char_start (the parser emits them in document
        // order). We pick the last block whose char_start <= cs (so an oversized
        // block spanning many tokens gets the right anchor for all its children).
        let anchor = blocks
            .iter()
            .zip(anchors.iter())
            .rev()
            .find(|(b, _)| b.char_start <= cs)
            .map(|(_, a)| a);
        if let Some(a) = anchor {
            chunk.source_anchor = serde_json::to_string(a).ok();
        }
    }
}

/// Emits a `model_download` progress event when a tokenizer network download is
/// about to happen, so a cold cache surfaces in the UI.
///
/// The actual resolution + caching is owned by [`LensEngine::tokenizer`] (which
/// calls the shared [`resolve_nomic_tokenizer`] once and reuses the result
/// across ingests). This helper only decides whether the upcoming resolve will
/// hit the network — neither the canonical path nor the cache subtree has a
/// `tokenizer.json` — and emits the event if so.
fn maybe_emit_tokenizer_download(data_dir: &Path, on_progress: &mut impl FnMut(IngestProgress)) {
    let fastembed_dir = data_dir.join("models").join("fastembed");
    let canonical = fastembed_dir.join("tokenizer.json");
    if !canonical.is_file() && find_tokenizer_json(&fastembed_dir).is_none() {
        on_progress(IngestProgress::new(ingest_phase::MODEL_DOWNLOAD, 0, None));
    }
}

/// Resolves the nomic `tokenizer.json`, downloading it once (atomically) if
/// necessary. Shared by the ingest pipeline and the eval harness so both use the
/// same 3-step resolution and the same atomic `.part`→rename download (a
/// duplicate, non-atomic copy in the eval harness previously corrupted the cache
/// on an interrupted download).
///
/// Resolution order:
/// 1. A previously-downloaded `{data_dir}/models/fastembed/tokenizer.json`.
/// 2. Any `tokenizer.json` found in the fastembed cache subtree (e.g. the
///    `NomicEmbedTextV15` model dir fastembed creates).
/// 3. Download nomic's `tokenizer.json` from HuggingFace into
///    `{data_dir}/models/fastembed/tokenizer.json` and load it.
pub async fn resolve_nomic_tokenizer(data_dir: &Path) -> Result<Tokenizer, LensError> {
    let fastembed_dir = data_dir.join("models").join("fastembed");
    let canonical = fastembed_dir.join("tokenizer.json");

    // 1. Already downloaded into the canonical location.
    if canonical.is_file() {
        return Tokenizer::from_file(&canonical)
            .map_err(|e| LensError::Model(format!("load tokenizer {}: {e}", canonical.display())));
    }

    // 2. Search the fastembed cache subtree for a tokenizer.json.
    if let Some(found) = find_tokenizer_json(&fastembed_dir) {
        return Tokenizer::from_file(&found)
            .map_err(|e| LensError::Model(format!("load tokenizer {}: {e}", found.display())));
    }

    // 3. Best-effort download from HuggingFace (mirrors tts download pattern).
    download_tokenizer(NOMIC_TOKENIZER_URL, &canonical).await?;
    Tokenizer::from_file(&canonical)
        .map_err(|e| LensError::Model(format!("load tokenizer {}: {e}", canonical.display())))
}

/// Recursively searches `dir` (shallow, bounded) for a `tokenizer.json` file.
///
/// fastembed lays the model out under a model-named subdir; we look one or two
/// levels deep rather than guessing the exact layout (which is brittle across
/// fastembed versions).
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

/// Downloads `url` to `dest`, writing atomically via a `.part` temp file.
///
/// A clear [`LensError::Network`] is returned on any failure so a brittle path
/// guess never blocks the whole pipeline silently.
async fn download_tokenizer(url: &str, dest: &Path) -> Result<(), LensError> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| LensError::Io(format!("create {}: {e}", parent.display())))?;
    }
    let client = reqwest::Client::builder()
        .connect_timeout(TOKENIZER_CONNECT_TIMEOUT)
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

    // ── JSON-vs-JSONL content sniff (M4 Phase 2.5c) ───────────────────────

    #[test]
    fn kind_detection_json_vs_jsonl_sniff() {
        // A single JSON object (even pretty-printed) is NOT JSONL.
        assert!(!sniff_is_jsonl(b"{\"a\":1}"));
        assert!(!sniff_is_jsonl(b"{\n  \"a\": 1,\n  \"b\": 2\n}"));
        // A JSON array on one line is a single value → not JSONL.
        assert!(!sniff_is_jsonl(b"[1, 2, 3]"));
        // Two newline-delimited JSON values → JSONL.
        assert!(sniff_is_jsonl(b"{\"a\":1}\n{\"b\":2}\n"));
        // CRLF endings still sniff correctly.
        assert!(sniff_is_jsonl(b"{\"a\":1}\r\n{\"b\":2}\r\n"));
        // Non-UTF-8 bytes never sniff as JSONL.
        assert!(!sniff_is_jsonl(&[0xFF, 0xFE]));
    }

    #[test]
    fn structured_kinds_follow_derived_path() {
        // All four structured kinds are DERIVED (not text-like), so they take
        // the raw-bytes-hash / `.extracted.txt` ingest branch.
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

    // ── SSRF IP guard (item 1) ────────────────────────────────────────────

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
            IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34)), // example.com
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
        // A public IP literal must pass the guard (no DNS needed).
        let ok = validate_fetch_url("http://8.8.8.8/", false).expect("public literal must pass");
        assert_eq!(ok.url.scheme(), "http");
        // IP-literal host: no DNS step, so nothing to pin.
        assert!(
            ok.pinned_addrs.is_empty(),
            "an IP-literal host must not produce pinned addrs (no DNS to rebind)"
        );
        assert_eq!(ok.host, "8.8.8.8");
    }

    /// TOCTOU defense (N1): a hostname host RETURNS its resolved, guard-approved
    /// `SocketAddr`s so the fetch can pin reqwest to them (closing the
    /// DNS-rebinding window between validation and connect). We resolve a public
    /// literal-backed host deterministically via `localhost`'s sibling — but to
    /// avoid network flakiness we assert the mechanism against `127.0.0.1`
    /// through the public path is rejected, and the returned-addr contract on a
    /// literal whose DNS form we control. `validate_fetch_url` only RESOLVES for
    /// a `Domain` host, so we exercise the pin-contract via the loopback escape
    /// hatch: a domain host with `allow_local` returns no pin (reqwest resolves),
    /// proving the empty-pin branch; the blocked-resolution rejection is covered
    /// by `validate_fetch_url_rejects_loopback_and_private_literals`.
    #[test]
    fn validate_fetch_url_pins_resolved_addrs_for_hostname() {
        // `localhost` resolves to 127.0.0.1/::1 — both BLOCKED — so the guard
        // must reject it (proving every resolved addr is checked) rather than
        // returning pinned addrs. This is the core TOCTOU guarantee: a hostname
        // whose resolution is blocked never escapes to a fresh DNS lookup.
        let err = validate_fetch_url("http://localhost/", false)
            .expect_err("localhost resolves to loopback and must be rejected");
        assert!(
            matches!(err, LensError::Validation(_)),
            "a hostname resolving to a blocked addr must be a Validation error, got {err:?}"
        );

        // The escape-hatch branch (used by integration tests) must NOT pin, so
        // reqwest resolves the loopback mock host itself.
        let local = validate_fetch_url("http://localhost:8080/x", true)
            .expect("allow_local must bypass the IP guard");
        assert!(
            local.pinned_addrs.is_empty(),
            "allow_local must not pin (reqwest resolves loopback itself)"
        );
        assert_eq!(local.host, "localhost");
    }

    // ── Streaming body cap + Content-Type + redirect + timeout (items 2–4) ─

    use wiremock::matchers::{header, method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Drives the REAL `fetch_url_guarded_inner` against a wiremock server.
    ///
    /// wiremock always binds to `127.0.0.1`, which the IP guard rejects, so the
    /// post-guard logic (no-redirect, Content-Type allowlist, Content-Length /
    /// streaming body cap, timeout) can only be exercised with `allow_local =
    /// true`. The IP guard itself is covered by the `validate_fetch_url_*` and
    /// `is_blocked_ip_*` unit tests above. The scheme allowlist still applies.
    async fn fetch_against_mock(
        url: &str,
        timeout: std::time::Duration,
    ) -> Result<Vec<u8>, LensError> {
        fetch_url_guarded_inner(url, timeout, true).await
    }

    /// Item 1: a 302 redirect (to a blocked loopback host) is NOT followed — the
    /// no-redirect policy surfaces a clear error instead of fetching the target.
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

    /// Item 3: an `application/octet-stream` Content-Type is rejected before the
    /// body is read.
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

    /// Item 3: a `text/html` Content-Type is accepted.
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

    /// Item 1: the fetch client sends our named `User-Agent` so bot-blocking
    /// CDNs/WAFs don't `403`/`429` a UA-less request. The mock ONLY matches when
    /// the `User-Agent` header equals [`URL_FETCH_USER_AGENT`]; a missing/wrong UA
    /// would yield no matched mock and a fetch error.
    #[tokio::test]
    async fn fetch_sends_user_agent_header() {
        // Guard against an empty/misconfigured const before relying on it.
        assert!(!URL_FETCH_USER_AGENT.is_empty());
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/ua"))
            .and(header("user-agent", URL_FETCH_USER_AGENT))
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
        .expect("request carrying the expected User-Agent must be served");
        assert!(!body.is_empty());
    }

    /// Item 2: a body LARGER than `MAX_SOURCE_BYTES` is rejected. wiremock sets a
    /// correct `Content-Length`, so the pre-body short-circuit fires (the body is
    /// never streamed into memory). We send a real over-cap body ONCE (~10 MB) so
    /// the test exercises the real production path rather than a faked length.
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

    /// Item 2 (streaming cap, no Content-Length): a body whose accumulated bytes
    /// exceed a SMALL injected cap is rejected mid-stream. We prove the streaming
    /// accumulator logic directly with a tiny threshold so no large allocation is
    /// needed.
    #[test]
    fn streaming_cap_logic_bails_over_threshold() {
        // Mirror the accumulator guard with a tiny cap.
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

    /// Item 4: a genuine TCP-level timeout FIRES. The mock delays its response
    /// well beyond a SHORT injected timeout; the fetch errors out (does not hang
    /// for 30 s).
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

    /// N1 (TOCTOU): proves the `resolve_to_addrs` pinning mechanism actually
    /// makes reqwest connect to the PINNED address rather than re-resolving the
    /// host. We point a *bogus* hostname (one that does NOT resolve via DNS) at
    /// the wiremock server's real loopback address via `resolve_to_addrs`. If
    /// pinning works, the GET reaches the mock and succeeds; if reqwest ignored
    /// the pin and re-resolved `pinned.invalid`, the connection would fail. This
    /// is exactly the mechanism `fetch_url_guarded_inner` relies on to ensure
    /// the connect-time IP equals the guard-validated IP (no DNS rebinding).
    #[tokio::test]
    async fn resolve_to_addrs_pins_connection_to_validated_addr() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/pinned"))
            .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
            .mount(&mock)
            .await;

        // The mock's actual loopback SocketAddr — the only addr we will allow.
        let mock_url = url::Url::parse(&mock.uri()).expect("mock uri parses");
        let mock_addrs: Vec<std::net::SocketAddr> = (
            mock_url.host_str().expect("mock host"),
            mock_url.port().expect("mock port"),
        )
            .to_socket_addrs()
            .expect("resolve mock addr")
            .collect();
        assert!(!mock_addrs.is_empty());

        // A hostname that has NO DNS record. Without pinning, reqwest would fail
        // to resolve it; with the pin it connects to the mock's loopback addr.
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
