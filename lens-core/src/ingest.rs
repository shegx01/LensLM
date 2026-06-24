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

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::chunk::{Chunk, chunk_blocks};
use crate::embedder::{EMBED_DIM, EMBED_MODEL_ID};
use crate::parse::{SourceKind, parse_blocks};
use crate::vector_store::{LanceVectorStore, VectorRow, VectorStore};
use crate::{LensEngine, LensError};

/// Canonical HuggingFace URL for the nomic-embed-text-v1.5 tokenizer.
const NOMIC_TOKENIZER_URL: &str =
    "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json";

/// Connect timeout for the tokenizer download (the file is small).
const TOKENIZER_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Embed batch size — documents are embedded in batches of this many texts to
/// bound peak memory while keeping the ONNX session warm.
const EMBED_BATCH: usize = 32;

/// Ingest progress phase labels (the [`IngestProgress::phase`] string values).
///
/// Single source of truth for the phase literals streamed to the progress sink,
/// mirroring the `source_status` mod in `notebooks.rs`. The public wire shape is
/// unchanged — these are the same strings, just no longer scattered as raw
/// literals. The lifecycle is `parsing → chunking → [model_download] →
/// embedding → indexing → done`.
pub(crate) mod ingest_phase {
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
    // Serialize the whole pipeline (single ONNX session — Decision D1 / M2).
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
            .update_source_status(source_id, crate::notebooks::source_status::ERROR)
            .await
        {
            tracing::warn!(
                source_id,
                "failed to mark source as error after ingest failure: {e}"
            );
        }
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

    // ── Load the source row + its file ────────────────────────────────────
    let source = {
        let repo = crate::notebooks::NotebookRepo::new(&pool);
        repo.get_source(source_id)
            .await?
            .ok_or_else(|| LensError::Validation(format!("no source with id {source_id}")))?
    };
    let kind = SourceKind::from_kind_str(&source.kind)?;
    let text = std::fs::read_to_string(&source.locator)
        .map_err(|e| LensError::Io(format!("read source {}: {e}", source.locator)))?;

    // ── Compute content hash + short-circuit unchanged indexed sources ────
    let content_hash = sha256_hex(text.as_bytes());
    if source.status == crate::notebooks::source_status::INDEXED
        && source.content_hash.as_deref() == Some(content_hash.as_str())
    {
        tracing::info!(
            source_id,
            "source already indexed with unchanged content; no-op"
        );
        on_progress(IngestProgress::new(ingest_phase::DONE, 1, Some(1)));
        return Ok(());
    }

    // ── Construct the vector store (per-ingest; cheap embedded connection) ─
    let store = LanceVectorStore::new(&data_dir, pool.clone());
    let notebook = source.notebook_id.clone();

    // ── PARSE ─────────────────────────────────────────────────────────────
    {
        let repo = crate::notebooks::NotebookRepo::new(&pool);
        // INVARIANT (load-bearing): status MUST move to a transient state
        // (`parsing`) BEFORE the cross-store wipe below, so that if the process
        // crashes mid-wipe the startup crash-recovery reset (`lib.rs init`, which
        // flips lingering `parsing`/`embedding` rows → `error`) can reclaim the
        // half-wiped source on next launch. Wiping while still `indexed` would
        // leave a row that looks complete but has lost its vectors.
        repo.update_source_status(source_id, crate::notebooks::source_status::PARSING)
            .await?;
    }
    on_progress(IngestProgress::new(ingest_phase::PARSING, 0, Some(1)));
    let blocks = parse_blocks(&text, kind);
    on_progress(IngestProgress::new(ingest_phase::PARSING, 1, Some(1)));

    // ── CHUNK ─────────────────────────────────────────────────────────────
    on_progress(IngestProgress::new(ingest_phase::CHUNKING, 0, None));
    // Emit a `model_download` event up front only when the tokenizer is not yet
    // cached on disk (a cold-cache fetch is about to happen); the engine then
    // resolves + caches the multi-MB tokenizer once and reuses it across ingests.
    maybe_emit_tokenizer_download(&data_dir, &mut on_progress);
    let tokenizer = engine.tokenizer().await?;
    let chunks = chunk_blocks(&text, &blocks, &tokenizer)?;
    let total_tokens: i64 = chunks
        .iter()
        .filter(|c| c.level == 0)
        .map(|c| c.token_end - c.token_start)
        .sum();
    on_progress(IngestProgress::new(
        "chunking",
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
    store
        .drop_source(&notebook, EMBED_MODEL_ID, EMBED_DIM, source_id)
        .await?;

    let mut tx = pool.begin().await?;
    delete_chunks_for_source(&mut tx, source_id).await?;
    insert_chunks(&mut tx, source_id, &chunks).await?;
    tx.commit().await?;
    // NOTE: `&mut tx` coerces to `&mut SqliteConnection` via `Transaction`'s
    // `DerefMut`; the helpers take `&mut SqliteConnection` so they run inside
    // this transaction rather than against the pool directly.

    // ── EMBED ─────────────────────────────────────────────────────────────
    {
        let repo = crate::notebooks::NotebookRepo::new(&pool);
        repo.update_source_status(source_id, crate::notebooks::source_status::EMBEDDING)
            .await?;
    }

    // Lazily get the cached embedder. Emit a `model_download` phase BEFORE the
    // first construction so a cold-cache download surfaces in the UI.
    on_progress(IngestProgress::new(ingest_phase::MODEL_DOWNLOAD, 0, None));
    let embedder = engine.embedder().await?;
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
    store
        .add(&notebook, EMBED_MODEL_ID, EMBED_DIM, rows)
        .await?;
    on_progress(IngestProgress::new(ingest_phase::INDEXING, 1, Some(1)));

    // ── Finalize: metadata + indexed status ──────────────────────────────
    {
        let repo = crate::notebooks::NotebookRepo::new(&pool);
        repo.update_source_metadata(source_id, total_tokens, &content_hash)
            .await?;
        repo.update_source_status(source_id, crate::notebooks::source_status::INDEXED)
            .await?;
    }
    on_progress(IngestProgress::new(ingest_phase::DONE, 1, Some(1)));
    Ok(())
}

/// SHA-256 of `bytes`, lowercase hex.
fn sha256_hex(bytes: &[u8]) -> String {
    crate::hex_encode(&Sha256::digest(bytes))
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
/// Each row binds 13 variables (`page` and `enrichment` are literal `NULL`, not
/// bound), so the per-statement variable count is `60 * 13 = 780`, comfortably
/// under SQLite's default 999-bound-variable limit (`SQLITE_MAX_VARIABLE_NUMBER`).
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
/// `page` and `enrichment` stay NULL in Phase 1 (emitted as SQL literals so they
/// don't consume bound-variable budget). The remaining 13 columns are bound per
/// row via [`sqlx::QueryBuilder::push_values`].
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
              token_start, token_end, page, char_start, char_end, block_type, enrichment, created_at) ",
    );
    qb.push_values(chunks, |mut b, chunk| {
        b.push_bind(&chunk.id)
            .push_bind(source_id)
            .push_bind(&chunk.parent_id)
            .push_bind(&chunk.kind)
            .push_bind(chunk.level)
            .push_bind(&chunk.section_path)
            .push_bind(&chunk.text)
            .push_bind(chunk.token_start)
            .push_bind(chunk.token_end)
            .push("NULL") // page
            .push_bind(chunk.char_start)
            .push_bind(chunk.char_end)
            .push_bind(&chunk.block_type)
            .push("NULL") // enrichment
            .push_bind(now);
    });
    qb.build().execute(&mut *conn).await?;
    Ok(())
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
