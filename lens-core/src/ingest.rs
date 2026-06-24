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
//! FIRST (`drop_source`), THEN deletes the SQLite `chunks` rows — so a completed
//! wipe can never leave orphan Lance rows, and a retry converges cleanly.
//!
//! # Tokenizer (integration wrinkle)
//!
//! `chunk_blocks` needs the nomic `tokenizers::Tokenizer`. `fastembed` downloads
//! the model into `{data_dir}/models/fastembed/` but does not expose its
//! tokenizer. We solve this with [`load_nomic_tokenizer`]: first we search the
//! fastembed cache subtree for a `tokenizer.json`; if none is found we download
//! nomic's `tokenizer.json` once (mirroring `tts::download_kokoro_model`) into
//! `{data_dir}/models/fastembed/tokenizer.json` and load it from there. The
//! tokenizer is small (a few MB), so loading it per-ingest is acceptable for
//! Phase 1.
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

/// One ingestion progress event. Serializes as the `T` payload carried by
/// `StreamEvent<IngestProgress>` over the command channel.
///
/// `phase` is one of `"parsing"`, `"chunking"`, `"model_download"`,
/// `"embedding"`, `"indexing"`, or `"done"`. `done`/`total` track per-phase
/// progress (`total` is `None` when the upper bound is unknown).
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
        on_progress(IngestProgress::new("done", 1, Some(1)));
        return Ok(());
    }

    // ── Construct the vector store (per-ingest; cheap embedded connection) ─
    let store = LanceVectorStore::new(&data_dir, pool.clone());
    let notebook = source.notebook_id.clone();

    // ── PARSE ─────────────────────────────────────────────────────────────
    {
        let repo = crate::notebooks::NotebookRepo::new(&pool);
        repo.update_source_status(source_id, crate::notebooks::source_status::PARSING)
            .await?;
    }
    on_progress(IngestProgress::new("parsing", 0, Some(1)));
    let blocks = parse_blocks(&text, kind);
    on_progress(IngestProgress::new("parsing", 1, Some(1)));

    // ── CHUNK ─────────────────────────────────────────────────────────────
    on_progress(IngestProgress::new("chunking", 0, None));
    let tokenizer = load_nomic_tokenizer(&data_dir, &mut on_progress).await?;
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
    on_progress(IngestProgress::new("model_download", 0, None));
    let embedder = engine.embedder().await?;
    on_progress(IngestProgress::new("model_download", 1, Some(1)));

    // Embed every chunk (parents AND children) in batches under spawn_blocking.
    let total = chunks.len() as u64;
    on_progress(IngestProgress::new("embedding", 0, Some(total)));

    let mut rows: Vec<VectorRow> = Vec::with_capacity(chunks.len());
    let mut embedded: u64 = 0;
    for batch in chunks.chunks(EMBED_BATCH) {
        let texts: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
        let embedder = embedder.clone();
        // MANDATORY: the synchronous fastembed embed() runs under spawn_blocking
        // so it never blocks a tokio worker (Decision M2).
        let vectors = tokio::task::spawn_blocking(move || {
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            embedder.embed_documents(&refs)
        })
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
        on_progress(IngestProgress::new("embedding", embedded, Some(total)));
    }

    // ── INDEX ─────────────────────────────────────────────────────────────
    on_progress(IngestProgress::new("indexing", 0, Some(1)));
    store
        .add(&notebook, EMBED_MODEL_ID, EMBED_DIM, rows)
        .await?;
    on_progress(IngestProgress::new("indexing", 1, Some(1)));

    // ── Finalize: metadata + indexed status ──────────────────────────────
    {
        let repo = crate::notebooks::NotebookRepo::new(&pool);
        repo.update_source_metadata(source_id, total_tokens, &content_hash)
            .await?;
        repo.update_source_status(source_id, crate::notebooks::source_status::INDEXED)
            .await?;
    }
    on_progress(IngestProgress::new("done", 1, Some(1)));
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

/// Inserts the parent + child chunk rows for `source_id`.
///
/// Parents (`level = 0`, `parent_id IS NULL`) are inserted before children so
/// the self-referencing `parent_id` FK always resolves at insert time.
async fn insert_chunks(
    conn: &mut sqlx::SqliteConnection,
    source_id: &str,
    chunks: &[Chunk],
) -> Result<(), LensError> {
    let now = chrono::Utc::now().to_rfc3339();
    // Parents first, then children (FK ordering).
    for chunk in chunks.iter().filter(|c| c.parent_id.is_none()) {
        insert_one_chunk(&mut *conn, source_id, chunk, &now).await?;
    }
    for chunk in chunks.iter().filter(|c| c.parent_id.is_some()) {
        insert_one_chunk(&mut *conn, source_id, chunk, &now).await?;
    }
    Ok(())
}

/// Inserts a single `chunks` row. `page` and `enrichment` stay NULL in Phase 1.
async fn insert_one_chunk(
    conn: &mut sqlx::SqliteConnection,
    source_id: &str,
    chunk: &Chunk,
    now: &str,
) -> Result<(), LensError> {
    sqlx::query(
        "INSERT INTO chunks \
             (id, source_id, parent_id, kind, level, section_path, text, \
              token_start, token_end, page, char_start, char_end, block_type, enrichment, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, NULL, ?)",
    )
    .bind(&chunk.id)
    .bind(source_id)
    .bind(&chunk.parent_id)
    .bind(&chunk.kind)
    .bind(chunk.level)
    .bind(&chunk.section_path)
    .bind(&chunk.text)
    .bind(chunk.token_start)
    .bind(chunk.token_end)
    .bind(chunk.char_start)
    .bind(chunk.char_end)
    .bind(&chunk.block_type)
    .bind(now)
    .execute(&mut *conn)
    .await?;
    Ok(())
}

/// Loads the nomic `tokenizer.json`, downloading it once if necessary, emitting
/// a `model_download` progress event before a network download so a cold cache
/// surfaces in the UI.
///
/// This is the ingest-pipeline wrapper around [`resolve_nomic_tokenizer`]: it
/// adds the progress event around the resolution. Both share the single resolver
/// below (Resolution order + atomic-rename download).
async fn load_nomic_tokenizer(
    data_dir: &Path,
    on_progress: &mut impl FnMut(IngestProgress),
) -> Result<Tokenizer, LensError> {
    let fastembed_dir = data_dir.join("models").join("fastembed");
    let canonical = fastembed_dir.join("tokenizer.json");
    // Only emit a `model_download` event when a network download is actually
    // about to happen (neither the canonical path nor the cache subtree has one).
    if !canonical.is_file() && find_tokenizer_json(&fastembed_dir).is_none() {
        on_progress(IngestProgress::new("model_download", 0, None));
    }
    resolve_nomic_tokenizer(data_dir).await
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
