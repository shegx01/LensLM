//! Shared test support for the ingest integration suites (`ingest.rs`,
//! `url_ingest.rs`).
//!
//! These helpers were byte-duplicated across the two integration-test binaries;
//! they are consolidated here and pulled in via `mod support;` in each test file.
//! Only the genuinely shared pieces live here — format-specific fixture builders
//! (PDF/DOCX writers, the `test-seam` fake extractors) stay in the test files that
//! own them, where their dev-dependency/feature coupling belongs.
//!
//! As an included module (not its own test binary) some helpers are used by only
//! one of the two suites; `#[allow(dead_code)]` keeps the module warning-clean
//! regardless of which binary compiles it.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use lens_core::LensEngine;
use lens_core::embedder::{CountingEmbedder, Embedder};
use tempfile::TempDir;
use tokenizers::Tokenizer;
use tokio::sync::OnceCell;

// ---------------------------------------------------------------------------
// Engine construction
// ---------------------------------------------------------------------------

/// Builds a file-backed engine over a fresh temp dir. Ingest tests need a
/// file-backed engine (text sources are written under `{data_dir}/sources/`),
/// not the in-memory `for_test()`. The tokenizer cache is seeded here so an ingest
/// on this engine never reaches for the network — every caller gets that for free
/// rather than each test remembering to ask.
pub async fn file_engine() -> (TempDir, LensEngine) {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = LensEngine::init(dir.path()).await.expect("engine init");
    seed_tokenizer(dir.path()).await;
    (dir, engine)
}

/// Injects a `CountingEmbedder` into an existing engine so the embedder never
/// downloads the ~130 MB model. The engine's `OnceCell` is pre-filled, so every
/// ingest reuses this one embedder.
pub fn inject_fake_embedder(engine: &LensEngine) {
    let load_count = Arc::new(AtomicUsize::new(0));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let e: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new(load_count, in_flight));
    engine
        .set_embedder_for_test(e, lens_core::EmbeddingBackend::Fastembed)
        .expect("embedder not yet initialized");
}

/// Builds a file-backed engine with an injected `CountingEmbedder` so ingest
/// tests avoid the 130 MB model (they still need the tokenizer for chunking).
pub async fn inject_counting_engine() -> (TempDir, LensEngine) {
    let (dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);
    (dir, engine)
}

// ---------------------------------------------------------------------------
// Tokenizer seeding / availability
// ---------------------------------------------------------------------------

/// Deliberately the path an engine built from `AppConfig::default()` resolves:
/// its `data_dir` is empty, so the cache root is CWD-relative and such an engine
/// reads this exact file.
fn shared_cache_path() -> PathBuf {
    PathBuf::from("models")
        .join("fastembed")
        .join("tokenizer.json")
}

static SHARED_TOKENIZER: OnceCell<Option<PathBuf>> = OnceCell::const_new();

/// Resolves the nomic tokenizer ONCE per test binary: `NOMIC_TOKENIZER_PATH`, else
/// the on-disk cache, else a single download. Serializing the acquisition is the
/// point — every tokenizer-dependent test used to fetch its own copy in parallel
/// and the provider cut the concurrent streams.
pub async fn shared_tokenizer() -> Option<PathBuf> {
    SHARED_TOKENIZER
        .get_or_init(|| async {
            if let Ok(path) = std::env::var("NOMIC_TOKENIZER_PATH") {
                let path = PathBuf::from(path);
                if path.is_file() {
                    return Some(path);
                }
            }
            let cached = shared_cache_path();
            if cached.is_file() {
                return Some(cached);
            }
            download_shared_tokenizer(&cached).await
        })
        .await
        .clone()
}

/// Published by rename so a killed run cannot leave a truncated tokenizer that
/// every later run would happily load.
async fn download_shared_tokenizer(dest: &Path) -> Option<PathBuf> {
    const URL: &str =
        "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json";
    std::fs::create_dir_all(dest.parent()?).ok()?;
    let bytes = reqwest::get(URL).await.ok()?.bytes().await.ok()?;
    let tmp = dest.with_extension("part");
    std::fs::write(&tmp, &bytes).ok()?;
    std::fs::rename(&tmp, dest).ok()?;
    Some(dest.to_path_buf())
}

/// Copies the shared tokenizer into `data_dir`'s fastembed cache so an ingest there
/// issues no download. No-op when none could be acquired (offline, no cache), which
/// leaves [`tokenizer_available`] to skip the test.
pub async fn seed_tokenizer(data_dir: &Path) {
    let Some(src) = shared_tokenizer().await else {
        return;
    };
    let dest = data_dir
        .join("models")
        .join("fastembed")
        .join("tokenizer.json");
    if let Some(parent) = dest.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::copy(&src, &dest);
}

/// Loads the tokenizer the ingest pipeline would use, seeding `data_dir` on the way
/// so a later ingest there stays offline. `None` when offline with no cache.
pub async fn tokenizer_for(data_dir: &Path) -> Option<Tokenizer> {
    let src = shared_tokenizer().await?;
    seed_tokenizer(data_dir).await;
    Tokenizer::from_file(&src).ok()
}

/// True if a tokenizer is available. Used to skip tokenizer-dependent tests cleanly
/// when offline with no cache.
pub async fn tokenizer_available() -> bool {
    shared_tokenizer().await.is_some()
}

// ---------------------------------------------------------------------------
// Audio test helpers (shared by audio_ingest.rs and audio_anchors.rs)
// ---------------------------------------------------------------------------

/// Writes a mono 16 kHz PCM16 WAV of `seconds` seconds carrying a 440 Hz tone
/// (nonzero, so it survives the all-silent guard) to `path`. At the default
/// ~30 s window this yields `ceil(seconds / 30)` decode windows — pass ≥ 61 s
/// for the ≥ 3 windows the deterministic cancel test needs.
pub fn write_tone_wav(path: &std::path::Path, seconds: u32) {
    const SAMPLE_RATE: u32 = 16_000;
    let n_samples = SAMPLE_RATE * seconds;
    let data_len = n_samples * 2; // 16-bit mono
    let mut buf: Vec<u8> = Vec::with_capacity(44 + data_len as usize);

    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    buf.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());

    for i in 0..n_samples {
        let t = i as f32 / SAMPLE_RATE as f32;
        let s = (t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5;
        let v = (s * i16::MAX as f32) as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }

    std::fs::write(path, &buf).expect("write tone wav");
}

/// Routes `LensEngine::transcribe` to the injected mock: the `apple_native`
/// backend with an injected engine hits the `(AppleNative, Some)` arm (the mock
/// is the Apple-native seam in tests).
pub async fn use_mock_asr(engine: &LensEngine, segments: Vec<lens_core::TranscriptSegment>) {
    let mut config = engine.config().await;
    config.asr.backend = "apple_native".to_string();
    engine.set_config(config).await;
    engine
        .set_asr_engine(Some(std::sync::Arc::new(lens_core::MockAsrEngine::new(
            segments,
        ))))
        .await;
}

// ---------------------------------------------------------------------------
// Lance vector-store readers
// ---------------------------------------------------------------------------

/// Returns the set of chunk ids stored in Lance for `source_id`. Reads the
/// physical table directly via a fresh lancedb connection to avoid coupling to
/// the (private) store internals. Returns an empty set when the store / table
/// does not exist yet (a never-ingested source).
pub async fn vector_chunk_ids(
    data_dir: &Path,
    notebook: &str,
    source_id: &str,
) -> std::collections::HashSet<String> {
    use arrow_array::StringArray;
    use futures_util::TryStreamExt;
    use lancedb::query::{ExecutableQuery, QueryBase};

    let root = data_dir.join("lancedb");
    let conn = match lancedb::connect(root.to_string_lossy().as_ref())
        .execute()
        .await
    {
        Ok(c) => c,
        Err(_) => return std::collections::HashSet::new(),
    };
    let table_name = format!(
        "vec__{notebook}__fastembed__nomic_v15__d{}",
        lens_core::DEFAULT_EMBED_DIM
    );
    let names = conn.table_names().execute().await.unwrap_or_default();
    if !names.iter().any(|n| n == &table_name) {
        return std::collections::HashSet::new();
    }
    let table = conn.open_table(&table_name).execute().await.unwrap();
    let stream = table
        .query()
        .only_if(format!("source_id = '{source_id}'"))
        .execute()
        .await
        .unwrap();
    let batches: Vec<_> = stream.try_collect().await.unwrap();
    let mut ids = std::collections::HashSet::new();
    for batch in &batches {
        let col = batch
            .column_by_name("chunk_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..batch.num_rows() {
            ids.insert(col.value(i).to_string());
        }
    }
    ids
}

/// Counts the Lance vector rows for a given source. Reads the physical table
/// directly (search-by-source is not a trait method) and is robust to a missing
/// store/table (returns 0), which the never-ingested-source tests rely on.
pub async fn vector_row_count(data_dir: &Path, notebook: &str, source_id: &str) -> usize {
    vector_chunk_ids(data_dir, notebook, source_id).await.len()
}
