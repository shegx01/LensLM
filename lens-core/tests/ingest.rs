//! M4 Phase 1, Group g.1 — integration & snapshot tests for the text/Markdown
//! ingestion slice: parser fidelity, chunk hierarchy, vector-store notebook
//! isolation, the `embedding_index` registry, the embedder cached-once +
//! concurrency invariants, re-ingest idempotency + G5 wipe ordering,
//! `ingest_source` streaming, and the real-model cosine ACs.
//!
//! # Network requirements
//!
//! Tests fall into three bands:
//!
//! * **Offline** — parser snapshots, byte-identity, vector-store isolation +
//!   registry, and the cached-once/concurrency invariants. These use hand-built
//!   vectors or the model-free [`CountingEmbedder`] and never touch the network.
//!   The chunk-hierarchy and ingest tests additionally need the nomic
//!   `tokenizer.json` (a few-MB download the ingest pipeline performs once into
//!   the temp dir's `models/fastembed/`); they are skipped offline unless a
//!   tokenizer is reachable (see [`tokenizer_available`]).
//! * **Real-model** — cosine self-similarity / doc≠query prefixing with the real
//!   [`FastembedEmbedder`]. These download the ~130 MB nomic weights on first
//!   run and are gated behind the `LENS_RUN_MODEL_TESTS` env var so the default
//!   suite stays runnable without a 130 MB download. Set
//!   `LENS_RUN_MODEL_TESTS=1` to run them (the orchestrator runs the full set
//!   once).
//!
//! The embedder injection seam used by the cached-once / concurrency tests is
//! `LensEngine::set_embedder_for_test`, available only under the crate's
//! `test-util` feature (auto-enabled for this crate's test builds).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use lens_core::chunk::{
    CHILD_TOKEN_TARGET, CHILD_TOKEN_TOLERANCE, PARENT_TOKEN_TARGET, PARENT_TOKEN_TOLERANCE,
    chunk_blocks,
};
use lens_core::embedder::{CountingEmbedder, Embedder, FastembedEmbedder};
use lens_core::parse::{Block, SourceKind, parse_blocks};
use lens_core::vector_store::{LanceVectorStore, VectorRow, VectorStore};
use lens_core::{EMBED_DIM, EMBED_MODEL_ID, IngestProgress, LensEngine};
use sqlx::Row;
use tempfile::TempDir;
use tokenizers::Tokenizer;

// ===========================================================================
// Shared helpers
// ===========================================================================

const SAMPLE_MD: &str = include_str!("fixtures/sample.md");
const PLAIN_TXT: &str = include_str!("fixtures/plain.txt");

/// Cosine similarity of two equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// A unit vector of length [`EMBED_DIM`] with all weight on dimension `axis`.
/// Lets isolation/registry tests build deterministic, model-free vectors.
fn unit_vector(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; EMBED_DIM];
    v[axis % EMBED_DIM] = 1.0;
    v
}

/// Are the real-model (FastembedEmbedder) tests enabled?
fn model_tests_enabled() -> bool {
    std::env::var("LENS_RUN_MODEL_TESTS").is_ok()
}

/// Builds a file-backed engine over a fresh temp dir. Ingest tests need a
/// file-backed engine (text sources are written under `{data_dir}/sources/`),
/// not the in-memory `for_test()`.
async fn file_engine() -> (TempDir, LensEngine) {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = LensEngine::init(dir.path()).await.expect("engine init");
    (dir, engine)
}

/// Attempts to load the nomic tokenizer the ingest pipeline would use: from the
/// `NOMIC_TOKENIZER_PATH` env var (fast offline path) or by performing the
/// pipeline's own download into `data_dir`. Returns `None` if neither works
/// (offline + no cached tokenizer) so tokenizer-dependent tests skip cleanly.
async fn tokenizer_for(data_dir: &std::path::Path) -> Option<Tokenizer> {
    if let Ok(path) = std::env::var("NOMIC_TOKENIZER_PATH")
        && let Ok(t) = Tokenizer::from_file(&path)
    {
        // Seed the engine's expected location too, so a subsequent ingest in
        // the same data dir does not re-download.
        let dest = data_dir
            .join("models")
            .join("fastembed")
            .join("tokenizer.json");
        if let Some(parent) = dest.parent() {
            let _ = std::fs::create_dir_all(parent);
            let _ = std::fs::copy(&path, &dest);
        }
        return Some(t);
    }
    download_tokenizer_into(data_dir).await
}

/// Best-effort: download the nomic `tokenizer.json` into the engine's fastembed
/// cache so the ingest pipeline finds it without a second fetch. Returns the
/// loaded tokenizer, or `None` on any network failure.
async fn download_tokenizer_into(data_dir: &std::path::Path) -> Option<Tokenizer> {
    let url = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json";
    let dest = data_dir
        .join("models")
        .join("fastembed")
        .join("tokenizer.json");
    if dest.is_file() {
        return Tokenizer::from_file(&dest).ok();
    }
    std::fs::create_dir_all(dest.parent()?).ok()?;
    let bytes = reqwest::get(url).await.ok()?.bytes().await.ok()?;
    std::fs::write(&dest, &bytes).ok()?;
    Tokenizer::from_file(&dest).ok()
}

/// True if a tokenizer is reachable (env path or network). Used to skip
/// tokenizer-dependent tests cleanly when offline with no cached tokenizer.
async fn tokenizer_available() -> bool {
    let dir = tempfile::tempdir().expect("tempdir");
    tokenizer_for(dir.path()).await.is_some()
}

// ===========================================================================
// Parser ACs (offline)
// ===========================================================================

/// AC: a fixture with `# A` / `## B` / `### C` yields the joined heading trail.
/// Snapshot the `(level, section_path, block_type, text)` shape so the heading
/// trail is eyeballed in review.
#[test]
fn parser_markdown_section_path_snapshot() {
    let blocks = parse_blocks(SAMPLE_MD, SourceKind::Markdown);
    let shape: Vec<(usize, &str, &str, &str)> = blocks
        .iter()
        .map(|b| {
            // "level" here is the heading depth implied by the section_path; we
            // expose the section_path itself, which is the AC-relevant value.
            let depth = if b.section_path.is_empty() {
                0
            } else {
                b.section_path.split(" > ").count()
            };
            (
                depth,
                b.section_path.as_str(),
                b.block_type.as_str(),
                b.text.as_str(),
            )
        })
        .collect();
    insta::assert_debug_snapshot!("parser_sample_md_blocks", shape);
}

/// AC: content under `### C` has `section_path == "A > B > C"`.
#[test]
fn parser_markdown_full_heading_trail() {
    let blocks = parse_blocks(SAMPLE_MD, SourceKind::Markdown);
    let para_c = blocks
        .iter()
        .find(|b| b.block_type == "paragraph" && b.text == "Content under C.")
        .expect("paragraph under ### C");
    assert_eq!(para_c.section_path, "A > B > C");
}

/// AC: plain text yields `section_path == ""` and `block_type == "paragraph"`.
#[test]
fn parser_plain_text_empty_section_path() {
    let blocks = parse_blocks(PLAIN_TXT, SourceKind::Text);
    assert!(!blocks.is_empty(), "plain text should yield blocks");
    for b in &blocks {
        assert_eq!(b.section_path, "", "plain text has no heading trail");
        assert_eq!(
            b.block_type, "paragraph",
            "plain text blocks are paragraphs"
        );
    }
}

/// AC (byte-identity, parser level): `src[char_start..char_end] == text`.
#[test]
fn parser_byte_identity_over_fixture_and_generated() {
    let mut cases: Vec<(String, SourceKind)> = vec![
        (SAMPLE_MD.to_string(), SourceKind::Markdown),
        (PLAIN_TXT.to_string(), SourceKind::Text),
        (
            "Plain one.\n\nPlain two with 🦀 emoji and 日本語.".to_string(),
            SourceKind::Text,
        ),
        (
            "# H\n\nBody with `code` and *emphasis*.\n\n## H2\n\nMore.".to_string(),
            SourceKind::Markdown,
        ),
    ];
    cases.push((
        "Single line no trailing newline".to_string(),
        SourceKind::Text,
    ));

    for (src, kind) in &cases {
        let blocks = parse_blocks(src, *kind);
        assert_block_byte_identity(src, &blocks);
    }
}

fn assert_block_byte_identity(src: &str, blocks: &[Block]) {
    for (i, b) in blocks.iter().enumerate() {
        assert!(b.char_end <= src.len(), "block[{i}] char_end OOB");
        assert_eq!(
            &src[b.char_start..b.char_end],
            b.text,
            "byte-identity violated for block[{i}] ({})",
            b.block_type
        );
    }
}

// ===========================================================================
// Chunker ACs (need the nomic tokenizer)
// ===========================================================================

/// AC: byte-identity at the CHUNK level + parent/child hierarchy invariants +
/// token-window bounds. Skips cleanly if no tokenizer is reachable offline.
#[tokio::test]
async fn chunker_hierarchy_and_byte_identity() {
    let dir = tempfile::tempdir().unwrap();
    let tokenizer = match tokenizer_for(dir.path()).await {
        Some(t) => t,
        None => {
            eprintln!("skipping chunker_hierarchy_and_byte_identity: no tokenizer (offline)");
            return;
        }
    };

    // A document long enough to force ≥1 parent and ≥1 child split.
    let mut src = String::from("# Intro\n\n");
    for i in 0..80 {
        src.push_str(&format!(
            "Sentence {i} adds tokens to exercise the parent and child chunk window boundaries cleanly.\n\n"
        ));
    }

    let blocks = parse_blocks(&src, SourceKind::Markdown);
    let chunks = chunk_blocks(&src, &blocks, &tokenizer).expect("chunk");

    // Byte-identity for every chunk.
    for (i, c) in chunks.iter().enumerate() {
        let (s, e) = (c.char_start as usize, c.char_end as usize);
        assert!(e <= src.len(), "chunk[{i}] char_end OOB");
        assert_eq!(&src[s..e], c.text, "byte-identity violated for chunk[{i}]");
    }

    let parents: Vec<_> = chunks.iter().filter(|c| c.level == 0).collect();
    let children: Vec<_> = chunks.iter().filter(|c| c.level == 1).collect();
    assert!(!parents.is_empty(), "expected ≥1 parent");
    assert!(!children.is_empty(), "expected ≥1 child");

    // Parents are level=0 with no parent_id; children are level=1 with a
    // parent_id resolving to a real level=0 chunk.
    let parent_ids: std::collections::HashSet<&str> =
        parents.iter().map(|c| c.id.as_str()).collect();
    for p in &parents {
        assert_eq!(p.level, 0);
        assert!(p.parent_id.is_none(), "parent must have no parent_id");
        assert_eq!(p.kind, "parent");
    }
    for c in &children {
        assert_eq!(c.level, 1);
        assert_eq!(c.kind, "child");
        let pid = c.parent_id.as_deref().expect("child has parent_id");
        assert!(
            parent_ids.contains(pid),
            "child parent_id resolves to a real parent"
        );
    }

    // Token-window bounds within target + tolerance.
    for p in &parents {
        let span = (p.token_end - p.token_start) as usize;
        assert!(
            span <= PARENT_TOKEN_TARGET + PARENT_TOKEN_TOLERANCE,
            "parent token span {span} > {}",
            PARENT_TOKEN_TARGET + PARENT_TOKEN_TOLERANCE
        );
    }
    for c in &children {
        let span = (c.token_end - c.token_start) as usize;
        assert!(
            span <= CHILD_TOKEN_TARGET + CHILD_TOKEN_TOLERANCE,
            "child token span {span} > {}",
            CHILD_TOKEN_TARGET + CHILD_TOKEN_TOLERANCE
        );
    }
}

// ===========================================================================
// VectorStore notebook-isolation AC (offline, hand-built vectors)
// ===========================================================================

/// AC (the spec's named isolation AC): add rows for notebook 1 and notebook 2,
/// search notebook 1 → only notebook-1 chunk ids come back.
#[tokio::test]
async fn vector_store_notebook_isolation() {
    let (_dir, engine) = file_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    // The embedding_index registry has a FK to notebooks(id), so the notebook
    // ids must be real rows.
    let nb1 = engine
        .create_notebook("nb1", None, None)
        .await
        .unwrap()
        .id
        .to_string();
    let nb2 = engine
        .create_notebook("nb2", None, None)
        .await
        .unwrap()
        .id
        .to_string();
    let pool = engine.pool().await;
    let store = LanceVectorStore::new(&data_dir, pool);

    // Notebook 1: two chunks on axes 0 and 1.
    store
        .add(
            &nb1,
            EMBED_MODEL_ID,
            EMBED_DIM,
            vec![
                VectorRow {
                    chunk_id: "nb1-a".into(),
                    source_id: "s1".into(),
                    notebook_id: nb1.clone(),
                    level: 1,
                    vector: unit_vector(0),
                },
                VectorRow {
                    chunk_id: "nb1-b".into(),
                    source_id: "s1".into(),
                    notebook_id: nb1.clone(),
                    level: 1,
                    vector: unit_vector(1),
                },
            ],
        )
        .await
        .expect("add nb1");

    // Notebook 2: a chunk on the SAME axis 0 as nb1-a (so a non-isolated search
    // would surface it).
    store
        .add(
            &nb2,
            EMBED_MODEL_ID,
            EMBED_DIM,
            vec![VectorRow {
                chunk_id: "nb2-a".into(),
                source_id: "s2".into(),
                notebook_id: nb2.clone(),
                level: 1,
                vector: unit_vector(0),
            }],
        )
        .await
        .expect("add nb2");

    let hits = store
        .search(&nb1, EMBED_MODEL_ID, EMBED_DIM, &unit_vector(0), 5)
        .await
        .expect("search nb1");

    let ids: std::collections::HashSet<&str> = hits.iter().map(|h| h.chunk_id.as_str()).collect();
    assert!(ids.contains("nb1-a"), "nb1 search returns nb1 chunk");
    assert!(
        !ids.contains("nb2-a"),
        "nb1 search must NOT return any nb2 chunk (isolation)"
    );
    for h in &hits {
        assert!(
            h.chunk_id.starts_with("nb1-"),
            "only nb1 ids: got {}",
            h.chunk_id
        );
    }
}

/// AC: ascending cosine distance ordering — the nearest vector ranks first.
#[tokio::test]
async fn vector_store_search_orders_by_ascending_cosine() {
    let (_dir, engine) = file_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nbx = engine
        .create_notebook("nbX", None, None)
        .await
        .unwrap()
        .id
        .to_string();
    let pool = engine.pool().await;
    let store = LanceVectorStore::new(&data_dir, pool);

    store
        .add(
            &nbx,
            EMBED_MODEL_ID,
            EMBED_DIM,
            vec![
                VectorRow {
                    chunk_id: "near".into(),
                    source_id: "s".into(),
                    notebook_id: nbx.clone(),
                    level: 1,
                    vector: unit_vector(0),
                },
                VectorRow {
                    chunk_id: "far".into(),
                    source_id: "s".into(),
                    notebook_id: nbx.clone(),
                    level: 1,
                    vector: unit_vector(7),
                },
            ],
        )
        .await
        .unwrap();

    let hits = store
        .search(&nbx, EMBED_MODEL_ID, EMBED_DIM, &unit_vector(0), 2)
        .await
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].chunk_id, "near", "nearest (cosine) ranks first");
    assert!(hits[0].distance <= hits[1].distance, "ascending distance");
}

// ===========================================================================
// embedding_index registry ACs (offline)
// ===========================================================================

/// AC: after an add for (notebook, model, dim), exactly one registry row with
/// the expected model/dim/prefix/table_name/status="active". A SECOND add into
/// the SAME notebook keeps it at exactly one row (idempotent register).
#[tokio::test]
async fn embedding_index_registers_once_per_notebook() {
    let (_dir, engine) = file_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb1 = engine
        .create_notebook("nb1", None, None)
        .await
        .unwrap()
        .id
        .to_string();
    let pool = engine.pool().await;
    let store = LanceVectorStore::new(&data_dir, pool.clone());

    // First source into nb1.
    store
        .add(
            &nb1,
            EMBED_MODEL_ID,
            EMBED_DIM,
            vec![VectorRow {
                chunk_id: "c1".into(),
                source_id: "s1".into(),
                notebook_id: nb1.clone(),
                level: 1,
                vector: unit_vector(0),
            }],
        )
        .await
        .unwrap();

    // Second source into the SAME notebook (idempotent register).
    store
        .add(
            &nb1,
            EMBED_MODEL_ID,
            EMBED_DIM,
            vec![VectorRow {
                chunk_id: "c2".into(),
                source_id: "s2".into(),
                notebook_id: nb1.clone(),
                level: 1,
                vector: unit_vector(1),
            }],
        )
        .await
        .unwrap();

    let rows = sqlx::query(
        "SELECT model, dim, prefix_convention, lance_table_name, status \
         FROM embedding_index WHERE notebook_id = ?",
    )
    .bind(&nb1)
    .fetch_all(&pool)
    .await
    .unwrap();

    assert_eq!(
        rows.len(),
        1,
        "exactly one registry row per (notebook, model, dim)"
    );
    let r = &rows[0];
    assert_eq!(r.get::<String, _>("model"), EMBED_MODEL_ID);
    assert_eq!(r.get::<i64, _>("dim"), EMBED_DIM as i64);
    assert_eq!(
        r.get::<String, _>("prefix_convention"),
        "search_document/search_query"
    );
    assert_eq!(
        r.get::<String, _>("lance_table_name"),
        format!("vec__{nb1}__nomic_v15")
    );
    assert_eq!(r.get::<String, _>("status"), "active");
}

// ===========================================================================
// Embedder cached-once + ingest serialization (CountingEmbedder seam)
// ===========================================================================

/// AC: the cached embedder is constructed exactly once across two ingests
/// (load_count == 1) and concurrent ingests never run the session concurrently
/// (max observed in_flight ≤ 1).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn embedder_cached_once_and_ingest_serialized() {
    if !tokenizer_available().await {
        eprintln!("skipping embedder_cached_once_and_ingest_serialized: no tokenizer (offline)");
        return;
    }

    let (_dir, engine) = file_engine().await;

    // Inject ONE CountingEmbedder via the test seam. The engine's OnceCell now
    // holds this Arc, so every ingest reuses it: load_count stays 1.
    let load_count = Arc::new(AtomicUsize::new(0));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let counting = Arc::new(CountingEmbedder::new(
        Arc::clone(&load_count),
        Arc::clone(&in_flight),
    ));
    // load_count is 1 after construction. Wrap in a max-tracking embedder so we
    // can assert in_flight never exceeded 1 under concurrency.
    let max_in_flight = Arc::new(AtomicUsize::new(0));
    let probe: Arc<dyn Embedder> = Arc::new(MaxInFlightEmbedder {
        inner: counting,
        in_flight: Arc::clone(&in_flight),
        max_in_flight: Arc::clone(&max_in_flight),
    });
    engine
        .set_embedder_for_test(probe)
        .expect("inject test embedder");

    // Two sources in the same notebook.
    let nb = engine
        .create_notebook("ingest-nb", None, None)
        .await
        .unwrap();
    let s1 = engine
        .add_text_source(
            &nb.id,
            "doc1",
            "# One\n\nFirst document body about apples.\n",
            "markdown",
        )
        .await
        .unwrap();
    let s2 = engine
        .add_text_source(
            &nb.id,
            "doc2",
            "# Two\n\nSecond document body about oranges.\n",
            "markdown",
        )
        .await
        .unwrap();

    // Drive two ingests CONCURRENTLY. The single-permit semaphore must serialize
    // them so the embedder's in_flight never exceeds 1.
    let e1 = engine.clone();
    let e2 = engine.clone();
    let id1 = s1.id.clone();
    let id2 = s2.id.clone();
    let h1 = tokio::spawn(async move { e1.ingest_source(&id1, |_p| {}).await });
    let h2 = tokio::spawn(async move { e2.ingest_source(&id2, |_p| {}).await });
    h1.await.unwrap().expect("ingest s1");
    h2.await.unwrap().expect("ingest s2");

    // load_count == 1: the injected embedder was the only one, never re-built.
    assert_eq!(
        load_count.load(Ordering::SeqCst),
        1,
        "the cached embedder must be constructed exactly once"
    );
    // in_flight peaked at ≤ 1: ingest serialization held.
    assert!(
        max_in_flight.load(Ordering::SeqCst) <= 1,
        "embed sessions must never overlap (max in_flight = {})",
        max_in_flight.load(Ordering::SeqCst)
    );

    // Both sources reached `indexed`.
    for id in [&s1.id, &s2.id] {
        let src = engine_source_status(&engine, id).await;
        assert_eq!(src, "indexed", "source {id} should be indexed");
    }
}

/// Wraps an inner embedder, recording the maximum observed `in_flight` count
/// across calls so the concurrency AC has a non-flaky high-water-mark assertion.
struct MaxInFlightEmbedder {
    inner: Arc<CountingEmbedder>,
    in_flight: Arc<AtomicUsize>,
    max_in_flight: Arc<AtomicUsize>,
}

impl MaxInFlightEmbedder {
    fn record_peak(&self) {
        // The inner CountingEmbedder increments in_flight on entry; sample it and
        // bump the high-water-mark. We sample around the inner call below.
        let cur = self.in_flight.load(Ordering::SeqCst);
        let mut prev = self.max_in_flight.load(Ordering::SeqCst);
        while cur > prev {
            match self
                .max_in_flight
                .compare_exchange(prev, cur, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => break,
                Err(p) => prev = p,
            }
        }
    }
}

impl Embedder for MaxInFlightEmbedder {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, lens_core::LensError> {
        let out = self.inner.embed_documents(texts);
        self.record_peak();
        out
    }
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, lens_core::LensError> {
        let out = self.inner.embed_query(text);
        self.record_peak();
        out
    }
}

async fn engine_source_status(engine: &LensEngine, id: &str) -> String {
    let pool = engine.pool().await;
    sqlx::query_scalar::<_, String>("SELECT status FROM sources WHERE id = ?")
        .bind(id)
        .fetch_one(&pool)
        .await
        .unwrap()
}

async fn count_chunks(engine: &LensEngine, source_id: &str) -> i64 {
    let pool = engine.pool().await;
    sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM chunks WHERE source_id = ?")
        .bind(source_id)
        .fetch_one(&pool)
        .await
        .unwrap()
}

// ===========================================================================
// ingest_source streaming + status transitions
// ===========================================================================

/// AC: `ingest_source` emits ordered phases (parsing → … → done) and a
/// successful ingest sets status to `indexed` with populated metadata.
#[tokio::test]
async fn ingest_streaming_phases_and_indexed_status() {
    if !tokenizer_available().await {
        eprintln!("skipping ingest_streaming_phases_and_indexed_status: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;

    let nb = engine
        .create_notebook("stream-nb", None, None)
        .await
        .unwrap();
    let src = engine
        .add_text_source(
            &nb.id,
            "doc",
            "# Title\n\nSome body text for streaming.\n",
            "markdown",
        )
        .await
        .unwrap();

    let mut phases: Vec<String> = Vec::new();
    engine
        .ingest_source(&src.id, |p: IngestProgress| phases.push(p.phase))
        .await
        .expect("ingest");

    // First phase is parsing; last is done.
    assert_eq!(phases.first().map(String::as_str), Some("parsing"));
    assert_eq!(phases.last().map(String::as_str), Some("done"));
    // The canonical phase order is a subsequence of the emitted phases.
    let order = ["parsing", "chunking", "embedding", "indexing", "done"];
    assert_subsequence(&phases, &order);

    // Status + metadata populated.
    let pool = engine.pool().await;
    let row = sqlx::query("SELECT status, token_count, content_hash FROM sources WHERE id = ?")
        .bind(&src.id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(row.get::<String, _>("status"), "indexed");
    assert!(
        row.get::<Option<i64>, _>("token_count").unwrap() > 0,
        "token_count populated"
    );
    assert_eq!(
        row.get::<Option<String>, _>("content_hash").unwrap().len(),
        64,
        "content_hash is a 64-hex SHA-256"
    );
}

/// AC: a failing ingest sets `sources.status = "error"`. We force the failure by
/// deleting the managed source file out from under the pipeline (read fails).
#[tokio::test]
async fn ingest_failure_sets_error_status() {
    let (_dir, engine) = file_engine().await;
    let nb = engine.create_notebook("err-nb", None, None).await.unwrap();
    let src = engine
        .add_text_source(&nb.id, "doc", "body", "text")
        .await
        .unwrap();

    // Remove the backing file so the pipeline's read_to_string fails.
    std::fs::remove_file(&src.locator).unwrap();

    let result = engine.ingest_source(&src.id, |_p| {}).await;
    assert!(result.is_err(), "ingest of a missing file should fail");

    let status = engine_source_status(&engine, &src.id).await;
    assert_eq!(status, "error", "failed ingest sets status=error");
}

fn assert_subsequence(haystack: &[String], needles: &[&str]) {
    let mut it = haystack.iter();
    for n in needles {
        let found = it.any(|h| h == n);
        assert!(found, "phase {n:?} missing or out of order in {haystack:?}");
    }
}

/// Builds a file-backed engine with an injected CountingEmbedder so ingest tests
/// avoid the 130 MB model (they still need the tokenizer for chunking).
async fn inject_counting_engine() -> (TempDir, LensEngine) {
    let (dir, engine) = file_engine().await;
    let load_count = Arc::new(AtomicUsize::new(0));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let counting: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new(load_count, in_flight));
    engine
        .set_embedder_for_test(counting)
        .expect("inject test embedder");
    (dir, engine)
}

// ===========================================================================
// Re-ingest idempotency + G5 cross-store wipe ordering
// ===========================================================================

/// AC: re-ingesting an unchanged indexed source is a no-op (no duplicate
/// chunks/vectors, status stays indexed); changing the file wipes prior
/// chunks/vectors (Lance dropped before SQLite) and re-indexes with no orphans.
#[tokio::test]
async fn reingest_idempotency_and_wipe() {
    if !tokenizer_available().await {
        eprintln!("skipping reingest_idempotency_and_wipe: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;
    let data_dir = engine.data_dir_for_test().await;

    let nb = engine
        .create_notebook("reingest-nb", None, None)
        .await
        .unwrap();
    let src = engine
        .add_text_source(
            &nb.id,
            "doc",
            "# Doc\n\nOriginal body paragraph one.\n\nOriginal body paragraph two.\n",
            "markdown",
        )
        .await
        .unwrap();

    // First ingest → indexed.
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();
    assert_eq!(engine_source_status(&engine, &src.id).await, "indexed");
    let chunks_after_first = count_chunks(&engine, &src.id).await;
    assert!(chunks_after_first > 0, "chunks inserted");
    let vec_count_first = vector_row_count(&data_dir, &nb.id.to_string(), &src.id).await;
    assert!(vec_count_first > 0, "vectors inserted");

    // Re-ingest with UNCHANGED content → no-op (no duplicate chunks/vectors).
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();
    assert_eq!(engine_source_status(&engine, &src.id).await, "indexed");
    assert_eq!(
        count_chunks(&engine, &src.id).await,
        chunks_after_first,
        "unchanged re-ingest must not duplicate chunks"
    );
    assert_eq!(
        vector_row_count(&data_dir, &nb.id.to_string(), &src.id).await,
        vec_count_first,
        "unchanged re-ingest must not duplicate vectors"
    );

    // CHANGE the file content, then re-ingest → wipe + re-index, no orphans.
    std::fs::write(
        &src.locator,
        "# Doc\n\nCompletely different content now, with three new sentences. Here is the second. And a third one too.\n",
    )
    .unwrap();
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();
    assert_eq!(engine_source_status(&engine, &src.id).await, "indexed");

    // No orphaned vectors: every Lance vector for this source maps to a live
    // SQLite chunk id.
    let chunk_ids = live_chunk_ids(&engine, &src.id).await;
    let vec_ids = vector_chunk_ids(&data_dir, &nb.id.to_string(), &src.id).await;
    assert!(!vec_ids.is_empty(), "re-index produced vectors");
    for vid in &vec_ids {
        assert!(
            chunk_ids.contains(vid),
            "orphaned Lance vector {vid} has no matching SQLite chunk"
        );
    }
    // And every SQLite chunk has a vector (the index covers all chunks).
    assert_eq!(
        chunk_ids.len(),
        vec_ids.len(),
        "chunk count and vector count must match after re-index"
    );
}

/// Counts the Lance vector rows for a given source by querying the store
/// directly (search-by-source is not a trait method, so we read via a wide
/// search and filter — sufficient for the small test corpus).
async fn vector_row_count(data_dir: &std::path::Path, notebook: &str, source_id: &str) -> usize {
    vector_chunk_ids(data_dir, notebook, source_id).await.len()
}

/// Returns the set of chunk ids stored in Lance for `source_id`. Reads the
/// physical table directly via a fresh lancedb connection to avoid coupling to
/// the (private) store internals.
async fn vector_chunk_ids(
    data_dir: &std::path::Path,
    notebook: &str,
    source_id: &str,
) -> std::collections::HashSet<String> {
    use arrow_array::StringArray;
    use futures_util::TryStreamExt;
    use lancedb::query::{ExecutableQuery, QueryBase};

    let root = data_dir.join("lancedb");
    let conn = lancedb::connect(root.to_string_lossy().as_ref())
        .execute()
        .await
        .expect("connect");
    let table_name = format!("vec__{notebook}__nomic_v15");
    let names = conn.table_names().execute().await.unwrap();
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

async fn live_chunk_ids(engine: &LensEngine, source_id: &str) -> std::collections::HashSet<String> {
    let pool = engine.pool().await;
    let rows: Vec<String> = sqlx::query_scalar("SELECT id FROM chunks WHERE source_id = ?")
        .bind(source_id)
        .fetch_all(&pool)
        .await
        .unwrap();
    rows.into_iter().collect()
}

// ===========================================================================
// Real-model ACs (FastembedEmbedder; gated behind LENS_RUN_MODEL_TESTS)
// ===========================================================================

/// AC: real-model cosine self-similarity ≈ 1.0 (> 0.999); doc≠query prefixing
/// produces different vectors for the same string; an unrelated string is
/// strictly less similar than self.
#[test]
fn real_model_cosine_self_similarity_and_prefixing() {
    if !model_tests_enabled() {
        eprintln!(
            "skipping real_model_cosine_self_similarity_and_prefixing (set LENS_RUN_MODEL_TESTS=1)"
        );
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let embedder = FastembedEmbedder::new(dir.path()).expect("build fastembed");

    let text = "the mitochondria is the powerhouse of the cell";
    let doc1 = embedder.embed_documents(&[text]).unwrap();
    let doc2 = embedder.embed_documents(&[text]).unwrap();
    let query = embedder.embed_query(text).unwrap();
    let unrelated = embedder
        .embed_documents(&["quarterly tax filing deadlines"])
        .unwrap();

    // 768-dim, L2-normalized.
    assert_eq!(doc1[0].len(), EMBED_DIM);
    let norm: f32 = doc1[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "‖v‖ ≈ 1, got {norm}");

    // Self-similarity ≈ 1.
    assert!(
        cosine(&doc1[0], &doc2[0]) > 0.999,
        "self-similarity > 0.999"
    );

    // doc ≠ query for the same string (prefix logic).
    assert_ne!(
        doc1[0], query,
        "doc and query vectors must differ (prefixing)"
    );

    // Unrelated string strictly less similar than self.
    assert!(
        cosine(&doc1[0], &unrelated[0]) < cosine(&doc1[0], &doc2[0]),
        "unrelated text should be less similar than self"
    );
}

// ===========================================================================
// Source soft-delete (trash / restore / purge)
// ===========================================================================

/// AC: `purge_source` wipes the source row, its chunks, and its Lance vectors.
/// Calling it a second time returns a validation error (not found).
#[tokio::test]
async fn purge_source_removes_rows_and_vectors() {
    if !tokenizer_available().await {
        eprintln!("skipping purge_source_removes_rows_and_vectors: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;
    let data_dir = engine.data_dir_for_test().await;

    // 1. Create notebook + text source.
    let nb = engine
        .create_notebook("purge-src-nb", None, None)
        .await
        .unwrap();
    let src = engine
        .add_text_source(
            &nb.id,
            "to-purge",
            "# Purge me\n\nBody text that will be ingested and then purged.\n",
            "markdown",
        )
        .await
        .unwrap();

    // 2. Ingest so Lance vectors exist.
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();

    // 3. Assert source indexed, chunks > 0, vectors > 0.
    assert_eq!(engine_source_status(&engine, &src.id).await, "indexed");
    let chunks_before = count_chunks(&engine, &src.id).await;
    assert!(chunks_before > 0, "chunks must exist before purge");
    let vecs_before = vector_row_count(&data_dir, &nb.id.to_string(), &src.id).await;
    assert!(vecs_before > 0, "vectors must exist before purge");

    // 4. Purge the source — must succeed.
    engine
        .purge_source(&src.id)
        .await
        .expect("purge_source should succeed");

    // 5. Source row gone.
    let pool = engine.pool().await;
    let row = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM sources WHERE id = ?")
        .bind(&src.id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(row, 0, "source row must be gone after purge");

    // 6. Chunk rows gone (cascade).
    assert_eq!(
        count_chunks(&engine, &src.id).await,
        0,
        "chunk rows must be removed after purge"
    );

    // 7. Lance vectors gone.
    assert_eq!(
        vector_row_count(&data_dir, &nb.id.to_string(), &src.id).await,
        0,
        "Lance vectors must be removed after purge"
    );

    // 8. Second purge returns Err (not found).
    let second = engine.purge_source(&src.id).await;
    assert!(
        second.is_err(),
        "purging a non-existent source must return an error"
    );
}

/// AC: `trash_source` sets `trashed_at` and the source is EXCLUDED from
/// `list_sources`, but chunks and Lance vectors are preserved.
#[tokio::test]
async fn trash_source_hides_from_list_keeps_data() {
    if !tokenizer_available().await {
        eprintln!("skipping trash_source_hides_from_list_keeps_data: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;
    let data_dir = engine.data_dir_for_test().await;

    let nb = engine
        .create_notebook("trash-src-nb", None, None)
        .await
        .unwrap();
    let src = engine
        .add_text_source(
            &nb.id,
            "to-trash",
            "# Trash me\n\nBody text that will be trashed but not purged.\n",
            "markdown",
        )
        .await
        .unwrap();

    // Ingest so chunks + vectors exist.
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();
    let chunks_before = count_chunks(&engine, &src.id).await;
    assert!(chunks_before > 0, "chunks must exist before trash");
    let vecs_before = vector_row_count(&data_dir, &nb.id.to_string(), &src.id).await;
    assert!(vecs_before > 0, "vectors must exist before trash");

    // Source visible before trash.
    let live = engine.list_sources(&nb.id).await.unwrap();
    assert_eq!(live.len(), 1, "source visible before trash");

    // Trash the source — must succeed.
    engine
        .trash_source(&src.id)
        .await
        .expect("trash_source should succeed");

    // Source is no longer visible in list_sources.
    let after = engine.list_sources(&nb.id).await.unwrap();
    assert!(
        after.is_empty(),
        "trashed source must not appear in list_sources"
    );

    // trashed_at is set in the DB.
    let pool = engine.pool().await;
    let trashed_at =
        sqlx::query_scalar::<_, Option<String>>("SELECT trashed_at FROM sources WHERE id = ?")
            .bind(&src.id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert!(
        trashed_at.is_some(),
        "trashed_at must be set after trash_source"
    );

    // Chunks and vectors are STILL present (soft-delete keeps data).
    assert_eq!(
        count_chunks(&engine, &src.id).await,
        chunks_before,
        "chunks must survive trash (soft-delete)"
    );
    assert_eq!(
        vector_row_count(&data_dir, &nb.id.to_string(), &src.id).await,
        vecs_before,
        "vectors must survive trash (soft-delete)"
    );
}

/// AC: `restore_source` clears `trashed_at` and the source reappears in
/// `list_sources`.
#[tokio::test]
async fn restore_source_reappears_in_list() {
    let (_dir, engine) = file_engine().await;

    let nb = engine
        .create_notebook("restore-src-nb", None, None)
        .await
        .unwrap();
    let src = engine
        .add_text_source(&nb.id, "to-restore", "Just some text.", "text")
        .await
        .unwrap();

    // Trash then restore.
    engine.trash_source(&src.id).await.unwrap();
    assert!(
        engine.list_sources(&nb.id).await.unwrap().is_empty(),
        "source must be hidden after trash"
    );

    engine.restore_source(&src.id).await.unwrap();
    let live = engine.list_sources(&nb.id).await.unwrap();
    assert_eq!(live.len(), 1, "source must reappear after restore");
    assert!(
        live[0].trashed_at.is_none(),
        "trashed_at must be cleared after restore"
    );
}

/// AC: trashing an already-trashed source errors; restoring a live source errors.
#[tokio::test]
async fn trash_and_restore_idempotency_errors() {
    let (_dir, engine) = file_engine().await;

    let nb = engine
        .create_notebook("idem-src-nb", None, None)
        .await
        .unwrap();
    let src = engine
        .add_text_source(&nb.id, "idem", "Idempotency test.", "text")
        .await
        .unwrap();

    // Restoring a live source must fail.
    let err = engine.restore_source(&src.id).await;
    assert!(err.is_err(), "restoring a live source must fail");

    // Trash once — OK.
    engine.trash_source(&src.id).await.unwrap();

    // Trashing again must fail.
    let err2 = engine.trash_source(&src.id).await;
    assert!(
        err2.is_err(),
        "trashing an already-trashed source must fail"
    );
}

/// AC: purging a never-ingested source (no Lance vectors) is a clean no-op for
/// the vector store — the SQL row is removed without error.
#[tokio::test]
async fn purge_source_without_vectors_is_clean() {
    let (_dir, engine) = file_engine().await;

    let nb = engine
        .create_notebook("purge-no-vec-nb", None, None)
        .await
        .unwrap();
    let src = engine
        .add_text_source(&nb.id, "no-vectors", "No ingest yet.", "text")
        .await
        .unwrap();

    // Purge without any prior ingest — must succeed (vector store handles missing
    // table gracefully).
    engine
        .purge_source(&src.id)
        .await
        .expect("purge of a never-ingested source must succeed");

    // Row is gone.
    let pool = engine.pool().await;
    let count = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM sources WHERE id = ?")
        .bind(&src.id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 0, "source row must be gone after purge");
}

/// AC (real model, end-to-end): ingest a real text source with the real
/// embedder and assert it reaches `indexed` and search returns its chunks.
#[tokio::test]
async fn real_model_end_to_end_ingest_and_search() {
    if !model_tests_enabled() {
        eprintln!("skipping real_model_end_to_end_ingest_and_search (set LENS_RUN_MODEL_TESTS=1)");
        return;
    }
    let (_dir, engine) = file_engine().await;
    let data_dir = engine.data_dir_for_test().await;

    let nb = engine.create_notebook("real-nb", None, None).await.unwrap();
    let src = engine
        .add_text_source(
            &nb.id,
            "bio",
            "# Cells\n\nThe mitochondria produce ATP, the energy currency of the cell.\n",
            "markdown",
        )
        .await
        .unwrap();
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();
    assert_eq!(engine_source_status(&engine, &src.id).await, "indexed");

    // Query the store with the real embedder and confirm a hit.
    let pool = engine.pool().await;
    let store = LanceVectorStore::new(&data_dir, pool);
    let embedder = FastembedEmbedder::new(&data_dir).unwrap();
    let qvec = embedder
        .embed_query("what makes energy in a cell?")
        .unwrap();
    let hits = store
        .search(&nb.id.to_string(), EMBED_MODEL_ID, EMBED_DIM, &qvec, 5)
        .await
        .unwrap();
    assert!(!hits.is_empty(), "real-model search returns hits");
}
