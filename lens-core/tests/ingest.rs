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

mod support;
use support::{
    file_engine, inject_counting_engine, tokenizer_available, tokenizer_for, vector_chunk_ids,
    vector_row_count,
};

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

    // 4. Trash then purge the source — purge requires a trashed source.
    engine
        .trash_source(&src.id)
        .await
        .expect("trash_source should succeed");
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
    // table gracefully). Purge requires a trashed source, so trash it first.
    engine
        .trash_source(&src.id)
        .await
        .expect("trash_source should succeed");
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

/// AC (disk leak): `purge_source` removes the managed
/// `{data_dir}/sources/{id}.ext` file written by `add_text_source` so "Delete
/// forever" does not leak it. No ingest needed — the file exists from
/// `add_text_source` alone.
#[tokio::test]
async fn purge_source_removes_managed_file() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("purge-file-nb", None, None)
        .await
        .unwrap();
    let src = engine
        .add_text_source(&nb.id, "doc", "Some managed body text.", "text")
        .await
        .unwrap();

    // The managed file exists right after add_text_source.
    let path = std::path::PathBuf::from(&src.locator);
    assert!(
        path.exists(),
        "managed source file should exist after add_text_source: {}",
        src.locator
    );

    // Trash then purge — purge requires a trashed source.
    engine.trash_source(&src.id).await.unwrap();
    engine.purge_source(&src.id).await.unwrap();

    // The managed file is gone.
    assert!(
        !path.exists(),
        "managed source file must be removed by purge_source: {}",
        src.locator
    );
}

/// AC (disk leak, notebook): `purge_notebook` removes the managed source files
/// of all its sources (live and trashed) so a notebook "Delete forever" does not
/// leak them.
#[tokio::test]
async fn purge_notebook_removes_managed_source_files() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("purge-nb-files", None, None)
        .await
        .unwrap();
    let live = engine
        .add_text_source(&nb.id, "live", "Live source body.", "text")
        .await
        .unwrap();
    let trashed = engine
        .add_text_source(&nb.id, "trashed", "Trashed source body.", "markdown")
        .await
        .unwrap();
    // Trash one source so the notebook holds both a live and a trashed source.
    engine.trash_source(&trashed.id).await.unwrap();

    let live_path = std::path::PathBuf::from(&live.locator);
    let trashed_path = std::path::PathBuf::from(&trashed.locator);
    assert!(live_path.exists() && trashed_path.exists());

    // Trash then purge the notebook.
    engine.trash_notebook(&nb.id).await.unwrap();
    engine.purge_notebook(&nb.id).await.unwrap();

    assert!(
        !live_path.exists(),
        "live source's managed file must be removed by purge_notebook"
    );
    assert!(
        !trashed_path.exists(),
        "trashed source's managed file must be removed by purge_notebook"
    );
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

// ===========================================================================
// Ingest robustness: size cap, empty-doc short-circuit, batch seam, tables
// ===========================================================================

/// AC (size cap): `add_text_source` with text larger than
/// [`MAX_SOURCE_BYTES`](lens_core::ingest::MAX_SOURCE_BYTES) is rejected with a
/// `Validation` error before anything is written/queued.
#[tokio::test]
async fn add_text_source_rejects_oversized_input() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("oversize-nb", None, None)
        .await
        .unwrap();

    // One byte over the cap.
    let huge = "x".repeat(lens_core::ingest::MAX_SOURCE_BYTES + 1);
    let err = engine.add_text_source(&nb.id, "huge", &huge, "text").await;
    assert!(
        matches!(err, Err(lens_core::LensError::Validation(_))),
        "oversized paste must be rejected with Validation, got {err:?}"
    );

    // Exactly at the cap is accepted (boundary).
    let ok = "y".repeat(lens_core::ingest::MAX_SOURCE_BYTES);
    assert!(
        engine
            .add_text_source(&nb.id, "at-cap", &ok, "text")
            .await
            .is_ok(),
        "input exactly at the cap must be accepted"
    );
}

/// AC (empty-doc short-circuit): ingesting an empty/whitespace-only source
/// reaches `indexed` with `token_count == 0` and zero chunks — WITHOUT loading
/// the embedder.
///
/// The "embedder not loaded" property is proven structurally: this test uses a
/// bare `file_engine()` (NO injected test embedder) and does NOT set
/// `LENS_RUN_MODEL_TESTS`, so if the pipeline tried to load the embedder it
/// would attempt the ~130 MB `FastembedEmbedder` download — the short-circuit is
/// what keeps this offline-clean. Chunking still needs the tokenizer, so the
/// test skips when no tokenizer is reachable.
#[tokio::test]
async fn empty_doc_indexes_without_loading_embedder() {
    if !tokenizer_available().await {
        eprintln!("skipping empty_doc_indexes_without_loading_embedder: no tokenizer (offline)");
        return;
    }
    // NOTE: deliberately NO embedder injection — the short-circuit must avoid the
    // embedder load entirely.
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("empty-nb", None, None)
        .await
        .unwrap();
    let src = engine
        .add_text_source(&nb.id, "empty", "   \n\t\n  \n", "text")
        .await
        .unwrap();

    let mut phases: Vec<String> = Vec::new();
    engine
        .ingest_source(&src.id, |p: IngestProgress| phases.push(p.phase))
        .await
        .expect("empty-doc ingest should succeed without an embedder");

    // Status indexed, token_count 0, zero chunks.
    assert_eq!(engine_source_status(&engine, &src.id).await, "indexed");
    assert_eq!(
        count_chunks(&engine, &src.id).await,
        0,
        "empty doc produces zero chunks"
    );
    let pool = engine.pool().await;
    let token_count =
        sqlx::query_scalar::<_, Option<i64>>("SELECT token_count FROM sources WHERE id = ?")
            .bind(&src.id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(token_count, Some(0), "empty doc has token_count 0");

    // The pipeline reached `done` but never entered `embedding`/`indexing`.
    assert_eq!(phases.last().map(String::as_str), Some("done"));
    assert!(
        !phases.iter().any(|p| p == "embedding"),
        "empty doc must not emit an embedding phase: {phases:?}"
    );
}

/// AC (>60-chunk batch seam): a document large enough to produce ≥61 chunks
/// spans more than one `CHUNK_INSERT_BATCH` (60). After ingest every child's
/// `parent_id` resolves to a real parent row (the FK holds across the batch
/// seam) and `token_start` is globally non-decreasing across parents.
#[tokio::test]
async fn ingest_spans_chunk_insert_batch_seam() {
    if !tokenizer_available().await {
        eprintln!("skipping ingest_spans_chunk_insert_batch_seam: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;

    // Build a large doc so the chunker emits ≥61 chunks (parents + children).
    // Parents pack ~512 tokens then split into ~128-token children, so a single
    // parent window yields several rows; we need enough total prose to span many
    // parent windows AND clear the 60-row insert batch. ~250 substantial
    // paragraphs is a comfortable margin.
    let mut doc = String::new();
    for i in 0..250 {
        doc.push_str(&format!(
            "# Section {i}\n\nParagraph {i}: the quick brown fox jumps over the lazy dog while \
             contemplating the nature of section number {i} and its many distinct sentences. \
             Here is a second sentence for section {i} with additional prose. And a third \
             sentence to make section {i} a substantial, tokenizable chunk of real text.\n\n"
        ));
    }
    let nb = engine.create_notebook("seam-nb", None, None).await.unwrap();
    let src = engine
        .add_text_source(&nb.id, "big", &doc, "markdown")
        .await
        .unwrap();
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();
    assert_eq!(engine_source_status(&engine, &src.id).await, "indexed");

    // (a) chunk count spans more than one insert batch.
    let total = count_chunks(&engine, &src.id).await;
    assert!(
        total >= 61,
        "expected ≥61 chunks to cross the 60-row insert batch, got {total}"
    );

    // Pull (id, parent_id, level, token_start) for all chunks.
    let pool = engine.pool().await;
    let rows = sqlx::query(
        "SELECT id, parent_id, level, token_start FROM chunks WHERE source_id = ? ORDER BY rowid",
    )
    .bind(&src.id)
    .fetch_all(&pool)
    .await
    .unwrap();

    let ids: std::collections::HashSet<String> =
        rows.iter().map(|r| r.get::<String, _>("id")).collect();

    // (b) every child's parent_id resolves to an existing row.
    for r in &rows {
        if let Some(parent_id) = r.get::<Option<String>, _>("parent_id") {
            assert!(
                ids.contains(&parent_id),
                "child parent_id {parent_id} has no matching parent row (FK broken across batch seam)"
            );
        }
    }

    // (c) parent token_start values are globally non-decreasing (in insert order).
    let mut last_parent_start: i64 = i64::MIN;
    for r in &rows {
        if r.get::<i64, _>("level") == 0 {
            let start = r.get::<i64, _>("token_start");
            assert!(
                start >= last_parent_start,
                "parent token_start went backwards: {start} < {last_parent_start}"
            );
            last_parent_start = start;
        }
    }
}

/// AC (table ingest-level): a GFM table flows through parse → chunk → insert and
/// lands as a `table` chunk row whose text carries the cell values.
#[tokio::test]
async fn ingest_preserves_gfm_table_block() {
    if !tokenizer_available().await {
        eprintln!("skipping ingest_preserves_gfm_table_block: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;
    let nb = engine
        .create_notebook("table-nb", None, None)
        .await
        .unwrap();
    // The GFM table from the spec leads the document so it is the FIRST block of
    // the first parent window — parents pack by token budget and carry the FIRST
    // block's `block_type`, so a leading table yields a `table`-typed chunk. The
    // trailing filler just gives the rest of the doc some body; it does not change
    // what the assertion checks: a `table` chunk carrying the cell values.
    let filler = "the quick brown fox jumps over the lazy dog. ".repeat(20);
    let doc = format!("| A | B |\n| - | - |\n| x | y |\n\n{filler}");
    let src = engine
        .add_text_source(&nb.id, "t", &doc, "markdown")
        .await
        .unwrap();
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();
    assert_eq!(engine_source_status(&engine, &src.id).await, "indexed");

    let total = count_chunks(&engine, &src.id).await;
    assert!(total > 0, "table doc must produce chunks");

    // A `table` chunk row exists whose text contains the cell values.
    let pool = engine.pool().await;
    let table_texts: Vec<String> =
        sqlx::query_scalar("SELECT text FROM chunks WHERE source_id = ? AND block_type = 'table'")
            .bind(&src.id)
            .fetch_all(&pool)
            .await
            .unwrap();
    assert!(
        !table_texts.is_empty(),
        "expected at least one chunk row with block_type 'table'"
    );
    assert!(
        table_texts
            .iter()
            .any(|t| t.contains('x') && t.contains('y')),
        "table chunk text must carry the cell values, got {table_texts:?}"
    );
}

// ===========================================================================
// Step 3 — Extractor-seam rewire: canonical buffer, hash split, two-stage
// guard, sibling persist + purge (AC3, AC4a, AC4b, AC4d)
// ===========================================================================

use lens_core::extract::test_seam::{
    FakeBinaryExtractor, clear_test_extractor_factory, set_test_extractor_factory,
};

/// A test-only DERIVED kind whose `Extractor` is injected via the `test-util`
/// seam. The ingest pipeline treats it as binary (raw-bytes hash, `.extracted.txt`
/// sibling, Stage-1 guard) because it is NOT in `is_text_like_kind`.
const FAKE_KIND: &str = "faketest";

/// Writes a raw binary file under `{data_dir}/sources/{id}.bin` and returns its
/// path string (the source locator for a fake-binary source).
fn write_raw_source_file(data_dir: &std::path::Path, id: &str, bytes: &[u8]) -> String {
    let sources_dir = data_dir.join("sources");
    std::fs::create_dir_all(&sources_dir).unwrap();
    let path = sources_dir.join(format!("{id}.bin"));
    std::fs::write(&path, bytes).unwrap();
    path.display().to_string()
}

/// Reads back the `(canonical_buffer, content_hash)` chunk-slice invariant and
/// asserts byte-identity: `canonical[char_start..char_end] == text` for every
/// chunk. `canonical` is the EXACT buffer the chunker was fed.
async fn assert_chunk_byte_identity(engine: &LensEngine, source_id: &str, canonical: &str) {
    let pool = engine.pool().await;
    let rows = sqlx::query(
        "SELECT text, char_start, char_end FROM chunks WHERE source_id = ? ORDER BY rowid",
    )
    .bind(source_id)
    .fetch_all(&pool)
    .await
    .unwrap();
    assert!(!rows.is_empty(), "expected chunks for {source_id}");
    for (i, r) in rows.iter().enumerate() {
        let s = r.get::<i64, _>("char_start") as usize;
        let e = r.get::<i64, _>("char_end") as usize;
        let text = r.get::<String, _>("text");
        assert!(e <= canonical.len(), "chunk[{i}] char_end OOB");
        assert_eq!(
            &canonical[s..e],
            text,
            "byte-identity violated for chunk[{i}] against the canonical buffer"
        );
    }
}

/// AC4a (byte-identity, text/MD end-to-end through the new Extractor seam):
/// for every chunk, `canonical[char_start..char_end] == chunk.text`, where
/// `canonical` is the ORIGINAL locator content (text/MD has no sibling).
#[tokio::test]
async fn seam_text_md_byte_identity_against_canonical() {
    if !tokenizer_available().await {
        eprintln!("skipping seam_text_md_byte_identity_against_canonical: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;
    let nb = engine.create_notebook("seam-id", None, None).await.unwrap();

    let text_body = "Plain paragraph one with 🦀 emoji.\n\nPlain paragraph two with 日本語 text.\n";
    let md_body =
        "# 日本語 Heading\n\nBody under heading with `code` and 🦀.\n\n## Sub\n\nMore body.\n";

    for (title, body, kind) in [
        ("txt-doc", text_body, "text"),
        ("md-doc", md_body, "markdown"),
    ] {
        let src = engine
            .add_text_source(&nb.id, title, body, kind)
            .await
            .unwrap();
        engine.ingest_source(&src.id, |_p| {}).await.unwrap();
        assert_eq!(engine_source_status(&engine, &src.id).await, "indexed");
        // The canonical buffer for text/MD is the original locator content.
        let canonical = std::fs::read_to_string(&src.locator).unwrap();
        assert_chunk_byte_identity(&engine, &src.id, &canonical).await;
    }
}

/// AC3 (single buffer): the buffer fed to `chunk_blocks` is the SAME buffer
/// whose hash is stored as `content_hash`. For text/MD that is the canonical
/// text, so the stored `content_hash` must equal `sha256(canonical)`. A second
/// read between chunk + hash would let them diverge — this asserts they don't.
#[tokio::test]
async fn seam_single_buffer_drives_chunk_and_hash() {
    if !tokenizer_available().await {
        eprintln!("skipping seam_single_buffer_drives_chunk_and_hash: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;
    let nb = engine
        .create_notebook("seam-buf", None, None)
        .await
        .unwrap();
    let body = "# Buffer\n\nOne buffer drives both chunking and hashing.\n";
    let src = engine
        .add_text_source(&nb.id, "buf", body, "markdown")
        .await
        .unwrap();
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();

    let canonical = std::fs::read_to_string(&src.locator).unwrap();
    // Byte-identity proves the chunker sliced `canonical`.
    assert_chunk_byte_identity(&engine, &src.id, &canonical).await;

    // The stored hash is over that SAME canonical buffer.
    use sha2::{Digest, Sha256};
    let expected: String = Sha256::digest(canonical.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let pool = engine.pool().await;
    let stored =
        sqlx::query_scalar::<_, Option<String>>("SELECT content_hash FROM sources WHERE id = ?")
            .bind(&src.id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        stored.as_deref(),
        Some(expected.as_str()),
        "content_hash must be over the same canonical buffer that was chunked"
    );
}

/// AC4b Stage 1: a DERIVED-kind raw input OVER `MAX_SOURCE_BYTES` is rejected
/// BEFORE extraction. The injected extractor is configured to PANIC if called,
/// so reaching it would fail the test — proving the Stage-1 guard fired first.
#[tokio::test]
async fn seam_stage1_guard_rejects_oversized_binary_before_extract() {
    let (_dir, engine) = file_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine.create_notebook("seam-s1", None, None).await.unwrap();

    // Inject a fake extractor that PANICS if `extract` is ever called.
    set_test_extractor_factory(FAKE_KIND, || {
        Box::new(FakeBinaryExtractor {
            calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            extracted_text: String::new(),
            panic_if_called: true,
        })
    });

    // Raw file one byte over the cap.
    let id = uuid::Uuid::now_v7().to_string();
    let oversized = vec![0u8; lens_core::ingest::MAX_SOURCE_BYTES + 1];
    let locator = write_raw_source_file(&data_dir, &id, &oversized);
    let pool = engine.pool().await;
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
    )
    .bind(&id)
    .bind(nb.id.to_string())
    .bind(FAKE_KIND)
    .bind("oversize")
    .bind("queued")
    .bind(&locator)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let result = engine.ingest_source(&id, |_p| {}).await;
    clear_test_extractor_factory(FAKE_KIND);
    assert!(
        matches!(result, Err(lens_core::LensError::Validation(_))),
        "oversized binary must be rejected with Validation before extraction, got {result:?}"
    );
}

/// AC4b Stage 2: a SMALL DERIVED-kind raw input whose extractor explodes it into
/// OVER-cap `extracted_text` is rejected AFTER extraction.
#[tokio::test]
async fn seam_stage2_guard_rejects_oversized_extracted_text() {
    let (_dir, engine) = file_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine.create_notebook("seam-s2", None, None).await.unwrap();

    // Tiny raw bytes, but the extractor returns over-cap text.
    let huge_text = "z".repeat(lens_core::ingest::MAX_SOURCE_BYTES + 1);
    set_test_extractor_factory(FAKE_KIND, {
        let t = huge_text.clone();
        move || {
            Box::new(FakeBinaryExtractor {
                calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                extracted_text: t.clone(),
                panic_if_called: false,
            })
        }
    });

    let id = uuid::Uuid::now_v7().to_string();
    let locator = write_raw_source_file(&data_dir, &id, b"tiny");
    insert_raw_source_locator(&engine, &nb.id.to_string(), FAKE_KIND, &locator, &id).await;

    let result = engine.ingest_source(&id, |_p| {}).await;
    clear_test_extractor_factory(FAKE_KIND);
    assert!(
        matches!(result, Err(lens_core::LensError::Validation(_))),
        "over-cap extracted_text must be rejected with Validation, got {result:?}"
    );
}

/// Inserts a raw source with a known id + locator (helper for the Stage-2 /
/// no-op tests where the id is needed to predict the `.extracted.txt` sibling).
async fn insert_raw_source_locator(
    engine: &LensEngine,
    notebook_id: &str,
    kind: &str,
    locator: &str,
    id: &str,
) {
    let pool = engine.pool().await;
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
    )
    .bind(id)
    .bind(notebook_id)
    .bind(kind)
    .bind("fake")
    .bind("queued")
    .bind(locator)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();
}

/// AC4d (re-ingest no-op WITHOUT re-extraction, DERIVED kind): ingest a fake
/// binary source twice with unchanged raw bytes. The second ingest must be a
/// no-op (status stays `indexed`) AND must NOT call the extractor (raw-bytes
/// hash matched). Also asserts the `.extracted.txt` sibling was written for the
/// derived kind and is removed on purge.
#[tokio::test]
async fn seam_derived_reingest_noop_without_reextract_and_sibling_lifecycle() {
    if !tokenizer_available().await {
        eprintln!(
            "skipping seam_derived_reingest_noop_without_reextract_and_sibling_lifecycle: no tokenizer (offline)"
        );
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine
        .create_notebook("seam-noop", None, None)
        .await
        .unwrap();

    // Shared call counter across every box the factory builds.
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let fake_text =
        "# Fake Binary\n\nDecoded canonical text for the fake binary source.\n".to_string();
    set_test_extractor_factory(FAKE_KIND, {
        let calls = std::sync::Arc::clone(&calls);
        let t = fake_text.clone();
        move || {
            Box::new(FakeBinaryExtractor {
                calls: std::sync::Arc::clone(&calls),
                extracted_text: t.clone(),
                panic_if_called: false,
            })
        }
    });

    let id = uuid::Uuid::now_v7().to_string();
    let locator = write_raw_source_file(&data_dir, &id, b"\x00\x01\x02 fake binary bytes \xff");
    insert_raw_source_locator(&engine, &nb.id.to_string(), FAKE_KIND, &locator, &id).await;

    // First ingest → indexed, extractor called once, sibling written.
    engine.ingest_source(&id, |_p| {}).await.unwrap();
    assert_eq!(engine_source_status(&engine, &id).await, "indexed");
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "extractor must run exactly once on the first ingest"
    );
    // The canonical buffer for a DERIVED kind is the `.extracted.txt` sibling.
    let sibling = data_dir.join("sources").join(format!("{id}.extracted.txt"));
    assert!(
        sibling.exists(),
        "derived kind must persist the .extracted.txt sibling"
    );
    let canonical = std::fs::read_to_string(&sibling).unwrap();
    assert_eq!(canonical, fake_text, "sibling holds the extracted_text");
    assert_chunk_byte_identity(&engine, &id, &canonical).await;

    // Second ingest, UNCHANGED raw bytes → no-op, extractor NOT called again.
    engine.ingest_source(&id, |_p| {}).await.unwrap();
    assert_eq!(engine_source_status(&engine, &id).await, "indexed");
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "unchanged-binary re-ingest must NOT re-run the extractor (raw-bytes hash matched)"
    );

    // Purge removes BOTH the locator AND the `.extracted.txt` sibling.
    engine.trash_source(&id).await.unwrap();
    engine.purge_source(&id).await.unwrap();
    clear_test_extractor_factory(FAKE_KIND);
    assert!(
        !std::path::Path::new(&locator).exists(),
        "purge must remove the original locator file"
    );
    assert!(
        !sibling.exists(),
        "purge must remove the .extracted.txt sibling"
    );
}

// ===========================================================================
// Step 4 — SourceAnchor persisted in a dedicated column (AC5)
// ===========================================================================

use lens_core::extract::SourceAnchor;

/// AC5 — migration 0004: the `chunks.source_anchor` column exists after
/// migrations are applied on a fresh DB (the migrator picks up `0004`).
#[tokio::test]
async fn migration_0004_adds_source_anchor_column() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;
    let rows = sqlx::query("PRAGMA table_info(chunks)")
        .fetch_all(&pool)
        .await
        .unwrap();
    let cols: std::collections::HashSet<String> = rows
        .into_iter()
        .map(|r| r.get::<String, _>("name"))
        .collect();
    assert!(
        cols.contains("source_anchor"),
        "migration 0004 must add chunks.source_anchor column; found: {cols:?}"
    );
    // enrichment is still there (not reused / removed).
    assert!(
        cols.contains("enrichment"),
        "chunks.enrichment must still exist after 0004"
    );
}

/// AC5 — anchor round-trip + enrichment stays NULL: ingest a text/MD source and
/// read back `source_anchor` (deserializable as `SourceAnchor`) and assert
/// `enrichment` is still NULL on those rows.
#[tokio::test]
async fn source_anchor_roundtrip_and_enrichment_null_for_text_source() {
    if !tokenizer_available().await {
        eprintln!(
            "skipping source_anchor_roundtrip_and_enrichment_null_for_text_source: no tokenizer (offline)"
        );
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;
    let nb = engine
        .create_notebook("anchor-rt-nb", None, None)
        .await
        .unwrap();
    let body = "# Anchor Test\n\nParagraph one.\n\nParagraph two.\n";
    let src = engine
        .add_text_source(&nb.id, "doc", body, "markdown")
        .await
        .unwrap();
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();
    assert_eq!(engine_source_status(&engine, &src.id).await, "indexed");

    let pool = engine.pool().await;
    let rows = sqlx::query(
        "SELECT source_anchor, enrichment FROM chunks WHERE source_id = ? ORDER BY rowid",
    )
    .bind(&src.id)
    .fetch_all(&pool)
    .await
    .unwrap();

    assert!(!rows.is_empty(), "expected chunks for the ingested source");

    for (i, row) in rows.iter().enumerate() {
        // enrichment must remain NULL (reserved for Phase-3 pass).
        let enrichment = row.get::<Option<String>, _>("enrichment");
        assert!(
            enrichment.is_none(),
            "chunk[{i}] enrichment must be NULL (reserved for Phase-3); got {enrichment:?}"
        );

        // source_anchor is set (text/MD yields SourceAnchor::Text for every block).
        let anchor_json = row.get::<Option<String>, _>("source_anchor");
        assert!(
            anchor_json.is_some(),
            "chunk[{i}] source_anchor must be non-NULL for a text/MD source"
        );
        let json = anchor_json.unwrap();
        let anchor: SourceAnchor = serde_json::from_str(&json).unwrap_or_else(|e| {
            panic!("chunk[{i}] source_anchor is not a valid SourceAnchor JSON: {e}\njson={json}")
        });
        // Text/MD blocks always carry SourceAnchor::Text.
        assert_eq!(
            anchor,
            SourceAnchor::Text,
            "chunk[{i}] source_anchor must be SourceAnchor::Text for a markdown source"
        );
    }
}

/// AC5 (forward-check) — PDF anchor maps to `chunks.page`: a `SourceAnchor::Pdf`
/// with `page: 3` (injected via the fake-binary extractor) is serialized and
/// stored, and the `chunks.page` column is populated with `3`.
///
/// This test exercises the anchor-to-page mapping unit-path even before the real
/// PDF extractor is wired up (Step 5), keeping the contract clear.
#[tokio::test]
async fn source_anchor_pdf_maps_page_to_chunk_page_column() {
    use lens_core::parse::Block;

    if !tokenizer_available().await {
        eprintln!(
            "skipping source_anchor_pdf_maps_page_to_chunk_page_column: no tokenizer (offline)"
        );
        return;
    }

    // Build a fake PDF extractor that returns a single block with a
    // SourceAnchor::Pdf { page: 3 }.
    struct FakePdfExtractor;
    impl lens_core::extract::Extractor for FakePdfExtractor {
        fn extract(
            &self,
            _raw: &[u8],
        ) -> Result<lens_core::extract::ExtractOutput, lens_core::LensError> {
            let text = "PDF page three paragraph content.\n".to_string();
            let block = Block {
                block_type: "paragraph".to_string(),
                section_path: String::new(),
                char_start: 0,
                char_end: text.len(),
                text: text.clone(),
            };
            Ok(lens_core::extract::ExtractOutput {
                extracted_text: text,
                blocks: vec![block],
                anchors: vec![SourceAnchor::Pdf {
                    page: 3,
                    bbox: [10.0, 20.0, 500.0, 700.0],
                }],
            })
        }
    }

    const PDF_KIND: &str = "fakepdf";
    set_test_extractor_factory(PDF_KIND, || Box::new(FakePdfExtractor));

    let (_dir, engine) = inject_counting_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine
        .create_notebook("pdf-page-nb", None, None)
        .await
        .unwrap();

    let id = uuid::Uuid::now_v7().to_string();
    let locator = write_raw_source_file(&data_dir, &id, b"fake pdf bytes");
    insert_raw_source_locator(&engine, &nb.id.to_string(), PDF_KIND, &locator, &id).await;

    engine.ingest_source(&id, |_p| {}).await.unwrap();
    clear_test_extractor_factory(PDF_KIND);
    assert_eq!(engine_source_status(&engine, &id).await, "indexed");

    let pool = engine.pool().await;
    let rows = sqlx::query(
        "SELECT source_anchor, page, enrichment FROM chunks WHERE source_id = ? ORDER BY rowid",
    )
    .bind(&id)
    .fetch_all(&pool)
    .await
    .unwrap();

    assert!(!rows.is_empty(), "expected chunks for the fake PDF source");

    for (i, row) in rows.iter().enumerate() {
        // enrichment still NULL.
        let enrichment = row.get::<Option<String>, _>("enrichment");
        assert!(
            enrichment.is_none(),
            "chunk[{i}] enrichment must be NULL; got {enrichment:?}"
        );

        // source_anchor round-trips as SourceAnchor::Pdf.
        let anchor_json = row
            .get::<Option<String>, _>("source_anchor")
            .unwrap_or_else(|| panic!("chunk[{i}] source_anchor must be non-NULL for PDF"));
        let anchor: SourceAnchor = serde_json::from_str(&anchor_json)
            .unwrap_or_else(|e| panic!("chunk[{i}] invalid anchor JSON: {e}\njson={anchor_json}"));
        assert!(
            matches!(anchor, SourceAnchor::Pdf { page: 3, .. }),
            "chunk[{i}] source_anchor must be SourceAnchor::Pdf {{ page: 3, .. }}, got {anchor:?}"
        );

        // chunks.page must be populated with 3 for PDF anchors.
        let page = row.get::<Option<i64>, _>("page");
        assert_eq!(
            page,
            Some(3),
            "chunk[{i}] chunks.page must be 3 for a SourceAnchor::Pdf {{ page: 3 }}"
        );
    }
}

/// Builds a tiny single-page PDF with NO text layer (image-only / scanned PDF
/// surrogate) as in-memory bytes, using `printpdf`. The REAL `PdfExtractor` will
/// extract empty text from it, exercising the end-to-end `needs_ocr` path.
fn build_no_text_layer_pdf_bytes() -> Vec<u8> {
    use printpdf::{Mm, PdfDocument};
    use std::io::BufWriter;
    let (doc, _page1, _layer1) =
        PdfDocument::new("no-text-fixture", Mm(210.0), Mm(297.0), "Layer 1");
    let mut buf = Vec::new();
    doc.save(&mut BufWriter::new(&mut buf))
        .expect("serialize no-text PDF fixture");
    buf
}

/// AC7 (end-to-end) — a real image-only / no-text-layer PDF ingested via the
/// `"pdf"` kind drives `run_ingest` through the REAL `PdfExtractor`, which yields
/// empty text, so the source ends at `status = needs_ocr` (Ok-with-status, NOT
/// Err, NOT indexed, zero chunks).
///
/// Skipped when libpdfium cannot bind on this platform (the vendored asset is the
/// macOS universal dylib; AC7 is gated on macOS dev / release).
#[tokio::test]
async fn pdf_image_only_source_sets_needs_ocr_end_to_end() {
    use lens_core::extract::extractor_for;

    if !tokenizer_available().await {
        eprintln!(
            "skipping pdf_image_only_source_sets_needs_ocr_end_to_end: no tokenizer (offline)"
        );
        return;
    }
    // Probe binding: skip (not fail) where the universal dylib can't load.
    let raw = build_no_text_layer_pdf_bytes();
    let extractor = extractor_for("pdf").expect("pdf extractor resolves");
    if extractor.extract(&raw).is_err() {
        eprintln!(
            "skipping pdf_image_only_source_sets_needs_ocr_end_to_end: libpdfium not bindable here"
        );
        return;
    }

    let (_dir, engine) = inject_counting_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine
        .create_notebook("pdf-ocr-nb", None, None)
        .await
        .unwrap();

    let id = uuid::Uuid::now_v7().to_string();
    // Write the PDF bytes to a real on-disk file the locator points at.
    let sources_dir = data_dir.join("sources");
    std::fs::create_dir_all(&sources_dir).unwrap();
    let path = sources_dir.join(format!("{id}.pdf"));
    std::fs::write(&path, &raw).unwrap();
    let locator = path.display().to_string();
    insert_raw_source_locator(&engine, &nb.id.to_string(), "pdf", &locator, &id).await;

    // The ingest must return Ok (NOT Err — an Err would flip status to `error`).
    let result = engine.ingest_source(&id, |_p| {}).await;
    assert!(
        result.is_ok(),
        "image-only PDF must return Ok-with-status, not Err: {result:?}"
    );

    // Status is needs_ocr (terminal-pending), not indexed and not error.
    assert_eq!(
        engine_source_status(&engine, &id).await,
        "needs_ocr",
        "image-only PDF must set status = needs_ocr"
    );

    // Nothing was indexed.
    let pool = engine.pool().await;
    let chunk_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chunks WHERE source_id = ?")
        .bind(&id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(chunk_count, 0, "needs_ocr source must index zero chunks");
}

/// AC4a + AC4c + AC5 (end-to-end) — a real text-layer PDF ingested via the `"pdf"`
/// kind reaches `indexed`, every chunk slices the canonical `.extracted.txt`
/// sibling byte-identically, the sentinel text is present, and every chunk's
/// `source_anchor` is a `SourceAnchor::Pdf` with a non-NULL `chunks.page`.
///
/// Skipped when libpdfium cannot bind on this platform.
#[tokio::test]
async fn pdf_text_layer_source_indexed_with_anchors_end_to_end() {
    use lens_core::extract::extractor_for;

    const SENTINEL: &str = "The quick brown fox jumps over the lazy dog.";

    if !tokenizer_available().await {
        eprintln!(
            "skipping pdf_text_layer_source_indexed_with_anchors_end_to_end: no tokenizer (offline)"
        );
        return;
    }

    // Build a text-layer PDF with a known sentinel.
    let raw = {
        use printpdf::{BuiltinFont, Mm, PdfDocument};
        use std::io::BufWriter;
        let (doc, page1, layer1) =
            PdfDocument::new("sentinel-fixture", Mm(210.0), Mm(297.0), "Layer 1");
        let layer = doc.get_page(page1).get_layer(layer1);
        let font = doc.add_builtin_font(BuiltinFont::Helvetica).unwrap();
        layer.use_text(SENTINEL, 14.0, Mm(20.0), Mm(270.0), &font);
        let mut buf = Vec::new();
        doc.save(&mut BufWriter::new(&mut buf)).unwrap();
        buf
    };

    let extractor = extractor_for("pdf").expect("pdf extractor resolves");
    if extractor.extract(&raw).is_err() {
        eprintln!(
            "skipping pdf_text_layer_source_indexed_with_anchors_end_to_end: libpdfium not bindable here"
        );
        return;
    }

    let (_dir, engine) = inject_counting_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine
        .create_notebook("pdf-text-nb", None, None)
        .await
        .unwrap();

    let id = uuid::Uuid::now_v7().to_string();
    let sources_dir = data_dir.join("sources");
    std::fs::create_dir_all(&sources_dir).unwrap();
    let path = sources_dir.join(format!("{id}.pdf"));
    std::fs::write(&path, &raw).unwrap();
    let locator = path.display().to_string();
    insert_raw_source_locator(&engine, &nb.id.to_string(), "pdf", &locator, &id).await;

    engine.ingest_source(&id, |_p| {}).await.unwrap();
    assert_eq!(engine_source_status(&engine, &id).await, "indexed");

    // The canonical buffer is the persisted `.extracted.txt` sibling.
    let sibling = sources_dir.join(format!("{id}.extracted.txt"));
    let canonical = std::fs::read_to_string(&sibling)
        .expect("a derived PDF source must persist its .extracted.txt sibling");

    // AC4c: sentinel present (whitespace-normalized — pdfium may resegment).
    let got: String = canonical.split_whitespace().collect::<Vec<_>>().join(" ");
    let want: String = SENTINEL.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(
        got.contains(&want),
        "sentinel missing from persisted canonical buffer; got {got:?}"
    );

    // AC4a: chunks slice the canonical buffer byte-identically.
    assert_chunk_byte_identity(&engine, &id, &canonical).await;

    // AC5: every chunk anchor is Pdf with a non-NULL page.
    let pool = engine.pool().await;
    let rows = sqlx::query("SELECT source_anchor, page FROM chunks WHERE source_id = ?")
        .bind(&id)
        .fetch_all(&pool)
        .await
        .unwrap();
    assert!(!rows.is_empty(), "expected chunks for the text-layer PDF");
    for (i, row) in rows.iter().enumerate() {
        let anchor_json = row
            .get::<Option<String>, _>("source_anchor")
            .unwrap_or_else(|| panic!("chunk[{i}] source_anchor must be non-NULL for PDF"));
        let anchor: SourceAnchor = serde_json::from_str(&anchor_json).unwrap();
        assert!(
            matches!(anchor, SourceAnchor::Pdf { .. }),
            "chunk[{i}] anchor must be SourceAnchor::Pdf, got {anchor:?}"
        );
        let page = row.get::<Option<i64>, _>("page");
        assert!(
            matches!(page, Some(p) if p >= 1),
            "chunk[{i}] chunks.page must be a populated 1-based page, got {page:?}"
        );
    }
}

// ===========================================================================
// add_file_source — real-binary DOCX/PDF end-to-end (REAL extractor, NOT the
// FakeBinaryExtractor seam): extension→kind detection, managed copy, real
// ingest, sibling + anchor + byte-identity + purge lifecycle.
// ===========================================================================

/// Builds a real `.docx` in memory: a Heading1, a body paragraph carrying a
/// known sentinel, and a 2×2 table. Mirrors the in-crate `build_fixture_docx`
/// helper but lives here so the integration test owns its own fixture.
fn build_e2e_docx_bytes(sentinel: &str) -> Vec<u8> {
    use docx_rs::{Docx, Paragraph, Run, Table, TableCell, TableRow};
    use std::io::Cursor;
    let docx = Docx::new()
        .add_paragraph(
            Paragraph::new()
                .add_run(Run::new().add_text("E2E Heading"))
                .style("Heading1"),
        )
        .add_paragraph(Paragraph::new().add_run(Run::new().add_text(sentinel)))
        .add_table(Table::new(vec![
            TableRow::new(vec![
                TableCell::new()
                    .add_paragraph(Paragraph::new().add_run(Run::new().add_text("Cell A1"))),
                TableCell::new()
                    .add_paragraph(Paragraph::new().add_run(Run::new().add_text("Cell A2"))),
            ]),
            TableRow::new(vec![
                TableCell::new()
                    .add_paragraph(Paragraph::new().add_run(Run::new().add_text("Cell B1"))),
                TableCell::new()
                    .add_paragraph(Paragraph::new().add_run(Run::new().add_text("Cell B2"))),
            ]),
        ]));
    let mut buf = Vec::new();
    docx.build()
        .pack(Cursor::new(&mut buf))
        .expect("fixture DOCX build failed");
    buf
}

/// AC (DOCX end-to-end, REAL extractor) — a real `.docx` added via
/// `add_file_source` (extension→kind = "docx", copied into managed storage) and
/// ingested through the REAL `DocxExtractor` reaches `indexed`; the
/// `{data_dir}/sources/{id}.extracted.txt` sibling exists with the sentinel;
/// every chunk carries a `SourceAnchor::Docx`; chunks slice the canonical sibling
/// byte-identically; and purge removes BOTH the managed original AND the sibling.
///
/// Runs everywhere (docx-rs is pure Rust — no platform-gated native dependency).
#[tokio::test]
async fn docx_file_source_indexed_with_anchors_end_to_end() {
    const SENTINEL: &str = "Sentinel body text for docx end-to-end extraction.";

    if !tokenizer_available().await {
        eprintln!(
            "skipping docx_file_source_indexed_with_anchors_end_to_end: no tokenizer (offline)"
        );
        return;
    }

    let (_dir, engine) = inject_counting_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine
        .create_notebook("docx-e2e-nb", None, None)
        .await
        .unwrap();

    // Write the `.docx` to a temp path OUTSIDE managed storage; `add_file_source`
    // copies it into `{data_dir}/sources/{id}.docx`.
    let src_dir = tempfile::tempdir().unwrap();
    let src_path = src_dir.path().join("report.docx");
    std::fs::write(&src_path, build_e2e_docx_bytes(SENTINEL)).unwrap();

    let source = engine
        .add_file_source(&nb.id, &src_path, None)
        .await
        .expect("add_file_source for a .docx must succeed");
    assert_eq!(source.kind, "docx", "extension .docx → kind docx");
    assert_eq!(source.status, "queued", "new file source is queued");
    assert_eq!(
        source.title, "report.docx",
        "title defaults to the file name"
    );
    // The managed copy lives under {data_dir}/sources and is what the ingest reads.
    assert!(
        std::path::Path::new(&source.locator).exists(),
        "managed copy must exist after add_file_source: {}",
        source.locator
    );

    engine.ingest_source(&source.id, |_p| {}).await.unwrap();
    assert_eq!(engine_source_status(&engine, &source.id).await, "indexed");

    // The canonical buffer is the persisted `.extracted.txt` sibling.
    let sibling = data_dir
        .join("sources")
        .join(format!("{}.extracted.txt", source.id));
    let canonical = std::fs::read_to_string(&sibling)
        .expect("a derived DOCX source must persist its .extracted.txt sibling");
    assert!(
        canonical.contains(SENTINEL),
        "sentinel missing from the persisted canonical buffer; got {canonical:?}"
    );

    // Chunks slice the canonical buffer byte-identically.
    assert_chunk_byte_identity(&engine, &source.id, &canonical).await;

    // Every chunk anchor is a SourceAnchor::Docx.
    let pool = engine.pool().await;
    let rows = sqlx::query("SELECT source_anchor FROM chunks WHERE source_id = ?")
        .bind(&source.id)
        .fetch_all(&pool)
        .await
        .unwrap();
    assert!(!rows.is_empty(), "expected chunks for the DOCX source");
    for (i, row) in rows.iter().enumerate() {
        let anchor_json = row
            .get::<Option<String>, _>("source_anchor")
            .unwrap_or_else(|| panic!("chunk[{i}] source_anchor must be non-NULL for DOCX"));
        let anchor: SourceAnchor = serde_json::from_str(&anchor_json).unwrap();
        assert!(
            matches!(anchor, SourceAnchor::Docx { .. }),
            "chunk[{i}] anchor must be SourceAnchor::Docx, got {anchor:?}"
        );
    }

    // Purge removes BOTH the managed original AND the `.extracted.txt` sibling.
    let managed = std::path::PathBuf::from(&source.locator);
    engine.trash_source(&source.id).await.unwrap();
    engine.purge_source(&source.id).await.unwrap();
    assert!(
        !managed.exists(),
        "purge must remove the managed DOCX original"
    );
    assert!(
        !sibling.exists(),
        "purge must remove the .extracted.txt sibling"
    );
}

/// AC (PDF end-to-end, REAL extractor) — a real text-layer PDF added via
/// `add_file_source` (extension→kind = "pdf", copied into managed storage) and
/// ingested through the REAL `PdfExtractor` reaches `indexed`; the
/// `.extracted.txt` sibling is written with the sentinel; chunks slice the
/// canonical sibling byte-identically; and every chunk carries a
/// `SourceAnchor::Pdf` with a non-NULL `chunks.page`.
///
/// `#[cfg(target_os = "macos")]`-gated, mirroring the `pdf.rs` extractor build
/// gating (the vendored libpdfium asset is the macOS universal dylib).
#[cfg(target_os = "macos")]
#[tokio::test]
async fn pdf_file_source_indexed_with_anchors_end_to_end() {
    use lens_core::extract::extractor_for;

    const SENTINEL: &str = "The quick brown fox jumps over the lazy dog.";

    if !tokenizer_available().await {
        eprintln!(
            "skipping pdf_file_source_indexed_with_anchors_end_to_end: no tokenizer (offline)"
        );
        return;
    }

    // Build a text-layer PDF with a known sentinel.
    let raw = {
        use printpdf::{BuiltinFont, Mm, PdfDocument};
        use std::io::BufWriter;
        let (doc, page1, layer1) =
            PdfDocument::new("pdf-file-source-fixture", Mm(210.0), Mm(297.0), "Layer 1");
        let layer = doc.get_page(page1).get_layer(layer1);
        let font = doc.add_builtin_font(BuiltinFont::Helvetica).unwrap();
        layer.use_text(SENTINEL, 14.0, Mm(20.0), Mm(270.0), &font);
        let mut buf = Vec::new();
        doc.save(&mut BufWriter::new(&mut buf)).unwrap();
        buf
    };

    // Probe binding: skip (not fail) where the universal dylib can't load.
    let extractor = extractor_for("pdf").expect("pdf extractor resolves");
    if extractor.extract(&raw).is_err() {
        eprintln!(
            "skipping pdf_file_source_indexed_with_anchors_end_to_end: libpdfium not bindable here"
        );
        return;
    }

    let (_dir, engine) = inject_counting_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine
        .create_notebook("pdf-file-nb", None, None)
        .await
        .unwrap();

    // Write the PDF to a temp path OUTSIDE managed storage; add_file_source copies.
    let src_dir = tempfile::tempdir().unwrap();
    let src_path = src_dir.path().join("paper.pdf");
    std::fs::write(&src_path, &raw).unwrap();

    let source = engine
        .add_file_source(&nb.id, &src_path, Some("Custom Title"))
        .await
        .expect("add_file_source for a .pdf must succeed");
    assert_eq!(source.kind, "pdf", "extension .pdf → kind pdf");
    assert_eq!(source.status, "queued");
    assert_eq!(source.title, "Custom Title", "supplied title is honored");

    engine.ingest_source(&source.id, |_p| {}).await.unwrap();
    assert_eq!(engine_source_status(&engine, &source.id).await, "indexed");

    // The canonical buffer is the persisted `.extracted.txt` sibling.
    let sibling = data_dir
        .join("sources")
        .join(format!("{}.extracted.txt", source.id));
    let canonical = std::fs::read_to_string(&sibling)
        .expect("a derived PDF source must persist its .extracted.txt sibling");

    // Sentinel present (whitespace-normalized — pdfium may resegment).
    let got: String = canonical.split_whitespace().collect::<Vec<_>>().join(" ");
    let want: String = SENTINEL.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(
        got.contains(&want),
        "sentinel missing from persisted canonical buffer; got {got:?}"
    );

    // Chunks slice the canonical buffer byte-identically.
    assert_chunk_byte_identity(&engine, &source.id, &canonical).await;

    // Every chunk anchor is Pdf with a non-NULL, 1-based page.
    let pool = engine.pool().await;
    let rows = sqlx::query("SELECT source_anchor, page FROM chunks WHERE source_id = ?")
        .bind(&source.id)
        .fetch_all(&pool)
        .await
        .unwrap();
    assert!(!rows.is_empty(), "expected chunks for the text-layer PDF");
    for (i, row) in rows.iter().enumerate() {
        let anchor_json = row
            .get::<Option<String>, _>("source_anchor")
            .unwrap_or_else(|| panic!("chunk[{i}] source_anchor must be non-NULL for PDF"));
        let anchor: SourceAnchor = serde_json::from_str(&anchor_json).unwrap();
        assert!(
            matches!(anchor, SourceAnchor::Pdf { .. }),
            "chunk[{i}] anchor must be SourceAnchor::Pdf, got {anchor:?}"
        );
        let page = row.get::<Option<i64>, _>("page");
        assert!(
            matches!(page, Some(p) if p >= 1),
            "chunk[{i}] chunks.page must be a populated 1-based page, got {page:?}"
        );
    }
}
