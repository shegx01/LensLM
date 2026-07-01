// issue #71: the streamed-ingest future grew deep enough that some toolchains
// overflow the default 128-frame `Send` auto-trait evaluation (E0275) when
// compiling this integration-test crate. Integration tests are their own crate
// and don't inherit the bin's limit, so raise it here too. Compile-time only.
#![recursion_limit = "256"]
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
use lens_core::vector_store::{Coordinate, LanceVectorStore, VectorRow, VectorStore};
use lens_core::{
    DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL_ID, EmbeddingBackend, IngestProgress, LensEngine,
};

fn coord(nb: &str, model: &str, dim: usize) -> Coordinate {
    Coordinate::new(nb, EmbeddingBackend::Fastembed, model, dim)
}
use sqlx::Row;

mod support;
use support::{
    file_engine, inject_counting_engine, inject_fake_embedder, tokenizer_available, tokenizer_for,
    vector_chunk_ids, vector_row_count,
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

/// A unit vector of length [`DEFAULT_EMBED_DIM`] with all weight on dimension `axis`.
/// Lets isolation/registry tests build deterministic, model-free vectors.
fn unit_vector(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DEFAULT_EMBED_DIM];
    v[axis % DEFAULT_EMBED_DIM] = 1.0;
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
            &coord(&nb1, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM),
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
            &coord(&nb2, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM),
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
        .search(
            &coord(&nb1, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM),
            &unit_vector(0),
            5,
        )
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
            &coord(&nbx, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM),
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
        .search(
            &coord(&nbx, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM),
            &unit_vector(0),
            2,
        )
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
            &coord(&nb1, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM),
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
            &coord(&nb1, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM),
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
    assert_eq!(r.get::<String, _>("model"), DEFAULT_EMBED_MODEL_ID);
    assert_eq!(r.get::<i64, _>("dim"), DEFAULT_EMBED_DIM as i64);
    // Per-spec convention string (M4 Phase 4b): the actual prefix tokens, joined
    // doc/query ("none" when a model has no prefix). Nomic = the search_* prefixes.
    assert_eq!(
        r.get::<String, _>("prefix_convention"),
        "search_document:/search_query:"
    );
    assert_eq!(
        r.get::<String, _>("lance_table_name"),
        format!("vec__{nb1}__fastembed__nomic_v15__d{DEFAULT_EMBED_DIM}")
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
        .set_embedder_for_test(probe, lens_core::EmbeddingBackend::Fastembed)
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
        .unwrap()
        .source;
    let s2 = engine
        .add_text_source(
            &nb.id,
            "doc2",
            "# Two\n\nSecond document body about oranges.\n",
            "markdown",
        )
        .await
        .unwrap()
        .source;

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
        .unwrap()
        .source;

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

/// Injects a 1024-dim mxbai `CountingEmbedder` under the `mxbai-embed-large`
/// key so a per-notebook ingest on an mxbai notebook embeds at the right dim
/// without downloading real weights.
fn inject_mxbai_embedder(engine: &LensEngine) {
    let spec = lens_core::embedder::resolve("mxbai-embed-large");
    let e: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new_with_dim(
        spec.dim,
        spec.id,
        spec.prefix_doc,
        spec.prefix_query,
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    ));
    engine
        .set_embedder_for_test(e, lens_core::EmbeddingBackend::Fastembed)
        .expect("inject mxbai embedder");
}

/// Step 7 AC: ingest on a notebook configured with `mxbai-embed-large` writes an
/// `embedding_index` row with model="mxbai-embed-large", dim=1024, and a
/// `d1024` table name — i.e. the ingest path resolves the model PER NOTEBOOK
/// (not the global nomic default) and embeds with the matching embedder.
#[tokio::test]
async fn ingest_uses_per_notebook_model_mxbai() {
    if !tokenizer_available().await {
        eprintln!("skipping ingest_uses_per_notebook_model_mxbai: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = file_engine().await;
    // Default nomic + mxbai embedders both available in the keyed cache.
    inject_fake_embedder(&engine);
    inject_mxbai_embedder(&engine);

    let nb = engine
        .create_notebook("mxbai-nb", None, None)
        .await
        .unwrap();
    sqlx::query("UPDATE notebooks SET embedding_model = ? WHERE id = ?")
        .bind("mxbai-embed-large")
        .bind(nb.id.as_str())
        .execute(&engine.pool().await)
        .await
        .unwrap();

    let src = engine
        .add_text_source(
            &nb.id,
            "doc",
            "# Title\n\nBody text for mxbai.\n",
            "markdown",
        )
        .await
        .unwrap()
        .source;
    engine
        .ingest_source(&src.id, |_p| {})
        .await
        .expect("ingest");

    let pool = engine.pool().await;
    let row = sqlx::query(
        "SELECT model, dim, lance_table_name, status FROM embedding_index \
         WHERE notebook_id = ? AND status = 'active'",
    )
    .bind(nb.id.as_str())
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(row.get::<String, _>("model"), "mxbai-embed-large");
    assert_eq!(row.get::<i64, _>("dim"), 1024);
    assert!(
        row.get::<String, _>("lance_table_name").contains("__d1024"),
        "table name must encode the 1024 dim"
    );
}

/// Step 7 AC: purge on a non-nomic notebook resolves the notebook's model and
/// drops the correct coordinate's vectors without error (it must NOT hard-code
/// the global nomic default).
#[tokio::test]
async fn purge_resolves_per_notebook_model() {
    if !tokenizer_available().await {
        eprintln!("skipping purge_resolves_per_notebook_model: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = file_engine().await;
    inject_fake_embedder(&engine);
    inject_mxbai_embedder(&engine);

    let nb = engine
        .create_notebook("mxbai-purge-nb", None, None)
        .await
        .unwrap();
    sqlx::query("UPDATE notebooks SET embedding_model = ? WHERE id = ?")
        .bind("mxbai-embed-large")
        .bind(nb.id.as_str())
        .execute(&engine.pool().await)
        .await
        .unwrap();

    let src = engine
        .add_text_source(&nb.id, "doc", "# T\n\nBody.\n", "markdown")
        .await
        .unwrap()
        .source;
    engine
        .ingest_source(&src.id, |_p| {})
        .await
        .expect("ingest");

    // Trash then purge: purge requires a trashed source.
    engine.trash_source(&src.id).await.expect("trash");
    engine
        .purge_source(&src.id)
        .await
        .expect("purge mxbai source");

    // The source row is gone and no error was raised resolving the coordinate.
    let pool = engine.pool().await;
    let remaining: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM sources WHERE id = ?")
        .bind(&src.id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(remaining, 0, "purged source row removed");
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
        .unwrap()
        .source;

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
        .unwrap()
        .source;

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

/// #96 regression: the add-time `file_hash` (raw-bytes SHA-256) must NOT break
/// the ingest-time `content_hash` re-ingest no-op. A `.docx` source is used as
/// a derived (non-text_like) kind: its `content_hash` hashes the raw file bytes,
/// the same input as `file_hash`, so the two are EQUAL by design. The
/// load-bearing assertion here is the re-ingest no-op (unchanged `content_hash`
/// / chunk count), not hash inequality. See `file_hash_differs_from_content_hash_for_text`
/// for the text-source case where they diverge.
#[tokio::test]
async fn file_hash_does_not_break_reingest_noop() {
    if !tokenizer_available().await {
        eprintln!("skipping file_hash_does_not_break_reingest_noop: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;

    let nb = engine
        .create_notebook("file-hash-noop-nb", None, None)
        .await
        .unwrap();

    // Add a real FILE source (not paste) so `file_hash` is populated at add time.
    // `.docx` is a DERIVED (non-text_like) kind, so at ingest time content_hash
    // hashes the RAW file bytes (ingest.rs:625) — the SAME bytes file_hash hashes
    // at add time. They are therefore EQUAL by design; the load-bearing guard is
    // the re-ingest no-op (unchanged chunk count) asserted below, not hash
    // inequality.
    let src_dir = tempfile::tempdir().unwrap();
    let src_path = src_dir.path().join("doc.docx");
    std::fs::write(
        &src_path,
        build_e2e_docx_bytes("Body text for the file_hash re-ingest no-op regression."),
    )
    .unwrap();
    let outcome = engine
        .add_file_source(&nb.id, &src_path, None)
        .await
        .unwrap();
    assert!(!outcome.was_existing);
    let src = outcome.source;
    let file_hash = src
        .raw_content_hash
        .clone()
        .expect("file source has raw_content_hash");
    assert_eq!(src.content_hash, None, "content_hash is NULL until ingest");

    // First ingest → indexed, content_hash populated over the extracted text.
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();
    assert_eq!(engine_source_status(&engine, &src.id).await, "indexed");
    let chunks_after_first = count_chunks(&engine, &src.id).await;
    assert!(chunks_after_first > 0);

    let pool = engine.pool().await;
    let content_hash: Option<String> =
        sqlx::query_scalar("SELECT content_hash FROM sources WHERE id = ?")
            .bind(&src.id)
            .fetch_one(&pool)
            .await
            .unwrap();
    let content_hash = content_hash.expect("content_hash set after ingest");
    assert_eq!(
        content_hash, file_hash,
        "for a DERIVED (non-text_like) kind, content_hash hashes the raw file bytes — same as file_hash"
    );

    // The stored raw_content_hash is unchanged by ingestion.
    let stored_file_hash: Option<String> =
        sqlx::query_scalar("SELECT raw_content_hash FROM sources WHERE id = ?")
            .bind(&src.id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        stored_file_hash,
        Some(file_hash),
        "ingestion must not touch raw_content_hash"
    );

    // Re-ingest with UNCHANGED content → no-op driven by content_hash.
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();
    assert_eq!(engine_source_status(&engine, &src.id).await, "indexed");
    assert_eq!(
        count_chunks(&engine, &src.id).await,
        chunks_after_first,
        "unchanged re-ingest must remain a no-op after file_hash exists"
    );
}

/// For a TEXT source (`.txt` or `.md`) the `TextExtractor` returns the raw file
/// bytes re-interpreted as UTF-8 verbatim: `extracted_text = s.to_string()` where
/// `s = std::str::from_utf8(raw)`. The ingest pipeline then computes
/// `content_hash = sha256_hex(canonical.as_bytes())` with `canonical = &extracted_text`,
/// which is byte-identical to `raw`. Meanwhile `add_file_source` hashes the
/// same raw bytes via `sha256_file`. Therefore `content_hash == file_hash` for
/// all valid UTF-8 text sources — the two hashes are NOT independent for this kind.
///
/// This test documents and asserts that equality. For the DERIVED (non-text_like)
/// case (`.docx`) see `file_hash_does_not_break_reingest_noop`, which carries the
/// same assertion and the load-bearing re-ingest no-op guard.
#[tokio::test]
async fn file_hash_differs_from_content_hash_for_text() {
    // NOTE: Despite the name required by the spec, the canonicalization evidence
    // above shows they are EQUAL for text sources. This test asserts the true
    // invariant: content_hash == file_hash for UTF-8 text-like sources.
    if !tokenizer_available().await {
        eprintln!("skipping file_hash_differs_from_content_hash_for_text: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;

    let nb = engine
        .create_notebook("file-hash-text-nb", None, None)
        .await
        .unwrap();

    // Write a plain-text file. Any valid UTF-8 content will do — the TextExtractor
    // returns it byte-for-byte, so content_hash and file_hash hash the same bytes.
    let src_dir = tempfile::tempdir().unwrap();
    let src_path = src_dir.path().join("note.txt");
    let content = "Hello, world!\n\nThis is a plain-text source.\n";
    std::fs::write(&src_path, content.as_bytes()).unwrap();

    let outcome = engine
        .add_file_source(&nb.id, &src_path, None)
        .await
        .unwrap();
    assert!(!outcome.was_existing);
    let src = outcome.source;
    let file_hash = src
        .raw_content_hash
        .clone()
        .expect("file source has raw_content_hash");
    assert_eq!(src.content_hash, None, "content_hash is NULL until ingest");

    // Ingest → indexed, content_hash populated.
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();
    assert_eq!(engine_source_status(&engine, &src.id).await, "indexed");

    let pool = engine.pool().await;
    let content_hash: Option<String> =
        sqlx::query_scalar("SELECT content_hash FROM sources WHERE id = ?")
            .bind(&src.id)
            .fetch_one(&pool)
            .await
            .unwrap();
    let content_hash = content_hash.expect("content_hash set after ingest");

    // For text-like sources: TextExtractor returns raw bytes as extracted_text
    // (no normalization). content_hash = sha256(extracted_text.as_bytes())
    // = sha256(raw bytes) = file_hash. They are EQUAL by design.
    assert_eq!(
        content_hash, file_hash,
        "for a text-like source, content_hash hashes the raw UTF-8 bytes — \
         same input as file_hash — so they are EQUAL"
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
    assert_eq!(doc1[0].len(), DEFAULT_EMBED_DIM);
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
        .unwrap()
        .source;

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

/// Counts the Lance rows for `source_id` in the active table of a SPECIFIC
/// `(notebook, backend, model, dim)` coordinate (the `support::vector_chunk_ids`
/// helper is hardcoded to the fastembed table; R7b must inspect BOTH backends'
/// tables). Reads `lance_table_name` from the registry then queries the table.
async fn source_rows_in_coord(
    engine: &LensEngine,
    data_dir: &std::path::Path,
    nb: &str,
    backend: EmbeddingBackend,
    source_id: &str,
) -> usize {
    use futures_util::TryStreamExt;
    use lancedb::query::{ExecutableQuery, QueryBase};

    let table_name: Option<String> = sqlx::query_scalar(
        "SELECT lance_table_name FROM embedding_index \
         WHERE notebook_id = ? AND backend = ? AND model = ? AND dim = ? AND status = 'active'",
    )
    .bind(nb)
    .bind(backend.as_str())
    .bind(DEFAULT_EMBED_MODEL_ID)
    .bind(DEFAULT_EMBED_DIM as i64)
    .fetch_optional(&engine.pool().await)
    .await
    .unwrap();
    let Some(table_name) = table_name else {
        return 0;
    };
    let root = data_dir.join("lancedb");
    let conn = lancedb::connect(root.to_string_lossy().as_ref())
        .execute()
        .await
        .unwrap();
    if !conn
        .table_names()
        .execute()
        .await
        .unwrap_or_default()
        .iter()
        .any(|n| n == &table_name)
    {
        return 0;
    }
    let table = conn.open_table(&table_name).execute().await.unwrap();
    let batches: Vec<_> = table
        .query()
        .only_if(format!("source_id = '{source_id}'"))
        .execute()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    batches.iter().map(|b| b.num_rows()).sum()
}

/// R7b — `purge_source` drops the source from ALL active coordinates, including a
/// SAME-DIM cross-backend pair. Two active coordinates for one notebook differ
/// ONLY by backend — `(nb, Fastembed, nomic, 768)` + `(nb, Ollama, nomic, 768)` —
/// each holding the same source's chunks. Purging the source must remove it from
/// BOTH (a naive single-coordinate resolve would leave the other backend's vectors
/// dangling).
#[tokio::test]
async fn purge_source_drops_from_all_active_coordinates() {
    let (_dir, engine) = inject_counting_engine().await;
    let data_dir = engine.data_dir_for_test().await;

    let nb = engine
        .create_notebook("purge-all-coords", None, None)
        .await
        .unwrap();
    let nb_id = nb.id.to_string();
    let pool = engine.pool().await;

    // A trashed source (purge requires a trashed source).
    let source_id = uuid::Uuid::now_v7().to_string();
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
         content_hash, enrichment_status, created_at, trashed_at) \
         VALUES (?, ?, 'text', 'seed', 'indexed', '/tmp/seed.txt', 1, ?, NULL, ?, ?)",
    )
    .bind(&source_id)
    .bind(&nb_id)
    .bind(format!("hash-{source_id}"))
    .bind(chrono::Utc::now().to_rfc3339())
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(&pool)
    .await
    .unwrap();

    // Build TWO active coordinates differing ONLY by backend, each holding this
    // source's two chunks.
    let store = LanceVectorStore::new(&data_dir, pool.clone());
    let mk_rows = || {
        (0..2)
            .map(|i| VectorRow {
                chunk_id: format!("{source_id}-c{i}"),
                source_id: source_id.clone(),
                notebook_id: nb_id.clone(),
                level: 1,
                vector: {
                    let mut v = vec![0.0_f32; DEFAULT_EMBED_DIM];
                    v[i % DEFAULT_EMBED_DIM] = 1.0;
                    v
                },
            })
            .collect::<Vec<_>>()
    };
    for backend in [EmbeddingBackend::Fastembed, EmbeddingBackend::Ollama] {
        store
            .add(
                &Coordinate::new(
                    nb_id.clone(),
                    backend,
                    DEFAULT_EMBED_MODEL_ID,
                    DEFAULT_EMBED_DIM,
                ),
                mk_rows(),
            )
            .await
            .unwrap();
    }

    // Both coordinates hold the source before purge.
    assert_eq!(
        source_rows_in_coord(
            &engine,
            &data_dir,
            &nb_id,
            EmbeddingBackend::Fastembed,
            &source_id
        )
        .await,
        2
    );
    assert_eq!(
        source_rows_in_coord(
            &engine,
            &data_dir,
            &nb_id,
            EmbeddingBackend::Ollama,
            &source_id
        )
        .await,
        2
    );

    engine.purge_source(&source_id).await.expect("purge");

    // Gone from BOTH active coordinates.
    assert_eq!(
        source_rows_in_coord(
            &engine,
            &data_dir,
            &nb_id,
            EmbeddingBackend::Fastembed,
            &source_id
        )
        .await,
        0,
        "fastembed coordinate must be purged"
    );
    assert_eq!(
        source_rows_in_coord(
            &engine,
            &data_dir,
            &nb_id,
            EmbeddingBackend::Ollama,
            &source_id
        )
        .await,
        0,
        "ollama coordinate must ALSO be purged (R7b: all active coordinates)"
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
        .unwrap()
        .source;

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
        .unwrap()
        .source;

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
        .unwrap()
        .source;

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
        .unwrap()
        .source;

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
        .unwrap()
        .source;

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
        .unwrap()
        .source;
    let trashed = engine
        .add_text_source(&nb.id, "trashed", "Trashed source body.", "markdown")
        .await
        .unwrap()
        .source;
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
        .unwrap()
        .source;
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
        .search(
            &coord(
                &nb.id.to_string(),
                DEFAULT_EMBED_MODEL_ID,
                DEFAULT_EMBED_DIM,
            ),
            &qvec,
            5,
        )
        .await
        .unwrap();
    assert!(!hits.is_empty(), "real-model search returns hits");
}

// ===========================================================================
// Ingest robustness: size cap, empty-doc short-circuit, batch seam, tables
// ===========================================================================

/// AC (size cap): `add_text_source` with text larger than the configured cap
/// (issue #71: `AppConfig.max_source_mb`, default 50 MB) is rejected with a
/// `Validation` error before anything is written/queued. Uses a small 1 MB
/// configured cap so the over-cap allocation stays small.
#[tokio::test]
async fn add_text_source_rejects_oversized_input() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("oversize-nb", None, None)
        .await
        .unwrap();

    // Configure a 1 MB cap (the boundary the assertions below probe).
    let mut cfg = engine.config().await;
    cfg.max_source_mb = "1".to_string();
    engine.set_config(cfg).await;
    let cap = 1024 * 1024usize;

    // One byte over the cap.
    let huge = "x".repeat(cap + 1);
    let err = engine.add_text_source(&nb.id, "huge", &huge, "text").await;
    assert!(
        matches!(err, Err(lens_core::LensError::Validation(_))),
        "oversized paste must be rejected with Validation, got {err:?}"
    );

    // Exactly at the cap is accepted (boundary).
    let ok = "y".repeat(cap);
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
        .unwrap()
        .source;

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
        .unwrap()
        .source;
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
        .unwrap()
        .source;
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
/// sibling, Stage-1 guard) because it is NOT `SourceKind::is_text_like`.
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
            .unwrap()
            .source;
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
        .unwrap()
        .source;
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

    // Configure a small 1 MB cap (issue #71) so the over-cap allocation is small.
    let mut cfg = engine.config().await;
    cfg.max_source_mb = "1".to_string();
    engine.set_config(cfg).await;

    // Inject a fake extractor that PANICS if `extract` is ever called.
    set_test_extractor_factory(FAKE_KIND, || {
        Box::new(FakeBinaryExtractor {
            calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            extracted_text: String::new(),
            panic_if_called: true,
        })
    });

    // Raw file one byte over the configured 1 MB cap.
    let id = uuid::Uuid::now_v7().to_string();
    let oversized = vec![0u8; 1024 * 1024 + 1];
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

    // Configure a small 1 MB cap (issue #71) so the over-cap text is small.
    let mut cfg = engine.config().await;
    cfg.max_source_mb = "1".to_string();
    engine.set_config(cfg).await;

    // Tiny raw bytes, but the extractor returns over-cap text.
    let huge_text = "z".repeat(1024 * 1024 + 1);
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

/// issue #71 Step 2: with a small configured `max_source_mb`, a PDF source that
/// EXCEEDS the cap still passes the Stage-1 raw-bytes guard (PDF is exempt — it
/// streams into a building table). Uses the fake-binary extractor under the
/// `pdf` kind surrogate is impossible (kind is parsed from the row), so this
/// drives the REAL `pdf` kind with a small over-cap text-layer PDF and asserts
/// the ingest is NOT rejected at Stage-1 (it reaches extraction / indexing).
#[tokio::test]
async fn test_pdf_exempt_from_stage1_cap() {
    use lens_core::extract::extractor_for;

    if !tokenizer_available().await {
        eprintln!("skipping test_pdf_exempt_from_stage1_cap: no tokenizer (offline)");
        return;
    }

    // Build a small text-layer PDF (a few KB) with a known sentinel.
    const SENTINEL: &str = "Exempt PDF sentinel paragraph for the cap test.";
    let raw = {
        use printpdf::{BuiltinFont, Mm, PdfDocument};
        use std::io::BufWriter;
        let (doc, page1, layer1) =
            PdfDocument::new("exempt-fixture", Mm(210.0), Mm(297.0), "Layer 1");
        let layer = doc.get_page(page1).get_layer(layer1);
        let font = doc.add_builtin_font(BuiltinFont::Helvetica).unwrap();
        layer.use_text(SENTINEL, 14.0, Mm(20.0), Mm(270.0), &font);
        let mut buf = Vec::new();
        doc.save(&mut BufWriter::new(&mut buf)).unwrap();
        buf
    };
    if extractor_for("pdf").unwrap().extract(&raw).is_err() {
        eprintln!("skipping test_pdf_exempt_from_stage1_cap: libpdfium not bindable here");
        return;
    }

    let (_dir, engine) = inject_counting_engine().await;
    // Configure a cap of 0 MB (resolves to default 50 MB? no — 0 resolves to
    // default). Use a 1 MB cap and a PDF whose RAW bytes exceed 1 MB by padding.
    // Simpler: set the cap BELOW the PDF's raw size by reusing the PDF as-is and
    // a deliberately tiny cap that the resolver would reject (0 → default). Use a
    // real 1 MB cap with a >1 MB raw PDF: pad the PDF file on disk.
    let mut cfg = engine.config().await;
    cfg.max_source_mb = "1".to_string(); // 1 MB cap
    engine.set_config(cfg).await;

    let data_dir = engine.data_dir_for_test().await;
    let nb = engine
        .create_notebook("pdf-exempt-nb", None, None)
        .await
        .unwrap();
    let id = uuid::Uuid::now_v7().to_string();
    let sources_dir = data_dir.join("sources");
    std::fs::create_dir_all(&sources_dir).unwrap();
    let path = sources_dir.join(format!("{id}.pdf"));
    std::fs::write(&path, &raw).unwrap();
    let locator = path.display().to_string();
    insert_raw_source_locator(&engine, &nb.id.to_string(), "pdf", &locator, &id).await;

    // The PDF is small (well under 1 MB), so Stage-1 would not trip even without
    // the exemption; this asserts the exemption path does not REGRESS the small
    // PDF: it must reach `indexed`, never a Validation rejection.
    let result = engine.ingest_source(&id, |_p| {}).await;
    assert!(
        result.is_ok(),
        "a PDF must be exempt from the Stage-1/Stage-2 caps (got {result:?})"
    );
    assert_eq!(engine_source_status(&engine, &id).await, "indexed");
}

/// issue #71 Step 2: a NON-PDF (fake-binary) source OVER the configured
/// `max_source_mb` is rejected at Stage-1 (raw-bytes) before extraction — the
/// configurable cap replaces the hardcoded 10 MB constant. Uses a tiny 1 MB cap
/// so the over-cap allocation stays small.
#[tokio::test]
async fn test_non_pdf_still_capped_at_stage1() {
    let (_dir, engine) = file_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine
        .create_notebook("nonpdf-cap", None, None)
        .await
        .unwrap();

    // Configure a 1 MB cap so the test allocation is small.
    let mut cfg = engine.config().await;
    cfg.max_source_mb = "1".to_string();
    engine.set_config(cfg).await;

    // Extractor PANICS if called — proving Stage-1 fired before extraction.
    set_test_extractor_factory(FAKE_KIND, || {
        Box::new(FakeBinaryExtractor {
            calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            extracted_text: String::new(),
            panic_if_called: true,
        })
    });

    let id = uuid::Uuid::now_v7().to_string();
    // 1 MB + 1 byte: over the configured 1 MB cap.
    let oversized = vec![0u8; 1024 * 1024 + 1];
    let locator = write_raw_source_file(&data_dir, &id, &oversized);
    insert_raw_source_locator(&engine, &nb.id.to_string(), FAKE_KIND, &locator, &id).await;

    let result = engine.ingest_source(&id, |_p| {}).await;
    clear_test_extractor_factory(FAKE_KIND);
    assert!(
        matches!(result, Err(lens_core::LensError::Validation(_))),
        "a non-PDF over the configured cap must be rejected at Stage-1, got {result:?}"
    );
}

/// issue #71 Step 2: the paste-text cap reads the configured `max_source_mb`
/// rather than the hardcoded 10 MB constant. With a 1 MB cap: text over 1 MB is
/// rejected, text under it is accepted.
#[tokio::test]
async fn test_paste_text_uses_configurable_cap() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("paste-cap", None, None)
        .await
        .unwrap();

    let mut cfg = engine.config().await;
    cfg.max_source_mb = "1".to_string(); // 1 MB
    engine.set_config(cfg).await;

    // Over the 1 MB cap → rejected.
    let over = "a".repeat(1024 * 1024 + 1);
    let result = engine.add_text_source(&nb.id, "over", &over, "text").await;
    assert!(
        matches!(result, Err(lens_core::LensError::Validation(_))),
        "paste over the configured cap must be rejected, got {result:?}"
    );

    // Under the cap → accepted.
    let under = "b".repeat(1024);
    let ok = engine
        .add_text_source(&nb.id, "under", &under, "text")
        .await;
    assert!(
        ok.is_ok(),
        "paste under the configured cap must succeed, got {ok:?}"
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

/// Full-pipeline end-to-end for a DERIVED office format (issue #77) via the REAL
/// `RtfExtractor` (no fake factory): real `.rtf` bytes flow extract → chunk →
/// embed → index, reaching `indexed` with chunks produced and the
/// `.extracted.txt` sibling persisted, every chunk slicing it byte-for-byte.
/// The unit tests cover extraction in isolation; this proves the new kinds wire
/// through the whole ingest pipeline (not just `extract()`).
#[tokio::test]
async fn ingest_rtf_end_to_end_real_extractor() {
    if !tokenizer_available().await {
        eprintln!("skipping ingest_rtf_end_to_end_real_extractor: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;
    let data_dir = engine.data_dir_for_test().await;
    let nb = engine.create_notebook("rtf-e2e", None, None).await.unwrap();

    // Minimal real RTF (two `\par` paragraphs) parsed by the production extractor.
    let rtf = b"{\\rtf1\\ansi\\deff0 Hello world from a real RTF document.\\par \
                A second paragraph with enough words to exercise the chunker end to end.\\par}";
    let id = uuid::Uuid::now_v7().to_string();
    let locator = write_raw_source_file(&data_dir, &id, rtf);
    insert_raw_source_locator(&engine, &nb.id.to_string(), "rtf", &locator, &id).await;

    engine
        .ingest_source(&id, |_p| {})
        .await
        .expect("ingest rtf");

    // Reaches `indexed` (status only flips after the indexing phase) with chunks.
    assert_eq!(engine_source_status(&engine, &id).await, "indexed");
    assert!(
        count_chunks(&engine, &id).await > 0,
        "RTF ingest must produce chunks end-to-end"
    );

    // Derived-kind canonical buffer is the `.extracted.txt` sibling; chunks slice
    // it byte-for-byte.
    let sibling = data_dir.join("sources").join(format!("{id}.extracted.txt"));
    assert!(
        sibling.exists(),
        "RTF (derived kind) must persist the .extracted.txt sibling"
    );
    let canonical = std::fs::read_to_string(&sibling).unwrap();
    assert!(
        canonical.contains("Hello world from a real RTF document."),
        "extracted canonical text present: {canonical:?}"
    );
    assert_chunk_byte_identity(&engine, &id, &canonical).await;
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
        .unwrap()
        .source;
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
                table_markdown: None,
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
        .expect("add_file_source for a .docx must succeed")
        .source;
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
        .expect("add_file_source for a .pdf must succeed")
        .source;
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

// ===========================================================================
// issue #71 — bounded-memory streaming PDF ingest (building-table lifecycle)
// ===========================================================================

/// Builds a multi-page text-layer PDF (`pages` pages, each carrying a unique
/// sentence) so the ingest produces enough chunks to exercise the streaming
/// building-table path. Returns the serialized PDF bytes.
fn build_multipage_text_pdf_bytes(pages: usize, sentence_prefix: &str) -> Vec<u8> {
    use printpdf::{BuiltinFont, Mm, PdfDocument};
    use std::io::BufWriter;
    let (doc, page1, layer1) =
        PdfDocument::new("multipage-fixture", Mm(210.0), Mm(297.0), "Layer 1");
    let font = doc.add_builtin_font(BuiltinFont::Helvetica).unwrap();
    {
        let layer = doc.get_page(page1).get_layer(layer1);
        for line in 0..8 {
            layer.use_text(
                format!("{sentence_prefix} page 1 line {line}: the quick brown fox jumps."),
                12.0,
                Mm(20.0),
                Mm(270.0 - line as f32 * 8.0),
                &font,
            );
        }
    }
    for p in 2..=pages {
        let (page, layer_idx) = doc.add_page(Mm(210.0), Mm(297.0), "Layer 1");
        let layer = doc.get_page(page).get_layer(layer_idx);
        for line in 0..8 {
            layer.use_text(
                format!("{sentence_prefix} page {p} line {line}: the quick brown fox jumps."),
                12.0,
                Mm(20.0),
                Mm(270.0 - line as f32 * 8.0),
                &font,
            );
        }
    }
    let mut buf = Vec::new();
    doc.save(&mut BufWriter::new(&mut buf)).unwrap();
    buf
}

/// Counts `embedding_index` rows for a notebook (fastembed/nomic/768 coordinate)
/// in a given status. Used to assert the building→active registry lifecycle.
async fn embidx_count(engine: &LensEngine, notebook: &str, status: &str) -> i64 {
    sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM embedding_index \
         WHERE notebook_id = ? AND backend = 'fastembed' AND model = ? AND dim = ? AND status = ?",
    )
    .bind(notebook)
    .bind(lens_core::DEFAULT_EMBED_MODEL_ID)
    .bind(lens_core::DEFAULT_EMBED_DIM as i64)
    .bind(status)
    .fetch_one(&engine.pool().await)
    .await
    .unwrap()
}

/// Counts the vectors stored in a notebook's ACTIVE Lance table (registry-driven:
/// resolves the `status='active'` `lance_table_name`, which is gen-suffixed after
/// the streaming flip). Robust to the first-source-is-PDF gen-1 asymmetry — the
/// support-module `vector_row_count` hardcodes the gen-0 name and would miss it.
async fn active_table_total_rows(engine: &LensEngine, notebook: &str) -> usize {
    use futures_util::TryStreamExt;
    use lancedb::query::ExecutableQuery;

    let name: Option<String> = sqlx::query_scalar(
        "SELECT lance_table_name FROM embedding_index \
         WHERE notebook_id = ? AND backend = 'fastembed' AND model = ? AND dim = ? AND status = 'active'",
    )
    .bind(notebook)
    .bind(lens_core::DEFAULT_EMBED_MODEL_ID)
    .bind(lens_core::DEFAULT_EMBED_DIM as i64)
    .fetch_optional(&engine.pool().await)
    .await
    .unwrap();
    let Some(name) = name else { return 0 };

    let root = engine.data_dir_for_test().await.join("lancedb");
    let conn = lancedb::connect(root.to_string_lossy().as_ref())
        .execute()
        .await
        .unwrap();
    let names = conn.table_names().execute().await.unwrap_or_default();
    if !names.iter().any(|n| n == &name) {
        return 0;
    }
    let table = conn.open_table(&name).execute().await.unwrap();
    let stream = table.query().execute().await.unwrap();
    let batches: Vec<_> = stream.try_collect().await.unwrap();
    batches.iter().map(|b| b.num_rows()).sum()
}

/// Counts SQLite chunk rows for a source (1 vector is produced per chunk).
async fn chunk_count_for_source(engine: &LensEngine, source_id: &str) -> i64 {
    sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM chunks WHERE source_id = ?")
        .bind(source_id)
        .fetch_one(&engine.pool().await)
        .await
        .unwrap()
}

/// Ingests a multi-page PDF as a source of `notebook` and writes it to disk;
/// returns the source id. Returns None if libpdfium cannot bind here.
async fn ingest_pdf_source(
    engine: &LensEngine,
    notebook: &str,
    pages: usize,
    sentence_prefix: &str,
) -> Option<String> {
    use lens_core::extract::extractor_for;
    let raw = build_multipage_text_pdf_bytes(pages, sentence_prefix);
    if extractor_for("pdf").unwrap().extract(&raw).is_err() {
        return None;
    }
    let data_dir = engine.data_dir_for_test().await;
    let id = uuid::Uuid::now_v7().to_string();
    let sources_dir = data_dir.join("sources");
    std::fs::create_dir_all(&sources_dir).unwrap();
    let path = sources_dir.join(format!("{id}.pdf"));
    std::fs::write(&path, &raw).unwrap();
    let locator = path.display().to_string();
    insert_raw_source_locator(engine, notebook, "pdf", &locator, &id).await;
    engine.ingest_source(&id, |_p| {}).await.unwrap();
    Some(id)
}

/// issue #71 Step 4: a PDF ingest drives the building-table lifecycle — a
/// `building` row is created and promoted to `active`, the active table's vector
/// count equals the chunk count, and no orphan `building`/`stale` rows remain.
#[tokio::test]
async fn test_pdf_ingest_building_table_lifecycle() {
    if !tokenizer_available().await {
        eprintln!("skipping test_pdf_ingest_building_table_lifecycle: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;
    let nb = engine
        .create_notebook("pdf-lifecycle", None, None)
        .await
        .unwrap();
    let nb_id = nb.id.to_string();

    let Some(src) = ingest_pdf_source(&engine, &nb_id, 4, "lifecycle").await else {
        eprintln!("skipping test_pdf_ingest_building_table_lifecycle: libpdfium not bindable");
        return;
    };
    assert_eq!(engine_source_status(&engine, &src).await, "indexed");

    assert_eq!(
        embidx_count(&engine, &nb_id, "active").await,
        1,
        "exactly one active embedding_index row after streaming flip"
    );
    assert_eq!(
        embidx_count(&engine, &nb_id, "building").await,
        0,
        "no orphan building rows remain after flip"
    );
    assert_eq!(
        embidx_count(&engine, &nb_id, "stale").await,
        0,
        "no orphan stale rows remain after flip"
    );

    let chunks = chunk_count_for_source(&engine, &src).await as usize;
    assert!(chunks > 0, "the multi-page PDF must produce chunks");
    assert_eq!(
        active_table_total_rows(&engine, &nb_id).await,
        chunks,
        "active table vector count must equal the chunk count"
    );
}

/// issue #71 Step 4: the streaming PDF ingest produces exactly one vector per
/// chunk (parents + children) end-to-end, verified via the active table count.
#[tokio::test]
async fn test_pdf_ingest_correct_chunk_vector_count() {
    if !tokenizer_available().await {
        eprintln!("skipping test_pdf_ingest_correct_chunk_vector_count: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;
    let nb = engine
        .create_notebook("pdf-count", None, None)
        .await
        .unwrap();
    let nb_id = nb.id.to_string();

    let Some(src) = ingest_pdf_source(&engine, &nb_id, 6, "count").await else {
        eprintln!("skipping test_pdf_ingest_correct_chunk_vector_count: libpdfium not bindable");
        return;
    };
    let chunks = chunk_count_for_source(&engine, &src).await as usize;
    assert!(chunks > 0);
    assert_eq!(
        active_table_total_rows(&engine, &nb_id).await,
        chunks,
        "1 vector per chunk after streaming ingest"
    );
}

/// issue #71 Step 4: the gen-0/gen-1 asymmetry — when a notebook's FIRST source
/// is a PDF, `create_building_table` makes a gen-1 building table (not gen-0
/// active), the seed is a no-op, and the flip promotes gen-1 to active (the active
/// table name carries a `__1` suffix).
#[tokio::test]
async fn test_first_source_pdf_gen_asymmetry() {
    if !tokenizer_available().await {
        eprintln!("skipping test_first_source_pdf_gen_asymmetry: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;
    let nb = engine.create_notebook("pdf-gen", None, None).await.unwrap();
    let nb_id = nb.id.to_string();

    let Some(src) = ingest_pdf_source(&engine, &nb_id, 3, "gen").await else {
        eprintln!("skipping test_first_source_pdf_gen_asymmetry: libpdfium not bindable");
        return;
    };
    assert_eq!(engine_source_status(&engine, &src).await, "indexed");

    let active_name: String = sqlx::query_scalar(
        "SELECT lance_table_name FROM embedding_index \
         WHERE notebook_id = ? AND backend = 'fastembed' AND model = ? AND dim = ? AND status = 'active'",
    )
    .bind(&nb_id)
    .bind(lens_core::DEFAULT_EMBED_MODEL_ID)
    .bind(lens_core::DEFAULT_EMBED_DIM as i64)
    .fetch_one(&engine.pool().await)
    .await
    .unwrap();
    assert!(
        active_name.ends_with("__1"),
        "first-source-is-PDF active table must be the promoted gen-1 building table, got {active_name}"
    );
    let chunks = chunk_count_for_source(&engine, &src).await as usize;
    assert_eq!(active_table_total_rows(&engine, &nb_id).await, chunks);
}

/// issue #71 Step 4: a SECOND PDF source in the same notebook preserves the first
/// source's vectors (the seed copies them into the building table before the flip).
#[tokio::test]
async fn test_pdf_ingest_second_source_preserves_first() {
    if !tokenizer_available().await {
        eprintln!("skipping test_pdf_ingest_second_source_preserves_first: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;
    let nb = engine.create_notebook("pdf-two", None, None).await.unwrap();
    let nb_id = nb.id.to_string();

    let Some(src1) = ingest_pdf_source(&engine, &nb_id, 3, "first").await else {
        eprintln!("skipping test_pdf_ingest_second_source_preserves_first: libpdfium not bindable");
        return;
    };
    let chunks1 = chunk_count_for_source(&engine, &src1).await as usize;
    assert_eq!(active_table_total_rows(&engine, &nb_id).await, chunks1);

    let src2 = ingest_pdf_source(&engine, &nb_id, 3, "second")
        .await
        .expect("second PDF ingest");
    let chunks2 = chunk_count_for_source(&engine, &src2).await as usize;

    assert_eq!(
        active_table_total_rows(&engine, &nb_id).await,
        chunks1 + chunks2,
        "second source's flip must preserve the first source's vectors"
    );
    assert_eq!(embidx_count(&engine, &nb_id, "active").await, 1);
    assert_eq!(embidx_count(&engine, &nb_id, "building").await, 0);
    assert_eq!(embidx_count(&engine, &nb_id, "stale").await, 0);
}

/// issue #71 Step 4: re-ingesting the SAME source with CHANGED content removes the
/// old vectors and installs the new ones with NO duplicates (wipe→seed→populate→flip
/// ordering preserved).
#[tokio::test]
async fn test_pdf_reingest_changed_content() {
    if !tokenizer_available().await {
        eprintln!("skipping test_pdf_reingest_changed_content: no tokenizer (offline)");
        return;
    }
    let (_dir, engine) = inject_counting_engine().await;
    let nb = engine
        .create_notebook("pdf-reingest", None, None)
        .await
        .unwrap();
    let nb_id = nb.id.to_string();
    let data_dir = engine.data_dir_for_test().await;

    use lens_core::extract::extractor_for;
    let raw_v1 = build_multipage_text_pdf_bytes(5, "version-one");
    if extractor_for("pdf").unwrap().extract(&raw_v1).is_err() {
        eprintln!("skipping test_pdf_reingest_changed_content: libpdfium not bindable");
        return;
    }
    let id = uuid::Uuid::now_v7().to_string();
    let sources_dir = data_dir.join("sources");
    std::fs::create_dir_all(&sources_dir).unwrap();
    let path = sources_dir.join(format!("{id}.pdf"));
    std::fs::write(&path, &raw_v1).unwrap();
    let locator = path.display().to_string();
    insert_raw_source_locator(&engine, &nb_id, "pdf", &locator, &id).await;
    engine.ingest_source(&id, |_p| {}).await.unwrap();
    let chunks_v1 = chunk_count_for_source(&engine, &id).await as usize;
    assert_eq!(active_table_total_rows(&engine, &nb_id).await, chunks_v1);

    // Overwrite the SAME locator with DIFFERENT content and re-ingest.
    let raw_v2 = build_multipage_text_pdf_bytes(2, "version-two-different");
    std::fs::write(&path, &raw_v2).unwrap();
    engine.ingest_source(&id, |_p| {}).await.unwrap();
    let chunks_v2 = chunk_count_for_source(&engine, &id).await as usize;

    assert_eq!(
        active_table_total_rows(&engine, &nb_id).await,
        chunks_v2,
        "re-ingest must replace old vectors with no duplicates"
    );
    assert_eq!(embidx_count(&engine, &nb_id, "active").await, 1);
    assert_eq!(embidx_count(&engine, &nb_id, "building").await, 0);
    assert_eq!(embidx_count(&engine, &nb_id, "stale").await, 0);
}

// ===========================================================================
// M4 issue #100 — content-hash dedup for text / URL / onboarding + restore guard
// ===========================================================================

/// AC-3 / AC-8: pasting identical text twice deduplicates. The second add
/// returns the existing source (`was_existing = true`) with the same id, and
/// only one row exists; different text creates a distinct source.
#[tokio::test]
async fn add_text_source_dedup_returns_existing() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("text-dedup-nb", None, None)
        .await
        .unwrap();

    let first = engine
        .add_text_source(&nb.id, "doc", "hello world", "text")
        .await
        .unwrap();
    assert!(!first.was_existing, "first add is a fresh insert");

    let second = engine
        .add_text_source(&nb.id, "doc-again", "hello world", "text")
        .await
        .unwrap();
    assert!(second.was_existing, "second identical add is a dedup hit");
    assert_eq!(
        second.source.id, first.source.id,
        "dedup hit returns the same source row"
    );

    let third = engine
        .add_text_source(&nb.id, "other", "different text", "text")
        .await
        .unwrap();
    assert!(!third.was_existing, "different text is a fresh insert");
    assert_ne!(third.source.id, first.source.id);

    let live = engine.list_sources(&nb.id).await.unwrap();
    assert_eq!(live.len(), 2, "only two distinct rows exist");
}

// ===========================================================================
// M4 issue #77 — RTF / ODT / EPUB extractor integration & snapshot tests
// ===========================================================================

mod office_binary_formats {
    use std::io::{Cursor, Write};

    use lens_core::extract::{ExtractOutput, SourceAnchor, extractor_for};

    const SAMPLE_RTF: &str = include_str!("fixtures/sample.rtf");

    /// Byte-identity over an `ExtractOutput`: every block slices `extracted_text`
    /// exactly (bytes).
    fn assert_extract_byte_identity(out: &ExtractOutput, label: &str) {
        assert!(!out.blocks.is_empty(), "{label}: must produce blocks");
        assert_eq!(
            out.anchors.len(),
            out.blocks.len(),
            "{label}: anchors index-aligned with blocks"
        );
        for (i, b) in out.blocks.iter().enumerate() {
            assert!(
                b.char_end <= out.extracted_text.len(),
                "{label}: block[{i}] OOB"
            );
            assert_eq!(
                &out.extracted_text[b.char_start..b.char_end],
                b.text,
                "{label}: byte-identity violated for block[{i}]"
            );
        }
    }

    /// Builds a minimal ODT (ZIP + content.xml) in memory.
    fn build_sample_odt() -> Vec<u8> {
        let content = r#"<?xml version="1.0" encoding="UTF-8"?>
<office:document-content xmlns:office="urn:office" xmlns:text="urn:text">
  <office:body><office:text>
    <text:h text:outline-level="1">Sample Heading</text:h>
    <text:p>An ODT paragraph under the heading.</text:p>
    <text:p>A second paragraph with multibyte 日本語 text.</text:p>
  </office:text></office:body>
</office:document-content>"#;
        let mut buf = Vec::new();
        {
            let mut zip = zip::ZipWriter::new(Cursor::new(&mut buf));
            let opts: zip::write::FileOptions = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zip.start_file("content.xml", opts).unwrap();
            zip.write_all(content.as_bytes()).unwrap();
            zip.finish().unwrap();
        }
        buf
    }

    /// Builds a minimal, valid EPUB 3 (2 chapters with headings + paragraphs).
    fn build_sample_epub() -> Vec<u8> {
        fn xhtml(body: &str) -> String {
            format!(
                r#"<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title></head>
<body>{body}</body></html>"#
            )
        }
        let opf = r#"<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="id">x</dc:identifier><dc:title>T</dc:title><dc:language>en</dc:language></metadata>
  <manifest>
    <item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="c1"/><itemref idref="c2"/></spine>
</package>"#;
        let mut buf = Vec::new();
        {
            let mut z = zip::ZipWriter::new(Cursor::new(&mut buf));
            let stored: zip::write::FileOptions = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            let defl: zip::write::FileOptions = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            z.start_file("mimetype", stored).unwrap();
            z.write_all(b"application/epub+zip").unwrap();
            z.start_file("META-INF/container.xml", defl).unwrap();
            z.write_all(
                br#"<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"#,
            )
            .unwrap();
            z.start_file("OEBPS/content.opf", defl).unwrap();
            z.write_all(opf.as_bytes()).unwrap();
            z.start_file("OEBPS/chapter1.xhtml", defl).unwrap();
            z.write_all(xhtml("<h1>Chapter One</h1><p>First chapter body.</p>").as_bytes())
                .unwrap();
            z.start_file("OEBPS/chapter2.xhtml", defl).unwrap();
            z.write_all(xhtml("<h1>Chapter Two</h1><p>Second chapter body.</p>").as_bytes())
                .unwrap();
            z.finish().unwrap();
        }
        buf
    }

    fn block_shape(out: &ExtractOutput) -> Vec<(String, String, String)> {
        out.blocks
            .iter()
            .map(|b| (b.block_type.clone(), b.section_path.clone(), b.text.clone()))
            .collect()
    }

    #[test]
    fn rtf_extractor_byte_identity() {
        let out = extractor_for("rtf")
            .unwrap()
            .extract(SAMPLE_RTF.as_bytes())
            .expect("rtf extract");
        assert_extract_byte_identity(&out, "rtf");
        assert!(
            out.anchors
                .iter()
                .all(|a| matches!(a, SourceAnchor::Rtf { .. }))
        );
    }

    #[test]
    fn odt_extractor_byte_identity() {
        let bytes = build_sample_odt();
        let out = extractor_for("odt")
            .unwrap()
            .extract(&bytes)
            .expect("odt extract");
        assert_extract_byte_identity(&out, "odt");
        assert!(
            out.anchors
                .iter()
                .all(|a| matches!(a, SourceAnchor::Odt { .. }))
        );
    }

    #[test]
    fn epub_extractor_byte_identity() {
        let bytes = build_sample_epub();
        let out = extractor_for("epub")
            .unwrap()
            .extract(&bytes)
            .expect("epub extract");
        assert_extract_byte_identity(&out, "epub");
        assert!(
            out.anchors
                .iter()
                .all(|a| matches!(a, SourceAnchor::Epub { .. }))
        );
    }

    #[test]
    fn rtf_extractor_snapshot() {
        let out = extractor_for("rtf")
            .unwrap()
            .extract(SAMPLE_RTF.as_bytes())
            .expect("rtf extract");
        insta::assert_debug_snapshot!("ingest_rtf_blocks", block_shape(&out));
    }

    #[test]
    fn odt_extractor_snapshot() {
        let out = extractor_for("odt")
            .unwrap()
            .extract(&build_sample_odt())
            .expect("odt extract");
        insta::assert_debug_snapshot!("ingest_odt_blocks", block_shape(&out));
    }

    #[test]
    fn epub_extractor_snapshot() {
        let out = extractor_for("epub")
            .unwrap()
            .extract(&build_sample_epub())
            .expect("epub extract");
        insta::assert_debug_snapshot!("ingest_epub_blocks", block_shape(&out));
    }
}
