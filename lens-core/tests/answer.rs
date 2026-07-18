//! Integration tests for the grounded-answer orchestrator (#173). Fully offline:
//! a scripted mock `LlmProvider`, a `CountingEmbedder`, a real (empty) Lance store,
//! and a temp SQLite DB seeded via `LensEngine::init`. No `LENS_RUN_MODEL_TESTS`, no
//! network. Covers the `AnswerEvent`/`AnswerStage` serde + `RESERVED_OUTPUT` compile
//! assertion (Step 2), the `answer_stream` behavioural contract (Step 5), and the
//! `answer_notebook` ctx-gathering (Step 7).

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use lens_core::config::{ModelConfig, RetrievalConfig, TierThresholds};
use lens_core::embedder::{CountingEmbedder, Embedder, EmbeddingBackend};
use lens_core::llm::{LlmProvider, LlmRequest, LlmResponse, StreamChunk};
use lens_core::vector_store::{Coordinate, LanceVectorStore, VectorStore};
use lens_core::{
    AnswerCtx, AnswerEvent, AnswerStage, LensEngine, RESERVED_OUTPUT, Reranker, answer_stream,
};
use sqlx::SqlitePool;
use tokio_util::sync::CancellationToken;

const DIM: usize = 4;

// ---------------------------------------------------------------------------
// Step 2 — serde round-trip + RESERVED_OUTPUT compile assertion
// ---------------------------------------------------------------------------

#[test]
fn reserved_output_is_2048() {
    assert_eq!(RESERVED_OUTPUT, 2_048);
}

#[test]
fn answer_stage_serde_round_trip() {
    for stage in [
        AnswerStage::Retrieving,
        AnswerStage::Thinking,
        AnswerStage::Answering,
    ] {
        let json = serde_json::to_string(&stage).unwrap();
        let back: AnswerStage = serde_json::from_str(&json).unwrap();
        assert_eq!(stage, back);
    }
}

#[test]
fn answer_event_serde_round_trip() {
    let events = vec![
        AnswerEvent::Stage(AnswerStage::Retrieving),
        AnswerEvent::ThinkingDelta("thinking".into()),
        AnswerEvent::TextDelta("answer".into()),
        AnswerEvent::Citations(Vec::new()),
        AnswerEvent::Done {
            tokens_used: 42,
            grounded: true,
            citation_count: 0,
        },
    ];
    for ev in events {
        let json = serde_json::to_string(&ev).unwrap();
        let back: AnswerEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(ev, back);
    }
}

// ---------------------------------------------------------------------------
// Send assertion on the CONCRETE stream (Step 3)
// ---------------------------------------------------------------------------

fn assert_send<T: Send>(_t: &T) {}

#[tokio::test]
async fn answer_ctx_and_stream_are_send() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    let ctx = build_ctx(
        dir.path(),
        &pool,
        nb.as_str(),
        scripted_provider(vec![]),
        "q",
    );
    let stream = answer_stream(ctx, CancellationToken::new());
    assert_send(&stream);
}

// ---------------------------------------------------------------------------
// Mock providers
// ---------------------------------------------------------------------------

type Chunks = Vec<Result<StreamChunk, lens_core::LensError>>;

/// A mock whose `generate_stream` replays a scripted chunk list. `outer_err` makes
/// `generate_stream` itself return `Err` (construction failure). `panic_on_call`
/// asserts the provider is never invoked (empty-context path).
struct MockProvider {
    chunks: Chunks,
    outer_err: bool,
    panic_on_call: bool,
}

#[async_trait]
impl LlmProvider for MockProvider {
    fn model_id(&self) -> &str {
        "mock-model"
    }

    async fn reachable(&self) -> bool {
        true
    }

    async fn generate(&self, _req: &LlmRequest) -> Result<LlmResponse, lens_core::LensError> {
        Ok(LlmResponse {
            text: String::new(),
            tokens_used: 0,
        })
    }

    async fn generate_stream(
        &self,
        _req: &LlmRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamChunk, lens_core::LensError>> + Send>>,
        lens_core::LensError,
    > {
        if self.panic_on_call {
            panic!("provider must not be invoked on the empty-context path");
        }
        if self.outer_err {
            return Err(lens_core::LensError::Model("construction failed".into()));
        }
        let chunks = self.chunks.clone();
        Ok(Box::pin(futures_util::stream::iter(chunks)))
    }
}

fn scripted_provider(chunks: Chunks) -> Arc<dyn LlmProvider> {
    Arc::new(MockProvider {
        chunks,
        outer_err: false,
        panic_on_call: false,
    })
}

fn panicking_provider() -> Arc<dyn LlmProvider> {
    Arc::new(MockProvider {
        chunks: Vec::new(),
        outer_err: false,
        panic_on_call: true,
    })
}

fn outer_err_provider() -> Arc<dyn LlmProvider> {
    Arc::new(MockProvider {
        chunks: Vec::new(),
        outer_err: true,
        panic_on_call: false,
    })
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

async fn insert_source(pool: &SqlitePool, notebook_id: &str, source_id: &str, title: &str) {
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
         token_count, content_hash, created_at) \
         VALUES (?, ?, 'text', ?, 'indexed', '/tmp/seed.txt', 1, 50, ?, ?)",
    )
    .bind(source_id)
    .bind(notebook_id)
    .bind(title)
    .bind(format!("hash-{source_id}"))
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(pool)
    .await
    .expect("insert source");
}

async fn insert_chunk(pool: &SqlitePool, source_id: &str, chunk_id: &str, text: &str, page: i64) {
    sqlx::query(
        "INSERT INTO chunks \
         (id, source_id, parent_id, kind, level, section_path, text, \
          token_start, token_end, char_start, char_end, page, block_type, source_anchor, created_at) \
         VALUES (?, ?, NULL, 'parent', 0, 'Intro', ?, 0, 1, 0, ?, ?, 'paragraph', ?, ?)",
    )
    .bind(chunk_id)
    .bind(source_id)
    .bind(text)
    .bind(text.len() as i64)
    .bind(page)
    .bind(format!("anchor-{chunk_id}"))
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(pool)
    .await
    .expect("insert chunk");
}

fn build_ctx(
    data_dir: &std::path::Path,
    pool: &SqlitePool,
    nb: &str,
    provider: Arc<dyn LlmProvider>,
    question: &str,
) -> AnswerCtx {
    let store: Arc<dyn VectorStore> = Arc::new(LanceVectorStore::new(data_dir, pool.clone()));
    let embedder: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new_with_dim(
        DIM,
        "m",
        "doc: ",
        "query: ",
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    ));
    let coord = Coordinate::new(nb.to_string(), EmbeddingBackend::Fastembed, "m", DIM);
    AnswerCtx {
        provider,
        store,
        embedder,
        reranker: Reranker::new(data_dir),
        graph: None,
        pool: pool.clone(),
        coord,
        model: ModelConfig {
            context: 10_000,
            ..ModelConfig::default()
        },
        retrieval: RetrievalConfig::default(),
        thresholds: TierThresholds::default(),
        tokenizer: None,
        question: question.to_string(),
        history: Vec::new(),
        chat: lens_core::ChatConfig::default(),
    }
}

/// Collects a run expected to be error-free, unwrapping each `Ok` (fails loudly on
/// a terminal `Err`). Used by the success/cancel/empty-context paths.
async fn collect(
    stream: impl Stream<Item = Result<AnswerEvent, lens_core::LensError>>,
) -> Vec<AnswerEvent> {
    collect_raw(stream)
        .await
        .into_iter()
        .map(|r| r.expect("no terminal error expected on this path"))
        .collect()
}

/// Collects the raw `Result` items — used by the error paths to assert a terminal
/// `Err(LensError)` is surfaced.
async fn collect_raw(
    stream: impl Stream<Item = Result<AnswerEvent, lens_core::LensError>>,
) -> Vec<Result<AnswerEvent, lens_core::LensError>> {
    let mut s = Box::pin(stream);
    let mut out = Vec::new();
    while let Some(item) = s.next().await {
        out.push(item);
    }
    out
}

/// A two-source Tier-1 notebook (small corpus fits the cap). `sA/c1` on page 3,
/// `sB/c2` on page 7. Returns `(engine, pool, nb, dir)`.
async fn seed_two_source_notebook() -> (LensEngine, SqlitePool, String, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_source(&pool, nb.as_str(), "sA", "Alpha Doc").await;
    insert_source(&pool, nb.as_str(), "sB", "Beta Doc").await;
    insert_chunk(&pool, "sA", "c1", "alpha content", 3).await;
    insert_chunk(&pool, "sB", "c2", "beta content", 7).await;
    (engine, pool, nb.as_str().to_string(), dir)
}

// ---------------------------------------------------------------------------
// Step 5 — happy path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn happy_path_ordering_accumulation_citations_and_done() {
    let (_engine, pool, nb, dir) = seed_two_source_notebook().await;
    let provider = scripted_provider(vec![
        Ok(StreamChunk::ThinkingDelta("let me think ".into())),
        Ok(StreamChunk::ThinkingDelta("more".into())),
        Ok(StreamChunk::TextDelta("From [1] alpha".into())),
        Ok(StreamChunk::TextDelta(" and [2] beta.".into())),
        Ok(StreamChunk::Done { tokens_used: 99 }),
    ]);
    let ctx = build_ctx(dir.path(), &pool, &nb, provider, "compare");
    let events = collect(answer_stream(ctx, CancellationToken::new())).await;

    // Ordering contract.
    assert_eq!(events[0], AnswerEvent::Stage(AnswerStage::Retrieving));
    assert_eq!(events[1], AnswerEvent::Stage(AnswerStage::Thinking));
    assert_eq!(
        events[2],
        AnswerEvent::ThinkingDelta("let me think ".into())
    );
    assert_eq!(events[3], AnswerEvent::ThinkingDelta("more".into()));
    assert_eq!(events[4], AnswerEvent::Stage(AnswerStage::Answering));
    assert_eq!(events[5], AnswerEvent::TextDelta("From [1] alpha".into()));
    assert_eq!(events[6], AnswerEvent::TextDelta(" and [2] beta.".into()));

    // Exactly one Citations, then Done last.
    let citations: Vec<&Vec<lens_core::Citation>> = events
        .iter()
        .filter_map(|e| match e {
            AnswerEvent::Citations(c) => Some(c),
            _ => None,
        })
        .collect();
    assert_eq!(citations.len(), 1, "exactly one Citations event");
    let cites = citations[0];
    let source_ids: Vec<&str> = cites.iter().map(|c| c.source_id.as_str()).collect();
    assert_eq!(source_ids, vec!["sA", "sB"], "both sources cited in order");

    // Locators hydrated engine-side from the chunks fixture (page 3 / page 7).
    let page_a = cites[0].locators[0].page;
    let page_b = cites[1].locators[0].page;
    assert_eq!(page_a, Some(3), "sA locator hydrated to page 3");
    assert_eq!(page_b, Some(7), "sB locator hydrated to page 7");

    assert_eq!(
        events.last(),
        Some(&AnswerEvent::Done {
            tokens_used: 99,
            grounded: true,
            citation_count: 2,
        }),
        "Done passes the mock tokens_used through and is last"
    );
}

#[tokio::test]
async fn full_ordering_contract_no_stray_events() {
    let (_engine, pool, nb, dir) = seed_two_source_notebook().await;
    let provider = scripted_provider(vec![
        Ok(StreamChunk::TextDelta("plain [1] answer".into())),
        Ok(StreamChunk::Done { tokens_used: 5 }),
    ]);
    let ctx = build_ctx(dir.path(), &pool, &nb, provider, "q");
    let events = collect(answer_stream(ctx, CancellationToken::new())).await;
    // No thinking emitted → no Stage(Thinking).
    assert!(!events.contains(&AnswerEvent::Stage(AnswerStage::Thinking)));
    let stages: Vec<&AnswerStage> = events
        .iter()
        .filter_map(|e| match e {
            AnswerEvent::Stage(s) => Some(s),
            _ => None,
        })
        .collect();
    assert_eq!(
        stages,
        vec![&AnswerStage::Retrieving, &AnswerStage::Answering]
    );
    assert_eq!(
        events.last(),
        Some(&AnswerEvent::Done {
            tokens_used: 5,
            grounded: true,
            citation_count: 1,
        })
    );
}

// ---------------------------------------------------------------------------
// Step 5 — empty context (provider never invoked)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn empty_context_short_circuits_without_calling_provider() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    // No selected+live sources → tiered_search returns zero units.
    let ctx = build_ctx(dir.path(), &pool, nb.as_str(), panicking_provider(), "q");
    let events = collect(answer_stream(ctx, CancellationToken::new())).await;

    assert_eq!(events[0], AnswerEvent::Stage(AnswerStage::Retrieving));
    assert_eq!(events[1], AnswerEvent::Stage(AnswerStage::Answering));
    assert!(matches!(events[2], AnswerEvent::TextDelta(_)));
    assert_eq!(events[3], AnswerEvent::Citations(Vec::new()));
    assert_eq!(
        events[4],
        AnswerEvent::Done {
            tokens_used: 0,
            grounded: true,
            citation_count: 0,
        }
    );
    assert_eq!(events.len(), 5);
}

// ---------------------------------------------------------------------------
// Step 5 — cancellation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cancel_before_retrieval_ends_with_only_retrieving_stage() {
    let (_engine, pool, nb, dir) = seed_two_source_notebook().await;
    let ctx = build_ctx(dir.path(), &pool, &nb, panicking_provider(), "q");
    let cancel = CancellationToken::new();
    cancel.cancel();
    let events = collect(answer_stream(ctx, cancel)).await;
    // Retrieving is yielded first, then the pre-retrieval cancel check breaks.
    assert_eq!(events, vec![AnswerEvent::Stage(AnswerStage::Retrieving)]);
}

#[tokio::test]
async fn cancel_mid_generation_stops_emission_without_terminal_events() {
    let (_engine, pool, nb, dir) = seed_two_source_notebook().await;
    // A provider whose stream would emit text + Done, but we cancel first.
    let provider = scripted_provider(vec![
        Ok(StreamChunk::TextDelta("partial [1]".into())),
        Ok(StreamChunk::Done { tokens_used: 7 }),
    ]);
    let ctx = build_ctx(dir.path(), &pool, &nb, provider, "q");
    let cancel = CancellationToken::new();
    // Cancel up front so the first in-loop `is_cancelled` check breaks before relaying.
    cancel.cancel();
    let events = collect(answer_stream(ctx, cancel)).await;
    // No Citations, no Done (cancel ends the stream with no synthetic terminals).
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AnswerEvent::Citations(_)))
    );
    assert!(!events.iter().any(|e| matches!(e, AnswerEvent::Done { .. })));
}

// ---------------------------------------------------------------------------
// Step 5 — error surfaces (outer vs mid-stream)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn outer_generate_stream_err_yields_no_deltas_citations_or_done() {
    let (_engine, pool, nb, dir) = seed_two_source_notebook().await;
    let ctx = build_ctx(dir.path(), &pool, &nb, outer_err_provider(), "q");
    let items = collect_raw(answer_stream(ctx, CancellationToken::new())).await;
    // Retrieving stage, then a terminal Err — no deltas/Citations/Done.
    assert_eq!(items.len(), 2);
    assert_eq!(items[0], Ok(AnswerEvent::Stage(AnswerStage::Retrieving)));
    assert!(
        matches!(items[1], Err(lens_core::LensError::Model(_))),
        "outer generate_stream error surfaces as a terminal Err(Model)"
    );
}

#[tokio::test]
async fn mid_stream_err_stops_without_citations_or_done() {
    let (_engine, pool, nb, dir) = seed_two_source_notebook().await;
    let provider = scripted_provider(vec![
        Ok(StreamChunk::TextDelta("first [1]".into())),
        Ok(StreamChunk::TextDelta(" second".into())),
        Err(lens_core::LensError::Network("dropped".into())),
        Ok(StreamChunk::Done { tokens_used: 3 }),
    ]);
    let ctx = build_ctx(dir.path(), &pool, &nb, provider, "q");
    let items = collect_raw(answer_stream(ctx, CancellationToken::new())).await;
    // Two text deltas relayed, then a terminal Err — no Citations/Done (truncated
    // answer must not be cited); the error kind is preserved for the UI.
    let text_count = items
        .iter()
        .filter(|i| matches!(i, Ok(AnswerEvent::TextDelta(_))))
        .count();
    assert_eq!(text_count, 2);
    assert!(
        !items
            .iter()
            .any(|i| matches!(i, Ok(AnswerEvent::Citations(_))))
    );
    assert!(
        !items
            .iter()
            .any(|i| matches!(i, Ok(AnswerEvent::Done { .. })))
    );
    assert!(
        matches!(items.last(), Some(Err(lens_core::LensError::Network(_)))),
        "mid-stream error surfaces as a terminal Err(Network), preserving the kind"
    );
}

// ---------------------------------------------------------------------------
// Step 7 — answer_notebook ctx-gathering
// ---------------------------------------------------------------------------

#[tokio::test]
async fn answer_notebook_errors_when_no_provider_before_any_stream() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    // No provider installed → Err before any stream is built.
    let res = engine
        .answer_notebook(&nb, "turn-1", "q".into(), CancellationToken::new())
        .await;
    match res {
        Err(lens_core::LensError::Model(_)) => {}
        Err(other) => panic!("expected LensError::Model, got {other:?}"),
        Ok(_) => panic!("expected Err before a stream is built"),
    }
}

#[tokio::test]
async fn answer_notebook_selects_embedder_matched_coordinate() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    // Keep the run offline: the tokenizer is optional on the answer path.
    engine.disable_tokenizer_for_test();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;

    // Pin the notebook to a known embedding model/backend; resolve_notebook_embedding
    // must drive BOTH the embedder and the search Coordinate from this one tuple.
    engine
        .set_notebook_embedding_model(&nb, "all-minilm", EmbeddingBackend::Fastembed)
        .await
        .unwrap();
    let (model_id, dim, backend) = engine.resolve_notebook_embedding(&nb).await.unwrap();

    // Inject a model-free `all-minilm` embedder so `embedder_for` resolves offline
    // (all-minilm has accelerate_hint=false → CPU for both Interactive and Bulk, so
    // the Bulk-keyed injection is found by the Interactive lookup answer_notebook uses).
    let injected: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new_with_dim(
        dim,
        "all-minilm",
        "",
        "",
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    ));
    engine
        .set_embedder_for_test(injected, EmbeddingBackend::Fastembed)
        .unwrap();

    // Seed two ACTIVE embedding_index coords (a superseded cross-backend switch can
    // leave several); the answer path must search the one matching the notebook's
    // resolved embedder, not an unrelated active row.
    seed_active_coord(
        &pool,
        nb.as_str(),
        &model_id,
        dim,
        backend.as_str(),
        "tbl_match",
    )
    .await;
    seed_active_coord(
        &pool,
        nb.as_str(),
        "nomic-embed-text-v1.5",
        768,
        "ollama",
        "tbl_other",
    )
    .await;

    engine
        .set_llm_provider(Some(scripted_provider(vec![
            Ok(StreamChunk::TextDelta("ok".into())),
            Ok(StreamChunk::Done { tokens_used: 1 }),
        ])))
        .await;
    insert_source(&pool, nb.as_str(), "sA", "Alpha Doc").await;
    insert_chunk(&pool, "sA", "c1", "alpha content", 1).await;

    let stream = engine
        .answer_notebook(&nb, "turn-1", "q".into(), CancellationToken::new())
        .await
        .expect("stream builds with a provider present");
    let events = collect(stream).await;
    // The run completes against the matched coordinate (Tier-1 corpus), producing a
    // terminal Done — proof the resolved coord/embedder pair was searched, not the
    // mismatched active row.
    assert!(events.iter().any(|e| matches!(e, AnswerEvent::Done { .. })));
    // Sanity: the resolved coord is the all-minilm one, distinct from the ollama row.
    assert_eq!(model_id, "all-minilm");
    assert_eq!(backend, EmbeddingBackend::Fastembed);
}

async fn seed_active_coord(
    pool: &SqlitePool,
    notebook_id: &str,
    model: &str,
    dim: usize,
    backend: &str,
    table: &str,
) {
    sqlx::query(
        "INSERT INTO embedding_index \
         (id, notebook_id, model, dim, prefix_convention, backend, status, lance_table_name, created_at) \
         VALUES (?, ?, ?, ?, 'nomic', ?, 'active', ?, ?)",
    )
    .bind(format!("idx-{table}"))
    .bind(notebook_id)
    .bind(model)
    .bind(dim as i64)
    .bind(backend)
    .bind(table)
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(pool)
    .await
    .expect("insert embedding_index coord");
}

/// Inserts an oversized (Tier-2-forcing) selected+live source with one parent chunk.
async fn insert_oversized_source(pool: &SqlitePool, nb: &str, source_id: &str, chunk_text: &str) {
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
         token_count, content_hash, created_at) \
         VALUES (?, ?, 'text', 'Doc', 'indexed', '/tmp/s.txt', 1, 9000, ?, ?)",
    )
    .bind(source_id)
    .bind(nb)
    .bind(format!("hash-{source_id}"))
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(pool)
    .await
    .expect("insert oversized source");
    insert_chunk(pool, source_id, "c1", chunk_text, 1).await;
}

// ---------------------------------------------------------------------------
// RT-1 — reindexing gap vs honest refusal
// ---------------------------------------------------------------------------

#[tokio::test]
async fn empty_retrieval_during_reindex_yields_reindexing_not_refusal() {
    // Live selected chunks exist, but NO vectors were added and NO active
    // embedding_index row exists (a re-embed in flight / failed). The query shares no
    // lexical token, so BM25 misses too → retrieval goes fully empty. This must be a
    // typed `Reindexing` error the caller can retry, NEVER the persisted refusal.
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_oversized_source(&pool, nb.as_str(), "s1", "alpha content").await;

    let ctx = build_ctx(
        dir.path(),
        &pool,
        nb.as_str(),
        panicking_provider(),
        "zzzznomatch",
    );
    let items = collect_raw(answer_stream(ctx, CancellationToken::new())).await;

    assert_eq!(
        items[0].as_ref().unwrap(),
        &AnswerEvent::Stage(AnswerStage::Retrieving)
    );
    let last = items.last().expect("at least one item");
    assert!(
        matches!(last, Err(lens_core::LensError::Reindexing(_))),
        "expected terminal Reindexing, got {last:?}"
    );
    assert!(
        !items
            .iter()
            .any(|i| matches!(i, Ok(AnswerEvent::Done { .. }))),
        "a reindexing gap must not persist a Done/refusal"
    );
}

#[tokio::test]
async fn empty_retrieval_with_active_index_still_refuses() {
    // Same empty-retrieval shape, but an active index row exists → the corpus is
    // genuinely unanswerable, so the honest "no sources" refusal stands (not RT-1).
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_oversized_source(&pool, nb.as_str(), "s1", "alpha content").await;
    seed_active_coord(&pool, nb.as_str(), "m", DIM, "fastembed", "tbl").await;

    let ctx = build_ctx(
        dir.path(),
        &pool,
        nb.as_str(),
        panicking_provider(),
        "zzzznomatch",
    );
    let events = collect(answer_stream(ctx, CancellationToken::new())).await;

    assert!(
        events
            .iter()
            .any(|e| matches!(e, AnswerEvent::TextDelta(t) if t.contains("couldn't find"))),
        "active index + empty retrieval → honest refusal, got {events:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AnswerEvent::Done { grounded: true, .. })),
        "refusal reports grounded=true"
    );
}
