//! Integration tests for the dialogue-script orchestrator (#26). Fully offline: a
//! scripted / barrier mock `LlmProvider`, a `CountingEmbedder`, a real (empty) Lance
//! store, and a temp SQLite DB seeded via `LensEngine::init`. Covers the public
//! serde surface, the pure `generate_dialogue` behavioural contract (parse/validate/
//! repair/cancel/empty-notebook), and the `LensEngine::generate_dialogue` grounded
//! ctx-gathering. The default suite is fully offline; a single real-local-model
//! end-to-end test is gated behind `LENS_RUN_MODEL_TESTS=1` + `--ignored`.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use async_trait::async_trait;
use lens_core::config::{ModelConfig, RetrievalConfig, TierThresholds};
use lens_core::embedder::{CountingEmbedder, Embedder, EmbeddingBackend};
use lens_core::llm::{LlmProvider, LlmRequest, LlmResponse};
use lens_core::vector_store::{Coordinate, LanceVectorStore, VectorStore};
use lens_core::{
    DialogueCtx, DialoguePhase, DialogueScript, Emotion, Length, LensEngine, Reranker, Speaker,
    generate_dialogue,
};
use sqlx::SqlitePool;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

const DIM: usize = 4;

// ---------------------------------------------------------------------------
// Public serde surface
// ---------------------------------------------------------------------------

#[test]
fn dialogue_script_serializes_as_turns_object() {
    let script = DialogueScript {
        turns: vec![lens_core::Turn {
            speaker: Speaker::Host,
            text: "hi".into(),
            emotion: Some(Emotion::Laugh),
            source_ids: vec!["sA".into()],
        }],
    };
    let v = serde_json::to_value(&script).unwrap();
    assert_eq!(v["turns"][0]["speaker"], "host");
    assert_eq!(v["turns"][0]["emotion"], "laugh");
    assert_eq!(v["turns"][0]["source_ids"][0], "sA");
    let back: DialogueScript = serde_json::from_value(v).unwrap();
    assert_eq!(script, back);
}

#[test]
fn length_serde_snake_case() {
    for (len, wire) in [
        (Length::Short, "short"),
        (Length::Medium, "medium"),
        (Length::Long, "long"),
    ] {
        assert_eq!(serde_json::to_value(len).unwrap(), wire);
    }
}

// ---------------------------------------------------------------------------
// Mock providers
// ---------------------------------------------------------------------------

/// Call-counted mock returning scripted text (cycling the last entry when
/// exhausted). Mirrors enrichment's `ScriptedProvider` but lives in this crate so it
/// can be shared by the orchestrator behavioural tests.
struct ScriptedProvider {
    calls: Arc<AtomicU32>,
    responses: Vec<String>,
}

impl ScriptedProvider {
    fn new(responses: Vec<&str>) -> (Arc<Self>, Arc<AtomicU32>) {
        let calls = Arc::new(AtomicU32::new(0));
        let p = Arc::new(Self {
            calls: calls.clone(),
            responses: responses.into_iter().map(|s| s.to_string()).collect(),
        });
        (p, calls)
    }
}

#[async_trait]
impl LlmProvider for ScriptedProvider {
    fn model_id(&self) -> &str {
        "mock-model"
    }
    async fn reachable(&self) -> bool {
        true
    }
    async fn generate(&self, _req: &LlmRequest) -> Result<LlmResponse, lens_core::LensError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) as usize;
        let text = self
            .responses
            .get(n)
            .or_else(|| self.responses.last())
            .cloned()
            .unwrap_or_default();
        Ok(LlmResponse {
            text,
            tokens_used: 10,
        })
    }
}

/// Asserts it is never called — proves the empty-notebook path issues ZERO LLM calls.
struct PanicProvider;

#[async_trait]
impl LlmProvider for PanicProvider {
    fn model_id(&self) -> &str {
        "panic-model"
    }
    async fn reachable(&self) -> bool {
        true
    }
    async fn generate(&self, _req: &LlmRequest) -> Result<LlmResponse, lens_core::LensError> {
        panic!("provider must not be invoked on the empty-notebook path");
    }
}

/// Parks in `generate` on a `Notify` barrier until released — the ONLY way to prove
/// the `tokio::select!` cancel race interrupts an in-flight call (C2a). A scripted
/// provider returns instantly and cannot exercise this.
struct BarrierProvider {
    parked: Arc<Notify>,
    release: Arc<Notify>,
    calls: Arc<AtomicU32>,
}

#[async_trait]
impl LlmProvider for BarrierProvider {
    fn model_id(&self) -> &str {
        "barrier-model"
    }
    async fn reachable(&self) -> bool {
        true
    }
    async fn generate(&self, _req: &LlmRequest) -> Result<LlmResponse, lens_core::LensError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.parked.notify_one();
        self.release.notified().await;
        Ok(LlmResponse {
            text: "[]".into(),
            tokens_used: 0,
        })
    }
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

async fn insert_chunk(pool: &SqlitePool, source_id: &str, chunk_id: &str, text: &str) {
    sqlx::query(
        "INSERT INTO chunks \
         (id, source_id, parent_id, kind, level, section_path, text, \
          token_start, token_end, char_start, char_end, page, block_type, source_anchor, created_at) \
         VALUES (?, ?, NULL, 'parent', 0, 'Intro', ?, 0, 1, 0, ?, 1, 'paragraph', ?, ?)",
    )
    .bind(chunk_id)
    .bind(source_id)
    .bind(text)
    .bind(text.len() as i64)
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
    selected_live_ids: HashSet<String>,
    length: Length,
) -> DialogueCtx {
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
    DialogueCtx {
        provider,
        store,
        embedder,
        reranker: Reranker::new(data_dir),
        graph: None,
        pool: pool.clone(),
        coord,
        model: ModelConfig {
            context: 20_000,
            ..ModelConfig::default()
        },
        retrieval: RetrievalConfig::default(),
        thresholds: TierThresholds::default(),
        tokenizer: None,
        length,
        selected_live_ids,
    }
}

/// A two-source Tier-1 notebook. Returns `(engine, pool, nb, ids, dir)`.
async fn seed_two_source_notebook() -> (
    LensEngine,
    SqlitePool,
    String,
    HashSet<String>,
    tempfile::TempDir,
) {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_source(&pool, nb.as_str(), "sA", "Alpha Doc").await;
    insert_source(&pool, nb.as_str(), "sB", "Beta Doc").await;
    insert_chunk(&pool, "sA", "c1", "alpha content").await;
    insert_chunk(&pool, "sB", "c2", "beta content").await;
    let ids: HashSet<String> = ["sA".to_string(), "sB".to_string()].into_iter().collect();
    (engine, pool, nb.as_str().to_string(), ids, dir)
}

fn no_phase() -> impl Fn(DialoguePhase) + Send {
    |_p| {}
}

/// A minimal valid Short script (>= 8 turns, both speakers, <=2 consecutive, cites
/// only sA/sB). Serialized as a bare JSON array.
fn valid_short_json() -> String {
    let mut turns = String::from("[");
    for i in 0..10 {
        let (speaker, sid) = if i % 2 == 0 {
            ("host", "sA")
        } else {
            ("guest", "sB")
        };
        if i > 0 {
            turns.push(',');
        }
        turns.push_str(&format!(
            r#"{{"speaker":"{speaker}","text":"turn {i}","source_ids":["{sid}"]}}"#
        ));
    }
    turns.push(']');
    turns
}

// ---------------------------------------------------------------------------
// Step 5 — happy path (one call)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn happy_path_one_call_returns_valid_grounded_script() {
    let (_engine, pool, nb, ids, dir) = seed_two_source_notebook().await;
    let (provider, calls) = ScriptedProvider::new(vec![&valid_short_json()]);
    let ctx = build_ctx(dir.path(), &pool, &nb, provider, ids, Length::Short);

    let phases = std::sync::Mutex::new(Vec::new());
    let script = generate_dialogue(ctx, CancellationToken::new(), |p| {
        phases.lock().unwrap().push(p)
    })
    .await
    .expect("valid script");

    assert_eq!(calls.load(Ordering::SeqCst), 1, "exactly one LLM call");
    assert!(script.turns.len() >= 8);
    assert!(script.turns.iter().any(|t| t.speaker == Speaker::Host));
    assert!(script.turns.iter().any(|t| t.speaker == Speaker::Guest));
    assert_eq!(
        *phases.lock().unwrap(),
        vec![
            DialoguePhase::Retrieving,
            DialoguePhase::Generating,
            DialoguePhase::Validating
        ]
    );
}

// ---------------------------------------------------------------------------
// Step 5 — malformed then valid (one repair, two calls)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn malformed_then_valid_repairs_in_two_calls() {
    let (_engine, pool, nb, ids, dir) = seed_two_source_notebook().await;
    let good = valid_short_json();
    let (provider, calls) = ScriptedProvider::new(vec!["not json at all", &good]);
    let ctx = build_ctx(dir.path(), &pool, &nb, provider, ids, Length::Short);
    let script = generate_dialogue(ctx, CancellationToken::new(), no_phase())
        .await
        .expect("repaired script");
    assert_eq!(calls.load(Ordering::SeqCst), 2, "one repair → two calls");
    assert!(script.turns.len() >= 8);
}

#[tokio::test]
async fn validation_failure_then_valid_repairs_in_two_calls() {
    let (_engine, pool, nb, ids, dir) = seed_two_source_notebook().await;
    // First response parses but is too short (fails validation) → one repair.
    let good = valid_short_json();
    let (provider, calls) =
        ScriptedProvider::new(vec![r#"[{"speaker":"host","text":"lonely"}]"#, &good]);
    let ctx = build_ctx(dir.path(), &pool, &nb, provider, ids, Length::Short);
    let script = generate_dialogue(ctx, CancellationToken::new(), no_phase())
        .await
        .expect("repaired script");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(script.turns.len() >= 8);
}

// ---------------------------------------------------------------------------
// Step 5 — malformed twice → Err (exactly one repair, two calls)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn malformed_twice_errors_after_one_repair() {
    let (_engine, pool, nb, ids, dir) = seed_two_source_notebook().await;
    let (provider, calls) = ScriptedProvider::new(vec!["garbage", "still garbage"]);
    let ctx = build_ctx(dir.path(), &pool, &nb, provider, ids, Length::Short);
    let err = generate_dialogue(ctx, CancellationToken::new(), no_phase())
        .await
        .expect_err("second failure is terminal");
    assert!(matches!(err, lens_core::LensError::Parse(_)));
    assert_eq!(calls.load(Ordering::SeqCst), 2, "capped at one repair");
}

// ---------------------------------------------------------------------------
// Step 5 — truncated Long response → repair
// ---------------------------------------------------------------------------

#[tokio::test]
async fn truncated_long_response_falls_to_repair() {
    let (_engine, pool, nb, ids, dir) = seed_two_source_notebook().await;
    // A Long array cut off mid-object: extract_json_array finds no balanced close.
    let truncated = r#"[{"speaker":"host","text":"intro","source_ids":["sA"]},{"speaker":"guest","text":"unterminat"#;
    let good = valid_short_json();
    let (provider, calls) = ScriptedProvider::new(vec![truncated, &good]);
    let ctx = build_ctx(dir.path(), &pool, &nb, provider, ids, Length::Long);
    // Long min_turns is 30; the repair `good` has 10 turns, so this still fails
    // validation on the repaired attempt — proving the truncation fell to repair.
    let err = generate_dialogue(ctx, CancellationToken::new(), no_phase())
        .await
        .expect_err("repaired-but-still-short is terminal");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "truncation triggered a repair"
    );
    assert!(matches!(err, lens_core::LensError::Validation(_)));
}

// ---------------------------------------------------------------------------
// Step 5 — empty notebook (Err::Validation, ZERO LLM calls, only Retrieving)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn empty_notebook_errors_with_zero_llm_calls() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    let ctx = build_ctx(
        dir.path(),
        &pool,
        nb.as_str(),
        Arc::new(PanicProvider),
        HashSet::new(),
        Length::Short,
    );
    let phases = std::sync::Mutex::new(Vec::new());
    let err = generate_dialogue(ctx, CancellationToken::new(), |p| {
        phases.lock().unwrap().push(p)
    })
    .await
    .expect_err("empty notebook is a validation error");
    assert!(matches!(err, lens_core::LensError::Validation(_)));
    // Only Retrieving fired — never Generating/Validating.
    assert_eq!(*phases.lock().unwrap(), vec![DialoguePhase::Retrieving]);
}

// ---------------------------------------------------------------------------
// Step 5 — cancel BEFORE generate (zero LLM calls)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cancel_before_generate_yields_cancelled_no_calls() {
    let (_engine, pool, nb, ids, dir) = seed_two_source_notebook().await;
    let (provider, calls) = ScriptedProvider::new(vec![&valid_short_json()]);
    let ctx = build_ctx(dir.path(), &pool, &nb, provider, ids, Length::Short);
    let cancel = CancellationToken::new();
    cancel.cancel();
    let err = generate_dialogue(ctx, cancel, no_phase())
        .await
        .expect_err("pre-cancelled run");
    assert!(matches!(err, lens_core::LensError::Cancelled(_)));
    assert_eq!(calls.load(Ordering::SeqCst), 0, "no LLM call after cancel");
}

// ---------------------------------------------------------------------------
// C2a — mid-call cancel via a barrier provider (the C1 acceptance gate)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cancel_mid_generate_interrupts_in_flight_call() {
    let (_engine, pool, nb, ids, dir) = seed_two_source_notebook().await;
    let parked = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let calls = Arc::new(AtomicU32::new(0));
    let provider = Arc::new(BarrierProvider {
        parked: parked.clone(),
        release: release.clone(),
        calls: calls.clone(),
    });
    let ctx = build_ctx(dir.path(), &pool, &nb, provider, ids, Length::Short);
    let cancel = CancellationToken::new();

    let cancel_for_task = cancel.clone();
    let handle =
        tokio::spawn(async move { generate_dialogue(ctx, cancel_for_task, no_phase()).await });

    // Wait until the provider is parked mid-generate, THEN cancel.
    parked.notified().await;
    cancel.cancel();

    let res = handle.await.expect("task joins");
    assert!(
        matches!(res, Err(lens_core::LensError::Cancelled(_))),
        "mid-call cancel returns Cancelled, got {res:?}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1, "the call was in flight");
    // Release so the abandoned future can drop cleanly.
    release.notify_waiters();
}

// ---------------------------------------------------------------------------
// Step 6 — LensEngine::generate_dialogue grounded ctx-gathering
// ---------------------------------------------------------------------------

#[tokio::test]
async fn engine_generate_dialogue_errors_when_no_provider() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    let res = engine
        .generate_dialogue(&nb, Length::Short, CancellationToken::new(), no_phase())
        .await;
    assert!(matches!(res, Err(lens_core::LensError::Model(_))));
}

#[tokio::test]
async fn engine_generate_dialogue_produces_grounded_script() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    engine.disable_tokenizer_for_test();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;

    engine
        .set_notebook_embedding_model(&nb, "all-minilm", EmbeddingBackend::Fastembed)
        .await
        .unwrap();
    let (model_id, dim, backend) = engine.resolve_notebook_embedding(&nb).await.unwrap();
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
    seed_active_coord(&pool, nb.as_str(), &model_id, dim, backend.as_str(), "tbl").await;

    let (provider, _calls) = ScriptedProvider::new(vec![&valid_short_json()]);
    engine.set_llm_provider(Some(provider)).await;
    insert_source(&pool, nb.as_str(), "sA", "Alpha Doc").await;
    insert_source(&pool, nb.as_str(), "sB", "Beta Doc").await;
    insert_chunk(&pool, "sA", "c1", "alpha content").await;
    insert_chunk(&pool, "sB", "c2", "beta content").await;

    let script = engine
        .generate_dialogue(&nb, Length::Short, CancellationToken::new(), no_phase())
        .await
        .expect("grounded script");

    let allowed: HashSet<&str> = ["sA", "sB"].into_iter().collect();
    for t in &script.turns {
        for id in &t.source_ids {
            assert!(
                allowed.contains(id.as_str()),
                "cited id {id} is selected+live"
            );
        }
    }
    assert!(script.turns.iter().any(|t| t.speaker == Speaker::Host));
    assert!(script.turns.iter().any(|t| t.speaker == Speaker::Guest));
}

// ---------------------------------------------------------------------------
// Step 9 — real-local-model end-to-end (gated behind LENS_RUN_MODEL_TESTS=1)
// ---------------------------------------------------------------------------

fn run_model_tests() -> bool {
    std::env::var("LENS_RUN_MODEL_TESTS").is_ok()
}

/// End-to-end against a real local LLM (Ollama, from config) + a real fastembed
/// embedder over a seeded notebook: proves a genuine model produces a script the
/// salvage-parse + validator accept. Opt-in — needs a reachable local model.
#[tokio::test]
#[ignore = "needs a real local LLM; run with LENS_RUN_MODEL_TESTS=1 --ignored"]
async fn real_model_generates_valid_grounded_dialogue() {
    if !run_model_tests() {
        eprintln!(
            "skipping real_model_generates_valid_grounded_dialogue (set LENS_RUN_MODEL_TESTS=1)"
        );
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_source(&pool, nb.as_str(), "sA", "Photosynthesis").await;
    insert_source(&pool, nb.as_str(), "sB", "Cellular Respiration").await;
    insert_chunk(
        &pool,
        "sA",
        "c1",
        "Photosynthesis converts light energy into chemical energy stored in glucose.",
    )
    .await;
    insert_chunk(
        &pool,
        "sB",
        "c2",
        "Cellular respiration breaks down glucose to release ATP for the cell.",
    )
    .await;

    let script = engine
        .generate_dialogue(&nb, Length::Short, CancellationToken::new(), no_phase())
        .await
        .expect("a reachable local model returns a valid grounded script");

    assert!(script.turns.len() >= Length::Short.preset().min_turns);
    assert!(script.turns.iter().any(|t| t.speaker == Speaker::Host));
    assert!(script.turns.iter().any(|t| t.speaker == Speaker::Guest));
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
