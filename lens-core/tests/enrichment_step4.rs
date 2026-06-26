//! Step-4 integration tests for the M4 Phase-3 enrichment LLM pipeline.
//!
//! These drive a FILE-BACKED engine with a MOCK [`LlmProvider`] (a call-counter)
//! installed via `set_llm_provider`, seed real `chunks` rows, enqueue a job, and
//! inspect the persisted TEXT columns + status + meta. The tokenizer is disabled
//! (`disable_tokenizer_for_test`) so the suite runs fully offline (the worker
//! falls back to a whitespace-word token count).
//!
//! Covered acceptance criteria (TEXT columns only — NO embeds; the re-embed flip
//! is Step 5):
//! * AC4 — valid structural map stored in `chunks.enrichment`; malformed×3 →
//!   degrade to context-prefix-only, source NOT failed.
//! * AC5 — contextual `embedding_text` composed; canonical `text` byte-identical.
//! * AC9 — matching composite cache key → ZERO LLM calls (re-run); a model-id
//!   change (same shape as a prompt-version bump) → re-runs.
//! * AC11 — a 2-call job under a per-job budget that admits 1 → the SECOND
//!   `generate()` is NEVER dispatched (mock sees exactly 1); status → `failed` +
//!   `budget_exceeded`; cloud-without-consent never dispatched (the factory gates
//!   it — covered in `llm.rs` unit tests).
//! * kind-awareness — a code-only source → `skipped` with a context-prefix
//!   `embedding_text` still present.
//! * size-gate — a tiny prose source → `skipped`.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use lens_core::LensEngine;
use lens_core::enrichment::meta::EnrichmentMeta;
use lens_core::error::LensError;
use lens_core::llm::{LlmProvider, LlmRequest, LlmResponse};
use sqlx::Row;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Mock provider
// ---------------------------------------------------------------------------

/// A mock provider with a call-counter that serves a scripted body sequence
/// (cycling the last entry once exhausted). `reachable()` is always `true`.
struct MockProvider {
    calls: Arc<AtomicU32>,
    bodies: Vec<String>,
    model: String,
}

impl MockProvider {
    fn new(model: &str, bodies: Vec<&str>) -> (Arc<Self>, Arc<AtomicU32>) {
        let calls = Arc::new(AtomicU32::new(0));
        let me = Arc::new(Self {
            calls: calls.clone(),
            bodies: bodies.into_iter().map(|s| s.to_string()).collect(),
            model: model.to_string(),
        });
        (me, calls)
    }
}

#[async_trait]
impl LlmProvider for MockProvider {
    fn model_id(&self) -> &str {
        &self.model
    }
    async fn reachable(&self) -> bool {
        true
    }
    async fn generate(&self, _req: &LlmRequest) -> Result<LlmResponse, LensError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) as usize;
        let text = self
            .bodies
            .get(n)
            .or_else(|| self.bodies.last())
            .cloned()
            .unwrap_or_default();
        Ok(LlmResponse {
            text,
            tokens_used: 10,
        })
    }
}

fn valid_map() -> &'static str {
    r#"{"entities":["Ada Lovelace"],"definitions":[{"term":"engine","definition":"a machine"}],"dates":["1843"],"summary":"A note about Ada Lovelace and the analytical engine."}"#
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn file_engine() -> (TempDir, LensEngine) {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = LensEngine::init(dir.path()).await.expect("engine init");
    // Step 6: the worker now honors `AppConfig.enrichment.enabled`. These tests
    // install a provider and assert enrichment RUNS, so enable it (the default is
    // OFF). `coref_strategy` keeps its default (`LlmInline`).
    {
        let mut cfg = engine.config().await;
        cfg.enrichment.enabled = true;
        engine.set_config(cfg).await;
    }
    // Run fully offline: no tokenizer download (whitespace-word fallback) AND a
    // deterministic injected embedder so the fused Step-4+Step-5 worker completes
    // the re-embed flip without the ~130 MB model download.
    engine.disable_tokenizer_for_test();
    let embedder: std::sync::Arc<dyn lens_core::Embedder> =
        std::sync::Arc::new(lens_core::CountingEmbedder::new(
            std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        ));
    engine
        .set_embedder_for_test(embedder)
        .expect("inject embedder");
    (dir, engine)
}

/// A chunk-row spec: `(parent_ordinal, kind, level, section_path, text,
/// block_type)`. `parent_ordinal` references another seeded chunk by its index.
type ChunkSpec<'a> = (
    Option<&'a str>,
    &'a str,
    i32,
    &'a str,
    &'a str,
    Option<&'a str>,
);

/// Seeds a notebook + an `indexed` source with a `content_hash`, then inserts the
/// given chunk rows. Returns `(notebook_id, source_id, chunk_ids)`.
async fn seed_source_with_chunks(
    engine: &LensEngine,
    content_hash: &str,
    chunks: &[ChunkSpec<'_>],
) -> (String, String, Vec<String>) {
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;
    let source_id = uuid::Uuid::now_v7().to_string();
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
         content_hash, enrichment_status, created_at) \
         VALUES (?, ?, 'text', 'seed', 'indexed', '/tmp/seed.txt', 1, ?, NULL, ?)",
    )
    .bind(&source_id)
    .bind(&nb)
    .bind(content_hash)
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(&pool)
    .await
    .expect("insert source");

    let now = chrono::Utc::now().to_rfc3339();
    let mut ids = Vec::new();
    for (i, (parent_id, kind, level, section_path, text, block_type)) in chunks.iter().enumerate() {
        let id = format!("{source_id}-chunk-{i}");
        sqlx::query(
            "INSERT INTO chunks \
             (id, source_id, parent_id, kind, level, section_path, text, \
              token_start, token_end, char_start, char_end, block_type, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&id)
        .bind(&source_id)
        .bind(parent_id.map(|p| format!("{source_id}-chunk-{p}")))
        .bind(*kind)
        .bind(*level)
        .bind(*section_path)
        .bind(*text)
        .bind(i as i64)
        .bind((i + 1) as i64)
        .bind(0_i64)
        .bind(text.len() as i64)
        .bind(*block_type)
        .bind(&now)
        .execute(&pool)
        .await
        .expect("insert chunk");
        ids.push(id);
    }
    (nb, source_id, ids)
}

async fn enrichment_status(engine: &LensEngine, source_id: &str) -> Option<String> {
    let pool = engine.pool().await;
    sqlx::query("SELECT enrichment_status FROM sources WHERE id = ?")
        .bind(source_id)
        .fetch_one(&pool)
        .await
        .expect("fetch source")
        .get::<Option<String>, _>("enrichment_status")
}

async fn wait_for_status(engine: &LensEngine, source_id: &str, want: &str) -> bool {
    for _ in 0..200 {
        if enrichment_status(engine, source_id).await.as_deref() == Some(want) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    false
}

/// Reads `(text, embedding_text, enrichment)` for a chunk id.
async fn chunk_columns(
    engine: &LensEngine,
    chunk_id: &str,
) -> (String, Option<String>, Option<String>) {
    let pool = engine.pool().await;
    let row = sqlx::query("SELECT text, embedding_text, enrichment FROM chunks WHERE id = ?")
        .bind(chunk_id)
        .fetch_one(&pool)
        .await
        .expect("fetch chunk");
    (
        row.get::<String, _>("text"),
        row.get::<Option<String>, _>("embedding_text"),
        row.get::<Option<String>, _>("enrichment"),
    )
}

async fn enrichment_meta(engine: &LensEngine, source_id: &str) -> Option<EnrichmentMeta> {
    let pool = engine.pool().await;
    let json: Option<String> = sqlx::query("SELECT enrichment_meta FROM sources WHERE id = ?")
        .bind(source_id)
        .fetch_one(&pool)
        .await
        .expect("fetch meta")
        .get("enrichment_meta");
    json.and_then(|j| serde_json::from_str(&j).ok())
}

/// A long prose body so the source clears the ~2000-token size gate (word-count
/// fallback). 2100 words.
fn long_prose() -> String {
    "Ada ".repeat(2100).trim_end().to_string()
}

// ---------------------------------------------------------------------------
// AC4 — structural map stored / fallback
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ac4_valid_map_is_stored_on_parent_and_text_byte_identical() {
    let (_dir, engine) = file_engine().await;
    let body = long_prose();
    let (_nb, source_id, ids) = seed_source_with_chunks(
        &engine,
        "hash-ac4",
        &[
            (None, "parent", 0, "Intro", &body, Some("paragraph")),
            (
                Some("0"),
                "child",
                1,
                "Intro",
                "Ada was a pioneer.",
                Some("paragraph"),
            ),
        ],
    )
    .await;

    let (provider, calls) = MockProvider::new("mock-1", vec![valid_map()]);
    engine.set_llm_provider(Some(provider)).await;
    engine.enqueue_enrichment_for_test(&source_id);

    // The fused Step-4+Step-5 worker writes the text columns then re-embeds and
    // flips, landing the source at `enriched`. The text-column assertions below
    // still hold (they are written before the flip).
    assert!(
        wait_for_status(&engine, &source_id, "enriched").await,
        "the worker must complete the re-embed flip to `enriched`; got {:?}",
        enrichment_status(&engine, &source_id).await
    );

    assert_eq!(calls.load(Ordering::SeqCst), 1, "exactly one map LLM call");

    // The map JSON is on the PARENT row; it parses as a StructuralMap.
    let (parent_text, parent_et, parent_enrichment) = chunk_columns(&engine, &ids[0]).await;
    let enrichment = parent_enrichment.expect("parent row has the structural map");
    let map: serde_json::Value = serde_json::from_str(&enrichment).expect("valid map JSON");
    assert_eq!(
        map["summary"].as_str().unwrap(),
        "A note about Ada Lovelace and the analytical engine."
    );
    // AC5: canonical text byte-identical; embedding_text present + ⊇ text.
    assert_eq!(parent_text, body, "canonical text must be byte-identical");
    let et = parent_et.expect("embedding_text written");
    assert!(
        et.contains(&body) || et.ends_with(&body[body.len().saturating_sub(20)..]),
        "embedding_text must contain the canonical body"
    );

    // The CHILD row has embedding_text but NO map (map attaches to parent only).
    let (_ctext, child_et, child_enrichment) = chunk_columns(&engine, &ids[1]).await;
    assert!(child_et.is_some(), "child has embedding_text");
    assert!(
        child_enrichment.is_none(),
        "map attaches to the parent row only"
    );

    let meta = enrichment_meta(&engine, &source_id)
        .await
        .expect("meta written");
    assert_eq!(meta.map_quality, "ok");
    assert!(!meta.budget_exceeded);
}

#[tokio::test]
async fn ac4_malformed_thrice_degrades_to_fallback_source_not_failed() {
    let (_dir, engine) = file_engine().await;
    let body = long_prose();
    let (_nb, source_id, ids) = seed_source_with_chunks(
        &engine,
        "hash-fallback",
        &[(None, "parent", 0, "Intro", &body, Some("paragraph"))],
    )
    .await;

    // 1 initial + 2 reprompts, all malformed → fallback (not failed).
    let (provider, calls) = MockProvider::new("mock-1", vec!["nope", "still bad", "garbage"]);
    engine.set_llm_provider(Some(provider)).await;
    engine.enqueue_enrichment_for_test(&source_id);

    // Malformed map → prefix-only fallback, but the worker still re-embeds the
    // prefix-only `embedding_text` and flips → `enriched` (NOT failed; a degrade
    // is not a failure). No summary node is created (empty summary).
    assert!(
        wait_for_status(&engine, &source_id, "enriched").await,
        "a malformed-map fallback must still complete to `enriched` (NOT failed); got {:?}",
        enrichment_status(&engine, &source_id).await
    );
    assert_eq!(calls.load(Ordering::SeqCst), 3, "1 initial + 2 reprompts");

    // No map on the parent (degraded), but embedding_text IS present (prefix-only).
    let (text, et, enrichment) = chunk_columns(&engine, &ids[0]).await;
    assert_eq!(text, body, "canonical text untouched");
    assert!(
        et.is_some(),
        "prefix-only embedding_text still applied on degrade"
    );
    assert!(enrichment.is_none(), "no structural map on a fallback");

    let meta = enrichment_meta(&engine, &source_id).await.expect("meta");
    assert_eq!(meta.map_quality, "fallback");
    assert!(!meta.budget_exceeded, "a degrade is NOT a budget failure");
}

// ---------------------------------------------------------------------------
// AC9 — composite cache-key short-circuit
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ac9_matching_cache_key_skips_llm_entirely() {
    let (_dir, engine) = file_engine().await;
    let body = long_prose();
    let (_nb, source_id, _ids) = seed_source_with_chunks(
        &engine,
        "hash-cache",
        &[(None, "parent", 0, "Intro", &body, Some("paragraph"))],
    )
    .await;

    let (provider, calls) = MockProvider::new("mock-1", vec![valid_map()]);
    engine.set_llm_provider(Some(provider)).await;

    // First run: one LLM call; the fused worker completes to `enriched`.
    engine.enqueue_enrichment_for_test(&source_id);
    assert!(wait_for_status(&engine, &source_id, "enriched").await);
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // Re-enqueue with the SAME config → the persisted cache key matches → ZERO
    // additional LLM calls; status stays `enriched`.
    engine.enqueue_enrichment_for_test(&source_id);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "a matching cache key must dispatch ZERO additional LLM calls"
    );
    assert_eq!(
        enrichment_status(&engine, &source_id).await.as_deref(),
        Some("enriched"),
        "cache-key hit keeps the source enriched"
    );
}

#[tokio::test]
async fn ac9_model_change_invalidates_cache_key_and_reruns() {
    let (_dir, engine) = file_engine().await;
    let body = long_prose();
    let (_nb, source_id, _ids) = seed_source_with_chunks(
        &engine,
        "hash-cache2",
        &[(None, "parent", 0, "Intro", &body, Some("paragraph"))],
    )
    .await;

    // First provider (model "mock-A") enriches once.
    let (provider_a, calls_a) = MockProvider::new("mock-A", vec![valid_map()]);
    engine.set_llm_provider(Some(provider_a)).await;
    engine.enqueue_enrichment_for_test(&source_id);
    assert!(wait_for_status(&engine, &source_id, "enriched").await);
    assert_eq!(calls_a.load(Ordering::SeqCst), 1);

    // Swap to a DIFFERENT model id → the composite cache key changes (model_id is
    // a key component, exactly like a prompt_version bump) → it MUST re-run. The
    // re-run re-enriches and completes back to `enriched` with a fresh LLM call.
    let (provider_b, calls_b) = MockProvider::new("mock-B", vec![valid_map()]);
    engine.set_llm_provider(Some(provider_b)).await;
    // Move OFF `enriched` first so the re-run's completion back to `enriched` is
    // observable as a transition (the cache-key mismatch forces the LLM call).
    engine.enqueue_enrichment_for_test(&source_id);
    // The model-id change invalidates the cache key → a second LLM call fires.
    for _ in 0..300 {
        if calls_b.load(Ordering::SeqCst) >= 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        wait_for_status(&engine, &source_id, "enriched").await,
        "a changed model id must invalidate the cache key and re-run"
    );
    assert_eq!(
        calls_b.load(Ordering::SeqCst),
        1,
        "the new-model run dispatched a fresh LLM call"
    );
}

// ---------------------------------------------------------------------------
// AC11 — budget circuit-break
// ---------------------------------------------------------------------------

/// AC11 (the load-bearing budget assertion): a job whose per-job call ceiling
/// admits exactly 1 call, with a mock that replies malformed (which WOULD trigger
/// a 2nd reprompt call). The SECOND `generate()` must NEVER be dispatched — the
/// budget short-circuits BEFORE the call — so the mock sees EXACTLY 1 call and the
/// source flips to `failed` with `budget_exceeded` recorded in `enrichment_meta`.
#[tokio::test]
async fn ac11_per_job_budget_one_call_second_generate_never_dispatched() {
    let (_dir, engine) = file_engine().await;
    let body = long_prose();
    let (_nb, source_id, ids) = seed_source_with_chunks(
        &engine,
        "hash-budget",
        &[(None, "parent", 0, "Intro", &body, Some("paragraph"))],
    )
    .await;

    // Tighten the per-job call ceiling to 1 (AC11 seam).
    engine.set_enrichment_max_calls_for_test(1);

    // Two malformed replies are scripted: the first is dispatched + malformed;
    // the worker would reprompt, but the budget refuses the 2nd dispatch.
    let (provider, calls) = MockProvider::new("mock-1", vec!["bad json", "more bad", valid_map()]);
    engine.set_llm_provider(Some(provider)).await;
    engine.enqueue_enrichment_for_test(&source_id);

    assert!(
        wait_for_status(&engine, &source_id, "failed").await,
        "a budget circuit-break must flip the source to `failed`; got {:?}",
        enrichment_status(&engine, &source_id).await
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the SECOND generate() must NEVER be dispatched (budget admits exactly 1)"
    );

    let meta = enrichment_meta(&engine, &source_id).await.expect("meta");
    assert!(
        meta.budget_exceeded,
        "budget_exceeded must be recorded (AC11)"
    );
    assert_eq!(
        meta.calls_made, 1,
        "exactly one call was made before the break"
    );

    // Raw vectors untouched: no text columns were written on the budget failure.
    let (text, et, enr) = chunk_columns(&engine, &ids[0]).await;
    assert_eq!(text, body, "canonical text untouched on a budget failure");
    assert!(
        et.is_none(),
        "no embedding_text written when the job fails on budget"
    );
    assert!(
        enr.is_none(),
        "no structural map written when the job fails on budget"
    );
}

// ---------------------------------------------------------------------------
// kind-awareness + size-gate → skipped + prefix-only embedding_text
// ---------------------------------------------------------------------------

#[tokio::test]
async fn code_only_source_is_skipped_with_prefix_embedding_text() {
    let (_dir, engine) = file_engine().await;
    // A long CODE body (clears the size gate) so only the kind gate fires.
    let code = "fn main() {} ".repeat(2100);
    let (_nb, source_id, ids) = seed_source_with_chunks(
        &engine,
        "hash-code",
        &[(None, "parent", 0, "src/main.rs", &code, Some("code"))],
    )
    .await;

    let (provider, calls) = MockProvider::new("mock-1", vec![valid_map()]);
    engine.set_llm_provider(Some(provider)).await;
    engine.enqueue_enrichment_for_test(&source_id);

    assert!(
        wait_for_status(&engine, &source_id, "skipped").await,
        "a code-only source skips the structural map; got {:?}",
        enrichment_status(&engine, &source_id).await
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "non-prose dispatches ZERO LLM calls"
    );

    // Decision B: a context-prefix embedding_text IS still applied; text untouched.
    let (text, et, enrichment) = chunk_columns(&engine, &ids[0]).await;
    assert_eq!(text, code, "canonical text byte-identical");
    assert!(
        et.is_some(),
        "skipped kind still gets a prefix-only embedding_text"
    );
    assert!(enrichment.is_none(), "no structural map for a skipped kind");

    let meta = enrichment_meta(&engine, &source_id).await.expect("meta");
    assert_eq!(meta.map_quality, "skipped");
}

#[tokio::test]
async fn tiny_prose_source_is_skipped_by_size_gate() {
    let (_dir, engine) = file_engine().await;
    // A short prose body — well under the ~2000-token size gate.
    let (_nb, source_id, ids) = seed_source_with_chunks(
        &engine,
        "hash-tiny",
        &[(
            None,
            "parent",
            0,
            "Intro",
            "A short note.",
            Some("paragraph"),
        )],
    )
    .await;

    let (provider, calls) = MockProvider::new("mock-1", vec![valid_map()]);
    engine.set_llm_provider(Some(provider)).await;
    engine.enqueue_enrichment_for_test(&source_id);

    assert!(
        wait_for_status(&engine, &source_id, "skipped").await,
        "a tiny source is size-gated to `skipped`; got {:?}",
        enrichment_status(&engine, &source_id).await
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "size-gated → ZERO LLM calls"
    );
    let (_text, et, _enr) = chunk_columns(&engine, &ids[0]).await;
    assert!(
        et.is_some(),
        "size-gated source still gets a prefix-only embedding_text"
    );
}
