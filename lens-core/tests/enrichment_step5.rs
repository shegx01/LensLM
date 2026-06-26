//! Step-5 integration + crash-injection tests for the re-embed new-table-flip.
//!
//! These drive the FULL worker pipeline (enqueue → Step-4 text columns → Step-5
//! re-embed flip) over a FILE-BACKED engine with:
//!   * a MOCK [`LlmProvider`] returning a valid structural map (so the prose path
//!     runs and produces a doc-summary), and
//!   * an injected [`CountingEmbedder`] (deterministic 768-dim vectors, no model
//!     download), and a disabled tokenizer (whitespace-word fallback, fully
//!     offline).
//!
//! Covered acceptance criteria (Step 5):
//!   * AC6 — the doc-summary RAPTOR node is created with the exact contract,
//!     embedded into the active table, and reclaimed by `purge_source`.
//!   * AC7 — `crash_between_flip_txn_commit_and_lance_drop` (active points at the
//!     complete enriched table; a stale row + orphan table remain; startup-GC
//!     reclaims the stale row even when its Lance table is already gone; search
//!     never returns mixed/empty) AND `crash_during_building_table_populate`
//!     (building orphan GC'd, active raw vectors untouched and still serve).
//!   * AC8 — search resolves the gen-suffixed active table via the registry after
//!     a flip; gen-0 == formula keeps the pre-flip path identical.

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use lens_core::error::LensError;
use lens_core::llm::{LlmProvider, LlmRequest, LlmResponse};
use lens_core::vector_store::{
    CRASH_AFTER_FLIP_TXN_BEFORE_LANCE_DROP, LanceVectorStore, VectorStore,
};
use lens_core::{CountingEmbedder, EMBED_DIM, EMBED_MODEL_ID, Embedder, LensEngine};
use sqlx::Row;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Flip-test serialization
// ---------------------------------------------------------------------------
//
// `CRASH_AFTER_FLIP_TXN_BEFORE_LANCE_DROP` (vector_store.rs) is a PROCESS-GLOBAL
// `AtomicBool` consumed by a `swap(false, …)` inside `flip_active`. Every test in
// this file reaches a `flip_active` (directly or via the worker), so if the
// crash-injection test arms the flag while another flip-touching test runs
// concurrently (cargo's default), the WRONG test consumes it — corrupting both.
// The fix is to run every flip-touching test under one process-wide async mutex so
// the arm→consume window is exclusive; the guard also resets the flag to a known
// `false` on entry so a panicked prior test cannot leak a `true` across the lock.
static FLIP_TEST_SERIAL: LazyLock<tokio::sync::Mutex<()>> =
    LazyLock::new(|| tokio::sync::Mutex::new(()));

/// Acquires the flip-test serialization lock and clears the global crash flag.
/// Hold the returned guard for the whole test body (the flag is consumed inside
/// `flip_active`, so the lock must outlive the flip).
async fn flip_serial_guard() -> tokio::sync::MutexGuard<'static, ()> {
    let guard = FLIP_TEST_SERIAL.lock().await;
    CRASH_AFTER_FLIP_TXN_BEFORE_LANCE_DROP.store(false, Ordering::SeqCst);
    guard
}

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

/// Always-reachable provider returning one fixed valid structural map.
struct MockProvider {
    calls: Arc<AtomicU32>,
}

#[async_trait]
impl LlmProvider for MockProvider {
    fn model_id(&self) -> &str {
        "mock-llm"
    }
    async fn reachable(&self) -> bool {
        true
    }
    async fn generate(&self, _req: &LlmRequest) -> Result<LlmResponse, LensError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(LlmResponse {
            text: r#"{"entities":["Ada"],"definitions":[{"term":"engine","definition":"a machine"}],"dates":["1843"],"summary":"Ada Lovelace wrote about the analytical engine."}"#
                .to_string(),
            tokens_used: 10,
        })
    }
}

/// An embedder that ALWAYS errors on `embed_documents` — used to simulate a crash
/// DURING the building-table populate (before the flip).
struct FailingEmbedder;

impl Embedder for FailingEmbedder {
    fn model_id(&self) -> &str {
        EMBED_MODEL_ID
    }
    fn dim(&self) -> usize {
        EMBED_DIM
    }
    fn embed_documents(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>, LensError> {
        Err(LensError::Model("injected embed failure".into()))
    }
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, LensError> {
        Err(LensError::Model("injected embed failure".into()))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Step 6: the worker now honors `AppConfig.enrichment.enabled` (default OFF).
/// These tests assert enrichment RUNS, so enable it AND PERSIST it to disk so a
/// `reopen_engine` (crash-restart) reloads it from `config.json`.
fn enable_enrichment_on_disk(dir: &TempDir) {
    let mut cfg = lens_core::config::AppConfig::load(dir.path()).expect("load config");
    cfg.enrichment.enabled = true;
    cfg.save(dir.path()).expect("save config");
}

async fn file_engine() -> (TempDir, LensEngine) {
    let dir = tempfile::tempdir().expect("tempdir");
    // Init once to materialize the default config, then enable enrichment on disk
    // and re-init so the in-memory config + the disk file both have it enabled.
    LensEngine::init(dir.path()).await.expect("engine init");
    enable_enrichment_on_disk(&dir);
    let engine = LensEngine::init(dir.path()).await.expect("engine re-init");
    engine.disable_tokenizer_for_test();
    (dir, engine)
}

async fn reopen_engine(dir: &TempDir) -> LensEngine {
    let engine = LensEngine::init(dir.path()).await.expect("engine re-init");
    engine.disable_tokenizer_for_test();
    engine
}

async fn install_mock_provider(engine: &LensEngine) -> Arc<AtomicU32> {
    let calls = Arc::new(AtomicU32::new(0));
    let provider: Arc<dyn LlmProvider> = Arc::new(MockProvider {
        calls: calls.clone(),
    });
    engine.set_llm_provider(Some(provider)).await;
    calls
}

fn install_counting_embedder(engine: &LensEngine) {
    let embedder: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new(
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    ));
    engine
        .set_embedder_for_test(embedder)
        .expect("inject embedder");
}

/// Seeds a notebook + an `indexed` source with a `content_hash` + a prose parent
/// chunk and one child, with PRE-EXISTING RAW vectors in the gen-0 active table
/// (mirroring a Phase-1/2 ingested source). Returns `(notebook_id, source_id)`.
async fn seed_indexed_prose_source(engine: &LensEngine) -> (String, String) {
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
         VALUES (?, ?, 'text', 'seed', 'indexed', '/tmp/seed.txt', 1, 'hash-1', NULL, ?)",
    )
    .bind(&source_id)
    .bind(&nb)
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(&pool)
    .await
    .expect("insert source");

    let now = chrono::Utc::now().to_rfc3339();
    let parent_id = format!("{source_id}-p0");
    let child_id = format!("{source_id}-c0");
    // The parent body must exceed the worker's size-gate (2000 tokens, counted by
    // the whitespace-word fallback) so the prose structural-map + re-embed path
    // runs instead of the size-gated `skipped` path. Pad with repeated prose.
    let parent_text = {
        let base =
            "Ada Lovelace wrote extensive notes about the analytical engine and its potential. ";
        let mut s = String::new();
        // ~12 words per repeat → 200 repeats ≈ 2400 words, comfortably over 2000.
        for _ in 0..200 {
            s.push_str(base);
        }
        s
    };
    let child_text = "She is widely regarded as the first computer programmer.";
    let parent_text_ref = parent_text.as_str();
    for (id, parent, kind, level, text, tok) in [
        (
            &parent_id,
            None::<&str>,
            "parent",
            0_i32,
            parent_text_ref,
            0_i64,
        ),
        (
            &child_id,
            Some(parent_id.as_str()),
            "child",
            1,
            child_text,
            1,
        ),
    ] {
        sqlx::query(
            "INSERT INTO chunks \
             (id, source_id, parent_id, kind, level, section_path, text, \
              token_start, token_end, char_start, char_end, block_type, created_at) \
             VALUES (?, ?, ?, ?, ?, '[\"Intro\"]', ?, ?, ?, 0, ?, 'paragraph', ?)",
        )
        .bind(id)
        .bind(&source_id)
        .bind(parent)
        .bind(kind)
        .bind(level)
        .bind(text)
        .bind(tok)
        .bind(tok + 1)
        .bind(text.len() as i64)
        .bind(&now)
        .execute(&pool)
        .await
        .expect("insert chunk");
    }

    // Seed RAW vectors into the gen-0 active table (the Phase-1/2 raw index) via
    // the production `add` path (which registers a `status='active'` gen-0 row).
    let data_dir = engine.data_dir_for_test().await;
    let store = LanceVectorStore::new(&data_dir, pool.clone());
    let rows = vec![
        row(&parent_id, &source_id, &nb, 0),
        row(&child_id, &source_id, &nb, 1),
    ];
    store
        .add(&nb, EMBED_MODEL_ID, EMBED_DIM, rows)
        .await
        .expect("seed raw active vectors");

    (nb, source_id)
}

fn row(
    chunk_id: &str,
    source_id: &str,
    notebook: &str,
    level: i32,
) -> lens_core::vector_store::VectorRow {
    // Distinct raw vectors per chunk (axis-aligned) so a "raw vs enriched" diff is
    // observable; the enriched re-embed overwrites these in a NEW table.
    let mut v = vec![0.0_f32; EMBED_DIM];
    v[(level as usize) % EMBED_DIM] = 1.0;
    lens_core::vector_store::VectorRow {
        chunk_id: chunk_id.to_string(),
        source_id: source_id.to_string(),
        notebook_id: notebook.to_string(),
        level,
        vector: v,
    }
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
    for _ in 0..300 {
        if enrichment_status(engine, source_id).await.as_deref() == Some(want) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    false
}

async fn registry_count(engine: &LensEngine, status: &str) -> i64 {
    let pool = engine.pool().await;
    sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM embedding_index WHERE status = ?")
        .bind(status)
        .fetch_one(&pool)
        .await
        .expect("count registry rows")
}

async fn active_table_name(engine: &LensEngine, notebook: &str) -> String {
    let pool = engine.pool().await;
    sqlx::query_scalar::<_, String>(
        "SELECT lance_table_name FROM embedding_index \
         WHERE notebook_id = ? AND status = 'active'",
    )
    .bind(notebook)
    .fetch_one(&pool)
    .await
    .expect("fetch active table name")
}

async fn live_lance_tables(data_dir: &std::path::Path) -> Vec<String> {
    let conn = lancedb::connect(data_dir.join("lancedb").to_string_lossy().as_ref())
        .execute()
        .await
        .unwrap();
    conn.table_names().execute().await.unwrap()
}

/// Runs a search through the production `VectorStore::search` for the parent
/// chunk's content and returns the hit chunk ids.
async fn search_chunk_ids(engine: &LensEngine, notebook: &str, query_text: &str) -> Vec<String> {
    let pool = engine.pool().await;
    let data_dir = engine.data_dir_for_test().await;
    let embedder =
        CountingEmbedder::new(Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let q = embedder.embed_query(query_text).expect("embed query");
    let store = LanceVectorStore::new(&data_dir, pool);
    store
        .search(notebook, EMBED_MODEL_ID, EMBED_DIM, &q, 10)
        .await
        .expect("search")
        .into_iter()
        .map(|h| h.chunk_id)
        .collect()
}

// ---------------------------------------------------------------------------
// AC6 — doc-summary RAPTOR node
// ---------------------------------------------------------------------------

/// AC6: after enrichment the summary row exists with the exact contract, is in
/// Lance with its `source_id`, and `purge_source` reclaims it (SQLite cascade +
/// Lance `.only_if`).
#[tokio::test]
async fn summary_node_created_embedded_and_reclaimed() {
    let _flip_guard = flip_serial_guard().await;
    let (_dir, engine) = file_engine().await;
    let _calls = install_mock_provider(&engine).await;
    install_counting_embedder(&engine);
    let (nb, source_id) = seed_indexed_prose_source(&engine).await;

    engine.enqueue_enrichment_for_test(&source_id);
    assert!(
        wait_for_status(&engine, &source_id, "enriched").await,
        "prose source must reach `enriched` via the re-embed flip; got {:?}",
        enrichment_status(&engine, &source_id).await
    );

    // The summary row exists with the exact AC6 contract.
    let pool = engine.pool().await;
    let row = sqlx::query(
        "SELECT id, parent_id, kind, level, section_path, source_id, enrichment, embedding_text \
         FROM chunks WHERE source_id = ? AND kind = 'summary'",
    )
    .bind(&source_id)
    .fetch_one(&pool)
    .await
    .expect("summary row exists");
    assert_eq!(row.get::<String, _>("kind"), "summary");
    assert_eq!(row.get::<i64, _>("level"), 2);
    assert!(row.get::<Option<String>, _>("parent_id").is_none());
    assert_eq!(row.get::<String, _>("section_path"), "");
    assert_eq!(row.get::<String, _>("source_id"), source_id);
    assert!(row.get::<Option<String>, _>("enrichment").is_none());
    let summary_id = row.get::<String, _>("id");

    // The summary vector is in the ACTIVE Lance table (search returns it).
    let hits = search_chunk_ids(&engine, &nb, "analytical engine").await;
    assert!(
        hits.iter().any(|c| c == &summary_id),
        "summary node must be searchable in the active table; hits: {hits:?}"
    );

    // purge_source reclaims it: trash then purge, then the row + vector are gone.
    sqlx::query("UPDATE sources SET trashed_at = ? WHERE id = ?")
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(&source_id)
        .execute(&pool)
        .await
        .unwrap();
    engine.purge_source(&source_id).await.expect("purge");
    let remaining: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM chunks WHERE source_id = ? AND kind = 'summary'")
            .bind(&source_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(remaining, 0, "summary row must cascade on purge");
    let hits = search_chunk_ids(&engine, &nb, "analytical engine").await;
    assert!(
        !hits.iter().any(|c| c == &summary_id),
        "summary vector must be reclaimed from Lance on purge"
    );
}

// ---------------------------------------------------------------------------
// AC8 — registry-driven search resolution after a flip
// ---------------------------------------------------------------------------

/// AC8: after a flip the active registry row + physical table are the gen-1
/// enriched table (gen-0 == formula is dropped); `search` resolves it via the
/// registry and returns the enriched vectors (incl. the summary node). One active
/// row per coordinate; gen-0 table physically gone.
#[tokio::test]
async fn search_resolves_gen_suffixed_active_table_after_flip() {
    let _flip_guard = flip_serial_guard().await;
    let (_dir, engine) = file_engine().await;
    let _calls = install_mock_provider(&engine).await;
    install_counting_embedder(&engine);
    let (nb, source_id) = seed_indexed_prose_source(&engine).await;
    let data_dir = engine.data_dir_for_test().await;

    // Pre-flip: the active table is the gen-0 formula name.
    assert_eq!(
        active_table_name(&engine, &nb).await,
        format!("vec__{nb}__nomic_v15"),
        "pre-flip active must be the gen-0 formula name"
    );

    engine.enqueue_enrichment_for_test(&source_id);
    assert!(
        wait_for_status(&engine, &source_id, "enriched").await,
        "must reach enriched; got {:?}",
        enrichment_status(&engine, &source_id).await
    );

    // Post-flip: exactly one active row, now the gen-1 enriched table.
    assert_eq!(registry_count(&engine, "active").await, 1);
    assert_eq!(
        active_table_name(&engine, &nb).await,
        format!("vec__{nb}__nomic_v15__1"),
        "post-flip active must be the gen-1 enriched table"
    );
    assert_eq!(registry_count(&engine, "stale").await, 0, "stale dropped");
    assert_eq!(
        registry_count(&engine, "building").await,
        0,
        "building promoted"
    );

    // The gen-0 raw table is physically dropped; the gen-1 enriched table lives.
    let tables = live_lance_tables(&data_dir).await;
    assert!(
        !tables.iter().any(|t| t == &format!("vec__{nb}__nomic_v15")),
        "the stale gen-0 table must be dropped; tables: {tables:?}"
    );
    assert!(
        tables
            .iter()
            .any(|t| t == &format!("vec__{nb}__nomic_v15__1")),
        "the enriched gen-1 table must exist; tables: {tables:?}"
    );

    // Search resolves the gen-1 table via the registry and is non-empty (parent +
    // child + summary).
    let hits = search_chunk_ids(&engine, &nb, "analytical engine").await;
    assert!(
        hits.len() >= 3,
        "search must return enriched parent+child+summary; hits: {hits:?}"
    );
}

// ---------------------------------------------------------------------------
// AC7 — crash injection
// ---------------------------------------------------------------------------

/// AC7 `crash_between_flip_txn_commit_and_lance_drop`: a crash AFTER the flip
/// SQLite txn commits but BEFORE the stale Lance table is dropped leaves (a) the
/// active row + table pointing at the COMPLETE enriched (gen-1) table, (b) a
/// `stale` row + orphan gen-0 Lance table, (c) search never mixed/empty; then
/// startup-GC reclaims the stale row AND tolerates a missing Lance table.
#[tokio::test]
async fn crash_between_flip_txn_commit_and_lance_drop() {
    let _flip_guard = flip_serial_guard().await;
    let (dir, engine) = file_engine().await;
    let _calls = install_mock_provider(&engine).await;
    install_counting_embedder(&engine);
    let (nb, source_id) = seed_indexed_prose_source(&engine).await;
    let data_dir = engine.data_dir_for_test().await;

    // Arm the crash: flip_active commits the txn then returns BEFORE dropping the
    // stale Lance table / deleting its row.
    CRASH_AFTER_FLIP_TXN_BEFORE_LANCE_DROP.store(true, Ordering::SeqCst);

    engine.enqueue_enrichment_for_test(&source_id);
    // The flip txn committed → active is the enriched gen-1 table → the worker
    // marks `enriched` (the stale-drop is what was skipped).
    assert!(
        wait_for_status(&engine, &source_id, "enriched").await,
        "the flip txn committed, so the source must reach enriched; got {:?}",
        enrichment_status(&engine, &source_id).await
    );

    // (a) active row points at the COMPLETE enriched gen-1 table.
    assert_eq!(
        active_table_name(&engine, &nb).await,
        format!("vec__{nb}__nomic_v15__1"),
        "active must point at the enriched gen-1 table after the committed flip"
    );
    // (b) a stale row + orphan gen-0 Lance table remain (the crash skipped the drop).
    assert_eq!(
        registry_count(&engine, "stale").await,
        1,
        "the demoted gen-0 row must remain `stale` (crash skipped the drop)"
    );
    let tables = live_lance_tables(&data_dir).await;
    assert!(
        tables.iter().any(|t| t == &format!("vec__{nb}__nomic_v15")),
        "the orphan gen-0 Lance table must still exist; tables: {tables:?}"
    );
    // (d) search never mixed/empty — it resolves the active gen-1 table only.
    let hits = search_chunk_ids(&engine, &nb, "analytical engine").await;
    assert!(
        hits.len() >= 3,
        "search must serve the enriched table, never empty/mixed; hits: {hits:?}"
    );

    // (c) startup-GC reclaims the stale row + drops its Lance table on restart.
    drop(engine);
    let engine2 = reopen_engine(&dir).await;
    assert_eq!(
        registry_count(&engine2, "stale").await,
        0,
        "startup-GC must reclaim the stale row"
    );
    assert_eq!(
        registry_count(&engine2, "active").await,
        1,
        "the active row must survive GC"
    );
    let tables = live_lance_tables(&data_dir).await;
    assert!(
        !tables.iter().any(|t| t == &format!("vec__{nb}__nomic_v15")),
        "GC must drop the orphan gen-0 Lance table; tables: {tables:?}"
    );
    assert!(
        tables
            .iter()
            .any(|t| t == &format!("vec__{nb}__nomic_v15__1")),
        "the enriched gen-1 table must survive GC; tables: {tables:?}"
    );
    // Search still works after GC.
    let hits = search_chunk_ids(&engine2, &nb, "analytical engine").await;
    assert!(
        hits.len() >= 3,
        "search must still serve after GC; hits: {hits:?}"
    );
}

/// AC7 `crash_during_building_table_populate`: a crash (injected embed failure)
/// DURING the building-table populate, BEFORE the flip, leaves a `building` orphan
/// (row + table), the `active` raw gen-0 table untouched and still serving search,
/// and the source `failed`; startup-GC then reclaims the `building` orphan.
#[tokio::test]
async fn crash_during_building_table_populate() {
    let _flip_guard = flip_serial_guard().await;
    let (dir, engine) = file_engine().await;
    let _calls = install_mock_provider(&engine).await;
    // Inject the FAILING embedder so the re-embed populate errors before the flip.
    let failing: Arc<dyn Embedder> = Arc::new(FailingEmbedder);
    engine
        .set_embedder_for_test(failing)
        .expect("inject failing embedder");
    let (nb, source_id) = seed_indexed_prose_source(&engine).await;
    let data_dir = engine.data_dir_for_test().await;

    engine.enqueue_enrichment_for_test(&source_id);
    // The re-embed fails before the flip → status `failed`; raw vectors untouched.
    assert!(
        wait_for_status(&engine, &source_id, "failed").await,
        "an embed failure before the flip must degrade to `failed`; got {:?}",
        enrichment_status(&engine, &source_id).await
    );

    // The active row is still the gen-0 raw table (never touched by the flip).
    assert_eq!(
        active_table_name(&engine, &nb).await,
        format!("vec__{nb}__nomic_v15"),
        "active must remain the raw gen-0 table"
    );
    // A building orphan row exists (the empty gen-1 table was created before the
    // populate failed).
    assert_eq!(
        registry_count(&engine, "building").await,
        1,
        "a building orphan row must remain after the pre-flip failure"
    );
    // Raw vectors still serve search (parent + child; no summary — flip never ran).
    let hits = search_chunk_ids(&engine, &nb, "analytical engine").await;
    assert_eq!(
        hits.len(),
        2,
        "raw vectors (parent+child only, no summary) must still serve; hits: {hits:?}"
    );

    // Startup-GC reclaims the building orphan on restart; active raw table stays.
    drop(engine);
    let engine2 = reopen_engine(&dir).await;
    assert_eq!(
        registry_count(&engine2, "building").await,
        0,
        "startup-GC must reclaim the building orphan"
    );
    assert_eq!(
        registry_count(&engine2, "active").await,
        1,
        "the active raw row must survive GC"
    );
    let tables = live_lance_tables(&data_dir).await;
    assert!(
        tables.iter().any(|t| t == &format!("vec__{nb}__nomic_v15")),
        "the active raw gen-0 table must survive GC; tables: {tables:?}"
    );
}
