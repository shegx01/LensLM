//! Integration tests for the M4 Phase-3 enrichment WIRING (Step 3).
//!
//! Covers the engine-side infrastructure: the background worker + `mpsc` queue
//! (AC3), startup-GC of orphaned `building`/`stale` re-embed tables (AC7-GC), the
//! `enriching → pending` crash-recovery reset + queue-rebuild (AC12), and the
//! graceful-degrade / back-fill rescan seam (AC10). The worker's job body is the
//! Step-3 stub (it flips `enrichment_status` through the lifecycle); the LLM pass
//! (Step 4) and re-embed flip (Step 5) are out of scope here.
//!
//! These tests build a FILE-BACKED engine via [`LensEngine::init`] over a tempdir
//! so the startup recovery/GC paths in `init` run for real (the in-memory
//! `for_test()` cannot re-`init` the same DB).

use std::sync::Arc;
use std::time::Duration;

use lens_core::vector_store::{Coordinate, LanceVectorStore, VectorRow, VectorStore};
use lens_core::{DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL_ID, EmbeddingBackend, LensEngine};

fn coord(nb: &str, model: &str, dim: usize) -> Coordinate {
    Coordinate::new(nb, EmbeddingBackend::Fastembed, model, dim)
}
use sqlx::Row;
use tempfile::TempDir;
use tokio::sync::Notify;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// File-backed engine over a fresh tempdir (so `init`'s recovery/GC/queue-rebuild
/// paths run for real). Returns the dir guard so it outlives the engine.
/// Step 6: the worker now honors `AppConfig.enrichment.enabled` (default OFF).
/// These tests exercise the worker body (status transitions, lock-freedom, dedup,
/// degrade, queue-rebuild across restart), so enrichment must be ON — and
/// PERSISTED to disk so a `reopen_engine` reloads it from `config.json`.
fn enable_enrichment_on_disk(dir: &TempDir) {
    let mut cfg = lens_core::config::AppConfig::load(dir.path()).expect("load config");
    cfg.enrichment.enabled = true;
    cfg.save(dir.path()).expect("save config");
}

async fn file_engine() -> (TempDir, LensEngine) {
    let dir = tempfile::tempdir().expect("tempdir");
    // Init once to materialize the default config, then enable enrichment on disk
    // and re-init so both the in-memory config and the disk file have it enabled.
    LensEngine::init(dir.path()).await.expect("engine init");
    enable_enrichment_on_disk(&dir);
    let engine = LensEngine::init(dir.path()).await.expect("engine re-init");
    (dir, engine)
}

/// Re-opens an engine over the SAME data dir (simulating an app restart). The new
/// engine's `init` runs crash-recovery + startup-GC + queue-rebuild against the
/// persisted DB.
async fn reopen_engine(dir: &TempDir) -> LensEngine {
    LensEngine::init(dir.path()).await.expect("engine re-init")
}

/// Inserts a notebook + an `indexed` source row directly (bypassing ingest, which
/// needs the embedder/tokenizer). Returns the source id.
async fn seed_indexed_source(
    engine: &LensEngine,
    enrichment_status: Option<&str>,
) -> (String, String) {
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
         enrichment_status, created_at) \
         VALUES (?, ?, 'text', 'seed', 'indexed', '/tmp/seed.txt', 1, ?, ?)",
    )
    .bind(&source_id)
    .bind(&nb)
    .bind(enrichment_status)
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(&pool)
    .await
    .expect("insert source");
    (nb, source_id)
}

/// Reads a source's `enrichment_status` column (NULL → `None`).
async fn enrichment_status(engine: &LensEngine, source_id: &str) -> Option<String> {
    let pool = engine.pool().await;
    sqlx::query("SELECT enrichment_status FROM sources WHERE id = ?")
        .bind(source_id)
        .fetch_one(&pool)
        .await
        .expect("fetch source")
        .get::<Option<String>, _>("enrichment_status")
}

/// Polls `enrichment_status` until it equals `want` or the timeout elapses.
async fn wait_for_status(engine: &LensEngine, source_id: &str, want: &str) -> bool {
    for _ in 0..200 {
        if enrichment_status(engine, source_id).await.as_deref() == Some(want) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    false
}

/// Inserts a registry row directly with the given physical table name + status.
async fn seed_registry_row(engine: &LensEngine, notebook: &str, table: &str, status: &str) {
    let pool = engine.pool().await;
    sqlx::query(
        "INSERT INTO embedding_index \
         (id, notebook_id, model, dim, prefix_convention, lance_table_name, status, created_at) \
         VALUES (?, ?, ?, ?, 'search_document', ?, ?, ?)",
    )
    .bind(uuid::Uuid::now_v7().to_string())
    .bind(notebook)
    .bind(DEFAULT_EMBED_MODEL_ID)
    .bind(DEFAULT_EMBED_DIM as i64)
    .bind(table)
    .bind(status)
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(&pool)
    .await
    .expect("insert registry row");
}

/// Counts registry rows for a given status.
async fn registry_count(engine: &LensEngine, status: &str) -> i64 {
    let pool = engine.pool().await;
    sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM embedding_index WHERE status = ?")
        .bind(status)
        .fetch_one(&pool)
        .await
        .expect("count registry rows")
}

fn unit_vector(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DEFAULT_EMBED_DIM];
    v[axis] = 1.0;
    v
}

// ---------------------------------------------------------------------------
// AC3 — background worker, mpsc enqueue, lock-free job body
// ---------------------------------------------------------------------------

/// AC3/AC10: a directly-enqueued source is picked up by the worker. With NO
/// chunks AND NO reachable provider (this wiring fixture seeds neither), the
/// Step-4 body degrades gracefully to `pending` (raw vectors serve; a later
/// rescan re-drives it). The observable signal is that the worker dequeued the
/// job and advanced the status off its seeded NULL.
#[tokio::test]
async fn worker_picks_up_and_degrades_chunkless_source() {
    let (_dir, engine) = file_engine().await;
    let (_nb, source_id) = seed_indexed_source(&engine, None).await;

    engine.enqueue_enrichment_for_test(&source_id);

    // No provider + no chunks ⇒ graceful degrade to `pending` (AC10).
    assert!(
        wait_for_status(&engine, &source_id, "pending").await,
        "worker should pick up the job and degrade to `pending`; got {:?}",
        enrichment_status(&engine, &source_id).await
    );
}

/// AC3: the worker holds NO `ingest_lock` permit during its job body, so a
/// concurrent `purge_source` is NOT blocked while a job is in flight. A test gate
/// pins a job inside the stub body; we assert `purge_source` completes while the
/// job is pinned, then release the gate.
#[tokio::test]
async fn worker_holds_no_ingest_permit_during_job() {
    let (_dir, engine) = file_engine().await;
    let (_nb, source_id) = seed_indexed_source(&engine, None).await;
    // A SECOND, trashed source we can purge concurrently (purge requires trashed).
    let (_nb2, purgeable) = seed_indexed_source(&engine, None).await;
    {
        let pool = engine.pool().await;
        sqlx::query("UPDATE sources SET trashed_at = ? WHERE id = ?")
            .bind(chrono::Utc::now().to_rfc3339())
            .bind(&purgeable)
            .execute(&pool)
            .await
            .unwrap();
    }

    // Install a gate so the worker blocks inside the stub job body.
    let gate = Arc::new(Notify::new());
    engine
        .set_enrichment_gate_for_test(Some(gate.clone()))
        .await;

    engine.enqueue_enrichment_for_test(&source_id);
    // Let the worker reach the gate (status flips to `enriching` first).
    assert!(
        wait_for_status(&engine, &source_id, "enriching").await,
        "worker should reach `enriching` and block on the gate"
    );

    // While the job is pinned in its body, a concurrent purge MUST proceed
    // (the worker holds no ingest_lock permit). If it held the permit this would
    // hang and the timeout below would fail.
    let purge = tokio::time::timeout(Duration::from_secs(5), engine.purge_source(&purgeable)).await;
    assert!(
        purge.is_ok(),
        "purge_source must not block on the in-flight enrichment job (worker holds no permit)"
    );
    purge.unwrap().expect("purge should succeed");

    // Release the gate; the chunkless/provider-less job degrades to `pending`.
    gate.notify_one();
    assert!(
        wait_for_status(&engine, &source_id, "pending").await,
        "released job should degrade to `pending` (no chunks/provider)"
    );
}

/// AC3: a `try_send` into a FULL channel does not deadlock (it logs + drops; the
/// rescan recovers). We fill the channel to capacity while the worker is blocked
/// on the gate, then issue one more enqueue — it must return promptly.
#[tokio::test]
async fn enqueue_into_full_channel_does_not_deadlock() {
    let (_dir, engine) = file_engine().await;
    let (_nb, blocker) = seed_indexed_source(&engine, None).await;

    // Pin the worker inside a job so it stops draining the channel.
    let gate = Arc::new(Notify::new());
    engine
        .set_enrichment_gate_for_test(Some(gate.clone()))
        .await;
    engine.enqueue_enrichment_for_test(&blocker);
    assert!(wait_for_status(&engine, &blocker, "enriching").await);

    // Fill the channel past capacity. The worker dequeued `blocker` (now blocked),
    // so the channel can hold `capacity` more. Push capacity + 64 — the overflow
    // enqueues must all return promptly (try_send drops on Full, never blocks).
    let cap = engine.enrichment_queue_capacity();
    let fill = tokio::time::timeout(Duration::from_secs(5), async {
        for i in 0..(cap + 64) {
            engine.enqueue_enrichment_for_test(&format!("ghost-{i}"));
        }
    })
    .await;
    assert!(
        fill.is_ok(),
        "enqueue into a full channel must not deadlock (try_send drops on Full)"
    );

    gate.notify_one();
}

/// AC13(b): a job whose source was purged before the worker dequeues it is dropped
/// without panic (the worker re-checks existence).
#[tokio::test]
async fn worker_drops_job_for_missing_source() {
    let (_dir, engine) = file_engine().await;
    // Enqueue a source id that does not exist. The worker must not panic/error.
    engine.enqueue_enrichment_for_test("does-not-exist");
    // Give the worker a moment to process + drop the job.
    tokio::time::sleep(Duration::from_millis(150)).await;
    // No assertion beyond "did not panic / engine still usable":
    let (_nb, source_id) = seed_indexed_source(&engine, None).await;
    engine.enqueue_enrichment_for_test(&source_id);
    assert!(
        wait_for_status(&engine, &source_id, "pending").await,
        "worker still processes valid jobs after a missing-source job (degrades to pending)"
    );
}

// ---------------------------------------------------------------------------
// AC7-GC — startup-GC of orphaned building/stale re-embed tables
// ---------------------------------------------------------------------------

/// AC7-GC: a seeded `building` row AND a `stale` row are BOTH reclaimed at startup
/// — including when the Lance table is already absent (the delete still succeeds).
/// The `active` row is untouched.
#[tokio::test]
async fn startup_gc_reclaims_building_and_stale_rows() {
    let (dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .unwrap()
        .id
        .to_string();
    let data_dir = engine.data_dir_for_test().await;

    // 1) An `active` row WITH a real Lance table (must survive GC).
    let store = LanceVectorStore::new(&data_dir, engine.pool().await);
    store
        .add(
            &coord(&nb, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM),
            vec![VectorRow {
                chunk_id: "live".into(),
                source_id: "s-live".into(),
                notebook_id: nb.clone(),
                level: 1,
                vector: unit_vector(0),
            }],
        )
        .await
        .expect("seed active table");
    // 2) A `building` row WITH a real physical Lance table (crash mid-populate),
    //    created via the real Step-5 `create_building_table` (gen-suffixed name +
    //    registered `building` row) then populated via `add_to_table`.
    let building_table = store
        .create_building_table(&coord(&nb, DEFAULT_EMBED_MODEL_ID, DEFAULT_EMBED_DIM))
        .await
        .expect("create building table");
    assert_eq!(
        building_table,
        format!("vec__{nb}__fastembed__nomic_v15__d{DEFAULT_EMBED_DIM}__1")
    );
    store
        .add_to_table(
            &building_table,
            vec![VectorRow {
                chunk_id: "build".into(),
                source_id: "s-build".into(),
                notebook_id: nb.clone(),
                level: 1,
                vector: unit_vector(1),
            }],
            DEFAULT_EMBED_DIM,
        )
        .await
        .expect("seed building table");
    // 3) A `stale` row whose Lance table is ALREADY GONE (crash between flip-txn
    //    commit and Lance-drop, then the drop happened but the row delete didn't).
    seed_registry_row(&engine, &nb, "vec__phantom__nomic_v15__0", "stale").await;

    assert_eq!(registry_count(&engine, "building").await, 1);
    assert_eq!(registry_count(&engine, "stale").await, 1);
    assert_eq!(registry_count(&engine, "active").await, 1);

    // Re-open the engine over the same dir → startup-GC runs.
    drop(engine);
    let engine2 = reopen_engine(&dir).await;

    assert_eq!(
        registry_count(&engine2, "building").await,
        0,
        "building rows must be reclaimed at startup"
    );
    assert_eq!(
        registry_count(&engine2, "stale").await,
        0,
        "stale rows must be reclaimed at startup (even with a missing Lance table)"
    );
    assert_eq!(
        registry_count(&engine2, "active").await,
        1,
        "the active row must be untouched by GC"
    );

    // The building Lance table must be physically dropped; the active table stays.
    let names = {
        let conn = lancedb::connect(data_dir.join("lancedb").to_string_lossy().as_ref())
            .execute()
            .await
            .unwrap();
        conn.table_names().execute().await.unwrap()
    };
    assert!(
        !names.iter().any(|n| n == &building_table),
        "the building Lance table must be dropped by GC; live tables: {names:?}"
    );
    assert!(
        names
            .iter()
            .any(|n| n == &format!("vec__{nb}__fastembed__nomic_v15__d{DEFAULT_EMBED_DIM}")),
        "the active Lance table must survive GC; live tables: {names:?}"
    );
}

// ---------------------------------------------------------------------------
// AC12 — enrichment crash-recovery + queue-rebuild
// ---------------------------------------------------------------------------

/// AC12: a source stranded in `enriching` by a crash is reset to `pending` on
/// restart (a SEPARATE reset from the SourceStatus one), and the queue-rebuild
/// re-enqueues it (so the worker walks it to `enriched`). The `SourceStatus`
/// stays `indexed` throughout.
#[tokio::test]
async fn restart_resets_enriching_to_pending_and_reenqueues() {
    let (dir, engine) = file_engine().await;
    let (_nb, source_id) = seed_indexed_source(&engine, Some("enriching")).await;
    drop(engine);

    let engine2 = reopen_engine(&dir).await;
    // The reset+rebuild ran in init: `enriching → pending`, then the queue-rebuild
    // re-enqueues it. With no chunks/provider the Step-4 worker degrades back to
    // `pending` (the stable terminal state); the key invariant is that the stranded
    // `enriching` was reset off `enriching` (never left stuck).
    assert!(
        wait_for_status(&engine2, &source_id, "pending").await,
        "a stranded `enriching` source must be reset to `pending` and re-enqueued; got {:?}",
        enrichment_status(&engine2, &source_id).await
    );
    assert_ne!(
        enrichment_status(&engine2, &source_id).await.as_deref(),
        Some("enriching"),
        "the stranded `enriching` must not survive the restart reset"
    );
    // SourceStatus is untouched (still indexed — searchable on raw vectors).
    let pool = engine2.pool().await;
    let status: String = sqlx::query_scalar("SELECT status FROM sources WHERE id = ?")
        .bind(&source_id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(status, "indexed", "SourceStatus must stay indexed");
}

/// AC12: the SourceStatus crash-recovery reset (`parsing`/`embedding` → `error`)
/// is UNCHANGED by the new enrichment reset — a `parsing` source still flips to
/// `error`, and a non-`enriching` enrichment status is left alone.
#[tokio::test]
async fn restart_sourcestatus_recovery_unchanged() {
    let (dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .unwrap()
        .id
        .to_string();
    let pool = engine.pool().await;
    let parsing = uuid::Uuid::now_v7().to_string();
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, created_at) \
         VALUES (?, ?, 'text', 'p', 'parsing', '/tmp/p.txt', 1, ?)",
    )
    .bind(&parsing)
    .bind(&nb)
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(&pool)
    .await
    .unwrap();
    drop(engine);

    let engine2 = reopen_engine(&dir).await;
    let pool = engine2.pool().await;
    let status: String = sqlx::query_scalar("SELECT status FROM sources WHERE id = ?")
        .bind(&parsing)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(
        status, "error",
        "the SourceStatus parsing→error recovery must be unchanged by the enrichment reset"
    );
}

/// AC12: the queue-rebuild enqueues only the eligible sources — `indexed` with
/// `enrichment_status` in (NULL, pending, failed) — and skips `enriched`/`skipped`.
#[tokio::test]
async fn queue_rebuild_enqueues_only_eligible_sources() {
    let (dir, engine) = file_engine().await;
    let (_n1, none_src) = seed_indexed_source(&engine, None).await;
    let (_n2, failed_src) = seed_indexed_source(&engine, Some("failed")).await;
    let (_n3, enriched_src) = seed_indexed_source(&engine, Some("enriched")).await;
    let (_n4, skipped_src) = seed_indexed_source(&engine, Some("skipped")).await;
    drop(engine);

    let engine2 = reopen_engine(&dir).await;

    // none + failed → re-enqueued → worker picks them up and (no chunks/provider)
    // degrades them to `pending`. The eligibility (re-enqueue) is what's asserted.
    assert!(
        wait_for_status(&engine2, &none_src, "pending").await,
        "a `none` source must be re-enqueued (and degrades to `pending`)"
    );
    assert!(
        wait_for_status(&engine2, &failed_src, "pending").await,
        "a `failed` source must be re-enqueued (and degrades to `pending`)"
    );
    // enriched stays enriched (dedup); skipped stays skipped (not eligible).
    assert_eq!(
        enrichment_status(&engine2, &enriched_src).await.as_deref(),
        Some("enriched"),
        "an `enriched` source must not be re-enqueued/changed"
    );
    assert_eq!(
        enrichment_status(&engine2, &skipped_src).await.as_deref(),
        Some("skipped"),
        "a `skipped` source must not be re-enqueued"
    );
}

// ---------------------------------------------------------------------------
// AC10 — graceful degrade + back-fill rescan seam
// ---------------------------------------------------------------------------

/// AC10: with no reachable provider the source stays eligible; installing a
/// provider via the rescan hook re-enqueues it. (Step 3 worker is a stub, so we
/// assert the rescan path re-binds + re-enqueues — the LLM dispatch is Step 4.)
#[tokio::test]
async fn rescan_rebinds_provider_and_reenqueues() {
    let (_dir, engine) = file_engine().await;
    // Seed a config with a local ollama model so the factory builds a provider.
    // Step 6: rescan now reads `enrichment.{enabled,cloud_consent}` from config
    // (the consent flag is no longer a threaded param), so enable enrichment.
    let mut config = engine.config().await;
    config.enrichment.enabled = true;
    // A fixed always-refused port (127.0.0.1:1), NOT the default Ollama port
    // (11434): the enrichment-model preflight (issue #90) now marks a source
    // `failed` when a REACHABLE Ollama is missing the configured model, so a real
    // Ollama running on 11434 (dev machines) would flip this test's expected
    // `pending` to `failed`. Port 1 is deterministically unreachable everywhere
    // (the parallel-test-safe pattern used across the suite), so `reachable()` is
    // false ⇒ the preflight proceeds ⇒ the worker degrades to `pending` as intended.
    config.models = vec![lens_core::config::ModelConfig {
        provider: "ollama".to_string(),
        base_url: "http://127.0.0.1:1".to_string(),
        model: "llama3".to_string(),
        ..Default::default()
    }];
    engine.set_config(config).await;

    assert!(
        engine.llm_provider().await.is_none(),
        "no provider installed initially"
    );

    let (_nb, source_id) = seed_indexed_source(&engine, Some("failed")).await;
    engine
        .rescan_enrichment_on_provider_change()
        .await
        .expect("rescan");

    assert!(
        engine.llm_provider().await.is_some(),
        "rescan must install the provider from config"
    );
    // The provider points at an always-refused localhost port (not reachable), and
    // the source has no chunks ⇒ the re-enqueued job degrades to `pending`. The
    // assertion is that the rescan RE-ENQUEUED it (the worker dequeued + advanced).
    assert!(
        wait_for_status(&engine, &source_id, "pending").await,
        "rescan must re-enqueue the eligible source (which then degrades to `pending`)"
    );
}
