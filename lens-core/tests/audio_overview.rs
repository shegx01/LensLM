//! Integration tests for per-notebook Audio Overview persistence + status (#29).
//! Fully offline: the success synth path needs a real TTS model and is out of scope
//! here (gated end-to-end lives with the TTS suites); these cover the persistence,
//! read-path reconciliation (missing/stale), and terminal-row semantics on the
//! failure and cancel paths using a scripted mock `LlmProvider` and no TTS backend.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use async_trait::async_trait;
use lens_core::embedder::{CountingEmbedder, Embedder, EmbeddingBackend};
use lens_core::llm::{LlmProvider, LlmRequest, LlmResponse};
use lens_core::{AudioOverviewStatus, Length, LensEngine, LensError};
use sqlx::SqlitePool;
use tokio_util::sync::CancellationToken;

// Fixtures

async fn insert_source(
    pool: &SqlitePool,
    notebook_id: &str,
    source_id: &str,
    raw_content_hash: &str,
) {
    sqlx::query(
        "INSERT INTO sources \
         (id, notebook_id, kind, title, status, locator, selected, token_count, \
          content_hash, raw_content_hash, created_at) \
         VALUES (?, ?, 'text', ?, 'indexed', '/tmp/seed.txt', 1, 50, ?, ?, ?)",
    )
    .bind(source_id)
    .bind(notebook_id)
    .bind(format!("title-{source_id}"))
    .bind(format!("content-{source_id}"))
    .bind(raw_content_hash)
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

/// Raw INSERT of a terminal row (the crate's `upsert_overview` is `pub(crate)`; tests
/// seed rows via SQL, mirroring the source/chunk fixtures).
async fn insert_overview_row(
    pool: &SqlitePool,
    notebook_id: &str,
    path: &str,
    status: &str,
    source_set_hash: &str,
) {
    sqlx::query(
        "INSERT INTO audio_overviews (notebook_id, path, generated_at, status, source_set_hash) \
         VALUES (?, ?, ?, ?, ?)",
    )
    .bind(notebook_id)
    .bind(path)
    .bind(chrono::Utc::now().to_rfc3339())
    .bind(status)
    .bind(source_set_hash)
    .execute(pool)
    .await
    .expect("insert audio_overviews row");
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

/// Scripted mock returning a fixed valid dialogue script — lets the dialogue phase
/// succeed so the synth phase (with no TTS backend) is the failure under test.
struct ScriptedProvider {
    calls: Arc<AtomicU32>,
    response: String,
}

#[async_trait]
impl LlmProvider for ScriptedProvider {
    fn model_id(&self) -> &str {
        "mock-model"
    }
    async fn reachable(&self) -> bool {
        true
    }
    async fn generate(&self, _req: &LlmRequest) -> Result<LlmResponse, LensError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(LlmResponse {
            text: self.response.clone(),
            tokens_used: 10,
        })
    }
}

/// A minimal valid Short script (>= 8 turns, both speakers, cites only sA/sB).
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

/// A two-source notebook wired so `engine.generate_dialogue` succeeds (embedder +
/// coord injected, scripted provider installed). No TTS backend is configured, so a
/// subsequent synth fails.
async fn seed_dialogue_ready_notebook() -> (LensEngine, SqlitePool, String, tempfile::TempDir) {
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

    let provider = Arc::new(ScriptedProvider {
        calls: Arc::new(AtomicU32::new(0)),
        response: valid_short_json(),
    });
    engine.set_llm_provider(Some(provider)).await;
    insert_source(&pool, nb.as_str(), "sA", "hash-a").await;
    insert_source(&pool, nb.as_str(), "sB", "hash-b").await;
    insert_chunk(&pool, "sA", "c1", "alpha content").await;
    insert_chunk(&pool, "sB", "c2", "beta content").await;

    (engine, pool, nb.as_str().to_string(), dir)
}

fn no_phase() -> impl Fn(lens_core::TtsPhase) + Send + Sync {
    |_p| {}
}

#[tokio::test]
async fn migration_applies_and_table_exists_at_count_23() {
    let engine = LensEngine::for_test().await;
    assert_eq!(engine.migration_count().await.unwrap(), 23);
    let pool = engine.pool().await;
    let exists: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='audio_overviews'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(exists, 1, "audio_overviews table must exist after 0023");
}

// Read-path round-trip + reconciliation

#[tokio::test]
async fn ready_row_reads_back_then_downgrades_to_missing_when_file_gone() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;

    // A ready row whose stored hash matches the (empty-source-set) current hash so it
    // does not read as stale.
    let hash = engine.source_set_hash_for_test(nb.as_str()).await.unwrap();
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), b"RIFF....WAVE").unwrap();
    let path = tmp.path().to_string_lossy().into_owned();
    insert_overview_row(&pool, nb.as_str(), &path, "ready", &hash).await;

    let rec = engine
        .get_audio_overview_status(nb.as_str())
        .await
        .unwrap()
        .expect("row present");
    assert_eq!(rec.status, AudioOverviewStatus::Ready);
    assert_eq!(rec.source_set_hash, hash);

    std::fs::remove_file(tmp.path()).unwrap();
    let rec = engine
        .get_audio_overview_status(nb.as_str())
        .await
        .unwrap()
        .expect("row present");
    assert_eq!(
        rec.status,
        AudioOverviewStatus::Missing,
        "ready row with a vanished file must reconcile to Missing"
    );
}

#[tokio::test]
async fn ready_row_with_changed_sources_reads_as_stale() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_source(&pool, nb.as_str(), "sA", "hash-a").await;

    let hash_before = engine.source_set_hash_for_test(nb.as_str()).await.unwrap();
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), b"RIFF....WAVE").unwrap();
    insert_overview_row(
        &pool,
        nb.as_str(),
        &tmp.path().to_string_lossy(),
        "ready",
        &hash_before,
    )
    .await;

    // Matching set → Ready.
    assert_eq!(
        engine
            .get_audio_overview_status(nb.as_str())
            .await
            .unwrap()
            .unwrap()
            .status,
        AudioOverviewStatus::Ready
    );

    // Add a second selected source → hash drifts → Stale.
    insert_source(&pool, nb.as_str(), "sB", "hash-b").await;
    assert_eq!(
        engine
            .get_audio_overview_status(nb.as_str())
            .await
            .unwrap()
            .unwrap()
            .status,
        AudioOverviewStatus::Stale
    );
}

#[tokio::test]
async fn failed_row_reads_back_as_failed() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_overview_row(
        &pool,
        nb.as_str(),
        "/nonexistent/overview.wav",
        "failed",
        "h",
    )
    .await;

    let rec = engine
        .get_audio_overview_status(nb.as_str())
        .await
        .unwrap()
        .unwrap();
    // A failed row is NOT reconciled against the file — it stays failed.
    assert_eq!(rec.status, AudioOverviewStatus::Failed);
}

#[tokio::test]
async fn status_is_none_when_never_generated() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    assert!(
        engine
            .get_audio_overview_status(nb.as_str())
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn source_set_hash_is_stable_and_reflects_ids_and_content() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;

    insert_source(&pool, nb.as_str(), "sB", "hash-b").await;
    insert_source(&pool, nb.as_str(), "sA", "hash-a").await;
    let h1 = engine.source_set_hash_for_test(nb.as_str()).await.unwrap();

    // Deterministic: the set is sorted before hashing, so repeated reads (and the
    // arbitrary DB row order) yield the same hash — the order-independence guarantee.
    let h1_again = engine.source_set_hash_for_test(nb.as_str()).await.unwrap();
    assert_eq!(h1, h1_again, "hash is stable across reads");

    // Different ids → different hash even with the same content hashes.
    let nb2 = engine.create_notebook("nb2", None, None).await.unwrap().id;
    insert_source(&pool, nb2.as_str(), "sA2", "hash-a").await;
    insert_source(&pool, nb2.as_str(), "sB2", "hash-b").await;
    let h_other_ids = engine.source_set_hash_for_test(nb2.as_str()).await.unwrap();
    assert_ne!(h1, h_other_ids, "ids participate in the hash");

    // Content change flips the hash.
    sqlx::query("UPDATE sources SET raw_content_hash = 'hash-a-v2' WHERE id = 'sA'")
        .execute(&pool)
        .await
        .unwrap();
    let h2 = engine.source_set_hash_for_test(nb.as_str()).await.unwrap();
    assert_ne!(h1, h2, "a content-hash change must change the set hash");
}

// generate_and_persist_overview terminal-row semantics

#[tokio::test]
async fn dialogue_phase_failure_persists_failed_row() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;

    // No provider configured → the dialogue phase errors (Model).
    let res = engine
        .generate_and_persist_overview(
            nb.as_str(),
            Length::Short,
            no_phase(),
            CancellationToken::new(),
        )
        .await;
    assert!(matches!(res, Err(LensError::Model(_))));

    let rec = engine
        .get_audio_overview_status(nb.as_str())
        .await
        .unwrap()
        .expect("failed row persisted on dialogue-phase failure");
    assert_eq!(rec.status, AudioOverviewStatus::Failed);
}

#[tokio::test]
async fn synth_phase_failure_persists_failed_row() {
    let (engine, _pool, nb, _dir) = seed_dialogue_ready_notebook().await;

    // Dialogue succeeds (scripted provider) but no TTS backend is downloaded → synth
    // errors (Tts).
    let res = engine
        .generate_and_persist_overview(&nb, Length::Short, no_phase(), CancellationToken::new())
        .await;
    assert!(
        matches!(res, Err(LensError::Tts(_))),
        "expected a synth-phase Tts failure, got {res:?}"
    );

    let rec = engine
        .get_audio_overview_status(&nb)
        .await
        .unwrap()
        .expect("failed row persisted on synth-phase failure");
    assert_eq!(rec.status, AudioOverviewStatus::Failed);
}

#[tokio::test]
async fn cancel_writes_no_failed_row_and_preserves_prior() {
    let (engine, pool, nb, _dir) = seed_dialogue_ready_notebook().await;

    // A prior ready row that a cancel must leave untouched.
    let hash = engine.source_set_hash_for_test(&nb).await.unwrap();
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), b"RIFF....WAVE").unwrap();
    insert_overview_row(&pool, &nb, &tmp.path().to_string_lossy(), "ready", &hash).await;

    let cancel = CancellationToken::new();
    cancel.cancel();
    let out = engine
        .generate_and_persist_overview(&nb, Length::Short, no_phase(), cancel)
        .await
        .expect("cancel is not an error");
    assert!(out.is_none(), "cancel yields an idle-equivalent None");

    let rec = engine
        .get_audio_overview_status(&nb)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        rec.status,
        AudioOverviewStatus::Ready,
        "prior ready row must survive a cancel (no failed overwrite)"
    );
    assert_eq!(rec.source_set_hash, hash);
}

#[tokio::test]
async fn purge_notebook_cascades_audio_overview_row() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_overview_row(&pool, nb.as_str(), "/tmp/overview.wav", "failed", "h").await;

    // Purge (the hard-delete path that fires FK cascade) requires a trashed notebook.
    engine.trash_notebook(&nb).await.unwrap();
    engine.purge_notebook(&nb).await.unwrap();

    let remaining: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM audio_overviews WHERE notebook_id = ?")
            .bind(nb.as_str())
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        remaining, 0,
        "FK ON DELETE CASCADE must remove the overview row"
    );
}

#[tokio::test]
async fn is_overview_generating_reflects_tts_registry() {
    let engine = LensEngine::for_test().await;
    assert!(
        !engine.is_overview_generating("nb-x"),
        "no token → not generating"
    );

    let token = engine.register_tts("nb-x");
    let _guard = engine.tts_cancel_guard("nb-x", token.clone());
    assert!(
        engine.is_overview_generating("nb-x"),
        "live token → generating"
    );

    token.cancel();
    assert!(
        !engine.is_overview_generating("nb-x"),
        "a cancelled-but-not-dropped token must read as not generating"
    );
}

#[test]
fn audio_overview_status_serde_snake_case() {
    for (status, wire) in [
        (AudioOverviewStatus::Ready, "ready"),
        (AudioOverviewStatus::Failed, "failed"),
        (AudioOverviewStatus::Stale, "stale"),
        (AudioOverviewStatus::Missing, "missing"),
    ] {
        assert_eq!(serde_json::to_value(status).unwrap(), wire);
    }
}
