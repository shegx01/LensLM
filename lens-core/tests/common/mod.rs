//! Shared entity-graph seed helpers for integration tests. Kept offline: no
//! model downloads, no LLM — callers hand-build nodes/edges/mentions.
//!
//! Compiled once per integration-test binary; not every test uses every helper.
#![allow(dead_code)]

use lens_core::LensEngine;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use tempfile::TempDir;

/// `LensEngine::init` spawns a model-catalog refresh; a fresh cache file makes
/// `refresh_if_stale` return before its HTTP call, keeping tests offline.
pub fn pin_catalog_offline(data_dir: &Path) {
    let path = data_dir.join(lens_core::MODELS_CATALOG_RELPATH);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create catalog dir");
    }
    std::fs::write(&path, b"{\"models\":[]}").expect("pin catalog");
}

/// A file-backed engine with the tokenizer disabled (offline, deterministic).
pub async fn file_engine() -> (TempDir, LensEngine) {
    let dir = tempfile::tempdir().expect("tempdir");
    pin_catalog_offline(dir.path());
    let engine = LensEngine::init(dir.path()).await.expect("engine init");
    engine.disable_tokenizer_for_test();
    (dir, engine)
}

/// A file-backed engine (relocation needs a real data dir) with a deterministic
/// embedder, an `active` coordinate, and two same-name nodes across two sources —
/// enough for `resolve_notebook_for_test` to reach the entity-vector write.
/// Returns the dir guard, the engine, and the notebook id.
pub async fn resolution_ready_engine() -> (TempDir, LensEngine, String) {
    use lens_core::{CountingEmbedder, Embedder, EmbeddingBackend};
    use std::sync::atomic::AtomicUsize;

    let (dir, engine) = file_engine().await;
    let embedder: Arc<dyn Embedder> = Arc::new(CountingEmbedder::new(
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    ));
    engine
        .set_embedder_for_test(embedder, EmbeddingBackend::Fastembed)
        .expect("inject embedder");

    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;
    let (model, dim, backend) = engine
        .resolve_notebook_embedding(&lens_core::NotebookId::from(nb.clone()))
        .await
        .expect("resolve embedding");
    sqlx::query(
        "INSERT INTO embedding_index \
         (id, notebook_id, model, dim, prefix_convention, lance_table_name, status, backend, created_at) \
         VALUES (?, ?, ?, ?, 'nomic', ?, 'active', ?, ?)",
    )
    .bind(uuid::Uuid::now_v7().to_string())
    .bind(&nb)
    .bind(&model)
    .bind(dim as i64)
    .bind(format!("chunks__{nb}"))
    .bind(backend.as_str())
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(&pool)
    .await
    .expect("seed active coord");

    for name in ["Gamma", "gamma"] {
        let source_id = uuid::Uuid::now_v7().to_string();
        sqlx::query(
            "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, created_at) \
             VALUES (?, ?, 'text', 'seed', 'indexed', '/tmp/seed.txt', 1, ?)",
        )
        .bind(&source_id)
        .bind(&nb)
        .bind(chrono::Utc::now().to_rfc3339())
        .execute(&pool)
        .await
        .expect("insert source");
        sqlx::query(
            "INSERT INTO entity_nodes (id, notebook_id, source_id, kind, name, definition, created_at) \
             VALUES (?, ?, ?, 'concept', ?, NULL, ?)",
        )
        .bind(uuid::Uuid::now_v7().to_string())
        .bind(&nb)
        .bind(&source_id)
        .bind(name)
        .bind(chrono::Utc::now().to_rfc3339())
        .execute(&pool)
        .await
        .expect("insert entity node");
    }
    (dir, engine, nb)
}

/// One captured `tracing` event.
pub struct CapturedEvent {
    pub level: tracing::Level,
    pub body: String,
}

static SINK: Mutex<Option<Arc<Mutex<Vec<CapturedEvent>>>>> = Mutex::new(None);

fn sink_slot() -> std::sync::MutexGuard<'static, Option<Arc<Mutex<Vec<CapturedEvent>>>>> {
    SINK.lock().unwrap_or_else(|e| e.into_inner())
}

struct FieldVisitor(String);

impl tracing::field::Visit for FieldVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write;
        let _ = write!(self.0, " {}={:?}", field.name(), value);
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        use std::fmt::Write;
        let _ = write!(self.0, " {}={}", field.name(), value);
    }
}

struct CaptureLayer;

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CaptureLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        let slot = sink_slot();
        let Some(buf) = slot.as_ref() else { return };
        let mut visitor = FieldVisitor(String::new());
        event.record(&mut visitor);
        if let Ok(mut events) = buf.lock() {
            events.push(CapturedEvent {
                level: *event.metadata().level(),
                body: visitor.0,
            });
        }
    }
}

/// Holds the capture buffer and the cross-test exclusion lock; drop clears the sink.
pub struct Capture {
    _exclusive: std::sync::MutexGuard<'static, ()>,
    buf: Arc<Mutex<Vec<CapturedEvent>>>,
}

impl Capture {
    /// True when some captured event's fields contain every needle.
    pub fn any_with(&self, needles: &[&str]) -> bool {
        let events = self.buf.lock().unwrap_or_else(|e| e.into_inner());
        events
            .iter()
            .any(|e| needles.iter().all(|n| e.body.contains(n)))
    }

    pub fn count_at(&self, level: tracing::Level) -> usize {
        let events = self.buf.lock().unwrap_or_else(|e| e.into_inner());
        events.iter().filter(|e| e.level == level).count()
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        *sink_slot() = None;
    }
}

/// Captures `tracing` events for the calling test, including those emitted from
/// `tokio::spawn` and `spawn_blocking` — the subscriber is process-global because a
/// thread-local one does not reach spawned tasks. Serialized across tests, so the
/// global sink only ever holds one test's buffer.
pub fn capture_logs() -> Capture {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    static EXCLUSIVE: Mutex<()> = Mutex::new(());
    static INSTALLED: OnceLock<()> = OnceLock::new();

    let exclusive = EXCLUSIVE.lock().unwrap_or_else(|e| e.into_inner());
    INSTALLED.get_or_init(|| {
        // INFO keeps every debug!/trace! site short-circuiting on the static max
        // instead of reaching the sink's mutex.
        let _ = tracing_subscriber::registry()
            .with(tracing_subscriber::filter::LevelFilter::INFO)
            .with(CaptureLayer)
            .try_init();
    });
    let buf = Arc::new(Mutex::new(Vec::new()));
    *sink_slot() = Some(buf.clone());
    Capture {
        _exclusive: exclusive,
        buf,
    }
}

/// Seeds a source row. `selected`: 1=active, 0=deselected.
/// `trashed_at`: `None` = live, `Some(ts)` = trashed.
pub async fn seed_source(
    pool: &sqlx::SqlitePool,
    source_id: &str,
    notebook_id: &str,
    selected: i64,
    trashed_at: Option<&str>,
) {
    let now = chrono::Utc::now().to_rfc3339();
    let sql = if trashed_at.is_some() {
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
         content_hash, enrichment_status, trashed_at, created_at) \
         VALUES (?, ?, 'text', 'seed', 'indexed', '/tmp/s.txt', ?, 'h', NULL, ?, ?)"
    } else {
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
         content_hash, enrichment_status, created_at) \
         VALUES (?, ?, 'text', 'seed', 'indexed', '/tmp/s.txt', ?, 'h', NULL, ?)"
    };
    if let Some(ts) = trashed_at {
        sqlx::query(sql)
            .bind(source_id)
            .bind(notebook_id)
            .bind(selected)
            .bind(ts)
            .bind(&now)
            .execute(pool)
            .await
            .expect("insert source");
    } else {
        sqlx::query(sql)
            .bind(source_id)
            .bind(notebook_id)
            .bind(selected)
            .bind(&now)
            .execute(pool)
            .await
            .expect("insert source");
    }
}

/// Seeds a chunk. `token_start` is nullable; pass `None` for a NULL.
pub async fn seed_chunk(
    pool: &sqlx::SqlitePool,
    chunk_id: &str,
    source_id: &str,
    level: i64,
    token_start: Option<i64>,
    text: &str,
) {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO chunks \
         (id, source_id, parent_id, kind, level, section_path, text, \
          token_start, token_end, char_start, char_end, block_type, created_at) \
         VALUES (?, ?, NULL, 'child', ?, 'Intro', ?, ?, NULL, 0, 100, 'paragraph', ?)",
    )
    .bind(chunk_id)
    .bind(source_id)
    .bind(level)
    .bind(text)
    .bind(token_start)
    .bind(&now)
    .execute(pool)
    .await
    .expect("insert chunk");
}

/// Seeds an entity_node. `definition` and `canonical_name` default to NULL.
pub async fn seed_entity_node(
    pool: &sqlx::SqlitePool,
    node_id: &str,
    notebook_id: &str,
    source_id: &str,
    kind: &str,
    name: &str,
    definition: Option<&str>,
) {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO entity_nodes \
         (id, notebook_id, source_id, kind, name, definition, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(node_id)
    .bind(notebook_id)
    .bind(source_id)
    .bind(kind)
    .bind(name)
    .bind(definition)
    .bind(&now)
    .execute(pool)
    .await
    .expect("insert entity node");
}

/// Sets `canonical_name`/`resolution_conf` on a node (simulates the #155 pass).
pub async fn set_canonical(
    pool: &sqlx::SqlitePool,
    node_id: &str,
    canonical_name: &str,
    resolution_conf: f64,
) {
    sqlx::query(
        "UPDATE entity_nodes SET canonical_name = ?, resolution_conf = ?, \
         resolution_prompt_version = 'res-v1' WHERE id = ?",
    )
    .bind(canonical_name)
    .bind(resolution_conf)
    .bind(node_id)
    .execute(pool)
    .await
    .expect("set canonical");
}

/// Seeds an entity_mention. `char_start` distinguishes multiple mentions in the
/// same (node, chunk) pair (UNIQUE is (entity_node_id, chunk_id, char_start, char_end)).
pub async fn seed_mention(
    pool: &sqlx::SqlitePool,
    mention_id: &str,
    notebook_id: &str,
    node_id: &str,
    chunk_id: &str,
    char_start: i64,
) {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO entity_mentions \
         (id, notebook_id, entity_node_id, chunk_id, char_start, char_end, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(mention_id)
    .bind(notebook_id)
    .bind(node_id)
    .bind(chunk_id)
    .bind(char_start)
    .bind(char_start + 5)
    .bind(&now)
    .execute(pool)
    .await
    .expect("insert mention");
}

/// Seeds an entity_edge. `relation` is a raw DB string (`co_occurs` or a semantic
/// predicate). `weight`/`confidence` are nullable. `from_node`/`to_node` are
/// per-source `entity_nodes.id` values.
#[allow(clippy::too_many_arguments)]
pub async fn seed_edge(
    pool: &sqlx::SqlitePool,
    edge_id: &str,
    notebook_id: &str,
    source_id: &str,
    chunk_id: &str,
    from_node: &str,
    to_node: &str,
    relation: &str,
    weight: Option<f64>,
    confidence: Option<f64>,
) {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO entity_edges \
         (id, notebook_id, source_id, chunk_id, from_node, to_node, relation, \
          weight, confidence, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(edge_id)
    .bind(notebook_id)
    .bind(source_id)
    .bind(chunk_id)
    .bind(from_node)
    .bind(to_node)
    .bind(relation)
    .bind(weight)
    .bind(confidence)
    .bind(&now)
    .execute(pool)
    .await
    .expect("insert edge");
}
