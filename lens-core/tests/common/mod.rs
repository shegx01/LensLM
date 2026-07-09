//! Shared entity-graph seed helpers for integration tests. Kept offline: no
//! model downloads, no LLM — callers hand-build nodes/edges/mentions.

use lens_core::LensEngine;
use tempfile::TempDir;

/// A file-backed engine with the tokenizer disabled (offline, deterministic).
pub async fn file_engine() -> (TempDir, LensEngine) {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = LensEngine::init(dir.path()).await.expect("engine init");
    engine.disable_tokenizer_for_test();
    (dir, engine)
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
