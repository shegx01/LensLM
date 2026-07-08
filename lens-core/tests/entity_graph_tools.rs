//! Integration tests for entity-graph lexical tools (#156a):
//! `entity_lookup` and `entity_evidence`.
//!
//! All tests are offline (no model downloads, no LLM). A seeded temp-DB
//! is built via `file_engine()` from the entity_graph test harness pattern.

use lens_core::LensEngine;
use lens_core::graph::{EntityKind, entity_evidence, entity_lookup};
use sqlx::Row;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

async fn file_engine() -> (TempDir, LensEngine) {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = LensEngine::init(dir.path()).await.expect("engine init");
    engine.disable_tokenizer_for_test();
    (dir, engine)
}

/// Seeds a source row. `selected`: 1=active, 0=deselected.
/// `trashed_at`: `None` = live, `Some(ts)` = trashed.
async fn seed_source(
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
async fn seed_chunk(
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
async fn seed_entity_node(
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

/// Seeds an entity_mention. `char_start` is used to distinguish multiple mentions
/// in the same (node, chunk) pair (the UNIQUE key is (entity_node_id, chunk_id, char_start, char_end)).
async fn seed_mention(
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// 1. exact > prefix > substring ranking
#[tokio::test]
async fn lookup_exact_gt_prefix_gt_substring() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    seed_chunk(&pool, "c1", "s1", 1, Some(0), "text").await;
    seed_chunk(&pool, "c2", "s1", 1, Some(1), "text").await;
    seed_chunk(&pool, "c3", "s1", 1, Some(2), "text").await;

    // "Alice" = exact (tier 0); "Alicesmith" = prefix match on "Alice%" (tier 1); "Malice" = substring (tier 2)
    seed_entity_node(&pool, "n-alice", &nb, "s1", "concept", "Alice", None).await;
    seed_entity_node(
        &pool,
        "n-alicesmith",
        &nb,
        "s1",
        "concept",
        "Alicesmith",
        None,
    )
    .await;
    seed_entity_node(&pool, "n-malice", &nb, "s1", "concept", "Malice", None).await;

    seed_mention(&pool, "m1", &nb, "n-alice", "c1", 0).await;
    seed_mention(&pool, "m2", &nb, "n-alicesmith", "c2", 0).await;
    seed_mention(&pool, "m3", &nb, "n-malice", "c3", 0).await;

    let results = entity_lookup(&pool, &nb, "Alice", 10)
        .await
        .expect("lookup ok");

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].name.to_lowercase(), "alice");
    assert_eq!(results[1].name.to_lowercase(), "alicesmith");
    assert_eq!(results[2].name.to_lowercase(), "malice");
}

/// 2. case-insensitive collapse across sources
#[tokio::test]
async fn lookup_collapse_across_sources() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;

    seed_source(&pool, "sA", &nb, 1, None).await;
    seed_source(&pool, "sB", &nb, 1, None).await;
    seed_chunk(&pool, "cA", "sA", 1, Some(0), "text").await;
    seed_chunk(&pool, "cB", "sB", 1, Some(0), "text").await;

    seed_entity_node(&pool, "nA", &nb, "sA", "concept", "Alice", None).await;
    seed_entity_node(&pool, "nB", &nb, "sB", "concept", "alice", None).await;

    seed_mention(&pool, "mA1", &nb, "nA", "cA", 0).await;
    seed_mention(&pool, "mA2", &nb, "nA", "cA", 10).await;
    seed_mention(&pool, "mB1", &nb, "nB", "cB", 0).await;

    let results = entity_lookup(&pool, &nb, "Alice", 10)
        .await
        .expect("lookup ok");

    assert_eq!(results.len(), 1, "must collapse to one entity");
    assert_eq!(
        results[0].source_count, 2,
        "source_count across both sources"
    );
    assert_eq!(results[0].mention_count, 3, "mention_count = sum of both");
}

/// 3. k truncation
#[tokio::test]
async fn lookup_k_truncation() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    for i in 0..5usize {
        let cid = format!("c{i}");
        let nid = format!("n{i}");
        let name = format!("Abc{i}");
        seed_chunk(&pool, &cid, "s1", 1, Some(i as i64), "text").await;
        seed_entity_node(&pool, &nid, &nb, "s1", "concept", &name, None).await;
        seed_mention(&pool, &format!("m{i}"), &nb, &nid, &cid, 0).await;
    }

    let results = entity_lookup(&pool, &nb, "Abc", 3)
        .await
        .expect("lookup ok");
    assert!(results.len() <= 3, "k=3 must cap results at 3");
}

/// 4. excludes trashed sources
#[tokio::test]
async fn lookup_excludes_trashed_source() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;

    seed_source(&pool, "s-trash", &nb, 1, Some("2026-01-01T00:00:00Z")).await;
    seed_source(&pool, "s-live", &nb, 1, None).await;

    seed_chunk(&pool, "c-trash", "s-trash", 1, Some(0), "text").await;
    seed_chunk(&pool, "c-live", "s-live", 1, Some(0), "text").await;

    seed_entity_node(&pool, "n-trash", &nb, "s-trash", "concept", "Alice", None).await;
    seed_entity_node(&pool, "n-live", &nb, "s-live", "concept", "Alice", None).await;

    seed_mention(&pool, "m-trash", &nb, "n-trash", "c-trash", 0).await;
    seed_mention(&pool, "m-live", &nb, "n-live", "c-live", 0).await;

    let results = entity_lookup(&pool, &nb, "Alice", 10)
        .await
        .expect("lookup ok");

    assert_eq!(results.len(), 1, "trashed source must be excluded");
    assert_eq!(results[0].source_count, 1, "only live source counted");
}

/// 5. excludes deselected sources
#[tokio::test]
async fn lookup_excludes_deselected_source() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;

    seed_source(&pool, "s-off", &nb, 0, None).await;
    seed_chunk(&pool, "c-off", "s-off", 1, Some(0), "text").await;
    seed_entity_node(&pool, "n-off", &nb, "s-off", "concept", "Alice", None).await;
    seed_mention(&pool, "m-off", &nb, "n-off", "c-off", 0).await;

    let results = entity_lookup(&pool, &nb, "Alice", 10)
        .await
        .expect("lookup ok");

    assert!(results.is_empty(), "deselected source must be excluded");
}

/// 6. canonical_name NULL does not break matching
#[tokio::test]
async fn lookup_canonical_name_inert() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    seed_chunk(&pool, "c1", "s1", 1, Some(0), "text").await;
    seed_entity_node(&pool, "n1", &nb, "s1", "concept", "Alice", None).await;
    seed_mention(&pool, "m1", &nb, "n1", "c1", 0).await;

    let results = entity_lookup(&pool, &nb, "Alice", 10)
        .await
        .expect("lookup ok");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].name.to_lowercase(), "alice");
}

/// 7. entity_evidence unions chunk_ids across two sources
#[tokio::test]
async fn evidence_union_across_sources() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;

    seed_source(&pool, "sA", &nb, 1, None).await;
    seed_source(&pool, "sB", &nb, 1, None).await;
    seed_chunk(&pool, "cA", "sA", 1, Some(0), "text").await;
    seed_chunk(&pool, "cB", "sB", 1, Some(0), "text").await;

    seed_entity_node(&pool, "nA", &nb, "sA", "concept", "Alice", None).await;
    seed_entity_node(&pool, "nB", &nb, "sB", "concept", "Alice", None).await;

    seed_mention(&pool, "mA", &nb, "nA", "cA", 0).await;
    seed_mention(&pool, "mB", &nb, "nB", "cB", 0).await;

    let chunk_ids = entity_evidence(&pool, &nb, "Alice", EntityKind::Concept, 10)
        .await
        .expect("evidence ok");

    assert!(chunk_ids.contains(&"cA".to_string()), "cA must be present");
    assert!(chunk_ids.contains(&"cB".to_string()), "cB must be present");
}

/// 8. evidence orders by mention_count DESC
#[tokio::test]
async fn evidence_ordering_mention_count_desc() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    seed_chunk(&pool, "cA", "s1", 1, Some(0), "text").await;
    seed_chunk(&pool, "cB", "s1", 1, Some(10), "text").await;

    seed_entity_node(&pool, "n1", &nb, "s1", "concept", "Alice", None).await;

    // 3 mentions in cA, 1 in cB; vary char_start to satisfy the UNIQUE constraint
    seed_mention(&pool, "m1", &nb, "n1", "cA", 0).await;
    seed_mention(&pool, "m2", &nb, "n1", "cA", 10).await;
    seed_mention(&pool, "m3", &nb, "n1", "cA", 20).await;
    seed_mention(&pool, "m4", &nb, "n1", "cB", 0).await;

    let chunk_ids = entity_evidence(&pool, &nb, "Alice", EntityKind::Concept, 10)
        .await
        .expect("evidence ok");

    assert_eq!(chunk_ids.len(), 2);
    assert_eq!(chunk_ids[0], "cA", "cA (3 mentions) must come first");
    assert_eq!(chunk_ids[1], "cB", "cB (1 mention) must come second");
}

/// 9. evidence orders by doc-order tiebreak (level ASC, token_start ASC)
#[tokio::test]
async fn evidence_ordering_doc_order_tiebreak() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    // cB has lower level than cA → comes first in doc order
    seed_chunk(&pool, "cA", "s1", 2, Some(100), "text").await;
    seed_chunk(&pool, "cB", "s1", 1, Some(10), "text").await;

    seed_entity_node(&pool, "n1", &nb, "s1", "concept", "Alice", None).await;

    // Equal mention count (1 each)
    seed_mention(&pool, "m1", &nb, "n1", "cA", 0).await;
    seed_mention(&pool, "m2", &nb, "n1", "cB", 0).await;

    let chunk_ids = entity_evidence(&pool, &nb, "Alice", EntityKind::Concept, 10)
        .await
        .expect("evidence ok");

    assert_eq!(chunk_ids.len(), 2);
    assert_eq!(chunk_ids[0], "cB", "cB (level 1) before cA (level 2)");
    assert_eq!(chunk_ids[1], "cA");
}

/// 10. evidence k truncation
#[tokio::test]
async fn evidence_k_truncation() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    seed_entity_node(&pool, "n1", &nb, "s1", "concept", "Alice", None).await;

    for i in 0..6usize {
        let cid = format!("c{i}");
        seed_chunk(&pool, &cid, "s1", 1, Some(i as i64), "text").await;
        seed_mention(&pool, &format!("m{i}"), &nb, "n1", &cid, 0).await;
    }

    let chunk_ids = entity_evidence(&pool, &nb, "Alice", EntityKind::Concept, 3)
        .await
        .expect("evidence ok");

    assert!(chunk_ids.len() <= 3, "k=3 must cap at 3");
}

/// 11. evidence excludes trashed and deselected sources
#[tokio::test]
async fn evidence_excludes_trashed_and_deselected() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;

    seed_source(&pool, "s-trash", &nb, 1, Some("2026-01-01T00:00:00Z")).await;
    seed_source(&pool, "s-off", &nb, 0, None).await;

    seed_chunk(&pool, "c-trash", "s-trash", 1, Some(0), "text").await;
    seed_chunk(&pool, "c-off", "s-off", 1, Some(0), "text").await;

    seed_entity_node(&pool, "n-trash", &nb, "s-trash", "concept", "Alice", None).await;
    seed_entity_node(&pool, "n-off", &nb, "s-off", "concept", "Alice", None).await;

    seed_mention(&pool, "m-trash", &nb, "n-trash", "c-trash", 0).await;
    seed_mention(&pool, "m-off", &nb, "n-off", "c-off", 0).await;

    let chunk_ids = entity_evidence(&pool, &nb, "Alice", EntityKind::Concept, 10)
        .await
        .expect("evidence ok");

    assert!(
        chunk_ids.is_empty(),
        "trashed and deselected chunks must be excluded"
    );
}

/// 12. evidence chunk_ids are real PKs that can hydrate text
#[tokio::test]
async fn evidence_chunk_ids_hydrate_compatible() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    seed_chunk(&pool, "c1", "s1", 1, Some(0), "hello world").await;

    seed_entity_node(&pool, "n1", &nb, "s1", "concept", "Alice", None).await;
    seed_mention(&pool, "m1", &nb, "n1", "c1", 0).await;

    let chunk_ids = entity_evidence(&pool, &nb, "Alice", EntityKind::Concept, 10)
        .await
        .expect("evidence ok");

    assert_eq!(chunk_ids, vec!["c1".to_string()]);

    let text: String = sqlx::query("SELECT text FROM chunks WHERE id = ?")
        .bind(&chunk_ids[0])
        .fetch_one(&pool)
        .await
        .expect("fetch chunk")
        .get("text");

    assert_eq!(text, "hello world");
}

/// 13. empty query returns empty vec without scanning
#[tokio::test]
async fn lookup_empty_query_returns_empty() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    seed_chunk(&pool, "c1", "s1", 1, Some(0), "text").await;
    seed_entity_node(&pool, "n1", &nb, "s1", "concept", "Alice", None).await;
    seed_mention(&pool, "m1", &nb, "n1", "c1", 0).await;

    let r1 = entity_lookup(&pool, &nb, "", 10).await.expect("empty ok");
    let r2 = entity_lookup(&pool, &nb, "  ", 10)
        .await
        .expect("whitespace ok");

    assert!(r1.is_empty(), "empty string must return empty");
    assert!(r2.is_empty(), "whitespace-only must return empty");
}

/// 14. LIKE metacharacters in query are escaped (no wildcard over-match)
#[tokio::test]
async fn lookup_wildcards_escaped() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    for (i, name) in ["50 dollars", "50%", "a_b", "axb"].iter().enumerate() {
        let cid = format!("c{i}");
        let nid = format!("n{i}");
        seed_chunk(&pool, &cid, "s1", 1, Some(i as i64), "text").await;
        seed_entity_node(&pool, &nid, &nb, "s1", "concept", name, None).await;
        seed_mention(&pool, &format!("m{i}"), &nb, &nid, &cid, 0).await;
    }

    // Query "50%" must match "50%" literally (exact tier), NOT "50 dollars" via wildcard
    let results = entity_lookup(&pool, &nb, "50%", 10)
        .await
        .expect("lookup ok");
    assert!(
        results.iter().any(|e| e.name == "50%"),
        "50% must be in results"
    );
    assert!(
        !results.iter().any(|e| e.name == "50 dollars"),
        "50 dollars must NOT be matched by 50% query"
    );

    // Query "a_b" must match "a_b" literally, NOT "axb" via _ wildcard
    let results2 = entity_lookup(&pool, &nb, "a_b", 10)
        .await
        .expect("lookup ok");
    assert!(
        results2.iter().any(|e| e.name == "a_b"),
        "a_b must be in results"
    );
    assert!(
        !results2.iter().any(|e| e.name == "axb"),
        "axb must NOT be matched by a_b query"
    );
}

/// 15. token_start NULL sorts LAST (NULLS LAST tiebreak)
#[tokio::test]
async fn evidence_nulls_last_tiebreak() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    // cA: token_start=100; cB: token_start=NULL — same level, equal mention count
    seed_chunk(&pool, "cA", "s1", 1, Some(100), "text").await;
    seed_chunk(&pool, "cB", "s1", 1, None, "text").await;

    seed_entity_node(&pool, "n1", &nb, "s1", "concept", "Alice", None).await;

    seed_mention(&pool, "m1", &nb, "n1", "cA", 0).await;
    seed_mention(&pool, "m2", &nb, "n1", "cB", 0).await;

    let chunk_ids = entity_evidence(&pool, &nb, "Alice", EntityKind::Concept, 10)
        .await
        .expect("evidence ok");

    assert_eq!(chunk_ids.len(), 2);
    assert_eq!(chunk_ids[0], "cA", "token_start=100 must come before NULL");
    assert_eq!(chunk_ids[1], "cB", "NULL token_start sorts last");
}

/// 16. both tools are notebook-scoped: notebook A never returns notebook B's rows
#[tokio::test]
async fn tools_scope_to_notebook() {
    let (_dir, engine) = file_engine().await;
    let nb_a = engine
        .create_notebook("a", None, None)
        .await
        .expect("create nb a")
        .id
        .to_string();
    let nb_b = engine
        .create_notebook("b", None, None)
        .await
        .expect("create nb b")
        .id
        .to_string();
    let pool = engine.pool().await;

    // Same entity name "Alice" seeded independently in each notebook.
    seed_source(&pool, "sa", &nb_a, 1, None).await;
    seed_chunk(&pool, "ca", "sa", 1, Some(0), "text").await;
    seed_entity_node(&pool, "na", &nb_a, "sa", "concept", "Alice", None).await;
    seed_mention(&pool, "ma", &nb_a, "na", "ca", 0).await;

    seed_source(&pool, "sb", &nb_b, 1, None).await;
    seed_chunk(&pool, "cb", "sb", 1, Some(0), "text").await;
    seed_entity_node(&pool, "nb1", &nb_b, "sb", "concept", "Alice", None).await;
    seed_mention(&pool, "mb", &nb_b, "nb1", "cb", 0).await;

    let hits = entity_lookup(&pool, &nb_a, "Alice", 10)
        .await
        .expect("lookup ok");
    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].source_count, 1,
        "must not count notebook B's source"
    );

    let evidence = entity_evidence(&pool, &nb_a, "Alice", EntityKind::Concept, 10)
        .await
        .expect("evidence ok");
    assert_eq!(evidence, vec!["ca"], "must not return notebook B's chunk");
}

/// 17. entity_evidence filters by kind: a name match with a different kind returns nothing
#[tokio::test]
async fn evidence_respects_kind() {
    let (_dir, engine) = file_engine().await;
    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    seed_chunk(&pool, "c1", "s1", 1, Some(0), "text").await;
    seed_entity_node(&pool, "n1", &nb, "s1", "concept", "Alice", None).await;
    seed_mention(&pool, "m1", &nb, "n1", "c1", 0).await;

    let wrong_kind = entity_evidence(&pool, &nb, "Alice", EntityKind::Person, 10)
        .await
        .expect("evidence ok");
    assert!(
        wrong_kind.is_empty(),
        "kind mismatch must return no evidence"
    );

    let right_kind = entity_evidence(&pool, &nb, "Alice", EntityKind::Concept, 10)
        .await
        .expect("evidence ok");
    assert_eq!(right_kind, vec!["c1"]);
}
