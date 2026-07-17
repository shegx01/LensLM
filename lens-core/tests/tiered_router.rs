//! Integration tests for the Tiered Context Router (issue #21). Offline —
//! hand-built vectors, no model downloads, reranker left disabled. Covers tier
//! selection (AC1), Tier-2 parent auto-merge + doc order (AC4), Tier-1 raw-corpus
//! assembly (AC8), the #39 dense pre-filter (AC7), graph fusion (AC5), and the
//! graph-OFF fusion-seam parity + input-divergence pair (AC6).

use lens_core::config::{ModelConfig, RetrievalConfig, TierThresholds};
use lens_core::embedder::EmbeddingBackend;
use lens_core::graph::NotebookGraph;
use lens_core::retrieval::router::tiered_search;
use lens_core::retrieval::{HitSource, Reranker, hybrid_search};
use lens_core::vector_store::{Coordinate, LanceVectorStore, VectorRow, VectorStore};
use lens_core::{LensEngine, Tier};
use sqlx::SqlitePool;

const DIM: usize = 4;

fn model_ctx(context: u32) -> ModelConfig {
    ModelConfig {
        context,
        ..ModelConfig::default()
    }
}

fn axis_vec(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[axis] = 1.0;
    v
}

async fn insert_source(pool: &SqlitePool, notebook_id: &str, source_id: &str, token_count: i64) {
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
         token_count, content_hash, created_at) \
         VALUES (?, ?, 'text', 'seed', 'indexed', '/tmp/seed.txt', 1, ?, ?, ?)",
    )
    .bind(source_id)
    .bind(notebook_id)
    .bind(token_count)
    .bind(format!("hash-{source_id}"))
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(pool)
    .await
    .expect("insert source");
}

#[allow(clippy::too_many_arguments)]
async fn insert_chunk(
    pool: &SqlitePool,
    source_id: &str,
    chunk_id: &str,
    parent_id: Option<&str>,
    kind: &str,
    level: i32,
    token_start: i64,
    text: &str,
) {
    sqlx::query(
        "INSERT INTO chunks \
         (id, source_id, parent_id, kind, level, section_path, text, \
          token_start, token_end, char_start, char_end, block_type, source_anchor, created_at) \
         VALUES (?, ?, ?, ?, ?, 'Intro', ?, ?, ?, 0, ?, 'paragraph', ?, ?)",
    )
    .bind(chunk_id)
    .bind(source_id)
    .bind(parent_id)
    .bind(kind)
    .bind(level)
    .bind(text)
    .bind(token_start)
    .bind(token_start + 1)
    .bind(text.len() as i64)
    .bind(format!("anchor-{chunk_id}"))
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(pool)
    .await
    .expect("insert chunk");
}

fn store_and_coord(
    engine: &LensEngine,
    data_dir: &std::path::Path,
    pool: SqlitePool,
    nb: &str,
) -> (LanceVectorStore, Coordinate) {
    let _ = engine;
    let store = LanceVectorStore::new(data_dir, pool);
    let coord = Coordinate::new(nb.to_string(), EmbeddingBackend::Fastembed, "m", DIM);
    (store, coord)
}

// ---------------------------------------------------------------------------
// AC1 — tier boundary
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tier1_when_corpus_fits_cap() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    // ctx 10000 -> tier1_cap ≈ 4836. token_count 100 fits.
    insert_source(&pool, &nb, "s1", 100).await;
    insert_chunk(&pool, "s1", "p1", None, "parent", 0, 0, "parent text one").await;

    let (store, coord) = store_and_coord(&engine, dir.path(), pool.clone(), &nb);
    let reranker = Reranker::new(dir.path());
    let out = tiered_search(
        &pool,
        &store,
        &reranker,
        None,
        &coord,
        "q",
        &axis_vec(0),
        &model_ctx(10_000),
        10,
        &RetrievalConfig::default(),
        None,
        &TierThresholds::default(),
        None,
        0,
    )
    .await
    .unwrap();
    assert_eq!(out.tier, Tier::Tier1);
    assert!(!out.units.is_empty());
}

#[tokio::test]
async fn tier2_when_corpus_overflows_cap() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    // Two sources each 3000 -> sum 6000 > tier1_cap (~4836) at ctx 10000.
    insert_source(&pool, &nb, "s1", 3_000).await;
    insert_source(&pool, &nb, "s2", 3_000).await;
    insert_chunk(&pool, "s1", "p1", None, "parent", 0, 0, "alpha").await;
    insert_chunk(&pool, "s2", "p2", None, "parent", 0, 0, "beta").await;

    let (store, coord) = store_and_coord(&engine, dir.path(), pool.clone(), &nb);
    let reranker = Reranker::new(dir.path());
    let out = tiered_search(
        &pool,
        &store,
        &reranker,
        None,
        &coord,
        "q",
        &axis_vec(0),
        &model_ctx(10_000),
        10,
        &RetrievalConfig::default(),
        None,
        &TierThresholds::default(),
        None,
        0,
    )
    .await
    .unwrap();
    assert_eq!(out.tier, Tier::Tier2);
}

#[tokio::test]
async fn single_oversized_source_forces_tier2() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    // One source alone whose count exceeds the cap.
    insert_source(&pool, &nb, "s1", 9_000).await;
    insert_chunk(&pool, "s1", "p1", None, "parent", 0, 0, "big").await;

    let (store, coord) = store_and_coord(&engine, dir.path(), pool.clone(), &nb);
    let reranker = Reranker::new(dir.path());
    let out = tiered_search(
        &pool,
        &store,
        &reranker,
        None,
        &coord,
        "q",
        &axis_vec(0),
        &model_ctx(10_000),
        10,
        &RetrievalConfig::default(),
        None,
        &TierThresholds::default(),
        None,
        0,
    )
    .await
    .unwrap();
    assert_eq!(out.tier, Tier::Tier2);
}

// ---------------------------------------------------------------------------
// AC8 — Tier-1 raw-corpus assembly
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tier1_units_are_parent_chunks_in_doc_order() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_source(&pool, &nb, "s1", 50).await;
    // Insert out of doc order; assembly must sort by level then token_start.
    insert_chunk(&pool, "s1", "p2", None, "parent", 0, 20, "second parent").await;
    insert_chunk(&pool, "s1", "p1", None, "parent", 0, 0, "first parent").await;
    // A child must NOT appear in Tier-1 (parents only).
    insert_chunk(&pool, "s1", "c1", Some("p1"), "child", 1, 0, "child body").await;

    let (store, coord) = store_and_coord(&engine, dir.path(), pool.clone(), &nb);
    let reranker = Reranker::new(dir.path());
    let out = tiered_search(
        &pool,
        &store,
        &reranker,
        None,
        &coord,
        "q",
        &axis_vec(0),
        &model_ctx(10_000),
        10,
        &RetrievalConfig::default(),
        None,
        &TierThresholds::default(),
        None,
        0,
    )
    .await
    .unwrap();
    assert_eq!(out.tier, Tier::Tier1);
    let ids: Vec<&str> = out.units.iter().map(|u| u.chunk_id.as_str()).collect();
    assert_eq!(ids, vec!["p1", "p2"], "parents in doc order, no child");
    for (i, u) in out.units.iter().enumerate() {
        assert_eq!(u.order_index, i, "order_index monotonic");
        assert_eq!(u.parent_id, None, "Tier-1 parents carry parent_id=None");
        assert!(!u.text.is_empty(), "hydrated");
    }
}

// ---------------------------------------------------------------------------
// AC4 — Tier-2 parent auto-merge (>=50%) + doc order
// ---------------------------------------------------------------------------

/// Builds a Tier-2 notebook: one parent p1 with 4 children; a vector added per
/// child so a dense query can surface a controlled subset. Returns the coord.
async fn seed_tier2_notebook(
    pool: &SqlitePool,
    store: &LanceVectorStore,
    coord: &Coordinate,
    nb: &str,
) {
    // Oversized so the router always picks Tier-2.
    insert_source(pool, nb, "s1", 9_000).await;
    insert_chunk(
        pool,
        "s1",
        "p1",
        None,
        "parent",
        0,
        0,
        "PARENT ONE full text",
    )
    .await;
    for (i, cid) in ["c1", "c2", "c3", "c4"].iter().enumerate() {
        insert_chunk(
            pool,
            "s1",
            cid,
            Some("p1"),
            "child",
            1,
            i as i64,
            &format!("child {cid}"),
        )
        .await;
    }
    let mut rows = Vec::new();
    for (i, cid) in ["c1", "c2", "c3", "c4"].iter().enumerate() {
        rows.push(VectorRow {
            chunk_id: cid.to_string(),
            source_id: "s1".to_string(),
            notebook_id: nb.to_string(),
            level: 1,
            vector: axis_vec(i),
        });
    }
    store.add(coord, rows).await.unwrap();
}

#[tokio::test]
async fn tier2_merges_parent_when_half_children_retrieved() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    let (store, coord) = store_and_coord(&engine, dir.path(), pool.clone(), &nb);
    seed_tier2_notebook(&pool, &store, &coord, &nb).await;

    // pool=2 dense hits (axis-0 and axis-1 nearest) => c1,c2 retrieved (2/4 = 50%)
    // => parent p1 auto-merges.
    let reranker = Reranker::new(dir.path());
    let cfg = RetrievalConfig {
        hybrid_enabled: false,
        ..RetrievalConfig::default()
    };
    let out = tiered_search(
        &pool,
        &store,
        &reranker,
        None,
        &coord,
        "irrelevant",
        &axis_vec(0),
        &model_ctx(10_000),
        2,
        &cfg,
        None,
        &TierThresholds::default(),
        None,
        0,
    )
    .await
    .unwrap();
    assert_eq!(out.tier, Tier::Tier2);
    let ids: Vec<&str> = out.units.iter().map(|u| u.chunk_id.as_str()).collect();
    assert!(
        ids.contains(&"p1"),
        "2/4 children retrieved must auto-merge the parent, got {ids:?}"
    );
    assert!(
        !ids.contains(&"c1") && !ids.contains(&"c2"),
        "merged parent replaces its children, got {ids:?}"
    );
}

#[tokio::test]
async fn tier2_no_merge_when_below_half() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    let (store, coord) = store_and_coord(&engine, dir.path(), pool.clone(), &nb);
    seed_tier2_notebook(&pool, &store, &coord, &nb).await;

    // pool=1 => only c1 (1/4 = 25% < 50%) => no merge; c1 stays.
    let reranker = Reranker::new(dir.path());
    let cfg = RetrievalConfig {
        hybrid_enabled: false,
        ..RetrievalConfig::default()
    };
    let out = tiered_search(
        &pool,
        &store,
        &reranker,
        None,
        &coord,
        "irrelevant",
        &axis_vec(0),
        &model_ctx(10_000),
        1,
        &cfg,
        None,
        &TierThresholds::default(),
        None,
        0,
    )
    .await
    .unwrap();
    let ids: Vec<&str> = out.units.iter().map(|u| u.chunk_id.as_str()).collect();
    assert!(
        ids.contains(&"c1"),
        "1/4 must NOT merge; child stays, got {ids:?}"
    );
    assert!(
        !ids.contains(&"p1"),
        "no parent merge below 50%, got {ids:?}"
    );
}

// ---------------------------------------------------------------------------
// AC7 — dense pre-filter + empty-selected-set semantics
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dense_prefilter_excludes_deselected_source_nearer_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    // A(selected) has a chunk on axis-1; B(deselected) has a chunk on axis-0 (the
    // query axis) so B is strictly nearer. The pre-filter must exclude B.
    insert_source(&pool, &nb, "sA", 9_000).await;
    insert_source(&pool, &nb, "sB", 9_000).await;
    insert_chunk(&pool, "sA", "cA", None, "parent", 0, 0, "alpha chunk").await;
    insert_chunk(&pool, "sB", "cB", None, "parent", 0, 0, "beta chunk").await;
    engine.set_source_selected("sB", false).await.unwrap();

    let (store, coord) = store_and_coord(&engine, dir.path(), pool.clone(), &nb);
    store
        .add(
            &coord,
            vec![
                VectorRow {
                    chunk_id: "cA".into(),
                    source_id: "sA".into(),
                    notebook_id: nb.to_string(),
                    level: 0,
                    vector: axis_vec(1),
                },
                VectorRow {
                    chunk_id: "cB".into(),
                    source_id: "sB".into(),
                    notebook_id: nb.to_string(),
                    level: 0,
                    vector: axis_vec(0),
                },
            ],
        )
        .await
        .unwrap();

    let reranker = Reranker::new(dir.path());
    let cfg = RetrievalConfig {
        hybrid_enabled: false,
        ..RetrievalConfig::default()
    };
    let out = tiered_search(
        &pool,
        &store,
        &reranker,
        None,
        &coord,
        "q",
        &axis_vec(0),
        &model_ctx(10_000),
        10,
        &cfg,
        None,
        &TierThresholds::default(),
        None,
        0,
    )
    .await
    .unwrap();
    let ids: Vec<&str> = out.units.iter().map(|u| u.chunk_id.as_str()).collect();
    assert!(
        !ids.contains(&"cB"),
        "deselected source's nearer chunk must be excluded, got {ids:?}"
    );
    assert!(ids.contains(&"cA"), "selected source retained, got {ids:?}");
}

#[tokio::test]
async fn empty_selected_set_returns_empty_not_notebook_scope() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_source(&pool, &nb, "s1", 9_000).await;
    insert_chunk(&pool, "s1", "c1", None, "parent", 0, 0, "content").await;
    engine.set_source_selected("s1", false).await.unwrap();

    let (store, coord) = store_and_coord(&engine, dir.path(), pool.clone(), &nb);
    store
        .add(
            &coord,
            vec![VectorRow {
                chunk_id: "c1".into(),
                source_id: "s1".into(),
                notebook_id: nb.to_string(),
                level: 0,
                vector: axis_vec(0),
            }],
        )
        .await
        .unwrap();

    let reranker = Reranker::new(dir.path());
    let out = tiered_search(
        &pool,
        &store,
        &reranker,
        None,
        &coord,
        "q",
        &axis_vec(0),
        &model_ctx(10_000),
        10,
        &RetrievalConfig::default(),
        None,
        &TierThresholds::default(),
        None,
        0,
    )
    .await
    .unwrap();
    assert!(
        out.units.is_empty(),
        "nothing selected must ground on nothing, got {:?}",
        out.units
    );
}

// ---------------------------------------------------------------------------
// AC6 — graph-OFF fusion-seam parity + input divergence
// ---------------------------------------------------------------------------

/// Seam parity: with graph OFF, tiered_search's Tier-2 fused chunk ids equal
/// hybrid_search's ids on the SAME selected corpus (both use fuse_and_rerank).
/// Reranker OFF and ON.
async fn assert_seam_parity(reranker_enabled: bool) {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_source(&pool, &nb, "s1", 9_000).await;
    // Children so both paths reach the same fused set; keep <50% per parent so no
    // merge changes the id set (one parent, one child each so parent has 1 child;
    // retrieving that child is 1/1 = merge — avoid by using level-1 orphans).
    insert_chunk(
        &pool,
        "s1",
        "c1",
        None,
        "child",
        1,
        0,
        "quokka rare lexical alpha",
    )
    .await;
    insert_chunk(
        &pool,
        "s1",
        "c2",
        None,
        "child",
        1,
        1,
        "wombat rare lexical beta",
    )
    .await;

    let (store, coord) = store_and_coord(&engine, dir.path(), pool.clone(), &nb);
    store
        .add(
            &coord,
            vec![
                VectorRow {
                    chunk_id: "c1".into(),
                    source_id: "s1".into(),
                    notebook_id: nb.to_string(),
                    level: 1,
                    vector: axis_vec(0),
                },
                VectorRow {
                    chunk_id: "c2".into(),
                    source_id: "s1".into(),
                    notebook_id: nb.to_string(),
                    level: 1,
                    vector: axis_vec(1),
                },
            ],
        )
        .await
        .unwrap();

    let reranker = Reranker::new(dir.path());
    let mut cfg = RetrievalConfig::default();
    cfg.reranker.enabled = reranker_enabled;
    let qvec = axis_vec(0);

    let hybrid = hybrid_search(
        &pool, &store, &reranker, &coord, "quokka", &qvec, None, None, 10, &cfg,
    )
    .await
    .unwrap();
    let hybrid_ids: Vec<&str> = hybrid.iter().map(|h| h.chunk_id.as_str()).collect();

    let out = tiered_search(
        &pool,
        &store,
        &reranker,
        None,
        &coord,
        "quokka",
        &qvec,
        &model_ctx(10_000),
        10,
        &cfg,
        None,
        &TierThresholds::default(),
        None,
        0,
    )
    .await
    .unwrap();
    // The fusion seam (rrf_merge3 + reranker) is shared by construction, so the SET
    // of surfaced chunks is identical with graph off; the router additionally re-sorts
    // survivors to document order, so compare membership (sorted), not rank order.
    let mut router_ids: Vec<&str> = out.units.iter().map(|u| u.chunk_id.as_str()).collect();
    let mut hybrid_sorted = hybrid_ids.clone();
    router_ids.sort_unstable();
    hybrid_sorted.sort_unstable();
    assert_eq!(
        router_ids, hybrid_sorted,
        "seam parity (reranker={reranker_enabled}): router Tier-2 chunk set must equal hybrid_search set"
    );
}

#[tokio::test]
async fn seam_parity_reranker_off() {
    assert_seam_parity(false).await;
}

#[tokio::test]
async fn seam_parity_reranker_on() {
    // Reranker ENABLED. To keep this in the offline suite (no model download), point
    // fastembed at an unreachable endpoint so init fails and rerank_with_fallback
    // returns RRF order — the same fallback path CI hits with no cached model. Parity
    // still holds because both callers share fuse_and_rerank; the enabled flag
    // exercises the reranker branch without the network. Mirrors rerank.rs's pattern;
    // nextest isolates each test in its own process, so the env var is safe.
    // SAFETY: single-threaded test process setting an env var before the init call.
    unsafe {
        std::env::set_var("HF_ENDPOINT", "http://127.0.0.1:1");
    }
    assert_seam_parity(true).await;
    unsafe {
        std::env::remove_var("HF_ENDPOINT");
    }
}

#[tokio::test]
async fn prefilter_diverges_from_hybrid_on_deselected_source() {
    // Input-divergence half of AC6: with a deselected source whose chunk is nearest,
    // tiered_search (pre-filter before top-N) excludes it entirely; hybrid_search
    // fetches it densely then post-filters. The final router output must not contain
    // the deselected chunk — the documented, deliberate difference.
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_source(&pool, &nb, "sA", 9_000).await;
    insert_source(&pool, &nb, "sB", 9_000).await;
    insert_chunk(&pool, "sA", "cA", None, "child", 1, 0, "alpha").await;
    insert_chunk(&pool, "sB", "cB", None, "child", 1, 0, "beta").await;
    engine.set_source_selected("sB", false).await.unwrap();

    let (store, coord) = store_and_coord(&engine, dir.path(), pool.clone(), &nb);
    store
        .add(
            &coord,
            vec![
                VectorRow {
                    chunk_id: "cA".into(),
                    source_id: "sA".into(),
                    notebook_id: nb.to_string(),
                    level: 1,
                    vector: axis_vec(1),
                },
                VectorRow {
                    chunk_id: "cB".into(),
                    source_id: "sB".into(),
                    notebook_id: nb.to_string(),
                    level: 1,
                    vector: axis_vec(0),
                },
            ],
        )
        .await
        .unwrap();

    let reranker = Reranker::new(dir.path());
    let cfg = RetrievalConfig {
        hybrid_enabled: false,
        ..RetrievalConfig::default()
    };
    let out = tiered_search(
        &pool,
        &store,
        &reranker,
        None,
        &coord,
        "q",
        &axis_vec(0),
        &model_ctx(10_000),
        10,
        &cfg,
        None,
        &TierThresholds::default(),
        None,
        0,
    )
    .await
    .unwrap();
    let ids: Vec<&str> = out.units.iter().map(|u| u.chunk_id.as_str()).collect();
    assert!(!ids.contains(&"cB"), "router excludes deselected chunk");
}

// ---------------------------------------------------------------------------
// AC5 — graph composition folds in as a third list
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn seed_entity_node(
    pool: &SqlitePool,
    node_id: &str,
    nb: &str,
    source_id: &str,
    kind: &str,
    name: &str,
) {
    sqlx::query(
        "INSERT INTO entity_nodes (id, notebook_id, source_id, kind, name, created_at) \
         VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(node_id)
    .bind(nb)
    .bind(source_id)
    .bind(kind)
    .bind(name)
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(pool)
    .await
    .expect("insert node");
}

async fn seed_mention(pool: &SqlitePool, id: &str, nb: &str, node_id: &str, chunk_id: &str) {
    sqlx::query(
        "INSERT INTO entity_mentions \
         (id, notebook_id, entity_node_id, chunk_id, char_start, char_end, created_at) \
         VALUES (?, ?, ?, ?, 0, 5, ?)",
    )
    .bind(id)
    .bind(nb)
    .bind(node_id)
    .bind(chunk_id)
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(pool)
    .await
    .expect("insert mention");
}

#[tokio::test]
async fn graph_arm_folds_graph_only_chunk_tagged_graph() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_source(&pool, &nb, "s1", 9_000).await;
    // cg is reachable ONLY via the graph arm (a mention of "Widget"); it has no
    // vector and no lexical match to the query, so dense/bm25 miss it.
    insert_chunk(&pool, "s1", "cd", None, "child", 1, 0, "dense body").await;
    insert_chunk(&pool, "s1", "cg", None, "child", 1, 1, "unrelated body").await;
    seed_entity_node(&pool, "n1", &nb, "s1", "concept", "Widget").await;
    seed_mention(&pool, "m1", &nb, "n1", "cg").await;

    let (store, coord) = store_and_coord(&engine, dir.path(), pool.clone(), &nb);
    store
        .add(
            &coord,
            vec![VectorRow {
                chunk_id: "cd".into(),
                source_id: "s1".into(),
                notebook_id: nb.to_string(),
                level: 1,
                vector: axis_vec(0),
            }],
        )
        .await
        .unwrap();

    let graph = NotebookGraph::load(&pool, &nb).await.unwrap();
    let reranker = Reranker::new(dir.path());
    let cfg = RetrievalConfig {
        hybrid_enabled: false,
        graph_retrieval_enabled: true,
        ..RetrievalConfig::default()
    };
    // Query carries a relational signal beyond the seed name "Widget".
    let out = tiered_search(
        &pool,
        &store,
        &reranker,
        Some(&graph),
        &coord,
        "Widget acquisition timeline details",
        &axis_vec(0),
        &model_ctx(10_000),
        10,
        &cfg,
        Some(true),
        &TierThresholds::default(),
        None,
        0,
    )
    .await
    .unwrap();
    let cg = out.units.iter().find(|u| u.chunk_id == "cg");
    assert!(cg.is_some(), "graph-only chunk folded into the fused set");
    let cg = cg.unwrap();
    assert_eq!(cg.provenance.source, HitSource::Graph);
    assert!(
        cg.provenance.graph_confidence.is_some(),
        "graph_confidence carried on provenance"
    );
}

#[tokio::test]
async fn graph_off_when_flag_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LensEngine::init(dir.path()).await.unwrap();
    let pool = engine.pool().await;
    let nb = engine.create_notebook("nb", None, None).await.unwrap().id;
    insert_source(&pool, &nb, "s1", 9_000).await;
    insert_chunk(&pool, "s1", "cd", None, "child", 1, 0, "dense body").await;
    insert_chunk(&pool, "s1", "cg", None, "child", 1, 1, "unrelated body").await;
    seed_entity_node(&pool, "n1", &nb, "s1", "concept", "Widget").await;
    seed_mention(&pool, "m1", &nb, "n1", "cg").await;

    let (store, coord) = store_and_coord(&engine, dir.path(), pool.clone(), &nb);
    store
        .add(
            &coord,
            vec![VectorRow {
                chunk_id: "cd".into(),
                source_id: "s1".into(),
                notebook_id: nb.to_string(),
                level: 1,
                vector: axis_vec(0),
            }],
        )
        .await
        .unwrap();

    let graph = NotebookGraph::load(&pool, &nb).await.unwrap();
    let reranker = Reranker::new(dir.path());
    let cfg = RetrievalConfig {
        hybrid_enabled: false,
        graph_retrieval_enabled: false,
        ..RetrievalConfig::default()
    };
    // Per-notebook override OFF beats app-wide even if it were on.
    let out = tiered_search(
        &pool,
        &store,
        &reranker,
        Some(&graph),
        &coord,
        "Widget acquisition timeline details",
        &axis_vec(0),
        &model_ctx(10_000),
        10,
        &cfg,
        Some(false),
        &TierThresholds::default(),
        None,
        0,
    )
    .await
    .unwrap();
    let ids: Vec<&str> = out.units.iter().map(|u| u.chunk_id.as_str()).collect();
    assert!(
        !ids.contains(&"cg"),
        "graph disabled must not surface the graph-only chunk, got {ids:?}"
    );
}
