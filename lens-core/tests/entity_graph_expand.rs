//! Integration tests for entity-graph gated tools (#156b):
//! `expand_neighbors` (SQL recursive-CTE traversal) and `ppr_expand`
//! (in-memory power-iteration personalized PageRank over a `NotebookGraph`).
//!
//! All tests are offline (no model downloads, no LLM). Graphs are hand-seeded
//! via the shared helpers in `common`.

mod common;

use common::{
    file_engine, seed_chunk, seed_edge, seed_entity_node, seed_mention, seed_source, set_canonical,
};
use lens_core::graph::{EntityKind, NotebookGraph, expand_neighbors, ppr_expand};

async fn new_notebook(engine: &lens_core::LensEngine) -> String {
    engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string()
}

fn seed(name: &str, kind: EntityKind) -> Vec<(String, EntityKind)> {
    vec![(name.to_string(), kind)]
}

// ---------------------------------------------------------------------------
// expand_neighbors
// ---------------------------------------------------------------------------

/// depth-1: A—B co_occurs → querying A returns B once.
#[tokio::test]
async fn expand_depth1_returns_direct_neighbor() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    seed_chunk(&pool, "cA", "s1", 1, Some(0), "t").await;
    seed_chunk(&pool, "cB", "s1", 1, Some(1), "t").await;
    seed_entity_node(&pool, "nA", &nb, "s1", "concept", "A", None).await;
    seed_entity_node(&pool, "nB", &nb, "s1", "concept", "B", None).await;
    seed_mention(&pool, "mA", &nb, "nA", "cA", 0).await;
    seed_mention(&pool, "mB", &nb, "nB", "cB", 0).await;
    seed_edge(
        &pool,
        "e1",
        &nb,
        "s1",
        "cA",
        "nA",
        "nB",
        "co_occurs",
        Some(2.0),
        None,
    )
    .await;

    let hits = expand_neighbors(&pool, &nb, &seed("A", EntityKind::Concept), 1)
        .await
        .expect("expand ok");

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].name, "B");
    assert_eq!(hits[0].chunk_ids, vec!["cB".to_string()]);
    assert!(
        (hits[0].graph_confidence - 1.0).abs() < 1e-6,
        "top hit normalizes to 1.0"
    );
    assert_eq!(hits[0].relation.as_deref(), Some("co_occurs"));
}

/// depth-2: A—B—C. Depth 1 returns only B; depth 2 returns B and C.
#[tokio::test]
async fn expand_depth2_reaches_two_hops() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    for (c, ts) in [("cA", 0), ("cB", 1), ("cC", 2)] {
        seed_chunk(&pool, c, "s1", 1, Some(ts), "t").await;
    }
    for (nid, name) in [("nA", "A"), ("nB", "B"), ("nC", "C")] {
        seed_entity_node(&pool, nid, &nb, "s1", "concept", name, None).await;
    }
    seed_mention(&pool, "mA", &nb, "nA", "cA", 0).await;
    seed_mention(&pool, "mB", &nb, "nB", "cB", 0).await;
    seed_mention(&pool, "mC", &nb, "nC", "cC", 0).await;
    seed_edge(
        &pool,
        "e1",
        &nb,
        "s1",
        "cA",
        "nA",
        "nB",
        "co_occurs",
        Some(2.0),
        None,
    )
    .await;
    seed_edge(
        &pool,
        "e2",
        &nb,
        "s1",
        "cB",
        "nB",
        "nC",
        "co_occurs",
        Some(2.0),
        None,
    )
    .await;

    let d1 = expand_neighbors(&pool, &nb, &seed("A", EntityKind::Concept), 1)
        .await
        .expect("expand ok");
    assert_eq!(d1.len(), 1, "depth 1 reaches only B");
    assert_eq!(d1[0].name, "B");

    let d2 = expand_neighbors(&pool, &nb, &seed("A", EntityKind::Concept), 2)
        .await
        .expect("expand ok");
    let names: Vec<&str> = d2.iter().map(|h| h.name.as_str()).collect();
    assert!(
        names.contains(&"B") && names.contains(&"C"),
        "depth 2 reaches B and C: {names:?}"
    );
}

/// depth clamps to 2: requesting depth 5 must not reach a 3rd hop.
#[tokio::test]
async fn expand_depth_clamped_to_two() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    // A—B—C—D chain.
    for (c, ts) in [("cA", 0), ("cB", 1), ("cC", 2), ("cD", 3)] {
        seed_chunk(&pool, c, "s1", 1, Some(ts), "t").await;
    }
    for (nid, name) in [("nA", "A"), ("nB", "B"), ("nC", "C"), ("nD", "D")] {
        seed_entity_node(&pool, nid, &nb, "s1", "concept", name, None).await;
    }
    for (m, nid, c) in [
        ("mA", "nA", "cA"),
        ("mB", "nB", "cB"),
        ("mC", "nC", "cC"),
        ("mD", "nD", "cD"),
    ] {
        seed_mention(&pool, m, &nb, nid, c, 0).await;
    }
    seed_edge(
        &pool,
        "e1",
        &nb,
        "s1",
        "cA",
        "nA",
        "nB",
        "co_occurs",
        Some(2.0),
        None,
    )
    .await;
    seed_edge(
        &pool,
        "e2",
        &nb,
        "s1",
        "cB",
        "nB",
        "nC",
        "co_occurs",
        Some(2.0),
        None,
    )
    .await;
    seed_edge(
        &pool,
        "e3",
        &nb,
        "s1",
        "cC",
        "nC",
        "nD",
        "co_occurs",
        Some(2.0),
        None,
    )
    .await;

    let hits = expand_neighbors(&pool, &nb, &seed("A", EntityKind::Concept), 5)
        .await
        .expect("expand ok");
    let names: Vec<&str> = hits.iter().map(|h| h.name.as_str()).collect();
    assert!(names.contains(&"C"), "depth-2 C reachable: {names:?}");
    assert!(
        !names.contains(&"D"),
        "depth clamps to 2, D unreachable: {names:?}"
    );
}

/// UNION-ALL bidirectional correctness: a pair stored as BOTH co_occurs (A,B) and
/// semantic (A,B) yields neighbor B exactly once with the MAX blended weight.
#[tokio::test]
async fn expand_union_all_dedupes_with_max_weight() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    seed_chunk(&pool, "cA", "s1", 1, Some(0), "t").await;
    seed_chunk(&pool, "cB", "s1", 1, Some(1), "t").await;
    seed_entity_node(&pool, "nA", &nb, "s1", "concept", "A", None).await;
    seed_entity_node(&pool, "nB", &nb, "s1", "concept", "B", None).await;
    seed_mention(&pool, "mA", &nb, "nA", "cA", 0).await;
    seed_mention(&pool, "mB", &nb, "nB", "cB", 0).await;
    // co_occurs weight 1 → ln_1p(1)=0.693; semantic confidence 1.0 → 3.0 (bigger).
    seed_edge(
        &pool,
        "e1",
        &nb,
        "s1",
        "cA",
        "nA",
        "nB",
        "co_occurs",
        Some(1.0),
        None,
    )
    .await;
    seed_edge(
        &pool,
        "e2",
        &nb,
        "s1",
        "cA",
        "nA",
        "nB",
        "founded",
        None,
        Some(1.0),
    )
    .await;

    let hits = expand_neighbors(&pool, &nb, &seed("A", EntityKind::Concept), 1)
        .await
        .expect("expand ok");

    assert_eq!(hits.len(), 1, "B appears once, not once per edge class");
    assert_eq!(hits[0].name, "B");
    assert_eq!(
        hits[0].relation.as_deref(),
        Some("founded"),
        "max-weight edge is semantic"
    );
}

/// Cycle + self-loop safety: A↔B cycle and a self-loop on A must terminate and
/// not double-count. Querying A returns B (self-loop ignored).
#[tokio::test]
async fn expand_cycle_and_self_loop_safe() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    seed_chunk(&pool, "cA", "s1", 1, Some(0), "t").await;
    seed_chunk(&pool, "cB", "s1", 1, Some(1), "t").await;
    seed_entity_node(&pool, "nA", &nb, "s1", "concept", "A", None).await;
    seed_entity_node(&pool, "nB", &nb, "s1", "concept", "B", None).await;
    seed_mention(&pool, "mA", &nb, "nA", "cA", 0).await;
    seed_mention(&pool, "mB", &nb, "nB", "cB", 0).await;
    seed_edge(
        &pool,
        "e1",
        &nb,
        "s1",
        "cA",
        "nA",
        "nB",
        "co_occurs",
        Some(2.0),
        None,
    )
    .await;
    seed_edge(
        &pool,
        "e2",
        &nb,
        "s1",
        "cB",
        "nB",
        "nA",
        "co_occurs",
        Some(2.0),
        None,
    )
    .await;
    seed_edge(
        &pool,
        "e3",
        &nb,
        "s1",
        "cA",
        "nA",
        "nA",
        "co_occurs",
        Some(9.0),
        None,
    )
    .await;

    let hits = expand_neighbors(&pool, &nb, &seed("A", EntityKind::Concept), 2)
        .await
        .expect("expand ok");
    let names: Vec<&str> = hits.iter().map(|h| h.name.as_str()).collect();
    assert!(names.contains(&"B"), "B reachable: {names:?}");
    assert!(
        !names.contains(&"A"),
        "self-loop must not surface A as its own neighbor: {names:?}"
    );
}

/// 50-triple cap: a star of 60 neighbors around A truncates to 50.
#[tokio::test]
async fn expand_caps_at_fifty_triples() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    seed_chunk(&pool, "cA", "s1", 1, Some(0), "t").await;
    seed_entity_node(&pool, "nA", &nb, "s1", "concept", "A", None).await;
    seed_mention(&pool, "mA", &nb, "nA", "cA", 0).await;
    for i in 0..60usize {
        let nid = format!("n{i}");
        let cid = format!("c{i}");
        seed_chunk(&pool, &cid, "s1", 1, Some(i as i64 + 1), "t").await;
        seed_entity_node(&pool, &nid, &nb, "s1", "concept", &format!("N{i:02}"), None).await;
        seed_mention(&pool, &format!("m{i}"), &nb, &nid, &cid, 0).await;
        // Distinct weights so ranking is deterministic.
        seed_edge(
            &pool,
            &format!("e{i}"),
            &nb,
            "s1",
            "cA",
            "nA",
            &nid,
            "co_occurs",
            Some(i as f64 + 1.0),
            None,
        )
        .await;
    }

    let hits = expand_neighbors(&pool, &nb, &seed("A", EntityKind::Concept), 1)
        .await
        .expect("expand ok");
    assert_eq!(hits.len(), 50, "truncated to the 50-triple cap");
}

/// Empty seeds → Ok(vec![]).
#[tokio::test]
async fn expand_empty_seeds_returns_empty() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;
    let hits = expand_neighbors(&pool, &nb, &[], 2).await.expect("ok");
    assert!(hits.is_empty());
}

/// Seeds matching zero nodes → Ok(vec![]).
#[tokio::test]
async fn expand_zero_match_seeds_returns_empty() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;
    seed_source(&pool, "s1", &nb, 1, None).await;
    seed_chunk(&pool, "cA", "s1", 1, Some(0), "t").await;
    seed_entity_node(&pool, "nA", &nb, "s1", "concept", "A", None).await;
    seed_mention(&pool, "mA", &nb, "nA", "cA", 0).await;

    let hits = expand_neighbors(&pool, &nb, &seed("Nonexistent", EntityKind::Concept), 2)
        .await
        .expect("ok");
    assert!(hits.is_empty());
}

/// expand_neighbors excludes trashed/deselected sources at both endpoints.
#[tokio::test]
async fn expand_excludes_dead_sources() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;

    seed_source(&pool, "s-live", &nb, 1, None).await;
    seed_source(&pool, "s-trash", &nb, 1, Some("2026-01-01T00:00:00Z")).await;
    seed_chunk(&pool, "cA", "s-live", 1, Some(0), "t").await;
    seed_chunk(&pool, "cB", "s-trash", 1, Some(1), "t").await;
    seed_entity_node(&pool, "nA", &nb, "s-live", "concept", "A", None).await;
    seed_entity_node(&pool, "nB", &nb, "s-trash", "concept", "B", None).await;
    seed_mention(&pool, "mA", &nb, "nA", "cA", 0).await;
    seed_mention(&pool, "mB", &nb, "nB", "cB", 0).await;
    // Edge lives in the trashed source → its endpoint is excluded.
    seed_edge(
        &pool,
        "e1",
        &nb,
        "s-trash",
        "cB",
        "nA",
        "nB",
        "co_occurs",
        Some(2.0),
        None,
    )
    .await;

    let hits = expand_neighbors(&pool, &nb, &seed("A", EntityKind::Concept), 2)
        .await
        .expect("ok");
    assert!(hits.is_empty(), "neighbor in a trashed source is excluded");
}

/// Multi-seed exclusion: with seeds A and C and an edge C->A, neither A nor C
/// appears in the results even though A is a depth-1 neighbor reached from C.
#[tokio::test]
async fn expand_multi_seed_excludes_all_seeds() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    for (c, ts) in [("cA", 0), ("cB", 1), ("cC", 2)] {
        seed_chunk(&pool, c, "s1", 1, Some(ts), "t").await;
    }
    for (nid, name) in [("nA", "A"), ("nB", "B"), ("nC", "C")] {
        seed_entity_node(&pool, nid, &nb, "s1", "concept", name, None).await;
    }
    seed_mention(&pool, "mA", &nb, "nA", "cA", 0).await;
    seed_mention(&pool, "mB", &nb, "nB", "cB", 0).await;
    seed_mention(&pool, "mC", &nb, "nC", "cC", 0).await;
    // A -> B (so B is a genuine non-seed neighbor) and C -> A (so seed A is reached
    // as a depth>0 neighbor from seed C).
    seed_edge(
        &pool,
        "e1",
        &nb,
        "s1",
        "cA",
        "nA",
        "nB",
        "co_occurs",
        Some(2.0),
        None,
    )
    .await;
    seed_edge(
        &pool,
        "e2",
        &nb,
        "s1",
        "cC",
        "nC",
        "nA",
        "co_occurs",
        Some(2.0),
        None,
    )
    .await;

    let seeds = vec![
        ("A".to_string(), EntityKind::Concept),
        ("C".to_string(), EntityKind::Concept),
    ];
    let hits = expand_neighbors(&pool, &nb, &seeds, 2)
        .await
        .expect("expand ok");
    let names: Vec<&str> = hits.iter().map(|h| h.name.as_str()).collect();
    assert!(
        !names.contains(&"A") && !names.contains(&"C"),
        "no seed leaks into results: {names:?}"
    );
    assert!(
        names.contains(&"B"),
        "genuine neighbor B present: {names:?}"
    );
}

// ---------------------------------------------------------------------------
// NotebookGraph::load — graph-state edge cases
// ---------------------------------------------------------------------------

/// co_occurs-only graph loads with both-direction edges.
#[tokio::test]
async fn load_cooccurs_only_bidirectional() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    seed_chunk(&pool, "cA", "s1", 1, Some(0), "t").await;
    seed_entity_node(&pool, "nA", &nb, "s1", "concept", "A", None).await;
    seed_entity_node(&pool, "nB", &nb, "s1", "concept", "B", None).await;
    seed_mention(&pool, "mA", &nb, "nA", "cA", 0).await;
    seed_mention(&pool, "mB", &nb, "nB", "cA", 5).await;
    seed_edge(
        &pool,
        "e1",
        &nb,
        "s1",
        "cA",
        "nA",
        "nB",
        "co_occurs",
        Some(2.0),
        None,
    )
    .await;

    let g = NotebookGraph::load(&pool, &nb).await.expect("load ok");
    assert_eq!(g.node_count(), 2);
    assert_eq!(g.edge_count(), 2, "one co_occurs pair → two directed edges");
}

/// semantic-only graph loads with subject→object edges only.
#[tokio::test]
async fn load_semantic_only_directed_one_way() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    seed_chunk(&pool, "cA", "s1", 1, Some(0), "t").await;
    seed_entity_node(&pool, "nA", &nb, "s1", "concept", "A", None).await;
    seed_entity_node(&pool, "nB", &nb, "s1", "concept", "B", None).await;
    seed_mention(&pool, "mA", &nb, "nA", "cA", 0).await;
    seed_mention(&pool, "mB", &nb, "nB", "cA", 5).await;
    seed_edge(
        &pool,
        "e1",
        &nb,
        "s1",
        "cA",
        "nA",
        "nB",
        "founded",
        None,
        Some(0.9),
    )
    .await;

    let g = NotebookGraph::load(&pool, &nb).await.expect("load ok");
    assert_eq!(g.node_count(), 2);
    assert_eq!(g.edge_count(), 1, "semantic edge is one-way");
}

/// all-NULL-resolution graph (no canonical_name) still loads with logical nodes
/// keyed by raw name.
#[tokio::test]
async fn load_all_null_resolution() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;

    seed_source(&pool, "s1", &nb, 1, None).await;
    seed_chunk(&pool, "cA", "s1", 1, Some(0), "t").await;
    seed_entity_node(&pool, "nA", &nb, "s1", "concept", "A", None).await;
    seed_entity_node(&pool, "nB", &nb, "s1", "concept", "B", None).await;
    seed_mention(&pool, "mA", &nb, "nA", "cA", 0).await;
    seed_mention(&pool, "mB", &nb, "nB", "cA", 5).await;
    seed_edge(
        &pool,
        "e1",
        &nb,
        "s1",
        "cA",
        "nA",
        "nB",
        "co_occurs",
        Some(2.0),
        None,
    )
    .await;

    let g = NotebookGraph::load(&pool, &nb).await.expect("load ok");
    assert_eq!(g.node_count(), 2);
    let hits = ppr_expand(&pool, &g, &seed("A", EntityKind::Concept), 5)
        .await
        .expect("ppr ok");
    assert!(hits.iter().any(|h| h.name == "B"));
}

/// Mixed-confidence collapse: source A has `canonical_name = "AWS"` set, source B
/// has the same raw name "aws-thing" with NULL canonical → TWO logical nodes.
#[tokio::test]
async fn load_mixed_confidence_two_logical_nodes() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;

    seed_source(&pool, "sA", &nb, 1, None).await;
    seed_source(&pool, "sB", &nb, 1, None).await;
    seed_chunk(&pool, "cA", "sA", 1, Some(0), "t").await;
    seed_chunk(&pool, "cB", "sB", 1, Some(0), "t").await;
    // Source A: raw "aws-thing" resolved to canonical "AWS".
    seed_entity_node(&pool, "nA", &nb, "sA", "concept", "aws-thing", None).await;
    set_canonical(&pool, "nA", "AWS", 0.95).await;
    // Source B: same raw name, NULL canonical (unresolved).
    seed_entity_node(&pool, "nB", &nb, "sB", "concept", "aws-thing", None).await;
    seed_mention(&pool, "mA", &nb, "nA", "cA", 0).await;
    seed_mention(&pool, "mB", &nb, "nB", "cB", 0).await;

    let g = NotebookGraph::load(&pool, &nb).await.expect("load ok");
    assert_eq!(
        g.node_count(),
        2,
        "canonical 'AWS' and raw 'aws-thing' are distinct logical nodes"
    );
}

/// Duplicate per-source edge rows for the same logical pair sum their weights.
#[tokio::test]
async fn load_sums_duplicate_edge_weights() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;

    // Same logical pair (AWS, DB) present in two sources via canonical collapse.
    seed_source(&pool, "sA", &nb, 1, None).await;
    seed_source(&pool, "sB", &nb, 1, None).await;
    seed_chunk(&pool, "cA", "sA", 1, Some(0), "t").await;
    seed_chunk(&pool, "cB", "sB", 1, Some(0), "t").await;
    for (nid, src, nm) in [
        ("nA1", "sA", "AWS"),
        ("nA2", "sB", "AWS"),
        ("nB1", "sA", "DB"),
        ("nB2", "sB", "DB"),
    ] {
        seed_entity_node(&pool, nid, &nb, src, "concept", nm, None).await;
    }
    seed_mention(&pool, "m1", &nb, "nA1", "cA", 0).await;
    seed_mention(&pool, "m2", &nb, "nB1", "cA", 5).await;
    seed_mention(&pool, "m3", &nb, "nA2", "cB", 0).await;
    seed_mention(&pool, "m4", &nb, "nB2", "cB", 5).await;
    seed_edge(
        &pool,
        "e1",
        &nb,
        "sA",
        "cA",
        "nA1",
        "nB1",
        "founded",
        None,
        Some(0.5),
    )
    .await;
    seed_edge(
        &pool,
        "e2",
        &nb,
        "sB",
        "cB",
        "nA2",
        "nB2",
        "founded",
        None,
        Some(0.5),
    )
    .await;

    let g = NotebookGraph::load(&pool, &nb).await.expect("load ok");
    assert_eq!(g.node_count(), 2, "AWS + DB collapse across sources");
    assert_eq!(
        g.edge_count(),
        1,
        "the two per-source rows are one logical edge"
    );
}

// ---------------------------------------------------------------------------
// ppr_expand
// ---------------------------------------------------------------------------

/// Builds the known-answer graph: 4 logical nodes A,B,C,D with semantic edges
/// whose blended weights are exactly {A->B:3.0, A->C:1.5, B->C:3.0, C->A:1.5,
/// C->D:3.0} (semantic weight = SEMANTIC_BOOST(3.0) * confidence; conf 1.0→3.0,
/// conf 0.5→1.5). D is dangling.
async fn seed_known_answer_graph(pool: &sqlx::SqlitePool, nb: &str) {
    seed_source(pool, "s1", nb, 1, None).await;
    for (c, ts) in [("cA", 0), ("cB", 1), ("cC", 2), ("cD", 3)] {
        seed_chunk(pool, c, "s1", 1, Some(ts), "t").await;
    }
    for (nid, name) in [("nA", "A"), ("nB", "B"), ("nC", "C"), ("nD", "D")] {
        seed_entity_node(pool, nid, nb, "s1", "concept", name, None).await;
    }
    for (m, nid, c) in [
        ("mA", "nA", "cA"),
        ("mB", "nB", "cB"),
        ("mC", "nC", "cC"),
        ("mD", "nD", "cD"),
    ] {
        seed_mention(pool, m, nb, nid, c, 0).await;
    }
    // confidence 1.0 → weight 3.0; confidence 0.5 → weight 1.5.
    seed_edge(
        pool,
        "e1",
        nb,
        "s1",
        "cA",
        "nA",
        "nB",
        "rel_ab",
        None,
        Some(1.0),
    )
    .await;
    seed_edge(
        pool,
        "e2",
        nb,
        "s1",
        "cA",
        "nA",
        "nC",
        "rel_ac",
        None,
        Some(0.5),
    )
    .await;
    seed_edge(
        pool,
        "e3",
        nb,
        "s1",
        "cB",
        "nB",
        "nC",
        "rel_bc",
        None,
        Some(1.0),
    )
    .await;
    seed_edge(
        pool,
        "e4",
        nb,
        "s1",
        "cC",
        "nC",
        "nA",
        "rel_ca",
        None,
        Some(0.5),
    )
    .await;
    seed_edge(
        pool,
        "e5",
        nb,
        "s1",
        "cC",
        "nC",
        "nD",
        "rel_cd",
        None,
        Some(1.0),
    )
    .await;
}

/// PPR known-answer: single seed A. Expected vector (numpy, converged, α=0.85):
/// A=0.36164185 B=0.20493038 C=0.27665602 D=0.15677174.
/// graph_confidence is score max-normalized over the returned NON-seed set, so
/// the top non-seed (C) → 1.0 and B/D scale to C.
#[tokio::test]
async fn ppr_known_answer_single_seed() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;
    seed_known_answer_graph(&pool, &nb).await;

    let g = NotebookGraph::load(&pool, &nb).await.expect("load ok");
    assert_eq!(g.node_count(), 4);
    assert_eq!(g.edge_count(), 5);

    let hits = ppr_expand(&pool, &g, &seed("A", EntityKind::Concept), 5)
        .await
        .expect("ppr ok");

    // A is the seed → excluded. Non-seed reference scores: C=0.27665602,
    // B=0.20493038, D=0.15677174. Max-normalized to C: C=1.0, B≈0.740748, D≈0.566672.
    let by_name = |n: &str| hits.iter().find(|h| h.name == n).expect("hit present");
    let c = by_name("C").graph_confidence;
    let b = by_name("B").graph_confidence;
    let d = by_name("D").graph_confidence;
    assert!(!hits.iter().any(|h| h.name == "A"), "seed excluded");

    assert!((c - 1.0).abs() < 1e-4, "C normalizes to 1.0, got {c}");
    assert!((b - 0.740_740_7).abs() < 1e-4, "B confidence off: {b}");
    assert!((d - 0.566_666_6).abs() < 1e-4, "D confidence off: {d}");

    // Ranking order by score: C > B > D.
    assert_eq!(hits[0].name, "C");
    assert_eq!(hits[1].name, "B");
    assert_eq!(hits[2].name, "D");
}

/// Multi-seed personalization biases toward both seeds. Seeds A+D; reference
/// (numpy): A=0.28181583 B=0.15969564 C=0.21558911 D=0.34289942. Non-seed set is
/// {B,C}; C outranks B.
#[tokio::test]
async fn ppr_multi_seed_biases_to_both() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;
    seed_known_answer_graph(&pool, &nb).await;

    let g = NotebookGraph::load(&pool, &nb).await.expect("load ok");
    let seeds = vec![
        ("A".to_string(), EntityKind::Concept),
        ("D".to_string(), EntityKind::Concept),
    ];
    let hits = ppr_expand(&pool, &g, &seeds, 5).await.expect("ppr ok");

    assert!(
        !hits.iter().any(|h| h.name == "A" || h.name == "D"),
        "seeds excluded"
    );
    assert_eq!(hits[0].name, "C", "C outranks B");
    assert_eq!(hits[1].name, "B");
    // Max-normalized: C=1.0, B = 0.15969564/0.21558911.
    let c = hits[0].graph_confidence;
    let b = hits[1].graph_confidence;
    assert!((c - 1.0).abs() < 1e-4, "C=1.0, got {c}");
    // Reference B/C = 0.15969564 / 0.21558911 = 0.7407407.
    assert!((b - 0.740_740_7).abs() < 1e-4, "B confidence off: {b}");
}

/// Dangling-node mass teleports to the seed set (NOT uniform). Assert D's exact
/// reference score contribution: with the known-answer single-seed graph, D
/// (dangling) still receives mass, and its confidence matches the reference ratio.
/// A wrong dangling handler (uniform teleport) would shift every score.
#[tokio::test]
async fn ppr_dangling_mass_to_seeds() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;
    seed_known_answer_graph(&pool, &nb).await;

    let g = NotebookGraph::load(&pool, &nb).await.expect("load ok");
    let hits = ppr_expand(&pool, &g, &seed("A", EntityKind::Concept), 5)
        .await
        .expect("ppr ok");

    let d = hits
        .iter()
        .find(|h| h.name == "D")
        .unwrap()
        .graph_confidence;
    assert!(
        (d - 0.566_666_6).abs() < 1e-4,
        "dangling D score wrong: {d}"
    );
}

/// Damping: a higher α propagates more mass to distant nodes. Here we assert the
/// converged single-seed vector is a proper distribution (sums to ~1 pre-exclusion)
/// by checking the top hit is C (2 hops via B) outranks D at 2 hops but... we
/// instead assert convergence stability: two loads produce identical scores.
#[tokio::test]
async fn ppr_convergence_deterministic() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;
    seed_known_answer_graph(&pool, &nb).await;

    let g = NotebookGraph::load(&pool, &nb).await.expect("load ok");
    let h1 = ppr_expand(&pool, &g, &seed("A", EntityKind::Concept), 5)
        .await
        .expect("ok");
    let h2 = ppr_expand(&pool, &g, &seed("A", EntityKind::Concept), 5)
        .await
        .expect("ok");
    assert_eq!(h1, h2, "PPR is deterministic across runs");
    assert_eq!(h1[0].name, "C");
}

/// Empty / zero-match seeds → Ok(vec![]).
#[tokio::test]
async fn ppr_empty_and_zero_match_seeds() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;
    seed_known_answer_graph(&pool, &nb).await;

    let g = NotebookGraph::load(&pool, &nb).await.expect("load ok");
    let empty = ppr_expand(&pool, &g, &[], 5).await.expect("ok");
    assert!(empty.is_empty());
    let zero = ppr_expand(&pool, &g, &seed("Zzz", EntityKind::Concept), 5)
        .await
        .expect("ok");
    assert!(zero.is_empty());
}

/// Case-insensitive seed resolution: a seed passed in a different case than the
/// stored node still resolves (PPR index is lowercased, matching COLLATE NOCASE).
#[tokio::test]
async fn ppr_seed_resolution_case_insensitive() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;
    seed_known_answer_graph(&pool, &nb).await;

    let g = NotebookGraph::load(&pool, &nb).await.expect("load ok");
    // Stored node name is "A"; seed with "a".
    let hits = ppr_expand(&pool, &g, &seed("a", EntityKind::Concept), 5)
        .await
        .expect("ppr ok");
    assert!(
        hits.iter().any(|h| h.name == "C"),
        "lowercase seed resolves to node A and returns its neighbors: {hits:?}"
    );
    assert!(
        !hits.iter().any(|h| h.name == "A"),
        "the resolved seed is excluded from results"
    );
}

/// Guard trip → fallback to expand_neighbors. We use the test-only capped entry
/// point via a tiny EDGE_CAP so a small graph trips the guard and the fallback
/// path returns expand_neighbors-equivalent results.
#[tokio::test]
async fn ppr_guard_trips_to_expand_neighbors() {
    let (_dir, engine) = file_engine().await;
    let nb = new_notebook(&engine).await;
    let pool = engine.pool().await;
    seed_known_answer_graph(&pool, &nb).await;

    let g = NotebookGraph::load(&pool, &nb).await.expect("load ok");
    let expanded = expand_neighbors(&pool, &nb, &seed("A", EntityKind::Concept), 2)
        .await
        .expect("expand ok");
    let tripped = lens_core::graph::ppr_expand_capped_for_test(
        &pool,
        &g,
        &seed("A", EntityKind::Concept),
        5,
        0,
        0,
    )
    .await
    .expect("ppr ok");
    assert_eq!(
        tripped, expanded,
        "guard fallback returns expand_neighbors results"
    );
}
