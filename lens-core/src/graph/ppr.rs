//! #156b in-memory personalized PageRank over a loaded notebook entity graph.
//!
//! [`NotebookGraph::load`] collapses the per-source node/edge rows into one
//! directed logical graph (cache-ready: #21 holds `Arc<NotebookGraph>`).
//! [`ppr_expand`] runs power-iteration PPR over it, falling back to the SQL
//! [`expand_neighbors`] traversal when the graph exceeds the size guard.

use std::collections::HashMap;

use petgraph::Direction;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use sqlx::{Row, SqlitePool};

use super::{EntityKind, GraphHit, Relation, blended_edge_weight, expand_neighbors};
use crate::LensError;

/// Guard ceilings: above either, `ppr_expand` skips the in-memory PPR and
/// delegates to the SQL traversal. Generous — co-occurrence edges are windowed +
/// top-N (see `build.rs`), so real counts sit well below the naive all-pairs bound.
const EDGE_CAP: usize = 750_000;
const NODE_CAP: usize = 100_000;

/// PPR damping factor (probability of following an edge vs. teleporting).
const ALPHA: f64 = 0.85;
const MAX_ITERS: usize = 100;
const CONVERGENCE_L1: f64 = 1e-6;

/// A logical node's identity plus the chunk evidence backing it.
struct LogicalNode {
    name: String,
    kind: EntityKind,
    chunk_ids: Vec<String>,
}

/// A notebook's entity graph collapsed to logical nodes (one per
/// `(COALESCE(canonical_name, name), kind)`), directed, with blended edge weights.
/// `co_occurs` edges are inserted both ways; semantic edges subject→object only.
pub struct NotebookGraph {
    notebook_id: String,
    graph: DiGraph<LogicalNode, f32>,
    index: HashMap<(String, EntityKind), NodeIndex>,
}

impl NotebookGraph {
    /// Loads and collapses the live-source portion of a notebook's entity graph.
    /// Live-source filtering (`trashed_at IS NULL AND selected = 1`) is applied at
    /// the node level; edges with an excluded endpoint are dropped.
    pub async fn load(pool: &SqlitePool, notebook_id: &str) -> Result<NotebookGraph, LensError> {
        // Per-source node rows scoped to live sources. `logical_key` collapses
        // cross-doc aliases the same way #156a does (COALESCE(canonical_name, name)).
        let node_rows = sqlx::query(
            "SELECT en.id,
                    COALESCE(en.canonical_name, en.name) AS logical_name,
                    en.kind
             FROM entity_nodes en
             JOIN sources s ON s.id = en.source_id
             WHERE en.notebook_id = ?1
               AND s.trashed_at IS NULL AND s.selected = 1",
        )
        .bind(notebook_id)
        .fetch_all(pool)
        .await?;

        let mut graph: DiGraph<LogicalNode, f32> = DiGraph::new();
        let mut index: HashMap<(String, EntityKind), NodeIndex> = HashMap::new();
        // Per-source node id → logical NodeIndex (for edge endpoint resolution).
        let mut node_to_logical: HashMap<String, NodeIndex> = HashMap::new();

        for row in &node_rows {
            let id: String = row.get("id");
            let logical_name: String = row.get("logical_name");
            let kind_str: String = row.get("kind");
            let kind = EntityKind::from_db(&kind_str)?;
            let key = (logical_name.clone(), kind);
            let node_index = *index.entry(key).or_insert_with(|| {
                graph.add_node(LogicalNode {
                    name: logical_name,
                    kind,
                    chunk_ids: Vec::new(),
                })
            });
            node_to_logical.insert(id, node_index);
        }

        // Chunk evidence per logical node, from mentions (live-source scoped).
        let mention_rows = sqlx::query(
            "SELECT DISTINCT em.entity_node_id, em.chunk_id
             FROM entity_mentions em
             JOIN entity_nodes en ON en.id = em.entity_node_id
             JOIN sources s ON s.id = en.source_id
             WHERE em.notebook_id = ?1
               AND s.trashed_at IS NULL AND s.selected = 1",
        )
        .bind(notebook_id)
        .fetch_all(pool)
        .await?;

        for row in &mention_rows {
            let node_id: String = row.get("entity_node_id");
            let chunk_id: String = row.get("chunk_id");
            if let Some(&node_index) = node_to_logical.get(&node_id) {
                let ln = &mut graph[node_index];
                if !ln.chunk_ids.contains(&chunk_id) {
                    ln.chunk_ids.push(chunk_id);
                }
            }
        }

        // Edges. `from_node`/`to_node` reference per-source `entity_nodes.id`; map
        // both to logical indices and drop the edge if either endpoint was excluded.
        let edge_rows = sqlx::query(
            "SELECT ee.from_node, ee.to_node, ee.relation, ee.weight, ee.confidence
             FROM entity_edges ee
             JOIN sources s ON s.id = ee.source_id
             WHERE ee.notebook_id = ?1
               AND s.trashed_at IS NULL AND s.selected = 1",
        )
        .bind(notebook_id)
        .fetch_all(pool)
        .await?;

        // Accumulate summed blended weight per directed logical pair before
        // materializing edges, so a logical pair present in N sources is one edge.
        let mut directed: HashMap<(NodeIndex, NodeIndex), f32> = HashMap::new();
        for row in &edge_rows {
            let from_id: String = row.get("from_node");
            let to_id: String = row.get("to_node");
            let (from_ix, to_ix) =
                match (node_to_logical.get(&from_id), node_to_logical.get(&to_id)) {
                    (Some(&f), Some(&t)) => (f, t),
                    _ => continue,
                };
            if from_ix == to_ix {
                continue;
            }
            let relation = Relation::from_db(&row.get::<String, _>("relation"));
            let weight: Option<f64> = row.get("weight");
            let confidence: Option<f64> = row.get("confidence");
            let w = blended_edge_weight(&relation, weight, confidence);

            match relation {
                Relation::CoOccurs => {
                    *directed.entry((from_ix, to_ix)).or_insert(0.0) += w;
                    *directed.entry((to_ix, from_ix)).or_insert(0.0) += w;
                }
                Relation::Semantic(_) => {
                    *directed.entry((from_ix, to_ix)).or_insert(0.0) += w;
                }
            }
        }
        for ((from_ix, to_ix), w) in directed {
            graph.add_edge(from_ix, to_ix, w);
        }

        Ok(NotebookGraph {
            notebook_id: notebook_id.to_string(),
            graph,
            index,
        })
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Resolves seed `(name, kind)` pairs to logical node indices, matching by
    /// logical key. Unmatched seeds are silently skipped.
    fn resolve_seeds(&self, seeds: &[(String, EntityKind)]) -> Vec<NodeIndex> {
        let mut out = Vec::new();
        for (name, kind) in seeds {
            if let Some(&ix) = self.index.get(&(name.clone(), *kind))
                && !out.contains(&ix)
            {
                out.push(ix);
            }
        }
        out
    }
}

/// Runs personalized PageRank from `seeds` over `graph`, returning the top-`k`
/// non-seed hits ranked by score (`graph_confidence` max-normalized; `relation`
/// always `None`). Guard: if the graph exceeds [`EDGE_CAP`]/[`NODE_CAP`], delegates
/// to the SQL [`expand_neighbors`] (depth 2) with a warning. Empty/zero-match
/// seeds → `Ok(vec![])`.
pub async fn ppr_expand(
    pool: &SqlitePool,
    graph: &NotebookGraph,
    seeds: &[(String, EntityKind)],
    k: usize,
) -> Result<Vec<GraphHit>, LensError> {
    ppr_expand_capped(pool, graph, seeds, k, EDGE_CAP, NODE_CAP).await
}

/// Test-only entry point exposing the injectable caps so the guard-trip fallback
/// can be exercised without materializing a graph beyond the production ceiling.
#[doc(hidden)]
pub async fn ppr_expand_capped_for_test(
    pool: &SqlitePool,
    graph: &NotebookGraph,
    seeds: &[(String, EntityKind)],
    k: usize,
    edge_cap: usize,
    node_cap: usize,
) -> Result<Vec<GraphHit>, LensError> {
    ppr_expand_capped(pool, graph, seeds, k, edge_cap, node_cap).await
}

/// [`ppr_expand`] with injectable caps so the guard-trip fallback is testable
/// without materializing a graph beyond the production ceiling.
async fn ppr_expand_capped(
    pool: &SqlitePool,
    graph: &NotebookGraph,
    seeds: &[(String, EntityKind)],
    k: usize,
    edge_cap: usize,
    node_cap: usize,
) -> Result<Vec<GraphHit>, LensError> {
    if graph.edge_count() > edge_cap || graph.node_count() > node_cap {
        tracing::warn!(
            notebook_id = %graph.notebook_id,
            nodes = graph.node_count(),
            edges = graph.edge_count(),
            "ppr_expand: graph exceeds size guard, falling back to expand_neighbors"
        );
        return expand_neighbors(pool, &graph.notebook_id, seeds, 2).await;
    }

    let seed_nodes = graph.resolve_seeds(seeds);
    if seed_nodes.is_empty() || k == 0 {
        return Ok(Vec::new());
    }

    let scores = power_iteration(&graph.graph, &seed_nodes);

    // Rank non-seed nodes by score desc.
    let mut ranked: Vec<(NodeIndex, f64)> = scores
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| {
            let ix = NodeIndex::new(i);
            if seed_nodes.contains(&ix) {
                None
            } else {
                Some((ix, s))
            }
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(k);

    let max_score = ranked.first().map(|(_, s)| *s).unwrap_or(0.0);
    let hits = ranked
        .into_iter()
        .map(|(ix, score)| {
            let node = &graph.graph[ix];
            let confidence = if max_score > 0.0 {
                (score / max_score) as f32
            } else {
                0.0
            };
            GraphHit {
                name: node.name.clone(),
                kind: node.kind,
                chunk_ids: node.chunk_ids.clone(),
                graph_confidence: confidence,
                relation: None,
            }
        })
        .collect();

    Ok(hits)
}

/// Power-iteration personalized PageRank. Teleport (and dangling-node mass) both
/// land on the personalization vector — uniform over `seed_nodes` — NOT uniform
/// over all nodes. Transition probability along an out-edge is proportional to its
/// blended weight (per-node sum-normalized). Converges at L1 delta < 1e-6 or 100 iters.
fn power_iteration(graph: &DiGraph<LogicalNode, f32>, seed_nodes: &[NodeIndex]) -> Vec<f64> {
    let n = graph.node_count();
    let mut personalization = vec![0.0f64; n];
    let seed_mass = 1.0 / seed_nodes.len() as f64;
    for &ix in seed_nodes {
        personalization[ix.index()] = seed_mass;
    }

    // Per-node out-weight sum (for transition normalization); zero = dangling.
    let mut out_sum = vec![0.0f64; n];
    for edge in graph.edge_references() {
        out_sum[edge.source().index()] += *edge.weight() as f64;
    }

    let mut rank = personalization.clone();
    for _ in 0..MAX_ITERS {
        let mut next = vec![0.0f64; n];
        let mut dangling_mass = 0.0f64;

        for i in 0..n {
            let ix = NodeIndex::new(i);
            if out_sum[i] <= 0.0 {
                dangling_mass += rank[i];
                continue;
            }
            for edge in graph.edges_directed(ix, Direction::Outgoing) {
                let share = (*edge.weight() as f64) / out_sum[i];
                next[edge.target().index()] += ALPHA * rank[i] * share;
            }
        }

        // Teleport + dangling mass both flow to the personalization vector.
        let redistributed = (1.0 - ALPHA) + ALPHA * dangling_mass;
        for j in 0..n {
            next[j] += redistributed * personalization[j];
        }

        let delta: f64 = next
            .iter()
            .zip(rank.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        rank = next;
        if delta < CONVERGENCE_L1 {
            break;
        }
    }
    rank
}
