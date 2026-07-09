//! Cross-document entity resolution (#155): the pure 3-tier cascade that decides
//! which per-source [`EntityNode`]s name the same real-world entity and stamps a
//! shared `canonical_name`/`resolution_conf` onto each member.
//!
//! Decoupled from I/O so it is unit-testable offline: the cascade takes an
//! in-memory node set plus hand-built vectors, a [`VectorStore`] handle, an
//! optional [`LlmProvider`], and an [`AdjudicationCache`] trait object. The
//! Step-4 worker wires the real DB/Lance/LLM handles; tests use mocks.
//!
//! Correctness bias is **under-merge** (Principle 2): over-merge is unrecoverable,
//! under-merge is recoverable by a `resolution_prompt_version` bump. Thresholds are
//! conservative and typed-veto is inherent (only same-`kind` names ever group).

pub mod worker;

pub use worker::{RESOLUTION_PROMPT_VERSION, ResolveNotebook, spawn_resolution_worker};

use std::collections::HashMap;

use async_trait::async_trait;
use sqlx::SqlitePool;

use crate::enrichment::MapError;
use crate::enrichment::map::run_llm_with_retries;
use crate::enrichment::meta::Budget;
use crate::error::LensError;
use crate::graph::EntityNode;
use crate::llm::LlmProvider;
use crate::vector_store::{Coordinate, VectorStore};

/// Per-notebook LLM-call ceiling for the resolution pass (own budget, never shares
/// the enrichment `SessionBudget`). Backstops the persisted adjudication cache.
pub const RESOLUTION_MAX_CALLS_PER_NOTEBOOK: u32 = 24;
/// Per-notebook output-token ceiling for the resolution pass.
pub const RESOLUTION_MAX_TOKENS_PER_NOTEBOOK: u32 = 48_000;
/// Max output tokens requested for a single adjudication call (bool + confidence).
pub const RESOLUTION_ADJUDICATION_MAX_TOKENS: u32 = 512;

/// ANN neighbours fetched per node in Tier 2.
const TIER2_ANN_K: usize = 8;
/// Cosine similarity at/above which Tier 2 auto-merges without asking the LLM.
const TIER2_AUTO_MERGE_SIM: f64 = 0.90;
/// Cosine similarity below which Tier 2 keeps nodes separate (never adjudicates).
const TIER2_KEEP_SEPARATE_SIM: f64 = 0.72;
/// Minimum final confidence to write a canonical assignment (exact cutoff).
const WRITE_CONFIDENCE_BAR: f64 = 0.88;

/// A resolution result for one node: the canonical name it now aliases and the
/// confidence by which it joined its group. Emitted only when `resolution_conf`
/// meets [`WRITE_CONFIDENCE_BAR`]. Singletons/sub-bar members get no update.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolutionUpdate {
    pub entity_node_id: String,
    pub canonical_name: String,
    pub resolution_conf: f64,
}

/// A persisted store of LLM adjudication verdicts keyed on `(normalized_pair,
/// resolution_prompt_version)` (Tier 3). A hit avoids re-paying the LLM on
/// full-notebook recompute; a version bump invalidates by key.
#[async_trait]
pub trait AdjudicationCache: Send + Sync {
    /// Returns the cached `(verdict, confidence)` for this pair+version, if any.
    async fn get(
        &self,
        normalized_pair: &str,
        version: &str,
    ) -> Result<Option<(bool, f64)>, LensError>;

    /// Persists a verdict for this pair+version.
    async fn put(
        &self,
        normalized_pair: &str,
        version: &str,
        notebook_id: &str,
        verdict: bool,
        confidence: f64,
    ) -> Result<(), LensError>;
}

/// Everything the cascade needs, all borrowed so callers own the lifetimes.
pub struct ResolveInput<'a> {
    /// All entity nodes in the notebook (across every source).
    pub nodes: &'a [EntityNode],
    /// `entity_node_id` → L2-normalized embedding (the worker builds these).
    pub vectors: &'a HashMap<String, Vec<f32>>,
    pub store: &'a dyn VectorStore,
    pub coord: &'a Coordinate,
    /// `None` OR transport failure degrades to Tier 1-2 only (never fails the pass).
    pub provider: Option<&'a dyn LlmProvider>,
    pub cache: &'a dyn AdjudicationCache,
    pub prompt_version: &'a str,
    /// `entity_node_id` pairs known-same from coref (Tier-1 seed).
    pub coref_pairs: &'a [(String, String)],
    pub notebook_id: &'a str,
}

/// Normalizes an entity name for exact-match grouping and cache keys: trims, collapses
/// internal whitespace to a single space, casefolds, and strips trailing punctuation.
pub fn normalize_name(name: &str) -> String {
    let collapsed = name.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = collapsed.trim_end_matches(|c: char| c.is_ascii_punctuation());
    trimmed.to_lowercase()
}

/// Builds the Tier-2 embedding text: `"{kind}: {name} — {definition}"`, or
/// `"{kind}: {name}"` when the node has no definition.
pub fn embedding_text(node: &EntityNode) -> String {
    match &node.definition {
        Some(def) if !def.trim().is_empty() => {
            format!("{}: {} — {}", node.kind.as_str(), node.name, def)
        }
        _ => format!("{}: {}", node.kind.as_str(), node.name),
    }
}

/// A disjoint-set forest over node indices that also records every join edge and its
/// confidence. Grouping uses plain union-find; per-node confidence is derived from the
/// recorded edges in [`node_confidences`] (a node binds to its group only as strongly
/// as the weakest edge on its path — conservative for multi-hop chains).
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    /// Every accepted union edge `(a, b, conf)`. Only edges that actually merged two
    /// disjoint sets are kept, so they form a spanning forest (one tree per group).
    edges: Vec<(usize, usize, f64)>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            edges: Vec::new(),
        }
    }

    fn find(&self, mut i: usize) -> usize {
        while self.parent[i] != i {
            i = self.parent[i];
        }
        i
    }

    /// Unions the sets of `a` and `b` with join confidence `conf`, recording the edge.
    /// Already-unioned inputs are a no-op (no edge recorded).
    fn union(&mut self, a: usize, b: usize, conf: f64) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        self.edges.push((a, b, conf));
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

/// Each node's binding confidence = the minimum edge weight on the tree path linking it
/// into its group (the weakest link is the conservative bound for a multi-hop chain). The
/// union edges form a spanning forest, so a non-singleton member always sits on ≥1 edge; a
/// singleton has none → `None`. The traversal anchor takes the group's overall minimum so
/// it never reads as more confident than the weakest link holding the group together.
fn node_confidences(n: usize, uf: &UnionFind) -> Vec<Option<f64>> {
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for &(a, b, conf) in &uf.edges {
        adj[a].push((b, conf));
        adj[b].push((a, conf));
    }

    let mut conf = vec![None; n];
    let mut visited = vec![false; n];
    for start in 0..n {
        if adj[start].is_empty() || visited[start] {
            continue;
        }
        let mut group_min = f64::INFINITY;
        let mut stack = vec![(start, f64::INFINITY)];
        visited[start] = true;
        while let Some((node, path_min)) = stack.pop() {
            if node != start {
                conf[node] = Some(path_min);
                group_min = group_min.min(path_min);
            }
            for &(next, edge) in &adj[node] {
                if !visited[next] {
                    visited[next] = true;
                    stack.push((next, path_min.min(edge)));
                }
            }
        }
        conf[start] = Some(group_min);
    }
    conf
}

/// Runs the 3-tier cascade over a notebook's entity nodes and returns the canonical
/// assignments for every node whose final confidence meets [`WRITE_CONFIDENCE_BAR`].
///
/// Degradation is silent: a missing/failing LLM provider or an exhausted budget
/// leaves the affected Tier-3 pairs unresolved but still returns the Tier 1-2
/// results (`Ok`). A hard `LensError` (store/cache I/O) aborts the pass.
pub async fn resolve_notebook(
    input: ResolveInput<'_>,
    budget: &mut Budget,
) -> Result<Vec<ResolutionUpdate>, LensError> {
    let nodes = input.nodes;
    if nodes.is_empty() {
        return Ok(Vec::new());
    }

    let mut uf = UnionFind::new(nodes.len());
    let index_by_id: HashMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id.as_str(), i))
        .collect();

    // --- Tier 1: exact normalized (name, kind) + coref seeds (confidence 1.0) ---
    tier1_exact_and_coref(nodes, &index_by_id, input.coref_pairs, &mut uf);

    // --- Tier 2: embedding ANN. >=0.90 auto-union; <0.72 skip; the band is Tier 3.
    let candidates = tier2_ann(&input, nodes, &index_by_id, &mut uf).await?;

    // --- Tier 3: LLM adjudication of the mid band (cache-first, degrade on failure).
    tier3_adjudicate(&input, nodes, &candidates, &mut uf, budget).await?;

    Ok(assign_canonical(nodes, &uf))
}

/// Tier 1: group nodes with identical `(normalize_name(name), kind)` and union every
/// coref pair. Typed veto is inherent — the key includes `kind`, so cross-kind names
/// never share a group.
fn tier1_exact_and_coref(
    nodes: &[EntityNode],
    index_by_id: &HashMap<&str, usize>,
    coref_pairs: &[(String, String)],
    uf: &mut UnionFind,
) {
    let mut first_of_key: HashMap<(String, &'static str), usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        let key = (normalize_name(&node.name), node.kind.as_str());
        match first_of_key.get(&key) {
            Some(&first) => uf.union(first, i, 1.0),
            None => {
                first_of_key.insert(key, i);
            }
        }
    }

    for (a, b) in coref_pairs {
        if let (Some(&ia), Some(&ib)) = (index_by_id.get(a.as_str()), index_by_id.get(b.as_str())) {
            uf.union(ia, ib, 1.0);
        }
    }
}

/// A mid-band candidate pair for Tier-3 adjudication, deduped by canonical id order.
type CandidatePair = (usize, usize, f64);

/// Tier 2: ANN neighbour scan per node. Auto-unions at similarity >= 0.90, skips
/// below 0.72, and collects the 0.72..0.90 band as Tier-3 candidates. Similarity is
/// `1.0 - cosine_distance` (lance returns distance). Typed veto via the `kind` filter.
async fn tier2_ann(
    input: &ResolveInput<'_>,
    nodes: &[EntityNode],
    index_by_id: &HashMap<&str, usize>,
    uf: &mut UnionFind,
) -> Result<Vec<CandidatePair>, LensError> {
    let mut seen_pairs: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    let mut candidates: Vec<CandidatePair> = Vec::new();

    for (i, node) in nodes.iter().enumerate() {
        let query = match input.vectors.get(&node.id) {
            Some(v) => v,
            None => continue, // no vector built for this node — Tier 1 only
        };
        let neighbors = input
            .store
            .entity_ann(input.coord, query, TIER2_ANN_K, Some(node.kind.as_str()))
            .await?;

        for (neighbor_id, distance) in neighbors {
            let j = match index_by_id.get(neighbor_id.as_str()) {
                Some(&j) if j != i => j,
                _ => continue, // self-hit or a node not in this pass
            };
            let sim = 1.0 - distance as f64;
            let pair = if i < j { (i, j) } else { (j, i) };
            if uf.find(i) == uf.find(j) {
                continue; // already same group
            }
            if sim >= TIER2_AUTO_MERGE_SIM {
                uf.union(i, j, sim);
            } else if sim < TIER2_KEEP_SEPARATE_SIM {
                continue;
            } else if seen_pairs.insert(pair) {
                candidates.push((pair.0, pair.1, sim));
            }
        }
    }
    Ok(candidates)
}

/// Tier 3: adjudicate each mid-band candidate against the cache, then the LLM on a
/// miss. Unions only on `verdict == true && confidence >= 0.88`. Budget exhaustion
/// stops further adjudication; a per-pair LLM/parse failure skips that pair. Both
/// are silent degradations — never a hard error.
async fn tier3_adjudicate(
    input: &ResolveInput<'_>,
    nodes: &[EntityNode],
    candidates: &[CandidatePair],
    uf: &mut UnionFind,
    budget: &mut Budget,
) -> Result<(), LensError> {
    for &(i, j, sim) in candidates {
        if uf.find(i) == uf.find(j) {
            continue; // a prior Tier-3 union already merged them transitively
        }
        let (na, nb) = (&nodes[i], &nodes[j]);
        let key = adjudication_key(na, nb);

        let verdict = match input.cache.get(&key, input.prompt_version).await? {
            Some(cached) => Some(cached),
            None => {
                match adjudicate_via_llm(input, na, nb, sim, &key, budget).await {
                    LlmVerdict::Verdict(v) => Some(v),
                    LlmVerdict::Skip => None, // provider absent / degraded — leave unresolved
                    LlmVerdict::BudgetStop => break, // stop adjudicating the rest
                }
            }
        };

        if let Some((same, conf)) = verdict {
            if same && conf >= WRITE_CONFIDENCE_BAR {
                uf.union(i, j, conf);
            } else {
                tracing::debug!(
                    pair = %key,
                    same,
                    confidence = conf,
                    "resolution: Tier-3 verdict discarded (below write bar or not-same)"
                );
            }
        }
    }
    Ok(())
}

/// Outcome of a single LLM adjudication attempt.
enum LlmVerdict {
    Verdict((bool, f64)),
    /// Provider absent, transport error, or malformed reply — skip this pair only.
    Skip,
    /// Budget exhausted — stop adjudicating the remaining pairs.
    BudgetStop,
}

const ADJUDICATION_SYSTEM_PROMPT: &str = "You judge whether two named entities refer to \
the SAME real-world entity. Respond with ONLY a JSON object, no prose, no markdown \
fences, with EXACTLY these keys: \"same\" (boolean), \"confidence\" (number between 0 \
and 1). Bias toward \"same\": false when uncertain.";

/// Calls the LLM for one pair (cache miss). Persists the verdict on success. Returns
/// [`LlmVerdict::Skip`] when there is no provider or the call degrades, and
/// [`LlmVerdict::BudgetStop`] on a pre-dispatch budget breach.
async fn adjudicate_via_llm(
    input: &ResolveInput<'_>,
    a: &EntityNode,
    b: &EntityNode,
    sim: f64,
    key: &str,
    budget: &mut Budget,
) -> LlmVerdict {
    let provider = match input.provider {
        Some(p) => p,
        None => return LlmVerdict::Skip,
    };

    let user_prompt = format!(
        "Entity A — kind: {}; name: {}; definition: {}\n\
         Entity B — kind: {}; name: {}; definition: {}\n\
         Embedding cosine similarity: {sim:.3}\n\
         Are A and B the same real-world entity?",
        a.kind.as_str(),
        a.name,
        a.definition.as_deref().unwrap_or("(none)"),
        b.kind.as_str(),
        b.name,
        b.definition.as_deref().unwrap_or("(none)"),
    );

    let result = run_llm_with_retries(
        provider,
        budget,
        ADJUDICATION_SYSTEM_PROMPT,
        &user_prompt,
        RESOLUTION_ADJUDICATION_MAX_TOKENS,
        parse_adjudication,
    )
    .await;

    match result {
        Ok(Some((same, conf))) => {
            if let Err(e) = input
                .cache
                .put(key, input.prompt_version, input.notebook_id, same, conf)
                .await
            {
                // A cache-write failure must not fail the pass; the verdict still stands.
                tracing::warn!(error = %e, pair = %key, "resolution: adjudication cache put failed");
            }
            LlmVerdict::Verdict((same, conf))
        }
        Ok(None) => LlmVerdict::Skip, // retries exhausted — degrade
        Err(MapError::Llm(_)) => LlmVerdict::Skip, // transport failure — degrade this pair
        Err(MapError::BudgetExceeded) => LlmVerdict::BudgetStop,
    }
}

/// Parses `{"same": bool, "confidence": number}` from an LLM reply, tolerating
/// markdown fences / preamble and clamping confidence to `[0, 1]`.
fn parse_adjudication(body: &str) -> Result<(bool, f64), LensError> {
    let json = crate::enrichment::meta::extract_json_object(body).unwrap_or(body);
    let value: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| LensError::Parse(format!("adjudication invalid: {e}")))?;
    let same = value
        .get("same")
        .and_then(|v| v.as_bool())
        .ok_or_else(|| LensError::Parse("adjudication missing bool `same`".into()))?;
    let confidence = value
        .get("confidence")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| LensError::Parse("adjudication missing number `confidence`".into()))?
        .clamp(0.0, 1.0);
    Ok((same, confidence))
}

/// Deterministic cache key: `"{kind}|{a}|{b}"` where `a,b` are the two normalized
/// names in sorted order (pairs are order-independent). Both nodes share a `kind`
/// (typed veto), so the pair's kind is unambiguous.
fn adjudication_key(a: &EntityNode, b: &EntityNode) -> String {
    let (mut n1, mut n2) = (normalize_name(&a.name), normalize_name(&b.name));
    if n1 > n2 {
        std::mem::swap(&mut n1, &mut n2);
    }
    format!("{}|{n1}|{n2}", a.kind.as_str())
}

/// Assigns each multi-member group a canonical name (longest by char count, ties to
/// lexicographically smallest) and emits an update for every member whose group
/// confidence meets [`WRITE_CONFIDENCE_BAR`]. Singletons and sub-bar members emit none.
fn assign_canonical(nodes: &[EntityNode], uf: &UnionFind) -> Vec<ResolutionUpdate> {
    let confidences = node_confidences(nodes.len(), uf);
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..nodes.len() {
        groups.entry(uf.find(i)).or_default().push(i);
    }

    let mut updates = Vec::new();
    for members in groups.values() {
        if members.len() < 2 {
            continue; // singleton — canonical_name stays NULL
        }
        let canonical = canonical_name(members.iter().map(|&i| nodes[i].name.as_str()));
        for &i in members {
            let conf = confidences[i].unwrap_or(0.0);
            if conf >= WRITE_CONFIDENCE_BAR {
                updates.push(ResolutionUpdate {
                    entity_node_id: nodes[i].id.clone(),
                    canonical_name: canonical.clone(),
                    resolution_conf: conf,
                });
            } else {
                tracing::debug!(
                    node = %nodes[i].id,
                    confidence = conf,
                    "resolution: group member below write bar, left unresolved"
                );
            }
        }
    }
    updates
}

/// Picks the canonical name: longest by char count, breaking ties by the
/// lexicographically smallest name.
fn canonical_name<'a>(names: impl Iterator<Item = &'a str>) -> String {
    names
        .max_by(|a, b| {
            a.chars()
                .count()
                .cmp(&b.chars().count())
                .then_with(|| b.cmp(a)) // reversed: on a tie, prefer the smaller name
        })
        .unwrap_or_default()
        .to_string()
}

/// SQLite-backed [`AdjudicationCache`] over the `adjudication_cache` table (0016).
/// Production uses this; tests use a mock. Cheap to construct (borrows the pool).
pub struct SqliteAdjudicationCache<'a> {
    pub pool: &'a SqlitePool,
}

impl SqliteAdjudicationCache<'_> {
    /// Deletes cache rows for `notebook_id` whose version differs from `current_version`
    /// (a version bump invalidates prior verdicts), so old-version rows don't accumulate.
    pub async fn gc_stale(
        &self,
        notebook_id: &str,
        current_version: &str,
    ) -> Result<(), LensError> {
        sqlx::query(
            "DELETE FROM adjudication_cache \
             WHERE notebook_id = ? AND resolution_prompt_version != ?",
        )
        .bind(notebook_id)
        .bind(current_version)
        .execute(self.pool)
        .await?;
        Ok(())
    }
}

#[async_trait]
impl AdjudicationCache for SqliteAdjudicationCache<'_> {
    async fn get(
        &self,
        normalized_pair: &str,
        version: &str,
    ) -> Result<Option<(bool, f64)>, LensError> {
        let row: Option<(i64, f64)> = sqlx::query_as(
            "SELECT verdict, confidence FROM adjudication_cache \
             WHERE normalized_pair = ? AND resolution_prompt_version = ?",
        )
        .bind(normalized_pair)
        .bind(version)
        .fetch_optional(self.pool)
        .await?;
        Ok(row.map(|(verdict, confidence)| (verdict != 0, confidence)))
    }

    async fn put(
        &self,
        normalized_pair: &str,
        version: &str,
        notebook_id: &str,
        verdict: bool,
        confidence: f64,
    ) -> Result<(), LensError> {
        sqlx::query(
            "INSERT INTO adjudication_cache \
             (normalized_pair, resolution_prompt_version, notebook_id, verdict, confidence, created_at) \
             VALUES (?, ?, ?, ?, ?, ?) \
             ON CONFLICT (normalized_pair, resolution_prompt_version) DO UPDATE SET \
             verdict = excluded.verdict, confidence = excluded.confidence, \
             notebook_id = excluded.notebook_id, created_at = excluded.created_at",
        )
        .bind(normalized_pair)
        .bind(version)
        .bind(notebook_id)
        .bind(i64::from(verdict))
        .bind(confidence)
        .bind(chrono::Utc::now().to_rfc3339())
        .execute(self.pool)
        .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
