//! Tiered Context Router (issue #21): the single integration point for grounded
//! retrieval. Given a pre-embedded query, it decides Tier-1 (inject the raw
//! selected corpus) vs Tier-2 (fused retrieval + parent auto-merge), applies the
//! #39 dense pre-filter, optionally folds in a deterministic graph arm, and returns
//! a budgeted, document-ordered, hydrated [`RouterOutput`]. It performs NO LLM call
//! and NO prompt assembly — that is a downstream `answer()` step.

use std::collections::{HashMap, HashSet};

use sqlx::SqlitePool;
use tokenizers::Tokenizer;

use crate::LensError;
use crate::chunk::kind;
use crate::config::{ModelConfig, RetrievalConfig, TierThresholds};
use crate::graph::{EntityKind, NotebookGraph, entity_evidence, entity_lookup, ppr_expand};
use crate::vector_store::{Coordinate, VectorStore};

use super::{HitSource, MAX_OVERFETCH, OVERFETCH, Reranker, RetrievalHit, bm25, live_chunk_ids};

/// Headroom reserved for the model's generation.
const RESERVED_OUTPUT: u32 = 2_048;
/// System/prompt-scaffold budget the downstream consumer will spend.
const SYSTEM_OVERHEAD: u32 = 512;
/// Near-cap band that triggers an exact tokenizer recount of a `chars/4` estimate.
const MARGIN: u32 = 256;

/// ABS_CAP band for `ctx <= 8_192` (spec line 38).
const ABS_CAP_SMALL: usize = 4_000;
/// ABS_CAP band for `ctx <= 32_768`.
const ABS_CAP_MEDIUM: usize = 20_000;
/// ABS_CAP band for `ctx >= 32_769` (covers 128K+).
const ABS_CAP_LARGE: usize = 48_000;
/// Fraction of `usable_input` allotted to the Tier-1 raw-corpus band (spec line 38).
const TIER1_FRACTION: f32 = 0.65;

/// Guard on the size of the `source_id IN (...)` pre-filter literal. Above this the
/// router falls back to notebook-scope search + the SQLite `live_chunk_ids`
/// post-filter (both correct; the fallback is just slower). Realistic per-notebook
/// source counts are tens, so the guard is purely defensive.
const MAX_PREFILTER_IDS: usize = 512;

/// Which tier the router selected for a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Raw selected corpus fits the budget — inject it directly.
    Tier1,
    /// Corpus overflows — fused retrieval with parent auto-merge.
    Tier2,
}

/// Per-unit provenance. `graph_confidence` is `Some(_)` only when the unit was
/// surfaced by the graph arm.
#[derive(Debug, Clone, PartialEq)]
pub struct Provenance {
    pub source: HitSource,
    pub graph_confidence: Option<f32>,
}

/// One hydrated, document-ordered context unit ready for injection (NOT a prompt).
#[derive(Debug, Clone, PartialEq)]
pub struct ContextUnit {
    pub text: String,
    pub source_id: String,
    pub chunk_id: String,
    /// `Some` when auto-merged from a parent (Tier-2); `None` for Tier-1 parents.
    pub parent_id: Option<String>,
    /// `source_anchor` / `section_path` locator.
    pub locator: Option<String>,
    /// Document order after the final re-sort.
    pub order_index: usize,
    pub provenance: Provenance,
}

/// The router's budgeted, tier-tagged, document-ordered output.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterOutput {
    pub tier: Tier,
    pub units: Vec<ContextUnit>,
    pub total_tokens: usize,
}

/// Derived per-tier token caps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TierCap {
    pub tier1_cap: usize,
    pub tier2_cap: usize,
}

/// Token-budget breakdown for a model context window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenBudget {
    /// `ctx_limit - RESERVED_OUTPUT - SYSTEM_OVERHEAD` (saturating, floored at 0).
    pub usable_input: usize,
    pub reserved_output: usize,
    pub system_overhead: usize,
}

/// Computes the usable-input budget for a model context window (saturating).
fn token_budget(ctx: u32) -> TokenBudget {
    let usable = ctx
        .saturating_sub(RESERVED_OUTPUT)
        .saturating_sub(SYSTEM_OVERHEAD);
    TokenBudget {
        usable_input: usable as usize,
        reserved_output: RESERVED_OUTPUT as usize,
        system_overhead: SYSTEM_OVERHEAD as usize,
    }
}

/// Absolute Tier-1 cap band for a context window (spec line 38).
fn abs_cap(ctx: u32) -> usize {
    match ctx {
        0..=8_192 => ABS_CAP_SMALL,
        8_193..=32_768 => ABS_CAP_MEDIUM,
        _ => ABS_CAP_LARGE,
    }
}

/// Derives the tier caps from a model context window. `ctx == 0` (unknown context)
/// falls back to the static [`TierThresholds`].
fn derive_tier_caps(ctx: u32, fallback: &TierThresholds) -> TierCap {
    if ctx == 0 {
        return TierCap {
            tier1_cap: fallback.tier1_token_cap as usize,
            tier2_cap: fallback.tier2_token_cap as usize,
        };
    }
    let budget = token_budget(ctx);
    let tier1_fraction = (TIER1_FRACTION * budget.usable_input as f32) as usize;
    let tier1_cap = tier1_fraction.min(abs_cap(ctx));
    TierCap {
        tier1_cap,
        tier2_cap: budget.usable_input,
    }
}

/// Routing token estimate: `chars/4` (spec line 37). Cheap, deliberately rough.
fn estimate_tokens(text: &str) -> usize {
    text.chars().count() / 4
}

/// Exact token count via the shared tokenizer. Used only for a near-cap recount of
/// a `chars/4` estimate (see [`estimate_within_margin`]).
fn exact_tokens(tokenizer: &Tokenizer, text: &str) -> Result<usize, LensError> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| LensError::Model(format!("tokenizer encode failed: {e}")))?;
    Ok(encoding.len())
}

/// True when a `chars/4` estimate lands within [`MARGIN`] of `cap` (either side),
/// i.e. close enough that the rough estimate could be on the wrong side of the cap
/// and an exact recount is warranted.
fn estimate_within_margin(estimate: usize, cap: usize) -> bool {
    let lo = cap.saturating_sub(MARGIN as usize);
    let hi = cap.saturating_add(MARGIN as usize);
    estimate >= lo && estimate <= hi
}

/// Tier-1 token sum over the selected+live sources. Uses each source's cached
/// `token_count`; a `None` count falls back to `chars/4` of the source text
/// (reconstructed from its parent chunks).
async fn tier1_sum(
    pool: &SqlitePool,
    sources: &[(String, Option<i64>)],
) -> Result<usize, LensError> {
    let mut sum = 0usize;
    for (source_id, token_count) in sources {
        match token_count {
            Some(n) if *n >= 0 => sum += *n as usize,
            _ => {
                let text: Option<String> = sqlx::query_scalar::<_, Option<String>>(
                    "SELECT group_concat(text, '') FROM chunks \
                     WHERE source_id = ? AND kind = ?",
                )
                .bind(source_id)
                .bind(kind::PARENT)
                .fetch_optional(pool)
                .await?
                .flatten();
                sum += estimate_tokens(text.as_deref().unwrap_or_default());
            }
        }
    }
    Ok(sum)
}

/// Resolves the selected+live source ids (with cached token counts) for a notebook,
/// using the exact predicate BM25 and `live_chunk_ids` use so all paths share one
/// corpus scope. Ordered by `created_at` for a stable document order across sources.
async fn resolve_selected_sources(
    pool: &SqlitePool,
    notebook_id: &str,
) -> Result<Vec<(String, Option<i64>)>, LensError> {
    let rows = sqlx::query_as::<_, (String, Option<i64>)>(
        "SELECT id, token_count FROM sources \
         WHERE notebook_id = ? AND selected = 1 AND trashed_at IS NULL \
         ORDER BY created_at ASC, id ASC",
    )
    .bind(notebook_id)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// A parent-chunk row in document order, used by both tier assemblies.
#[derive(Debug, Clone)]
struct ParentRow {
    chunk_id: String,
    source_id: String,
    text: String,
    locator: Option<String>,
}

/// Loads the parent chunks for a set of sources in document order (source order,
/// then `level ASC, token_start ASC`). `source_anchor` falls back to `section_path`.
async fn load_parent_rows(
    pool: &SqlitePool,
    source_ids: &[String],
) -> Result<Vec<ParentRow>, LensError> {
    let mut rows = Vec::new();
    for source_id in source_ids {
        let recs = sqlx::query_as::<_, (String, String, Option<String>, String)>(
            "SELECT id, text, source_anchor, section_path FROM chunks \
             WHERE source_id = ? AND kind = ? \
             ORDER BY level ASC, token_start ASC",
        )
        .bind(source_id)
        .bind(kind::PARENT)
        .fetch_all(pool)
        .await?;
        for (chunk_id, text, source_anchor, section_path) in recs {
            rows.push(ParentRow {
                chunk_id,
                source_id: source_id.clone(),
                text,
                locator: source_anchor.or(Some(section_path)),
            });
        }
    }
    Ok(rows)
}

/// Tier-1 raw-corpus assembly: the selected+live sources' parent chunks in document
/// order, each hydrated, `parent_id = None`, `order_index` monotonic. Non-graph
/// provenance (raw inject). No retrieval, no fusion.
async fn assemble_tier1(
    pool: &SqlitePool,
    source_ids: &[String],
) -> Result<Vec<ContextUnit>, LensError> {
    let parents = load_parent_rows(pool, source_ids).await?;
    Ok(parents
        .into_iter()
        .enumerate()
        .map(|(order_index, p)| ContextUnit {
            text: p.text,
            source_id: p.source_id,
            chunk_id: p.chunk_id,
            parent_id: None,
            locator: p.locator,
            order_index,
            provenance: Provenance {
                source: HitSource::Dense,
                graph_confidence: None,
            },
        })
        .collect())
}

/// A retrieved child chunk with its parent linkage and provenance, the input to
/// Tier-2 auto-merge.
#[derive(Debug, Clone)]
struct RetrievedChunk {
    chunk_id: String,
    source_id: String,
    parent_id: Option<String>,
    text: String,
    locator: Option<String>,
    source: HitSource,
    graph_confidence: Option<f32>,
}

/// Loads the parent/child linkage + text for a set of retrieved chunk ids,
/// preserving the input (fused rank) order. Chunks that vanished are dropped.
async fn hydrate_retrieved(
    pool: &SqlitePool,
    hits: &[RetrievalHit],
    graph_conf: &HashMap<String, f32>,
) -> Result<Vec<RetrievedChunk>, LensError> {
    let mut out = Vec::with_capacity(hits.len());
    for h in hits {
        let rec = sqlx::query_as::<_, (String, Option<String>, String, Option<String>, String)>(
            "SELECT source_id, parent_id, text, source_anchor, section_path FROM chunks \
             WHERE id = ?",
        )
        .bind(&h.chunk_id)
        .fetch_optional(pool)
        .await?;
        if let Some((source_id, parent_id, text, source_anchor, section_path)) = rec {
            out.push(RetrievedChunk {
                chunk_id: h.chunk_id.clone(),
                source_id,
                parent_id,
                text,
                locator: source_anchor.or(Some(section_path)),
                source: h.source,
                graph_confidence: graph_conf.get(&h.chunk_id).copied(),
            });
        }
    }
    Ok(out)
}

/// The number of children a parent has (`chunks.parent_id = parent_id`), used to
/// decide the ≥50% auto-merge boundary.
async fn parent_child_count(pool: &SqlitePool, parent_id: &str) -> Result<usize, LensError> {
    let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chunks WHERE parent_id = ?")
        .bind(parent_id)
        .fetch_optional(pool)
        .await?
        .unwrap_or(0);
    Ok(n.max(0) as usize)
}

/// A parent chunk's row (text + locator + source), loaded when a parent is merged.
async fn load_parent(pool: &SqlitePool, parent_id: &str) -> Result<Option<ParentRow>, LensError> {
    let rec = sqlx::query_as::<_, (String, String, Option<String>, String)>(
        "SELECT source_id, text, source_anchor, section_path FROM chunks WHERE id = ?",
    )
    .bind(parent_id)
    .fetch_optional(pool)
    .await?;
    Ok(
        rec.map(|(source_id, text, source_anchor, section_path)| ParentRow {
            chunk_id: parent_id.to_string(),
            source_id,
            text,
            locator: source_anchor.or(Some(section_path)),
        }),
    )
}

/// Document-order key for a chunk id (`(source rank, level, token_start)`), so the
/// final units re-sort to reading order regardless of retrieval rank.
async fn doc_order_key(
    pool: &SqlitePool,
    chunk_id: &str,
    source_rank: &HashMap<String, usize>,
) -> Result<(usize, i32, i64), LensError> {
    let rec = sqlx::query_as::<_, (String, i32, i64)>(
        "SELECT source_id, level, token_start FROM chunks WHERE id = ?",
    )
    .bind(chunk_id)
    .fetch_optional(pool)
    .await?;
    match rec {
        Some((source_id, level, token_start)) => {
            let rank = source_rank.get(&source_id).copied().unwrap_or(usize::MAX);
            Ok((rank, level, token_start))
        }
        None => Ok((usize::MAX, i32::MAX, i64::MAX)),
    }
}

/// Tier-2 assembly (spec/AC4): a parent is auto-merged when **≥50%** of its children
/// are in the retrieved set (2/4 merges, 1/4 does not); a merged parent replaces its
/// retrieved children; overlapping parents are deduped; survivors are trimmed to
/// `tier2_cap` (by `chars/4` estimate) and re-sorted to document order.
async fn assemble_tier2(
    pool: &SqlitePool,
    hits: &[RetrievalHit],
    graph_conf: &HashMap<String, f32>,
    source_rank: &HashMap<String, usize>,
    tier2_cap: usize,
) -> Result<Vec<ContextUnit>, LensError> {
    let retrieved = hydrate_retrieved(pool, hits, graph_conf).await?;

    // Group retrieved children by parent to evaluate the ≥50% merge boundary.
    let mut by_parent: HashMap<String, Vec<RetrievedChunk>> = HashMap::new();
    let mut orphans: Vec<RetrievedChunk> = Vec::new();
    for rc in retrieved {
        match &rc.parent_id {
            Some(pid) => by_parent.entry(pid.clone()).or_default().push(rc),
            None => orphans.push(rc),
        }
    }

    // Provenance of a merged parent unit: merge the child sources (graph membership
    // wins its own tag; otherwise the fused source of the children).
    #[derive(Clone)]
    struct Candidate {
        chunk_id: String,
        source_id: String,
        parent_id: Option<String>,
        text: String,
        locator: Option<String>,
        provenance: Provenance,
    }

    let mut candidates: Vec<Candidate> = Vec::new();
    let mut merged_parent_ids: HashSet<String> = HashSet::new();

    for (parent_id, children) in by_parent {
        let total = parent_child_count(pool, &parent_id).await?;
        let retrieved_count = children.len();
        let merge = total > 0 && retrieved_count * 2 >= total;
        if merge {
            if merged_parent_ids.insert(parent_id.clone())
                && let Some(parent) = load_parent(pool, &parent_id).await?
            {
                let provenance = merged_provenance(&children);
                candidates.push(Candidate {
                    chunk_id: parent.chunk_id,
                    source_id: parent.source_id,
                    parent_id: Some(parent_id.clone()),
                    text: parent.text,
                    locator: parent.locator,
                    provenance,
                });
            }
        } else {
            for rc in children {
                candidates.push(Candidate {
                    chunk_id: rc.chunk_id,
                    source_id: rc.source_id,
                    parent_id: rc.parent_id,
                    text: rc.text,
                    locator: rc.locator,
                    provenance: Provenance {
                        source: rc.source,
                        graph_confidence: rc.graph_confidence,
                    },
                });
            }
        }
    }
    for rc in orphans {
        candidates.push(Candidate {
            chunk_id: rc.chunk_id,
            source_id: rc.source_id,
            parent_id: rc.parent_id,
            text: rc.text,
            locator: rc.locator,
            provenance: Provenance {
                source: rc.source,
                graph_confidence: rc.graph_confidence,
            },
        });
    }

    // Re-sort survivors to document order.
    let mut keyed: Vec<((usize, i32, i64), Candidate)> = Vec::with_capacity(candidates.len());
    for c in candidates {
        let key = doc_order_key(pool, &c.chunk_id, source_rank).await?;
        keyed.push((key, c));
    }
    keyed.sort_by(|a, b| a.0.cmp(&b.0));

    // Trim to the tier-2 cap by running `chars/4` estimate.
    let mut units = Vec::new();
    let mut running = 0usize;
    for (order_index, (_, c)) in keyed.into_iter().enumerate() {
        let est = estimate_tokens(&c.text);
        if running + est > tier2_cap && !units.is_empty() {
            break;
        }
        running += est;
        units.push(ContextUnit {
            text: c.text,
            source_id: c.source_id,
            chunk_id: c.chunk_id,
            parent_id: c.parent_id,
            locator: c.locator,
            order_index,
            provenance: c.provenance,
        });
    }
    Ok(units)
}

/// Provenance for a merged-parent unit: if any child came from the graph arm, keep
/// its graph tag + max confidence; otherwise the dominant fused source of the
/// children (Both if mixed dense/bm25).
fn merged_provenance(children: &[RetrievedChunk]) -> Provenance {
    let graph_conf = children
        .iter()
        .filter(|c| c.source == HitSource::Graph)
        .filter_map(|c| c.graph_confidence)
        .fold(None::<f32>, |acc, x| Some(acc.map_or(x, |a| a.max(x))));
    if let Some(conf) = graph_conf {
        return Provenance {
            source: HitSource::Graph,
            graph_confidence: Some(conf),
        };
    }
    let mut has_dense = false;
    let mut has_bm25 = false;
    for c in children {
        match c.source {
            HitSource::Dense => has_dense = true,
            HitSource::Bm25 => has_bm25 = true,
            HitSource::Both => {
                has_dense = true;
                has_bm25 = true;
            }
            HitSource::Graph => {}
        }
    }
    let source = match (has_dense, has_bm25) {
        (true, true) => HitSource::Both,
        (false, true) => HitSource::Bm25,
        _ => HitSource::Dense,
    };
    Provenance {
        source,
        graph_confidence: None,
    }
}

/// Small stop-word set for the relational-predicate signal. Deterministic and
/// LLM-free — the signal only needs to distinguish a bare entity lookup from a
/// query carrying relational intent, so a tiny function-word list suffices.
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "of", "to", "in", "on", "at", "is", "are", "was", "were", "and", "or", "for",
    "with", "how", "what", "who", "when", "where", "why", "does", "do", "did", "by", "from",
    "about", "between",
];

/// Max token width of a candidate seed span. Entity names are rarely wider; caps
/// the per-query `entity_lookup` fan-out (mirrors the graph module's cap discipline).
const MAX_SEED_SPAN: usize = 5;

/// Tokenizes on whitespace/punctuation, lowercased.
fn content_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// The deterministic graph gate outcome: the resolved seeds plus whether the
/// relational-predicate signal fires.
struct SeedResolution {
    seeds: Vec<(String, EntityKind)>,
    signal: bool,
}

/// Deterministic, LLM-free seed resolution (spec §5.2, corrected). `entity_lookup`
/// matches the WHOLE query string against entity names, so passing a multi-word
/// query never yields seeds; instead we tokenize and look up per contiguous span
/// (longest-first, capped at [`MAX_SEED_SPAN`]), claiming matched token positions.
/// The relational signal fires when `>= 2` non-stop-word tokens remain unclaimed by
/// any seed name — the "content tokens beyond the matched seed name(s)" the plan's
/// worked examples assume.
async fn resolve_seeds(
    pool: &SqlitePool,
    notebook_id: &str,
    query: &str,
) -> Result<SeedResolution, LensError> {
    let tokens = content_tokens(query);
    let mut claimed = vec![false; tokens.len()];
    let mut seeds: Vec<(String, EntityKind)> = Vec::new();
    let mut seen: HashSet<(String, EntityKind)> = HashSet::new();

    // Longest-first spans so a multi-word entity name is matched before its parts.
    for width in (1..=MAX_SEED_SPAN.min(tokens.len())).rev() {
        for start in 0..=tokens.len().saturating_sub(width) {
            let end = start + width;
            if claimed[start..end].iter().any(|c| *c) {
                continue;
            }
            let span = tokens[start..end].join(" ");
            let hits = entity_lookup(pool, notebook_id, &span, 1).await?;
            let Some(entity) = hits.into_iter().next() else {
                continue;
            };
            // Only claim the span when the matched name's tokens are covered by it,
            // so a loose substring hit does not swallow unrelated query words.
            let name_tokens = content_tokens(&entity.name);
            let span_set: HashSet<&String> = tokens[start..end].iter().collect();
            if !name_tokens.iter().all(|t| span_set.contains(t)) {
                continue;
            }
            for c in &mut claimed[start..end] {
                *c = true;
            }
            let key = (entity.name.to_lowercase(), entity.kind);
            if seen.insert(key) {
                seeds.push((entity.name, entity.kind));
            }
        }
    }

    let residual = tokens
        .iter()
        .enumerate()
        .filter(|(i, _)| !claimed[*i])
        .filter(|(_, t)| !STOP_WORDS.contains(&t.as_str()))
        .count();

    Ok(SeedResolution {
        seeds,
        signal: residual >= 2,
    })
}

/// Resolves the effective per-notebook graph-retrieval opt-in: the per-notebook
/// override wins (`Some(true)` on, `Some(false)` off); `None` inherits the app-wide
/// `RetrievalConfig::graph_retrieval_enabled`.
fn effective_graph_flag(notebook_flag: Option<bool>, app_wide: bool) -> bool {
    notebook_flag.unwrap_or(app_wide)
}

/// Deterministic, LLM-free graph gate (spec §5.2). Fires only when ALL hold:
/// (a) >= 1 resolved seed, (b) the effective flag is on, (c) the relational signal
/// fires. Graph is additive — it only enables the third fusion list.
fn should_run_graph(resolution: &SeedResolution, effective_flag: bool) -> bool {
    !resolution.seeds.is_empty() && effective_flag && resolution.signal
}

/// Pair-preserving graph composition extracted from `eval::graph_arm` (spec §5.1):
/// `entity_evidence` (fixed conf 1.0) + `ppr_expand` (traversal conf), merged with
/// the LOCKED merge invariant (dedup keeps max conf; evidence precedes equal-conf
/// expansion; truncate to `k`) but returning `(chunk_id, graph_confidence)` pairs so
/// `Provenance.graph_confidence` survives. `eval::graph_arm` maps these back to
/// `Vec<String>`.
pub(crate) async fn graph_compose(
    pool: &SqlitePool,
    graph: &NotebookGraph,
    seeds: &[(String, EntityKind)],
    k: usize,
) -> Result<Vec<(String, f32)>, LensError> {
    let notebook_id = graph.notebook_id().to_string();
    let mut evidence = Vec::new();
    for (name, kind) in seeds {
        evidence.extend(entity_evidence(pool, &notebook_id, name, *kind, k).await?);
    }
    let expansion: Vec<(String, f32)> = ppr_expand(pool, graph, seeds, k)
        .await?
        .into_iter()
        .flat_map(|hit| {
            hit.chunk_ids
                .into_iter()
                .map(move |c| (c, hit.graph_confidence))
        })
        .collect();
    Ok(merge_ranked_pairs(evidence, expansion, k))
}

/// The pair-preserving merge (the source of `eval::merge_ranked`'s ordering): dedup
/// by chunk id keeping the highest confidence; stable sort conf DESC, first-seen
/// tie-break (evidence at fixed 1.0 precedes equal-conf expansion); truncate to `k`.
/// Returns `(chunk_id, best_conf)` pairs.
fn merge_ranked_pairs(
    evidence: Vec<String>,
    expansion: Vec<(String, f32)>,
    k: usize,
) -> Vec<(String, f32)> {
    let mut best: HashMap<String, (f32, usize)> = HashMap::new();
    let mut seen = 0usize;
    let mut insert = |best: &mut HashMap<String, (f32, usize)>, chunk_id: String, conf: f32| {
        best.entry(chunk_id)
            .and_modify(|(c, _)| *c = c.max(conf))
            .or_insert_with(|| {
                let ord = seen;
                seen += 1;
                (conf, ord)
            });
    };
    for chunk_id in evidence {
        insert(&mut best, chunk_id, 1.0);
    }
    for (chunk_id, conf) in expansion {
        insert(&mut best, chunk_id, conf);
    }
    let mut ranked: Vec<(String, f32, usize)> =
        best.into_iter().map(|(id, (c, o))| (id, c, o)).collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.2.cmp(&b.2))
    });
    ranked.truncate(k);
    ranked.into_iter().map(|(id, c, _)| (id, c)).collect()
}

/// The Tiered Context Router entry point (issue #21): decides Tier-1 vs Tier-2,
/// applies the #39 dense pre-filter, optionally folds in a deterministic graph arm,
/// and returns a budgeted, document-ordered, hydrated [`RouterOutput`]. Performs NO
/// LLM call and NO prompt assembly.
#[allow(clippy::too_many_arguments)]
pub async fn tiered_search(
    pool_db: &SqlitePool,
    store: &dyn VectorStore,
    reranker: &Reranker,
    graph: Option<&NotebookGraph>,
    coord: &Coordinate,
    query_text: &str,
    query_vec: &[f32],
    model: &ModelConfig,
    pool: usize,
    retrieval: &RetrievalConfig,
    notebook_graph_flag: Option<bool>,
    thresholds: &TierThresholds,
    tokenizer: Option<&Tokenizer>,
) -> Result<RouterOutput, LensError> {
    let caps = derive_tier_caps(model.context, thresholds);
    let sources = resolve_selected_sources(pool_db, &coord.notebook).await?;

    // Empty selected+live set → ground on nothing (do NOT fall back to notebook
    // scope, which would leak deselected/trashed chunks — the #39 bug inverted).
    if sources.is_empty() {
        return Ok(RouterOutput {
            tier: Tier::Tier2,
            units: Vec::new(),
            total_tokens: 0,
        });
    }

    let source_ids: Vec<String> = sources.iter().map(|(id, _)| id.clone()).collect();
    let source_rank: HashMap<String, usize> = source_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.clone(), i))
        .collect();

    // Tier selection: Σ cached token_count vs the derived tier-1 cap. A single
    // oversized source (its own count > cap) forces Tier-2.
    let total_sum = tier1_sum(pool_db, &sources).await?;
    let any_oversized = sources
        .iter()
        .any(|(_, tc)| tc.map(|n| n as usize > caps.tier1_cap).unwrap_or(false));

    if total_sum <= caps.tier1_cap && !any_oversized {
        // Near-cap recount: if the rough sum lands within MARGIN of the cap and a
        // tokenizer is available, an exact recount decides the boundary precisely.
        let confirmed = if estimate_within_margin(total_sum, caps.tier1_cap) {
            match tokenizer {
                Some(tk) => exact_tier1_sum(pool_db, &source_ids, tk).await? <= caps.tier1_cap,
                None => true,
            }
        } else {
            true
        };
        if confirmed {
            let units = assemble_tier1(pool_db, &source_ids).await?;
            let total_tokens = units.iter().map(|u| estimate_tokens(&u.text)).sum();
            return Ok(RouterOutput {
                tier: Tier::Tier1,
                units,
                total_tokens,
            });
        }
    }

    // Tier-2: fused retrieval (dense pre-filter + bm25 + optional graph) → assembly.
    // Over-fetch per path IDENTICALLY to hybrid_search so the fusion seam matches.
    let overfetch = pool.clamp(OVERFETCH, MAX_OVERFETCH);

    // DENSE: pre-filter to selected+live BEFORE top-N (#39 fix) when the id-set is
    // small; above the guard, notebook-scope search + the live_chunk_ids post-filter
    // (both correct; the fallback is just slower).
    let use_prefilter = source_ids.len() <= MAX_PREFILTER_IDS;
    let dense_hits = if use_prefilter {
        store
            .search_filtered(coord, query_vec, overfetch, &source_ids)
            .await?
    } else {
        store.search(coord, query_vec, overfetch).await?
    };
    let dense_ids: Vec<String> = dense_hits.into_iter().map(|h| h.chunk_id).collect();
    // Correctness backstop regardless of store impl (the default trait body cannot
    // narrow by source): guarantee selected+live via the SQLite post-filter too.
    let dense_ids = live_chunk_ids(pool_db, &dense_ids, None, None).await?;

    // BM25: same selected+live scope (its sources JOIN already enforces it).
    let bm25_ids = if retrieval.hybrid_enabled {
        bm25::bm25_search(pool_db, &coord.notebook, None, None, query_text, overfetch).await?
    } else {
        Vec::new()
    };

    // GRAPH (additive third list), gated deterministically.
    let mut graph_conf: HashMap<String, f32> = HashMap::new();
    let graph_ids: Vec<String> = match graph {
        Some(g) => {
            let resolution = resolve_seeds(pool_db, &coord.notebook, query_text).await?;
            let flag = effective_graph_flag(notebook_graph_flag, retrieval.graph_retrieval_enabled);
            if should_run_graph(&resolution, flag) {
                let pairs = graph_compose(pool_db, g, &resolution.seeds, overfetch).await?;
                let mut ids = Vec::with_capacity(pairs.len());
                for (id, conf) in pairs {
                    graph_conf.entry(id.clone()).or_insert(conf);
                    ids.push(id);
                }
                // Restrict graph hits to the selected+live corpus too.
                live_chunk_ids(pool_db, &ids, None, None).await?
            } else {
                Vec::new()
            }
        }
        None => Vec::new(),
    };

    let fused = super::fuse_and_rerank(
        pool_db, reranker, query_text, &dense_ids, &bm25_ids, &graph_ids, pool, retrieval,
    )
    .await?;

    let units = assemble_tier2(pool_db, &fused, &graph_conf, &source_rank, caps.tier2_cap).await?;
    let total_tokens = units.iter().map(|u| estimate_tokens(&u.text)).sum();
    Ok(RouterOutput {
        tier: Tier::Tier2,
        units,
        total_tokens,
    })
}

/// Exact-recount Tier-1 sum used only near the cap boundary: re-tokenizes each
/// source's reconstructed parent text with the real tokenizer.
async fn exact_tier1_sum(
    pool: &SqlitePool,
    source_ids: &[String],
    tokenizer: &Tokenizer,
) -> Result<usize, LensError> {
    let mut sum = 0usize;
    for source_id in source_ids {
        let text: Option<String> = sqlx::query_scalar::<_, Option<String>>(
            "SELECT group_concat(text, '') FROM chunks WHERE source_id = ? AND kind = ?",
        )
        .bind(source_id)
        .bind(kind::PARENT)
        .fetch_optional(pool)
        .await?
        .flatten();
        sum += exact_tokens(tokenizer, text.as_deref().unwrap_or_default())?;
    }
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thresholds() -> TierThresholds {
        TierThresholds::default()
    }

    #[test]
    fn cap_small_band_is_bounded_by_4000() {
        let caps = derive_tier_caps(8_192, &thresholds());
        assert!(caps.tier1_cap <= ABS_CAP_SMALL, "{}", caps.tier1_cap);
    }

    #[test]
    fn cap_medium_band_is_bounded_by_20000() {
        let caps = derive_tier_caps(32_000, &thresholds());
        assert!(caps.tier1_cap <= ABS_CAP_MEDIUM, "{}", caps.tier1_cap);
    }

    #[test]
    fn cap_large_band_is_bounded_by_48000() {
        let caps = derive_tier_caps(200_000, &thresholds());
        assert!(caps.tier1_cap <= ABS_CAP_LARGE, "{}", caps.tier1_cap);
        // 0.65 * usable dominates below the abs cap for a very large window? No —
        // 0.65 * (200000-2560) ≈ 128k, so the abs cap binds.
        assert_eq!(caps.tier1_cap, ABS_CAP_LARGE);
    }

    #[test]
    fn cap_fraction_binds_when_below_abs_cap() {
        // ctx 10000 -> usable 7440 -> 0.65*7440 = 4836, abs cap (medium band) 20000.
        // The fraction binds.
        let caps = derive_tier_caps(10_000, &thresholds());
        let budget = token_budget(10_000);
        let expected = (TIER1_FRACTION * budget.usable_input as f32) as usize;
        assert_eq!(caps.tier1_cap, expected);
        assert!(caps.tier1_cap < ABS_CAP_MEDIUM);
    }

    #[test]
    fn context_zero_falls_back_to_tier_thresholds() {
        let caps = derive_tier_caps(0, &thresholds());
        assert_eq!(caps.tier1_cap, 4_000);
        assert_eq!(caps.tier2_cap, 16_000);
    }

    #[test]
    fn token_budget_subtracts_reserved_and_overhead() {
        let b = token_budget(10_000);
        assert_eq!(b.usable_input, 10_000 - 2_048 - 512);
        assert_eq!(b.reserved_output, 2_048);
        assert_eq!(b.system_overhead, 512);
    }

    #[test]
    fn token_budget_saturates_at_zero_for_tiny_context() {
        let b = token_budget(100);
        assert_eq!(b.usable_input, 0);
    }

    #[test]
    fn estimate_tokens_is_chars_over_4() {
        assert_eq!(estimate_tokens("abcdefgh"), 2);
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abc"), 0);
    }

    #[test]
    fn estimate_within_margin_detects_near_cap() {
        // cap 4000, margin 256 -> band [3744, 4256].
        assert!(estimate_within_margin(4_000, 4_000));
        assert!(estimate_within_margin(3_800, 4_000));
        assert!(estimate_within_margin(4_200, 4_000));
        assert!(!estimate_within_margin(3_000, 4_000));
        assert!(!estimate_within_margin(5_000, 4_000));
    }

    /// Model-gated: exercises the exact-recount path against the real nomic
    /// tokenizer. Skipped offline (unset `NOMIC_TOKENIZER_PATH`) so the suite stays
    /// runnable without model weights (mirrors `chunk.rs`'s `load_tokenizer`).
    #[test]
    fn exact_tokens_matches_encoded_length() {
        let Some(path) = std::env::var("NOMIC_TOKENIZER_PATH").ok() else {
            return;
        };
        let Ok(tokenizer) = Tokenizer::from_file(&path) else {
            return;
        };
        let text = "The quick brown fox jumps over the lazy dog.";
        let n = exact_tokens(&tokenizer, text).expect("encode");
        let expected = tokenizer.encode(text, false).expect("encode").len();
        assert_eq!(n, expected);
        assert!(n > 0);
    }

    // --- graph composition merge (ported from eval::merge_ranked tests) ---

    #[test]
    fn merge_ranks_evidence_before_equal_confidence_expansion() {
        // Evidence chunk "e" (conf 1.0) must outrank an expansion chunk "x" even
        // when expansion confidence also normalizes to 1.0 (first-seen tie-break).
        let out = merge_ranked_pairs(
            vec!["e".into()],
            vec![("x".into(), 1.0), ("y".into(), 0.4)],
            5,
        );
        let ids: Vec<&str> = out.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, vec!["e", "x", "y"]);
        assert_eq!(out[0].1, 1.0);
    }

    #[test]
    fn merge_dedups_keeping_highest_confidence_and_truncates() {
        // "d" appears as both evidence (1.0) and low-conf expansion (0.2): one
        // entry at conf 1.0. Expansion sorts by confidence DESC. Truncate to k=2.
        let out = merge_ranked_pairs(
            vec!["d".into()],
            vec![("d".into(), 0.2), ("hi".into(), 0.9), ("lo".into(), 0.1)],
            2,
        );
        let ids: Vec<&str> = out.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, vec!["d", "hi"]);
        assert_eq!(out[0].1, 1.0, "d keeps evidence conf 1.0, not 0.2");
    }

    // --- graph gate ---

    fn seed(name: &str) -> (String, EntityKind) {
        (name.to_string(), EntityKind::Concept)
    }

    fn resolution(seeds: Vec<(String, EntityKind)>, signal: bool) -> SeedResolution {
        SeedResolution { seeds, signal }
    }

    #[test]
    fn effective_flag_override_wins_over_app_wide() {
        assert!(effective_graph_flag(Some(true), false));
        assert!(!effective_graph_flag(Some(false), true));
        assert!(effective_graph_flag(None, true));
        assert!(!effective_graph_flag(None, false));
    }

    #[test]
    fn should_run_graph_requires_seeds_flag_and_signal() {
        let with = || resolution(vec![seed("Acme")], true);
        assert!(should_run_graph(&with(), true));
        // flag off
        assert!(!should_run_graph(&with(), false));
        // no seeds
        assert!(!should_run_graph(&resolution(vec![], true), true));
        // seeds but no relational signal (bare-name query)
        assert!(!should_run_graph(
            &resolution(vec![seed("Acme")], false),
            true
        ));
    }

    // --- merged provenance ---

    fn rc(source: HitSource, conf: Option<f32>) -> RetrievedChunk {
        RetrievedChunk {
            chunk_id: "c".into(),
            source_id: "s".into(),
            parent_id: Some("p".into()),
            text: "t".into(),
            locator: None,
            source,
            graph_confidence: conf,
        }
    }

    #[test]
    fn merged_provenance_graph_wins_with_max_conf() {
        let children = vec![
            rc(HitSource::Dense, None),
            rc(HitSource::Graph, Some(0.4)),
            rc(HitSource::Graph, Some(0.7)),
        ];
        let p = merged_provenance(&children);
        assert_eq!(p.source, HitSource::Graph);
        assert_eq!(p.graph_confidence, Some(0.7));
    }

    #[test]
    fn merged_provenance_mixes_dense_and_bm25_to_both() {
        let children = vec![rc(HitSource::Dense, None), rc(HitSource::Bm25, None)];
        let p = merged_provenance(&children);
        assert_eq!(p.source, HitSource::Both);
        assert_eq!(p.graph_confidence, None);
    }
}
