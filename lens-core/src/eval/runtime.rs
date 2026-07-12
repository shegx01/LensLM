//! M13 #158a runtime per-notebook eval harness (gated behind `LENS_RUN_MODEL_TESTS`
//! for the networked path; the plumbing is exercised offline with a mock LLM). Never
//! mutates `graph_retrieval_enabled`. Provenance-gold rationale: see [`super`].

use std::collections::HashSet;
use std::time::Instant;

use serde::Deserialize;
use sqlx::SqlitePool;

use super::{graph_arm, hybrid_arm, mean, recall_at_k};
use crate::LensError;
use crate::config::RetrievalConfig;
use crate::embedder::Embedder;
use crate::enrichment::meta::{Budget, SessionBudget};
use crate::graph::{EntityKind, NotebookGraph};
use crate::llm::LlmProvider;
use crate::retrieval::Reranker;
use crate::vector_store::{Coordinate, VectorStore};

/// Recall cutoff and the retrieval-arm k for the runtime harness.
const EVAL_K: usize = 5;

/// Minimum live sources / chunks a notebook must have for an eval to be meaningful
/// (R6). Below either floor the harness returns `Skipped` rather than log noise.
const MIN_SOURCES: i64 = 3;
const MIN_CHUNKS: i64 = 50;

/// Promotion bar: graph must beat hybrid by ≥5pp on the bridging+rollup subset and
/// stay under the latency budget. Recorded on the log row; never flips the flag.
const MIN_DELTA_PP: f32 = 5.0;
const MAX_P95_MS: f32 = 500.0;

/// Target per-kind question count for the LLM QA-gen pass (20–30 total).
const TARGET_PER_KIND: usize = 10;

/// Per-job LLM ceilings for the QA-gen pass. Generous single-call budget: one JSON
/// array of ~30 questions with gold, with retry headroom for reprompt-on-malformed.
const QA_MAX_TOKENS: u32 = 4096;
const QA_MAX_TOKENS_PER_JOB: u32 = 20_000;
const QA_MAX_CALLS_PER_JOB: u32 = 8;

/// Max entity-dense raw chunks fed to the QA-gen prompt (bounds the input budget).
const QA_MAX_CONTEXT_CHUNKS: usize = 40;

/// QA-gen prompt lineage. Bump on any prompt/contract change so `eval_questions`
/// and `notebook_eval_log` rows carry the version they were generated under.
pub const QA_PROMPT_VERSION: &str = "158a-qa-v2";

const QA_SYSTEM_PROMPT: &str = "You generate an evaluation set of questions grounded in a \
document corpus. Respond with ONLY a JSON array (no prose, no markdown fences). Each element \
is an object with EXACTLY these keys: \"kind\" (one of \"single_hop\", \"bridging\", \
\"rollup\"), \"question\" (string), \"seed_entities\" (array of {\"name\": string, \"kind\": \
string}), \"gold_chunk_ids\" (array of strings — the EXACT chunk ids, taken verbatim from the \
prefix lines, that contain the answer to the question). A \"single_hop\" question is \
answerable from one entity in one place (the control). A \"bridging\" question is answerable \
ONLY by combining information about the named entities across sources. A \"rollup\" question \
requires aggregating information about the named entities across the whole corpus. \
seed_entities MUST be the entities the question is grounded in. gold_chunk_ids MUST be copied \
verbatim from the [chunk_id: …] prefix lines in the corpus — do not invent ids. Aim for \
about 10 of each kind.";

/// One question kind. Rust-enum-validated; stored as lowercase snake_case TEXT
/// (no SQL CHECK, per repo convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionKind {
    SingleHop,
    Bridging,
    Rollup,
}

impl QuestionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            QuestionKind::SingleHop => "single_hop",
            QuestionKind::Bridging => "bridging",
            QuestionKind::Rollup => "rollup",
        }
    }

    pub fn from_db(s: &str) -> Result<Self, LensError> {
        match s {
            "single_hop" => Ok(QuestionKind::SingleHop),
            "bridging" => Ok(QuestionKind::Bridging),
            "rollup" => Ok(QuestionKind::Rollup),
            _ => Err(LensError::Parse(format!("unknown question kind: {s}"))),
        }
    }

    /// Bridging + rollup are the scored subset for `delta_pp`; single_hop is the
    /// advisory control.
    fn is_scored(self) -> bool {
        !matches!(self, QuestionKind::SingleHop)
    }
}

/// A seed entity a question is grounded in. `kind` is validated against
/// [`EntityKind::from_db`] at parse time; a seed whose `(name, kind)` is absent from
/// the notebook graph is dropped before scoring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedEntity {
    pub name: String,
    pub kind: EntityKind,
}

/// A generated, graph-resolved question ready to score.
#[derive(Debug, Clone)]
struct EvalQuestion {
    kind: QuestionKind,
    question: String,
    seeds: Vec<SeedEntity>,
    /// Provenance gold: LLM-emitted chunk ids validated against the fed corpus.
    gold_chunk_ids: Vec<String>,
}

/// Outcome of a runtime eval attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum EvalOutcome {
    /// The notebook did not meet the sample floor; nothing was written.
    Skipped { reason: String },
    /// The eval ran and logged one `notebook_eval_log` row.
    Ran(EvalReport),
}

/// The latest logged eval for a notebook: the `EvalReport` plus the `ran_at`
/// timestamp of the row it was read from (`EvalReport` itself carries no `ran_at`).
#[derive(Debug, Clone, PartialEq)]
pub struct LatestEval {
    pub report: EvalReport,
    pub ran_at: String,
}

/// Coarse progress phase for a user-triggered `run_graph_eval`. Only these two
/// phases can fire: `run_notebook_eval` is monolithic with no internal progress
/// hook, so the caller emits `GeneratingQa` immediately before the run and `Done`
/// immediately after. Per-arm progress is a deferred change (would require threading
/// a callback into `run_notebook_eval`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalPhase {
    GeneratingQa,
    Done,
}

/// The ablation summary logged for one eval run.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalReport {
    pub graph_recall: f32,
    pub hybrid_recall: f32,
    pub delta_pp: f32,
    pub p95_ms: f32,
    pub passed: bool,
    pub sample_n: usize,
    /// Questions dropped for zero valid provenance gold (all emitted ids were
    /// hallucinated or not live), or for zero graph-resolvable seeds.
    pub dropped_n: usize,
    pub graph_enabled: bool,
    pub prompt_version: String,
}

/// Dependencies for one runtime eval run. `graph_enabled` is the caller-resolved
/// effective flag snapshot — recorded as context only, NEVER mutated here.
pub struct RunEvalDeps<'a> {
    pub pool: &'a SqlitePool,
    pub store: &'a dyn VectorStore,
    pub reranker: &'a Reranker,
    pub coord: &'a Coordinate,
    pub embedder: &'a dyn Embedder,
    pub llm: &'a dyn LlmProvider,
    pub config: &'a RetrievalConfig,
    pub graph_enabled: bool,
}

/// Raw QA element as emitted by the LLM (pre-validation).
#[derive(Debug, Deserialize)]
struct RawQuestion {
    kind: String,
    question: String,
    seed_entities: Vec<RawSeed>,
    gold_chunk_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawSeed {
    name: String,
    kind: String,
}

/// The validated, pre-graph-resolution form of one question.
#[derive(Debug)]
struct RawParsed {
    kind: QuestionKind,
    question: String,
    seeds: Vec<SeedEntity>,
    /// Gold ids that appeared in `fed_ids` (not yet live-checked).
    gold_chunk_ids: Vec<String>,
}

fn parse_qa_inner(json: &str, fed_ids: &HashSet<String>) -> Result<Vec<RawParsed>, LensError> {
    let raw: Vec<RawQuestion> = serde_json::from_str(json)?;
    let mut out = Vec::with_capacity(raw.len());
    for q in raw {
        let kind = QuestionKind::from_db(&q.kind)?;
        let mut seeds = Vec::with_capacity(q.seed_entities.len());
        for s in q.seed_entities {
            seeds.push(SeedEntity {
                name: s.name,
                kind: EntityKind::from_db(&s.kind)?,
            });
        }
        // Reject any hallucinated gold id. This causes a parse failure so the
        // retry loop re-prompts with the error appended.
        for id in &q.gold_chunk_ids {
            if !fed_ids.contains(id) {
                return Err(LensError::Parse(format!(
                    "gold_chunk_id {id:?} was not in the fed corpus"
                )));
            }
        }
        out.push(RawParsed {
            kind,
            question: q.question,
            seeds,
            gold_chunk_ids: q.gold_chunk_ids,
        });
    }
    Ok(out)
}

/// Runs the per-notebook eval: floor → QA-gen (with provenance gold) → validate +
/// drop → measure both arms → log. Returns `Skipped` below the sample floor;
/// otherwise writes exactly one `notebook_eval_log` row and the accepted
/// `eval_questions` rows, and returns the `EvalReport`.
/// Never writes `notebooks.graph_retrieval_enabled`.
pub async fn run_notebook_eval(
    deps: &RunEvalDeps<'_>,
    notebook_id: &str,
) -> Result<EvalOutcome, LensError> {
    // (a) FLOOR — ≥3 live sources AND ≥50 live chunks.
    let source_count = live_source_count(deps.pool, notebook_id).await?;
    if source_count < MIN_SOURCES {
        return Ok(EvalOutcome::Skipped {
            reason: format!("only {source_count} live sources (need ≥{MIN_SOURCES})"),
        });
    }
    let chunk_count = live_chunk_count(deps.pool, notebook_id).await?;
    if chunk_count < MIN_CHUNKS {
        return Ok(EvalOutcome::Skipped {
            reason: format!("only {chunk_count} live chunks (need ≥{MIN_CHUNKS})"),
        });
    }

    // (b) QA-GEN over entity-dense raw chunks, fed WITH their ids so the LLM can
    // emit provenance gold_chunk_ids verbatim from the corpus.
    let (context, fed_ids) = entity_dense_context(deps.pool, notebook_id).await?;
    let raw_parsed = generate_qa(deps.llm, &context, &fed_ids).await?;

    // Resolve seeds against the graph; drop seeds absent from it and questions left
    // with zero resolvable seeds.
    let graph = NotebookGraph::load(deps.pool, notebook_id).await?;
    let mut dropped_n = 0usize;
    let mut questions: Vec<EvalQuestion> = Vec::new();
    for rp in raw_parsed {
        let mut resolved_seeds: Vec<SeedEntity> = Vec::new();
        for s in rp.seeds {
            if seed_in_graph(deps.pool, notebook_id, &s.name, s.kind).await? {
                resolved_seeds.push(s);
            }
        }
        if resolved_seeds.is_empty() {
            tracing::debug!(question = %rp.question, "eval: dropping — no graph-resolvable seeds");
            dropped_n += 1;
            continue;
        }
        // Validate gold ids are live (live-source scope, same as retrieval). Drop
        // those that are not; drop the whole question if none remain.
        let mut valid_gold: Vec<String> = Vec::new();
        for id in rp.gold_chunk_ids {
            if live_chunk_id(deps.pool, &id).await? {
                valid_gold.push(id);
            }
        }
        if valid_gold.is_empty() {
            tracing::debug!(question = %rp.question, "eval: dropping — zero valid provenance gold");
            dropped_n += 1;
            continue;
        }
        questions.push(EvalQuestion {
            kind: rp.kind,
            question: rp.question,
            seeds: resolved_seeds,
            gold_chunk_ids: valid_gold,
        });
    }

    // Persist accepted rows (question + seeds + provenance gold).
    for q in &questions {
        persist_question(deps.pool, notebook_id, q).await?;
    }

    // (c) MEASURE — recall@5 per arm, timing RETRIEVAL wall-time only (excl. embed).
    // Both arms are scored against the same independent provenance gold.
    let mut graph_recalls: Vec<f32> = Vec::with_capacity(questions.len());
    let mut hybrid_recalls: Vec<f32> = Vec::with_capacity(questions.len());
    let mut scored_graph: Vec<f32> = Vec::new();
    let mut scored_hybrid: Vec<f32> = Vec::new();
    let mut latencies_ms: Vec<f32> = Vec::with_capacity(questions.len());

    for q in &questions {
        let query_vec = deps.embedder.embed_query(&q.question)?;
        let seed_pairs: Vec<(String, EntityKind)> =
            q.seeds.iter().map(|s| (s.name.clone(), s.kind)).collect();

        let started = Instant::now();
        let graph_hits = graph_arm(deps.pool, &graph, &seed_pairs, EVAL_K).await?;
        let hybrid_hits = hybrid_arm(
            deps.pool,
            deps.store,
            deps.reranker,
            deps.coord,
            &q.question,
            &query_vec,
            EVAL_K,
            deps.config,
        )
        .await?;
        // p95 covers retrieval wall-time only; query embedding is excluded (it is
        // shared by both arms and hardware-bound, not a graph-tool signal).
        latencies_ms.push(started.elapsed().as_secs_f32() * 1000.0);

        let g = recall_at_k(&graph_hits, &q.gold_chunk_ids, EVAL_K);
        let h = recall_at_k(&hybrid_hits, &q.gold_chunk_ids, EVAL_K);
        graph_recalls.push(g);
        hybrid_recalls.push(h);
        if q.kind.is_scored() {
            scored_graph.push(g);
            scored_hybrid.push(h);
        }
    }

    let graph_recall = mean(&graph_recalls);
    let hybrid_recall = mean(&hybrid_recalls);
    // delta_pp on the bridging+rollup subset only (single_hop is advisory control).
    let delta_pp = (mean(&scored_graph) - mean(&scored_hybrid)) * 100.0;
    // ~20-30 samples → this p95 is effectively a max-ish latency guard.
    let p95_ms = percentile95(&latencies_ms);
    let passed = delta_pp >= MIN_DELTA_PP && p95_ms < MAX_P95_MS;

    let report = EvalReport {
        graph_recall,
        hybrid_recall,
        delta_pp,
        p95_ms,
        passed,
        sample_n: questions.len(),
        dropped_n,
        graph_enabled: deps.graph_enabled,
        prompt_version: QA_PROMPT_VERSION.to_string(),
    };

    // (d) LOG one observational row. Never touches the flag.
    persist_log(deps.pool, notebook_id, &report).await?;

    Ok(EvalOutcome::Ran(report))
}

/// Drives the QA-gen LLM call with its OWN fresh budget, mapping the enrichment
/// retry/parse loop's error onto [`LensError`]. `fed_ids` is the set of chunk ids
/// shown in the prompt; the parse closure rejects any gold id not in that set.
async fn generate_qa(
    llm: &dyn LlmProvider,
    context: &str,
    fed_ids: &HashSet<String>,
) -> Result<Vec<RawParsed>, LensError> {
    let mut budget = Budget::with_caps(
        SessionBudget::with_max_tokens(QA_MAX_TOKENS_PER_JOB),
        QA_MAX_TOKENS_PER_JOB,
        QA_MAX_CALLS_PER_JOB,
    );
    let user_prompt = format!(
        "Generate about {TARGET_PER_KIND} single_hop, {TARGET_PER_KIND} bridging, and \
         {TARGET_PER_KIND} rollup questions grounded in the following source excerpts. \
         Each [chunk_id: …] prefix is the exact id to use in gold_chunk_ids.\n\n{context}"
    );
    // The parse closure captures `fed_ids` and validates gold ids on each attempt.
    let parse_fn = |json: &str| parse_qa_inner(json, fed_ids);
    let parsed = crate::enrichment::map::run_llm_with_retries(
        llm,
        &mut budget,
        QA_SYSTEM_PROMPT,
        &user_prompt,
        QA_MAX_TOKENS,
        parse_fn,
    )
    .await
    .map_err(map_err_to_lens)?;
    Ok(parsed.unwrap_or_default())
}

/// Maps the enrichment `MapError` onto [`LensError`] (a budget breach is treated as
/// a model-side failure — QA-gen yields nothing rather than erroring).
fn map_err_to_lens(err: crate::enrichment::MapError) -> LensError {
    match err {
        crate::enrichment::MapError::Llm(e) => e,
        crate::enrichment::MapError::BudgetExceeded => {
            LensError::Model("QA-gen budget exceeded".to_string())
        }
    }
}

/// Live source count (`trashed_at IS NULL`) for the notebook.
async fn live_source_count(pool: &SqlitePool, notebook_id: &str) -> Result<i64, LensError> {
    let n: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sources WHERE notebook_id = ? AND trashed_at IS NULL",
    )
    .bind(notebook_id)
    .fetch_one(pool)
    .await?;
    Ok(n)
}

/// Live chunk count (same scope as retrieval: `trashed_at IS NULL AND selected = 1`).
async fn live_chunk_count(pool: &SqlitePool, notebook_id: &str) -> Result<i64, LensError> {
    let n: i64 = sqlx::query_scalar(
        "SELECT COUNT(DISTINCT c.id) FROM chunks c JOIN sources s ON c.source_id = s.id \
         WHERE s.notebook_id = ? AND s.trashed_at IS NULL AND s.selected = 1",
    )
    .bind(notebook_id)
    .fetch_one(pool)
    .await?;
    Ok(n)
}

/// Whether a single chunk id belongs to a live, selected source.
async fn live_chunk_id(pool: &SqlitePool, chunk_id: &str) -> Result<bool, LensError> {
    let hit: Option<i64> = sqlx::query_scalar(
        "SELECT 1 FROM chunks c JOIN sources s ON s.id = c.source_id \
         WHERE c.id = ? AND s.trashed_at IS NULL AND s.selected = 1 \
         LIMIT 1",
    )
    .bind(chunk_id)
    .fetch_optional(pool)
    .await?;
    Ok(hit.is_some())
}

/// Whether `(name, kind)` resolves to a live-source entity node in the notebook
/// graph. Matches [`NotebookGraph::load`] semantics: live-source scope, canonical-or-
/// raw name, case-insensitive (`COLLATE NOCASE`). A seed absent here is dropped.
async fn seed_in_graph(
    pool: &SqlitePool,
    notebook_id: &str,
    name: &str,
    kind: EntityKind,
) -> Result<bool, LensError> {
    let hit: Option<i64> = sqlx::query_scalar(
        "SELECT 1 FROM entity_nodes en \
         JOIN sources s ON s.id = en.source_id \
         WHERE en.notebook_id = ? \
           AND (en.name = ? COLLATE NOCASE OR en.canonical_name = ? COLLATE NOCASE) \
           AND en.kind = ? \
           AND s.trashed_at IS NULL AND s.selected = 1 \
         LIMIT 1",
    )
    .bind(notebook_id)
    .bind(name)
    .bind(name)
    .bind(kind.as_str())
    .fetch_optional(pool)
    .await?;
    Ok(hit.is_some())
}

/// Fetches the notebook's most entity-dense live chunk texts, prefixing each with its
/// id so the LLM can copy ids verbatim into `gold_chunk_ids`. Returns the formatted
/// context string and the set of fed chunk ids (for gold validation).
async fn entity_dense_context(
    pool: &SqlitePool,
    notebook_id: &str,
) -> Result<(String, HashSet<String>), LensError> {
    use sqlx::Row;
    let rows = sqlx::query(
        "SELECT c.id, c.text, COUNT(em.id) AS mentions \
         FROM chunks c \
         JOIN sources s ON c.source_id = s.id \
         LEFT JOIN entity_mentions em ON em.chunk_id = c.id \
         WHERE s.notebook_id = ? AND s.trashed_at IS NULL AND s.selected = 1 \
         GROUP BY c.id \
         ORDER BY mentions DESC, c.id ASC \
         LIMIT ?",
    )
    .bind(notebook_id)
    .bind(QA_MAX_CONTEXT_CHUNKS as i64)
    .fetch_all(pool)
    .await?;

    let mut parts: Vec<String> = Vec::with_capacity(rows.len());
    let mut ids: HashSet<String> = HashSet::with_capacity(rows.len());
    for row in &rows {
        let id: String = row.get("id");
        let text: String = row.get("text");
        parts.push(format!("[chunk_id: {id}]\n{text}"));
        ids.insert(id);
    }
    Ok((parts.join("\n\n"), ids))
}

/// Persists one accepted question with its seeds + provenance gold. `seed_entities`
/// and `gold_chunk_ids` are JSON arrays.
async fn persist_question(
    pool: &SqlitePool,
    notebook_id: &str,
    q: &EvalQuestion,
) -> Result<(), LensError> {
    let seeds_json: Vec<serde_json::Value> = q
        .seeds
        .iter()
        .map(|s| serde_json::json!({ "name": s.name, "kind": s.kind.as_str() }))
        .collect();
    let seeds = serde_json::to_string(&seeds_json)?;
    let gold = serde_json::to_string(&q.gold_chunk_ids)?;
    sqlx::query(
        "INSERT INTO eval_questions \
         (id, notebook_id, kind, question, seed_entities, gold_chunk_ids, prompt_version, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(uuid::Uuid::now_v7().to_string())
    .bind(notebook_id)
    .bind(q.kind.as_str())
    .bind(&q.question)
    .bind(seeds)
    .bind(gold)
    .bind(QA_PROMPT_VERSION)
    .bind(chrono::Utc::now().to_rfc3339())
    .execute(pool)
    .await?;
    Ok(())
}

/// Persists one `notebook_eval_log` row. Booleans stored as INTEGER.
async fn persist_log(
    pool: &SqlitePool,
    notebook_id: &str,
    report: &EvalReport,
) -> Result<(), LensError> {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO notebook_eval_log \
         (id, notebook_id, ran_at, graph_recall, hybrid_recall, delta_pp, p95_ms, passed, \
          sample_n, dropped_n, graph_enabled, prompt_version, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(uuid::Uuid::now_v7().to_string())
    .bind(notebook_id)
    .bind(&now)
    .bind(report.graph_recall as f64)
    .bind(report.hybrid_recall as f64)
    .bind(report.delta_pp as f64)
    .bind(report.p95_ms as f64)
    .bind(report.passed as i64)
    .bind(report.sample_n as i64)
    .bind(report.dropped_n as i64)
    .bind(report.graph_enabled as i64)
    .bind(&report.prompt_version)
    .bind(&now)
    .execute(pool)
    .await?;
    Ok(())
}

/// 95th-percentile (nearest-rank) of `xs`; empty → `0.0`. With ~20-30 samples this
/// is effectively a max-ish guard, not a statistically robust p95.
fn percentile95(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = ((0.95 * sorted.len() as f32).ceil() as usize).max(1) - 1;
    sorted[rank.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn question_kind_roundtrips() {
        for (k, s) in [
            (QuestionKind::SingleHop, "single_hop"),
            (QuestionKind::Bridging, "bridging"),
            (QuestionKind::Rollup, "rollup"),
        ] {
            assert_eq!(k.as_str(), s);
            assert_eq!(QuestionKind::from_db(s).unwrap(), k);
        }
        assert!(QuestionKind::from_db("multi").is_err());
    }

    #[test]
    fn parse_qa_inner_validates_kinds_and_fed_ids() {
        let mut fed = HashSet::new();
        fed.insert("chunk-1".to_string());
        fed.insert("chunk-2".to_string());

        let json = r#"[
            {"kind":"single_hop","question":"q1",
             "seed_entities":[{"name":"Alice","kind":"person"}],
             "gold_chunk_ids":["chunk-1"]},
            {"kind":"bridging","question":"q2",
             "seed_entities":[{"name":"Voyager","kind":"concept"}],
             "gold_chunk_ids":["chunk-2"]}
        ]"#;
        let parsed = parse_qa_inner(json, &fed).expect("valid");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].kind, QuestionKind::SingleHop);
        assert_eq!(parsed[1].seeds[0].kind, EntityKind::Concept);
        assert_eq!(parsed[0].gold_chunk_ids, vec!["chunk-1"]);
    }

    #[test]
    fn parse_qa_inner_rejects_unknown_kind() {
        let fed: HashSet<String> = ["chunk-1".to_string()].into();
        let json = r#"[{"kind":"multi","question":"q","seed_entities":[],"gold_chunk_ids":[]}]"#;
        assert!(parse_qa_inner(json, &fed).is_err());
    }

    #[test]
    fn parse_qa_inner_rejects_unknown_seed_kind() {
        let fed: HashSet<String> = ["chunk-1".to_string()].into();
        let json = r#"[{"kind":"rollup","question":"q",
            "seed_entities":[{"name":"X","kind":"bogus"}],
            "gold_chunk_ids":["chunk-1"]}]"#;
        assert!(parse_qa_inner(json, &fed).is_err());
    }

    #[test]
    fn parse_qa_inner_rejects_hallucinated_gold_id() {
        let fed: HashSet<String> = ["chunk-1".to_string()].into();
        let json = r#"[{"kind":"single_hop","question":"q",
            "seed_entities":[{"name":"A","kind":"concept"}],
            "gold_chunk_ids":["chunk-HALLUCINATED"]}]"#;
        let err = parse_qa_inner(json, &fed).expect_err("hallucinated id must fail");
        assert!(matches!(err, LensError::Parse(_)));
    }

    #[test]
    fn percentile95_and_mean() {
        assert_eq!(mean(&[]), 0.0);
        assert_eq!(mean(&[1.0, 3.0]), 2.0);
        assert_eq!(percentile95(&[]), 0.0);
        // 20 samples: rank ceil(0.95*20)-1 = 18 → the 19th-smallest value.
        let xs: Vec<f32> = (1..=20).map(|i| i as f32).collect();
        assert_eq!(percentile95(&xs), 19.0);
    }
}
