//! The structural-map LLM map-reduce (AC4) — pure orchestration over an
//! [`LlmProvider`] trait object so it is unit-testable against a mock with a
//! call-counter, with NO DB / embedder / network in the loop.
//!
//! The worker maps over a source's `level=0` parent chunks, batching their text
//! to the LLM window; for over-window docs it hierarchically reduces (map each
//! batch to a partial map, then reduce the partial summaries into a final map).
//! The LLM response is validated against the strict [`StructuralMap`] serde
//! schema with up to [`ENRICHMENT_MAX_RETRIES`] reprompts; on persistent
//! malformed output it DEGRADES to context-prefix-only (the caller keeps the
//! source `enriching`/`enriched` with `map_quality="fallback"` — never failed).
//!
//! Budget + circuit-break (AC11) is enforced HERE, before every `generate()`:
//! the shared [`Budget`] is checked and a breach short-circuits WITHOUT
//! dispatching the call, surfacing [`MapError::BudgetExceeded`] so the worker
//! flips the source to `failed` + `budget_exceeded`.

use crate::error::LensError;
use crate::llm::{LlmProvider, LlmRequest};

use super::meta::{
    Budget, BudgetCheck, ENRICHMENT_BATCH_CHAR_BUDGET, ENRICHMENT_MAP_MAX_TOKENS,
    ENRICHMENT_MAX_RETRIES, StructuralMap,
};

/// Soft input-character budget for a single map batch (the shared enrichment batch
/// budget). Sized well under a typical local-model context so several parents batch
/// together but a huge doc splits into multiple map calls (triggering the
/// hierarchical reduce). Shared with the coref pass via
/// [`ENRICHMENT_BATCH_CHAR_BUDGET`] so the two batchers stay in sync.
const MAP_BATCH_CHAR_BUDGET: usize = ENRICHMENT_BATCH_CHAR_BUDGET;

/// The system prompt that pins the LLM to emit STRICT JSON matching
/// [`StructuralMap`]. Kept terse; the strict serde validation is the real guard.
const MAP_SYSTEM_PROMPT: &str = "You extract a structural map from a document. \
Respond with ONLY a JSON object, no prose, no markdown fences, with EXACTLY these \
keys: \"entities\" (array of strings), \"definitions\" (array of {\"term\",\"definition\"}), \
\"dates\" (array of strings), \"summary\" (string). Do not add any other keys.";

/// The outcome of a structural-map attempt over a source's parents (AC4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapOutcome {
    /// A validated structural map (`map_quality="ok"`).
    Ok(StructuralMap),
    /// Every reprompt produced malformed JSON → degrade to context-prefix-only
    /// (`map_quality="fallback"`, status stays enriching/enriched per AC4).
    Fallback,
}

/// A non-degrade failure of the structural-map pass.
#[derive(Debug)]
pub enum MapError {
    /// A budget/circuit-break short-circuit BEFORE a dispatch (AC11): the source
    /// must flip to `failed` with `budget_exceeded` in `enrichment_meta`.
    BudgetExceeded,
    /// A transport/provider error (LLM down/429): accumulate nothing, `failed`,
    /// raw vectors untouched (AC13 d).
    Llm(LensError),
}

impl From<LensError> for MapError {
    fn from(e: LensError) -> Self {
        MapError::Llm(e)
    }
}

/// Splits parent texts into batches whose concatenated length stays under
/// [`MAP_BATCH_CHAR_BUDGET`]. A single parent that alone exceeds the budget forms
/// its own batch (the provider truncates if needed; never a panic).
fn batch_parents(parent_texts: &[String]) -> Vec<String> {
    let mut batches: Vec<String> = Vec::new();
    let mut current = String::new();
    for text in parent_texts {
        if !current.is_empty() && current.len() + text.len() + 2 > MAP_BATCH_CHAR_BUDGET {
            batches.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(text);
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

/// Shared LLM retry/budget/parse loop for the enrichment passes (DRY across the
/// structural map and coref). Calls `provider.generate` with up to
/// [`ENRICHMENT_MAX_RETRIES`] reprompts, parsing each reply with `parse`. Returns:
/// * `Ok(Some(value))` — `parse` succeeded;
/// * `Ok(None)` — exhausted retries with malformed output (caller degrades);
/// * `Err(BudgetExceeded)` — a pre-dispatch budget breach (caller fails);
/// * `Err(Llm(_))` — a transport/provider error (caller fails).
///
/// Budget is checked BEFORE every `generate()` (AC11) using `max_tokens` as the
/// projected per-call cost: on a breach the call is NEVER dispatched. Every call
/// pins `temperature: 0.0, json: true` for deterministic, machine-parseable output.
pub(super) async fn run_llm_with_retries<T>(
    provider: &dyn LlmProvider,
    budget: &mut Budget,
    system_prompt: &str,
    user_prompt: &str,
    max_tokens: u32,
    parse: impl Fn(&str) -> Result<T, LensError>,
) -> Result<Option<T>, MapError> {
    // 1 initial attempt + ENRICHMENT_MAX_RETRIES reprompts.
    let total_attempts = ENRICHMENT_MAX_RETRIES + 1;
    let mut last_body = String::new();
    for attempt in 0..total_attempts {
        // AC11: check the budget BEFORE dispatching. A breach short-circuits.
        if budget.check_before_dispatch(max_tokens) == BudgetCheck::Exceeded {
            return Err(MapError::BudgetExceeded);
        }

        // On a reprompt, append the prior malformed body so the model can correct.
        let prompt = if attempt == 0 {
            user_prompt.to_string()
        } else {
            format!(
                "{user_prompt}\n\nYour previous reply was not valid JSON for the required \
                 schema:\n{last_body}\n\nReply again with ONLY the JSON object."
            )
        };

        let req = LlmRequest {
            system: Some(system_prompt.to_string()),
            prompt,
            max_tokens,
            // Determinism: greedy decode + JSON mode. The system prompt already
            // demands strict JSON; pinning temperature 0.0 + json maximizes
            // reproducible, machine-parseable output.
            temperature: 0.0,
            json: true,
        };
        let resp = provider.generate(&req).await?;
        budget.record(resp.tokens_used);

        match parse(&resp.text) {
            Ok(value) => return Ok(Some(value)),
            Err(_) => {
                last_body = resp.text;
                // loop to reprompt (if any attempts remain)
            }
        }
    }
    Ok(None)
}

/// Calls the LLM for one batch's map (thin wrapper over [`run_llm_with_retries`]
/// pinned to the [`StructuralMap`] schema + the map system prompt + token budget).
async fn map_one_batch(
    provider: &dyn LlmProvider,
    budget: &mut Budget,
    user_prompt: &str,
) -> Result<Option<StructuralMap>, MapError> {
    run_llm_with_retries(
        provider,
        budget,
        MAP_SYSTEM_PROMPT,
        user_prompt,
        ENRICHMENT_MAP_MAX_TOKENS,
        StructuralMap::parse_strict,
    )
    .await
}

/// Runs the full structural-map pass over `parent_texts` (the source's level-0
/// parent chunk bodies), with hierarchical reduce for over-window docs (AC4).
///
/// * Single batch → one map call (with reprompts).
/// * Multiple batches → map each to a partial map, then reduce the partials'
///   summaries into one final map call (reduce-of-reduces).
///
/// Returns [`MapOutcome::Fallback`] when validation never succeeds (the source
/// stays enriched-with-fallback, NOT failed — AC4); returns [`MapError`] only for
/// a budget breach or a provider/transport failure (AC11/AC13).
pub async fn build_structural_map(
    provider: &dyn LlmProvider,
    budget: &mut Budget,
    parent_texts: &[String],
) -> Result<MapOutcome, MapError> {
    if parent_texts.is_empty() {
        return Ok(MapOutcome::Fallback);
    }

    let batches = batch_parents(parent_texts);

    // Single-batch fast path: one map call.
    if batches.len() == 1 {
        let prompt = format!("Document:\n{}", batches[0]);
        return Ok(match map_one_batch(provider, budget, &prompt).await? {
            Some(map) => MapOutcome::Ok(map),
            None => MapOutcome::Fallback,
        });
    }

    // Hierarchical reduce: map each batch, collect the partial maps, then reduce.
    let mut partials: Vec<StructuralMap> = Vec::with_capacity(batches.len());
    for batch in &batches {
        let prompt = format!("Document section:\n{batch}");
        match map_one_batch(provider, budget, &prompt).await? {
            Some(map) => partials.push(map),
            // A malformed partial does not fail the whole doc — skip it; if ALL
            // partials are malformed the reduce input is empty → fallback.
            None => continue,
        }
    }
    if partials.is_empty() {
        return Ok(MapOutcome::Fallback);
    }

    // Reduce: feed the partial summaries (+ merged entities/dates) into one final
    // map call so the result is a single coherent doc map.
    let reduce_input = render_partials_for_reduce(&partials);
    let reduce_prompt = format!(
        "These are partial structural maps of different sections of ONE document. \
         Merge them into a single structural map for the whole document:\n{reduce_input}"
    );
    Ok(
        match map_one_batch(provider, budget, &reduce_prompt).await? {
            Some(map) => MapOutcome::Ok(map),
            // Reduce failed validation but we DO have partials → degrade to a merged
            // best-effort map rather than throwing away the work (still `ok` quality
            // is not claimed — the caller treats a merged map as a real map).
            None => MapOutcome::Ok(merge_partials(partials)),
        },
    )
}

/// Renders partial maps as compact text for the reduce prompt.
fn render_partials_for_reduce(partials: &[StructuralMap]) -> String {
    partials
        .iter()
        .enumerate()
        .map(|(i, m)| {
            format!(
                "Section {}: entities={:?}; dates={:?}; summary={}",
                i + 1,
                m.entities,
                m.dates,
                m.summary
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Deterministically merges partial maps (the reduce-call fallback): unions
/// entities/dates/definitions (dedup, order-preserving) and concatenates
/// summaries. Used only when the final reduce call's JSON failed to validate but
/// we already hold valid partials.
fn merge_partials(partials: Vec<StructuralMap>) -> StructuralMap {
    let mut entities: Vec<String> = Vec::new();
    let mut dates: Vec<String> = Vec::new();
    let mut definitions = Vec::new();
    let mut summaries: Vec<String> = Vec::new();
    for p in partials {
        for e in p.entities {
            if !entities.contains(&e) {
                entities.push(e);
            }
        }
        for d in p.dates {
            if !dates.contains(&d) {
                dates.push(d);
            }
        }
        definitions.extend(p.definitions);
        if !p.summary.trim().is_empty() {
            summaries.push(p.summary);
        }
    }
    StructuralMap {
        entities,
        definitions,
        dates,
        summary: summaries.join(" "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enrichment::meta::{Budget, SessionBudget};
    use crate::enrichment::test_util::ScriptedProvider;
    use crate::error::LensError;
    use crate::llm::{LlmProvider, LlmRequest, LlmResponse};

    use async_trait::async_trait;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A provider that always errors (LLM death / 429).
    struct DeadProvider {
        calls: Arc<AtomicU32>,
    }
    #[async_trait]
    impl LlmProvider for DeadProvider {
        fn model_id(&self) -> &str {
            "dead"
        }
        async fn reachable(&self) -> bool {
            false
        }
        async fn generate(&self, _req: &LlmRequest) -> Result<LlmResponse, LensError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(LensError::Network("connection refused".into()))
        }
    }

    fn valid_map() -> &'static str {
        r#"{"entities":["Ada"],"definitions":[],"dates":["1843"],"summary":"ok"}"#
    }

    #[tokio::test]
    async fn valid_first_attempt_returns_map_one_call() {
        let (provider, calls) = ScriptedProvider::new(vec![valid_map()]);
        let mut budget = Budget::new(SessionBudget::new());
        let out = build_structural_map(&provider, &mut budget, &["doc text".to_string()])
            .await
            .unwrap();
        assert!(matches!(out, MapOutcome::Ok(m) if m.summary == "ok"));
        assert_eq!(calls.load(Ordering::SeqCst), 1, "exactly one LLM call");
    }

    #[tokio::test]
    async fn malformed_then_valid_succeeds_after_retry() {
        let (provider, calls) = ScriptedProvider::new(vec!["not json", valid_map()]);
        let mut budget = Budget::new(SessionBudget::new());
        let out = build_structural_map(&provider, &mut budget, &["doc".to_string()])
            .await
            .unwrap();
        assert!(matches!(out, MapOutcome::Ok(_)));
        assert_eq!(calls.load(Ordering::SeqCst), 2, "one reprompt");
    }

    #[tokio::test]
    async fn malformed_thrice_degrades_to_fallback_not_failed() {
        // 1 initial + 2 reprompts = 3 malformed → fallback (AC4).
        let (provider, calls) = ScriptedProvider::new(vec!["nope", "still nope", "garbage"]);
        let mut budget = Budget::new(SessionBudget::new());
        let out = build_structural_map(&provider, &mut budget, &["doc".to_string()])
            .await
            .unwrap();
        assert_eq!(out, MapOutcome::Fallback, "degrades, never errors");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "1 initial + ENRICHMENT_MAX_RETRIES(2) reprompts"
        );
    }

    #[tokio::test]
    async fn budget_short_circuits_before_second_dispatch() {
        // AC11 seam: a per-job budget admitting exactly 1 call. The mock must see
        // EXACTLY 1 generate() even though the first reply is malformed (which
        // would otherwise reprompt).
        let (provider, calls) = ScriptedProvider::new(vec!["bad json", valid_map()]);
        let mut budget = Budget::with_caps(SessionBudget::new(), 1_000_000, 1);
        let err = build_structural_map(&provider, &mut budget, &["doc".to_string()])
            .await
            .expect_err("must circuit-break");
        assert!(matches!(err, MapError::BudgetExceeded));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "the SECOND generate() must NEVER be dispatched"
        );
        assert_eq!(budget.job_calls(), 1);
    }

    #[tokio::test]
    async fn budget_blocks_the_very_first_dispatch_when_zero() {
        let (provider, calls) = ScriptedProvider::new(vec![valid_map()]);
        let mut budget = Budget::with_caps(SessionBudget::new(), 1_000_000, 0);
        let err = build_structural_map(&provider, &mut budget, &["doc".to_string()])
            .await
            .expect_err("zero-call budget breaks immediately");
        assert!(matches!(err, MapError::BudgetExceeded));
        assert_eq!(calls.load(Ordering::SeqCst), 0, "no call dispatched at all");
    }

    #[tokio::test]
    async fn provider_error_propagates_as_llm_error() {
        let calls = Arc::new(AtomicU32::new(0));
        let provider = DeadProvider {
            calls: calls.clone(),
        };
        let mut budget = Budget::new(SessionBudget::new());
        let err = build_structural_map(&provider, &mut budget, &["doc".to_string()])
            .await
            .expect_err("dead provider errors");
        assert!(matches!(err, MapError::Llm(_)));
    }

    #[tokio::test]
    async fn over_window_doc_hierarchically_reduces() {
        // Two huge parents force two map batches + one reduce = 3 calls.
        let big = "x ".repeat(MAP_BATCH_CHAR_BUDGET);
        let parents = vec![big.clone(), big];
        let (provider, calls) = ScriptedProvider::new(vec![valid_map()]); // cycles valid_map
        let mut budget = Budget::new(SessionBudget::new());
        let out = build_structural_map(&provider, &mut budget, &parents)
            .await
            .unwrap();
        assert!(matches!(out, MapOutcome::Ok(_)));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "2 map batches + 1 reduce call"
        );
    }
}
