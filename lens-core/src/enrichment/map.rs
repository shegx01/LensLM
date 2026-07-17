//! Structural-map LLM map-reduce (AC4) over an [`LlmProvider`] trait object.
//!
//! Maps over level-0 parent chunks; for over-window docs reduces batches hierarchically.
//! Persistent malformed output degrades to context-prefix-only (never fails the source).
//! Budget (AC11) is checked before every `generate()`; a breach surfaces
//! `MapError::BudgetExceeded` so the worker flips to `failed + budget_exceeded`.

use crate::error::LensError;
use crate::llm::{LlmProvider, LlmRequest};

use super::batching::batch_by_char_budget;
use super::meta::{
    Budget, BudgetCheck, ENRICHMENT_BATCH_BYTE_BUDGET, ENRICHMENT_MAP_MAX_TOKENS,
    ENRICHMENT_MAX_RETRIES, StructuralMap,
};

/// Shared input-byte budget per map batch (synced with the coref pass).
const MAP_BATCH_BYTE_BUDGET: usize = ENRICHMENT_BATCH_BYTE_BUDGET;

const PARENT_SEPARATOR_LEN: usize = 2;

const MAP_SYSTEM_PROMPT: &str = "You extract a structural map from a document. \
Respond with ONLY a JSON object, no prose, no markdown fences, with EXACTLY these \
keys: \"entities\" (array of strings), \"definitions\" (array of {\"term\",\"definition\"}), \
\"dates\" (array of strings), \"summary\" (string). Do not add any other keys.";

/// Outcome of a structural-map attempt (AC4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapOutcome {
    Ok(StructuralMap),
    /// All reprompts returned malformed JSON — degrade to context-prefix-only.
    Fallback,
}

/// Non-degrade failure of the structural-map pass.
#[derive(Debug)]
pub enum MapError {
    /// Pre-dispatch budget breach (AC11) — source flips to `failed + budget_exceeded`.
    BudgetExceeded,
    /// Transport/provider error — source flips to `failed`, raw vectors untouched.
    Llm(LensError),
}

impl From<LensError> for MapError {
    fn from(e: LensError) -> Self {
        MapError::Llm(e)
    }
}

fn batch_parents(parent_texts: &[String]) -> Vec<String> {
    batch_by_char_budget(
        parent_texts.iter(),
        MAP_BATCH_BYTE_BUDGET,
        PARENT_SEPARATOR_LEN,
        |text| text.len(),
    )
    .into_iter()
    .map(|group| {
        group
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    })
    .collect()
}

/// Shared LLM retry/budget/parse loop (DRY for structural-map and coref passes).
/// Returns `Ok(Some)` on success, `Ok(None)` on exhausted retries (caller degrades),
/// `Err(BudgetExceeded)` on pre-dispatch breach, `Err(Llm)` on transport failure.
/// Budget is checked before every `generate()`; calls pin `temperature=0, json=true`.
pub(crate) async fn run_llm_with_retries<T>(
    provider: &dyn LlmProvider,
    budget: &mut Budget,
    system_prompt: &str,
    user_prompt: &str,
    max_tokens: u32,
    parse: impl Fn(&str) -> Result<T, LensError>,
) -> Result<Option<T>, MapError> {
    let total_attempts = ENRICHMENT_MAX_RETRIES + 1;
    let mut last_body = String::new();
    for attempt in 0..total_attempts {
        if budget.check_before_dispatch(max_tokens) == BudgetCheck::Exceeded {
            return Err(MapError::BudgetExceeded);
        }

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
            temperature: 0.0,
            json: true,
            thinking: false,
            reasoning_effort: None,
            messages: Vec::new(),
        };
        let resp = provider.generate(&req).await?;
        budget.record(resp.tokens_used);

        match parse(&resp.text) {
            Ok(value) => return Ok(Some(value)),
            Err(_) => {
                last_body = resp.text;
            }
        }
    }
    Ok(None)
}

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

/// Builds the structural map over `parent_texts` with hierarchical reduce for
/// over-window docs (AC4). Returns `Fallback` when validation never succeeds (not
/// a failure); returns `MapError` only on budget breach or provider failure.
pub async fn build_structural_map(
    provider: &dyn LlmProvider,
    budget: &mut Budget,
    parent_texts: &[String],
) -> Result<MapOutcome, MapError> {
    if parent_texts.is_empty() {
        return Ok(MapOutcome::Fallback);
    }

    let batches = batch_parents(parent_texts);

    if batches.len() == 1 {
        let prompt = format!("Document:\n{}", batches[0]);
        return Ok(match map_one_batch(provider, budget, &prompt).await? {
            Some(map) => MapOutcome::Ok(map),
            None => MapOutcome::Fallback,
        });
    }

    let mut partials: Vec<StructuralMap> = Vec::with_capacity(batches.len());
    for batch in &batches {
        let prompt = format!("Document section:\n{batch}");
        match map_one_batch(provider, budget, &prompt).await? {
            Some(map) => partials.push(map),
            None => continue, // malformed partial — skip; all-malformed → fallback
        }
    }
    if partials.is_empty() {
        return Ok(MapOutcome::Fallback);
    }

    let reduce_input = render_partials_for_reduce(&partials);
    let reduce_prompt = format!(
        "These are partial structural maps of different sections of ONE document. \
         Merge them into a single structural map for the whole document:\n{reduce_input}"
    );
    Ok(
        match map_one_batch(provider, budget, &reduce_prompt).await? {
            Some(map) => MapOutcome::Ok(map),
            // Reduce failed but partials exist → best-effort merge (not `fallback`).
            None => MapOutcome::Ok(merge_partials(partials)),
        },
    )
}

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

/// Merges partial maps (reduce-call fallback): unions entities/dates/definitions
/// (dedup, order-preserving) and concatenates summaries.
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

    use std::sync::atomic::Ordering;

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
        let (provider, _calls) = ScriptedProvider::dead();
        let mut budget = Budget::new(SessionBudget::new());
        let err = build_structural_map(&provider, &mut budget, &["doc".to_string()])
            .await
            .expect_err("dead provider errors");
        assert!(matches!(err, MapError::Llm(_)));
    }

    #[tokio::test]
    async fn over_window_doc_hierarchically_reduces() {
        let big = "x ".repeat(MAP_BATCH_BYTE_BUDGET);
        let parents = vec![big.clone(), big];
        let (provider, calls) = ScriptedProvider::new(vec![valid_map()]);
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
