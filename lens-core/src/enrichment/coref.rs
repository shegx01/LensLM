//! Real LLM-driven coreference resolution for the `LlmInline` strategy (Step 4 /
//! M4 Phase-3) — schema-constrained SUBSTITUTION extraction, applied
//! DETERMINISTICALLY in Rust.
//!
//! The LLM only IDENTIFIES referential expressions (pronouns; definite
//! descriptions like "the company"/"this approach") whose antecedent is a named
//! entity present in the passage OR the provided entity list, returning each as a
//! `{mention, char_start, char_end, antecedent}` substitution into the chunk's
//! OWN text. Our Rust code then APPLIES the surviving substitutions with strict
//! validation (offset/char-boundary/mention/antecedent checks) and a
//! RIGHT-TO-LEFT splice so earlier offsets stay valid. This bounds hallucination
//! (an invented antecedent that is not in the allow-list is dropped) and keeps the
//! transform deterministic + cacheable.
//!
//! ## What is and is NOT mutated
//!
//! Coref resolution only ever rewrites the BODY that feeds a chunk's
//! `embedding_text`; the canonical `chunks.text` is NEVER touched (it remains the
//! immutable citation text). When coref degrades (malformed output, budget) the
//! worker falls back to the raw body — coref never fails the source.
//!
//! ## Budget + circuit-break (AC11)
//!
//! Like the structural map, the shared [`Budget`](super::meta::Budget) is checked
//! BEFORE every `generate()`; a breach surfaces [`MapError::BudgetExceeded`] so the
//! worker fails the source. The coref pass shares the SAME `Budget` instance as the
//! map, so the per-job circuit-break covers both passes together.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::llm::LlmProvider;

use super::map::{MapError, run_llm_with_retries};
use super::meta::{Budget, ENRICHMENT_COREF_MAX_TOKENS};

/// Soft input-character budget for a single coref batch. Chunk bodies are batched
/// under this ceiling (each tagged with its `id`); a single chunk larger than the
/// budget forms its own batch (the provider truncates if needed; never a panic).
const COREF_BATCH_CHAR_BUDGET: usize = 8 * 1024;

/// The system prompt pinning the LLM to emit STRICT coref-substitution JSON.
///
/// It identifies ONLY referential expressions whose antecedent is a named entity
/// in the passage or the supplied entity list, returns char offsets into THAT
/// chunk's text, resolves in the document's own language, and NEVER invents an
/// antecedent (empty `subs` when there is nothing to resolve). The strict serde
/// parse + the Rust-side validation in [`apply_substitutions`] are the real guards.
const COREF_SYSTEM_PROMPT: &str = "You resolve coreferences in document passages. \
For each passage you are given its integer id and its text. Identify ONLY referential \
expressions — pronouns (it, they, he, she, this, that, …) and definite descriptions \
(\"the company\", \"this approach\", …) — whose antecedent is a NAMED ENTITY that appears \
in the same passage OR in the provided entity list. For each one, report the exact mention \
substring, its character offsets [char_start, char_end) into THAT passage's text, and the \
antecedent entity it refers to. Use the document's own language. NEVER invent an antecedent: \
if a reference has no clear named-entity antecedent, omit it. If a passage has nothing to \
resolve, return an empty subs array for it. Respond with ONLY a JSON object, no prose, no \
markdown fences, with EXACTLY this shape: \
{\"results\":[{\"id\":<int>,\"subs\":[{\"mention\":<str>,\"char_start\":<int>,\"char_end\":<int>,\"antecedent\":<str>}]}]}. \
Do not add any other keys.";

/// The strict serde schema for the coref response. Unknown fields are rejected so
/// a garbled/hallucinated shape triggers a reprompt rather than silent acceptance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorefResponse {
    /// Per-chunk substitution sets, each tagged with the chunk's batch `id`.
    pub results: Vec<ChunkCoref>,
}

/// The coref substitutions identified for a single chunk (by its batch `id`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkCoref {
    /// The chunk's batch id (the positional index supplied in the request).
    pub id: usize,
    /// The substitutions to apply to this chunk's text.
    pub subs: Vec<CorefSub>,
}

/// A single coref substitution: replace `text[char_start..char_end]` (which must
/// equal `mention`) with `antecedent`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorefSub {
    /// The referential expression as it appears in the chunk text.
    pub mention: String,
    /// Inclusive start char offset of the mention in the chunk text.
    pub char_start: usize,
    /// Exclusive end char offset of the mention in the chunk text.
    pub char_end: usize,
    /// The named-entity antecedent to substitute in.
    pub antecedent: String,
}

impl CorefResponse {
    /// Parses + strictly validates an LLM response body as a [`CorefResponse`],
    /// tolerating markdown fences / chat preamble by extracting the first balanced
    /// `{...}` object before parsing (mirrors [`super::meta::StructuralMap::parse_strict`]).
    /// Returns `Err` on any parse/validation miss so the caller can reprompt.
    pub fn parse_strict(body: &str) -> Result<Self, crate::LensError> {
        let json = super::meta::extract_json_object(body).unwrap_or(body);
        serde_json::from_str::<CorefResponse>(json)
            .map_err(|e| crate::LensError::Parse(format!("coref response invalid: {e}")))
    }
}

/// Splits `(id, text)` chunks into batches whose concatenated text stays under
/// [`COREF_BATCH_CHAR_BUDGET`]. A single chunk that alone exceeds the budget forms
/// its own batch.
fn batch_chunks<'a>(chunks: &[(usize, &'a str)]) -> Vec<Vec<(usize, &'a str)>> {
    let mut batches: Vec<Vec<(usize, &str)>> = Vec::new();
    let mut current: Vec<(usize, &str)> = Vec::new();
    let mut current_len = 0usize;
    for &(id, text) in chunks {
        if !current.is_empty() && current_len + text.len() + 2 > COREF_BATCH_CHAR_BUDGET {
            batches.push(std::mem::take(&mut current));
            current_len = 0;
        }
        current_len += text.len() + 2;
        current.push((id, text));
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

/// Renders one batch's `(id, text)` chunks + the doc entity list into the coref
/// user prompt.
fn render_batch_prompt(batch: &[(usize, &str)], entities: &[String]) -> String {
    let entity_line = if entities.is_empty() {
        "(none provided)".to_string()
    } else {
        entities.join(", ")
    };
    let mut prompt = String::new();
    prompt.push_str("Known named entities: ");
    prompt.push_str(&entity_line);
    prompt.push_str("\n\nPassages:\n");
    for (id, text) in batch {
        prompt.push_str(&format!("[id={id}]\n{text}\n\n"));
    }
    prompt
}

/// Resolves coreferences over `chunks` (each `(id, text)`), returning the
/// per-`id` surviving substitutions. The same [`Budget`] instance as the
/// structural map is threaded in so the per-job circuit-break covers both passes
/// (AC11): a pre-dispatch budget breach surfaces [`MapError::BudgetExceeded`] so
/// the worker fails the source.
///
/// On persistent malformed output for a batch the result DEGRADES to empty subs
/// for that batch's chunks (the source is NOT failed — the worker falls back to the
/// raw body). A provider/transport error propagates as [`MapError::Llm`].
pub async fn resolve_coref_batch(
    provider: &dyn LlmProvider,
    budget: &mut Budget,
    chunks: &[(usize, &str)],
    entities: &[String],
) -> Result<HashMap<usize, Vec<CorefSub>>, MapError> {
    let mut out: HashMap<usize, Vec<CorefSub>> = HashMap::new();
    if chunks.is_empty() {
        return Ok(out);
    }

    for batch in batch_chunks(chunks) {
        let user_prompt = render_batch_prompt(&batch, entities);
        // Shared retry/budget/parse loop (DRY with the structural map). On
        // persistent malformed output → `None` → degrade to empty subs for this
        // batch; a budget breach / provider error propagates as `MapError`.
        let parsed = run_llm_with_retries(
            provider,
            budget,
            COREF_SYSTEM_PROMPT,
            &user_prompt,
            ENRICHMENT_COREF_MAX_TOKENS,
            CorefResponse::parse_strict,
        )
        .await?;

        match parsed {
            Some(resp) => {
                for chunk_coref in resp.results {
                    out.entry(chunk_coref.id)
                        .or_default()
                        .extend(chunk_coref.subs);
                }
            }
            // Malformed×retries → degrade to empty subs for this batch (the worker
            // falls back to the raw body for these chunks; never a source failure).
            None => continue,
        }
    }

    Ok(out)
}

/// Applies the surviving coref substitutions to `text`, returning the resolved
/// text. PURE + deterministic — never panics on bad offsets.
///
/// Validation rules (a sub is DROPPED on any failure):
/// 1. `char_start <= char_end <= text.len()` and both are on UTF-8 char
///    boundaries (an offset mid-codepoint is invalid);
/// 2. `text[char_start..char_end] == mention` (the LLM's offsets must actually
///    point at the claimed mention);
/// 3. `antecedent` is non-empty;
/// 4. `antecedent` appears in `allowed_antecedents` (the doc entities / summary
///    terms) — this is the hallucination guard: an invented antecedent that is not
///    in the allow-list is dropped.
///
/// Surviving subs are applied RIGHT-TO-LEFT (sorted by `char_start` descending) so
/// each splice leaves the offsets of the not-yet-applied (earlier) subs valid.
/// Overlapping subs are skipped (a later splice whose range intersects an
/// already-applied one is dropped) so the result stays well-defined.
pub fn apply_substitutions(
    text: &str,
    subs: &[CorefSub],
    allowed_antecedents: &[String],
) -> String {
    // Validate + collect the survivors.
    let mut valid: Vec<&CorefSub> = subs
        .iter()
        .filter(|s| is_valid_sub(text, s, allowed_antecedents))
        .collect();

    if valid.is_empty() {
        return text.to_string();
    }

    // Right-to-left: descending by char_start so earlier offsets stay valid as we
    // splice from the end.
    valid.sort_by(|a, b| b.char_start.cmp(&a.char_start));

    let mut result = text.to_string();
    let mut last_applied_start = usize::MAX;
    for sub in valid {
        // Skip an overlap with an already-applied (further-right) sub.
        if sub.char_end > last_applied_start {
            continue;
        }
        // Offsets validated above; this splice can never panic.
        result.replace_range(sub.char_start..sub.char_end, &sub.antecedent);
        last_applied_start = sub.char_start;
    }
    result
}

/// Whether a single sub passes every validation rule against `text`.
fn is_valid_sub(text: &str, sub: &CorefSub, allowed_antecedents: &[String]) -> bool {
    if sub.antecedent.trim().is_empty() {
        return false;
    }
    if sub.char_start > sub.char_end || sub.char_end > text.len() {
        return false;
    }
    if !text.is_char_boundary(sub.char_start) || !text.is_char_boundary(sub.char_end) {
        return false;
    }
    if text[sub.char_start..sub.char_end] != sub.mention {
        return false;
    }
    // Hallucination guard: the antecedent must be a known doc entity / term.
    allowed_antecedents.contains(&sub.antecedent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enrichment::meta::{Budget, SessionBudget};
    use crate::error::LensError;
    use crate::llm::{LlmProvider, LlmRequest, LlmResponse};

    use async_trait::async_trait;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A mock provider with a call-counter returning a scripted response sequence
    /// (cycling the last entry once exhausted).
    struct ScriptedProvider {
        calls: Arc<AtomicU32>,
        responses: Vec<String>,
    }

    impl ScriptedProvider {
        fn new(responses: Vec<&str>) -> (Self, Arc<AtomicU32>) {
            let calls = Arc::new(AtomicU32::new(0));
            (
                Self {
                    calls: calls.clone(),
                    responses: responses.into_iter().map(|s| s.to_string()).collect(),
                },
                calls,
            )
        }
    }

    #[async_trait]
    impl LlmProvider for ScriptedProvider {
        fn model_id(&self) -> &str {
            "mock-model"
        }
        async fn reachable(&self) -> bool {
            true
        }
        async fn generate(&self, _req: &LlmRequest) -> Result<LlmResponse, LensError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst) as usize;
            let text = self
                .responses
                .get(n)
                .or_else(|| self.responses.last())
                .cloned()
                .unwrap_or_default();
            Ok(LlmResponse {
                text,
                tokens_used: 10,
            })
        }
    }

    // --- schema parse -------------------------------------------------------

    #[test]
    fn parse_strict_accepts_valid_coref() {
        let body = r#"{"results":[{"id":0,"subs":[{"mention":"She","char_start":0,"char_end":3,"antecedent":"Ada"}]}]}"#;
        let resp = CorefResponse::parse_strict(body).expect("valid");
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].id, 0);
        assert_eq!(resp.results[0].subs[0].mention, "She");
        assert_eq!(resp.results[0].subs[0].antecedent, "Ada");
    }

    #[test]
    fn parse_strict_tolerates_markdown_fences() {
        let body = "Sure, here:\n```json\n{\"results\":[]}\n```\n";
        let resp = CorefResponse::parse_strict(body).expect("fenced still parses");
        assert!(resp.results.is_empty());
    }

    #[test]
    fn parse_strict_rejects_malformed_and_unknown_fields() {
        assert!(CorefResponse::parse_strict("not json").is_err());
        // Unknown top-level field.
        assert!(CorefResponse::parse_strict(r#"{"results":[],"extra":1}"#).is_err());
        // Unknown sub field.
        assert!(
            CorefResponse::parse_strict(
                r#"{"results":[{"id":0,"subs":[{"mention":"x","char_start":0,"char_end":1,"antecedent":"y","z":1}]}]}"#
            )
            .is_err()
        );
    }

    // --- apply_substitutions -----------------------------------------------

    fn sub(mention: &str, start: usize, end: usize, antecedent: &str) -> CorefSub {
        CorefSub {
            mention: mention.to_string(),
            char_start: start,
            char_end: end,
            antecedent: antecedent.to_string(),
        }
    }

    #[test]
    fn apply_valid_single_sub() {
        let text = "She wrote the first algorithm.";
        let subs = vec![sub("She", 0, 3, "Ada Lovelace")];
        let allowed = vec!["Ada Lovelace".to_string()];
        assert_eq!(
            apply_substitutions(text, &subs, &allowed),
            "Ada Lovelace wrote the first algorithm."
        );
    }

    #[test]
    fn apply_multi_sub_right_to_left_keeps_offsets_valid() {
        // Two subs in one text; applying left-to-right would shift the 2nd offset.
        // Right-to-left keeps both valid.
        let text = "It cited it again.";
        // "It" @ [0,2), "it" @ [9,11)
        let subs = vec![sub("It", 0, 2, "Babbage"), sub("it", 9, 11, "the engine")];
        let allowed = vec!["Babbage".to_string(), "the engine".to_string()];
        assert_eq!(
            apply_substitutions(text, &subs, &allowed),
            "Babbage cited the engine again."
        );
    }

    #[test]
    fn drop_sub_when_mention_mismatches_offsets() {
        let text = "She wrote it.";
        // Offsets point at "She" but mention claims "He" → drop.
        let subs = vec![sub("He", 0, 3, "Ada")];
        let allowed = vec!["Ada".to_string()];
        assert_eq!(apply_substitutions(text, &subs, &allowed), text);
    }

    #[test]
    fn drop_sub_when_offset_not_on_char_boundary() {
        // "é" is 2 bytes; an offset of 1 lands mid-codepoint → drop (no panic).
        let text = "café it";
        // bytes: c(0) a(1) f(2) é(3,4) space(5) i(6) t(7) -> len 8
        let subs = vec![sub("é", 3, 4, "Coffee")]; // end 4 is mid-é
        let allowed = vec!["Coffee".to_string()];
        assert_eq!(apply_substitutions(text, &subs, &allowed), text);
    }

    #[test]
    fn drop_sub_with_invented_antecedent_not_in_entities() {
        let text = "It is fast.";
        let subs = vec![sub("It", 0, 2, "Imaginary Corp")];
        // allow-list does NOT contain the antecedent → drop (hallucination guard).
        let allowed = vec!["Real Entity".to_string()];
        assert_eq!(apply_substitutions(text, &subs, &allowed), text);
    }

    #[test]
    fn drop_sub_with_empty_antecedent() {
        let text = "It runs.";
        let subs = vec![sub("It", 0, 2, "")];
        let allowed = vec!["".to_string()];
        assert_eq!(apply_substitutions(text, &subs, &allowed), text);
    }

    #[test]
    fn drop_sub_with_out_of_range_offset() {
        let text = "It";
        let subs = vec![sub("It", 0, 99, "Entity")];
        let allowed = vec!["Entity".to_string()];
        assert_eq!(apply_substitutions(text, &subs, &allowed), text);
    }

    #[test]
    fn empty_subs_leaves_text_unchanged() {
        let text = "Nothing to resolve here.";
        assert_eq!(apply_substitutions(text, &[], &["X".to_string()]), text);
    }

    #[test]
    fn multibyte_text_resolves_on_valid_boundaries() {
        // Resolve a pronoun in non-ASCII text. "Elle" → "Marie Curie".
        let text = "Elle a gagné le prix Nobel.";
        let subs = vec![sub("Elle", 0, 4, "Marie Curie")];
        let allowed = vec!["Marie Curie".to_string()];
        assert_eq!(
            apply_substitutions(text, &subs, &allowed),
            "Marie Curie a gagné le prix Nobel."
        );
    }

    // --- resolve_coref_batch (mock provider) --------------------------------

    fn valid_coref_for(id: usize, mention: &str, start: usize, end: usize, ant: &str) -> String {
        format!(
            r#"{{"results":[{{"id":{id},"subs":[{{"mention":"{mention}","char_start":{start},"char_end":{end},"antecedent":"{ant}"}}]}}]}}"#
        )
    }

    #[tokio::test]
    async fn resolve_returns_subs_one_call() {
        let resp = valid_coref_for(0, "She", 0, 3, "Ada");
        let (provider, calls) = ScriptedProvider::new(vec![&resp]);
        let mut budget = Budget::new(SessionBudget::new());
        let chunks = vec![(0usize, "She wrote code.")];
        let out = resolve_coref_batch(&provider, &mut budget, &chunks, &["Ada".to_string()])
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1, "exactly one coref call");
        assert_eq!(out.get(&0).unwrap()[0].antecedent, "Ada");
    }

    #[tokio::test]
    async fn malformed_thrice_degrades_to_empty_subs_not_failed() {
        let (provider, calls) = ScriptedProvider::new(vec!["nope", "still bad", "garbage"]);
        let mut budget = Budget::new(SessionBudget::new());
        let chunks = vec![(0usize, "She wrote code.")];
        let out = resolve_coref_batch(&provider, &mut budget, &chunks, &["Ada".to_string()])
            .await
            .expect("degrades, never errors");
        assert!(out.is_empty(), "malformed coref degrades to empty subs map");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "1 initial + ENRICHMENT_MAX_RETRIES(2) reprompts"
        );
    }

    #[tokio::test]
    async fn budget_short_circuits_before_second_batch_dispatch() {
        // Two batches (each chunk just over the batch budget), budget admits exactly
        // 1 call. The mock must see EXACTLY 1 generate() — the 2nd batch is never
        // dispatched — and the error is BudgetExceeded.
        let big = "x ".repeat(COREF_BATCH_CHAR_BUDGET);
        let resp = valid_coref_for(0, "x", 0, 1, "Ada");
        let (provider, calls) = ScriptedProvider::new(vec![&resp]);
        let mut budget = Budget::with_caps(SessionBudget::new(), 1_000_000, 1);
        let chunks = vec![(0usize, big.as_str()), (1usize, big.as_str())];
        let err = resolve_coref_batch(&provider, &mut budget, &chunks, &["Ada".to_string()])
            .await
            .expect_err("must circuit-break");
        assert!(matches!(err, MapError::BudgetExceeded));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "the SECOND batch generate() must NEVER be dispatched"
        );
    }

    #[tokio::test]
    async fn deterministic_same_input_same_resolved_text() {
        // Same mock script + same input + same allow-list → IDENTICAL resolved text
        // on every run (greedy temperature 0.0 + deterministic Rust application).
        let resp = valid_coref_for(0, "She", 0, 3, "Ada Lovelace");
        let allowed = vec!["Ada Lovelace".to_string()];
        let body = "She wrote the first algorithm.";

        let mut first = String::new();
        for run in 0..3 {
            let (provider, _calls) = ScriptedProvider::new(vec![&resp]);
            let mut budget = Budget::new(SessionBudget::new());
            let chunks = vec![(0usize, body)];
            let subs = resolve_coref_batch(&provider, &mut budget, &chunks, &allowed)
                .await
                .unwrap();
            let resolved = apply_substitutions(body, subs.get(&0).unwrap(), &allowed);
            if run == 0 {
                first = resolved;
                assert_eq!(first, "Ada Lovelace wrote the first algorithm.");
            } else {
                assert_eq!(resolved, first, "coref resolution must be deterministic");
            }
        }
    }

    #[tokio::test]
    async fn empty_chunks_makes_no_call() {
        let (provider, calls) = ScriptedProvider::new(vec!["{\"results\":[]}"]);
        let mut budget = Budget::new(SessionBudget::new());
        let out = resolve_coref_batch(&provider, &mut budget, &[], &[])
            .await
            .unwrap();
        assert!(out.is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), 0, "no chunks → no call");
    }
}
