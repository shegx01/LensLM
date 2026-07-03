//! LLM-driven coreference resolution for the `LlmInline` strategy (Step 4).
//!
//! The LLM identifies referential expressions whose antecedent is a named entity in
//! the passage or entity list, returning `{mention, char_start, char_end, antecedent}`
//! substitutions with Unicode codepoint offsets. Rust converts offsets to byte ranges,
//! validates, and applies surviving subs RIGHT-TO-LEFT (deterministic + cacheable).
//! Invented antecedents are dropped (hallucination guard). `chunks.text` is never
//! mutated; coref rewrites the body only in `embedding_text`. Budget shared with the
//! structural-map pass (AC11).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::llm::LlmProvider;

use super::batching::batch_by_char_budget;
use super::map::{MapError, run_llm_with_retries};
use super::meta::{
    Budget, COREF_MAX_RESULTS, COREF_MAX_SUBS_PER_CHUNK, ENRICHMENT_BATCH_BYTE_BUDGET,
    ENRICHMENT_COREF_MAX_TOKENS, STRUCTURAL_MAP_MAX_FIELD_CHARS, truncate_chars,
};

/// Soft input-byte budget per coref batch. Shared with the structural-map pass so
/// the two batchers stay in sync.
const COREF_BATCH_BYTE_BUDGET: usize = ENRICHMENT_BATCH_BYTE_BUDGET;

/// Per-item byte overhead added to the batch-cost accounting.
const COREF_SEPARATOR_LEN: usize = 2;

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

/// Strict serde schema for the coref response (`deny_unknown_fields` triggers a
/// reprompt rather than silent acceptance of a garbled shape).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorefResponse {
    pub results: Vec<ChunkCoref>,
}

/// Coref substitutions for a single chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkCoref {
    /// Positional batch id supplied in the request.
    pub id: usize,
    pub subs: Vec<CorefSub>,
}

/// A single coref substitution: replace `mention` at the Unicode codepoint range
/// `[char_start, char_end)` with `antecedent`. Offsets are codepoint indices (not
/// bytes); Rust converts them to byte ranges before slicing so multibyte text is
/// handled correctly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorefSub {
    pub mention: String,
    /// Inclusive codepoint start offset of the mention.
    pub char_start: usize,
    /// Exclusive codepoint end offset of the mention.
    pub char_end: usize,
    pub antecedent: String,
}

impl CorefResponse {
    /// Parses + validates an LLM response, tolerating markdown fences / preamble.
    /// Returns `Err` on any parse miss so the caller can reprompt.
    pub fn parse_strict(body: &str) -> Result<Self, crate::LensError> {
        let json = super::meta::extract_json_object(body).unwrap_or(body);
        let mut resp = serde_json::from_str::<CorefResponse>(json)
            .map_err(|e| crate::LensError::Parse(format!("coref response invalid: {e}")))?;
        resp.bound_sizes();
        Ok(resp)
    }

    /// Caps collections + string lengths to their bounds by TRUNCATING (never
    /// rejecting) so a bloated response degrades rather than failing the source.
    fn bound_sizes(&mut self) {
        self.results.truncate(COREF_MAX_RESULTS);
        for chunk in &mut self.results {
            chunk.subs.truncate(COREF_MAX_SUBS_PER_CHUNK);
            for sub in &mut chunk.subs {
                truncate_chars(&mut sub.mention, STRUCTURAL_MAP_MAX_FIELD_CHARS);
                truncate_chars(&mut sub.antecedent, STRUCTURAL_MAP_MAX_FIELD_CHARS);
            }
        }
    }
}

fn batch_chunks<'a>(chunks: &[(usize, &'a str)]) -> Vec<Vec<(usize, &'a str)>> {
    batch_by_char_budget(
        chunks.iter().copied(),
        COREF_BATCH_BYTE_BUDGET,
        COREF_SEPARATOR_LEN,
        |&(_, text)| text.len(),
    )
}

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

/// Resolves coreferences over `chunks`, returning per-`id` surviving substitutions.
/// Shares the structural-map `Budget` so the per-job circuit-break (AC11) covers both
/// passes. Persistent malformed output degrades to empty subs (not a source failure).
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
            None => continue, // malformed×retries → degrade to empty subs for this batch
        }
    }

    Ok(out)
}

/// Applies surviving coref substitutions to `text`. Pure + deterministic; never
/// panics on bad offsets. Subs are validated (range check, mention match,
/// non-empty antecedent, allow-list membership case-insensitively) and applied
/// RIGHT-TO-LEFT so earlier offsets stay valid. Overlapping subs are skipped.
pub fn apply_substitutions(
    text: &str,
    subs: &[CorefSub],
    allowed_antecedents: &[String],
) -> String {
    let mut valid: Vec<&CorefSub> = subs
        .iter()
        .filter(|s| is_valid_sub(text, s, allowed_antecedents))
        .collect();

    if valid.is_empty() {
        return text.to_string();
    }

    valid.sort_by(|a, b| b.char_start.cmp(&a.char_start));

    let mut result = text.to_string();
    let mut last_applied_start = usize::MAX;
    for sub in valid {
        if sub.char_end > last_applied_start {
            tracing::debug!(
                mention = %sub.mention,
                "coref: dropped sub (overlaps an already-applied substitution)"
            );
            continue;
        }
        // Right-to-left order means byte offsets from the original `text` are still
        // valid against `result` up to `char_end`. Validated above → `Some`.
        let Some((byte_start, byte_end)) =
            codepoint_range_to_bytes(text, sub.char_start, sub.char_end)
        else {
            continue;
        };
        result.replace_range(byte_start..byte_end, &sub.antecedent);
        last_applied_start = sub.char_start;
    }
    result
}

/// Converts `[char_start, char_end)` codepoint indices to byte offsets in `text`.
/// Returns `None` when either index is out of range or `char_start > char_end`.
fn codepoint_range_to_bytes(
    text: &str,
    char_start: usize,
    char_end: usize,
) -> Option<(usize, usize)> {
    if char_start > char_end {
        return None;
    }
    let mut byte_start: Option<usize> = None;
    let mut byte_end: Option<usize> = None;
    for (cp_idx, (byte_idx, _)) in text.char_indices().enumerate() {
        if cp_idx == char_start {
            byte_start = Some(byte_idx);
        }
        if cp_idx == char_end {
            byte_end = Some(byte_idx);
        }
    }
    // char_indices() never yields the past-the-end index; map it manually.
    let total = text.chars().count();
    if char_start == total {
        byte_start = Some(text.len());
    }
    if char_end == total {
        byte_end = Some(text.len());
    }
    match (byte_start, byte_end) {
        (Some(s), Some(e)) => Some((s, e)),
        _ => None,
    }
}

/// Returns true if `sub` passes all validation rules (range, mention match,
/// non-empty antecedent, allow-list membership). Emits `tracing::debug!` on drop.
fn is_valid_sub(text: &str, sub: &CorefSub, allowed_antecedents: &[String]) -> bool {
    if sub.antecedent.trim().is_empty() {
        tracing::debug!(
            mention = %sub.mention,
            "coref: dropped sub (empty antecedent)"
        );
        return false;
    }
    let Some((byte_start, byte_end)) = codepoint_range_to_bytes(text, sub.char_start, sub.char_end)
    else {
        tracing::debug!(
            mention = %sub.mention,
            char_start = sub.char_start,
            char_end = sub.char_end,
            "coref: dropped sub (codepoint offsets out of range)"
        );
        return false;
    };
    if text[byte_start..byte_end] != sub.mention {
        tracing::debug!(
            mention = %sub.mention,
            "coref: dropped sub (mention does not match text at offsets)"
        );
        return false;
    }
    // Hallucination guard: case-insensitive Unicode comparison tolerates casing drift.
    let antecedent_lower = sub.antecedent.to_lowercase();
    let allowed = allowed_antecedents
        .iter()
        .any(|a| a.to_lowercase() == antecedent_lower);
    if !allowed {
        tracing::debug!(
            antecedent = %sub.antecedent,
            "coref: dropped sub (antecedent not in allow-list — hallucination guard)"
        );
    }
    allowed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enrichment::meta::{Budget, SessionBudget};
    use crate::enrichment::test_util::ScriptedProvider;

    use std::sync::atomic::Ordering;

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
        let subs = vec![sub("He", 0, 3, "Ada")];
        let allowed = vec!["Ada".to_string()];
        assert_eq!(apply_substitutions(text, &subs, &allowed), text);
    }

    #[test]
    fn codepoint_offsets_address_multibyte_chars_not_bytes() {
        let text = "café it"; // 7 codepoints; é is 2 bytes
        let subs = vec![sub("it", 5, 7, "Coffee")];
        let allowed = vec!["Coffee".to_string()];
        assert_eq!(apply_substitutions(text, &subs, &allowed), "café Coffee");
    }

    #[test]
    fn drop_sub_with_invented_antecedent_not_in_entities() {
        let text = "It is fast.";
        let subs = vec![sub("It", 0, 2, "Imaginary Corp")];
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

    #[test]
    fn multibyte_before_mention_resolves_at_codepoint_offset() {
        // HEADLINE multilingual proof: multibyte (accented) text BEFORE the
        // referential mention. With BYTE offsets the mention guard would mismatch
        // (the accented prefix occupies more bytes than codepoints) and the sub
        // would be silently dropped — coref no-ops on non-English text. With
        // CODEPOINT offsets it resolves correctly.
        // Text: "À Paris, elle a vécu." — codepoints:
        //   À(0) ' '(1) P(2) a(3) r(4) i(5) s(6) ,(7) ' '(8) e(9) l(10) l(11) e(12)
        // so "elle" is codepoints [9,13). (Byte offset of "elle" is 10 — the À is 2
        // bytes — so a byte reading would point one char early and mismatch.)
        let text = "À Paris, elle a vécu.";
        assert_eq!(text.chars().nth(9), Some('e'));
        let subs = vec![sub("elle", 9, 13, "Marie Curie")];
        let allowed = vec!["Marie Curie".to_string()];
        assert_eq!(
            apply_substitutions(text, &subs, &allowed),
            "À Paris, Marie Curie a vécu."
        );
    }

    #[test]
    fn cjk_before_mention_resolves_at_codepoint_offset() {
        // A CJK prefix (each char 3 bytes) before an ASCII mention. Codepoint
        // offsets keep the slice correct where byte offsets would not.
        // Text: "東京。it is large." — codepoints: 東(0) 京(1) 。(2) ' '(3) i(4) t(5)
        let text = "東京。 it is large.";
        assert_eq!(text.chars().nth(4), Some('i'));
        let subs = vec![sub("it", 4, 6, "Tokyo")];
        let allowed = vec!["Tokyo".to_string()];
        assert_eq!(
            apply_substitutions(text, &subs, &allowed),
            "東京。 Tokyo is large."
        );
    }

    #[test]
    fn drop_sub_with_out_of_range_codepoint_index_no_panic() {
        // A codepoint index past the end of the (multibyte) string must DROP the
        // sub, never panic.
        let text = "café"; // 4 codepoints, 5 bytes
        let subs = vec![sub("é", 3, 99, "Coffee")];
        let allowed = vec!["Coffee".to_string()];
        assert_eq!(apply_substitutions(text, &subs, &allowed), text);
    }

    #[test]
    fn antecedent_allow_list_match_is_case_insensitive() {
        let text = "It is fast.";
        let subs = vec![sub("It", 0, 2, "ACME corp")];
        let allowed = vec!["Acme Corp".to_string()];
        assert_eq!(
            apply_substitutions(text, &subs, &allowed),
            "ACME corp is fast."
        );
    }

    #[test]
    fn overlapping_subs_apply_only_one() {
        let text = "the big engine ran";
        let subs = vec![
            sub("big engine", 4, 14, "Babbage"),
            sub("engine", 8, 14, "the device"),
        ];
        let allowed = vec!["Babbage".to_string(), "the device".to_string()];
        assert_eq!(
            apply_substitutions(text, &subs, &allowed),
            "the big the device ran"
        );
    }

    #[test]
    fn parse_strict_truncates_oversized_coref_response() {
        use crate::enrichment::meta::{
            COREF_MAX_RESULTS, COREF_MAX_SUBS_PER_CHUNK, STRUCTURAL_MAP_MAX_FIELD_CHARS,
        };
        // One chunk with too many subs, and an over-long mention/antecedent.
        let long_mention = "m".repeat(STRUCTURAL_MAP_MAX_FIELD_CHARS + 100);
        let long_antecedent = "a".repeat(STRUCTURAL_MAP_MAX_FIELD_CHARS + 100);
        let mut subs: Vec<serde_json::Value> = Vec::new();
        subs.push(serde_json::json!({
            "mention": long_mention,
            "char_start": 0,
            "char_end": 1,
            "antecedent": long_antecedent,
        }));
        for _ in 0..(COREF_MAX_SUBS_PER_CHUNK + 50) {
            subs.push(serde_json::json!({
                "mention": "x",
                "char_start": 0,
                "char_end": 1,
                "antecedent": "y",
            }));
        }
        let results: Vec<serde_json::Value> = (0..(COREF_MAX_RESULTS + 50))
            .map(|i| serde_json::json!({ "id": i, "subs": subs.clone() }))
            .collect();
        let body = serde_json::json!({ "results": results }).to_string();

        let resp = CorefResponse::parse_strict(&body).expect("oversized response truncates");
        assert_eq!(resp.results.len(), COREF_MAX_RESULTS);
        assert_eq!(resp.results[0].subs.len(), COREF_MAX_SUBS_PER_CHUNK);
        assert_eq!(
            resp.results[0].subs[0].mention.chars().count(),
            STRUCTURAL_MAP_MAX_FIELD_CHARS
        );
        assert_eq!(
            resp.results[0].subs[0].antecedent.chars().count(),
            STRUCTURAL_MAP_MAX_FIELD_CHARS
        );
    }

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
        let big = "x ".repeat(COREF_BATCH_BYTE_BUDGET);
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
