//! Enrichment tuning constants, the composite cache key (AC9), the per-job /
//! per-session token+call budget with circuit-break (AC11), and the
//! [`EnrichmentMeta`] JSON persisted to `sources.enrichment_meta`.
//!
//! This module is deliberately free of I/O: every piece here is a pure function
//! or a plain counter so the Step-4 LLM phase can be unit-tested against a mock
//! [`LlmProvider`](crate::llm::LlmProvider) with a call-counter, with no DB,
//! embedder, or network in the loop.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Decision C — concrete tuning constants (defaults; tunable later)
// ---------------------------------------------------------------------------

/// Structural-map / coref prompt version. A component of the composite cache key
/// (AC9, lock #5): bumping it changes the key so an already-enriched source
/// re-runs the LLM pass. Bump on ANY prompt change.
///
/// `2`: real LLM-driven coref-substitution resolution replaced the static prefix
/// "[Resolve pronouns…]" hint clause — the coref prompt + schema (and thus the
/// composed `embedding_text` body) changed, so the composite key MUST invalidate
/// every prior enrichment.
pub const ENRICHMENT_PROMPT_VERSION: u32 = 2;

/// Skip enrichment for docs whose total token count is below this gate (Decision
/// C): small docs don't benefit from a structural map and aren't worth an LLM
/// call.
pub const ENRICHMENT_SIZE_GATE_TOKENS: usize = 2_000;

/// Per-job output-token ceiling (Decision C). Checked BEFORE each `generate()`.
pub const ENRICHMENT_MAX_TOKENS_PER_JOB: u32 = 50_000;

/// Per-session output-token ceiling (Decision C). The session counter accumulates
/// across every job in a worker's lifetime.
pub const ENRICHMENT_MAX_TOKENS_PER_SESSION: u32 = 500_000;

/// Per-job LLM-call ceiling (Decision C). The AC11 "2-call job, budget=1" test
/// tightens this seam to 1 via [`Budget::with_caps`].
pub const ENRICHMENT_MAX_CALLS_PER_JOB: u32 = 8;

/// Structural-map reprompt retries on malformed JSON (Decision C). `2` reprompts
/// → up to 3 total attempts per map call; then degrade to context-prefix-only.
pub const ENRICHMENT_MAX_RETRIES: u32 = 2;

/// Max output tokens requested for a single structural-map LLM call.
pub const ENRICHMENT_MAP_MAX_TOKENS: u32 = 1_024;

/// Max output tokens requested for a single coref-resolution LLM call (also the
/// projected pre-dispatch budget cost for a coref batch, AC11). Coref output is a
/// compact substitution list, so it is modest.
pub const ENRICHMENT_COREF_MAX_TOKENS: u32 = 1_024;

/// The embedder's input token window (nomic-embed-text-v1.5 = 2048). The composed
/// `embedding_text` must fit within this window AFTER the hard-applied
/// `"search_document: "` prefix is accounted for (`embedder.rs:207`).
pub const EMBEDDER_TOKEN_WINDOW: usize = 2_048;

/// Token cost of the embedder's hard-applied `"search_document: "` prefix
/// (`embedder.rs:207`). The composed `embedding_text` budget is
/// `EMBEDDER_TOKEN_WINDOW - SEARCH_DOCUMENT_PREFIX_TOKENS`. Conservatively `4`
/// (the wordpiece tokenization of `search_document: ` is small and fixed).
pub const SEARCH_DOCUMENT_PREFIX_TOKENS: usize = 4;

// ---------------------------------------------------------------------------
// StructuralMap size bounds — defense-in-depth against a prompt-injected/bloated
// LLM response inflating SQLite. Applied AFTER parse by TRUNCATING (not rejecting)
// so enrichment degrades rather than failing the source.
// ---------------------------------------------------------------------------

/// Max entities retained in a [`StructuralMap`]; the vec is truncated to this.
pub const STRUCTURAL_MAP_MAX_ENTITIES: usize = 200;
/// Max definitions retained in a [`StructuralMap`]; the vec is truncated to this.
pub const STRUCTURAL_MAP_MAX_DEFINITIONS: usize = 200;
/// Max dates retained in a [`StructuralMap`]; the vec is truncated to this.
pub const STRUCTURAL_MAP_MAX_DATES: usize = 200;
/// Max chars retained in a [`StructuralMap`]'s `summary`; truncated to this.
pub const STRUCTURAL_MAP_MAX_SUMMARY_CHARS: usize = 4_000;
/// Max chars retained in any individual string field (entity / date / definition
/// term + definition); each over-long string is truncated to this.
pub const STRUCTURAL_MAP_MAX_FIELD_CHARS: usize = 1_000;

// ---------------------------------------------------------------------------
// StructuralMap — the strict serde schema validated from the LLM JSON (AC4)
// ---------------------------------------------------------------------------

/// The per-doc structural map produced by the LLM (AC4) and stored as JSON in
/// `chunks.enrichment` on the representative parent row.
///
/// Validation is strict: the JSON MUST parse into exactly these fields. Unknown
/// fields are rejected (`deny_unknown_fields`) so a hallucinated/garbled shape
/// triggers a reprompt rather than being silently accepted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralMap {
    /// Named entities mentioned in the document.
    pub entities: Vec<String>,
    /// Term → definition pairs the document establishes.
    pub definitions: Vec<Definition>,
    /// Significant dates the document references.
    pub dates: Vec<String>,
    /// A short prose summary of the document.
    pub summary: String,
}

/// A single term/definition pair inside a [`StructuralMap`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Definition {
    /// The term being defined.
    pub term: String,
    /// Its definition.
    pub definition: String,
}

impl StructuralMap {
    /// Parses + strictly validates an LLM response body as a [`StructuralMap`].
    ///
    /// Tolerates a response that wraps the JSON in markdown fences or surrounding
    /// prose by extracting the first balanced `{...}` object before parsing
    /// (small, local models frequently add a "Here is the JSON:" preamble).
    /// Returns `Err` on any parse/validation miss so the caller can reprompt.
    pub fn parse_strict(body: &str) -> Result<Self, crate::LensError> {
        let json = extract_json_object(body).unwrap_or(body);
        let mut map = serde_json::from_str::<StructuralMap>(json)
            .map_err(|e| crate::LensError::Parse(format!("structural map invalid: {e}")))?;
        map.bound_sizes();
        Ok(map)
    }

    /// Caps the collections + string lengths to the `STRUCTURAL_MAP_MAX_*`
    /// bounds, TRUNCATING (never rejecting) so a bloated/prompt-injected LLM
    /// response can't inflate SQLite while enrichment still degrades gracefully.
    fn bound_sizes(&mut self) {
        self.entities.truncate(STRUCTURAL_MAP_MAX_ENTITIES);
        self.definitions.truncate(STRUCTURAL_MAP_MAX_DEFINITIONS);
        self.dates.truncate(STRUCTURAL_MAP_MAX_DATES);
        for entity in &mut self.entities {
            truncate_chars(entity, STRUCTURAL_MAP_MAX_FIELD_CHARS);
        }
        for date in &mut self.dates {
            truncate_chars(date, STRUCTURAL_MAP_MAX_FIELD_CHARS);
        }
        for def in &mut self.definitions {
            truncate_chars(&mut def.term, STRUCTURAL_MAP_MAX_FIELD_CHARS);
            truncate_chars(&mut def.definition, STRUCTURAL_MAP_MAX_FIELD_CHARS);
        }
        truncate_chars(&mut self.summary, STRUCTURAL_MAP_MAX_SUMMARY_CHARS);
    }
}

/// Truncates `s` in place to at most `max_chars` characters, on a UTF-8 char
/// boundary (never panics on multibyte input).
fn truncate_chars(s: &mut String, max_chars: usize) {
    if let Some((byte_idx, _)) = s.char_indices().nth(max_chars) {
        s.truncate(byte_idx);
    }
}

/// Extracts the first balanced top-level `{...}` JSON object from `text`, so a
/// response wrapped in ```json fences or chat preamble still validates. Returns
/// `None` when no balanced object is found (caller falls back to the raw body).
///
/// Shared by [`StructuralMap::parse_strict`] and the coref parser
/// ([`super::coref::CorefResponse::parse_strict`]).
pub(super) fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let bytes = text.as_bytes();
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Composite cache key (AC9, lock #5)
// ---------------------------------------------------------------------------

/// The composite enrichment cache key persisted in `sources.enrichment_meta`.
///
/// Key = `hash(content_hash ‖ llm_model_id ‖ prompt_version ‖ coref_strategy)`.
/// On re-enqueue the worker recomputes this from the live source + provider +
/// config; if it matches the persisted key the LLM pass is short-circuited
/// (status stays `enriched`, ZERO LLM calls — AC9).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKeyParts {
    /// The source's `content_hash` (changes when the document body changes).
    pub content_hash: String,
    /// The LLM model id used to enrich (`LlmProvider::model_id`).
    pub llm_model_id: String,
    /// [`ENRICHMENT_PROMPT_VERSION`] at enrichment time.
    pub prompt_version: u32,
    /// The coref strategy used (`"llm_inline"` / `"none"`).
    pub coref_strategy: String,
}

impl CacheKeyParts {
    /// Computes the composite key: a hex SHA-256 over the four components joined
    /// with a NUL separator (NUL can't appear in any component, so the join is
    /// unambiguous — no `a‖bc` vs `ab‖c` collision).
    pub fn compute(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.content_hash.as_bytes());
        hasher.update([0u8]);
        hasher.update(self.llm_model_id.as_bytes());
        hasher.update([0u8]);
        hasher.update(self.prompt_version.to_le_bytes());
        hasher.update([0u8]);
        hasher.update(self.coref_strategy.as_bytes());
        crate::hex_encode(&hasher.finalize())
    }
}

/// The JSON persisted to `sources.enrichment_meta` (AC9/AC11).
///
/// Records the composite cache key for short-circuiting on re-enqueue, the
/// quality of the structural map (`"ok"` / `"fallback"` / `"skipped"`), and —
/// when a budget circuit-break fired — the `budget_exceeded` flag (AC11) so the
/// failure is never silent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct EnrichmentMeta {
    /// The composite cache key (`CacheKeyParts::compute`).
    pub cache_key: String,
    /// Structural-map quality: `"ok"` (validated), `"fallback"` (malformed×3 →
    /// context-prefix only), or `"skipped"` (non-prose / size-gated).
    #[serde(default)]
    pub map_quality: String,
    /// Set `true` when a per-job / per-session budget check circuit-broke this
    /// job to `failed` (AC11). Defaults `false`.
    #[serde(default)]
    pub budget_exceeded: bool,
    /// Total LLM tokens this job consumed (observability).
    #[serde(default)]
    pub tokens_spent: u32,
    /// Total LLM calls this job dispatched (observability).
    #[serde(default)]
    pub calls_made: u32,
}

// ---------------------------------------------------------------------------
// Budget + circuit-break (AC11)
// ---------------------------------------------------------------------------

/// `map_quality` value: structural map validated successfully.
pub const MAP_QUALITY_OK: &str = "ok";
/// `map_quality` value: structural map degraded to context-prefix-only after
/// exhausting reprompts (Open Question 3 → status stays `enriched`).
pub const MAP_QUALITY_FALLBACK: &str = "fallback";
/// `map_quality` value: structural map skipped (non-prose kind / size-gated).
pub const MAP_QUALITY_SKIPPED: &str = "skipped";

/// Per-session token + call counters, OWNED by the worker (AC11, Decision C).
///
/// `Arc<AtomicU32>` so they survive across every job in a worker's lifetime and
/// are injectable into the test worker. Reset to 0 once when the worker spawns.
#[derive(Debug, Clone)]
pub struct SessionBudget {
    /// Accumulated output tokens across all jobs this session.
    tokens: Arc<AtomicU32>,
    /// Accumulated LLM calls across all jobs this session.
    calls: Arc<AtomicU32>,
    /// Session token ceiling (default [`ENRICHMENT_MAX_TOKENS_PER_SESSION`]).
    max_tokens: u32,
}

impl Default for SessionBudget {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionBudget {
    /// A fresh session budget with the default ceiling, counters at 0.
    pub fn new() -> Self {
        Self {
            tokens: Arc::new(AtomicU32::new(0)),
            calls: Arc::new(AtomicU32::new(0)),
            max_tokens: ENRICHMENT_MAX_TOKENS_PER_SESSION,
        }
    }

    /// A session budget with a custom token ceiling (test seam — AC11 sets a
    /// tiny budget so the mock provider sees exactly the admitted call count).
    pub fn with_max_tokens(max_tokens: u32) -> Self {
        Self {
            max_tokens,
            ..Self::new()
        }
    }

    /// Tokens consumed so far this session.
    pub fn tokens_used(&self) -> u32 {
        self.tokens.load(Ordering::Relaxed)
    }

    /// Calls dispatched so far this session.
    pub fn calls_made(&self) -> u32 {
        self.calls.load(Ordering::Relaxed)
    }
}

/// A per-job budget view layered over the shared [`SessionBudget`] (AC11).
///
/// Per-job counters are plain locals reset at job start; the session counters are
/// the shared atomics. [`Budget::check_before_dispatch`] is called BEFORE every
/// `generate()` and short-circuits (never dispatching) when the next call's
/// projected spend would breach EITHER the per-job or per-session ceiling.
pub struct Budget {
    session: SessionBudget,
    job_tokens: u32,
    job_calls: u32,
    max_tokens_per_job: u32,
    max_calls_per_job: u32,
}

/// Outcome of a pre-dispatch budget check (AC11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetCheck {
    /// The next call fits both ceilings — dispatch is permitted.
    Ok,
    /// The next call would breach a ceiling — circuit-break, do NOT dispatch.
    Exceeded,
}

impl Budget {
    /// A per-job budget over `session` using the default per-job ceilings.
    pub fn new(session: SessionBudget) -> Self {
        Self {
            session,
            job_tokens: 0,
            job_calls: 0,
            max_tokens_per_job: ENRICHMENT_MAX_TOKENS_PER_JOB,
            max_calls_per_job: ENRICHMENT_MAX_CALLS_PER_JOB,
        }
    }

    /// A per-job budget with custom per-job ceilings (AC11 test seam — set
    /// `max_calls_per_job = 1` so a 2-call job circuit-breaks on the 2nd call).
    pub fn with_caps(
        session: SessionBudget,
        max_tokens_per_job: u32,
        max_calls_per_job: u32,
    ) -> Self {
        Self {
            session,
            job_tokens: 0,
            job_calls: 0,
            max_tokens_per_job,
            max_calls_per_job,
        }
    }

    /// Checks — BEFORE dispatching — whether the next call (projected to cost
    /// `projected_tokens`) fits BOTH the per-job and per-session ceilings. Does
    /// NOT mutate any counter; the caller dispatches only on [`BudgetCheck::Ok`]
    /// and then records the actual spend via [`Budget::record`].
    pub fn check_before_dispatch(&self, projected_tokens: u32) -> BudgetCheck {
        let next_calls = self.job_calls.saturating_add(1);
        if next_calls > self.max_calls_per_job {
            return BudgetCheck::Exceeded;
        }
        let next_job_tokens = self.job_tokens.saturating_add(projected_tokens);
        if next_job_tokens > self.max_tokens_per_job {
            return BudgetCheck::Exceeded;
        }
        let next_session_tokens = self
            .session
            .tokens
            .load(Ordering::Relaxed)
            .saturating_add(projected_tokens);
        if next_session_tokens > self.session.max_tokens {
            return BudgetCheck::Exceeded;
        }
        BudgetCheck::Ok
    }

    /// Records an actual dispatched call's spend against both the per-job and the
    /// shared per-session counters. Call exactly once per successful `generate()`.
    pub fn record(&mut self, tokens_used: u32) {
        self.job_calls = self.job_calls.saturating_add(1);
        self.job_tokens = self.job_tokens.saturating_add(tokens_used);
        self.session
            .tokens
            .fetch_add(tokens_used, Ordering::Relaxed);
        self.session.calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Per-job tokens spent so far.
    pub fn job_tokens(&self) -> u32 {
        self.job_tokens
    }

    /// Per-job calls dispatched so far.
    pub fn job_calls(&self) -> u32 {
        self.job_calls
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- StructuralMap strict validation (AC4) ------------------------------

    fn valid_map_json() -> &'static str {
        r#"{
            "entities": ["Ada Lovelace", "Charles Babbage"],
            "definitions": [{"term": "Analytical Engine", "definition": "a mechanical computer"}],
            "dates": ["1843"],
            "summary": "An overview of early computing."
        }"#
    }

    #[test]
    fn parse_strict_accepts_valid_map() {
        let map = StructuralMap::parse_strict(valid_map_json()).expect("valid");
        assert_eq!(map.entities.len(), 2);
        assert_eq!(map.definitions[0].term, "Analytical Engine");
        assert_eq!(map.dates, vec!["1843".to_string()]);
        assert!(map.summary.contains("computing"));
    }

    #[test]
    fn parse_strict_tolerates_markdown_fenced_json() {
        let fenced = format!(
            "Sure! Here is the JSON:\n```json\n{}\n```\n",
            valid_map_json()
        );
        let map = StructuralMap::parse_strict(&fenced).expect("fenced still parses");
        assert_eq!(map.entities.len(), 2);
    }

    #[test]
    fn parse_strict_rejects_malformed() {
        assert!(StructuralMap::parse_strict("not json at all").is_err());
        // Missing required field `summary`.
        assert!(
            StructuralMap::parse_strict(r#"{"entities":[],"definitions":[],"dates":[]}"#).is_err()
        );
        // Unknown field rejected by deny_unknown_fields.
        assert!(
            StructuralMap::parse_strict(
                r#"{"entities":[],"definitions":[],"dates":[],"summary":"x","extra":1}"#
            )
            .is_err()
        );
    }

    #[test]
    fn parse_strict_truncates_oversized_map() {
        // Build a map that blows past every cap, then assert it's truncated (not
        // rejected) to the STRUCTURAL_MAP_MAX_* bounds.
        let entities: Vec<String> = (0..STRUCTURAL_MAP_MAX_ENTITIES + 50)
            .map(|i| format!("entity-{i}"))
            .collect();
        let dates: Vec<String> = (0..STRUCTURAL_MAP_MAX_DATES + 50)
            .map(|i| format!("date-{i}"))
            .collect();
        let definitions: Vec<serde_json::Value> = (0..STRUCTURAL_MAP_MAX_DEFINITIONS + 50)
            .map(|i| serde_json::json!({"term": format!("t{i}"), "definition": format!("d{i}")}))
            .collect();
        let long_entity = "e".repeat(STRUCTURAL_MAP_MAX_FIELD_CHARS + 100);
        let long_summary = "s".repeat(STRUCTURAL_MAP_MAX_SUMMARY_CHARS + 100);
        let long_term = "u".repeat(STRUCTURAL_MAP_MAX_FIELD_CHARS + 100);

        let mut entities_with_long = entities.clone();
        entities_with_long[0] = long_entity;
        let mut defs_with_long = definitions.clone();
        defs_with_long[0] = serde_json::json!({"term": long_term, "definition": "ok"});

        let json = serde_json::json!({
            "entities": entities_with_long,
            "definitions": defs_with_long,
            "dates": dates,
            "summary": long_summary,
        })
        .to_string();

        let map = StructuralMap::parse_strict(&json).expect("oversized map truncates, not rejects");
        assert_eq!(map.entities.len(), STRUCTURAL_MAP_MAX_ENTITIES);
        assert_eq!(map.definitions.len(), STRUCTURAL_MAP_MAX_DEFINITIONS);
        assert_eq!(map.dates.len(), STRUCTURAL_MAP_MAX_DATES);
        assert_eq!(
            map.entities[0].chars().count(),
            STRUCTURAL_MAP_MAX_FIELD_CHARS
        );
        assert_eq!(
            map.definitions[0].term.chars().count(),
            STRUCTURAL_MAP_MAX_FIELD_CHARS
        );
        assert_eq!(
            map.summary.chars().count(),
            STRUCTURAL_MAP_MAX_SUMMARY_CHARS
        );
    }

    // --- composite cache key (AC9) ------------------------------------------

    fn parts() -> CacheKeyParts {
        CacheKeyParts {
            content_hash: "abc123".to_string(),
            llm_model_id: "llama3".to_string(),
            prompt_version: ENRICHMENT_PROMPT_VERSION,
            coref_strategy: "llm_inline".to_string(),
        }
    }

    #[test]
    fn cache_key_is_stable_for_identical_inputs() {
        assert_eq!(parts().compute(), parts().compute());
    }

    #[test]
    fn cache_key_changes_when_prompt_version_bumps() {
        let base = parts().compute();
        let mut bumped = parts();
        bumped.prompt_version = ENRICHMENT_PROMPT_VERSION + 1;
        assert_ne!(base, bumped.compute());
    }

    #[test]
    fn cache_key_changes_on_content_model_or_coref() {
        let base = parts().compute();
        let mut p = parts();
        p.content_hash = "different".into();
        assert_ne!(base, p.compute());
        let mut p = parts();
        p.llm_model_id = "gpt-4o".into();
        assert_ne!(base, p.compute());
        let mut p = parts();
        p.coref_strategy = "none".into();
        assert_ne!(base, p.compute());
    }

    #[test]
    fn cache_key_join_is_unambiguous() {
        // "a" ‖ "bc" must differ from "ab" ‖ "c" (NUL separator guards this).
        let mut a = parts();
        a.content_hash = "a".into();
        a.llm_model_id = "bc".into();
        let mut b = parts();
        b.content_hash = "ab".into();
        b.llm_model_id = "c".into();
        assert_ne!(a.compute(), b.compute());
    }

    // --- budget + circuit-break (AC11) --------------------------------------

    #[test]
    fn per_job_call_ceiling_circuit_breaks() {
        // AC11 seam: budget admits exactly 1 call.
        let mut budget = Budget::with_caps(SessionBudget::new(), 1_000_000, 1);
        assert_eq!(budget.check_before_dispatch(100), BudgetCheck::Ok);
        budget.record(100);
        // The SECOND call must be refused before dispatch.
        assert_eq!(budget.check_before_dispatch(100), BudgetCheck::Exceeded);
        assert_eq!(budget.job_calls(), 1);
    }

    #[test]
    fn per_job_token_ceiling_circuit_breaks() {
        let mut budget = Budget::with_caps(SessionBudget::new(), 150, 100);
        assert_eq!(budget.check_before_dispatch(100), BudgetCheck::Ok);
        budget.record(100);
        // 100 + 100 = 200 > 150 → refused.
        assert_eq!(budget.check_before_dispatch(100), BudgetCheck::Exceeded);
    }

    #[test]
    fn per_session_ceiling_accumulates_across_jobs() {
        let session = SessionBudget::with_max_tokens(150);
        // Job 1 spends 100 (session now 100).
        let mut job1 = Budget::new(session.clone());
        assert_eq!(job1.check_before_dispatch(100), BudgetCheck::Ok);
        job1.record(100);
        // Job 2: per-job is fresh (0), but session is at 100 → 100 + 100 > 150.
        let job2 = Budget::new(session.clone());
        assert_eq!(job2.check_before_dispatch(100), BudgetCheck::Exceeded);
        assert_eq!(session.tokens_used(), 100);
        assert_eq!(session.calls_made(), 1);
    }
}
