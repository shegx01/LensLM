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

/// Structural-map / coref prompt version (AC9 cache key component). Bump on any
/// prompt change to invalidate prior enrichments. `2`: real coref replaced the
/// static "[Resolve pronouns…]" hint clause — prior enrichments must re-run.
pub const ENRICHMENT_PROMPT_VERSION: u32 = 2;

/// Minimum total tokens for structural-map enrichment (Decision C).
pub const ENRICHMENT_SIZE_GATE_TOKENS: usize = 2_000;

/// Per-job output-token ceiling (Decision C). Checked BEFORE each `generate()`.
pub const ENRICHMENT_MAX_TOKENS_PER_JOB: u32 = 50_000;

/// Per-session output-token ceiling (Decision C).
pub const ENRICHMENT_MAX_TOKENS_PER_SESSION: u32 = 500_000;

/// Per-job LLM-call ceiling (Decision C). Tests tighten this via [`Budget::with_caps`].
pub const ENRICHMENT_MAX_CALLS_PER_JOB: u32 = 8;

/// Structural-map reprompt retries on malformed JSON (Decision C). `2` reprompts
/// → up to 3 total attempts per map call; then degrade to context-prefix-only.
pub const ENRICHMENT_MAX_RETRIES: u32 = 2;

/// Soft input-byte budget per enrichment LLM batch (shared by the structural-map
/// and coref batchers so they stay in sync). A single item exceeding the budget
/// forms its own batch; the provider truncates if needed.
pub(super) const ENRICHMENT_BATCH_BYTE_BUDGET: usize = 8 * 1024;

/// Max output tokens requested for a single structural-map LLM call.
pub const ENRICHMENT_MAP_MAX_TOKENS: u32 = 1_024;

/// Max output tokens for a single coref-resolution LLM call (also the pre-dispatch
/// projected cost for AC11 budget checks).
pub const ENRICHMENT_COREF_MAX_TOKENS: u32 = 1_024;

/// Embedder input window (nomic-embed-text-v1.5). `embedding_text` must fit after
/// the hard-applied `"search_document: "` prefix (`embedder.rs:207`) is accounted for.
pub const EMBEDDER_TOKEN_WINDOW: usize = 2_048;

/// Token cost of the hard-applied `"search_document: "` prefix (`embedder.rs:207`).
pub const SEARCH_DOCUMENT_PREFIX_TOKENS: usize = 4;

/// Size bounds applied after parse by TRUNCATING (not rejecting) — defense-in-depth
/// against a bloated LLM response inflating SQLite.
pub const STRUCTURAL_MAP_MAX_ENTITIES: usize = 200;
pub const STRUCTURAL_MAP_MAX_DEFINITIONS: usize = 200;
pub const STRUCTURAL_MAP_MAX_DATES: usize = 200;
pub const STRUCTURAL_MAP_MAX_SUMMARY_CHARS: usize = 4_000;
/// Per-field char cap for entity/date/definition strings; also reused for coref
/// `mention`/`antecedent` fields.
pub const STRUCTURAL_MAP_MAX_FIELD_CHARS: usize = 1_000;
pub const COREF_MAX_RESULTS: usize = 256;
pub const COREF_MAX_SUBS_PER_CHUNK: usize = 128;

/// Per-chunk cap on entities considered for co-occurrence edges (M13). Overflow
/// beyond the first `N` (insertion order) is dropped and counted for observability.
pub const CO_OCCURRENCE_MAX_ENTITIES: usize = 30;

/// Per-doc structural map produced by the LLM (AC4). Unknown fields are rejected
/// (`deny_unknown_fields`) so a garbled shape triggers a reprompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralMap {
    pub entities: Vec<String>,
    pub definitions: Vec<Definition>,
    pub dates: Vec<String>,
    pub summary: String,
}

/// A term/definition pair inside a [`StructuralMap`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Definition {
    pub term: String,
    pub definition: String,
}

impl StructuralMap {
    /// Parses + validates an LLM response, tolerating markdown fences / preamble.
    /// Returns `Err` on any parse miss so the caller can reprompt.
    pub fn parse_strict(body: &str) -> Result<Self, crate::LensError> {
        let json = extract_json_object(body).unwrap_or(body);
        let mut map = serde_json::from_str::<StructuralMap>(json)
            .map_err(|e| crate::LensError::Parse(format!("structural map invalid: {e}")))?;
        map.bound_sizes();
        Ok(map)
    }

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

/// Truncates `s` in place to at most `max_chars` codepoints on a UTF-8 char boundary.
pub(super) fn truncate_chars(s: &mut String, max_chars: usize) {
    if let Some((byte_idx, _)) = s.char_indices().nth(max_chars) {
        s.truncate(byte_idx);
    }
}

/// Extracts the first balanced `{...}` JSON object from `text` so a response
/// wrapped in markdown fences or preamble still parses. Returns `None` if not found.
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

/// Composite enrichment cache key (AC9): `hash(content_hash ‖ model ‖ prompt_version ‖ coref)`.
/// A match on re-enqueue short-circuits the LLM pass entirely.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKeyParts {
    pub content_hash: String,
    pub llm_model_id: String,
    pub prompt_version: u32,
    pub coref_strategy: String,
}

impl CacheKeyParts {
    /// SHA-256 over the four components joined with NUL (unambiguous separator).
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

/// JSON persisted to `sources.enrichment_meta` (AC9/AC11).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct EnrichmentMeta {
    pub cache_key: String,
    /// `"ok"` / `"fallback"` / `"skipped"`.
    #[serde(default)]
    pub map_quality: String,
    #[serde(default)]
    pub budget_exceeded: bool,
    #[serde(default)]
    pub tokens_spent: u32,
    #[serde(default)]
    pub calls_made: u32,
    /// Human-readable failure reason (issue #90). `#[serde(default)]` for
    /// backward compat — old rows without this field deserialize as `None`.
    #[serde(default)]
    pub failure_reason: Option<String>,
}

/// `map_quality` value: structural map validated successfully.
pub const MAP_QUALITY_OK: &str = "ok";
/// `map_quality` value: structural map degraded to context-prefix-only.
pub const MAP_QUALITY_FALLBACK: &str = "fallback";
/// `map_quality` value: structural map skipped (non-prose kind / size-gated).
pub const MAP_QUALITY_SKIPPED: &str = "skipped";

/// Per-session token + call counters shared across all jobs in the worker's lifetime
/// (AC11). `Arc<AtomicU32>` so they survive across jobs and are injectable in tests.
#[derive(Debug, Clone)]
pub struct SessionBudget {
    tokens: Arc<AtomicU32>,
    calls: Arc<AtomicU32>,
    max_tokens: u32,
}

impl Default for SessionBudget {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionBudget {
    pub fn new() -> Self {
        Self {
            tokens: Arc::new(AtomicU32::new(0)),
            calls: Arc::new(AtomicU32::new(0)),
            max_tokens: ENRICHMENT_MAX_TOKENS_PER_SESSION,
        }
    }

    /// Custom token ceiling (AC11 test seam).
    pub fn with_max_tokens(max_tokens: u32) -> Self {
        Self {
            max_tokens,
            ..Self::new()
        }
    }

    pub fn tokens_used(&self) -> u32 {
        self.tokens.load(Ordering::Relaxed)
    }

    pub fn calls_made(&self) -> u32 {
        self.calls.load(Ordering::Relaxed)
    }
}

/// Per-job budget layered over [`SessionBudget`] (AC11). `check_before_dispatch`
/// is called before every `generate()`; a breach short-circuits without dispatching.
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
    Ok,
    Exceeded,
}

impl Budget {
    pub fn new(session: SessionBudget) -> Self {
        Self {
            session,
            job_tokens: 0,
            job_calls: 0,
            max_tokens_per_job: ENRICHMENT_MAX_TOKENS_PER_JOB,
            max_calls_per_job: ENRICHMENT_MAX_CALLS_PER_JOB,
        }
    }

    /// Custom per-job ceilings (AC11 test seam).
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

    /// Checks whether the next call fits both ceilings without mutating counters.
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

    /// Records a dispatched call's spend. Call once per successful `generate()`.
    pub fn record(&mut self, tokens_used: u32) {
        self.job_calls = self.job_calls.saturating_add(1);
        self.job_tokens = self.job_tokens.saturating_add(tokens_used);
        self.session
            .tokens
            .fetch_add(tokens_used, Ordering::Relaxed);
        self.session.calls.fetch_add(1, Ordering::Relaxed);
    }

    pub fn job_tokens(&self) -> u32 {
        self.job_tokens
    }

    pub fn job_calls(&self) -> u32 {
        self.job_calls
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut a = parts();
        a.content_hash = "a".into();
        a.llm_model_id = "bc".into();
        let mut b = parts();
        b.content_hash = "ab".into();
        b.llm_model_id = "c".into();
        assert_ne!(a.compute(), b.compute());
    }

    #[test]
    fn per_job_call_ceiling_circuit_breaks() {
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
        assert_eq!(budget.check_before_dispatch(100), BudgetCheck::Exceeded);
    }

    #[test]
    fn per_session_ceiling_accumulates_across_jobs() {
        let session = SessionBudget::with_max_tokens(150);
        let mut job1 = Budget::new(session.clone());
        assert_eq!(job1.check_before_dispatch(100), BudgetCheck::Ok);
        job1.record(100);
        let job2 = Budget::new(session.clone());
        assert_eq!(job2.check_before_dispatch(100), BudgetCheck::Exceeded);
        assert_eq!(session.tokens_used(), 100);
        assert_eq!(session.calls_made(), 1);
    }

    #[test]
    fn enrichment_meta_failure_reason_roundtrip() {
        let meta = EnrichmentMeta {
            cache_key: "k".to_string(),
            failure_reason: Some("Model 'x' not found in Ollama".to_string()),
            ..EnrichmentMeta::default()
        };
        let json = serde_json::to_string(&meta).unwrap();
        let back: EnrichmentMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.failure_reason,
            Some("Model 'x' not found in Ollama".to_string())
        );
    }

    #[test]
    fn enrichment_meta_backward_compat() {
        let old_json = r#"{
            "cache_key": "k",
            "map_quality": "ok",
            "budget_exceeded": false,
            "tokens_spent": 10,
            "calls_made": 2
        }"#;
        let meta: EnrichmentMeta = serde_json::from_str(old_json).unwrap();
        assert_eq!(meta.failure_reason, None);
        assert_eq!(meta.cache_key, "k");
        assert_eq!(meta.tokens_spent, 10);
    }

    #[test]
    fn enrichment_meta_explicit_none() {
        let json = r#"{
            "cache_key": "k",
            "map_quality": "ok",
            "budget_exceeded": false,
            "tokens_spent": 0,
            "calls_made": 0,
            "failure_reason": null
        }"#;
        let meta: EnrichmentMeta = serde_json::from_str(json).unwrap();
        assert_eq!(meta.failure_reason, None);
    }
}
