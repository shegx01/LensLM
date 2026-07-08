//! `embedding_text` composition (AC5) and the coref strategy enum.
//!
//! Each chunk's `embedding_text` = context prefix + canonical body. `chunks.text`
//! is never mutated; `embedding_text` is a separate column read via
//! `COALESCE(embedding_text, text)` in Step 5.
//!
//! Truncation invariant: when the composed text would exceed the embedder window,
//! the PREFIX is dropped first, never the body. Token accounting uses the nomic
//! tokenizer in production and a whitespace-word approximation in pure-logic tests.

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use super::meta::{EMBEDDER_TOKEN_WINDOW, SEARCH_DOCUMENT_PREFIX_TOKENS};

/// Coreference-resolution strategy for `embedding_text` composition (AC9 cache key
/// component). Serializes to stable snake_case; `"dedicated_model"` (legacy) maps
/// to `LlmInline` for backward compat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorefStrategy {
    /// No coref — body embedded verbatim under the context prefix.
    None,
    /// LLM-driven substitutions applied to the body (default). The prefix is
    /// identical to `None`; only the body changes.
    #[default]
    LlmInline,
}

impl CorefStrategy {
    /// Stable AC9 cache-key string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::LlmInline => "llm_inline",
        }
    }

    /// Parses the persisted/config string. Unknown values (including the legacy
    /// `"dedicated_model"`) default to `LlmInline`.
    pub fn from_config(value: &str) -> Self {
        match value {
            "none" => Self::None,
            _ => Self::LlmInline,
        }
    }
}

/// Opt-in strategy for the LLM relation-extraction pass (#154), mirroring
/// [`CorefStrategy`]. Default `Off` so shipping causes zero behavioral change and
/// zero cache churn (see [`CacheKeyParts::compute`](super::meta::CacheKeyParts)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationsStrategy {
    /// No relation extraction (default).
    #[default]
    Off,
    /// Extract typed semantic relations with the configured LLM.
    On,
}

impl RelationsStrategy {
    /// Stable AC2 cache-key string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::On => "on",
        }
    }

    /// Parses the persisted/config string. Unknown values default to `Off`.
    pub fn from_config(value: &str) -> Self {
        match value {
            "on" => Self::On,
            _ => Self::Off,
        }
    }
}

/// Counts tokens using the tokenizer when supplied, else whitespace-word count.
/// A tokenizer failure falls back to the word approximation (never panics).
pub(crate) fn count_tokens(text: &str, tokenizer: Option<&Tokenizer>) -> usize {
    match tokenizer {
        Some(tk) => tk
            .encode(text, false)
            .map(|e| e.len())
            .unwrap_or_else(|_| text.split_whitespace().count()),
        None => text.split_whitespace().count(),
    }
}

/// Composes the context prefix: `"[Document: {summary}] [Section: {path}] "`.
/// Empty clauses are omitted. Coref does not affect the prefix — resolution
/// happens in the body; the prefix is the same regardless of strategy.
pub fn compose_prefix(doc_summary: &str, section_path: &str) -> String {
    let mut clauses: Vec<String> = Vec::with_capacity(2);
    let summary = doc_summary.trim();
    if !summary.is_empty() {
        clauses.push(format!("[Document: {summary}]"));
    }
    let section = section_path.trim();
    if !section.is_empty() {
        clauses.push(format!("[Section: {section}]"));
    }
    if clauses.is_empty() {
        String::new()
    } else {
        format!("{} ", clauses.join(" "))
    }
}

/// Composes `embedding_text = prefix + body`, truncating the PREFIX (never the body)
/// so the result fits the embedder window (AC5). Returns `body` alone when the prefix
/// would not fit or the body already fills the window.
pub fn compose_embedding_text(prefix: &str, body: &str, tokenizer: Option<&Tokenizer>) -> String {
    let budget = EMBEDDER_TOKEN_WINDOW.saturating_sub(SEARCH_DOCUMENT_PREFIX_TOKENS);
    let body_tokens = count_tokens(body, tokenizer);

    if prefix.is_empty() || body_tokens >= budget {
        return body.to_string();
    }

    let prefix_budget = budget - body_tokens;
    let prefix_tokens = count_tokens(prefix, tokenizer);
    if prefix_tokens <= prefix_budget {
        return format!("{prefix}{body}");
    }

    let truncated = truncate_to_token_budget(prefix, prefix_budget, tokenizer);
    if truncated.is_empty() {
        body.to_string()
    } else {
        format!("{truncated} {body}")
    }
}

/// Truncates `text` to at most `budget` tokens by dropping trailing words.
fn truncate_to_token_budget(text: &str, budget: usize, tokenizer: Option<&Tokenizer>) -> String {
    if budget == 0 {
        return String::new();
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut kept: Vec<&str> = Vec::new();
    for w in words {
        let candidate = if kept.is_empty() {
            w.to_string()
        } else {
            format!("{} {w}", kept.join(" "))
        };
        if count_tokens(&candidate, tokenizer) <= budget {
            kept.push(w);
        } else {
            break;
        }
    }
    kept.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coref_strategy_round_trips_and_defaults() {
        assert_eq!(CorefStrategy::from_config("none"), CorefStrategy::None);
        assert_eq!(
            CorefStrategy::from_config("llm_inline"),
            CorefStrategy::LlmInline
        );
        assert_eq!(
            CorefStrategy::from_config("dedicated_model"),
            CorefStrategy::LlmInline
        );
        assert_eq!(
            CorefStrategy::from_config("future_strategy"),
            CorefStrategy::LlmInline
        );
        assert_eq!(CorefStrategy::LlmInline.as_str(), "llm_inline");
        assert_eq!(CorefStrategy::None.as_str(), "none");
    }

    #[test]
    fn relations_strategy_round_trips_and_defaults() {
        assert_eq!(RelationsStrategy::default(), RelationsStrategy::Off);
        assert_eq!(
            RelationsStrategy::from_config("off"),
            RelationsStrategy::Off
        );
        assert_eq!(RelationsStrategy::from_config("on"), RelationsStrategy::On);
        assert_eq!(
            RelationsStrategy::from_config("future"),
            RelationsStrategy::Off
        );
        assert_eq!(RelationsStrategy::Off.as_str(), "off");
        assert_eq!(RelationsStrategy::On.as_str(), "on");
    }

    #[test]
    fn prefix_includes_summary_and_section() {
        let p = compose_prefix("A doc about Ada", "Intro > Bio");
        assert!(p.contains("[Document: A doc about Ada]"));
        assert!(p.contains("[Section: Intro > Bio]"));
        assert!(p.ends_with(' '), "prefix must end with a separating space");
    }

    #[test]
    fn prefix_omits_empty_clauses_but_keeps_section_for_skipped() {
        let p = compose_prefix("", "Appendix");
        assert!(!p.contains("[Document:"));
        assert!(p.contains("[Section: Appendix]"));
        assert!(!p.contains("Resolve pronouns"));
    }

    #[test]
    fn prefix_carries_no_coref_hint() {
        let p = compose_prefix("ctx", "S");
        assert!(!p.contains("Resolve pronouns"));
    }

    #[test]
    fn embedding_text_is_prefix_plus_body_and_superset_of_body() {
        let prefix = compose_prefix("ctx summary", "Sec");
        let body = "The canonical chunk body.";
        let et = compose_embedding_text(&prefix, body, None);
        assert!(
            et.ends_with(body),
            "body must be present verbatim at the end"
        );
        assert!(et.contains(body), "embedding_text ⊇ text");
        assert!(et.starts_with(&prefix[..prefix.len() - 1]) || et.starts_with(&prefix));
    }

    #[test]
    fn empty_prefix_returns_body_verbatim() {
        let body = "just the body";
        assert_eq!(compose_embedding_text("", body, None), body);
    }

    #[test]
    fn oversized_context_truncates_prefix_not_body() {
        let budget = EMBEDDER_TOKEN_WINDOW - SEARCH_DOCUMENT_PREFIX_TOKENS;
        let body_words = budget - 5; // leaves room for ~5 prefix tokens
        let body = "body ".repeat(body_words);
        let body = body.trim_end();
        let huge_prefix = format!("{} ", "context ".repeat(budget * 2));

        let et = compose_embedding_text(&huge_prefix, body, None);
        // The body survives verbatim as a suffix.
        assert!(et.ends_with(body), "body must never be truncated");
        // The prefix was truncated: total token count fits the budget.
        assert!(
            et.split_whitespace().count() <= budget,
            "composed text must fit the embedder budget"
        );
    }

    #[test]
    fn body_at_or_over_budget_returns_body_alone() {
        let budget = EMBEDDER_TOKEN_WINDOW - SEARCH_DOCUMENT_PREFIX_TOKENS;
        let body = "w ".repeat(budget + 10);
        let body = body.trim_end();
        let prefix = "[Document: ctx] ";
        let et = compose_embedding_text(prefix, body, None);
        assert_eq!(et, body);
    }
}
