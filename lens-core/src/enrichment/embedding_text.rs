//! Contextual `embedding_text` composition (AC5) and the coref strategy enum.
//!
//! The worker derives a per-chunk `embedding_text` = a doc/section CONTEXT PREFIX
//! (+ optional inline-coref hint) followed by the chunk's CANONICAL body. The
//! canonical `chunks.text` is NEVER mutated; `embedding_text` is a SEPARATE column
//! the re-embed pass (Step 5) reads via `COALESCE(embedding_text, text)`.
//!
//! ## Truncation invariant (the load-bearing part of AC5)
//!
//! When the composed `embedding_text` would exceed the embedder's input window —
//! accounting for the hard-applied `"search_document: "` prefix
//! (`embedder.rs:207`) — the CONTEXT/PREFIX is dropped FIRST, never the canonical
//! body. The body is the citation text and must survive verbatim so retrieval
//! still grounds on the real chunk content. If the body alone already exceeds the
//! window the body is returned unprefixed (the embedder applies its own internal
//! truncation downstream; we never corrupt the canonical bytes here).
//!
//! Token accounting uses a real tokenizer when one is supplied (the production
//! path threads the nomic tokenizer) and a conservative whitespace-word
//! approximation otherwise (pure-logic unit tests, no model download).

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use super::meta::{EMBEDDER_TOKEN_WINDOW, SEARCH_DOCUMENT_PREFIX_TOKENS};

/// The coreference-resolution strategy applied while composing `embedding_text`.
///
/// This is the CANONICAL coref enum: it is the typed `coref_strategy` on
/// [`crate::config::EnrichmentConfig`] (Step 6) AND the runtime strategy used by
/// the worker, so there is a single source of truth (no stringly-typed config).
/// It serializes to the same stable snake_case strings used in the composite
/// cache key (AC9) and mirrored on the TS `CorefStrategy` union (`none` /
/// `llm_inline`), so an existing `config.json` round-trips. A legacy
/// `"dedicated_model"` value (written by an older build that shipped that stub)
/// is accepted on read and maps to [`LlmInline`](CorefStrategy::LlmInline) so old
/// configs never panic — but the strategy itself is gone: no stub ships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorefStrategy {
    /// No coref resolution — the chunk body is embedded verbatim under the
    /// doc/section context prefix.
    None,
    /// Real LLM-driven coref-substitution resolution applied to the BODY (the
    /// default). The worker runs [`super::coref::resolve_coref_batch`] and
    /// deterministically applies the surviving substitutions to each chunk's body
    /// before composing `embedding_text`; the PREFIX is unchanged from `None`.
    #[default]
    LlmInline,
}

impl CorefStrategy {
    /// The stable cache-key string (a component of the AC9 composite key).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::LlmInline => "llm_inline",
        }
    }

    /// Parses the persisted/config string. An unrecognized value — including the
    /// legacy `"dedicated_model"` stub that no longer exists — defaults to
    /// [`LlmInline`](CorefStrategy::LlmInline), so an old config round-trips
    /// without panicking the worker.
    pub fn from_config(value: &str) -> Self {
        match value {
            "none" => Self::None,
            // "llm_inline", the legacy "dedicated_model", and any unknown value
            // → the default.
            _ => Self::LlmInline,
        }
    }
}

/// Counts tokens in `text`. Uses the real tokenizer when supplied (production +
/// integration), else a conservative whitespace-word count (pure-logic tests).
pub(crate) fn count_tokens(text: &str, tokenizer: Option<&Tokenizer>) -> usize {
    match tokenizer {
        Some(tk) => tk
            .encode(text, false)
            .map(|e| e.len())
            // A tokenizer failure must never corrupt the body — fall back to the
            // word approximation rather than dropping the chunk.
            .unwrap_or_else(|_| text.split_whitespace().count()),
        None => text.split_whitespace().count(),
    }
}

/// The composed context prefix for a chunk, given the per-doc summary and the
/// chunk's section path.
///
/// Shape: `"[Document: {summary}] [Section: {section_path}] "`. Each clause is
/// omitted when its input is empty, so a skipped/non-prose chunk (empty summary)
/// still gets a `[Section: …]`-only prefix (Decision B). The trailing space
/// separates the prefix from the body.
///
/// **Coref does NOT affect the prefix.** Real coref resolution happens in the BODY:
/// under [`CorefStrategy::LlmInline`] the worker applies validated coref
/// substitutions to the chunk text before composing `embedding_text` (the old
/// static "[Resolve pronouns…]" hint clause was a no-op placeholder and has been
/// removed). The prefix is identical regardless of coref strategy, so this function
/// no longer takes a `CorefStrategy` — a caller that needs to branch on strategy
/// does so at the worker level (on the body), not here.
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

/// Composes a chunk's `embedding_text` = `prefix + body`, truncating the PREFIX
/// (never the body) so the result fits the embedder window after the hard-applied
/// `"search_document: "` prefix is accounted for (AC5).
///
/// * `body` is the canonical `chunks.text` — returned VERBATIM and always present
///   in the output (the truncation invariant: drop context, preserve body).
/// * Returns `body` unchanged (no prefix) when the prefix would not fit.
///
/// The budget is `EMBEDDER_TOKEN_WINDOW - SEARCH_DOCUMENT_PREFIX_TOKENS`. When the
/// body alone meets/exceeds that budget there is no room for any prefix → return
/// the body alone (the embedder truncates downstream; we never alter the body).
pub fn compose_embedding_text(prefix: &str, body: &str, tokenizer: Option<&Tokenizer>) -> String {
    let budget = EMBEDDER_TOKEN_WINDOW.saturating_sub(SEARCH_DOCUMENT_PREFIX_TOKENS);
    let body_tokens = count_tokens(body, tokenizer);

    // No prefix, or the body already fills the window → body alone (verbatim).
    if prefix.is_empty() || body_tokens >= budget {
        return body.to_string();
    }

    let prefix_budget = budget - body_tokens;
    let prefix_tokens = count_tokens(prefix, tokenizer);
    if prefix_tokens <= prefix_budget {
        // Whole prefix fits.
        return format!("{prefix}{body}");
    }

    // The prefix must be truncated to its token budget. Truncate by WORDS (a
    // safe char-boundary unit) until it fits — never touch the body.
    let truncated = truncate_to_token_budget(prefix, prefix_budget, tokenizer);
    if truncated.is_empty() {
        body.to_string()
    } else {
        format!("{truncated} {body}")
    }
}

/// Truncates `text` to at most `budget` tokens by dropping trailing words. Returns
/// the longest leading whitespace-delimited prefix whose token count fits. Empty
/// when even the first word overflows. Word boundaries keep the result on valid
/// char boundaries (no panics on multi-byte text).
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
        // The legacy `dedicated_model` stub is gone: it must map to LlmInline so
        // an old config round-trips without panic (no stub ships).
        assert_eq!(
            CorefStrategy::from_config("dedicated_model"),
            CorefStrategy::LlmInline
        );
        // Unknown → default LlmInline (never panics).
        assert_eq!(
            CorefStrategy::from_config("future_strategy"),
            CorefStrategy::LlmInline
        );
        assert_eq!(CorefStrategy::LlmInline.as_str(), "llm_inline");
        assert_eq!(CorefStrategy::None.as_str(), "none");
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
        // Decision B: a skipped/non-prose chunk has an empty summary but still gets
        // a section-only prefix.
        let p = compose_prefix("", "Appendix");
        assert!(!p.contains("[Document:"));
        assert!(p.contains("[Section: Appendix]"));
        // The static coref hint is gone — coref now resolves in the body.
        assert!(!p.contains("Resolve pronouns"));
    }

    #[test]
    fn prefix_carries_no_coref_hint() {
        // Coref no longer changes the PREFIX (real resolution happens in the body),
        // so `compose_prefix` no longer takes a strategy and never emits the
        // obsolete static hint clause.
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
        // Body just under the budget; an enormous prefix must be dropped, the body
        // kept byte-identical. Word-count approximation (tokenizer=None).
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
        // No room for any prefix → body alone, verbatim.
        assert_eq!(et, body);
    }
}
