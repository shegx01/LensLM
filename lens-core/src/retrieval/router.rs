//! Tiered Context Router (issue #21): the single integration point for grounded
//! retrieval. Given a pre-embedded query, it decides Tier-1 (inject the raw
//! selected corpus) vs Tier-2 (fused retrieval + parent auto-merge), applies the
//! #39 dense pre-filter, optionally folds in a deterministic graph arm, and returns
//! a budgeted, document-ordered, hydrated [`RouterOutput`]. It performs NO LLM call
//! and NO prompt assembly — that is a downstream `answer()` step.

use sqlx::SqlitePool;
use tokenizers::Tokenizer;

use crate::LensError;
use crate::chunk::kind;
use crate::config::TierThresholds;
use crate::graph::{EntityKind, GraphEntity};

use super::HitSource;

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
    let usable = ctx.saturating_sub(RESERVED_OUTPUT).saturating_sub(SYSTEM_OVERHEAD);
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

/// Maps `entity_lookup` results into the `(name, kind)` seed shape `ppr_expand` and
/// [`graph_compose`] consume, preserving order.
fn seeds_from_entities(entities: &[GraphEntity]) -> Vec<(String, EntityKind)> {
    entities.iter().map(|e| (e.name.clone(), e.kind)).collect()
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

    #[test]
    fn seeds_from_entities_preserves_order_and_shape() {
        let ents = vec![
            GraphEntity {
                name: "Acme".into(),
                kind: EntityKind::Org,
                definition: None,
                source_count: 1,
                mention_count: 1,
            },
            GraphEntity {
                name: "Beta".into(),
                kind: EntityKind::Concept,
                definition: None,
                source_count: 1,
                mention_count: 1,
            },
        ];
        let seeds = seeds_from_entities(&ents);
        assert_eq!(
            seeds,
            vec![
                ("Acme".to_string(), EntityKind::Org),
                ("Beta".to_string(), EntityKind::Concept),
            ]
        );
    }
}
