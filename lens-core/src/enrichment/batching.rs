//! A generic char-budget batcher shared by the enrichment passes (the
//! structural-map map-reduce and the coref pass).
//!
//! Both passes accumulate items into batches whose combined size stays under a
//! soft byte budget ([`super::meta::ENRICHMENT_BATCH_BYTE_BUDGET`]), with the SAME
//! two invariants:
//!   * a single item that alone exceeds the budget forms its OWN batch (the
//!     provider truncates if needed — never a panic / never silently dropped);
//!   * consecutive items pack together until the next one would breach the budget,
//!     accounting for a fixed per-item separator overhead.
//!
//! [`batch_by_char_budget`] captures that one algorithm once so the two call sites
//! can't drift (the prior copies even disagreed on the `+separator` accounting).

/// Splits `items` into batches whose accumulated size (each item's `len_fn` plus a
/// fixed `separator_len` of overhead per item) stays at or under `budget`.
///
/// The accounting matches the original coref batcher: every item — including the
/// first in a batch — contributes `len_fn(item) + separator_len` to the running
/// total, and the next item is flushed into a new batch when adding it would push
/// the running total strictly past `budget`. A single oversized item (one whose own
/// `len_fn + separator_len` already exceeds the budget) forms its own batch rather
/// than being dropped.
///
/// Returns the items grouped (by value) into batches, preserving input order. An
/// empty input yields no batches.
pub(super) fn batch_by_char_budget<T>(
    items: impl IntoIterator<Item = T>,
    budget: usize,
    separator_len: usize,
    len_fn: impl Fn(&T) -> usize,
) -> Vec<Vec<T>> {
    let mut batches: Vec<Vec<T>> = Vec::new();
    let mut current: Vec<T> = Vec::new();
    let mut current_len = 0usize;
    for item in items {
        let item_cost = len_fn(&item) + separator_len;
        if !current.is_empty() && current_len + item_cost > budget {
            batches.push(std::mem::take(&mut current));
            current_len = 0;
        }
        current_len += item_cost;
        current.push(item);
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `len_fn` for a `&str` batch is its byte length.
    fn str_len(s: &&str) -> usize {
        s.len()
    }

    #[test]
    fn empty_input_yields_no_batches() {
        let batches = batch_by_char_budget(Vec::<&str>::new(), 100, 2, str_len);
        assert!(batches.is_empty());
    }

    #[test]
    fn items_under_budget_pack_into_one_batch() {
        // "aa"(2)+2, "bb"(2)+2 → running 4 then 8, both <= 100 → one batch.
        let batches = batch_by_char_budget(vec!["aa", "bb"], 100, 2, str_len);
        assert_eq!(batches, vec![vec!["aa", "bb"]]);
    }

    #[test]
    fn flushes_when_next_item_would_breach_budget() {
        // budget 10, sep 2. "aaaa"(4)+2=6 → current_len 6. Next "bbbb"(4)+2=6 →
        // 6+6=12 > 10 → flush, new batch.
        let batches = batch_by_char_budget(vec!["aaaa", "bbbb"], 10, 2, str_len);
        assert_eq!(batches, vec![vec!["aaaa"], vec!["bbbb"]]);
    }

    #[test]
    fn single_oversized_item_forms_its_own_batch() {
        let big = "x".repeat(50);
        let batches = batch_by_char_budget(vec![big.as_str(), "small"], 10, 2, str_len);
        // The oversized item is alone (current empty when seen → no flush, but its
        // cost exceeds budget); "small" then can't join it → its own batch.
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], vec![big.as_str()]);
        assert_eq!(batches[1], vec!["small"]);
    }
}
