//! Generic char-budget batcher shared by the structural-map and coref passes.
//! An oversized item forms its own batch; consecutive items pack until the next
//! would breach the budget (accounting for per-item separator overhead).

/// Groups `items` into batches whose accumulated `len_fn(item) + separator_len`
/// stays at or under `budget`. A single oversized item forms its own batch.
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
        let batches = batch_by_char_budget(vec!["aa", "bb"], 100, 2, str_len);
        assert_eq!(batches, vec![vec!["aa", "bb"]]);
    }

    #[test]
    fn flushes_when_next_item_would_breach_budget() {
        let batches = batch_by_char_budget(vec!["aaaa", "bbbb"], 10, 2, str_len);
        assert_eq!(batches, vec![vec!["aaaa"], vec!["bbbb"]]);
    }

    #[test]
    fn single_oversized_item_forms_its_own_batch() {
        let big = "x".repeat(50);
        let batches = batch_by_char_budget(vec![big.as_str(), "small"], 10, 2, str_len);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], vec![big.as_str()]);
        assert_eq!(batches[1], vec!["small"]);
    }
}
