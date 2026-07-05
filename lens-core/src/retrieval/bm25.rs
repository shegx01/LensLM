//! BM25 lexical retrieval over `chunks_fts` (issue #39). Notebook scope and the
//! trashed-source exclusion require a JOIN through `sources` — `chunks` carries no
//! `notebook_id`, and `trashed_at` is mutable so it must be checked live. The
//! INNER JOIN on `chunks` also neutralizes orphan `chunks_fts` rows left by a
//! FK-cascade delete (which does not fire the AFTER DELETE trigger).

use sqlx::SqlitePool;

use crate::LensError;

/// FTS5 bareword operators (case-sensitive in FTS5: only uppercase are operators).
const FTS5_OPERATORS: &[&str] = &["AND", "OR", "NOT", "NEAR"];

/// Sanitizes a user query into a safe FTS5 `MATCH` expression. Any character that
/// is not a Unicode alphanumeric is mapped to a SPACE (not deleted) — this both
/// neutralizes every FTS5 operator/special char (`* ^ " ( ) + - : ? .` …) AND
/// mirrors the default unicode61 tokenizer, which splits indexed text on the same
/// boundaries (so a query `ZX9-4471Q` becomes bare tokens `zx9 4471q` that match
/// the indexed tokens). Uppercase bareword operators (`AND`/`OR`/`NOT`/`NEAR`) are
/// dropped. Remaining tokens are joined by spaces (implicit-OR matching). We do NOT
/// phrase-quote the whole query — that would be a single FTS5 phrase and gut recall.
/// Returns `None` when nothing survives; the caller then yields zero BM25 hits.
pub fn sanitize_fts_query(query: &str) -> Option<String> {
    let mapped: String = query
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();

    let tokens: Vec<&str> = mapped
        .split_whitespace()
        .filter(|t| !FTS5_OPERATORS.contains(t))
        .collect();

    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" "))
    }
}

/// Runs a notebook-scoped BM25 search over `chunks_fts`, returning chunk ids ordered
/// best-first (most-negative `bm25()` = best), excluding trashed sources. Optional
/// `source_id`/`level` pre-filters narrow the scope further. An empty-after-sanitize
/// query yields zero hits (not an error).
pub async fn bm25_search(
    pool: &SqlitePool,
    notebook_id: &str,
    source_id: Option<&str>,
    level: Option<i32>,
    query: &str,
    limit: usize,
) -> Result<Vec<String>, LensError> {
    let Some(match_expr) = sanitize_fts_query(query) else {
        return Ok(Vec::new());
    };

    // bm25() is more-negative = better, so ORDER BY ascending puts best first.
    let mut sql = String::from(
        "SELECT f.chunk_id FROM chunks_fts f \
         JOIN chunks c ON c.id = f.chunk_id \
         JOIN sources s ON s.id = c.source_id \
         WHERE chunks_fts MATCH ? AND s.notebook_id = ? AND s.trashed_at IS NULL",
    );
    if source_id.is_some() {
        sql.push_str(" AND c.source_id = ?");
    }
    if level.is_some() {
        sql.push_str(" AND c.level = ?");
    }
    sql.push_str(" ORDER BY bm25(chunks_fts) LIMIT ?");

    let mut q = sqlx::query_scalar::<_, String>(&sql)
        .bind(match_expr)
        .bind(notebook_id);
    if let Some(sid) = source_id {
        q = q.bind(sid);
    }
    if let Some(lvl) = level {
        q = q.bind(lvl);
    }
    let ids = q.bind(limit as i64).fetch_all(pool).await?;
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_multi_word_query_passes_bare_tokens() {
        assert_eq!(
            sanitize_fts_query("golden record voyager"),
            Some("golden record voyager".to_string())
        );
    }

    #[test]
    fn maps_fts5_special_chars_to_spaces_without_erroring() {
        // Each special char becomes a token boundary; the alphanumeric payload survives.
        assert_eq!(
            sanitize_fts_query(r#"foo* "bar" (baz) +qux ^rank"#),
            Some("foo bar baz qux rank".to_string())
        );
    }

    #[test]
    fn question_mark_and_punctuation_do_not_error() {
        assert_eq!(
            sanitize_fts_query("How do plants turn sunlight into sugar?"),
            Some("How do plants turn sunlight into sugar".to_string())
        );
    }

    #[test]
    fn hyphenated_code_splits_to_match_tokenizer() {
        // The default unicode61 tokenizer indexes "ZX9-4471Q" as tokens zx9/4471q;
        // the query must split identically to match.
        assert_eq!(
            sanitize_fts_query("ZX9-4471Q"),
            Some("ZX9 4471Q".to_string())
        );
    }

    #[test]
    fn drops_uppercase_bareword_operators() {
        assert_eq!(
            sanitize_fts_query("cats AND dogs OR birds NOT fish NEAR trees"),
            Some("cats dogs birds fish trees".to_string())
        );
    }

    #[test]
    fn lowercase_operators_are_kept_as_tokens() {
        // FTS5 operators are case-sensitive; lowercase "and" is a normal token.
        assert_eq!(
            sanitize_fts_query("cats and dogs"),
            Some("cats and dogs".to_string())
        );
    }

    #[test]
    fn empty_after_strip_returns_none() {
        assert_eq!(sanitize_fts_query(""), None);
        assert_eq!(sanitize_fts_query("   "), None);
        assert_eq!(sanitize_fts_query(r#"* ^ " ( ) + -"#), None);
        assert_eq!(sanitize_fts_query("AND OR NOT"), None);
    }

    #[test]
    fn colon_splits_so_no_column_filter_syntax_leaks() {
        // A leading "col:" would be an FTS5 column filter; mapping ':' to a space
        // splits it into two harmless tokens.
        assert_eq!(
            sanitize_fts_query("text:secret"),
            Some("text secret".to_string())
        );
    }
}
