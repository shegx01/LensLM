//! Citation extraction (issue #23a): the engine half of source-grounded answers.
//!
//! A grounded answer emits inline `[n]` markers where `n` is **1-based** into the
//! injected [`ContextUnit`] sequence the model saw. Extraction maps `[n]` to
//! `units[n-1]` **by Vec slice position** — never by searching for
//! `order_index == n-1` — because the caller passes the units in the exact order
//! the model was shown them; a position search would silently mis-cite any future
//! filtered/re-sorted slice.
//!
//! Three distinct numbering spaces coexist and must not be confused: the 1-based
//! marker value `n` (into `units`), the 0-based `ContextUnit.order_index`, and the
//! 1-based [`Citation::ordinal`] (fresh first-appearance rank over surviving
//! citations; dropped markers do not consume an ordinal).
//!
//! Grammar (ASCII, hand-rolled scan — no `regex` dep): `[`, then a comma-separated
//! list of one or more base-10 numbers with optional surrounding/internal
//! whitespace, then `]`. So `[1]`, `[1,2]`, `[1, 2]`, and `[ 1 ]` all parse (models
//! emit these despite the preferred `[1][2]` instruction, and dropping them silently
//! loses citations). A marker parses only when EVERY bracketed part is a bare
//! integer — `[1-3]` ranges, prose, empty parts, non-ASCII digits, and
//! `usize`-overflow reject the marker WHOLESALE (never a partial/false citation).
//! `[[1]]` matches the inner `[1]`; `[1](url)` yields `n=1` (the paren is not
//! consumed — not markdown-link aware). A `[1]` echoed inside a code fence IS
//! scanned (the scanner is context-free by design). Out-of-range/malformed markers
//! are dropped and logged at `debug`, never `Err`, never `panic!`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

use crate::LensError;
use crate::retrieval::router::ContextUnit;

/// Prompt-instruction template a future answer-generation step injects so the
/// model emits `[n]` markers this extractor accepts. Parameterless by design: it
/// describes the grammar, not the specific units.
pub const CITATION_PROMPT_INSTRUCTION: &str = "\
When a statement is supported by a provided source, cite it inline with a bracketed \
number: `[1]` refers to the first source unit you were given, `[2]` the second, and \
so on. Place the marker immediately after the supported statement. Emit separate \
markers when several sources support one statement — write `[1][2]`, never `[1,2]`, \
`[1-2]`, or `[ 1 ]`. Multiple markers may point at the same source. Use only the \
source numbers you were shown.";

/// A source-precise pointer to the cited passage. `chunk_id`/`anchor` come from the
/// [`ContextUnit`] at extraction time; `section_path`/`page`/`char_*` stay `None`
/// until [`hydrate_locators`] fills them from a [`ChunkLocatorRow`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Locator {
    pub chunk_id: String,
    /// Opaque `ContextUnit.locator` (`source_anchor` ‖ `section_path`); honestly
    /// named because it is not always a section path.
    pub anchor: Option<String>,
    pub section_path: Option<String>,
    pub page: Option<u32>,
    pub char_start: Option<usize>,
    pub char_end: Option<usize>,
}

/// One cited source, grouping every locator the answer pointed at it through.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Citation {
    pub source_id: String,
    /// 1-based first-appearance rank among surviving citations.
    pub ordinal: u32,
    pub locators: Vec<Locator>,
}

/// The hydration inputs for one chunk. `section_path` is `chunks.section_path`
/// (NOT NULL in the schema, hence `String`); the others are nullable columns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkLocatorRow {
    pub section_path: String,
    pub page: Option<u32>,
    pub char_start: Option<usize>,
    pub char_end: Option<usize>,
}

/// Scans `answer` for citation markers, returning `(byte_pos, n)` for every source
/// number referenced, in appearance order (a list marker like `[1,2]` yields one
/// entry per number, all sharing the marker's byte position). See the module header
/// for the accepted grammar.
fn parse_markers(answer: &str) -> Vec<(usize, usize)> {
    let bytes = answer.as_bytes();
    let mut markers = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'[' {
            i += 1;
            continue;
        }
        let start = i;
        // Scan to the first closing bracket after this `[`.
        let close = match answer[start + 1..].find(']') {
            Some(off) => start + 1 + off,
            None => {
                i = start + 1;
                continue;
            }
        };
        match parse_marker_list(&answer[start + 1..close]) {
            Some(ns) => {
                for n in ns {
                    markers.push((start, n));
                }
                i = close + 1;
            }
            // Not a marker: step past this `[` so `[[1]]` still finds the inner `[1]`.
            None => i = start + 1,
        }
    }
    markers
}

/// Parses the text between `[` and `]` as a comma-separated list of base-10 source
/// numbers, tolerating whitespace. Returns `None` unless EVERY part is a bare
/// integer, so ranges (`1-3`), prose, and empty parts reject the whole marker rather
/// than emit a partial/false citation. `usize`-overflow is dropped (logged `debug`).
fn parse_marker_list(inner: &str) -> Option<Vec<usize>> {
    let mut ns = Vec::new();
    for part in inner.split(',') {
        let t = part.trim();
        if t.is_empty() || !t.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        match t.parse::<usize>() {
            Ok(n) => ns.push(n),
            Err(_) => {
                tracing::debug!(marker = %t, "malformed marker: digits overflow usize");
                return None;
            }
        }
    }
    (!ns.is_empty()).then_some(ns)
}

/// Extracts citations from a grounded `answer` by parsing `[n]` markers and mapping
/// each to `units[n-1]` **by slice position**. Pure and synchronous: out-of-range /
/// malformed markers are dropped and logged, never an `Err`; duplicate markers per
/// source collapse to one [`Citation`] with locators deduped by `chunk_id`.
pub fn extract_citations(answer: &str, units: &[ContextUnit]) -> Vec<Citation> {
    // Insertion-ordered grouping by source_id (no indexmap dep): parallel vec of
    // (source_id, locators) with a linear membership scan — group counts are small.
    let mut groups: Vec<(String, Vec<Locator>)> = Vec::new();

    for (_pos, n) in parse_markers(answer) {
        if n == 0 || n > units.len() {
            tracing::debug!(
                marker = n,
                units = units.len(),
                "out-of-range marker dropped"
            );
            continue;
        }
        let unit = &units[n - 1];
        let locator = Locator {
            chunk_id: unit.chunk_id.clone(),
            anchor: unit.locator.clone(),
            section_path: None,
            page: None,
            char_start: None,
            char_end: None,
        };
        match groups.iter_mut().find(|(sid, _)| *sid == unit.source_id) {
            Some((_, locators)) => {
                if !locators.iter().any(|l| l.chunk_id == locator.chunk_id) {
                    locators.push(locator);
                }
            }
            None => groups.push((unit.source_id.clone(), vec![locator])),
        }
    }

    // Survivor count is bounded by the marker count, well under u32::MAX.
    groups
        .into_iter()
        .enumerate()
        .map(|(idx, (source_id, locators))| Citation {
            source_id,
            ordinal: (idx + 1) as u32,
            locators,
        })
        .collect()
}

/// Fills each [`Locator`]'s `section_path`/`page`/`char_*` from `rows` keyed by
/// `chunk_id`. Pure — no DB access. A `chunk_id` absent from `rows` is left as-is
/// (fields stay `None`). `anchor` is never touched: it comes from the
/// [`ContextUnit`], not re-read here, avoiding a double source of truth.
pub fn hydrate_locators(citations: &mut [Citation], rows: &HashMap<String, ChunkLocatorRow>) {
    for citation in citations.iter_mut() {
        for locator in citation.locators.iter_mut() {
            if let Some(row) = rows.get(&locator.chunk_id) {
                locator.section_path = Some(row.section_path.clone());
                locator.page = row.page;
                locator.char_start = row.char_start;
                locator.char_end = row.char_end;
            }
        }
    }
}

/// Loads [`ChunkLocatorRow`]s for `chunk_ids` from the `chunks` table — the single
/// owned DB path so #23b can reach real page/char via
/// `load_chunk_locators` → [`hydrate_locators`]. Chunks the `IN (…)` list under the
/// bind limit. Absent ids are simply missing from the returned map.
pub async fn load_chunk_locators(
    pool: &SqlitePool,
    chunk_ids: &[String],
) -> Result<HashMap<String, ChunkLocatorRow>, LensError> {
    let mut out = HashMap::new();
    if chunk_ids.is_empty() {
        return Ok(out);
    }
    for batch in chunk_ids.chunks(crate::db::BIND_BATCH) {
        let placeholders = crate::db::in_placeholders(batch.len());
        let sql = format!(
            "SELECT id, section_path, page, char_start, char_end \
             FROM chunks WHERE id IN ({placeholders})"
        );
        let mut q =
            sqlx::query_as::<_, (String, String, Option<i64>, Option<i64>, Option<i64>)>(&sql);
        for id in batch {
            q = q.bind(id);
        }
        for (id, section_path, page, char_start, char_end) in q.fetch_all(pool).await? {
            out.insert(
                id,
                ChunkLocatorRow {
                    section_path,
                    page: page.and_then(|p| u32::try_from(p).ok()),
                    char_start: char_start.and_then(|c| usize::try_from(c).ok()),
                    char_end: char_end.and_then(|c| usize::try_from(c).ok()),
                },
            );
        }
    }
    Ok(out)
}

/// Batched `SELECT id, title FROM sources` over the distinct `ids`, into a
/// `source_id -> title` map. Chunks the `IN (…)` list under the SQLite bind limit;
/// dedups the input. Absent ids are simply missing from the returned map. Shared
/// by the answer orchestrator (prompt titles) and notes (frozen source tag).
pub(crate) async fn source_titles(
    pool: &SqlitePool,
    ids: &[&str],
) -> Result<HashMap<String, String>, LensError> {
    let mut ids: Vec<&str> = ids.to_vec();
    ids.sort_unstable();
    ids.dedup();

    let mut out = HashMap::new();
    if ids.is_empty() {
        return Ok(out);
    }
    for batch in ids.chunks(crate::db::BIND_BATCH) {
        let placeholders = crate::db::in_placeholders(batch.len());
        let sql = format!("SELECT id, title FROM sources WHERE id IN ({placeholders})");
        let mut q = sqlx::query_as::<_, (String, String)>(&sql);
        for id in batch {
            q = q.bind(*id);
        }
        for (id, title) in q.fetch_all(pool).await? {
            out.insert(id, title);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ns(pairs: &[(usize, usize)]) -> Vec<usize> {
        pairs.iter().map(|(_, n)| *n).collect()
    }

    #[test]
    fn parses_separate_markers() {
        assert_eq!(ns(&parse_markers("a[1][2]b")), vec![1, 2]);
    }

    #[test]
    fn parses_multi_digit_marker() {
        assert_eq!(ns(&parse_markers("[12]")), vec![12]);
    }

    #[test]
    fn parses_comma_separated_list() {
        assert_eq!(ns(&parse_markers("[1,2]")), vec![1, 2]);
        assert_eq!(ns(&parse_markers("[1, 2]")), vec![1, 2]);
        assert_eq!(ns(&parse_markers("[10, 2, 3]")), vec![10, 2, 3]);
    }

    #[test]
    fn parses_internal_and_surrounding_whitespace() {
        assert_eq!(ns(&parse_markers("[ 1 ]")), vec![1]);
        assert_eq!(ns(&parse_markers("[  12  ]")), vec![12]);
    }

    #[test]
    fn drops_range_markers() {
        // Ranges are ambiguous to expand, so a `-` rejects the whole marker.
        assert!(parse_markers("[1-3]").is_empty());
    }

    #[test]
    fn mixed_valid_and_prose_rejects_whole_marker() {
        // Every comma part must be a bare integer — no partial/false citation.
        assert!(parse_markers("[1, and more]").is_empty());
        assert!(parse_markers("[1,]").is_empty());
        assert!(parse_markers("[see 1, 2 below]").is_empty());
    }

    #[test]
    fn nested_brackets_match_inner_marker() {
        assert_eq!(ns(&parse_markers("[[1]]")), vec![1]);
    }

    #[test]
    fn markdown_link_paren_is_ignored() {
        // The parser consumes only `[digits]`, never a following paren group.
        assert_eq!(ns(&parse_markers("[1](http://x)")), vec![1]);
    }

    #[test]
    fn non_ascii_digits_dropped() {
        // Arabic-Indic digits are not ASCII 0-9.
        assert!(parse_markers("[\u{0661}]").is_empty());
    }

    #[test]
    fn overflow_dropped_no_panic() {
        let huge = format!("[{}]", "9".repeat(40));
        assert!(parse_markers(&huge).is_empty());
    }

    #[test]
    fn empty_brackets_and_letters_dropped() {
        assert!(parse_markers("[]").is_empty());
        assert!(parse_markers("[abc]").is_empty());
    }
}
