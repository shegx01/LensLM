//! Shared rendering helpers for the tabular extractors (XLSX/XLS/CSV — issue #76).
//!
//! [`render_table_markdown`] turns a single sheet's `headers` + `rows` into a
//! pipe-delimited markdown table. This rendering is produced DURING extraction
//! (the rows are already in memory — no second parse) and carried on
//! [`ExtractOutput::table_markdown`](super::ExtractOutput::table_markdown). It is
//! persisted by ingest as the `{id}.tables.md` sibling but is NEVER embedded and
//! NEVER part of `extracted_text`.

/// Hard ceiling on the number of columns a tabular source may have. Set to
/// Excel's hard column ceiling (the XFD column — 16,384) so we never reject a
/// valid spreadsheet, while still bounding the per-row `Vec<String>` +
/// verbalized `String` allocation a pathological million-column input would
/// otherwise force BEFORE any byte-size cap can trigger. Enforced by both the
/// CSV and spreadsheet extractors against the header (and, for CSV's flexible
/// parsing, any data row that exceeds it).
pub(crate) const MAX_COLUMNS: usize = 16_384;

/// Normalizes a tabular source's first row into headers: if every cell is
/// blank/whitespace, synthesizes `"Column 1"`, `"Column 2"`, ... using the
/// row's width (preserving the no-header fallback); otherwise the row is used
/// verbatim (blank/duplicate headers kept per spec). Shared by the CSV and
/// spreadsheet extractors so the fallback can never diverge.
pub(crate) fn normalize_headers(first: Vec<String>) -> Vec<String> {
    if first.iter().all(|c| c.trim().is_empty()) {
        (0..first.len())
            .map(|i| format!("Column {}", i + 1))
            .collect()
    } else {
        first
    }
}

/// Renders one sheet's `headers` + `rows` into a pipe-delimited markdown table.
///
/// Output shape (GitHub-flavoured markdown):
/// ```text
/// | H1 | H2 |
/// | --- | --- |
/// | v1 | v2 |
/// ```
///
/// Short rows are padded with empty cells and long rows are truncated to the
/// header width, so every emitted row has exactly `headers.len()` columns (the
/// same shape the row-verbalization uses). Pipes and newlines inside a cell are
/// escaped (`\|`, space) so they cannot break the table grid. An empty `headers`
/// slice yields an empty string.
pub(crate) fn render_table_markdown(headers: &[String], rows: &[Vec<String>]) -> String {
    if headers.is_empty() {
        return String::new();
    }
    let ncols = headers.len();
    let mut out = String::new();

    // Header row.
    out.push_str("| ");
    out.push_str(
        &headers
            .iter()
            .map(|h| escape_cell(h))
            .collect::<Vec<_>>()
            .join(" | "),
    );
    out.push_str(" |\n");

    // Separator row.
    out.push_str("| ");
    out.push_str(&vec!["---"; ncols].join(" | "));
    out.push_str(" |\n");

    // Data rows (padded/truncated to header width).
    for row in rows {
        out.push_str("| ");
        let cells: Vec<String> = (0..ncols)
            .map(|i| escape_cell(row.get(i).map(String::as_str).unwrap_or("")))
            .collect();
        out.push_str(&cells.join(" | "));
        out.push_str(" |\n");
    }

    out
}

/// Verbalizes one data row into `"H1: v1; H2: v2; ..."`, padding short rows with
/// empty values and truncating long rows to `headers.len()`. Shared by the CSV
/// and spreadsheet extractors so the embedded text shape stays byte-identical.
pub(crate) fn verbalize_row(headers: &[String], row: &[String]) -> String {
    headers
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let v = row.get(i).map(String::as_str).unwrap_or("");
            format!("{h}: {v}")
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Escapes a cell value so it cannot break the markdown table grid: pipes and
/// backticks are backslash-escaped and newlines/carriage-returns are flattened
/// to spaces. A lone backtick would otherwise open a GFM code span that can
/// swallow the cell's trailing ` | ` delimiter and break the table grid.
fn escape_cell(s: &str) -> String {
    // Escape backslashes BEFORE pipes/backticks so the escape backslash we add
    // isn't itself re-escaped.
    s.replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('`', "\\`")
        .replace(['\n', '\r'], " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_table_markdown_simple() {
        let headers = vec!["Name".to_string(), "Age".to_string()];
        let rows = vec![
            vec!["Alice".to_string(), "30".to_string()],
            vec!["Bob".to_string(), "25".to_string()],
        ];
        let md = render_table_markdown(&headers, &rows);
        assert_eq!(
            md,
            "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |\n"
        );
    }

    #[test]
    fn render_table_markdown_pads_short_and_truncates_long_rows() {
        let headers = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let rows = vec![
            vec!["1".to_string()], // short
            vec![
                "1".to_string(),
                "2".to_string(),
                "3".to_string(),
                "4".to_string(),
            ], // long
        ];
        let md = render_table_markdown(&headers, &rows);
        // Short row padded to 3 cols; long row truncated to 3 cols.
        assert!(md.contains("| 1 |  |  |\n"), "short row padded: {md:?}");
        assert!(md.contains("| 1 | 2 | 3 |\n"), "long row truncated: {md:?}");
    }

    #[test]
    fn render_table_markdown_escapes_pipes_and_newlines() {
        let headers = vec!["H".to_string()];
        let rows = vec![vec!["a|b\nc".to_string()]];
        let md = render_table_markdown(&headers, &rows);
        assert!(md.contains("| a\\|b c |\n"), "escaped cell: {md:?}");
    }

    #[test]
    fn render_table_markdown_escapes_backticks() {
        // A lone backtick must be escaped so it cannot open a GFM code span that
        // swallows the trailing ` | ` delimiter and breaks the table grid.
        let headers = vec!["H".to_string()];
        let rows = vec![vec!["a`b".to_string()]];
        let md = render_table_markdown(&headers, &rows);
        assert!(md.contains("| a\\`b |\n"), "escaped backtick cell: {md:?}");
        // The grid is intact: the row still has exactly one trailing delimiter.
        assert!(md.lines().all(|l| l.is_empty() || l.ends_with(" |")));
    }

    #[test]
    fn normalize_headers_synthesizes_columns_for_blank_first_row() {
        let headers = normalize_headers(vec![String::new(), "  ".to_string()]);
        assert_eq!(headers, vec!["Column 1", "Column 2"]);
    }

    #[test]
    fn normalize_headers_keeps_nonblank_verbatim() {
        let headers =
            normalize_headers(vec!["Name".to_string(), String::new(), "Name".to_string()]);
        assert_eq!(headers, vec!["Name", "", "Name"]);
    }

    #[test]
    fn render_table_markdown_empty_headers_is_empty() {
        assert_eq!(render_table_markdown(&[], &[]), "");
    }

    #[test]
    fn render_table_markdown_multi_sheet_concatenation() {
        // The caller concatenates per-sheet sections with a `## {sheet}` heading.
        let s1 = render_table_markdown(&["Name".to_string()], &[vec!["Alice".to_string()]]);
        let s2 = render_table_markdown(&["City".to_string()], &[vec!["NYC".to_string()]]);
        let doc = format!("## People\n\n{s1}\n## Cities\n\n{s2}");
        assert!(doc.contains("## People"));
        assert!(doc.contains("## Cities"));
        assert!(doc.contains("| Name |"));
        assert!(doc.contains("| City |"));
    }
}
