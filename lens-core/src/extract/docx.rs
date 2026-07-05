//! DOCX extractor (M4 Phase 2, Step 6).
//!
//! [`DocxExtractor`] parses a `.docx` file from raw bytes using `docx-rs` and
//! produces a canonical [`ExtractOutput`] where:
//! - `extracted_text` is the linearised UTF-8 text of all paragraphs and tables
//!   (one `\n` separator between consecutive blocks, two `\n` at the end of each
//!   table cell row to separate cell content).
//! - `blocks` carries one [`Block`] per:
//!   - Heading paragraph → `block_type = "heading"`, `section_path` updated,
//!   - Table → one `block_type = "table"` block carrying the linearised cell text
//!     (mirrors the Group-0 GFM table convention: one block, not one per cell).
//!   - Body paragraph → `block_type = "paragraph"`.
//! - `anchors` carries one [`SourceAnchor::Docx { node_path }`] per block,
//!   where `node_path` is a compact XPath-ish string like `"body/p[0]"` or
//!   `"body/tbl[1]"`, 0-indexed among **all** `DocumentChild` elements.

use std::io::{Cursor, Read};

use docx_rs::{
    DocumentChild, ParagraphChild, RunChild, Table, TableCellContent, TableChild, read_docx,
};

use crate::LensError;
use crate::parse::{Block, BlockType, SectionPathStack};

use super::{ExtractOutput, Extractor, SourceAnchor};

const MAX_DECOMPRESSED_BYTES: u64 = 256 * 1024 * 1024;

fn guard_docx_inflation(raw: &[u8], max: u64) -> Result<(), LensError> {
    let mut archive = zip::ZipArchive::new(Cursor::new(raw))
        .map_err(|e| LensError::Parse(format!("DOCX is not a valid ZIP container: {e}")))?;
    let mut total: u64 = 0;
    for i in 0..archive.len() {
        let entry = archive
            .by_index(i)
            .map_err(|e| LensError::Parse(format!("DOCX ZIP entry {i} unreadable: {e}")))?;
        let budget = max - total;
        let read = std::io::copy(&mut entry.take(budget + 1), &mut std::io::sink())
            .map_err(|e| LensError::Parse(format!("DOCX ZIP entry decompression failed: {e}")))?;
        total += read;
        if total > max {
            return Err(LensError::Validation(format!(
                "DOCX decompresses to more than the {max}-byte limit \
                 (possible decompression bomb)"
            )));
        }
    }
    Ok(())
}

/// Returns the heading level (1–6) for a Word heading style id
/// (`Heading1`…`Heading6`, case-insensitive, optional space/dash separator),
/// or `None`.
fn heading_level(style_id: &str) -> Option<u8> {
    let s = style_id.to_lowercase();
    let s = s.trim();
    let digits = s
        .strip_prefix("heading")
        .map(|rest| rest.trim_matches(|c: char| c == ' ' || c == '-' || c == '_'));
    match digits {
        Some("1") => Some(1),
        Some("2") => Some(2),
        Some("3") => Some(3),
        Some("4") => Some(4),
        Some("5") => Some(5),
        Some("6") => Some(6),
        _ => None,
    }
}

/// Extracts plain text from a `docx_rs::Paragraph` by concatenating all
/// `RunChild::Text` values. Non-text run children are ignored.
fn para_text(p: &docx_rs::Paragraph) -> String {
    let mut buf = String::new();
    for pc in &p.children {
        if let ParagraphChild::Run(run) = pc {
            for rc in &run.children {
                if let RunChild::Text(t) = rc {
                    buf.push_str(&t.text);
                }
            }
        }
    }
    buf
}

/// Extracts linearised table text (cells separated by two spaces, rows by `\n`).
///
/// The `#[allow(irrefutable_let_patterns)]` guards are deliberate future-proofing:
/// `TableChild`/`TableRowChild` are single-variant today, but if an additive
/// upstream variant is added the `else { continue }` degrades gracefully instead
/// of failing to compile.
const MAX_TABLE_DEPTH: usize = 16;

fn table_text(tbl: &Table) -> String {
    table_text_depth(tbl, 0)
}

fn table_text_depth(tbl: &Table, depth: usize) -> String {
    let mut rows: Vec<String> = Vec::new();
    for row_child in &tbl.rows {
        #[allow(irrefutable_let_patterns)]
        let TableChild::TableRow(row) = row_child else {
            continue;
        };
        let mut cells: Vec<String> = Vec::new();
        for cell_child in &row.cells {
            #[allow(irrefutable_let_patterns)]
            let docx_rs::TableRowChild::TableCell(tc) = cell_child else {
                continue;
            };
            let mut cell_buf = String::new();
            for cc in &tc.children {
                let t = match cc {
                    TableCellContent::Paragraph(p) => para_text(p),
                    TableCellContent::Table(nested) if depth < MAX_TABLE_DEPTH => {
                        table_text_depth(nested, depth + 1)
                    }
                    _ => continue,
                };
                if !t.is_empty() {
                    if !cell_buf.is_empty() {
                        cell_buf.push(' ');
                    }
                    cell_buf.push_str(&t);
                }
            }
            cells.push(cell_buf);
        }
        rows.push(cells.join("  "));
    }
    rows.join("\n")
}

/// DOCX extractor — implements [`Extractor`] via `docx-rs`.
pub struct DocxExtractor;

impl Extractor for DocxExtractor {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError> {
        guard_docx_inflation(raw, MAX_DECOMPRESSED_BYTES)?;
        let docx = read_docx(raw)
            .map_err(|e| LensError::Parse(format!("docx-rs failed to parse DOCX: {e:?}")))?;

        let mut extracted_text = String::new();
        let mut blocks: Vec<Block> = Vec::new();
        let mut anchors: Vec<SourceAnchor> = Vec::new();

        let mut section_path = SectionPathStack::new();

        // Per-type counters ensure distinct node_paths (para and table never share one).
        let mut para_count: usize = 0;
        let mut tbl_count: usize = 0;

        for child in &docx.document.children {
            match child {
                DocumentChild::Paragraph(p) => {
                    let node_path = format!("body/p[{para_count}]");
                    // Increment before the empty-paragraph skip so node_path stays
                    // stable regardless of which paragraphs are skipped.
                    para_count += 1;

                    let text = para_text(p);
                    if text.is_empty() {
                        continue; // skip blank paragraphs
                    }

                    let style_id = p
                        .property
                        .style
                        .as_ref()
                        .map(|s| s.val.as_str())
                        .unwrap_or("");
                    let level = heading_level(style_id);

                    // Update before emitting so the heading carries its own trail.
                    if let Some(lvl) = level {
                        section_path.push(lvl, &text);
                    }

                    let btype = if level.is_some() {
                        BlockType::Heading.as_str()
                    } else {
                        BlockType::Paragraph.as_str()
                    };

                    let sp = section_path.current();

                    let char_start = extracted_text.len();
                    extracted_text.push_str(&text);
                    extracted_text.push('\n');
                    let char_end = extracted_text.len() - 1; // trailing \n excluded from block slice

                    blocks.push(Block {
                        block_type: btype.to_string(),
                        section_path: sp,
                        text: text.clone(),
                        char_start,
                        char_end,
                    });
                    anchors.push(SourceAnchor::Docx { node_path });
                }

                DocumentChild::Table(tbl) => {
                    let node_path = format!("body/tbl[{tbl_count}]");
                    tbl_count += 1;

                    let text = table_text(tbl);
                    if text.is_empty() {
                        continue;
                    }

                    let sp = section_path.current();

                    let char_start = extracted_text.len();
                    extracted_text.push_str(&text);
                    extracted_text.push('\n');
                    let char_end = extracted_text.len() - 1; // trailing \n excluded

                    blocks.push(Block {
                        block_type: BlockType::Table.as_str().to_string(),
                        section_path: sp,
                        text,
                        char_start,
                        char_end,
                    });
                    anchors.push(SourceAnchor::Docx { node_path });
                }

                // Bookmarks, comments, section properties, SDTs, ToC — skip.
                _ => {}
            }
        }

        // The trailing '\n' was never part of any block's char_end; safe to trim.
        while extracted_text.ends_with('\n') {
            extracted_text.pop();
        }

        Ok(ExtractOutput {
            extracted_text,
            blocks,
            anchors,
            table_markdown: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::extract::ExtractOutput;

    /// Builds a minimal fixture DOCX: Heading1, body paragraph, 2×2 table.
    fn build_fixture_docx() -> Vec<u8> {
        use docx_rs::{Docx, Paragraph, Run, Table, TableCell, TableRow};
        let docx = Docx::new()
            .add_paragraph(
                Paragraph::new()
                    .add_run(Run::new().add_text("Test Heading"))
                    .style("Heading1"),
            )
            .add_paragraph(
                Paragraph::new().add_run(Run::new().add_text("Sentinel body text for extraction.")),
            )
            .add_table(Table::new(vec![
                TableRow::new(vec![
                    TableCell::new()
                        .add_paragraph(Paragraph::new().add_run(Run::new().add_text("Cell A1"))),
                    TableCell::new()
                        .add_paragraph(Paragraph::new().add_run(Run::new().add_text("Cell A2"))),
                ]),
                TableRow::new(vec![
                    TableCell::new()
                        .add_paragraph(Paragraph::new().add_run(Run::new().add_text("Cell B1"))),
                    TableCell::new()
                        .add_paragraph(Paragraph::new().add_run(Run::new().add_text("Cell B2"))),
                ]),
            ]));

        let mut buf = Vec::new();
        docx.build()
            .pack(Cursor::new(&mut buf))
            .expect("fixture DOCX build failed");
        buf
    }

    fn extract_fixture() -> ExtractOutput {
        let raw = build_fixture_docx();
        DocxExtractor
            .extract(&raw)
            .expect("fixture DOCX extraction must succeed")
    }

    #[test]
    fn inflation_guard_passes_legit_docx_under_ceiling() {
        let raw = build_fixture_docx();
        guard_docx_inflation(&raw, MAX_DECOMPRESSED_BYTES)
            .expect("a legitimate DOCX must pass the inflation guard");
    }

    #[test]
    fn inflation_guard_rejects_decompression_beyond_cap() {
        let raw = build_fixture_docx();
        let err = guard_docx_inflation(&raw, 16)
            .expect_err("decompression past the cap must be rejected");
        assert!(
            matches!(err, LensError::Validation(_)),
            "an over-inflation is a Validation error, got {err:?}"
        );
    }

    #[test]
    fn inflation_guard_rejects_non_zip_input() {
        let err = guard_docx_inflation(b"not a zip at all", MAX_DECOMPRESSED_BYTES)
            .expect_err("non-ZIP bytes must not reach read_docx");
        assert!(
            matches!(err, LensError::Parse(_)),
            "a non-ZIP container is a Parse error, got {err:?}"
        );
    }

    #[test]
    fn docx_byte_identity() {
        let out = extract_fixture();
        assert!(
            !out.blocks.is_empty(),
            "fixture must produce at least one block"
        );
        for (i, b) in out.blocks.iter().enumerate() {
            assert_eq!(
                &out.extracted_text[b.char_start..b.char_end],
                b.text,
                "byte-identity violated for block[{i}] (type={:?})",
                b.block_type
            );
        }
    }

    #[test]
    fn docx_sentinel_body_text_present() {
        let out = extract_fixture();
        assert!(
            out.extracted_text
                .contains("Sentinel body text for extraction."),
            "sentinel body text missing from extracted_text; got: {:?}",
            out.extracted_text
        );
    }

    #[test]
    fn docx_table_cell_present() {
        let out = extract_fixture();
        assert!(
            out.extracted_text.contains("Cell A1"),
            "table cell 'Cell A1' missing from extracted_text; got: {:?}",
            out.extracted_text
        );
        assert!(
            out.extracted_text.contains("Cell B2"),
            "table cell 'Cell B2' missing from extracted_text; got: {:?}",
            out.extracted_text
        );
    }

    #[test]
    fn docx_anchors_index_aligned() {
        let out = extract_fixture();
        assert_eq!(
            out.anchors.len(),
            out.blocks.len(),
            "anchors.len() must equal blocks.len()"
        );
        for (i, a) in out.anchors.iter().enumerate() {
            assert!(
                matches!(a, SourceAnchor::Docx { .. }),
                "anchor[{i}] must be SourceAnchor::Docx"
            );
        }
    }

    #[test]
    fn docx_node_paths_distinct() {
        let out = extract_fixture();
        let mut seen = std::collections::HashSet::new();
        for (i, a) in out.anchors.iter().enumerate() {
            let SourceAnchor::Docx { node_path } = a else {
                panic!("anchor[{i}] is not Docx");
            };
            assert!(
                seen.insert(node_path.clone()),
                "duplicate node_path {node_path:?} at anchor[{i}]"
            );
        }
    }

    #[test]
    fn docx_heading_block_present() {
        let out = extract_fixture();
        let heading = out
            .blocks
            .iter()
            .find(|b| b.block_type == BlockType::Heading.as_str())
            .expect("at least one heading block expected");
        assert_eq!(heading.text, "Test Heading");
        assert!(
            heading.section_path.contains("Test Heading"),
            "heading section_path must include the heading text; got {:?}",
            heading.section_path
        );
    }

    #[test]
    fn docx_paragraph_block_under_heading() {
        let out = extract_fixture();
        let para = out
            .blocks
            .iter()
            .find(|b| b.block_type == BlockType::Paragraph.as_str())
            .expect("at least one paragraph block expected");
        assert_eq!(para.text, "Sentinel body text for extraction.");
        assert!(
            para.section_path.contains("Test Heading"),
            "paragraph section_path must inherit the heading trail; got {:?}",
            para.section_path
        );
    }

    #[test]
    fn docx_table_block_present() {
        let out = extract_fixture();
        let tbl = out
            .blocks
            .iter()
            .find(|b| b.block_type == BlockType::Table.as_str())
            .expect("at least one table block expected");
        assert!(
            tbl.text.contains("Cell A1"),
            "table block text must include cell content; got {:?}",
            tbl.text
        );
    }

    #[test]
    fn docx_nested_table_cell_text_present() {
        use docx_rs::{Docx, Paragraph, Run, Table, TableCell, TableRow};
        let inner_table =
            Table::new(vec![TableRow::new(vec![TableCell::new().add_paragraph(
                Paragraph::new().add_run(Run::new().add_text("NestedCell")),
            )])]);
        let outer_table = Table::new(vec![TableRow::new(vec![
            TableCell::new().add_paragraph(Paragraph::new().add_run(Run::new().add_text("Outer"))),
            TableCell::new().add_table(inner_table),
        ])]);
        let docx = Docx::new().add_table(outer_table);
        let mut buf = Vec::new();
        docx.build()
            .pack(Cursor::new(&mut buf))
            .expect("build nested-table DOCX");
        let out = DocxExtractor
            .extract(&buf)
            .expect("extract nested-table DOCX");
        assert!(
            out.extracted_text.contains("NestedCell"),
            "nested cell text must appear in extracted_text; got: {:?}",
            out.extracted_text
        );
    }

    #[test]
    fn docx_block_order_heading_para_table() {
        let out = extract_fixture();
        let types: Vec<&str> = out.blocks.iter().map(|b| b.block_type.as_str()).collect();
        // We expect at minimum: heading, paragraph, table (in that order).
        let heading_pos = types.iter().position(|&t| t == BlockType::Heading.as_str());
        let para_pos = types
            .iter()
            .position(|&t| t == BlockType::Paragraph.as_str());
        let table_pos = types.iter().position(|&t| t == BlockType::Table.as_str());
        assert!(heading_pos.is_some(), "no heading block");
        assert!(para_pos.is_some(), "no paragraph block");
        assert!(table_pos.is_some(), "no table block");
        assert!(heading_pos < para_pos, "heading must come before paragraph");
        assert!(para_pos < table_pos, "paragraph must come before table");
    }

    #[test]
    fn docx_snapshot_block_structure() {
        let out = extract_fixture();
        #[derive(serde::Serialize)]
        struct BlockSnapshot<'a> {
            block_type: &'a str,
            section_path: &'a str,
            text: &'a str,
        }
        let snaps: Vec<BlockSnapshot<'_>> = out
            .blocks
            .iter()
            .map(|b| BlockSnapshot {
                block_type: &b.block_type,
                section_path: &b.section_path,
                text: &b.text,
            })
            .collect();
        insta::assert_json_snapshot!("docx_block_structure", snaps);
    }

    #[test]
    fn heading_level_detection() {
        assert_eq!(heading_level("Heading1"), Some(1));
        assert_eq!(heading_level("Heading2"), Some(2));
        assert_eq!(heading_level("Heading6"), Some(6));
        assert_eq!(heading_level("heading1"), Some(1));
        assert_eq!(heading_level("Normal"), None);
        assert_eq!(heading_level(""), None);
        assert_eq!(heading_level("HeadingX"), None);
    }

    #[test]
    fn docx_invalid_bytes_returns_parse_error() {
        let err = DocxExtractor
            .extract(b"not a docx file at all")
            .expect_err("invalid bytes must error");
        assert!(
            matches!(err, crate::LensError::Parse(_)),
            "expected LensError::Parse, got: {err:?}"
        );
    }

    #[test]
    fn section_path_push_and_current() {
        let mut sp = SectionPathStack::new();
        assert_eq!(sp.current(), "");
        sp.push(1, "Chapter 1");
        assert_eq!(sp.current(), "Chapter 1");
        sp.push(2, "Section 1.1");
        assert_eq!(sp.current(), "Chapter 1 > Section 1.1");
        sp.push(2, "Section 1.2");
        assert_eq!(sp.current(), "Chapter 1 > Section 1.2");
        sp.push(1, "Chapter 2");
        assert_eq!(sp.current(), "Chapter 2");
    }

    #[test]
    fn docx_multi_heading_section_path() {
        use docx_rs::{Docx, Paragraph, Run};
        let docx = Docx::new()
            .add_paragraph(
                Paragraph::new()
                    .add_run(Run::new().add_text("Chapter"))
                    .style("Heading1"),
            )
            .add_paragraph(
                Paragraph::new()
                    .add_run(Run::new().add_text("Section"))
                    .style("Heading2"),
            )
            .add_paragraph(Paragraph::new().add_run(Run::new().add_text("Body text")));

        let mut buf = Vec::new();
        docx.build()
            .pack(Cursor::new(&mut buf))
            .expect("build DOCX");
        let out = DocxExtractor.extract(&buf).expect("extract");

        let para = out
            .blocks
            .iter()
            .find(|b| b.text == "Body text")
            .expect("body paragraph");
        assert_eq!(
            para.section_path, "Chapter > Section",
            "body paragraph must inherit full heading trail"
        );
    }
}
