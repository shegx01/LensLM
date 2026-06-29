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

use docx_rs::{
    DocumentChild, ParagraphChild, RunChild, Table, TableCellContent, TableChild, read_docx,
};

use crate::LensError;
use crate::parse::{Block, BlockType, SectionPathStack};

use super::{ExtractOutput, Extractor, SourceAnchor};

// ---------------------------------------------------------------------------
// Heading-style detection
// ---------------------------------------------------------------------------

/// Returns the heading level (1–6) if `style_id` looks like a Word heading style
/// (`Heading1`…`Heading6`, case-insensitive, with optional space/dash separating
/// the number), otherwise `None`.
fn heading_level(style_id: &str) -> Option<u8> {
    let s = style_id.to_lowercase();
    let s = s.trim();
    // Accept "heading1", "heading 1", "heading-1" etc.
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

// ---------------------------------------------------------------------------
// Text extraction helpers
// ---------------------------------------------------------------------------

/// Extracts the plain text from a single `docx_rs::Paragraph`.
///
/// Concatenates the `text` field of every [`RunChild::Text`] across every
/// [`ParagraphChild::Run`] in document order.  All other run children
/// (line-breaks, tab stops, field chars, etc.) are silently ignored — this is
/// appropriate for Phase 2's linearisation scope.
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

/// Extracts the plain text from a `docx_rs::Table` by walking rows → cells →
/// paragraphs.  Cells are separated by `"  "` (two spaces), rows by a single
/// `"\n"`.  The result is the linearised table text used as the single
/// `"table"` block's content.
fn table_text(tbl: &Table) -> String {
    let mut rows: Vec<String> = Vec::new();
    for row_child in &tbl.rows {
        // `TableChild` / `TableRowChild` are single-variant enums in the current
        // docx-rs, so these patterns are irrefutable today; the `else { continue }`
        // is deliberate future-proofing so an additive upstream variant degrades
        // to "skip the unknown child" rather than failing to compile (matching the
        // skip-unknown pattern used for `DocumentChild` in `extract` below).
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
                if let TableCellContent::Paragraph(p) = cc {
                    let t = para_text(p);
                    if !t.is_empty() {
                        if !cell_buf.is_empty() {
                            cell_buf.push(' ');
                        }
                        cell_buf.push_str(&t);
                    }
                }
            }
            cells.push(cell_buf);
        }
        rows.push(cells.join("  "));
    }
    rows.join("\n")
}

// ---------------------------------------------------------------------------
// DocxExtractor
// ---------------------------------------------------------------------------

/// DOCX extractor — implements [`Extractor`] via `docx-rs`.
///
/// Parses the raw `.docx` bytes and walks the top-level document children:
/// - Heading paragraphs → `"heading"` blocks with an updated `section_path`.
/// - Tables → a single `"table"` block (linearised cell text).
/// - Body paragraphs → `"paragraph"` blocks.
///
/// Empty paragraphs (no run text) are skipped to avoid spurious blank blocks.
///
/// The `extracted_text` buffer is built incrementally; each block's
/// `char_start..char_end` indexes into it byte-identically.
pub struct DocxExtractor;

impl Extractor for DocxExtractor {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError> {
        // ZIP-BOMB RISK: `read_docx` (docx-rs 0.4.20 → zip 0.6.6) decompresses the
        // DOCX (a ZIP) with NO intermediate inflation limit, so a small crafted
        // archive could inflate to large transient memory here. Current mitigation
        // is bounded and accepted for Phase 2:
        //   1. Stage-1 raw-bytes cap (the configurable `max_source_bytes`, default
        //      50 MB via `AppConfig.max_source_mb`) rejects the source in
        //      `run_ingest` BEFORE this call, capping the compressed input.
        //   2. Stage-2 caps the resulting `extracted_text` length post-extraction.
        //   3. This whole extraction runs under `spawn_blocking`, so a panic
        //      (e.g. allocator abort) is isolated to the task, not the runtime.
        // Stage-1 + Stage-2 bound BOTH the input and the retained output; only the
        // transient in-`read_docx` inflation is unbounded.
        // TODO(phase-3): bound intermediate inflation (zip backend swap / size-checked reader).
        let docx = read_docx(raw)
            .map_err(|e| LensError::Parse(format!("docx-rs failed to parse DOCX: {e:?}")))?;

        let mut extracted_text = String::new();
        let mut blocks: Vec<Block> = Vec::new();
        let mut anchors: Vec<SourceAnchor> = Vec::new();

        let mut section_path = SectionPathStack::new();

        // Paragraph counter and table counter — used to build unique node_paths.
        // Each counter increments only for its own element type, so node_paths
        // are always distinct (a paragraph and a table can never share a path).
        let mut para_count: usize = 0;
        let mut tbl_count: usize = 0;

        for child in &docx.document.children {
            match child {
                DocumentChild::Paragraph(p) => {
                    let node_path = format!("body/p[{para_count}]");
                    // Increment BEFORE the empty-paragraph skip: node_path is the
                    // document-child index, which must stay stable regardless of
                    // which paragraphs are skipped (so anchors round-trip to the
                    // original DOCX position even past blank paragraphs).
                    para_count += 1;

                    let text = para_text(p);
                    if text.is_empty() {
                        continue; // skip blank paragraphs
                    }

                    // Detect heading style.
                    let style_id = p
                        .property
                        .style
                        .as_ref()
                        .map(|s| s.val.as_str())
                        .unwrap_or("");
                    let level = heading_level(style_id);

                    // Update section_path BEFORE emitting the heading block so
                    // the heading itself carries the full trail it introduces.
                    if let Some(lvl) = level {
                        section_path.push(lvl, &text);
                    }

                    let btype = if level.is_some() {
                        BlockType::Heading.as_str()
                    } else {
                        BlockType::Paragraph.as_str()
                    };

                    // The section_path for a heading is the trail INCLUDING itself
                    // (consistent with parse.rs: the heading block's section_path
                    // includes its own text).
                    let sp = section_path.current();

                    let char_start = extracted_text.len();
                    extracted_text.push_str(&text);
                    extracted_text.push('\n');
                    let char_end = extracted_text.len() - 1; // exclude the trailing \n from the block slice

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
                    let char_end = extracted_text.len() - 1; // exclude trailing \n

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

        // Trim trailing newlines from extracted_text while keeping all block
        // offsets valid (blocks were built against indices before trimming, so
        // we only trim content that's beyond the last block's char_end).
        // This is safe because the last '\n' was never included in any block's
        // char_end.
        while extracted_text.ends_with('\n') {
            extracted_text.pop();
        }

        Ok(ExtractOutput {
            extracted_text,
            blocks,
            anchors,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests (TDD: RED written first, GREEN implemented above)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::extract::ExtractOutput;

    // -----------------------------------------------------------------------
    // Fixture builder
    // -----------------------------------------------------------------------

    /// Builds a tiny DOCX in memory using `docx-rs`'s write API:
    ///   - one Heading1 paragraph ("Test Heading")
    ///   - one body paragraph ("Sentinel body text for extraction.")
    ///   - one 2×2 table with cells "Cell A1" / "Cell A2" / "Cell B1" / "Cell B2"
    ///
    /// This is the SAME DOCX reused by all tests in this module.
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

    // -----------------------------------------------------------------------
    // AC4a — byte-identity: extracted_text[b.char_start..b.char_end] == b.text
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // AC4c — extraction fidelity: known substrings present in extracted_text
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // AC5 — anchors: len == blocks.len(), all Docx, node_paths distinct
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // AC5 / AC10 — structural: block types, section_path, heading detection
    // -----------------------------------------------------------------------

    #[test]
    fn docx_heading_block_present() {
        let out = extract_fixture();
        let heading = out
            .blocks
            .iter()
            .find(|b| b.block_type == BlockType::Heading.as_str())
            .expect("at least one heading block expected");
        assert_eq!(heading.text, "Test Heading");
        // The heading's section_path includes its own text (mirrors parse.rs).
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
        // The paragraph is nested under the H1 heading.
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

    // -----------------------------------------------------------------------
    // AC10 — snapshot: stable block structure
    // -----------------------------------------------------------------------

    #[test]
    fn docx_snapshot_block_structure() {
        let out = extract_fixture();
        // Serialise just the structural fields (not char offsets, which are
        // layout-dependent) so the snapshot remains stable.
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

    // -----------------------------------------------------------------------
    // Heading-level helper
    // -----------------------------------------------------------------------

    #[test]
    fn heading_level_detection() {
        assert_eq!(heading_level("Heading1"), Some(1));
        assert_eq!(heading_level("Heading2"), Some(2));
        assert_eq!(heading_level("Heading6"), Some(6));
        assert_eq!(heading_level("heading1"), Some(1)); // case-insensitive
        assert_eq!(heading_level("Normal"), None);
        assert_eq!(heading_level(""), None);
        assert_eq!(heading_level("HeadingX"), None);
    }

    // -----------------------------------------------------------------------
    // Error handling: invalid bytes
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Section-path stack
    // -----------------------------------------------------------------------

    #[test]
    fn section_path_push_and_current() {
        let mut sp = SectionPathStack::new();
        assert_eq!(sp.current(), "");
        sp.push(1, "Chapter 1");
        assert_eq!(sp.current(), "Chapter 1");
        sp.push(2, "Section 1.1");
        assert_eq!(sp.current(), "Chapter 1 > Section 1.1");
        // A second H2 replaces the previous H2 and all H3+.
        sp.push(2, "Section 1.2");
        assert_eq!(sp.current(), "Chapter 1 > Section 1.2");
        // An H1 clears everything below level 1 and replaces it.
        sp.push(1, "Chapter 2");
        assert_eq!(sp.current(), "Chapter 2");
    }

    // -----------------------------------------------------------------------
    // Multi-heading section_path inheritance
    // -----------------------------------------------------------------------

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
