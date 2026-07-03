//! CSV extractor (M4 issue #76) — TableRAG row-verbalization.
//!
//! [`CsvExtractor`] parses a comma-delimited, RFC-4180 CSV from raw bytes (via
//! the `csv` crate) and produces a canonical [`ExtractOutput`] where each DATA
//! row becomes one header-paired, key:value [`Block`]:
//! `"Header1: val1; Header2: val2; ..."`. Headers are embedded inline so column
//! semantics drive retrieval (`section_path`/`block_type` are metadata only).
//!
//! The first row is the header. If it is empty or absent, synthetic
//! `"Column 1"`, `"Column 2"`, ... headers are used. Short data rows are padded
//! with empty values; long rows are truncated to the header width. Blank and
//! duplicate headers are kept verbatim (per spec).
//!
//! A pipe-delimited markdown rendering is produced DURING this single parse and
//! carried on [`ExtractOutput::table_markdown`] (never embedded; persisted by
//! ingest as the `{id}.tables.md` sibling).
//!
//! Byte-identity invariant: each block's `char_start..char_end` slices the
//! canonical buffer exactly (build-as-you-go append, offsets via `String::len()`).
//! Anchors are `SourceAnchor::Structured { path: "/row[{n}]" }` (1-indexed data
//! rows; NO sheet prefix — CSV has no sheet concept).

use crate::LensError;
use crate::parse::{Block, BlockType};

use super::tabular_utils::{MAX_COLUMNS, normalize_headers, render_table_markdown, verbalize_row};
use super::{ExtractOutput, Extractor, SourceAnchor};

/// Strips a leading UTF-8 BOM (`EF BB BF`); the `csv` crate does not strip it,
/// so an Excel-exported CSV would otherwise carry BOM into the first header.
fn strip_bom(raw: &[u8]) -> &[u8] {
    if raw.starts_with(&[0xEF, 0xBB, 0xBF]) {
        &raw[3..]
    } else {
        raw
    }
}

/// CSV extractor — implements [`Extractor`] via the `csv` crate.
pub struct CsvExtractor;

impl Extractor for CsvExtractor {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError> {
        let raw = strip_bom(raw);

        let mut reader = ::csv::ReaderBuilder::new()
            .flexible(true)
            .has_headers(false)
            .from_reader(raw);

        let mut records: Vec<Vec<String>> = Vec::new();
        for result in reader.records() {
            let rec = result.map_err(|e| LensError::Parse(format!("invalid CSV: {e}")))?;
            records.push(rec.iter().map(str::to_string).collect());
        }

        if records.is_empty() {
            return Ok(ExtractOutput {
                extracted_text: String::new(),
                blocks: Vec::new(),
                anchors: Vec::new(),
                table_markdown: None,
            });
        }

        let first = records.remove(0);
        let headers = normalize_headers(first);

        // Guard the widest record seen; `flexible(true)` means a data row can exceed the header width.
        let max_cols = headers
            .len()
            .max(records.iter().map(Vec::len).max().unwrap_or(0));
        if max_cols > MAX_COLUMNS {
            return Err(LensError::Validation(format!(
                "tabular source has {max_cols} columns, exceeding the {MAX_COLUMNS}-column limit"
            )));
        }

        let data_rows = records;

        let mut extracted_text = String::new();
        let mut blocks: Vec<Block> = Vec::new();
        let mut anchors: Vec<SourceAnchor> = Vec::new();

        for (i, row) in data_rows.iter().enumerate() {
            let line = verbalize_row(&headers, row);
            let char_start = extracted_text.len();
            extracted_text.push_str(&line);
            let char_end = extracted_text.len();
            extracted_text.push('\n');

            blocks.push(Block {
                block_type: BlockType::Table.as_str().to_string(),
                section_path: String::new(),
                text: line,
                char_start,
                char_end,
            });
            anchors.push(SourceAnchor::Structured {
                path: format!("/row[{}]", i + 1),
            });
        }

        let table_markdown = render_table_markdown(&headers, &data_rows);

        Ok(ExtractOutput {
            extracted_text,
            blocks,
            anchors,
            table_markdown: Some(table_markdown),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(src: &str) -> ExtractOutput {
        CsvExtractor
            .extract(src.as_bytes())
            .expect("csv extraction")
    }

    fn extract_bytes(raw: &[u8]) -> ExtractOutput {
        CsvExtractor.extract(raw).expect("csv extraction")
    }

    fn assert_byte_identity(out: &ExtractOutput) {
        for (i, b) in out.blocks.iter().enumerate() {
            assert_eq!(
                &out.extracted_text[b.char_start..b.char_end],
                b.text,
                "byte-identity violated for block[{i}]"
            );
        }
    }

    #[test]
    fn csv_simple_3col() {
        let out = extract("Name,Age,City\nAlice,30,NYC\nBob,25,LA\n");
        assert_eq!(out.blocks.len(), 2, "two data rows");
        for b in &out.blocks {
            assert_eq!(b.block_type, BlockType::Table.as_str());
            assert_eq!(b.section_path, "", "CSV has no sheet name");
        }
        assert_eq!(out.blocks[0].text, "Name: Alice; Age: 30; City: NYC");
        assert_eq!(out.blocks[1].text, "Name: Bob; Age: 25; City: LA");

        let paths: Vec<&str> = out
            .anchors
            .iter()
            .map(|a| match a {
                SourceAnchor::Structured { path } => path.as_str(),
                _ => panic!("non-structured anchor"),
            })
            .collect();
        assert_eq!(paths, vec!["/row[1]", "/row[2]"]);
    }

    #[test]
    fn csv_byte_identity() {
        let out = extract("Name,Age,City\nAlice,30,NYC\nBob,25,LA\n");
        assert!(!out.blocks.is_empty());
        assert_byte_identity(&out);
    }

    #[test]
    fn csv_no_header_fallback() {
        let out = extract(",,\nAlice,30,NYC\n");
        assert_eq!(out.blocks.len(), 1);
        assert_eq!(
            out.blocks[0].text,
            "Column 1: Alice; Column 2: 30; Column 3: NYC"
        );
        assert_byte_identity(&out);
    }

    #[test]
    fn csv_empty_file() {
        let out = extract("");
        assert!(out.blocks.is_empty());
        assert!(out.extracted_text.is_empty());
        assert!(out.table_markdown.is_none(), "no records → no markdown");
    }

    #[test]
    fn csv_header_only() {
        let out = extract("Name,Age,City\n");
        assert!(out.blocks.is_empty(), "header-only → no data rows");
        assert!(out.extracted_text.is_empty());
        let md = out.table_markdown.expect("markdown present");
        assert!(md.contains("| Name | Age | City |"));
    }

    #[test]
    fn csv_embedded_commas_quotes() {
        let out = extract("Name,Nickname\nAl,\"Big, Al\"\n");
        assert_eq!(out.blocks.len(), 1);
        assert_eq!(out.blocks[0].text, "Name: Al; Nickname: Big, Al");
        assert_byte_identity(&out);
    }

    #[test]
    fn csv_multibyte_utf8() {
        let out = extract("名前,年齢\nアリス,30\n");
        assert_eq!(out.blocks.len(), 1);
        assert_byte_identity(&out);
        let b = &out.blocks[0];
        assert_eq!(b.char_end - b.char_start, b.text.len());
        assert!(b.text.contains("アリス"));
    }

    #[test]
    fn csv_ragged_rows_short_padded() {
        let out = extract("A,B,C\n1,2\n");
        assert_eq!(out.blocks.len(), 1);
        assert_eq!(out.blocks[0].text, "A: 1; B: 2; C: ");
        assert_byte_identity(&out);
    }

    #[test]
    fn csv_ragged_rows_long_truncated() {
        let out = extract("A,B,C\n1,2,3,4\n");
        assert_eq!(out.blocks.len(), 1);
        assert_eq!(out.blocks[0].text, "A: 1; B: 2; C: 3");
        assert!(!out.blocks[0].text.contains("4"), "extra column dropped");
        assert_byte_identity(&out);
    }

    #[test]
    fn csv_duplicate_blank_headers() {
        let out = extract("Name,Name,\na,b,c\n");
        assert_eq!(out.blocks.len(), 1);
        assert_eq!(out.blocks[0].text, "Name: a; Name: b; : c");
        assert_byte_identity(&out);
    }

    #[test]
    fn csv_bom_header() {
        let mut raw: Vec<u8> = vec![0xEF, 0xBB, 0xBF];
        raw.extend_from_slice(b"Name,Age\nAlice,30\n");
        let out = extract_bytes(&raw);
        assert_eq!(out.blocks.len(), 1);
        assert_eq!(out.blocks[0].text, "Name: Alice; Age: 30");
        assert!(
            !out.blocks[0].text.contains('\u{FEFF}'),
            "BOM must not leak into the first header"
        );
    }

    #[test]
    fn csv_table_markdown_field() {
        let out = extract("Name,Age\nAlice,30\nBob,25\n");
        let md = out.table_markdown.expect("markdown present");
        assert!(md.contains("| Name | Age |"), "header row: {md:?}");
        assert!(md.contains("| --- | --- |"), "separator row: {md:?}");
        assert!(md.contains("| Alice | 30 |"), "data row: {md:?}");
        assert!(
            !out.extracted_text.contains('|'),
            "markdown must not leak into extracted_text"
        );
    }

    #[test]
    fn csv_too_many_columns_rejected() {
        let header: String = (0..=MAX_COLUMNS)
            .map(|i| format!("H{i}"))
            .collect::<Vec<_>>()
            .join(",");
        let src = format!("{header}\n");
        let err = CsvExtractor
            .extract(src.as_bytes())
            .expect_err("over-wide header must be rejected");
        match err {
            LensError::Validation(msg) => {
                assert!(msg.contains("columns"), "msg: {msg}");
                assert!(msg.contains(&MAX_COLUMNS.to_string()), "msg: {msg}");
            }
            other => panic!("expected Validation, got {other:?}"),
        }

        let ok_header: String = (0..MAX_COLUMNS)
            .map(|i| format!("H{i}"))
            .collect::<Vec<_>>()
            .join(",");
        CsvExtractor
            .extract(format!("{ok_header}\n").as_bytes())
            .expect("exactly MAX_COLUMNS columns must be accepted");
    }

    #[test]
    fn csv_snapshot_block_structure() {
        let out = extract("Name,Age,City\nAlice,30,NYC\nBob,25,LA\n");
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
        insta::assert_json_snapshot!("csv_block_structure", snaps);
    }
}
