//! Spreadsheet extractor (M4 issue #76) — TableRAG row-verbalization for XLSX/XLS.
//!
//! [`SpreadsheetExtractor`] opens a workbook from raw bytes via
//! `calamine::open_workbook_auto_from_rs(Cursor::new(raw))` (serving BOTH XLSX
//! and XLS) and produces a canonical [`ExtractOutput`] where each DATA row of
//! each sheet becomes one header-paired, key:value [`Block`]:
//! `"Header1: val1; Header2: val2; ..."`. Headers are embedded inline so column
//! semantics drive retrieval; the sheet name is carried as `section_path`
//! (metadata only — NOT embedded).
//!
//! Per sheet, the first row is the header. If it is empty/all-blank, synthetic
//! `"Column 1"`, `"Column 2"`, ... headers are used. Short rows are padded with
//! empty values; long rows are truncated to the header width. Blank/duplicate
//! headers are kept verbatim.
//!
//! A pipe-delimited markdown rendering (one `## {sheet}` section per sheet) is
//! produced DURING this single parse and carried on
//! [`ExtractOutput::table_markdown`] (never embedded; persisted by ingest as the
//! `{id}.tables.md` sibling).
//!
//! Byte-identity invariant: each block's `char_start..char_end` slices the
//! canonical buffer exactly (build-as-you-go append, offsets via `String::len()`).
//! Anchors are `SourceAnchor::Structured { path: "/{sheet}/row[{n}]" }`
//! (1-indexed data rows).

use std::io::Cursor;

use calamine::{Data, Reader, open_workbook_auto_from_rs};

use crate::LensError;
use crate::parse::{Block, BlockType};

use super::tabular_utils::{MAX_COLUMNS, normalize_headers, render_table_markdown, verbalize_row};
use super::{ExtractOutput, Extractor, SourceAnchor};

/// Cumulative cell-data byte ceiling (decompression-bomb guard). `calamine`
/// decompresses XLSX internally, so the 50 MB stage-1 compressed cap is not
/// enough; a high-ratio workbook could OOM before stage-2. We sum cell byte
/// lengths as we iterate and bail past this cap.
///
/// Residual: `calamine` fully materialises a sheet's range before we can
/// measure cells, so this bounds downstream amplification only.
const MAX_DECOMPRESSED_BYTES: usize = 256 * 1024 * 1024;

/// Converts a calamine [`Data`] cell to its verbalized string form.
///
/// Float integers render without a trailing `.0` (`30.0` → `"30"`). The i64
/// range guard prevents saturation on out-of-range values (e.g. `1e19`).
pub(crate) fn cell_to_string(cell: &Data) -> String {
    match cell {
        Data::String(s) => s.clone(),
        Data::Float(f) => {
            // Only use integer formatting when the value is a whole number AND
            // inside i64 range; otherwise `*f as i64` saturates silently.
            if f.fract() == 0.0 && f.is_finite() && *f >= i64::MIN as f64 && *f <= i64::MAX as f64 {
                format!("{}", *f as i64)
            } else {
                format!("{f}")
            }
        }
        Data::Int(i) => i.to_string(),
        Data::Bool(b) => b.to_string(),
        Data::DateTime(dt) => format!("{dt}"),
        Data::DateTimeIso(s) => s.clone(),
        Data::DurationIso(s) => s.clone(),
        Data::Error(e) => format!("#ERR:{e:?}"),
        Data::Empty => String::new(),
    }
}

/// Byte cost proxy for a cell (used by the decompression-bomb guard).
/// `String` charges its length; all other variants charge a flat `8`.
fn cell_data_bytes(cell: &Data) -> usize {
    match cell {
        Data::String(s) => s.len(),
        _ => 8,
    }
}

/// Spreadsheet extractor — implements [`Extractor`] via `calamine` (XLSX + XLS).
pub struct SpreadsheetExtractor;

impl Extractor for SpreadsheetExtractor {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError> {
        super::guard_zip_entry_count(raw)?;
        // `open_workbook_auto_from_rs` needs `Read+Seek+Clone`; owned Vec cursor.
        let cursor = Cursor::new(raw.to_vec());
        let mut workbook = open_workbook_auto_from_rs(cursor)
            .map_err(|e| LensError::Parse(format!("calamine failed to open workbook: {e}")))?;

        let mut extracted_text = String::new();
        let mut blocks: Vec<Block> = Vec::new();
        let mut anchors: Vec<SourceAnchor> = Vec::new();
        let mut table_markdown = String::new();
        let mut total_cell_bytes: usize = 0;

        let sheet_names = workbook.sheet_names().to_vec();
        for sheet_name in sheet_names {
            let range = workbook
                .worksheet_range(&sheet_name)
                .map_err(|e| LensError::Parse(format!("calamine sheet {sheet_name:?}: {e}")))?;

            // Collect every row as owned String cells (single parse).
            let mut rows_iter = range.rows();
            let header_row = match rows_iter.next() {
                Some(r) => r,
                None => continue, // empty sheet → no blocks
            };

            total_cell_bytes += header_row.iter().map(cell_data_bytes).sum::<usize>();
            if total_cell_bytes > MAX_DECOMPRESSED_BYTES {
                return Err(LensError::Validation(format!(
                    "spreadsheet decompresses to >{MAX_DECOMPRESSED_BYTES} bytes of \
                     cell data (possible decompression bomb)"
                )));
            }

            let first: Vec<String> = header_row.iter().map(cell_to_string).collect();
            let headers = normalize_headers(first);

            // Column guard: calamine fixes the range width at the header, so
            // guarding the header count bounds every data row too.
            if headers.len() > MAX_COLUMNS {
                return Err(LensError::Validation(format!(
                    "tabular source has {} columns, exceeding the {MAX_COLUMNS}-column limit",
                    headers.len()
                )));
            }

            let mut data_rows: Vec<Vec<String>> = Vec::new();
            for r in rows_iter {
                total_cell_bytes += r.iter().map(cell_data_bytes).sum::<usize>();
                if total_cell_bytes > MAX_DECOMPRESSED_BYTES {
                    return Err(LensError::Validation(format!(
                        "spreadsheet decompresses to >{MAX_DECOMPRESSED_BYTES} bytes of \
                         cell data (possible decompression bomb)"
                    )));
                }
                data_rows.push(r.iter().map(cell_to_string).collect());
            }

            for (i, row) in data_rows.iter().enumerate() {
                let line = verbalize_row(&headers, row);
                let char_start = extracted_text.len();
                extracted_text.push_str(&line);
                let char_end = extracted_text.len();
                extracted_text.push('\n');

                blocks.push(Block {
                    block_type: BlockType::Table.as_str().to_string(),
                    section_path: sheet_name.clone(),
                    text: line,
                    char_start,
                    char_end,
                });
                anchors.push(SourceAnchor::Structured {
                    path: format!("/{sheet_name}/row[{}]", i + 1),
                });
            }

            // Render now so `data_rows` can be dropped before the next sheet.
            table_markdown.push_str(&format!("## {sheet_name}\n\n"));
            table_markdown.push_str(&render_table_markdown(&headers, &data_rows));
            table_markdown.push('\n');
            drop(data_rows);
        }

        // Emit None (not Some("")) for empty workbooks, matching CsvExtractor.
        let table_markdown = if table_markdown.is_empty() {
            None
        } else {
            Some(table_markdown)
        };

        Ok(ExtractOutput {
            extracted_text,
            blocks,
            anchors,
            table_markdown,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calamine::{CellErrorType, ExcelDateTime, ExcelDateTimeType};

    /// Regenerates `tests/fixtures/sample.xlsx`. Run with `--ignored`.
    #[test]
    #[ignore = "fixture generator; run with --ignored to regenerate sample.xlsx"]
    fn regenerate_sample_xlsx() {
        use rust_xlsxwriter::Workbook;

        let mut wb = Workbook::new();

        let people = wb.add_worksheet().set_name("People").unwrap();
        people.write_string(0, 0, "Name").unwrap();
        people.write_string(0, 1, "Age").unwrap();
        people.write_string(1, 0, "Alice").unwrap();
        people.write_number(1, 1, 30.0).unwrap();
        people.write_string(2, 0, "Bob").unwrap();
        people.write_number(2, 1, 25.0).unwrap();

        let cities = wb.add_worksheet().set_name("Cities").unwrap();
        cities.write_string(0, 0, "City").unwrap();
        cities.write_string(0, 1, "Pop").unwrap();
        cities.write_string(1, 0, "NYC").unwrap();
        cities.write_number(1, 1, 8_000_000.0).unwrap();

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/sample.xlsx");
        wb.save(path).expect("write sample.xlsx fixture");
    }

    fn fixture_bytes() -> Vec<u8> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/sample.xlsx");
        std::fs::read(path).expect("sample.xlsx fixture must exist (run regenerate_sample_xlsx)")
    }

    fn extract_fixture() -> ExtractOutput {
        SpreadsheetExtractor
            .extract(&fixture_bytes())
            .expect("xlsx extraction")
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

    fn build_xlsx(sheets: &[(&str, &[&[&str]])]) -> Vec<u8> {
        use rust_xlsxwriter::Workbook;
        let mut wb = Workbook::new();
        for (name, rows) in sheets {
            let ws = wb.add_worksheet().set_name(*name).unwrap();
            for (r, row) in rows.iter().enumerate() {
                for (c, val) in row.iter().enumerate() {
                    ws.write_string(r as u32, c as u16, *val).unwrap();
                }
            }
        }
        wb.save_to_buffer().expect("save xlsx to buffer")
    }

    #[test]
    fn xlsx_multibyte_utf8() {
        let bytes = build_xlsx(&[("S", &[&["名前", "年齢"], &["アリス", "30"]])]);
        let out = SpreadsheetExtractor.extract(&bytes).expect("extract");
        assert_eq!(out.blocks.len(), 1);
        assert_byte_identity(&out);
        let b = &out.blocks[0];
        assert_eq!(b.char_end - b.char_start, b.text.len(), "BYTE offsets");
        assert!(b.text.contains("アリス"));
    }

    #[test]
    fn xlsx_empty_sheet() {
        let bytes = build_xlsx(&[("Empty", &[])]);
        let out = SpreadsheetExtractor.extract(&bytes).expect("extract");
        assert!(out.blocks.is_empty(), "empty sheet → no blocks");
    }

    #[test]
    fn xlsx_header_only_sheet() {
        let bytes = build_xlsx(&[("H", &[&["Name", "Age"]])]);
        let out = SpreadsheetExtractor.extract(&bytes).expect("extract");
        assert!(out.blocks.is_empty(), "header-only → no data rows");
        // Markdown still carries the header + separator.
        let md = out.table_markdown.expect("markdown present");
        assert!(md.contains("## H"));
        assert!(md.contains("| Name | Age |"));
    }

    // NOTE: "blank first row" is not tested end-to-end — calamine collapses a
    // fully-blank written row, so the synthetic-header path is covered by
    // `xlsx_no_header_fallback` instead.

    #[test]
    fn xlsx_simple_sheet() {
        let out = extract_fixture();
        let people: Vec<&Block> = out
            .blocks
            .iter()
            .filter(|b| b.section_path == "People")
            .collect();
        assert_eq!(people.len(), 2, "two data rows on People");
        for b in &people {
            assert_eq!(b.block_type, BlockType::Table.as_str());
        }
        assert_eq!(people[0].text, "Name: Alice; Age: 30");
        assert_eq!(people[1].text, "Name: Bob; Age: 25");
    }

    #[test]
    fn xlsx_byte_identity() {
        let out = extract_fixture();
        assert!(!out.blocks.is_empty());
        assert_byte_identity(&out);
    }

    #[test]
    fn xlsx_multi_sheet() {
        let out = extract_fixture();
        // People + Cities blocks, with correct section_path + anchors.
        let cities: Vec<&Block> = out
            .blocks
            .iter()
            .filter(|b| b.section_path == "Cities")
            .collect();
        assert_eq!(cities.len(), 1, "one data row on Cities");
        assert_eq!(cities[0].text, "City: NYC; Pop: 8000000");

        let paths: Vec<&str> = out
            .anchors
            .iter()
            .map(|a| match a {
                SourceAnchor::Structured { path } => path.as_str(),
                _ => panic!("non-structured anchor"),
            })
            .collect();
        assert!(paths.contains(&"/People/row[1]"));
        assert!(paths.contains(&"/People/row[2]"));
        assert!(paths.contains(&"/Cities/row[1]"));
    }

    #[test]
    fn xlsx_table_markdown_field() {
        let out = extract_fixture();
        let md = out.table_markdown.expect("markdown present");
        assert!(md.contains("## People"), "People section: {md:?}");
        assert!(md.contains("## Cities"), "Cities section: {md:?}");
        assert!(md.contains("| Name | Age |"));
        assert!(md.contains("| Alice | 30 |"));
        assert!(
            !out.extracted_text.contains('|'),
            "markdown must not leak into extracted_text"
        );
    }

    #[test]
    fn xlsx_all_empty_workbook_markdown_none() {
        // Every sheet empty/skipped → no blocks AND table_markdown == None
        // (matching CsvExtractor's None-for-empty behavior, not Some("")).
        let bytes = build_xlsx(&[("Empty", &[])]);
        let out = SpreadsheetExtractor.extract(&bytes).expect("extract");
        assert!(out.blocks.is_empty(), "no data → no blocks");
        assert!(out.extracted_text.is_empty());
        assert!(
            out.table_markdown.is_none(),
            "all-empty workbook → table_markdown None, got {:?}",
            out.table_markdown
        );
    }

    #[test]
    fn xlsx_snapshot_block_structure() {
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
        insta::assert_json_snapshot!("xlsx_block_structure", snaps);
    }

    #[test]
    fn xlsx_invalid_bytes_returns_error() {
        let err = SpreadsheetExtractor
            .extract(b"definitely not a workbook")
            .expect_err("invalid bytes must error");
        assert!(matches!(err, LensError::Parse(_)), "got {err:?}");
    }

    #[test]
    fn xlsx_cell_string_variant() {
        assert_eq!(cell_to_string(&Data::String("hi".to_string())), "hi");
    }

    #[test]
    fn xlsx_cell_float_integer_no_trailing_zero() {
        assert_eq!(cell_to_string(&Data::Float(30.0)), "30");
    }

    #[test]
    fn xlsx_cell_float_fraction() {
        assert_eq!(cell_to_string(&Data::Float(2.5)), "2.5");
    }

    #[test]
    fn xlsx_cell_float_negative_integer() {
        assert_eq!(cell_to_string(&Data::Float(-30.0)), "-30");
        assert_eq!(cell_to_string(&Data::Float(30.0)), "30");
        assert_eq!(cell_to_string(&Data::Float(30.5)), "30.5");
    }

    #[test]
    fn xlsx_cell_float_out_of_i64_range_does_not_saturate() {
        // 1e19 > i64::MAX; must fall through to Display rather than saturate.
        assert_eq!(cell_to_string(&Data::Float(1e19)), "10000000000000000000");
        assert_ne!(
            cell_to_string(&Data::Float(1e19)),
            i64::MAX.to_string(),
            "must not saturate to i64::MAX"
        );
    }

    #[test]
    fn xlsx_cell_float_non_finite_falls_through() {
        assert_eq!(cell_to_string(&Data::Float(f64::NAN)), "NaN");
        assert_eq!(cell_to_string(&Data::Float(f64::INFINITY)), "inf");
        assert_eq!(cell_to_string(&Data::Float(f64::NEG_INFINITY)), "-inf");
    }

    #[test]
    fn xlsx_decompression_bomb_guard() {
        // Exercise the helper directly; a real 256 MB+ workbook is infeasible in-test.
        assert_eq!(cell_data_bytes(&Data::String("x".repeat(100))), 100);
        assert_eq!(cell_data_bytes(&Data::Int(42)), 8);
        assert_eq!(cell_data_bytes(&Data::Float(30.0)), 8);
        assert_eq!(cell_data_bytes(&Data::Empty), 8);

        let big_cell = Data::String("a".repeat(MAX_DECOMPRESSED_BYTES + 1));
        let mut total = 0usize;
        total += cell_data_bytes(&big_cell);
        assert!(
            total > MAX_DECOMPRESSED_BYTES,
            "synthetic cell-byte total must exceed the cap"
        );

        let small_total: usize = [&Data::Int(1), &Data::String("hi".to_string())]
            .iter()
            .map(|c| cell_data_bytes(c))
            .sum();
        assert!(small_total <= MAX_DECOMPRESSED_BYTES);
    }

    #[test]
    fn xlsx_too_many_columns_rejected() {
        // >16384-column workbook is infeasible in-test; drive the guard predicate directly.
        let guard = |ncols: usize| -> Option<LensError> {
            (ncols > MAX_COLUMNS).then(|| {
                LensError::Validation(format!(
                    "tabular source has {ncols} columns, exceeding the {MAX_COLUMNS}-column limit"
                ))
            })
        };

        assert!(guard(MAX_COLUMNS).is_none(), "exactly MAX_COLUMNS is valid");

        match guard(MAX_COLUMNS + 1) {
            Some(LensError::Validation(msg)) => {
                assert!(msg.contains("columns"), "msg: {msg}");
                assert!(msg.contains(&MAX_COLUMNS.to_string()), "msg: {msg}");
            }
            other => panic!("expected Some(Validation), got {other:?}"),
        }
    }

    #[test]
    fn xlsx_cell_int() {
        assert_eq!(cell_to_string(&Data::Int(42)), "42");
    }

    #[test]
    fn xlsx_cell_bool() {
        assert_eq!(cell_to_string(&Data::Bool(true)), "true");
        assert_eq!(cell_to_string(&Data::Bool(false)), "false");
    }

    #[test]
    fn xlsx_cell_datetime_uses_display() {
        let dt = ExcelDateTime::new(44484.0, ExcelDateTimeType::DateTime, false);
        assert_eq!(cell_to_string(&Data::DateTime(dt)), format!("{dt}"));
    }

    #[test]
    fn xlsx_cell_datetime_iso_clones_string() {
        assert_eq!(
            cell_to_string(&Data::DateTimeIso("2024-01-15T00:00:00".to_string())),
            "2024-01-15T00:00:00"
        );
    }

    #[test]
    fn xlsx_cell_duration_iso_clones_string() {
        assert_eq!(
            cell_to_string(&Data::DurationIso("PT1H30M".to_string())),
            "PT1H30M"
        );
    }

    #[test]
    fn xlsx_cell_error() {
        assert_eq!(
            cell_to_string(&Data::Error(CellErrorType::Div0)),
            "#ERR:Div0"
        );
    }

    #[test]
    fn xlsx_cell_empty() {
        assert_eq!(cell_to_string(&Data::Empty), "");
    }

    #[test]
    fn xlsx_no_header_fallback() {
        let blank: Vec<String> = vec![String::new(), String::new()];
        let headers: Vec<String> = if blank.iter().all(|c| c.trim().is_empty()) {
            (0..blank.len())
                .map(|i| format!("Column {}", i + 1))
                .collect()
        } else {
            blank
        };
        let row = vec!["Alice".to_string(), "30".to_string()];
        assert_eq!(
            verbalize_row(&headers, &row),
            "Column 1: Alice; Column 2: 30"
        );
    }

    #[test]
    fn xlsx_duplicate_blank_headers() {
        let headers = vec!["Name".to_string(), "Name".to_string(), String::new()];
        let row = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(verbalize_row(&headers, &row), "Name: a; Name: b; : c");
    }

    #[test]
    fn xlsx_ragged_short_padded_and_long_truncated() {
        let headers = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        assert_eq!(
            verbalize_row(&headers, &["1".to_string(), "2".to_string()]),
            "A: 1; B: 2; C: "
        );
        assert_eq!(
            verbalize_row(
                &headers,
                &[
                    "1".to_string(),
                    "2".to_string(),
                    "3".to_string(),
                    "4".to_string()
                ]
            ),
            "A: 1; B: 2; C: 3"
        );
    }
}
