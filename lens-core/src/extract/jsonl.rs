//! JSONL / NDJSON extractor (M4 Phase 2.5c).
//!
//! [`JsonlExtractor`] treats the input as line-delimited JSON: each non-empty
//! line is parsed independently as a `serde_json::Value` and verbalized with the
//! SAME depth-first key-path walk as [`super::json`] (reused via
//! [`super::json::walk_value`]). Each record's path is seeded with a record-index
//! discriminant: the `section_path` top segment is `[N]` and the anchor path is
//! `/N/...` for the 0-based record index `N`.
//!
//! Robustness: a leading UTF-8 BOM is stripped, trailing `\r` is trimmed before
//! parse (CRLF line endings), and empty / unparseable lines are skipped with a
//! `tracing::warn!` (never silent, never an error).

use serde_json::Value;

use crate::LensError;

use super::json::{Segment, Sink, strip_bom, validate_utf8, walk_value};
use super::{ExtractOutput, Extractor};

/// JSONL extractor — implements [`Extractor`].
pub struct JsonlExtractor;

impl Extractor for JsonlExtractor {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError> {
        let s = validate_utf8(raw)?;
        let s = strip_bom(s);

        let mut sink = Sink::new();
        // 0-based record index, incremented ONLY for lines that parse — so the
        // record discriminant matches the count of successfully-parsed records.
        let mut record_index: usize = 0;

        for (line_no, raw_line) in s.split('\n').enumerate() {
            // Trim a trailing `\r` so Windows CRLF endings parse cleanly.
            let line = raw_line.trim_end_matches('\r');
            if line.trim().is_empty() {
                continue; // skip blank lines (not an error)
            }
            match serde_json::from_str::<Value>(line) {
                Ok(value) => {
                    let mut segments: Vec<Segment<'_>> = vec![Segment::Record(record_index)];
                    walk_value(&value, &mut segments, 0, &mut sink, "jsonl");
                    record_index += 1;
                }
                Err(e) => {
                    tracing::warn!(
                        line = line_no,
                        error = %e,
                        "jsonl line is not valid JSON; skipped"
                    );
                }
            }
        }

        Ok(sink.finish())
    }
}

// ---------------------------------------------------------------------------
// Tests (TDD: RED first)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::SourceAnchor;

    fn extract(src: &str) -> ExtractOutput {
        JsonlExtractor.extract(src.as_bytes()).expect("extraction")
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
    fn jsonl_two_records_two_blocks_minimum() {
        let out = extract("{\"a\":1}\n{\"b\":2}\n");
        assert!(out.blocks.len() >= 2);
        assert!(out.blocks.iter().any(|b| b.text == "/0/a: 1"));
        assert!(out.blocks.iter().any(|b| b.text == "/1/b: 2"));
    }

    #[test]
    fn jsonl_byte_identity() {
        let out = extract("{\"a\":1}\n{\"name\":\"x\",\"vals\":[1,2]}\n");
        assert!(!out.blocks.is_empty());
        assert_byte_identity(&out);
    }

    #[test]
    fn jsonl_section_path_includes_record_index() {
        let out = extract("{\"a\":1}\n{\"b\":2}\n");
        let first = out.blocks.iter().find(|b| b.text == "/0/a: 1").unwrap();
        let second = out.blocks.iter().find(|b| b.text == "/1/b: 2").unwrap();
        assert!(first.section_path.starts_with("[0]"));
        assert!(second.section_path.starts_with("[1]"));
    }

    #[test]
    fn jsonl_anchors_index_aligned() {
        let out = extract("{\"a\":1}\n{\"b\":2}\n");
        assert_eq!(out.anchors.len(), out.blocks.len());
        for a in &out.anchors {
            assert!(matches!(a, SourceAnchor::Structured { .. }));
        }
    }

    #[test]
    fn jsonl_anchor_path_includes_record_index() {
        let out = extract("{\"a\":1}\n");
        let SourceAnchor::Structured { path } = &out.anchors[0] else {
            panic!("structured");
        };
        assert_eq!(path, "/0/a");
    }

    #[test]
    fn jsonl_empty_lines_skipped() {
        let out = extract("{\"a\":1}\n\n\n{\"b\":2}\n");
        assert!(out.blocks.iter().any(|b| b.text == "/0/a: 1"));
        // The second record is index 1 (blank lines do not bump the index).
        assert!(out.blocks.iter().any(|b| b.text == "/1/b: 2"));
    }

    #[test]
    fn jsonl_invalid_line_skipped_with_warning() {
        let out = extract("{\"a\":1}\nnot json at all\n{\"b\":2}\n");
        // The valid records still extract; the bad line is skipped (not an error).
        assert!(out.blocks.iter().any(|b| b.text == "/0/a: 1"));
        assert!(out.blocks.iter().any(|b| b.text == "/1/b: 2"));
    }

    #[test]
    fn jsonl_single_line_still_works() {
        let out = extract("{\"name\":\"Alice\"}");
        assert!(out.blocks.iter().any(|b| b.text == "/0/name: Alice"));
    }

    #[test]
    fn jsonl_empty_input() {
        let out = extract("");
        assert!(out.extracted_text.is_empty());
        assert!(out.blocks.is_empty());
    }

    #[test]
    fn jsonl_crlf_line_endings() {
        let out = extract("{\"a\":1}\r\n{\"b\":2}\r\n");
        assert!(out.blocks.iter().any(|b| b.text == "/0/a: 1"));
        assert!(out.blocks.iter().any(|b| b.text == "/1/b: 2"));
        assert_byte_identity(&out);
    }

    #[test]
    fn jsonl_bom_stripped() {
        let with_bom = format!("\u{FEFF}{}", "{\"a\":1}\n");
        let a = JsonlExtractor.extract(with_bom.as_bytes()).expect("bom ok");
        let b = extract("{\"a\":1}\n");
        assert_eq!(a.extracted_text, b.extracted_text);
        assert_eq!(a.blocks, b.blocks);
    }

    #[test]
    fn jsonl_invalid_utf8_returns_validation_error() {
        let err = JsonlExtractor
            .extract(&[0xFF, 0xFE, 0x00])
            .expect_err("invalid UTF-8 errors");
        assert!(matches!(err, LensError::Validation(_)), "got {err:?}");
    }

    #[test]
    fn jsonl_multibyte_byte_identity() {
        let out = extract("{\"日本語\":\"🦀\"}\n");
        assert_byte_identity(&out);
        let b = &out.blocks[0];
        assert_eq!(b.char_end - b.char_start, b.text.len());
    }

    #[test]
    fn jsonl_snapshot_block_structure() {
        let out = extract("{\"name\":\"Alice\",\"age\":30}\n{\"name\":\"Bob\"}\n");
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
        insta::assert_json_snapshot!("jsonl_block_structure", snaps);
    }
}
