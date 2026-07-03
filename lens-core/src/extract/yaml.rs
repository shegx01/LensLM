//! YAML extractor (M4 Phase 2.5c).
//!
//! [`YamlExtractor`] parses YAML into a `serde_json::Value` via
//! `serde-saphyr` (`=0.0.28`, verified by the Step 0a spike) and verbalizes it
//! with the SAME depth-first key-path walk as [`super::json`] (reused via
//! [`super::json::walk_value`]).
//!
//! Multi-document handling: `serde_saphyr::from_str` deserializes only the FIRST
//! document, so multi-document streams are split FIRST with a spec-aware boundary
//! scan ([`split_documents`]) — a `---` line at column 0 (NOT inside an indented
//! block scalar) starts a new document; a `...` line ends one. When more than one
//! document is present, each document's path is seeded with a `[N]` record
//! discriminant (`section_path`) / `/N` (anchor path); a single document carries
//! NO `[N]` prefix.
//!
//! Determinism: like JSON, mapping keys traverse in alphabetical (BTreeMap)
//! order, so the canonical buffer is deterministic regardless of source key
//! order. A leading UTF-8 BOM is stripped before parsing.

use serde_json::Value;
use serde_saphyr::Options;

use crate::LensError;

use super::json::{Segment, Sink, strip_bom, validate_utf8, walk_value};
use super::{ExtractOutput, Extractor};

/// Cap on documents from a multi-doc stream; excess are dropped with `tracing::warn!`.
const MAX_YAML_DOCUMENTS: usize = 1024;

/// Tighter nesting budget than serde_saphyr's default (2000); rejects adversarial
/// input far earlier while staying above any legitimate config document.
const YAML_MAX_DEPTH: usize = 128;

/// Large stack for the YAML parse thread. serde_saphyr enforces `YAML_MAX_DEPTH`
/// WHILE the recursive deserializer pulls events, recursing to ~3 MB at depth 128
/// BEFORE the breach fires. A `spawn_blocking` thread's ~2 MB default overflows
/// before the budget error is returned; 8 MB guarantees a clean `Err` instead.
const YAML_PARSE_STACK_BYTES: usize = 8 * 1024 * 1024;

fn parse_yaml_document(doc_src: &str) -> Result<Value, LensError> {
    let owned = doc_src.to_owned();
    let handle = std::thread::Builder::new()
        .stack_size(YAML_PARSE_STACK_BYTES)
        .spawn(move || {
            serde_saphyr::from_str_with_options::<Value>(&owned, yaml_options())
                .map_err(|e| LensError::Parse(format!("invalid YAML: {e}")))
        })
        .map_err(|e| LensError::Parse(format!("failed to spawn YAML parse thread: {e}")))?;

    handle
        .join()
        .map_err(|_| LensError::Parse("YAML parse thread panicked".to_string()))?
}

/// Builds serde_saphyr options with our tightened depth budget. `options!`/`budget!`
/// macros are the crate's supported path (direct struct construction is `#[deprecated]`).
fn yaml_options() -> Options {
    serde_saphyr::options! {
        budget: serde_saphyr::budget! {
            max_depth: YAML_MAX_DEPTH,
        },
    }
}

/// Splits a YAML stream on spec-aware `---`/`...` boundaries at column 0.
/// Block-scalar content is always indented so a `---` inside a block scalar
/// never appears at column 0 — no explicit indentation tracking is needed.
fn split_documents(s: &str) -> Vec<&str> {
    let mut docs: Vec<&str> = Vec::new();
    let mut doc_start: usize = 0;
    let mut pos: usize = 0;

    while pos <= s.len() {
        let line_end = s[pos..].find('\n').map(|i| pos + i + 1).unwrap_or(s.len());
        let line = &s[pos..line_end];
        let trimmed_nl = line.strip_suffix('\n').unwrap_or(line);
        let trimmed_nl = trimmed_nl.strip_suffix('\r').unwrap_or(trimmed_nl);

        let is_doc_start = trimmed_nl == "---" || trimmed_nl.starts_with("--- ");
        let is_doc_end = trimmed_nl == "..." || trimmed_nl.starts_with("... ");

        if is_doc_start && pos != 0 {
            // Close the current document at the start of this marker line.
            let slice = &s[doc_start..pos];
            if !slice.trim().is_empty() {
                docs.push(slice);
            }
            // The next document begins AFTER this marker line.
            doc_start = line_end;
        } else if is_doc_start && pos == 0 {
            // A leading `---` just opens the first document; skip the marker.
            doc_start = line_end;
        } else if is_doc_end {
            let slice = &s[doc_start..pos];
            if !slice.trim().is_empty() {
                docs.push(slice);
            }
            doc_start = line_end;
        }

        if line_end > pos {
            pos = line_end;
        } else {
            break;
        }
        if pos >= s.len() {
            break;
        }
    }

    if doc_start < s.len() {
        let slice = &s[doc_start..];
        if !slice.trim().is_empty() {
            docs.push(slice);
        }
    }

    docs
}

/// YAML extractor — implements [`Extractor`].
pub struct YamlExtractor;

impl Extractor for YamlExtractor {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError> {
        let s = validate_utf8(raw)?;
        let s = strip_bom(s);

        let docs = split_documents(s);
        let multi_doc = docs.len() > 1;

        let docs: &[&str] = if docs.len() > MAX_YAML_DOCUMENTS {
            tracing::warn!(
                total = docs.len(),
                cap = MAX_YAML_DOCUMENTS,
                dropped = docs.len() - MAX_YAML_DOCUMENTS,
                "YAML stream exceeds MAX_YAML_DOCUMENTS; trailing documents truncated"
            );
            &docs[..MAX_YAML_DOCUMENTS]
        } else {
            &docs[..]
        };

        let mut sink = Sink::new();

        for (doc_index, doc_src) in docs.iter().enumerate() {
            let value: Value = parse_yaml_document(doc_src)?;

            let mut segments: Vec<Segment<'_>> = Vec::new();
            if multi_doc {
                segments.push(Segment::Record(doc_index));
            }
            walk_value(&value, &mut segments, 0, &mut sink, "yaml");
        }

        Ok(sink.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::SourceAnchor;

    fn extract(src: &str) -> ExtractOutput {
        YamlExtractor.extract(src.as_bytes()).expect("extraction")
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
    fn yaml_simple_mapping_extracted() {
        let out = extract("name: Alice\nage: 30\n");
        assert!(
            out.extracted_text.contains("/name: Alice"),
            "{:?}",
            out.extracted_text
        );
        assert!(
            out.extracted_text.contains("/age: 30"),
            "{:?}",
            out.extracted_text
        );
    }

    #[test]
    fn yaml_byte_identity() {
        let out = extract("a: 1\nb:\n  c: x\nd:\n  - 1\n  - 2\n");
        assert!(!out.blocks.is_empty());
        assert_byte_identity(&out);
    }

    #[test]
    fn yaml_multibyte_utf8_byte_identity() {
        let out = extract("日本語: 🦀\n");
        assert_eq!(out.blocks.len(), 1);
        assert_byte_identity(&out);
        let b = &out.blocks[0];
        assert_eq!(b.char_end - b.char_start, b.text.len());
        assert!(b.text.contains("日本語"));
        assert!(b.text.contains("🦀"));
    }

    #[test]
    fn yaml_key_order_is_alphabetical() {
        let out = extract("z: 1\na: 2\n");
        let a_idx = out
            .blocks
            .iter()
            .position(|b| b.text == "/a: 2")
            .expect("a block");
        let z_idx = out
            .blocks
            .iter()
            .position(|b| b.text == "/z: 1")
            .expect("z block");
        assert!(a_idx < z_idx, "alphabetical key order");
    }

    #[test]
    fn yaml_section_path_reflects_nesting() {
        let out = extract("a:\n  b:\n    c: 1\n");
        let b = out
            .blocks
            .iter()
            .find(|b| b.section_path == "a > b > c")
            .expect("nested section_path");
        assert_eq!(b.text, "/a/b/c: 1");
    }

    #[test]
    fn yaml_anchors_index_aligned() {
        let out = extract("a: 1\nb: 2\n");
        assert_eq!(out.anchors.len(), out.blocks.len());
        for a in &out.anchors {
            assert!(matches!(a, SourceAnchor::Structured { .. }));
        }
    }

    #[test]
    fn yaml_multi_document_two_docs() {
        let out = extract("---\na: 1\n---\nb: 2\n");
        let first = out
            .blocks
            .iter()
            .find(|b| b.text == "/0/a: 1")
            .expect("first doc block");
        let second = out
            .blocks
            .iter()
            .find(|b| b.text == "/1/b: 2")
            .expect("second doc block");
        assert!(first.section_path.starts_with("[0]"));
        assert!(second.section_path.starts_with("[1]"));
        assert_byte_identity(&out);
    }

    #[test]
    fn yaml_single_document_no_separator() {
        let out = extract("a: 1\n");
        let b = &out.blocks[0];
        assert_eq!(b.text, "/a: 1");
        assert!(!b.section_path.starts_with('['));
    }

    #[test]
    fn yaml_multi_document_separator_in_block_scalar() {
        // The `---` inside the block scalar must NOT split into two documents.
        let out = extract("value: |\n  ---\n  not a split\n");
        // Single document → no `[N]` prefix.
        assert!(out.blocks.iter().all(|b| !b.section_path.starts_with('[')));
        // The block scalar content is preserved as the value.
        let v = out.blocks.iter().find(|b| b.section_path == "value");
        assert!(v.is_some(), "value key present: {:?}", out.blocks);
        assert!(
            v.unwrap().text.contains("---"),
            "block scalar retains its '---' content: {:?}",
            v.unwrap().text
        );
    }

    #[test]
    fn yaml_multi_document_trailing_ellipsis() {
        let out = extract("a: 1\n...\n");
        assert_eq!(out.blocks.len(), 1);
        assert_eq!(out.blocks[0].text, "/a: 1");
    }

    #[test]
    fn yaml_multi_document_count_capped() {
        let extra = 50;
        let total = MAX_YAML_DOCUMENTS + extra;
        let mut s = String::new();
        for i in 0..total {
            s.push_str("---\n");
            s.push_str(&format!("k: {i}\n"));
        }
        let out = extract(&s);
        assert_eq!(
            out.blocks.len(),
            MAX_YAML_DOCUMENTS,
            "only the first {MAX_YAML_DOCUMENTS} documents are processed"
        );
        let last_idx = MAX_YAML_DOCUMENTS - 1;
        assert!(
            out.blocks
                .iter()
                .any(|b| b.text == format!("/{last_idx}/k: {last_idx}")),
            "last in-cap document present"
        );
        let dropped_idx = total - 1;
        assert!(
            !out.blocks
                .iter()
                .any(|b| b.text.contains(&format!("k: {dropped_idx}"))),
            "dropped document must be absent"
        );
        assert_byte_identity(&out);
    }

    #[test]
    fn yaml_deeply_nested_rejected_by_budget() {
        // Exercises the PRODUCTION path on the default test thread (not a hand-rolled
        // large-stack thread), proving the 8 MB parse-thread guard returns a clean Err
        // regardless of the caller's stack. See YAML_PARSE_STACK_BYTES for context.
        let depth = 200;
        let mut s = String::new();
        for i in 0..depth {
            for _ in 0..i {
                s.push_str("  ");
            }
            s.push_str("k:\n");
        }
        for _ in 0..depth {
            s.push(' ');
        }
        s.push_str("leaf\n");
        let err = YamlExtractor
            .extract(s.as_bytes())
            .expect_err("over-deep YAML must be rejected by the depth budget");
        assert!(matches!(err, LensError::Parse(_)), "got {err:?}");
    }

    #[test]
    fn yaml_root_scalar() {
        let out = extract("42\n");
        assert_eq!(out.blocks.len(), 1);
        assert_eq!(out.blocks[0].text, ": 42");
        assert_eq!(out.blocks[0].section_path, "");
    }

    #[test]
    fn yaml_empty_input() {
        let out = extract("");
        assert!(out.extracted_text.is_empty());
        assert!(out.blocks.is_empty());
    }

    #[test]
    fn yaml_bom_stripped() {
        let with_bom = format!("\u{FEFF}{}", "a: 1\n");
        let a = YamlExtractor.extract(with_bom.as_bytes()).expect("bom ok");
        let b = extract("a: 1\n");
        assert_eq!(a.extracted_text, b.extracted_text);
        assert_eq!(a.blocks, b.blocks);
    }

    #[test]
    fn yaml_invalid_yaml_returns_parse_error() {
        let err = YamlExtractor
            .extract(b"a: [1, 2\nb: }")
            .expect_err("malformed YAML errors");
        assert!(matches!(err, LensError::Parse(_)), "got {err:?}");
    }

    #[test]
    fn yaml_invalid_utf8_returns_validation_error() {
        let err = YamlExtractor
            .extract(&[0xFF, 0xFE, 0x00])
            .expect_err("invalid UTF-8 errors");
        assert!(matches!(err, LensError::Validation(_)), "got {err:?}");
    }

    #[test]
    fn yaml_snapshot_block_structure() {
        let out =
            extract("title: Doc\ntags:\n  - a\n  - b\nmeta:\n  author: Alice\n  year: 2024\n");
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
        insta::assert_json_snapshot!("yaml_block_structure", snaps);
    }
}
