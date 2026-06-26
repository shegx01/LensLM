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

/// Maximum number of YAML documents processed from a multi-document stream.
/// A pathological stream of thousands of `---`-separated documents would
/// otherwise fan out unbounded work; documents beyond this cap are dropped with
/// a `tracing::warn!` (never silently).
const MAX_YAML_DOCUMENTS: usize = 1024;

/// Maximum YAML structural nesting depth enforced by `serde_saphyr`'s pre-parse
/// budget (finding H2). serde_saphyr's own default is `max_depth: 2000`; this
/// tighter ceiling rejects pathologically deep YAML far earlier while staying
/// comfortably above any legitimate configuration document.
const YAML_MAX_DEPTH: usize = 128;

/// Stack size for the dedicated thread on which `serde_saphyr` deserializes.
///
/// WHY a scoped large stack: production calls [`YamlExtractor::extract`] under
/// `tokio::task::spawn_blocking` (see `ingest.rs`), whose blocking-pool threads
/// carry the default ~2 MB stack. serde_saphyr enforces [`YAML_MAX_DEPTH`] as a
/// pre-parse *budget*, but it does so WHILE the recursive deserializer pulls
/// events — so the deserializer recurses to ~`YAML_MAX_DEPTH` (≈3 MB of stack at
/// depth 128) BEFORE the budget breach fires and returns `Err`. On a 2 MB stack
/// that recursion overflows and crashes the worker instead of producing a clean
/// error. Running the parse on a thread with this 8 MB stack (comfortably above
/// the ~3 MB the depth-128 budget needs) guarantees the budget breach reliably
/// returns `Err` regardless of the caller's stack — closing a DoS hole.
const YAML_PARSE_STACK_BYTES: usize = 8 * 1024 * 1024;

/// Deserializes one YAML document into a [`Value`] on a dedicated thread with a
/// [`YAML_PARSE_STACK_BYTES`] stack, so the [`YAML_MAX_DEPTH`] budget breach
/// returns an `Err` instead of overflowing the caller's (possibly ~2 MB) stack.
/// A panic inside the parse thread (or a join failure) is converted into a
/// [`LensError::Parse`] rather than being allowed to unwind into a crash.
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

/// Builds the `serde_saphyr` deserialization options with our tightened nesting
/// budget. All other budget limits keep their library defaults. The `options!`
/// / `budget!` macros are the crate's supported construction path (direct
/// struct-literal construction of `Options`/`Budget` is `#[deprecated]`).
fn yaml_options() -> Options {
    serde_saphyr::options! {
        budget: serde_saphyr::budget! {
            max_depth: YAML_MAX_DEPTH,
        },
    }
}

/// Splits a YAML stream into its constituent document source slices using a
/// spec-aware boundary scan: a boundary marker must be at column 0 (no leading
/// whitespace). Per the YAML spec, block-scalar content (`|` / `>`) is ALWAYS
/// indented relative to its key, so a `---` or `...` appearing inside a block
/// scalar is never at column 0 — column-0 checking is therefore sufficient and
/// no explicit block-scalar indentation tracking is performed.
///
/// A document boundary is a line whose TRIMMED-of-trailing form is exactly `---`
/// or starts with `--- ` (a directives-end marker at column 0). A line that is
/// exactly `...` (or starts with `... `) at column 0 ends the current document.
///
/// Returns the non-empty document slices in order. An input with no explicit
/// boundary markers yields a single document (the whole input).
fn split_documents(s: &str) -> Vec<&str> {
    // Byte offsets at which each document's content begins (after the marker).
    let mut docs: Vec<&str> = Vec::new();
    let mut doc_start: usize = 0;
    let mut pos: usize = 0;

    // Walk line by line tracking byte offsets so slices are exact.
    while pos <= s.len() {
        let line_end = s[pos..].find('\n').map(|i| pos + i + 1).unwrap_or(s.len());
        let line = &s[pos..line_end];
        // The line content without the trailing newline, for marker detection.
        let trimmed_nl = line.strip_suffix('\n').unwrap_or(line);
        let trimmed_nl = trimmed_nl.strip_suffix('\r').unwrap_or(trimmed_nl);

        // A boundary marker must be at column 0 (no leading whitespace).
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
            // End the current document at the marker; the next document (if any)
            // begins after it. A `...` with no following `---` simply terminates.
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

    // Push the trailing document (content after the last boundary).
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

        // Cap the number of documents processed to bound fan-out on a
        // pathological multi-document stream; warn (never silently) on truncation.
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

// ---------------------------------------------------------------------------
// Tests (TDD: RED first)
// ---------------------------------------------------------------------------

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
        // Both documents contribute blocks, prefixed with [0] / [1].
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
        // No `[N]` prefix for a single document.
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
        // A trailing `...` terminator must not produce an extra empty document.
        let out = extract("a: 1\n...\n");
        // Single document, no `[N]` prefix.
        assert_eq!(out.blocks.len(), 1);
        assert_eq!(out.blocks[0].text, "/a: 1");
    }

    #[test]
    fn yaml_multi_document_count_capped() {
        // Build a stream of MAX_YAML_DOCUMENTS + 50 single-key documents.
        // Only the first MAX_YAML_DOCUMENTS must be processed (one block each).
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
        // The last processed document is index MAX_YAML_DOCUMENTS - 1.
        let last_idx = MAX_YAML_DOCUMENTS - 1;
        assert!(
            out.blocks
                .iter()
                .any(|b| b.text == format!("/{last_idx}/k: {last_idx}")),
            "last in-cap document present"
        );
        // A dropped document (the very last one) must NOT appear.
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
        // A YAML mapping nested far beyond YAML_MAX_DEPTH must be rejected by
        // serde_saphyr's depth budget (a Parse error), not silently accepted —
        // and crucially, WITHOUT crashing on the caller's stack.
        //
        // serde_saphyr enforces the budget WHILE the recursive deserializer
        // pulls events, so the deserializer recurses to ~depth before the breach
        // fires. At YAML_MAX_DEPTH (128) that recursion needs ~3 MB of stack —
        // more than a `spawn_blocking` worker's (or this test harness's) default
        // ~2 MB stack. The guard now runs the parse on its own
        // YAML_PARSE_STACK_BYTES (8 MB) thread, so it returns a clean `Err`
        // regardless of the caller's stack. This test therefore exercises the
        // PRODUCTION path directly — calling `YamlExtractor::extract` on the
        // default test thread, NOT a hand-rolled large-stack thread — proving the
        // guard no longer depends on the caller providing a big stack.
        let depth = 200; // well beyond YAML_MAX_DEPTH (128)
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
        // Unbalanced flow mapping is invalid YAML.
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
