//! JSON extractor (M4 Phase 2.5c).
//!
//! [`JsonExtractor`] parses a single JSON value (`serde_json::Value`) from raw
//! bytes and produces a canonical [`ExtractOutput`] via a **key-path
//! verbalization**: the `serde_json::Value` tree is walked depth-first and each
//! leaf scalar is emitted as one line `"{path}: {value}\n"` in a growing buffer.
//! The buffer IS the `extracted_text`; each block's `char_start..char_end`
//! slices it byte-identically (offsets recorded at append time via
//! `String::len()`, so multibyte UTF-8 keys/values are handled correctly).
//!
//! The depth-first tree-walk ([`walk_value`]) is shared `pub(crate)` with the
//! JSONL and YAML extractors (both verbalize `serde_json::Value` trees the same
//! way).

use serde_json::Value;

use crate::LensError;
use crate::parse::{Block, BlockType};

use super::{ExtractOutput, Extractor, SourceAnchor};

/// Maximum nesting depth before a subtree is collapsed to a single block.
pub(crate) const MAX_NESTING_DEPTH: usize = 64;
/// Maximum number of array elements emitted individually before truncation.
pub(crate) const MAX_ARRAY_ELEMENTS: usize = 10_000;
/// Maximum byte length of a collapsed over-deep subtree. UTF-8-safe truncation,
/// marked with a ` …[truncated]` suffix so no content is silently lost.
pub(crate) const MAX_COLLAPSED_BLOCK_BYTES: usize = 8 * 1024;

/// Strips a leading UTF-8 BOM so `serde_json` (which rejects BOM per RFC 8259)
/// and the other parsers see clean input.
pub(crate) fn strip_bom(s: &str) -> &str {
    s.strip_prefix('\u{FEFF}').unwrap_or(s)
}

pub(crate) fn validate_utf8(raw: &[u8]) -> Result<&str, LensError> {
    std::str::from_utf8(raw)
        .map_err(|_| LensError::Validation("source is not valid UTF-8".to_string()))
}

/// One path segment of the JSON-pointer-ish address.
#[derive(Clone)]
pub(crate) enum Segment<'a> {
    Key(&'a str),
    Index(usize),
    /// JSONL/YAML discriminant: renders `[N]` in `section_path`, `/N` in the anchor path.
    Record(usize),
}

impl Segment<'_> {
    fn section_part(&self) -> String {
        match self {
            Segment::Key(k) => (*k).to_string(),
            Segment::Index(i) => i.to_string(),
            Segment::Record(i) => format!("[{i}]"),
        }
    }

    fn path_part(&self) -> String {
        match self {
            Segment::Key(k) => (*k).to_string(),
            Segment::Index(i) => i.to_string(),
            Segment::Record(i) => i.to_string(),
        }
    }
}

pub(crate) struct Sink {
    pub buf: String,
    pub blocks: Vec<Block>,
    pub anchors: Vec<SourceAnchor>,
}

impl Sink {
    pub(crate) fn new() -> Self {
        Self {
            buf: String::new(),
            blocks: Vec::new(),
            anchors: Vec::new(),
        }
    }

    pub(crate) fn finish(self) -> ExtractOutput {
        ExtractOutput {
            extracted_text: self.buf,
            blocks: self.blocks,
            anchors: self.anchors,
            table_markdown: None,
        }
    }

    fn emit_leaf(&mut self, section_path: &str, anchor_path: &str, value: &str) {
        let line = format!("{anchor_path}: {value}");
        let char_start = self.buf.len();
        self.buf.push_str(&line);
        let char_end = self.buf.len();
        self.buf.push('\n');
        self.blocks.push(Block {
            block_type: BlockType::Paragraph.as_str().to_string(),
            section_path: section_path.to_string(),
            text: line,
            char_start,
            char_end,
        });
        self.anchors.push(SourceAnchor::Structured {
            path: anchor_path.to_string(),
        });
    }
}

fn scalar_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

fn truncate_collapsed(s: &str) -> String {
    if s.len() <= MAX_COLLAPSED_BLOCK_BYTES {
        return s.to_string();
    }
    let mut end = MAX_COLLAPSED_BLOCK_BYTES;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    tracing::warn!(
        original_bytes = s.len(),
        cap = MAX_COLLAPSED_BLOCK_BYTES,
        "collapsed subtree serialization truncated to MAX_COLLAPSED_BLOCK_BYTES"
    );
    format!("{} …[truncated]", &s[..end])
}

fn section_path_of(segments: &[Segment<'_>]) -> String {
    segments
        .iter()
        .map(Segment::section_part)
        .collect::<Vec<_>>()
        .join(" > ")
}

fn anchor_path_of(segments: &[Segment<'_>]) -> String {
    if segments.is_empty() {
        return String::new();
    }
    let mut p = String::new();
    for s in segments {
        p.push('/');
        p.push_str(&s.path_part());
    }
    p
}

/// Walks a `serde_json::Value` tree depth-first, appending one leaf block per scalar.
/// Deeper than [`MAX_NESTING_DEPTH`]: collapse the subtree into one block. Shared
/// with JSONL/YAML extractors, which seed `segments` with a record/document prefix.
pub(crate) fn walk_value<'a>(
    value: &'a Value,
    segments: &mut Vec<Segment<'a>>,
    depth: usize,
    sink: &mut Sink,
    format: &str,
) {
    if depth > MAX_NESTING_DEPTH {
        let section_path = section_path_of(segments);
        let anchor_path = anchor_path_of(segments);
        tracing::warn!(
            format,
            depth,
            path = %anchor_path,
            "{format} nesting exceeds MAX_NESTING_DEPTH ({MAX_NESTING_DEPTH}); subtree collapsed"
        );
        let serialized = truncate_collapsed(&value.to_string());
        sink.emit_leaf(&section_path, &anchor_path, &serialized);
        return;
    }

    match value {
        Value::Object(map) => {
            // Explicit sort is a defensive determinism guard: if `preserve_order`
            // is ever enabled (switching BTreeMap → IndexMap), the canonical buffer
            // stays deterministic regardless of source key order.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for k in keys {
                segments.push(Segment::Key(k.as_str()));
                walk_value(&map[k], segments, depth + 1, sink, format);
                segments.pop();
            }
        }
        Value::Array(items) => {
            let n = items.len();
            let capped = n.min(MAX_ARRAY_ELEMENTS);
            for (i, item) in items.iter().take(capped).enumerate() {
                segments.push(Segment::Index(i));
                walk_value(item, segments, depth + 1, sink, format);
                segments.pop();
            }
            if n > MAX_ARRAY_ELEMENTS {
                let remaining = n - MAX_ARRAY_ELEMENTS;
                let section_path = section_path_of(segments);
                let anchor_path = anchor_path_of(segments);
                tracing::warn!(
                    format,
                    total = n,
                    cap = MAX_ARRAY_ELEMENTS,
                    remaining,
                    path = %anchor_path,
                    "{format} array exceeds MAX_ARRAY_ELEMENTS; {remaining} elements truncated"
                );
                sink.emit_leaf(
                    &section_path,
                    &anchor_path,
                    &format!("[... {remaining} more elements truncated]"),
                );
            }
        }
        scalar => {
            let section_path = section_path_of(segments);
            let anchor_path = anchor_path_of(segments);
            sink.emit_leaf(&section_path, &anchor_path, &scalar_to_string(scalar));
        }
    }
}

/// JSON extractor — implements [`Extractor`].
pub struct JsonExtractor;

impl Extractor for JsonExtractor {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError> {
        let s = validate_utf8(raw)?;
        let s = strip_bom(s);
        let value: Value =
            serde_json::from_str(s).map_err(|e| LensError::Parse(format!("invalid JSON: {e}")))?;

        let mut sink = Sink::new();
        let mut segments: Vec<Segment<'_>> = Vec::new();
        walk_value(&value, &mut segments, 0, &mut sink, "json");
        Ok(sink.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(src: &str) -> ExtractOutput {
        JsonExtractor.extract(src.as_bytes()).expect("extraction")
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
    fn json_simple_object_extracted_text() {
        let out = extract(r#"{"name":"Alice","age":30}"#);
        assert!(
            out.extracted_text.contains("/name: Alice"),
            "extracted_text = {:?}",
            out.extracted_text
        );
        assert!(
            out.extracted_text.contains("/age: 30"),
            "extracted_text = {:?}",
            out.extracted_text
        );
    }

    #[test]
    fn json_byte_identity() {
        let out = extract(r#"{"a":1,"b":{"c":"x"},"d":[1,2]}"#);
        assert!(!out.blocks.is_empty());
        assert_byte_identity(&out);
    }

    #[test]
    fn json_multibyte_utf8_byte_identity() {
        let out = extract(r#"{"日本語":"🦀"}"#);
        assert_eq!(out.blocks.len(), 1);
        assert_byte_identity(&out);
        let b = &out.blocks[0];
        assert_eq!(b.char_end - b.char_start, b.text.len());
        assert!(b.text.contains("日本語"));
        assert!(b.text.contains("🦀"));
    }

    #[test]
    fn json_section_path_reflects_nesting() {
        let out = extract(r#"{"a":{"b":{"c":1}}}"#);
        let b = out
            .blocks
            .iter()
            .find(|b| b.section_path == "a > b > c")
            .expect("nested section_path");
        assert_eq!(b.text, "/a/b/c: 1");
    }

    #[test]
    fn json_key_order_is_alphabetical() {
        let out = extract(r#"{"z":1,"a":2}"#);
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
    fn json_anchors_index_aligned() {
        let out = extract(r#"{"a":1,"b":2}"#);
        assert_eq!(out.anchors.len(), out.blocks.len());
        for a in &out.anchors {
            assert!(matches!(a, SourceAnchor::Structured { .. }));
        }
    }

    #[test]
    fn json_anchor_path_matches_key() {
        let out = extract(r#"{"a":{"b":1}}"#);
        let SourceAnchor::Structured { path } = &out.anchors[0] else {
            panic!("expected Structured anchor");
        };
        assert_eq!(path, "/a/b");
    }

    #[test]
    fn json_array_elements_indexed() {
        let out = extract(r#"{"items":[1,2,3]}"#);
        let paths: Vec<&str> = out
            .anchors
            .iter()
            .map(|a| match a {
                SourceAnchor::Structured { path } => path.as_str(),
                _ => panic!("non-structured anchor"),
            })
            .collect();
        assert!(paths.contains(&"/items/0"));
        assert!(paths.contains(&"/items/1"));
        assert!(paths.contains(&"/items/2"));
    }

    #[test]
    fn json_deeply_nested_capped() {
        let mut s = String::new();
        for _ in 0..100 {
            s.push_str(r#"{"k":"#);
        }
        s.push('1');
        for _ in 0..100 {
            s.push('}');
        }
        let out = JsonExtractor
            .extract(s.as_bytes())
            .expect("no panic on deep nesting");
        assert_eq!(out.blocks.len(), 1);
        assert_byte_identity(&out);
    }

    #[test]
    fn json_over_deep_collapsed_block_truncated() {
        let depth = MAX_NESTING_DEPTH + 5;
        let big_value = "x".repeat(64 * 1024);
        let mut s = String::new();
        for _ in 0..depth {
            s.push_str(r#"{"k":"#);
        }
        s.push('"');
        s.push_str(&big_value);
        s.push('"');
        for _ in 0..depth {
            s.push('}');
        }
        let out = JsonExtractor
            .extract(s.as_bytes())
            .expect("deep nesting collapses without panic");
        assert_eq!(out.blocks.len(), 1, "exactly one collapsed block");
        let block = &out.blocks[0];
        assert!(
            block.text.len() <= MAX_COLLAPSED_BLOCK_BYTES + 256,
            "collapsed block must be capped near {MAX_COLLAPSED_BLOCK_BYTES}; got {}",
            block.text.len()
        );
        assert!(
            block.text.ends_with("…[truncated]"),
            "truncation marker present; got tail {:?}",
            &block.text[block.text.len().saturating_sub(32)..]
        );
        assert_byte_identity(&out);
    }

    #[test]
    fn json_large_array_capped() {
        let mut s = String::from("[");
        for i in 0..10_001 {
            if i > 0 {
                s.push(',');
            }
            s.push('1');
        }
        s.push(']');
        let out = JsonExtractor
            .extract(s.as_bytes())
            .expect("extract large array");
        assert_eq!(out.blocks.len(), MAX_ARRAY_ELEMENTS + 1);
        assert!(
            out.blocks
                .last()
                .unwrap()
                .text
                .contains("more elements truncated")
        );
        assert_byte_identity(&out);
    }

    #[test]
    fn json_empty_object() {
        let out = extract("{}");
        assert!(out.extracted_text.is_empty());
        assert!(out.blocks.is_empty());
    }

    #[test]
    fn json_empty_array() {
        let out = extract("[]");
        assert!(out.extracted_text.is_empty());
        assert!(out.blocks.is_empty());
    }

    #[test]
    fn json_root_null() {
        let out = extract("null");
        assert_eq!(out.blocks.len(), 1);
        assert_eq!(out.blocks[0].text, ": null");
        assert_eq!(out.blocks[0].section_path, "");
        assert_byte_identity(&out);
    }

    #[test]
    fn json_root_scalar() {
        let out = extract("42");
        assert_eq!(out.blocks.len(), 1);
        assert_eq!(out.blocks[0].text, ": 42");
        assert_eq!(out.blocks[0].section_path, "");
        assert_eq!(out.extracted_text, ": 42\n");
        let SourceAnchor::Structured { path } = &out.anchors[0] else {
            panic!("structured");
        };
        assert_eq!(path, "");
    }

    #[test]
    fn json_bom_stripped() {
        let with_bom = format!("\u{FEFF}{}", r#"{"a":1}"#);
        let a = JsonExtractor.extract(with_bom.as_bytes()).expect("bom ok");
        let b = extract(r#"{"a":1}"#);
        assert_eq!(a.extracted_text, b.extracted_text);
        assert_eq!(a.blocks, b.blocks);
    }

    #[test]
    fn json_invalid_syntax_returns_parse_error() {
        let err = JsonExtractor
            .extract(b"{bad}")
            .expect_err("malformed JSON errors");
        assert!(matches!(err, LensError::Parse(_)), "got {err:?}");
    }

    #[test]
    fn json_invalid_utf8_returns_validation_error() {
        let err = JsonExtractor
            .extract(&[0xFF, 0xFE, 0x00])
            .expect_err("invalid UTF-8 errors");
        assert!(matches!(err, LensError::Validation(_)), "got {err:?}");
    }

    #[test]
    fn json_snapshot_block_structure() {
        let out =
            extract(r#"{"title":"Doc","tags":["a","b"],"meta":{"author":"Alice","year":2024}}"#);
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
        insta::assert_json_snapshot!("json_block_structure", snaps);
    }
}
