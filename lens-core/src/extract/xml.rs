//! XML extractor (M4 Phase 2.5c).
//!
//! [`XmlExtractor`] parses XML with `roxmltree` (`=0.21.1`) and walks the DOM
//! depth-first, emitting one [`Block`] per element that carries direct text
//! content (CDATA included). The element path becomes the JSON-pointer-ish
//! anchor path (`/root/child`) and the ` > `-joined `section_path`
//! (`root > child`). Element attributes are prepended to the block text as
//! `@attr=value ` before the text content.
//!
//! The canonical buffer is built incrementally (`String::len()` byte offsets),
//! so byte-identity holds for multibyte UTF-8 text. A leading UTF-8 BOM is
//! stripped before parsing; a non-UTF-8 `encoding="..."` declaration is rejected
//! with a clear `Validation` error (the bytes ARE valid UTF-8, but the declared
//! encoding is unsupported).

use roxmltree::{Document, Node, NodeType, ParsingOptions};

use crate::LensError;
use crate::parse::{Block, BlockType};

use super::json::{MAX_NESTING_DEPTH, strip_bom, validate_utf8};
use super::{ExtractOutput, Extractor, SourceAnchor};

/// Safe ceiling on XML element nesting depth, enforced by a cheap single-pass
/// scan BEFORE `roxmltree` parses (and recursively builds) the DOM. roxmltree
/// 0.21.1 has no element-nesting limit of its own (its `LoopDetector` only
/// bounds entity expansion), so an adversarial deeply-nested document — well
/// under the configurable raw-bytes cap (`max_source_bytes`) — would overflow the stack during
/// the parser's own recursive descent. This ceiling is comfortably above any
/// real document yet far below the stack-overflow threshold; the pre-scan is
/// intentionally conservative (it may over-count slightly) since its only job
/// is to prevent unbounded recursion.
const XML_MAX_PARSE_DEPTH: usize = 256;

/// Upper bound on the number of parsed nodes, passed to `roxmltree` to cap
/// memory amplification on adversarial input (finding M2). One million nodes is
/// far above any legitimate document while still bounding worst-case allocation.
const XML_MAX_NODES: u32 = 1_000_000;

/// Detects an XML prolog `encoding="..."` declaration that is NOT UTF-8
/// (case-insensitive). Returns `true` when an unsupported encoding is declared.
///
/// Scans only the prolog (the first `<?xml ... ?>` processing instruction); a
/// missing declaration or an explicit UTF-8 declaration is fine.
fn declares_non_utf8_encoding(s: &str) -> bool {
    let prolog = match s.find("?>") {
        Some(end) if s.trim_start().starts_with("<?xml") => &s[..end],
        _ => return false,
    };
    let Some(enc_pos) = prolog.find("encoding") else {
        return false;
    };
    // Find the quoted value after `encoding`.
    let after = &prolog[enc_pos + "encoding".len()..];
    let Some(q_rel) = after.find(['"', '\'']) else {
        return false;
    };
    let quote = after.as_bytes()[q_rel] as char;
    let val_start = q_rel + 1;
    let Some(q_end_rel) = after[val_start..].find(quote) else {
        return false;
    };
    let enc = &after[val_start..val_start + q_end_rel];
    !enc.eq_ignore_ascii_case("utf-8")
}

/// Cheap single-pass scan of (BOM-stripped, UTF-8-validated) XML that tracks
/// element nesting depth and returns the maximum depth observed.
///
/// It increments on a start tag (`<name ...>`), decrements on an end tag
/// (`</name>`), and ignores self-closing tags (`<name ... />`), comments
/// (`<!-- ... -->`), processing instructions / declarations (`<? ... ?>`,
/// `<! ... >`), CDATA sections (`<![CDATA[ ... ]]>`), and any `<`/`>` that
/// appear inside a quoted attribute value. It does NOT validate the XML — it is
/// a conservative resource guard run BEFORE `roxmltree`'s own recursive parse,
/// so over-counting slightly is acceptable (the ceiling is generous).
fn max_element_nesting_depth(s: &str) -> usize {
    let bytes = s.as_bytes();
    let mut i = 0;
    let len = bytes.len();
    let mut depth: usize = 0;
    let mut max_depth: usize = 0;

    while i < len {
        if bytes[i] != b'<' {
            i += 1;
            continue;
        }
        // We are at a `<`. Classify what follows.
        let next = bytes.get(i + 1).copied();
        match next {
            // Comment, CDATA, or declaration: `<!...`. None of these change
            // element depth. Skip past the appropriate terminator.
            Some(b'!') => {
                if bytes[i..].starts_with(b"<!--") {
                    // Comment: skip to `-->`.
                    if let Some(end) = find_subslice(&bytes[i + 4..], b"-->") {
                        i = i + 4 + end + 3;
                    } else {
                        break;
                    }
                } else if bytes[i..].starts_with(b"<![CDATA[") {
                    // CDATA: skip to `]]>`.
                    if let Some(end) = find_subslice(&bytes[i + 9..], b"]]>") {
                        i = i + 9 + end + 3;
                    } else {
                        break;
                    }
                } else {
                    // Other declaration (e.g. `<!DOCTYPE ...>`): skip to `>`.
                    if let Some(end) = find_byte(&bytes[i + 1..], b'>') {
                        i = i + 1 + end + 1;
                    } else {
                        break;
                    }
                }
            }
            // Processing instruction / prolog: `<? ... ?>`. No depth change.
            Some(b'?') => {
                if let Some(end) = find_subslice(&bytes[i + 2..], b"?>") {
                    i = i + 2 + end + 2;
                } else {
                    break;
                }
            }
            // End tag: `</name>`. Decrement depth.
            Some(b'/') => {
                if let Some(end) = scan_tag_end(bytes, i + 2) {
                    depth = depth.saturating_sub(1);
                    i = end + 1;
                } else {
                    break;
                }
            }
            // Start tag (or self-closing): `<name ...>` / `<name ... />`.
            Some(_) => {
                if let Some((end, self_closing)) = scan_start_tag(bytes, i + 1) {
                    if !self_closing {
                        depth += 1;
                        if depth > max_depth {
                            max_depth = depth;
                        }
                    }
                    i = end + 1;
                } else {
                    break;
                }
            }
            // Trailing `<` at EOF: nothing to do.
            None => break,
        }
    }

    max_depth
}

/// Scans from `start` to the closing `>` of an end tag, returning the index of
/// the `>` (quotes shouldn't appear in an end tag, but we skip them defensively).
fn scan_tag_end(bytes: &[u8], start: usize) -> Option<usize> {
    find_byte(&bytes[start..], b'>').map(|off| start + off)
}

/// Scans a start tag beginning at `start` (the byte after `<`), honoring quoted
/// attribute values so a `>` inside an attribute does not end the tag. Returns
/// `(index_of_closing_gt, is_self_closing)`.
fn scan_start_tag(bytes: &[u8], start: usize) -> Option<(usize, bool)> {
    let len = bytes.len();
    let mut i = start;
    let mut quote: Option<u8> = None;
    while i < len {
        let b = bytes[i];
        match quote {
            Some(q) => {
                if b == q {
                    quote = None;
                }
            }
            None => match b {
                b'"' | b'\'' => quote = Some(b),
                b'>' => {
                    let self_closing = i > start && bytes[i - 1] == b'/';
                    return Some((i, self_closing));
                }
                _ => {}
            },
        }
        i += 1;
    }
    None
}

/// Finds the first occurrence of `byte` in `haystack`, returning its index.
fn find_byte(haystack: &[u8], byte: u8) -> Option<usize> {
    haystack.iter().position(|&b| b == byte)
}

/// Finds the first occurrence of `needle` in `haystack`, returning its start index.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Recursively walks `node`, appending one block per text-bearing element.
///
/// `segments` is the current element-name path. `depth` enforces
/// [`MAX_NESTING_DEPTH`]. Each element's DIRECT text children (and CDATA) are
/// concatenated; if non-empty, a block is emitted with the element's attributes
/// prepended as `@name=value `.
fn walk_element(
    node: Node<'_, '_>,
    segments: &mut Vec<String>,
    depth: usize,
    buf: &mut String,
    blocks: &mut Vec<Block>,
    anchors: &mut Vec<SourceAnchor>,
) {
    let tag = node.tag_name().name();
    // Preserve a namespace prefix if the source used one (roxmltree exposes the
    // resolved prefix separately from the local name).
    let qualified = match node.lookup_prefix(node.tag_name().namespace().unwrap_or("")) {
        Some(prefix) if !prefix.is_empty() => format!("{prefix}:{tag}"),
        _ => tag.to_string(),
    };
    segments.push(qualified);

    if depth > MAX_NESTING_DEPTH {
        tracing::warn!(
            depth,
            path = %anchor_path_of(segments),
            "xml nesting exceeds MAX_NESTING_DEPTH ({MAX_NESTING_DEPTH}); subtree skipped"
        );
        segments.pop();
        return;
    }

    // Collect this element's DIRECT text + CDATA (not descendant element text).
    let mut text = String::new();
    for child in node.children() {
        if matches!(child.node_type(), NodeType::Text)
            && let Some(t) = child.text()
        {
            text.push_str(t);
        }
    }
    let text = text.trim().to_string();

    // Prepend attributes (if any) so they are searchable in the canonical buffer.
    let mut rendered = String::new();
    for attr in node.attributes() {
        rendered.push('@');
        rendered.push_str(attr.name());
        rendered.push('=');
        rendered.push_str(attr.value());
        rendered.push(' ');
    }
    rendered.push_str(&text);
    let rendered = rendered.trim_end().to_string();

    if !rendered.is_empty() {
        let section_path = segments.join(" > ");
        let anchor_path = anchor_path_of(segments);
        let char_start = buf.len();
        buf.push_str(&rendered);
        let char_end = buf.len();
        buf.push('\n');
        blocks.push(Block {
            block_type: BlockType::Paragraph.as_str().to_string(),
            section_path,
            text: rendered,
            char_start,
            char_end,
        });
        anchors.push(SourceAnchor::Structured { path: anchor_path });
    }

    // Recurse into child elements.
    for child in node.children() {
        if child.is_element() {
            walk_element(child, segments, depth + 1, buf, blocks, anchors);
        }
    }

    segments.pop();
}

/// Joins element-name segments into a `/`-separated anchor path.
fn anchor_path_of(segments: &[String]) -> String {
    let mut p = String::new();
    for s in segments {
        p.push('/');
        p.push_str(s);
    }
    p
}

/// XML extractor — implements [`Extractor`].
pub struct XmlExtractor;

impl Extractor for XmlExtractor {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError> {
        let s = validate_utf8(raw)?;
        let s = strip_bom(s);

        if declares_non_utf8_encoding(s) {
            return Err(LensError::Validation(
                "non-UTF-8 encoding not supported".to_string(),
            ));
        }

        // Resource guard: bound element nesting BEFORE handing the input to
        // `roxmltree`, whose own parse is a recursive descent that would
        // overflow the stack on adversarial deep nesting (it has no element
        // depth limit of its own).
        if max_element_nesting_depth(s) > XML_MAX_PARSE_DEPTH {
            return Err(LensError::Validation(
                "XML nesting depth exceeds supported limit".to_string(),
            ));
        }

        // `allow_dtd` stays at its `false` default to preserve XXE /
        // billion-laughs protection; `nodes_limit` bounds memory amplification.
        let opts = ParsingOptions {
            nodes_limit: XML_MAX_NODES,
            ..ParsingOptions::default()
        };
        let doc = Document::parse_with_options(s, opts)
            .map_err(|e| LensError::Parse(format!("invalid XML: {e}")))?;

        let mut buf = String::new();
        let mut blocks: Vec<Block> = Vec::new();
        let mut anchors: Vec<SourceAnchor> = Vec::new();
        let mut segments: Vec<String> = Vec::new();

        walk_element(
            doc.root_element(),
            &mut segments,
            0,
            &mut buf,
            &mut blocks,
            &mut anchors,
        );

        Ok(ExtractOutput {
            extracted_text: buf,
            blocks,
            anchors,
            table_markdown: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests (TDD: RED first)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(src: &str) -> ExtractOutput {
        XmlExtractor.extract(src.as_bytes()).expect("extraction")
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
    fn xml_simple_elements_extracted() {
        let out = extract("<root><name>Alice</name><age>30</age></root>");
        assert!(out.blocks.iter().any(|b| b.text == "Alice"));
        assert!(out.blocks.iter().any(|b| b.text == "30"));
    }

    #[test]
    fn xml_byte_identity() {
        let out = extract("<root><a>x</a><b><c>y</c></b></root>");
        assert!(!out.blocks.is_empty());
        assert_byte_identity(&out);
    }

    #[test]
    fn xml_multibyte_utf8_byte_identity() {
        let out = extract("<root><lang>日本語</lang><emoji>🦀</emoji></root>");
        assert_byte_identity(&out);
        let b = out.blocks.iter().find(|b| b.text == "日本語").unwrap();
        assert_eq!(b.char_end - b.char_start, b.text.len());
        let e = out.blocks.iter().find(|b| b.text == "🦀").unwrap();
        assert_eq!(e.char_end - e.char_start, e.text.len());
    }

    #[test]
    fn xml_section_path_reflects_element_nesting() {
        let out = extract("<a><b><c>text</c></b></a>");
        let b = out.blocks.iter().find(|b| b.text == "text").unwrap();
        assert_eq!(b.section_path, "a > b > c");
    }

    #[test]
    fn xml_anchors_index_aligned() {
        let out = extract("<root><x>1</x><y>2</y></root>");
        assert_eq!(out.anchors.len(), out.blocks.len());
        for a in &out.anchors {
            assert!(matches!(a, SourceAnchor::Structured { .. }));
        }
    }

    #[test]
    fn xml_anchor_path_is_element_path() {
        let out = extract("<root><name>Alice</name></root>");
        let b_idx = out.blocks.iter().position(|b| b.text == "Alice").unwrap();
        let SourceAnchor::Structured { path } = &out.anchors[b_idx] else {
            panic!("structured");
        };
        assert_eq!(path, "/root/name");
    }

    #[test]
    fn xml_attributes_included() {
        let out = extract(r#"<root><item id="1">text</item></root>"#);
        let b = out.blocks.iter().find(|b| b.text.contains("text")).unwrap();
        assert!(
            b.text.contains("@id=1"),
            "attribute must be in block text; got {:?}",
            b.text
        );
    }

    #[test]
    fn xml_mixed_content() {
        let out = extract("<root>lead text<child>child text</child></root>");
        // The root's own direct text is emitted as a block, and the child's too.
        assert!(out.blocks.iter().any(|b| b.text.contains("lead text")));
        assert!(out.blocks.iter().any(|b| b.text == "child text"));
        assert_byte_identity(&out);
    }

    #[test]
    fn xml_empty_elements_skipped() {
        let out = extract("<root><empty/><name>Alice</name></root>");
        // `<empty/>` produces no block.
        assert!(!out.blocks.iter().any(|b| b.section_path.ends_with("empty")));
        assert!(out.blocks.iter().any(|b| b.text == "Alice"));
    }

    #[test]
    fn xml_cdata_handled() {
        let out = extract("<root><data><![CDATA[raw <content> here]]></data></root>");
        let b = out
            .blocks
            .iter()
            .find(|b| b.section_path == "root > data")
            .expect("data block");
        assert!(
            b.text.contains("raw <content> here"),
            "CDATA text extracted; got {:?}",
            b.text
        );
    }

    #[test]
    fn xml_namespace_prefixes_preserved() {
        let out =
            extract(r#"<root xmlns:ns="http://example.com/ns"><ns:item>val</ns:item></root>"#);
        let b = out.blocks.iter().find(|b| b.text == "val").unwrap();
        assert!(
            b.section_path.contains("ns:item"),
            "namespace prefix preserved in path; got {:?}",
            b.section_path
        );
    }

    #[test]
    fn xml_bom_stripped() {
        let with_bom = format!("\u{FEFF}{}", "<root><a>x</a></root>");
        let a = XmlExtractor.extract(with_bom.as_bytes()).expect("bom ok");
        let b = extract("<root><a>x</a></root>");
        assert_eq!(a.extracted_text, b.extracted_text);
        assert_eq!(a.blocks, b.blocks);
    }

    #[test]
    fn xml_invalid_syntax_returns_parse_error() {
        let err = XmlExtractor
            .extract(b"<root><unclosed></root>")
            .expect_err("malformed XML errors");
        assert!(matches!(err, LensError::Parse(_)), "got {err:?}");
    }

    #[test]
    fn xml_invalid_utf8_returns_validation_error() {
        let err = XmlExtractor
            .extract(&[0xFF, 0xFE, 0x00])
            .expect_err("invalid UTF-8 errors");
        assert!(
            matches!(&err, LensError::Validation(m) if m == "source is not valid UTF-8"),
            "got {err:?}"
        );
    }

    #[test]
    fn xml_non_utf8_encoding_declaration_returns_validation_error() {
        let err = XmlExtractor
            .extract(br#"<?xml version="1.0" encoding="ISO-8859-1"?><root/>"#)
            .expect_err("non-UTF-8 encoding declaration errors");
        assert!(
            matches!(&err, LensError::Validation(m) if m == "non-UTF-8 encoding not supported"),
            "got {err:?}"
        );
    }

    #[test]
    fn xml_utf8_encoding_declaration_ok() {
        let out = extract(r#"<?xml version="1.0" encoding="UTF-8"?><root><a>x</a></root>"#);
        assert!(out.blocks.iter().any(|b| b.text == "x"));
    }

    #[test]
    fn xml_deeply_nested_capped() {
        let mut s = String::new();
        for _ in 0..100 {
            s.push_str("<n>");
        }
        s.push_str("deep");
        for _ in 0..100 {
            s.push_str("</n>");
        }
        let out = XmlExtractor
            .extract(s.as_bytes())
            .expect("no panic on deep nesting");
        // The deepest text is past the cap, so it is skipped — but no panic and
        // byte-identity still holds for whatever WAS emitted.
        assert_byte_identity(&out);
    }

    #[test]
    fn xml_extreme_nesting_rejected_without_crash() {
        // ~50,000 nested `<a>` elements: well under MAX_SOURCE_BYTES, but far
        // beyond what roxmltree's recursive parse could handle without
        // overflowing the stack. The pre-scan guard must reject it as an Err
        // (NOT panic / crash) WITHOUT invoking roxmltree.
        let n = 50_000;
        let mut s = String::with_capacity(n * 7);
        for _ in 0..n {
            s.push_str("<a>");
        }
        s.push_str("deep");
        for _ in 0..n {
            s.push_str("</a>");
        }
        let err = XmlExtractor
            .extract(s.as_bytes())
            .expect_err("extreme nesting must be rejected, not crash");
        assert!(
            matches!(&err, LensError::Validation(m) if m == "XML nesting depth exceeds supported limit"),
            "got {err:?}"
        );
    }

    #[test]
    fn xml_normal_depth_still_ok() {
        // A normal-depth document (50 levels) is below both the walk-time
        // MAX_NESTING_DEPTH (64) and the new pre-scan ceiling
        // XML_MAX_PARSE_DEPTH (256), so it must parse AND emit the deepest text.
        let n = 50;
        let mut s = String::new();
        for _ in 0..n {
            s.push_str("<a>");
        }
        s.push_str("deep");
        for _ in 0..n {
            s.push_str("</a>");
        }
        let out = XmlExtractor
            .extract(s.as_bytes())
            .expect("normal-depth XML must parse");
        assert!(
            out.blocks.iter().any(|b| b.text == "deep"),
            "deepest text emitted below the walk cap"
        );
    }

    #[test]
    fn xml_self_closing_and_comments_not_counted_as_depth() {
        // Self-closing tags, comments, PIs, and CDATA must not inflate the
        // depth count. A flat document with many of these stays at depth 1.
        let mut s = String::from("<root>");
        for _ in 0..1000 {
            s.push_str("<empty/><!-- c --><![CDATA[x]]>");
        }
        s.push_str("</root>");
        // Depth is 1 (root), so this must parse fine despite 1000 self-closers.
        assert_eq!(max_element_nesting_depth(&s), 1);
        let out = XmlExtractor.extract(s.as_bytes()).expect("flat XML parses");
        assert!(out.blocks.iter().any(|b| b.text.contains('x')));
    }

    #[test]
    fn xml_gt_inside_attribute_not_a_tag_boundary() {
        // A `>` inside a quoted attribute value must not be treated as the end
        // of the start tag by the depth scanner.
        let src = r#"<root><item note="a > b">text</item></root>"#;
        assert_eq!(max_element_nesting_depth(src), 2);
    }

    #[test]
    fn xml_snapshot_block_structure() {
        let out = extract(
            r#"<doc title="Doc"><section><para>First.</para><para>Second.</para></section></doc>"#,
        );
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
        insta::assert_json_snapshot!("xml_block_structure", snaps);
    }
}
