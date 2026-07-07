//! ODT extractor (M4 issue #77).
//!
//! [`OdtExtractor`] reads the `content.xml` entry from an ODF (OpenDocument Text)
//! ZIP container and walks it with `quick-xml`, emitting one [`Block`] per
//! `<text:h>` (heading) and `<text:p>` (paragraph) element. It produces a
//! canonical [`ExtractOutput`] where:
//! - `extracted_text` is the linearised UTF-8 text of all non-empty headings and
//!   paragraphs (one `\n` separator between consecutive blocks).
//! - `blocks` carries `block_type = "heading"` for `<text:h>` (with the
//!   `text:outline-level` attribute mapped to levels 1–6, clamped) and
//!   `block_type = "paragraph"` for `<text:p>`. The shared [`SectionPathStack`]
//!   tracks the heading trail.
//! - `anchors` carries one [`SourceAnchor::Odt { node_path }`] per block, where
//!   `node_path` is `"body/text:h[N]"` or `"body/text:p[N]"` (separate 0-based
//!   counters per element type, mirroring the DOCX `body/p[N]` pattern).
//!
//! Heading detection uses the normative ODF representation (`<text:h>` with
//! `text:outline-level`), per ODF 1.3 §5.1.4 — NOT style-name heuristics.

use std::io::{Cursor, Read};

use quick_xml::events::BytesStart;

use crate::LensError;
use crate::parse::{Block, SectionPathStack};

use super::xml_blocks::{BlockKind, walk_xml_blocks};
use super::{ExtractOutput, Extractor, SourceAnchor};

/// Hard ceiling on the DECOMPRESSED size of `content.xml` (decompression-bomb
/// guard). A `.odt` is a ZIP under the 50 MB stage-1 raw-bytes cap, but a
/// high-ratio entry could inflate to GBs and OOM the backend BEFORE the stage-2
/// extracted-text guard runs (see ingest.rs stage-1/stage-2). We bound the
/// `read_to_string` so decompression stops well before that. 256 MB is far above
/// any legitimate document body yet a hard cap on attacker-controlled inflation.
const MAX_DECOMPRESSED_BYTES: u64 = 256 * 1024 * 1024;

/// ODT extractor — implements [`Extractor`].
pub struct OdtExtractor;

impl Extractor for OdtExtractor {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError> {
        super::guard_zip_entry_count(raw)?;
        let mut archive = zip::ZipArchive::new(Cursor::new(raw))
            .map_err(|e| LensError::Parse(format!("ODT is not a valid ZIP container: {e}")))?;

        let entry = archive
            .by_name("content.xml")
            .map_err(|e| LensError::Parse(format!("ODT missing content.xml: {e}")))?;
        let content = read_capped(entry, MAX_DECOMPRESSED_BYTES)?;

        let mut extracted_text = String::new();
        let mut blocks: Vec<Block> = Vec::new();
        let mut anchors: Vec<SourceAnchor> = Vec::new();
        let mut section_path = SectionPathStack::new();

        // Counters advance even for empty elements to keep node_paths stable.
        let mut h_count: usize = 0;
        let mut p_count: usize = 0;

        walk_xml_blocks(
            &content,
            "ODT content.xml",
            |e: &BytesStart<'_>| match e.name().as_ref() {
                b"text:h" => Some(BlockKind::Heading(outline_level(e).unwrap_or(1))),
                b"text:p" => Some(BlockKind::Paragraph),
                _ => None,
            },
            // ODF inline whitespace elements (LibreOffice/Google Docs).
            |e: &BytesStart<'_>| match e.name().as_ref() {
                b"text:s" => Some(text_s_spaces(e)),
                b"text:tab" => Some("\t"),
                b"text:line-break" => Some("\n"),
                _ => None,
            },
            |is_heading: bool| {
                let node_path = if is_heading {
                    let path = format!("body/text:h[{h_count}]");
                    h_count += 1;
                    path
                } else {
                    let path = format!("body/text:p[{p_count}]");
                    p_count += 1;
                    path
                };
                SourceAnchor::Odt { node_path }
            },
            &mut section_path,
            &mut extracted_text,
            &mut blocks,
            &mut anchors,
        )?;

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

/// Reads `reader` into a `String`, rejecting entries that decompress beyond
/// `max` bytes (decompression-bomb guard).
fn read_capped<R: Read>(reader: R, max: u64) -> Result<String, LensError> {
    let mut content = String::new();
    let read = reader
        .take(max + 1)
        .read_to_string(&mut content)
        .map_err(|e| LensError::Parse(format!("ODT content.xml read failed: {e}")))?;
    if read as u64 > max {
        return Err(LensError::Validation(format!(
            "ODT content.xml decompresses to more than the {max}-byte limit \
             (possible decompression bomb)"
        )));
    }
    Ok(content)
}

/// Maps `<text:s text:c="n"/>` to `n` spaces per ODF 1.3 §6.1.3. Missing
/// `text:c` defaults to 1; `text:c="0"` yields zero; unparseable falls back
/// to 1. Clamped to 32 to avoid allocation on pathological input.
fn text_s_spaces(e: &BytesStart<'_>) -> &'static str {
    const SPACES: &str = "                                "; // 32 spaces
    let n = match e
        .attributes()
        .flatten()
        .find(|a| a.key.as_ref() == b"text:c")
    {
        None => 1,
        Some(a) => std::str::from_utf8(a.value.as_ref())
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(1),
    }
    .min(SPACES.len());
    &SPACES[..n]
}

/// Returns the `text:outline-level` attribute (1–6, clamped), or `None`.
fn outline_level(e: &quick_xml::events::BytesStart<'_>) -> Option<u8> {
    for attr in e.attributes().flatten() {
        if attr.key.as_ref() == b"text:outline-level" {
            let v = std::str::from_utf8(attr.value.as_ref()).ok()?;
            let n: u8 = v.trim().parse().ok()?;
            return Some(n.clamp(1, 6));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use crate::parse::BlockType;

    fn build_odt(content_xml: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut zip = zip::ZipWriter::new(Cursor::new(&mut buf));
            let opts: zip::write::FileOptions = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zip.start_file("mimetype", opts).expect("start mimetype");
            zip.write_all(b"application/vnd.oasis.opendocument.text")
                .expect("write mimetype");
            zip.start_file("content.xml", opts).expect("start content");
            zip.write_all(content_xml.as_bytes())
                .expect("write content");
            zip.finish().expect("finish zip");
        }
        buf
    }

    const CONTENT: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<office:document-content xmlns:office="urn:office" xmlns:text="urn:text">
  <office:body><office:text>
    <text:h text:outline-level="1">Chapter One</text:h>
    <text:p>An introductory paragraph.</text:p>
    <text:h text:outline-level="2">Section A</text:h>
    <text:p>Body under section A.</text:p>
    <text:p></text:p>
  </office:text></office:body>
</office:document-content>"#;

    fn extract(content_xml: &str) -> ExtractOutput {
        let bytes = build_odt(content_xml);
        OdtExtractor.extract(&bytes).expect("odt extraction")
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
    fn odt_byte_identity() {
        let out = extract(CONTENT);
        assert!(!out.blocks.is_empty());
        assert_byte_identity(&out);
    }

    #[test]
    fn odt_heading_detection() {
        let out = extract(CONTENT);
        let h = out
            .blocks
            .iter()
            .find(|b| b.text == "Chapter One")
            .expect("heading present");
        assert_eq!(h.block_type, BlockType::Heading.as_str());
        assert_eq!(h.section_path, "Chapter One");
    }

    #[test]
    fn odt_section_path_inheritance() {
        let out = extract(CONTENT);
        let body = out
            .blocks
            .iter()
            .find(|b| b.text == "Body under section A.")
            .expect("body paragraph present");
        assert_eq!(body.block_type, BlockType::Paragraph.as_str());
        assert_eq!(body.section_path, "Chapter One > Section A");
    }

    #[test]
    fn odt_empty_paragraph_skipped() {
        let out = extract(CONTENT);
        // The trailing `<text:p></text:p>` produces no block.
        assert!(out.blocks.iter().all(|b| !b.text.is_empty()));
    }

    #[test]
    fn odt_anchors_index_aligned() {
        let out = extract(CONTENT);
        assert_eq!(out.anchors.len(), out.blocks.len());
        for (i, a) in out.anchors.iter().enumerate() {
            assert!(
                matches!(a, SourceAnchor::Odt { .. }),
                "anchor[{i}] must be SourceAnchor::Odt"
            );
        }
    }

    #[test]
    fn odt_node_paths_distinct() {
        let out = extract(CONTENT);
        let mut seen = std::collections::HashSet::new();
        for (i, a) in out.anchors.iter().enumerate() {
            let SourceAnchor::Odt { node_path } = a else {
                panic!("anchor[{i}] not Odt");
            };
            assert!(
                seen.insert(node_path.clone()),
                "duplicate node_path {node_path:?}"
            );
        }
    }

    #[test]
    fn odt_invalid_bytes_returns_error() {
        let err = OdtExtractor
            .extract(b"not a zip")
            .expect_err("non-ZIP must error");
        assert!(matches!(err, LensError::Parse(_)), "got {err:?}");
    }

    #[test]
    fn odt_missing_content_xml_returns_error() {
        // A valid ZIP without a content.xml entry.
        let mut buf = Vec::new();
        {
            let mut zip = zip::ZipWriter::new(Cursor::new(&mut buf));
            let opts: zip::write::FileOptions = zip::write::FileOptions::default();
            zip.start_file("other.xml", opts).unwrap();
            zip.write_all(b"<x/>").unwrap();
            zip.finish().unwrap();
        }
        let err = OdtExtractor
            .extract(&buf)
            .expect_err("missing content.xml must error");
        assert!(matches!(err, LensError::Parse(_)), "got {err:?}");
    }

    #[test]
    fn odt_empty_document() {
        let out = extract(
            r#"<?xml version="1.0"?><office:body xmlns:office="urn:o"><office:text/></office:body>"#,
        );
        assert!(out.blocks.is_empty());
        assert!(out.extracted_text.is_empty());
    }

    #[test]
    fn odt_entities_and_inline_spans() {
        // `&amp;` resolves; inline <text:span> text is folded into the parent.
        let content = r#"<?xml version="1.0"?>
<office:document-content xmlns:office="urn:o" xmlns:text="urn:t">
  <office:body><office:text>
    <text:p>Tom &amp; Jerry <text:span>and friends</text:span>.</text:p>
  </office:text></office:body>
</office:document-content>"#;
        let out = extract(content);
        assert_byte_identity(&out);
        let p = &out.blocks[0];
        assert_eq!(p.text, "Tom & Jerry and friends.");
    }

    #[test]
    fn odt_inline_whitespace_elements() {
        // ODF inline whitespace elements (LibreOffice/Google-Docs emit these):
        // <text:s text:c="3"/> → 3 spaces, <text:tab/> → tab, <text:line-break/>
        // → newline. Without handling them the surrounding runs would collapse.
        let content = r#"<?xml version="1.0"?>
<office:document-content xmlns:office="urn:o" xmlns:text="urn:t">
  <office:body><office:text>
    <text:p>A<text:s text:c="3"/>B<text:tab/>C<text:line-break/>D</text:p>
  </office:text></office:body>
</office:document-content>"#;
        let out = extract(content);
        assert_byte_identity(&out);
        assert_eq!(out.blocks[0].text, "A   B\tC\nD");
    }

    #[test]
    fn odt_single_space_default() {
        // `<text:s/>` with no text:c attribute is a single space (ODF default).
        let content = r#"<?xml version="1.0"?>
<office:document-content xmlns:office="urn:o" xmlns:text="urn:t">
  <office:body><office:text>
    <text:p>X<text:s/>Y</text:p>
  </office:text></office:body>
</office:document-content>"#;
        let out = extract(content);
        assert_byte_identity(&out);
        assert_eq!(out.blocks[0].text, "X Y");
    }

    #[test]
    fn odt_nested_paragraph_does_not_close_early() {
        // An ODF footnote nests a <text:p> inside <text:note> inside the outer
        // <text:p>. The nested </text:p> must NOT close the outer block — its
        // trailing text ("the end.") must survive.
        let content = r#"<?xml version="1.0"?>
<office:document-content xmlns:office="urn:o" xmlns:text="urn:t">
  <office:body><office:text>
    <text:p>Before<text:note><text:note-body><text:p>footnote body</text:p></text:note-body></text:note> the end.</text:p>
  </office:text></office:body>
</office:document-content>"#;
        let out = extract(content);
        assert_byte_identity(&out);
        // Exactly ONE block (the outer paragraph); the nested <text:p> folds in.
        assert_eq!(out.blocks.len(), 1, "nested <text:p> must not emit a block");
        assert_eq!(out.blocks[0].text, "Beforefootnote body the end.");
    }

    #[test]
    fn odt_read_capped_rejects_overflow() {
        // Test the bounded-read helper directly with a small cap.
        let data = vec![b'a'; 8192];
        let err =
            read_capped(Cursor::new(&data), 1024).expect_err("entry over the cap must be rejected");
        assert!(
            matches!(&err, LensError::Validation(m) if m.contains("decompression bomb")),
            "got {err:?}"
        );
    }

    #[test]
    fn odt_read_capped_accepts_at_limit() {
        let data = vec![b'x'; 1024];
        let s = read_capped(Cursor::new(&data), 1024).expect("at-limit entry accepted");
        assert_eq!(s.len(), 1024, "full content returned, not truncated");
    }

    #[test]
    fn odt_bounded_reader_accepts_large_valid_doc() {
        // ~4 MB of repeated text — well under the 256 MB ceiling.
        let big = "lorem ipsum ".repeat(350_000);
        let content = format!(
            r#"<?xml version="1.0"?>
<office:document-content xmlns:office="urn:o" xmlns:text="urn:t">
  <office:body><office:text><text:p>{big}</text:p></office:text></office:body>
</office:document-content>"#
        );
        let out = extract(&content);
        assert_byte_identity(&out);
        assert_eq!(out.blocks.len(), 1);
        assert!(out.blocks[0].text.len() >= 4_000_000, "full body extracted");
    }

    #[test]
    fn odt_text_s_zero_emits_no_space_and_large_clamps() {
        // `text:c="0"` is valid ODF and means ZERO spaces (must NOT become 1);
        // a huge `text:c` clamps to 32 rather than the literal count.
        let content = r#"<?xml version="1.0"?>
<office:document-content xmlns:office="urn:o" xmlns:text="urn:t">
  <office:body><office:text>
    <text:p>A<text:s text:c="0"/>B</text:p>
    <text:p>C<text:s text:c="99999"/>D</text:p>
  </office:text></office:body>
</office:document-content>"#;
        let out = extract(content);
        assert_byte_identity(&out);
        assert_eq!(out.blocks[0].text, "AB", "text:c=0 must emit zero spaces");
        // 32 spaces (clamped) between C and D.
        assert_eq!(out.blocks[1].text, format!("C{}D", " ".repeat(32)));
    }

    #[test]
    fn odt_self_closing_block_advances_node_path() {
        // A self-closing `<text:p/>` emits no block but MUST advance the p counter
        // so the following paragraph keeps a stable node_path (parity with the
        // paired-empty `<text:p></text:p>` case).
        let content = r#"<?xml version="1.0"?>
<office:document-content xmlns:office="urn:o" xmlns:text="urn:t">
  <office:body><office:text>
    <text:p>First</text:p>
    <text:p/>
    <text:p>Third</text:p>
  </office:text></office:body>
</office:document-content>"#;
        let out = extract(content);
        assert_byte_identity(&out);
        // Two emitted blocks (the self-closing middle one is empty).
        assert_eq!(out.blocks.len(), 2);
        let third = out
            .blocks
            .iter()
            .position(|b| b.text == "Third")
            .expect("third paragraph");
        let SourceAnchor::Odt { node_path } = &out.anchors[third] else {
            panic!("odt anchor");
        };
        assert_eq!(
            node_path, "body/text:p[2]",
            "self-closing <text:p/> must advance the counter (Third == p[2], not p[1])"
        );
    }

    #[test]
    fn odt_multibyte_byte_identity_through_walker() {
        // CJK + emoji passing through the shared XML walker must preserve
        // byte-identity (offsets are byte offsets, never mid-codepoint).
        let content = r#"<?xml version="1.0"?>
<office:document-content xmlns:office="urn:o" xmlns:text="urn:t">
  <office:body><office:text>
    <text:h text:outline-level="1">日本語の見出し</text:h>
    <text:p>café — naïve — 🚀 résumé</text:p>
  </office:text></office:body>
</office:document-content>"#;
        let out = extract(content);
        assert_byte_identity(&out);
        assert_eq!(out.blocks[0].text, "日本語の見出し");
        assert_eq!(out.blocks[1].text, "café — naïve — 🚀 résumé");
    }

    #[test]
    fn odt_real_world_fixture_full_fidelity() {
        // Real pandoc-generated ODT: full OASIS namespaces, bookmark start-tags,
        // nested list/table <text:p>, inline spans, smart quotes, multi-byte content.
        let bytes = include_bytes!("../../tests/fixtures/real_pandoc.odt");
        let out = OdtExtractor.extract(bytes).expect("real ODT extraction");
        assert_byte_identity(&out);

        let h1 = out
            .blocks
            .iter()
            .find(|b| b.text == "Chapter One: Introduction")
            .expect("h1 present");
        assert_eq!(h1.block_type, BlockType::Heading.as_str());
        let sect_a = out
            .blocks
            .iter()
            .find(|b| b.text == "Section A")
            .expect("h2 present");
        assert_eq!(
            sect_a.section_path, "Chapter One: Introduction > Section A",
            "heading trail must nest"
        );

        assert!(
            out.blocks
                .iter()
                .any(|b| b.text.contains("café & “quotes”")),
            "inline-formatted paragraph with entities must be captured verbatim"
        );

        for cell in ["Alice", "Engineer", "Bob", "Designer"] {
            assert!(
                out.blocks.iter().any(|b| b.text == cell),
                "table cell {cell:?} must be extracted"
            );
        }

        assert!(
            out.blocks
                .iter()
                .any(|b| b.text.contains("日本語") && b.text.contains('🚀')),
            "CJK + emoji must survive"
        );

        assert_eq!(out.anchors.len(), out.blocks.len());
        assert!(
            out.anchors
                .iter()
                .all(|a| matches!(a, SourceAnchor::Odt { .. })),
            "all anchors must be Odt"
        );
    }

    #[test]
    fn odt_snapshot_block_structure() {
        let out = extract(CONTENT);
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
        insta::assert_json_snapshot!("odt_block_structure", snaps);
    }
}
