//! EPUB extractor (M4 issue #77).
//!
//! [`EpubExtractor`] reads an EPUB (2 or 3) from raw bytes using `rbook`, then
//! parses each spine entry's XHTML with `quick-xml`, emitting one [`Block`] per
//! `<p>` (paragraph) and `<h1>`..`<h6>` (heading) element. It produces a
//! canonical [`ExtractOutput`] where:
//! - `extracted_text` is a SINGLE buffer concatenating every spine entry's text
//!   (block offsets are global byte offsets into this buffer).
//! - `blocks` carries `block_type = "heading"` for `<h1>`..`<h6>` (level taken
//!   from the tag) and `block_type = "paragraph"` for `<p>`. Other block-level
//!   elements (lists, tables) are NOT recognised as their own blocks for v1;
//!   any `<p>` nested inside them is still captured as a paragraph.
//! - `anchors` carries one [`SourceAnchor::Epub { spine_index, href }`] per block.
//!
//! ## Inter-spine separator protocol
//!
//! No extra separators are inserted between spine entries — each block already
//! gets its trailing `'\n'` from the build-as-you-go pattern. An empty spine
//! entry (no `<p>` / `<h*>`) contributes nothing to the buffer. The
//! [`SectionPathStack`] is reset per spine entry (each XHTML document has its own
//! heading hierarchy) by reassigning a fresh stack.
//!
//! ## API note (verified against rbook 0.7.9)
//!
//! The reader is obtained via `Epub::read(Cursor)` then iterated with
//! `epub.reader()` (an `Iterator<Item = ReaderResult<EpubReaderContent>>`). Each
//! item exposes `.position()` (the spine index), `.manifest_entry().href()` (the
//! content document path), and `.content()` (the XHTML string).

use std::io::Cursor;

use quick_xml::events::BytesStart;
use rbook::Epub;

use crate::LensError;
use crate::parse::{Block, SectionPathStack};

use super::xml_blocks::{BlockKind, walk_xml_blocks};
use super::{ExtractOutput, Extractor, SourceAnchor};

/// Hard ceiling on the cumulative extracted-text size while walking the EPUB
/// spine (decompression-bomb / runaway-content guard). An EPUB is a ZIP under the
/// 50 MB stage-1 raw-bytes cap, but a high-ratio spine could inflate well past
/// it; this early-exit bounds the in-memory buffer BEFORE the stage-2 guard runs
/// (see ingest.rs stage-1/stage-2). 256 MB matches the ODT ceiling.
const MAX_DECOMPRESSED_BYTES: usize = 256 * 1024 * 1024;

/// EPUB extractor — implements [`Extractor`] via `rbook` + `quick-xml`.
///
/// Byte-identity offsets follow the DOCX build-as-you-go pattern (see docx.rs:204-207).
///
/// No extra separators between spine entries — each block gets its trailing '\n'
/// from the build-as-you-go pattern. Empty spine entries contribute nothing.
pub struct EpubExtractor;

impl Extractor for EpubExtractor {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError> {
        // `rbook` needs an owned, Seek-able source; copy the slice into a Cursor.
        // NOTE: rbook 0.7.9's `Epub::read` requires `Read + Seek + 'static`, so a
        // borrowed `&[u8]` cursor does not satisfy the bound — the owned copy is
        // unavoidable here. The decompression-bomb risk is instead bounded by the
        // per-spine early-exit ceiling below (and the upstream 50 MB raw cap).
        let epub = Epub::read(Cursor::new(raw.to_vec()))
            .map_err(|e| LensError::Parse(format!("rbook failed to read EPUB: {e}")))?;

        let mut extracted_text = String::new();
        let mut blocks: Vec<Block> = Vec::new();
        let mut anchors: Vec<SourceAnchor> = Vec::new();

        for item in epub.reader() {
            let content =
                item.map_err(|e| LensError::Parse(format!("EPUB spine entry read failed: {e}")))?;
            let spine_index = content.position() as u64;
            let href = content.manifest_entry().href().as_str().to_string();
            let xhtml = content.content();

            // Decompression-bomb guard (pre-walk): rbook fully inflates the spine
            // entry into `xhtml` before we get here, so reject an oversized single
            // entry BEFORE parsing it — this bounds peak memory more tightly than
            // the cumulative post-walk check below. (rbook owns decompression, so
            // a detect-after-inflate guard is the best available without its
            // lower-level zip API.)
            if xhtml.len() > MAX_DECOMPRESSED_BYTES {
                return Err(LensError::Validation(format!(
                    "EPUB spine entry decompresses to more than the \
                     {MAX_DECOMPRESSED_BYTES}-byte limit (possible decompression bomb)"
                )));
            }

            // Each spine entry is its own XHTML document with its own heading
            // hierarchy: reset the section-path stack per entry.
            let mut section_path = SectionPathStack::new();

            walk_xml_blocks(
                xhtml,
                "EPUB XHTML",
                // classify: <h1>..<h6> are headings, <p> a paragraph. Match on the
                // LOCAL name so namespace-prefixed XHTML (e.g. `<html:p>`) is still
                // recognised rather than silently dropping the whole chapter.
                |e: &BytesStart<'_>| {
                    let local = e.local_name();
                    match heading_level(local.as_ref()) {
                        Some(lvl) => Some(BlockKind::Heading(lvl)),
                        None if local.as_ref() == b"p" => Some(BlockKind::Paragraph),
                        None => None,
                    }
                },
                // inline_whitespace: XHTML `<br/>` is a hard line break.
                |e: &BytesStart<'_>| {
                    if e.local_name().as_ref() == b"br" {
                        Some("\n")
                    } else {
                        None
                    }
                },
                // make_anchor: the spine index + href identify the block.
                |_is_heading: bool| SourceAnchor::Epub {
                    spine_index,
                    href: href.clone(),
                },
                &mut section_path,
                &mut extracted_text,
                &mut blocks,
                &mut anchors,
            )?;

            // Decompression-bomb / runaway-content early exit: bound the in-memory
            // buffer per spine entry BEFORE the stage-2 guard runs.
            if extracted_text.len() > MAX_DECOMPRESSED_BYTES {
                return Err(LensError::Validation(format!(
                    "EPUB extracted text exceeds the {MAX_DECOMPRESSED_BYTES}-byte \
                     limit (possible decompression bomb)"
                )));
            }
        }

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

/// Returns the heading level (1–6) for an `<h1>`..`<h6>` tag name, else `None`.
fn heading_level(name: &[u8]) -> Option<u8> {
    match name {
        b"h1" => Some(1),
        b"h2" => Some(2),
        b"h3" => Some(3),
        b"h4" => Some(4),
        b"h5" => Some(5),
        b"h6" => Some(6),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use crate::parse::BlockType;

    fn xhtml(title: &str, body: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>{title}</title></head>
<body>{body}</body></html>"#
        )
    }

    /// Builds a minimal, valid EPUB 3 in memory (mimetype + container + OPF +
    /// the given chapters). Hand-crafted (format structure only — no licensed
    /// content). `chapters` is `(href, xhtml_body)`.
    fn build_epub(chapters: &[(&str, &str)]) -> Vec<u8> {
        let mut manifest = String::new();
        let mut spine = String::new();
        for (i, (href, _)) in chapters.iter().enumerate() {
            manifest.push_str(&format!(
                r#"<item id="c{i}" href="{href}" media-type="application/xhtml+xml"/>"#
            ));
            spine.push_str(&format!(r#"<itemref idref="c{i}"/>"#));
        }
        let opf = format!(
            r#"<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="id">x</dc:identifier><dc:title>T</dc:title><dc:language>en</dc:language></metadata>
  <manifest>{manifest}</manifest>
  <spine>{spine}</spine>
</package>"#
        );

        let mut buf = Vec::new();
        {
            let mut z = zip::ZipWriter::new(Cursor::new(&mut buf));
            let stored: zip::write::FileOptions = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            let defl: zip::write::FileOptions = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            z.start_file("mimetype", stored).unwrap();
            z.write_all(b"application/epub+zip").unwrap();
            z.start_file("META-INF/container.xml", defl).unwrap();
            z.write_all(
                br#"<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"#,
            )
            .unwrap();
            z.start_file("OEBPS/content.opf", defl).unwrap();
            z.write_all(opf.as_bytes()).unwrap();
            for (href, body) in chapters {
                z.start_file(format!("OEBPS/{href}"), defl).unwrap();
                z.write_all(xhtml(href, body).as_bytes()).unwrap();
            }
            z.finish().unwrap();
        }
        buf
    }

    fn extract(chapters: &[(&str, &str)]) -> ExtractOutput {
        let bytes = build_epub(chapters);
        EpubExtractor.extract(&bytes).expect("epub extraction")
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

    const CHAPTERS: &[(&str, &str)] = &[
        (
            "chapter1.xhtml",
            "<h1>Chapter One</h1><p>First para &amp; more.</p><h2>Sub A</h2><p>Under sub A.</p>",
        ),
        (
            "chapter2.xhtml",
            "<h1>Chapter Two</h1><p>Second <em>emphasised</em> para.</p>",
        ),
    ];

    #[test]
    fn epub_byte_identity() {
        let out = extract(CHAPTERS);
        assert!(!out.blocks.is_empty());
        assert_byte_identity(&out);
    }

    #[test]
    fn epub_heading_blocks() {
        let out = extract(CHAPTERS);
        let h1 = out
            .blocks
            .iter()
            .find(|b| b.text == "Chapter One")
            .expect("h1 present");
        assert_eq!(h1.block_type, BlockType::Heading.as_str());
        assert_eq!(h1.section_path, "Chapter One");

        let under = out
            .blocks
            .iter()
            .find(|b| b.text == "Under sub A.")
            .expect("nested paragraph present");
        assert_eq!(under.section_path, "Chapter One > Sub A");
    }

    #[test]
    fn epub_inline_elements_folded() {
        let out = extract(CHAPTERS);
        let p = out
            .blocks
            .iter()
            .find(|b| b.text.starts_with("Second"))
            .expect("second-chapter paragraph");
        assert_eq!(p.text, "Second emphasised para.");
    }

    #[test]
    fn epub_entities_resolved() {
        let out = extract(CHAPTERS);
        assert!(
            out.blocks.iter().any(|b| b.text == "First para & more."),
            "entity &amp; must resolve; blocks: {:#?}",
            out.blocks.iter().map(|b| &b.text).collect::<Vec<_>>()
        );
    }

    #[test]
    fn epub_multi_spine_section_path_resets() {
        let out = extract(CHAPTERS);
        // Chapter Two's H1 must NOT inherit Chapter One's trail.
        let ch2 = out
            .blocks
            .iter()
            .find(|b| b.text == "Chapter Two")
            .expect("chapter two heading");
        assert_eq!(ch2.section_path, "Chapter Two");
    }

    #[test]
    fn epub_anchors_index_aligned() {
        let out = extract(CHAPTERS);
        assert_eq!(out.anchors.len(), out.blocks.len());
        for (i, a) in out.anchors.iter().enumerate() {
            assert!(
                matches!(a, SourceAnchor::Epub { .. }),
                "anchor[{i}] must be SourceAnchor::Epub"
            );
        }
        // The first block comes from spine_index 0; a later block from index 1.
        let SourceAnchor::Epub { spine_index, href } = &out.anchors[0] else {
            panic!("epub anchor");
        };
        assert_eq!(*spine_index, 0);
        assert!(href.ends_with("chapter1.xhtml"), "href: {href}");
    }

    #[test]
    fn epub_invalid_bytes_returns_error() {
        let err = EpubExtractor
            .extract(b"not an epub")
            .expect_err("garbage must error");
        assert!(matches!(err, LensError::Parse(_)), "got {err:?}");
    }

    #[test]
    fn epub_empty_spine_entry_handled() {
        // A chapter with no <p>/<h*> contributes nothing; the other chapter still
        // yields blocks.
        let out = extract(&[
            ("empty.xhtml", "<div><span>no block elements</span></div>"),
            ("real.xhtml", "<p>Real content.</p>"),
        ]);
        assert!(out.blocks.iter().any(|b| b.text == "Real content."));
        assert_byte_identity(&out);
    }

    #[test]
    fn epub_br_becomes_newline() {
        // XHTML `<br/>` inside a paragraph is a hard line break.
        let out = extract(&[("c.xhtml", "<p>Line one<br/>Line two</p>")]);
        assert_byte_identity(&out);
        assert_eq!(out.blocks[0].text, "Line one\nLine two");
    }

    #[test]
    fn epub_nested_paragraph_does_not_close_early() {
        // A nested <p> inside a <blockquote> inside the outer <p>: the inner </p>
        // must NOT close the outer block, and the trailing text must survive.
        let out = extract(&[(
            "c.xhtml",
            "<p>Start<blockquote><p>quoted</p></blockquote> finish.</p>",
        )]);
        assert_byte_identity(&out);
        assert_eq!(
            out.blocks.len(),
            1,
            "nested <p> must not emit a separate block"
        );
        assert_eq!(out.blocks[0].text, "Startquoted finish.");
    }

    #[test]
    fn epub_large_valid_doc_extracts() {
        // A legitimate large body (~2 MB in one paragraph) is well under the
        // 256 MB ceiling and must extract cleanly (proving the early-exit guard
        // does not reject valid content under cap).
        let big = "word ".repeat(400_000); // ~2 MB
        let out = extract(&[("c.xhtml", &format!("<p>{big}</p>"))]);
        assert_byte_identity(&out);
        assert_eq!(out.blocks.len(), 1);
        assert!(out.blocks[0].text.len() >= 1_900_000, "full body extracted");
    }

    #[test]
    fn epub_namespaced_block_elements_extracted() {
        // XHTML that prefixes its block elements (valid XML: `<x:h1>`, `<x:p>`
        // bound to the XHTML namespace) must still be recognised via local-name
        // matching — otherwise an entire chapter would be silently dropped.
        let body = r#"<x:h1 xmlns:x="http://www.w3.org/1999/xhtml">Prefixed Heading</x:h1><x:p xmlns:x="http://www.w3.org/1999/xhtml">Prefixed paragraph.</x:p>"#;
        let out = extract(&[("c.xhtml", body)]);
        assert_byte_identity(&out);
        assert!(
            out.blocks.iter().any(|b| b.text == "Prefixed Heading"),
            "namespaced <x:h1> must be extracted; blocks: {:?}",
            out.blocks.iter().map(|b| &b.text).collect::<Vec<_>>()
        );
        assert!(
            out.blocks.iter().any(|b| b.text == "Prefixed paragraph."),
            "namespaced <x:p> must be extracted"
        );
    }

    #[test]
    fn epub_snapshot_block_structure() {
        let out = extract(CHAPTERS);
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
        insta::assert_json_snapshot!("epub_block_structure", snaps);
    }
}
