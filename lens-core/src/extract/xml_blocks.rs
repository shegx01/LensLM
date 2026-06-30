//! Shared `quick-xml` block walker for the ODT and EPUB extractors (issue #77).
//!
//! ODT (`content.xml`) and EPUB (XHTML spine entries) share the same event-loop
//! shape: walk the document, open a block at the OUTERMOST block-level element
//! (`<text:h>`/`<text:p>` for ODF, `<h1>`..`<h6>`/`<p>` for XHTML), fold inline
//! text / CDATA / entity references into the open block, and close it when the
//! matching end tag returns to the depth it was opened at — emitting one
//! [`Block`] (build-as-you-go byte offsets) and letting the caller build the
//! format-native anchor.
//!
//! The per-format differences are injected as closures:
//! - `classify`: maps a start tag to [`BlockKind`] (heading level, paragraph, or
//!   not-a-block).
//! - `inline_whitespace`: maps an EMPTY tag (`<text:s/>`, `<br/>`, …) to the
//!   literal whitespace it represents, so real LibreOffice/Google-Docs/EPUB files
//!   do not run their text together.
//! - `make_anchor`: builds the [`SourceAnchor`] for the just-closed block.
//!
//! Invariants preserved (must stay green): byte-identity
//! (`extracted_text[char_start..char_end] == block.text`) and the build-as-you-go
//! global-offset pattern (see docx.rs:204-207).

use quick_xml::events::{BytesStart, Event};
use quick_xml::reader::Reader;

use crate::LensError;
use crate::parse::{Block, BlockType, SectionPathStack};

use super::SourceAnchor;

/// The block classification of a start tag.
pub(crate) enum BlockKind {
    /// A heading at the given outline level (1–6).
    Heading(u8),
    /// A paragraph (or other block-level element treated as one).
    Paragraph,
}

/// Resolves a quick-xml `GeneralRef` (an `&name;` / `&#NN;` entity reference)
/// into its textual replacement, appending it to `dst`.
///
/// Predefined XML entities (`amp`/`lt`/`gt`/`quot`/`apos`) and numeric character
/// references are resolved; an unknown entity is preserved verbatim as
/// `&name;` so no content is silently lost. Shared by the ODT and EPUB walkers.
pub(crate) fn push_general_ref(dst: &mut String, raw: &str, resolved_char: Option<char>) {
    if let Some(c) = resolved_char {
        // Numeric character reference (`&#NN;` / `&#xNN;`).
        dst.push(c);
    } else if let Some(replacement) = quick_xml::escape::resolve_predefined_entity(raw) {
        dst.push_str(replacement);
    } else {
        // Unknown entity: keep it verbatim rather than dropping content.
        dst.push('&');
        dst.push_str(raw);
        dst.push(';');
    }
}

/// In-flight accumulation state for the currently-open block.
struct OpenBlock {
    /// Heading outline level (1–6); 0 for a paragraph.
    level: u8,
    /// Element nesting depth at which this block was opened (relative to the
    /// open event). A matching end tag finalizes the block only when this returns
    /// to 0, so a NESTED block-level element (e.g. an ODF footnote `<text:p>`
    /// inside `<text:note>`) cannot close the outer block early.
    depth: u32,
    text: String,
}

/// Walks `xhtml`/`content.xml` text, appending one [`Block`] per block-level
/// element to the shared buffers (build-as-you-go global byte offsets) and one
/// anchor per block via `make_anchor`.
///
/// `format` is used only in error messages (e.g. `"ODT"`, `"EPUB"`). `classify`
/// decides which start tags open a block; `inline_whitespace` maps EMPTY tags to
/// literal whitespace; `make_anchor` builds the per-format anchor for each
/// emitted block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn walk_xml_blocks<C, W, A>(
    xml: &str,
    format: &str,
    mut classify: C,
    mut inline_whitespace: W,
    mut make_anchor: A,
    section_path: &mut SectionPathStack,
    extracted_text: &mut String,
    blocks: &mut Vec<Block>,
    anchors: &mut Vec<SourceAnchor>,
) -> Result<(), LensError>
where
    C: FnMut(&BytesStart<'_>) -> Option<BlockKind>,
    W: FnMut(&BytesStart<'_>) -> Option<&'static str>,
    A: FnMut(bool) -> SourceAnchor,
{
    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();
    // Only the OUTERMOST block-level element opens a block; nested inline (and
    // nested block-level, via the depth counter) elements fold in.
    let mut current: Option<OpenBlock> = None;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                if let Some(cur) = current.as_mut() {
                    // Already inside a block: count nesting depth so a nested
                    // block-level end tag does not close the outer block early.
                    cur.depth += 1;
                } else if let Some(kind) = classify(&e) {
                    let level = match kind {
                        BlockKind::Heading(l) => l,
                        BlockKind::Paragraph => 0,
                    };
                    current = Some(OpenBlock {
                        level,
                        depth: 0,
                        text: String::new(),
                    });
                }
            }
            Ok(Event::Empty(e)) => {
                if let Some(cur) = current.as_mut() {
                    // Inside a block: an empty element (e.g. `<text:s/>`,
                    // `<text:tab/>`, `<br/>`) contributes whitespace but no
                    // open/close depth change.
                    if let Some(ws) = inline_whitespace(&e) {
                        cur.text.push_str(ws);
                    }
                } else if let Some(kind) = classify(&e) {
                    // A self-closing BLOCK element (e.g. `<text:p/>`): it is
                    // definitionally empty so emits no block, but the per-element
                    // counter must still advance so node_path positions stay
                    // stable regardless of `<text:p/>` vs `<text:p></text:p>`
                    // serialization (parity with the paired-empty case below).
                    let is_heading = matches!(kind, BlockKind::Heading(_));
                    let _ = make_anchor(is_heading);
                }
            }
            Ok(Event::Text(t)) => {
                if let Some(cur) = current.as_mut() {
                    let decoded = t.xml10_content().map_err(|e| {
                        LensError::Parse(format!("{format} text decode failed: {e}"))
                    })?;
                    cur.text.push_str(decoded.as_ref());
                }
            }
            Ok(Event::CData(c)) => {
                if let Some(cur) = current.as_mut() {
                    let decoded = c.decode().map_err(|e| {
                        LensError::Parse(format!("{format} CDATA decode failed: {e}"))
                    })?;
                    cur.text.push_str(decoded.as_ref());
                }
            }
            Ok(Event::GeneralRef(r)) => {
                if let Some(cur) = current.as_mut() {
                    let resolved = r.resolve_char_ref().ok().flatten();
                    let raw_ref = std::str::from_utf8(r.as_ref()).map_err(|e| {
                        LensError::Parse(format!("{format} entity ref not UTF-8: {e}"))
                    })?;
                    push_general_ref(&mut cur.text, raw_ref, resolved);
                }
            }
            Ok(Event::End(_)) => {
                if let Some(cur) = current.as_mut() {
                    if cur.depth > 0 {
                        // A nested element closed; the outer block stays open.
                        cur.depth -= 1;
                    } else {
                        // The block-level element itself closed: finalize.
                        let cur = current.take().expect("current is Some");
                        let is_heading = cur.level > 0;
                        let text = cur.text.trim().to_string();

                        // The anchor is built whether or not the block is empty
                        // so per-element counters advance for stable positions
                        // (ODT node_path); the caller decides what to do.
                        let anchor = make_anchor(is_heading);

                        if text.is_empty() {
                            continue; // skip empty elements
                        }

                        let (btype, sp) = if is_heading {
                            // Update the trail BEFORE emitting so the heading
                            // carries the full trail it introduces (DOCX parity).
                            section_path.push(cur.level, &text);
                            (BlockType::Heading.as_str(), section_path.current())
                        } else {
                            (BlockType::Paragraph.as_str(), section_path.current())
                        };

                        let char_start = extracted_text.len();
                        extracted_text.push_str(&text);
                        extracted_text.push('\n');
                        let char_end = extracted_text.len() - 1;

                        blocks.push(Block {
                            block_type: btype.to_string(),
                            section_path: sp,
                            text,
                            char_start,
                            char_end,
                        });
                        anchors.push(anchor);
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(LensError::Parse(format!("{format} XML parse error: {e}")));
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(())
}
