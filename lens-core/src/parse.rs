//! Block parser for plain-text and Markdown sources.
//!
//! Converts raw source text into a flat `Vec<Block>` where every block carries:
//! * The verbatim source bytes (`text`), slice-verified against `char_start..char_end`.
//! * The heading trail (`section_path`) in force at that point in the document.
//! * A semantic label (`block_type`): `"heading"`, `"paragraph"`, `"code"`, or
//!   `"list_item"`.
//!
//! **Byte-identity invariant:** for every returned block,
//! `src[block.char_start..block.char_end] == block.text` must hold byte-for-byte.
//! The parser slices the source string by the byte offsets the parser reports —
//! it never reconstructs text from event data, which would break multi-byte
//! characters and normalisation differences.

use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};

use crate::LensError;

/// Selects the document format for [`parse_blocks`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    /// Plain text (no markup). Blocks are paragraphs split on blank lines.
    Text,
    /// CommonMark / GitHub-flavoured Markdown.
    Markdown,
}

impl SourceKind {
    /// Maps the `sources.kind` string stored in SQLite (`"text"` | `"markdown"`)
    /// to the enum variant.  Unknown values are an input-validation error.
    pub fn from_kind_str(s: &str) -> Result<Self, LensError> {
        match s {
            "text" => Ok(Self::Text),
            "markdown" => Ok(Self::Markdown),
            other => Err(LensError::Validation(format!(
                "unknown source kind: {other:?}; expected \"text\" or \"markdown\""
            ))),
        }
    }
}

/// Semantic block-type labels (the `Block::block_type` / `chunks.block_type`
/// column values).
///
/// Single source of truth for the block-type string literals the parser emits.
pub(crate) mod block_type {
    /// A markdown/plain heading.
    pub const HEADING: &str = "heading";
    /// A paragraph of prose.
    pub const PARAGRAPH: &str = "paragraph";
    /// A fenced or indented code block.
    pub const CODE: &str = "code";
    /// A single list item.
    pub const LIST_ITEM: &str = "list_item";
}

/// A bounded, semantically-labelled span within a source document.
///
/// Every field is derived from the source text so that callers can reconstruct
/// the original bytes without re-reading the file:
///
/// ```text
/// assert_eq!(&src[block.char_start..block.char_end], block.text);
/// ```
///
/// The name `char_start`/`char_end` mirrors the `chunks` table column names but
/// the values are **byte offsets**, not Unicode scalar-value offsets — consistent
/// with how Rust string slicing and `pulldown-cmark` ranges work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// Semantic block type: `"heading"`, `"paragraph"`, `"code"`, or `"list_item"`.
    pub block_type: String,
    /// Slash-joined heading trail, e.g. `"Intro > Background > Detail"`.
    /// Empty string for top-level or plain-text content.
    pub section_path: String,
    /// Verbatim bytes from the source document for this block.
    ///
    /// Invariant: `src[char_start..char_end] == text` (bytes, not chars).
    pub text: String,
    /// Byte offset of the first byte of this block in the source string.
    pub char_start: usize,
    /// Byte offset one-past the last byte of this block in the source string.
    pub char_end: usize,
}

/// Parses `src` into a flat list of [`Block`]s according to `kind`.
///
/// Every block satisfies:
/// ```text
/// src[block.char_start..block.char_end] == block.text   // bytes, not chars
/// ```
///
/// # Markdown behaviour
/// * A heading tag updates the internal heading stack and emits a `block_type =
///   "heading"` block whose `text` is the trimmed heading content.
/// * Paragraph content is emitted as `"paragraph"` blocks.
/// * Fenced / indented code is emitted as `"code"` blocks.
/// * List items are emitted as `"list_item"` blocks.
/// * `section_path` is the `" > "`-joined heading trail active at the start of
///   each block (e.g. `"A > B > C"` for content nested under `# A / ## B / ### C`).
///
/// # Plain-text behaviour
/// The whole document is split on runs of blank lines.  Every segment is trimmed,
/// non-empty segments are emitted as `block_type = "paragraph"` blocks with
/// `section_path = ""`.
pub fn parse_blocks(src: &str, kind: SourceKind) -> Vec<Block> {
    match kind {
        SourceKind::Text => parse_text(src),
        SourceKind::Markdown => parse_markdown(src),
    }
}

// ---------------------------------------------------------------------------
// Plain-text path
// ---------------------------------------------------------------------------

/// Splits `src` on one-or-more blank lines and emits each non-empty segment as
/// a `"paragraph"` block.  Byte offsets are computed by scanning the raw source
/// so the invariant `src[start..end] == text` holds.
fn parse_text(src: &str) -> Vec<Block> {
    let mut blocks = Vec::new();

    // Walk the source byte-by-byte tracking paragraph boundaries.
    // A paragraph starts at the first non-whitespace-line byte after either the
    // beginning of the string or a blank line; it ends just before the blank line.
    let len = src.len();
    let mut pos = 0usize;

    while pos < len {
        // Skip leading blank lines / whitespace-only lines.
        let para_start = skip_blank_lines(src, pos);
        if para_start >= len {
            break;
        }

        // Advance until we hit a blank line (two consecutive newlines with only
        // whitespace between) or the end of the string.
        let para_end = find_paragraph_end(src, para_start);

        // Trim the segment to remove trailing newline/whitespace without losing
        // the byte-start anchor.  The `text` field holds the trimmed content but
        // `char_start` points to the un-trimmed start; we re-anchor char_start to
        // the first non-whitespace byte so the invariant still holds.
        let raw = &src[para_start..para_end];
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            pos = para_end;
            continue;
        }
        // `trimmed` is a sub-slice of `raw` which is a sub-slice of `src`, so
        // pointer subtraction yields the exact byte offset within `src`.
        let trim_offset = trimmed.as_ptr() as usize - src.as_ptr() as usize;
        let trim_end = trim_offset + trimmed.len();

        blocks.push(Block {
            block_type: block_type::PARAGRAPH.to_string(),
            section_path: String::new(),
            text: trimmed.to_string(),
            char_start: trim_offset,
            char_end: trim_end,
        });

        pos = para_end;
    }

    blocks
}

/// Returns the byte index of the first non-blank-line character at or after
/// `pos` in `src`.
fn skip_blank_lines(src: &str, mut pos: usize) -> usize {
    let len = src.len();
    while pos < len {
        // Find the end of this line.
        let line_start = pos;
        let line_end = src[pos..].find('\n').map(|i| pos + i + 1).unwrap_or(len);
        let line = &src[line_start..line_end];
        if line.trim().is_empty() {
            pos = line_end;
        } else {
            break;
        }
    }
    pos
}

/// Returns the byte index of the end of the current paragraph starting at
/// `start` (i.e. just before the first blank line, or `src.len()`).
fn find_paragraph_end(src: &str, start: usize) -> usize {
    let len = src.len();
    let mut pos = start;
    loop {
        if pos >= len {
            return len;
        }
        let line_end = src[pos..].find('\n').map(|i| pos + i + 1).unwrap_or(len);
        let line = &src[pos..line_end];
        if line.trim().is_empty() {
            // This line is blank — the paragraph ended on the previous line.
            return pos;
        }
        pos = line_end;
        if pos >= len {
            return len;
        }
    }
}

// ---------------------------------------------------------------------------
// Markdown path
// ---------------------------------------------------------------------------

/// Heading-level index (H1=0 … H6=5) used for the stack.
fn heading_index(level: HeadingLevel) -> usize {
    match level {
        HeadingLevel::H1 => 0,
        HeadingLevel::H2 => 1,
        HeadingLevel::H3 => 2,
        HeadingLevel::H4 => 3,
        HeadingLevel::H5 => 4,
        HeadingLevel::H6 => 5,
    }
}

/// Builds the `" > "`-joined heading trail from the stack.
fn build_section_path(stack: &[Option<String>; 6]) -> String {
    stack
        .iter()
        .filter_map(|opt| opt.as_deref())
        .collect::<Vec<_>>()
        .join(" > ")
}

/// State machine context for block accumulation inside a Markdown container tag.
#[derive(Debug)]
enum BlockContext {
    Paragraph,
    CodeBlock,
    ListItem,
    /// The heading level is recorded on the `TagEnd::Heading` event, not stored
    /// here — this variant exists only to identify the open-heading context.
    Heading,
}

/// Parses `src` as Markdown using `pulldown_cmark` and returns one [`Block`] per
/// logical unit (paragraph, code block, list item, heading).
///
/// # Byte-identity
/// `pulldown_cmark::Parser::into_offset_iter()` yields `(Event, Range<usize>)`
/// where the range is byte offsets into `src`.  We record the **outermost**
/// `Tag::{Paragraph,CodeBlock,Item,Heading}` range and slice `src` directly so
/// `block.text = src[start..end].trim()` — the actual stored `text` equals
/// `src[char_start..char_end]`.  We re-anchor `char_start/char_end` to the trimmed
/// span so the invariant is exact.
fn parse_markdown(src: &str) -> Vec<Block> {
    let parser = Parser::new_ext(src, Options::all()).into_offset_iter();

    let mut blocks: Vec<Block> = Vec::new();
    // heading_stack[0] = current H1 text, [1] = H2, … [5] = H6.
    let mut heading_stack: [Option<String>; 6] = Default::default();
    // Stack of (context, byte_start, section_path at open time).
    let mut ctx_stack: Vec<(BlockContext, usize, String)> = Vec::new();

    for (event, range) in parser {
        match event {
            // ── Opening tags ──────────────────────────────────────────────
            Event::Start(Tag::Paragraph) => {
                let path = build_section_path(&heading_stack);
                ctx_stack.push((BlockContext::Paragraph, range.start, path));
            }
            Event::Start(Tag::CodeBlock(_)) => {
                let path = build_section_path(&heading_stack);
                ctx_stack.push((BlockContext::CodeBlock, range.start, path));
            }
            Event::Start(Tag::Item) => {
                let path = build_section_path(&heading_stack);
                ctx_stack.push((BlockContext::ListItem, range.start, path));
            }
            Event::Start(Tag::Heading { .. }) => {
                let path = build_section_path(&heading_stack);
                ctx_stack.push((BlockContext::Heading, range.start, path));
            }

            // ── Closing tags ──────────────────────────────────────────────
            Event::End(TagEnd::Paragraph) => {
                if let Some((BlockContext::Paragraph, byte_start, path)) = ctx_stack.pop() {
                    let byte_end = range.end;
                    emit_block(
                        src,
                        block_type::PARAGRAPH,
                        &path,
                        byte_start,
                        byte_end,
                        &mut blocks,
                    );
                }
            }
            Event::End(TagEnd::CodeBlock) => {
                if let Some((BlockContext::CodeBlock, byte_start, path)) = ctx_stack.pop() {
                    let byte_end = range.end;
                    emit_block(
                        src,
                        block_type::CODE,
                        &path,
                        byte_start,
                        byte_end,
                        &mut blocks,
                    );
                }
            }
            Event::End(TagEnd::Item) => {
                if let Some((BlockContext::ListItem, byte_start, path)) = ctx_stack.pop() {
                    let byte_end = range.end;
                    emit_block(
                        src,
                        block_type::LIST_ITEM,
                        &path,
                        byte_start,
                        byte_end,
                        &mut blocks,
                    );
                }
            }
            Event::End(TagEnd::Heading(level)) => {
                if let Some((BlockContext::Heading, byte_start, path)) = ctx_stack.pop() {
                    let byte_end = range.end;
                    // Collect the raw heading bytes from src and extract the
                    // inner text (strips Markdown syntax like `## `).
                    let raw = &src[byte_start..byte_end];
                    // The heading *text* is the trimmed text content within the
                    // tag.  We extract it by stripping leading `#` markers and
                    // whitespace from the raw slice.
                    let heading_text = extract_heading_text(raw);

                    // Update the heading stack: clear all deeper levels.
                    let idx = heading_index(level);
                    heading_stack[idx] = Some(heading_text.to_string());
                    for deeper in heading_stack.iter_mut().skip(idx + 1) {
                        *deeper = None;
                    }

                    // Emit a "heading" block. `text` is the trimmed heading
                    // text; `char_start/char_end` anchor to that text inside src.
                    let (start, end) = locate_in_src(src, heading_text, byte_start, byte_end);
                    if end > start {
                        blocks.push(Block {
                            block_type: block_type::HEADING.to_string(),
                            section_path: path,
                            text: heading_text.to_string(),
                            char_start: start,
                            char_end: end,
                        });
                    }
                }
            }

            // All other events (text, soft breaks, hard breaks, …) are handled
            // by the outer tag events above through the raw source slice — we do
            // not reconstruct text from individual Event::Text nodes.
            _ => {}
        }
    }

    blocks
}

/// Emits a `Block` for a completed container tag.
///
/// Slices `src[byte_start..byte_end]`, trims it, then re-anchors `char_start`
/// and `char_end` to the trimmed span so the invariant
/// `src[char_start..char_end] == text` is exact.  Empty spans after trimming are
/// silently dropped.
fn emit_block(
    src: &str,
    block_type: &str,
    section_path: &str,
    byte_start: usize,
    byte_end: usize,
    blocks: &mut Vec<Block>,
) {
    // Guard against out-of-range or inverted ranges (defensive).
    if byte_start >= byte_end || byte_end > src.len() {
        return;
    }
    let raw = &src[byte_start..byte_end];
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return;
    }
    let (start, end) = locate_in_src(src, trimmed, byte_start, byte_end);
    if end <= start {
        return;
    }
    blocks.push(Block {
        block_type: block_type.to_string(),
        section_path: section_path.to_string(),
        text: trimmed.to_string(),
        char_start: start,
        char_end: end,
    });
}

/// Returns `(char_start, char_end)` as byte offsets in `src` for the `needle`
/// substring, searching only within `[window_start..window_end]`.
///
/// If `needle` is found, the returned range satisfies
/// `src[char_start..char_end] == needle` by construction.
fn locate_in_src(
    src: &str,
    needle: &str,
    window_start: usize,
    window_end: usize,
) -> (usize, usize) {
    if needle.is_empty() {
        return (window_start, window_start);
    }
    let window = &src[window_start..window_end.min(src.len())];
    // `needle` was derived from `window.trim()`, so its bytes must be a
    // contiguous sub-slice of `window`.  We locate it by pointer arithmetic.
    let needle_ptr = needle.as_ptr() as usize;
    let window_ptr = window.as_ptr() as usize;
    if needle_ptr >= window_ptr && needle_ptr + needle.len() <= window_ptr + window.len() {
        let offset = needle_ptr - window_ptr;
        let start = window_start + offset;
        let end = start + needle.len();
        (start, end)
    } else {
        // Fallback: search within the window.
        if let Some(pos) = window.find(needle) {
            (window_start + pos, window_start + pos + needle.len())
        } else {
            (window_start, window_start)
        }
    }
}

/// Extracts the plain text of a heading from its raw Markdown source slice.
///
/// Strips leading `#` characters and surrounding whitespace to obtain the
/// heading content as it appears in the document (e.g. `## My Heading` →
/// `"My Heading"`).  This is a best-effort approach that works for the common
/// ATX-style headings that `pulldown_cmark` emits in offset ranges; setext
/// headings are handled by the fallback trim.
///
/// An ATX *closing* `#` sequence is only valid when preceded by whitespace
/// (CommonMark §4.2): `## A #` closes to `"A"`, but `# C#` is NOT a closing
/// marker and must stay `"C#"`. We therefore only strip a trailing `#` run when
/// it is preceded by whitespace (or is the entire remaining content).
fn extract_heading_text(raw: &str) -> &str {
    // ATX headings start with one or more '#' characters followed by whitespace.
    // Trim trailing whitespace first so any ATX closing `#` run sits at the very
    // end of the slice (the raw range may carry a trailing newline).
    let stripped = raw.trim_start_matches('#').trim();
    // Strip an ATX closing `#` run only when it is preceded by whitespace (or is
    // the whole remainder) — a `#` glued to a word (e.g. `C#`) is part of the
    // heading text, not a closing marker.
    let stripped = strip_atx_closing(stripped).trim_end();
    // If nothing was stripped fall back to a plain trim (setext headings).
    if stripped.is_empty() {
        raw.trim()
    } else {
        stripped
    }
}

/// Removes a trailing ATX closing `#` run from `s` iff it is a valid closing
/// marker: the `#` run must be at the end of the string and either preceded by
/// whitespace or constitute the entire string. A `#` run glued directly to a
/// non-whitespace character (e.g. `C#`) is left intact.
fn strip_atx_closing(s: &str) -> &str {
    let without_hashes = s.trim_end_matches('#');
    if without_hashes.len() == s.len() {
        // No trailing `#` at all.
        return s;
    }
    // The closing marker is valid only when the char immediately before the `#`
    // run is whitespace, or the whole remainder is `#` (e.g. `#` / `###`).
    match without_hashes.chars().next_back() {
        None => without_hashes, // entire remainder was `#`s
        Some(c) if c.is_whitespace() => without_hashes,
        Some(_) => s, // glued `#` — keep it
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the byte-identity invariant for every block returned by the parser.
    fn assert_byte_identity(src: &str, blocks: &[Block]) {
        for (i, b) in blocks.iter().enumerate() {
            assert!(
                b.char_end <= src.len(),
                "block[{i}] char_end {} > src.len() {}",
                b.char_end,
                src.len()
            );
            assert_eq!(
                &src[b.char_start..b.char_end],
                b.text,
                "byte-identity invariant violated for block[{i}] ({:?})",
                b.block_type
            );
        }
    }

    #[test]
    fn plain_text_single_paragraph() {
        let src = "Hello, world!";
        let blocks = parse_blocks(src, SourceKind::Text);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].block_type, "paragraph");
        assert_eq!(blocks[0].section_path, "");
        assert_eq!(blocks[0].text, "Hello, world!");
        assert_byte_identity(src, &blocks);
    }

    #[test]
    fn plain_text_multiple_paragraphs() {
        let src = "First paragraph.\n\nSecond paragraph.\n\nThird.";
        let blocks = parse_blocks(src, SourceKind::Text);
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].text, "First paragraph.");
        assert_eq!(blocks[1].text, "Second paragraph.");
        assert_eq!(blocks[2].text, "Third.");
        assert_byte_identity(src, &blocks);
    }

    #[test]
    fn plain_text_empty() {
        let src = "";
        let blocks = parse_blocks(src, SourceKind::Text);
        assert!(blocks.is_empty());
    }

    #[test]
    fn plain_text_only_whitespace() {
        let src = "   \n\n  \n";
        let blocks = parse_blocks(src, SourceKind::Text);
        assert!(blocks.is_empty());
    }

    #[test]
    fn markdown_headings_section_path() {
        let src =
            "# A\n\nContent under A.\n\n## B\n\nContent under B.\n\n### C\n\nContent under C.\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        assert_byte_identity(src, &blocks);

        // Find the paragraph block under ### C
        let para_c = blocks
            .iter()
            .find(|b| b.block_type == "paragraph" && b.text == "Content under C.")
            .expect("paragraph under ### C should exist");
        assert_eq!(para_c.section_path, "A > B > C");
    }

    #[test]
    fn markdown_section_path_resets_on_shallow_heading() {
        let src = "# A\n\n## B\n\nText under B.\n\n# New A\n\nText under New A.\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        assert_byte_identity(src, &blocks);

        // "Text under B." should have section_path "A > B".
        let under_b = blocks
            .iter()
            .find(|b| b.block_type == "paragraph" && b.text == "Text under B.")
            .expect("paragraph under ## B");
        assert_eq!(under_b.section_path, "A > B");

        // "# New A" replaces H1 and clears H2+, so the paragraph beneath it
        // has section_path "New A" (only H1 active, no H2).
        let under_new_a = blocks
            .iter()
            .find(|b| b.block_type == "paragraph" && b.text == "Text under New A.")
            .expect("paragraph under # New A");
        assert_eq!(under_new_a.section_path, "New A");
    }

    #[test]
    fn markdown_code_block() {
        let src = "# Code\n\n```rust\nfn main() {}\n```\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        assert_byte_identity(src, &blocks);
        let code = blocks
            .iter()
            .find(|b| b.block_type == "code")
            .expect("code block should exist");
        assert_eq!(code.section_path, "Code");
    }

    #[test]
    fn source_kind_from_str() {
        assert_eq!(SourceKind::from_kind_str("text").unwrap(), SourceKind::Text);
        assert_eq!(
            SourceKind::from_kind_str("markdown").unwrap(),
            SourceKind::Markdown
        );
        assert!(SourceKind::from_kind_str("pdf").is_err());
    }

    #[test]
    fn markdown_heading_hash_not_corrupted() {
        // `# C#` — the trailing `#` is glued to `C`, so it is NOT an ATX closing
        // marker and must be preserved as part of the heading text.
        let src = "# C#\n\nBody.\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        assert_byte_identity(src, &blocks);
        let heading = blocks
            .iter()
            .find(|b| b.block_type == "heading")
            .expect("heading should exist");
        assert_eq!(heading.text, "C#");
        // The propagated section_path on the body inherits the intact heading.
        let body = blocks
            .iter()
            .find(|b| b.block_type == "paragraph")
            .expect("paragraph should exist");
        assert_eq!(body.section_path, "C#");
    }

    #[test]
    fn markdown_heading_atx_closing_marker_stripped() {
        // `## A #` — the trailing `#` is preceded by whitespace, so it IS a valid
        // ATX closing marker and must be stripped, leaving "A".
        let src = "## A #\n\nBody.\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        assert_byte_identity(src, &blocks);
        let heading = blocks
            .iter()
            .find(|b| b.block_type == "heading")
            .expect("heading should exist");
        assert_eq!(heading.text, "A");
    }

    #[test]
    fn markdown_multibyte_byte_identity() {
        // Emoji and CJK: ensure byte offsets, not char offsets, are used.
        let src = "# 日本語\n\nこんにちは 🦀 world\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        assert_byte_identity(src, &blocks);
    }
}
