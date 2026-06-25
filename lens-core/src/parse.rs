//! Block parser for plain-text and Markdown sources.
//!
//! Converts raw source text into a flat `Vec<Block>` where every block carries:
//! * The verbatim source bytes (`text`), slice-verified against `char_start..char_end`.
//! * The heading trail (`section_path`) in force at that point in the document.
//! * A semantic label (`block_type`): `"heading"`, `"paragraph"`, `"code"`,
//!   `"list_item"`, `"table"`, or `"html"`.
//!
//! **Byte-identity invariant:** for every returned block,
//! `src[block.char_start..block.char_end] == block.text` must hold byte-for-byte.
//! The parser slices the source string by the byte offsets the parser reports —
//! it never reconstructs text from event data, which would break multi-byte
//! characters and normalisation differences.

use std::str::FromStr;

use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};

use crate::LensError;

/// The `sources.kind` discriminant: the source's document format.
///
/// Serialized to / parsed from the EXACT legacy `sources.kind` strings stored in
/// SQLite (`"text"`/`"markdown"`/`"pdf"`/`"docx"`/`"url"`) via [`as_str`](Self::as_str)
/// and [`from_kind_str`](Self::from_kind_str) — the wire/DB format is unchanged.
/// The DB-row / IPC boundary keeps a raw `String`; logic converts to this enum
/// immediately and dispatches via exhaustive `match` so adding a variant is a
/// compile error everywhere it matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    /// Plain text (no markup). Blocks are paragraphs split on blank lines.
    Text,
    /// CommonMark / GitHub-flavoured Markdown.
    Markdown,
    /// A PDF document (pdfium text + per-segment bbox extraction).
    Pdf,
    /// A DOCX document (docx-rs XML walk).
    Docx,
    /// A remote URL (HTML fetched then content-extracted).
    Url,
}

impl SourceKind {
    /// The EXACT `sources.kind` string stored in SQLite for this variant.
    ///
    /// Inverse of [`from_kind_str`](Self::from_kind_str). These strings are the
    /// persisted wire format and MUST NOT change (no DB migration).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Markdown => "markdown",
            Self::Pdf => "pdf",
            Self::Docx => "docx",
            Self::Url => "url",
        }
    }

    /// Maps the `sources.kind` string stored in SQLite to the enum variant.
    /// Unknown values are an input-validation error.
    pub fn from_kind_str(s: &str) -> Result<Self, LensError> {
        match s {
            "text" => Ok(Self::Text),
            "markdown" => Ok(Self::Markdown),
            "pdf" => Ok(Self::Pdf),
            "docx" => Ok(Self::Docx),
            "url" => Ok(Self::Url),
            other => Err(LensError::Validation(format!(
                "unknown source kind: {other:?}; expected one of \"text\", \"markdown\", \"pdf\", \"docx\", \"url\""
            ))),
        }
    }

    /// Whether this kind is *text-like* (`Text`/`Markdown`) — the original
    /// locator content IS the canonical buffer — vs. *derived* (`Pdf`/`Docx`/`Url`),
    /// whose canonical buffer is the persisted `.extracted.txt` sibling. Single
    /// point of truth for the ingest read-path asymmetry (Decision A1).
    pub fn is_text_like(&self) -> bool {
        match self {
            Self::Text | Self::Markdown => true,
            Self::Pdf | Self::Docx | Self::Url => false,
        }
    }
}

impl FromStr for SourceKind {
    type Err = LensError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_kind_str(s)
    }
}

/// The `Block::block_type` / `chunks.block_type` discriminant: a block's semantic
/// label.
///
/// Single source of truth for the block-type string vocabulary the parser and
/// every extractor emit. Serialized to / parsed from the EXACT legacy strings via
/// [`as_str`](Self::as_str) and [`FromStr`] — the persisted `chunks.block_type`
/// values are unchanged. `Block::block_type` stays a `String` at the struct
/// boundary; construction sites use `BlockType::X.as_str()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockType {
    /// A markdown/plain heading.
    Heading,
    /// A paragraph of prose.
    Paragraph,
    /// A fenced or indented code block.
    Code,
    /// A single list item.
    ListItem,
    /// A GFM table, emitted as one block carrying the raw table markdown.
    Table,
    /// A raw HTML block, emitted verbatim.
    Html,
}

impl BlockType {
    /// The EXACT `block_type` string emitted for this variant (the persisted
    /// `chunks.block_type` value). Inverse of [`FromStr`].
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Heading => "heading",
            Self::Paragraph => "paragraph",
            Self::Code => "code",
            Self::ListItem => "list_item",
            Self::Table => "table",
            Self::Html => "html",
        }
    }
}

impl FromStr for BlockType {
    type Err = LensError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "heading" => Ok(Self::Heading),
            "paragraph" => Ok(Self::Paragraph),
            "code" => Ok(Self::Code),
            "list_item" => Ok(Self::ListItem),
            "table" => Ok(Self::Table),
            "html" => Ok(Self::Html),
            other => Err(LensError::Validation(format!(
                "unknown block type: {other:?}; expected one of \"heading\", \"paragraph\", \
                 \"code\", \"list_item\", \"table\", \"html\""
            ))),
        }
    }
}

/// Shared heading-trail stack used by every extractor that tracks `section_path`
/// (the Markdown path in this module and the DOCX extractor in
/// [`crate::extract::docx`]).
///
/// Headings are 1–6 (H1…H6). [`push`](Self::push) records a heading at a level,
/// clearing every deeper level so a new H2 drops any H3+ beneath it;
/// [`current`](Self::current) returns the ` > `-joined trail of every level still
/// in force (e.g. `"A > B > C"`). The trail is empty at the document top.
///
/// A fixed `[Option<String>; 6]` (rather than a growable stack) keeps the level →
/// slot mapping direct and matches the shape the Markdown path already used.
#[derive(Debug, Default)]
pub(crate) struct SectionPathStack {
    /// `levels[0]` = current H1 text, `[1]` = H2, … `[5]` = H6. `None` when that
    /// level is not currently in force.
    levels: [Option<String>; 6],
}

impl SectionPathStack {
    /// Creates an empty stack (no headings in force).
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Records `text` as the heading at `level` (1–6, clamped) and clears every
    /// deeper level (a new heading at `level` invalidates any sub-heading below
    /// it). Levels outside 1–6 are clamped into range so callers cannot panic.
    pub(crate) fn push(&mut self, level: u8, text: &str) {
        let idx = (level.clamp(1, 6) - 1) as usize;
        self.levels[idx] = Some(text.to_string());
        for deeper in self.levels.iter_mut().skip(idx + 1) {
            *deeper = None;
        }
    }

    /// Returns the current ` > `-joined heading trail (empty at the document top).
    pub(crate) fn current(&self) -> String {
        self.levels
            .iter()
            .filter_map(|opt| opt.as_deref())
            .collect::<Vec<_>>()
            .join(" > ")
    }
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
        SourceKind::Markdown => parse_markdown(src),
        SourceKind::Text => parse_text(src),
        // The derived kinds (`Pdf`/`Docx`/`Url`) must never reach here — each has
        // a dedicated `Extractor` and only `Text`/`Markdown` build a
        // `TextExtractor` (see `extract::extractor_for`). A `debug_assert!`
        // catches a mis-call in debug/test builds; release falls through to
        // `parse_text` as a safe, non-panicking default rather than crashing.
        SourceKind::Pdf | SourceKind::Docx | SourceKind::Url => {
            debug_assert!(
                false,
                "parse_blocks called with derived kind {kind:?} — use an Extractor"
            );
            parse_text(src)
        }
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
            block_type: BlockType::Paragraph.as_str().to_string(),
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

/// Heading level (H1=1 … H6=6) for [`SectionPathStack::push`].
fn heading_level_u8(level: HeadingLevel) -> u8 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

/// State machine context for block accumulation inside a Markdown container tag.
#[derive(Debug)]
enum BlockContext {
    Paragraph,
    CodeBlock,
    /// A list item. `emitted` flips to `true` once the item's own-text has been
    /// accounted for — either by emitting the leading own-text slice at the
    /// item's FIRST block-level child (nested list / table / html block / code
    /// block / wrapped paragraph), or by `End(Item)` emitting the whole span for
    /// a tight leaf item with no block-level child. The own-text slice is always
    /// clamped to the byte BEFORE the first block-level child so a block child is
    /// never re-included in the item span (B2 loose-list & table-in-item fix).
    ///
    /// A loose item's own text is itself a `Paragraph` child, which the paragraph
    /// arm emits — so the clamped leading slice there is just the bare marker
    /// (empty after trim) and is skipped (B5).
    ListItem {
        emitted: bool,
    },
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
///
/// # Tables (B1)
/// A GFM table's cell text is inline (not wrapped in `Paragraph`), so without a
/// dedicated arm it would hit the `_ => {}` catch-all and be lost entirely. We
/// emit the **whole table** as ONE `"table"` block whose `text` is the raw table
/// markdown `src[table_range]` (byte-identical — never linearised, which would
/// break byte-identity), then swallow every inner table event via `table_depth`
/// until `End(Table)` so cell text never pollutes another arm.
///
/// # Nested list items (B2)
/// `into_offset_iter` gives an outer `Item` a range spanning its nested sublist.
/// Slicing that whole range would duplicate the inner item's text in the outer
/// block and emit it out of reading order. Instead each item emits only its
/// **own** text: the span from the item start to the start of its first nested
/// `Tag::List` (or to `End(Item)` if none), so blocks come out in document order
/// with no duplication.
fn parse_markdown(src: &str) -> Vec<Block> {
    let parser = Parser::new_ext(src, Options::all()).into_offset_iter();

    let mut blocks: Vec<Block> = Vec::new();
    // Shared heading-trail stack (H1…H6); see [`SectionPathStack`].
    let mut heading_stack = SectionPathStack::new();
    // Stack of (context, byte_start, section_path at open time).
    let mut ctx_stack: Vec<(BlockContext, usize, String)> = Vec::new();
    // Depth of open `Tag::Table`s. While > 0 we swallow every event (the whole
    // table was already emitted as one block at `Start(Table)`), so inner
    // TableHead/TableRow/TableCell + their inline text never reach another arm.
    let mut table_depth: usize = 0;

    for (event, range) in parser {
        // Inside a table everything is already captured by the single table
        // block; only track table nesting and drop all other events.
        if table_depth > 0 {
            match event {
                Event::Start(Tag::Table(_)) => table_depth += 1,
                Event::End(TagEnd::Table) => table_depth -= 1,
                _ => {}
            }
            continue;
        }

        // Any block-level child opening directly inside an open list item is that
        // item's FIRST block-level child iff the item has not yet emitted. Emit
        // the item's leading own-text — clamped to the byte BEFORE this child — so
        // a list / table / html / code / paragraph child is never re-included in
        // the item span, fixing loose-list duplication (B1), table-in-item
        // double-emit (B2), and bare-marker artifacts (B5). The leading slice is
        // skipped when it is empty or only a list marker after trimming.
        if matches!(
            event,
            Event::Start(
                Tag::Paragraph | Tag::CodeBlock(_) | Tag::List(_) | Tag::Table(_) | Tag::HtmlBlock
            )
        ) {
            emit_item_own_text_before_child(src, &mut ctx_stack, range.start, &mut blocks);
        }

        match event {
            // ── Opening tags ──────────────────────────────────────────────
            Event::Start(Tag::Paragraph) => {
                let path = heading_stack.current();
                ctx_stack.push((BlockContext::Paragraph, range.start, path));
            }
            Event::Start(Tag::CodeBlock(_)) => {
                let path = heading_stack.current();
                ctx_stack.push((BlockContext::CodeBlock, range.start, path));
            }
            Event::Start(Tag::Item) => {
                let path = heading_stack.current();
                ctx_stack.push((BlockContext::ListItem { emitted: false }, range.start, path));
            }
            Event::Start(Tag::List(_)) => {
                // The nested list itself emits nothing here — the enclosing item's
                // own text (if any) was already emitted above, clamped to this
                // list's start. Nested items emit via their own `Start(Item)` /
                // child / `End(Item)` flow.
            }
            Event::Start(Tag::Table(_)) => {
                // Emit the WHOLE table as one block (raw markdown, byte-identical)
                // then swallow all inner events until `End(Table)` (B1).
                let path = heading_stack.current();
                emit_block(
                    src,
                    BlockType::Table.as_str(),
                    &path,
                    range.start,
                    range.end,
                    &mut blocks,
                );
                table_depth += 1;
            }
            Event::Start(Tag::HtmlBlock) => {
                // Emit the raw HTML block verbatim as one `"html"` block (B3). The
                // `Start..End` range covers the whole block, so a single slice is
                // byte-identical; the inner `Event::Html` text events that follow
                // fall through the `_ => {}` arm and are not re-emitted.
                let path = heading_stack.current();
                emit_block(
                    src,
                    BlockType::Html.as_str(),
                    &path,
                    range.start,
                    range.end,
                    &mut blocks,
                );
            }
            Event::Start(Tag::Heading { .. }) => {
                let path = heading_stack.current();
                ctx_stack.push((BlockContext::Heading, range.start, path));
            }

            // ── Closing tags ──────────────────────────────────────────────
            Event::End(TagEnd::Paragraph) => {
                if let Some((BlockContext::Paragraph, byte_start, path)) = ctx_stack.pop() {
                    let byte_end = range.end;
                    emit_block(
                        src,
                        BlockType::Paragraph.as_str(),
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
                        BlockType::Code.as_str(),
                        &path,
                        byte_start,
                        byte_end,
                        &mut blocks,
                    );
                }
            }
            Event::End(TagEnd::Item) => {
                if let Some((BlockContext::ListItem { emitted }, byte_start, path)) =
                    ctx_stack.pop()
                {
                    // A tight leaf item (no block-level child) emits its full span
                    // here; an item that had a block-level child already emitted
                    // its leading own-text clamped to that child (B1/B2), so skip
                    // it. Drop a bare-marker span (B5).
                    if !emitted && byte_start < range.end {
                        let raw = &src[byte_start..range.end];
                        if !is_blank_or_marker(raw.trim()) {
                            emit_block(
                                src,
                                BlockType::ListItem.as_str(),
                                &path,
                                byte_start,
                                range.end,
                                &mut blocks,
                            );
                        }
                    }
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

                    // Update the heading stack: record this heading at its level
                    // and clear all deeper levels (handled by `push`).
                    heading_stack.push(heading_level_u8(level), heading_text);

                    // Emit a "heading" block. `text` is the trimmed heading
                    // text; `char_start/char_end` anchor to that text inside src.
                    let (start, end) = locate_in_src(src, heading_text, byte_start, byte_end);
                    if end > start {
                        blocks.push(Block {
                            block_type: BlockType::Heading.as_str().to_string(),
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

/// If the innermost open context is a list item that has not yet emitted its
/// own-text, emit the item's leading own-text — the span from the item start to
/// `child_start` (the byte at which its first block-level child begins) — and
/// mark the item emitted. The leading slice is dropped when it is empty or only
/// a list marker (`-`, `*`, `+`, or `N.` / `N)`) after trimming (B5), so an item
/// whose only content is a nested list never produces a bare-marker block.
///
/// This is the single clamp point for list-item own-text: every block-level
/// child (paragraph, code, nested list, table, html block) routes through here
/// before being handled, so the item span never re-includes a block child —
/// fixing loose-list duplication and table-in-item double-emit.
fn emit_item_own_text_before_child(
    src: &str,
    ctx_stack: &mut [(BlockContext, usize, String)],
    child_start: usize,
    blocks: &mut Vec<Block>,
) {
    if let Some((BlockContext::ListItem { emitted }, byte_start, path)) = ctx_stack.last_mut()
        && !*emitted
    {
        *emitted = true;
        let byte_start = *byte_start;
        let path = path.clone();
        if byte_start < child_start {
            let raw = &src[byte_start..child_start];
            if !is_blank_or_marker(raw.trim()) {
                emit_block(
                    src,
                    BlockType::ListItem.as_str(),
                    &path,
                    byte_start,
                    child_start,
                    blocks,
                );
            }
        }
    }
}

/// Returns `true` when `s` is empty or consists solely of a single leading list
/// marker (`-`, `*`, `+`, or an ordered marker like `1.` / `2)`) with no real
/// own-text — used to skip bare-marker `list_item` artifacts (B5).
fn is_blank_or_marker(s: &str) -> bool {
    if s.is_empty() {
        return true;
    }
    // Bullet markers.
    if matches!(s, "-" | "*" | "+") {
        return true;
    }
    // Ordered markers: digits followed by a single `.` or `)`.
    let mut chars = s.chars();
    let mut saw_digit = false;
    for c in chars.by_ref() {
        if c.is_ascii_digit() {
            saw_digit = true;
        } else {
            return saw_digit && (c == '.' || c == ')') && chars.next().is_none();
        }
    }
    // All digits, no terminator → not a marker on its own.
    false
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
/// `"My Heading"`).
///
/// Setext headings (a title line followed by a `===`/`---` underline) are
/// detected and narrowed to the title line only — the underline (and the
/// newline before it) is excluded (B3). Returning the title sub-slice keeps
/// byte-identity exact: `locate_in_src` finds it as a contiguous span of `src`,
/// so the heading block's `char_start..char_end` covers exactly the title text
/// and the polluted underline never propagates into `section_path`.
///
/// An ATX *closing* `#` sequence is only valid when preceded by whitespace
/// (CommonMark §4.2): `## A #` closes to `"A"`, but `# C#` is NOT a closing
/// marker and must stay `"C#"`. We therefore only strip a trailing `#` run when
/// it is preceded by whitespace (or is the entire remaining content).
fn extract_heading_text(raw: &str) -> &str {
    // Setext: the raw range is `title\n  underline\n` where `underline` is a run
    // of `=` or `-`. Narrow to the title line (everything before the underline).
    if let Some(title) = setext_title(raw) {
        return title;
    }

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

/// If `raw` is a setext heading (`title` line followed by a `===` or `---`
/// underline line), returns the trimmed title line. Returns `None` for ATX
/// headings or anything that is not a recognisable setext shape.
///
/// The returned slice is a contiguous sub-slice of `raw` (hence of `src`), so
/// the caller's `locate_in_src` recovers an exact byte span — preserving
/// byte-identity for the narrowed heading block.
fn setext_title(raw: &str) -> Option<&str> {
    // Collect non-empty lines with their trimmed forms.
    let mut lines: Vec<&str> = raw.lines().filter(|l| !l.trim().is_empty()).collect();
    // Need at least a title and an underline.
    if lines.len() < 2 {
        return None;
    }
    let underline = lines.pop()?.trim();
    if underline.is_empty() {
        return None;
    }
    let is_setext_underline = underline.chars().all(|c| c == '=')
        || (underline.chars().all(|c| c == '-') && !underline.is_empty());
    if !is_setext_underline {
        return None;
    }
    // The title is the first remaining content line, trimmed. A setext title is a
    // single line in CommonMark; use the first non-empty line.
    let title = lines.first()?.trim();
    if title.is_empty() { None } else { Some(title) }
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
        assert_eq!(SourceKind::from_kind_str("pdf").unwrap(), SourceKind::Pdf);
        assert_eq!(SourceKind::from_kind_str("docx").unwrap(), SourceKind::Docx);
        assert_eq!(SourceKind::from_kind_str("url").unwrap(), SourceKind::Url);
        assert!(SourceKind::from_kind_str("nonsense").is_err());
    }

    #[test]
    fn source_kind_roundtrip_and_wire_strings() {
        use std::str::FromStr;
        // Lock the EXACT persisted wire strings (no DB migration).
        let cases = [
            (SourceKind::Text, "text"),
            (SourceKind::Markdown, "markdown"),
            (SourceKind::Pdf, "pdf"),
            (SourceKind::Docx, "docx"),
            (SourceKind::Url, "url"),
        ];
        for (kind, s) in cases {
            assert_eq!(kind.as_str(), s, "as_str must equal legacy wire string");
            assert_eq!(
                SourceKind::from_str(s).unwrap(),
                kind,
                "FromStr round-trips as_str"
            );
        }
    }

    #[test]
    fn source_kind_from_str_rejects_unknown() {
        let err = SourceKind::from_str("nope").expect_err("unknown kind must error");
        assert!(matches!(err, LensError::Validation(_)));
    }

    #[test]
    fn source_kind_is_text_like() {
        assert!(SourceKind::Text.is_text_like());
        assert!(SourceKind::Markdown.is_text_like());
        assert!(!SourceKind::Pdf.is_text_like());
        assert!(!SourceKind::Docx.is_text_like());
        assert!(!SourceKind::Url.is_text_like());
    }

    #[test]
    fn block_type_roundtrip_and_wire_strings() {
        use std::str::FromStr;
        // Lock the EXACT persisted `chunks.block_type` strings.
        let cases = [
            (BlockType::Heading, "heading"),
            (BlockType::Paragraph, "paragraph"),
            (BlockType::Code, "code"),
            (BlockType::ListItem, "list_item"),
            (BlockType::Table, "table"),
            (BlockType::Html, "html"),
        ];
        for (bt, s) in cases {
            assert_eq!(bt.as_str(), s, "as_str must equal legacy block-type string");
            assert_eq!(
                BlockType::from_str(s).unwrap(),
                bt,
                "FromStr round-trips as_str"
            );
        }
    }

    #[test]
    fn block_type_from_str_rejects_unknown() {
        use std::str::FromStr;
        let err = BlockType::from_str("bogus").expect_err("unknown block type must error");
        assert!(matches!(err, LensError::Validation(_)));
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

    /// B1 — GFM table content must not be silently dropped. A 2-column table is
    /// emitted as exactly ONE `"table"` block whose raw markdown contains every
    /// cell's text. Demonstrates the content-loss bug on the old `_ => {}` arm.
    #[test]
    fn markdown_gfm_table_emitted_as_one_block() {
        let src = "# T\n\n\
                   | Fruit | Color |\n\
                   | ----- | ----- |\n\
                   | Apple | Red |\n\
                   | Lime | Green |\n\n\
                   After table.\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        assert_byte_identity(src, &blocks);

        let tables: Vec<&Block> = blocks.iter().filter(|b| b.block_type == "table").collect();
        assert_eq!(tables.len(), 1, "exactly one table block");
        let t = tables[0];
        // Every cell's content survives in the single table block.
        for cell in ["Fruit", "Color", "Apple", "Red", "Lime", "Green"] {
            assert!(
                t.text.contains(cell),
                "table block lost cell {cell:?}; text = {:?}",
                t.text
            );
        }
        // section_path is the active heading trail.
        assert_eq!(t.section_path, "T");
        // The table events must not pollute other arms: no cell text leaks into a
        // paragraph/list_item block.
        for b in &blocks {
            if b.block_type != "table" {
                for cell in ["Fruit", "Apple", "Lime", "Green"] {
                    assert!(
                        !b.text.contains(cell),
                        "cell {cell:?} leaked into a {} block: {:?}",
                        b.block_type,
                        b.text
                    );
                }
            }
        }
    }

    /// B2 — nested list items must each emit their OWN text only, in document
    /// order, with no duplication. Demonstrates the outer-includes-inner +
    /// out-of-order double-emit bug on the old per-Item slice.
    #[test]
    fn markdown_nested_list_no_duplication_in_order() {
        let src = "- outer\n  - inner\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        assert_byte_identity(src, &blocks);

        let items: Vec<&Block> = blocks
            .iter()
            .filter(|b| b.block_type == "list_item")
            .collect();
        assert_eq!(items.len(), 2, "exactly two list_item blocks");

        // "inner" appears in EXACTLY one block (the inner one).
        let inner_count = items.iter().filter(|b| b.text.contains("inner")).count();
        assert_eq!(inner_count, 1, "'inner' must appear in exactly one block");

        // The outer block must NOT contain the inner text (no duplication).
        let outer = items
            .iter()
            .find(|b| b.text.contains("outer"))
            .expect("an outer block exists");
        assert!(
            !outer.text.contains("inner"),
            "outer list_item must not include nested 'inner' text; got {:?}",
            outer.text
        );

        // Document order: the outer block precedes the inner block.
        let outer_idx = items.iter().position(|b| b.text.contains("outer")).unwrap();
        let inner_idx = items.iter().position(|b| b.text.contains("inner")).unwrap();
        assert!(
            outer_idx < inner_idx,
            "outer list_item must come before inner in document order"
        );
    }

    /// B3 — a raw HTML block is captured verbatim as one `"html"` block.
    #[test]
    fn markdown_html_block_captured() {
        let src = "# H\n\n<div class=\"note\">\n  <p>Raw HTML body.</p>\n</div>\n\nAfter.\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        assert_byte_identity(src, &blocks);

        let html: Vec<&Block> = blocks.iter().filter(|b| b.block_type == "html").collect();
        assert_eq!(html.len(), 1, "exactly one html block");
        assert!(html[0].text.contains("<div"), "html block keeps the markup");
        assert!(
            html[0].text.contains("Raw HTML body."),
            "html block keeps inner content"
        );
        assert_eq!(html[0].section_path, "H");
    }

    /// B2-loose — a loose list (blank line between items) wraps each item's
    /// own-text in a `Paragraph`. The `Paragraph` arm already emits that text, so
    /// the item must NOT re-emit a spanning slice at the nested-list start or at
    /// `End(Item)`. Repro: "a" must appear in exactly one block, not two.
    #[test]
    fn markdown_loose_list_item_own_text_not_duplicated() {
        let src = "- a\n\n  - b\n\n  trailing\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        assert_byte_identity(src, &blocks);

        // The loose item's own-text "a" must appear in exactly one block. The bug
        // emits it twice: once as the wrapped paragraph "a" and again as the
        // spanning re-slice "- a" at the nested-list start.
        let a_count = blocks
            .iter()
            .filter(|b| b.text == "a" || b.text == "- a")
            .count();
        assert_eq!(
            a_count, 1,
            "loose item own-text duplicated across blocks: {:#?}",
            blocks
        );

        // No block may span both the item's own-text and the nested item text.
        for b in &blocks {
            assert!(
                !(b.text.contains("a") && b.text.contains("b") && b.text.contains("trailing")),
                "a block spans the whole loose item (duplication): {:?}",
                b.text
            );
        }
    }

    /// B2-table — a list item containing a table must emit the table exactly once
    /// (via the `Table` arm), and the item own-text must clamp to the table start
    /// so `End(Item)` does not re-emit the whole item span (which would duplicate
    /// the raw table markdown).
    #[test]
    fn markdown_table_in_list_item_not_double_emitted() {
        let src = "- item\n\n  | A | B |\n  | - | - |\n  | 1 | 2 |\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        assert_byte_identity(src, &blocks);

        let tables: Vec<&Block> = blocks.iter().filter(|b| b.block_type == "table").collect();
        assert_eq!(
            tables.len(),
            1,
            "exactly one table block; got {:#?}",
            blocks
        );

        // Cell content must live only in the table block, never in a list_item.
        for b in &blocks {
            if b.block_type != "table" {
                for cell in ["A", "B", "1", "2"] {
                    // Allow incidental single-char overlap only outside table cells:
                    // the bug manifests as the FULL table markdown (pipes) leaking
                    // into the item span, so assert no pipe-delimited row leaks.
                    let _ = cell;
                }
                assert!(
                    !b.text.contains("| 1 | 2 |"),
                    "table markdown leaked into a {} block: {:?}",
                    b.block_type,
                    b.text
                );
            }
        }
    }

    /// B5 — an item whose only own-text is whitespace plus a list marker must NOT
    /// emit a bare "-" `list_item` block.
    #[test]
    fn markdown_bare_marker_own_text_skipped() {
        let src = "- \n  - x\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        assert_byte_identity(src, &blocks);

        for b in &blocks {
            assert_ne!(
                b.text, "-",
                "emitted a bare list-marker artifact block: {:#?}",
                blocks
            );
        }
        // "x" (the only real content) must still be present exactly once.
        let x_count = blocks.iter().filter(|b| b.text.contains('x')).count();
        assert_eq!(x_count, 1, "nested 'x' should appear exactly once");
    }

    /// B3-setext — a setext heading's text must be the title line only (the
    /// `===`/`---` underline must be excluded), and that clean text must propagate
    /// into descendants' section_path. Byte-identity must still hold.
    #[test]
    fn markdown_setext_heading_clean_text_and_section_path() {
        let src = "My Heading\n==========\n\nbody\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        assert_byte_identity(src, &blocks);

        let heading = blocks
            .iter()
            .find(|b| b.block_type == "heading")
            .expect("setext heading should exist");
        assert_eq!(
            heading.text, "My Heading",
            "setext heading text must exclude the underline"
        );

        // No newline or '=' may appear in any heading text or section_path.
        for b in &blocks {
            if b.block_type == "heading" {
                assert!(
                    !b.text.contains('\n') && !b.text.contains('='),
                    "heading text polluted: {:?}",
                    b.text
                );
            }
            assert!(
                !b.section_path.contains('\n') && !b.section_path.contains('='),
                "section_path polluted for {:?}: {:?}",
                b.block_type,
                b.section_path
            );
        }

        let body = blocks
            .iter()
            .find(|b| b.block_type == "paragraph" && b.text == "body")
            .expect("body paragraph should exist");
        assert_eq!(body.section_path, "My Heading");
    }

    /// Setext H2 (`---` underline) must be handled identically.
    #[test]
    fn markdown_setext_h2_clean_text() {
        let src = "Subtitle\n--------\n\nbody\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        assert_byte_identity(src, &blocks);
        let heading = blocks
            .iter()
            .find(|b| b.block_type == "heading")
            .expect("setext h2 should exist");
        assert_eq!(heading.text, "Subtitle");
    }

    /// Deeper nesting still emits one block per item, no duplication, in order.
    #[test]
    fn markdown_nested_list_three_levels() {
        let src = "- a\n  - b\n    - c\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        assert_byte_identity(src, &blocks);

        let items: Vec<&Block> = blocks
            .iter()
            .filter(|b| b.block_type == "list_item")
            .collect();
        assert_eq!(items.len(), 3, "one block per item");
        // Each item's own block ends with its marker letter and does NOT contain
        // any deeper letter (no parent includes its children's text).
        let block_for = |letter: char| -> &Block {
            items
                .iter()
                .copied()
                .find(|b| b.text.trim_end().ends_with(letter))
                .unwrap_or_else(|| panic!("a block ending in {letter:?}"))
        };
        let a = block_for('a');
        let b = block_for('b');
        let c = block_for('c');
        assert!(
            !a.text.contains('b') && !a.text.contains('c'),
            "a excludes b,c"
        );
        assert!(!b.text.contains('c'), "b excludes c");
        // Document order a < b < c.
        let pos = |target: &Block| items.iter().position(|x| std::ptr::eq(*x, target)).unwrap();
        assert!(pos(a) < pos(b) && pos(b) < pos(c), "document order a<b<c");
    }
}
