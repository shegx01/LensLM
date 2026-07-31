//! Document outline extraction (#279).
//!
//! [`build_sections`] derives a per-source outline (the `sections` table) from the
//! flat [`Block`] list every extractor already produces. It relies on the shared
//! convention that a heading block's `section_path` includes itself (see
//! [`crate::parse::SectionPathStack`]): a heading's **level** is the segment count
//! of its trail, its **title** is the heading text, and its span runs until the next
//! heading of the same-or-shallower level. Structure-aware retrieval resolves
//! positional queries ("summary of chapter 2") against these ordinals instead of
//! matching the lossy heading string.

use crate::parse::{Block, BlockType};

/// One outline entry, ready to become a `sections` row (the caller adds
/// `id`/`source_id`/`created_at`). `char_start`/`char_end` are byte offsets in the
/// same coordinate space as the source's chunks, so a section's chunks are the ones
/// whose `char_start` falls in `[char_start, char_end)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    /// Heading depth, 1..=6.
    pub level: u8,
    /// 1-based position among same-level siblings under the same parent, in
    /// document order (top-level ordinal is the "chapter number").
    pub ordinal: u32,
    /// Heading text.
    pub title: String,
    /// Byte offset where the section starts (its heading block).
    pub char_start: usize,
    /// Byte offset one-past the section's last content (next same-or-shallower
    /// heading, or document end).
    pub char_end: usize,
}

/// Builds the outline from an extractor's ordered block list. Returns an empty vec
/// when the document has no headings (plain text, data formats) — such sources fall
/// back to normal retrieval.
pub(crate) fn build_sections(blocks: &[Block]) -> Vec<Section> {
    let heading = BlockType::Heading.as_str();
    let doc_end = blocks.iter().map(|b| b.char_end).max().unwrap_or(0);

    // First pass: heading level (trail segment count) + title + start, in order.
    let mut raw: Vec<(u8, String, usize)> = Vec::new();
    for b in blocks {
        if b.block_type != heading || b.section_path.is_empty() {
            continue;
        }
        let level = (b.section_path.split(" > ").count() as u8).clamp(1, 6);
        raw.push((level, b.text.clone(), b.char_start));
    }

    // Second pass: sibling ordinals (a shallower heading resets deeper counters) and
    // the char span (next same-or-shallower heading start, else document end).
    let mut counters = [0u32; 7];
    let mut out = Vec::with_capacity(raw.len());
    for (i, (level, title, char_start)) in raw.iter().enumerate() {
        let lvl = *level as usize;
        counters[lvl] += 1;
        for deeper in counters.iter_mut().skip(lvl + 1) {
            *deeper = 0;
        }
        let char_end = raw[i + 1..]
            .iter()
            .find(|(l, _, _)| *l <= *level)
            .map(|(_, _, start)| *start)
            .unwrap_or(doc_end);
        out.push(Section {
            level: *level,
            ordinal: counters[lvl],
            title: title.clone(),
            char_start: *char_start,
            char_end,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heading(trail: &str, title: &str, start: usize) -> Block {
        Block {
            block_type: BlockType::Heading.as_str().to_string(),
            section_path: trail.to_string(),
            text: title.to_string(),
            char_start: start,
            char_end: start + title.len(),
        }
    }

    fn para(trail: &str, text: &str, start: usize) -> Block {
        Block {
            block_type: BlockType::Paragraph.as_str().to_string(),
            section_path: trail.to_string(),
            text: text.to_string(),
            char_start: start,
            char_end: start + text.len(),
        }
    }

    #[test]
    fn no_headings_yields_no_sections() {
        let blocks = vec![para("", "just prose", 0), para("", "more", 20)];
        assert!(build_sections(&blocks).is_empty());
    }

    #[test]
    fn flat_top_level_headings_number_sequentially() {
        let blocks = vec![
            heading("Chapter 1", "Chapter 1", 0),
            para("Chapter 1", "intro", 10),
            heading("Chapter 2", "Chapter 2", 100),
            para("Chapter 2", "body", 110),
        ];
        let s = build_sections(&blocks);
        assert_eq!(s.len(), 2);
        assert_eq!((s[0].level, s[0].ordinal, s[0].title.as_str()), (1, 1, "Chapter 1"));
        assert_eq!((s[1].level, s[1].ordinal, s[1].title.as_str()), (1, 2, "Chapter 2"));
        // Section 1 ends where section 2 begins; section 2 runs to document end.
        assert_eq!(s[0].char_start, 0);
        assert_eq!(s[0].char_end, 100);
        assert_eq!(s[1].char_start, 100);
        assert_eq!(s[1].char_end, 114);
    }

    #[test]
    fn nested_headings_get_per_parent_ordinals_and_bounded_spans() {
        let blocks = vec![
            heading("A", "A", 0),
            heading("A > B", "B", 10),
            heading("A > C", "C", 20),
            heading("D", "D", 30),
            heading("D > E", "E", 40),
        ];
        let s = build_sections(&blocks);
        // A(l1,o1), B(l2,o1), C(l2,o2), D(l1,o2), E(l2,o1 — resets under new parent D)
        let got: Vec<(u8, u32, &str)> =
            s.iter().map(|x| (x.level, x.ordinal, x.title.as_str())).collect();
        assert_eq!(
            got,
            vec![(1, 1, "A"), (2, 1, "B"), (2, 2, "C"), (1, 2, "D"), (2, 1, "E")]
        );
        // A spans to the next level<=1 heading (D@30); B spans to C@20; D spans to doc end.
        assert_eq!(s[0].char_end, 30);
        assert_eq!(s[1].char_end, 20);
        assert_eq!(s[3].char_end, 41);
    }

    #[test]
    fn heading_without_a_trail_is_ignored() {
        // A malformed heading block with empty section_path contributes nothing.
        let blocks = vec![heading("", "orphan", 0), heading("Real", "Real", 10)];
        let s = build_sections(&blocks);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].title, "Real");
    }
}
