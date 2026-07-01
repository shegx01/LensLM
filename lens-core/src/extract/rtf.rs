//! RTF extractor (M4 issue #77).
//!
//! [`RtfExtractor`] parses a `.rtf` document from raw bytes using `rtf-parser`
//! and produces a canonical [`ExtractOutput`] where:
//! - `extracted_text` is the linearised UTF-8 text of all non-empty paragraphs
//!   (one `\n` separator between consecutive blocks).
//! - `blocks` carries one [`Block`] per non-empty paragraph. RTF has NO semantic
//!   heading tags (`\heading1` does not exist — heading styles are an
//!   application-level font/bold convention), so every block is
//!   `block_type = "paragraph"` with an empty `section_path`. This is a
//!   documented gap (issue #77 spec non-goal: "paragraphs only").
//! - `anchors` carries one [`SourceAnchor::Rtf { text_offset }`] per block, where
//!   `text_offset` is the block's byte start in the canonical buffer.
//!
//! ## API deviation from the plan
//!
//! The plan assumed `RtfDocument::body` yields one `StyleBlock` per paragraph.
//! In reality `rtf-parser 0.4.3` groups text into a new `StyleBlock` ONLY when
//! the painter or paragraph *style* changes — the `\par` control word emits no
//! text and no style change, so consecutive same-styled paragraphs collapse into
//! a single `StyleBlock`. To recover paragraph boundaries we run a tiny pre-pass
//! that rewrites the `\par` control word to `\line` (which the parser turns into
//! a `\n` inside the block text), then split the concatenated body text on `\n`.

use rtf_parser::RtfDocument;

use crate::LensError;
use crate::parse::{Block, BlockType};

use super::{ExtractOutput, Extractor, SourceAnchor};

/// Rewrites the `\par` RTF control word to `\line` so paragraph breaks survive
/// as `\n` in the parsed block text.
///
/// Matches `\par` ONLY when terminated by a non-letter (space, `\`, `{`, `}`,
/// digit, or end-of-input) so longer control words such as `\pard`,
/// `\pararsid`, or `\pardeftab` are left untouched. RTF control words are ASCII,
/// so byte-level scanning is safe; non-ASCII bytes are copied through verbatim.
fn par_to_line(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\' && b[i + 1..].starts_with(b"par") {
            let after = b.get(i + 4).copied();
            let is_par = match after {
                Some(c) => !c.is_ascii_alphabetic(),
                None => true,
            };
            if is_par {
                out.push_str("\\line");
                i += 4; // consumed `\par`
                continue;
            }
        }
        // Copy one UTF-8 char starting at byte `i` (control words are ASCII, but
        // plain text may be multibyte — step by the char's byte length). A
        // checked `next()` so a malformed/non-boundary index can never panic the
        // extractor — we simply stop scanning at the bad byte.
        let ch = match s[i..].chars().next() {
            Some(c) => c,
            None => break,
        };
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// RTF extractor — implements [`Extractor`] via `rtf-parser`.
///
/// Byte-identity offsets follow the DOCX build-as-you-go pattern (see docx.rs:204-207).
///
/// RTF has no semantic heading structure, so every emitted block is a
/// `"paragraph"` with an empty `section_path` (issue #77 spec non-goal).
pub struct RtfExtractor;

impl Extractor for RtfExtractor {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError> {
        // RTF is nominally ASCII with escapes, but `rtf-parser` takes a `&str`.
        let s = std::str::from_utf8(raw)
            .map_err(|_| LensError::Validation("source is not valid UTF-8".to_string()))?;

        // Pre-pass: recover paragraph boundaries (see module-level doc).
        let prepared = par_to_line(s);

        let doc = RtfDocument::try_from(prepared.as_str())
            .map_err(|e| LensError::Parse(format!("rtf-parser failed to parse RTF: {e:?}")))?;

        // Concatenate every StyleBlock's text in document order, then split on
        // `\n` (the paragraph separators we injected via `\line`) into blocks.
        let full: String = doc.body.iter().map(|sb| sb.text.as_str()).collect();

        let mut extracted_text = String::new();
        let mut blocks: Vec<Block> = Vec::new();
        let mut anchors: Vec<SourceAnchor> = Vec::new();

        for line in full.split('\n') {
            let text = line.trim();
            if text.is_empty() {
                continue; // skip blank paragraphs
            }

            // Build-as-you-go: offsets are correct by construction.
            let char_start = extracted_text.len();
            extracted_text.push_str(text);
            extracted_text.push('\n');
            let char_end = extracted_text.len() - 1; // exclude the trailing \n

            blocks.push(Block {
                block_type: BlockType::Paragraph.as_str().to_string(),
                section_path: String::new(),
                text: text.to_string(),
                char_start,
                char_end,
            });
            anchors.push(SourceAnchor::Rtf {
                text_offset: char_start as u64,
            });
        }

        // Trim trailing newlines (the last '\n' was never part of any block).
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

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(raw: &str) -> ExtractOutput {
        RtfExtractor
            .extract(raw.as_bytes())
            .expect("rtf extraction")
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

    // A minimal hand-crafted RTF with three paragraphs separated by `\par`.
    const SAMPLE_RTF: &str =
        r"{\rtf1\ansi\deff0 First paragraph.\par Second paragraph.\par Third paragraph.\par}";

    #[test]
    fn rtf_byte_identity() {
        let out = extract(SAMPLE_RTF);
        assert!(!out.blocks.is_empty(), "fixture must produce blocks");
        assert_byte_identity(&out);
    }

    #[test]
    fn rtf_splits_paragraphs_on_par() {
        let out = extract(SAMPLE_RTF);
        assert_eq!(out.blocks.len(), 3, "three \\par-separated paragraphs");
        assert_eq!(out.blocks[0].text, "First paragraph.");
        assert_eq!(out.blocks[1].text, "Second paragraph.");
        assert_eq!(out.blocks[2].text, "Third paragraph.");
    }

    #[test]
    fn rtf_paragraph_blocks() {
        let out = extract(SAMPLE_RTF);
        for b in &out.blocks {
            assert_eq!(
                b.block_type,
                BlockType::Paragraph.as_str(),
                "all RTF blocks are paragraphs"
            );
            assert_eq!(b.section_path, "", "RTF has no heading trail");
        }
    }

    #[test]
    fn rtf_anchors_index_aligned() {
        let out = extract(SAMPLE_RTF);
        assert_eq!(out.anchors.len(), out.blocks.len());
        for (i, (a, b)) in out.anchors.iter().zip(&out.blocks).enumerate() {
            let SourceAnchor::Rtf { text_offset } = a else {
                panic!("anchor[{i}] must be SourceAnchor::Rtf");
            };
            assert_eq!(
                *text_offset, b.char_start as u64,
                "anchor offset == block start"
            );
        }
    }

    #[test]
    fn rtf_invalid_bytes_returns_error() {
        let err = RtfExtractor
            .extract(b"not an rtf file at all")
            .expect_err("invalid bytes must error");
        assert!(matches!(err, LensError::Parse(_)), "got {err:?}");
    }

    #[test]
    fn rtf_invalid_utf8_returns_validation_error() {
        let err = RtfExtractor
            .extract(&[0xFF, 0xFE, 0x00])
            .expect_err("invalid UTF-8 must error");
        assert!(
            matches!(&err, LensError::Validation(m) if m == "source is not valid UTF-8"),
            "got {err:?}"
        );
    }

    #[test]
    fn rtf_empty_document() {
        let out = extract(r"{\rtf1\ansi\deff0}");
        assert!(out.blocks.is_empty(), "empty RTF yields no blocks");
        assert!(out.extracted_text.is_empty());
    }

    #[test]
    fn rtf_pard_not_rewritten() {
        // `\pard` and `\pardeftab` must NOT be treated as `\par`; the document
        // must still parse and split correctly on the real `\par`.
        let out = extract(r"{\rtf1\ansi\deff0\pard\pardeftab720 Alpha.\par Beta.\par}");
        assert_byte_identity(&out);
        assert_eq!(out.blocks.len(), 2);
        assert_eq!(out.blocks[0].text, "Alpha.");
        assert_eq!(out.blocks[1].text, "Beta.");
    }

    #[test]
    fn rtf_blank_paragraphs_skipped() {
        // Two consecutive `\par` produce a blank paragraph that must be skipped.
        let out = extract(r"{\rtf1\ansi\deff0 One.\par\par Two.\par}");
        assert_eq!(out.blocks.len(), 2);
        assert_eq!(out.blocks[0].text, "One.");
        assert_eq!(out.blocks[1].text, "Two.");
        assert_byte_identity(&out);
    }

    #[test]
    fn rtf_multibyte_text_byte_identity() {
        // Literal multibyte UTF-8 content (€ and CJK) must survive verbatim with
        // byte-identity offsets intact.
        let out = extract(r"{\rtf1\ansi\deff0 Price is €5 here.\par More 日本語 text.\par}");
        assert_byte_identity(&out);
        assert!(
            out.extracted_text.contains('€') && out.extracted_text.contains("日本語"),
            "literal multibyte content must survive; got {:?}",
            out.extracted_text
        );
    }

    #[test]
    fn rtf_unicode_escape_documented_behavior() {
        // DOCUMENTED LIBRARY BEHAVIOR (rtf-parser 0.4.3): the `\uNNN?` Unicode
        // escape is NOT decoded — the parser DROPS it entirely (it does not emit
        // the substitute char `?` nor the U+NNNN code point). So `Caf\u233? au`
        // yields "Cafau" (note: the escape AND the space following the `?`
        // fallback are both consumed). This test pins that real behavior; if a
        // future rtf-parser upgrade decodes `\u233?` to 'é', this assertion will
        // fail and force a deliberate re-evaluation.
        let out = extract(r"{\rtf1\ansi\deff0 Caf\u233? au lait.\par}");
        assert!(
            !out.extracted_text.contains('\u{00E9}'),
            "rtf-parser 0.4.3 does NOT decode \\u233? to é; got {:?}",
            out.extracted_text
        );
        assert!(
            out.extracted_text.contains("Caf"),
            "the literal text around the dropped escape survives; got {:?}",
            out.extracted_text
        );
    }

    #[test]
    fn rtf_snapshot_block_structure() {
        let out = extract(SAMPLE_RTF);
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
        insta::assert_json_snapshot!("rtf_block_structure", snaps);
    }
}
