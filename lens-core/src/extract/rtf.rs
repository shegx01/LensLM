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

fn decode_unicode_escapes(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    let mut uc: usize = 1;
    while i < b.len() {
        if b[i] == b'\\' {
            if b[i + 1..].starts_with(b"uc") && b.get(i + 3).is_some_and(u8::is_ascii_digit) {
                let mut j = i + 3;
                let mut n = 0usize;
                while j < b.len() && b[j].is_ascii_digit() {
                    n = n.saturating_mul(10).saturating_add((b[j] - b'0') as usize);
                    j += 1;
                }
                uc = n;
                if b.get(j) == Some(&b' ') {
                    j += 1;
                }
                i = j;
                continue;
            }
            if b.get(i + 1) == Some(&b'u')
                && b.get(i + 2)
                    .is_some_and(|c| c.is_ascii_digit() || *c == b'-')
            {
                let mut j = i + 2;
                let neg = b[j] == b'-';
                if neg {
                    j += 1;
                }
                let mut n: i64 = 0;
                let digits_start = j;
                while j < b.len() && b[j].is_ascii_digit() {
                    n = n.saturating_mul(10).saturating_add((b[j] - b'0') as i64);
                    j += 1;
                }
                if j > digits_start {
                    let code = if neg { 65536 - n } else { n };
                    if let Some(ch) = u32::try_from(code).ok().and_then(char::from_u32) {
                        match ch {
                            '\\' => out.push_str("\\\\"),
                            '{' => out.push_str("\\{"),
                            '}' => out.push_str("\\}"),
                            _ => out.push(ch),
                        }
                    }
                }
                if b.get(j) == Some(&b' ') {
                    j += 1;
                }
                let mut skipped = 0;
                while skipped < uc && j < b.len() {
                    if b[j] == b'\\'
                        && b.get(j + 1) == Some(&b'\'')
                        && j + 3 < b.len()
                        && b[j + 2].is_ascii_hexdigit()
                        && b[j + 3].is_ascii_hexdigit()
                    {
                        j += 4;
                    } else {
                        match s[j..].chars().next() {
                            Some(c) => j += c.len_utf8(),
                            None => break,
                        }
                    }
                    skipped += 1;
                }
                i = j;
                continue;
            }
        }
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
pub struct RtfExtractor;

impl Extractor for RtfExtractor {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError> {
        let s = std::str::from_utf8(raw)
            .map_err(|_| LensError::Validation("source is not valid UTF-8".to_string()))?;

        let prepared = par_to_line(&decode_unicode_escapes(s));

        let doc = RtfDocument::try_from(prepared.as_str())
            .map_err(|e| LensError::Parse(format!("rtf-parser failed to parse RTF: {e:?}")))?;

        let full: String = doc.body.iter().map(|sb| sb.text.as_str()).collect();

        let mut extracted_text = String::new();
        let mut blocks: Vec<Block> = Vec::new();
        let mut anchors: Vec<SourceAnchor> = Vec::new();

        for line in full.split('\n') {
            let text = line.trim();
            if text.is_empty() {
                continue;
            }

            let char_start = extracted_text.len();
            extracted_text.push_str(text);
            extracted_text.push('\n');
            let char_end = extracted_text.len() - 1; // trailing \n excluded

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
    fn rtf_unicode_escape_is_decoded() {
        let out = extract(r"{\rtf1\ansi\deff0 Caf\u233? au lait.\par}");
        assert!(
            out.extracted_text.contains("Café au lait."),
            "\\u233? must decode to é with its single fallback char skipped; got {:?}",
            out.extracted_text
        );
    }

    #[test]
    fn rtf_unicode_escape_decodes_cjk() {
        let bs = '\\';
        let rtf = format!("{{{bs}rtf1{bs}ansi{bs}deff0 {bs}u26085?{bs}u26412?{bs}u35486?{bs}par}}");
        let out = extract(&rtf);
        assert!(
            out.extracted_text.contains("日本語"),
            "consecutive unicode escapes must decode to CJK; got {:?}",
            out.extracted_text
        );
    }

    #[test]
    fn rtf_unicode_escape_consumes_delimiter_space() {
        let out = extract(r"{\rtf1\ansi\deff0 A\u233 ?B\par}");
        assert!(
            out.extracted_text.contains("AéB"),
            "the RTF delimiter space after the numeric arg must not leak into output; got {:?}",
            out.extracted_text
        );
    }

    #[test]
    fn rtf_unicode_escape_honors_uc_count() {
        let out = extract(r"{\rtf1\ansi\deff0 \uc2 A\u233 xxB\par}");
        assert!(
            out.extracted_text.contains("AéB"),
            "\\uc2 must skip two fallback chars after the escape; got {:?}",
            out.extracted_text
        );
    }

    #[test]
    fn rtf_unicode_escape_negative_bmp_codepoint() {
        let out = extract(r"{\rtf1\ansi\deff0 \u-3585?\par}");
        assert!(
            out.extracted_text.contains('\u{F1FF}'),
            "a negative \\u value must decode as its codepoint + 65536; got {:?}",
            out.extracted_text
        );
    }

    #[test]
    fn rtf_unicode_escape_uc0_skips_no_fallback() {
        let out = extract(r"{\rtf1\ansi\deff0 \uc0 A\u233 B\par}");
        assert!(
            out.extracted_text.contains("AéB"),
            "\\uc0 must skip zero fallback chars; got {:?}",
            out.extracted_text
        );
    }

    #[test]
    fn rtf_unicode_escape_hex_fallback_char() {
        let bs = '\\';
        let rtf = format!("{{{bs}rtf1{bs}ansi{bs}deff0 A{bs}u233 {bs}'e9B{bs}par}}");
        let out = extract(&rtf);
        assert!(
            out.extracted_text.contains("AéB"),
            "a \\'hh hex fallback char must count as one skip; got {:?}",
            out.extracted_text
        );
    }

    #[test]
    fn rtf_unicode_escape_non_ascii_after_hex_marker_does_not_panic() {
        let bs = '\\';
        let rtf = format!("{{{bs}rtf1{bs}ansi{bs}deff0 A{bs}u233 {bs}'€B{bs}par}}");
        let out = extract(&rtf);
        assert!(
            out.extracted_text.contains('é'),
            "malformed non-ASCII after \\' must not panic and still decode the escape; got {:?}",
            out.extracted_text
        );
    }

    #[test]
    fn rtf_unicode_escape_decoded_backslash_survives_as_literal() {
        let bs = '\\';
        let rtf = format!("{{{bs}rtf1{bs}ansi{bs}deff0 A{bs}u92?B{bs}par}}");
        let out = extract(&rtf);
        assert!(
            out.extracted_text.contains("A\\B"),
            "a decoded backslash must survive as literal text, not corrupt parsing; got {:?}",
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
