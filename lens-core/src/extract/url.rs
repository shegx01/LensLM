//! URL extractor: fetches HTML bytes (async, outside this trait), then
//! extracts content with `rs-trafilatura`.
//!
//! # Trait boundary
//!
//! [`Extractor::extract`] is sync `(&self, raw: &[u8]) -> Result<ExtractOutput>`.
//! The async `reqwest` GET therefore lives in `run_ingest` (in `ingest.rs`), not
//! here. This extractor receives the already-fetched HTML bytes.
//!
//! # Block model
//!
//! `rs_trafilatura` returns a flat `content_text` string — it does NOT expose a
//! structured block tree. We produce one block per non-empty line-group of the
//! extracted text, with `block_type = "paragraph"` and `section_path = ""`.
//! Each block's `text_offset` is the block's start byte index as a decimal
//! string (a stable, round-trippable pointer into the canonical text buffer).
//!
//! # needs_js detection
//!
//! If the extracted text is too short (absolute floor) or too small a fraction
//! of the raw HTML (ratio floor), the caller (`run_ingest`) handles the
//! `needs_js` status transition. This extractor is not aware of thresholds —
//! it simply returns the `ExtractOutput` (which may have empty or near-empty
//! `extracted_text`).

use crate::LensError;
use crate::parse::{Block, BlockType};

use super::{ExtractOutput, Extractor, SourceAnchor};

/// Extracts content from raw HTML bytes using `rs-trafilatura`.
///
/// Constructed once per ingest via [`super::extractor_for`] for `kind = "url"`.
/// The instance is stateless — `extract` is a pure function of the input bytes.
pub struct UrlExtractor;

impl Extractor for UrlExtractor {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError> {
        // `extract_bytes` handles character-encoding detection internally
        // (reads `<meta charset>` / `Content-Type` meta and transcodes to UTF-8).
        let result = rs_trafilatura::extract_bytes(raw)
            .map_err(|e| LensError::Validation(format!("URL content extraction failed: {e}")))?;

        let extracted_text = result.content_text;

        if extracted_text.is_empty() {
            return Ok(ExtractOutput {
                extracted_text,
                blocks: vec![],
                anchors: vec![],
            });
        }

        // Split the extracted flat text into paragraph blocks on double-newline
        // boundaries (trafilatura separates paragraphs with `\n\n`). Single
        // newlines within a paragraph are preserved verbatim — they may be
        // structural (list items). Blank-line groups become one block each.
        let mut blocks = Vec::new();
        let mut anchors = Vec::new();

        // Manual byte-offset splitter (not `str::split("\n\n")`): we must track
        // each block's exact byte `block_start`/`block_end` to set `char_start`/
        // `char_end` byte-identically into `extracted_text` (the byte-identity
        // invariant). `str::split` discards offsets, so we walk positions by hand.
        let mut pos = 0usize;
        let bytes = extracted_text.as_bytes();
        let len = bytes.len();

        while pos < len {
            // Skip leading newlines between blocks.
            while pos < len && bytes[pos] == b'\n' {
                pos += 1;
            }
            if pos == len {
                break;
            }
            let block_start = pos;

            // Find the end of this block: a double-newline or EOF.
            while pos < len {
                if pos + 1 < len && bytes[pos] == b'\n' && bytes[pos + 1] == b'\n' {
                    break;
                }
                pos += 1;
            }
            let block_end = pos;

            // The slice [block_start..block_end] is one paragraph.
            // Skip empty slices (shouldn't happen given leading-newline skip, but
            // be defensive).
            if block_start == block_end {
                continue;
            }

            let text = extracted_text[block_start..block_end].to_string();
            blocks.push(Block {
                block_type: BlockType::Paragraph.as_str().to_string(),
                section_path: String::new(),
                char_start: block_start,
                char_end: block_end,
                text,
            });
            // text_offset = decimal BYTE offset of the block's first character in
            // the extracted text buffer (a stable, round-trippable pointer).
            anchors.push(SourceAnchor::Url {
                text_offset: block_start.to_string(),
            });

            // Ensure we make forward progress when the split point lands on
            // exactly two newlines.
            if pos < len && bytes[pos] == b'\n' {
                pos += 1;
            }
        }

        Ok(ExtractOutput {
            extracted_text,
            blocks,
            anchors,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plain article HTML page with enough text to clear the needs_js threshold.
    const ARTICLE_HTML: &[u8] = br#"<!DOCTYPE html>
<html>
<head><title>Test Article</title></head>
<body>
<article>
<h1>Test Article Title</h1>
<p>This is the first paragraph of the article. It contains enough text to be extracted properly by trafilatura. The content is meaningful and represents real prose text.</p>
<p>This is the second paragraph. It also has substantial content that the extractor can work with. More text here to make it longer and more realistic as web content.</p>
</article>
</body>
</html>"#;

    /// A JS-shell page: minimal real text, all content behind JavaScript.
    const JS_SHELL_HTML: &[u8] = br#"<!DOCTYPE html>
<html>
<head><title>App</title></head>
<body>
<div id="root"></div>
<script>window.__INITIAL_STATE__ = {};</script>
<script src="/static/js/main.chunk.js"></script>
<script src="/static/js/2.chunk.js"></script>
<script src="/static/js/bundle.js"></script>
</body>
</html>"#;

    #[test]
    fn article_page_extracts_non_empty_text() {
        let extractor = UrlExtractor;
        let out = extractor
            .extract(ARTICLE_HTML)
            .expect("extraction should succeed");
        assert!(
            !out.extracted_text.is_empty(),
            "article page must produce non-empty extracted_text"
        );
        assert_eq!(
            out.blocks.len(),
            out.anchors.len(),
            "blocks and anchors must be index-aligned"
        );
        // Every anchor must be Url variant.
        for a in &out.anchors {
            assert!(
                matches!(a, SourceAnchor::Url { .. }),
                "URL extractor must produce SourceAnchor::Url anchors"
            );
        }
    }

    #[test]
    fn js_shell_extracts_empty_or_tiny_text() {
        let extractor = UrlExtractor;
        let out = extractor
            .extract(JS_SHELL_HTML)
            .expect("extraction should succeed");
        // trafilatura should find near-nothing in a pure JS shell.
        assert!(
            out.extracted_text.len() < 200,
            "JS shell should produce <200 chars (got {})",
            out.extracted_text.len()
        );
    }

    #[test]
    fn blocks_are_char_aligned() {
        let extractor = UrlExtractor;
        let out = extractor
            .extract(ARTICLE_HTML)
            .expect("extraction should succeed");
        for (i, block) in out.blocks.iter().enumerate() {
            assert_eq!(
                &out.extracted_text[block.char_start..block.char_end],
                &block.text,
                "byte-identity violated for block[{i}]"
            );
        }
    }

    #[test]
    fn empty_html_produces_empty_output() {
        let extractor = UrlExtractor;
        let out = extractor
            .extract(b"<html><body></body></html>")
            .expect("should not error");
        assert!(out.blocks.is_empty(), "empty page should produce no blocks");
        assert!(
            out.anchors.is_empty(),
            "empty page should produce no anchors"
        );
    }
}
