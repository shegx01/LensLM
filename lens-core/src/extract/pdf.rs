//! PDF extractor — `pdfium-render` binding that dynamically loads `libpdfium`.
//!
//! An image-only (no-text-layer) PDF returns empty `extracted_text` and no
//! blocks (not `Err`) so `run_ingest` can set `needs_ocr` via the Ok-with-status
//! mechanism without flipping the source to `error`.
//!
//! The `libpdfium` binding is resolved exactly once (lazily) in this order:
//!   1. `PDFIUM_DYLIB_PATH` env var,
//!   2. macOS-only vendored path (`src-tauri/frameworks/libpdfium.dylib`),
//!   3. `Pdfium::bind_to_system_library()`.

#[cfg(target_os = "macos")]
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use pdfium_render::prelude::{Pdfium, PdfiumError};

use crate::LensError;
use crate::parse::{Block, BlockType};

use super::{ExtractOutput, Extractor, SourceAnchor};

/// Segments whose font size is at least this ratio above the page modal are
/// treated (best-effort) as headings.
const HEADING_FONT_RATIO: f32 = 1.2;

/// Heading candidates must also be short; a long big-font run is likely body text.
const HEADING_MAX_CHARS: usize = 120;

/// Process-global libpdfium binding (`Pdfium` is `Send + Sync` via the
/// `thread_safe` feature).
static PDFIUM: OnceLock<Result<Pdfium, String>> = OnceLock::new();

/// Serializes whole-document extraction. pdfium-render's `thread_safe` feature
/// only guards individual FFI calls — a multi-call sequence holding live
/// `PdfDocument`/`PdfPage` handles across threads segfaults in libpdfium. This
/// mutex holds for the full `extract` body and is load-bearing for the parallel
/// test runner.
static PDFIUM_EXTRACT_LOCK: Mutex<()> = Mutex::new(());

/// Returns the vendored `libpdfium.dylib` path relative to this crate's manifest
/// dir (`<workspace>/src-tauri/frameworks/libpdfium.dylib`). macOS-only.
#[cfg(target_os = "macos")]
fn vendored_dylib_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("src-tauri")
        .join("frameworks")
        .join("libpdfium.dylib")
}

/// Binds libpdfium exactly once (env override → vendored path → system) and
/// returns the cached [`Pdfium`]. Returns [`LensError::Parse`] if no binding
/// could be established.
fn pdfium() -> Result<&'static Pdfium, LensError> {
    let cell = PDFIUM.get_or_init(|| {
        // 1. PDFIUM_DYLIB_PATH env override.
        if let Ok(path) = std::env::var("PDFIUM_DYLIB_PATH") {
            match Pdfium::bind_to_library(&path) {
                Ok(bindings) => return Ok(Pdfium::new(bindings)),
                Err(e) => {
                    return Err(format!(
                        "PDFIUM_DYLIB_PATH={path:?} set but bind_to_library failed: {e:?}"
                    ));
                }
            }
        }

        // 2. Vendored path (macOS only — compiled out on other platforms).
        #[cfg(target_os = "macos")]
        {
            let vendored = vendored_dylib_path();
            if vendored.exists() {
                match Pdfium::bind_to_library(&vendored) {
                    Ok(bindings) => return Ok(Pdfium::new(bindings)),
                    Err(e) => {
                        return Err(format!(
                            "vendored libpdfium at {} failed to bind: {e:?}",
                            vendored.display()
                        ));
                    }
                }
            }
        }

        // 3. System library fallback.
        match Pdfium::bind_to_system_library() {
            Ok(bindings) => Ok(Pdfium::new(bindings)),
            Err(e) => Err(format!(
                "could not bind libpdfium: no usable PDFIUM_DYLIB_PATH, \
                 no bound vendored dylib (macOS-only; bundled-pdfium for other \
                 platforms is post-MVP), and bind_to_system_library failed: {e:?}"
            )),
        }
    });

    match cell {
        Ok(p) => Ok(p),
        Err(msg) => Err(LensError::Parse(format!("libpdfium binding failed: {msg}"))),
    }
}

/// PDF extractor — implements [`Extractor`] via `pdfium-render`.
pub struct PdfExtractor;

impl Extractor for PdfExtractor {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError> {
        let pdfium = pdfium()?;

        // Only one live `PdfDocument` at a time (see `PDFIUM_EXTRACT_LOCK`).
        let _extract_guard = PDFIUM_EXTRACT_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let document = pdfium
            .load_pdf_from_byte_slice(raw, None)
            .map_err(|e: PdfiumError| {
                LensError::Parse(format!("pdfium failed to load PDF: {e:?}"))
            })?;

        let mut extracted_text = String::new();
        let mut blocks: Vec<Block> = Vec::new();
        let mut anchors: Vec<SourceAnchor> = Vec::new();

        // page_number is 1-based (PDF convention; matches `chunks.page`).
        for (page_index, page) in document.pages().iter().enumerate() {
            let page_number = (page_index as u32) + 1;

            let text = match page.text() {
                Ok(t) => t,
                Err(_) => continue,
            };

            let modal_font = modal_font_size(&text);

            for segment in text.segments().iter() {
                let seg_text = segment.text();
                if seg_text.trim().is_empty() {
                    continue;
                }

                let bounds = segment.bounds();
                let bbox = [
                    bounds.left().value,
                    bounds.bottom().value,
                    bounds.right().value,
                    bounds.top().value,
                ];

                let seg_font = segment_font_size(&segment);
                let is_heading = seg_text.chars().count() <= HEADING_MAX_CHARS
                    && modal_font > 0.0
                    && seg_font >= modal_font * HEADING_FONT_RATIO;
                let btype = if is_heading {
                    BlockType::Heading.as_str()
                } else {
                    BlockType::Paragraph.as_str()
                };

                let char_start = extracted_text.len();
                extracted_text.push_str(&seg_text);
                let char_end = extracted_text.len();
                extracted_text.push('\n');

                blocks.push(Block {
                    block_type: btype.to_string(),
                    section_path: String::new(),
                    text: seg_text,
                    char_start,
                    char_end,
                });
                anchors.push(SourceAnchor::Pdf {
                    page: page_number,
                    bbox,
                });
            }
        }

        // The trailing '\n' is never inside any block slice; safe to trim.
        while extracted_text.ends_with('\n') {
            extracted_text.pop();
        }

        // needs_ocr signal: empty output (not Err) so run_ingest sets the status
        // via the Ok-with-status mechanism rather than flipping to error.
        if extracted_text.trim().is_empty() {
            return Ok(ExtractOutput {
                extracted_text: String::new(),
                blocks: Vec::new(),
                anchors: Vec::new(),
                table_markdown: None,
            });
        }

        Ok(ExtractOutput {
            extracted_text,
            blocks,
            anchors,
            table_markdown: None,
        })
    }
}

/// Returns the most frequent (rounded-to-int) scaled font size across the page,
/// or `0.0` if the page has no characters.
fn modal_font_size(text: &pdfium_render::prelude::PdfPageText) -> f32 {
    use std::collections::HashMap;
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for ch in text.chars().iter() {
        let size = ch.scaled_font_size().value.round() as i32;
        if size > 0 {
            *counts.entry(size).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(size, _)| size as f32)
        .unwrap_or(0.0)
}

/// Returns the maximum scaled font size across a segment's characters, or `0.0`.
fn segment_font_size(segment: &pdfium_render::prelude::PdfPageTextSegment) -> f32 {
    let chars = match segment.chars() {
        Ok(c) => c,
        Err(_) => return 0.0,
    };
    let mut max = 0.0f32;
    for ch in chars.iter() {
        let size = ch.scaled_font_size().value;
        if size > max {
            max = size;
        }
    }
    max
}

// Tests require the libpdfium dylib. On platforms where binding fails (e.g. the
// Linux CI runner) the binding-dependent tests are skipped, not failed.
#[cfg(test)]
mod tests {
    use super::*;

    const SENTINEL: &str = "The quick brown fox jumps over the lazy dog.";

    /// Builds a tiny single-page A4 PDF with a text layer containing [`SENTINEL`].
    fn build_text_layer_pdf() -> Vec<u8> {
        use printpdf::{BuiltinFont, Mm, PdfDocument};
        use std::io::BufWriter;

        let (doc, page1, layer1) =
            PdfDocument::new("sentinel-fixture", Mm(210.0), Mm(297.0), "Layer 1");
        let layer = doc.get_page(page1).get_layer(layer1);
        let font = doc
            .add_builtin_font(BuiltinFont::Helvetica)
            .expect("add builtin font");
        layer.use_text(SENTINEL, 14.0, Mm(20.0), Mm(270.0), &font);

        let mut buf = Vec::new();
        doc.save(&mut BufWriter::new(&mut buf))
            .expect("serialize fixture PDF to bytes");
        buf
    }

    /// Builds a single-page PDF with no text layer (blank page, no glyphs).
    fn build_no_text_layer_pdf() -> Vec<u8> {
        use printpdf::{Mm, PdfDocument};
        use std::io::BufWriter;

        let (doc, _page1, _layer1) =
            PdfDocument::new("no-text-fixture", Mm(210.0), Mm(297.0), "Layer 1");

        let mut buf = Vec::new();
        doc.save(&mut BufWriter::new(&mut buf))
            .expect("serialize no-text fixture PDF to bytes");
        buf
    }

    /// Returns `None` (skip) when libpdfium is unavailable; panics on other errors.
    fn try_extract(raw: &[u8]) -> Option<ExtractOutput> {
        if let Err(e) = pdfium() {
            eprintln!("skipping PDF extractor test: libpdfium not bindable here: {e:?}");
            return None;
        }
        Some(
            PdfExtractor
                .extract(raw)
                .expect("PDF extraction must succeed once pdfium is bound"),
        )
    }

    #[test]
    fn pdf_extractor_binds_pdfium_and_extracts() {
        let raw = build_text_layer_pdf();
        let Some(out) = try_extract(&raw) else {
            return;
        };
        assert!(
            !out.extracted_text.trim().is_empty(),
            "a text-layer PDF must produce non-empty extracted_text"
        );
        assert!(
            !out.blocks.is_empty(),
            "a text-layer PDF must produce at least one block"
        );
    }

    #[test]
    fn pdf_sentinel_text_present() {
        let raw = build_text_layer_pdf();
        let Some(out) = try_extract(&raw) else {
            return;
        };
        let got: String = out
            .extracted_text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let want: String = SENTINEL.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(
            got.contains(&want),
            "sentinel sentence missing from extracted_text;\n  want substring: {want:?}\n  got: {got:?}"
        );
    }

    #[test]
    fn pdf_byte_identity() {
        let raw = build_text_layer_pdf();
        let Some(out) = try_extract(&raw) else {
            return;
        };
        assert!(!out.blocks.is_empty(), "fixture must produce blocks");
        for (i, b) in out.blocks.iter().enumerate() {
            assert_eq!(
                &out.extracted_text[b.char_start..b.char_end],
                b.text,
                "byte-identity violated for block[{i}] (type={:?})",
                b.block_type
            );
        }
    }

    #[test]
    fn pdf_anchors_index_aligned_pages_populated() {
        let raw = build_text_layer_pdf();
        let Some(out) = try_extract(&raw) else {
            return;
        };
        assert_eq!(
            out.anchors.len(),
            out.blocks.len(),
            "anchors.len() must equal blocks.len()"
        );
        for (i, a) in out.anchors.iter().enumerate() {
            match a {
                SourceAnchor::Pdf { page, .. } => assert!(
                    *page >= 1,
                    "anchor[{i}] page must be 1-based (>= 1), got {page}"
                ),
                other => panic!("anchor[{i}] must be SourceAnchor::Pdf, got {other:?}"),
            }
        }
    }

    #[test]
    fn pdf_no_text_layer_yields_empty_output() {
        let raw = build_no_text_layer_pdf();
        // Probe binding; skip on platforms that can't load the universal dylib.
        if pdfium().is_err() {
            eprintln!(
                "skipping pdf_no_text_layer_yields_empty_output: libpdfium not bindable here"
            );
            return;
        }
        let out = PdfExtractor
            .extract(&raw)
            .expect("a no-text-layer PDF must return Ok (empty), NEVER Err");
        assert!(
            out.extracted_text.trim().is_empty(),
            "no-text-layer PDF must produce empty extracted_text (needs_ocr signal); got {:?}",
            out.extracted_text
        );
        assert!(
            out.blocks.is_empty(),
            "no-text-layer PDF must produce no blocks"
        );
        assert!(
            out.anchors.is_empty(),
            "no-text-layer PDF must produce no anchors"
        );
    }

    #[test]
    fn pdf_invalid_bytes_returns_parse_error() {
        if pdfium().is_err() {
            eprintln!(
                "skipping pdf_invalid_bytes_returns_parse_error: libpdfium not bindable here"
            );
            return;
        }
        let err = PdfExtractor
            .extract(b"this is definitely not a pdf")
            .expect_err("non-PDF bytes must error");
        assert!(
            matches!(err, LensError::Parse(_)),
            "expected LensError::Parse, got: {err:?}"
        );
    }
}
