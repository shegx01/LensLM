//! PDF extractor (M4 Phase 2, Step 5).
//!
//! [`PdfExtractor`] parses a PDF from raw bytes using `pdfium-render` (a pure-Rust
//! binding that DYNAMICALLY LOADS the prebuilt `libpdfium` dylib at runtime) and
//! produces a canonical [`ExtractOutput`] where:
//! - `extracted_text` is the concatenation of every page's text (one `\n` between
//!   consecutive blocks within a page; one `\n` between pages). The block offsets
//!   are built together with this buffer so byte-identity holds by construction.
//! - `blocks` carries one [`Block`] per text segment-group:
//!   - `block_type = "heading"` (best-effort) when a short segment's font size is
//!     `>= HEADING_FONT_RATIO ×` the page's modal body font size,
//!   - `block_type = "paragraph"` otherwise. Tables are hard in PDF and are NOT
//!     detected — they degrade to `paragraph` (flat) for Phase 2.
//! - `anchors` carries one [`SourceAnchor::Pdf { page, bbox }`] per block, where
//!   `page` is **1-based** (page 1 is the first page — matches the PDF page
//!   convention and the `chunks.page` column) and `bbox = [x0, y0, x1, y1]` in PDF
//!   user-space points (origin bottom-left), taken from the segment's bounds.
//!
//! # needs_ocr signalling
//!
//! An image-only / no-text-layer PDF yields empty (or all-whitespace) text. The
//! extractor returns an [`ExtractOutput`] with **empty `extracted_text` and no
//! blocks** in that case (it does NOT return `Err`). `run_ingest` (ingest.rs)
//! detects the empty extraction for `kind == "pdf"` and sets
//! `status = needs_ocr` via the SAME Ok-with-status mechanism Step 7 added for
//! `needs_js` — never an `Err` (which would flip the source to `error`), never an
//! empty indexed source.
//!
//! # libpdfium binding
//!
//! Pdfium's bindings live in a process-global `OnceCell` inside `pdfium-render`
//! (only ONE [`pdfium_render::prelude::Pdfium`] may be constructed per process).
//! [`pdfium`] binds it exactly once, lazily, in this ORDER (the bare two-step bind
//! fails on a clean checkout with no system pdfium):
//!   1. `PDFIUM_DYLIB_PATH` env var → `Pdfium::bind_to_library`,
//!   2. **macOS only** — the vendored repo path
//!      `src-tauri/frameworks/libpdfium.dylib` (resolved relative to
//!      `CARGO_MANIFEST_DIR` — tests run from `lens-core/`, so the workspace
//!      `src-tauri/frameworks/` is `../src-tauri/frameworks/`). The bundled
//!      pdfium asset is the macOS universal `.dylib`; M4 ships macOS only, so this
//!      step is `#[cfg(target_os = "macos")]`-gated and does NOT exist on other
//!      platforms (no silent "look for a `.dylib` that can't exist on Linux").
//!      Bundling pdfium for other platforms is post-MVP.
//!   3. `Pdfium::bind_to_system_library()`.
//!
//! The result is cached so every `extract` reuses the one binding.

// `PathBuf` is only used by the macOS-only vendored-dylib resolver below.
#[cfg(target_os = "macos")]
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use pdfium_render::prelude::{Pdfium, PdfiumError};

use crate::LensError;
use crate::parse::{Block, BlockType};

use super::{ExtractOutput, Extractor, SourceAnchor};

/// A segment whose font size is at least this multiple of the page's modal body
/// font size is treated (best-effort) as a `"heading"` rather than a `"paragraph"`.
const HEADING_FONT_RATIO: f32 = 1.2;

/// A heading candidate must also be short (PDF gives no semantic markup, so a long
/// run in a big font is almost certainly body text, not a heading).
const HEADING_MAX_CHARS: usize = 120;

/// Process-global libpdfium binding. `Pdfium` is `Send + Sync` (the `thread_safe`
/// feature), so it is safe to share across parallel ingest/test threads; the
/// per-binding-call FFI is serialized inside pdfium-render.
static PDFIUM: OnceLock<Result<Pdfium, String>> = OnceLock::new();

/// Serializes whole-document extraction. pdfium-render's `thread_safe` feature
/// only guards INDIVIDUAL binding calls — it does NOT make a multi-call sequence
/// that holds live `PdfDocument`/`PdfPage` handles (load → iterate pages → read
/// segments) safe across concurrent threads, which segfaults in libpdfium. This
/// mutex holds for the full `extract` body so at most one document is live at a
/// time across the process. In production this is moot (ingest already holds a
/// single permit); it is load-bearing for the parallel test runner.
static PDFIUM_EXTRACT_LOCK: Mutex<()> = Mutex::new(());

/// Resolves the vendored libpdfium path relative to this crate's manifest dir.
///
/// `CARGO_MANIFEST_DIR` for this crate is `<workspace>/lens-core`, so the vendored
/// dylib lives at `<workspace>/src-tauri/frameworks/libpdfium.dylib`.
///
/// macOS-only: the bundled asset is the macOS universal `.dylib`. M4 ships macOS;
/// bundling pdfium for other platforms is post-MVP, so this path does not exist
/// off macOS (the `pdfium()` bind order below is `#[cfg]`-split to match).
#[cfg(target_os = "macos")]
fn vendored_dylib_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("src-tauri")
        .join("frameworks")
        .join("libpdfium.dylib")
}

/// Binds libpdfium exactly once (env override → vendored path → system) and
/// returns a shared reference to the cached [`Pdfium`].
///
/// Returns a clear [`LensError::Parse`] if no binding could be established (no
/// usable dylib found on any of the three paths).
fn pdfium() -> Result<&'static Pdfium, LensError> {
    let cell = PDFIUM.get_or_init(|| {
        // 1. PDFIUM_DYLIB_PATH env override (dev / CI escape hatch).
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

        // 2. Vendored repo path (macOS ONLY) — the dylib fetched by
        //    scripts/fetch-pdfium.sh. The bundled asset is the macOS universal
        //    `.dylib`; M4 ships macOS only and bundling pdfium for other platforms
        //    is post-MVP, so this step is compiled out entirely off macOS rather
        //    than probing for a `.dylib` that cannot exist there.
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

        // 3. System library fallback. On non-macOS (where the macOS-only vendored
        //    `.dylib` step above is compiled out) this is the only path after the
        //    `PDFIUM_DYLIB_PATH` env override — a system libpdfium must be present.
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
///
/// Iterates pages, extracts each page's text segments + per-segment bounding
/// boxes, classifies each as `"heading"` (best-effort by font size) or
/// `"paragraph"`, and builds the canonical `extracted_text` buffer together with
/// the block offsets so byte-identity holds.
pub struct PdfExtractor;

impl Extractor for PdfExtractor {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError> {
        let pdfium = pdfium()?;

        // Serialize whole-document work: only one live `PdfDocument` at a time
        // across the process (see `PDFIUM_EXTRACT_LOCK`). Held for the duration of
        // `extract` — released when `_extract_guard` drops with `document`.
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

        // `page_number` is 1-based (PDF convention; matches `chunks.page`).
        for (page_index, page) in document.pages().iter().enumerate() {
            let page_number = (page_index as u32) + 1;

            let text = match page.text() {
                Ok(t) => t,
                // A page with no text layer yields no text object; skip it.
                Err(_) => continue,
            };

            // Determine the page's modal (most common, rounded) font size across
            // all chars, used as the body-text baseline for heading detection.
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

                // Heading heuristic: short + font noticeably larger than body.
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

        // Trim the trailing separator newline without invalidating any block
        // offset (the last '\n' is never inside a block slice).
        while extracted_text.ends_with('\n') {
            extracted_text.pop();
        }

        // needs_ocr signal: an image-only / no-text-layer PDF produces no usable
        // text. Return EMPTY output (not Err) so `run_ingest` sets needs_ocr via
        // the Ok-with-status mechanism (Step 7) rather than indexing nothing.
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

/// Returns the modal (most frequent, rounded-to-int) scaled font size across all
/// characters on the page, or `0.0` if the page has no characters.
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

/// Returns the maximum scaled font size across a segment's characters (the
/// segment's effective size for heading detection), or `0.0` if it has none.
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

// ---------------------------------------------------------------------------
// Tests (TDD: RED written first, GREEN implemented above)
// ---------------------------------------------------------------------------
//
// These tests need the libpdfium dylib to bind. The vendored
// `src-tauri/frameworks/libpdfium.dylib` (fetched by `scripts/fetch-pdfium.sh`)
// is the macOS *universal* binary, so `pdfium()` only binds successfully on
// macOS (or when `PDFIUM_DYLIB_PATH` points at a loadable build). On a platform
// where binding fails (e.g. the Linux CI runner, which fetches + checksum-verifies
// the dylib but cannot load it), the binding-dependent tests are skipped with a
// printed note rather than failing — the AC6 dylib-load gate is verified on
// macOS dev / release. The fixture-builder and threshold logic are
// platform-independent.
#[cfg(test)]
mod tests {
    use super::*;

    /// A known sentinel sentence written into the text-layer fixture PDF (AC4c).
    const SENTINEL: &str = "The quick brown fox jumps over the lazy dog.";

    /// Builds a tiny single-page A4 PDF with a text layer containing [`SENTINEL`],
    /// returned as in-memory bytes. Uses `printpdf` (a dev-dependency, pure-Rust
    /// PDF *writer*) so no binary fixture is committed (Step 5 TDD).
    fn build_text_layer_pdf() -> Vec<u8> {
        use printpdf::{BuiltinFont, Mm, PdfDocument};
        use std::io::BufWriter;

        let (doc, page1, layer1) =
            PdfDocument::new("sentinel-fixture", Mm(210.0), Mm(297.0), "Layer 1");
        let layer = doc.get_page(page1).get_layer(layer1);
        let font = doc
            .add_builtin_font(BuiltinFont::Helvetica)
            .expect("add builtin font");
        // Write the sentinel near the top of the page (y measured from bottom).
        layer.use_text(SENTINEL, 14.0, Mm(20.0), Mm(270.0), &font);

        let mut buf = Vec::new();
        doc.save(&mut BufWriter::new(&mut buf))
            .expect("serialize fixture PDF to bytes");
        buf
    }

    /// Builds a tiny single-page PDF with NO text layer (an image-only / scanned
    /// PDF surrogate): a blank page with no glyphs at all. pdfium extracts empty
    /// text from it, exercising the `needs_ocr` empty-output path (AC7).
    fn build_no_text_layer_pdf() -> Vec<u8> {
        use printpdf::{Mm, PdfDocument};
        use std::io::BufWriter;

        // A document with one page and one (empty) layer — no text objects.
        let (doc, _page1, _layer1) =
            PdfDocument::new("no-text-fixture", Mm(210.0), Mm(297.0), "Layer 1");

        let mut buf = Vec::new();
        doc.save(&mut BufWriter::new(&mut buf))
            .expect("serialize no-text fixture PDF to bytes");
        buf
    }

    /// Attempts a PDF extraction, returning `None` (with a printed skip note) when
    /// the libpdfium binding is unavailable on this platform. Panics on any OTHER
    /// extraction error (a real failure, not a missing-dylib skip).
    fn try_extract(raw: &[u8]) -> Option<ExtractOutput> {
        // Probe the binding first so a missing dylib is a SKIP, not a failure.
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

    /// AC6 — the vendored dylib loads: constructing `PdfExtractor` and extracting a
    /// real text PDF succeeds (pdfium bound via env / vendored path). On a platform
    /// where the universal dylib cannot load, this is skipped (see module note).
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

    /// AC4c — extraction fidelity: the known sentinel sentence is PRESENT in
    /// `extracted_text` (catches silent content loss). pdfium may split the line
    /// into segments, so compare on a whitespace-normalized form.
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

    /// AC4a — byte-identity: every block slices `extracted_text` exactly.
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

    /// AC5 — anchors: index-aligned, all `Pdf`, with a populated (1-based) page.
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

    /// AC7 — needs_ocr signal: a no-text-layer PDF yields EMPTY output (no Err),
    /// so `run_ingest` can set `needs_ocr` via the Ok-with-status mechanism. This
    /// asserts the extractor's contract directly (empty `extracted_text`, no
    /// blocks, no anchors — never an Err).
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

    /// Loading malformed bytes that are not a PDF at all is a clear `Err`
    /// (`LensError::Parse`) — distinct from the empty-output no-text-layer case.
    /// Skipped when pdfium is not bindable (the load call needs the binding).
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
