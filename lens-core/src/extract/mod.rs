//! The `Extractor` trait seam — the single new abstraction through which every
//! source format produces a canonical extraction result for the (unchanged)
//! Phase-1 `chunk → embed → index` pipeline.
//!
//! Each [`Extractor`] takes the RAW source bytes (binary formats such as
//! PDF/DOCX are not UTF-8) and returns an [`ExtractOutput`] binding three
//! index-aligned facts:
//! * `extracted_text` — the canonical UTF-8 buffer that chunk `char_start/end`
//!   offsets index into;
//! * `blocks` — the structural [`Block`]s (reusing [`parse::Block`]) whose
//!   `char_start..char_end` slice `extracted_text` byte-identically;
//! * `anchors` — one [`SourceAnchor`] per block (index-aligned with `blocks`),
//!   carrying the format-native coordinates needed to re-locate the block in the
//!   ORIGINAL source (PDF page+bbox, DOCX node path, URL DOM anchor, or
//!   `Text` = whole-doc / none for plain text & Markdown).
//!
//! Phase 2 scope (honest caveat, Principle 5): this unifies the `Extractor`
//! trait SURFACE only — it does NOT change the ingest read path. Text/Markdown
//! is refactored onto the trait here ([`TextExtractor`]); PDF and URL extractors
//! are added in later steps. The dispatcher [`extractor_for`] returns a clear
//! [`LensError`] for kinds not yet implemented.

pub mod docx;
pub mod epub;
pub mod json;
pub mod jsonl;
pub mod odt;
pub mod pdf;
pub mod rtf;
pub mod url;
pub mod xml;
pub mod xml_blocks;
pub mod yaml;

use serde::{Deserialize, Serialize};

use crate::LensError;
use crate::parse::{Block, SourceKind, parse_blocks};

/// Format-native coordinates that re-locate a [`Block`] inside its ORIGINAL
/// source document.
///
/// Serialized (tagged) as JSON for persistence in the dedicated
/// `chunks.source_anchor` column (added in a later step). The tag/content shape
/// is stable: each variant serializes independently so adding the binary/URL
/// variants cannot change the wire shape of the existing ones.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum SourceAnchor {
    /// A PDF block: its page number (0-based or 1-based per the extractor) and
    /// its bounding box `[x0, y0, x1, y1]` in PDF user-space points.
    Pdf { page: u32, bbox: [f32; 4] },
    /// A DOCX block: a path identifying the node (paragraph/table) it came from.
    Docx { node_path: String },
    /// A URL block: the decimal BYTE OFFSET (as a string) of the block's first
    /// character in the canonical extracted-text buffer — NOT a DOM selector.
    Url { text_offset: String },
    /// No format-native coordinate is meaningful — the whole document IS the
    /// canonical buffer (plain text & Markdown).
    Text,
    /// A block from a structured data format (JSON/JSONL/YAML/XML): its
    /// JSON-pointer-ish `path` within the source document (e.g. `/users/0/name`).
    ///
    /// Additive to the `#[serde(tag = "kind")]` enum — each variant serializes
    /// independently under its own `kind` tag, so adding `Structured` cannot
    /// change the wire shape of `Text`/`Pdf`/`Docx`/`Url`.
    Structured { path: String },
    /// An RTF block: its byte offset in the canonical extracted-text buffer.
    ///
    /// Additive under its own `kind` tag (issue #77); cannot change the wire
    /// shape of any pre-existing variant. `u64` (not `usize`) so the persisted
    /// JSON wire shape is identical on 32- and 64-bit targets.
    Rtf { text_offset: u64 },
    /// An ODT block: the node path in `content.xml` (e.g. `"body/text:h[0]"`).
    ///
    /// Additive under its own `kind` tag (issue #77).
    Odt { node_path: String },
    /// An EPUB block: the spine index and the href of the content document.
    ///
    /// Additive under its own `kind` tag (issue #77). `spine_index` is `u64`
    /// (not `usize`) for a target-independent JSON wire shape.
    Epub { spine_index: u64, href: String },
}

/// The canonical extraction result for a single source.
///
/// `blocks` and `anchors` are index-aligned: `anchors[i]` is the format-native
/// coordinate for `blocks[i]`. Every block's `char_start..char_end` indexes into
/// `extracted_text` byte-identically (the Phase-1 byte-identity invariant).
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractOutput {
    /// The canonical UTF-8 buffer that chunk offsets index into.
    pub extracted_text: String,
    /// Structural blocks; `char_start/char_end` index into `extracted_text`.
    pub blocks: Vec<Block>,
    /// One anchor per block, index-aligned with `blocks`.
    pub anchors: Vec<SourceAnchor>,
}

/// The single new seam: turns raw source bytes into a canonical [`ExtractOutput`].
///
/// `Send + Sync` so a `Box<dyn Extractor>` can be resolved per-kind and used from
/// the async ingest pipeline. Takes `raw: &[u8]` (NOT `&str`) because PDF/DOCX are
/// binary; text-based extractors validate UTF-8 themselves.
pub trait Extractor: Send + Sync {
    /// Extracts the canonical text, structural blocks, and per-block anchors from
    /// the raw source bytes.
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError>;
}

/// Extractor for plain-text and Markdown sources.
///
/// Wraps the existing [`parse_blocks`] (`parse.rs`) so text/MD behaviour is
/// byte-identical to Phase 1: the input bytes ARE the canonical buffer, blocks
/// come straight from `parse_blocks`, and every block gets a [`SourceAnchor::Text`]
/// (no format-native coordinate is meaningful for plain text).
pub struct TextExtractor {
    kind: SourceKind,
}

impl TextExtractor {
    /// Builds a `TextExtractor` for the given (text/markdown) [`SourceKind`].
    pub fn new(kind: SourceKind) -> Self {
        Self { kind }
    }
}

impl Extractor for TextExtractor {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError> {
        // Text/MD must be valid UTF-8; surface a clear validation error otherwise.
        let s = std::str::from_utf8(raw)
            .map_err(|e| LensError::Validation(format!("source is not valid UTF-8: {e}")))?;
        let blocks = parse_blocks(s, self.kind);
        // One whole-doc `Text` anchor per block, index-aligned.
        let anchors = vec![SourceAnchor::Text; blocks.len()];
        Ok(ExtractOutput {
            extracted_text: s.to_string(),
            blocks,
            anchors,
        })
    }
}

/// Resolves the [`Extractor`] for a `sources.kind` string.
///
/// Parses the boundary `&str` into a [`SourceKind`] and dispatches via an
/// EXHAUSTIVE match — adding a [`SourceKind`] variant is a compile error here
/// until an extractor is wired for it. `Text`/`Markdown` map to a
/// [`TextExtractor`], `Pdf` to [`pdf::PdfExtractor`], `Docx` to
/// [`docx::DocxExtractor`], and `Url` to [`url::UrlExtractor`]. An unknown kind
/// string is a [`LensError::Validation`] (from [`SourceKind::from_kind_str`]).
///
/// Under the `test-util` feature a test may register an injected extractor for an
/// otherwise-unknown kind (see [`set_test_extractor_factory`]); that injection is
/// consulted first so integration tests can drive a fake binary kind end-to-end.
pub fn extractor_for(kind: &str) -> Result<Box<dyn Extractor>, LensError> {
    #[cfg(feature = "test-util")]
    if let Some(injected) = test_seam::injected_extractor(kind) {
        return Ok(injected);
    }
    match SourceKind::from_kind_str(kind)? {
        SourceKind::Text => Ok(Box::new(TextExtractor::new(SourceKind::Text))),
        SourceKind::Markdown => Ok(Box::new(TextExtractor::new(SourceKind::Markdown))),
        SourceKind::Docx => Ok(Box::new(docx::DocxExtractor)),
        // URL extractor — rs-trafilatura-based HTML content extraction. The async
        // reqwest GET lives in run_ingest (ingest.rs); this extractor receives
        // already-fetched bytes.
        SourceKind::Url => Ok(Box::new(url::UrlExtractor)),
        // PDF extractor — pdfium-render text + per-segment bbox extraction. A
        // no-text-layer (scanned) PDF yields empty output → run_ingest sets
        // needs_ocr (Ok-with-status, never Err).
        SourceKind::Pdf => Ok(Box::new(pdf::PdfExtractor)),
        // Structured-format extractors (M4 Phase 2.5c): key-path verbalization
        // with byte-identity offsets and `SourceAnchor::Structured` anchors.
        SourceKind::Json => Ok(Box::new(json::JsonExtractor)),
        SourceKind::Jsonl => Ok(Box::new(jsonl::JsonlExtractor)),
        SourceKind::Yaml => Ok(Box::new(yaml::YamlExtractor)),
        SourceKind::Xml => Ok(Box::new(xml::XmlExtractor)),
        // Office/binary-format extractors (M4 issue #77): RTF flat paragraphs,
        // ODT/EPUB structural blocks with byte-identity offsets.
        SourceKind::Rtf => Ok(Box::new(rtf::RtfExtractor)),
        SourceKind::Odt => Ok(Box::new(odt::OdtExtractor)),
        SourceKind::Epub => Ok(Box::new(epub::EpubExtractor)),
    }
}

/// Test-only injection seam: lets integration tests register a fake binary
/// [`Extractor`] for an arbitrary `sources.kind` so the ingest pipeline can be
/// driven through the DERIVED-kind path (raw-bytes hash, `.extracted.txt`
/// sibling, two-stage guard) without a real PDF/DOCX/URL backend.
///
/// Gated behind `test-util` so it is absent from production builds. The factory
/// is a thread-local so concurrent tests on different threads never see each
/// other's injection.
#[cfg(feature = "test-util")]
pub mod test_seam {
    use super::{Extractor, LensError};
    use std::cell::RefCell;
    use std::collections::HashMap;

    type Factory = Box<dyn Fn() -> Box<dyn Extractor>>;

    thread_local! {
        static FACTORIES: RefCell<HashMap<String, Factory>> = RefCell::new(HashMap::new());
    }

    /// Registers a factory that builds the [`Extractor`] for `kind`. Subsequent
    /// [`extractor_for`](super::extractor_for) calls (on this thread) for `kind`
    /// return a fresh box from `factory`.
    pub fn set_test_extractor_factory<F>(kind: &str, factory: F)
    where
        F: Fn() -> Box<dyn Extractor> + 'static,
    {
        FACTORIES.with(|m| {
            m.borrow_mut().insert(kind.to_string(), Box::new(factory));
        });
    }

    /// Clears any injected factory for `kind` (test cleanup).
    pub fn clear_test_extractor_factory(kind: &str) {
        FACTORIES.with(|m| {
            m.borrow_mut().remove(kind);
        });
    }

    /// Resolves an injected extractor for `kind`, if any was registered on this
    /// thread.
    pub(super) fn injected_extractor(kind: &str) -> Option<Box<dyn Extractor>> {
        FACTORIES.with(|m| m.borrow().get(kind).map(|f| f()))
    }

    /// A configurable fake binary extractor for ingest tests.
    ///
    /// `extract` increments a shared call counter (so a test can assert the
    /// extractor was NOT re-run on a no-op re-ingest) and returns a fixed
    /// single-block [`ExtractOutput`]. It panics if `panic_if_called` is set —
    /// used to prove the Stage-1 size guard fires BEFORE extraction.
    pub struct FakeBinaryExtractor {
        /// Bumped once per `extract` call.
        pub calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        /// The canonical text this fake "decodes" the raw bytes into.
        pub extracted_text: String,
        /// If true, `extract` panics — proving it must not be reached.
        pub panic_if_called: bool,
    }

    impl Extractor for FakeBinaryExtractor {
        fn extract(&self, _raw: &[u8]) -> Result<super::ExtractOutput, LensError> {
            assert!(
                !self.panic_if_called,
                "FakeBinaryExtractor::extract was called but the Stage-1 guard \
                 should have rejected the input first"
            );
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let text = self.extracted_text.clone();
            let block = super::Block {
                block_type: crate::parse::BlockType::Paragraph.as_str().to_string(),
                section_path: String::new(),
                char_start: 0,
                char_end: text.len(),
                text: text.clone(),
            };
            Ok(super::ExtractOutput {
                extracted_text: text,
                blocks: vec![block],
                anchors: vec![super::SourceAnchor::Pdf {
                    page: 1,
                    bbox: [0.0, 0.0, 0.0, 0.0],
                }],
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AC1+AC2 — each text-based kind drives through `Box<dyn Extractor>` via
    /// `extractor_for` and returns the SAME blocks as the existing `parse_blocks`
    /// (byte-identical text/offsets), `extracted_text` equals the input, every
    /// anchor is `SourceAnchor::Text`, and `anchors.len() == blocks.len()`.
    fn assert_text_kind_matches_parse_blocks(kind_str: &str, source_kind: SourceKind, src: &str) {
        let extractor = extractor_for(kind_str).expect("extractor for known kind");
        let out = extractor
            .extract(src.as_bytes())
            .expect("text/MD extraction is infallible for valid UTF-8");

        // extracted_text is the input verbatim.
        assert_eq!(out.extracted_text, src, "extracted_text must equal input");

        // Blocks are byte-identical to the direct parse_blocks call.
        let expected = parse_blocks(src, source_kind);
        assert_eq!(
            out.blocks, expected,
            "blocks must match parse_blocks exactly"
        );

        // Anchors are index-aligned, one per block, all `Text`.
        assert_eq!(
            out.anchors.len(),
            out.blocks.len(),
            "one anchor per block (index-aligned)"
        );
        for a in &out.anchors {
            assert_eq!(*a, SourceAnchor::Text, "text/MD anchors are all `Text`");
        }

        // Byte-identity: each block slices extracted_text exactly.
        for (i, b) in out.blocks.iter().enumerate() {
            assert_eq!(
                &out.extracted_text[b.char_start..b.char_end],
                b.text,
                "byte-identity violated for block[{i}]"
            );
        }
    }

    #[test]
    fn text_kind_drives_through_trait_object() {
        assert_text_kind_matches_parse_blocks(
            "text",
            SourceKind::Text,
            "First paragraph.\n\nSecond paragraph.\n\nThird.",
        );
    }

    #[test]
    fn markdown_kind_drives_through_trait_object() {
        assert_text_kind_matches_parse_blocks(
            "markdown",
            SourceKind::Markdown,
            "# A\n\nContent under A.\n\n## B\n\nContent under B.\n",
        );
    }

    #[test]
    fn markdown_kind_multibyte_byte_identity() {
        // Emoji + CJK: prove byte offsets survive the trait round-trip.
        assert_text_kind_matches_parse_blocks(
            "markdown",
            SourceKind::Markdown,
            "# 日本語\n\nこんにちは 🦀 world\n",
        );
    }

    #[test]
    fn unknown_kind_returns_err() {
        let err = match extractor_for("nonsense") {
            Ok(_) => panic!("unknown kind must error"),
            Err(e) => e,
        };
        assert!(matches!(err, LensError::Validation(_)));
    }

    #[test]
    fn pdf_kind_resolves_to_extractor() {
        // pdf is a known kind; extractor_for resolves it to an Extractor.
        let result = extractor_for("pdf");
        assert!(
            result.is_ok(),
            "extractor_for(\"pdf\") must resolve an extractor"
        );
    }

    #[test]
    fn docx_kind_resolves_to_extractor() {
        // docx is a known kind; extractor_for resolves it to an Extractor.
        let result = extractor_for("docx");
        assert!(
            result.is_ok(),
            "extractor_for(\"docx\") must resolve an extractor"
        );
    }

    #[test]
    fn url_kind_resolves_to_extractor() {
        // url is a known kind; extractor_for resolves it to an Extractor.
        let result = extractor_for("url");
        assert!(
            result.is_ok(),
            "extractor_for(\"url\") must resolve an extractor"
        );
    }

    #[test]
    fn text_extractor_rejects_invalid_utf8() {
        let extractor = extractor_for("text").unwrap();
        // 0xFF is never valid in UTF-8.
        let err = extractor
            .extract(&[0xFF, 0xFE, 0x00])
            .expect_err("invalid UTF-8 must be a validation error");
        assert!(matches!(err, LensError::Validation(_)));
    }

    #[test]
    fn source_anchor_roundtrips_through_serde_json() {
        let anchors = vec![
            SourceAnchor::Text,
            SourceAnchor::Pdf {
                page: 3,
                bbox: [1.0, 2.5, 100.0, 200.25],
            },
            SourceAnchor::Docx {
                node_path: "body/p[4]".to_string(),
            },
            SourceAnchor::Url {
                text_offset: "42".to_string(),
            },
        ];
        for a in anchors {
            let json = serde_json::to_string(&a).expect("serialize anchor");
            let back: SourceAnchor = serde_json::from_str(&json).expect("deserialize anchor");
            assert_eq!(a, back, "anchor must round-trip through serde_json");
        }
    }

    #[test]
    fn source_anchor_structured_roundtrips_through_serde_json() {
        let a = SourceAnchor::Structured {
            path: "/a/b".to_string(),
        };
        let json = serde_json::to_string(&a).expect("serialize structured anchor");
        let back: SourceAnchor =
            serde_json::from_str(&json).expect("deserialize structured anchor");
        assert_eq!(
            a, back,
            "Structured anchor must round-trip through serde_json"
        );
    }

    #[test]
    fn source_anchor_structured_does_not_change_existing_wire_shape() {
        // Regression guard: adding `Structured` must NOT change the serialized
        // JSON of any pre-existing variant. These literals are the exact wire
        // shapes locked before Phase 2.5c.
        assert_eq!(
            serde_json::to_string(&SourceAnchor::Text).unwrap(),
            r#"{"kind":"Text"}"#
        );
        assert_eq!(
            serde_json::to_string(&SourceAnchor::Pdf {
                page: 3,
                bbox: [1.0, 2.5, 100.0, 200.25],
            })
            .unwrap(),
            r#"{"kind":"Pdf","page":3,"bbox":[1.0,2.5,100.0,200.25]}"#
        );
        assert_eq!(
            serde_json::to_string(&SourceAnchor::Docx {
                node_path: "body/p[4]".to_string(),
            })
            .unwrap(),
            r#"{"kind":"Docx","node_path":"body/p[4]"}"#
        );
        assert_eq!(
            serde_json::to_string(&SourceAnchor::Url {
                text_offset: "42".to_string(),
            })
            .unwrap(),
            r#"{"kind":"Url","text_offset":"42"}"#
        );
    }

    #[test]
    fn source_anchor_office_binary_roundtrips_through_serde_json() {
        let anchors = vec![
            SourceAnchor::Rtf { text_offset: 42 },
            SourceAnchor::Odt {
                node_path: "body/text:h[0]".to_string(),
            },
            SourceAnchor::Epub {
                spine_index: 2,
                href: "OEBPS/chapter1.xhtml".to_string(),
            },
        ];
        for a in anchors {
            let json = serde_json::to_string(&a).expect("serialize office-binary anchor");
            let back: SourceAnchor =
                serde_json::from_str(&json).expect("deserialize office-binary anchor");
            assert_eq!(a, back, "anchor must round-trip through serde_json");
        }
    }

    #[test]
    fn source_anchor_office_binary_does_not_change_existing_wire_shape() {
        // Regression guard: adding `Rtf`/`Odt`/`Epub` must NOT change the
        // serialized JSON of any pre-existing variant (locked literals).
        assert_eq!(
            serde_json::to_string(&SourceAnchor::Text).unwrap(),
            r#"{"kind":"Text"}"#
        );
        assert_eq!(
            serde_json::to_string(&SourceAnchor::Docx {
                node_path: "body/p[4]".to_string(),
            })
            .unwrap(),
            r#"{"kind":"Docx","node_path":"body/p[4]"}"#
        );
        assert_eq!(
            serde_json::to_string(&SourceAnchor::Url {
                text_offset: "42".to_string(),
            })
            .unwrap(),
            r#"{"kind":"Url","text_offset":"42"}"#
        );
        assert_eq!(
            serde_json::to_string(&SourceAnchor::Structured {
                path: "/a/b".to_string(),
            })
            .unwrap(),
            r#"{"kind":"Structured","path":"/a/b"}"#
        );
        // And lock the new variants' own wire shapes.
        assert_eq!(
            serde_json::to_string(&SourceAnchor::Rtf { text_offset: 42 }).unwrap(),
            r#"{"kind":"Rtf","text_offset":42}"#
        );
        assert_eq!(
            serde_json::to_string(&SourceAnchor::Odt {
                node_path: "body/text:h[0]".to_string(),
            })
            .unwrap(),
            r#"{"kind":"Odt","node_path":"body/text:h[0]"}"#
        );
        assert_eq!(
            serde_json::to_string(&SourceAnchor::Epub {
                spine_index: 2,
                href: "OEBPS/chapter1.xhtml".to_string(),
            })
            .unwrap(),
            r#"{"kind":"Epub","spine_index":2,"href":"OEBPS/chapter1.xhtml"}"#
        );
    }

    #[test]
    fn rtf_kind_resolves_to_extractor() {
        assert!(extractor_for("rtf").is_ok());
    }

    #[test]
    fn odt_kind_resolves_to_extractor() {
        assert!(extractor_for("odt").is_ok());
    }

    #[test]
    fn epub_kind_resolves_to_extractor() {
        assert!(extractor_for("epub").is_ok());
    }

    #[test]
    fn json_kind_resolves_to_extractor() {
        assert!(extractor_for("json").is_ok());
    }

    #[test]
    fn jsonl_kind_resolves_to_extractor() {
        assert!(extractor_for("jsonl").is_ok());
    }

    #[test]
    fn yaml_kind_resolves_to_extractor() {
        assert!(extractor_for("yaml").is_ok());
    }

    #[test]
    fn xml_kind_resolves_to_extractor() {
        assert!(extractor_for("xml").is_ok());
    }
}
