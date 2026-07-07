//! The `Extractor` trait seam — turns raw source bytes into a canonical
//! [`ExtractOutput`] (text buffer + index-aligned blocks + per-block anchors)
//! for the `chunk → embed → index` pipeline.

pub mod csv;
pub mod docx;
pub mod epub;
pub mod json;
pub mod jsonl;
pub mod odt;
pub mod pdf;
pub mod rtf;
pub mod spreadsheet;
pub mod tabular_utils;
pub mod url;
pub mod xml;
pub mod xml_blocks;
pub mod yaml;

use serde::{Deserialize, Serialize};

use crate::LensError;
use crate::parse::{Block, SourceKind, parse_blocks};

/// Format-native coordinates that re-locate a [`Block`] in its original source.
///
/// Tagged JSON for `chunks.source_anchor`. Each variant serializes under its
/// own `kind` tag so adding new variants cannot change the wire shape of
/// pre-existing ones.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum SourceAnchor {
    /// PDF page (1-based) and bounding box `[x0, y0, x1, y1]` in user-space points.
    Pdf { page: u32, bbox: [f32; 4] },
    /// DOCX node path (e.g. `"body/p[0]"`).
    Docx { node_path: String },
    /// Decimal byte offset of the block's first character in the extracted-text buffer.
    Url { text_offset: String },
    /// No format-native coordinate (plain text / Markdown).
    Text,
    /// JSON-pointer-ish path within a structured source (e.g. `/users/0/name`).
    Structured { path: String },
    /// RTF byte offset in the extracted-text buffer. `u64` for target-independent wire shape.
    Rtf { text_offset: u64 },
    /// ODT node path in `content.xml` (e.g. `"body/text:h[0]"`).
    Odt { node_path: String },
    /// EPUB spine index and content-document href. `u64` for target-independent wire shape.
    Epub { spine_index: u64, href: String },
    /// Transcript timestamps (seconds) for an audio/video source (issue #44). A
    /// chunk spanning multiple segments carries `[min start, max end]` over them.
    Audio { start_second: f32, end_second: f32 },
}

/// Canonical extraction result for a single source.
///
/// `blocks` and `anchors` are index-aligned: `anchors[i]` is the format-native
/// coordinate for `blocks[i]`. Every block's `char_start..char_end` indexes into
/// `extracted_text` byte-identically.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractOutput {
    pub extracted_text: String,
    pub blocks: Vec<Block>,
    /// One anchor per block, index-aligned.
    pub anchors: Vec<SourceAnchor>,
    /// Pipe-delimited markdown for tabular sources (XLSX/XLS/CSV). Persisted as
    /// `{id}.tables.md` by ingest; never embedded in `extracted_text`.
    pub table_markdown: Option<String>,
}

/// Turns raw source bytes into a canonical [`ExtractOutput`].
///
/// Takes `raw: &[u8]` (not `&str`) because binary formats (PDF/DOCX) are not
/// UTF-8; text-based extractors validate UTF-8 themselves.
pub trait Extractor: Send + Sync {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError>;
}

/// Extractor for plain-text and Markdown sources.
pub struct TextExtractor {
    kind: SourceKind,
}

impl TextExtractor {
    pub fn new(kind: SourceKind) -> Self {
        Self { kind }
    }
}

impl Extractor for TextExtractor {
    fn extract(&self, raw: &[u8]) -> Result<ExtractOutput, LensError> {
        let s = std::str::from_utf8(raw)
            .map_err(|e| LensError::Validation(format!("source is not valid UTF-8: {e}")))?;
        let blocks = parse_blocks(s, self.kind);
        let anchors = vec![SourceAnchor::Text; blocks.len()];
        Ok(ExtractOutput {
            extracted_text: s.to_string(),
            blocks,
            anchors,
            table_markdown: None,
        })
    }
}

/// Resolves the [`Extractor`] for a `sources.kind` string.
///
/// Dispatches via exhaustive match — adding a `SourceKind` variant is a compile
/// error until an extractor is wired. Under the `test-util` feature an injected
/// factory is consulted first (see [`test_seam::set_test_extractor_factory`]).
pub fn extractor_for(kind: &str) -> Result<Box<dyn Extractor>, LensError> {
    #[cfg(feature = "test-util")]
    if let Some(injected) = test_seam::injected_extractor(kind) {
        return Ok(injected);
    }
    match SourceKind::from_kind_str(kind)? {
        SourceKind::Text => Ok(Box::new(TextExtractor::new(SourceKind::Text))),
        SourceKind::Markdown => Ok(Box::new(TextExtractor::new(SourceKind::Markdown))),
        SourceKind::Docx => Ok(Box::new(docx::DocxExtractor)),
        SourceKind::Url => Ok(Box::new(url::UrlExtractor)),
        SourceKind::Pdf => Ok(Box::new(pdf::PdfExtractor)),
        SourceKind::Json => Ok(Box::new(json::JsonExtractor)),
        SourceKind::Jsonl => Ok(Box::new(jsonl::JsonlExtractor)),
        SourceKind::Yaml => Ok(Box::new(yaml::YamlExtractor)),
        SourceKind::Xml => Ok(Box::new(xml::XmlExtractor)),
        SourceKind::Rtf => Ok(Box::new(rtf::RtfExtractor)),
        SourceKind::Odt => Ok(Box::new(odt::OdtExtractor)),
        SourceKind::Epub => Ok(Box::new(epub::EpubExtractor)),
        SourceKind::Xlsx | SourceKind::Xls => Ok(Box::new(spreadsheet::SpreadsheetExtractor)),
        SourceKind::Csv => Ok(Box::new(csv::CsvExtractor)),
        // Audio uses the decode+transcribe path (the audio ingest branch returns
        // before reaching here); this arm is a defensive backstop only (issue #43).
        SourceKind::Audio => Err(LensError::Internal(
            "audio sources use the decode+transcribe path, not an extractor".into(),
        )),
    }
}

pub(crate) const MAX_ZIP_ENTRIES: usize = 10_000;

/// Rejects a ZIP container whose entry count exceeds [`MAX_ZIP_ENTRIES`] (a
/// many-small-entries bomb the uncompressed-size ceiling does not catch). Input
/// that is not a ZIP passes through: the caller's own open validates the format.
pub(crate) fn guard_zip_entry_count(raw: &[u8]) -> Result<(), LensError> {
    let Ok(archive) = zip::ZipArchive::new(std::io::Cursor::new(raw)) else {
        return Ok(());
    };
    if archive.len() > MAX_ZIP_ENTRIES {
        return Err(LensError::Validation(format!(
            "ZIP container has {} entries, exceeding the {MAX_ZIP_ENTRIES}-entry \
             limit (possible zip bomb)",
            archive.len()
        )));
    }
    Ok(())
}

/// Test-only injection seam: lets integration tests register a fake [`Extractor`]
/// for an arbitrary kind to drive the ingest pipeline end-to-end.
///
/// Gated behind `test-util`; absent from production builds. Thread-local so
/// concurrent tests on different threads never see each other's injection.
#[cfg(feature = "test-util")]
pub mod test_seam {
    use super::{Extractor, LensError};
    use std::cell::RefCell;
    use std::collections::HashMap;

    type Factory = Box<dyn Fn() -> Box<dyn Extractor>>;

    thread_local! {
        static FACTORIES: RefCell<HashMap<String, Factory>> = RefCell::new(HashMap::new());
    }

    /// Registers a factory for `kind`; subsequent `extractor_for` calls on this
    /// thread return a fresh box from `factory`.
    pub fn set_test_extractor_factory<F>(kind: &str, factory: F)
    where
        F: Fn() -> Box<dyn Extractor> + 'static,
    {
        FACTORIES.with(|m| {
            m.borrow_mut().insert(kind.to_string(), Box::new(factory));
        });
    }

    /// Clears the injected factory for `kind`.
    pub fn clear_test_extractor_factory(kind: &str) {
        FACTORIES.with(|m| {
            m.borrow_mut().remove(kind);
        });
    }

    /// Returns an injected extractor for `kind` if one was registered on this thread.
    pub(super) fn injected_extractor(kind: &str) -> Option<Box<dyn Extractor>> {
        FACTORIES.with(|m| m.borrow().get(kind).map(|f| f()))
    }

    /// Fake binary extractor for ingest tests.
    ///
    /// Increments `calls` per `extract` invocation; panics if `panic_if_called`
    /// is set (proving the Stage-1 size guard fires before extraction).
    pub struct FakeBinaryExtractor {
        pub calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        pub extracted_text: String,
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
                table_markdown: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_zip_entry_count_rejects_excessive_entries() {
        use std::io::Cursor;
        let mut buf = Vec::new();
        {
            let mut zip = zip::ZipWriter::new(Cursor::new(&mut buf));
            let opts: zip::write::FileOptions = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            for i in 0..(MAX_ZIP_ENTRIES + 1) {
                zip.start_file(format!("e{i}.txt"), opts)
                    .expect("start entry");
            }
            zip.finish().expect("finish zip");
        }
        let err = guard_zip_entry_count(&buf).expect_err("excess entries must be rejected");
        assert!(matches!(err, LensError::Validation(_)), "got {err:?}");
    }

    #[test]
    fn guard_zip_entry_count_allows_small_zip_and_passes_non_zip() {
        use std::io::Cursor;
        let mut buf = Vec::new();
        {
            let mut zip = zip::ZipWriter::new(Cursor::new(&mut buf));
            let opts: zip::write::FileOptions = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("only.txt", opts).expect("start entry");
            zip.finish().expect("finish zip");
        }
        guard_zip_entry_count(&buf).expect("small zip allowed");
        guard_zip_entry_count(b"not a zip at all").expect("non-zip passes through");
    }

    fn assert_text_kind_matches_parse_blocks(kind_str: &str, source_kind: SourceKind, src: &str) {
        let extractor = extractor_for(kind_str).expect("extractor for known kind");
        let out = extractor
            .extract(src.as_bytes())
            .expect("text/MD extraction is infallible for valid UTF-8");

        assert_eq!(out.extracted_text, src, "extracted_text must equal input");

        let expected = parse_blocks(src, source_kind);
        assert_eq!(
            out.blocks, expected,
            "blocks must match parse_blocks exactly"
        );

        assert_eq!(
            out.anchors.len(),
            out.blocks.len(),
            "one anchor per block (index-aligned)"
        );
        for a in &out.anchors {
            assert_eq!(*a, SourceAnchor::Text, "text/MD anchors are all `Text`");
        }

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
        let result = extractor_for("pdf");
        assert!(
            result.is_ok(),
            "extractor_for(\"pdf\") must resolve an extractor"
        );
    }

    #[test]
    fn docx_kind_resolves_to_extractor() {
        let result = extractor_for("docx");
        assert!(
            result.is_ok(),
            "extractor_for(\"docx\") must resolve an extractor"
        );
    }

    #[test]
    fn url_kind_resolves_to_extractor() {
        let result = extractor_for("url");
        assert!(
            result.is_ok(),
            "extractor_for(\"url\") must resolve an extractor"
        );
    }

    #[test]
    fn text_extractor_rejects_invalid_utf8() {
        let extractor = extractor_for("text").unwrap();
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
        // Regression guard: wire shapes locked before Phase 2.5c.
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
        // Regression guard: wire shapes locked before issue #77.
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
    fn source_anchor_audio_roundtrips_through_serde_json() {
        let a = SourceAnchor::Audio {
            start_second: 12.5,
            end_second: 47.25,
        };
        let json = serde_json::to_string(&a).expect("serialize audio anchor");
        let back: SourceAnchor = serde_json::from_str(&json).expect("deserialize audio anchor");
        assert_eq!(a, back, "Audio anchor must round-trip through serde_json");
    }

    #[test]
    fn source_anchor_audio_wire_shape() {
        assert_eq!(
            serde_json::to_string(&SourceAnchor::Audio {
                start_second: 1.5,
                end_second: 2.25,
            })
            .unwrap(),
            r#"{"kind":"Audio","start_second":1.5,"end_second":2.25}"#
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

    #[test]
    fn xlsx_kind_resolves_to_extractor() {
        assert!(extractor_for("xlsx").is_ok());
    }

    #[test]
    fn xls_kind_resolves_to_extractor() {
        assert!(extractor_for("xls").is_ok());
    }

    #[test]
    fn csv_kind_resolves_to_extractor() {
        assert!(extractor_for("csv").is_ok());
    }
}
