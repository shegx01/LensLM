//! Speech-to-text (ASR) abstractions: the async [`AsrEngine`] trait, its core
//! value types ([`TranscriptSegment`], [`Lang`], [`TranscribeConfig`]), the
//! [`AsrBackend`] selector enum, and a model-free [`MockAsrEngine`] for tests.
//!
//! Headless: no tauri/UI/OS-window deps. Input PCM is #41's 16 kHz mono f32.
//! The Whisper engine, router, and registry arrive in later units.

pub mod registry;
mod router;
#[cfg(feature = "local-whisper")]
pub mod whisper;

pub use registry::{
    DEFAULT_WHISPER_MODEL_ID, WHISPER_REGISTRY, WhisperModelSpec, download_whisper_model,
    resolve_whisper, whisper_model_downloaded, whisper_model_path,
};
pub use router::{MIN_MACOS_FOR_APPLE_ASR, Platform, select_asr_backend};
#[cfg(feature = "local-whisper")]
pub use whisper::WhisperEngine;

use serde::{Deserialize, Serialize};

use crate::LensError;

/// One transcribed span. Timestamps are in seconds; they anchor Citations (#43).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub text: String,
    pub start_second: f32,
    pub end_second: f32,
}

/// Language selector. Enum, not a magic string (strong-typing rule). A minimal
/// common set plus an `Other(String)` escape hatch; per-source language UI is #43.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Lang {
    En,
    De,
    Fr,
    Es,
    It,
    Pt,
    Nl,
    Ru,
    Zh,
    Ja,
    Ko,
    /// Any language outside the common set, by BCP-47-ish code (e.g. `"ar"`).
    Other(String),
}

/// Per-call engine config. Multilingual by default: `language == None` ⇒
/// auto-detect and transcribe in the source language. `translate == true` ⇒
/// translate to English (the Whisper translate task).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TranscribeConfig {
    pub language: Option<Lang>,
    pub translate: bool,
}

/// Which speech-to-text backend runs a transcription. Strong-typed; selection is
/// explicit/router-driven, so there is deliberately no `Default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsrBackend {
    AppleNative,
    LocalWhisper,
}

impl AsrBackend {
    /// Storage/config token for this backend.
    pub fn as_str(&self) -> &'static str {
        match self {
            AsrBackend::AppleNative => "apple_native",
            AsrBackend::LocalWhisper => "local_whisper",
        }
    }

    /// Parses a config token. Unknown/empty yields `None` (no default backend —
    /// selection is router-driven), mirroring `EmbeddingBackend::from_opt_str`'s
    /// null-tolerant, non-`FromStr` shape.
    pub fn from_opt_str(s: Option<&str>) -> Option<AsrBackend> {
        match s {
            Some("apple_native") => Some(AsrBackend::AppleNative),
            Some("local_whisper") => Some(AsrBackend::LocalWhisper),
            _ => None,
        }
    }
}

/// Object-safe, async speech-to-text seam. `Send + Sync` so it is held behind
/// `Arc<dyn AsrEngine>`/`Box<dyn AsrEngine>` in the engine. Input is #41's 16 kHz
/// mono f32 PCM; `progress_tx` (if any) receives values in `0.0..=1.0`.
#[async_trait::async_trait]
pub trait AsrEngine: Send + Sync {
    async fn transcribe_pcm(
        &self,
        pcm: &[f32],
        config: &TranscribeConfig,
        progress_tx: Option<tokio::sync::mpsc::UnboundedSender<f32>>,
    ) -> Result<Vec<TranscriptSegment>, LensError>;
}

/// Deterministic, model-free engine for offline tests (mirrors `CountingEmbedder`).
/// Returns caller-supplied canned segments and emits a couple of progress values.
#[cfg(any(test, feature = "test-util"))]
pub struct MockAsrEngine {
    segments: Vec<TranscriptSegment>,
}

#[cfg(any(test, feature = "test-util"))]
impl MockAsrEngine {
    pub fn new(segments: Vec<TranscriptSegment>) -> Self {
        Self { segments }
    }
}

#[cfg(any(test, feature = "test-util"))]
#[async_trait::async_trait]
impl AsrEngine for MockAsrEngine {
    async fn transcribe_pcm(
        &self,
        _pcm: &[f32],
        _config: &TranscribeConfig,
        progress_tx: Option<tokio::sync::mpsc::UnboundedSender<f32>>,
    ) -> Result<Vec<TranscriptSegment>, LensError> {
        if let Some(tx) = progress_tx {
            let _ = tx.send(0.5);
            let _ = tx.send(1.0);
        }
        Ok(self.segments.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asr_backend_from_opt_str() {
        assert_eq!(
            AsrBackend::from_opt_str(Some("apple_native")),
            Some(AsrBackend::AppleNative)
        );
        assert_eq!(
            AsrBackend::from_opt_str(Some("local_whisper")),
            Some(AsrBackend::LocalWhisper)
        );
        assert_eq!(AsrBackend::from_opt_str(None), None);
        assert_eq!(AsrBackend::from_opt_str(Some("")), None);
        assert_eq!(AsrBackend::from_opt_str(Some("bogus")), None);
        // as_str round-trips through from_opt_str.
        for b in [AsrBackend::AppleNative, AsrBackend::LocalWhisper] {
            assert_eq!(AsrBackend::from_opt_str(Some(b.as_str())), Some(b));
        }
    }

    #[test]
    fn transcript_segment_serde_roundtrip() {
        let seg = TranscriptSegment {
            text: "hello world".to_string(),
            start_second: 0.5,
            end_second: 1.75,
        };
        let json = serde_json::to_string(&seg).expect("serialize segment");
        let back: TranscriptSegment = serde_json::from_str(&json).expect("deserialize segment");
        assert_eq!(seg, back);
    }

    #[tokio::test]
    async fn mock_engine_returns_canned_segments() {
        let canned = vec![
            TranscriptSegment {
                text: "one".to_string(),
                start_second: 0.0,
                end_second: 1.0,
            },
            TranscriptSegment {
                text: "two".to_string(),
                start_second: 1.0,
                end_second: 2.5,
            },
        ];
        let engine = MockAsrEngine::new(canned.clone());

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let out = engine
            .transcribe_pcm(&[0.0_f32; 8], &TranscribeConfig::default(), Some(tx))
            .await
            .expect("mock transcribe");

        assert_eq!(out, canned);
        for seg in &out {
            assert!(seg.start_second < seg.end_second);
        }

        let mut progress = Vec::new();
        while let Ok(p) = rx.try_recv() {
            progress.push(p);
        }
        assert!(!progress.is_empty(), "mock should emit progress");
    }

    #[test]
    fn asr_engine_is_object_safe() {
        let _: Box<dyn AsrEngine> = Box::new(MockAsrEngine::new(vec![TranscriptSegment {
            text: "obj-safe".to_string(),
            start_second: 0.0,
            end_second: 0.1,
        }]));
    }
}
