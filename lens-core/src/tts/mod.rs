//! Text-to-speech provider seam + audio pipeline (#190).
//!
//! Replaces the download-only Kokoro scaffold with a provider abstraction: the
//! object-safe [`TtsProvider`] trait (mirroring `LlmProvider`), the
//! [`TtsBackend`]/[`CloudTtsKind`] selector enums, the [`resolve_tts_provider`]
//! router, the [`registry`] model catalog, the [`audio`] pipeline
//! ([`AudioBuffer`] + resample + speaker-aware stitch + WAV), the [`sidecar`]
//! injection seam, and [`TtsPhase`] progress. No concrete synthesis engine ships
//! in #190 — every resolution path returns `None`/a typed no-backend
//! [`LensError::Tts`], so #191 (Orpheus) is a pure match-arm addition. Kokoro
//! stays vestigial (`kokoro`), removed in #192.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::config::{TtsConfig, VoiceConfig};
use crate::dialogue::{DialogueScript, Emotion, Speaker, Turn};
use crate::error::LensError;

pub mod audio;
mod kokoro;
pub mod registry;
pub mod sidecar;

pub use audio::{AudioBuffer, stitch_turns, write_wav_16bit};
pub use kokoro::{
    DownloadProgress, Gender, KOKORO_MODEL_FILENAME, KOKORO_MODEL_RELPATH, KOKORO_MODEL_URL,
    TtsVoice, download_kokoro_model, kokoro_model_path, list_tts_voices,
};
pub use registry::{
    TTS_REGISTRY, TtsModelSpec, download_tts_model, resolve_tts, tts_model_downloaded,
    tts_model_path,
};
pub use sidecar::TtsSidecar;

/// The synthesis backend, selecting a [`TtsProvider`]. Strong-typed, not a magic
/// string ([[strong-typing-no-stringly-domain]]). `Cloud` wraps the specific
/// cloud API kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsBackend {
    #[default]
    Kokoro,
    Orpheus,
    MossLocal,
    MossTtsd,
    Cloud(CloudTtsKind),
}

impl TtsBackend {
    /// Stable storage/label token. `Cloud` collapses to `"cloud"` — the specific
    /// [`CloudTtsKind`] rides in `TtsConfig.cloud`, not this discriminant.
    pub fn as_str(&self) -> &'static str {
        match self {
            TtsBackend::Kokoro => "kokoro",
            TtsBackend::Orpheus => "orpheus",
            TtsBackend::MossLocal => "moss_local",
            TtsBackend::MossTtsd => "moss_ttsd",
            TtsBackend::Cloud(_) => "cloud",
        }
    }

    /// Parses a stored token; `None`/empty/unknown resolves to the default
    /// (`Kokoro`). Deliberately INFALLIBLE (an absent value is a normal case).
    /// `"cloud"` resolves to `Cloud` with the default [`CloudTtsKind`].
    pub fn from_opt_str(s: Option<&str>) -> Self {
        match s.unwrap_or("") {
            "orpheus" => TtsBackend::Orpheus,
            "moss_local" => TtsBackend::MossLocal,
            "moss_ttsd" => TtsBackend::MossTtsd,
            "cloud" => TtsBackend::Cloud(CloudTtsKind::default()),
            _ => TtsBackend::default(),
        }
    }
}

/// The specific cloud TTS API dialect for [`TtsBackend::Cloud`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CloudTtsKind {
    #[default]
    OpenAiCompatible,
    Deepgram,
    ElevenLabs,
}

/// Static metadata about a resolved provider ([`TtsProvider::info`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtsProviderInfo {
    pub backend: TtsBackend,
    pub model: String,
}

/// Honest phase markers streamed over the `synthesize_overview` progress channel.
/// `Synthesizing` fires once per turn; `Stitching` once before concatenation;
/// `Encoding` exactly once (emitted by the engine, not the default
/// [`TtsProvider::synthesize_script`]) immediately before the WAV write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsPhase {
    Synthesizing { turn: usize, total: usize },
    Stitching,
    Encoding,
}

/// An async, object-safe TTS backend held behind `Arc<dyn TtsProvider>`. Mirrors
/// [`LlmProvider`](crate::llm::LlmProvider): concrete `synthesize_turn` plus a
/// defaulted `synthesize_script` that loops turns and stitches.
#[async_trait]
pub trait TtsProvider: Send + Sync {
    /// Backend + model identity.
    fn info(&self) -> TtsProviderInfo;

    /// Synthesizes one dialogue turn into a canonical [`AudioBuffer`], racing the
    /// call against `cancel`.
    async fn synthesize_turn(
        &self,
        turn: &Turn,
        voices: &VoiceConfig,
        cancel: &CancellationToken,
    ) -> Result<AudioBuffer, LensError>;

    /// Synthesizes the whole script into one stitched overview buffer. The default
    /// loops turns emitting `Synthesizing { turn, total }`, races each
    /// `synthesize_turn` against `cancel`, then emits `Stitching` and returns
    /// [`stitch_turns`]. `on_phase` is a `&dyn Fn(TtsPhase) + Send + Sync` (not a
    /// generic `impl Fn`) because an object-safe trait method cannot take a generic
    /// closure, and a `&dyn Fn` held across `.await` in a `Send` future needs the
    /// referent `+ Sync`. The engine emits `Encoding` separately, so this method
    /// never does (no double phase event).
    async fn synthesize_script(
        &self,
        script: &DialogueScript,
        voices: &VoiceConfig,
        on_phase: &(dyn Fn(TtsPhase) + Send + Sync),
        cancel: &CancellationToken,
    ) -> Result<AudioBuffer, LensError> {
        let total = script.turns.len();
        let mut buffers: Vec<(Speaker, AudioBuffer)> = Vec::with_capacity(total);
        for (i, turn) in script.turns.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err(LensError::Cancelled("tts synthesis cancelled".into()));
            }
            on_phase(TtsPhase::Synthesizing { turn: i + 1, total });
            let buf = tokio::select! {
                r = self.synthesize_turn(turn, voices, cancel) => r?,
                _ = cancel.cancelled() => {
                    return Err(LensError::Cancelled("tts synthesis cancelled".into()));
                }
            };
            buffers.push((turn.speaker, buf));
        }
        on_phase(TtsPhase::Stitching);
        stitch_turns(&buffers)
    }
}

/// Resolves a [`TtsBackend`] to a concrete [`TtsProvider`]. **Wildcard-free**
/// exhaustive match so enum growth and #191 are compiler-guided: every backend
/// returns `None` in #190 (no adapter ships). #191 replaces `Orpheus => None` with
/// `Orpheus => Some(Arc::new(OrpheusAdapter::new(cfg)))`, etc.
pub fn resolve_tts_provider(backend: TtsBackend, _cfg: &TtsConfig) -> Option<Arc<dyn TtsProvider>> {
    match backend {
        TtsBackend::Kokoro => None,
        TtsBackend::Orpheus => None,
        TtsBackend::MossLocal => None,
        TtsBackend::MossTtsd => None,
        TtsBackend::Cloud(_) => None,
    }
}

/// Maps a dialogue [`Emotion`] to a backend-specific inline tag. A scaffold in
/// #190: always `None` (drop-if-unsupported). Real per-backend tables land with
/// the adapters (#191 Orpheus tags, #195 SSML).
pub fn emotion_tag(_emotion: Emotion, _backend: TtsBackend) -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_returns_none_for_every_backend() {
        let cfg = TtsConfig::default();
        for backend in [
            TtsBackend::Kokoro,
            TtsBackend::Orpheus,
            TtsBackend::MossLocal,
            TtsBackend::MossTtsd,
            TtsBackend::Cloud(CloudTtsKind::OpenAiCompatible),
            TtsBackend::Cloud(CloudTtsKind::Deepgram),
            TtsBackend::Cloud(CloudTtsKind::ElevenLabs),
        ] {
            assert!(resolve_tts_provider(backend, &cfg).is_none());
        }
    }

    #[test]
    fn emotion_tag_is_none_scaffold() {
        for emotion in [
            Emotion::Neutral,
            Emotion::Laugh,
            Emotion::Sigh,
            Emotion::Excited,
            Emotion::Thoughtful,
        ] {
            assert!(emotion_tag(emotion, TtsBackend::Orpheus).is_none());
        }
    }

    #[test]
    fn backend_default_is_kokoro() {
        assert_eq!(TtsBackend::default(), TtsBackend::Kokoro);
    }

    #[test]
    fn backend_as_str_and_from_opt_str_round_trip() {
        for b in [
            TtsBackend::Kokoro,
            TtsBackend::Orpheus,
            TtsBackend::MossLocal,
            TtsBackend::MossTtsd,
        ] {
            assert_eq!(TtsBackend::from_opt_str(Some(b.as_str())), b);
        }
        // Cloud collapses to the default cloud kind on the string round-trip.
        assert_eq!(
            TtsBackend::Cloud(CloudTtsKind::ElevenLabs).as_str(),
            "cloud"
        );
        assert_eq!(
            TtsBackend::from_opt_str(Some("cloud")),
            TtsBackend::Cloud(CloudTtsKind::default())
        );
        // None / empty / unknown → default.
        assert_eq!(TtsBackend::from_opt_str(None), TtsBackend::Kokoro);
        assert_eq!(TtsBackend::from_opt_str(Some("")), TtsBackend::Kokoro);
        assert_eq!(TtsBackend::from_opt_str(Some("nope")), TtsBackend::Kokoro);
    }

    #[test]
    fn backend_serde_round_trips_including_cloud() {
        for b in [
            TtsBackend::Kokoro,
            TtsBackend::Orpheus,
            TtsBackend::Cloud(CloudTtsKind::ElevenLabs),
        ] {
            let json = serde_json::to_string(&b).unwrap();
            let back: TtsBackend = serde_json::from_str(&json).unwrap();
            assert_eq!(b, back);
        }
    }

    /// A fake provider proving the defaulted `synthesize_script` compiles behind
    /// `Arc<dyn TtsProvider>` and stitches per-turn buffers.
    struct FakeProvider;

    #[async_trait]
    impl TtsProvider for FakeProvider {
        fn info(&self) -> TtsProviderInfo {
            TtsProviderInfo {
                backend: TtsBackend::Kokoro,
                model: "fake".to_string(),
            }
        }
        async fn synthesize_turn(
            &self,
            _turn: &Turn,
            _voices: &VoiceConfig,
            _cancel: &CancellationToken,
        ) -> Result<AudioBuffer, LensError> {
            Ok(AudioBuffer::mono(vec![0.5; 1000], audio::TARGET_RATE))
        }
    }

    #[tokio::test]
    async fn default_synthesize_script_stitches_behind_arc_dyn() {
        let provider: Arc<dyn TtsProvider> = Arc::new(FakeProvider);
        let script = DialogueScript {
            turns: vec![
                Turn {
                    speaker: Speaker::Host,
                    text: "a".into(),
                    emotion: None,
                    source_ids: Vec::new(),
                },
                Turn {
                    speaker: Speaker::Guest,
                    text: "b".into(),
                    emotion: None,
                    source_ids: Vec::new(),
                },
            ],
        };
        let voices = VoiceConfig::default();
        let cancel = CancellationToken::new();
        let phases = std::sync::Mutex::new(Vec::new());
        let on_phase = |p: TtsPhase| phases.lock().unwrap().push(p);
        let out = provider
            .synthesize_script(&script, &voices, &on_phase, &cancel)
            .await
            .unwrap();
        // Two 1000-sample turns + one speaker-change gap (10800) @ 24 kHz.
        assert_eq!(out.samples.len(), 1000 + 10_800 + 1000);
        let recorded = phases.lock().unwrap();
        assert_eq!(recorded[0], TtsPhase::Synthesizing { turn: 1, total: 2 });
        assert_eq!(recorded[1], TtsPhase::Synthesizing { turn: 2, total: 2 });
        assert_eq!(recorded[2], TtsPhase::Stitching);
    }

    #[tokio::test]
    async fn synthesize_script_honors_cancel() {
        let provider: Arc<dyn TtsProvider> = Arc::new(FakeProvider);
        let script = DialogueScript {
            turns: vec![Turn {
                speaker: Speaker::Host,
                text: "a".into(),
                emotion: None,
                source_ids: Vec::new(),
            }],
        };
        let voices = VoiceConfig::default();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let noop = |_p: TtsPhase| {};
        let err = provider
            .synthesize_script(&script, &voices, &noop, &cancel)
            .await
            .unwrap_err();
        assert!(matches!(err, LensError::Cancelled(_)));
    }
}
