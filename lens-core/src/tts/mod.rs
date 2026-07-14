use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::config::{TtsConfig, VoiceConfig};
use crate::dialogue::{DialogueScript, Emotion, Speaker, Turn};
use crate::error::LensError;

pub mod audio;
mod kokoro;
pub mod orpheus;
pub mod registry;
pub mod sidecar;
pub mod snac;

pub use audio::AudioBuffer;
pub(crate) use audio::write_wav_16bit;
pub use kokoro::{
    DownloadProgress, Gender, KOKORO_MODEL_FILENAME, KOKORO_MODEL_RELPATH, KOKORO_MODEL_URL,
    TtsVoice, download_kokoro_model, kokoro_model_path, list_tts_voices,
};
pub use registry::{
    TTS_REGISTRY, TtsModelSpec, download_tts_model, resolve_tts, tts_model_downloaded,
    tts_model_path,
};
pub use sidecar::TtsSidecar;

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
    pub fn as_str(&self) -> &'static str {
        match self {
            TtsBackend::Kokoro => "kokoro",
            TtsBackend::Orpheus => "orpheus",
            TtsBackend::MossLocal => "moss_local",
            TtsBackend::MossTtsd => "moss_ttsd",
            TtsBackend::Cloud(_) => "cloud",
        }
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CloudTtsKind {
    #[default]
    OpenAiCompatible,
    Deepgram,
    ElevenLabs,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtsProviderInfo {
    pub backend: TtsBackend,
    pub model: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsPhase {
    Synthesizing { turn: usize, total: usize },
    Stitching,
    Encoding,
}

const CANCELLED_MSG: &str = "tts synthesis cancelled";

type TurnSynthFuture<'a> = std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<AudioBuffer, LensError>> + Send + 'a>,
>;

pub(crate) async fn synthesize_and_stitch<'t, F>(
    turns: &'t [Turn],
    on_phase: &(dyn Fn(TtsPhase) + Send + Sync),
    cancel: &CancellationToken,
    mut synth_turn: F,
) -> Result<AudioBuffer, LensError>
where
    F: FnMut(&'t Turn) -> TurnSynthFuture<'t>,
{
    let total = turns.len();
    let mut buffers: Vec<(Speaker, AudioBuffer)> = Vec::with_capacity(total);
    for (i, turn) in turns.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(LensError::Cancelled(CANCELLED_MSG.into()));
        }
        on_phase(TtsPhase::Synthesizing { turn: i + 1, total });
        let buf = tokio::select! {
            r = synth_turn(turn) => r?,
            _ = cancel.cancelled() => {
                return Err(LensError::Cancelled(CANCELLED_MSG.into()));
            }
        };
        buffers.push((turn.speaker, buf));
    }
    on_phase(TtsPhase::Stitching);
    audio::stitch_turns(&buffers)
}

#[async_trait]
pub trait TtsProvider: Send + Sync {
    fn info(&self) -> TtsProviderInfo;

    async fn synthesize_turn(
        &self,
        turn: &Turn,
        voices: &VoiceConfig,
        cancel: &CancellationToken,
    ) -> Result<AudioBuffer, LensError>;

    // `&dyn Fn` (not `impl Fn`) for object-safety; `+ Sync` because it is held across `.await` in a `Send` future.
    async fn synthesize_script(
        &self,
        script: &DialogueScript,
        voices: &VoiceConfig,
        on_phase: &(dyn Fn(TtsPhase) + Send + Sync),
        cancel: &CancellationToken,
    ) -> Result<AudioBuffer, LensError> {
        synthesize_and_stitch(&script.turns, on_phase, cancel, |turn| {
            self.synthesize_turn(turn, voices, cancel)
        })
        .await
    }
}

/// Resolves a [`TtsProvider`] for `backend`. Construction is cheap: an embedded
/// provider (Orpheus) holds only its model paths + config and lazy-loads weights
/// on first synth (see [`orpheus::OrpheusAdapter`]). `data_dir` supplies the
/// model relpaths; availability (whether those files exist) is a separate cheap
/// probe (`tts_model_downloaded`), so a missing artifact surfaces as a lazy-load
/// `LensError::Tts`, never a silent `None`.
pub fn resolve_tts_provider(
    backend: TtsBackend,
    _cfg: &TtsConfig,
    data_dir: &Path,
) -> Option<Arc<dyn TtsProvider>> {
    match backend {
        TtsBackend::Orpheus => {
            let orpheus = tts_model_path(data_dir, "orpheus")?;
            let snac = tts_model_path(data_dir, "snac")?;
            Some(Arc::new(orpheus::OrpheusAdapter::new(orpheus, snac)))
        }
        TtsBackend::Kokoro => None,
        TtsBackend::MossLocal => None,
        TtsBackend::MossTtsd => None,
        TtsBackend::Cloud(_) => None,
    }
}

pub fn emotion_tag(emotion: Emotion, backend: TtsBackend) -> Option<String> {
    match backend {
        TtsBackend::Orpheus => match emotion {
            Emotion::Laugh => Some("<laugh>".to_string()),
            Emotion::Sigh => Some("<sigh>".to_string()),
            Emotion::Neutral | Emotion::Excited | Emotion::Thoughtful => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_returns_none_for_non_embedded_backends() {
        let cfg = TtsConfig::default();
        let data_dir = Path::new("/data");
        for backend in [
            TtsBackend::Kokoro,
            TtsBackend::MossLocal,
            TtsBackend::MossTtsd,
            TtsBackend::Cloud(CloudTtsKind::OpenAiCompatible),
            TtsBackend::Cloud(CloudTtsKind::Deepgram),
            TtsBackend::Cloud(CloudTtsKind::ElevenLabs),
        ] {
            assert!(resolve_tts_provider(backend, &cfg, data_dir).is_none());
        }
    }

    #[test]
    fn resolve_returns_orpheus_adapter_cheaply() {
        // Cheap construct: an adapter is returned even when the weights are
        // absent (paths only, no load); availability is a separate file probe.
        let cfg = TtsConfig::default();
        let provider = resolve_tts_provider(TtsBackend::Orpheus, &cfg, Path::new("/data"))
            .expect("orpheus resolves to an adapter");
        assert_eq!(provider.info().backend, TtsBackend::Orpheus);
    }

    #[test]
    fn emotion_tag_orpheus_table() {
        assert_eq!(emotion_tag(Emotion::Neutral, TtsBackend::Orpheus), None);
        assert_eq!(
            emotion_tag(Emotion::Laugh, TtsBackend::Orpheus).as_deref(),
            Some("<laugh>")
        );
        assert_eq!(
            emotion_tag(Emotion::Sigh, TtsBackend::Orpheus).as_deref(),
            Some("<sigh>")
        );
        assert_eq!(emotion_tag(Emotion::Excited, TtsBackend::Orpheus), None);
        assert_eq!(emotion_tag(Emotion::Thoughtful, TtsBackend::Orpheus), None);
    }

    #[test]
    fn emotion_tag_none_for_non_orpheus_backends() {
        for emotion in [
            Emotion::Neutral,
            Emotion::Laugh,
            Emotion::Sigh,
            Emotion::Excited,
            Emotion::Thoughtful,
        ] {
            assert!(emotion_tag(emotion, TtsBackend::Kokoro).is_none());
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
        assert_eq!(
            TtsBackend::Cloud(CloudTtsKind::ElevenLabs).as_str(),
            "cloud"
        );
        assert_eq!(
            TtsBackend::from_opt_str(Some("cloud")),
            TtsBackend::Cloud(CloudTtsKind::default())
        );
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
