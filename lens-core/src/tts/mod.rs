use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::config::{TtsConfig, VoiceConfig};
use crate::dialogue::{DialogueScript, Emotion, Speaker, Turn};
use crate::error::LensError;

pub mod audio;
pub mod moss;
pub mod orpheus;
pub mod registry;
pub mod sidecar;
pub mod snac;

pub(crate) use audio::write_wav_16bit;
pub use audio::{AudioBuffer, read_wav_mono16};
pub use moss::{
    MOSS_REFERENCE_VOICES, MossLocalAdapter, MossReferenceVoice, MossTtsdAdapter,
    moss_reference_voice,
};
pub use registry::{
    ArtifactKind, TTS_REGISTRY, TtsModelSpec, download_tts_model, resolve_tts,
    tts_model_downloaded, tts_model_path, unpack_zip,
};
pub use sidecar::TtsSidecar;

/// Speaker gender. Serializes lowercase to match the `'male' | 'female'` union in the Svelte
/// client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Gender {
    Male,
    Female,
}

/// One selectable named voice. Frozen IPC contract — mirrored in the Svelte client as
/// `TtsVoice { id, name, gender }`. The catalog is adapter-driven via [`TtsProvider::voices`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtsVoice {
    pub id: String,
    pub name: String,
    pub gender: Gender,
}

impl TtsVoice {
    pub(crate) fn new(id: &str, name: &str, gender: Gender) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            gender,
        }
    }
}

/// Download progress. Frozen IPC contract — mirrored in the Svelte client as
/// `{ received, total, done }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub received: u64,
    pub total: Option<u64>,
    pub done: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsBackend {
    #[default]
    Orpheus,
    MossLocal,
    MossTtsd,
    Cloud(CloudTtsKind),
}

impl TtsBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
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

    /// Registry ids of every model artifact this backend needs on disk to be
    /// usable. Non-embedded backends (cloud, not-yet-wired local) return `&[]`.
    pub fn required_model_ids(&self) -> &'static [&'static str] {
        match self {
            TtsBackend::Orpheus => &["orpheus", "snac"],
            TtsBackend::MossLocal => &["moss_sidecar_bin", "moss_model"],
            TtsBackend::MossTtsd | TtsBackend::Cloud(_) => &[],
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

    /// Adapter-driven named-voice catalog. Empty when the backend enumerates no
    /// fixed voices (e.g. a clone-only backend).
    fn voices(&self) -> Vec<TtsVoice>;

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

/// Resolves a [`TtsProvider`] for `backend`, given an optional injected `sidecar`.
/// This is the single dispatch path; both [`resolve_tts_provider`] (the `None`
/// wrapper) and `synthesize_overview` route through it so the two entry points
/// cannot diverge. Construction is cheap: an embedded provider (Orpheus) holds
/// only its model paths and lazy-loads weights on first synth. Sidecar-backed
/// backends (MossLocal) resolve only when a sidecar is present; a missing artifact
/// surfaces as a lazy-load `LensError::Tts`, never a silent `None`.
pub fn resolve_tts_provider_full(
    backend: TtsBackend,
    _cfg: &TtsConfig,
    data_dir: &Path,
    sidecar: Option<Arc<dyn TtsSidecar>>,
) -> Option<Arc<dyn TtsProvider>> {
    match backend {
        TtsBackend::Orpheus => {
            let orpheus = tts_model_path(data_dir, "orpheus")?;
            let snac = tts_model_path(data_dir, "snac")?;
            Some(Arc::new(orpheus::OrpheusAdapter::new(orpheus, snac)))
        }
        TtsBackend::MossLocal => {
            sidecar.map(|s| Arc::new(moss::MossLocalAdapter::new(s)) as Arc<dyn TtsProvider>)
        }
        TtsBackend::MossTtsd => None,
        TtsBackend::Cloud(_) => None,
    }
}

/// Thin wrapper over [`resolve_tts_provider_full`] with no sidecar. Sidecar-backed
/// backends (MossLocal) therefore return `None` here by design; call `_full` with
/// the injected sidecar when one is required (see `synthesize_overview`). Keeps the
/// 3-arg signature used by `system.rs` voice resolution and existing tests.
pub fn resolve_tts_provider(
    backend: TtsBackend,
    cfg: &TtsConfig,
    data_dir: &Path,
) -> Option<Arc<dyn TtsProvider>> {
    resolve_tts_provider_full(backend, cfg, data_dir, None)
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
    fn resolve_full_moss_local_needs_sidecar() {
        let cfg = TtsConfig::default();
        let data_dir = Path::new("/data");
        // No sidecar → the wrapper's behavior: MossLocal is None.
        assert!(resolve_tts_provider_full(TtsBackend::MossLocal, &cfg, data_dir, None).is_none());
        assert!(resolve_tts_provider(TtsBackend::MossLocal, &cfg, data_dir).is_none());

        // With an injected sidecar → the MossLocal adapter.
        let sidecar: Arc<dyn TtsSidecar> = Arc::new(NoopSidecar);
        let provider =
            resolve_tts_provider_full(TtsBackend::MossLocal, &cfg, data_dir, Some(sidecar))
                .expect("moss_local resolves with a sidecar");
        assert_eq!(provider.info().backend, TtsBackend::MossLocal);
        assert_eq!(provider.info().model, "moss-tts-local-int8");
    }

    #[test]
    fn resolve_full_moss_ttsd_is_none_even_with_sidecar() {
        let cfg = TtsConfig::default();
        let sidecar: Arc<dyn TtsSidecar> = Arc::new(NoopSidecar);
        assert!(
            resolve_tts_provider_full(
                TtsBackend::MossTtsd,
                &cfg,
                Path::new("/data"),
                Some(sidecar)
            )
            .is_none()
        );
    }

    #[test]
    fn resolve_orpheus_via_wrapper_ignores_absent_sidecar() {
        // Orpheus resolves through the None-wrapper with no sidecar injected.
        let cfg = TtsConfig::default();
        let provider = resolve_tts_provider(TtsBackend::Orpheus, &cfg, Path::new("/data"))
            .expect("orpheus resolves without a sidecar");
        assert_eq!(provider.info().backend, TtsBackend::Orpheus);
    }

    struct NoopSidecar;

    #[async_trait]
    impl crate::tts::sidecar::TtsSidecar for NoopSidecar {
        async fn start(&self) -> Result<(), LensError> {
            Ok(())
        }
        async fn stop(&self) -> Result<(), LensError> {
            Ok(())
        }
        async fn health(&self) -> bool {
            true
        }
        async fn synthesize_turn(
            &self,
            _turn: &Turn,
            _voices: &VoiceConfig,
            _cancel: &CancellationToken,
        ) -> Result<AudioBuffer, LensError> {
            Ok(AudioBuffer::mono(vec![0.0; 8], audio::TARGET_RATE))
        }
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
            assert!(emotion_tag(emotion, TtsBackend::MossLocal).is_none());
        }
    }

    #[test]
    fn backend_default_is_orpheus() {
        assert_eq!(TtsBackend::default(), TtsBackend::Orpheus);
    }

    #[test]
    fn gender_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&Gender::Male).unwrap(), "\"male\"");
        assert_eq!(
            serde_json::to_string(&Gender::Female).unwrap(),
            "\"female\""
        );
    }

    #[test]
    fn backend_as_str_and_from_opt_str_round_trip() {
        for b in [
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
        assert_eq!(TtsBackend::from_opt_str(None), TtsBackend::Orpheus);
        assert_eq!(TtsBackend::from_opt_str(Some("")), TtsBackend::Orpheus);
        assert_eq!(TtsBackend::from_opt_str(Some("nope")), TtsBackend::Orpheus);
    }

    #[test]
    fn backend_serde_round_trips_including_cloud() {
        for b in [
            TtsBackend::Orpheus,
            TtsBackend::MossLocal,
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
                backend: TtsBackend::Orpheus,
                model: "fake".to_string(),
            }
        }
        fn voices(&self) -> Vec<TtsVoice> {
            Vec::new()
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
