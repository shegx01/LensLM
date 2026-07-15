use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::config::VoiceConfig;
use crate::dialogue::Turn;
use crate::error::LensError;
use crate::tts::sidecar::TtsSidecar;
use crate::tts::{AudioBuffer, Gender, TtsBackend, TtsProvider, TtsProviderInfo, TtsVoice};

/// A bundled MOSS reference clip: the clone-only backend synthesizes each turn
/// against one of these. Transcripts are static engine data; the clip file *path*
/// is resolved by `src-tauri` (it owns the bundled resource dir), so only the
/// filename lives here.
pub struct MossReferenceVoice {
    pub id: &'static str,
    pub display_name: &'static str,
    pub gender: Gender,
    pub transcript: &'static str,
    pub clip_filename: &'static str,
}

/// The four bundled MOSS reference voices. Transcripts are provisional pending a
/// human clip-alignment audition (#193 A0/audition).
pub static MOSS_REFERENCE_VOICES: &[MossReferenceVoice] = &[
    MossReferenceVoice {
        id: "librivox-clarke",
        display_name: "David (measured)",
        gender: Gender::Male,
        clip_filename: "librivox-clarke.wav",
        transcript: "To Sherlock Holmes she is always the woman. I have seldom heard him mention \
                     her under any other name. In his eyes she eclipses and predominates the \
                     whole of her sex.",
    },
    MossReferenceVoice {
        id: "librivox-chenevert",
        display_name: "Phil (warm)",
        gender: Gender::Male,
        clip_filename: "librivox-chenevert.wav",
        transcript: "Chapter One, In Jolly Kimbaloo. The King of Kimbaloo was kinda jolly, and \
                     kinda jolly was the King of Kimbaloo.",
    },
    MossReferenceVoice {
        id: "librivox-klett",
        display_name: "Elizabeth (clear)",
        gender: Gender::Female,
        clip_filename: "librivox-klett.wav",
        transcript: "No one who had ever seen Catherine Morland in her infancy would have \
                     supposed her born to be an heroine. Her situation in life, the character of \
                     her father and mother, her own person and disposition, were all equally \
                     against her.",
    },
    MossReferenceVoice {
        id: "librivox-savage",
        display_name: "Karen (expressive)",
        gender: Gender::Female,
        clip_filename: "librivox-savage.wav",
        transcript: "Paris, September, 1792. A surging, seething, murmuring crowd of beings that \
                     are human only in name, for to the eye and ear they seem naught but savage \
                     creatures, animated by vile passions and by the lust of vengeance and of \
                     hate.",
    },
];

/// Resolves a bundled reference voice by id (used by `src-tauri` to map a
/// `VoiceRef::Named(id)` to its transcript + clip filename).
pub fn moss_reference_voice(id: &str) -> Option<&'static MossReferenceVoice> {
    MOSS_REFERENCE_VOICES.iter().find(|v| v.id == id)
}

/// In-process adapter for MOSS-TTS-Local. Owns no model weights: every turn is
/// delegated to an out-of-process MLX sidecar (`src-tauri`), so `lens-core` stays
/// headless. Reuses the default [`TtsProvider::synthesize_script`] stitch/cancel
/// pipeline; voices are the bundled clone-reference clips ([`MOSS_REFERENCE_VOICES`]).
pub struct MossLocalAdapter {
    sidecar: Arc<dyn TtsSidecar>,
}

impl MossLocalAdapter {
    pub fn new(sidecar: Arc<dyn TtsSidecar>) -> Self {
        Self { sidecar }
    }
}

#[async_trait]
impl TtsProvider for MossLocalAdapter {
    fn info(&self) -> TtsProviderInfo {
        TtsProviderInfo {
            backend: TtsBackend::MossLocal,
            model: "moss-tts-local-int8".to_string(),
        }
    }

    fn voices(&self) -> Vec<TtsVoice> {
        MOSS_REFERENCE_VOICES
            .iter()
            .map(|v| TtsVoice::new(v.id, v.display_name, v.gender))
            .collect()
    }

    async fn synthesize_turn(
        &self,
        turn: &Turn,
        voices: &VoiceConfig,
        cancel: &CancellationToken,
    ) -> Result<AudioBuffer, LensError> {
        self.sidecar.synthesize_turn(turn, voices, cancel).await
    }
}

/// Placeholder for the MOSS-TTSD multi-speaker backend (#189). Registered so the
/// resolver has an explicit arm; synthesis is a typed not-implemented error rather
/// than silent dead code.
pub struct MossTtsdAdapter;

#[async_trait]
impl TtsProvider for MossTtsdAdapter {
    fn info(&self) -> TtsProviderInfo {
        TtsProviderInfo {
            backend: TtsBackend::MossTtsd,
            model: "moss-ttsd".to_string(),
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
        Err(LensError::Tts("moss-ttsd not implemented (#189)".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialogue::{DialogueScript, Speaker};
    use crate::tts::TtsPhase;
    use crate::tts::audio::TARGET_RATE;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const PER_TURN: usize = 2_400; // 0.1s @ 24 kHz
    const CHANGE_GAP: usize = 10_800; // 450 ms @ 24 kHz (speaker change)

    /// Mock sidecar returning valid 24 kHz buffers. When `cancel_on_turn` matches
    /// the (1-based) call count it cancels the shared token mid-turn and never
    /// returns — so `synthesize_and_stitch`'s `select!` resolves via cancellation.
    struct MockSidecar {
        calls: AtomicUsize,
        cancel_on_turn: Option<usize>,
    }

    impl MockSidecar {
        fn new(cancel_on_turn: Option<usize>) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                cancel_on_turn,
            }
        }
    }

    #[async_trait]
    impl TtsSidecar for MockSidecar {
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
            cancel: &CancellationToken,
        ) -> Result<AudioBuffer, LensError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if self.cancel_on_turn == Some(n) {
                cancel.cancel();
                std::future::pending::<()>().await;
            }
            Ok(AudioBuffer::mono(vec![0.1; PER_TURN], TARGET_RATE))
        }
    }

    fn two_turn_script() -> DialogueScript {
        DialogueScript {
            turns: vec![
                Turn {
                    speaker: Speaker::Host,
                    text: "hello".into(),
                    emotion: None,
                    source_ids: Vec::new(),
                },
                Turn {
                    speaker: Speaker::Guest,
                    text: "hi".into(),
                    emotion: None,
                    source_ids: Vec::new(),
                },
            ],
        }
    }

    #[tokio::test]
    async fn adapter_info_and_voices() {
        let adapter = MossLocalAdapter::new(Arc::new(MockSidecar::new(None)));
        assert_eq!(adapter.info().backend, TtsBackend::MossLocal);
        assert_eq!(adapter.info().model, "moss-tts-local-int8");

        let voices = adapter.voices();
        assert_eq!(voices.len(), 4);
        let ids: Vec<&str> = voices.iter().map(|v| v.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "librivox-clarke",
                "librivox-chenevert",
                "librivox-klett",
                "librivox-savage"
            ]
        );
        assert_eq!(
            voices.iter().filter(|v| v.gender == Gender::Male).count(),
            2
        );
        assert_eq!(
            voices.iter().filter(|v| v.gender == Gender::Female).count(),
            2
        );
    }

    #[test]
    fn reference_voice_lookup_resolves_transcript_and_clip() {
        let v = moss_reference_voice("librivox-klett").expect("known voice");
        assert_eq!(v.gender, Gender::Female);
        assert_eq!(v.clip_filename, "librivox-klett.wav");
        assert!(v.transcript.contains("Catherine Morland"));
        assert!(moss_reference_voice("nope").is_none());
    }

    #[tokio::test]
    async fn synthesizes_multi_turn_at_24khz_with_phase_sequence() {
        let adapter = MossLocalAdapter::new(Arc::new(MockSidecar::new(None)));
        let script = two_turn_script();
        let voices = VoiceConfig::default();
        let cancel = CancellationToken::new();
        let phases = Mutex::new(Vec::new());
        let on_phase = |p: TtsPhase| phases.lock().unwrap().push(p);

        let out = adapter
            .synthesize_script(&script, &voices, &on_phase, &cancel)
            .await
            .expect("mock synth succeeds");

        assert_eq!(out.sample_rate, TARGET_RATE);
        assert_eq!(out.channels, 1);
        assert_eq!(out.samples.len(), PER_TURN + CHANGE_GAP + PER_TURN);

        let recorded = phases.lock().unwrap();
        assert_eq!(recorded[0], TtsPhase::Synthesizing { turn: 1, total: 2 });
        assert_eq!(recorded[1], TtsPhase::Synthesizing { turn: 2, total: 2 });
        assert_eq!(recorded[2], TtsPhase::Stitching);
    }

    #[tokio::test]
    async fn mid_turn_cancel_yields_cancelled_then_clean_rerun() {
        // Same adapter/sidecar: turn 1 self-cancels mid-turn.
        let adapter = MossLocalAdapter::new(Arc::new(MockSidecar::new(Some(1))));
        let script = two_turn_script();
        let voices = VoiceConfig::default();
        let noop = |_p: TtsPhase| {};

        let cancel = CancellationToken::new();
        let err = adapter
            .synthesize_script(&script, &voices, &noop, &cancel)
            .await
            .unwrap_err();
        assert!(matches!(err, LensError::Cancelled(_)), "got {err:?}");

        // A fresh run (new token) against the still-warm mock succeeds — no desync.
        let cancel2 = CancellationToken::new();
        let out = adapter
            .synthesize_script(&script, &voices, &noop, &cancel2)
            .await
            .expect("subsequent run is clean");
        assert_eq!(out.sample_rate, TARGET_RATE);
        assert_eq!(out.samples.len(), PER_TURN + CHANGE_GAP + PER_TURN);
    }

    #[tokio::test]
    async fn moss_ttsd_stub_is_typed_not_implemented() {
        let adapter = MossTtsdAdapter;
        assert_eq!(adapter.info().backend, TtsBackend::MossTtsd);
        let turn = Turn {
            speaker: Speaker::Host,
            text: "x".into(),
            emotion: None,
            source_ids: Vec::new(),
        };
        let err = adapter
            .synthesize_turn(&turn, &VoiceConfig::default(), &CancellationToken::new())
            .await
            .unwrap_err();
        match err {
            LensError::Tts(m) => assert!(m.contains("#189"), "msg was {m:?}"),
            other => panic!("expected Tts, got {other:?}"),
        }
    }
}
