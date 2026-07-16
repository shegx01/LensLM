use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::config::VoiceConfig;
use crate::dialogue::Turn;
use crate::error::LensError;
use crate::tts::sidecar::TtsSidecar;
use crate::tts::{AudioBuffer, Gender, TtsBackend, TtsProvider, TtsProviderInfo, TtsVoice};

/// A Qwen3-TTS CustomVoice preset: a fixed speaker selected by `id` whose delivery
/// is steered by an `instruct` string. No reference clip and no transcript —
/// CustomVoice is a preset+instruct engine, not a cloning one.
pub struct QwenVoice {
    pub id: &'static str,
    pub display_name: &'static str,
    pub gender: Gender,
    pub instruct: &'static str,
}

/// Default instruct applied to every preset until per-preset tuning lands: an
/// energetic podcast-host delivery (the same string benchmarked in the spike).
const DEFAULT_INSTRUCT: &str = "Upbeat, energetic podcast host, conversational and lively.";

/// The four surfaced Qwen3-TTS CustomVoice presets (the model supports more via
/// `get_supported_speakers()`). Ids are the model's canonical lowercase speaker
/// ids; the sidecar resolves them case-insensitively.
pub static QWEN_VOICES: &[QwenVoice] = &[
    QwenVoice {
        id: "dylan",
        display_name: "Dylan",
        gender: Gender::Male,
        instruct: DEFAULT_INSTRUCT,
    },
    QwenVoice {
        id: "aiden",
        display_name: "Aiden",
        gender: Gender::Male,
        instruct: DEFAULT_INSTRUCT,
    },
    QwenVoice {
        id: "serena",
        display_name: "Serena",
        gender: Gender::Female,
        instruct: DEFAULT_INSTRUCT,
    },
    QwenVoice {
        // Canonical model speaker id is "ono_anna" (not "anna") — do not normalize.
        id: "ono_anna",
        display_name: "Anna",
        gender: Gender::Female,
        instruct: DEFAULT_INSTRUCT,
    },
];

/// Resolves a preset voice by id (used by `src-tauri` to map a `VoiceRef::Named(id)`
/// to its speaker id + instruct string).
pub fn qwen_voice(id: &str) -> Option<&'static QwenVoice> {
    QWEN_VOICES.iter().find(|v| v.id == id)
}

/// In-process adapter for Qwen3-TTS CustomVoice. Owns no model weights: every turn
/// is delegated to an out-of-process MLX sidecar (`src-tauri`), so `lens-core` stays
/// headless. Voices are the fixed presets ([`QWEN_VOICES`]).
pub struct QwenLocalAdapter {
    sidecar: Arc<dyn TtsSidecar>,
}

impl QwenLocalAdapter {
    pub fn new(sidecar: Arc<dyn TtsSidecar>) -> Self {
        Self { sidecar }
    }
}

#[async_trait]
impl TtsProvider for QwenLocalAdapter {
    fn info(&self) -> TtsProviderInfo {
        TtsProviderInfo {
            backend: TtsBackend::Qwen3Local,
            model: "qwen3-tts-customvoice".to_string(),
        }
    }

    fn voices(&self) -> Vec<TtsVoice> {
        QWEN_VOICES
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
        let adapter = QwenLocalAdapter::new(Arc::new(MockSidecar::new(None)));
        assert_eq!(adapter.info().backend, TtsBackend::Qwen3Local);
        assert_eq!(adapter.info().model, "qwen3-tts-customvoice");

        let voices = adapter.voices();
        assert_eq!(voices.len(), 4);
        let ids: Vec<&str> = voices.iter().map(|v| v.id.as_str()).collect();
        assert_eq!(ids, ["dylan", "aiden", "serena", "ono_anna"]);
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
    fn qwen_voice_lookup_resolves_preset_and_instruct() {
        let v = qwen_voice("serena").expect("known voice");
        assert_eq!(v.gender, Gender::Female);
        assert_eq!(v.display_name, "Serena");
        assert!(!v.instruct.is_empty());
        assert_eq!(
            qwen_voice("ono_anna").expect("known voice").display_name,
            "Anna"
        );
        assert!(qwen_voice("nope").is_none());
    }

    #[tokio::test]
    async fn synthesizes_multi_turn_at_24khz_with_phase_sequence() {
        let adapter = QwenLocalAdapter::new(Arc::new(MockSidecar::new(None)));
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
        let adapter = QwenLocalAdapter::new(Arc::new(MockSidecar::new(Some(1))));
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
}
