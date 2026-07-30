//! Cloud TTS tier (#195): an opt-in [`TtsProvider`] that synthesizes via a cloud
//! provider, gated by select-Cloud + a non-empty API key. Mirrors the `asr/cloud/`
//! module layout + per-request internals (bearer auth, status→error mapping,
//! wiremock tests) but NOT its engine loop — the adapter implements only
//! [`TtsProvider::synthesize_turn`] and inherits the shared `synthesize_script`
//! (stitch + edge-fades + phase events).

pub mod elevenlabs;
pub mod google;
pub mod openai_compat;
pub mod ssml;

use crate::dialogue::DialogueScript;

use std::time::Duration;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::config::{VoiceConfig, VoiceRef};
use crate::dialogue::{Speaker, Turn};
use crate::error::LensError;
use crate::tts::audio::AudioBuffer;
use crate::tts::{
    CloudTtsKind, Gender, TtsBackend, TtsPhase, TtsProvider, TtsProviderInfo, TtsVoice,
};

const CANCELLED_MSG: &str = "tts synthesis cancelled";

/// Default OpenAI-compatible TTS model when `TtsConfig.model` is empty.
pub const DEFAULT_CLOUD_TTS_MODEL: &str = "gpt-4o-mini-tts";

/// Default ElevenLabs dialogue model. Text-to-Dialogue is only supported on
/// `eleven_v3` — do not substitute another id.
pub const ELEVENLABS_DIALOGUE_MODEL: &str = "eleven_v3";

/// Default Gemini multi-speaker TTS model (Gemini API `generateContent`).
pub const GEMINI_TTS_MODEL: &str = "gemini-2.5-flash-preview-tts";

/// ElevenLabs Text-to-Dialogue caps the combined length of all dialogue lines at
/// 2000 characters per request; scene-chunking is sized against this.
pub const ELEVENLABS_DIALOGUE_CHAR_LIMIT: usize = 2000;

/// Conservative per-request character budget for Gemini `generateContent`. The real
/// bound is a ~32k-token context; chunking well under it keeps every chunk safely
/// within one request.
pub const GEMINI_DIALOGUE_CHAR_LIMIT: usize = 5000;

/// Default API base URL for a cloud kind, applied when the stored base URL is
/// empty so a provider selection works without the user pasting an endpoint.
pub fn default_base_url(kind: CloudTtsKind) -> &'static str {
    match kind {
        CloudTtsKind::OpenAiCompatible => "https://api.openai.com",
        CloudTtsKind::ElevenLabs => "https://api.elevenlabs.io",
        CloudTtsKind::GoogleCloud => "https://generativelanguage.googleapis.com",
        CloudTtsKind::Deepgram => "https://api.deepgram.com",
    }
}

/// Default model id for a cloud kind, applied when `TtsConfig.model` is empty.
pub fn default_model(kind: CloudTtsKind) -> &'static str {
    match kind {
        CloudTtsKind::OpenAiCompatible | CloudTtsKind::Deepgram => DEFAULT_CLOUD_TTS_MODEL,
        CloudTtsKind::ElevenLabs => ELEVENLABS_DIALOGUE_MODEL,
        CloudTtsKind::GoogleCloud => GEMINI_TTS_MODEL,
    }
}

/// Per-request input character budget for a dialogue kind (used by scene-chunking).
/// Per-turn providers (OpenAI/Deepgram) are not chunked, so they return `None`.
pub fn dialogue_char_limit(kind: CloudTtsKind) -> Option<usize> {
    match kind {
        CloudTtsKind::ElevenLabs => Some(ELEVENLABS_DIALOGUE_CHAR_LIMIT),
        CloudTtsKind::GoogleCloud => Some(GEMINI_DIALOGUE_CHAR_LIMIT),
        CloudTtsKind::OpenAiCompatible | CloudTtsKind::Deepgram => None,
    }
}

/// Mirror the LLM path's bounded timeouts (`llm.rs`): a cloud TTS turn is a single
/// short dialogue turn, comparable to an LLM completion in latency.
const CLOUD_TTS_CONNECT_TIMEOUT: Duration = Duration::from_secs(1);
const CLOUD_TTS_TIMEOUT: Duration = Duration::from_secs(30);

/// Curated OpenAI TTS voice set. `gender` is a UX host/guest bucket only — OpenAI
/// does NOT expose a gender attribute; it is our display grouping for voice pickers.
pub const OPENAI_VOICES: &[(&str, &str, Gender)] = &[
    ("alloy", "Alloy", Gender::Female),
    ("ash", "Ash", Gender::Male),
    ("ballad", "Ballad", Gender::Male),
    ("coral", "Coral", Gender::Female),
    ("echo", "Echo", Gender::Male),
    ("fable", "Fable", Gender::Male),
    ("onyx", "Onyx", Gender::Male),
    ("nova", "Nova", Gender::Female),
    ("sage", "Sage", Gender::Female),
    ("shimmer", "Shimmer", Gender::Female),
    ("verse", "Verse", Gender::Male),
];

/// Opt-in cloud text-to-speech adapter. Dispatches by [`CloudTtsKind`]; only
/// `OpenAiCompatible` is wired — other kinds fail with a clear "not yet supported".
///
/// No `#[derive(Debug)]`: the struct holds a plaintext `api_key` that must never
/// reach logs or IPC.
pub struct CloudTtsAdapter {
    kind: CloudTtsKind,
    base_url: String,
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl CloudTtsAdapter {
    /// Builds an adapter with the hardened, no-redirect HTTP client. Used in production.
    pub fn new(
        kind: CloudTtsKind,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        let client = crate::http::hardened_client(CLOUD_TTS_CONNECT_TIMEOUT, CLOUD_TTS_TIMEOUT);
        Self::with_client(kind, base_url, api_key, model, client)
    }

    /// Builds an adapter with a caller-supplied client, so tests can point `base_url`
    /// at a wiremock server without the no-redirect/timeout policy.
    pub fn with_client(
        kind: CloudTtsKind,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        client: reqwest::Client,
    ) -> Self {
        Self {
            kind,
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            client,
        }
    }
}

/// Resolves a turn's [`VoiceRef`] to a cloud voice id. Cloning
/// (`VoiceRef::Reference`) is rejected (mirrors `orpheus.rs`); an unset named voice
/// falls back to a per-speaker default; a non-empty name passes through (free-text
/// voice id honored).
/// Resolves a turn's [`VoiceRef`] to a provider voice id, using `default` for an
/// unset named voice (the default set differs per provider — OpenAI voice names,
/// ElevenLabs ids, Gemini prebuilt names). Cloning (`VoiceRef::Reference`) is
/// unsupported; a non-empty name passes through (free-text voice id honored).
pub(crate) fn resolve_voice_with(
    voice: &VoiceRef,
    speaker: Speaker,
    default: impl Fn(Speaker) -> &'static str,
) -> Result<String, LensError> {
    match voice {
        VoiceRef::Reference { .. } => Err(LensError::Tts(
            "voice cloning (VoiceRef::Reference) is unsupported by the cloud TTS backend; \
             use a named voice id"
                .into(),
        )),
        VoiceRef::Named(name) if name.is_empty() => Ok(default(speaker).to_string()),
        VoiceRef::Named(name) => Ok(name.clone()),
    }
}

fn resolve_voice(voice: &VoiceRef, speaker: Speaker) -> Result<String, LensError> {
    resolve_voice_with(voice, speaker, default_voice)
}

fn default_voice(speaker: Speaker) -> &'static str {
    match speaker {
        Speaker::Host => "alloy",
        Speaker::Guest => "onyx",
    }
}

/// Reads an HTTP response body with a running byte cap (the [`MAX_TURN_WAV_BYTES`]
/// ceiling), so a hostile/buggy endpoint can't force an unbounded allocation even
/// when Content-Length is absent or understated. Shared by every cloud adapter.
pub(crate) async fn read_body_capped(resp: reqwest::Response) -> Result<Vec<u8>, LensError> {
    let cap = crate::tts::audio::MAX_TURN_WAV_BYTES;
    if let Some(len) = resp.content_length()
        && len > cap
    {
        return Err(LensError::Validation("cloud TTS response too large".into()));
    }
    let mut resp = resp;
    let mut bytes: Vec<u8> = Vec::new();
    loop {
        match resp.chunk().await {
            Ok(Some(chunk)) => {
                if bytes.len() as u64 + chunk.len() as u64 > cap {
                    return Err(LensError::Validation("cloud TTS response too large".into()));
                }
                bytes.extend_from_slice(&chunk);
            }
            Ok(None) => break,
            Err(e) => {
                tracing::warn!(
                    timeout = e.is_timeout(),
                    connect = e.is_connect(),
                    "cloud TTS response read failed"
                );
                return Err(LensError::Tts("cloud TTS response read failed".into()));
            }
        }
    }
    Ok(bytes)
}

/// Maps a provider HTTP status to a [`LensError`] without leaking provider internals.
/// Misconfiguration (401/403) + oversize (413) → `Validation`; connectivity-class
/// (429/5xx) → `Network`; else `Tts`. Mirrors `asr/cloud/mod.rs::map_status_error`.
pub(crate) fn map_status_error(status: u16) -> LensError {
    match status {
        401 | 403 => LensError::Validation("cloud TTS rejected the API key".into()),
        413 => LensError::Validation("cloud TTS request payload too large".into()),
        429 => LensError::Network("cloud TTS rate limited".into()),
        500..=599 => LensError::Network(format!("cloud TTS provider error ({status})")),
        _ => LensError::Tts(format!("cloud TTS unexpected status ({status})")),
    }
}

/// Whole-exchange synthesis loop for dialogue engines (#40) — keyless sibling of
/// [`crate::tts::synthesize_and_stitch`]. Scene-chunks `turns` to fit `char_limit`
/// (sized via `sized_len` so provider markup counts), renders each chunk as one
/// dialogue request, then stitches. `Synthesizing.turn` is the chunk index here; a
/// render is raced against `cancel`. Never degrades to per-turn rendering (AC4).
pub(crate) async fn synthesize_chunks<'t, F, Fut>(
    turns: &'t [Turn],
    char_limit: usize,
    sized_len: impl Fn(&Turn) -> usize,
    on_phase: &(dyn Fn(TtsPhase) + Send + Sync),
    cancel: &CancellationToken,
    mut render_chunk: F,
) -> Result<AudioBuffer, LensError>
where
    F: FnMut(&'t [Turn]) -> Fut,
    Fut: std::future::Future<Output = Result<AudioBuffer, LensError>> + Send,
{
    let chunks = crate::tts::chunk::scene_chunks(turns, char_limit, sized_len)?;
    let total = chunks.len();
    let mut buffers: Vec<AudioBuffer> = Vec::with_capacity(total);
    for (i, range) in chunks.into_iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(LensError::Cancelled(CANCELLED_MSG.into()));
        }
        on_phase(TtsPhase::Synthesizing { turn: i + 1, total });
        let buf = tokio::select! {
            r = render_chunk(&turns[range]) => r?,
            _ = cancel.cancelled() => {
                return Err(LensError::Cancelled(CANCELLED_MSG.into()));
            }
        };
        buffers.push(buf);
    }
    on_phase(TtsPhase::Stitching);
    crate::tts::audio::stitch_chunks(&buffers)
}

#[async_trait]
impl TtsProvider for CloudTtsAdapter {
    fn info(&self) -> TtsProviderInfo {
        TtsProviderInfo {
            backend: TtsBackend::Cloud(self.kind),
            model: self.model.clone(),
        }
    }

    fn voices(&self) -> Vec<TtsVoice> {
        let list: &[(&str, &str, Gender)] = match self.kind {
            CloudTtsKind::ElevenLabs => elevenlabs::ELEVENLABS_VOICES,
            CloudTtsKind::GoogleCloud => google::GEMINI_VOICES,
            CloudTtsKind::OpenAiCompatible | CloudTtsKind::Deepgram => OPENAI_VOICES,
        };
        list.iter()
            .map(|&(id, name, gender)| TtsVoice::new(id, name, gender))
            .collect()
    }

    async fn synthesize_turn(
        &self,
        turn: &Turn,
        voices: &VoiceConfig,
        cancel: &CancellationToken,
    ) -> Result<AudioBuffer, LensError> {
        if self.api_key.is_empty() {
            return Err(LensError::Validation(
                "no cloud TTS API key configured".into(),
            ));
        }

        let voice_ref = match turn.speaker {
            Speaker::Host => &voices.host,
            Speaker::Guest => &voices.guest,
        };
        // Resolve + validate the voice before any network work so an unsupported
        // clone reference errors up front.
        let voice = resolve_voice(voice_ref, turn.speaker)?;

        if cancel.is_cancelled() {
            return Err(LensError::Cancelled("tts synthesis cancelled".into()));
        }

        match self.kind {
            CloudTtsKind::OpenAiCompatible => {
                openai_compat::synthesize_turn(
                    &self.client,
                    &self.base_url,
                    &self.model,
                    &self.api_key,
                    &voice,
                    turn,
                )
                .await
            }
            CloudTtsKind::Deepgram => Err(LensError::Tts(
                "Deepgram cloud TTS is not yet supported".into(),
            )),
            // Dialogue-only engines render the whole exchange; they override
            // `synthesize_script` and never route a single turn here via
            // `synthesize_overview`. A direct call is a programming/config error.
            CloudTtsKind::ElevenLabs | CloudTtsKind::GoogleCloud => Err(LensError::Validation(
                "this cloud voice provider renders whole dialogues, not single turns".into(),
            )),
        }
    }

    async fn synthesize_script(
        &self,
        script: &DialogueScript,
        voices: &VoiceConfig,
        on_phase: &(dyn Fn(TtsPhase) + Send + Sync),
        cancel: &CancellationToken,
    ) -> Result<AudioBuffer, LensError> {
        match self.kind {
            // Per-turn engines keep the shared default (verbatim #195 behavior): a
            // single-utterance API has no cross-turn dynamics to preserve.
            CloudTtsKind::OpenAiCompatible | CloudTtsKind::Deepgram => {
                crate::tts::synthesize_and_stitch(&script.turns, on_phase, cancel, |turn| {
                    self.synthesize_turn(turn, voices, cancel)
                })
                .await
            }
            CloudTtsKind::ElevenLabs => {
                if self.api_key.is_empty() {
                    return Err(LensError::Validation(
                        "no cloud TTS API key configured".into(),
                    ));
                }
                synthesize_chunks(
                    &script.turns,
                    ELEVENLABS_DIALOGUE_CHAR_LIMIT,
                    elevenlabs::sized_len,
                    on_phase,
                    cancel,
                    |chunk| {
                        elevenlabs::render_dialogue_chunk(
                            &self.client,
                            &self.base_url,
                            &self.api_key,
                            &self.model,
                            voices,
                            chunk,
                        )
                    },
                )
                .await
            }
            CloudTtsKind::GoogleCloud => {
                if self.api_key.is_empty() {
                    return Err(LensError::Validation(
                        "no cloud TTS API key configured".into(),
                    ));
                }
                synthesize_chunks(
                    &script.turns,
                    GEMINI_DIALOGUE_CHAR_LIMIT,
                    google::sized_len,
                    on_phase,
                    cancel,
                    |chunk| {
                        google::render_dialogue_chunk(
                            &self.client,
                            &self.base_url,
                            &self.api_key,
                            &self.model,
                            voices,
                            chunk,
                        )
                    },
                )
                .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_status_error_matrix() {
        assert!(matches!(map_status_error(401), LensError::Validation(_)));
        assert!(matches!(map_status_error(403), LensError::Validation(_)));
        assert!(matches!(map_status_error(413), LensError::Validation(_)));
        assert!(matches!(map_status_error(429), LensError::Network(_)));
        assert!(matches!(map_status_error(500), LensError::Network(_)));
        assert!(matches!(map_status_error(503), LensError::Network(_)));
        assert!(matches!(map_status_error(418), LensError::Tts(_)));
    }

    #[test]
    fn resolve_voice_passes_named_and_defaults_unset() {
        assert_eq!(
            resolve_voice(&VoiceRef::Named("nova".into()), Speaker::Host).unwrap(),
            "nova"
        );
        assert_eq!(
            resolve_voice(&VoiceRef::Named(String::new()), Speaker::Host).unwrap(),
            "alloy"
        );
        assert_eq!(
            resolve_voice(&VoiceRef::Named(String::new()), Speaker::Guest).unwrap(),
            "onyx"
        );
    }

    #[test]
    fn resolve_voice_rejects_reference_clone() {
        let r = VoiceRef::Reference {
            clip_path: "/x.wav".into(),
            transcript: "hi".into(),
        };
        assert!(matches!(
            resolve_voice(&r, Speaker::Host),
            Err(LensError::Tts(_))
        ));
    }

    #[test]
    fn openai_voices_have_display_metadata() {
        let voices = CloudTtsAdapter::new(
            CloudTtsKind::OpenAiCompatible,
            "https://api.openai.com",
            "k",
            "gpt-4o-mini-tts",
        )
        .voices();
        assert_eq!(voices.len(), OPENAI_VOICES.len());
        assert!(voices.iter().any(|v| v.id == "alloy"));
        assert!(voices.iter().any(|v| v.id == "onyx"));
    }

    #[test]
    fn default_base_url_and_model_per_kind() {
        assert_eq!(
            default_base_url(CloudTtsKind::OpenAiCompatible),
            "https://api.openai.com"
        );
        assert_eq!(
            default_base_url(CloudTtsKind::ElevenLabs),
            "https://api.elevenlabs.io"
        );
        assert_eq!(
            default_base_url(CloudTtsKind::GoogleCloud),
            "https://generativelanguage.googleapis.com"
        );
        assert_eq!(default_model(CloudTtsKind::ElevenLabs), "eleven_v3");
        assert_eq!(
            default_model(CloudTtsKind::OpenAiCompatible),
            DEFAULT_CLOUD_TTS_MODEL
        );
        assert_eq!(dialogue_char_limit(CloudTtsKind::ElevenLabs), Some(2000));
        assert_eq!(dialogue_char_limit(CloudTtsKind::OpenAiCompatible), None);
    }

    fn host_turn(text: &str) -> Turn {
        Turn {
            speaker: Speaker::Host,
            text: text.to_string(),
            emotion: None,
            source_ids: Vec::new(),
        }
    }

    #[tokio::test]
    async fn synthesize_chunks_emits_chunk_phases_and_stitches() {
        // 3 turns of 800 chars, limit 2000 -> chunks [0..2],[2..3] => 2 renders.
        let turns: Vec<Turn> = (0..3).map(|_| host_turn(&"a".repeat(800))).collect();
        let phases = std::sync::Mutex::new(Vec::new());
        let on_phase = |p: TtsPhase| phases.lock().unwrap().push(p);
        let cancel = CancellationToken::new();
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let out = synthesize_chunks(
            &turns,
            2000,
            |t: &Turn| t.text.chars().count(),
            &on_phase,
            &cancel,
            |_chunk| {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async move {
                    Ok(AudioBuffer::mono(
                        vec![0.1; 100],
                        crate::tts::audio::TARGET_RATE,
                    ))
                }
            },
        )
        .await
        .expect("chunk synth");

        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
        let recorded = phases.lock().unwrap();
        assert_eq!(recorded[0], TtsPhase::Synthesizing { turn: 1, total: 2 });
        assert_eq!(recorded[1], TtsPhase::Synthesizing { turn: 2, total: 2 });
        assert_eq!(recorded[2], TtsPhase::Stitching);
        // Two 100-sample chunks joined by one scene gap.
        assert!(
            out.samples.len() > 200,
            "stitched len {}",
            out.samples.len()
        );
        assert_eq!(out.sample_rate, crate::tts::audio::TARGET_RATE);
    }

    #[tokio::test]
    async fn synthesize_chunks_cancel_before_first_chunk() {
        let turns = vec![host_turn("x")];
        let cancel = CancellationToken::new();
        cancel.cancel();
        let noop = |_p: TtsPhase| {};
        let err = synthesize_chunks(
            &turns,
            2000,
            |t: &Turn| t.text.chars().count(),
            &noop,
            &cancel,
            |_chunk| async {
                Ok(AudioBuffer::mono(
                    vec![0.0; 10],
                    crate::tts::audio::TARGET_RATE,
                ))
            },
        )
        .await
        .expect_err("cancelled");
        assert!(matches!(err, LensError::Cancelled(_)));
    }

    #[tokio::test]
    async fn synthesize_chunks_single_over_limit_turn_is_validation() {
        let turns = vec![host_turn(&"z".repeat(2500))];
        let cancel = CancellationToken::new();
        let noop = |_p: TtsPhase| {};
        let err = synthesize_chunks(
            &turns,
            2000,
            |t: &Turn| t.text.chars().count(),
            &noop,
            &cancel,
            |_chunk| async {
                Ok(AudioBuffer::mono(
                    vec![0.0; 10],
                    crate::tts::audio::TARGET_RATE,
                ))
            },
        )
        .await
        .expect_err("over-limit turn");
        assert!(matches!(err, LensError::Validation(_)));
    }
}
