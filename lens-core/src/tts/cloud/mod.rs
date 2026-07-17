//! Cloud TTS tier (#195): an opt-in [`TtsProvider`] that synthesizes via a cloud
//! provider, gated by select-Cloud + a non-empty API key. Mirrors the `asr/cloud/`
//! module layout + per-request internals (bearer auth, status→error mapping,
//! wiremock tests) but NOT its engine loop — the adapter implements only
//! [`TtsProvider::synthesize_turn`] and inherits the shared `synthesize_script`
//! (stitch + edge-fades + phase events).

pub mod openai_compat;
pub mod ssml;

use std::time::Duration;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::config::{VoiceConfig, VoiceRef};
use crate::dialogue::{Speaker, Turn};
use crate::error::LensError;
use crate::tts::audio::AudioBuffer;
use crate::tts::{CloudTtsKind, Gender, TtsBackend, TtsProvider, TtsProviderInfo, TtsVoice};

/// Default OpenAI-compatible TTS model when `TtsConfig.model` is empty.
pub const DEFAULT_CLOUD_TTS_MODEL: &str = "gpt-4o-mini-tts";

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
fn resolve_voice(voice: &VoiceRef, speaker: Speaker) -> Result<String, LensError> {
    match voice {
        VoiceRef::Reference { .. } => Err(LensError::Tts(
            "voice cloning (VoiceRef::Reference) is unsupported by the cloud TTS backend; \
             use a named voice id"
                .into(),
        )),
        VoiceRef::Named(name) if name.is_empty() => Ok(default_voice(speaker).to_string()),
        VoiceRef::Named(name) => Ok(name.clone()),
    }
}

fn default_voice(speaker: Speaker) -> &'static str {
    match speaker {
        Speaker::Host => "alloy",
        Speaker::Guest => "onyx",
    }
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

#[async_trait]
impl TtsProvider for CloudTtsAdapter {
    fn info(&self) -> TtsProviderInfo {
        TtsProviderInfo {
            backend: TtsBackend::Cloud(self.kind),
            model: self.model.clone(),
        }
    }

    fn voices(&self) -> Vec<TtsVoice> {
        OPENAI_VOICES
            .iter()
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
            CloudTtsKind::ElevenLabs => Err(LensError::Tts(
                "ElevenLabs cloud TTS is not yet supported".into(),
            )),
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
}
