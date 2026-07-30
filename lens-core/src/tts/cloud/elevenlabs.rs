//! ElevenLabs Text-to-Dialogue adapter (#40): the whole-exchange cloud engine.
//! Submits an ordered list of `{text, voice_id}` lines to `POST /v1/text-to-dialogue`
//! (auth header `xi-api-key`, `output_format=pcm_24000` → raw S16LE PCM @ 24 kHz),
//! preserving cross-turn dynamics. Per-line emotion is an inline bracketed audio tag
//! prepended to the line text. Called per scene chunk by `synthesize_chunks`.

use serde::Serialize;

use crate::config::VoiceConfig;
use crate::dialogue::{Emotion, Speaker, Turn};
use crate::error::LensError;
use crate::tts::Gender;
use crate::tts::audio::{self, AudioBuffer};

/// Curated ElevenLabs preset voices (public library voice ids). `gender` is a UX
/// display bucket only. A user-supplied free-text voice id overrides these.
pub const ELEVENLABS_VOICES: &[(&str, &str, Gender)] = &[
    ("21m00Tcm4TlvDq8ikWAM", "Rachel", Gender::Female),
    ("EXAVITQu4vr4xnSDxMaL", "Bella", Gender::Female),
    ("pNInz6obpgDQGcFmaJgB", "Adam", Gender::Male),
    ("29vD33N1CtxCmqQRPOHJ", "Drew", Gender::Male),
];

fn default_voice(speaker: Speaker) -> &'static str {
    match speaker {
        Speaker::Host => "21m00Tcm4TlvDq8ikWAM",
        Speaker::Guest => "pNInz6obpgDQGcFmaJgB",
    }
}

/// Inline bracketed v3 audio tag for a per-line emotion, from the central table
/// (documented tags only — an undocumented tag is spoken literally). Prepended to the
/// line text so it scopes to that speaker.
fn emotion_to_audio_tag(emotion: Emotion) -> Option<&'static str> {
    crate::tts::emotion_render(emotion).elevenlabs
}

/// Submitted length of one line — text plus any prepended emotion tag (+ a space) —
/// so scene-chunking measures the real request size, not the raw text (#40 minor).
pub(crate) fn sized_len(turn: &Turn) -> usize {
    let tag = turn
        .emotion
        .and_then(emotion_to_audio_tag)
        .map(|t| t.len() + 1)
        .unwrap_or(0);
    turn.text.chars().count() + tag
}

fn line_text(turn: &Turn) -> String {
    match turn.emotion.and_then(emotion_to_audio_tag) {
        Some(tag) => format!("{tag} {}", turn.text),
        None => turn.text.clone(),
    }
}

#[derive(Serialize)]
struct DialogueInput {
    text: String,
    voice_id: String,
}

#[derive(Serialize)]
struct DialogueRequest<'a> {
    inputs: &'a [DialogueInput],
    model_id: &'a str,
}

/// Renders one scene chunk (a contiguous slice of turns) as a single dialogue
/// request. The whole chunk is submitted at once so cross-turn dynamics are
/// preserved; the response is raw `pcm_24000` decoded to a 24 kHz mono buffer.
pub async fn render_dialogue_chunk(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    voices: &VoiceConfig,
    chunk: &[Turn],
) -> Result<AudioBuffer, LensError> {
    let inputs = chunk
        .iter()
        .map(|turn| {
            let vref = match turn.speaker {
                Speaker::Host => &voices.host,
                Speaker::Guest => &voices.guest,
            };
            let voice_id = super::resolve_voice_with(vref, turn.speaker, default_voice)?;
            Ok(DialogueInput {
                text: line_text(turn),
                voice_id,
            })
        })
        .collect::<Result<Vec<_>, LensError>>()?;

    let body = DialogueRequest {
        inputs: &inputs,
        model_id: model,
    };
    let url = format!(
        "{}/v1/text-to-dialogue?output_format=pcm_24000",
        base_url.trim_end_matches('/')
    );
    let resp = client
        .post(url)
        .header("xi-api-key", api_key)
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            tracing::warn!(
                timeout = e.is_timeout(),
                connect = e.is_connect(),
                "elevenlabs dialogue request failed"
            );
            LensError::Network("cloud TTS request failed".into())
        })?;

    let status = resp.status();
    if !status.is_success() {
        return Err(super::map_status_error(status.as_u16()));
    }
    let bytes = super::read_body_capped(resp).await?;
    audio::decode_pcm16_mono(&bytes, audio::TARGET_RATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(speaker: Speaker, text: &str, emotion: Option<Emotion>) -> Turn {
        Turn {
            speaker,
            text: text.to_string(),
            emotion,
            source_ids: Vec::new(),
        }
    }

    #[test]
    fn sized_len_counts_prepended_tag() {
        let plain = t(Speaker::Host, "hello", None);
        assert_eq!(sized_len(&plain), 5);
        let emote = t(Speaker::Host, "hello", Some(Emotion::Laugh));
        // "[laughs]" (8) + 1 space + 5 = 14.
        assert_eq!(sized_len(&emote), 14);
    }

    #[test]
    fn line_text_prepends_only_for_non_neutral() {
        assert_eq!(line_text(&t(Speaker::Host, "hi", None)), "hi");
        assert_eq!(
            line_text(&t(Speaker::Host, "hi", Some(Emotion::Neutral))),
            "hi"
        );
        assert_eq!(
            line_text(&t(Speaker::Guest, "hi", Some(Emotion::Excited))),
            "[excited] hi"
        );
    }

    #[test]
    fn default_voices_differ_per_speaker() {
        assert_ne!(default_voice(Speaker::Host), default_voice(Speaker::Guest));
    }
}
