//! Google multi-speaker TTS adapter (#40) via the Gemini API `generateContent`
//! surface (`generativelanguage.googleapis.com`) — the Google dialogue path callable
//! with a plain `x-goog-api-key` (the Cloud TTS `text:synthesize` product is
//! OAuth-only and deliberately unused). Submits the whole chunk as a labelled
//! transcript with a `multiSpeakerVoiceConfig`; the response is base64 raw S16LE PCM
//! (`audio/L16;rate=24000`). Per-line emotion is an inline bracketed cue.

use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::config::VoiceConfig;
use crate::dialogue::{Emotion, Speaker, Turn};
use crate::error::LensError;
use crate::tts::Gender;
use crate::tts::audio::{self, AudioBuffer};

/// Curated Gemini prebuilt voice names. `gender` is a UX display bucket only (Gemini
/// exposes no gender attribute). A user-supplied voice name overrides these.
pub const GEMINI_VOICES: &[(&str, &str, Gender)] = &[
    ("Kore", "Kore", Gender::Female),
    ("Aoede", "Aoede", Gender::Female),
    ("Leda", "Leda", Gender::Female),
    ("Puck", "Puck", Gender::Male),
    ("Charon", "Charon", Gender::Male),
    ("Fenrir", "Fenrir", Gender::Male),
];

fn default_voice(speaker: Speaker) -> &'static str {
    match speaker {
        Speaker::Host => "Kore",
        Speaker::Guest => "Puck",
    }
}

fn speaker_label(speaker: Speaker) -> &'static str {
    match speaker {
        Speaker::Host => "Host",
        Speaker::Guest => "Guest",
    }
}

fn emotion_cue(emotion: Emotion) -> Option<&'static str> {
    match emotion {
        Emotion::Neutral => None,
        Emotion::Laugh => Some("[laughing]"),
        Emotion::Sigh => Some("[sighing]"),
        Emotion::Excited => Some("[excited]"),
        Emotion::Thoughtful => Some("[thoughtful]"),
    }
}

fn line(turn: &Turn) -> String {
    let label = speaker_label(turn.speaker);
    match turn.emotion.and_then(emotion_cue) {
        Some(cue) => format!("{label}: {cue} {}", turn.text),
        None => format!("{label}: {}", turn.text),
    }
}

/// Submitted length of one transcript line ("Host: [cue] text"), so scene-chunking
/// measures the real request size, not the raw text.
pub(crate) fn sized_len(turn: &Turn) -> usize {
    line(turn).chars().count() + 1
}

#[derive(Serialize)]
struct GenerateRequest {
    contents: Vec<Content>,
    #[serde(rename = "generationConfig")]
    generation_config: GenerationConfig,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<TextPart>,
}

#[derive(Serialize)]
struct TextPart {
    text: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig {
    response_modalities: Vec<&'static str>,
    speech_config: SpeechConfig,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SpeechConfig {
    multi_speaker_voice_config: MultiSpeaker,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MultiSpeaker {
    speaker_voice_configs: Vec<SpeakerVoiceConfig>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SpeakerVoiceConfig {
    speaker: String,
    voice_config: PrebuiltWrap,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PrebuiltWrap {
    prebuilt_voice_config: Prebuilt,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Prebuilt {
    voice_name: String,
}

#[derive(Deserialize)]
struct GenResponse {
    #[serde(default)]
    candidates: Vec<Candidate>,
}

#[derive(Deserialize)]
struct Candidate {
    #[serde(default)]
    content: Option<CandidateContent>,
}

#[derive(Deserialize)]
struct CandidateContent {
    #[serde(default)]
    parts: Vec<RespPart>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RespPart {
    #[serde(default)]
    inline_data: Option<InlineData>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct InlineData {
    #[serde(default)]
    mime_type: String,
    data: String,
}

/// Parses the sample rate out of an `audio/L16;codec=pcm;rate=24000` mime type.
fn parse_rate(mime: &str) -> Option<u32> {
    mime.split(';')
        .find_map(|p| p.trim().strip_prefix("rate="))
        .and_then(|r| r.parse().ok())
}

/// Renders one scene chunk as a single multi-speaker `generateContent` request. The
/// whole chunk is one request so cross-turn dynamics are preserved; the base64 L16
/// PCM response is decoded to a mono buffer at its declared rate.
pub async fn render_dialogue_chunk(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    voices: &VoiceConfig,
    chunk: &[Turn],
) -> Result<AudioBuffer, LensError> {
    let transcript = chunk.iter().map(line).collect::<Vec<_>>().join("\n");

    // A speaker config per distinct speaker present, in first-appearance order.
    // Gemini caps multi-speaker at 2 — our Speaker enum (Host/Guest) guarantees it.
    let mut present: Vec<Speaker> = Vec::new();
    for turn in chunk {
        if !present.contains(&turn.speaker) {
            present.push(turn.speaker);
        }
    }
    let speaker_voice_configs = present
        .iter()
        .map(|sp| {
            let vref = match sp {
                Speaker::Host => &voices.host,
                Speaker::Guest => &voices.guest,
            };
            let voice = super::resolve_voice_with(vref, *sp, default_voice)?;
            Ok(SpeakerVoiceConfig {
                speaker: speaker_label(*sp).to_string(),
                voice_config: PrebuiltWrap {
                    prebuilt_voice_config: Prebuilt { voice_name: voice },
                },
            })
        })
        .collect::<Result<Vec<_>, LensError>>()?;

    let body = GenerateRequest {
        contents: vec![Content {
            parts: vec![TextPart { text: transcript }],
        }],
        generation_config: GenerationConfig {
            response_modalities: vec!["AUDIO"],
            speech_config: SpeechConfig {
                multi_speaker_voice_config: MultiSpeaker {
                    speaker_voice_configs,
                },
            },
        },
    };

    let url = format!(
        "{}/v1beta/models/{}:generateContent",
        base_url.trim_end_matches('/'),
        model
    );
    let resp = client
        .post(url)
        .header("x-goog-api-key", api_key)
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            tracing::warn!(
                timeout = e.is_timeout(),
                connect = e.is_connect(),
                "gemini tts request failed"
            );
            LensError::Network("cloud TTS request failed".into())
        })?;

    let status = resp.status();
    if !status.is_success() {
        return Err(super::map_status_error(status.as_u16()));
    }
    let bytes = super::read_body_capped(resp).await?;
    let parsed: GenResponse = serde_json::from_slice(&bytes).map_err(|e| {
        tracing::warn!(error = %e, "gemini tts response parse failed");
        LensError::Tts("cloud TTS returned an unreadable response".into())
    })?;

    // A safety-blocked or text-only 200 has empty/absent candidates or a non-audio
    // part — surface as Tts, never an index panic (`.first()`, not `[0]`).
    let inline = parsed
        .candidates
        .first()
        .and_then(|c| c.content.as_ref())
        .and_then(|c| c.parts.first())
        .and_then(|p| p.inline_data.as_ref())
        .ok_or_else(|| LensError::Tts("cloud TTS returned no audio".into()))?;

    let pcm = base64::engine::general_purpose::STANDARD
        .decode(inline.data.as_bytes())
        .map_err(|_| LensError::Tts("cloud TTS returned undecodable audio".into()))?;
    let rate = parse_rate(&inline.mime_type).unwrap_or(audio::TARGET_RATE);
    audio::decode_pcm16_mono(&pcm, rate)
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
    fn line_labels_speaker_and_prepends_cue() {
        assert_eq!(line(&t(Speaker::Host, "hi", None)), "Host: hi");
        assert_eq!(
            line(&t(Speaker::Guest, "yo", Some(Emotion::Excited))),
            "Guest: [excited] yo"
        );
    }

    #[test]
    fn parse_rate_extracts_from_mime() {
        assert_eq!(parse_rate("audio/L16;codec=pcm;rate=24000"), Some(24_000));
        assert_eq!(parse_rate("audio/L16"), None);
    }

    #[test]
    fn no_audio_response_maps_to_tts_not_panic() {
        // Empty candidates (safety block) must not index-panic.
        let parsed: GenResponse = serde_json::from_str(r#"{"candidates":[]}"#).unwrap();
        let inline = parsed
            .candidates
            .first()
            .and_then(|c| c.content.as_ref())
            .and_then(|c| c.parts.first())
            .and_then(|p| p.inline_data.as_ref());
        assert!(inline.is_none());
    }

    #[test]
    fn default_voices_differ_per_speaker() {
        assert_ne!(default_voice(Speaker::Host), default_voice(Speaker::Guest));
    }
}
