//! OpenAI-compatible cloud TTS adapter (#195): OpenAI / any `base_url` speaking the
//! `POST /v1/audio/speech` contract. Sends `{model, input, voice, response_format}`
//! (+ best-effort `instructions`) as JSON with `Authorization: Bearer {api_key}`,
//! and decodes the WAV response body into a mono [`AudioBuffer`].

use serde::Serialize;

use super::map_status_error;
use super::ssml::emotion_to_instruction;
use crate::dialogue::Turn;
use crate::error::LensError;
use crate::tts::audio::{self, AudioBuffer};

#[derive(Serialize)]
struct SpeechRequest<'a> {
    model: &'a str,
    input: &'a str,
    voice: &'a str,
    response_format: &'a str,
    // Best-effort delivery hint (`gpt-4o-mini-tts`); omitted when the turn has no
    // emotion so a strict compatible server never sees an unknown/empty field.
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
}

/// Synthesizes one turn via `POST {base_url}/v1/audio/speech`, requesting a WAV body.
/// The real sample rate + channel count are read back from the WAV header; the shared
/// stitch step resamples to 24 kHz mono.
pub async fn synthesize_turn(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    api_key: &str,
    voice: &str,
    turn: &Turn,
) -> Result<AudioBuffer, LensError> {
    let instructions = turn.emotion.and_then(emotion_to_instruction);
    let body = SpeechRequest {
        model,
        input: &turn.text,
        voice,
        response_format: "wav",
        instructions,
    };

    let url = format!("{}/v1/audio/speech", base_url.trim_end_matches('/'));
    let resp = client
        .post(url)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            tracing::warn!(
                timeout = e.is_timeout(),
                connect = e.is_connect(),
                "cloud TTS HTTP request failed"
            );
            LensError::Network("cloud TTS request failed".into())
        })?;

    let status = resp.status();
    if !status.is_success() {
        return Err(map_status_error(status.as_u16()));
    }

    // Pre-check before buffering the body; the authoritative cap is re-applied
    // by byte length inside `decode_wav_mono16`.
    if let Some(len) = resp.content_length()
        && len > audio::MAX_TURN_WAV_BYTES
    {
        return Err(LensError::Validation("cloud TTS response too large".into()));
    }

    let bytes = resp.bytes().await.map_err(|e| {
        tracing::warn!(
            timeout = e.is_timeout(),
            connect = e.is_connect(),
            "cloud TTS response read failed"
        );
        LensError::Tts("cloud TTS response read failed".into())
    })?;

    audio::decode_wav_mono16(&bytes).map_err(|e| {
        tracing::warn!(error = %e, "cloud TTS audio decode failed");
        LensError::Tts("cloud TTS returned undecodable audio".into())
    })
}
