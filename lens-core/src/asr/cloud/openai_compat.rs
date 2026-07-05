//! OpenAI-compatible cloud ASR adapter (#45): OpenAI / Groq / any `base_url`.
//! Wraps PCM in WAV, uploads a multipart form, and maps the `verbose_json`
//! `segments[]` response into [`TranscriptSegment`]s (float seconds).

use serde::Deserialize;

use super::{lang_code, map_status_error, wav};
use crate::LensError;
use crate::asr::{TranscribeConfig, TranscriptSegment};

#[derive(Deserialize)]
struct VerboseJson {
    #[serde(default)]
    segments: Vec<VerboseSegment>,
}

#[derive(Deserialize)]
struct VerboseSegment {
    start: f32,
    end: f32,
    text: String,
}

/// Transcribes one PCM window via a `POST {base_url}/v1/audio/transcriptions`
/// multipart upload with `Authorization: Bearer {api_key}`.
pub async fn transcribe(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    api_key: &str,
    pcm: &[f32],
    sample_rate: u32,
    config: &TranscribeConfig,
) -> Result<Vec<TranscriptSegment>, LensError> {
    let wav_bytes = wav::pcm_to_wav(pcm, sample_rate)?;
    let file_part = reqwest::multipart::Part::bytes(wav_bytes)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .map_err(|e| {
            tracing::warn!(error = %e, "cloud ASR form build failed");
            LensError::Transcription("cloud ASR form build failed".into())
        })?;

    let mut form = reqwest::multipart::Form::new()
        .part("file", file_part)
        .text("model", model.to_string())
        .text("response_format", "verbose_json")
        .text("timestamp_granularities[]", "segment");
    // Translation uses a separate OpenAI endpoint; this adapter transcribes
    // in-language and forwards the pinned language when set.
    if let Some(lang) = &config.language {
        form = form.text("language", lang_code(lang).to_string());
    }

    let url = format!("{}/v1/audio/transcriptions", base_url.trim_end_matches('/'));
    let resp = client
        .post(url)
        .bearer_auth(api_key)
        .multipart(form)
        .send()
        .await
        .map_err(|e| {
            tracing::warn!(error = %e, "cloud ASR HTTP request failed");
            LensError::Network("cloud ASR request failed".into())
        })?;

    let status = resp.status();
    if !status.is_success() {
        return Err(map_status_error(status.as_u16()));
    }

    let parsed: VerboseJson = resp.json().await.map_err(|e| {
        tracing::warn!(error = %e, "cloud ASR response parse failed");
        LensError::Transcription("cloud ASR bad response".into())
    })?;

    Ok(parsed
        .segments
        .into_iter()
        .map(|s| TranscriptSegment {
            text: s.text.trim().to_string(),
            start_second: s.start,
            end_second: s.end,
        })
        .collect())
}
