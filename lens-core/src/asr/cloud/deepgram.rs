//! Deepgram cloud ASR adapter (#45): sends the app's raw 16 kHz mono f32 PCM
//! directly as `encoding=linear32` (zero conversion), and maps `utterances[]`
//! into [`TranscriptSegment`]s (float seconds).

use serde::Deserialize;

use super::{lang_code, map_status_error};
use crate::LensError;
use crate::asr::{TranscribeConfig, TranscriptSegment};

#[derive(Deserialize)]
struct DeepgramResponse {
    #[serde(default)]
    results: DeepgramResults,
}

#[derive(Deserialize, Default)]
struct DeepgramResults {
    #[serde(default)]
    utterances: Vec<Utterance>,
}

#[derive(Deserialize)]
struct Utterance {
    start: f32,
    end: f32,
    transcript: String,
}

/// Little-endian f32 bytes for Deepgram's `linear32` wire encoding. Explicit
/// per-sample `to_le_bytes` guarantees the on-wire byte order regardless of host
/// endianness (no `transmute`, no extra dep).
fn pcm_to_le_bytes(pcm: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(pcm));
    for &s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Transcribes one PCM window via `POST {base_url}/v1/listen` with the raw f32
/// body and `Authorization: Token {api_key}`.
pub async fn transcribe(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    api_key: &str,
    pcm: &[f32],
    sample_rate: u32,
    config: &TranscribeConfig,
) -> Result<Vec<TranscriptSegment>, LensError> {
    let language = config.language.as_ref().map(lang_code).unwrap_or("multi");
    let base = format!("{}/v1/listen", base_url.trim_end_matches('/'));
    let url = url::Url::parse_with_params(
        &base,
        &[
            ("model", model),
            ("encoding", "linear32"),
            ("sample_rate", &sample_rate.to_string()),
            ("utterances", "true"),
            ("language", language),
        ],
    )
    .map_err(|e| {
        tracing::warn!(error = %e, "cloud ASR URL construction failed");
        LensError::Validation("cloud ASR URL construction failed".into())
    })?;

    let resp = client
        .post(url)
        .header(reqwest::header::AUTHORIZATION, format!("Token {api_key}"))
        .header(reqwest::header::CONTENT_TYPE, "audio/raw")
        .body(pcm_to_le_bytes(pcm))
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

    let parsed: DeepgramResponse = resp.json().await.map_err(|e| {
        tracing::warn!(error = %e, "cloud ASR response parse failed");
        LensError::Transcription("cloud ASR bad response".into())
    })?;

    Ok(parsed
        .results
        .utterances
        .into_iter()
        .map(|u| TranscriptSegment {
            text: u.transcript.trim().to_string(),
            start_second: u.start,
            end_second: u.end,
        })
        .collect())
}
