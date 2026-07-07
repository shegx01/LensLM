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
    #[serde(default)]
    text: String,
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

    let segments: Vec<TranscriptSegment> = parsed
        .segments
        .into_iter()
        .map(|s| TranscriptSegment {
            text: s.text.trim().to_string(),
            start_second: s.start,
            end_second: s.end,
        })
        .collect();

    if segments.is_empty() {
        let text = parsed.text.trim();
        if !text.is_empty() {
            tracing::warn!(
                "cloud ASR returned no segment timestamps; emitting one whole-window \
                 segment (model may not support timestamp_granularities)"
            );
            let duration = pcm.len() as f32 / sample_rate.max(1) as f32;
            return Ok(vec![TranscriptSegment {
                text: text.to_string(),
                start_second: 0.0,
                end_second: duration,
            }]);
        }
    }

    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::TranscribeConfig;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn empty_segments_with_text_degrade_to_single_whole_window_segment() {
        let server = MockServer::start().await;
        let body = serde_json::json!({ "text": "hello world", "segments": [] });
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;

        let sample_rate = 16_000u32;
        let pcm = vec![0.0f32; sample_rate as usize * 3];
        let out = transcribe(
            &reqwest::Client::new(),
            &server.uri(),
            "gpt-4o-transcribe",
            "test-key",
            &pcm,
            sample_rate,
            &TranscribeConfig::default(),
        )
        .await
        .expect("degraded transcription");

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "hello world");
        assert_eq!(out[0].start_second, 0.0);
        assert!(
            (out[0].end_second - 3.0).abs() < 1e-3,
            "end {}",
            out[0].end_second
        );
    }

    #[tokio::test]
    async fn empty_segments_and_empty_text_yield_no_segments() {
        let server = MockServer::start().await;
        let body = serde_json::json!({ "text": "", "segments": [] });
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;

        let out = transcribe(
            &reqwest::Client::new(),
            &server.uri(),
            "gpt-4o-transcribe",
            "k",
            &vec![0.0f32; 16_000],
            16_000,
            &TranscribeConfig::default(),
        )
        .await
        .expect("transcription");

        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn present_segments_are_mapped_directly() {
        let server = MockServer::start().await;
        let body = serde_json::json!({
            "text": "ignored when segments present",
            "segments": [
                { "start": 0.0, "end": 1.5, "text": " Hello " },
                { "start": 1.5, "end": 2.0, "text": "world" }
            ]
        });
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;

        let out = transcribe(
            &reqwest::Client::new(),
            &server.uri(),
            "whisper-1",
            "k",
            &vec![0.0f32; 16_000],
            16_000,
            &TranscribeConfig::default(),
        )
        .await
        .expect("transcription");

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text, "Hello");
        assert_eq!(out[0].end_second, 1.5);
        assert_eq!(out[1].text, "world");
    }
}
