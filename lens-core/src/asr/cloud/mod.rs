//! Cloud ASR tier (#45): an opt-in third [`AsrEngine`] that transcribes via a
//! cloud provider, gated by explicit `audio_cloud_consent`. Two reference
//! adapters — OpenAI-compatible (WAV multipart) and Deepgram (raw f32 PCM) —
//! validate the seam across both wire formats. Chunking, WAV wrapping, and the
//! consent pre-flight live here; the fallback-to-local cascade lives in
//! `lib.rs::transcribe`.

pub mod chunk;
pub mod deepgram;
pub mod openai_compat;
pub mod wav;

use crate::asr::Lang;

/// Maps a [`Lang`] to the wire language code; `Other` passes through unchanged.
pub(crate) fn lang_code(lang: &Lang) -> &str {
    match lang {
        Lang::En => "en",
        Lang::De => "de",
        Lang::Fr => "fr",
        Lang::Es => "es",
        Lang::It => "it",
        Lang::Pt => "pt",
        Lang::Nl => "nl",
        Lang::Ru => "ru",
        Lang::Zh => "zh",
        Lang::Ja => "ja",
        Lang::Ko => "ko",
        Lang::Other(code) => code.as_str(),
    }
}

use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc::UnboundedSender;

use crate::LensError;
use crate::asr::{AsrEngine, TranscribeConfig, TranscriptSegment};
use crate::config::{AppConfig, CloudAsrProvider};

/// Bounded connect timeout for cloud ASR. Slightly larger than the LLM path's 1 s
/// because a TLS handshake to a large-upload endpoint can be slower to establish.
const ASR_CLOUD_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Read/overall timeout. Larger than the 30 s LLM timeout: audio upload plus
/// server-side inference (and per-chunk retries on long files) are slow.
const ASR_CLOUD_TIMEOUT: Duration = Duration::from_secs(120);

const ASR_CLOUD_MAX_RETRIES: u32 = 3;
const ASR_CLOUD_RETRY_BASE_DELAY: Duration = Duration::from_millis(500);

/// The 16 kHz mono f32 PCM sample rate produced by #41 decode/resample.
const SAMPLE_RATE: u32 = 16_000;

/// Cloud speech-to-text engine. Dispatches to a provider adapter by
/// [`CloudAsrProvider`]; handles size-bounded chunking + timestamp stitching.
pub struct CloudAsrEngine {
    provider: CloudAsrProvider,
    base_url: String,
    model: String,
    api_key: String,
    client: reqwest::Client,
    max_retries: u32,
    retry_base_delay: Duration,
}

impl CloudAsrEngine {
    /// Builds an engine with the hardened, no-redirect HTTP client and cloud-ASR
    /// timeouts. Used in production.
    pub fn new(
        provider: CloudAsrProvider,
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        let client = crate::http::hardened_client(ASR_CLOUD_CONNECT_TIMEOUT, ASR_CLOUD_TIMEOUT);
        Self::with_client(provider, base_url, model, api_key, client)
    }

    /// Builds an engine with a caller-supplied client, so tests can point the
    /// `base_url` at a wiremock server without the no-redirect/timeout policy.
    pub fn with_client(
        provider: CloudAsrProvider,
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: impl Into<String>,
        client: reqwest::Client,
    ) -> Self {
        Self {
            provider,
            base_url: base_url.into(),
            model: model.into(),
            api_key: api_key.into(),
            client,
            max_retries: ASR_CLOUD_MAX_RETRIES,
            retry_base_delay: ASR_CLOUD_RETRY_BASE_DELAY,
        }
    }

    pub fn with_retry_policy(mut self, max_retries: u32, base_delay: Duration) -> Self {
        self.max_retries = max_retries;
        self.retry_base_delay = base_delay;
        self
    }

    async fn transcribe_chunk_with_retry(
        &self,
        pcm: &[f32],
        config: &TranscribeConfig,
    ) -> Result<Vec<TranscriptSegment>, LensError> {
        let mut attempt = 0u32;
        loop {
            match self.transcribe_chunk(pcm, config).await {
                Ok(segments) => return Ok(segments),
                Err(err) => {
                    let retryable = matches!(err, LensError::Network(_));
                    if !retryable || attempt >= self.max_retries {
                        return Err(err);
                    }
                    let backoff = self
                        .retry_base_delay
                        .saturating_mul(2u32.saturating_pow(attempt));
                    tracing::warn!(
                        provider = ?self.provider,
                        attempt = attempt + 1,
                        max_retries = self.max_retries,
                        backoff_ms = backoff.as_millis() as u64,
                        "cloud ASR chunk transient failure; retrying"
                    );
                    tokio::time::sleep(backoff).await;
                    attempt += 1;
                }
            }
        }
    }

    /// Transcribes a single PCM window through the configured provider adapter.
    async fn transcribe_chunk(
        &self,
        pcm: &[f32],
        config: &TranscribeConfig,
    ) -> Result<Vec<TranscriptSegment>, LensError> {
        match self.provider {
            CloudAsrProvider::OpenAiCompatible => {
                openai_compat::transcribe(
                    &self.client,
                    &self.base_url,
                    &self.model,
                    &self.api_key,
                    pcm,
                    SAMPLE_RATE,
                    config,
                )
                .await
            }
            CloudAsrProvider::Deepgram => {
                deepgram::transcribe(
                    &self.client,
                    &self.base_url,
                    &self.model,
                    &self.api_key,
                    pcm,
                    SAMPLE_RATE,
                    config,
                )
                .await
            }
        }
    }
}

#[async_trait]
impl AsrEngine for CloudAsrEngine {
    async fn transcribe_pcm(
        &self,
        pcm: &[f32],
        config: &TranscribeConfig,
        progress_tx: Option<UnboundedSender<f32>>,
    ) -> Result<Vec<TranscriptSegment>, LensError> {
        tracing::info!(provider = ?self.provider, model = %self.model, "cloud ASR started");

        let chunks = chunk::split_if_needed(pcm, self.provider, SAMPLE_RATE);
        let total = chunks.len().max(1);
        let mut stitched_input: Vec<(f32, Vec<TranscriptSegment>)> = Vec::with_capacity(total);

        for (idx, c) in chunks.iter().enumerate() {
            let segments = self.transcribe_chunk_with_retry(c.data, config).await?;
            stitched_input.push((c.start_second, segments));
            if let Some(tx) = &progress_tx {
                let fraction = (idx + 1) as f32 / total as f32;
                let _ = tx.send(fraction.clamp(0.0, 1.0));
            }
        }

        Ok(chunk::stitch_segments(&stitched_input))
    }
}

/// Consent + config pre-flight (#45), run BEFORE any cloud request. Order is
/// deliberate: consent is checked first so a mis-set backend never leaks audio.
/// No reachability probe — the unreachable case is handled by runtime fallback.
pub fn preflight_check(config: &AppConfig) -> Result<(), LensError> {
    let consent = config.audio_cloud_consent;
    let key_present = !config.asr.cloud_api_key.is_empty();
    let provider_set = config.asr.cloud_provider.is_some();
    tracing::debug!(consent, key_present, provider_set, "cloud ASR pre-flight");

    if !consent {
        return Err(LensError::Validation(
            "audio cloud consent not granted".into(),
        ));
    }
    if !key_present {
        return Err(LensError::Validation(
            "no cloud ASR API key configured".into(),
        ));
    }
    if !provider_set {
        return Err(LensError::Validation(
            "no cloud ASR provider configured".into(),
        ));
    }
    Ok(())
}

/// Maps a provider HTTP status to a [`LensError`] without leaking provider
/// internals. Connectivity-class statuses (429/5xx) → `Network`; misconfiguration
/// (401/403) → `Validation`; oversize (413) → `Validation`; else `Transcription`.
pub(crate) fn map_status_error(status: u16) -> LensError {
    match status {
        401 | 403 => LensError::Validation("cloud ASR rejected the API key".into()),
        413 => LensError::Validation("cloud ASR audio payload too large".into()),
        429 => LensError::Network("cloud ASR rate limited".into()),
        500..=599 => LensError::Network(format!("cloud ASR provider error ({status})")),
        _ => LensError::Transcription(format!("cloud ASR unexpected status ({status})")),
    }
}
