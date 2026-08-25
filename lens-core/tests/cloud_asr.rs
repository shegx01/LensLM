// issue #71: deep `Send` auto-trait evaluation can overflow the default 128-frame
// limit under stricter toolchains.
#![recursion_limit = "256"]
//! Offline tests for the cloud ASR fallback tier (#45).
//!
//! # Coverage
//!
//! Unit: WAV header, chunk split/stitch, config backward-compat, api_key
//! redaction, consent isolation, preflight gates.
//!
//! Integration (wiremock): OpenAI + Deepgram happy paths, HTTP error matrix
//! (401/413/429/500/malformed-200), pre-flight zero-call assertions,
//! chunked multi-request transcription, effective-backend transparency.
//!
//! All offline — no live network, no `LENS_RUN_MODEL_TESTS` gate.

use std::sync::Arc;

use lens_core::LensEngine;
use lens_core::asr::cloud::chunk::{split_if_needed, stitch_segments};
use lens_core::asr::cloud::wav::{WAV_HEADER_BYTES, pcm_to_wav};
use lens_core::asr::cloud::{CloudAsrEngine, default_base_url, default_model, preflight_check};
use lens_core::asr::{
    AsrBackend, AsrEngine, MockAsrEngine, TranscribeConfig, TranscriptOutput, TranscriptSegment,
};
use lens_core::config::{AppConfig, AsrConfig, CloudAsrProvider};
use rstest::rstest;
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ===========================================================================
// Unit: WAV header validity
// ===========================================================================

#[test]
fn wav_header_is_44_bytes() {
    let wav = pcm_to_wav(&[0.0_f32; 0], 16_000).expect("wav encode");
    assert_eq!(
        wav.len(),
        WAV_HEADER_BYTES,
        "empty PCM must produce exactly a 44-byte header"
    );
}

#[test]
fn wav_header_riff_magic() {
    let wav = pcm_to_wav(&[0.0_f32; 4], 16_000).expect("wav encode");
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(&wav[12..16], b"fmt ");
    assert_eq!(&wav[36..40], b"data");
}

#[test]
fn wav_total_length_matches_samples() {
    let n_samples: usize = 1024;
    let wav = pcm_to_wav(&vec![0.25_f32; n_samples], 16_000).expect("wav encode");
    // 44-byte header + n_samples * 2 bytes (16-bit)
    assert_eq!(wav.len(), WAV_HEADER_BYTES + n_samples * 2);
}

#[test]
fn wav_header_fields_16khz_mono_16bit() {
    let wav = pcm_to_wav(&[0.0_f32; 0], 16_000).expect("wav encode");

    // fmt chunk size: bytes 16..20 = 16u32 LE
    let fmt_chunk_size = u32::from_le_bytes(wav[16..20].try_into().unwrap());
    assert_eq!(fmt_chunk_size, 16, "PCM fmt chunk is 16 bytes");

    // audio format: bytes 20..22 = 1u16 (PCM)
    let audio_fmt = u16::from_le_bytes(wav[20..22].try_into().unwrap());
    assert_eq!(audio_fmt, 1, "format must be PCM (1)");

    // channels: bytes 22..24 = 1u16 (mono)
    let channels = u16::from_le_bytes(wav[22..24].try_into().unwrap());
    assert_eq!(channels, 1, "must be mono");

    // sample rate: bytes 24..28 = 16000u32 LE
    let sample_rate = u32::from_le_bytes(wav[24..28].try_into().unwrap());
    assert_eq!(sample_rate, 16_000);

    // bits per sample: bytes 34..36 = 16u16
    let bits = u16::from_le_bytes(wav[34..36].try_into().unwrap());
    assert_eq!(bits, 16);
}

#[test]
fn wav_riff_chunk_size_correct() {
    let n_samples: usize = 100;
    let wav = pcm_to_wav(&vec![0.0_f32; n_samples], 16_000).expect("wav encode");
    // RIFF chunk size = 36 + data_len; data_len = n_samples * 2
    let data_len = (n_samples * 2) as u32;
    let expected_riff = 36 + data_len;
    let actual_riff = u32::from_le_bytes(wav[4..8].try_into().unwrap());
    assert_eq!(actual_riff, expected_riff);
}

#[test]
fn wav_data_chunk_size_correct() {
    let n_samples: usize = 50;
    let wav = pcm_to_wav(&vec![0.5_f32; n_samples], 16_000).expect("wav encode");
    let data_len = u32::from_le_bytes(wav[40..44].try_into().unwrap());
    assert_eq!(data_len, (n_samples * 2) as u32);
}

#[test]
fn wav_samples_are_clamped_and_scaled() {
    // 1.0 → i16::MAX, -1.0 → -i16::MAX, 0.0 → 0
    let wav = pcm_to_wav(&[1.0_f32, -1.0_f32, 0.0_f32], 16_000).expect("wav encode");
    let s0 = i16::from_le_bytes(wav[44..46].try_into().unwrap());
    let s1 = i16::from_le_bytes(wav[46..48].try_into().unwrap());
    let s2 = i16::from_le_bytes(wav[48..50].try_into().unwrap());
    assert_eq!(s0, i16::MAX);
    assert_eq!(s1, -i16::MAX);
    assert_eq!(s2, 0);
}

// ===========================================================================
// Unit: chunk::split_if_needed / stitch_segments
// ===========================================================================

#[test]
fn split_if_needed_under_limit_returns_single_chunk() {
    // 1 second at 16kHz = 16000 f32 samples, WAV ~= 44 + 32000 bytes ≈ 32 KB → way under 25 MB
    let pcm: Vec<f32> = vec![0.1_f32; 16_000];
    let chunks = split_if_needed(&pcm, CloudAsrProvider::OpenAiCompatible, 16_000);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].start_second, 0.0);
    assert_eq!(chunks[0].data.len(), pcm.len());
}

#[test]
fn split_if_needed_deepgram_short_audio_single_chunk() {
    let pcm: Vec<f32> = vec![0.1_f32; 16_000 * 60];
    let chunks = split_if_needed(&pcm, CloudAsrProvider::Deepgram, 16_000);
    assert_eq!(chunks.len(), 1);
}

#[test]
fn split_if_needed_deepgram_over_duration_cap_produces_multiple_chunks() {
    let sample_rate = 16_000usize;
    let n = sample_rate * 60 * 20;
    let pcm: Vec<f32> = vec![0.1_f32; n];
    let chunks = split_if_needed(&pcm, CloudAsrProvider::Deepgram, sample_rate as u32);
    assert!(
        chunks.len() >= 2,
        "20 minutes of Deepgram audio must split on the duration cap, got {}",
        chunks.len()
    );
    let max_chunk_samples = 480 * sample_rate;
    for (i, c) in chunks.iter().enumerate() {
        assert!(
            c.data.len() <= max_chunk_samples,
            "chunk {i} of {} samples exceeds the ~8-minute duration cap ({max_chunk_samples})",
            c.data.len()
        );
    }
    let total: usize = chunks.iter().map(|c| c.data.len()).sum();
    assert_eq!(total, n, "all samples must be covered across chunks");
    let mut prev = -1.0f32;
    for c in &chunks {
        assert!(
            c.start_second > prev,
            "chunk starts must be strictly increasing"
        );
        prev = c.start_second;
    }
}

#[test]
fn split_if_needed_zero_sample_rate_terminates() {
    let pcm: Vec<f32> = vec![0.1_f32; 10];
    let chunks = split_if_needed(&pcm, CloudAsrProvider::Deepgram, 0);
    let total: usize = chunks.iter().map(|c| c.data.len()).sum();
    assert_eq!(
        total,
        pcm.len(),
        "a zero sample rate must not lose or loop over samples"
    );
}

#[test]
fn split_if_needed_over_openai_25mb_cap_produces_multiple_chunks() {
    // OpenAI WAV cap: 25 MB. 16-bit WAV: each sample = 2 bytes.
    // 25 MB = 26_214_400 bytes. Data portion = 25 MB - 44 header = 26_214_356 bytes
    // → max ~13_107_178 samples. Use 14 million to be safely over.
    let n = 14_000_000usize;
    let pcm: Vec<f32> = vec![0.1_f32; n];
    let chunks = split_if_needed(&pcm, CloudAsrProvider::OpenAiCompatible, 16_000);
    assert!(
        chunks.len() >= 2,
        "over-limit PCM must split into at least 2 chunks, got {}",
        chunks.len()
    );
    // Every chunk's encoded WAV size must be ≤ 25 MB
    let cap = 25 * 1024 * 1024usize;
    for (i, c) in chunks.iter().enumerate() {
        let encoded = WAV_HEADER_BYTES + c.data.len() * 2;
        assert!(
            encoded <= cap,
            "chunk {i} encoded size {encoded} exceeds 25 MB cap"
        );
    }
}

#[test]
fn split_if_needed_chunk_start_seconds_are_monotonic() {
    let n = 14_000_000usize;
    let pcm: Vec<f32> = vec![0.0_f32; n];
    let chunks = split_if_needed(&pcm, CloudAsrProvider::OpenAiCompatible, 16_000);
    let mut prev = -1.0f32;
    for c in &chunks {
        assert!(
            c.start_second > prev,
            "chunk start_second must be strictly increasing"
        );
        prev = c.start_second;
    }
    // First chunk always starts at 0.0
    assert_eq!(chunks[0].start_second, 0.0);
}

#[test]
fn split_if_needed_chunks_cover_all_samples() {
    let n = 14_000_000usize;
    let pcm: Vec<f32> = vec![0.1_f32; n];
    let chunks = split_if_needed(&pcm, CloudAsrProvider::OpenAiCompatible, 16_000);
    let total: usize = chunks.iter().map(|c| c.data.len()).sum();
    assert_eq!(total, n, "all samples must be covered across chunks");
}

#[test]
fn stitch_segments_empty_input() {
    let out = stitch_segments(&[]);
    assert!(out.is_empty());
}

#[test]
fn stitch_segments_single_chunk_passthrough() {
    let segs = vec![
        TranscriptSegment {
            text: "hello".into(),
            start_second: 0.0,
            end_second: 1.0,
        },
        TranscriptSegment {
            text: "world".into(),
            start_second: 1.0,
            end_second: 2.5,
        },
    ];
    let out = stitch_segments(&[(0.0, segs.clone())]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].text, "hello");
    assert!((out[0].start_second - 0.0).abs() < 1e-4);
    assert!((out[1].start_second - 1.0).abs() < 1e-4);
}

#[test]
fn stitch_segments_reoffsets_second_chunk() {
    // Chunk 1: 10s long, segments at 0–5 and 5–10
    // Chunk 2: starts at t=10s (start_second=10.0), segments at 0–3 and 3–6
    let chunk1_segs = vec![
        TranscriptSegment {
            text: "a".into(),
            start_second: 0.0,
            end_second: 5.0,
        },
        TranscriptSegment {
            text: "b".into(),
            start_second: 5.0,
            end_second: 10.0,
        },
    ];
    let chunk2_segs = vec![
        TranscriptSegment {
            text: "c".into(),
            start_second: 0.0,
            end_second: 3.0,
        },
        TranscriptSegment {
            text: "d".into(),
            start_second: 3.0,
            end_second: 6.0,
        },
    ];
    let out = stitch_segments(&[(0.0, chunk1_segs), (10.0, chunk2_segs)]);
    assert_eq!(out.len(), 4);
    // "a": unchanged
    assert!(
        (out[0].start_second - 0.0).abs() < 1e-3,
        "a.start={}",
        out[0].start_second
    );
    assert!((out[0].end_second - 5.0).abs() < 1e-3);
    // "b": unchanged
    assert!((out[1].start_second - 5.0).abs() < 1e-3);
    assert!((out[1].end_second - 10.0).abs() < 1e-3);
    // "c": offset by 10.0
    assert!(
        (out[2].start_second - 10.0).abs() < 1e-3,
        "c.start={}",
        out[2].start_second
    );
    assert!((out[2].end_second - 13.0).abs() < 1e-3);
    // "d": offset by 10.0
    assert!(
        (out[3].start_second - 13.0).abs() < 1e-3,
        "d.start={}",
        out[3].start_second
    );
    assert!((out[3].end_second - 16.0).abs() < 1e-3);
}

#[test]
fn stitch_segments_global_monotonic_non_decreasing() {
    // Three chunks whose individual segments are all relative to chunk start.
    let c1 = vec![TranscriptSegment {
        text: "x".into(),
        start_second: 0.0,
        end_second: 4.0,
    }];
    let c2 = vec![TranscriptSegment {
        text: "y".into(),
        start_second: 0.0,
        end_second: 3.0,
    }];
    let c3 = vec![TranscriptSegment {
        text: "z".into(),
        start_second: 0.0,
        end_second: 5.0,
    }];
    let out = stitch_segments(&[(0.0, c1), (4.0, c2), (7.0, c3)]);
    for w in out.windows(2) {
        assert!(
            w[1].start_second >= w[0].end_second - 1e-4,
            "segments not monotonic: {:?} then {:?}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn stitch_segments_total_coverage_preserved() {
    // Sum of each chunk's max end_second should ≈ total duration
    let c1 = vec![TranscriptSegment {
        text: "a".into(),
        start_second: 0.0,
        end_second: 8.0,
    }];
    let c2 = vec![TranscriptSegment {
        text: "b".into(),
        start_second: 0.0,
        end_second: 7.0,
    }];
    let out = stitch_segments(&[(0.0, c1), (8.0, c2)]);
    let last_end = out.last().unwrap().end_second;
    // Expected ~15.0 (8 + 7)
    assert!((last_end - 15.0).abs() < 0.1, "last_end={last_end}");
}

// ===========================================================================
// Unit: config backward-compat, serde, and api_key redaction
// ===========================================================================

#[test]
fn asr_config_cloud_fields_default_to_empty() {
    let cfg = AsrConfig::default();
    assert!(cfg.cloud_provider.is_none());
    assert!(cfg.cloud_base_url.is_empty());
    assert!(cfg.cloud_model.is_empty());
    assert!(cfg.cloud_api_key.is_empty());
}

#[test]
fn app_config_audio_cloud_consent_defaults_false() {
    // AppConfig::default() is the authoritative default; audio_cloud_consent must be false.
    let cfg = AppConfig::default();
    assert!(
        !cfg.audio_cloud_consent,
        "audio_cloud_consent must default to false"
    );

    // Old on-disk JSON that predates #45 also gets false via #[serde(default)].
    let json = r#"{"theme":"dark","user_name":"","embedding_model":"","embedding_backend":"","max_source_mb":"","models":[],"endpoints":{},"voices":{"host":"","guest":""},"paths":{"data_dir":""},"tier_thresholds":{"tier1_token_cap":4000,"tier2_token_cap":16000},"onboarding_complete":false}"#;
    let cfg2: AppConfig = serde_json::from_str(json).expect("old config must parse");
    assert!(
        !cfg2.audio_cloud_consent,
        "audio_cloud_consent must default to false when absent from old JSON"
    );
}

#[test]
fn app_config_tts_cloud_consent_defaults_false() {
    let cfg = AppConfig::default();
    assert!(
        !cfg.tts_cloud_consent,
        "tts_cloud_consent must default to false"
    );

    // Every existing on-disk config predates this field; absent must read as withheld,
    // never grandfathered on from audio_cloud_consent or a saved key.
    let json = r#"{"theme":"dark","user_name":"","embedding_model":"","embedding_backend":"","max_source_mb":"","models":[],"endpoints":{},"voices":{"host":"","guest":""},"audio_cloud_consent":true,"tts":{"version":1,"backend":{"cloud":"open_ai_compatible"},"model":"","clouds":{"open_ai_compatible":{"api_key":"sk-key","base_url":""}}},"paths":{"data_dir":""},"tier_thresholds":{"tier1_token_cap":4000,"tier2_token_cap":16000},"onboarding_complete":false}"#;
    let cfg2: AppConfig = serde_json::from_str(json).expect("old config must parse");
    assert!(
        !cfg2.tts_cloud_consent,
        "tts_cloud_consent must default to false when absent from old JSON"
    );
    assert!(cfg2.audio_cloud_consent, "the ASR consent key is unrenamed");
}

#[test]
fn asr_config_backward_compat_old_json_no_cloud_fields() {
    // Old config without cloud keys must parse fine, cloud fields get defaults
    let json = r#"{"backend":"local_whisper","whisper_model":"base"}"#;
    let cfg: AsrConfig = serde_json::from_str(json).expect("old asr config must parse");
    assert!(cfg.cloud_provider.is_none());
    assert!(cfg.cloud_api_key.is_empty());
}

#[test]
fn cloud_asr_provider_serde_snake_case_roundtrip() {
    let oai = CloudAsrProvider::OpenAiCompatible;
    let dg = CloudAsrProvider::Deepgram;

    let oai_json = serde_json::to_string(&oai).unwrap();
    let dg_json = serde_json::to_string(&dg).unwrap();

    assert_eq!(oai_json, r#""open_ai_compatible""#);
    assert_eq!(dg_json, r#""deepgram""#);

    let back_oai: CloudAsrProvider = serde_json::from_str(&oai_json).unwrap();
    let back_dg: CloudAsrProvider = serde_json::from_str(&dg_json).unwrap();
    assert_eq!(back_oai, CloudAsrProvider::OpenAiCompatible);
    assert_eq!(back_dg, CloudAsrProvider::Deepgram);
}

#[test]
fn asr_config_debug_redacts_api_key() {
    let cfg = AsrConfig {
        cloud_api_key: "super-secret-key".to_string(),
        cloud_provider: Some(CloudAsrProvider::OpenAiCompatible),
        ..AsrConfig::default()
    };
    let debug_str = format!("{cfg:?}");
    assert!(
        !debug_str.contains("super-secret-key"),
        "api_key must not appear in Debug output: {debug_str}"
    );
    assert!(
        debug_str.contains("***"),
        "Debug output must show *** for non-empty key: {debug_str}"
    );
}

#[test]
fn asr_config_debug_shows_empty_for_absent_key() {
    let cfg = AsrConfig {
        cloud_api_key: String::new(),
        ..AsrConfig::default()
    };
    let debug_str = format!("{cfg:?}");
    // Empty key shows as "" not ***
    assert!(
        !debug_str.contains("***"),
        "empty key must not show ***: {debug_str}"
    );
}

#[test]
fn audio_cloud_consent_independent_from_enrichment_cloud_consent() {
    use lens_core::config::EnrichmentConfig;

    // Case 1: audio consent ON, enrichment consent OFF — must be independent
    let cfg = AppConfig {
        audio_cloud_consent: true,
        enrichment: EnrichmentConfig {
            cloud_consent: false,
            ..EnrichmentConfig::default()
        },
        ..AppConfig::default()
    };
    assert!(cfg.audio_cloud_consent, "audio_cloud_consent must be true");
    assert!(
        !cfg.enrichment.cloud_consent,
        "enrichment.cloud_consent must remain false"
    );

    // Case 2: audio consent OFF, enrichment consent ON — must be independent
    let cfg2 = AppConfig {
        audio_cloud_consent: false,
        enrichment: EnrichmentConfig {
            cloud_consent: true,
            ..EnrichmentConfig::default()
        },
        ..AppConfig::default()
    };
    assert!(
        !cfg2.audio_cloud_consent,
        "audio_cloud_consent must be false"
    );
    assert!(
        cfg2.enrichment.cloud_consent,
        "enrichment.cloud_consent must be true"
    );

    // Case 3: serde round-trip preserves each flag independently
    let src = AppConfig {
        audio_cloud_consent: true,
        enrichment: EnrichmentConfig {
            cloud_consent: false,
            ..EnrichmentConfig::default()
        },
        ..AppConfig::default()
    };
    let json = serde_json::to_string(&src).unwrap();
    let back: AppConfig = serde_json::from_str(&json).unwrap();
    assert!(back.audio_cloud_consent);
    assert!(!back.enrichment.cloud_consent);
}

// ===========================================================================
// Unit: preflight_check gates (no network)
// ===========================================================================

fn app_config_with_cloud_endpoint(
    consent: bool,
    api_key: &str,
    provider: Option<CloudAsrProvider>,
    base_url: &str,
    model: &str,
) -> AppConfig {
    AppConfig {
        audio_cloud_consent: consent,
        asr: AsrConfig {
            cloud_provider: provider,
            cloud_api_key: api_key.to_string(),
            cloud_base_url: base_url.to_string(),
            cloud_model: model.to_string(),
            ..AsrConfig::default()
        },
        ..AppConfig::default()
    }
}

fn app_config_with_cloud(
    consent: bool,
    api_key: &str,
    provider: Option<CloudAsrProvider>,
) -> AppConfig {
    app_config_with_cloud_endpoint(
        consent,
        api_key,
        provider,
        "https://api.openai.com",
        "whisper-1",
    )
}

#[test]
fn preflight_consent_false_returns_validation_error() {
    let cfg = app_config_with_cloud(false, "sk-test", Some(CloudAsrProvider::OpenAiCompatible));
    let err = preflight_check(&cfg).unwrap_err();
    assert_eq!(
        err.kind(),
        "Validation",
        "no consent → Validation, got {err:?}"
    );
    assert!(
        err.message().contains("consent"),
        "error must mention consent: {}",
        err.message()
    );
}

#[test]
fn preflight_empty_key_returns_validation_error() {
    let cfg = app_config_with_cloud(true, "", Some(CloudAsrProvider::OpenAiCompatible));
    let err = preflight_check(&cfg).unwrap_err();
    assert_eq!(err.kind(), "Validation");
    assert!(err.message().contains("key") || err.message().contains("API"));
}

#[test]
fn preflight_no_provider_returns_validation_error() {
    let cfg = app_config_with_cloud(true, "sk-test", None);
    let err = preflight_check(&cfg).unwrap_err();
    assert_eq!(err.kind(), "Validation");
    assert!(
        err.message().contains("provider"),
        "error must mention provider: {}",
        err.message()
    );
}

#[rstest]
#[case::empty(CloudAsrProvider::OpenAiCompatible, "")]
#[case::spaces(CloudAsrProvider::OpenAiCompatible, "   ")]
#[case::tab_newline(CloudAsrProvider::OpenAiCompatible, " \t\n ")]
#[case::deepgram_empty(CloudAsrProvider::Deepgram, "")]
#[case::deepgram_spaces(CloudAsrProvider::Deepgram, "   ")]
fn preflight_blank_base_url_returns_validation_error(
    #[case] provider: CloudAsrProvider,
    #[case] base_url: &str,
) {
    let cfg =
        app_config_with_cloud_endpoint(true, "sk-test", Some(provider), base_url, "whisper-1");
    let err = preflight_check(&cfg).unwrap_err();
    assert_eq!(err.kind(), "Validation");
    assert!(
        err.message().contains("base URL"),
        "error must name the base URL: {}",
        err.message()
    );
}

#[rstest]
#[case::empty(CloudAsrProvider::OpenAiCompatible, "")]
#[case::spaces(CloudAsrProvider::OpenAiCompatible, "   ")]
#[case::tab_newline(CloudAsrProvider::OpenAiCompatible, " \t\n ")]
#[case::deepgram_empty(CloudAsrProvider::Deepgram, "")]
#[case::deepgram_spaces(CloudAsrProvider::Deepgram, "   ")]
fn preflight_blank_model_returns_validation_error(
    #[case] provider: CloudAsrProvider,
    #[case] model: &str,
) {
    let cfg = app_config_with_cloud_endpoint(
        true,
        "sk-test",
        Some(provider),
        default_base_url(provider),
        model,
    );
    let err = preflight_check(&cfg).unwrap_err();
    assert_eq!(err.kind(), "Validation");
    assert!(
        err.message().contains("model"),
        "error must name the model: {}",
        err.message()
    );
}

#[test]
fn preflight_blank_base_url_and_model_messages_are_distinct() {
    let blank_url = app_config_with_cloud_endpoint(
        true,
        "sk-test",
        Some(CloudAsrProvider::OpenAiCompatible),
        "",
        "whisper-1",
    );
    let blank_model = app_config_with_cloud_endpoint(
        true,
        "sk-test",
        Some(CloudAsrProvider::OpenAiCompatible),
        "https://api.openai.com",
        "",
    );
    let url_err = preflight_check(&blank_url).unwrap_err();
    let model_err = preflight_check(&blank_model).unwrap_err();
    assert_ne!(
        url_err.message(),
        model_err.message(),
        "blank base URL and blank model must report distinct messages"
    );
}

#[test]
fn preflight_reports_the_base_url_first_when_both_fields_are_blank() {
    let cfg = app_config_with_cloud_endpoint(
        true,
        "sk-test",
        Some(CloudAsrProvider::OpenAiCompatible),
        "",
        "",
    );
    let err = preflight_check(&cfg).unwrap_err();
    assert!(
        err.message().contains("base URL"),
        "base URL is the earlier gate: {}",
        err.message()
    );
}

#[test]
fn preflight_whitespace_only_key_returns_validation_error() {
    let cfg = app_config_with_cloud(true, "   ", Some(CloudAsrProvider::OpenAiCompatible));
    let err = preflight_check(&cfg).unwrap_err();
    assert_eq!(err.kind(), "Validation");
    assert!(err.message().contains("key") || err.message().contains("API"));
}

// SYNC-CHECK: this table and the accept table below mirror the `isValidBaseUrl` tables in
// `TranscriptionCloudPane.svelte.test.ts`; a case added here belongs there too.
#[rstest]
#[case::plain_http("http://api.openai.com")]
#[case::plain_http_other_host("http://api.example.com")]
#[case::http_lookalike_host("http://localhost.evil.example")]
#[case::rfc1918_lookalike_domain("http://10.0.0.1.evil.com")]
#[case::below_the_172_16_block("http://172.15.0.1")]
#[case::above_the_172_31_block("http://172.32.0.1")]
#[case::link_local("http://169.254.1.1")]
#[case::unspecified("http://0.0.0.0")]
#[case::public_ipv4("http://93.184.216.34")]
#[case::ftp("ftp://x")]
#[case::relative("api.example.com")]
#[case::garbage("not a url at all")]
#[case::file("file:///etc/passwd")]
fn preflight_rejects_a_base_url_that_is_not_transport_safe(#[case] base_url: &str) {
    let cfg = app_config_with_cloud_endpoint(
        true,
        "sk-test",
        Some(CloudAsrProvider::OpenAiCompatible),
        base_url,
        "whisper-1",
    );
    let err = preflight_check(&cfg).unwrap_err();
    assert_eq!(err.kind(), "Validation");
    assert!(
        err.message().contains("https"),
        "error must name the required scheme: {}",
        err.message()
    );
    assert!(
        !err.message().contains(base_url),
        "error must not echo the configured URL: {}",
        err.message()
    );
}

#[rstest]
#[case::https("https://api.openai.com")]
#[case::https_custom_port("https://asr.internal.example:8443/v1")]
#[case::loopback_name("http://localhost:9000")]
#[case::loopback_v4("http://127.0.0.1")]
#[case::loopback_v4_upper_range("http://127.1.2.3")]
#[case::loopback_v6("http://[::1]:1234")]
#[case::rfc1918_10("http://10.0.0.5:9000")]
#[case::rfc1918_172_low("http://172.16.0.1")]
#[case::rfc1918_172_high("http://172.31.255.254")]
#[case::rfc1918_192_168("http://192.168.1.5:9000")]
#[case::mdns_local("http://whisper.local")]
fn preflight_accepts_a_transport_safe_base_url(#[case] base_url: &str) {
    let cfg = app_config_with_cloud_endpoint(
        true,
        "sk-test",
        Some(CloudAsrProvider::OpenAiCompatible),
        base_url,
        "whisper-1",
    );
    assert!(
        preflight_check(&cfg).is_ok(),
        "{base_url} must pass: {:?}",
        preflight_check(&cfg)
    );
}

#[test]
fn preflight_consent_fails_before_the_endpoint_gates() {
    let cfg =
        app_config_with_cloud_endpoint(false, "", Some(CloudAsrProvider::OpenAiCompatible), "", "");
    let err = preflight_check(&cfg).unwrap_err();
    assert!(
        err.message().contains("consent"),
        "consent must be the first gate reported: {}",
        err.message()
    );
}

// Every case carries a set provider so the consent/key gates are reached with the
// secrets present, rather than short-circuiting before the message is built.
#[test]
fn preflight_messages_leak_no_internals() {
    let secret_key = "sk-super-secret-key";
    let secret_url = "https://internal.corp.example/asr";
    let secret_model = "internal-model-v9";
    let openai = Some(CloudAsrProvider::OpenAiCompatible);
    let cases = [
        app_config_with_cloud_endpoint(false, secret_key, openai, secret_url, secret_model),
        app_config_with_cloud_endpoint(true, "", openai, secret_url, secret_model),
        app_config_with_cloud_endpoint(true, secret_key, None, secret_url, secret_model),
        app_config_with_cloud_endpoint(true, secret_key, openai, "", secret_model),
        app_config_with_cloud_endpoint(
            true,
            secret_key,
            Some(CloudAsrProvider::Deepgram),
            secret_url,
            "",
        ),
        app_config_with_cloud_endpoint(
            true,
            secret_key,
            openai,
            "http://internal.corp.example/asr",
            secret_model,
        ),
    ];
    for cfg in cases {
        let err = preflight_check(&cfg).unwrap_err();
        let msg = err.message();
        for leak in [
            secret_key,
            secret_url,
            secret_model,
            "internal.corp.example",
            "OpenAiCompatible",
            "Deepgram",
        ] {
            assert!(
                !msg.contains(leak),
                "pre-flight message leaked {leak:?}: {msg}"
            );
        }
    }
}

/// Reads `<field>: '<value>'` out of one entry of the live `CLOUD_ASR_PRESETS`
/// literal in `src/lib/asr/catalog.ts` — parsed, not grepped, so a comment
/// mentioning the value cannot satisfy the check.
fn frontend_preset(entry: &str, field: &str) -> String {
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../src/lib/asr/catalog.ts");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let decl = src
        .find("CLOUD_ASR_PRESETS")
        .expect("CLOUD_ASR_PRESETS not found in catalog.ts");
    let body = &src[decl..];
    let estart = body
        .find(&format!("{entry}:"))
        .unwrap_or_else(|| panic!("{entry} missing from CLOUD_ASR_PRESETS"));
    let obj = &body[estart..];
    let obj_end = obj.find('}').expect("preset entry never closes");
    let obj = &obj[..obj_end];
    let fstart = obj
        .find(&format!("{field}:"))
        .unwrap_or_else(|| panic!("{entry}.{field} missing from CLOUD_ASR_PRESETS"));
    let after = &obj[fstart..];
    let q = after.find(['\'', '"']).expect("quoted value expected");
    let quote = after.as_bytes()[q] as char;
    let rest = &after[q + 1..];
    let end = rest.find(quote).expect("closing quote expected");
    rest[..end].to_string()
}

#[rstest]
#[case::openai(CloudAsrProvider::OpenAiCompatible, "open_ai_compatible")]
#[case::deepgram(CloudAsrProvider::Deepgram, "deepgram")]
fn cloud_asr_defaults_match_the_frontend_presets(
    #[case] provider: CloudAsrProvider,
    #[case] entry: &str,
) {
    assert_eq!(
        default_base_url(provider),
        frontend_preset(entry, "base_url"),
        "{entry}.base_url drifted from CLOUD_ASR_PRESETS in src/lib/asr/catalog.ts"
    );
    assert_eq!(
        default_model(provider),
        frontend_preset(entry, "model"),
        "{entry}.model drifted from CLOUD_ASR_PRESETS in src/lib/asr/catalog.ts"
    );
}

// ===========================================================================
// Unit: AppConfig::normalize + save wiring (no network)
// ===========================================================================

/// Consent is granted throughout: a persisted `cloud` backend WITHOUT it is its own
/// defect with its own case, and these cases isolate the endpoint-fill behaviour.
fn asr_cloud_config(
    backend: &str,
    provider: Option<CloudAsrProvider>,
    base_url: &str,
    model: &str,
) -> AppConfig {
    AppConfig {
        audio_cloud_consent: true,
        asr: AsrConfig {
            backend: backend.to_string(),
            cloud_provider: provider,
            cloud_base_url: base_url.to_string(),
            cloud_model: model.to_string(),
            ..AsrConfig::default()
        },
        ..AppConfig::default()
    }
}

/// `save` is the only normalization funnel, so these cases exercise it rather than
/// the crate-private routine — the returned value is what both memory and disk hold.
fn normalized(mut cfg: AppConfig) -> AppConfig {
    let dir = tempfile::tempdir().unwrap();
    cfg.save(dir.path()).unwrap();
    cfg
}

#[rstest]
#[case::openai(
    CloudAsrProvider::OpenAiCompatible,
    "https://api.openai.com",
    "whisper-1"
)]
#[case::deepgram(CloudAsrProvider::Deepgram, "https://api.deepgram.com", "nova-3")]
fn normalize_fills_blank_endpoint_fields_under_a_set_provider(
    #[case] provider: CloudAsrProvider,
    #[case] base_url: &str,
    #[case] model: &str,
) {
    let cfg = normalized(asr_cloud_config("local_whisper", Some(provider), "", ""));
    assert_eq!(cfg.asr.cloud_base_url, base_url);
    assert_eq!(cfg.asr.cloud_model, model);
    assert_eq!(
        cfg.asr.backend, "local_whisper",
        "a non-cloud backend must not be demoted"
    );
}

// The end state a bare provider choice promises: nothing else
// filled in, is enough for the cloud pre-flight to pass.
#[rstest]
#[case::openai_inactive(CloudAsrProvider::OpenAiCompatible, "")]
#[case::openai_active(CloudAsrProvider::OpenAiCompatible, "cloud")]
#[case::deepgram_inactive(CloudAsrProvider::Deepgram, "")]
#[case::deepgram_active(CloudAsrProvider::Deepgram, "cloud")]
fn normalize_makes_a_bare_provider_choice_pass_preflight(
    #[case] provider: CloudAsrProvider,
    #[case] backend: &str,
) {
    let mut cfg = asr_cloud_config(backend, Some(provider), "", "");
    cfg.audio_cloud_consent = true;
    cfg.asr.cloud_api_key = "sk-test".to_string();
    let cfg = normalized(cfg);
    assert!(
        preflight_check(&cfg).is_ok(),
        "normalized config must clear pre-flight: {:?}",
        preflight_check(&cfg)
    );
}

#[rstest]
#[case::blank_both("", "")]
#[case::blank_base_url("", "whisper-1")]
#[case::blank_model("https://api.openai.com", "")]
#[case::whitespace_both("   ", " \t ")]
#[case::already_normalized("https://api.openai.com", "whisper-1")]
fn normalize_is_idempotent(#[case] base_url: &str, #[case] model: &str) {
    let once = normalized(asr_cloud_config(
        "cloud",
        Some(CloudAsrProvider::OpenAiCompatible),
        base_url,
        model,
    ));
    let twice = normalized(once.clone());
    assert_eq!(twice, once, "a second normalize must be a no-op");
}

#[test]
fn normalize_is_a_noop_without_a_provider() {
    let cfg = normalized(asr_cloud_config("cloud", None, "", ""));
    assert_eq!(cfg.asr.cloud_base_url, "");
    assert_eq!(cfg.asr.cloud_model, "");
    assert_eq!(cfg.asr.backend, "cloud");
}

#[test]
fn normalize_leaves_populated_fields_unchanged() {
    let before = asr_cloud_config(
        "cloud",
        Some(CloudAsrProvider::Deepgram),
        "https://asr.internal.example",
        "custom-model",
    );
    assert_eq!(normalized(before.clone()), before);
}

// The fill is what makes the demotion load-bearing: without it the blank would be
// refilled on the next pass and cloud would silently re-arm.
#[rstest]
#[case::openai_blank_base_url(CloudAsrProvider::OpenAiCompatible, "", "whisper-1")]
#[case::openai_blank_model(CloudAsrProvider::OpenAiCompatible, "https://api.openai.com", "")]
#[case::openai_whitespace_base_url(CloudAsrProvider::OpenAiCompatible, "   ", "whisper-1")]
#[case::deepgram_blank_base_url(CloudAsrProvider::Deepgram, "", "nova-3")]
#[case::deepgram_blank_model(CloudAsrProvider::Deepgram, "https://api.deepgram.com", "")]
fn normalize_demotes_an_active_cloud_backend_before_filling(
    #[case] provider: CloudAsrProvider,
    #[case] base_url: &str,
    #[case] model: &str,
) {
    let cfg = normalized(asr_cloud_config("cloud", Some(provider), base_url, model));
    assert_eq!(cfg.asr.backend, "", "active cloud backend must be demoted");
    assert_eq!(cfg.asr.cloud_base_url, default_base_url(provider));
    assert_eq!(cfg.asr.cloud_model, default_model(provider));
}

#[test]
fn normalize_fills_when_no_cloud_backend_is_active() {
    let cfg = normalized(asr_cloud_config(
        "",
        Some(CloudAsrProvider::Deepgram),
        "",
        "",
    ));
    assert_eq!(cfg.asr.backend, "", "an inactive backend stays untouched");
    assert_eq!(cfg.asr.cloud_base_url, "https://api.deepgram.com");
    assert_eq!(cfg.asr.cloud_model, "nova-3");
}

// `load` is a read; normalizing there would rewrite a config the caller never
// asked to change, and would mask a `save` that forgot to normalize.
#[test]
fn load_does_not_normalize() {
    let dir = tempfile::tempdir().unwrap();
    let raw = serde_json::to_string_pretty(&asr_cloud_config(
        "cloud",
        Some(CloudAsrProvider::Deepgram),
        "",
        "",
    ))
    .unwrap();
    std::fs::write(dir.path().join("config.json"), raw).unwrap();

    let loaded = AppConfig::load(dir.path()).unwrap();
    assert_eq!(loaded.asr.backend, "cloud");
    assert_eq!(loaded.asr.cloud_base_url, "");
    assert_eq!(loaded.asr.cloud_model, "");
}

/// The saved config holds a plaintext cloud API key, so it must never be readable
/// by another account — and the temp file it is staged through must not survive.
#[cfg(unix)]
#[test]
fn save_leaves_the_config_private_and_no_temp_file_behind() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let mut cfg =
        app_config_with_cloud(true, "sk-secret", Some(CloudAsrProvider::OpenAiCompatible));
    cfg.save(dir.path()).unwrap();

    let mode = std::fs::metadata(dir.path().join("config.json"))
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o600, "mode was {:o}", mode & 0o777);

    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .filter(|n| n != "config.json")
        .collect();
    assert!(leftovers.is_empty(), "stray files: {leftovers:?}");
}

#[test]
fn save_normalizes_before_writing_to_disk() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = asr_cloud_config("local_whisper", Some(CloudAsrProvider::Deepgram), "", "");
    cfg.save(dir.path()).unwrap();

    let loaded = AppConfig::load(dir.path()).unwrap();
    assert_eq!(loaded.asr.cloud_base_url, "https://api.deepgram.com");
    assert_eq!(loaded.asr.cloud_model, "nova-3");
    assert_eq!(loaded.asr, cfg.asr, "in-memory and disk must agree");
}

#[test]
fn save_persists_the_demotion_of_an_active_cloud_backend() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = asr_cloud_config(
        "cloud",
        Some(CloudAsrProvider::OpenAiCompatible),
        "",
        "whisper-1",
    );
    cfg.save(dir.path()).unwrap();

    let loaded = AppConfig::load(dir.path()).unwrap();
    assert_eq!(loaded.asr.backend, "");
    assert_eq!(loaded.asr.cloud_base_url, "https://api.openai.com");
}

// `set_config` is an in-memory apply, not a persist; normalizing here would
// diverge memory from disk on any path that applies without saving.
#[tokio::test]
async fn set_config_does_not_normalize() {
    let engine = LensEngine::for_test().await;
    let mut cfg = engine.config().await;
    cfg.asr.backend = "cloud".to_string();
    cfg.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    cfg.asr.cloud_base_url = String::new();
    cfg.asr.cloud_model = String::new();
    engine.set_config(cfg).await;

    let back = engine.config().await;
    assert_eq!(back.asr.backend, "cloud");
    assert_eq!(back.asr.cloud_base_url, "");
    assert_eq!(back.asr.cloud_model, "");
}

#[test]
fn preflight_all_present_returns_ok() {
    let cfg = app_config_with_cloud(true, "sk-test", Some(CloudAsrProvider::OpenAiCompatible));
    assert!(preflight_check(&cfg).is_ok());
}

// ===========================================================================
// Integration helpers
// ===========================================================================

/// Minimal flat PCM — just enough samples to produce a non-empty WAV body.
fn tiny_pcm() -> Vec<f32> {
    vec![0.1_f32; 160] // 10 ms at 16 kHz
}

fn openai_segments_response() -> serde_json::Value {
    serde_json::json!({
        "segments": [
            { "start": 0.0, "end": 1.5, "text": " hello from openai" },
            { "start": 1.5, "end": 3.0, "text": " goodbye from openai" }
        ]
    })
}

fn deepgram_utterances_response() -> serde_json::Value {
    serde_json::json!({
        "results": {
            "utterances": [
                { "start": 0.0, "end": 1.2, "transcript": "hello from deepgram" },
                { "start": 1.2, "end": 2.8, "transcript": "goodbye from deepgram" }
            ]
        }
    })
}

// ===========================================================================
// Integration: OpenAI happy path
// ===========================================================================

#[tokio::test]
async fn openai_happy_path_maps_segments_correctly() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::OpenAiCompatible,
        server.uri(),
        "whisper-1",
        "sk-test",
        reqwest::Client::new(),
    );

    let TranscriptOutput { segments: out, .. } = engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect("happy-path openai");

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].text, "hello from openai");
    assert!((out[0].start_second - 0.0).abs() < 1e-4);
    assert!((out[0].end_second - 1.5).abs() < 1e-4);
    assert_eq!(out[1].text, "goodbye from openai");
    assert!((out[1].start_second - 1.5).abs() < 1e-4);
    assert!((out[1].end_second - 3.0).abs() < 1e-4);
}

#[tokio::test]
async fn openai_request_carries_bearer_auth_and_multipart_fields() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .and(header("Authorization", "Bearer sk-bearer-test"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::OpenAiCompatible,
        server.uri(),
        "whisper-1",
        "sk-bearer-test",
        reqwest::Client::new(),
    );

    engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect("bearer auth test");

    let calls = server.received_requests().await.unwrap();
    assert_eq!(calls.len(), 1);
    let body_str = String::from_utf8_lossy(&calls[0].body);
    // multipart should contain model and response_format
    assert!(
        body_str.contains("whisper-1"),
        "body must contain model name: {body_str}"
    );
    assert!(
        body_str.contains("verbose_json"),
        "body must contain response_format: {body_str}"
    );
}

#[tokio::test]
async fn cloud_chunk_retries_then_succeeds_on_transient_5xx() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(2)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .with_priority(2)
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::OpenAiCompatible,
        server.uri(),
        "whisper-1",
        "sk-test",
        reqwest::Client::new(),
    )
    .with_retry_policy(3, std::time::Duration::from_millis(1));

    engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect("two transient 5xx then success must recover via retry");

    let calls = server.received_requests().await.unwrap();
    assert_eq!(calls.len(), 3, "expected 2 failed attempts + 1 success");
}

#[tokio::test]
async fn cloud_chunk_gives_up_after_max_retries() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::OpenAiCompatible,
        server.uri(),
        "whisper-1",
        "sk-test",
        reqwest::Client::new(),
    )
    .with_retry_policy(2, std::time::Duration::from_millis(1));

    let err = engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect_err("persistent 5xx must propagate after retries are exhausted");
    assert!(
        matches!(err, lens_core::LensError::Network(_)),
        "a retried-out 5xx surfaces as Network, got {err:?}"
    );

    let calls = server.received_requests().await.unwrap();
    assert_eq!(calls.len(), 3, "expected 1 initial attempt + 2 retries");
}

// ===========================================================================
// Integration: Deepgram happy path
// ===========================================================================

#[tokio::test]
async fn deepgram_happy_path_maps_utterances_correctly() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/listen"))
        .respond_with(ResponseTemplate::new(200).set_body_json(deepgram_utterances_response()))
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::Deepgram,
        server.uri(),
        "nova-3",
        "dg-key",
        reqwest::Client::new(),
    );

    let TranscriptOutput { segments: out, .. } = engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect("happy-path deepgram");

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].text, "hello from deepgram");
    assert!((out[0].start_second - 0.0).abs() < 1e-4);
    assert!((out[0].end_second - 1.2).abs() < 1e-4);
    assert_eq!(out[1].text, "goodbye from deepgram");
}

#[tokio::test]
async fn deepgram_request_carries_token_auth_and_query_params() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/listen"))
        .and(header("Authorization", "Token dg-token-test"))
        .and(query_param("encoding", "linear32"))
        .and(query_param("utterances", "true"))
        .respond_with(ResponseTemplate::new(200).set_body_json(deepgram_utterances_response()))
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::Deepgram,
        server.uri(),
        "nova-3",
        "dg-token-test",
        reqwest::Client::new(),
    );

    engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect("deepgram token auth + query params");
}

#[tokio::test]
async fn deepgram_request_content_type_is_audio_raw() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/listen"))
        .and(header("Content-Type", "audio/raw"))
        .respond_with(ResponseTemplate::new(200).set_body_json(deepgram_utterances_response()))
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::Deepgram,
        server.uri(),
        "nova-3",
        "dg-key",
        reqwest::Client::new(),
    );

    engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect("deepgram content-type audio/raw");
}

// ===========================================================================
// Integration: error mapping for both providers
// ===========================================================================

/// Helper: assert a given HTTP status maps to the expected LensError kind.
async fn assert_status_maps_to(
    provider: CloudAsrProvider,
    path_str: &str,
    status: u16,
    expected_kind: &str,
) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(path_str))
        .respond_with(ResponseTemplate::new(status))
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        provider,
        server.uri(),
        "model",
        "key",
        reqwest::Client::new(),
    )
    .with_retry_policy(0, std::time::Duration::ZERO);

    let err = engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect_err(&format!("status {status} must error"));
    assert_eq!(
        err.kind(),
        expected_kind,
        "status {status}: expected {expected_kind}, got {} ({})",
        err.kind(),
        err.message()
    );
}

#[tokio::test]
async fn openai_401_maps_to_validation() {
    assert_status_maps_to(
        CloudAsrProvider::OpenAiCompatible,
        "/v1/audio/transcriptions",
        401,
        "Validation",
    )
    .await;
}

#[tokio::test]
async fn openai_413_maps_to_validation() {
    assert_status_maps_to(
        CloudAsrProvider::OpenAiCompatible,
        "/v1/audio/transcriptions",
        413,
        "Validation",
    )
    .await;
}

#[tokio::test]
async fn openai_429_maps_to_network() {
    assert_status_maps_to(
        CloudAsrProvider::OpenAiCompatible,
        "/v1/audio/transcriptions",
        429,
        "Network",
    )
    .await;
}

#[tokio::test]
async fn openai_500_maps_to_network() {
    assert_status_maps_to(
        CloudAsrProvider::OpenAiCompatible,
        "/v1/audio/transcriptions",
        500,
        "Network",
    )
    .await;
}

#[tokio::test]
async fn deepgram_401_maps_to_validation() {
    assert_status_maps_to(CloudAsrProvider::Deepgram, "/v1/listen", 401, "Validation").await;
}

#[tokio::test]
async fn deepgram_413_maps_to_validation() {
    assert_status_maps_to(CloudAsrProvider::Deepgram, "/v1/listen", 413, "Validation").await;
}

#[tokio::test]
async fn deepgram_429_maps_to_network() {
    assert_status_maps_to(CloudAsrProvider::Deepgram, "/v1/listen", 429, "Network").await;
}

#[tokio::test]
async fn deepgram_500_maps_to_network() {
    assert_status_maps_to(CloudAsrProvider::Deepgram, "/v1/listen", 500, "Network").await;
}

#[tokio::test]
async fn openai_malformed_json_200_maps_to_transcription_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("not valid json {{{")
                .insert_header("content-type", "application/json"),
        )
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::OpenAiCompatible,
        server.uri(),
        "whisper-1",
        "sk-test",
        reqwest::Client::new(),
    );

    let err = engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect_err("malformed JSON must error");
    assert_eq!(err.kind(), "Transcription", "malformed 200: got {err:?}");
}

#[tokio::test]
async fn deepgram_malformed_json_200_maps_to_transcription_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/listen"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("{bad json")
                .insert_header("content-type", "application/json"),
        )
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::Deepgram,
        server.uri(),
        "nova-3",
        "dg-key",
        reqwest::Client::new(),
    );

    let err = engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect_err("malformed JSON must error");
    assert_eq!(err.kind(), "Transcription", "malformed 200: got {err:?}");
}

// ===========================================================================
// Integration: pre-flight gates → zero wiremock requests
// ===========================================================================

/// Runs `LensEngine::transcribe` with the cloud backend configured. Asserts the
/// wiremock received zero requests (pre-flight blocked), and that the fallback
/// produced the mock's canned segments.
async fn assert_preflight_blocks_with_zero_requests(
    server: &MockServer,
    cfg_override: impl FnOnce(&mut AppConfig),
) {
    let engine = LensEngine::for_test().await;

    // Set backend=cloud but override via the caller's closure
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_base_url = server.uri();
    config.asr.cloud_model = "whisper-1".to_string();
    cfg_override(&mut config);
    engine.set_config(config).await;

    // Inject mock Apple engine as fallback (uses apple_native seam)
    let canned = vec![TranscriptSegment {
        text: "local fallback".into(),
        start_second: 0.0,
        end_second: 1.0,
    }];
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(canned.clone()))))
        .await;

    let pcm = tiny_pcm();
    let (out, _backend) = engine
        .transcribe(&pcm, &TranscribeConfig::default(), None, None)
        .await
        .expect("pre-flight blocked → local fallback should succeed");

    assert_eq!(out, canned, "fallback must return mock segments");

    let received = server.received_requests().await.unwrap();
    assert_eq!(
        received.len(),
        0,
        "pre-flight must issue ZERO cloud requests, got {}",
        received.len()
    );
}

#[tokio::test]
async fn preflight_no_consent_zero_cloud_requests() {
    let server = MockServer::start().await;
    // Mount a catch-all so any request would be recorded
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    assert_preflight_blocks_with_zero_requests(&server, |cfg| {
        cfg.audio_cloud_consent = false;
        cfg.asr.cloud_api_key = "sk-test".to_string();
        cfg.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    })
    .await;
}

#[tokio::test]
async fn preflight_no_key_zero_cloud_requests() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    assert_preflight_blocks_with_zero_requests(&server, |cfg| {
        cfg.audio_cloud_consent = true;
        cfg.asr.cloud_api_key = "".to_string(); // empty key
        cfg.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    })
    .await;
}

#[tokio::test]
async fn preflight_no_provider_zero_cloud_requests() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    assert_preflight_blocks_with_zero_requests(&server, |cfg| {
        cfg.audio_cloud_consent = true;
        cfg.asr.cloud_api_key = "sk-test".to_string();
        cfg.asr.cloud_provider = None; // no provider
    })
    .await;
}

// Zero-requests alone cannot discriminate here: a blank base URL also fails while
// BUILDING the request, so the count is 0 with the gate deleted too. Asserting the
// surfaced error instead — with no local engine to fall back to — does discriminate.
#[tokio::test]
async fn preflight_blank_base_url_surfaces_the_gate_error_with_zero_cloud_requests() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.audio_cloud_consent = true;
    config.asr.cloud_api_key = "sk-test".to_string();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    config.asr.cloud_base_url = "   ".to_string();
    config.asr.cloud_model = "whisper-1".to_string();
    engine.set_config(config).await;

    let err = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .expect_err("no local engine is injected, so the cloud error must surface");
    assert_eq!(err.kind(), "Validation", "got {err:?}");
    assert_eq!(
        err.message(),
        "no cloud ASR base URL configured",
        "must be the blank-field gate, not the scheme gate or a request-build failure"
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 0);
}

// The helper persists via `set_config` (in-memory, no `save`), so `normalize`
// never runs and the blank reaches pre-flight intact.
#[tokio::test]
async fn preflight_blank_model_zero_cloud_requests() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    assert_preflight_blocks_with_zero_requests(&server, |cfg| {
        cfg.audio_cloud_consent = true;
        cfg.asr.cloud_api_key = "sk-test".to_string();
        cfg.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
        cfg.asr.cloud_model = String::new();
    })
    .await;
}

// ===========================================================================
// Integration: router gate 1 — a demoted backend can never reach Cloud
// ===========================================================================

/// What makes a demoted backend durable: with `backend: ""` the router has no
/// explicit override, so the Cloud arm is unreachable however usable the cloud
/// block is. A Cloud selection would show up as a `(fallback)` marker instead.
#[tokio::test]
async fn empty_backend_never_selects_cloud_despite_a_usable_cloud_block() {
    let engine = LensEngine::for_test().await;

    let mut config = engine.config().await;
    config.audio_cloud_consent = true;
    config.asr.backend = String::new();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    config.asr.cloud_base_url = "https://api.openai.com".to_string();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-test".to_string();
    engine.set_config(config).await;

    let canned = vec![TranscriptSegment {
        text: "local only".into(),
        start_second: 0.0,
        end_second: 1.0,
    }];
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(canned.clone()))))
        .await;

    let resolved = engine
        .resolve_asr_backend(None, false)
        .await
        .expect("resolve");
    assert_ne!(
        resolved,
        AsrBackend::Cloud,
        "the resolution seam the UI reads must not report cloud either"
    );

    let (out, backend) = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .expect("router must resolve to the injected local engine");

    assert_eq!(out, canned);
    assert_eq!(
        backend, "apple_native",
        "a demoted backend must not route through cloud, got {backend}"
    );
}

// ===========================================================================
// Integration: LensEngine::transcribe cloud → fallback on HTTP errors
// ===========================================================================

/// Configures `LensEngine` for cloud with a wiremock that returns `status`, then
/// asserts the error triggers fallback to the injected MockAsrEngine.
async fn assert_cloud_error_falls_back_to_mock(
    server: &MockServer,
    _route_path: &str,
    status: u16,
    expected_label: &str,
) {
    let engine = LensEngine::for_test().await;

    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_base_url = server.uri();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-test".to_string();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    config.audio_cloud_consent = true;
    engine.set_config(config).await;

    let canned = vec![TranscriptSegment {
        text: "fallback segment".into(),
        start_second: 0.0,
        end_second: 1.0,
    }];
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(canned.clone()))))
        .await;

    let (out, label) = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .unwrap_or_else(|e| panic!("status {status} must fallback, not hard-fail: {e:?}"));

    assert_eq!(out, canned, "fallback segments must match mock");
    assert_eq!(
        label, expected_label,
        "status {status} must produce its own degrade marker"
    );
}

/// A rejected key only surfaces at request time — pre-flight cannot know the provider
/// will refuse it — so this is where the misconfigured marker has to be earned.
#[tokio::test]
async fn cloud_401_falls_back_to_local_and_is_marked_misconfigured() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(401))
        .mount(&server)
        .await;
    assert_cloud_error_falls_back_to_mock(
        &server,
        "/v1/audio/transcriptions",
        401,
        "apple_native (cloud misconfigured)",
    )
    .await;
}

/// 413 is `Validation` like a rejected key, but an oversized payload is not something
/// the user fixes in Settings — tagging on the variant would misdirect them.
#[tokio::test]
async fn cloud_413_triggers_fallback_to_local() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(413))
        .mount(&server)
        .await;
    assert_cloud_error_falls_back_to_mock(
        &server,
        "/v1/audio/transcriptions",
        413,
        "apple_native (fallback)",
    )
    .await;
}

#[tokio::test]
async fn cloud_429_triggers_fallback_to_local() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(429))
        .mount(&server)
        .await;
    assert_cloud_error_falls_back_to_mock(
        &server,
        "/v1/audio/transcriptions",
        429,
        "apple_native (fallback)",
    )
    .await;
}

#[tokio::test]
async fn cloud_500_triggers_fallback_to_local() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    assert_cloud_error_falls_back_to_mock(
        &server,
        "/v1/audio/transcriptions",
        500,
        "apple_native (fallback)",
    )
    .await;
}

#[tokio::test]
async fn cloud_malformed_200_triggers_fallback_to_local() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("{bad}")
                .insert_header("content-type", "application/json"),
        )
        .mount(&server)
        .await;
    assert_cloud_error_falls_back_to_mock(
        &server,
        "/v1/audio/transcriptions",
        200,
        "apple_native (fallback)",
    )
    .await;
}

// ===========================================================================
// Integration: effective-backend transparency via LensEngine::transcribe
// ===========================================================================

#[tokio::test]
async fn transcribe_cloud_success_returns_cloud_label() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_base_url = server.uri();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-label-test".to_string();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    config.audio_cloud_consent = true;
    engine.set_config(config).await;

    let (_segs, label) = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .expect("cloud success");
    assert_eq!(label, "cloud");
}

#[tokio::test]
async fn transcribe_cloud_fallback_returns_fallback_label() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_base_url = server.uri();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-fallback-test".to_string();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    config.audio_cloud_consent = true;
    engine.set_config(config).await;

    let canned = vec![TranscriptSegment {
        text: "fallback".into(),
        start_second: 0.0,
        end_second: 1.0,
    }];
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(canned))))
        .await;

    let (_segs, label) = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .expect("500 → fallback");
    assert!(
        label.contains("fallback"),
        "label must contain fallback: {label}"
    );
}

/// Drives the Cloud arm against `base_url` with a local engine injected, returning the
/// `effective_backend` label the degradation produced.
async fn cloud_degradation_label(base_url: &str) -> String {
    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_base_url = base_url.to_string();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-label-test".to_string();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    config.audio_cloud_consent = true;
    engine.set_config(config).await;

    let canned = vec![TranscriptSegment {
        text: "local".into(),
        start_second: 0.0,
        end_second: 1.0,
    }];
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(canned))))
        .await;

    let (_segs, label) = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .expect("a cloud failure must degrade to local, not hard-fail");
    label.to_string()
}

/// A pre-flight rejection is the user's to fix and permanent until they do; a provider
/// 500 is neither. Collapsing the two markers would bury the actionable case inside the
/// retryable one, so both exact strings and their inequality are pinned.
#[tokio::test]
async fn a_preflight_rejection_and_a_transient_failure_get_distinct_labels() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    let transient = cloud_degradation_label(&server.uri()).await;

    // Cleartext on a public host: rejected by the transport gate before any request, and
    // parseable, so deleting that gate would emit a request and flip this to `(fallback)`.
    let misconfigured = cloud_degradation_label("http://asr.example.com").await;

    assert_eq!(transient, "apple_native (fallback)");
    assert_eq!(misconfigured, "apple_native (cloud misconfigured)");
    assert_ne!(misconfigured, transient);
}

// ===========================================================================
// Integration: chunked transcription (multiple wiremock calls, stitched timestamps)
// ===========================================================================

#[tokio::test]
async fn chunked_transcription_calls_server_multiple_times_and_stitches() {
    let server = MockServer::start().await;

    // The mock returns different responses each call:
    // call 1 → chunk 1 segments; call 2 → chunk 2 segments
    // wiremock 0.6 doesn't support per-call ordering easily, so return the same body
    // but we verify 2+ calls and that the stitched output has re-offset timestamps.
    let chunk_response = serde_json::json!({
        "segments": [
            { "start": 0.0, "end": 2.0, "text": "chunk segment" }
        ]
    });
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chunk_response))
        .mount(&server)
        .await;

    // Build a PCM buffer large enough to split into 2 chunks (> 25 MB WAV)
    // 14 million f32 samples → WAV ~= 44 + 28_000_000 bytes ≈ 26.7 MB > 25 MB cap
    let n = 14_000_000usize;
    let pcm = vec![0.1_f32; n];

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::OpenAiCompatible,
        server.uri(),
        "whisper-1",
        "sk-chunk-test",
        reqwest::Client::new(),
    );

    let TranscriptOutput { segments: out, .. } = engine
        .transcribe_pcm(&pcm, &TranscribeConfig::default(), None)
        .await
        .expect("chunked transcription");

    let received = server.received_requests().await.unwrap();
    assert!(
        received.len() >= 2,
        "over-limit PCM must produce 2+ requests, got {}",
        received.len()
    );

    // Stitched output should have segments from all chunks; timestamps must be non-decreasing
    assert!(out.len() >= 2, "stitched output must have 2+ segments");
    for w in out.windows(2) {
        assert!(
            w[1].start_second >= w[0].start_second - 1e-4,
            "stitched segments must be monotonic: {:?} then {:?}",
            w[0],
            w[1]
        );
    }
    // Second chunk's segment must be offset from 0 (the chunk starts after the first chunk)
    assert!(
        out[1].start_second > 0.0,
        "second chunk segment must be offset from zero, got {}",
        out[1].start_second
    );
}

// ===========================================================================
// Integration: consent isolation — audio_cloud_consent independent from enrichment.cloud_consent
// ===========================================================================

#[tokio::test]
async fn consent_isolation_audio_consent_true_enrichment_false_allows_cloud() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_base_url = server.uri();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-isolation-test".to_string();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    // audio consent ON, enrichment consent OFF
    config.audio_cloud_consent = true;
    config.enrichment.cloud_consent = false;
    engine.set_config(config).await;

    let (_segs, label) = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .expect(
            "audio_cloud_consent=true must allow cloud even when enrichment.cloud_consent=false",
        );
    assert_eq!(label, "cloud", "must use cloud: {label}");

    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1, "must have made exactly 1 cloud request");
}

#[tokio::test]
async fn consent_isolation_audio_consent_false_enrichment_true_blocks_cloud() {
    let server = MockServer::start().await;
    // Mount a catch-all to detect any unauthorized request
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_base_url = server.uri();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-isolation-test".to_string();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    // audio consent OFF, enrichment consent ON
    config.audio_cloud_consent = false;
    config.enrichment.cloud_consent = true;
    engine.set_config(config).await;

    // Inject mock for fallback
    let canned = vec![TranscriptSegment {
        text: "local only".into(),
        start_second: 0.0,
        end_second: 1.0,
    }];
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(canned.clone()))))
        .await;

    let (out, _label) = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .expect("audio_cloud_consent=false must fallback, not hard-fail");
    assert_eq!(out, canned, "must use local fallback");

    let received = server.received_requests().await.unwrap();
    assert_eq!(
        received.len(),
        0,
        "enrichment.cloud_consent=true must NOT unblock audio cloud requests"
    );
}

// ===========================================================================
// Integration: dead-port (unreachable) triggers fallback (1 failed request expected)
// ===========================================================================

#[tokio::test]
async fn unreachable_cloud_triggers_local_fallback() {
    // Use a port that is almost certainly not listening (high ephemeral port)
    let dead_url = "http://127.0.0.1:19823";

    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_base_url = dead_url.to_string();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-dead-test".to_string();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    config.audio_cloud_consent = true;
    engine.set_config(config).await;

    let canned = vec![TranscriptSegment {
        text: "local result".into(),
        start_second: 0.0,
        end_second: 1.0,
    }];
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(canned.clone()))))
        .await;

    let (out, label) = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .expect("dead port must degrade gracefully to local fallback");

    assert_eq!(out, canned);
    assert!(
        label.contains("fallback"),
        "label must indicate fallback: {label}"
    );
}

// ===========================================================================
// Unit: select_asr_backend Cloud gate-1 passthrough
// ===========================================================================

#[test]
fn select_asr_backend_cloud_config_override_is_passthrough() {
    use lens_core::asr::{Platform, select_asr_backend};

    // Gate-1: explicit Cloud override wins regardless of platform/availability.
    let non_apple = Platform {
        is_apple_silicon_macos: false,
        macos_major: None,
    };
    let result = select_asr_backend(Some(lens_core::AsrBackend::Cloud), non_apple, false, false);
    assert_eq!(
        result,
        lens_core::AsrBackend::Cloud,
        "explicit Cloud config override must pass through unconditionally"
    );
}

// ===========================================================================
// Unit: IngestProgress effective_backend serde
// ===========================================================================

#[test]
fn ingest_progress_effective_backend_roundtrip_and_omit() {
    // With effective_backend set: must survive a JSON round-trip.
    let with_backend = lens_core::IngestProgress {
        phase: "transcribing".to_string(),
        done: 1,
        total: Some(1),
        effective_backend: Some("cloud".to_string()),
    };
    let json = serde_json::to_string(&with_backend).expect("serialize with backend");
    let back: lens_core::IngestProgress = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.effective_backend, Some("cloud".to_string()));

    // Without effective_backend: the key must be absent from the JSON wire format.
    let without_backend = lens_core::IngestProgress {
        phase: "transcribing".to_string(),
        done: 1,
        total: Some(1),
        effective_backend: None,
    };
    let json_none = serde_json::to_string(&without_backend).expect("serialize without backend");
    assert!(
        !json_none.contains("effective_backend"),
        "None effective_backend must be omitted from JSON: {json_none}"
    );
}

/// The pre-flight validates the TRIMMED base URL, so the request has to use the same
/// value — otherwise a stray space clears the gate and dies at request time, which is
/// exactly the outcome the gate exists to prevent.
#[tokio::test]
async fn cloud_request_uses_the_trimmed_endpoint_the_preflight_validated() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    config.asr.cloud_base_url = format!("  {}  ", server.uri());
    config.asr.cloud_model = "  whisper-1  ".to_string();
    config.asr.cloud_api_key = "  sk-test  ".to_string();
    config.audio_cloud_consent = true;
    engine.set_config(config).await;

    let (_segments, label) = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .expect("a padded but otherwise valid endpoint must still reach the cloud");
    assert_eq!(label, "cloud");
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

/// When nothing produces segments, the broken cloud config is the failure the user
/// can act on; the local error that followed it is a symptom and must not mask it.
#[tokio::test]
async fn a_cloud_misconfiguration_outranks_the_local_failure_that_follows_it() {
    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    config.asr.cloud_base_url = "https://api.openai.com".to_string();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-test".to_string();
    config.audio_cloud_consent = false;
    engine.set_config(config).await;
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::failing(
            "apple on-device asset missing",
        ))))
        .await;

    let err = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .expect_err("neither cloud nor local can produce segments");
    assert_eq!(err.kind(), "Validation", "got {err:?}");
    assert_eq!(err.message(), "audio cloud consent not granted");
}

// ===========================================================================
// Integration: cancellation during an in-flight cloud request (#135)
// ===========================================================================

fn fallback_segments() -> Vec<TranscriptSegment> {
    vec![TranscriptSegment {
        text: "local fallback ran".into(),
        start_second: 0.0,
        end_second: 1.0,
    }]
}

/// Builds a cloud-backed engine pointed at `base_url` with `local` injected as the Apple
/// seam the cloud arm degrades to.
async fn cloud_engine_with(base_url: &str, local: MockAsrEngine) -> LensEngine {
    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_base_url = base_url.to_string();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-cancel-test".to_string();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    config.audio_cloud_consent = true;
    engine.set_config(config).await;

    engine.set_asr_engine(Some(Arc::new(local))).await;
    engine
}

/// The injected local engine SUCCEEDS, so any result other than `Cancelled` proves a
/// fallback ran. The returned sink stays empty unless that engine was actually invoked,
/// which states the "zero local-engine invocation" criterion directly rather than
/// inferring it from the returned error.
async fn cloud_engine_with_working_local(
    base_url: &str,
) -> (LensEngine, Arc<std::sync::Mutex<Option<TranscribeConfig>>>) {
    let sink = Arc::new(std::sync::Mutex::new(None));
    let local = MockAsrEngine::new(fallback_segments()).recording_config(Arc::clone(&sink));
    (cloud_engine_with(base_url, local).await, sink)
}

fn assert_local_untouched(sink: &Arc<std::sync::Mutex<Option<TranscribeConfig>>>) {
    assert!(
        sink.lock().expect("sink lock").is_none(),
        "the local fallback engine was invoked for a cancelled clip"
    );
}

/// Trips `token` when the request arrives, then answers with `response`. Tripping from
/// the responder makes "the cancel landed mid-request" deterministic — a timer-based
/// cancel races the scheduler and flakes on a loaded runner.
struct CancelOnRequest {
    token: tokio_util::sync::CancellationToken,
    response: ResponseTemplate,
}

impl wiremock::Respond for CancelOnRequest {
    fn respond(&self, _request: &wiremock::Request) -> ResponseTemplate {
        self.token.cancel();
        self.response.clone()
    }
}

/// #135 core AC: a cancel that lands while the request is in flight must abort the
/// request itself, not wait it out. The endpoint stalls, so the elapsed-time assertion is
/// what separates a real in-flight abort from a post-request boundary check.
#[tokio::test]
async fn a_cancel_during_an_in_flight_cloud_request_aborts_without_local_fallback() {
    let server = MockServer::start().await;
    let token = tokio_util::sync::CancellationToken::new();
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(CancelOnRequest {
            token: token.clone(),
            response: ResponseTemplate::new(200)
                .set_body_json(openai_segments_response())
                .set_delay(std::time::Duration::from_secs(30)),
        })
        .mount(&server)
        .await;

    let (engine, sink) = cloud_engine_with_working_local(&server.uri()).await;

    let started = std::time::Instant::now();
    let err = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, Some(token))
        .await
        .expect_err("a cancelled clip must not return cloud or fallback segments");
    let elapsed = started.elapsed();

    assert_eq!(err.kind(), "Cancelled", "message: {}", err.message());
    assert_local_untouched(&sink);
    // Load-independent proof this was an in-flight abort and not a never-issued request:
    // wiremock records on receipt, and the responder is the only thing that trips the token.
    assert!(
        !server
            .received_requests()
            .await
            .expect("the mock server records requests")
            .is_empty(),
        "the request was never issued, so this proves nothing about in-flight aborts"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "the in-flight request was waited out rather than aborted: {elapsed:?}"
    );
}

/// A cancel arriving while a NON-cancel cloud failure is in flight; the responder makes
/// that race deterministic. Ungated, the 500 would degrade to local and return segments.
#[tokio::test]
async fn a_cancel_racing_a_cloud_failure_does_not_start_the_local_fallback() {
    let token = tokio_util::sync::CancellationToken::new();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(CancelOnRequest {
            token: token.clone(),
            response: ResponseTemplate::new(500),
        })
        .mount(&server)
        .await;

    let (engine, sink) = cloud_engine_with_working_local(&server.uri()).await;

    let err = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, Some(token))
        .await
        .expect_err("a cancel racing a cloud failure must not degrade to local");

    assert_eq!(err.kind(), "Cancelled", "message: {}", err.message());
    assert_local_untouched(&sink);
}

/// The cloud→local cascade has its own Apple leg, equally uninterruptible: a cancel
/// landing there must discard the fallback's result rather than return it. The 418 is
/// deliberately non-retryable, so the cascade reaches Apple without any backoff sleeps.
#[tokio::test]
async fn a_cancel_during_the_fallback_apple_run_discards_its_result() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(418))
        .mount(&server)
        .await;

    let token = tokio_util::sync::CancellationToken::new();
    let local = MockAsrEngine::new(fallback_segments()).cancelling(token.clone());
    let engine = cloud_engine_with(&server.uri(), local).await;

    let err = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, Some(token))
        .await
        .expect_err("a cancel during the fallback Apple run must not return its segments");

    assert_eq!(err.kind(), "Cancelled", "message: {}", err.message());
}

/// An already-cancelled clip must issue ZERO requests — nothing sent, billed, or logged
/// upstream.
#[tokio::test]
async fn an_already_cancelled_clip_issues_no_cloud_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    let (engine, sink) = cloud_engine_with_working_local(&server.uri()).await;
    let token = tokio_util::sync::CancellationToken::new();
    token.cancel();

    let err = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, Some(token))
        .await
        .expect_err("an already-cancelled clip must not transcribe");

    assert_eq!(err.kind(), "Cancelled", "message: {}", err.message());
    assert_local_untouched(&sink);
    let requests = server
        .received_requests()
        .await
        .expect("the mock server records requests");
    assert!(
        requests.is_empty(),
        "a cancelled clip must not reach the provider: {} request(s)",
        requests.len()
    );
}

/// The gate must key on the token being TRIPPED, not merely present. The 418 is
/// non-retryable, so this control costs no backoff sleeps.
#[tokio::test]
async fn a_live_token_leaves_the_cloud_fallback_intact() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(418))
        .mount(&server)
        .await;

    let (engine, _sink) = cloud_engine_with_working_local(&server.uri()).await;

    let (segments, label) = engine
        .transcribe(
            &tiny_pcm(),
            &TranscribeConfig::default(),
            None,
            Some(tokio_util::sync::CancellationToken::new()),
        )
        .await
        .expect("an uncancelled 500 must still degrade to local");

    assert_eq!(label, "apple_native (fallback)");
    assert_eq!(segments[0].text, "local fallback ran");
}
