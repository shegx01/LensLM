//! Size-bounded PCM chunking + timestamp re-offset + stitching (#45).
//!
//! When a provider payload would exceed its byte cap, the PCM is split into
//! fixed-duration windows (boundaries refined to the nearest zero crossing to
//! avoid a click, NO VAD). Each chunk is transcribed independently; on stitch,
//! every segment's `start_second`/`end_second` is re-offset by the chunk's start
//! time so the merged timeline is global and monotonic.

use super::wav;
use crate::asr::TranscriptSegment;
use crate::config::CloudAsrProvider;

/// A window of the source PCM plus the second-offset of its first sample.
pub struct PcmChunk<'a> {
    pub data: &'a [f32],
    pub start_second: f32,
}

/// Max request payload per provider. OpenAI caps at 25 MB (WAV-encoded);
/// Deepgram accepts effectively unbounded raw PCM for our on-device sizes.
fn max_payload_bytes(provider: CloudAsrProvider) -> usize {
    match provider {
        CloudAsrProvider::OpenAiCompatible => 25 * 1024 * 1024,
        CloudAsrProvider::Deepgram => 2 * 1024 * 1024 * 1024,
    }
}

/// Encoded byte size of `sample_count` samples for `provider` (WAV for OpenAI,
/// raw 4-byte f32 for Deepgram).
fn encoded_len(provider: CloudAsrProvider, sample_count: usize) -> usize {
    match provider {
        CloudAsrProvider::OpenAiCompatible => wav::wav_encoded_len(sample_count),
        CloudAsrProvider::Deepgram => sample_count * std::mem::size_of::<f32>(),
    }
}

/// Splits `pcm` into windows whose encoded payload stays under the provider cap.
/// Under-limit input passes through as a single chunk. Window boundaries are
/// refined toward the nearest zero crossing within a small search radius.
pub fn split_if_needed(
    pcm: &[f32],
    provider: CloudAsrProvider,
    sample_rate: u32,
) -> Vec<PcmChunk<'_>> {
    if pcm.is_empty() || encoded_len(provider, pcm.len()) <= max_payload_bytes(provider) {
        return vec![PcmChunk {
            data: pcm,
            start_second: 0.0,
        }];
    }

    let max_samples = max_samples_per_chunk(provider);
    let radius = (sample_rate as usize / 100).max(1); // ±10 ms zero-crossing search
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < pcm.len() {
        let ideal_end = (start + max_samples).min(pcm.len());
        let end = if ideal_end >= pcm.len() {
            pcm.len()
        } else {
            nearest_zero_crossing(pcm, ideal_end, radius)
        };
        chunks.push(PcmChunk {
            data: &pcm[start..end],
            start_second: start as f32 / sample_rate as f32,
        });
        start = end;
    }
    chunks
}

/// Largest sample count whose encoded payload fits the provider cap, with a small
/// safety margin for the WAV header / multipart framing.
fn max_samples_per_chunk(provider: CloudAsrProvider) -> usize {
    let budget = max_payload_bytes(provider).saturating_sub(wav::WAV_HEADER_BYTES + 4096);
    match provider {
        CloudAsrProvider::OpenAiCompatible => budget / 2, // 16-bit samples
        CloudAsrProvider::Deepgram => budget / std::mem::size_of::<f32>(),
    }
}

/// Finds the sample index within `±radius` of `ideal` whose value is closest to
/// zero, so a split does not fall mid-waveform. Clamped to buffer bounds.
fn nearest_zero_crossing(pcm: &[f32], ideal: usize, radius: usize) -> usize {
    let lo = ideal.saturating_sub(radius);
    let hi = (ideal + radius).min(pcm.len().saturating_sub(1));
    let mut best = ideal.min(pcm.len().saturating_sub(1));
    let mut best_abs = f32::INFINITY;
    for (i, &v) in pcm.iter().enumerate().take(hi + 1).skip(lo) {
        let a = v.abs();
        if a < best_abs {
            best_abs = a;
            best = i;
        }
    }
    // Never return 0 for a non-first window (would make an empty chunk).
    best.max(lo.max(1))
}

/// Merges per-chunk segments into one global timeline, re-offsetting each
/// segment by its chunk start time (f64 intermediate to avoid f32 drift on long
/// files) and clamping any tiny non-monotonic overlap at chunk seams.
pub fn stitch_segments(chunks: &[(f32, Vec<TranscriptSegment>)]) -> Vec<TranscriptSegment> {
    let mut out: Vec<TranscriptSegment> = Vec::new();
    let mut last_end = 0.0f64;
    for (start_second, segments) in chunks {
        let offset = *start_second as f64;
        for seg in segments {
            let start = (seg.start_second as f64 + offset).max(last_end);
            let end = (seg.end_second as f64 + offset).max(start);
            last_end = end;
            out.push(TranscriptSegment {
                text: seg.text.clone(),
                start_second: start as f32,
                end_second: end as f32,
            });
        }
    }
    out
}
