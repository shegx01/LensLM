//! In-memory PCM→WAV wrapper for cloud providers that require a container (#45).
//! Converts 16 kHz mono f32 PCM to 16-bit signed PCM and prepends a 44-byte
//! RIFF/WAV header. No temp file: the whole WAV is built in a `Vec<u8>`.

use crate::LensError;

/// Bytes in a canonical PCM WAV header (RIFF + fmt + data chunk descriptors).
pub const WAV_HEADER_BYTES: usize = 44;

const BITS_PER_SAMPLE: u16 = 16;
const NUM_CHANNELS: u16 = 1;
const PCM_FORMAT: u16 = 1;

/// Wraps 16 kHz mono f32 PCM in an in-memory 16-bit PCM WAV. Samples are clamped
/// to `[-1.0, 1.0]` and scaled to `i16`; the header carries the exact data length.
/// Returns `Err` when the PCM data length overflows the 32-bit WAV data chunk field.
pub fn pcm_to_wav(pcm: &[f32], sample_rate: u32) -> Result<Vec<u8>, LensError> {
    let bytes_per_sample = (BITS_PER_SAMPLE / 8) as u32;
    // Compute data_len in u64 first to detect overflow before casting.
    let data_len_u64 = pcm.len() as u64 * bytes_per_sample as u64;
    if data_len_u64 > u32::MAX as u64 {
        return Err(LensError::Validation(
            "PCM too large for WAV container".into(),
        ));
    }
    let data_len = data_len_u64 as u32;
    let byte_rate = sample_rate * NUM_CHANNELS as u32 * bytes_per_sample;
    let block_align = NUM_CHANNELS * (BITS_PER_SAMPLE / 8);
    // RIFF chunk size = 36 + data (everything after the first 8 bytes).
    let riff_size = 36u32.saturating_add(data_len);

    let mut out = Vec::with_capacity(WAV_HEADER_BYTES + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&PCM_FORMAT.to_le_bytes());
    out.extend_from_slice(&NUM_CHANNELS.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());

    for &s in pcm {
        let clamped = s.clamp(-1.0, 1.0);
        let scaled = (clamped * i16::MAX as f32) as i16;
        out.extend_from_slice(&scaled.to_le_bytes());
    }
    Ok(out)
}

/// Encoded WAV size (header + 16-bit samples) for `n` f32 samples, without
/// allocating — used by the chunker to size windows against a provider byte cap.
pub fn wav_encoded_len(sample_count: usize) -> usize {
    WAV_HEADER_BYTES + sample_count * (BITS_PER_SAMPLE as usize / 8)
}
