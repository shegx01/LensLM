use std::path::Path;

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use crate::dialogue::Speaker;
use crate::error::LensError;

pub const TARGET_RATE: u32 = 24_000;

const WITHIN_GAP_MS: u32 = 350;
const CHANGE_GAP_MS: u32 = 450;
const FADE_MS: u32 = 30;
const PEAK_DBFS: f32 = -1.0;

const RESAMPLE_CHUNK_FRAMES: usize = 1024;

const MONO: usize = 0;

#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioBuffer {
    pub fn mono(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
            channels: 1,
        }
    }

    pub fn resample_to(&self, target: u32) -> Result<AudioBuffer, LensError> {
        if self.sample_rate == target {
            return Ok(self.clone());
        }
        let ratio = target as f64 / self.sample_rate as f64;
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            oversampling_factor: 128,
            interpolation: SincInterpolationType::Cubic,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler =
            SincFixedIn::<f32>::new(ratio, 1.0, params, RESAMPLE_CHUNK_FRAMES, 1)
                .map_err(|e| LensError::Internal(format!("tts resampler init failed: {e}")))?;

        // SincFixedIn requires fixed-size chunks; a one-shot whole-buffer feed errors, so chunk + process_partial tail-flush.
        let mut out: Vec<f32> = Vec::new();
        let mut idx = 0usize;
        while self.samples.len() - idx >= RESAMPLE_CHUNK_FRAMES {
            let chunk = self.samples[idx..idx + RESAMPLE_CHUNK_FRAMES].to_vec();
            let res = resampler.process(&[chunk], None).map_err(resample_err)?;
            out.extend_from_slice(&res[MONO]);
            idx += RESAMPLE_CHUNK_FRAMES;
        }
        if idx < self.samples.len() {
            let tail = self.samples[idx..].to_vec();
            let res = resampler
                .process_partial(Some(&[tail]), None)
                .map_err(resample_err)?;
            out.extend_from_slice(&res[MONO]);
        }
        let flushed = resampler
            .process_partial(None::<&[Vec<f32>]>, None)
            .map_err(resample_err)?;
        out.extend_from_slice(&flushed[MONO]);

        let expected =
            (self.samples.len() as u64 * target as u64 / self.sample_rate as u64) as usize;
        out.resize(expected, 0.0);

        Ok(AudioBuffer::mono(out, target))
    }
}

pub(crate) fn stitch_turns(buffers: &[(Speaker, AudioBuffer)]) -> Result<AudioBuffer, LensError> {
    let fade_samples = ms_to_samples(FADE_MS);
    let mut out: Vec<f32> = Vec::new();
    let mut prev: Option<Speaker> = None;

    for (speaker, buf) in buffers {
        if let Some(prev_speaker) = prev {
            let gap_ms = if prev_speaker == *speaker {
                WITHIN_GAP_MS
            } else {
                CHANGE_GAP_MS
            };
            out.resize(out.len() + ms_to_samples(gap_ms), 0.0);
        }
        let mut samples = buf.resample_to(TARGET_RATE)?.samples;
        // Edge fades over silence-separated turns, NOT an overlap-add crossfade.
        apply_edge_fades(&mut samples, fade_samples);
        out.extend_from_slice(&samples);
        prev = Some(*speaker);
    }

    peak_normalize(&mut out, PEAK_DBFS);
    Ok(AudioBuffer::mono(out, TARGET_RATE))
}

pub(crate) fn write_wav_16bit(buffer: &AudioBuffer, path: &Path) -> Result<(), LensError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: buffer.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(hound_err)?;
    for &s in &buffer.samples {
        writer
            .write_sample(crate::f32_sample_to_i16(s))
            .map_err(hound_err)?;
    }
    writer.finalize().map_err(hound_err)?;
    Ok(())
}

fn ms_to_samples(ms: u32) -> usize {
    (ms * TARGET_RATE / 1000) as usize
}

fn apply_edge_fades(samples: &mut [f32], fade: usize) {
    let n = samples.len();
    let fade = fade.min(n / 2);
    if fade == 0 {
        return;
    }
    for i in 0..fade {
        let t = i as f32 / fade as f32;
        let gain = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
        samples[i] *= gain;
        samples[n - 1 - i] *= gain;
    }
}

fn peak_normalize(samples: &mut [f32], target_dbfs: f32) {
    let max_abs = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if max_abs == 0.0 {
        return;
    }
    let target_amp = 10f32.powf(target_dbfs / 20.0);
    let gain = target_amp / max_abs;
    for s in samples.iter_mut() {
        *s *= gain;
    }
}

fn resample_err(err: rubato::ResampleError) -> LensError {
    LensError::Internal(format!("tts resample failed: {err}"))
}

fn hound_err(err: hound::Error) -> LensError {
    LensError::Io(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buf(samples: Vec<f32>) -> AudioBuffer {
        AudioBuffer::mono(samples, TARGET_RATE)
    }

    #[test]
    fn stitch_length_is_sum_of_turns_plus_gaps_with_zero_gap_region() {
        let a = buf(vec![0.5; 1000]);
        let b = buf(vec![0.5; 1000]);
        let out = stitch_turns(&[(Speaker::Host, a), (Speaker::Host, b)]).unwrap();
        let within_gap = (WITHIN_GAP_MS * TARGET_RATE / 1000) as usize;
        assert_eq!(within_gap, 8400);
        assert_eq!(out.samples.len(), 1000 + within_gap + 1000);
        assert_eq!(out.sample_rate, TARGET_RATE);
        assert_eq!(out.channels, 1);
        for &s in &out.samples[1000..1000 + within_gap] {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn stitch_speaker_change_uses_longer_gap() {
        let a = buf(vec![0.5; 500]);
        let b = buf(vec![0.5; 500]);
        let out = stitch_turns(&[(Speaker::Host, a), (Speaker::Guest, b)]).unwrap();
        let change_gap = (CHANGE_GAP_MS * TARGET_RATE / 1000) as usize;
        assert_eq!(change_gap, 10800);
        assert_eq!(out.samples.len(), 500 + change_gap + 500);
        for &s in &out.samples[500..500 + change_gap] {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn short_turn_shorter_than_fade_window_does_not_panic() {
        let out = stitch_turns(&[(Speaker::Host, buf(vec![0.9; 4]))]).unwrap();
        assert_eq!(out.samples.len(), 4);
    }

    #[test]
    fn silent_buffer_normalizes_without_divide_by_zero() {
        let out = stitch_turns(&[(Speaker::Host, buf(vec![0.0; 2000]))]).unwrap();
        assert_eq!(out.samples.len(), 2000);
        assert!(out.samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn peak_normalizes_to_minus_one_dbfs() {
        let out = stitch_turns(&[(Speaker::Host, buf(vec![0.2; 10_000]))]).unwrap();
        let peak = out.samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!((peak - 0.891_25).abs() < 1e-3, "peak was {peak}");
    }

    #[test]
    fn resample_equal_rate_is_noop_clone() {
        let src = AudioBuffer::mono(vec![0.1, -0.2, 0.3], TARGET_RATE);
        let out = src.resample_to(TARGET_RATE).unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn resample_upsample_and_downsample_length_ratio() {
        let n = 4800;
        let up = AudioBuffer::mono(vec![0.1; n], 12_000)
            .resample_to(24_000)
            .unwrap();
        assert_eq!(up.sample_rate, 24_000);
        assert_eq!(up.samples.len(), n * 24_000 / 12_000);

        let down = AudioBuffer::mono(vec![0.1; n], 48_000)
            .resample_to(24_000)
            .unwrap();
        assert_eq!(down.sample_rate, 24_000);
        assert_eq!(down.samples.len(), n * 24_000 / 48_000);
    }

    #[test]
    fn wav_round_trips_spec_and_values() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.wav");
        let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25];
        write_wav_16bit(&AudioBuffer::mono(samples.clone(), TARGET_RATE), &path).unwrap();

        let mut reader = hound::WavReader::open(&path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, TARGET_RATE);
        assert_eq!(spec.bits_per_sample, 16);
        let read: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect();
        assert_eq!(read.len(), samples.len());
        for (got, want) in read.iter().zip(samples.iter()) {
            assert!((got - want).abs() < 1.0 / 32_000.0, "got {got} want {want}");
        }
    }
}
