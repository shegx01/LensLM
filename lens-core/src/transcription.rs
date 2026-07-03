//! Audio decode + resample to 16 kHz mono f32 PCM (issue #41).
//!
//! Pure, CPU-only input-normalization: turns a compressed audio file on disk
//! into the buffer shape a speech-recognition model (#42) requires — raw
//! **16 kHz, mono, f32 PCM** — without writing any temp file. It is the audio
//! analogue of the PDF/DOCX extractors: clean raw material, nothing intelligent.
//! Transcription, embedding, Tauri wiring, and UI belong to #42/#43/#44.
//!
//! # Note on module scope
//! The module is named `transcription` because #42 will add the
//! `LensAudioProcessor` trait and related transcription types here. Today it
//! holds only the decode/resample leaf functions.
//!
//! # Two public entry points (both SYNC)
//! Heavy work runs off the async runtime via `spawn_blocking` at the caller
//! (#43 owns threading; these functions are synchronous by design).
//!
//! * [`decode_resample_windows`] — **genuinely streaming**, bounded-memory
//!   iterator. Peak RAM ≈ O(window + one codec frame + one resample chunk),
//!   independent of file length. rubato's sinc FIR retains filter state across
//!   window boundaries; concatenating all windows reproduces the whole-buffer
//!   signal within float tolerance.
//! * [`decode_and_resample_audio`] — streams internally and materialises the
//!   whole buffer; additionally enforces silence rejection (the streaming path
//!   emits silent windows without error). Use when downstream code needs a
//!   contiguous `&[f32]` (e.g. #42's `process_audio`).
//!
//! # Codec scope
//! Whatever Symphonia 0.5.5 decodes with `features = ["all"]`: mp3, m4a
//! (AAC + ALAC), aac, wav, flac, caf, aiff, ogg/vorbis, adpcm, …  Opus is NOT
//! decodable in 0.5.5 (roadmap only) and maps to
//! [`LensError::UnsupportedMediaCodec`]. All failures are typed errors —
//! this module never panics on untrusted input.

use std::fs::File;
use std::path::{Path, PathBuf};

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{CODEC_TYPE_NULL, Decoder, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::LensError;

/// The target sample rate for speech recognition (#42): 16 kHz.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Fixed input-chunk size (source-rate frames) fed to the sinc resampler per
/// call. Small enough to keep one resample call's working set in L2/L3 cache;
/// large enough to amortise per-call overhead. The resampler retains FIR state
/// across calls, so this constant does not affect the output signal.
const RESAMPLE_CHUNK_FRAMES: usize = 1024;

/// Channel index for the single (mono) output channel of `SincFixedIn`.
const MONO: usize = 0;

/// Configuration for the bounded-memory windowing of [`decode_resample_windows`].
///
/// A window is a memory-bounding block on the **output** (16 kHz) side; it
/// carries no semantic meaning (semantic windowing / Whisper framing is #42's
/// job). The default is ~30 s, aligning with the Whisper input frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowConfig {
    /// Number of 16 kHz output frames per emitted window. The final window may
    /// be shorter (it carries the resampler's flushed filter tail).
    pub window_frames: usize,
}

impl WindowConfig {
    /// ~30 s at the 16 kHz target rate.
    pub const DEFAULT_WINDOW_FRAMES: usize = (TARGET_SAMPLE_RATE as usize) * 30;
}

impl Default for WindowConfig {
    fn default() -> Self {
        WindowConfig {
            window_frames: Self::DEFAULT_WINDOW_FRAMES,
        }
    }
}

/// Decodes and resamples `path` to a single 16 kHz mono f32 PCM buffer in RAM.
///
/// Streams internally via [`decode_resample_windows`] and materialises the
/// whole result. Additionally rejects empty and all-silent audio (the streaming
/// path emits silent windows without error; this fn is the only place that
/// enforces the silence-rejection policy). Callers that need a contiguous
/// `&[f32]` (e.g. the #42 `process_audio` signature) use this fn.
///
/// # Errors
/// * [`LensError::Io`] — the file could not be opened.
/// * [`LensError::UnsupportedMediaCodec`] — the codec/container is not
///   decodable by this build (e.g. `.opus`).
/// * [`LensError::MediaDecodeFailed`] — the bitstream is corrupt, truncated,
///   or produced a mid-stream format change.
/// * [`LensError::EmptyAudio`] — the file decoded successfully but yielded no
///   usable audio (empty or entirely silent PCM).
pub fn decode_and_resample_audio(path: &Path) -> Result<Vec<f32>, LensError> {
    let mut out = Vec::new();
    for window in decode_resample_windows(path, WindowConfig::default())? {
        out.extend(window?);
    }
    if out.is_empty() {
        return Err(LensError::EmptyAudio(format!(
            "{}: decoded to no audio samples",
            path.display()
        )));
    }
    // Silence check: all-zero output (e.g. silence_16000_mono.wav) has nothing
    // to transcribe. This is the single authoritative EmptyAudio guard for the
    // whole-buffer path; the streaming iterator emits silent windows without
    // error (detecting silence mid-stream would require buffering all output).
    if out.iter().all(|s| *s == 0.0) {
        return Err(LensError::EmptyAudio(format!(
            "{}: audio is empty or entirely silent",
            path.display()
        )));
    }
    Ok(out)
}

/// Decodes and resamples `path` to 16 kHz mono f32 PCM, yielded as
/// bounded-memory windows (default ~30 s each at 16 kHz; see [`WindowConfig`]).
///
/// ## Memory bound
/// Peak RAM ≈ O(`window_frames` output samples + one codec frame + one
/// `RESAMPLE_CHUNK_FRAMES` input chunk). The open file, Symphonia format
/// reader, and rubato resampler live inside the returned iterator for the
/// duration of iteration — there is no per-file buffer that grows with file
/// length.
///
/// ## Filter-state continuity
/// rubato's sinc FIR retains its internal delay line across `next()` calls, so
/// the concatenation of all yielded windows is the same signal (within float
/// tolerance) as the output of [`decode_and_resample_audio`].
///
/// ## Error model
/// * Construction errors (bad path, unsupported codec) → `Err` returned here.
/// * Mid-stream errors (corrupt packet, format-change) → `Some(Err(...))` from
///   `Iterator::next`, so the caller can decide whether to abort or skip.
///
/// ## Window sizes
/// All windows except the last are exactly `window_frames`. The final window
/// may be shorter (it drains whatever remains after EOF) or carry up to ~one
/// extra resampler chunk of flush tail — this is harmless; #42 re-frames for
/// Whisper regardless.
///
/// ## Silence
/// Silent windows are emitted without error; silence rejection is the
/// whole-buffer fn's responsibility (see [`decode_and_resample_audio`]).
pub fn decode_resample_windows(
    path: &Path,
    window: WindowConfig,
) -> Result<impl Iterator<Item = Result<Vec<f32>, LensError>>, LensError> {
    WindowIter::open(path, window)
}

/// Stateful streaming iterator returned by [`decode_resample_windows`].
///
/// Owns the Symphonia format reader + decoder and the rubato resampler for the
/// lifetime of iteration, so peak RAM is O(window) regardless of file length.
struct WindowIter {
    /// Symphonia demuxer. Drives `next_packet()`.
    format: Box<dyn FormatReader>,
    /// Symphonia codec decoder for the selected track.
    decoder: Box<dyn Decoder>,
    /// The track ID we care about (skip packets from other tracks).
    track_id: u32,
    /// rubato sinc resampler (mono, `source_rate → 16 kHz`). `None` when the
    /// source is already 16 kHz (near-identity fast path: copy only).
    resampler: Option<SincFixedIn<f32>>,
    /// Source sample rate latched on the first decoded frame.
    source_rate: u32,
    /// Channel count latched on the first decoded frame.
    channels: usize,
    /// Accumulator for source-rate mono samples not yet fed to the resampler.
    /// Carries up to `RESAMPLE_CHUNK_FRAMES - 1` leftover samples between
    /// `next()` calls.
    src_pending: Vec<f32>,
    /// Accumulator for resampled 16 kHz output not yet emitted as a window.
    out_pending: Vec<f32>,
    /// Number of 16 kHz output frames per window.
    window_frames: usize,
    /// Path retained for error messages only (no I/O after open).
    path: PathBuf,
    /// Set when the Symphonia stream reaches clean EOF; prevents further
    /// `next_packet()` calls and triggers the tail-flush sequence.
    eof: bool,
    /// Set after the tail flush has been emitted; causes `Iterator::next` to
    /// return `None`.
    done: bool,
}

impl WindowIter {
    /// Opens `path`, probes the format, selects the default audio track, and
    /// constructs the resampler. Returns `Err` for unsupported or missing files;
    /// per-packet failures are deferred to `Iterator::next`.
    fn open(path: &Path, window: WindowConfig) -> Result<Self, LensError> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        let file = File::open(path)?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let mut hint = Hint::new();
        if !ext.is_empty() {
            hint.with_extension(&ext);
        }

        let probed = symphonia::default::get_probe()
            .format(
                &hint,
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(|e| unsupported_or_decode_err(&ext, path, e))?;

        let format = probed.format;

        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| {
                LensError::UnsupportedMediaCodec(format!(
                    "{}: no decodable audio track (ext: {ext})",
                    path.display()
                ))
            })?;
        let track_id = track.id;

        let decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .map_err(|e| unsupported_or_decode_err(&ext, path, e))?;

        Ok(WindowIter {
            format,
            decoder,
            track_id,
            resampler: None, // built lazily on first decoded frame
            source_rate: 0,
            channels: 0,
            src_pending: Vec::new(),
            out_pending: Vec::new(),
            window_frames: window.window_frames.max(1),
            path: path.to_owned(),
            eof: false,
            done: false,
        })
    }

    /// Initialises `source_rate`, `channels`, and `resampler` from the first
    /// decoded frame's spec (no-op after the first call), or returns an error
    /// if a subsequent frame carries a different rate or channel count.
    ///
    /// The mid-stream spec-change check is delegated to [`verify_spec_unchanged`]
    /// so that logic can be unit-tested independently.
    fn check_and_init_spec(&mut self, rate: u32, ch: usize) -> Result<(), LensError> {
        if self.source_rate != 0 {
            return verify_spec_unchanged(self.source_rate, self.channels, rate, ch, &self.path);
        }

        self.source_rate = rate;
        self.channels = ch;

        if rate != TARGET_SAMPLE_RATE {
            let ratio = TARGET_SAMPLE_RATE as f64 / rate as f64;
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                oversampling_factor: 128,
                interpolation: SincInterpolationType::Cubic,
                window: WindowFunction::BlackmanHarris2,
            };
            self.resampler = Some(
                SincFixedIn::<f32>::new(ratio, 1.0, params, RESAMPLE_CHUNK_FRAMES, 1).map_err(
                    |e| {
                        LensError::MediaDecodeFailed(format!(
                            "{}: resampler init failed: {e}",
                            self.path.display()
                        ))
                    },
                )?,
            );
        }
        Ok(())
    }

    /// Decodes the next packet and downmixes its samples into `src_pending`.
    /// Returns `Ok(true)` = data appended, `Ok(false)` = clean EOF,
    /// `Err` = hard error (corrupt, unsupported mid-stream, spec-change).
    fn decode_next_packet(&mut self) -> Result<bool, LensError> {
        loop {
            let packet = match self.format.next_packet() {
                Ok(p) => p,
                Err(SymphoniaError::IoError(io))
                    if io.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(false); // clean EOF
                }
                Err(SymphoniaError::ResetRequired) => return Ok(false),
                Err(SymphoniaError::IoError(io)) => {
                    return Err(LensError::MediaDecodeFailed(format!(
                        "{}: read error: {io}",
                        self.path.display()
                    )));
                }
                Err(e) => {
                    return Err(LensError::MediaDecodeFailed(format!(
                        "{}: {}",
                        self.path.display(),
                        e
                    )));
                }
            };

            if packet.track_id() != self.track_id {
                continue;
            }

            let decoded = match self.decoder.decode(&packet) {
                Ok(d) => d,
                // Per Symphonia's contract: tolerate individual packet decode
                // errors (skip the packet). A broken stream eventually produces
                // IoError/EOF or too many errors to yield any output.
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(SymphoniaError::IoError(io))
                    if io.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(false);
                }
                Err(e) => {
                    return Err(LensError::MediaDecodeFailed(format!(
                        "{}: decode error: {e}",
                        self.path.display()
                    )));
                }
            };

            // Read spec scalars before `push_decoded_to_mono` consumes `decoded`
            // (it borrows `self.decoder`), so the later `&mut self` call is legal.
            let pkt_rate = decoded.spec().rate;
            let pkt_ch = decoded.spec().channels.count();
            push_decoded_to_mono(decoded, pkt_ch, &mut self.src_pending);

            self.check_and_init_spec(pkt_rate, pkt_ch)?;

            return Ok(true);
        }
    }

    /// Feeds as many complete `RESAMPLE_CHUNK_FRAMES` blocks from `src_pending`
    /// as possible through the resampler (or identity copy for 16 kHz sources),
    /// appending clamped output to `out_pending`. Leaves leftover source frames
    /// in `src_pending` for the next call.
    fn pump_resampler(&mut self) -> Result<(), LensError> {
        while self.src_pending.len() >= RESAMPLE_CHUNK_FRAMES {
            let chunk: Vec<f32> = self.src_pending.drain(..RESAMPLE_CHUNK_FRAMES).collect();
            let out = self.resample_chunk(chunk, false)?;
            self.out_pending.extend(out);
        }
        Ok(())
    }

    /// Flushes the remaining `src_pending` (short final chunk) and the
    /// resampler's internal delay line, appending clamped output to
    /// `out_pending`.
    fn flush_resampler(&mut self) -> Result<(), LensError> {
        if !self.src_pending.is_empty() {
            let tail = std::mem::take(&mut self.src_pending);
            let out = self.resample_chunk(tail, true)?;
            self.out_pending.extend(out);
        }
        if let Some(ref mut rs) = self.resampler {
            let flushed = rs
                .process_partial(None::<&[Vec<f32>]>, None)
                .map_err(|e| resample_err(&self.path, e))?;
            self.out_pending
                .extend(flushed[MONO].iter().map(|&s| s.clamp(-1.0, 1.0)));
        }
        Ok(())
    }

    /// Passes `chunk` through the resampler (or copies it for 16 kHz sources)
    /// and returns clamped output. `partial` = true uses `process_partial` for
    /// a short final chunk.
    fn resample_chunk(&mut self, chunk: Vec<f32>, partial: bool) -> Result<Vec<f32>, LensError> {
        match self.resampler {
            None => Ok(chunk.into_iter().map(|s| s.clamp(-1.0, 1.0)).collect()),
            Some(ref mut rs) => {
                let resampled = if partial {
                    rs.process_partial(Some(&[chunk]), None)
                } else {
                    rs.process(&[chunk], None)
                }
                .map_err(|e| resample_err(&self.path, e))?;
                Ok(resampled[MONO]
                    .iter()
                    .map(|&s| s.clamp(-1.0, 1.0))
                    .collect())
            }
        }
    }
}

impl Iterator for WindowIter {
    type Item = Result<Vec<f32>, LensError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        loop {
            if self.out_pending.len() >= self.window_frames {
                let window: Vec<f32> = self.out_pending.drain(..self.window_frames).collect();
                return Some(Ok(window));
            }

            if self.eof {
                if let Err(e) = self.flush_resampler() {
                    self.done = true;
                    return Some(Err(e));
                }
                self.done = true;
                if self.out_pending.is_empty() {
                    return None;
                }
                let last = std::mem::take(&mut self.out_pending);
                return Some(Ok(last));
            }

            match self.decode_next_packet() {
                Ok(false) => {
                    self.eof = true;
                }
                Ok(true) => {
                    if let Err(e) = self.pump_resampler() {
                        self.done = true;
                        return Some(Err(e));
                    }
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            }
        }
    }
}

/// Maps a Symphonia probe/decoder-construction error to the right
/// `LensError`. `Unsupported` → `UnsupportedMediaCodec`; everything else →
/// `MediaDecodeFailed`. Both messages include `path.display()`.
fn unsupported_or_decode_err(ext: &str, path: &Path, err: SymphoniaError) -> LensError {
    match err {
        SymphoniaError::Unsupported(msg) => {
            LensError::UnsupportedMediaCodec(format!("{} (ext: {ext}): {msg}", path.display()))
        }
        other => LensError::MediaDecodeFailed(format!("{}: {other}", path.display())),
    }
}

fn resample_err(path: &Path, err: rubato::ResampleError) -> LensError {
    LensError::MediaDecodeFailed(format!("{}: resample failed: {err}", path.display()))
}

/// Copies a decoded Symphonia frame into interleaved f32 and downmixes it to
/// mono by averaging channels (D7), appending the mono samples to `out`.
///
/// Allocates a **fresh** `SampleBuffer` sized from `decoded.capacity()` on
/// every call. This is the fix for C1 (the SampleBuffer-reuse panic): Symphonia
/// asserts `capacity() >= frames×channels` when `copy_interleaved_ref` is
/// called on a buffer sized for a SMALLER frame. Variable-blocksize FLAC and
/// lossy-codec priming frames produce exactly this growing-capacity sequence.
/// Allocating per-packet is cheap relative to the decode itself.
///
/// Extracted as a free function so a unit test can drive it with synthetic
/// `AudioBuffer` frames (capacities [256, 1024, 4096]) and assert no panic.
pub(crate) fn push_decoded_to_mono(
    decoded: AudioBufferRef<'_>,
    latched_channels: usize,
    out: &mut Vec<f32>,
) {
    let spec = *decoded.spec();
    let capacity = decoded.capacity() as u64;
    let mut sample_buf = SampleBuffer::<f32>::new(capacity, spec);
    sample_buf.copy_interleaved_ref(decoded);
    let samples = sample_buf.samples();
    let ch = latched_channels.max(1);
    if ch <= 1 {
        out.extend_from_slice(samples);
    } else {
        for frame in samples.chunks_exact(ch) {
            let sum: f32 = frame.iter().sum();
            out.push(sum / ch as f32);
        }
    }
}

/// Checks whether a mid-stream packet's spec matches the latched (first-frame)
/// spec. Returns `Ok(())` when they match; `Err(MediaDecodeFailed)` when they
/// diverge (a rate or channel-count change would silently garble the resampled
/// output).
///
/// Pure function — extracted so it can be unit-tested without a real stream.
pub(crate) fn verify_spec_unchanged(
    latched_rate: u32,
    latched_ch: usize,
    pkt_rate: u32,
    pkt_ch: usize,
    path: &Path,
) -> Result<(), LensError> {
    if latched_rate != pkt_rate || latched_ch != pkt_ch {
        return Err(LensError::MediaDecodeFailed(format!(
            "{}: mid-stream audio spec change \
             (rate {latched_rate}->{pkt_rate}, channels {latched_ch}->{pkt_ch}); \
             single-spec files only",
            path.display(),
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/audio")
            .join(name)
    }

    /// Goertzel detector: returns the (unnormalized) power at `target_hz`.
    /// Avoids an FFT dependency while being sufficient to confirm a pure tone.
    fn goertzel_power(signal: &[f32], sample_rate: f32, target_hz: f32) -> f32 {
        let n = signal.len();
        if n == 0 {
            return 0.0;
        }
        let k = (0.5 + (n as f32 * target_hz) / sample_rate).floor();
        let omega = (2.0 * std::f32::consts::PI * k) / n as f32;
        let coeff = 2.0 * omega.cos();
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for &x in signal {
            let s = x + coeff * s1 - s2;
            s2 = s1;
            s1 = s;
        }
        s2 * s2 + s1 * s1 - coeff * s1 * s2
    }

    fn dominant_frequency(signal: &[f32], sample_rate: f32) -> f32 {
        let mut best_hz = 0.0f32;
        let mut best_power = f32::MIN;
        let mut hz = 50.0f32;
        while hz <= 2000.0 {
            let p = goertzel_power(signal, sample_rate, hz);
            if p > best_power {
                best_power = p;
                best_hz = hz;
            }
            hz += 5.0;
        }
        best_hz
    }

    /// Structural assertion for a 1 s 440 Hz tone resampled to 16 kHz.
    /// The window is [15500, 17500] to absorb:
    ///   (a) rubato sinc filter tail (hundreds of extra output frames after
    ///       the signal ends) — common for all lossless sources.
    ///   (b) Lossy-codec encoder priming/padding (MP3/AAC decode slightly
    ///       more than the source duration) — adds up to ~1042 frames here.
    fn assert_valid_16k_mono(pcm: &[f32]) {
        assert!(!pcm.is_empty(), "output must be non-empty");
        assert!(
            (15_500..=17_500).contains(&pcm.len()),
            "expected ~16000 samples [15500,17500], got {}",
            pcm.len()
        );
        for &s in pcm {
            assert!(
                (-1.0..=1.0).contains(&s),
                "sample {s} out of [-1.0, 1.0] range — clamp missing?"
            );
        }
    }

    #[test]
    fn decodes_wav_44100_stereo() {
        assert_valid_16k_mono(
            &decode_and_resample_audio(&fixture("tone_44100_stereo.wav")).unwrap(),
        );
    }

    #[test]
    fn decodes_wav_48000_mono() {
        assert_valid_16k_mono(&decode_and_resample_audio(&fixture("tone_48000_mono.wav")).unwrap());
    }

    #[test]
    fn decodes_wav_16000_mono_near_identity() {
        assert_valid_16k_mono(&decode_and_resample_audio(&fixture("tone_16000_mono.wav")).unwrap());
    }

    #[test]
    fn decodes_mp3() {
        assert_valid_16k_mono(&decode_and_resample_audio(&fixture("tone.mp3")).unwrap());
    }

    #[test]
    fn decodes_aac_m4a() {
        assert_valid_16k_mono(&decode_and_resample_audio(&fixture("tone_aac.m4a")).unwrap());
    }

    #[test]
    fn decodes_alac_m4a() {
        assert_valid_16k_mono(&decode_and_resample_audio(&fixture("tone_alac.m4a")).unwrap());
    }

    #[test]
    fn decodes_adts_aac() {
        assert_valid_16k_mono(&decode_and_resample_audio(&fixture("tone.aac")).unwrap());
    }

    #[test]
    fn decodes_flac() {
        assert_valid_16k_mono(&decode_and_resample_audio(&fixture("tone.flac")).unwrap());
    }

    #[test]
    fn decodes_aiff() {
        assert_valid_16k_mono(&decode_and_resample_audio(&fixture("tone.aiff")).unwrap());
    }

    #[test]
    fn decodes_caf() {
        assert_valid_16k_mono(&decode_and_resample_audio(&fixture("tone.caf")).unwrap());
    }

    #[test]
    fn preserves_440hz_tone_wav() {
        let pcm = decode_and_resample_audio(&fixture("tone_44100_stereo.wav")).unwrap();
        let hz = dominant_frequency(&pcm, TARGET_SAMPLE_RATE as f32);
        assert!(
            (hz - 440.0).abs() <= 8.0,
            "dominant freq {hz} Hz, want ~440"
        );
    }

    #[test]
    fn preserves_440hz_tone_mp3() {
        let pcm = decode_and_resample_audio(&fixture("tone.mp3")).unwrap();
        let hz = dominant_frequency(&pcm, TARGET_SAMPLE_RATE as f32);
        assert!(
            (hz - 440.0).abs() <= 8.0,
            "dominant freq {hz} Hz, want ~440"
        );
    }

    #[test]
    fn preserves_440hz_tone_flac() {
        let pcm = decode_and_resample_audio(&fixture("tone.flac")).unwrap();
        let hz = dominant_frequency(&pcm, TARGET_SAMPLE_RATE as f32);
        assert!(
            (hz - 440.0).abs() <= 8.0,
            "dominant freq {hz} Hz, want ~440"
        );
    }

    #[test]
    fn preserves_440hz_tone_m4a() {
        let pcm = decode_and_resample_audio(&fixture("tone_aac.m4a")).unwrap();
        let hz = dominant_frequency(&pcm, TARGET_SAMPLE_RATE as f32);
        assert!(
            (hz - 440.0).abs() <= 8.0,
            "dominant freq {hz} Hz, want ~440"
        );
    }

    #[test]
    fn downmixes_stereo_to_mono() {
        let pcm = decode_and_resample_audio(&fixture("tone_44100_stereo.wav")).unwrap();
        assert_valid_16k_mono(&pcm);
        let hz = dominant_frequency(&pcm, TARGET_SAMPLE_RATE as f32);
        assert!(
            (hz - 440.0).abs() <= 8.0,
            "downmixed freq {hz} Hz, want ~440"
        );
    }

    #[test]
    fn windows_flatten_equals_whole_buffer() {
        let path = fixture("tone_44100_stereo.wav");
        let whole = decode_and_resample_audio(&path).unwrap();
        let windowed: Vec<f32> = decode_resample_windows(&path, WindowConfig::default())
            .unwrap()
            .flat_map(|w| w.unwrap())
            .collect();
        assert_eq!(whole.len(), windowed.len(), "lengths must match");
        for (i, (a, b)) in whole.iter().zip(windowed.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "sample {i} differs: whole={a}, windowed={b}"
            );
        }
    }

    #[test]
    fn small_window_yields_multiple_windows() {
        let path = fixture("tone_48000_mono.wav");
        let whole = decode_and_resample_audio(&path).unwrap();
        let cfg = WindowConfig {
            window_frames: 4000,
        };
        let windows: Vec<Vec<f32>> = decode_resample_windows(&path, cfg)
            .unwrap()
            .map(|w| w.unwrap())
            .collect();
        assert!(
            windows.len() >= 3,
            "expected multiple windows, got {}",
            windows.len()
        );
        let flat: Vec<f32> = windows.into_iter().flatten().collect();
        assert_eq!(flat.len(), whole.len());
    }

    /// Regression: a 35 s stereo source crosses the default 30 s window
    /// boundary, so the streaming iterator emits ≥ 2 windows on the REAL path
    /// (not just from slicing a pre-built buffer). Asserts that:
    ///   1. At least 2 windows are yielded (file is genuinely longer than 1 window).
    ///   2. Concatenated windows == whole-buffer output within float tolerance.
    #[test]
    fn streaming_multi_window_crosses_30s_boundary() {
        let path = fixture("tone_44100_stereo_35s.wav");
        let whole = decode_and_resample_audio(&path).unwrap();

        let windows: Vec<Vec<f32>> = decode_resample_windows(&path, WindowConfig::default())
            .unwrap()
            .map(|w| w.unwrap())
            .collect();

        assert!(
            windows.len() >= 2,
            "expected ≥2 windows for a 35s file, got {}",
            windows.len()
        );
        assert_eq!(
            windows[0].len(),
            WindowConfig::DEFAULT_WINDOW_FRAMES,
            "first window must be full-size"
        );

        let flat: Vec<f32> = windows.into_iter().flatten().collect();
        assert_eq!(
            flat.len(),
            whole.len(),
            "streamed windows must concatenate to the whole-buffer output"
        );
        for (i, (a, b)) in whole.iter().zip(flat.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "sample {i} differs across streaming vs whole-buffer paths"
            );
        }
    }

    /// Regression: the sinc resampler (Cubic/BlackmanHarris2) exhibits Gibbs
    /// ringing that pushes output past ±1.0 on near-full-scale input. The
    /// clamp ensures the contract is met.
    #[test]
    fn nearclip_output_within_range() {
        let pcm = decode_and_resample_audio(&fixture("tone_nearclip_44100_mono.wav")).unwrap();
        assert!(!pcm.is_empty());
        for &s in &pcm {
            assert!(
                (-1.0..=1.0).contains(&s),
                "sample {s} exceeds [-1,1] — clamp not applied"
            );
        }
        let hz = dominant_frequency(&pcm, TARGET_SAMPLE_RATE as f32);
        assert!(
            (hz - 440.0).abs() <= 8.0,
            "dominant freq {hz} Hz, want ~440"
        );
    }

    #[test]
    fn opus_is_unsupported_codec() {
        let err = decode_and_resample_audio(&fixture("tone.opus")).unwrap_err();
        assert!(
            matches!(err, LensError::UnsupportedMediaCodec(_)),
            "expected UnsupportedMediaCodec, got {err:?}"
        );
    }

    #[test]
    fn corrupt_mp3_is_decode_failed() {
        let err = decode_and_resample_audio(&fixture("corrupt.mp3")).unwrap_err();
        assert!(
            matches!(
                err,
                LensError::MediaDecodeFailed(_) | LensError::UnsupportedMediaCodec(_)
            ),
            "expected MediaDecodeFailed/UnsupportedMediaCodec, got {err:?}"
        );
    }

    #[test]
    fn unsupported_extension_is_unsupported_codec() {
        let err = decode_and_resample_audio(&fixture("unsupported.xyz")).unwrap_err();
        assert!(
            matches!(
                err,
                LensError::UnsupportedMediaCodec(_) | LensError::MediaDecodeFailed(_)
            ),
            "expected UnsupportedMediaCodec/MediaDecodeFailed, got {err:?}"
        );
    }

    #[test]
    fn all_silent_is_empty_audio() {
        let err = decode_and_resample_audio(&fixture("silence_16000_mono.wav")).unwrap_err();
        assert!(
            matches!(err, LensError::EmptyAudio(_)),
            "expected EmptyAudio, got {err:?}"
        );
    }

    #[test]
    fn missing_file_is_io_error() {
        let err = decode_and_resample_audio(&fixture("does_not_exist.wav")).unwrap_err();
        assert!(matches!(err, LensError::Io(_)), "expected Io, got {err:?}");
    }

    /// Regression for C1: the OLD code sized a single `SampleBuffer` from the
    /// FIRST decoded frame's `capacity()` and reused it for every subsequent
    /// packet. Symphonia asserts `capacity() >= frames*channels` on
    /// `copy_interleaved_ref`, so a later packet with LARGER capacity panicked.
    ///
    /// This test constructs a sequence of synthetic `AudioBuffer<f32>` frames
    /// with INCREASING capacities — [256, 1024, 4096] — exactly the shape that
    /// would trip the old code — and feeds each through `push_decoded_to_mono`.
    /// Under the old first-frame-sizing code the second call panics; under the
    /// fixed per-packet allocation all three complete and the total mono sample
    /// count equals the sum of frame lengths.
    #[test]
    fn growing_frame_capacity_no_panic() {
        use symphonia::core::audio::{AsAudioBufferRef, AudioBuffer, Layout, Signal, SignalSpec};

        let spec = SignalSpec::new_with_layout(44100, Layout::Mono);
        let frame_sizes = [256usize, 1024, 4096];
        let mut total_mono = Vec::new();

        for &frames in &frame_sizes {
            let mut buf: AudioBuffer<f32> = AudioBuffer::new(frames as u64, spec);
            buf.render_reserved(Some(frames));
            for s in buf.chan_mut(0) {
                *s = 0.5_f32;
            }
            push_decoded_to_mono(buf.as_audio_buffer_ref(), 1, &mut total_mono);
        }

        let expected: usize = frame_sizes.iter().sum();
        assert_eq!(
            total_mono.len(),
            expected,
            "total mono samples must equal sum of frame lengths ({expected})"
        );
        for &s in &total_mono {
            assert!(
                (s - 0.5).abs() < 1e-6,
                "expected 0.5, got {s} — downmix logic wrong"
            );
        }
    }

    #[test]
    fn verify_spec_unchanged_same_spec_is_ok() {
        let path = std::path::Path::new("dummy.wav");
        assert!(
            verify_spec_unchanged(44100, 2, 44100, 2, path).is_ok(),
            "identical spec must be Ok"
        );
    }

    #[test]
    fn verify_spec_unchanged_rate_change_is_err() {
        let path = std::path::Path::new("dummy.wav");
        let err = verify_spec_unchanged(44100, 1, 48000, 1, path).unwrap_err();
        assert!(
            matches!(err, LensError::MediaDecodeFailed(_)),
            "rate change must be MediaDecodeFailed, got {err:?}"
        );
        let msg = err.message();
        assert!(msg.contains("44100"), "message missing old rate: {msg}");
        assert!(msg.contains("48000"), "message missing new rate: {msg}");
    }

    #[test]
    fn verify_spec_unchanged_channel_change_is_err() {
        let path = std::path::Path::new("dummy.wav");
        let err = verify_spec_unchanged(44100, 1, 44100, 2, path).unwrap_err();
        assert!(
            matches!(err, LensError::MediaDecodeFailed(_)),
            "channel change must be MediaDecodeFailed, got {err:?}"
        );
    }

    /// Near-identity streaming path (source = 16 kHz → resampler = None).
    /// Uses a small window to force multiple `next()` calls so the resampler=None
    /// branch in `resample_chunk` is exercised by the iterator, not just the
    /// whole-buffer fn.
    #[test]
    fn near_identity_streaming_multiple_windows() {
        let path = fixture("tone_16000_mono.wav");
        let whole = decode_and_resample_audio(&path).unwrap();
        // 4000-frame windows forces ≥ 3 iterations on a ~16000-sample output.
        let cfg = WindowConfig {
            window_frames: 4000,
        };
        let windows: Vec<Vec<f32>> = decode_resample_windows(&path, cfg)
            .unwrap()
            .map(|w| w.unwrap())
            .collect();
        assert!(
            windows.len() >= 3,
            "expected ≥3 windows for near-identity path, got {}",
            windows.len()
        );
        let flat: Vec<f32> = windows.into_iter().flatten().collect();
        assert_eq!(
            flat.len(),
            whole.len(),
            "streamed near-identity must equal whole-buffer"
        );
    }

    /// Upsampling: 8 kHz → 16 kHz (ratio 2×). Asserts structural correctness,
    /// signal fidelity (440 Hz), and that all samples are clamped to [-1, 1].
    #[test]
    fn decodes_8000_mono_upsampled_to_16k() {
        let pcm = decode_and_resample_audio(&fixture("tone_8000_mono.wav")).unwrap();

        // 1 s at 8 kHz → 8000 input frames → ~16000 output frames after 2× upsample.
        // At a 2× ratio the sinc resampler (sinc_len=256) emits a proportionally
        // larger filter tail than for downsampling from 44.1/48kHz; up to ~2200
        // extra output frames (~137 ms) are acceptable. Lower bound stays tight
        // to catch a failed upsample (would give ~8000).
        assert!(
            (15_500..=19_000).contains(&pcm.len()),
            "expected ~16000 samples for 8kHz→16kHz upsample [15500,19000], got {}",
            pcm.len()
        );
        for &s in &pcm {
            assert!(
                (-1.0..=1.0).contains(&s),
                "upsampled sample {s} out of [-1, 1]"
            );
        }
        let hz = dominant_frequency(&pcm, TARGET_SAMPLE_RATE as f32);
        assert!(
            (hz - 440.0).abs() <= 8.0,
            "dominant freq {hz} Hz after 8kHz→16kHz upsample, want ~440"
        );
    }

    /// Cross-validates our rubato resample output against an ffmpeg swr reference.
    ///
    /// Different resamplers are never bit-identical, so we use **Pearson correlation**
    /// on the aligned prefix (truncated to min length, as rubato's sinc tail makes
    /// ours slightly longer). Empirically measured: correlation = 0.9939 on
    /// `tone_44100_stereo.wav` (rubato SincFixedIn vs ffmpeg swr). Threshold ≥ 0.98
    /// gives a 1.4% margin while catching gross errors (wrong ratio, gain, channel
    /// swap, garbage output). The reference file was generated with:
    ///   `ffmpeg -y -v error -i tone_44100_stereo.wav -ar 16000 -ac 1 -f f32le tone_44100_stereo.ref16k.f32le`
    #[test]
    fn cross_validates_against_ffmpeg_reference() {
        use std::io::Read as _;

        let ref_path = fixture("tone_44100_stereo.ref16k.f32le");
        let mut f = std::fs::File::open(&ref_path)
            .expect("reference fixture missing — regenerate with generate.sh");
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).unwrap();
        let ref_pcm: Vec<f32> = buf
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        let our_pcm = decode_and_resample_audio(&fixture("tone_44100_stereo.wav")).unwrap();

        let n = ref_pcm.len().min(our_pcm.len());
        assert!(n >= 15_000, "too few aligned samples: {n}");

        let (r, o) = (&ref_pcm[..n], &our_pcm[..n]);
        let mr: f32 = r.iter().sum::<f32>() / n as f32;
        let mo: f32 = o.iter().sum::<f32>() / n as f32;
        let num: f32 = r
            .iter()
            .zip(o.iter())
            .map(|(a, b)| (a - mr) * (b - mo))
            .sum();
        let dr: f32 = r.iter().map(|a| (a - mr).powi(2)).sum::<f32>().sqrt();
        let do_: f32 = o.iter().map(|b| (b - mo).powi(2)).sum::<f32>().sqrt();
        let corr = num / (dr * do_);

        assert!(
            corr >= 0.98,
            "Pearson correlation {corr:.4} < 0.98 (measured 0.9939 on rubato vs ffmpeg swr); \
             possible wrong ratio, gain, channel swap, or garbage output"
        );
    }

    /// Validates the sinc anti-aliasing filter rejects a 12 kHz tone (above the
    /// 8 kHz Nyquist for a 16 kHz output rate). If anti-aliasing were broken, the
    /// 12 kHz component would alias to |16000 − 12000| = 4000 Hz. We measure the
    /// Goertzel power at 4 kHz in the 12 kHz output and compare it to the 4 kHz
    /// Goertzel power of the 440 Hz reference signal (noise floor). Empirically:
    /// alias@4kHz = 3.94e−4, reference inband@440Hz = 499754 → ratio ≈ 1.27 billion.
    /// Threshold ≥ 1_000_000× gives enormous margin; a broken anti-alias filter
    /// would collapse the ratio to near 1× (the alias would be as loud as a real tone).
    #[test]
    fn sinc_filter_rejects_above_nyquist_alias() {
        let pcm12k = decode_and_resample_audio(&fixture("tone_12k_44100_mono.wav")).unwrap();
        let pcm440 = decode_and_resample_audio(&fixture("tone_44100_stereo.wav")).unwrap();

        let alias_power = goertzel_power(&pcm12k, TARGET_SAMPLE_RATE as f32, 4000.0);
        let n440 = pcm440.len().min(16_000);
        let inband_power = goertzel_power(&pcm440[..n440], TARGET_SAMPLE_RATE as f32, 440.0);

        let rejection_ratio = inband_power / alias_power.max(1e-9);
        assert!(
            rejection_ratio >= 1_000_000.0,
            "alias rejection ratio {rejection_ratio:.0}x < 1_000_000× \
             (measured 1.27 billion on rubato SincFixedIn BlackmanHarris2); \
             anti-aliasing filter may be broken — 12 kHz leaking as alias @4 kHz"
        );
    }
}
