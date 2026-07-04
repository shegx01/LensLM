//! Apple-native ASR backend (issue #42): [`AppleSpeechEngine`] drives macOS 26's
//! SpeechAnalyzer/SpeechTranscriber through the Swift `@_cdecl` C-ABI bridge
//! (`bridge.swift`, declared in `bridge.h`). This whole module is gated to
//! `aarch64-apple-darwin` + the `apple-native-asr` feature; it is the ONLY place
//! Apple/OS speech code lives (lens-core stays headless).
//!
//! FFI ownership contract (see `bridge.h` for the authoritative statement):
//! (a) result buffers are Swift-allocated, freed ONLY via `lens_asr_free` — the
//!     [`ResultGuard`] RAII wrapper enforces this even on the error/panic path;
//!     the borrowed `pcm`/`lang_code` inputs are Rust-owned, never freed by Swift.
//! (b) transcript text crosses as UTF-8 bytes + length, decoded via CHECKED
//!     [`String::from_utf8`] — never `_unchecked`, never assumed NUL-terminated.
//! (c) the Swift body never traps; failures return null + an owned error message.

use std::ffi::{CString, c_char};
use std::os::raw::c_float;

use async_trait::async_trait;

use lens_core::{AsrEngine, Lang, LensError, TranscribeConfig, TranscriptSegment};

// ------------------------------- C ABI (mirror of bridge.h) -------------------------------

/// One transcribed span. Layout MUST match `LensAsrSegment` in `bridge.h`.
#[repr(C)]
struct CSegment {
    text_ptr: *const u8,
    text_len: usize,
    start_second: f64,
    end_second: f64,
}

/// Owned result buffer. Layout MUST match `LensAsrResult` in `bridge.h`.
#[repr(C)]
struct CResult {
    segments: *const CSegment,
    segment_count: usize,
}

/// Owned error message (UTF-8 bytes + len). Layout MUST match `LensAsrError`.
#[repr(C)]
struct CError {
    message_ptr: *const u8,
    message_len: usize,
}

unsafe extern "C" {
    /// See `bridge.h` — returns an owned `*mut CResult` or null (then `*out_error`
    /// is set to an owned `*mut CError`).
    fn lens_asr_transcribe(
        pcm: *const c_float,
        pcm_len: usize,
        sample_rate: i32,
        lang_code: *const c_char,
        translate: i32,
        out_error: *mut *mut CError,
    ) -> *mut CResult;

    /// Frees a `CResult` (and everything it owns). No-op on null. Clause (a).
    fn lens_asr_free(result: *mut CResult);

    /// Frees a `CError`. No-op on null. Clause (a).
    fn lens_asr_free_error(error: *mut CError);

    /// Non-zero if the Apple transcriber supports `lang_code`. Never traps.
    fn lens_asr_supports_locale(lang_code: *const c_char) -> i32;
}

// ------------------------------- RAII guards (clause a) -------------------------------

/// Owns a Swift-allocated `*mut CResult` and frees it via `lens_asr_free` on drop
/// (clause a) — so the buffer is released even if decoding panics or returns early.
struct ResultGuard(*mut CResult);

impl Drop for ResultGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: `self.0` is a non-null pointer returned by
            // `lens_asr_transcribe` and freed exactly once here (this guard owns
            // it and is not Copy/Clone); `lens_asr_free` is its paired allocator.
            unsafe { lens_asr_free(self.0) };
        }
    }
}

/// Owns a Swift-allocated `*mut CError` and frees it via `lens_asr_free_error` on
/// drop (clause a), after its message bytes have been copied into an owned String.
struct ErrorGuard(*mut CError);

impl Drop for ErrorGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: `self.0` is a non-null pointer written by
            // `lens_asr_transcribe` to `out_error` and freed exactly once here;
            // `lens_asr_free_error` is its paired allocator.
            unsafe { lens_asr_free_error(self.0) };
        }
    }
}

// ------------------------------- Engine -------------------------------

/// The Apple on-device speech engine. Zero-sized: it holds no state — each call
/// drives a fresh SpeechAnalyzer session in the Swift bridge. Constructing it does
/// NOT probe availability (src-tauri gates platform/version before injection); use
/// [`apple_supports_locale`] for the router's per-locale check.
pub struct AppleSpeechEngine;

impl AppleSpeechEngine {
    /// Builds the engine. Infallible: the platform/version gate lives in the caller
    /// (`main.rs` `.setup`), and per-locale support is queried separately.
    pub fn new() -> Self {
        AppleSpeechEngine
    }
}

impl Default for AppleSpeechEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Maps a [`Lang`] to the BCP-47 code the Apple bridge expects (`"en"`, `"de"`…).
/// The `Other` escape hatch passes its code through unchanged.
fn lang_to_bcp47(lang: &Lang) -> &str {
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

/// Runs the (blocking) C bridge call and decodes the result. Kept separate from
/// the async trait method so the whole `unsafe` FFI section runs inside
/// `spawn_blocking` with owned inputs. `pcm` is owned here so it outlives the call.
fn run_bridge(
    pcm: &[f32],
    lang_code: Option<&str>,
    translate: bool,
) -> Result<Vec<TranscriptSegment>, LensError> {
    // Borrowed, NUL-terminated lang code (clause a: Rust owns it; Swift borrows).
    // A NUL-containing code is invalid BCP-47 → typed error rather than a panic.
    let lang_c = match lang_code {
        Some(code) => Some(
            CString::new(code)
                .map_err(|_| LensError::Transcription("language code contains a NUL byte".into()))?,
        ),
        None => None,
    };
    let lang_ptr = lang_c
        .as_ref()
        .map(|c| c.as_ptr())
        .unwrap_or(std::ptr::null());

    let sample_rate: i32 = 16_000;
    let translate_flag: i32 = i32::from(translate);
    let mut err_ptr: *mut CError = std::ptr::null_mut();

    // SAFETY: `pcm` is a valid Rust slice living for this call; `lang_ptr` is
    // either null or a valid NUL-terminated buffer owned by `lang_c` (kept alive
    // by staying in scope). `out_error` points to our stack `err_ptr`. The bridge
    // borrows `pcm`/`lang_ptr` (never frees them — clause a) and returns either a
    // Swift-owned `*mut CResult` OR null with `*out_error` set (clause c: no trap).
    let result_ptr = unsafe {
        lens_asr_transcribe(
            pcm.as_ptr(),
            pcm.len(),
            sample_rate,
            lang_ptr,
            translate_flag,
            &mut err_ptr,
        )
    };

    if result_ptr.is_null() {
        // Failure path: take ownership of the error message (if any), copy it out
        // via CHECKED UTF-8 (clause b), and free it via its paired allocator.
        let guard = ErrorGuard(err_ptr);
        let message = if guard.0.is_null() {
            "apple transcription failed (no error detail returned)".to_string()
        } else {
            // SAFETY: `guard.0` is non-null (checked) and points to a Swift-owned
            // CError whose `message_ptr`/`message_len` describe a valid UTF-8 byte
            // run for `message_len` bytes (clause b). We only read it here, before
            // the guard's Drop frees it.
            let cerr = unsafe { &*guard.0 };
            read_utf8(cerr.message_ptr, cerr.message_len)
                .unwrap_or_else(|| "apple transcription failed (error message not UTF-8)".into())
        };
        return Err(LensError::Transcription(message));
    }

    // Success path: RAII-own the result so it is freed on every exit (clause a).
    let guard = ResultGuard(result_ptr);
    // SAFETY: `guard.0` is the non-null pointer just returned by the bridge and
    // owned by `guard`; `&*` reads the `CResult` header (segments ptr + count).
    let result = unsafe { &*guard.0 };

    let count = result.segment_count;
    let mut segments = Vec::with_capacity(count);
    if count > 0 {
        // SAFETY: on success the bridge guarantees `segments` points to `count`
        // contiguous, initialised `CSegment`s (clause a/b); the slice borrows them
        // for this scope only (their backing buffer is freed later by the guard).
        let items = unsafe { std::slice::from_raw_parts(result.segments, count) };
        for item in items {
            let text = read_utf8(item.text_ptr, item.text_len).ok_or_else(|| {
                LensError::Transcription("transcript segment text was not valid UTF-8".into())
            })?;
            segments.push(TranscriptSegment {
                text,
                start_second: item.start_second as f32,
                end_second: item.end_second as f32,
            });
        }
    }
    Ok(segments)
    // `guard` drops here → `lens_asr_free` releases the whole Swift buffer.
}

/// Copies `len` bytes at `ptr` into an owned `String` via CHECKED UTF-8 decoding
/// (clause b). Returns `None` on invalid UTF-8; `Some("")` for an empty run (a
/// null `ptr` is permitted only when `len == 0`).
fn read_utf8(ptr: *const u8, len: usize) -> Option<String> {
    if len == 0 {
        return Some(String::new());
    }
    if ptr.is_null() {
        return None;
    }
    // SAFETY: caller (the bridge, clause b) guarantees `ptr` addresses `len`
    // readable, initialised bytes; we copy them out immediately (`.to_vec()`)
    // before any Drop frees the source, so the borrow does not outlive the data.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    String::from_utf8(bytes.to_vec()).ok()
}

#[async_trait]
impl AsrEngine for AppleSpeechEngine {
    async fn transcribe_pcm(
        &self,
        pcm: &[f32],
        config: &TranscribeConfig,
        progress_tx: Option<tokio::sync::mpsc::UnboundedSender<f32>>,
    ) -> Result<Vec<TranscriptSegment>, LensError> {
        // Own the inputs so they outlive the blocking task. The Swift bridge runs
        // the SpeechAnalyzer pipeline to completion synchronously (it joins its own
        // async work on a semaphore), so the whole FFI call must run OFF the async
        // runtime — mirror the WhisperEngine's `spawn_blocking` shape.
        let pcm = pcm.to_vec();
        let lang_code = config.language.as_ref().map(|l| lang_to_bcp47(l).to_string());
        let translate = config.translate;

        // The bridge reports no incremental progress; emit a terminal 1.0 (the
        // trait's `0.0..=1.0` convention) so a listening onboarding bar completes.
        let out = tokio::task::spawn_blocking(move || {
            run_bridge(&pcm, lang_code.as_deref(), translate)
        })
        .await
        .map_err(|e| LensError::Transcription(format!("apple transcription task failed: {e}")))??;

        if let Some(tx) = progress_tx {
            let _ = tx.send(1.0);
        }
        Ok(out)
    }
}

/// Whether the Apple on-device transcriber supports `lang`'s locale. Feeds the
/// router's `apple_supports_locale` gate. Delegates to the Swift bridge, which
/// queries `SpeechTranscriber.supportedLocales`; a NUL-containing code or any
/// bridge error is treated as unsupported (returns `false`, never panics).
pub fn apple_supports_locale(lang: &Lang) -> bool {
    let code = lang_to_bcp47(lang);
    let Ok(c_code) = CString::new(code) else {
        return false;
    };
    // SAFETY: `c_code` is a valid NUL-terminated buffer, borrowed for this call
    // only; the bridge never frees it (clause a) and never traps (clause c).
    let supported = unsafe { lens_asr_supports_locale(c_code.as_ptr()) };
    supported != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lang_to_bcp47_maps_common_and_passthrough() {
        assert_eq!(lang_to_bcp47(&Lang::En), "en");
        assert_eq!(lang_to_bcp47(&Lang::Ja), "ja");
        assert_eq!(lang_to_bcp47(&Lang::Other("ar".into())), "ar");
    }

    #[test]
    fn read_utf8_handles_empty_and_null() {
        assert_eq!(read_utf8(std::ptr::null(), 0), Some(String::new()));
        assert_eq!(read_utf8(std::ptr::null(), 4), None);
        let bytes = b"hi\xC3\xA9"; // "hié" in UTF-8
        assert_eq!(
            read_utf8(bytes.as_ptr(), bytes.len()),
            Some("hié".to_string())
        );
    }

    #[test]
    fn read_utf8_rejects_invalid() {
        let bad = [0xFF_u8, 0xFE];
        assert_eq!(read_utf8(bad.as_ptr(), bad.len()), None);
    }

    #[test]
    fn engine_constructs() {
        let _engine = AppleSpeechEngine::new();
        let _default = AppleSpeechEngine;
    }

    // Gated on-device smoke test (Unit 6 gate): drives the real SpeechAnalyzer
    // bridge end-to-end. Requires a macOS-26 aarch64 host with an installed model.
    // Run: `LENS_RUN_MODEL_TESTS=1 cargo test -p lenslm --features apple-native-asr
    //       -- --ignored apple_speech`
    #[tokio::test]
    #[ignore = "requires macOS 26 + LENS_RUN_MODEL_TESTS + an on-device speech model"]
    async fn apple_speech_transcribes_fixture() {
        if std::env::var("LENS_RUN_MODEL_TESTS").is_err() {
            return;
        }
        // 1 s of silence at 16 kHz. A real fixture would carry speech; silence
        // exercises the full pipeline without asserting specific transcript text.
        let pcm = vec![0.0_f32; 16_000];
        let engine = AppleSpeechEngine::new();
        let out = engine
            .transcribe_pcm(&pcm, &TranscribeConfig::default(), None)
            .await
            .expect("apple transcription");
        for seg in &out {
            assert!(
                seg.start_second <= seg.end_second,
                "segment timestamps must be ordered"
            );
        }
    }
}
