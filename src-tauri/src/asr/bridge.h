/*
 * bridge.h — C ABI for the Apple-native ASR bridge (issue #42).
 *
 * This header is the SINGLE SOURCE OF TRUTH for the FFI surface: the Swift
 * `@_cdecl` definitions in bridge.swift and the Rust `extern "C"` block in
 * mod.rs must both match these declarations EXACTLY (struct layout + signatures).
 *
 * ============================ OWNERSHIP CONTRACT ============================
 * Three non-negotiable clauses (Pre-Mortem #2 of the #42 plan). Every allocation
 * is freed by the side that made it.
 *
 * (a) ALLOCATOR SYMMETRY.
 *     Every non-null pointer returned by `lens_asr_transcribe` (the LensAsrResult
 *     itself, its `segments` array, each segment's `text_ptr` bytes, and the
 *     `out_error` message) is SWIFT-ALLOCATED and must be freed ONLY by
 *     `lens_asr_free` (for the result) / `lens_asr_free_error` (for a returned
 *     error). Rust NEVER calls free()/dealloc on any of these pointers itself.
 *     Conversely the `pcm` and `lang_code` inputs are RUST-OWNED, BORROWED for the
 *     duration of the call only; Swift NEVER frees or retains them past return.
 *
 * (b) TEXT CROSSES AS UTF-8 BYTES + LENGTH.
 *     `LensAsrSegment.text_ptr` / `text_len` is a length-prefixed UTF-8 byte run,
 *     NOT a NUL-terminated C string (transcripts contain arbitrary Unicode, may
 *     embed NUL, and are not guaranteed NUL-terminated). Rust decodes via checked
 *     `String::from_utf8`. Likewise the error message is UTF-8 bytes + length.
 *
 * (c) NO TRAPS ACROSS THE BOUNDARY.
 *     The Swift body wraps every path in do/catch with zero force-unwraps and
 *     zero fatalError. On ANY failure it returns NULL from `lens_asr_transcribe`
 *     and writes an owned error message to `*out_error` (freed via
 *     `lens_asr_free_error`). A Swift trap across C is uncatchable UB.
 * ===========================================================================
 */

#ifndef LENS_ASR_BRIDGE_H
#define LENS_ASR_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One transcribed span. `text_ptr`/`text_len` is UTF-8 bytes + length (clause b);
 * timestamps are seconds. `text_ptr` may be null only if `text_len == 0`. */
typedef struct LensAsrSegment {
    const uint8_t *text_ptr;
    size_t text_len;
    double start_second;
    double end_second;
} LensAsrSegment;

/* An owned, Swift-allocated transcription result. Free ONLY via `lens_asr_free`
 * (clause a). `segments` points to `segment_count` contiguous LensAsrSegment. */
typedef struct LensAsrResult {
    const LensAsrSegment *segments;
    size_t segment_count;
} LensAsrResult;

/* An owned, Swift-allocated error message (UTF-8 bytes + length, clause b). Free
 * ONLY via `lens_asr_free_error` (clause a). Written to `*out_error` when
 * `lens_asr_transcribe` returns NULL. */
typedef struct LensAsrError {
    const uint8_t *message_ptr;
    size_t message_len;
} LensAsrError;

/*
 * Transcribe borrowed 16 kHz mono Float32 PCM.
 *
 *   pcm         borrowed, Rust-owned, `pcm_len` floats (NEVER freed by Swift).
 *   pcm_len     number of f32 samples.
 *   sample_rate PCM sample rate in Hz (expected 16000; #41 output).
 *   lang_code   borrowed NUL-terminated BCP-47 code (e.g. "en"), or NULL = auto.
 *   translate   non-zero → translate to English (best-effort; may be ignored if
 *               the Apple pipeline does not support translation for the locale).
 *   out_error   on NULL return, receives an owned LensAsrError* (clause a/b).
 *
 * Returns an owned LensAsrResult* on success (free via `lens_asr_free`), or NULL
 * on failure (then `*out_error` is set). Never traps (clause c).
 */
LensAsrResult *lens_asr_transcribe(const float *pcm,
                                   size_t pcm_len,
                                   int32_t sample_rate,
                                   const char *lang_code,
                                   int32_t translate,
                                   LensAsrError **out_error);

/* Frees a LensAsrResult* returned by `lens_asr_transcribe` and everything it owns
 * (segments array + each segment's text bytes). No-op on NULL. Clause (a). */
void lens_asr_free(LensAsrResult *result);

/* Frees a LensAsrError* written to `out_error`. No-op on NULL. Clause (a). */
void lens_asr_free_error(LensAsrError *error);

/* Returns non-zero if the Apple on-device transcriber supports `lang_code`
 * (borrowed NUL-terminated BCP-47). Best-effort; feeds the router's locale gate.
 * Never traps: on any internal error it returns 0 (treated as "unsupported"). */
int32_t lens_asr_supports_locale(const char *lang_code);

#ifdef __cplusplus
}
#endif

#endif /* LENS_ASR_BRIDGE_H */
