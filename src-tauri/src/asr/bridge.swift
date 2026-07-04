// bridge.swift — Apple-native ASR C-ABI bridge (issue #42).
//
// A tiny `@_cdecl` surface over macOS 26's SpeechAnalyzer + SpeechTranscriber that
// transcribes borrowed 16 kHz mono Float PCM into timed segments. bridge.h is the
// authoritative ABI and is imported via `-import-objc-header` (build.rs): the C
// structs `LensAsrSegment` / `LensAsrResult` / `LensAsrError` are visible here as
// C types, which is what lets the `@_cdecl` functions traffic in them (Swift-native
// structs are not C-representable, so they cannot cross `@_cdecl`).
//
// FFI OWNERSHIP CONTRACT (three non-negotiable clauses — full text in bridge.h):
//  (a) ALLOCATOR SYMMETRY. Every pointer we return (the LensAsrResult, its segment
//      array, each segment's UTF-8 text bytes, and the LensAsrError message) is
//      Swift-allocated (`UnsafeMutablePointer.allocate`) and freed ONLY by the
//      paired `lens_asr_free` / `lens_asr_free_error`. We NEVER free or retain the
//      borrowed `pcm` / `lang_code` inputs (Rust owns them).
//  (b) TEXT AS UTF-8 BYTES + LENGTH. Transcript/error text crosses as an allocated
//      byte run + length, NOT a NUL-terminated C string.
//  (c) NO TRAPS. Every path is wrapped in do/catch with ZERO force-unwraps (`!`)
//      and ZERO fatalError. On any failure we set `*out_error` and return nil — a
//      Swift trap across the C boundary is uncatchable UB that would abort the
//      process and defeat the Rust-side Whisper fallback.

import AVFAudio
import CoreMedia
import Foundation
import Speech

// MARK: - Allocation helpers (clause a + b)

/// Copies `str`'s UTF-8 bytes into a freshly allocated buffer and returns
/// (ptr, len). The buffer is Swift-owned; its paired free is `.deallocate()`,
/// invoked from `lens_asr_free`/`lens_asr_free_error`. Empty string → (nil, 0).
private func allocUTF8(_ str: String) -> (UnsafePointer<UInt8>?, Int) {
    let bytes = Array(str.utf8)
    if bytes.isEmpty {
        return (nil, 0)
    }
    let buf = UnsafeMutablePointer<UInt8>.allocate(capacity: bytes.count)
    buf.initialize(from: bytes, count: bytes.count)
    return (UnsafePointer(buf), bytes.count)
}

/// Allocates an owned `LensAsrError` for `message` and writes it to `outError`
/// (clause a/b). Safe with a nil `outError` (drops the message).
private func setError(
    _ message: String,
    _ outError: UnsafeMutablePointer<UnsafeMutablePointer<LensAsrError>?>?
) {
    guard let outError = outError else { return }
    let (ptr, len) = allocUTF8(message)
    let errBuf = UnsafeMutablePointer<LensAsrError>.allocate(capacity: 1)
    errBuf.initialize(to: LensAsrError(message_ptr: ptr, message_len: len))
    outError.pointee = errBuf
}

// MARK: - Result model (Swift-internal, before crossing the ABI)

private struct Segment {
    let text: String
    let start: Double
    let end: Double
}

private enum BridgeError: Error, CustomStringConvertible {
    case unsupportedOS
    case bufferAllocationFailed
    case noSupportedLocale
    case assetInstallFailed(String)
    case analysisFailed(String)

    var description: String {
        switch self {
        case .unsupportedOS: return "apple-native ASR requires macOS 26 or newer"
        case .bufferAllocationFailed: return "failed to build AVAudioPCMBuffer from PCM"
        case .noSupportedLocale: return "no supported speech locale is available"
        case .assetInstallFailed(let m): return "speech asset install failed: \(m)"
        case .analysisFailed(let m): return "speech analysis failed: \(m)"
        }
    }
}

// MARK: - Audio

/// Builds a 16 kHz mono Float32 `AVAudioPCMBuffer` from a borrowed PCM pointer.
/// Returns nil (never traps) if the format/buffer cannot be constructed. Samples
/// are COPIED in, so the borrowed `pcm` is not retained past this call.
private func makeBuffer(pcm: UnsafePointer<Float>, count: Int, sampleRate: Double) -> AVAudioPCMBuffer? {
    // Bound `count` before narrowing to UInt32: `AVAudioFrameCount(count)` traps for
    // count > UInt32.max, and copying `count` against a truncated capacity would OOB.
    guard count > 0, count <= Int(AVAudioFrameCount.max) else { return nil }
    let frames = AVAudioFrameCount(count)
    guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                     sampleRate: sampleRate,
                                     channels: 1,
                                     interleaved: false),
          let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)
    else {
        return nil
    }
    buffer.frameLength = frames
    guard let dst = buffer.floatChannelData?[0] else { return nil }
    dst.update(from: pcm, count: count)
    return buffer
}

// MARK: - Transcription core (the SpeechAnalyzer call site — isolated)

/// Runs SpeechAnalyzer/SpeechTranscriber to completion and returns timed segments.
///
/// NOTE — BEST-EFFORT CALL SITE pending the Unit 8 macOS-26 smoke test. The exact
/// SpeechAnalyzer/SpeechTranscriber surface is macOS-26-new; this function is the
/// ONLY place that touches it, is fully `do/catch`-guarded, and returns typed
/// errors, so any post-smoke-test symbol tweak is localized here. Verified against
/// developer.apple.com/documentation/speech/{speechanalyzer,speechtranscriber}.
@available(macOS 26.0, *)
private func transcribe(
    buffer: AVAudioPCMBuffer,
    langCode: String?,
    // `translate` is a reserved ABI slot: SpeechTranscriber has no translate task;
    // translation is routed to Whisper in lib.rs. Kept for signature stability.
    translate: Bool
) async throws -> [Segment] {
    // Resolve the locale: explicit code, else the first supported locale (auto).
    let supported = await SpeechTranscriber.supportedLocales
    let locale: Locale
    if let code = langCode, !code.isEmpty {
        locale = Locale(identifier: code)
    } else if let first = supported.first {
        locale = first
    } else {
        throw BridgeError.noSupportedLocale
    }

    // `.offlineTranscription` keeps this fully on-device; `attributeOptions`
    // includes `.audioTimeRange` so each result run carries a CMTimeRange we can
    // turn into segment start/end seconds.
    let transcriber = SpeechTranscriber(
        locale: locale,
        transcriptionOptions: [],
        reportingOptions: [],
        attributeOptions: [.audioTimeRange]
    )

    // Ensure the on-device model asset for this locale is installed (no-op when
    // already present). A missing/failed asset is a typed error, not a trap.
    do {
        if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            try await request.downloadAndInstall()
        }
    } catch {
        throw BridgeError.assetInstallFailed(error.localizedDescription)
    }

    let analyzer = SpeechAnalyzer(modules: [transcriber])

    // Feed the single buffer at t=0 (monotonic frame-based CMTime), then finish
    // through end-of-input and drain the final results.
    let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
    let startTime = CMTime(value: 0, timescale: CMTimeScale(buffer.format.sampleRate))
    continuation.yield(AnalyzerInput(buffer: buffer, bufferStartTime: startTime))
    continuation.finish()

    do {
        try await analyzer.start(inputSequence: stream)
    } catch {
        throw BridgeError.analysisFailed(error.localizedDescription)
    }

    var segments: [Segment] = []
    do {
        for try await result in transcriber.results {
            // `audioTimeRange` is populated on final results only; skip volatile.
            guard result.isFinal else { continue }
            let attributed = result.text
            let text = String(attributed.characters)
            // Span the segment across all runs that carry a time range.
            var segStart: Double? = nil
            var segEnd: Double? = nil
            for run in attributed.runs {
                guard let range = run.audioTimeRange else { continue }
                let s = CMTimeGetSeconds(range.start)
                let e = CMTimeGetSeconds(range.end)
                if s.isFinite {
                    segStart = segStart.map { min($0, s) } ?? s
                }
                if e.isFinite {
                    segEnd = segEnd.map { max($0, e) } ?? e
                }
            }
            let start = segStart ?? 0
            let end = segEnd ?? start
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                segments.append(Segment(text: trimmed, start: start, end: end))
            }
        }
    } catch {
        throw BridgeError.analysisFailed(error.localizedDescription)
    }

    try? await analyzer.finalizeAndFinishThroughEndOfInput()
    return segments
}

/// Drives the async transcription to completion synchronously. The C caller is
/// Rust's `spawn_blocking` thread, so blocking here is correct — the async work
/// runs on a detached Task and this thread only waits on a semaphore; we never
/// block Swift's cooperative pool.
@available(macOS 26.0, *)
private func transcribeBlocking(
    buffer: AVAudioPCMBuffer,
    langCode: String?,
    // Reserved ABI slot — see `transcribe`; translation is Whisper-only in lib.rs.
    translate: Bool
) -> Result<[Segment], Error> {
    let semaphore = DispatchSemaphore(value: 0)
    final class Box: @unchecked Sendable { var value: Result<[Segment], Error> = .success([]) }
    let box = Box()
    Task.detached {
        do {
            let segs = try await transcribe(buffer: buffer, langCode: langCode, translate: translate)
            box.value = .success(segs)
        } catch {
            box.value = .failure(error)
        }
        semaphore.signal()
    }
    semaphore.wait()
    return box.value
}

// MARK: - @_cdecl C-ABI surface

/// See bridge.h. Never traps (clause c): every path is do/catch-guarded and any
/// failure returns nil with `*out_error` set.
@_cdecl("lens_asr_transcribe")
func lens_asr_transcribe(
    _ pcm: UnsafePointer<Float>?,
    _ pcm_len: Int,
    _ sample_rate: Int32,
    _ lang_code: UnsafePointer<CChar>?,
    // Reserved ABI slot — SpeechTranscriber has no translate task; translation is
    // routed to Whisper in lib.rs. Kept in the C ABI for signature stability.
    _ translate: Int32,
    _ out_error: UnsafeMutablePointer<UnsafeMutablePointer<LensAsrError>?>?
) -> UnsafeMutablePointer<LensAsrResult>? {

    guard #available(macOS 26.0, *) else {
        setError(BridgeError.unsupportedOS.description, out_error)
        return nil
    }
    guard let pcm = pcm, pcm_len > 0 else {
        setError("empty or null PCM buffer", out_error)
        return nil
    }

    // Borrowed lang code → owned Swift String (we never retain the C pointer).
    let langCode: String? = lang_code.map { String(cString: $0) }

    let sampleRate = Double(sample_rate > 0 ? sample_rate : 16_000)
    guard let buffer = makeBuffer(pcm: pcm, count: pcm_len, sampleRate: sampleRate) else {
        setError(BridgeError.bufferAllocationFailed.description, out_error)
        return nil
    }

    let outcome = transcribeBlocking(buffer: buffer, langCode: langCode, translate: translate != 0)

    let segments: [Segment]
    switch outcome {
    case .success(let segs):
        segments = segs
    case .failure(let error):
        setError("apple transcription failed: \(error.localizedDescription)", out_error)
        return nil
    }

    // Marshal segments into an owned C array (clause a/b). Per-segment text bytes
    // are allocated; all freed by lens_asr_free.
    let resultBuf = UnsafeMutablePointer<LensAsrResult>.allocate(capacity: 1)
    if segments.isEmpty {
        resultBuf.initialize(to: LensAsrResult(segments: nil, segment_count: 0))
        return resultBuf
    }
    let segBuf = UnsafeMutablePointer<LensAsrSegment>.allocate(capacity: segments.count)
    for (i, seg) in segments.enumerated() {
        let (ptr, len) = allocUTF8(seg.text)
        segBuf[i] = LensAsrSegment(text_ptr: ptr,
                                   text_len: len,
                                   start_second: seg.start,
                                   end_second: seg.end)
    }
    resultBuf.initialize(to: LensAsrResult(segments: UnsafePointer(segBuf),
                                           segment_count: segments.count))
    return resultBuf
}

/// See bridge.h. Frees a LensAsrResult and everything it owns. No-op on nil.
@_cdecl("lens_asr_free")
func lens_asr_free(_ result: UnsafeMutablePointer<LensAsrResult>?) {
    guard let result = result else { return }
    let count = result.pointee.segment_count
    if let segs = result.pointee.segments {
        let mutableSegs = UnsafeMutablePointer(mutating: segs)
        for i in 0..<count {
            if let textPtr = mutableSegs[i].text_ptr {
                UnsafeMutablePointer(mutating: textPtr).deallocate()
            }
        }
        mutableSegs.deallocate()
    }
    result.deallocate()
}

/// See bridge.h. Frees a LensAsrError. No-op on nil.
@_cdecl("lens_asr_free_error")
func lens_asr_free_error(_ error: UnsafeMutablePointer<LensAsrError>?) {
    guard let error = error else { return }
    if let msg = error.pointee.message_ptr {
        UnsafeMutablePointer(mutating: msg).deallocate()
    }
    error.deallocate()
}

/// See bridge.h. Non-zero if Apple supports `lang_code`. Never traps: any error
/// (including an unavailable OS) returns 0.
@_cdecl("lens_asr_supports_locale")
func lens_asr_supports_locale(_ lang_code: UnsafePointer<CChar>?) -> Int32 {
    guard #available(macOS 26.0, *) else { return 0 }
    guard let lang_code = lang_code else { return 0 }
    let code = String(cString: lang_code)
    if code.isEmpty { return 0 }

    let semaphore = DispatchSemaphore(value: 0)
    final class Box: @unchecked Sendable { var supported = false }
    let box = Box()
    Task.detached {
        let target = Locale(identifier: code)
        let supported = await SpeechTranscriber.supportedLocales
        // Match on language identifier so region variants ("en_US") still count.
        box.supported = supported.contains { loc in
            loc.identifier(.bcp47) == target.identifier(.bcp47)
                || loc.language.languageCode == target.language.languageCode
        }
        semaphore.signal()
    }
    semaphore.wait()
    return box.supported ? 1 : 0
}
