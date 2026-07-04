//! ASR backend selection policy (#42). A pure function of (config override,
//! feature availability, OS facts, locale support) — mirrors the gate-ordered,
//! off-Mac-testable style of [`select_compute`](crate::embedder::device::select_compute).
//!
//! Keeping selection pure is what makes the LocalWhisper path testable ON Apple
//! hardware: force it via `config_backend`, or build without the Apple engine so
//! `apple_available == false`. lens-core stays OS-probe-free — authoritative OS
//! facts are supplied by src-tauri (which pre-gates before injecting the engine).

use super::AsrBackend;

/// Platform facts the router needs, injected so it is unit-testable off-Mac.
/// Populated authoritatively by src-tauri; lens-core never probes the OS itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Platform {
    pub is_apple_silicon_macos: bool,
    pub macos_major: Option<u32>,
}

/// Minimum macOS major version for the Apple-native ASR path (SpeechAnalyzer).
pub const MIN_MACOS_FOR_APPLE_ASR: u32 = 26;

/// Selects the ASR backend. Priority-ordered gates (each a testable arm):
/// 1. `config_backend` — explicit override ALWAYS wins (forces Whisper on Apple
///    for tests, or Apple when available).
/// 2. `!apple_available` — no Apple engine compiled/injected → LocalWhisper.
/// 3. not (Apple Silicon macOS ≥ 26) → LocalWhisper.
/// 4. `!apple_supports_locale` → LocalWhisper.
/// 5. else → AppleNative.
pub fn select_asr_backend(
    config_backend: Option<AsrBackend>,
    platform: Platform,
    apple_available: bool,
    apple_supports_locale: bool,
) -> AsrBackend {
    if let Some(b) = config_backend {
        return b; // gate 1: explicit config override wins
    }
    if !apple_available {
        return AsrBackend::LocalWhisper; // gate 2: no Apple engine present
    }
    let apple_capable = platform.is_apple_silicon_macos
        && platform
            .macos_major
            .is_some_and(|v| v >= MIN_MACOS_FOR_APPLE_ASR);
    if !apple_capable {
        return AsrBackend::LocalWhisper; // gate 3: old/non-Apple platform
    }
    if !apple_supports_locale {
        return AsrBackend::LocalWhisper; // gate 4: locale unsupported by Apple
    }
    AsrBackend::AppleNative // gate 5: capable + supported → Apple
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apple_platform() -> Platform {
        Platform {
            is_apple_silicon_macos: true,
            macos_major: Some(MIN_MACOS_FOR_APPLE_ASR),
        }
    }

    #[test]
    fn config_override_forces_backend() {
        // On a non-Apple platform, an explicit AppleNative override still wins.
        let non_apple = Platform {
            is_apple_silicon_macos: false,
            macos_major: None,
        };
        assert_eq!(
            select_asr_backend(Some(AsrBackend::AppleNative), non_apple, false, true),
            AsrBackend::AppleNative
        );
    }

    #[test]
    fn config_forces_whisper_on_apple() {
        // The testability guarantee: force Whisper even on capable Apple hardware.
        assert_eq!(
            select_asr_backend(Some(AsrBackend::LocalWhisper), apple_platform(), true, true),
            AsrBackend::LocalWhisper
        );
    }

    #[test]
    fn config_forces_apple_when_available() {
        assert_eq!(
            select_asr_backend(Some(AsrBackend::AppleNative), apple_platform(), true, true),
            AsrBackend::AppleNative
        );
    }

    #[test]
    fn no_apple_engine_is_whisper() {
        assert_eq!(
            select_asr_backend(None, apple_platform(), false, true),
            AsrBackend::LocalWhisper
        );
    }

    #[test]
    fn old_macos_is_whisper() {
        let old = Platform {
            is_apple_silicon_macos: true,
            macos_major: Some(MIN_MACOS_FOR_APPLE_ASR - 1),
        };
        assert_eq!(
            select_asr_backend(None, old, true, true),
            AsrBackend::LocalWhisper
        );
    }

    #[test]
    fn non_apple_platform_is_whisper() {
        let non_apple = Platform {
            is_apple_silicon_macos: false,
            macos_major: None,
        };
        assert_eq!(
            select_asr_backend(None, non_apple, true, true),
            AsrBackend::LocalWhisper
        );
    }

    #[test]
    fn unsupported_locale_falls_back_to_whisper() {
        assert_eq!(
            select_asr_backend(None, apple_platform(), true, false),
            AsrBackend::LocalWhisper
        );
    }

    #[test]
    fn apple_silicon_26_supported_locale_is_apple() {
        assert_eq!(
            select_asr_backend(None, apple_platform(), true, true),
            AsrBackend::AppleNative
        );
    }
}
