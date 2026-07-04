// Used only by the aarch64-apple-darwin Swift-bridge build below; gated so the
// Linux fmt/clippy jobs (which never compile that path) don't see unused imports.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::{env, path::PathBuf, process::Command};

fn main() {
    tauri_build::build();

    // The Apple-native ASR bridge (issue #42) is compiled ONLY on the shipping
    // aarch64-apple-darwin target AND only when its feature is on. On every other
    // target / with the feature off this is a no-op, so `swiftc` never runs (there
    // is no Mac in CI — the Linux fmt+clippy jobs must not invoke it).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    build_apple_asr_bridge();
}

/// Compiles `src/asr/bridge.swift` into a static library and emits the link flags
/// for it plus the Apple frameworks it drives (SpeechAnalyzer/AVAudioPCMBuffer/
/// CMTime). Gated by the `apple-native-asr` cargo feature via the env var Cargo
/// sets for it (`CARGO_FEATURE_APPLE_NATIVE_ASR`), so `#[cfg(feature=...)]` in a
/// build script is unnecessary and the feature check stays a runtime env read.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn build_apple_asr_bridge() {
    // Cargo exposes each enabled feature as CARGO_FEATURE_<UPPER_SNAKE>.
    if env::var_os("CARGO_FEATURE_APPLE_NATIVE_ASR").is_none() {
        return;
    }

    let swift_src = "src/asr/bridge.swift";
    let swift_header = "src/asr/bridge.h";
    println!("cargo:rerun-if-changed={swift_src}");
    println!("cargo:rerun-if-changed={swift_header}");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is always set by cargo"));
    let lib_name = "lens_asr_bridge";
    let lib_path = out_dir.join(format!("lib{lib_name}.a"));

    // Invoke swiftc via `xcrun` when available: `xcrun` sets DEVELOPER_DIR/SDKROOT
    // so the Swift stdlib + macOS SDK resolve (running the raw swiftc path directly
    // fails with "unable to load standard library"). Fall back to PATH `swiftc`.
    let use_xcrun = Command::new("xcrun")
        .args(["-f", "swiftc"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let sdk_path = Command::new("xcrun")
        .args(["--show-sdk-path"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());

    // Build a static archive: `-emit-library -static` produces a `.a` with the
    // Swift @_cdecl symbols. `-parse-as-library` avoids top-level-code (main.swift)
    // semantics. `-import-objc-header bridge.h` makes the C structs/signatures in
    // the header visible to Swift as C types — the ONLY way `@_cdecl` functions can
    // traffic in those structs (Swift-native structs are not C/Obj-C representable).
    let mut cmd = if use_xcrun {
        let mut c = Command::new("xcrun");
        c.arg("swiftc");
        c
    } else {
        Command::new("swiftc")
    };
    cmd.args([
        "-emit-library",
        "-static",
        "-parse-as-library",
        "-O",
        "-target",
        "arm64-apple-macos26.0",
        "-module-name",
        lib_name,
        "-import-objc-header",
        swift_header,
    ]);
    if let Some(sdk) = &sdk_path {
        cmd.args(["-sdk", sdk]);
    }
    cmd.arg("-o").arg(&lib_path).arg(swift_src);
    let status = cmd
        .status()
        .expect("failed to spawn swiftc for the Apple ASR bridge");
    assert!(
        status.success(),
        "swiftc failed to build the Apple ASR bridge ({swift_src})"
    );

    // Link the static bridge archive.
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={lib_name}");

    // Swift runtime + the frameworks the bridge calls into. The Swift static
    // libraries live under the toolchain's macosx lib dir; add it as a search path.
    if let Some(swift_lib_dir) = swift_static_lib_dir() {
        println!("cargo:rustc-link-search=native={swift_lib_dir}");
    }
    // Frameworks driven by bridge.swift: Speech (SpeechAnalyzer/SpeechTranscriber/
    // AssetInventory), AVFAudio (AVAudioPCMBuffer/AVAudioFormat), CoreMedia (CMTime/
    // CMTimeRange), Foundation (AttributedString/Locale).
    for framework in ["Speech", "AVFAudio", "CoreMedia", "Foundation"] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
    // The Swift standard-library/runtime symbols the archive references.
    for lib in ["swiftCore", "swiftFoundation"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    // Ensure the dynamic loader can find the Swift runtime at run time.
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
}

/// Locates the toolchain's static Swift stdlib dir (`.../lib/swift_static/macosx`)
/// via `xcrun --show-sdk-platform-path`-relative layout, falling back to the
/// developer dir. Returns `None` if it cannot be resolved; the default loader
/// search paths then apply.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn swift_static_lib_dir() -> Option<String> {
    let dev_dir = Command::new("xcode-select")
        .arg("-p")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())?;
    let candidate =
        PathBuf::from(dev_dir).join("Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx");
    candidate
        .is_dir()
        .then(|| candidate.to_string_lossy().into_owned())
}
