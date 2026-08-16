use std::env;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

const SWIFT_SRC: &str = "src/asr/bridge.swift";
const SWIFT_HEADER: &str = "src/asr/bridge.h";
const LIB_NAME: &str = "lens_asr_bridge";
/// The only target that links the bridge. Host arch is irrelevant: the swiftc
/// invocation below is already a cross-compile (`-target` plus an explicit `-sdk`)
/// and Xcode's Swift toolchain is universal.
const BRIDGE_TARGET: &str = "aarch64-apple-darwin";
/// SpeechAnalyzer/SpeechTranscriber floor — the deployment target the bridge is
/// compiled against, and the SDK major the probe requires.
const MACOS_TARGET_MAJOR: u32 = 26;
/// `xcode-select -s` rewrites this symlink and touches no env var, so it has to be
/// a tracked input or Cargo replays a stale toolchain decision.
const XCODE_SELECT_LINK: &str = "/var/db/xcode_select_link";

/// How a missing or broken Swift toolchain is treated.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BridgeMode {
    AllowMissing,
    Require,
}

fn main() {
    // `commands/system.rs` reads this cfg via the `cfg!()` macro on every target,
    // so the name must be declared unconditionally or `unexpected_cfgs` fires.
    println!("cargo:rustc-check-cfg=cfg(apple_asr_bridge)");
    tauri_build::build();

    // Above every early return: a bridge edit must invalidate the cached build
    // script even on a run that skips the compile.
    println!("cargo:rerun-if-changed={SWIFT_SRC}");
    println!("cargo:rerun-if-changed={SWIFT_HEADER}");
    for var in ["LENS_ASR_BRIDGE", "DEVELOPER_DIR", "SDKROOT"] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let shipping_target = env::var("TARGET").as_deref() == Ok(BRIDGE_TARGET);
    let mode = bridge_mode(shipping_target);
    if !shipping_target {
        // An explicit `require` is a promise about THIS artifact, so it must fail
        // here too — otherwise the flag is a no-op on every non-Apple machine.
        if mode == BridgeMode::Require {
            bridge_unavailable(
                mode,
                &format!(
                    "the build target is not {BRIDGE_TARGET}, which is the only one that links it"
                ),
            );
        }
        return;
    }

    build_apple_asr_bridge(mode);
}

/// `LENS_ASR_BRIDGE` overrides the profile default; unset means strict for a release
/// build of the shipping target (a bundle users install must carry the bridge) and
/// lenient everywhere else.
fn bridge_mode(shipping_target: bool) -> BridgeMode {
    match env::var("LENS_ASR_BRIDGE").as_deref() {
        Ok("require") => BridgeMode::Require,
        Ok("allow-missing") => BridgeMode::AllowMissing,
        Ok(other) => panic!(
            "LENS_ASR_BRIDGE={other:?} is not a recognized value; use `require`, `allow-missing`, or leave it unset"
        ),
        Err(_) if shipping_target && env::var("PROFILE").as_deref() == Ok("release") => {
            BridgeMode::Require
        }
        Err(_) => BridgeMode::AllowMissing,
    }
}

/// Reports a bridge the build could not produce, emitting no cfg or link
/// directives either way.
fn bridge_unavailable(mode: BridgeMode, reason: &str) {
    match mode {
        BridgeMode::AllowMissing => {
            println!("cargo:warning=Apple-native ASR bridge not built: {reason}")
        }
        BridgeMode::Require => panic!("Apple-native ASR bridge is required, but {reason}"),
    }
}

/// Compiles `bridge.swift` into a static library and emits the link flags for it
/// plus the Apple frameworks it drives (SpeechAnalyzer/AVAudioPCMBuffer/CMTime).
/// Success is what sets `apple_asr_bridge`; nothing else does.
fn build_apple_asr_bridge(mode: BridgeMode) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is always set by cargo"));
    let lib_path = out_dir.join(format!("lib{LIB_NAME}.a"));

    let sdk_path = match probe_toolchain() {
        Ok(sdk) => sdk,
        Err(reason) => {
            abandon_bridge(mode, &lib_path, &reason);
            return;
        }
    };

    // `-emit-library -static` produces a `.a` carrying the Swift @_cdecl symbols;
    // `-parse-as-library` avoids main.swift top-level-code semantics; the imported
    // header is the ONLY way `@_cdecl` functions can traffic in C structs.
    let mut cmd = Command::new("xcrun");
    cmd.arg("swiftc").args([
        "-emit-library",
        "-static",
        "-parse-as-library",
        "-O",
        "-target",
        &format!("arm64-apple-macos{MACOS_TARGET_MAJOR}.0"),
        "-module-name",
        LIB_NAME,
        "-import-objc-header",
        SWIFT_HEADER,
    ]);
    if let Some(sdk) = &sdk_path {
        cmd.args(["-sdk", sdk]);
    }
    cmd.arg("-o").arg(&lib_path).arg(SWIFT_SRC);
    let compiled = match cmd.status() {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!("swiftc exited with {status} compiling {SWIFT_SRC}")),
        Err(e) => Err(format!("swiftc could not be spawned: {e}")),
    };
    if let Err(reason) = compiled {
        abandon_bridge(mode, &lib_path, &reason);
        return;
    }
    // The cfg promises a linkable archive. A replayed-but-cleaned OUT_DIR would
    // otherwise emit link directives pointing at nothing.
    assert!(
        lib_path.exists(),
        "swiftc reported success but {} is missing",
        lib_path.display()
    );

    println!("cargo:rustc-cfg=apple_asr_bridge");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={LIB_NAME}");

    // The Swift static libraries live under the toolchain's macosx lib dir.
    if let Some(swift_lib_dir) = swift_static_lib_dir() {
        println!("cargo:rustc-link-search=native={swift_lib_dir}");
    }
    // Frameworks driven by bridge.swift: Speech (SpeechAnalyzer/SpeechTranscriber/
    // AssetInventory), AVFAudio (AVAudioPCMBuffer/AVAudioFormat), CoreMedia (CMTime/
    // CMTimeRange), Foundation (AttributedString/Locale).
    for framework in ["Speech", "AVFAudio", "CoreMedia", "Foundation"] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
    for lib in ["swiftCore", "swiftFoundation"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    // Ensure the dynamic loader can find the Swift runtime at run time. `scripts/
    // signoff.sh` greps the linked binary for this rpath as the bridge's proof.
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
}

/// Drops the archive an earlier capable build left behind — it would otherwise
/// stay linkable — then reports per `mode`.
fn abandon_bridge(mode: BridgeMode, lib_path: &Path, reason: &str) {
    let _ = fs::remove_file(lib_path);
    bridge_unavailable(mode, reason);
}

/// Cheap capability gate: `swiftc` must resolve through `xcrun` (running it off
/// PATH fails to load the Swift stdlib) and the SDK must be new enough for the
/// deployment target. The bridge compile itself is the authoritative trial, so
/// this only avoids a run that is guaranteed to fail. Returns the SDK path.
fn probe_toolchain() -> Result<Option<String>, String> {
    // Installing Xcode or switching toolchains changes neither a source file nor
    // an env var, so the inputs this decision really depends on are tracked here.
    println!("cargo:rerun-if-changed={XCODE_SELECT_LINK}");

    if xcrun_capture(&["-f", "swiftc"]).is_none() {
        return Err("swiftc was not found via `xcrun -f swiftc`".to_string());
    }
    let Some(version) = xcrun_capture(&["--show-sdk-version"]) else {
        return Err(
            "the macOS SDK version is unreadable via `xcrun --show-sdk-version`".to_string(),
        );
    };
    let sdk_path = xcrun_capture(&["--show-sdk-path"]);
    if let Some(sdk) = &sdk_path {
        println!(
            "cargo:rerun-if-changed={}",
            Path::new(sdk).join("SDKSettings.plist").display()
        );
    }
    match version
        .split('.')
        .next()
        .and_then(|m| m.parse::<u32>().ok())
    {
        Some(major) if major >= MACOS_TARGET_MAJOR => Ok(sdk_path),
        Some(major) => Err(format!(
            "the macOS SDK is version {major}, but the bridge targets macOS {MACOS_TARGET_MAJOR}"
        )),
        None => Err(format!("the macOS SDK version {version:?} is unparseable")),
    }
}

/// Trimmed stdout of a successful `xcrun` call, `None` on failure or empty output.
fn xcrun_capture(args: &[&str]) -> Option<String> {
    Command::new("xcrun")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Locates the toolchain's static Swift stdlib dir (`.../lib/swift/macosx`).
/// Returns `None` if it cannot be resolved; the default loader search paths
/// then apply.
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
