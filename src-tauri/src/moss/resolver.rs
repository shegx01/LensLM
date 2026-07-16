//! Runtime resolution of the MOSS sidecar launch spec (issue #193, [161e]).
//!
//! Finds a usable `uv`: a system install (`uv --version` on PATH) is preferred;
//! otherwise Astral's pre-signed static `uv` is downloaded into `<app_data>/bin/`
//! (SHA-pinned) and reused thereafter. The sidecar then runs via `uv run` against
//! the bundled project, with `HF_HOME` pointed at an app-data cache so
//! `mlx-speech` auto-downloads the model there. All failures map through [`tts_err`].

use std::path::{Path, PathBuf};
use std::process::Stdio;

use sha2::{Digest, Sha256};

use super::{SidecarSpawn, tts_err};
use lens_core::LensError;

/// Pinned Astral `uv` release for `aarch64-apple-darwin`. The SHA is over the
/// published `.tar.gz` asset (verified against the real release download).
const UV_VERSION: &str = "0.11.29";
const UV_URL: &str =
    "https://github.com/astral-sh/uv/releases/download/0.11.29/uv-aarch64-apple-darwin.tar.gz";
const UV_SHA256: &str = "61c04acc52a33ef0f331e494bdfbedcdb6c26c6970c022ed3699e5860f8930e3";
/// Path of the `uv` binary inside the extracted archive.
const UV_ARCHIVE_BIN: &str = "uv-aarch64-apple-darwin/uv";

/// Ceiling on the one-time `uv` download (a ~35 MB static binary).
const UV_DOWNLOAD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// Builds the production [`SidecarSpawn`]: resolve `uv`, ensure the HF cache dir,
/// and run the bundled sidecar under `uv run` with `HF_HOME` set. May download
/// `uv`, so it is invoked lazily off the async runtime.
pub fn resolve_sidecar_spawn(
    app_data_dir: &Path,
    sidecar_dir: &Path,
    hf_cache_dir: &Path,
) -> Result<SidecarSpawn, LensError> {
    let uv = find_or_provision_uv(app_data_dir)?;

    // huggingface_hub reads HF_HOME at import → the dir must exist and be in the
    // spawned process env (see `envs` below), not just the parent's.
    std::fs::create_dir_all(hf_cache_dir)
        .map_err(warn_tts("failed to create MOSS HF cache dir"))?;

    // `uv run --project <dir>` writes its virtualenv to `<dir>/.venv` by default;
    // in a packaged app `<dir>` is inside the read-only, signed .app bundle, so uv
    // could not create it and the sidecar would never start. Redirect both the
    // project env and uv's cache into writable app-data.
    let venv_dir = app_data_dir.join("moss-venv");
    let uv_cache_dir = app_data_dir.join("uv-cache");
    for dir in [&venv_dir, &uv_cache_dir] {
        std::fs::create_dir_all(dir).map_err(warn_tts("failed to create uv runtime dir"))?;
    }

    let script = sidecar_dir.join("mlx_speech_sidecar.py");
    let args = vec![
        "run".to_string(),
        // `--frozen`: use uv.lock verbatim — never re-resolve (which would bypass
        // the lock's pinned hashes) and never try to rewrite the lock inside the
        // read-only .app bundle.
        "--frozen".to_string(),
        "--project".to_string(),
        sidecar_dir.to_string_lossy().into_owned(),
        "python".to_string(),
        // Absolute so the launch is independent of the child's working directory.
        script.to_string_lossy().into_owned(),
    ];
    let envs = vec![
        (
            "HF_HOME".to_string(),
            hf_cache_dir.to_string_lossy().into_owned(),
        ),
        (
            "UV_PROJECT_ENVIRONMENT".to_string(),
            venv_dir.to_string_lossy().into_owned(),
        ),
        (
            "UV_CACHE_DIR".to_string(),
            uv_cache_dir.to_string_lossy().into_owned(),
        ),
    ];

    Ok(SidecarSpawn {
        program: uv,
        args,
        envs,
    })
}

/// Builds a `map_err` closure that logs `msg` at `warn` and maps to [`tts_err`],
/// folding the repeated log-then-map-to-generic-error boilerplate at each fallible step.
fn warn_tts<E: std::fmt::Display>(msg: &'static str) -> impl FnOnce(E) -> LensError {
    move |e| {
        tracing::warn!(error = %e, "{msg}");
        tts_err()
    }
}

/// Prefer a system `uv`; else reuse a previously provisioned one; else download it.
///
/// A system `uv` on PATH is trusted the same as any developer tool (a PATH-hijack
/// is out of scope). A previously-provisioned `uv` is reused only when its
/// `--version` still matches the pinned [`UV_VERSION`] — a truncated/broken
/// install, or one left over from before a pin bump, falls through and
/// re-provisions instead of being executed forever.
fn find_or_provision_uv(app_data_dir: &Path) -> Result<PathBuf, LensError> {
    if let Some(uv) = system_uv() {
        return Ok(uv);
    }
    let bin_dir = app_data_dir.join("bin");
    let uv_path = bin_dir.join("uv");
    if uv_path.is_file() && uv_is_pinned(&uv_path) {
        return Ok(uv_path);
    }
    let _ = std::fs::remove_file(&uv_path);
    tracing::info!(
        version = UV_VERSION,
        "provisioning Astral uv for MOSS sidecar"
    );
    download_uv(&bin_dir)
}

/// A system `uv` whose version is recent enough to support `uv run --project` and
/// the pinned lockfile format; older ones fall through to the pinned download.
fn system_uv() -> Option<PathBuf> {
    let out = std::process::Command::new("uv")
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .ok()?;
    if out.status.success() && uv_version_supported(&String::from_utf8_lossy(&out.stdout)) {
        Some(PathBuf::from("uv"))
    } else {
        None
    }
}

/// Parses `uv --version` output (e.g. `uv 0.11.29 (…)`) and accepts `>= 0.8.0` —
/// the floor that reads the bundled `uv.lock` (revision 3). Older `uv` would fail
/// to parse the lock (or try to rewrite it), so it falls through to the pin.
fn uv_version_supported(version_line: &str) -> bool {
    version_line
        .split_whitespace()
        .find_map(|tok| {
            let mut parts = tok.split('.');
            let major = parts.next()?.parse::<u32>().ok()?;
            let minor = parts.next()?.parse::<u32>().ok()?;
            Some((major, minor))
        })
        .is_some_and(|(major, minor)| major > 0 || minor >= 8)
}

/// True if `uv_path --version` runs successfully AND its output contains the
/// pinned [`UV_VERSION`] (re-hashing is not possible here — `UV_SHA256` is over
/// the tarball, not the extracted binary; a version mismatch after a pin bump
/// is the case this exists to catch).
fn uv_is_pinned(uv_path: &Path) -> bool {
    std::process::Command::new(uv_path)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .is_ok_and(|out| {
            out.status.success() && String::from_utf8_lossy(&out.stdout).contains(UV_VERSION)
        })
}

/// Downloads the pinned `uv` archive, verifies its SHA256, extracts the binary
/// via the OS `tar` (gzip-capable on macOS — no `tar`/`flate2` crate), installs
/// it at `<bin_dir>/uv` (chmod 0o755), and cleans up the intermediate files.
fn download_uv(bin_dir: &Path) -> Result<PathBuf, LensError> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::create_dir_all(bin_dir).map_err(warn_tts("failed to create uv bin dir"))?;

    let client = reqwest::blocking::Client::builder()
        .timeout(UV_DOWNLOAD_TIMEOUT)
        .build()
        .map_err(warn_tts("failed to build uv download client"))?;
    let bytes = client
        .get(UV_URL)
        .send()
        .and_then(|r| r.error_for_status())
        .and_then(|r| r.bytes())
        .map_err(warn_tts("uv download failed"))?;

    // Integrity gate: reject the bytes unless they match the pinned release SHA.
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let mut actual = String::with_capacity(64);
    for b in hasher.finalize() {
        use std::fmt::Write;
        let _ = write!(actual, "{b:02x}");
    }
    if actual != UV_SHA256 {
        tracing::warn!("uv download failed integrity check");
        return Err(tts_err());
    }

    let tar_path = bin_dir.join("uv-download.tar.gz");
    std::fs::write(&tar_path, &bytes).map_err(warn_tts("failed to write uv archive"))?;

    let extract_dir = bin_dir.join("uv-extract");
    let _ = std::fs::remove_dir_all(&extract_dir);
    std::fs::create_dir_all(&extract_dir).map_err(warn_tts("failed to create uv extract dir"))?;
    // Unsandboxed extraction is safe because the archive bytes were SHA256-pinned
    // above: `tar` only ever sees the known-good Astral release, and we copy out
    // only the fixed `UV_ARCHIVE_BIN` path afterward.
    let status = std::process::Command::new("tar")
        .arg("xzf")
        .arg(&tar_path)
        .arg("-C")
        .arg(&extract_dir)
        .status()
        .map_err(warn_tts("failed to run tar for uv extract"))?;
    if !status.success() {
        tracing::warn!("tar failed to extract uv archive");
        return Err(tts_err());
    }

    // Atomic install: copy+chmod a staging file, then rename onto the final path,
    // so a crash mid-install never leaves a truncated `uv` that `is_file()` would
    // accept and execute forever.
    let extracted = extract_dir.join(UV_ARCHIVE_BIN);
    let uv_path = bin_dir.join("uv");
    let staged = bin_dir.join("uv.partial");
    std::fs::copy(&extracted, &staged).map_err(warn_tts("failed to install uv binary"))?;
    std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o755))
        .map_err(warn_tts("failed to chmod uv binary"))?;
    std::fs::rename(&staged, &uv_path).map_err(warn_tts("failed to finalize uv binary"))?;

    let _ = std::fs::remove_file(&tar_path);
    let _ = std::fs::remove_dir_all(&extract_dir);
    Ok(uv_path)
}

#[cfg(test)]
mod tests {
    use super::{uv_is_pinned, uv_version_supported};

    /// Writes an executable shell stub at `dir/uv` that prints `version_line` to
    /// stdout for `--version` and exits 0, mimicking a real `uv` binary closely
    /// enough for `uv_is_pinned`'s output-matching to exercise.
    fn stub_uv(dir: &std::path::Path, version_line: &str) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join("uv");
        std::fs::write(&path, format!("#!/bin/sh\necho '{version_line}'\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[test]
    fn uv_is_pinned_accepts_matching_version_and_rejects_drift() {
        let dir = tempfile::tempdir().unwrap();
        let matching = stub_uv(dir.path(), "uv 0.11.29 (abc 2026-01-01)");
        assert!(uv_is_pinned(&matching));

        let drifted_dir = tempfile::tempdir().unwrap();
        let drifted = stub_uv(drifted_dir.path(), "uv 0.9.0 (abc 2025-01-01)");
        assert!(!uv_is_pinned(&drifted));
    }

    #[test]
    fn version_gate_accepts_recent_and_rejects_old_or_garbage() {
        assert!(uv_version_supported("uv 0.11.29 (abc 2026-01-01)"));
        assert!(uv_version_supported("uv 0.8.0"));
        assert!(uv_version_supported("uv 1.0.0"));
        // Below the revision-3 lock floor (0.8.0) — must fall through to the pin.
        assert!(!uv_version_supported("uv 0.7.20"));
        assert!(!uv_version_supported("uv 0.5.0"));
        assert!(!uv_version_supported("uv 0.1.2"));
        assert!(!uv_version_supported("not a version"));
        assert!(!uv_version_supported(""));
    }
}
