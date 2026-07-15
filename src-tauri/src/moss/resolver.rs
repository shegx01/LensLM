//! Runtime resolution of the MOSS sidecar launch spec (issue #193, [161e]).
//!
//! Finds a usable `uv`: a system install (`uv --version` on PATH) is preferred;
//! otherwise Astral's pre-signed static `uv` is downloaded into `<app_data>/bin/`
//! (SHA-pinned) and reused thereafter. The sidecar then runs via `uv run` against
//! the bundled project, with `HF_HOME` pointed at an app-data cache so
//! `mlx-speech` auto-downloads the model there. All failures map to the generic
//! [`tts_err`] so no path/internal detail crosses the IPC boundary.

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
    std::fs::create_dir_all(hf_cache_dir).map_err(|e| {
        tracing::warn!(error = %e, "failed to create MOSS HF cache dir");
        tts_err()
    })?;

    let script = sidecar_dir.join("mlx_speech_sidecar.py");
    let args = vec![
        "run".to_string(),
        "--project".to_string(),
        sidecar_dir.to_string_lossy().into_owned(),
        "python".to_string(),
        // Absolute so the launch is independent of the child's working directory.
        script.to_string_lossy().into_owned(),
    ];
    let envs = vec![(
        "HF_HOME".to_string(),
        hf_cache_dir.to_string_lossy().into_owned(),
    )];

    Ok(SidecarSpawn {
        program: uv,
        args,
        envs,
    })
}

/// Prefer a system `uv`; else reuse a previously provisioned one; else download it.
fn find_or_provision_uv(app_data_dir: &Path) -> Result<PathBuf, LensError> {
    if system_uv_present() {
        return Ok(PathBuf::from("uv"));
    }
    let bin_dir = app_data_dir.join("bin");
    let uv_path = bin_dir.join("uv");
    if uv_path.is_file() {
        return Ok(uv_path);
    }
    tracing::info!(
        version = UV_VERSION,
        "provisioning Astral uv for MOSS sidecar"
    );
    download_uv(&bin_dir)
}

fn system_uv_present() -> bool {
    std::process::Command::new("uv")
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Downloads the pinned `uv` archive, verifies its SHA256, extracts the binary
/// via the OS `tar` (gzip-capable on macOS — no `tar`/`flate2` crate), installs
/// it at `<bin_dir>/uv` (chmod 0o755), and cleans up the intermediate files.
fn download_uv(bin_dir: &Path) -> Result<PathBuf, LensError> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::create_dir_all(bin_dir).map_err(|e| {
        tracing::warn!(error = %e, "failed to create uv bin dir");
        tts_err()
    })?;

    let client = reqwest::blocking::Client::builder()
        .timeout(UV_DOWNLOAD_TIMEOUT)
        .build()
        .map_err(|e| {
            tracing::warn!(error = %e, "failed to build uv download client");
            tts_err()
        })?;
    let bytes = client
        .get(UV_URL)
        .send()
        .and_then(|r| r.error_for_status())
        .and_then(|r| r.bytes())
        .map_err(|e| {
            tracing::warn!(error = %e, "uv download failed");
            tts_err()
        })?;

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
    std::fs::write(&tar_path, &bytes).map_err(|e| {
        tracing::warn!(error = %e, "failed to write uv archive");
        tts_err()
    })?;

    let extract_dir = bin_dir.join("uv-extract");
    let _ = std::fs::remove_dir_all(&extract_dir);
    std::fs::create_dir_all(&extract_dir).map_err(|e| {
        tracing::warn!(error = %e, "failed to create uv extract dir");
        tts_err()
    })?;
    let status = std::process::Command::new("tar")
        .arg("xzf")
        .arg(&tar_path)
        .arg("-C")
        .arg(&extract_dir)
        .status()
        .map_err(|e| {
            tracing::warn!(error = %e, "failed to run tar for uv extract");
            tts_err()
        })?;
    if !status.success() {
        tracing::warn!("tar failed to extract uv archive");
        return Err(tts_err());
    }

    let extracted = extract_dir.join(UV_ARCHIVE_BIN);
    let uv_path = bin_dir.join("uv");
    std::fs::copy(&extracted, &uv_path).map_err(|e| {
        tracing::warn!(error = %e, "failed to install uv binary");
        tts_err()
    })?;
    std::fs::set_permissions(&uv_path, std::fs::Permissions::from_mode(0o755)).map_err(|e| {
        tracing::warn!(error = %e, "failed to chmod uv binary");
        tts_err()
    })?;

    let _ = std::fs::remove_file(&tar_path);
    let _ = std::fs::remove_dir_all(&extract_dir);
    Ok(uv_path)
}
