//! Shared streaming downloader with SHA256 integrity verification.
//!
//! Used by the TTS and Whisper ASR model downloaders so the
//! `.part` → verify → atomic-rename flow stays in one place.

use std::path::Path;

use futures_util::StreamExt;
use sha2::{Digest, Sha256};

use crate::LensError;
use crate::tts::DownloadProgress;

const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Idle read timeout: resets on each received chunk, not a total-request deadline, so a
/// large legitimate download never times out — but a fully stalled response body does
/// (see `llm::LLM_GENERATION_IDLE_TIMEOUT` for the same idiom).
const IDLE_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Free bytes available to an unprivileged writer under `dir`, or `None` when the
/// filesystem cannot be queried (a missing directory reports `ENOENT`, not zero).
/// Callers treat `None` as "abstain", never as "no space".
pub fn available_space_bytes(dir: &Path) -> Option<u64> {
    match fs4::available_space(dir) {
        Ok(bytes) => Some(bytes),
        Err(e) => {
            tracing::warn!(dir = %dir.display(), error = %e, "free-space probe failed");
            None
        }
    }
}

/// Reads `Content-Length` from response headers (works for HEAD responses too).
fn content_length_header(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

/// Streams `url` into `dest` with progress reporting and SHA256 verification.
///
/// Flow: HEAD probe for idempotency → stream to `{dest}.part`, hashing as bytes
/// arrive → verify digest (mismatch deletes `.part`, returns `Err`) → atomic rename
/// to `dest` → emit final `done` progress event. `expected_sha256 = None` skips
/// verification (use only in tests).
///
/// Integrity (SHA256) is enforced only on FRESH downloads; the size-match skip path
/// does NOT re-hash a pre-existing file (atomic rename means a landed file was
/// previously verified before it took the final name).
pub(crate) async fn download_verified<F>(
    url: &str,
    dest: &Path,
    expected_sha256: Option<&str>,
    mut on_progress: F,
) -> Result<(), LensError>
where
    F: FnMut(DownloadProgress),
{
    let client = reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(IDLE_READ_TIMEOUT)
        .build()
        .map_err(|e| LensError::Network(format!("download client init failed: {e}")))?;

    // HEAD probe gives the expected size for idempotency without streaming the body.
    // Redirects are NOT disabled: HuggingFace /resolve/ 302-redirects to a CDN.
    let expected_len = client
        .head(url)
        .send()
        .await
        .ok()
        .filter(|r| r.status().is_success())
        .and_then(|r| content_length_header(r.headers()));

    if let Ok(meta) = std::fs::metadata(dest) {
        let on_disk = meta.len();
        if on_disk > 0 && expected_len.is_some_and(|n| n == on_disk) {
            on_progress(DownloadProgress {
                received: on_disk,
                total: expected_len,
                done: true,
            });
            return Ok(());
        }
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| LensError::Io(format!("create {}: {e}", parent.display())))?;
    }

    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| LensError::Network(format!("download request failed: {e}")))?;
    if !resp.status().is_success() {
        return Err(LensError::Network(format!(
            "download failed with status {}",
            resp.status()
        )));
    }
    let total = content_length_header(resp.headers()).or(expected_len);

    let tmp = dest.with_extension("part");
    let mut file = match std::fs::File::create(&tmp) {
        Ok(file) => file,
        Err(e) => return Err(LensError::Io(format!("create {}: {e}", tmp.display()))),
    };

    use std::io::Write;
    let mut hasher = Sha256::new();
    let mut received: u64 = 0;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(e) => {
                drop(file);
                let _ = std::fs::remove_file(&tmp);
                return Err(LensError::Network(format!("download stream error: {e}")));
            }
        };
        if let Err(e) = file.write_all(&chunk) {
            drop(file);
            let _ = std::fs::remove_file(&tmp);
            return Err(LensError::Io(format!("write {}: {e}", tmp.display())));
        }
        hasher.update(&chunk);
        received += chunk.len() as u64;
        on_progress(DownloadProgress {
            received,
            total,
            done: false,
        });
    }
    if let Err(e) = file.flush() {
        drop(file);
        let _ = std::fs::remove_file(&tmp);
        return Err(LensError::Io(format!("flush {}: {e}", tmp.display())));
    }
    drop(file);

    if let Some(expected) = expected_sha256 {
        let actual = crate::hex_encode(&hasher.finalize());
        if !actual.eq_ignore_ascii_case(expected) {
            let _ = std::fs::remove_file(&tmp);
            return Err(LensError::Network(format!(
                "downloaded file failed integrity check: expected sha256 {expected}, got {actual}"
            )));
        }
    }

    if let Err(e) = std::fs::rename(&tmp, dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(LensError::Io(format!("finalize {}: {e}", dest.display())));
    }

    on_progress(DownloadProgress {
        received,
        total,
        done: true,
    });
    Ok(())
}
