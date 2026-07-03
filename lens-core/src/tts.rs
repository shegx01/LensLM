//! Text-to-speech (Kokoro): static voice catalog and ONNX model download.
//!
//! Owns the pure pieces (voice list, [`DownloadProgress`] IPC type, streaming download with a
//! progress closure); the Tauri command layer adapts the closure onto a `tauri::ipc::Channel`.

use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::LensError;

/// HuggingFace URL for the quantized Kokoro-82M ONNX model (~86 MiB).
/// Tests inject a mock-server URL so they never touch the network.
pub const KOKORO_MODEL_URL: &str =
    "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_q8f16.onnx";

pub const KOKORO_MODEL_FILENAME: &str = "model_q8f16.onnx";
pub const KOKORO_MODEL_RELPATH: &str = "models/kokoro/model_q8f16.onnx";

/// SHA256 from the HuggingFace LFS `oid` (`lfs.oid` IS the file SHA256). Verified
/// after download, before the `.part → final` rename, to reject a corrupted transfer.
const KOKORO_MODEL_SHA256: Option<&str> =
    Some("04c658aec1b6008857c2ad10f8c589d4180d0ec427e7e6118ceb487e215c3cd0");

/// Only the connect phase is bounded — the body stream can take minutes over a slow link.
const DOWNLOAD_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Single source of truth for the model path; shared by the system-check TTS probe and the
/// downloader so they can never disagree about the location.
pub fn kokoro_model_path(data_dir: &Path) -> PathBuf {
    data_dir
        .join("models")
        .join("kokoro")
        .join(KOKORO_MODEL_FILENAME)
}

/// Reads `Content-Length` directly from headers rather than `Response::content_length()`
/// because the latter is `None` for HEAD responses even when the server advertises it.
fn content_length_header(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

/// Speaker gender. Serializes lowercase to match the `'male' | 'female'` union in the Svelte
/// client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Gender {
    Male,
    Female,
}

/// One selectable Kokoro voice. Frozen IPC contract — mirrored in the Svelte client as
/// `TtsVoice { id, name, gender }`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtsVoice {
    pub id: String,
    pub name: String,
    pub gender: Gender,
}

impl TtsVoice {
    fn new(id: &str, name: &str, gender: Gender) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            gender,
        }
    }
}

/// Static Kokoro-82M voice catalog from the model card. Fixed list: voices ship with the
/// weights, so no runtime enumeration is needed. Female first, then male, in model-card order.
pub fn list_tts_voices() -> Vec<TtsVoice> {
    use Gender::{Female, Male};
    vec![
        TtsVoice::new("af_heart", "Heart", Female),
        TtsVoice::new("af_bella", "Bella", Female),
        TtsVoice::new("af_nicole", "Nicole", Female),
        TtsVoice::new("af_sarah", "Sarah", Female),
        TtsVoice::new("af_sky", "Sky", Female),
        TtsVoice::new("am_michael", "Michael", Male),
        TtsVoice::new("am_puck", "Puck", Male),
        TtsVoice::new("am_adam", "Adam", Male),
        TtsVoice::new("am_echo", "Echo", Male),
        TtsVoice::new("am_onyx", "Onyx", Male),
    ]
}

/// Download progress. Frozen IPC contract — mirrored in the Svelte client as
/// `{ received, total, done }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub received: u64,
    pub total: Option<u64>,
    pub done: bool,
}

/// Downloads the Kokoro ONNX model from `url` into `dest` with streaming progress.
/// Idempotent: skips if `dest` already exists with the right size. A size mismatch
/// (e.g. truncated prior run) re-downloads. Verifies against [`KOKORO_MODEL_SHA256`]
/// before the `.part → final` rename.
pub async fn download_kokoro_model<F>(
    url: &str,
    dest: &Path,
    on_progress: F,
) -> Result<(), LensError>
where
    F: FnMut(DownloadProgress),
{
    download_kokoro_model_verified(url, dest, KOKORO_MODEL_SHA256, on_progress).await
}

/// Like [`download_kokoro_model`] but with an injectable hash so tests can exercise the
/// integrity gate. `None` skips verification.
async fn download_kokoro_model_verified<F>(
    url: &str,
    dest: &Path,
    expected_sha256: Option<&str>,
    mut on_progress: F,
) -> Result<(), LensError>
where
    F: FnMut(DownloadProgress),
{
    // HEAD probe gives the expected size for idempotency without streaming the body.
    // Redirects are NOT disabled: HuggingFace /resolve/ 302-redirects to a CDN.
    let client = reqwest::Client::builder()
        .connect_timeout(DOWNLOAD_CONNECT_TIMEOUT)
        .build()
        .map_err(|e| LensError::Network(format!("download client init failed: {e}")))?;

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

    // TODO(M2): disk-space pre-check — before streaming ~86 MiB, verify the
    // target volume has enough free space and fail early with a clear error
    // rather than mid-stream on ENOSPC. Deferred (needs a cross-platform
    // free-space probe); tracked in the M1 onboarding review notes.
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

    // Write to `.part`, then atomically rename on success.
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

    // Compare the streamed digest to the pinned hash; a mismatch deletes `.part`.
    if let Some(expected) = expected_sha256 {
        let actual = crate::hex_encode(&hasher.finalize());
        if !actual.eq_ignore_ascii_case(expected) {
            let _ = std::fs::remove_file(&tmp);
            return Err(LensError::Network(format!(
                "downloaded model failed integrity check: expected sha256 {expected}, got {actual}"
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

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn voice_catalog_has_five_female_and_five_male() {
        let voices = list_tts_voices();
        assert_eq!(voices.len(), 10);
        let female = voices.iter().filter(|v| v.gender == Gender::Female).count();
        let male = voices.iter().filter(|v| v.gender == Gender::Male).count();
        assert_eq!(female, 5);
        assert_eq!(male, 5);
        assert!(
            voices
                .iter()
                .any(|v| v.id == "af_heart" && v.name == "Heart")
        );
        assert!(
            voices
                .iter()
                .any(|v| v.id == "am_michael" && v.name == "Michael")
        );
    }

    #[test]
    fn gender_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&Gender::Male).unwrap(), "\"male\"");
        assert_eq!(
            serde_json::to_string(&Gender::Female).unwrap(),
            "\"female\""
        );
    }

    #[tokio::test]
    async fn download_writes_file_and_emits_progress() {
        let body = vec![7u8; 2048];
        let server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-length", "2048"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("models/kokoro/model_q8f16.onnx");

        let mut events = Vec::new();
        download_kokoro_model_verified(&server.uri(), &dest, None, |p| events.push(p))
            .await
            .unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert!(!events.is_empty());
        let last = events.last().unwrap();
        assert!(last.done);
        assert_eq!(last.received, 2048);
        assert_eq!(last.total, Some(2048));
        assert!(events.iter().any(|e| !e.done));
    }

    #[tokio::test]
    async fn download_is_idempotent_when_file_present_with_right_size() {
        let body = vec![1u8; 512];
        let server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-length", "512"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("models/kokoro/model_q8f16.onnx");
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        std::fs::write(&dest, &body).unwrap();

        let mut events = Vec::new();
        download_kokoro_model(&server.uri(), &dest, |p| events.push(p))
            .await
            .unwrap();

        assert_eq!(events.len(), 1);
        assert!(events[0].done);
        assert_eq!(events[0].received, 512);
    }

    #[tokio::test]
    async fn download_errors_on_non_success_status() {
        let server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("models/kokoro/model_q8f16.onnx");

        let err = download_kokoro_model(&server.uri(), &dest, |_| {})
            .await
            .unwrap_err();
        assert!(matches!(err, LensError::Network(_)));
        assert!(!dest.exists());
    }

    #[test]
    fn kokoro_model_path_joins_under_data_dir() {
        let path = kokoro_model_path(Path::new("/data"));
        assert!(path.ends_with("models/kokoro/model_q8f16.onnx"));
        assert!(KOKORO_MODEL_RELPATH.ends_with(KOKORO_MODEL_FILENAME));
    }

    #[test]
    fn pinned_kokoro_sha256_is_present_and_well_formed() {
        // 64-char lowercase hex — a placeholder can never silently disable the integrity gate.
        let hash = KOKORO_MODEL_SHA256.expect("Kokoro model sha256 must be pinned");
        assert_eq!(hash.len(), 64);
        assert!(hash.bytes().all(|b| b.is_ascii_hexdigit()));
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        crate::hex_encode(&Sha256::digest(bytes))
    }

    #[tokio::test]
    async fn download_succeeds_when_hash_matches() {
        let body = vec![42u8; 4096];
        let expected = sha256_hex(&body);
        let server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-length", "4096"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("models/kokoro/model_q8f16.onnx");

        download_kokoro_model_verified(&server.uri(), &dest, Some(&expected), |_| {})
            .await
            .unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert!(!dest.with_extension("part").exists());
    }

    #[tokio::test]
    async fn download_fails_and_cleans_up_when_hash_mismatches() {
        let body = vec![42u8; 4096];
        let wrong_hash = sha256_hex(b"some other content entirely");
        let server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-length", "4096"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("models/kokoro/model_q8f16.onnx");

        let err = download_kokoro_model_verified(&server.uri(), &dest, Some(&wrong_hash), |_| {})
            .await
            .unwrap_err();

        assert!(matches!(err, LensError::Network(_)));
        assert!(!dest.exists());
        assert!(!dest.with_extension("part").exists());
    }
}
