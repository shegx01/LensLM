//! Text-to-speech (Kokoro) support: the static voice catalog and the engine
//! (ONNX model) download.
//!
//! `lens-core` stays Tauri-free, so this module owns only the pure pieces:
//! the voice list, the [`DownloadProgress`] IPC type, and a streaming download
//! routine that reports progress through a caller-supplied closure. The Tauri
//! command layer adapts that closure onto a `tauri::ipc::Channel`.

use std::path::Path;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use crate::LensError;

/// Canonical HuggingFace URL for the quantized Kokoro-82M ONNX model
/// (`model_q8f16.onnx`, ~86 MiB). Used by the Tauri command; tests inject a
/// mock-server URL instead so they never touch the network.
pub const KOKORO_MODEL_URL: &str =
    "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_q8f16.onnx";

/// Relative path (under the app data dir) the Kokoro model is written to.
pub const KOKORO_MODEL_RELPATH: &str = "models/kokoro/model_q8f16.onnx";

/// Reads the `Content-Length` header as a `u64`, if present and parseable.
///
/// We read the header directly rather than via `Response::content_length()`
/// because the latter is `None` for a HEAD response (no body to measure) even
/// when the server advertises the length — which is exactly the value the
/// idempotency check needs.
fn content_length_header(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

/// Speaker gender for a TTS voice. Serializes lowercase (`"male"` / `"female"`)
/// to match the `'male' | 'female'` union mirrored in the Svelte client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Gender {
    /// Male-presenting voice.
    Male,
    /// Female-presenting voice.
    Female,
}

/// One selectable Kokoro voice. Frozen IPC contract — mirrored in the Svelte
/// client (`TtsVoice { id, name, gender }`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtsVoice {
    /// Stable Kokoro voice id (e.g. `"af_heart"`).
    pub id: String,
    /// Human-readable display name (e.g. `"Heart"`).
    pub name: String,
    /// Speaker gender.
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

/// The static Kokoro-82M voice catalog (from the model card).
///
/// A fixed list is correct: these ship with the model weights, so there is no
/// runtime enumeration to perform. Female voices first, then male — both groups
/// in model-card order.
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

/// Progress for the Kokoro engine download. Frozen IPC contract — mirrored in
/// the Svelte client as `{ received, total, done }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownloadProgress {
    /// Bytes written to disk so far.
    pub received: u64,
    /// Total bytes, if the server advertised `Content-Length`.
    pub total: Option<u64>,
    /// Whether the download has finished (file fully written / already present).
    pub done: bool,
}

/// Downloads the Kokoro ONNX model from `url` into `dest`, streaming the body to
/// disk and invoking `on_progress` as bytes arrive.
///
/// Idempotent: if `dest` already exists and its size matches the server's
/// advertised `Content-Length`, the download is skipped and a single `done`
/// progress event is emitted. (A size mismatch — e.g. a truncated prior run —
/// re-downloads.)
///
/// `url` is a parameter rather than a hard-coded constant so tests can point it
/// at a mock server; production passes [`KOKORO_MODEL_URL`].
pub async fn download_kokoro_model<F>(
    url: &str,
    dest: &Path,
    mut on_progress: F,
) -> Result<(), LensError>
where
    F: FnMut(DownloadProgress),
{
    // A HEAD probe gives us the expected size for the idempotency check without
    // streaming the (large) body. If the server doesn't support HEAD or omits
    // Content-Length, we fall through to a normal download.
    let client = reqwest::Client::new();

    let expected_len = client
        .head(url)
        .send()
        .await
        .ok()
        .filter(|r| r.status().is_success())
        .and_then(|r| content_length_header(r.headers()));

    // Idempotent fast path: a complete file already on disk.
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

    // Write to a temp file in the same dir, then atomically rename on success so
    // a partial download never masquerades as a complete model on disk.
    let tmp = dest.with_extension("part");
    let mut file = std::fs::File::create(&tmp)
        .map_err(|e| LensError::Io(format!("create {}: {e}", tmp.display())))?;

    use std::io::Write;
    let mut received: u64 = 0;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| LensError::Network(format!("download stream error: {e}")))?;
        file.write_all(&chunk)
            .map_err(|e| LensError::Io(format!("write {}: {e}", tmp.display())))?;
        received += chunk.len() as u64;
        on_progress(DownloadProgress {
            received,
            total,
            done: false,
        });
    }
    file.flush()
        .map_err(|e| LensError::Io(format!("flush {}: {e}", tmp.display())))?;
    drop(file);

    std::fs::rename(&tmp, dest)
        .map_err(|e| LensError::Io(format!("finalize {}: {e}", dest.display())))?;

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
        // Spot-check the model-card ids/names.
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
        download_kokoro_model(&server.uri(), &dest, |p| events.push(p))
            .await
            .unwrap();

        // File written with the right contents.
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        // Progress emitted, last event is `done` with received == file size.
        assert!(!events.is_empty());
        let last = events.last().unwrap();
        assert!(last.done);
        assert_eq!(last.received, 2048);
        assert_eq!(last.total, Some(2048));
        // A non-final progress event was emitted while streaming.
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
        // A GET mock that would PANIC the test if hit: expect ZERO calls.
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

        // Exactly one `done` event, no re-download (GET never called → mock's
        // expect(0) is verified on server drop).
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
        // No file left behind on failure.
        assert!(!dest.exists());
    }
}
