//! TTS model registry and download helpers (#190). Mirrors `asr/registry.rs`:
//! a static allowlist of model specs with pinned URLs + SHA256 hashes, path
//! helpers, and a streaming downloader delegating to `download::download_verified`.
//! In #190 the only entry is the vestigial Kokoro model (reusing the pinned
//! `kokoro::KOKORO_MODEL_*` consts); #191 adds the Orpheus GGUF entry.

use std::path::{Path, PathBuf};

use crate::LensError;
use crate::tts::DownloadProgress;
use crate::tts::kokoro::{KOKORO_MODEL_RELPATH, KOKORO_MODEL_SHA256_HEX, KOKORO_MODEL_URL};

/// Static description of one supported TTS model. Same shape as
/// [`WhisperModelSpec`](crate::asr::WhisperModelSpec).
pub struct TtsModelSpec {
    /// Short id used as the registry key (e.g. `"kokoro"`).
    pub id: &'static str,
    /// Pinned resolve URL for the single-file model binary.
    pub url: &'static str,
    /// SHA256 from the HF LFS `oid` pointer — verified before the `.part → final` rename.
    pub sha256: &'static str,
    /// On-disk path relative to `data_dir`.
    pub relpath: &'static str,
}

/// All TTS models the seam knows how to fetch. The Kokoro entry is vestigial
/// (removed with the rest of Kokoro in #192 once #191's Orpheus adapter ships).
pub static TTS_REGISTRY: &[TtsModelSpec] = &[TtsModelSpec {
    id: "kokoro",
    url: KOKORO_MODEL_URL,
    sha256: KOKORO_MODEL_SHA256_HEX,
    relpath: KOKORO_MODEL_RELPATH,
}];

/// Resolves a model id to its [`TtsModelSpec`]. Returns `None` for unknown ids;
/// this allowlist is what makes the path helpers immune to `..` traversal.
pub fn resolve_tts(id: &str) -> Option<&'static TtsModelSpec> {
    TTS_REGISTRY.iter().find(|s| s.id == id)
}

/// On-disk path for a registered model: `{data_dir}/{spec.relpath}`. Returns
/// `None` for an unknown id, so a caller can never build a path from an
/// unvalidated (possibly traversing) id.
pub fn tts_model_path(data_dir: &Path, id: &str) -> Option<PathBuf> {
    resolve_tts(id).map(|spec| data_dir.join(spec.relpath))
}

/// Whether the resolved model's file exists on disk. Unknown/`..`-containing ids
/// resolve to `None` → `false`, so the probed path always stays under `data_dir`.
pub fn tts_model_downloaded(data_dir: &Path, id: &str) -> bool {
    match tts_model_path(data_dir, id) {
        Some(path) => path.is_file(),
        None => false,
    }
}

/// Downloads the requested TTS model with streaming progress and SHA256
/// verification. Unknown id → `LensError::Validation`; otherwise delegates to the
/// shared `download_verified` helper (`.part` → verify → atomic rename) and
/// returns the final path. Idempotent: skips a correctly-sized existing file.
pub async fn download_tts_model<F>(
    data_dir: &Path,
    id: &str,
    on_progress: F,
) -> Result<PathBuf, LensError>
where
    F: FnMut(DownloadProgress),
{
    let spec = resolve_tts(id)
        .ok_or_else(|| LensError::Validation(format!("unknown TTS model id: {id:?}")))?;
    let dest = data_dir.join(spec.relpath);
    crate::download::download_verified(spec.url, &dest, Some(spec.sha256), on_progress).await?;
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn sha256_hex(bytes: &[u8]) -> String {
        crate::hex_encode(&Sha256::digest(bytes))
    }

    #[test]
    fn resolve_known_kokoro() {
        let spec = resolve_tts("kokoro").expect("kokoro must be registered");
        assert_eq!(spec.id, "kokoro");
        assert!(!spec.url.is_empty());
        assert_eq!(spec.sha256.len(), 64);
        assert!(spec.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
        assert!(spec.relpath.ends_with("model_q8f16.onnx"));
    }

    #[test]
    fn resolve_unknown_returns_none() {
        assert!(resolve_tts("does-not-exist").is_none());
        assert!(resolve_tts("").is_none());
    }

    #[test]
    fn path_joins_under_data_dir() {
        let p = tts_model_path(Path::new("/data"), "kokoro").expect("known id resolves");
        assert!(p.ends_with("models/kokoro/model_q8f16.onnx"));
    }

    #[test]
    fn rejects_traversal_id() {
        let dir = tempfile::tempdir().unwrap();
        for bad in ["../../etc/passwd", "..", "kokoro/../../secret", ""] {
            assert!(
                resolve_tts(bad).is_none(),
                "traversal id {bad:?} must not resolve"
            );
            assert!(tts_model_path(dir.path(), bad).is_none());
            assert!(!tts_model_downloaded(dir.path(), bad));
        }
    }

    #[test]
    fn downloaded_false_when_absent_true_when_present() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!tts_model_downloaded(dir.path(), "kokoro"));
        let path = tts_model_path(dir.path(), "kokoro").unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"fake").unwrap();
        assert!(tts_model_downloaded(dir.path(), "kokoro"));
    }

    #[tokio::test]
    async fn download_unknown_id_is_validation_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = download_tts_model(dir.path(), "nope", |_| {})
            .await
            .unwrap_err();
        assert!(matches!(err, LensError::Validation(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn download_writes_file_and_emits_progress() {
        let body = vec![9u8; 1024];
        let expected = sha256_hex(&body);
        let server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-length", "1024"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .mount(&server)
            .await;

        // The registry URL is a hardcoded HF endpoint; exercise the shared
        // downloader against the mock (mirrors the asr registry test).
        let dir = tempfile::tempdir().unwrap();
        let dest = tts_model_path(dir.path(), "kokoro").unwrap();
        let mut events = Vec::new();
        crate::download::download_verified(&server.uri(), &dest, Some(&expected), |p| {
            events.push(p)
        })
        .await
        .unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert!(events.last().unwrap().done);
        assert!(events.iter().any(|e| !e.done));
        assert!(!dest.with_extension("part").exists());
    }
}
