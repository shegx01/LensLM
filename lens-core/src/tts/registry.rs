use std::path::{Path, PathBuf};

use crate::LensError;
use crate::tts::DownloadProgress;
use crate::tts::orpheus::{
    ORPHEUS_MODEL_ID, ORPHEUS_MODEL_RELPATH, ORPHEUS_MODEL_SHA256_HEX, ORPHEUS_MODEL_URL,
};
use crate::tts::snac::{SNAC_MODEL_ID, SNAC_MODEL_RELPATH, SNAC_MODEL_SHA256_HEX, SNAC_MODEL_URL};

pub struct TtsModelSpec {
    pub id: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub relpath: &'static str,
}

pub static TTS_REGISTRY: &[TtsModelSpec] = &[
    // issue #191 [161c]: SNAC 24 kHz neural-codec decoder weights (upstream
    // PyTorch `.bin`; load mechanism documented at the snac.rs call site).
    TtsModelSpec {
        id: SNAC_MODEL_ID,
        url: SNAC_MODEL_URL,
        sha256: SNAC_MODEL_SHA256_HEX,
        relpath: SNAC_MODEL_RELPATH,
    },
    // issue #191 [161c]: Orpheus-3B Q4_K_M GGUF (llama.cpp) — emits SNAC audio
    // tokens. Paired with the SNAC decoder above; both required for the backend.
    TtsModelSpec {
        id: ORPHEUS_MODEL_ID,
        url: ORPHEUS_MODEL_URL,
        sha256: ORPHEUS_MODEL_SHA256_HEX,
        relpath: ORPHEUS_MODEL_RELPATH,
    },
];

pub fn resolve_tts(id: &str) -> Option<&'static TtsModelSpec> {
    TTS_REGISTRY.iter().find(|s| s.id == id)
}

pub fn tts_model_path(data_dir: &Path, id: &str) -> Option<PathBuf> {
    resolve_tts(id).map(|spec| data_dir.join(spec.relpath))
}

pub fn tts_model_downloaded(data_dir: &Path, id: &str) -> bool {
    match tts_model_path(data_dir, id) {
        Some(path) => path.is_file(),
        None => false,
    }
}

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
    fn resolve_known_snac() {
        let spec = resolve_tts("snac").expect("snac must be registered");
        assert_eq!(spec.id, "snac");
        assert!(spec.url.starts_with("https://") && spec.url.contains("snac_24khz"));
        assert_eq!(spec.sha256.len(), 64);
        assert!(spec.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_eq!(spec.relpath, "models/snac/pytorch_model.bin");
    }

    #[test]
    fn resolve_known_orpheus() {
        let spec = resolve_tts("orpheus").expect("orpheus must be registered");
        assert_eq!(spec.id, "orpheus");
        assert!(spec.url.starts_with("https://") && spec.url.ends_with(".gguf"));
        assert_eq!(spec.sha256.len(), 64);
        assert!(spec.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_eq!(spec.relpath, "models/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf");
    }

    #[test]
    fn resolve_unknown_returns_none() {
        assert!(resolve_tts("does-not-exist").is_none());
        assert!(resolve_tts("").is_none());
    }

    #[test]
    fn path_joins_under_data_dir() {
        let p = tts_model_path(Path::new("/data"), "orpheus").expect("known id resolves");
        assert!(p.ends_with("models/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf"));
    }

    #[test]
    fn rejects_traversal_id() {
        let dir = tempfile::tempdir().unwrap();
        for bad in ["../../etc/passwd", "..", "orpheus/../../secret", ""] {
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
        assert!(!tts_model_downloaded(dir.path(), "orpheus"));
        let path = tts_model_path(dir.path(), "orpheus").unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"fake").unwrap();
        assert!(tts_model_downloaded(dir.path(), "orpheus"));
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

        let dir = tempfile::tempdir().unwrap();
        let dest = tts_model_path(dir.path(), "orpheus").unwrap();
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
