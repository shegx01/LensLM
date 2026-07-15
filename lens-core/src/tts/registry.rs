use std::path::{Path, PathBuf};

use crate::LensError;
use crate::tts::DownloadProgress;
use crate::tts::orpheus::{
    ORPHEUS_MODEL_ID, ORPHEUS_MODEL_RELPATH, ORPHEUS_MODEL_SHA256_HEX, ORPHEUS_MODEL_URL,
};
use crate::tts::snac::{SNAC_MODEL_ID, SNAC_MODEL_RELPATH, SNAC_MODEL_SHA256_HEX, SNAC_MODEL_URL};

/// How a downloaded artifact materializes on disk. `File` lands at `relpath` as-is
/// (SNAC/Orpheus). `Archive` downloads a `.zip` to `relpath`, then unpacks it via
/// an atomic rename; availability is keyed on a post-unpack `sentinel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    File,
    Archive,
}

pub struct TtsModelSpec {
    pub id: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    /// `download_verified` destination (a file path, incl. the `.zip` for archives).
    pub relpath: &'static str,
    pub kind: ArtifactKind,
    /// Post-unpack availability key for `Archive` (a file that exists only after a
    /// complete unpack). `None` for `File`, whose availability is `relpath` itself.
    pub sentinel: Option<&'static str>,
}

pub const MOSS_SIDECAR_BIN_ID: &str = "moss_sidecar_bin";
pub const MOSS_MODEL_ID: &str = "moss_model";

// TODO(#193 A0/A3): placeholder URL+SHA. The real GitHub Releases URL and the
// SHA-256 over the signed+notarized+stapled binary come from the human freeze
// step; the model URL+SHA come from the published Hugging Face `.zip`.
const MOSS_SIDECAR_BIN_URL: &str =
    "https://example.invalid/PLACEHOLDER/mlx-speech-sidecar-macos-aarch64";
const MOSS_SIDECAR_BIN_SHA256_HEX: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";
const MOSS_SIDECAR_BIN_RELPATH: &str = "bin/mlx-speech-sidecar";

const MOSS_MODEL_URL: &str = "https://example.invalid/PLACEHOLDER/moss-tts-local-int8.zip";
const MOSS_MODEL_SHA256_HEX: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";
const MOSS_MODEL_RELPATH: &str = "models/moss.zip";
const MOSS_MODEL_SENTINEL: &str = "models/moss/config.json";

pub static TTS_REGISTRY: &[TtsModelSpec] = &[
    // issue #191 [161c]: SNAC 24 kHz neural-codec decoder weights (upstream
    // PyTorch `.bin`; load mechanism documented at the snac.rs call site).
    TtsModelSpec {
        id: SNAC_MODEL_ID,
        url: SNAC_MODEL_URL,
        sha256: SNAC_MODEL_SHA256_HEX,
        relpath: SNAC_MODEL_RELPATH,
        kind: ArtifactKind::File,
        sentinel: None,
    },
    // issue #191 [161c]: Orpheus-3B Q4_K_M GGUF (llama.cpp) — emits SNAC audio
    // tokens. Paired with the SNAC decoder above; both required for the backend.
    TtsModelSpec {
        id: ORPHEUS_MODEL_ID,
        url: ORPHEUS_MODEL_URL,
        sha256: ORPHEUS_MODEL_SHA256_HEX,
        relpath: ORPHEUS_MODEL_RELPATH,
        kind: ArtifactKind::File,
        sentinel: None,
    },
    // issue #193 [161e]: frozen MLX `mlx-speech` sidecar binary (Apple Silicon).
    // Marked executable after download; consumed out-of-process by src-tauri.
    TtsModelSpec {
        id: MOSS_SIDECAR_BIN_ID,
        url: MOSS_SIDECAR_BIN_URL,
        sha256: MOSS_SIDECAR_BIN_SHA256_HEX,
        relpath: MOSS_SIDECAR_BIN_RELPATH,
        kind: ArtifactKind::File,
        sentinel: None,
    },
    // issue #193 [161e]: MOSS-TTS-Local int8 weights, shipped as a single `.zip`
    // (Store/Deflate only) and unpacked in-tree. Availability = the sentinel below.
    TtsModelSpec {
        id: MOSS_MODEL_ID,
        url: MOSS_MODEL_URL,
        sha256: MOSS_MODEL_SHA256_HEX,
        relpath: MOSS_MODEL_RELPATH,
        kind: ArtifactKind::Archive,
        sentinel: Some(MOSS_MODEL_SENTINEL),
    },
];

pub fn resolve_tts(id: &str) -> Option<&'static TtsModelSpec> {
    TTS_REGISTRY.iter().find(|s| s.id == id)
}

pub fn tts_model_path(data_dir: &Path, id: &str) -> Option<PathBuf> {
    resolve_tts(id).map(|spec| data_dir.join(spec.relpath))
}

pub fn tts_model_downloaded(data_dir: &Path, id: &str) -> bool {
    match resolve_tts(id) {
        // For archives the download destination (`relpath`) is a transient `.zip`;
        // availability is the post-unpack sentinel.
        Some(spec) => data_dir.join(spec.sentinel.unwrap_or(spec.relpath)).is_file(),
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
    match spec.kind {
        ArtifactKind::File => download_file_artifact(data_dir, spec, on_progress).await,
        ArtifactKind::Archive => download_archive_artifact(data_dir, spec, on_progress).await,
    }
}

async fn download_file_artifact<F>(
    data_dir: &Path,
    spec: &TtsModelSpec,
    on_progress: F,
) -> Result<PathBuf, LensError>
where
    F: FnMut(DownloadProgress),
{
    let dest = data_dir.join(spec.relpath);
    crate::download::download_verified(spec.url, &dest, Some(spec.sha256), on_progress).await?;
    // The sidecar ships without an exec bit over HTTP; make it runnable.
    #[cfg(unix)]
    if spec.id == MOSS_SIDECAR_BIN_ID {
        set_executable(&dest)?;
    }
    Ok(dest)
}

async fn download_archive_artifact<F>(
    data_dir: &Path,
    spec: &TtsModelSpec,
    on_progress: F,
) -> Result<PathBuf, LensError>
where
    F: FnMut(DownloadProgress),
{
    let zip_dest = data_dir.join(spec.relpath);
    let final_dir = zip_dest.with_extension("");
    let partial_dir = final_dir.with_extension("partial");
    let sentinel = data_dir.join(
        spec.sentinel
            .ok_or_else(|| LensError::Internal("archive spec missing sentinel".into()))?,
    );

    let mut on_progress = on_progress;

    // Idempotency: a re-enable must never re-pull the multi-GB archive.
    if sentinel.is_file() {
        on_progress(DownloadProgress {
            received: 0,
            total: None,
            done: true,
        });
        return Ok(final_dir);
    }

    // `download_verified` emits its terminal `done:true` when the zip *bytes* land —
    // before unpack. Suppress it here; the unpack is a distinct terminal phase and
    // the synthetic `done:true` is emitted only after the atomic rename succeeds.
    let mut last = DownloadProgress {
        received: 0,
        total: None,
        done: false,
    };
    {
        let mut wrapped = |p: DownloadProgress| {
            if !p.done {
                last = p;
                on_progress(p);
            }
        };
        crate::download::download_verified(spec.url, &zip_dest, Some(spec.sha256), &mut wrapped)
            .await?;
    }

    // A crashed prior unpack must not poison the rename source.
    if partial_dir.exists() {
        std::fs::remove_dir_all(&partial_dir)
            .map_err(|e| LensError::Io(format!("clear partial unpack dir: {e}")))?;
    }
    std::fs::create_dir_all(&partial_dir)
        .map_err(|e| LensError::Io(format!("create partial unpack dir: {e}")))?;

    unpack_zip(&zip_dest, &partial_dir)?;

    if final_dir.exists() {
        std::fs::remove_dir_all(&final_dir)
            .map_err(|e| LensError::Io(format!("clear stale model dir: {e}")))?;
    }
    std::fs::rename(&partial_dir, &final_dir)
        .map_err(|e| LensError::Io(format!("finalize model unpack: {e}")))?;
    let _ = std::fs::remove_file(&zip_dest);

    on_progress(DownloadProgress {
        done: true,
        ..last
    });
    Ok(final_dir)
}

/// Unpacks a Store/Deflate `.zip` into `dest_dir`. Zip-slip guard: any entry whose
/// normalized path is not safely contained (`..`, absolute) is rejected via
/// `enclosed_name()` → [`LensError::Validation`].
pub fn unpack_zip(zip_path: &Path, dest_dir: &Path) -> Result<(), LensError> {
    let file = std::fs::File::open(zip_path)
        .map_err(|e| LensError::Io(format!("open archive: {e}")))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| LensError::Validation(format!("invalid zip archive: {e}")))?;
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| LensError::Validation(format!("zip entry {i} unreadable: {e}")))?;
        let out_path = match entry.enclosed_name() {
            Some(rel) => dest_dir.join(rel),
            None => {
                return Err(LensError::Validation(
                    "zip entry has an unsafe path (traversal or absolute)".into(),
                ));
            }
        };
        if entry.is_dir() {
            std::fs::create_dir_all(&out_path)
                .map_err(|e| LensError::Io(format!("create dir in unpack: {e}")))?;
        } else {
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| LensError::Io(format!("create parent in unpack: {e}")))?;
            }
            let mut out = std::fs::File::create(&out_path)
                .map_err(|e| LensError::Io(format!("create file in unpack: {e}")))?;
            std::io::copy(&mut entry, &mut out)
                .map_err(|e| LensError::Io(format!("write file in unpack: {e}")))?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn set_executable(path: &Path) -> Result<(), LensError> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
        .map_err(|e| LensError::Io(format!("set executable bit: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::io::Write;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};
    use zip::write::FileOptions;

    fn sha256_hex(bytes: &[u8]) -> String {
        crate::hex_encode(&Sha256::digest(bytes))
    }

    fn write_zip(path: &Path, entries: &[(&str, &[u8])], dirs: &[&str]) {
        let file = std::fs::File::create(path).unwrap();
        let mut zw = zip::ZipWriter::new(file);
        let opts = FileOptions::default().compression_method(zip::CompressionMethod::Stored);
        for d in dirs {
            zw.add_directory(*d, opts).unwrap();
        }
        for (name, body) in entries {
            zw.start_file(*name, opts).unwrap();
            zw.write_all(body).unwrap();
        }
        zw.finish().unwrap();
    }

    #[test]
    fn resolve_known_snac() {
        let spec = resolve_tts("snac").expect("snac must be registered");
        assert_eq!(spec.id, "snac");
        assert!(spec.url.starts_with("https://") && spec.url.contains("snac_24khz"));
        assert_eq!(spec.sha256.len(), 64);
        assert!(spec.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_eq!(spec.relpath, "models/snac/pytorch_model.bin");
        assert_eq!(spec.kind, ArtifactKind::File);
        assert!(spec.sentinel.is_none());
    }

    #[test]
    fn resolve_known_orpheus() {
        let spec = resolve_tts("orpheus").expect("orpheus must be registered");
        assert_eq!(spec.id, "orpheus");
        assert!(spec.url.starts_with("https://") && spec.url.ends_with(".gguf"));
        assert_eq!(spec.sha256.len(), 64);
        assert!(spec.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_eq!(spec.relpath, "models/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf");
        assert_eq!(spec.kind, ArtifactKind::File);
    }

    #[test]
    fn resolve_known_moss_specs() {
        let bin = resolve_tts("moss_sidecar_bin").expect("moss binary must be registered");
        assert_eq!(bin.kind, ArtifactKind::File);
        assert_eq!(bin.relpath, "bin/mlx-speech-sidecar");
        assert!(bin.sentinel.is_none());
        assert_eq!(bin.sha256.len(), 64);

        let model = resolve_tts("moss_model").expect("moss model must be registered");
        assert_eq!(model.kind, ArtifactKind::Archive);
        assert_eq!(model.relpath, "models/moss.zip");
        assert_eq!(model.sentinel, Some("models/moss/config.json"));
        assert_eq!(model.sha256.len(), 64);
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

    #[test]
    fn archive_downloaded_keys_on_sentinel_not_zip() {
        let dir = tempfile::tempdir().unwrap();
        // The transient `.zip` at `relpath` must NOT report the model available.
        let zip = dir.path().join("models/moss.zip");
        std::fs::create_dir_all(zip.parent().unwrap()).unwrap();
        std::fs::write(&zip, b"zip").unwrap();
        assert!(!tts_model_downloaded(dir.path(), "moss_model"));
        // Only the post-unpack sentinel counts.
        let sentinel = dir.path().join("models/moss/config.json");
        std::fs::create_dir_all(sentinel.parent().unwrap()).unwrap();
        std::fs::write(&sentinel, b"{}").unwrap();
        assert!(tts_model_downloaded(dir.path(), "moss_model"));
    }

    #[test]
    fn unpack_zip_round_trips_files_and_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let zip = dir.path().join("m.zip");
        write_zip(
            &zip,
            &[
                ("config.json", b"{\"k\":1}"),
                ("weights/model.bin", &[1u8, 2, 3, 4]),
            ],
            &["weights/"],
        );
        let dest = dir.path().join("out");
        unpack_zip(&zip, &dest).unwrap();
        assert_eq!(
            std::fs::read(dest.join("config.json")).unwrap(),
            b"{\"k\":1}"
        );
        assert_eq!(
            std::fs::read(dest.join("weights/model.bin")).unwrap(),
            vec![1u8, 2, 3, 4]
        );
    }

    #[test]
    fn unpack_zip_rejects_path_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let zip = dir.path().join("evil.zip");
        write_zip(&zip, &[("../escape.txt", b"pwned")], &[]);
        let dest = dir.path().join("out");
        let err = unpack_zip(&zip, &dest).unwrap_err();
        assert!(matches!(err, LensError::Validation(_)), "got {err:?}");
        // Nothing escaped the destination.
        assert!(!dir.path().join("escape.txt").exists());
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
    async fn download_archive_unpacks_and_emits_done_after_rename() {
        let dir = tempfile::tempdir().unwrap();
        // Build a zip whose root holds `config.json`; unpacked under `models/moss/`
        // it becomes the sentinel `models/moss/config.json`.
        let src_zip = dir.path().join("src.zip");
        write_zip(&src_zip, &[("config.json", b"{}")], &[]);
        let body = std::fs::read(&src_zip).unwrap();
        let expected = sha256_hex(&body);

        let server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-length", body.len().to_string().as_str()),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .mount(&server)
            .await;

        // Drive the archive path directly with a server-backed spec.
        let spec = TtsModelSpec {
            id: "moss_model",
            url: Box::leak(server.uri().into_boxed_str()),
            sha256: Box::leak(expected.into_boxed_str()),
            relpath: "models/moss.zip",
            kind: ArtifactKind::Archive,
            sentinel: Some("models/moss/config.json"),
        };
        let mut events = Vec::new();
        let out = download_archive_artifact(dir.path(), &spec, |p| events.push(p))
            .await
            .unwrap();

        assert_eq!(out, dir.path().join("models/moss"));
        assert!(tts_model_downloaded(dir.path(), "moss_model"));
        // The transient zip is removed after a successful unpack.
        assert!(!dir.path().join("models/moss.zip").exists());
        assert!(!dir.path().join("models/moss.partial").exists());
        // Exactly one terminal `done` and it is the last event.
        assert!(events.last().unwrap().done);
        assert_eq!(events.iter().filter(|e| e.done).count(), 1);

        // Idempotent re-enable: sentinel present → no re-download, immediate done.
        let mut again = Vec::new();
        download_archive_artifact(dir.path(), &spec, |p| again.push(p))
            .await
            .unwrap();
        assert_eq!(again.len(), 1);
        assert!(again[0].done);
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
