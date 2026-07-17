//! One-shot Qwen model "prepare" (#194): explicit progress-streamed download of
//! the MLX model plus an on-disk presence check. Kept separate from the persistent
//! synth protocol in `mod.rs` so it can't perturb `run_turn_locked`/`drain_until`.

use std::path::Path;
use std::process::Stdio;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::Command;
use tokio::time::{Duration, timeout};

use lens_core::{DownloadProgress, LensError};

use super::{MAX_REPLY_BYTES, SpawnResolver, tts_err};

/// huggingface_hub's on-disk cache subdir for the Qwen model (under `<hf>/hub`).
/// Kept in lockstep with the sidecar's `CACHE_SUBDIR`.
const QWEN_SNAPSHOT_DIR: &str = "models--mlx-community--Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16";

/// Ceiling on the whole prepare download; matches the serve handshake budget
/// (~4.5 GB over a modest link).
const PREPARE_TIMEOUT: Duration = Duration::from_secs(1800);

#[derive(serde::Deserialize)]
struct PrepareLine {
    progress: Option<PrepareProgress>,
    #[serde(default)]
    done: bool,
    error: Option<String>,
}

#[derive(serde::Deserialize)]
struct PrepareProgress {
    received: u64,
    total: Option<u64>,
}

/// Drives the one-shot `--prepare` sidecar, forwarding streamed byte progress to
/// `on_progress` then a terminal `done:true`. Reuses the serve resolver, appending
/// `--prepare`; every failure maps to a detail-free [`LensError::Tts`].
pub async fn run_prepare<F>(resolver: SpawnResolver, mut on_progress: F) -> Result<(), LensError>
where
    F: FnMut(DownloadProgress) + Send,
{
    // Resolve off the async runtime (first resolution may download `uv`), then
    // append the mode flag so the serve resolution stays byte-for-byte untouched.
    let spec = tokio::task::spawn_blocking(move || {
        resolver().map(|mut s| {
            s.args.push("--prepare".to_string());
            s
        })
    })
    .await
    .map_err(|e| {
        tracing::warn!(error = %e, "Qwen prepare resolver task panicked/cancelled");
        tts_err()
    })??;

    let mut child = Command::new(&spec.program)
        .args(&spec.args)
        .envs(spec.envs)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        // Sidecar diagnostics (and tqdm bars) go to our stderr — never the pipe.
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| {
            tracing::warn!(error = %e, "failed to spawn Qwen prepare sidecar");
            tts_err()
        })?;

    let stdout = child.stdout.take().ok_or_else(tts_err)?;
    let mut reader = BufReader::new(stdout);

    let outcome = timeout(PREPARE_TIMEOUT, async {
        let mut last = DownloadProgress {
            received: 0,
            total: None,
            done: false,
        };
        let mut line = String::new();
        loop {
            line.clear();
            // Cap each line so a never-newline child can't OOM the host.
            let n = (&mut reader)
                .take(MAX_REPLY_BYTES)
                .read_line(&mut line)
                .await
                .map_err(|e| {
                    tracing::warn!(error = %e, "Qwen prepare read failed");
                    tts_err()
                })?;
            if n == 0 {
                tracing::warn!("Qwen prepare closed stdout before done (EOF)");
                return Err(tts_err());
            }
            if !line.ends_with('\n') {
                tracing::warn!(bytes = n, "Qwen prepare line exceeded cap or truncated");
                return Err(tts_err());
            }
            let Ok(msg) = serde_json::from_str::<PrepareLine>(line.trim()) else {
                // Non-JSON stdout (a stray sidecar log) — ignore and keep reading.
                continue;
            };
            if let Some(err) = msg.error {
                tracing::warn!(error = %err, "Qwen prepare reported error");
                return Err(tts_err());
            }
            if let Some(p) = msg.progress {
                last = DownloadProgress {
                    received: p.received,
                    total: p.total,
                    done: false,
                };
                on_progress(last);
            }
            if msg.done {
                return Ok(last);
            }
        }
    })
    .await;

    let last = match outcome {
        Ok(Ok(last)) => last,
        Ok(Err(e)) => {
            let _ = child.start_kill();
            return Err(e);
        }
        Err(_) => {
            tracing::warn!("Qwen prepare timed out");
            let _ = child.start_kill();
            return Err(tts_err());
        }
    };

    // Require a clean exit before declaring success (and reap the child).
    match child.wait().await {
        Ok(status) if status.success() => {}
        Ok(_) => {
            tracing::warn!("Qwen prepare exited non-zero after reporting done");
            return Err(tts_err());
        }
        Err(e) => {
            tracing::warn!(error = %e, "failed to reap Qwen prepare child");
            return Err(tts_err());
        }
    }

    on_progress(DownloadProgress {
        received: last.received,
        total: last.total,
        done: true,
    });
    Ok(())
}

/// Whether the Qwen model snapshot is present AND complete under `hf_cache_dir`
/// (which the resolver points `HF_HOME` at): a revision must resolve BOTH
/// `config.json` and a `*.safetensors` weight through its symlinks, with no
/// `blobs/*.incomplete` partial (huggingface_hub leaves those mid-download).
/// Assumes the plain-HTTPS cache layout forced by `HF_HUB_DISABLE_XET=1`.
pub fn qwen_snapshot_present(hf_cache_dir: &Path) -> bool {
    let model_dir = hf_cache_dir.join("hub").join(QWEN_SNAPSHOT_DIR);
    if has_incomplete_blob(&model_dir.join("blobs")) {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(model_dir.join("snapshots")) else {
        return false;
    };
    for entry in entries.flatten() {
        let rev = entry.path();
        // Require BOTH the config sentinel and a weight file: an interrupt between
        // files (config symlinked, weights not yet started → no `.incomplete`)
        // must read as absent, not silently re-fetch multi-GB at first synth.
        // `exists()` follows each symlink into `blobs/`, so a dangling link (blob
        // not yet downloaded) correctly reads as absent.
        if rev.is_dir() && rev.join("config.json").exists() && has_resolvable_safetensors(&rev) {
            return true;
        }
    }
    false
}

/// Whether the revision dir holds at least one `*.safetensors` file that resolves
/// through its symlink — covers both single-file `model.safetensors` and a sharded
/// `model-0000N-of-...safetensors` layout.
fn has_resolvable_safetensors(rev: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(rev) else {
        return false;
    };
    entries.flatten().any(|e| {
        let p = e.path();
        e.file_name()
            .to_str()
            .is_some_and(|n| n.ends_with(".safetensors"))
            && p.exists()
    })
}

fn has_incomplete_blob(blobs: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(blobs) else {
        return false;
    };
    entries.flatten().any(|e| {
        e.file_name()
            .to_str()
            .is_some_and(|n| n.ends_with(".incomplete"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen::{SidecarSpawn, SpawnResolver};
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    /// Writes `body` to `dir/stub.py` and returns its path. Run as an argument to
    /// the system `python3`, so no execute bit is needed.
    fn write_stub(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("stub.py");
        std::fs::write(&path, body).unwrap();
        path
    }

    fn resolver_for(stub: PathBuf) -> SpawnResolver {
        Arc::new(move || {
            Ok(SidecarSpawn {
                program: PathBuf::from("python3"),
                args: vec![stub.to_string_lossy().into_owned()],
                envs: vec![],
            })
        })
    }

    #[tokio::test]
    async fn run_prepare_streams_progress_then_done() {
        let tmp = tempfile::tempdir().unwrap();
        // The resolver appends `--prepare`; a real sidecar keys off it, the stub
        // just emits the contract lines and exits 0.
        let stub = write_stub(
            tmp.path(),
            concat!(
                "import sys, json\n",
                "def e(o):\n",
                "    sys.stdout.write(json.dumps(o)+'\\n'); sys.stdout.flush()\n",
                "assert '--prepare' in sys.argv[1:]\n",
                "e({'progress': {'received': 100, 'total': 1000}})\n",
                "e({'progress': {'received': 1000, 'total': 1000}})\n",
                "e({'done': True})\n",
            ),
        );
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        run_prepare(resolver_for(stub), move |p| sink.lock().unwrap().push(p))
            .await
            .expect("stub prepare succeeds");

        let got = events.lock().unwrap().clone();
        assert_eq!(got.len(), 3, "two progress + one done: {got:?}");
        assert_eq!(
            got[0],
            DownloadProgress {
                received: 100,
                total: Some(1000),
                done: false
            }
        );
        assert_eq!(
            got[1],
            DownloadProgress {
                received: 1000,
                total: Some(1000),
                done: false
            }
        );
        assert_eq!(
            got[2],
            DownloadProgress {
                received: 1000,
                total: Some(1000),
                done: true
            }
        );
    }

    #[tokio::test]
    async fn run_prepare_maps_sidecar_error_to_tts() {
        let tmp = tempfile::tempdir().unwrap();
        let stub = write_stub(
            tmp.path(),
            concat!(
                "import sys, json\n",
                "sys.stdout.write(json.dumps({'error': 'boom'})+'\\n'); sys.stdout.flush()\n",
                "sys.exit(1)\n",
            ),
        );
        let err = run_prepare(resolver_for(stub), |_| {})
            .await
            .expect_err("sidecar error surfaces");
        assert!(matches!(err, LensError::Tts(_)));
    }

    #[tokio::test]
    async fn run_prepare_eof_without_done_is_error() {
        let tmp = tempfile::tempdir().unwrap();
        // Emits progress, then exits 0 WITHOUT a done line — must not be success.
        let stub = write_stub(
            tmp.path(),
            concat!(
                "import sys, json\n",
                "sys.stdout.write(json.dumps({'progress': {'received': 5, 'total': 9}})+'\\n')\n",
                "sys.stdout.flush()\n",
            ),
        );
        let err = run_prepare(resolver_for(stub), |_| {})
            .await
            .expect_err("EOF before done is an error");
        assert!(matches!(err, LensError::Tts(_)));
    }

    #[tokio::test]
    async fn run_prepare_resolver_failure_is_tts() {
        let resolver: SpawnResolver = Arc::new(|| Err(tts_err()));
        let err = run_prepare(resolver, |_| {})
            .await
            .expect_err("resolver failure surfaces");
        assert!(matches!(err, LensError::Tts(_)));
    }

    /// Builds a complete HF snapshot layout under `hf`: a real (symlinked) blob
    /// and a `config.json` resolving through the snapshot dir.
    fn write_complete_snapshot(hf: &Path) {
        let model = hf.join("hub").join(QWEN_SNAPSHOT_DIR);
        let blobs = model.join("blobs");
        let rev = model.join("snapshots").join("abc123");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&rev).unwrap();
        let blob = blobs.join("deadbeef");
        std::fs::write(&blob, b"{}").unwrap();
        std::os::unix::fs::symlink(&blob, rev.join("config.json")).unwrap();
        let weight = blobs.join("cafef00d");
        std::fs::write(&weight, b"weights").unwrap();
        std::os::unix::fs::symlink(&weight, rev.join("model.safetensors")).unwrap();
    }

    #[test]
    fn snapshot_present_true_for_complete_layout() {
        let tmp = tempfile::tempdir().unwrap();
        write_complete_snapshot(tmp.path());
        assert!(qwen_snapshot_present(tmp.path()));
    }

    #[test]
    fn snapshot_present_false_when_incomplete_blob_remains() {
        let tmp = tempfile::tempdir().unwrap();
        write_complete_snapshot(tmp.path());
        // A partial blob from an interrupted pull invalidates the cache.
        let blobs = tmp.path().join("hub").join(QWEN_SNAPSHOT_DIR).join("blobs");
        std::fs::write(blobs.join("deadbeef.incomplete"), b"partial").unwrap();
        assert!(!qwen_snapshot_present(tmp.path()));
    }

    #[test]
    fn snapshot_present_false_when_weights_missing() {
        // The exact between-files gap: config.json resolves + NO `*.incomplete`,
        // but the weight artifact was never fetched → must NOT report ready.
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("hub").join(QWEN_SNAPSHOT_DIR);
        let blobs = model.join("blobs");
        let rev = model.join("snapshots").join("abc123");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&rev).unwrap();
        let blob = blobs.join("deadbeef");
        std::fs::write(&blob, b"{}").unwrap();
        std::os::unix::fs::symlink(&blob, rev.join("config.json")).unwrap();
        assert!(!qwen_snapshot_present(tmp.path()));
    }

    #[test]
    fn snapshot_present_false_when_config_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let rev = tmp
            .path()
            .join("hub")
            .join(QWEN_SNAPSHOT_DIR)
            .join("snapshots")
            .join("abc123");
        std::fs::create_dir_all(&rev).unwrap();
        assert!(!qwen_snapshot_present(tmp.path()));
    }

    #[test]
    fn snapshot_present_false_for_dangling_config_symlink() {
        let tmp = tempfile::tempdir().unwrap();
        let rev = tmp
            .path()
            .join("hub")
            .join(QWEN_SNAPSHOT_DIR)
            .join("snapshots")
            .join("abc123");
        std::fs::create_dir_all(&rev).unwrap();
        // Points at a blob that was never downloaded — `exists()` must read false.
        std::os::unix::fs::symlink(tmp.path().join("nope"), rev.join("config.json")).unwrap();
        assert!(!qwen_snapshot_present(tmp.path()));
    }

    #[test]
    fn snapshot_present_false_for_empty_cache() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!qwen_snapshot_present(tmp.path()));
    }
}
