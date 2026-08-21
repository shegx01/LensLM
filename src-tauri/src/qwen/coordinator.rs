//! Single-flight + cancel coordination for the Qwen `--prepare` download (#202):
//! serializes concurrent prepares to one download (second caller re-checks the on-disk
//! snapshot after the gate) and holds the cancel slot. Bridge-crate only, like the sidecar.

use std::path::Path;
use std::sync::{Arc, Mutex};

use tokio::sync::Mutex as AsyncMutex;
use tokio_util::sync::CancellationToken;

use lens_core::{DownloadProgress, LensError, available_space_bytes, path_size_bytes};

use super::prepare::QWEN_SNAPSHOT_DIR;
use super::{SidecarPaths, SpawnResolver, qwen_snapshot_present, run_prepare};

/// The `~4.5 GB` snapshot size `system.rs` documents; no fetcher here exposes a
/// real `Content-Length` to probe instead.
const QWEN_SNAPSHOT_APPROX_BYTES: u64 = 4_500_000_000;

const DISK_HEADROOM_BYTES: u64 = 256 * 1024 * 1024;

pub struct QwenPrepareCoordinator {
    /// Serializes prepares: a second caller blocks here until the first finishes.
    gate: AsyncMutex<()>,
    /// The in-flight prepare's cancel token (single slot), owner-matched on clear.
    current: Mutex<Option<Arc<CancellationToken>>>,
}

impl Default for QwenPrepareCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl QwenPrepareCoordinator {
    pub fn new() -> Self {
        Self {
            gate: AsyncMutex::new(()),
            current: Mutex::new(None),
        }
    }

    /// Runs a single-flight `--prepare`: serializes on the gate, re-checks the on-disk
    /// snapshot after acquiring (a just-finished predecessor short-circuits, no redundant
    /// download), then drives [`run_prepare`] under a cancel token in the single slot.
    pub async fn run_single_flight<F>(
        &self,
        paths: &SidecarPaths,
        resolver: SpawnResolver,
        on_progress: F,
    ) -> Result<(), LensError>
    where
        F: FnMut(DownloadProgress) + Send,
    {
        self.run_single_flight_with_probe(paths, resolver, on_progress, available_space_bytes)
            .await
    }

    async fn run_single_flight_with_probe<F, P>(
        &self,
        paths: &SidecarPaths,
        resolver: SpawnResolver,
        on_progress: F,
        probe: P,
    ) -> Result<(), LensError>
    where
        F: FnMut(DownloadProgress) + Send,
        P: Fn(&Path) -> Option<u64> + Send + Sync,
    {
        // Declared FIRST so it outlives the clear guard (reverse drop order): the
        // slot is cleared before the gate releases, so the next caller queued
        // behind us never observes a stale token.
        let _gate_guard = self.gate.lock().await;

        // Re-check after acquiring: the caller we queued behind may have just
        // completed the download, so there is nothing left to do (AC-2).
        if qwen_snapshot_present(&paths.hf_cache_dir) {
            return Ok(());
        }

        // Ordering is the whole safety argument: the short-circuit above is what
        // keeps a fully-cached snapshot from ever reaching this guard.
        let snapshot_dir = paths.hf_cache_dir.join("hub").join(QWEN_SNAPSHOT_DIR);
        let on_disk = path_size_bytes(&snapshot_dir).unwrap_or(0);
        let required = QWEN_SNAPSHOT_APPROX_BYTES
            .saturating_sub(on_disk)
            .saturating_add(DISK_HEADROOM_BYTES);
        match probe(&paths.hf_cache_dir) {
            Some(available) if available < required => {
                return Err(LensError::InsufficientSpace(format!(
                    "the Qwen voice model needs {required} bytes of free space but only {available} bytes are available"
                )));
            }
            Some(_) => {}
            None => tracing::warn!(
                dir = %paths.hf_cache_dir.display(),
                "Qwen disk pre-flight skipped: the free-space probe failed"
            ),
        }

        let token = Arc::new(CancellationToken::new());
        {
            let mut cur = self.current.lock().unwrap_or_else(|e| e.into_inner());
            *cur = Some(token.clone());
        }
        let _clear_guard = ClearGuard {
            current: &self.current,
            owner: token.clone(),
        };

        run_prepare(resolver, token.as_ref().clone(), on_progress).await
    }

    /// Cancels the in-flight prepare (if any) by flipping its token. Returns `true`
    /// when one was in flight, `false` otherwise (mirrors `cancel_dialogue`).
    pub fn cancel(&self) -> bool {
        let cur = self.current.lock().unwrap_or_else(|e| e.into_inner());
        match cur.as_ref() {
            Some(token) => {
                token.cancel();
                true
            }
            None => false,
        }
    }
}

/// Clears the coordinator's cancel slot on scope exit, but only when the exiting
/// run is still the owner (`Arc::ptr_eq`) — so a superseded run's teardown never
/// evicts a newer run's token. Mirrors `remove_dialogue_if_owner`.
struct ClearGuard<'a> {
    current: &'a Mutex<Option<Arc<CancellationToken>>>,
    owner: Arc<CancellationToken>,
}

impl Drop for ClearGuard<'_> {
    fn drop(&mut self) {
        let mut cur = self.current.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(current) = cur.as_ref()
            && Arc::ptr_eq(current, &self.owner)
        {
            *cur = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen::{SidecarSpawn, SpawnResolver};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    fn paths_at(hf: &Path) -> SidecarPaths {
        SidecarPaths {
            app_data_dir: hf.to_path_buf(),
            sidecar_dir: hf.to_path_buf(),
            hf_cache_dir: hf.to_path_buf(),
        }
    }

    /// A resolver that runs `python3 <stub> <extra...>` and counts each invocation
    /// — one per real `run_prepare`, i.e. one per actual download attempt.
    fn counting_resolver(
        stub: PathBuf,
        extra: Vec<String>,
        count: Arc<AtomicUsize>,
    ) -> SpawnResolver {
        Arc::new(move || {
            count.fetch_add(1, Ordering::SeqCst);
            let mut args = vec![stub.to_string_lossy().into_owned()];
            args.extend(extra.clone());
            Ok(SidecarSpawn {
                program: PathBuf::from("python3"),
                args,
                envs: vec![],
            })
        })
    }

    fn write_stub(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("stub.py");
        std::fs::write(&path, body).unwrap();
        path
    }

    /// Writes a complete HF snapshot layout under argv[1] (hf cache dir) using the
    /// snapshot dir name in argv[2], then reports progress + done and exits 0.
    const SNAPSHOT_STUB: &str = r#"
import sys, json, os
hf = sys.argv[1]
dirname = sys.argv[2]
snap = os.path.join(hf, 'hub', dirname)
blobs = os.path.join(snap, 'blobs')
rev = os.path.join(snap, 'snapshots', 'abc123')
os.makedirs(blobs, exist_ok=True)
os.makedirs(rev, exist_ok=True)
cb = os.path.join(blobs, 'deadbeef')
open(cb, 'w').write('x')
try:
    os.symlink(cb, os.path.join(rev, 'config.json'))
except FileExistsError:
    pass
wb = os.path.join(blobs, 'cafef00d')
open(wb, 'w').write('w')
try:
    os.symlink(wb, os.path.join(rev, 'model.safetensors'))
except FileExistsError:
    pass
def e(o):
    sys.stdout.write(json.dumps(o) + '\n')
    sys.stdout.flush()
e({'progress': {'received': 100, 'total': 1000}})
e({'done': True})
"#;

    /// Never emits `done`; sleeps so the read loop blocks until a cancel arrives.
    const SLOW_STUB: &str = r#"
import time
time.sleep(30)
"#;

    /// Reports an error and exits non-zero.
    const FAILING_STUB: &str = r#"
import sys, json
sys.stdout.write(json.dumps({'error': 'boom'}) + '\n')
sys.stdout.flush()
sys.exit(1)
"#;

    #[tokio::test]
    async fn concurrent_prepares_download_once_when_snapshot_completes() {
        let tmp = tempfile::tempdir().unwrap();
        let hf = tmp.path().join("hf");
        std::fs::create_dir_all(&hf).unwrap();
        let stub = write_stub(tmp.path(), SNAPSHOT_STUB);
        let count = Arc::new(AtomicUsize::new(0));
        let extra = vec![
            hf.to_string_lossy().into_owned(),
            crate::qwen::prepare::QWEN_SNAPSHOT_DIR.to_string(),
        ];
        let coord = QwenPrepareCoordinator::new();
        let paths = paths_at(&hf);

        let (a, b) = tokio::join!(
            coord.run_single_flight(
                &paths,
                counting_resolver(stub.clone(), extra.clone(), count.clone()),
                |_| {},
            ),
            coord.run_single_flight(
                &paths,
                counting_resolver(stub.clone(), extra.clone(), count.clone()),
                |_| {},
            ),
        );
        a.expect("first prepare succeeds");
        b.expect("second prepare short-circuits on the present snapshot");
        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "single-flight: exactly one download"
        );
    }

    #[tokio::test]
    async fn cancel_flips_token_and_returns_cancelled() {
        let tmp = tempfile::tempdir().unwrap();
        let hf = tmp.path().join("hf");
        std::fs::create_dir_all(&hf).unwrap();
        let stub = write_stub(tmp.path(), SLOW_STUB);
        let count = Arc::new(AtomicUsize::new(0));
        let coord = QwenPrepareCoordinator::new();
        let paths = paths_at(&hf);
        let resolver = counting_resolver(stub, vec![], count);

        let run = coord.run_single_flight(&paths, resolver, |_| {});
        tokio::pin!(run);
        let result = loop {
            tokio::select! {
                r = &mut run => break r,
                _ = tokio::time::sleep(Duration::from_millis(50)) => {
                    coord.cancel();
                }
            }
        };
        assert!(
            matches!(result, Err(LensError::Cancelled(_))),
            "cancel must abort the prepare, got {result:?}"
        );
        assert!(!coord.cancel(), "slot cleared once the run ends");
    }

    #[tokio::test]
    async fn failed_first_run_lets_second_reenter() {
        let tmp = tempfile::tempdir().unwrap();
        let hf = tmp.path().join("hf");
        std::fs::create_dir_all(&hf).unwrap();
        let stub = write_stub(tmp.path(), FAILING_STUB);
        let count = Arc::new(AtomicUsize::new(0));
        let coord = QwenPrepareCoordinator::new();
        let paths = paths_at(&hf);

        let e1 = coord
            .run_single_flight(
                &paths,
                counting_resolver(stub.clone(), vec![], count.clone()),
                |_| {},
            )
            .await
            .expect_err("failing stub errors");
        assert!(matches!(e1, LensError::Tts(_)));

        let e2 = coord
            .run_single_flight(
                &paths,
                counting_resolver(stub, vec![], count.clone()),
                |_| {},
            )
            .await
            .expect_err("second re-enters (snapshot never completed) and also errors");
        assert!(matches!(e2, LensError::Tts(_)));
        assert_eq!(
            count.load(Ordering::SeqCst),
            2,
            "a failed run does not block re-entry"
        );
    }

    #[tokio::test]
    async fn insufficient_probe_rejects_before_any_prepare() {
        let tmp = tempfile::tempdir().unwrap();
        let hf = tmp.path().join("hf");
        std::fs::create_dir_all(&hf).unwrap();
        let stub = write_stub(tmp.path(), SNAPSHOT_STUB);
        let count = Arc::new(AtomicUsize::new(0));
        let extra = vec![
            hf.to_string_lossy().into_owned(),
            crate::qwen::prepare::QWEN_SNAPSHOT_DIR.to_string(),
        ];
        let coord = QwenPrepareCoordinator::new();
        let paths = paths_at(&hf);

        let result = coord
            .run_single_flight_with_probe(
                &paths,
                counting_resolver(stub, extra, count.clone()),
                |_| {},
                |_: &Path| Some(0u64),
            )
            .await;

        assert!(
            matches!(result, Err(LensError::InsufficientSpace(_))),
            "an insufficient probe must reject, got {result:?}"
        );
        assert_eq!(
            count.load(Ordering::SeqCst),
            0,
            "no prepare must be spawned when the pre-flight rejects"
        );
    }

    #[tokio::test]
    async fn present_snapshot_short_circuits_even_with_zero_space_probe() {
        let tmp = tempfile::tempdir().unwrap();
        let hf = tmp.path().join("hf");
        std::fs::create_dir_all(&hf).unwrap();
        let stub = write_stub(tmp.path(), SNAPSHOT_STUB);
        let count = Arc::new(AtomicUsize::new(0));
        let extra = vec![
            hf.to_string_lossy().into_owned(),
            crate::qwen::prepare::QWEN_SNAPSHOT_DIR.to_string(),
        ];
        let coord = QwenPrepareCoordinator::new();
        let paths = paths_at(&hf);

        coord
            .run_single_flight(
                &paths,
                counting_resolver(stub.clone(), extra.clone(), count.clone()),
                |_| {},
            )
            .await
            .expect("first prepare populates the on-disk snapshot");

        let result = coord
            .run_single_flight_with_probe(
                &paths,
                counting_resolver(stub, extra, count.clone()),
                |_| {},
                |_: &Path| Some(0u64),
            )
            .await;

        assert!(
            result.is_ok(),
            "a fully-cached snapshot must short-circuit before the disk guard, got {result:?}"
        );
        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "the second call must not spawn a prepare"
        );
    }

    #[tokio::test]
    async fn probe_that_cannot_answer_lets_prepare_proceed() {
        let tmp = tempfile::tempdir().unwrap();
        let hf = tmp.path().join("hf");
        std::fs::create_dir_all(&hf).unwrap();
        let stub = write_stub(tmp.path(), SNAPSHOT_STUB);
        let count = Arc::new(AtomicUsize::new(0));
        let extra = vec![
            hf.to_string_lossy().into_owned(),
            crate::qwen::prepare::QWEN_SNAPSHOT_DIR.to_string(),
        ];
        let coord = QwenPrepareCoordinator::new();
        let paths = paths_at(&hf);

        let result = coord
            .run_single_flight_with_probe(
                &paths,
                counting_resolver(stub, extra, count.clone()),
                |_| {},
                |_: &Path| None,
            )
            .await;

        assert!(
            result.is_ok(),
            "a probe that cannot answer must abstain, got {result:?}"
        );
        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "abstaining must let the prepare proceed"
        );
    }

    #[test]
    fn stale_clear_guard_spares_a_newer_run() {
        let coord = QwenPrepareCoordinator::new();
        let token_a = Arc::new(CancellationToken::new());
        *coord.current.lock().unwrap() = Some(token_a.clone());
        // A newer run supersedes the slot with its own token.
        let token_b = Arc::new(CancellationToken::new());
        *coord.current.lock().unwrap() = Some(token_b.clone());
        // Run A's teardown fires while it is no longer the owner: must not evict B.
        drop(ClearGuard {
            current: &coord.current,
            owner: token_a,
        });
        let cur = coord.current.lock().unwrap();
        assert!(
            cur.as_ref().is_some_and(|t| Arc::ptr_eq(t, &token_b)),
            "stale clear must not evict the newer run's token"
        );
        assert!(!token_b.is_cancelled());
    }
}
