//! Shared streaming downloader for the TTS and Whisper model artifacts.
//!
//! One place owns the whole hardening story: a free-space pre-flight, a bounded
//! retry loop that resumes an interrupted transfer via HTTP `Range`, cooperative
//! cancellation, and a SHA256 taken from the finished `.part` on disk immediately
//! before the atomic rename.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use tokio::sync::Mutex as AsyncMutex;
use tokio_util::sync::CancellationToken;

use crate::LensError;
use crate::tts::DownloadProgress;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Idle read timeout: resets on each received chunk rather than bounding the whole
/// request, so a large legitimate download never expires but a stalled body does.
/// `MAX_ATTEMPTS` of these plus the backoffs is the ~96 s worst case before an error surfaces.
const IDLE_READ_TIMEOUT: Duration = Duration::from_secs(30);

const DISK_HEADROOM_BYTES: u64 = 256 * 1024 * 1024;

const MID_STREAM_CHECK_INTERVAL: u64 = 64 * 1024 * 1024;

// The floor must exceed the check interval: a check that passes at exactly the
// interval is followed by that many more writes, so a smaller floor could only
// ever fire after ENOSPC had already filled the disk.
const MID_STREAM_FLOOR: u64 = MID_STREAM_CHECK_INTERVAL + DISK_HEADROOM_BYTES;

const MAX_ATTEMPTS: u32 = 3;
const BASE_BACKOFF: Duration = Duration::from_secs(2);

const DIGEST_READ_BUF_BYTES: usize = 1024 * 1024;

type SpaceProbe = Arc<dyn Fn(&Path) -> Option<u64> + Send + Sync>;

/// Guard thresholds and retry budget, injectable so tests can drive both.
#[derive(Clone)]
struct DownloadPolicy {
    probe: SpaceProbe,
    headroom: u64,
    check_interval: u64,
    mid_stream_floor: u64,
    max_attempts: u32,
    base_backoff: Duration,
}

impl Default for DownloadPolicy {
    fn default() -> Self {
        Self {
            probe: Arc::new(available_space_bytes),
            headroom: DISK_HEADROOM_BYTES,
            check_interval: MID_STREAM_CHECK_INTERVAL,
            mid_stream_floor: MID_STREAM_FLOOR,
            max_attempts: MAX_ATTEMPTS,
            base_backoff: BASE_BACKOFF,
        }
    }
}

/// Free bytes available under `dir`, or `None` when the filesystem cannot be
/// queried — including a missing directory, which reports `ENOENT` and not zero.
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

/// Per-`.part` write locks, keyed on the `.part` path — the resource actually shared —
/// so different artifacts still download in parallel. Callers must re-check the
/// finished-file skip after acquiring: a predecessor may have completed while they queued.
fn part_write_lock(part: &Path) -> Arc<AsyncMutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<AsyncMutex<()>>>>> = OnceLock::new();
    let mut map = LOCKS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    Arc::clone(map.entry(part.to_path_buf()).or_default())
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

/// Full artifact length from a `Content-Range: bytes {start}-{end}/{total}` header.
fn content_range_total(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::CONTENT_RANGE)?
        .to_str()
        .ok()?
        .rsplit('/')
        .next()?
        .trim()
        .parse()
        .ok()
}

fn cancelled() -> LensError {
    LensError::Cancelled("download cancelled".into())
}

/// A failed attempt, tagged with whether the retry loop may try again.
enum AttemptFailure {
    Retryable(LensError),
    Fatal(LensError),
}

/// Classifies a `.part` write failure: a full disk is fatal, anything else retryable.
fn map_write_error(err: &std::io::Error, tmp: &Path) -> AttemptFailure {
    if err.kind() == std::io::ErrorKind::StorageFull || err.raw_os_error() == Some(28) {
        return AttemptFailure::Fatal(LensError::InsufficientSpace(
            "the disk filled up while downloading; free some space and try again".into(),
        ));
    }
    AttemptFailure::Retryable(LensError::Io(format!("write {}: {err}", tmp.display())))
}

fn completed_skip(dest: &Path, expected_len: Option<u64>) -> Option<DownloadProgress> {
    let on_disk = std::fs::metadata(dest).ok()?.len();
    (on_disk > 0 && expected_len.is_some_and(|n| n == on_disk)).then_some(DownloadProgress {
        received: on_disk,
        total: expected_len,
        done: true,
    })
}

fn finalize(tmp: &Path, dest: &Path) -> Result<(), AttemptFailure> {
    std::fs::rename(tmp, dest).map_err(|e| {
        AttemptFailure::Retryable(LensError::Io(format!("finalize {}: {e}", dest.display())))
    })
}

/// SHA256 of `path`, streamed in fixed-size reads on a blocking thread so a
/// multi-gigabyte artifact neither allocates its own size nor stalls a tokio worker.
async fn sha256_file(path: &Path) -> Result<String, LensError> {
    let owned = path.to_path_buf();
    let digest = tokio::task::spawn_blocking(move || {
        let mut file = std::fs::File::open(&owned)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; DIGEST_READ_BUF_BYTES];
        loop {
            let read = file.read(&mut buf)?;
            if read == 0 {
                break;
            }
            hasher.update(&buf[..read]);
        }
        Ok::<_, std::io::Error>(crate::hex_encode(&hasher.finalize()))
    })
    .await?
    .map_err(|e| LensError::Io(format!("hash {}: {e}", path.display())))?;
    Ok(digest)
}

struct AttemptCtx<'a> {
    client: &'a reqwest::Client,
    url: &'a str,
    dest: &'a Path,
    tmp: &'a Path,
    expected_sha256: Option<&'a str>,
    expected_len: Option<u64>,
    cancel: &'a CancellationToken,
    policy: DownloadPolicy,
}

/// Streams `url` into `dest` with progress reporting and SHA256 verification.
///
/// Flow: HEAD probe for the expected length, one writer per `.part` path, then up to
/// [`MAX_ATTEMPTS`] attempts that resume via `Range`; each attempt SHA256s the finished
/// `.part` read back from disk before the atomic rename, so integrity is enforced on
/// every published file — including one an earlier attempt left complete.
/// `expected_sha256 = None` skips verification (tests only). `cancel` aborts at any
/// await and retains the `.part` so a later call resumes it.
pub(crate) async fn download_verified<F>(
    url: &str,
    dest: &Path,
    expected_sha256: Option<&str>,
    cancel: &CancellationToken,
    on_progress: F,
) -> Result<(), LensError>
where
    F: FnMut(DownloadProgress),
{
    download_verified_with(
        url,
        dest,
        expected_sha256,
        cancel,
        on_progress,
        DownloadPolicy::default(),
    )
    .await
}

async fn download_verified_with<F>(
    url: &str,
    dest: &Path,
    expected_sha256: Option<&str>,
    cancel: &CancellationToken,
    on_progress: F,
    policy: DownloadPolicy,
) -> Result<(), LensError>
where
    F: FnMut(DownloadProgress),
{
    tokio::select! {
        biased;
        () = cancel.cancelled() => Err(cancelled()),
        result = download_all_attempts(url, dest, expected_sha256, cancel, on_progress, policy) => result,
    }
}

async fn download_all_attempts<F>(
    url: &str,
    dest: &Path,
    expected_sha256: Option<&str>,
    cancel: &CancellationToken,
    mut on_progress: F,
    policy: DownloadPolicy,
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

    if let Some(progress) = completed_skip(dest, expected_len) {
        on_progress(progress);
        return Ok(());
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| LensError::Io(format!("create {}: {e}", parent.display())))?;
    }

    let tmp = dest.with_extension("part");
    let lock = part_write_lock(&tmp);
    let _writer = lock.lock().await;
    if let Some(progress) = completed_skip(dest, expected_len) {
        on_progress(progress);
        return Ok(());
    }

    let ctx = AttemptCtx {
        client: &client,
        url,
        dest,
        tmp: &tmp,
        expected_sha256,
        expected_len,
        cancel,
        policy,
    };

    let mut attempt: u32 = 1;
    loop {
        match download_attempt(&ctx, &mut on_progress).await {
            Ok(()) => return Ok(()),
            Err(AttemptFailure::Fatal(err)) => return Err(err),
            Err(AttemptFailure::Retryable(err)) => {
                if attempt >= ctx.policy.max_attempts {
                    return Err(err);
                }
                let backoff = ctx.policy.base_backoff * 2u32.pow(attempt - 1);
                tracing::info!(attempt, ?backoff, error = %err, "download attempt failed; retrying");
                tokio::select! {
                    biased;
                    () = cancel.cancelled() => return Err(cancelled()),
                    () = tokio::time::sleep(backoff) => {}
                }
                attempt += 1;
            }
        }
    }
}

/// One attempt. Emits a progress tick before any network work so a caller's stall
/// watchdog re-arms across a retry that has not yet produced a byte.
async fn download_attempt<F>(ctx: &AttemptCtx<'_>, on_progress: &mut F) -> Result<(), AttemptFailure>
where
    F: FnMut(DownloadProgress),
{
    if ctx.cancel.is_cancelled() {
        return Err(AttemptFailure::Fatal(cancelled()));
    }

    // The resume offset always comes from the `.part` length on disk, never an
    // in-memory counter: bytes are written before they are counted, and a cancel
    // dropping an in-flight chunk would leave a counter ahead of the file.
    let mut part_len = std::fs::metadata(ctx.tmp).map(|m| m.len()).unwrap_or(0);

    on_progress(DownloadProgress {
        received: part_len,
        total: ctx.expected_len,
        done: false,
    });

    if let Some(expected) = ctx.expected_len {
        if part_len > expected {
            let _ = std::fs::remove_file(ctx.tmp);
            part_len = 0;
        } else if part_len == expected {
            match complete_part_matches(ctx).await {
                Ok(true) => {
                    finalize(ctx.tmp, ctx.dest)?;
                    on_progress(DownloadProgress {
                        received: part_len,
                        total: ctx.expected_len,
                        done: true,
                    });
                    return Ok(());
                }
                Ok(false) => {
                    let _ = std::fs::remove_file(ctx.tmp);
                    part_len = 0;
                }
                Err(err) => return Err(AttemptFailure::Retryable(err)),
            }
        }
    }

    preflight_space(ctx, part_len)?;

    let mut request = ctx.client.get(ctx.url);
    if part_len > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={part_len}-"));
        tracing::info!(
            part_len,
            expected_len = ?ctx.expected_len,
            "resuming download from the retained partial file"
        );
    }
    let response = request.send().await.map_err(|e| {
        AttemptFailure::Retryable(LensError::Network(format!("download request failed: {e}")))
    })?;

    let status = response.status();
    let (mut file, mut received, total) = if status == reqwest::StatusCode::PARTIAL_CONTENT {
        // On a 206 the response `Content-Length` is the PARTIAL body length, so the
        // full artifact size must come from the HEAD probe or `Content-Range`.
        let total = ctx
            .expected_len
            .or_else(|| content_range_total(response.headers()));
        let file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(ctx.tmp)
            .map_err(|e| {
                AttemptFailure::Retryable(LensError::Io(format!(
                    "append {}: {e}",
                    ctx.tmp.display()
                )))
            })?;
        (file, part_len, total)
    } else if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        let _ = std::fs::remove_file(ctx.tmp);
        return Err(AttemptFailure::Retryable(LensError::Network(
            "the server rejected the resume range; the partial file was discarded".into(),
        )));
    } else if status.is_success() {
        let total = content_length_header(response.headers()).or(ctx.expected_len);
        let file = std::fs::File::create(ctx.tmp).map_err(|e| {
            AttemptFailure::Retryable(LensError::Io(format!("create {}: {e}", ctx.tmp.display())))
        })?;
        (file, 0, total)
    } else {
        return Err(AttemptFailure::Retryable(LensError::Network(format!(
            "download failed with status {status}"
        ))));
    };

    let mut stream = response.bytes_stream();
    let mut since_space_check: u64 = 0;
    loop {
        let chunk = tokio::select! {
            biased;
            () = ctx.cancel.cancelled() => return Err(AttemptFailure::Fatal(cancelled())),
            item = stream.next() => match item {
                None => break,
                Some(Ok(chunk)) => chunk,
                Some(Err(e)) => {
                    return Err(AttemptFailure::Retryable(LensError::Network(format!(
                        "download stream error: {e}"
                    ))));
                }
            },
        };

        if let Err(e) = file.write_all(&chunk) {
            return Err(map_write_error(&e, ctx.tmp));
        }
        received = received.saturating_add(chunk.len() as u64);
        on_progress(DownloadProgress {
            received,
            total,
            done: false,
        });

        since_space_check = since_space_check.saturating_add(chunk.len() as u64);
        if since_space_check >= ctx.policy.check_interval {
            since_space_check = 0;
            if let Some(err) = mid_stream_space_error(ctx, received) {
                return Err(AttemptFailure::Fatal(err));
            }
        }
    }

    file.flush().map_err(|e| map_write_error(&e, ctx.tmp))?;
    // `sync_all` before the digest is what makes the hash cover DURABLE bytes: a
    // digest read back through the page cache would still let a crash between the
    // rename and writeback publish a file whose blocks never landed.
    file.sync_all().map_err(|e| map_write_error(&e, ctx.tmp))?;
    drop(file);

    if let Some(expected) = ctx.expected_sha256 {
        let actual = sha256_file(ctx.tmp)
            .await
            .map_err(AttemptFailure::Retryable)?;
        if !actual.eq_ignore_ascii_case(expected) {
            let _ = std::fs::remove_file(ctx.tmp);
            let hint = if part_len > 0 {
                " the resumed partial file was discarded, so starting the download again is safe"
            } else {
                ""
            };
            return Err(AttemptFailure::Fatal(LensError::Network(format!(
                "downloaded file failed integrity check: expected sha256 {expected}, got {actual}.{hint}"
            ))));
        }
    }

    finalize(ctx.tmp, ctx.dest)?;
    on_progress(DownloadProgress {
        received,
        total,
        done: true,
    });
    Ok(())
}

/// Whether a `.part` that already reached the expected length verifies. An absent
/// `expected_sha256` has nothing to check, so it counts as a match.
async fn complete_part_matches(ctx: &AttemptCtx<'_>) -> Result<bool, LensError> {
    match ctx.expected_sha256 {
        None => Ok(true),
        Some(expected) => Ok(sha256_file(ctx.tmp).await?.eq_ignore_ascii_case(expected)),
    }
}

fn preflight_space(ctx: &AttemptCtx<'_>, part_len: u64) -> Result<(), AttemptFailure> {
    let Some(expected) = ctx.expected_len else {
        tracing::warn!(
            url = %ctx.url,
            "disk-space pre-flight skipped: the server reported no Content-Length"
        );
        return Ok(());
    };
    let parent = ctx.dest.parent().unwrap_or_else(|| Path::new("."));
    let Some(available) = (ctx.policy.probe)(parent) else {
        tracing::warn!(
            dir = %parent.display(),
            "disk-space pre-flight skipped: the free-space probe failed"
        );
        return Ok(());
    };
    let required = expected
        .saturating_sub(part_len)
        .saturating_add(ctx.policy.headroom);
    if available < required {
        return Err(AttemptFailure::Fatal(LensError::InsufficientSpace(format!(
            "this download needs {required} bytes of free space but only {available} bytes are available"
        ))));
    }
    Ok(())
}

fn mid_stream_space_error(ctx: &AttemptCtx<'_>, received: u64) -> Option<LensError> {
    let parent = ctx.dest.parent().unwrap_or_else(|| Path::new("."));
    let available = (ctx.policy.probe)(parent)?;
    // `available - remaining` is invariant under our OWN writes, so the sized arm
    // only ever trips on foreign consumption; a header-less download has no
    // remainder to compare and can defend only the floor.
    let required = match ctx.expected_len {
        Some(expected) => expected.saturating_sub(received),
        None => ctx.policy.mid_stream_floor,
    };
    (available < required).then(|| {
        LensError::InsufficientSpace(format!(
            "the disk ran low while downloading: {required} bytes still needed but only {available} bytes are available"
        ))
    })
}
