//! Shared streaming downloader for the TTS and Whisper model artifacts.

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
/// `MAX_ATTEMPTS` of these plus the backoffs is the ~96 s worst case before an
/// error surfaces.
const IDLE_READ_TIMEOUT: Duration = Duration::from_secs(30);

pub const DISK_HEADROOM_BYTES: u64 = 256 * 1024 * 1024;

const MID_STREAM_CHECK_INTERVAL: u64 = 64 * 1024 * 1024;

// The floor must exceed the check interval: a check that passes at exactly the
// interval is followed by that many more writes, so a smaller floor could only
// ever fire after ENOSPC had already filled the disk.
const MID_STREAM_FLOOR: u64 = MID_STREAM_CHECK_INTERVAL + DISK_HEADROOM_BYTES;
const _: () = assert!(MID_STREAM_FLOOR > MID_STREAM_CHECK_INTERVAL);

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
/// finished-file skip after acquiring: a predecessor may have finished while
/// they queued.
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

enum AttemptFailure {
    Retryable(LensError),
    Fatal(LensError),
}

/// The raw `ENOSPC` check backs up `StorageFull`, which is only std's *mapping* of it and
/// is lost whenever an error reaches us through a layer that preserved just the errno.
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

/// Streamed in fixed-size reads on a blocking thread so a multi-gigabyte artifact
/// neither allocates its own size nor stalls a tokio worker.
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
/// Up to [`MAX_ATTEMPTS`] attempts resume via `Range`, each hashing the finished `.part`
/// from disk before the rename; `None` skips hashing (tests only). A pre-existing `dest`
/// of the expected length is trusted on size alone — the hash guards only the
/// files we rename.
/// `cancel` aborts at any await and retains the `.part` so a later call resumes it.
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

/// Emits a progress tick before any network work so a caller's stall
/// watchdog re-arms across a retry that has not yet produced a byte.
async fn download_attempt<F>(
    ctx: &AttemptCtx<'_>,
    on_progress: &mut F,
) -> Result<(), AttemptFailure>
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
    } else if status.is_client_error()
        && status != reqwest::StatusCode::REQUEST_TIMEOUT
        && status != reqwest::StatusCode::TOO_MANY_REQUESTS
    {
        // A 4xx is the server's verdict on this request, not a transient fault, so
        // retrying just makes the user wait out the backoffs for the same answer.
        // 408 and 429 are the two that do invite one. (416 is handled above.)
        return Err(AttemptFailure::Fatal(LensError::Network(format!(
            "download failed with status {status}"
        ))));
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

/// An absent `expected_sha256` has nothing to check, so it counts as a match.
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
        return Err(AttemptFailure::Fatal(LensError::InsufficientSpace(
            format!(
                "this download needs {required} bytes of free space but only {available} bytes are available"
            ),
        )));
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use wiremock::matchers::{header, header_exists, method};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    fn sha256_hex(bytes: &[u8]) -> String {
        crate::hex_encode(&Sha256::digest(bytes))
    }

    fn fixed_probe(bytes: u64) -> SpaceProbe {
        Arc::new(move |_: &Path| Some(bytes))
    }

    fn abstaining_probe() -> SpaceProbe {
        Arc::new(|_: &Path| None)
    }

    /// Fast retries and disarmed space guards, so each test re-arms only the knob
    /// it is about.
    fn policy() -> DownloadPolicy {
        DownloadPolicy {
            probe: fixed_probe(u64::MAX),
            headroom: 0,
            check_interval: u64::MAX,
            mid_stream_floor: 0,
            max_attempts: MAX_ATTEMPTS,
            base_backoff: Duration::from_millis(1),
        }
    }

    fn scratch() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("models").join("artifact.bin");
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        (dir, dest)
    }

    fn seed_part(dest: &Path, bytes: &[u8]) {
        std::fs::write(dest.with_extension("part"), bytes).unwrap();
    }

    async fn mount_head(server: &MockServer, len: u64) {
        Mock::given(method("HEAD"))
            .respond_with(
                ResponseTemplate::new(200).insert_header("content-length", len.to_string()),
            )
            .mount(server)
            .await;
    }

    async fn mount_failing_head(server: &MockServer) {
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(404))
            .mount(server)
            .await;
    }

    async fn run(
        url: &str,
        dest: &Path,
        sha: Option<&str>,
        cancel: &CancellationToken,
        policy: DownloadPolicy,
    ) -> (Result<(), LensError>, Vec<DownloadProgress>) {
        let mut events = Vec::new();
        let result =
            download_verified_with(url, dest, sha, cancel, |p| events.push(p), policy).await;
        (result, events)
    }

    #[derive(Clone)]
    struct Recorder {
        log: Arc<Mutex<Vec<String>>>,
    }

    impl Recorder {
        fn new() -> Self {
            Self {
                log: Arc::new(Mutex::new(Vec::new())),
            }
        }
        fn push(&self, entry: impl Into<String>) {
            self.log
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(entry.into());
        }
        fn entries(&self) -> Vec<String> {
            self.log.lock().unwrap_or_else(|e| e.into_inner()).clone()
        }
    }

    struct RecordingResponse {
        recorder: Recorder,
        label: &'static str,
        status: u16,
        body: Option<Vec<u8>>,
    }

    impl Respond for RecordingResponse {
        fn respond(&self, _: &Request) -> ResponseTemplate {
            self.recorder.push(self.label);
            let template = ResponseTemplate::new(self.status);
            match &self.body {
                Some(bytes) => template.set_body_bytes(bytes.clone()),
                None => template,
            }
        }
    }

    /// Serves HEAD with the full length, truncates the first GET's body mid-stream
    /// (a real `IncompleteMessage`, which wiremock cannot express), and answers later
    /// GETs with a 206 from the requested offset. Returns the base URL + request log.
    async fn truncating_then_resuming_server(body: Vec<u8>) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let log_for_task = Arc::clone(&log);
        tokio::spawn(async move {
            let total = body.len();
            let mut gets = 0usize;
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                let mut buf = vec![0u8; 8192];
                let Ok(read) = socket.read(&mut buf).await else {
                    continue;
                };
                let request = String::from_utf8_lossy(&buf[..read]).to_ascii_lowercase();
                log_for_task
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(request.clone());

                if request.starts_with("head") {
                    let head = format!(
                        "HTTP/1.1 200 OK\r\ncontent-length: {total}\r\naccept-ranges: bytes\r\nconnection: close\r\n\r\n"
                    );
                    let _ = socket.write_all(head.as_bytes()).await;
                } else {
                    gets += 1;
                    if gets == 1 {
                        let head = format!(
                            "HTTP/1.1 200 OK\r\ncontent-length: {total}\r\nconnection: close\r\n\r\n"
                        );
                        let _ = socket.write_all(head.as_bytes()).await;
                        let _ = socket.write_all(&body[..total / 2]).await;
                    } else {
                        let start = request
                            .split("range: bytes=")
                            .nth(1)
                            .and_then(|rest| rest.split('-').next())
                            .and_then(|n| n.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        let head = format!(
                            "HTTP/1.1 206 Partial Content\r\ncontent-length: {}\r\ncontent-range: bytes {}-{}/{}\r\nconnection: close\r\n\r\n",
                            total - start,
                            start,
                            total - 1,
                            total
                        );
                        let _ = socket.write_all(head.as_bytes()).await;
                        let _ = socket.write_all(&body[start..]).await;
                    }
                }
                let _ = socket.flush().await;
                let _ = socket.shutdown().await;
            }
        });
        (format!("http://{addr}"), log)
    }

    #[tokio::test]
    async fn cancel_mid_stream_returns_cancelled_and_retains_the_partial() {
        let body = vec![5u8; 64 * 1024];
        let server = MockServer::start().await;
        mount_head(&server, body.len() as u64).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .expect(1)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        let mut last_received = 0u64;
        let result = download_verified_with(
            &server.uri(),
            &dest,
            None,
            &cancel,
            |p| {
                if p.received > 0 && !p.done {
                    last_received = p.received;
                    trigger.cancel();
                }
            },
            policy(),
        )
        .await;

        assert!(
            matches!(result, Err(LensError::Cancelled(_))),
            "got {result:?}"
        );
        let part = dest.with_extension("part");
        assert!(part.exists(), ".part must be retained across a cancel");
        assert_eq!(std::fs::metadata(&part).unwrap().len(), last_received);
        assert!(!dest.exists());
    }

    #[tokio::test]
    async fn cancel_before_the_first_byte_makes_no_request() {
        let server = MockServer::start().await;
        mount_head(&server, 1024).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let (result, events) = run(&server.uri(), &dest, None, &cancel, policy()).await;

        assert!(
            matches!(result, Err(LensError::Cancelled(_))),
            "got {result:?}"
        );
        assert!(events.is_empty());
        assert!(
            server
                .received_requests()
                .await
                .unwrap_or_default()
                .is_empty(),
            "an already-cancelled download must not touch the network"
        );
    }

    #[tokio::test]
    async fn cancel_during_backoff_returns_promptly() {
        let server = MockServer::start().await;
        mount_head(&server, 1024).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(502))
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            trigger.cancel();
        });

        let slow = DownloadPolicy {
            base_backoff: Duration::from_secs(5),
            ..policy()
        };
        let started = std::time::Instant::now();
        let (result, _) = run(&server.uri(), &dest, None, &cancel, slow).await;
        let elapsed = started.elapsed();

        assert!(
            matches!(result, Err(LensError::Cancelled(_))),
            "got {result:?}"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "cancel must not wait out the 5 s backoff; took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn cancel_during_a_slow_head_returns_promptly() {
        let server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-length", "1024")
                    .set_delay(Duration::from_secs(5)),
            )
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            trigger.cancel();
        });

        let started = std::time::Instant::now();
        let (result, _) = run(&server.uri(), &dest, None, &cancel, policy()).await;
        let elapsed = started.elapsed();

        assert!(
            matches!(result, Err(LensError::Cancelled(_))),
            "got {result:?}"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "the hoisted HEAD must be cancellable well inside its 5 s delay; took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn transient_stream_error_preserves_the_partial_and_the_retry_sends_range() {
        let body: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let expected = sha256_hex(&body);
        let (url, log) = truncating_then_resuming_server(body.clone()).await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        let (result, _) = run(&url, &dest, Some(&expected), &cancel, policy()).await;

        assert!(result.is_ok(), "got {result:?}");
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert!(!dest.with_extension("part").exists());

        let requests = log.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let gets: Vec<&String> = requests.iter().filter(|r| r.starts_with("get")).collect();
        assert_eq!(
            gets.len(),
            2,
            "expected exactly one retry, saw {requests:?}"
        );
        assert!(
            gets[1].contains(&format!("range: bytes={}-", body.len() / 2)),
            "the retry must resume from the retained .part, not restart at zero: {}",
            gets[1]
        );
    }

    #[tokio::test]
    async fn zero_byte_retry_emits_progress_before_the_second_get() {
        let body = vec![1u8; 1024];
        let recorder = Recorder::new();
        let server = MockServer::start().await;
        mount_head(&server, body.len() as u64).await;
        Mock::given(method("GET"))
            .respond_with(RecordingResponse {
                recorder: recorder.clone(),
                label: "get",
                status: 502,
                body: None,
            })
            .up_to_n_times(1)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(RecordingResponse {
                recorder: recorder.clone(),
                label: "get",
                status: 200,
                body: Some(body.clone()),
            })
            .with_priority(2)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        let timeline = recorder.clone();
        let result = download_verified_with(
            &server.uri(),
            &dest,
            None,
            &cancel,
            |p| {
                if !p.done {
                    timeline.push(format!("progress:{}", p.received));
                }
            },
            policy(),
        )
        .await;
        assert!(result.is_ok(), "got {result:?}");

        let entries = recorder.entries();
        let gets: Vec<usize> = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| *e == "get")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(gets.len(), 2, "timeline: {entries:?}");
        assert!(
            entries[gets[0] + 1..gets[1]]
                .iter()
                .any(|e| e == "progress:0"),
            "an attempt-start tick must land between the two GETs so a stall watchdog re-arms; timeline: {entries:?}"
        );
    }

    #[tokio::test]
    async fn resume_without_content_length_keeps_the_partial_and_sends_range() {
        let body: Vec<u8> = (0..1024u32).map(|i| (i % 199) as u8).collect();
        let expected = sha256_hex(&body);
        let server = MockServer::start().await;
        mount_failing_head(&server).await;
        Mock::given(method("GET"))
            .and(header("range", "bytes=100-"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header(
                        "content-range",
                        format!("bytes 100-{}/{}", body.len() - 1, body.len()),
                    )
                    .set_body_bytes(body[100..].to_vec()),
            )
            .expect(1)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        seed_part(&dest, &body[..100]);
        let cancel = CancellationToken::new();
        let (result, events) = run(&server.uri(), &dest, Some(&expected), &cancel, policy()).await;

        assert!(result.is_ok(), "got {result:?}");
        assert_eq!(
            std::fs::read(&dest).unwrap(),
            body,
            "the retained 100 bytes must survive an unknown-length resume"
        );
        assert!(
            events.iter().any(|e| e.total == Some(body.len() as u64)),
            "with no HEAD length the total must come from Content-Range: {events:?}"
        );
        assert!(
            !events.iter().any(|e| e.total == Some(924)),
            "the 924-byte partial body must never be reported as the total: {events:?}"
        );
    }

    #[tokio::test]
    async fn partial_content_progress_reports_the_full_artifact_total() {
        let body: Vec<u8> = (0..1024u32).map(|i| (i % 97) as u8).collect();
        let expected = sha256_hex(&body);
        let server = MockServer::start().await;
        mount_head(&server, body.len() as u64).await;
        Mock::given(method("GET"))
            .and(header("range", "bytes=900-"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header(
                        "content-range",
                        format!("bytes 900-{}/{}", body.len() - 1, body.len()),
                    )
                    .set_body_bytes(body[900..].to_vec()),
            )
            .expect(1)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        seed_part(&dest, &body[..900]);
        let cancel = CancellationToken::new();
        let (result, events) = run(&server.uri(), &dest, Some(&expected), &cancel, policy()).await;

        assert!(result.is_ok(), "got {result:?}");
        assert!(
            events.iter().all(|e| e.total == Some(1024)),
            "the 124-byte partial body must never be reported as the total: {events:?}"
        );
        assert_eq!(events.first().map(|e| e.received), Some(900));
        let received: Vec<u64> = events.iter().map(|e| e.received).collect();
        let mut sorted = received.clone();
        sorted.sort_unstable();
        assert_eq!(
            received, sorted,
            "received must be monotonic across a resume"
        );
    }

    #[tokio::test]
    async fn two_hundred_answer_to_a_ranged_request_truncates_the_partial() {
        let body = vec![7u8; 1024];
        let expected = sha256_hex(&body);
        let server = MockServer::start().await;
        mount_head(&server, body.len() as u64).await;
        Mock::given(method("GET"))
            .and(header("range", "bytes=100-"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .expect(1)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        seed_part(&dest, &[0xAAu8; 100]);
        let cancel = CancellationToken::new();
        let (result, events) = run(&server.uri(), &dest, Some(&expected), &cancel, policy()).await;

        assert!(result.is_ok(), "got {result:?}");
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert!(events.iter().any(|e| e.received == 1024 && e.done));
    }

    #[tokio::test]
    async fn range_not_satisfiable_discards_the_partial_then_succeeds() {
        let body = vec![3u8; 1024];
        let expected = sha256_hex(&body);
        let server = MockServer::start().await;
        mount_head(&server, body.len() as u64).await;
        Mock::given(method("GET"))
            .and(header_exists("range"))
            .respond_with(ResponseTemplate::new(416))
            .with_priority(1)
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .with_priority(2)
            .expect(1)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        seed_part(&dest, &[9u8; 100]);
        let cancel = CancellationToken::new();
        let (result, _) = run(&server.uri(), &dest, Some(&expected), &cancel, policy()).await;

        assert!(result.is_ok(), "got {result:?}");
        assert_eq!(std::fs::read(&dest).unwrap(), body);
    }

    #[tokio::test]
    async fn oversized_partial_is_discarded_and_refetched_without_a_range() {
        let body = vec![4u8; 1024];
        let expected = sha256_hex(&body);
        let server = MockServer::start().await;
        mount_head(&server, body.len() as u64).await;
        Mock::given(method("GET"))
            .and(header_exists("range"))
            .respond_with(ResponseTemplate::new(500))
            .with_priority(1)
            .expect(0)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .with_priority(2)
            .expect(1)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        seed_part(&dest, &[0u8; 4096]);
        let cancel = CancellationToken::new();
        let (result, _) = run(&server.uri(), &dest, Some(&expected), &cancel, policy()).await;

        assert!(result.is_ok(), "got {result:?}");
        assert_eq!(std::fs::read(&dest).unwrap(), body);
    }

    #[tokio::test]
    async fn complete_partial_with_a_matching_hash_is_renamed_without_a_get() {
        let body = vec![6u8; 1024];
        let expected = sha256_hex(&body);
        let server = MockServer::start().await;
        mount_head(&server, body.len() as u64).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        seed_part(&dest, &body);
        let cancel = CancellationToken::new();
        let (result, events) = run(&server.uri(), &dest, Some(&expected), &cancel, policy()).await;

        assert!(result.is_ok(), "got {result:?}");
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert!(!dest.with_extension("part").exists());
        assert!(events.last().is_some_and(|e| e.done && e.received == 1024));
    }

    #[tokio::test]
    async fn complete_partial_with_a_bad_hash_is_refetched_in_the_same_attempt() {
        let body = vec![8u8; 1024];
        let expected = sha256_hex(&body);
        let server = MockServer::start().await;
        mount_head(&server, body.len() as u64).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .expect(1)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        seed_part(&dest, &[0xFFu8; 1024]);
        let cancel = CancellationToken::new();
        let (result, events) = run(&server.uri(), &dest, Some(&expected), &cancel, policy()).await;

        assert!(result.is_ok(), "got {result:?}");
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert!(
            !events.iter().any(|e| e.received == 0),
            "only an attempt start reports zero bytes, so a second one means the stale \
             partial burned a retry instead of being refetched in place: {events:?}"
        );
    }

    #[tokio::test]
    async fn retry_is_bounded_and_surfaces_the_last_network_error() {
        let server = MockServer::start().await;
        mount_head(&server, 1024).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(502))
            .expect(3)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        let (result, _) = run(&server.uri(), &dest, None, &cancel, policy()).await;

        assert!(
            matches!(result, Err(LensError::Network(_))),
            "got {result:?}"
        );
    }

    #[tokio::test]
    async fn a_client_error_is_not_retried() {
        let server = MockServer::start().await;
        mount_head(&server, 1024).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(404))
            .expect(1)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        let (result, _) = run(&server.uri(), &dest, None, &cancel, policy()).await;

        assert!(
            matches!(result, Err(LensError::Network(_))),
            "got {result:?}"
        );
    }

    /// Scopes the 4xx exemption: a 5xx must still exhaust the retry budget, or the
    /// fatal arm has swallowed the transient failures the retry loop exists for.
    #[tokio::test]
    async fn a_server_error_is_still_retried() {
        let server = MockServer::start().await;
        mount_head(&server, 1024).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(503))
            .expect(3)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        let (result, _) = run(&server.uri(), &dest, None, &cancel, policy()).await;

        assert!(
            matches!(result, Err(LensError::Network(_))),
            "got {result:?}"
        );
    }

    /// 429 is a client error that *does* invite a retry, so it must not fall into
    /// the fatal arm with the rest of the 4xx family.
    #[tokio::test]
    async fn too_many_requests_is_retried() {
        let server = MockServer::start().await;
        mount_head(&server, 1024).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(429))
            .expect(3)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        let (result, _) = run(&server.uri(), &dest, None, &cancel, policy()).await;

        assert!(
            matches!(result, Err(LensError::Network(_))),
            "got {result:?}"
        );
    }

    #[tokio::test]
    async fn hash_mismatch_fails_without_a_retry_and_discards_the_partial() {
        let body = vec![2u8; 1024];
        let wrong = sha256_hex(b"a completely different artifact");
        let server = MockServer::start().await;
        mount_head(&server, body.len() as u64).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body))
            .expect(1)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        let (result, _) = run(&server.uri(), &dest, Some(&wrong), &cancel, policy()).await;

        assert!(
            matches!(result, Err(LensError::Network(_))),
            "got {result:?}"
        );
        assert!(!dest.exists());
        assert!(!dest.with_extension("part").exists());
    }

    #[test]
    fn available_space_probes_a_real_directory_and_abstains_on_a_missing_one() {
        let dir = tempfile::tempdir().unwrap();
        assert!(available_space_bytes(dir.path()).is_some_and(|n| n > 0));
        assert!(
            available_space_bytes(&dir.path().join("nope")).is_none(),
            "a missing directory reports ENOENT and must abstain, not read as zero space"
        );
    }

    #[test]
    fn a_write_time_enospc_is_a_fatal_insufficient_space() {
        let tmp = Path::new("artifact.part");
        assert!(
            matches!(
                map_write_error(&std::io::Error::from_raw_os_error(28), tmp),
                AttemptFailure::Fatal(LensError::InsufficientSpace(_))
            ),
            "raw ENOSPC must map to a non-retryable InsufficientSpace"
        );
        assert!(
            matches!(
                map_write_error(&std::io::Error::from(std::io::ErrorKind::StorageFull), tmp),
                AttemptFailure::Fatal(LensError::InsufficientSpace(_))
            ),
            "ErrorKind::StorageFull must map to a non-retryable InsufficientSpace"
        );
        assert!(
            matches!(
                map_write_error(
                    &std::io::Error::from(std::io::ErrorKind::PermissionDenied),
                    tmp
                ),
                AttemptFailure::Retryable(LensError::Io(_))
            ),
            "any other write failure must stay a retryable Io error"
        );
    }

    #[tokio::test]
    async fn preflight_rejects_before_issuing_a_get() {
        let server = MockServer::start().await;
        mount_head(&server, 4096).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        let tight = DownloadPolicy {
            probe: fixed_probe(4195),
            headroom: 100,
            ..policy()
        };
        let (result, _) = run(&server.uri(), &dest, None, &cancel, tight).await;

        match result {
            Err(LensError::InsufficientSpace(msg)) => assert!(
                msg.contains("4196") && msg.contains("4195"),
                "the message must name bytes required and available: {msg}"
            ),
            other => panic!("expected InsufficientSpace, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tight_but_sufficient_space_completes() {
        let body = vec![1u8; 4096];
        let expected = sha256_hex(&body);
        let server = MockServer::start().await;
        mount_head(&server, body.len() as u64).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        let exact = DownloadPolicy {
            probe: fixed_probe(4196),
            headroom: 100,
            check_interval: 512,
            mid_stream_floor: 0,
            ..policy()
        };
        let (result, _) = run(&server.uri(), &dest, Some(&expected), &cancel, exact).await;

        assert!(
            result.is_ok(),
            "exactly-enough space must not abort: {result:?}"
        );
        assert_eq!(std::fs::read(&dest).unwrap(), body);
    }

    #[tokio::test]
    async fn foreign_consumption_mid_stream_aborts_and_preserves_the_partial() {
        let body = vec![1u8; 256 * 1024];
        let server = MockServer::start().await;
        mount_head(&server, body.len() as u64).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .expect(1)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        let calls = Arc::new(AtomicU64::new(0));
        let seen = Arc::clone(&calls);
        let plenty_then_full: SpaceProbe = Arc::new(move |_: &Path| {
            Some(if seen.fetch_add(1, Ordering::SeqCst) == 0 {
                u64::MAX
            } else {
                0
            })
        });
        let draining = DownloadPolicy {
            probe: plenty_then_full,
            headroom: 0,
            check_interval: 1024,
            ..policy()
        };
        let (result, _) = run(&server.uri(), &dest, None, &cancel, draining).await;

        assert!(
            matches!(result, Err(LensError::InsufficientSpace(_))),
            "got {result:?}"
        );
        assert!(
            dest.with_extension("part").exists(),
            ".part must survive so the user can resume after freeing space"
        );
        assert!(
            calls.load(Ordering::SeqCst) >= 2,
            "the mid-stream guard never ran"
        );
    }

    #[tokio::test]
    async fn header_less_mid_stream_check_defends_the_floor() {
        let body = vec![1u8; 4096];
        let server = MockServer::start().await;
        mount_failing_head(&server).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body))
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        let starved = DownloadPolicy {
            probe: fixed_probe(1023),
            check_interval: 512,
            mid_stream_floor: 1024,
            ..policy()
        };
        let (result, _) = run(&server.uri(), &dest, None, &cancel, starved).await;

        assert!(
            matches!(result, Err(LensError::InsufficientSpace(_))),
            "got {result:?}"
        );
        assert!(dest.with_extension("part").exists());
    }

    #[tokio::test]
    async fn missing_content_length_abstains_from_the_preflight() {
        let body = vec![1u8; 1024];
        let expected = sha256_hex(&body);
        let server = MockServer::start().await;
        mount_failing_head(&server).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        let starved = DownloadPolicy {
            probe: fixed_probe(0),
            headroom: DISK_HEADROOM_BYTES,
            ..policy()
        };
        let (result, _) = run(&server.uri(), &dest, Some(&expected), &cancel, starved).await;

        assert!(
            result.is_ok(),
            "a header-less server must not become a download outage: {result:?}"
        );
        assert_eq!(std::fs::read(&dest).unwrap(), body);
    }

    #[tokio::test]
    async fn a_failing_space_probe_abstains_from_the_preflight() {
        let body = vec![1u8; 1024];
        let expected = sha256_hex(&body);
        let server = MockServer::start().await;
        mount_head(&server, body.len() as u64).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        let blind = DownloadPolicy {
            probe: abstaining_probe(),
            headroom: DISK_HEADROOM_BYTES,
            ..policy()
        };
        let (result, _) = run(&server.uri(), &dest, Some(&expected), &cancel, blind).await;

        assert!(
            result.is_ok(),
            "an unstattable filesystem must not become a download outage: {result:?}"
        );
        assert_eq!(std::fs::read(&dest).unwrap(), body);
    }

    #[tokio::test]
    async fn concurrent_downloads_of_one_artifact_issue_a_single_get() {
        let body = vec![1u8; 1024];
        let expected = sha256_hex(&body);
        let server = MockServer::start().await;
        mount_head(&server, body.len() as u64).await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(body.clone())
                    .set_delay(Duration::from_millis(150)),
            )
            .expect(1)
            .mount(&server)
            .await;

        let (_dir, dest) = scratch();
        let cancel = CancellationToken::new();
        let uri = server.uri();
        let (first, second) = tokio::join!(
            run(&uri, &dest, Some(&expected), &cancel, policy()),
            run(&uri, &dest, Some(&expected), &cancel, policy()),
        );

        assert!(first.0.is_ok(), "got {:?}", first.0);
        assert!(second.0.is_ok(), "got {:?}", second.0);
        assert_eq!(std::fs::read(&dest).unwrap(), body);
    }

    #[test]
    fn write_locks_are_per_part_path() {
        let alpha = part_write_lock(Path::new("/models/alpha.part"));
        let beta = part_write_lock(Path::new("/models/beta.part"));
        let alpha_again = part_write_lock(Path::new("/models/alpha.part"));
        assert!(
            !Arc::ptr_eq(&alpha, &beta),
            "a single global gate would serialize concurrent downloads of different artifacts"
        );
        assert!(
            Arc::ptr_eq(&alpha, &alpha_again),
            "the same .part path must reuse its lock or the guard is bypassable"
        );
    }

    #[tokio::test]
    async fn concurrent_downloads_of_different_artifacts_each_fetch() {
        let body = vec![1u8; 1024];
        let expected = sha256_hex(&body);
        let server = MockServer::start().await;
        mount_head(&server, body.len() as u64).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .expect(2)
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let alpha = dir.path().join("alpha.bin");
        let beta = dir.path().join("beta.bin");
        let cancel = CancellationToken::new();
        let uri = server.uri();
        let (a, b) = tokio::join!(
            run(&uri, &alpha, Some(&expected), &cancel, policy()),
            run(&uri, &beta, Some(&expected), &cancel, policy()),
        );

        assert!(a.0.is_ok() && b.0.is_ok(), "{:?} / {:?}", a.0, b.0);
        assert_eq!(std::fs::read(&alpha).unwrap(), body);
        assert_eq!(std::fs::read(&beta).unwrap(), body);
    }
}
