//! Qwen3-TTS CustomVoice sidecar host ([161e]): drives an out-of-process MLX
//! Python sidecar over line-delimited JSON-over-stdio as lens-core's
//! [`TtsSidecar`]. Gated to `aarch64-apple-darwin` (the only target MLX runs on).
//!
//! The sidecar runs under `uv` (`uv run` against the bundled `pyproject.toml`);
//! `mlx-audio` auto-downloads the model into the `HF_HOME` cache passed in the
//! [`SidecarSpawn`]. Launch details are resolved lazily via an injected resolver
//! (see [`resolver::resolve_sidecar_spawn`]) so provisioning never blocks startup.
//!
//! All failures map through [`tts_err`].

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::time::{Duration, timeout};
use tokio_util::sync::CancellationToken;

use lens_core::{
    AudioBuffer, LensError, Speaker, TtsSidecar, Turn, VoiceConfig, VoiceRef, qwen_voice,
    read_wav_mono16,
};

mod prepare;
mod resolver;
pub use prepare::{qwen_snapshot_present, run_prepare};
pub use resolver::resolve_sidecar_spawn;

/// The app-data-derived paths the Qwen sidecar launches against. Single source of
/// truth so both the startup injection (`main.rs`) and the on-demand `--prepare`
/// command resolve identical `HF_HOME`/sidecar locations — never a divergent path.
pub struct SidecarPaths {
    pub app_data_dir: PathBuf,
    pub sidecar_dir: PathBuf,
    pub hf_cache_dir: PathBuf,
}

/// Resolves the [`SidecarPaths`] from an [`AppHandle`](tauri::AppHandle): app-data
/// via Tauri, the bundled sidecar dir (source tree in dev, `_up_/sidecar` in a
/// packaged build), and the `hf-cache` subdir the resolver points `HF_HOME` at.
pub fn sidecar_paths(app: &tauri::AppHandle) -> Result<SidecarPaths, LensError> {
    use tauri::Manager;
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| LensError::Io(e.to_string()))?;
    let sidecar_dir = if cfg!(debug_assertions) {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sidecar/qwen3-tts")
    } else {
        app.path()
            .resource_dir()
            .map_err(|e| LensError::Io(e.to_string()))?
            .join("_up_/sidecar/qwen3-tts")
    };
    let hf_cache_dir = app_data_dir.join("hf-cache");
    Ok(SidecarPaths {
        app_data_dir,
        sidecar_dir,
        hf_cache_dir,
    })
}

/// Builds the lazy [`SpawnResolver`] for [`resolve_sidecar_spawn`] from resolved
/// [`SidecarPaths`]. Used by both the serve host and the `--prepare` one-shot.
pub fn spawn_resolver(paths: &SidecarPaths) -> SpawnResolver {
    let app_data = paths.app_data_dir.clone();
    let sidecar_dir = paths.sidecar_dir.clone();
    let hf_cache_dir = paths.hf_cache_dir.clone();
    Arc::new(move || resolve_sidecar_spawn(&app_data, &sidecar_dir, &hf_cache_dir))
}

/// A fully-resolved sidecar launch: the program to exec, its args, and the extra
/// environment it needs (e.g. `HF_HOME` so `huggingface_hub` caches into app-data).
/// Produced by a [`SpawnResolver`] so [`QwenSidecar`] stays testable with a stub.
pub struct SidecarSpawn {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub envs: Vec<(String, String)>,
}

/// Lazily produces the [`SidecarSpawn`]. Resolution may download `uv`, so it is
/// invoked off the async runtime (via `spawn_blocking`) on first spawn, never at
/// app startup. Tests inject a resolver returning a stub spec.
pub type SpawnResolver = Arc<dyn Fn() -> Result<SidecarSpawn, LensError> + Send + Sync>;

/// Bound on the one-time model-load handshake: large (30 min) to survive a slow
/// first-run ~5 GB download over a modest link; a warm restart returns as soon
/// as `{"ready":true}` prints, so the ceiling only caps the worst case.
const READY_TIMEOUT: Duration = Duration::from_secs(1800);

/// Per-turn reply-read ceiling. A mid-write cancel can leave the child mid-synth;
/// without a bound the read (and the whole overview) would hang forever, so on
/// elapse we treat the child as dead and respawn a clean one for the next turn.
const SYNTH_TIMEOUT: Duration = Duration::from_secs(300);

/// Byte ceiling on a single reply line so a runaway/never-newline child cannot
/// OOM the host; a real reply (id + temp-WAV path) is well under this.
const MAX_REPLY_BYTES: u64 = 64 * 1024;

/// A live sidecar: the child process plus its owned stdin sink and buffered stdout
/// source. Held together so acquiring the cell's guard grants exclusive IO access.
type ChildTriple = (Child, ChildStdin, BufReader<ChildStdout>);

/// The default host/guest preset voices (stable ids; the picker to switch among
/// the presets is #194). Applied when the config leaves a voice unset — a male
/// host (`aiden`) and a female guest (`serena`), the audition-selected pair.
pub fn default_qwen_voices() -> VoiceConfig {
    VoiceConfig {
        host: VoiceRef::Named("aiden".to_string()),
        guest: VoiceRef::Named("serena".to_string()),
    }
}

/// Distinguishes a recoverable sidecar-side failure (child stays warm) from true
/// child death (must respawn) so `synthesize_turn` reacts correctly.
#[derive(Debug)]
enum TurnError {
    /// Child died (EOF / broken pipe / exit): kill+clear the cell, respawn next turn.
    Dead,
    /// Sidecar returned `ok:false` or an undecodable reply: child stays warm.
    Failed,
}

/// Generic, detail-free TTS error. Used for every host failure so no binary path
/// or internal source error crosses the IPC boundary.
fn tts_err() -> LensError {
    LensError::Tts("text-to-speech synthesis failed".into())
}

#[derive(serde::Deserialize)]
struct ReadyMsg {
    #[serde(default)]
    ready: bool,
}

#[derive(serde::Deserialize)]
struct SynthReply {
    id: u64,
    #[serde(default)]
    ok: bool,
    #[serde(default)]
    path: Option<String>,
}

pub struct QwenSidecar {
    /// Yields the launch spec on first spawn (may download `uv`); see [`SpawnResolver`].
    resolver: SpawnResolver,
    child: Mutex<Option<ChildTriple>>,
    ready: AtomicBool,
    /// Lifetime-monotonic request id; never resets. See `run_turn_locked`/
    /// [`drain_until`] for how stale (smaller-id) replies are drained.
    next_id: AtomicU64,
    invocation_id: String,
}

impl QwenSidecar {
    pub fn new(resolver: SpawnResolver) -> Self {
        Self {
            resolver,
            child: Mutex::new(None),
            ready: AtomicBool::new(false),
            next_id: AtomicU64::new(1),
            invocation_id: format!("qwen-sidecar-{}", uuid::Uuid::now_v7()),
        }
    }

    /// Resolves the per-speaker voice to `(speaker_id, instruct)`: a `Named(id)`
    /// maps through lens-core's preset catalog, an unset ref falls back to
    /// [`default_qwen_voices`]. A `Reference` clip is a typed error (unsupported).
    fn resolve_voice(
        &self,
        speaker: Speaker,
        voices: &VoiceConfig,
    ) -> Result<(String, String), LensError> {
        let requested = match speaker {
            Speaker::Host => &voices.host,
            Speaker::Guest => &voices.guest,
        };
        let effective = if requested.is_unset() {
            let defaults = default_qwen_voices();
            match speaker {
                Speaker::Host => defaults.host,
                Speaker::Guest => defaults.guest,
            }
        } else {
            requested.clone()
        };

        match effective {
            VoiceRef::Named(id) => {
                let voice = qwen_voice(&id).ok_or_else(tts_err)?;
                Ok((voice.id.to_string(), voice.instruct.to_string()))
            }
            VoiceRef::Reference { .. } => Err(tts_err()),
        }
    }

    /// Spawns the child and awaits its `{"ready":true}` line on the already-held
    /// cell guard (shared init for both `start` and the lazy synth path). On
    /// handshake failure the local `child` drops (kill_on_drop) and the cell stays empty.
    async fn spawn_locked(&self, cell: &mut Option<ChildTriple>) -> Result<(), LensError> {
        if cell.is_some() {
            return Ok(());
        }
        self.ready.store(false, Ordering::SeqCst);
        tracing::debug!(invocation_id = %self.invocation_id, "spawning Qwen3-TTS sidecar");

        // Resolve the launch spec off the async runtime: the first resolution may
        // download `uv`, which must not stall a runtime worker thread.
        let resolver = self.resolver.clone();
        let spec = tokio::task::spawn_blocking(move || resolver())
            .await
            .map_err(|e| {
                tracing::warn!(error = %e, "Qwen spawn resolver task panicked/cancelled");
                tts_err()
            })??;

        let mut child = Command::new(&spec.program)
            .args(&spec.args)
            .envs(spec.envs)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Sidecar diagnostics go to our stderr — never onto the JSON pipe.
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                tracing::warn!(error = %e, "failed to spawn Qwen sidecar");
                tts_err()
            })?;

        let stdin = child.stdin.take().ok_or_else(tts_err)?;
        let stdout = child.stdout.take().ok_or_else(tts_err)?;
        let mut reader = BufReader::new(stdout);

        // We hold the cell guard across this bounded, cancellable wait (acceptable
        // for a one-time cold start); app-quit force-kills via `kill_on_drop`.
        tracing::info!(invocation_id = %self.invocation_id, "awaiting Qwen sidecar readiness (first run may download ~5 GB)");
        let handshake = timeout(READY_TIMEOUT, async {
            let mut line = String::new();
            loop {
                line.clear();
                // Cap the handshake line so a never-newline child can't OOM us.
                let n = (&mut reader)
                    .take(MAX_REPLY_BYTES)
                    .read_line(&mut line)
                    .await
                    .map_err(|e| {
                        tracing::warn!(error = %e, "Qwen handshake read failed");
                        tts_err()
                    })?;
                if n == 0 {
                    tracing::warn!("Qwen sidecar closed stdout before ready (EOF)");
                    return Err(tts_err()); // EOF before ready — model load failed/exited
                }
                if !line.ends_with('\n') {
                    tracing::warn!(bytes = n, "Qwen handshake line exceeded cap or truncated");
                    return Err(tts_err());
                }
                if let Ok(msg) = serde_json::from_str::<ReadyMsg>(line.trim())
                    && msg.ready
                {
                    return Ok::<(), LensError>(());
                }
                // Non-ready stdout lines (sidecar logs) are ignored until ready.
            }
        })
        .await;

        if !matches!(handshake, Ok(Ok(()))) {
            tracing::warn!(invocation_id = %self.invocation_id, "Qwen sidecar handshake failed (timeout/EOF/over-cap)");
            return Err(tts_err());
        }

        self.ready.store(true, Ordering::SeqCst);
        *cell = Some((child, stdin, reader));
        Ok(())
    }

    /// Kills and clears the child so the next turn respawns cleanly.
    fn kill_locked(&self, cell: &mut Option<ChildTriple>) {
        self.ready.store(false, Ordering::SeqCst);
        if let Some((mut child, _stdin, _reader)) = cell.take()
            && let Err(e) = child.start_kill()
        {
            tracing::warn!(error = %e, "failed to kill Qwen sidecar child");
        }
    }

    /// Validates a sidecar-returned WAV path before any read/delete: it must be a
    /// `qwen-turn-*.wav` whose canonical location lives under the OS temp dir.
    /// Anything else is rejected untouched (defends against arbitrary-file access).
    fn is_valid_turn_wav(path: &Path) -> bool {
        let name_ok = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with("qwen-turn-") && n.ends_with(".wav"))
            .unwrap_or(false);
        if !name_ok {
            return false;
        }
        let (Ok(canon), Ok(tmp)) = (path.canonicalize(), std::env::temp_dir().canonicalize())
        else {
            return false;
        };
        canon.starts_with(&tmp)
    }

    /// Writes one synth request, then drains the reply via [`drain_until`] (see
    /// its doc for the stale/EOF/timeout handling), bounded by [`SYNTH_TIMEOUT`].
    async fn run_turn_locked(
        &self,
        cell: &mut Option<ChildTriple>,
        id: u64,
        text: &str,
        speaker: &str,
        instruct: &str,
    ) -> Result<AudioBuffer, TurnError> {
        let (_, stdin, reader) = cell.as_mut().ok_or(TurnError::Dead)?;

        // "auto" until #28/#161 — see validate_qwen_language in lens-core/src/tts/catalog.rs.
        let request = serde_json::json!({
            "id": id,
            "op": "synth",
            "text": text,
            "speaker": speaker,
            "instruct": instruct,
            "language": "auto",
        });
        let mut line = serde_json::to_string(&request).map_err(|e| {
            tracing::warn!(error = %e, "failed to serialize Qwen synth request");
            TurnError::Failed
        })?;
        line.push('\n');
        stdin.write_all(line.as_bytes()).await.map_err(|e| {
            tracing::warn!(error = %e, "failed to write Qwen synth request");
            TurnError::Dead
        })?;
        stdin.flush().await.map_err(|e| {
            tracing::warn!(error = %e, "failed to flush Qwen synth request");
            TurnError::Dead
        })?;

        let reply = match timeout(SYNTH_TIMEOUT, drain_until(reader, id)).await {
            Ok(r) => r?,
            Err(_) => {
                tracing::warn!(id, "Qwen synth reply timed out");
                return Err(TurnError::Dead);
            }
        };

        if !reply.ok {
            tracing::warn!(id, "Qwen sidecar returned ok:false for synth");
            return Err(TurnError::Failed);
        }
        let path = reply.path.ok_or(TurnError::Failed)?;
        if !Self::is_valid_turn_wav(Path::new(&path)) {
            tracing::warn!("Qwen sidecar returned an out-of-policy WAV path; rejecting");
            return Err(TurnError::Failed);
        }
        let audio = read_wav_mono16(Path::new(&path));
        let _ = std::fs::remove_file(&path); // best-effort cleanup, both paths
        audio.map_err(|e| {
            tracing::warn!(error = %e, "failed to decode Qwen turn WAV");
            TurnError::Failed
        })
    }
}

/// Pure reply-drain: reads lines from `reader` until one echoes `id`, against a
/// still-warm child. Unparseable lines (a truncated JSON tail from a cancelled
/// turn) resync on the next newline; stale replies (a mismatched id) are skipped
/// and their temp WAV best-effort deleted so a cancelled turn's file does not
/// leak. EOF and an over-cap line (no newline within [`MAX_REPLY_BYTES`]) map to
/// [`TurnError::Dead`]. The [`SYNTH_TIMEOUT`] bound stays in the caller so this
/// fn stays pure (and deterministically testable without it).
async fn drain_until<R: tokio::io::AsyncBufRead + Unpin>(
    reader: &mut R,
    id: u64,
) -> Result<SynthReply, TurnError> {
    loop {
        let mut buf = String::new();
        let n = (&mut *reader)
            .take(MAX_REPLY_BYTES)
            .read_line(&mut buf)
            .await
            .map_err(|e| {
                tracing::warn!(error = %e, "Qwen sidecar reply read failed");
                TurnError::Dead
            })?;
        if n == 0 {
            tracing::warn!("Qwen sidecar closed stdout (EOF) awaiting reply");
            return Err(TurnError::Dead); // EOF — child exited
        }
        if !buf.ends_with('\n') {
            tracing::warn!(
                bytes = n,
                "Qwen reply exceeded cap or truncated; treating as dead"
            );
            return Err(TurnError::Dead);
        }
        let reply: SynthReply = match serde_json::from_str(buf.trim()) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(error = %e, "skipping unparseable Qwen reply line");
                continue; // truncated tail from a cancelled turn — resync next line
            }
        };
        if reply.id != id {
            // Same allow-list as the matching path: only delete a validated
            // qwen-turn temp WAV, so a rogue/buggy sidecar can't unlink an
            // arbitrary file via a stale reply.
            if let Some(p) = reply.path.as_deref().map(Path::new)
                && QwenSidecar::is_valid_turn_wav(p)
            {
                let _ = std::fs::remove_file(p); // cancelled turn's WAV — don't leak it
            }
            continue; // stale reply from a previously cancelled turn
        }
        return Ok(reply);
    }
}

#[async_trait]
impl TtsSidecar for QwenSidecar {
    async fn start(&self) -> Result<(), LensError> {
        let mut guard = self.child.lock().await;
        self.spawn_locked(&mut guard).await
    }

    async fn stop(&self) -> Result<(), LensError> {
        let mut guard = self.child.lock().await;
        self.ready.store(false, Ordering::SeqCst);
        if let Some((mut child, stdin, _reader)) = guard.take() {
            // Closing stdin ends the sidecar's read loop (graceful); the kill+wait
            // then guarantees a reap so no zombie survives shutdown.
            drop(stdin);
            if let Err(e) = child.kill().await {
                tracing::warn!(error = %e, "failed to reap Qwen sidecar on stop");
            }
        }
        Ok(())
    }

    async fn health(&self) -> bool {
        if !self.ready.load(Ordering::SeqCst) {
            return false;
        }
        let mut guard = self.child.lock().await;
        match guard.as_mut() {
            Some((child, _, _)) => matches!(child.try_wait(), Ok(None)),
            None => false,
        }
    }

    async fn synthesize_turn(
        &self,
        turn: &Turn,
        voices: &VoiceConfig,
        cancel: &CancellationToken,
    ) -> Result<AudioBuffer, LensError> {
        if cancel.is_cancelled() {
            return Err(LensError::Cancelled("tts synthesis cancelled".into()));
        }
        let (speaker, instruct) = self.resolve_voice(turn.speaker, voices)?;

        // One guard for the whole turn: turns are sequential in the caller's
        // stitch loop, so a single acquisition prevents request interleaving.
        let mut guard = self.child.lock().await;
        self.spawn_locked(&mut guard).await?;
        // Re-confirm readiness before writing (defensive against a child left
        // mid-load by an earlier cancel); respawn if it is not.
        if !self.ready.load(Ordering::SeqCst) {
            self.kill_locked(&mut guard);
            self.spawn_locked(&mut guard).await?;
        }

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        match self
            .run_turn_locked(&mut guard, id, &turn.text, &speaker, &instruct)
            .await
        {
            Ok(audio) => Ok(audio),
            Err(TurnError::Failed) => Err(tts_err()),
            Err(TurnError::Dead) => {
                self.kill_locked(&mut guard);
                Err(tts_err())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sidecar() -> QwenSidecar {
        // These tests exercise only `resolve_voice`, which never spawns, so the
        // resolver is a stub that is never invoked.
        let resolver: SpawnResolver = Arc::new(|| {
            Ok(SidecarSpawn {
                program: PathBuf::from("/nonexistent/uv"),
                args: vec![],
                envs: vec![],
            })
        });
        QwenSidecar::new(resolver)
    }

    #[test]
    fn resolve_named_voice_maps_speaker_and_instruct() {
        let qwen = sidecar();
        let voices = VoiceConfig {
            host: VoiceRef::Named("dylan".to_string()),
            guest: VoiceRef::Named("serena".to_string()),
        };

        let (speaker, instruct) = qwen
            .resolve_voice(Speaker::Host, &voices)
            .expect("known host id resolves");
        assert_eq!(speaker, "dylan");
        assert!(!instruct.is_empty());

        let (guest_speaker, _) = qwen
            .resolve_voice(Speaker::Guest, &voices)
            .expect("known guest id resolves");
        assert_eq!(guest_speaker, "serena");
    }

    #[test]
    fn resolve_unset_voice_falls_back_to_defaults() {
        let qwen = sidecar();
        let voices = VoiceConfig::default(); // both refs unset

        let (host_speaker, _) = qwen
            .resolve_voice(Speaker::Host, &voices)
            .expect("unset host falls back");
        assert_eq!(host_speaker, "aiden");

        let (guest_speaker, _) = qwen
            .resolve_voice(Speaker::Guest, &voices)
            .expect("unset guest falls back");
        assert_eq!(guest_speaker, "serena");
    }

    #[test]
    fn resolve_unknown_voice_id_errors() {
        let qwen = sidecar();
        let voices = VoiceConfig {
            host: VoiceRef::Named("not-a-voice".to_string()),
            guest: VoiceRef::Named("serena".to_string()),
        };
        assert!(matches!(
            qwen.resolve_voice(Speaker::Host, &voices),
            Err(LensError::Tts(_))
        ));
    }

    #[test]
    fn resolve_reference_clip_is_unsupported() {
        // A `Reference` voice is a typed error, not a silent fallback.
        let qwen = sidecar();
        let voices = VoiceConfig {
            host: VoiceRef::Reference {
                clip_path: PathBuf::from("/some/clip.wav"),
                transcript: "hello".to_string(),
            },
            guest: VoiceRef::Named("serena".to_string()),
        };
        assert!(matches!(
            qwen.resolve_voice(Speaker::Host, &voices),
            Err(LensError::Tts(_))
        ));
    }

    #[test]
    fn default_qwen_voices_uses_preset_ids() {
        let voices = default_qwen_voices();
        assert_eq!(voices.host, VoiceRef::Named("aiden".to_string()));
        assert_eq!(voices.guest, VoiceRef::Named("serena".to_string()));
        // Both defaults must resolve against the static preset catalog.
        assert!(qwen_voice("aiden").is_some());
        assert!(qwen_voice("serena").is_some());
    }

    #[tokio::test]
    async fn drain_skips_stale_complete_reply_then_matches() {
        let data = concat!(
            r#"{"id":1,"ok":true,"path":"/nonexistent/qwen-turn-stale.wav"}"#,
            "\n",
            r#"{"id":2,"ok":true,"path":"/nonexistent/qwen-turn-live.wav"}"#,
            "\n",
        );
        let mut reader = data.as_bytes();
        let reply = drain_until(&mut reader, 2).await.expect("matches id 2");
        assert_eq!(reply.id, 2);
        assert!(reply.ok);
    }

    #[tokio::test]
    async fn drain_resyncs_after_truncated_line() {
        // A truncated JSON tail (still newline-terminated) fails parse and is
        // skipped; the drain resyncs on the next complete line.
        let data = concat!(
            r#"{"id":2,"ok":tru"#,
            "\n",
            r#"{"id":2,"ok":true,"path":"/nonexistent/qwen-turn-x.wav"}"#,
            "\n",
        );
        let mut reader = data.as_bytes();
        let reply = drain_until(&mut reader, 2)
            .await
            .expect("resyncs to the valid line");
        assert_eq!(reply.id, 2);
        assert!(reply.ok);
    }

    #[tokio::test]
    async fn drain_eof_maps_to_dead() {
        // A stale line then EOF before the awaited id → Dead (respawn).
        let data = concat!(r#"{"id":1,"ok":true}"#, "\n");
        let mut reader = data.as_bytes();
        assert!(matches!(
            drain_until(&mut reader, 2).await,
            Err(TurnError::Dead)
        ));
    }

    #[tokio::test]
    async fn drain_over_cap_line_maps_to_dead() {
        // A single line larger than the cap with no newline is a runaway child.
        let big = "x".repeat((MAX_REPLY_BYTES as usize) + 1024);
        let mut reader = big.as_bytes();
        assert!(matches!(
            drain_until(&mut reader, 1).await,
            Err(TurnError::Dead)
        ));
    }

    #[tokio::test]
    async fn drain_returns_matching_ok_false_reply() {
        // A matching-id `ok:false` is returned to the caller, which maps it to
        // `TurnError::Failed` (child stays warm) — the pure fn just surfaces it.
        let data = concat!(r#"{"id":5,"ok":false}"#, "\n");
        let mut reader = data.as_bytes();
        let reply = drain_until(&mut reader, 5)
            .await
            .expect("matching id returns the reply");
        assert_eq!(reply.id, 5);
        assert!(!reply.ok);
    }

    #[test]
    fn valid_turn_wav_accepts_temp_and_rejects_others() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("qwen-turn-abc.wav");
        std::fs::write(&good, b"riff").unwrap();
        assert!(QwenSidecar::is_valid_turn_wav(&good));

        // Wrong name in temp: rejected without touching the fs.
        let bad_name = dir.path().join("evil.wav");
        std::fs::write(&bad_name, b"riff").unwrap();
        assert!(!QwenSidecar::is_valid_turn_wav(&bad_name));

        // Correct name but outside the OS temp dir: rejected.
        assert!(!QwenSidecar::is_valid_turn_wav(Path::new(
            "/etc/qwen-turn-x.wav"
        )));
    }
}

/// End-to-end correctness harness: drives the REAL `QwenSidecar` (real
/// `tokio::process` spawn + JSON-over-stdio + correlation-id drain +
/// `read_wav_mono16` + stitch) through the production `resolve_tts_provider_full`
/// dispatch, against a stdlib-only Python stub that speaks the sidecar contract
/// but needs no MLX. Validates every line of net-new plumbing offline; only real
/// MLX voice quality (D4 audition) remains human-gated. Requires `python3`.
#[cfg(test)]
mod e2e_tests {
    use super::*;
    use lens_core::{
        DialogueScript, Speaker, TtsBackend, TtsConfig, TtsPhase, Turn, resolve_tts_provider_full,
    };
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::{Arc, Mutex as StdMutex};
    use std::time::Duration;

    /// Stub sidecar: honors the exact wire contract (ready line, id-echoed
    /// ping/synth replies, temp-WAV path). Sleeps per-synth only when the `slow`
    /// argv marker is present (the cancel test injects it; no `uv`/MLX needed).
    const STUB: &str = r#"#!/usr/bin/env python3
import sys, json, os, struct, wave, tempfile, math, time
def emit(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
slow = "slow" in sys.argv[1:]
def make_wav():
    fd, p = tempfile.mkstemp(prefix="qwen-turn-", suffix=".wav"); os.close(fd)
    r = 24000; n = int(0.4 * r)
    with wave.open(p, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(r)
        w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(2*math.pi*220*i/r))) for i in range(n)))
    return p
emit({"ready": True})
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except Exception:
        continue
    rid = req.get("id"); op = req.get("op")
    if op == "ping":
        emit({"id": rid, "ok": True, "pong": True})
    elif op == "synth":
        # Pin the wire field name: a renamed `speaker` field must fail the e2e.
        if not isinstance(req.get("speaker"), str) or not req.get("speaker"):
            emit({"id": rid, "ok": False, "error": "missing speaker"}); continue
        if slow:
            time.sleep(1.2)
        try:
            emit({"id": rid, "ok": True, "path": make_wav()})
        except Exception as e:
            emit({"id": rid, "ok": False, "error": str(e)[:200]})
    else:
        emit({"id": rid, "ok": False, "error": "unknown op"})
"#;

    /// Writes the stub to `dir/stub.py` (chmod +x, harmless — it is run as an
    /// argument to `python3`) and returns the path.
    fn write_stub(dir: &Path) -> PathBuf {
        let path = dir.join("stub.py");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(STUB.as_bytes()).unwrap();
        f.flush().unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn two_turn_script() -> DialogueScript {
        DialogueScript {
            turns: vec![
                Turn {
                    speaker: Speaker::Host,
                    text: "Welcome to the overview.".to_string(),
                    emotion: None,
                    source_ids: vec![],
                },
                Turn {
                    speaker: Speaker::Guest,
                    text: "Glad to be here.".to_string(),
                    emotion: None,
                    source_ids: vec![],
                },
            ],
        }
    }

    fn provider_over_stub(tmp: &Path, slow: bool) -> Arc<dyn lens_core::TtsProvider> {
        let stub = write_stub(tmp);
        // Inject a resolver that returns the "direct spawn" spec: run the stdlib
        // stub under the system `python3` (no `uv`/MLX). `slow` is passed as an
        // argv marker so the cancel test can force a mid-turn sleep.
        let mut args = vec![stub.to_string_lossy().into_owned()];
        if slow {
            args.push("slow".to_string());
        }
        let resolver: SpawnResolver = Arc::new(move || {
            Ok(SidecarSpawn {
                program: PathBuf::from("python3"),
                args: args.clone(),
                envs: vec![],
            })
        });
        let sidecar: Arc<dyn TtsSidecar> = Arc::new(QwenSidecar::new(resolver));
        resolve_tts_provider_full(
            TtsBackend::Qwen3Local,
            &TtsConfig::default(),
            Path::new("/unused"),
            Some(sidecar),
        )
        .expect("Qwen3Local resolves to an adapter when a sidecar is present")
    }

    fn phase_sink() -> (
        Arc<StdMutex<Vec<TtsPhase>>>,
        impl Fn(TtsPhase) + Send + Sync,
    ) {
        let phases = Arc::new(StdMutex::new(Vec::new()));
        let sink = phases.clone();
        (phases, move |p: TtsPhase| sink.lock().unwrap().push(p))
    }

    #[tokio::test]
    async fn e2e_two_turn_synthesizes_stitched_24k() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = provider_over_stub(tmp.path(), false);
        let (phases, on_phase) = phase_sink();
        let cancel = CancellationToken::new();

        let buf = provider
            .synthesize_script(
                &two_turn_script(),
                &VoiceConfig::default(),
                &on_phase,
                &cancel,
            )
            .await
            .expect("real subprocess synthesis + stitch succeeds");

        assert_eq!(buf.sample_rate, 24_000);
        // Two ~0.4s turns + an inter-speaker gap — comfortably over one turn.
        assert!(
            buf.samples.len() > (0.4 * 24_000.0) as usize,
            "stitched output too short: {} samples",
            buf.samples.len()
        );
        let recorded = phases.lock().unwrap().clone();
        assert_eq!(recorded[0], TtsPhase::Synthesizing { turn: 1, total: 2 });
        assert_eq!(recorded[1], TtsPhase::Synthesizing { turn: 2, total: 2 });
        assert_eq!(recorded[2], TtsPhase::Stitching);
    }

    #[tokio::test]
    async fn e2e_warm_child_serves_two_scripts() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = provider_over_stub(tmp.path(), false);
        let cancel = CancellationToken::new();
        let (_p1, cb1) = phase_sink();
        let (_p2, cb2) = phase_sink();

        provider
            .synthesize_script(&two_turn_script(), &VoiceConfig::default(), &cb1, &cancel)
            .await
            .expect("first script");
        // Same warm child (monotonic ids continue) serves a second script.
        provider
            .synthesize_script(&two_turn_script(), &VoiceConfig::default(), &cb2, &cancel)
            .await
            .expect("second script on the warm child");
    }

    #[tokio::test]
    async fn e2e_midturn_cancel_then_clean_rerun() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = provider_over_stub(tmp.path(), true); // slow sidecar
        let script = two_turn_script();

        // Cancel while turn 1 is in flight (child sleeping 1.2s).
        let cancel = CancellationToken::new();
        let fire = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(500)).await;
            fire.cancel();
        });
        let (_p, cb) = phase_sink();
        let cancelled = provider
            .synthesize_script(&script, &VoiceConfig::default(), &cb, &cancel)
            .await;
        assert!(
            matches!(cancelled, Err(lens_core::LensError::Cancelled(_))),
            "mid-turn cancel must abort, got {cancelled:?}"
        );

        // The stale reply from the cancelled turn is still in the pipe; a fresh
        // run must drain past it (by id) against the still-warm child and succeed.
        let fresh = CancellationToken::new();
        let (_p2, cb2) = phase_sink();
        let recovered = provider
            .synthesize_script(&script, &VoiceConfig::default(), &cb2, &fresh)
            .await;
        assert!(
            recovered.is_ok(),
            "run after cancel must recover via correlation-id drain, got {recovered:?}"
        );
        assert_eq!(recovered.unwrap().sample_rate, 24_000);
    }
}
