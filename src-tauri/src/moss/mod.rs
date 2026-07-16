//! MOSS-TTS-Local sidecar host (issue #193, [161e]): drives an out-of-process
//! MLX Python sidecar over line-delimited JSON-over-stdio as lens-core's
//! [`TtsSidecar`]. Gated to `aarch64-apple-darwin` (the only target MLX runs on).
//!
//! The sidecar runs under `uv` (`uv run` against the bundled `pyproject.toml`);
//! `mlx-speech` auto-downloads the model into the `HF_HOME` cache passed in the
//! [`SidecarSpawn`]. Launch details are resolved lazily via an injected resolver
//! (see [`resolver::resolve_sidecar_spawn`]) so provisioning never blocks startup.
//!
//! All failures map to a generic [`LensError::Tts`] (no path/internal detail —
//! it crosses the Tauri IPC boundary); real errors are logged server-side only.

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
    AudioBuffer, LensError, Speaker, TtsSidecar, Turn, VoiceConfig, VoiceRef, moss_reference_voice,
    read_wav_mono16,
};

mod resolver;
pub use resolver::resolve_sidecar_spawn;

/// A fully-resolved sidecar launch: the program to exec, its args, and the extra
/// environment it needs (e.g. `HF_HOME` so `huggingface_hub` caches into app-data).
/// Produced by a [`SpawnResolver`] so [`MossSidecar`] stays testable with a stub.
pub struct SidecarSpawn {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub envs: Vec<(String, String)>,
}

/// Lazily produces the [`SidecarSpawn`]. Resolution may download `uv`, so it is
/// invoked off the async runtime (via `spawn_blocking`) on first spawn, never at
/// app startup. Tests inject a resolver returning a stub spec.
pub type SpawnResolver = Arc<dyn Fn() -> Result<SidecarSpawn, LensError> + Send + Sync>;

/// Bound on the one-time model-load handshake. A cold first run also provisions
/// the `uv` env and `mlx-speech` lazily fetches ~5 GB of weights, so the ceiling
/// is deliberately large (30 min) to survive a slow first-run download on a
/// modest link; a genuine hang is still bounded, and user-cancel interrupts it.
/// A warm restart returns as soon as `{"ready":true}` prints, so the large
/// ceiling only caps the worst case. Interrupted first runs resume from the
/// `HF_HOME`/uv caches on retry rather than re-downloading.
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

/// The bundled default host/guest voices (stable ids, not paths — the picker to
/// switch among the four bundled clips is #194). Applied when the config leaves a
/// voice unset: MOSS is clone-only, so every turn needs a real reference clip.
pub fn default_moss_voices() -> VoiceConfig {
    VoiceConfig {
        host: VoiceRef::Named("librivox-chenevert".to_string()),
        guest: VoiceRef::Named("librivox-klett".to_string()),
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

pub struct MossSidecar {
    /// Yields the launch spec on first spawn (may download `uv`); see [`SpawnResolver`].
    resolver: SpawnResolver,
    /// Directory containing the bundled `voices/` reference clips.
    resource_dir: PathBuf,
    child: Mutex<Option<ChildTriple>>,
    ready: AtomicBool,
    /// Lifetime-monotonic request id; never resets. See `run_turn_locked`/
    /// [`drain_until`] for how stale (smaller-id) replies are drained.
    next_id: AtomicU64,
    invocation_id: String,
}

impl MossSidecar {
    pub fn new(resolver: SpawnResolver, resource_dir: PathBuf) -> Self {
        Self {
            resolver,
            resource_dir,
            child: Mutex::new(None),
            ready: AtomicBool::new(false),
            next_id: AtomicU64::new(1),
            invocation_id: format!("moss-sidecar-{}", uuid::Uuid::now_v7()),
        }
    }

    /// Resolves the per-speaker voice to `(clip_path, transcript)`: a `Named(id)`
    /// maps through lens-core's catalog + this host's resource dir, a `Reference`
    /// passes through, an unset ref falls back to [`default_moss_voices`].
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
            let defaults = default_moss_voices();
            match speaker {
                Speaker::Host => defaults.host,
                Speaker::Guest => defaults.guest,
            }
        } else {
            requested.clone()
        };

        match effective {
            VoiceRef::Named(id) => {
                let voice = moss_reference_voice(&id).ok_or_else(tts_err)?;
                let clip = self.resource_dir.join("voices").join(voice.clip_filename);
                Ok((
                    clip.to_string_lossy().into_owned(),
                    voice.transcript.to_string(),
                ))
            }
            VoiceRef::Reference {
                clip_path,
                transcript,
            } => Ok((clip_path.to_string_lossy().into_owned(), transcript)),
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
        tracing::debug!(invocation_id = %self.invocation_id, "spawning MOSS-TTS-Local sidecar");

        // Resolve the launch spec off the async runtime: the first resolution may
        // download `uv`, which must not stall a runtime worker thread.
        let resolver = self.resolver.clone();
        let spec = tokio::task::spawn_blocking(move || resolver())
            .await
            .map_err(|e| {
                tracing::warn!(error = %e, "MOSS spawn resolver task panicked/cancelled");
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
                tracing::warn!(error = %e, "failed to spawn MOSS sidecar");
                tts_err()
            })?;

        let stdin = child.stdin.take().ok_or_else(tts_err)?;
        let stdout = child.stdout.take().ok_or_else(tts_err)?;
        let mut reader = BufReader::new(stdout);

        // The first run downloads the uv env + ~5 GB of weights before the child
        // prints `{"ready":true}` (progress goes to inherited stderr). We hold the
        // cell guard across this bounded, cancellable wait — acceptable for a
        // one-time cold start; app-quit force-kills via `kill_on_drop`.
        tracing::info!(invocation_id = %self.invocation_id, "awaiting MOSS sidecar readiness (first run may download ~5 GB)");
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
                        tracing::warn!(error = %e, "MOSS handshake read failed");
                        tts_err()
                    })?;
                if n == 0 {
                    tracing::warn!("MOSS sidecar closed stdout before ready (EOF)");
                    return Err(tts_err()); // EOF before ready — model load failed/exited
                }
                if !line.ends_with('\n') {
                    tracing::warn!(bytes = n, "MOSS handshake line exceeded cap or truncated");
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
            tracing::warn!(invocation_id = %self.invocation_id, "MOSS sidecar handshake failed (timeout/EOF/over-cap)");
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
            tracing::warn!(error = %e, "failed to kill MOSS sidecar child");
        }
    }

    /// Validates a sidecar-returned WAV path before any read/delete: it must be a
    /// `moss-turn-*.wav` whose canonical location lives under the OS temp dir.
    /// Anything else is rejected untouched (defends against arbitrary-file access).
    fn is_valid_turn_wav(path: &Path) -> bool {
        let name_ok = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with("moss-turn-") && n.ends_with(".wav"))
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

    /// Writes one synth request and drains replies (via [`drain_until`]) until the
    /// echoed id matches, bounded by [`SYNTH_TIMEOUT`]. Stale/unparseable lines are
    /// skipped against the still-warm child; EOF/timeout/over-cap map to [`TurnError::Dead`].
    async fn run_turn_locked(
        &self,
        cell: &mut Option<ChildTriple>,
        id: u64,
        text: &str,
        clip: &str,
        transcript: &str,
    ) -> Result<AudioBuffer, TurnError> {
        let (_, stdin, reader) = cell.as_mut().ok_or(TurnError::Dead)?;

        let request = serde_json::json!({
            "id": id,
            "op": "synth",
            "text": text,
            "emotion": serde_json::Value::Null,
            "ref_clip": clip,
            "ref_transcript": transcript,
            "audio_temperature": 1.0,
        });
        let mut line = serde_json::to_string(&request).map_err(|e| {
            tracing::warn!(error = %e, "failed to serialize MOSS synth request");
            TurnError::Failed
        })?;
        line.push('\n');
        stdin.write_all(line.as_bytes()).await.map_err(|e| {
            tracing::warn!(error = %e, "failed to write MOSS synth request");
            TurnError::Dead
        })?;
        stdin.flush().await.map_err(|e| {
            tracing::warn!(error = %e, "failed to flush MOSS synth request");
            TurnError::Dead
        })?;

        // Bound the drain so a mid-synth child (e.g. after a mid-write cancel)
        // cannot hang the overview forever; on elapse treat as a dead child.
        let reply = match timeout(SYNTH_TIMEOUT, drain_until(reader, id)).await {
            Ok(r) => r?,
            Err(_) => {
                tracing::warn!(id, "MOSS synth reply timed out");
                return Err(TurnError::Dead);
            }
        };

        if !reply.ok {
            tracing::warn!(id, "MOSS sidecar returned ok:false for synth");
            return Err(TurnError::Failed);
        }
        let path = reply.path.ok_or(TurnError::Failed)?;
        if !Self::is_valid_turn_wav(Path::new(&path)) {
            tracing::warn!("MOSS sidecar returned an out-of-policy WAV path; rejecting");
            return Err(TurnError::Failed);
        }
        let audio = read_wav_mono16(Path::new(&path));
        let _ = std::fs::remove_file(&path); // best-effort cleanup, both paths
        audio.map_err(|e| {
            tracing::warn!(error = %e, "failed to decode MOSS turn WAV");
            TurnError::Failed
        })
    }
}

/// Pure reply-drain: reads lines from `reader` until one echoes `id`, against a
/// still-warm child. Unparseable lines (a truncated JSON tail from a cancelled
/// turn) resync on the next newline; stale replies (a mismatched id) are skipped
/// and their temp WAV best-effort deleted so a cancelled turn's file does not
/// leak. EOF and an over-cap line (no newline within [`MAX_REPLY_BYTES`]) map to
/// [`TurnError::Dead`]. Extracted from `run_turn_locked` for deterministic tests;
/// the [`SYNTH_TIMEOUT`] bound stays in the caller so this fn stays pure.
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
                tracing::warn!(error = %e, "MOSS sidecar reply read failed");
                TurnError::Dead
            })?;
        if n == 0 {
            tracing::warn!("MOSS sidecar closed stdout (EOF) awaiting reply");
            return Err(TurnError::Dead); // EOF — child exited
        }
        if !buf.ends_with('\n') {
            tracing::warn!(
                bytes = n,
                "MOSS reply exceeded cap or truncated; treating as dead"
            );
            return Err(TurnError::Dead);
        }
        let reply: SynthReply = match serde_json::from_str(buf.trim()) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(error = %e, "skipping unparseable MOSS reply line");
                continue; // truncated tail from a cancelled turn — resync next line
            }
        };
        if reply.id != id {
            if let Some(p) = &reply.path {
                let _ = std::fs::remove_file(p); // cancelled turn's WAV — don't leak it
            }
            continue; // stale reply from a previously cancelled turn
        }
        return Ok(reply);
    }
}

#[async_trait]
impl TtsSidecar for MossSidecar {
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
                tracing::warn!(error = %e, "failed to reap MOSS sidecar on stop");
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
        let (clip, transcript) = self.resolve_voice(turn.speaker, voices)?;

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
            .run_turn_locked(&mut guard, id, &turn.text, &clip, &transcript)
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

    fn sidecar_with_resources(resource_dir: PathBuf) -> MossSidecar {
        // These tests exercise only `resolve_voice`, which never spawns, so the
        // resolver is a stub that is never invoked.
        let resolver: SpawnResolver = Arc::new(|| {
            Ok(SidecarSpawn {
                program: PathBuf::from("/nonexistent/uv"),
                args: vec![],
                envs: vec![],
            })
        });
        MossSidecar::new(resolver, resource_dir)
    }

    #[test]
    fn resolve_named_voice_maps_path_and_transcript() {
        let resource_dir = PathBuf::from("/fake/resources");
        let moss = sidecar_with_resources(resource_dir.clone());
        let voices = VoiceConfig {
            host: VoiceRef::Named("librivox-clarke".to_string()),
            guest: VoiceRef::Named("librivox-savage".to_string()),
        };

        let (clip, transcript) = moss
            .resolve_voice(Speaker::Host, &voices)
            .expect("known host id resolves");
        assert_eq!(clip, "/fake/resources/voices/librivox-clarke.wav");
        assert_eq!(
            transcript,
            moss_reference_voice("librivox-clarke").unwrap().transcript
        );

        let (guest_clip, _) = moss
            .resolve_voice(Speaker::Guest, &voices)
            .expect("known guest id resolves");
        assert_eq!(guest_clip, "/fake/resources/voices/librivox-savage.wav");
    }

    #[test]
    fn resolve_unset_voice_falls_back_to_defaults() {
        let moss = sidecar_with_resources(PathBuf::from("/fake/resources"));
        let voices = VoiceConfig::default(); // both refs unset

        let (host_clip, _) = moss
            .resolve_voice(Speaker::Host, &voices)
            .expect("unset host falls back");
        assert_eq!(host_clip, "/fake/resources/voices/librivox-chenevert.wav");

        let (guest_clip, _) = moss
            .resolve_voice(Speaker::Guest, &voices)
            .expect("unset guest falls back");
        assert_eq!(guest_clip, "/fake/resources/voices/librivox-klett.wav");
    }

    #[test]
    fn resolve_unknown_voice_id_errors() {
        let moss = sidecar_with_resources(PathBuf::from("/fake/resources"));
        let voices = VoiceConfig {
            host: VoiceRef::Named("not-a-voice".to_string()),
            guest: VoiceRef::Named("librivox-klett".to_string()),
        };
        assert!(matches!(
            moss.resolve_voice(Speaker::Host, &voices),
            Err(LensError::Tts(_))
        ));
    }

    #[test]
    fn default_moss_voices_uses_bundled_ids() {
        let voices = default_moss_voices();
        assert_eq!(
            voices.host,
            VoiceRef::Named("librivox-chenevert".to_string())
        );
        assert_eq!(voices.guest, VoiceRef::Named("librivox-klett".to_string()));
        // Both defaults must resolve against the static catalog.
        assert!(moss_reference_voice("librivox-chenevert").is_some());
        assert!(moss_reference_voice("librivox-klett").is_some());
    }

    #[tokio::test]
    async fn drain_skips_stale_complete_reply_then_matches() {
        let data = concat!(
            r#"{"id":1,"ok":true,"path":"/nonexistent/moss-turn-stale.wav"}"#,
            "\n",
            r#"{"id":2,"ok":true,"path":"/nonexistent/moss-turn-live.wav"}"#,
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
            r#"{"id":2,"ok":true,"path":"/nonexistent/moss-turn-x.wav"}"#,
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
        let good = dir.path().join("moss-turn-abc.wav");
        std::fs::write(&good, b"riff").unwrap();
        assert!(MossSidecar::is_valid_turn_wav(&good));

        // Wrong name in temp: rejected without touching the fs.
        let bad_name = dir.path().join("evil.wav");
        std::fs::write(&bad_name, b"riff").unwrap();
        assert!(!MossSidecar::is_valid_turn_wav(&bad_name));

        // Correct name but outside the OS temp dir: rejected.
        assert!(!MossSidecar::is_valid_turn_wav(Path::new(
            "/etc/moss-turn-x.wav"
        )));
    }
}

/// End-to-end correctness harness: drives the REAL `MossSidecar` (real
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
    fd, p = tempfile.mkstemp(prefix="moss-turn-", suffix=".wav"); os.close(fd)
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
        let sidecar: Arc<dyn TtsSidecar> = Arc::new(MossSidecar::new(resolver, tmp.to_path_buf()));
        resolve_tts_provider_full(
            TtsBackend::MossLocal,
            &TtsConfig::default(),
            Path::new("/unused"),
            Some(sidecar),
        )
        .expect("MossLocal resolves to an adapter when a sidecar is present")
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
