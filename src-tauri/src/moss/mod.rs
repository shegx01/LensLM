//! MOSS-TTS-Local sidecar host (issue #193, [161e]): [`MossSidecar`] drives an
//! out-of-process MLX Python sidecar (`sidecar/mlx-speech`, frozen to a binary)
//! over line-delimited JSON-over-stdio, implementing lens-core's [`TtsSidecar`]
//! trait so the headless engine synthesizes a #26 `DialogueScript` without ever
//! seeing Python/MLX. This whole module is gated to `aarch64-apple-darwin` (the
//! only target MLX runs on); it compiles out everywhere else.
//!
//! IPC invariant: all process/IO/parse failures map to a GENERIC
//! [`LensError::Tts`] with no binary path or internal detail — the error crosses
//! the Tauri boundary, so it must never leak host internals. Audio never rides
//! the pipe: a synth reply carries a temp-WAV path, decoded here via lens-core's
//! `read_wav_mono16` and deleted (best-effort) after read.
//!
//! Teardown: the child is spawned with `kill_on_drop(true)` (backstop) and an
//! explicit `stop()` reap. A stale/truncated reply line after a mid-turn cancel is
//! NOT a poison event — the correlation-id drain skips it against the still-warm
//! child; only true child death (EOF / broken pipe / exit / start timeout) clears
//! the cell so the next turn respawns a clean process.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::time::{Duration, timeout};
use tokio_util::sync::CancellationToken;

use lens_core::{
    AudioBuffer, LensError, Speaker, TtsSidecar, Turn, VoiceConfig, VoiceRef, moss_reference_voice,
    read_wav_mono16,
};

/// Bound on the one-time model-load handshake. The int8 model is ~3.3 GB, so a
/// cold load can take tens of seconds; a generous ceiling avoids a false failure
/// while still guaranteeing the lazy `start` cannot hang forever.
const READY_TIMEOUT: Duration = Duration::from_secs(300);

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
    binary_path: PathBuf,
    model_dir: PathBuf,
    /// Directory containing the bundled `voices/` reference clips.
    resource_dir: PathBuf,
    child: Mutex<Option<ChildTriple>>,
    ready: AtomicBool,
    /// Lifetime-monotonic request id. Never resets — it outlives any child, so a
    /// stale reply from a respawned-over turn always carries a strictly smaller id
    /// and is skipped by the correlation drain.
    next_id: AtomicU64,
    invocation_id: String,
}

impl MossSidecar {
    pub fn new(binary_path: PathBuf, model_dir: PathBuf, resource_dir: PathBuf) -> Self {
        Self {
            binary_path,
            model_dir,
            resource_dir,
            child: Mutex::new(None),
            ready: AtomicBool::new(false),
            next_id: AtomicU64::new(1),
            invocation_id: format!("moss-sidecar-{}", uuid::Uuid::now_v7()),
        }
    }

    /// Resolves the per-speaker voice to `(clip_path, transcript)`. A `Named(id)`
    /// maps through lens-core's static catalog (transcript) + this host's bundled
    /// resource dir (path); a `Reference` passes through directly. An unset ref
    /// falls back to [`default_moss_voices`]; an unknown id is a generic error.
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

    /// Spawns the child and awaits its `{"ready":true}` line, operating on the
    /// already-held cell guard. Lock-free by construction — the caller holds the
    /// guard — so it is the shared init for both `start` and the lazy synth path
    /// with no reentrant lock. On timeout/handshake failure the local `child`
    /// drops (killed via `kill_on_drop`) and the cell stays empty for a clean
    /// respawn.
    async fn spawn_locked(&self, cell: &mut Option<ChildTriple>) -> Result<(), LensError> {
        if cell.is_some() {
            return Ok(());
        }
        self.ready.store(false, Ordering::SeqCst);
        tracing::debug!(invocation_id = %self.invocation_id, "spawning MOSS-TTS-Local sidecar");

        let mut child = Command::new(&self.binary_path)
            .arg("--model-dir")
            .arg(&self.model_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Sidecar diagnostics go to our stderr — never onto the JSON pipe.
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|_| tts_err())?;

        let stdin = child.stdin.take().ok_or_else(tts_err)?;
        let stdout = child.stdout.take().ok_or_else(tts_err)?;
        let mut reader = BufReader::new(stdout);

        let handshake = timeout(READY_TIMEOUT, async {
            let mut line = String::new();
            loop {
                line.clear();
                let n = reader.read_line(&mut line).await.map_err(|_| tts_err())?;
                if n == 0 {
                    return Err(tts_err()); // EOF before ready — model load failed/exited
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
            return Err(tts_err());
        }

        self.ready.store(true, Ordering::SeqCst);
        *cell = Some((child, stdin, reader));
        Ok(())
    }

    /// Kills and clears the child so the next turn respawns cleanly. Reserved for
    /// true child death — a merely stale reply is skipped, not a poison event.
    fn kill_locked(&self, cell: &mut Option<ChildTriple>) {
        self.ready.store(false, Ordering::SeqCst);
        if let Some((mut child, _stdin, _reader)) = cell.take() {
            let _ = child.start_kill();
        }
    }

    /// Writes one synth request and drains replies until the echoed id matches.
    /// Unparseable lines (a truncated tail from a cancelled turn) and replies with
    /// a smaller id (stale, since ids strictly increase) are skipped against the
    /// still-warm child; EOF/broken-pipe map to [`TurnError::Dead`].
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
        let mut line = serde_json::to_string(&request).map_err(|_| TurnError::Failed)?;
        line.push('\n');
        stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|_| TurnError::Dead)?;
        stdin.flush().await.map_err(|_| TurnError::Dead)?;

        loop {
            let mut buf = String::new();
            let n = reader
                .read_line(&mut buf)
                .await
                .map_err(|_| TurnError::Dead)?;
            if n == 0 {
                return Err(TurnError::Dead); // EOF — child exited
            }
            let reply: SynthReply = match serde_json::from_str(buf.trim()) {
                Ok(r) => r,
                Err(_) => continue, // truncated tail from a cancelled turn — resync next line
            };
            if reply.id != id {
                continue; // stale reply from a previously cancelled turn
            }
            if !reply.ok {
                return Err(TurnError::Failed);
            }
            let path = reply.path.ok_or(TurnError::Failed)?;
            let audio = read_wav_mono16(Path::new(&path));
            let _ = std::fs::remove_file(&path); // best-effort cleanup, both paths
            return audio.map_err(|_| TurnError::Failed);
        }
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
            let _ = child.kill().await;
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
        MossSidecar::new(
            PathBuf::from("/nonexistent/bin/mlx-speech-sidecar"),
            PathBuf::from("/nonexistent/models/moss"),
            resource_dir,
        )
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
    /// ping/synth replies, temp-WAV path). Sleeps per-synth only when a `SLOW`
    /// marker file exists in `--model-dir` (parallel-safe, no shared env).
    const STUB: &str = r#"#!/usr/bin/env python3
import sys, json, os, struct, wave, tempfile, math, time
def emit(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
model_dir = ""
argv = sys.argv[1:]
for i, x in enumerate(argv):
    if x == "--model-dir" and i + 1 < len(argv):
        model_dir = argv[i + 1]
slow = os.path.exists(os.path.join(model_dir, "SLOW"))
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

    /// Writes the stub to `dir/stub.py` (chmod +x) so `MossSidecar` can spawn it
    /// directly via its shebang, and returns the path.
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
        let model_dir = tmp.join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        if slow {
            std::fs::write(model_dir.join("SLOW"), b"1").unwrap();
        }
        let sidecar: Arc<dyn TtsSidecar> =
            Arc::new(MossSidecar::new(stub, model_dir, tmp.to_path_buf()));
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
            .synthesize_script(&two_turn_script(), &VoiceConfig::default(), &on_phase, &cancel)
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
