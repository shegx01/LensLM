#!/usr/bin/env python3
"""Qwen3-TTS CustomVoice IPC sidecar for LensLM (161e).

Spawned by src-tauri's QwenSidecar as `uv run --project <this dir> python
qwen3_tts_sidecar.py`, driven over stdio as line-delimited JSON. Apple-Silicon
only (MLX requires it). See README.md for the wire contract and runtime.

CustomVoice is a fixed-preset-speaker + instruct-string engine: no reference
clip, no transcript, no ASR — the speaker id selects a bundled voice and the
instruct string steers its delivery.

Two modes: the default serve loop (synthesis over stdio), and a one-shot
`--prepare` mode (#194) that explicitly downloads the model, streaming
`{"progress":{"received","total"}}` lines and a final `{"done":true}`, then
exits — without importing MLX or loading the model.
"""

import json
import math
import os
import sys
import tempfile
import threading
import wave

import numpy as np

# Community MLX weights (Apache-2.0). `mlx-audio` fetches this lazily into the
# HF cache on first synth (~4.5 GB); the Rust side sets HF_HOME to app-data.
MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"

# huggingface_hub's on-disk cache subdir for MODEL_ID (under `$HF_HOME/hub`).
# Kept in lockstep with the Rust presence check (`qwen::QWEN_SNAPSHOT_DIR`).
CACHE_SUBDIR = "models--" + MODEL_ID.replace("/", "--")

# Symmetric guards fall back to these when a request omits or malforms a param.
# The Rust host currently sends neither, so these are the effective values.
DEFAULT_TEMPERATURE = 0.9
DEFAULT_MAX_TOKENS = 4096

# Prepare-mode progress cadence: poll the cache dir this often (seconds) and emit
# only when cumulative bytes advanced by at least this many bytes, so a multi-GB
# pull streams ~progress without spamming the pipe.
PREPARE_POLL_INTERVAL = 1.0
PREPARE_MIN_EMIT_BYTES = 2 * 1024 * 1024


def load_qwen():
    """Load the Qwen3-TTS CustomVoice model and its case-folded speaker map.

    The `mlx_audio` import is deferred here (not module-scope) so the offline
    `resolve_gen_params` pytest can import this module without MLX installed.
    """
    from mlx_audio.tts.utils import load_model

    model = load_model(MODEL_ID)
    # Canonical speaker ids are lowercase; fold so a request may name any case.
    speaker_map = {s.lower(): s for s in model.get_supported_speakers()}
    return model, speaker_map


def resolve_gen_params(req: dict) -> tuple[float, int]:
    """Resolve (temperature, max_tokens) from a synth request with symmetric guards.

    `temperature`: a finite number in (0, 2] wins; anything else (absent, bool,
    NaN, out of range, non-numeric) falls back to the default. `max_tokens`: a
    positive non-bool int wins; else the default. Bools are rejected explicitly
    because `bool` is a subclass of `int`/numbers in Python.
    """
    temperature = DEFAULT_TEMPERATURE
    raw_temp = req.get("temperature")
    if isinstance(raw_temp, (int, float)) and not isinstance(raw_temp, bool):
        candidate = float(raw_temp)
        if math.isfinite(candidate) and 0.0 < candidate <= 2.0:
            temperature = candidate

    max_tokens = DEFAULT_MAX_TOKENS
    raw_max = req.get("max_tokens")
    if isinstance(raw_max, int) and not isinstance(raw_max, bool) and raw_max > 0:
        max_tokens = raw_max

    return temperature, max_tokens


def write_wav_mono(waveform: np.ndarray, sample_rate: int) -> str:
    # Writes at the model's actual rate (24 kHz); the Rust side resamples to its
    # target (AudioBuffer::resample_to / stitch_turns), so no rate guard here.
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")

    with tempfile.NamedTemporaryFile(prefix="qwen-turn-", suffix=".wav", delete=False) as f:
        path = f.name
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return path


def send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def handle_synth(model, speaker_map: dict, req: dict) -> str:
    text = req["text"]
    speaker = req.get("speaker")
    if not isinstance(speaker, str) or not speaker:
        raise ValueError("speaker must be a non-empty string")
    canonical = speaker_map.get(speaker.lower())
    if canonical is None:
        raise ValueError(f"unknown speaker: {speaker!r}")

    instruct = req.get("instruct")
    # Language is driven by the Rust host (#194); "auto" lets Qwen3-TTS detect it.
    language = req.get("language", "auto")
    temperature, max_tokens = resolve_gen_params(req)

    results = list(
        model.generate_custom_voice(
            text=text,
            speaker=canonical,
            language=language,
            instruct=instruct,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )
    output = results[0]
    return write_wav_mono(np.array(output.audio), int(output.sample_rate))


def serve() -> None:
    """Default mode: load the model, announce readiness, then serve synth over stdio."""
    try:
        model, speaker_map = load_qwen()
    except Exception as exc:
        sys.stderr.write(f"model load failed: {exc}\n")
        sys.stderr.flush()
        sys.exit(1)

    send({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            # No id to echo a reply against; log and keep the loop alive.
            sys.stderr.write(f"unparseable request line: {line[:200]}\n")
            sys.stderr.flush()
            continue

        req_id = req.get("id")
        op = req.get("op")
        try:
            if op == "ping":
                send({"id": req_id, "ok": True, "pong": True})
            elif op == "synth":
                path = handle_synth(model, speaker_map, req)
                send({"id": req_id, "ok": True, "path": path})
            else:
                send({"id": req_id, "ok": False, "error": f"unknown op: {op}"})
        except Exception as exc:
            send({"id": req_id, "ok": False, "error": str(exc)[:200]})


def parse_prepare(argv: list) -> bool:
    return "--prepare" in argv


def dir_size_bytes(path: str) -> int:
    """Cumulative size of every file under `path` (incl. huggingface_hub's
    `blobs/*.incomplete` partials). Best-effort: unreadable entries count as 0."""
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
    except OSError:
        return total
    return total


def model_cache_dir() -> str:
    """The on-disk cache dir for MODEL_ID under `$HF_HOME/hub` (the Rust resolver
    sets HF_HOME; snapshot_download honors it). Falls back to the HF default home."""
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    return os.path.join(hf_home, "hub", CACHE_SUBDIR)


def _poll_progress(cache_dir, total, stop_event, emit, interval) -> None:
    """Poll `cache_dir` size until `stop_event`, emitting throttled cumulative
    byte progress. Byte counts (not huggingface_hub's per-file tqdm) are the
    source of truth: in huggingface_hub 1.x, `snapshot_download`'s `tqdm_class`
    drives only the outer file-count bar, so disk polling is what yields bytes."""
    last = -1
    while not stop_event.wait(interval):
        received = dir_size_bytes(cache_dir)
        if received > 0 and (last < 0 or received - last >= PREPARE_MIN_EMIT_BYTES):
            last = received
            emit({"progress": {"received": received, "total": total}})


def _compute_total_bytes(hf):
    """Best-effort total download size (bytes); None when metadata is unavailable
    (the Rust/UI side then shows an indeterminate bar)."""
    try:
        info = hf.HfApi().model_info(MODEL_ID, files_metadata=True)
        total = sum(s.size for s in (info.siblings or []) if getattr(s, "size", None))
        return total or None
    except Exception:
        return None


def run_prepare(hf=None, emit=send, interval=PREPARE_POLL_INTERVAL) -> int:
    """One-shot: download the MODEL_ID snapshot into `$HF_HOME`, streaming
    cumulative byte progress, then a final `{"done":true}`. Returns the intended
    process exit code (0 ok, 1 error). MLX is never imported. `hf`/`interval` are
    injectable for offline tests."""
    if hf is None:
        try:
            import huggingface_hub as hf  # noqa: PLC0415 (lazy: avoid MLX/env cost in serve mode)
        except Exception as exc:
            emit({"error": f"huggingface_hub import failed: {exc}"[:200]})
            return 1

    total = _compute_total_bytes(hf)
    cache_dir = model_cache_dir()
    stop_event = threading.Event()
    poller = threading.Thread(
        target=_poll_progress,
        args=(cache_dir, total, stop_event, emit, interval),
        daemon=True,
    )
    poller.start()
    try:
        # snapshot_download honors HF_HOME (set by the resolver) → no explicit cache_dir.
        hf.snapshot_download(MODEL_ID)
    except Exception as exc:
        stop_event.set()
        poller.join(timeout=2)
        emit({"error": str(exc)[:200]})
        return 1

    stop_event.set()
    poller.join(timeout=2)
    # Deterministic final tick at the true on-disk size, then completion.
    emit({"progress": {"received": dir_size_bytes(cache_dir), "total": total}})
    emit({"done": True})
    return 0


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if parse_prepare(argv):
        sys.exit(run_prepare())
    serve()


if __name__ == "__main__":
    main()
