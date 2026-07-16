#!/usr/bin/env python3
"""Qwen3-TTS CustomVoice IPC sidecar for LensLM (161e).

Spawned by src-tauri's QwenSidecar as `uv run --project <this dir> python
qwen3_tts_sidecar.py`, driven over stdio as line-delimited JSON. Apple-Silicon
only (MLX requires it). See README.md for the wire contract and runtime.

CustomVoice is a fixed-preset-speaker + instruct-string engine: no reference
clip, no transcript, no ASR — the speaker id selects a bundled voice and the
instruct string steers its delivery.
"""

import json
import math
import sys
import tempfile
import wave

import numpy as np

# Community MLX weights (Apache-2.0). `mlx-audio` fetches this lazily into the
# HF cache on first synth (~4.5 GB); the Rust side sets HF_HOME to app-data.
MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"

# Symmetric guards fall back to these when a request omits or malforms a param.
# The Rust host currently sends neither, so these are the effective values.
DEFAULT_TEMPERATURE = 0.9
DEFAULT_MAX_TOKENS = 4096


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
    # No longer hardcoded to English — Qwen3Local is multilingual.
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


def main() -> None:
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


if __name__ == "__main__":
    main()
