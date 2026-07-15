#!/usr/bin/env python3
"""MOSS-TTS-Local (mlx-speech) IPC sidecar for LensLM (#193 / 161e).

Spawned by src-tauri's MossSidecar and driven over stdio as line-delimited
JSON. Audio never rides the pipe — a "synth" reply carries a path to a temp
WAV file that the Rust side reads and deletes. Contract (must match
lens-core/src-tauri MossSidecar byte-for-byte):

  startup:  load the model once, then print exactly one line {"ready": true}
  request:  {"id": <int>, "op": "ping"}
            {"id": <int>, "op": "synth", "text": str, "emotion": str|null,
             "ref_clip": str, "ref_transcript": str, "audio_temperature": float}
  reply:    {"id": <echo>, "ok": true, "pong": true}                    (ping)
            {"id": <echo>, "ok": true, "path": "<temp wav path>"}       (synth)
            {"id": <echo>, "ok": false, "error": "<short message>"}     (either, on failure)

Every reply echoes the request id and is flushed immediately so the Rust
side can resync on the next newline after a mid-line cancel. Apple-Silicon
only (MLX requires it); see README.md for the PyInstaller-onefile freeze.
"""

import argparse
import json
import os
import sys
import tempfile
import wave

import numpy as np

TARGET_SAMPLE_RATE = 24_000


def load_model(model_dir: str):
    import mlx_speech

    # TODO(A0): verify mlx_speech.tts.load() accepts a local directory path
    # for the pre-downloaded int8 model (not just a registered alias/HF repo id).
    return mlx_speech.tts.load(model_dir)


def write_wav_mono24k(waveform: np.ndarray, sample_rate: int) -> str:
    if sample_rate != TARGET_SAMPLE_RATE:
        # TODO(A0): confirm MOSS-TTS-Local always emits 24 kHz on real hardware;
        # resampling here is out of scope for #193 if it doesn't.
        raise RuntimeError(f"unexpected sample rate {sample_rate}, expected {TARGET_SAMPLE_RATE}")

    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")

    fd, path = tempfile.mkstemp(prefix="moss-turn-", suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return path


def send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def handle_synth(model, req: dict) -> str:
    text = req["text"]
    ref_clip = req["ref_clip"]
    ref_transcript = req["ref_transcript"]
    audio_temperature = req.get("audio_temperature", 1.0)
    # emotion fidelity is deferred for #193 (plan C8) — accepted, not forwarded.
    # TODO(A0): if mlx-speech exposes a cheap emotion/tag kwarg, wire
    # req.get("emotion") through as a follow-up; do not block #193 on it.

    # TODO(A0): confirm generate()'s kwarg names (reference_audio/reference_text/
    # temperature) and return shape (result.waveform/result.sample_rate) against
    # the installed mlx-speech version.
    result = model.generate(
        text,
        reference_audio=ref_clip,
        reference_text=ref_transcript,
        temperature=audio_temperature,
    )

    return write_wav_mono24k(np.array(result.waveform), int(result.sample_rate))


def main() -> None:
    parser = argparse.ArgumentParser(description="MOSS-TTS-Local mlx-speech sidecar")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing the downloaded MOSS-TTS-Local int8 model",
    )
    args = parser.parse_args()

    try:
        model = load_model(args.model_dir)
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
                path = handle_synth(model, req)
                send({"id": req_id, "ok": True, "path": path})
            else:
                send({"id": req_id, "ok": False, "error": f"unknown op: {op}"})
        except Exception as exc:
            send({"id": req_id, "ok": False, "error": str(exc)[:200]})


if __name__ == "__main__":
    main()
