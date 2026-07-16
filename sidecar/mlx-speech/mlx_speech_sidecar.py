#!/usr/bin/env python3
"""MOSS-TTS-Local (mlx-speech) IPC sidecar for LensLM (#193 / 161e).

Spawned by src-tauri's MossSidecar as `uv run --project <this dir> python
mlx_speech_sidecar.py`, driven over stdio as line-delimited JSON. Apple-Silicon
only (MLX requires it). See README.md for the wire contract and runtime.
"""

import argparse
import json
import sys
import tempfile
import wave

import numpy as np

# Pinned HF revisions (commit SHAs, not branches) so the sidecar's model
# behavior doesn't float across machines/time. ~5.27 GB combined.
MOSS_TTS_REPO = "appautomaton/openmoss-tts-local-mlx"
MOSS_TTS_REVISION = "c4951c75b9b44be20a87d0444b3638597e020ca0"
MOSS_CODEC_REPO = "appautomaton/openmoss-audio-tokenizer-mlx"
MOSS_CODEC_REVISION = "5d0020462d191cdf67c362ee0a9da1775666923e"


def load_moss_local():
    """Load the MOSS-TTS-Local transformer + audio codec with pinned revisions.

    Bypasses `mlx_speech.tts.load()` (single `revision` kwarg can't pin two
    repos independently) and instead resolves each repo via the hub helper,
    then builds the adapter directly to reach its held `_model`/`_processor`/
    `_codec`, which the conversation API needs as separate arguments.
    """
    from mlx_speech._hub import get_model_path
    from mlx_speech.tts._adapters.moss_local import MossLocalAdapter

    model_dir = get_model_path(MOSS_TTS_REPO, revision=MOSS_TTS_REVISION)
    codec_dir = get_model_path(MOSS_CODEC_REPO, revision=MOSS_CODEC_REVISION)
    adapter = MossLocalAdapter.from_dir(model_dir, codec_dir=codec_dir)
    return adapter._model, adapter._processor, adapter._codec


def write_wav_mono(waveform: np.ndarray, sample_rate: int) -> str:
    # Writes at the model's actual rate; the Rust side resamples to its
    # target (AudioBuffer::resample_to / stitch_turns), so no rate guard here.
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")

    with tempfile.NamedTemporaryFile(prefix="moss-turn-", suffix=".wav", delete=False) as f:
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


def handle_synth(model, processor, codec, req: dict) -> str:
    text = req["text"]
    ref_clip = req["ref_clip"]
    # emotion + ref_transcript are accepted for wire compatibility but unused:
    # MOSS clones from the reference clip alone (no transcript), and emotion
    # fidelity is out of scope for #193 (plan C8).

    from mlx_speech.generation.moss_local import (
        MossTTSLocalGenerationConfig,
        synthesize_moss_tts_local_conversations,
    )

    # Bound generation near the turn's length so a missed stochastic EOS can't
    # ramble past the text. `app_defaults()` leaves max_new_tokens=1024 (~82s);
    # the codec is ~12.5 rows/s and speech ~2.5 words/s (~5 rows/word), so cap at
    # ~2x the estimate, floored at the library's canonical clone cap (160 ≈ 12.8s).
    max_rows = max(160, len(text.split()) * 10)
    config = MossTTSLocalGenerationConfig(
        max_new_tokens=max_rows,
        audio_temperature=1.7,
        audio_top_k=25,
        audio_top_p=0.8,
        audio_repetition_penalty=1.0,
    )

    result = synthesize_moss_tts_local_conversations(
        model,
        processor,
        codec,
        conversations=[[processor.build_user_message(text=text, reference=[ref_clip])]],
        mode="generation",
        config=config,
    )
    output = result.outputs[0]
    return write_wav_mono(np.array(output.waveform), int(output.sample_rate))


def main() -> None:
    parser = argparse.ArgumentParser(description="MOSS-TTS-Local mlx-speech sidecar")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Deprecated/ignored: the model is fetched via pinned HF revisions "
        "under HF_HOME, not a caller-supplied directory. Kept for wire/test compat.",
    )
    args, _unknown = parser.parse_known_args()
    _ = args

    try:
        model, processor, codec = load_moss_local()
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
                path = handle_synth(model, processor, codec, req)
                send({"id": req_id, "ok": True, "path": path})
            else:
                send({"id": req_id, "ok": False, "error": f"unknown op: {op}"})
        except Exception as exc:
            send({"id": req_id, "ok": False, "error": str(exc)[:200]})


if __name__ == "__main__":
    main()
