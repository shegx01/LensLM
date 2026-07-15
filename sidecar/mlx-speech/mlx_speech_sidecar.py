#!/usr/bin/env python3
"""MOSS-TTS-Local (mlx-speech) IPC sidecar for LensLM (#193 / 161e).

Spawned by src-tauri's MossSidecar as `uv run --project <this dir> python
mlx_speech_sidecar.py` and driven over stdio as line-delimited JSON. Audio
never rides the pipe — a "synth" reply carries a path to a temp WAV file
that the Rust side reads and deletes. Contract (must match lens-core/
src-tauri MossSidecar byte-for-byte):

  startup:  load the model once, then print exactly one line {"ready": true}
  request:  {"id": <int>, "op": "ping"}
            {"id": <int>, "op": "synth", "text": str, "emotion": str|null,
             "ref_clip": str, "ref_transcript": str|null, "audio_temperature": float}
  reply:    {"id": <echo>, "ok": true, "pong": true}                    (ping)
            {"id": <echo>, "ok": true, "path": "<temp wav path>"}       (synth)
            {"id": <echo>, "ok": false, "error": "<short message>"}     (either, on failure)

Every reply echoes the request id and is flushed immediately so the Rust
side can resync on the next newline after a mid-line cancel.

Cloning uses the mlx-speech CONVERSATION API
(`synthesize_moss_tts_local_conversations`) — the simpler `generate(
reference_audio=...)` path is inert at every layer in mlx-speech 0.4.2/0.4.3
(verified against source). MOSS clones from the reference audio clip alone;
it does not use a reference transcript, so `ref_transcript` is accepted for
wire compatibility but otherwise unused.

The TTS transformer and audio codec live in two separate HuggingFace repos.
`mlx_speech.tts.load()` only takes a single `revision` kwarg, which cannot
pin both repos independently, so this sidecar resolves each repo directly
via `mlx_speech._hub.get_model_path()` with its own pinned revision and
constructs the adapter directly (see `load_moss_local()` below).

Apple-Silicon only (MLX requires it); see README.md for the uv-provisioned
runtime this sidecar runs under.
"""

import argparse
import json
import sys
import tempfile
import wave

import numpy as np

TARGET_SAMPLE_RATE = 24_000

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
    from mlx_speech.generation.moss_local import MossTTSLocalGenerationConfig
    from mlx_speech.tts._adapters.moss_local import MossLocalAdapter

    model_dir = get_model_path(MOSS_TTS_REPO, revision=MOSS_TTS_REVISION)
    codec_dir = get_model_path(MOSS_CODEC_REPO, revision=MOSS_CODEC_REVISION)
    adapter = MossLocalAdapter.from_dir(model_dir, codec_dir=codec_dir)
    config = MossTTSLocalGenerationConfig.app_defaults()
    return adapter._model, adapter._processor, adapter._codec, config


def write_wav_mono24k(waveform: np.ndarray, sample_rate: int) -> str:
    if sample_rate != TARGET_SAMPLE_RATE:
        # TODO(A0): confirm on hardware that the downloaded checkpoint's
        # audio-codec config keeps the library default of 24 kHz (it is the
        # documented rate, but the pinned revision's config.json is the
        # ground truth and isn't inspectable offline).
        raise RuntimeError(f"unexpected sample rate {sample_rate}, expected {TARGET_SAMPLE_RATE}")

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


def handle_synth(model, processor, codec, config, req: dict) -> str:
    text = req["text"]
    ref_clip = req["ref_clip"]
    # emotion + ref_transcript are accepted for wire compatibility but unused:
    # MOSS clones from the reference clip alone (no transcript), and emotion
    # fidelity is out of scope for #193 (plan C8).

    from mlx_speech.generation.moss_local import synthesize_moss_tts_local_conversations

    result = synthesize_moss_tts_local_conversations(
        model,
        processor,
        codec,
        conversations=[[processor.build_user_message(text=text, reference=[ref_clip])]],
        mode="generation",
        config=config,
    )
    output = result.outputs[0]
    return write_wav_mono24k(np.array(output.waveform), int(output.sample_rate))


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
        model, processor, codec, config = load_moss_local()
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
                path = handle_synth(model, processor, codec, config, req)
                send({"id": req_id, "ok": True, "path": path})
            else:
                send({"id": req_id, "ok": False, "error": f"unknown op: {op}"})
        except Exception as exc:
            send({"id": req_id, "ok": False, "error": str(exc)[:200]})


if __name__ == "__main__":
    main()
