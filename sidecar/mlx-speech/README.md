# mlx-speech sidecar (MOSS-TTS-Local)

Apple-Silicon-only Python sidecar for the `MossLocal` TTS backend (#193 /
161e). `src-tauri`'s `MossSidecar` spawns this process and drives it over
stdio with line-delimited JSON; it never enters `lens-core`, and MLX/Python
never leave this process. This file covers running the sidecar itself; the
stdio contract it must honor is specified in the Contract section below.

## Requirements

Apple Silicon Mac (M1 or later), Python 3.13+. MLX has no Linux/Windows/Intel
build, so this sidecar is never invoked outside `macos`+`aarch64` — the Rust
side compiles the whole `MossSidecar` module out on other targets.

## Runtime: `uv`, no freezing

The app runs this sidecar via system `uv` (or an auto-provisioned one) —
there is no PyInstaller build and nothing to sign or notarize. `uv`, its
managed Python, and the MLX wheels it fetches are all vendor-signed, so a
programmatic download clears Gatekeeper without any bespoke packaging step:

```bash
uv run --project sidecar/mlx-speech python mlx_speech_sidecar.py
```

`src-tauri` launches exactly this command and sets `HF_HOME` in the child's
environment (before spawn) to redirect the HuggingFace cache into the app's
data directory — `huggingface_hub` reads `HF_HOME` from the environment, so
the Python side never sets it itself.

## Model download

On first run the sidecar pulls two pinned HuggingFace revisions (~5.27 GB
combined, both Apache-2.0), matching the exact commits baked into
`mlx_speech_sidecar.py`:

- `appautomaton/openmoss-tts-local-mlx` @ `c4951c75b9b44be20a87d0444b3638597e020ca0` (~3.26 GB)
- `appautomaton/openmoss-audio-tokenizer-mlx` @ `5d0020462d191cdf67c362ee0a9da1775666923e` (~2.0 GB, mandatory codec)

## Contract

Startup: load the model once, then print one line `{"ready": true}`. Then
read one JSON request per stdin line and reply one JSON line per request,
always echoing the request's `id`:

- `{"id", "op": "ping"}` → `{"id", "ok": true, "pong": true}`
- `{"id", "op": "synth", "text", "emotion", "ref_clip", "ref_transcript", "audio_temperature"}`
  → `{"id", "ok": true, "path": "<temp wav path>"}` (mono WAV at the model's
  native sample rate; the Rust side resamples)
- any failure → `{"id", "ok": false, "error": "<short message>"}`

Cloning uses `ref_clip` (an audio path) only — MOSS does not use a reference
transcript, so `ref_transcript` is accepted for wire compatibility and
otherwise ignored. Audio never rides the pipe — only the temp WAV path does;
the Rust side reads and deletes it.

## Dev run

```bash
cd sidecar/mlx-speech
uv sync
echo '{"id": 1, "op": "ping"}' | uv run python mlx_speech_sidecar.py
```

Feed it a synth request on stdin (first run downloads the pinned model):

```bash
echo '{"id": 1, "op": "synth", "text": "hello", "emotion": null, "ref_clip": "/path/to/clip.wav", "ref_transcript": null, "audio_temperature": 1.0}' \
  | uv run python mlx_speech_sidecar.py
```
