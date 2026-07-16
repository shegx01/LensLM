# qwen3-tts sidecar (Qwen3-TTS CustomVoice)

Apple-Silicon-only Python sidecar for the `Qwen3Local` TTS backend (161e).
`src-tauri`'s `QwenSidecar` spawns this process and drives it over stdio with
line-delimited JSON; it never enters `lens-core`, and MLX/Python never leave
this process. This file covers running the sidecar itself; the stdio contract
it must honor is in the Contract section below.

## Requirements

Apple Silicon Mac (M1 or later), Python 3.13+. MLX has no Linux/Windows/Intel
build, so this sidecar is never invoked outside `macos`+`aarch64` — the Rust
side compiles the whole `QwenSidecar` module out on other targets.

## Runtime: `uv`, no freezing

The app runs this sidecar via system `uv` (or an auto-provisioned one) — there
is no PyInstaller build and nothing to sign or notarize. `uv`, its managed
Python, and the MLX wheels it fetches are all vendor-signed, so a programmatic
download clears Gatekeeper without any bespoke packaging step:

```bash
# from the repo root — note the script path is passed in full, since `--project`
# only sets the environment dir, not the working directory:
uv run --frozen --project sidecar/qwen3-tts python sidecar/qwen3-tts/qwen3_tts_sidecar.py
```

`src-tauri` launches this shape with an **absolute** script path and `--frozen`
(use `uv.lock` verbatim), and sets `HF_HOME` + `HF_HUB_DISABLE_XET=1` in the
child's environment (before spawn) to redirect the HuggingFace cache into the
app's data directory and disable Xet transfer. `huggingface_hub` reads these
from the environment, so the Python side never sets them itself.

## Model download

On first synth `mlx-audio` pulls the CustomVoice weights (~4.5 GB, Apache-2.0)
into the HF cache:

- `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16`

## Voices

CustomVoice ships fixed preset speakers selected by id (case-insensitive) with
delivery steered by an `instruct` string — no reference clip, no transcript.
LensLM surfaces four (`dylan`, `aiden`, `serena`, `ono_anna`); the model
supports more (`get_supported_speakers()`).

## Contract

Startup: load the model once, then print one line `{"ready": true}`. Then read
one JSON request per stdin line and reply one JSON line per request, always
echoing the request's `id`:

- `{"id", "op": "ping"}` → `{"id", "ok": true, "pong": true}`
- `{"id", "op": "synth", "text", "speaker", "instruct", "temperature"?, "max_tokens"?}`
  → `{"id", "ok": true, "path": "<temp wav path>"}` (mono WAV at the model's
  native 24 kHz; the Rust side resamples)
- any failure (unknown speaker, generation error) → `{"id", "ok": false, "error": "<short message>"}`

`speaker` selects a preset (resolved case-insensitively; unknown → `ok:false`).
`instruct` steers delivery (omitted → model default). `temperature`/`max_tokens`
are optional and clamped by `resolve_gen_params` (defaults 0.9 / 4096). Audio
never rides the pipe — only the temp WAV path does; the Rust side reads and
deletes it.

## Dev run

```bash
cd sidecar/qwen3-tts
uv sync
echo '{"id": 1, "op": "ping"}' | uv run python qwen3_tts_sidecar.py
```

Feed it a synth request on stdin (first run downloads the model):

```bash
echo '{"id": 1, "op": "synth", "text": "hello there", "speaker": "dylan", "instruct": "Upbeat, energetic podcast host, conversational and lively."}' \
  | uv run python qwen3_tts_sidecar.py
```

Run the offline param-guard tests (no MLX needed):

```bash
uv run --group dev pytest sidecar/qwen3-tts/test_resolve_gen_params.py
```
