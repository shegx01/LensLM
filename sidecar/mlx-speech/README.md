# mlx-speech sidecar (MOSS-TTS-Local)

Apple-Silicon-only Python sidecar for the `MossLocal` TTS backend (#193 /
161e). `src-tauri`'s `MossSidecar` spawns this process and drives it over
stdio with line-delimited JSON; it never enters `lens-core`, and MLX/Python
never leave this process. See
`.omc/plans/issue-193-moss-local-consensus-plan.md` §B1/B2/C7 for the full
design; this file only covers building and running the sidecar itself.

## Requirements

Apple Silicon Mac (M1 or later), Python 3.13+. MLX has no Linux/Windows/Intel
build, so this sidecar is never invoked outside `macos`+`aarch64` — the Rust
side compiles the whole `MossSidecar` module out on other targets.

## Contract

Startup: load the model once from `--model-dir`, then print one line
`{"ready": true}`. Then read one JSON request per stdin line and reply one
JSON line per request, always echoing the request's `id`:

- `{"id", "op": "ping"}` → `{"id", "ok": true, "pong": true}`
- `{"id", "op": "synth", "text", "emotion", "ref_clip", "ref_transcript", "audio_temperature"}`
  → `{"id", "ok": true, "path": "<temp wav path>"}` (24 kHz mono WAV)
- any failure → `{"id", "ok": false, "error": "<short message>"}`

Audio never rides the pipe — only the temp WAV path does; the Rust side
reads and deletes it.

## A0 freeze spike — human sign-off required

The three `# TODO(A0): ...` comments in `mlx_speech_sidecar.py` mark API
surface written against the _documented_ `mlx-speech` shape without a real
Apple-Silicon machine to run it on. Before this sidecar is trusted:

1. Run it unfrozen (see below) against a real downloaded MOSS int8 model dir
   and synthesize one turn.
2. Confirm/adjust the three TODO(A0) lines against the installed
   `mlx-speech` version's actual API (`load()` accepting a local dir,
   `generate()`'s kwarg names, and the returned sample rate).
3. Only then proceed to the PyInstaller freeze below and, later, signing/
   notarization/stapling (plan A3, separate step).

## Dev run (unfrozen, uv-managed venv)

```bash
cd sidecar/mlx-speech
uv sync
uv run python mlx_speech_sidecar.py --model-dir /path/to/moss/int8/model
```

Feed it requests on stdin, one JSON object per line, e.g.:

```bash
echo '{"id": 1, "op": "ping"}' | uv run python mlx_speech_sidecar.py --model-dir /path/to/model
```

## Freezing for distribution

**Primary path — PyInstaller `--onefile`.** MLX ships native dylibs and
Metal `.metallib` shader binaries that are not Python modules, so
PyInstaller's import-graph analysis won't find them on its own —
`--collect-all` is required for both `mlx` and `mlx_speech`:

```bash
cd sidecar/mlx-speech
uv sync --extra freeze
uv run pyinstaller --onefile \
  --name mlx-speech-sidecar \
  --collect-all mlx \
  --collect-all mlx_speech \
  mlx_speech_sidecar.py
```

If a Metal shader library still isn't discovered by `--collect-all` (missing
`.metallib` at first run), add it explicitly with `--add-binary` once its
path is known on the build machine — this is exactly the kind of thing the
A0 spike must observe and record.

The frozen binary is what gets Developer-ID signed, notarized, and stapled
(plan A3) before being published to GitHub Releases and SHA-pinned in the
registry (plan A1) as `moss_sidecar_bin`.

**Fallback — uv-managed venv, no freeze.** If the PyInstaller-onefile
freeze proves infeasible for MLX's native binaries, ship this directory's
`pyproject.toml` + `uv.lock` instead and have `src-tauri` spawn
`uv run --project <this dir> mlx_speech_sidecar.py --model-dir <dir>`
against a `uv sync`'d venv on first use. This avoids freezing MLX's native
code entirely at the cost of requiring `uv` (or a pre-provisioned venv) on
the target machine.
