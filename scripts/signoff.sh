#!/usr/bin/env bash
# Run the Rust test suite locally, then post the `signoff` status for HEAD.
# The suite runs locally (not in CI) and `signoff` is its required check — see
# docs/ci.md. Requires: gh extension install basecamp/gh-signoff.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fetch libpdfium only if missing: src-tauri needs it to compile, and
# fetch-pdfium re-downloads on every call (so don't call it unconditionally).
if [ ! -f "${SCRIPT_DIR}/../src-tauri/frameworks/libpdfium.dylib" ]; then
  echo "signoff: libpdfium.dylib missing — fetching…"
  bash "${SCRIPT_DIR}/fetch-pdfium.sh"
fi

cargo test --workspace

# issue #42: build.rs sets `apple_asr_bridge` only when the Swift bridge really
# compiled, and `LENS_ASR_BRIDGE` picks the strictness — `require` fails the build
# when it cannot, `allow-missing` never does, unset is strict under --release only.
# The shipping host demands that compile+link proof here.

# The clean is load-bearing: `tauri_build::build()` emits its own rerun-if-changed
# directives, disabling Cargo's "any package file changed" heuristic, and
# rerun-if-env-changed fires only on a CHANGED value — so a second consecutive run
# would replay cached link directives against a stale archive.
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  echo "signoff: proving the Apple-native ASR bridge compiles (LENS_ASR_BRIDGE=require)…"
  cargo clean -p lenslm
  LENS_ASR_BRIDGE=require cargo test -p lenslm --no-run
fi

gh signoff
