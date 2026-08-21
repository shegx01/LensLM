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

# issue #42: prove the Apple-native ASR bridge on the shipping host. The clean is
# load-bearing: `tauri_build::build()` emits its own rerun-if-changed directives, so
# a second consecutive run would replay cached link directives without rebuilding.
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  echo "signoff: proving the Apple-native ASR bridge links (LENS_ASR_BRIDGE=require)…"
  cargo clean -p lenslm
  LENS_ASR_BRIDGE=require cargo build -p lenslm

  # A green build only proves build.rs did not panic. `LC_RPATH /usr/lib/swift` is
  # emitted solely on the bridge success path and, unlike a symbol, survives
  # `[profile.release] strip = true` — so the binary itself is the evidence.
  BRIDGE_BIN="${CARGO_TARGET_DIR:-${SCRIPT_DIR}/../target}/debug/LensLM"
  if ! otool -l "$BRIDGE_BIN" | grep -q '/usr/lib/swift'; then
    echo "signoff: FAILED — $BRIDGE_BIN carries no Swift runtime rpath; the bridge is not linked in." >&2
    exit 1
  fi
fi

gh signoff
