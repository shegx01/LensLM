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

# issue #42: the Apple-native ASR bridge (SpeechAnalyzer via a Swift @_cdecl C ABI)
# is behind the `apple-native-asr` feature, which `cargo test --workspace` does NOT
# enable (a binary crate cannot self-activate a feature per-target — that would be a
# Cargo cycle). On the shipping aarch64-apple-darwin host, compile it explicitly so
# the Swift build (build.rs → swiftc) and the FFI declarations are proven under
# signoff. The runtime transcription test stays `#[ignore]` (needs a real macOS-26
# on-device model); this is a compile+link proof only.
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  echo "signoff: compiling the Apple-native ASR bridge (--features apple-native-asr)…"
  cargo test -p lenslm --features apple-native-asr --no-run
fi

gh signoff
