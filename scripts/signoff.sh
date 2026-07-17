#!/usr/bin/env bash
# Prove the macOS-gated Apple-native ASR bridge, then post the `signoff` status
# for HEAD. The bulk Rust test suite now runs in CI (the 3 ubuntu shards); this
# signoff covers ONLY what no Linux runner can build — see docs/ci.md.
# Requires: gh extension install basecamp/gh-signoff.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# issue #42: the Apple-native ASR bridge (SpeechAnalyzer via a Swift @_cdecl C ABI)
# is behind the `apple-native-asr` feature, which the CI shards do NOT enable
# (a binary crate cannot self-activate a feature per-target — that would be a Cargo
# cycle). On the shipping aarch64-apple-darwin host, compile it explicitly so the
# Swift build (build.rs → swiftc) and the FFI declarations are proven under signoff.
# The runtime transcription test stays `#[ignore]` (needs a real macOS-26 on-device
# model); this is a compile+link proof only.
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  # src-tauri needs libpdfium to compile; fetch only if missing (fetch-pdfium
  # re-downloads on every call, so don't call it unconditionally).
  if [ ! -f "${SCRIPT_DIR}/../src-tauri/frameworks/libpdfium.dylib" ]; then
    echo "signoff: libpdfium.dylib missing — fetching…"
    bash "${SCRIPT_DIR}/fetch-pdfium.sh"
  fi
  echo "signoff: compiling the Apple-native ASR bridge (--features apple-native-asr)…"
  cargo test -p lenslm --features apple-native-asr --no-run
else
  # Nothing to prove off arm64-macOS — make the no-op loud so a green status here
  # isn't mistaken for the ASR bridge having been verified.
  echo "signoff: not arm64-macOS ($(uname -s)/$(uname -m)) — ASR bridge NOT compiled; posting attestation only." >&2
fi

gh signoff
