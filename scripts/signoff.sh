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
gh signoff
