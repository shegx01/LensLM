#!/usr/bin/env bash
#
# fetch-pdfium.sh — vendor (or verify) the prebuilt libpdfium.dylib used by the
# PDF extractor (lens-core, M4 Phase 2, Step 5).
#
# Pinned binary (DECISION, locked):
#   repo : bblanchon/pdfium-binaries
#   tag  : chromium/7906   (URL-encoded in the download URL as chromium%2F7906)
#   build: NON-V8, universal macOS asset  →  pdfium-mac-univ.tgz
#   dylib: lib/libpdfium.dylib inside the archive
#
# Behaviour:
#   * Downloads + extracts the pinned asset to a temp dir, copies the dylib to
#     src-tauri/frameworks/libpdfium.dylib.
#   * RECORD mode (default when CHECKSUMS is absent, or forced with --record):
#     computes the dylib's SHA-256 and writes src-tauri/frameworks/CHECKSUMS,
#     then prints the recorded hash.
#   * VERIFY mode (default when CHECKSUMS exists): re-computes the dylib's SHA-256
#     and FAILS LOUDLY if it does not match the recorded CHECKSUMS entry.
#
# CHECKSUMS format (one entry):  <sha256>  libpdfium.dylib
#
# The .dylib itself is gitignored (fetched, never committed); scripts/ + CHECKSUMS
# ARE committed so CI can re-fetch and verify supply-chain integrity.
#
# Usage:
#   scripts/fetch-pdfium.sh            # auto: record if no CHECKSUMS, else verify
#   scripts/fetch-pdfium.sh --record   # force re-record CHECKSUMS (intentional bump)
#   scripts/fetch-pdfium.sh --verify   # force verify (fail if CHECKSUMS missing)

set -euo pipefail

# --- Pinned coordinates -----------------------------------------------------
readonly PDFIUM_TAG="chromium/7906"
readonly PDFIUM_TAG_ENC="chromium%2F7906"
readonly PDFIUM_ASSET="pdfium-mac-univ.tgz"
readonly PDFIUM_URL="https://github.com/bblanchon/pdfium-binaries/releases/download/${PDFIUM_TAG_ENC}/${PDFIUM_ASSET}"

# --- Paths (resolve relative to this script's repo root) --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly FRAMEWORKS_DIR="${REPO_ROOT}/src-tauri/frameworks"
readonly DYLIB_DST="${FRAMEWORKS_DIR}/libpdfium.dylib"
readonly CHECKSUMS="${FRAMEWORKS_DIR}/CHECKSUMS"
readonly CHECKSUM_NAME="libpdfium.dylib"

# --- Mode selection ---------------------------------------------------------
MODE="auto"
case "${1:-}" in
  --record) MODE="record" ;;
  --verify) MODE="verify" ;;
  "")       MODE="auto" ;;
  *) echo "error: unknown flag '$1' (expected --record | --verify)" >&2; exit 2 ;;
esac

# --- Helpers ----------------------------------------------------------------
sha256_of() {
  # Prints just the bare hex digest of $1, portable across macOS/Linux.
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    echo "error: neither shasum nor sha256sum is available" >&2
    exit 1
  fi
}

recorded_hash() {
  # Reads the recorded hash for CHECKSUM_NAME from CHECKSUMS, or empty string.
  [ -f "${CHECKSUMS}" ] || { echo ""; return; }
  awk -v name="${CHECKSUM_NAME}" '$2 == name { print $1; exit }' "${CHECKSUMS}"
}

# --- Resolve effective mode -------------------------------------------------
if [ "${MODE}" = "auto" ]; then
  if [ -f "${CHECKSUMS}" ] && [ -n "$(recorded_hash)" ]; then
    MODE="verify"
  else
    MODE="record"
  fi
fi

echo "fetch-pdfium: mode=${MODE} tag=${PDFIUM_TAG} asset=${PDFIUM_ASSET}"

# --- Download + extract -----------------------------------------------------
TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

echo "fetch-pdfium: downloading ${PDFIUM_URL}"
curl -fL --retry 3 --retry-delay 2 -o "${TMP_DIR}/${PDFIUM_ASSET}" "${PDFIUM_URL}"

echo "fetch-pdfium: extracting ${PDFIUM_ASSET}"
tar xzf "${TMP_DIR}/${PDFIUM_ASSET}" -C "${TMP_DIR}"

EXTRACTED_DYLIB="${TMP_DIR}/lib/libpdfium.dylib"
if [ ! -f "${EXTRACTED_DYLIB}" ]; then
  echo "error: expected dylib at lib/libpdfium.dylib inside the archive, not found" >&2
  echo "       archive contents:" >&2
  find "${TMP_DIR}" -maxdepth 2 -type f >&2
  exit 1
fi

mkdir -p "${FRAMEWORKS_DIR}"
cp "${EXTRACTED_DYLIB}" "${DYLIB_DST}"

DYLIB_HASH="$(sha256_of "${DYLIB_DST}")"
DYLIB_SIZE="$(wc -c < "${DYLIB_DST}" | tr -d ' ')"
echo "fetch-pdfium: vendored ${DYLIB_DST}"
echo "fetch-pdfium:   size   = ${DYLIB_SIZE} bytes"
echo "fetch-pdfium:   sha256 = ${DYLIB_HASH}"

# --- Record or verify -------------------------------------------------------
case "${MODE}" in
  record)
    printf '%s  %s\n' "${DYLIB_HASH}" "${CHECKSUM_NAME}" > "${CHECKSUMS}"
    echo "fetch-pdfium: RECORDED checksum to ${CHECKSUMS}"
    echo "fetch-pdfium:   ${DYLIB_HASH}  ${CHECKSUM_NAME}"
    ;;
  verify)
    EXPECTED="$(recorded_hash)"
    if [ -z "${EXPECTED}" ]; then
      echo "error: --verify requested but no recorded hash for ${CHECKSUM_NAME} in ${CHECKSUMS}" >&2
      exit 1
    fi
    if [ "${DYLIB_HASH}" != "${EXPECTED}" ]; then
      echo "error: SHA-256 MISMATCH for ${CHECKSUM_NAME}" >&2
      echo "       expected: ${EXPECTED}" >&2
      echo "       actual  : ${DYLIB_HASH}" >&2
      echo "       (the downloaded binary does not match the recorded checksum — refusing)" >&2
      exit 1
    fi
    echo "fetch-pdfium: VERIFIED — sha256 matches recorded CHECKSUMS"
    ;;
esac
