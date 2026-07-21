#!/usr/bin/env bash
#
# fetch-provider-logos.sh — vendor (or verify) provider brand-mark SVGs used by
# the AI Model settings pane (Providers section, logo chip).
#
# Source (DECISION, locked):
#   url    : https://models.dev/logos/{id}.svg
#   licence: MIT (github.com/sst/models.dev) — same permissively-licensed source
#            already vendored for the model catalog (fetch-models-catalog.sh).
#   usage  : these are provider BRAND MARKS, used nominatively for identification
#            only (a small logo chip naming which provider a row/config refers to)
#            — not used to imply endorsement. Providers with no bundled mark (e.g.
#            `openai-compatible`, a generic custom endpoint) fall back to a
#            monogram at render time (`src/lib/models/provider-logos.ts`).
#
# Provider ids vendored (mirrors `CLOUD_PROVIDERS` in
# `src/lib/onboarding/cloud-providers.ts` + the local `ollama` provider):
#   openai anthropic google groq deepseek xai cohere zai ollama-cloud ollama
# `openai-compatible` is intentionally NOT fetched — it has no brand mark.
#
# models.dev has no distinct mark for the LOCAL `ollama` provider: it serves a
# generic placeholder icon (HTTP 200) for any unknown logo id, indistinguishable
# from a truly missing asset. This script detects that sentinel by comparing
# against a deliberately-bogus id fetched at runtime, and for `ollama` falls back
# to the `ollama-cloud` mark (same brand) instead of vendoring the placeholder.
#
# Normalization + sanitization (each fetched SVG feeds `{@html}` — see
# `ProviderLogo.svelte`): strip `width`/`height` (size is controlled by the
# chip), coerce hard-coded hex/rgb `fill`/`stroke` to `currentColor` (theme
# tokens drive color), keep `viewBox`, and REJECT any asset containing
# `<script>`, an `on*=` handler attribute, `<foreignObject>`, `<use>`,
# `<image>`, `<style>`, or a `href`/`xlink:href` using a `javascript:`/`data:`
# scheme (defense-in-depth against click/SMIL-triggered and referenced-content
# vectors in future vendored marks).
#
# CHECKSUMS format (one line per vendored SVG): `<sha256>  <filename>` — same
# record/verify pattern as `src-tauri/frameworks/CHECKSUMS` (fetch-pdfium.sh).
#
# Usage:
#   scripts/fetch-provider-logos.sh            # auto: record if no CHECKSUMS, else verify
#   scripts/fetch-provider-logos.sh --record   # force re-record CHECKSUMS (intentional bump)
#   scripts/fetch-provider-logos.sh --verify   # force verify (fail if CHECKSUMS missing)

set -euo pipefail

# --- Source coordinates (locked) --------------------------------------------
readonly LOGO_BASE_URL="https://models.dev/logos"
readonly PROVIDER_IDS=(openai anthropic google groq deepseek xai cohere zai ollama-cloud ollama)
readonly SENTINEL_PROBE_ID="__lenslm_sentinel_probe__"

# --- Paths (resolve relative to this script's repo root) --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly LOGOS_DIR="${REPO_ROOT}/src/lib/assets/provider-logos"
readonly CHECKSUMS="${LOGOS_DIR}/CHECKSUMS"

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
  # Reads the recorded hash for filename $1 from CHECKSUMS, or empty string.
  [ -f "${CHECKSUMS}" ] || { echo ""; return; }
  awk -v name="$1" '$2 == name { print $1; exit }' "${CHECKSUMS}"
}

if ! command -v node >/dev/null 2>&1; then
  echo "error: node is required to sanitize fetched SVGs" >&2
  exit 1
fi

# --- Resolve effective mode -------------------------------------------------
if [ "${MODE}" = "auto" ]; then
  if [ -f "${CHECKSUMS}" ]; then
    MODE="verify"
  else
    MODE="record"
  fi
fi

echo "fetch-provider-logos: mode=${MODE}"

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

# --- Sanitizer (Node): strip width/height, coerce fill/stroke to currentColor,
# reject <script>/on*=/<foreignObject>/<use>/<image>/<style>/javascript:|data: hrefs.
# Reads $1, writes sanitized output to $2.
cat > "${TMP_DIR}/sanitize-svg.mjs" <<'JS'
import { readFileSync, writeFileSync } from 'node:fs';

const [, , src, dst] = process.argv;
let svg = readFileSync(src, 'utf8');

const forbidden =
  /<script[\s>]|on[a-z]+\s*=|<foreignObject[\s>]|<use\b|<image\b|<style\b|(?:xlink:)?href\s*=\s*["']?\s*(?:javascript|data):/i;
if (forbidden.test(svg)) {
  console.error(
    `sanitize-svg: REJECTED ${src} — forbidden <script>/on*=/<foreignObject>/<use>/<image>/<style>/javascript:|data: href content`
  );
  process.exit(1);
}

if (!/^\s*(<\?xml[^>]*\?>\s*)?<svg[\s>]/i.test(svg)) {
  console.error(`sanitize-svg: REJECTED ${src} — not an <svg> root element`);
  process.exit(1);
}

svg = svg.replace(/<\?xml[^>]*\?>\s*/i, '');
svg = svg.replace(/\s(width|height)="[^"]*"/gi, '');
svg = svg.replace(/\s(fill|stroke)="(#[0-9a-fA-F]{3,8}|rgba?\([^)]*\))"/gi, ' $1="currentColor"');
svg = svg.trim() + '\n';

writeFileSync(dst, svg);
JS

fetch_logo() {
  # Downloads models.dev/logos/$1.svg to $2. Returns non-zero on HTTP failure.
  curl -fsSL --retry 3 --retry-delay 2 -o "$2" "${LOGO_BASE_URL}/$1.svg"
}

echo "fetch-provider-logos: probing sentinel (models.dev's generic no-logo fallback)"
fetch_logo "${SENTINEL_PROBE_ID}" "${TMP_DIR}/sentinel.svg"
readonly SENTINEL_HASH="$(sha256_of "${TMP_DIR}/sentinel.svg")"

mkdir -p "${LOGOS_DIR}"

declare -a VENDORED=()
declare -a SKIPPED=()
OLLAMA_CLOUD_SANITIZED=""

for id in "${PROVIDER_IDS[@]}"; do
  raw="${TMP_DIR}/${id}.raw.svg"
  out="${LOGOS_DIR}/${id}.svg"

  if ! fetch_logo "${id}" "${raw}"; then
    echo "fetch-provider-logos: ${id} — 404/fetch failed, skipping (monogram fallback)"
    SKIPPED+=("${id}")
    continue
  fi

  if [ "$(sha256_of "${raw}")" = "${SENTINEL_HASH}" ]; then
    if [ "${id}" = "ollama" ] && [ -n "${OLLAMA_CLOUD_SANITIZED}" ]; then
      echo "fetch-provider-logos: ollama — no distinct mark, reusing ollama-cloud mark"
      cp "${OLLAMA_CLOUD_SANITIZED}" "${out}"
      VENDORED+=("${id}")
      continue
    fi
    echo "fetch-provider-logos: ${id} — no distinct mark (generic placeholder), skipping (monogram fallback)"
    SKIPPED+=("${id}")
    continue
  fi

  node "${TMP_DIR}/sanitize-svg.mjs" "${raw}" "${out}"
  echo "fetch-provider-logos: ${id} — vendored"
  VENDORED+=("${id}")

  if [ "${id}" = "ollama-cloud" ]; then
    OLLAMA_CLOUD_SANITIZED="${out}"
  fi
done

# --- Record or verify checksums over every vendored SVG ---------------------
NEW_CHECKSUMS="${TMP_DIR}/CHECKSUMS.new"
: > "${NEW_CHECKSUMS}"
for id in "${VENDORED[@]}"; do
  f="${LOGOS_DIR}/${id}.svg"
  printf '%s  %s\n' "$(sha256_of "${f}")" "${id}.svg" >> "${NEW_CHECKSUMS}"
done
sort -k2 -o "${NEW_CHECKSUMS}" "${NEW_CHECKSUMS}"

case "${MODE}" in
  record)
    cp "${NEW_CHECKSUMS}" "${CHECKSUMS}"
    echo "fetch-provider-logos: RECORDED checksums to ${CHECKSUMS}"
    ;;
  verify)
    if [ ! -f "${CHECKSUMS}" ]; then
      echo "error: --verify requested but ${CHECKSUMS} is missing" >&2
      exit 1
    fi
    MISMATCH=0
    while read -r hash name; do
      expected="$(recorded_hash "${name}")"
      if [ -z "${expected}" ]; then
        echo "error: no recorded checksum for ${name} (new asset — re-run with --record)" >&2
        MISMATCH=1
        continue
      fi
      if [ "${hash}" != "${expected}" ]; then
        echo "error: SHA-256 MISMATCH for ${name}" >&2
        echo "       expected: ${expected}" >&2
        echo "       actual  : ${hash}" >&2
        MISMATCH=1
      fi
    done < "${NEW_CHECKSUMS}"
    if [ "${MISMATCH}" -ne 0 ]; then
      echo "error: provider-logo checksum verification failed (refusing)" >&2
      exit 1
    fi
    echo "fetch-provider-logos: VERIFIED — sha256 matches recorded CHECKSUMS"
    ;;
esac

echo "fetch-provider-logos: vendored [${VENDORED[*]:-}]"
if [ "${#SKIPPED[@]}" -gt 0 ]; then
  echo "fetch-provider-logos: skipped (monogram fallback) [${SKIPPED[*]}]"
fi
