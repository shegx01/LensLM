#!/usr/bin/env bash
#
# fetch-models-catalog.sh — vendor the FULL models.dev catalog used as the
# OFFLINE FLOOR for the typed model catalog (lens-core, M4 Phase 3).
#
# Why this exists:
#   `lens-core::model_catalog` fetches the live catalog at runtime + caches it,
#   but onboarding (first run, empty cache) needs a usable catalog BEFORE any
#   fetch completes. The committed bundle is that floor. It used to be a tiny
#   hand-curated slice (a handful of models per provider), which silently went
#   stale as providers shipped/retired models. This script vendors the FULL
#   catalog instead, so the offline floor tracks reality at vendor time and the
#   live fetch supersedes it at runtime.
#
# Source (DECISION, locked):
#   url   : https://models.dev/api.json   (the same endpoint refreshed at runtime)
#   format: a single JSON object keyed by provider id (~2.4 MB raw)
#   licence: MIT (github.com/sst/models.dev) — redistribution permitted.
#
# Storage decision (DECISION, locked):
#   The full catalog is ~2.4 MB raw. `flate2` is ALREADY in the dependency tree
#   (transitive via fastembed/lancedb), so we GZIP the bundle and decompress it
#   in `ModelCatalog::bundled()` via that already-present crate — no NEW external
#   crate, a much smaller committed blob (~250–350 KB). The committed artifact is
#   therefore `bundled-catalog.json.gz` (the `include_bytes!` path).
#
# Behaviour:
#   * Downloads the live catalog to a temp file, sanity-checks it parses as JSON
#     and carries the core providers, then gzips it into place + prints sizes.
#
# Usage:
#   scripts/fetch-models-catalog.sh        # vendor the current full catalog
#
# Run this per release to refresh the offline floor. The runtime live fetch
# (lens-core::model_catalog::refresh_if_stale) supersedes this bundle whenever
# the user is online.

set -euo pipefail

# --- Source coordinates (locked) --------------------------------------------
readonly CATALOG_URL="https://models.dev/api.json"

# --- Paths (resolve relative to this script's repo root) --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly CATALOG_DIR="${REPO_ROOT}/lens-core/src/model_catalog"
readonly CATALOG_GZ="${CATALOG_DIR}/bundled-catalog.json.gz"

# --- Core CLOUD providers the offline floor MUST cover ----------------------
# models.dev catalogs CLOUD providers only — there is intentionally NO plain
# `ollama` key (local Ollama models are user-pulled and validated live via
# `/api/tags`, never the catalog). So the local `ollama` provider is NOT in this
# set; `ollama-cloud` (a hosted provider) is.
readonly CORE_PROVIDERS=("anthropic" "openai" "google" "ollama-cloud" "zai")

echo "fetch-models-catalog: source=${CATALOG_URL}"

# --- Download ---------------------------------------------------------------
TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

readonly TMP_JSON="${TMP_DIR}/api.json"
echo "fetch-models-catalog: downloading ${CATALOG_URL}"
curl -fL --retry 3 --retry-delay 2 -o "${TMP_JSON}" "${CATALOG_URL}"

# --- Sanity-check the download before committing it -------------------------
# A valid catalog is a JSON object keyed by provider id. Refuse to vendor a body
# that does not parse or is missing the core providers (a CDN/error page must not
# silently overwrite the floor).
if command -v python3 >/dev/null 2>&1; then
  python3 - "${TMP_JSON}" "${CORE_PROVIDERS[@]}" <<'PY'
import json
import sys

path = sys.argv[1]
required = sys.argv[2:]
with open(path, "rb") as f:
    data = json.load(f)
if not isinstance(data, dict):
    sys.exit("error: catalog root is not a JSON object")
missing = [p for p in required if p not in data]
if missing:
    sys.exit(f"error: catalog is missing core providers: {missing}")
n_models = sum(len(v.get("models", {})) for v in data.values() if isinstance(v, dict))
print(f"fetch-models-catalog: parsed {len(data)} providers, {n_models} models total")
PY
else
  echo "fetch-models-catalog: python3 not found — skipping JSON sanity check" >&2
fi

# --- Gzip into place --------------------------------------------------------
mkdir -p "${CATALOG_DIR}"
# `gzip -9 -c` writes a standard gzip stream that `flate2::read::GzDecoder` reads.
# `-n` omits the original name/timestamp so re-running with the same input yields
# a byte-stable artifact (no spurious diff from the embedded mtime).
gzip -9 -n -c "${TMP_JSON}" > "${CATALOG_GZ}"

RAW_SIZE="$(wc -c < "${TMP_JSON}" | tr -d ' ')"
GZ_SIZE="$(wc -c < "${CATALOG_GZ}" | tr -d ' ')"
echo "fetch-models-catalog: vendored ${CATALOG_GZ}"
echo "fetch-models-catalog:   raw  size = ${RAW_SIZE} bytes"
echo "fetch-models-catalog:   gzip size = ${GZ_SIZE} bytes"
