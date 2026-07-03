#!/usr/bin/env bash
# Run the Rust test suite locally, then post the `signoff` status for HEAD.
# The suite runs locally (not in CI) and `signoff` is its required check — see
# docs/ci.md. Requires: gh extension install basecamp/gh-signoff.
set -euo pipefail

cargo test --workspace
gh signoff
