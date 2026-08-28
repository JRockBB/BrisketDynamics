#!/usr/bin/env bash
# Build the Tesseract containers (run inside WSL2/Linux with Docker).
#
#   ./scripts/build_containers.sh [monolith]
#
# Stages a copy of the Julia package into each tesseract's build context
# (docker COPY cannot reach outside the context), then runs `tesseract build`.
# The tesseract CLI is expected at /opt/tess/bin/tesseract (see docs) or on
# PATH.
set -euo pipefail
cd "$(dirname "$0")/.."

TESS=${TESSERACT_BIN:-/opt/tess/bin/tesseract}
command -v "$TESS" >/dev/null 2>&1 || TESS=tesseract

stage_and_build() {
  local name=$1
  local ctx="tesseracts/$name"
  echo "=== staging Julia package into $ctx/BrisketDynamics ==="
  rm -rf "$ctx/BrisketDynamics"
  mkdir -p "$ctx/BrisketDynamics"
  cp -r src Project.toml Manifest.toml "$ctx/BrisketDynamics/"
  echo "=== tesseract build $ctx ==="
  "$TESS" build "$ctx"
}

case "${1:-all}" in
  monolith|"") stage_and_build monolith ;;
  *) echo "usage: $0 [monolith]"; exit 2 ;;
esac
