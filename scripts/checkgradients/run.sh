#!/usr/bin/env bash
# Full check-gradients sweep on both Tesseracts' trajectory inputs.
# EXPENSIVE (~1 h+ for the meat side at tight tolerances) — intended as a
# one-time submission artifact (Phase D, ideally inside the built containers),
# not a per-change check. See docs/verification.md for interpretation notes:
# parameter vectors are excluded because a single absolute --eps cannot span
# 12 orders of magnitude; they are covered by test/test_gradients.jl.
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONIOENCODING=utf-8   # tool prints emoji; Windows cp1252 chokes

echo "=== pit: u_knots, qmeat_knots ==="
TESSERACT_API_PATH=../../tesseracts/pit/tesseract_api.py \
  uv run tesseract-runtime check-gradients "@pit_payload.json" \
  --input-paths u_knots --input-paths qmeat_knots --max-evals 200 --seed 42

echo "=== meat: tpit_knots ==="
TESSERACT_API_PATH=../../tesseracts/meat/tesseract_api.py \
  uv run tesseract-runtime check-gradients "@meat_payload.json" \
  --input-paths tpit_knots --max-evals 25 --seed 42
