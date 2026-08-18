#!/usr/bin/env bash
# In-container variant of run.sh: same sweep, executed inside the built
# images (the submission artifact — verifies the *shipped* containers, not
# the dev environment). Run from WSL2 with the images built:
#   bash scripts/checkgradients/run_container.sh
# See run.sh / docs/verification.md for scope notes (parameter vectors are
# covered by test/test_gradients.jl; a single --eps cannot span 12 decades).
set -uo pipefail
cd "$(dirname "$0")"

# same CLI resolution as build_containers.sh
TESS=${TESSERACT_BIN:-/opt/tess/bin/tesseract}
command -v "$TESS" >/dev/null 2>&1 || TESS=tesseract

echo "=== pit (in-container): u_knots, qmeat_knots ==="
"$TESS" run brisket-pit:0.1.0 check-gradients "@pit_payload.json" \
  --input-paths u_knots --input-paths qmeat_knots --max-evals 200 --seed 42 \
  || { echo "PIT SWEEP FAILED (exit $?)"; exit 1; }

echo "=== meat (in-container): tpit_knots ==="
"$TESS" run brisket-meat:0.1.0 check-gradients "@meat_payload.json" \
  --input-paths tpit_knots --max-evals 25 --seed 42 \
  || { echo "MEAT SWEEP FAILED (exit $?)"; exit 1; }

echo "SWEEP COMPLETE"
