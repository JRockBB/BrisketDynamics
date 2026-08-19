#!/usr/bin/env bash
# In-container variant of run.sh: same sweep, executed inside the built
# images (the submission artifact — verifies the *shipped* containers, not
# the dev environment). Run from WSL2 with the images built:
#   bash scripts/checkgradients/run_container.sh
# See run.sh / docs/verification.md for scope notes (parameter vectors are
# covered by test/test_gradients.jl; a single --eps cannot span 12 decades)
# and for the interpretation of the tool-flagged failures (all benign FD
# artifacts at measure-zero kink/onset elements; analyze_failures.py
# classifies them mechanically).
#
# TODO(next sweep): meat_payload.json predates the t_wrap input (defaults
# to never-wrap, so existing results stand); add t_wrap to the payload and
# an --input-paths t_wrap direction to cover the wrap gradient in-container.
#
# Notes from the first run (2026-08-18):
# - `tesseract run` does NOT forward check-gradients flags — invoke the
#   container runtime entrypoint directly via `docker run`.
# - WSL tears down idle VMs and SIGTERMs detached containers mid-Julia-load;
#   run this script FOREGROUND from a live WSL session (or under a process
#   that holds the session open, e.g. `docker logs -f`).
set -uo pipefail
cd "$(dirname "$0")"

HERE=$(pwd)
DR="docker run --rm -v $HERE:/pl"

echo "=== pit (in-container): u_knots, qmeat_knots ==="
$DR brisket-pit:0.1.0 check-gradients @/pl/pit_payload.json \
  --input-paths u_knots --input-paths qmeat_knots \
  --max-evals 200 --max-failures 500 --seed 42
echo "pit exit: $?"

echo "=== meat (in-container): tpit_knots ==="
$DR brisket-meat:0.1.0 check-gradients @/pl/meat_payload.json \
  --input-paths tpit_knots --max-evals 25 --max-failures 500 --seed 42
echo "meat exit: $?"

echo "SWEEP COMPLETE — classify flagged failures with:"
echo "  python analyze_failures.py <saved logs>"
