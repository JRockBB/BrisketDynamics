# Install story

What it takes to run this stack. The native clone is the supported path;
the container path exists for Linux/WSL2. Publishing to PyPI or a Julia
registry is NOT planned — the clone (or a container) is the
distribution. The analysis below records what publishing would take, so
the decision is documented rather than open.

## Supported today: clone + uv (native)

The stack is REPO-ANCHORED: the Julia package (`Project.toml` at the
root), the Python app (`app/`, its own uv project), the canonical
scripts (`scripts/`), the log registry and calibrations (`data/`,
`tools/loghunt/logs/`) all resolve relative to the clone. Prerequisites
are Julia >= 1.10 and uv; the full recipe is README "Reproducing
(native, no Docker)", and the user-facing chain is the `brisket` CLI
(README "Using it on your own cook"). First runs JIT-compile the Julia
side (~1 min); the Tesseract runs in-process (LocalClient), no Docker.

## Containers (Linux/WSL2)

`scripts/build_containers.sh` builds the packaged Tesseract images
(`Tesseract.from_image`) — the hackathon submission path. Works where
Docker works; the native path is faster for development.

## What blocks `pip install brisket` (the honest gap)

1. **Julia dependency resolution.** The app activates the CLONED Julia
   project by path (`Pkg.activate(REPO)`). A wheel would need a
   `juliapkg.json` declaring BrisketDynamics + deps so juliacall can
   resolve them into a managed depot on first import — mechanical, but
   it requires publishing the Julia package (a registry, or a pinned
   git source) and versioning the two halves together.
2. **Repo-relative layout.** `cli.py`, the scripts and the registry
   resolve `REPO/...` paths (`data/calibration`, `figures`,
   `tools/loghunt`). A package needs a data-dir convention (platformdirs
   for outputs; packaged read-only data for priors/registry; scripts
   folded into the package as modules instead of `runpy` targets).
3. **Weight.** JAX + juliacall + a stiff-ODE Julia stack is a heavy
   first-install regardless of packaging; the container is the better
   distribution unit for non-developers.

The decision: none of (1)-(3) is planned. The clone is the package; a
container is the hand-off unit if one is ever needed. This section
exists so the trade-off stays documented, not as a to-do.
