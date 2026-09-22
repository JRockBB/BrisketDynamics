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

### Windows: one Julia for the app and the package

The app drives the Julia package in-process through juliacall, which keeps a
Julia project of its own (`app/.venv/julia_env`) and, left to itself, picks a
Julia of its own through juliaup. On this machine that was the `release`
channel (1.12) while the package, the estimator and CI run 1.11: juliacall's
project would not precompile PythonCall (its own dependency JSON was never
instantiated for that Julia), the Tesseract API failed to load, and every
Julia-backed app test errored at setup. Two things fix it:

- both Tesseract APIs (`tesseracts/monolith/tesseract_api.py`,
  `tesseracts/slab2d/tesseract_api.py`) now point juliacall at the `julia`
  on PATH (`PYTHON_JULIAPKG_EXE`, resolved through juliaup's shim) and at
  the repository project itself (`PYTHON_JULIAPKG_PROJECT`; PythonCall is a
  direct dependency of the package), so one manifest decides every version
  and the app uses the Julia the repository was instantiated with. A project
  of juliacall's own loads `Preferences` at another version, and the
  package's dependencies then cannot precompile inside the same session.
  Set either variable yourself to choose otherwise;
- juliapkg then manages the repository project: it keeps `PythonCall` pinned
  exactly, `OpenSSL_jll` among the dependencies, and drops a `lock.pid` at the
  root (ignored). If the package will not load, instantiate the repository
  project with that same Julia: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`.

When the Julia runtime cannot start, the app suite reports it once: the
first Julia-backed test fails with the cause and this repair, and the other
Julia-backed tests are skipped (`app/tests/conftest.py`).

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
