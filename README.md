# Differentiable BBQ — BrisketDynamics

[![CI](https://github.com/JRock95/BrisketDynamics/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JRock95/BrisketDynamics/actions/workflows/CI.yml)

**Bayesian calibration and optimal control of a coupled smoker–brisket system,
built for the [Tesseract Hackathon 2026](https://pasteurlabs.ai/tesseract-hackathon-2026/)
(track 04: differentiable inference & UQ; cross-track 02: multi-physics & coupled systems).
Technical writeup: [docs/writeup.md](docs/writeup.md) · full equations: [docs/model.md](docs/model.md) ([diagram](figures/math_diagram.png)).**

Real BBQ is a coupled multi-physics control problem: a charcoal pit with its own
thermal dynamics drives heat into a brisket whose internal physics — coupled heat
conduction and moisture transport with surface evaporation — produce the famous
**stall**, a multi-hour plateau in internal temperature caused by evaporative
cooling. This repo wraps the whole coupled physics as one Julia [Tesseract](https://github.com/pasteurlabs/tesseract-core)
into one end-to-end differentiable function orchestrated from JAX:

| Component | Physics | States | Differentiation strategy |
|---|---|---|---|
| **Monolith Tesseract** | Pit ODE (energy balance + fuel burn + Minion ignition spread) + 1D slab heat-moisture PDE (denaturation expulsion, fat rendering, smoothed evaporative surface BC, collagen kinetics, Maillard + smoke bark states) + quasi-steady pit humidity, as ONE stiff system | T, W, C, F × 40 nodes + B, S + T_pit, m_fuel, ignited (165 ODEs, FBDF) | **Forward-mode ForwardDiff**: `jacobian_vector_product` = one Dual solve, `vector_jacobian_product` / `jacobian` = one chunked pass over the requested inputs |

Pit temperature → meat boundary condition and meat evaporative + sensible
load → pit energy balance are integrated together in one stiff solve, so
there is no coupling truncation. Derivatives are forward-mode through the
solve: a JVP is one Dual solve, a VJP or full Jacobian one chunked pass —
which is what makes Gauss–Newton calibration, Fisher covariances and
second-order control affordable ([docs/monolith.md](docs/monolith.md)). The
two-Tesseract composition this replaced (meat adjoint + pit Jacobian, K=2
Picard in JAX) is in the git history; its measured seam cost is in the same
doc.

## Results

| | |
|---|---|
| **Gradient correctness** | 3-level FD verification: Julia suite (physics; monolith Jacobian / JVP vs FD; Picard→monolith convergence) and the app suite (composed `jax.grad` gate, bark chain, MPC invariants). See [docs/verification.md](docs/verification.md) |
| **Synthetic calibration** | MAP + NUTS on the coupled model; the k_m–h_c ridge is the stall's cooling/heating trade-off ([corner](figures/corner.png), [predictive band](figures/posterior_predictive.png)) |
| **Real-cook capstone** | 19-h WSM cook (digitized with provenance): pit hold tracked ±3 °C, core stall **and** stall break reproduced ([fit](figures/cooklog_fit_map.png), [data](data/cook_log/PROVENANCE.md)). NUTS posterior: [corner](figures/corner_real.png), [predictive band](figures/posterior_predictive_real.png) |
| **Optimal control** | doneness 0.26 → **0.900** on target, surface under bark limit ([before/after](figures/control_before_after.png)) |
| **Robust control** | expected-objective schedule vs MAP-optimal, evaluated on posterior draws ([comparison](figures/control_robust.png)) |
| **MPC** | receding-horizon replan from the full meat field (T/W/C/F/B/S) handed through the Tesseract ([demo](figures/mpc_demo.png)) |
| **Cook-time planning** | quality scan over duration (9 h pre-color → 17 h with bark). The chance layer reports an **empirical fraction over N draws**, not a 95% certification ([figure](figures/timeplan.png)) |
| **Differentiable Texas crutch** | wrap time optimized jointly with the vents against predicted bark color ([figure](figures/wraptime.png)) |
| **Mechanistic bark model** | Maillard + smoke deposition as meat states, smoke flux = pit burn rate; hot-fast vs low-slow vs wrap reproduced ([figure](figures/bark_model.png)) |
| **The full composition** | duration × wrap × vents jointly optimized; current artifacts wrap near 3.5–3.8 h on a ~17 h horizon ([figure](figures/fullplan.png)) |
| **Animated cook** | calibrated T/W fields evolving through the real cook ([gif](figures/cook_animation.gif)) |

## Layout

```
src/, test/            Julia package: component solvers, the coupled monolith, physics + derivative tests
tesseracts/monolith/   Tesseract API (Pydantic schemas + juliacall bridge, threading notes)
app/                   uv-managed Python: JAX orchestrator, NumPyro calibration,
                       Optax control, plots; boundary + composed-gradient tests
scripts/               one script per result (see below) + container build
data/                  synthetic dataset, real cook log (+provenance), posteriors
docs/                  writeup, verification record, project brief
figures/               all generated figures
```

## Reproducing (native, no Docker)

Requires Julia ≥ 1.10 and [uv](https://docs.astral.sh/uv/). The Tesseract runs
in-process via `Tesseract.from_tesseract_api` (LocalClient); everything below
runs on a laptop (Windows included).

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl        # physics + Julia-level gradient tests

cd app
uv run pytest tests/ -q                   # boundary tests + composed FD gradient gate

# each result is one script (run from app/):
uv run python ../scripts/forward_demo.py        # coupled 12-h cook figure
uv run python ../scripts/run_map.py             # synthetic MAP + recovery table
uv run python ../scripts/run_nuts.py            # NUTS (checkpointed; resumable by re-running)
uv run python ../scripts/run_control.py         # damper-schedule optimization
uv run python ../scripts/run_robust_control.py  # control under posterior uncertainty
uv run python ../scripts/replay_cooklog.py      # real cook, uncalibrated replay
uv run python ../scripts/run_map_real.py        # real cook MAP calibration
uv run python ../scripts/run_vi.py --real       # real cook ADVI (evening-scale; checkpointed)
uv run python ../scripts/run_nuts.py --real     # real cook NUTS (long; checkpointed)
uv run python ../scripts/run_mpc.py             # receding-horizon MPC demo
uv run python ../scripts/run_timeplan.py        # cook-time-as-decision-variable
uv run python ../scripts/run_wraptime.py        # differentiable wrap-time (crutch)
uv run python ../scripts/run_bark.py            # bark model forward validation
uv run python ../scripts/run_fullplan.py        # duration x wrap x schedule
uv run python ../scripts/render_animation.py    # animated calibrated cook
```

NUTS runs checkpoint **every 5 steps including warmup** (power-loss-proof;
survived four battery deaths during development) — re-run the same command to
resume. ADVI (`run_vi.py`) checkpoints the SVI state the same way; it is
the overnight calibration path and does not replace NUTS.

## Containers (Linux/WSL2 + Docker)

```bash
./scripts/build_containers.sh all   # stages the Julia package, builds both images
```

The Julia toolchain and precompiled package are baked at build time
(`package_data` + custom build steps); `BRISKET_JL_DIRECT=1` in the container
env bypasses the LocalClient thread-executor (no JAX inside the container).
`scripts/checkgradients/run.sh` runs the full endpoint gradient sweep as a
verification artifact.

## License & data

Apache 2.0; all code authored during the hackathon window (Aug 3–31, 2026).
Real cook log digitized from a public 2013 post by "Mache" (The Virtual Weber
Bulletin Board) with attribution — see
[data/cook_log/PROVENANCE.md](data/cook_log/PROVENANCE.md).
