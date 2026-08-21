# Differentiable BBQ — BrisketDynamics

[![CI](https://github.com/JRock95/BrisketDynamics/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JRock95/BrisketDynamics/actions/workflows/CI.yml)

**Bayesian calibration and optimal control of a coupled smoker–brisket system,
built for the [Tesseract Hackathon 2026](https://pasteurlabs.ai/tesseract-hackathon-2026/)
(track 04: differentiable inference & UQ; cross-track 02: multi-physics & coupled systems).
Technical writeup: [docs/writeup.md](docs/writeup.md).**

Real BBQ is a coupled multi-physics control problem: a charcoal pit with its own
thermal dynamics drives heat into a brisket whose internal physics — coupled heat
conduction and moisture transport with surface evaporation — produce the famous
**stall**, a multi-hour plateau in internal temperature caused by evaporative
cooling. This repo composes two Julia [Tesseracts](https://github.com/pasteurlabs/tesseract-core)
into one end-to-end differentiable function orchestrated from JAX:

| Component | Physics | States | Differentiation strategy |
|---|---|---|---|
| **Meat Tesseract** | 1D slab heat + moisture PDE (denaturation expulsion + fat rendering), smoothed evaporative surface BC, collagen kinetics, bark states (Maillard + smoke deposition) | T, W, C, F × 40 nodes + B, S (162 ODEs, stiff, FBDF) | SciMLSensitivity **continuous adjoint** (`InterpolatingAdjoint` + `EnzymeVJP`) |
| **Pit Tesseract** | Lumped charcoal pit: energy balance + fuel burn + Minion ignition spread | T_pit, m_fuel, ignited fraction (Tsit5) | **Exact dense ForwardDiff Jacobian**, transposed |

The two are coupled **two-way** (pit temperature → meat boundary condition;
meat evaporative + sensible load → pit energy balance) via K=2 whole-horizon
Picard iteration, unrolled in JAX — so `jax.grad` of the composed cook crosses
the Julia↔JAX Tesseract VJP boundary four times, and the gradients are exact
for the unrolled map. To our knowledge (survey of pasteurlabs/mosaic, the
official examples, and the si-tesseract forum) this is the first public example
of a SciMLSensitivity continuous adjoint exposed as a Tesseract
`vector_jacobian_product`.

## Results

| | |
|---|---|
| **Gradient correctness** | 3-level FD verification: Julia suite 92 tests (33 adjoint-vs-FD), app suite 21 incl. the composed `jax.grad` gate over every input class and the cross-component bark chain ([docs/verification.md](docs/verification.md)) |
| **Synthetic calibration** | MAP recovers 7/10 params ≤10%; NUTS posterior with truths inside everywhere; the k_m–h_c ridge is the stall's cooling/heating trade-off ([corner](figures/corner.png), [predictive band](figures/posterior_predictive.png)) |
| **Real-cook capstone** | 19-h WSM cook (digitized with provenance): pit hold tracked ±3 °C, core stall **and** stall break reproduced ([fit](figures/cooklog_fit_map.png), [data](data/cook_log/PROVENANCE.md)) |
| **Optimal control** | doneness 0.26 → **0.900** on target, surface under bark limit ([before/after](figures/control_before_after.png)) |
| **Robust control** | expected-objective schedule halves doneness spread across the posterior: 0.780±0.123 → 0.863±0.069 ([comparison](figures/control_robust.png)) |
| **MPC** | hourly re-planning absorbs an unmodeled −15 K cold front: doneness 0.92 vs 0.68 open-loop ([demo](figures/mpc_demo.png)) |
| **Cook-time planning** | the answer moved as quality completed (9 h pre-color → 17 h with bark); the posterior honestly refuses to certify an open-loop start ([figure](figures/timeplan.png)) |
| **Differentiable Texas crutch** | wrap time optimized jointly with the vents against predicted bark color ([figure](figures/wraptime.png)) |
| **Mechanistic bark model** | Maillard + smoke deposition as meat states, smoke flux = pit burn rate; hot-fast vs low-slow vs wrap reproduced ([figure](figures/bark_model.png)) |
| **The full composition** | duration × wrap × vents jointly optimized rediscovers the traditional ~17 h unwrapped low-and-slow ([figure](figures/fullplan.png)) |
| **Animated cook** | calibrated T/W fields evolving through the real cook ([gif](figures/cook_animation.gif)) |

## Layout

```
src/, test/            Julia package: solvers, adjoint VJPs, physics + gradient tests
tesseracts/{meat,pit}/ Tesseract APIs (Pydantic schemas + juliacall bridge, threading notes)
app/                   uv-managed Python: JAX orchestrator, NumPyro calibration,
                       Optax control, plots; boundary + composed-gradient tests
scripts/               one script per result (see below) + container build
data/                  synthetic dataset, real cook log (+provenance), posteriors
docs/                  writeup, verification record, project brief
figures/               all generated figures
```

## Reproducing (native, no Docker)

Requires Julia ≥ 1.10 and [uv](https://docs.astral.sh/uv/). The Tesseracts run
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
resume.

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
