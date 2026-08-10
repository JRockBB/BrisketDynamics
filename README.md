# Differentiable BBQ — BrisketDynamics

[![CI](https://github.com/JRock95/BrisketDynamics/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JRock95/BrisketDynamics/actions/workflows/CI.yml)

**Bayesian calibration and optimal control of a coupled smoker–brisket system,
built for the [Tesseract Hackathon 2026](https://pasteurlabs.ai/tesseract-hackathon-2026/)
(track 04: differentiable inference & UQ; cross-track 02: multi-physics & coupled systems).**

Real BBQ is a coupled multi-physics control problem: a charcoal pit with its own
thermal dynamics drives heat into a brisket whose internal physics — coupled heat
conduction and moisture transport with surface evaporation — produce the famous
**stall**, a multi-hour plateau in internal temperature caused by evaporative
cooling. This repo composes two Julia [Tesseracts](https://github.com/pasteurlabs/tesseract-core)
into one end-to-end differentiable function orchestrated from JAX:

| Component | Physics | States | Differentiation strategy |
|---|---|---|---|
| **Meat Tesseract** | 1D slab heat + moisture PDE, smoothed evaporative surface BC, collagen kinetics | T, W, C × 40 nodes (120 ODEs, stiff, FBDF) | SciMLSensitivity **continuous adjoint** (`InterpolatingAdjoint` + ReverseDiff VJP) |
| **Pit Tesseract** | Lumped charcoal pit: energy balance + fuel burn | T_pit, m_fuel (Tsit5) | **Exact dense ForwardDiff Jacobian**, transposed |

The two are coupled **two-way** (pit temperature → meat boundary condition;
meat evaporative + sensible load → pit energy balance) via K=2 whole-horizon
Picard iteration, unrolled in JAX — so `jax.grad` of the composed cook crosses
the Julia↔JAX Tesseract VJP boundary four times, and the gradients are exact
for the unrolled map. To our knowledge this is the first public example of a
SciMLSensitivity continuous adjoint exposed as a Tesseract
`vector_jacobian_product`.

## Status (hackathon in progress)

- ✅ **Phase A** — physics + infrastructure: stall reproduced (Julia tests assert
  the plateau quantitatively), WSM-like pit hold, both Tesseracts serving
  `apply`/`abstract_eval`/`vector_jacobian_product` via juliacall, composed
  12-h forward cook, and a composed finite-difference gradient gate
  (**13/13 pass** — see [docs/verification.md](docs/verification.md))
- ⬜ **Phase B** — NumPyro calibration (MAP → NUTS) against a cook log
- ⬜ **Phase C** — Optax damper-schedule optimization for target doneness
- ⬜ **Phase D** — containerized builds, writeup, submission

## Layout

```
src/, test/            Julia package: solvers, VJPs, physics + gradient tests
tesseracts/{meat,pit}/ Tesseract API definitions (Pydantic schemas + juliacall bridge)
app/                   uv-managed Python: JAX orchestrator, tests; later calibration + control
scripts/               forward_demo.py (12-h coupled cook figure)
docs/                  project brief, verification record
```

## Running (native, no Docker)

Requires Julia ≥ 1.10 and [uv](https://docs.astral.sh/uv/). The Tesseracts run
in-process via `Tesseract.from_tesseract_api` (LocalClient) — containerized
builds come in Phase D.

```bash
# Julia physics + gradient tests
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl

# Python boundary + composed-gradient tests
cd app
uv run pytest tests/ -q

# 12-h coupled forward demo (writes figures/forward_cook.png)
uv run python ../scripts/forward_demo.py
```

## License

Apache 2.0. All code authored during the hackathon window (Aug 3–31, 2026).
