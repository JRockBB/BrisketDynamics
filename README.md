# Differentiable BBQ — BrisketDynamics

[![CI](https://github.com/JRock95/BrisketDynamics/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JRock95/BrisketDynamics/actions/workflows/CI.yml)

**Bayesian calibration and optimal control of a coupled smoker–brisket system,
built for the [Tesseract Hackathon 2026](https://pasteurlabs.ai/tesseract-hackathon-2026/)
(track 04: differentiable inference & UQ; cross-track 02: multi-physics & coupled systems).
Technical writeup: [docs/writeup.md](docs/writeup.md) · results record: [docs/results.md](docs/results.md) · pooling write-up: [docs/pooling.md](docs/pooling.md) · install story: [docs/install.md](docs/install.md) · full equations: [docs/model.md](docs/model.md) ([diagram](figures/model/math_diagram.png)).**

Real BBQ is a coupled multi-physics control problem: a charcoal pit with its own
thermal dynamics drives heat into a brisket whose internal physics — coupled heat
conduction and moisture transport with surface evaporation — produce the famous
**stall**, a multi-hour plateau in internal temperature caused by evaporative
cooling. This repo wraps the whole coupled physics as one Julia [Tesseract](https://github.com/pasteurlabs/tesseract-core)
into one end-to-end differentiable function orchestrated from JAX:

| Component | Physics | States | Differentiation strategy |
|---|---|---|---|
| **Monolith Tesseract** | Pit ODE (energy balance + fuel burn + Minion ignition spread, water pan, two-zone firebox) + 1D slab heat–moisture–salt PDE (denaturation expulsion, fat rendering, smoothed evaporative surface BC, two-route collagen kinetics, Maillard + smoke bark states, smoke ring, state-triggered wrap latch) + quasi-steady pit humidity, as ONE stiff system | five fields (T, W, C, F, salt) × nx + four meat scalars (B, S, R, wrap latch) + six pit states (T_pit, m_fuel, ignited, m_pan, T_pan, T_firebox): 5·40 + 4 + 6 = **210 ODEs** at the default nx = 40, FBDF | **Forward-mode ForwardDiff**: `jacobian_vector_product` = one Dual solve, `vector_jacobian_product` / `jacobian` = one chunked pass over the requested inputs |

Pit temperature → meat boundary condition and meat evaporative + sensible
load → pit energy balance are integrated together in one stiff solve, so
there is no coupling truncation. Derivatives are forward-mode through the
solve: a JVP is one Dual solve, a VJP or full Jacobian one chunked pass —
which is what makes Gauss–Newton calibration, Fisher covariances and
second-order control affordable ([docs/monolith.md](docs/monolith.md)).

## Results

**Estimator status (2026-09-27):** `sampled_init` and `sampled_fit` provide joint non-Gaussian online
and offline inference. See the [API and limitations](docs/sampled-estimation.md) and
[current validation record](docs/validation/2026-09-27-online/README.md). The four-suite comparison
failed: one final online update exhausted sampling precision after about 94 minutes. Live-use performance
and real-cook calibration remain unresolved; the existing Gaussian pilot is a separate path. The estimator
results below describe earlier implementations, not a general consistency guarantee for the current backend.

| | |
|---|---|
| **Derivative correctness** | 3-level FD verification: Julia (residual Jacobian / JVP vs FD; Picard→monolith convergence), the Tesseract endpoints against each other (jvp = J·v, vjp = Jᵀ·w), and `jax.grad` through `coupled_cook`. See [docs/verification.md](docs/verification.md) |
| **Real-cook calibration** | 19-h WSM cook (digitized with provenance): Gauss–Newton MAP from the prior medians in 14 iterations / 103 s, pit rms 1.3 K, stall **and** stall break reproduced ([fit](figures/calibration/cooklog_fit_map.png), [data](data/cook_log/PROVENANCE.md)) |
| **Posterior** | NUTS on the monolith, 2 chains × (300 + 600) in 21.6 h each: 0 divergences, split-r̂ ≤ 1.010, ESS 169–900 ([corner](figures/calibration/corner_nuts_real.png), [predictive](figures/calibration/posterior_predictive_nuts_real.png)). It confirms the curvature: the Gauss–Newton Fisher and the Laplace match it within ~10–20 % on 13 of 15 sites, ADVI is 5–10× too narrow ([widths](figures/calibration/posterior_widths.png)) — meat side pinned, pit side along two ridges |
| **The plan** | cook length × wrap × pull × chimney × spritz policy × vents optimized jointly, quality scored at serve: the 11 h cook wraps at 7.1 h, pulls at 8.4 h, rests 2.6 h, adds no fuel and does not spritz ([figure](figures/control/fullplan.png)) — 13 cook lengths in 95 min |
| **MPC** | hourly replans of vents, crutch, pull, chimney and spritz policy from the full meat field through the Tesseract; a 15 K cold front the open-loop plan does not survive (doneness 0.57) is absorbed (0.945) ([demo](figures/control/mpc_demo.png)) — 4 cooks in 68 min |
| **Robust control** | E[J] over 8 posterior draws vs the plan-optimal schedule on 30 held-out draws: half the spread, the hedge is in the wrap and pull times ([comparison](figures/control/control_robust.png)) |
| **Online estimator** | the calibration as an RxInfer factor graph in Julia ([estimator/](estimator/)): one node per whitened residual, one latent block, the monolith affine around a point the graph itself re-centers from inside `infer` (the loop lives in the graph). On the real 19 h cook it reproduces the Gauss–Newton MAP within a third of a standard deviation and the Fisher widths within 3 %, 17 outer iterations in about two minutes; the channel registry then ranks sensors by Fisher information before they exist: humidity first, then a scale, then a second probe; the noise of every channel is identified as the cook goes (RxInfer's AR node on the residuals, the calibration's posterior as the prior); hourly streaming refits on mock plants show it working (with the right physics, humidity plus a scale put every latent within 1.5 sd of truth, on a slab or on a cylinder read by the 2-D engine) and failing honestly (the wrong physics is confident and 17 sd wrong, and the misfit check every refit reports flags it, from the first hour through the humidity sensor); it reads like a real cook (channels at their own rates, a censored humidity sensor, the pit's noise level a process on the graph that follows the fire) and its pilot replans when a replan is worth making rather than on a clock, closing the loop on a mock plant read like a real cook with goals in a pitmaster's terms; it optimises the app's own quality objective, ported term for term, in the app's own decision space (a serve bounded at 24 h, priors that are numerically nothing, a scan across the damper's two basins from a warm and a cold start), and starts the next cook from the calibration chain's posterior written site by site in the graph's own families by the pool's roles (the cooker's chain, the meat widened by the between-cook variance, the flat's thickness measured); against the app's MPC on the same brisket under that one objective on an honest bench the batch pilot landed at J 0.84 to 0.68 for the MPC flown as an oracle and 0.76 for the MPC flown with the calibration's parameters (a 256 °F pit wrapped at 10.8 h and pulled at 15.7 h at 91 °C), and the pilot on the filter lands at 0.87 on the same plant (a 260 °F pit wrapped at 10.3 h and pulled at 16.1 h at 90 °C, no burn, the fire's sites within 1.3 sd of the truth) once an event the belief propagates through blind leaves its state uncertain where the next reading can see it ([docs/results.md](docs/results.md), the entries of 2026-09-17); a coverage test over 21 cooks with truths drawn from that prior finds the posterior calibrated on the identified meat sites (spread within 7 % of one, 95 % inside two sd) and overconfident by two to three on the fire's ridge and the late water sites, no bias overall ([coverage](figures/estimator/coverage_realcook.png), [docs/results.md](docs/results.md)); on a plant whose parameters MOVE (the leak and the surface coefficients on random walks, the fuel add's charge jumping) the batch refit corrupts even the meat sites (pooled spread 2.6, 8 % of errors beyond 4 sd) and a smoother that carries the moving sites piecewise constant over the full history with a random-walk prior between the pieces, the cook solved in segments with the state handed off exactly, is as calibrated there as the batch estimator was on a constant plant (spread 1.47, 86 % inside two sd, at the same cost) ([smoother coverage](figures/estimator/coverage_realcook_driftsmoother.png)); flown against the batch pilot on three drifting plants under one planner the smoother pilot won every cook (J 0.80 against 0.91); and the belief is updated at every arrival of a sample, on the same graph, with the plant's state and every instrument's discrepancy in it (`estimator/src/sequential.jl`), carried through the physics by sigma points rather than along a tangent, its state widths earned on a reproducible bench and judged on seeds it never saw: on nine fresh drifting cooks the belief's pit, surface and centre widths are calibrated (NEES/3 of 0.92), the latents pool at 1.34 with every cook between 0.6 and 2.4, and 93 % of the errors sit within two sd, the four bounded-family sites' tails the open item ([the running belief](figures/estimator/filter_belief_drift.png), [the consistency tests](figures/estimator/filter_consistency_drift.png)) |
| **2-D shadow** | the planned cook re-simulated to serve on the axisymmetric 2-D Tesseract, pull and rest included: true cold-spot lethality and come-up time, bark distribution, shrunk dimensions, a doubled-grid discretization check, radius derived from the 1-D exposed area ([figure](figures/slab2d/slab2d_plan.png), [docs/results.md](docs/results.md)) |
| **Mechanistic bark model** | Maillard + smoke deposition as meat states, smoke flux = pit burn rate; hot-fast vs low-slow vs wrap reproduced ([figure](figures/model/bark_model.png)) |
| **Animated cook** | calibrated T/W fields evolving through the real cook ([gif](figures/model/cook_animation.gif)) |

## Layout

```
src/, test/            Julia package: component solvers, the coupled monolith, physics + derivative tests
tesseracts/monolith/   Tesseract API (Pydantic schemas + juliacall bridge, threading notes)
app/                   uv-managed Python: JAX orchestrator, NumPyro calibration,
                       Optax control, plots; boundary + composed-gradient tests
estimator/             Julia subproject: the online estimator as an RxInfer factor graph (its own Project.toml)
scripts/               one script per result (see below) + container build
data/                  synthetic dataset, real cook log (+provenance), posteriors
docs/                  writeup, equation sheet, results record, roadmap, review, verification record
figures/               one subdirectory per topic (model, calibration, validation, control, estimator, slab2d); figures/README.md lists every figure with its script
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
uv run python ../scripts/run_control.py         # damper-schedule optimization
uv run python ../scripts/run_robust_control.py  # control under posterior uncertainty
uv run python ../scripts/replay_cooklog.py      # real cook, uncalibrated replay
uv run python ../scripts/run_gn_map.py          # real cook Gauss-Newton MAP + Fisher
uv run python ../scripts/run_laplace.py         # Gauss-Newton Laplace draws + figures
uv run python ../scripts/run_tmb_ignition.py    # TMB/INLA nested-Laplace ignition grid
uv run python ../scripts/run_psis_laplace.py --proposal tmb --moment-match
                                                # PSIS check vs the true ODE (Pareto-khat,
                                                # moment-matching repair when khat is high)
uv run python ../scripts/run_hier_synth.py      # two-cook hierarchical pool (synthetic)
uv run python ../scripts/compare_approx.py      # W1 + width scoreboard vs NUTS
uv run python ../scripts/run_nuts_mono.py --chain 0   # real cook NUTS, one chain per process (long)
uv run python ../scripts/run_mpc.py             # receding-horizon MPC demo
uv run python ../scripts/run_bark.py            # bark model forward validation
uv run python ../scripts/run_sous_vide.py       # geometry modes vs Baldwin + exact conduction
uv run python ../scripts/run_fullplan.py        # J(t_f) landscape scan — the scout & the figure
uv run python ../scripts/run_duration_opt.py    # the chooser: joint gradient over (t_f, wrap, pull,
                                                # schedule), seeded from the scan (s = t/t_f)
uv run python ../scripts/run_start_time.py --serve 18:00 [--plan robust]
                                                # deadline -> when to light the fire (done-fraction
                                                # over posterior draws; the hedged plan needs NO
                                                # margin where the nominal needs 4.6 h)
uv run python ../scripts/render_animation.py    # animated calibrated cook
```

`run_nuts_mono.py` runs one chain per process and pickles the kernel state
**every 5 steps, warmup included** (`data/calibration/nuts_mono_real/`) —
re-run the same command to resume; `merge_nuts_mono.py` pools the chains and
reports r̂ / ESS. The cheap-posterior scoreboard (per-site 1-Wasserstein to
NUTS, Pareto-khat, wall time) lands in `data/calibration/approx_summary.md`.

## Using it on your own cook (`brisket` CLI)

The experimental deterministic factor-graph work is described in
[conditional estimation](docs/conditional-estimation.md), including offline and
online interfaces, nonlinear uncertainty transport, and the validation limits.
It has not replaced the live controller or established real-cook consistency.

The supported chain has one entry point (run from `app/`):

```bash
uv run brisket logs                       # the cook logs the repo knows
uv run brisket digitize my/log.toml       # chart screenshot -> CSV (tools/loghunt)
uv run brisket calibrate --csv my.csv --t-amb-k 294 --name mycook
uv run brisket plan  --params fit_mycook.json          # schedule optimum
uv run brisket pilot --params fit_mycook.json --pit http://heatermeter.local \
    --apikey XXXX --shadow                              # non-actuating first flight
```

`calibrate` fits any registry log or your own CSV (column convention in
[tools/loghunt/README.md](tools/loghunt/README.md): `t_min, pit_F, meat_F,
amb_F, fan_pct, setpoint_F, lid_event`, t = 0 at meat-on) with the
Gauss-Newton MAP + Laplace chain, and writes a `--params`-compatible
calibration that `plan` and `pilot` consume directly. `pilot` flies a cook
against a real HeaterMeter (LinkMeter v14 REST API) — or against
`brisket mock-pit`, the calibrated sim plant behind the same API, for a
hardware-free dry run. Real hardware uses real-time polling and explicit
human event acknowledgement by default; remove `--shadow` only when
actuation is intended. Multi-cook pooling: `brisket pool --set keg`.

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

Apache 2.0. The hackathon entry is the Aug 3–31, 2026 window; development
continued after it (the 2-D line, salt, the water pan, pooling, the
2026-09-07 review fixes — dated entries in [docs/results.md](docs/results.md)).
Real cook log digitized from a public 2013 post by "Mache" (The Virtual Weber
Bulletin Board) with attribution — see
[data/cook_log/PROVENANCE.md](data/cook_log/PROVENANCE.md).
