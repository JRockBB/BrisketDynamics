# Project Plan: Differentiable BBQ — Bayesian Calibration & Optimal Control of a Coupled Smoker–Brisket System

**Target:** Tesseract Hackathon 2026 (Aug 3–31, submission due Aug 31 11:59 PM AoE)
**Primary track:** 04 — Differentiable inference & UQ. **Cross-track:** 02 — Multi-physics & coupled systems.
**Team:** Solo, weekend-warrior cadence. Remaining working windows: Aug 8–9, 15–16, 22–23, 29–30.
**Developer profile:** Expert Julia (SciML ecosystem), comfortable with JAX/NumPyro.

---

## 0. Instructions to Cursor

Your job is to turn this document into a detailed, phased implementation plan (not code yet), which will be reviewed against the hackathon judging rubric before any implementation begins. Constraints on your plan:

1. **Honesty over ambition.** Every claim in the eventual writeup must be literally true of the code. Do not plan demos that fake gradients, hard-code results, or use Tesseract as a wrapper around functionality that never crosses a real boundary. The judges explicitly penalize "Tesseract as a costume."
2. **The boundary must be load-bearing.** The architecture below is designed so that Julia adjoint sensitivities compose with JAX autodiff *only through* the Tesseract VJP interface. Your plan must preserve this: no reimplementing the solvers in JAX, no pure-Julia fallback for the outer loops.
3. **Stage everything with fallbacks.** Each phase must produce a working, demoable state even if later phases slip. Flag any step where you believe Tesseract lacks a needed feature — forking/patching Tesseract is allowed and rewarded, but must be identified early.
4. **All code written during Aug 3–31,** licensed Apache 2.0, in a public GitHub repo with a reproducible README.
5. Where this document leaves a modeling decision open (marked **[DECIDE]**), propose a default with justification rather than silently choosing.

---

## 1. One-paragraph pitch

Real BBQ is a coupled multi-physics control problem: a firebox with its own thermal dynamics drives heat into a brisket whose internal physics — coupled heat conduction and moisture transport with surface evaporation — produce the famous "stall," a multi-hour plateau in internal temperature caused by evaporative cooling. We compose two Tesseracts (a lumped-parameter pit ODE model and a stiff 1D heat–moisture PDE meat model, both Julia with continuous adjoint sensitivities) into a single end-to-end differentiable function orchestrated from JAX. Gradients do two jobs: (1) NumPyro NUTS performs full Bayesian calibration of unidentifiable-from-literature pit and meat parameters against a real public cook log, where every leapfrog step pulls adjoint gradients across the Julia↔JAX boundary; (2) Optax optimizes the damper/fuel schedule to hit a target doneness profile, with gradients flowing through the two-way pit↔meat coupling. Stretch: schedule optimization in expectation over the posterior — control under calibrated uncertainty.

---

## 2. Why Tesseract (judging criterion 3 — write the plan to protect this)

- **AD-strategy boundary:** The meat PDE is stiff with a moving evaporation front; the correct differentiation strategy is continuous adjoint via SciMLSensitivity, not naive autodiff. The outer loops (NUTS, Optax) live in JAX. There is no native way to `jax.grad` through a Julia ODE solve. Each Julia Tesseract exposes its adjoint as a custom VJP; JAX consumes the composed pipeline as one differentiable function. This is exactly the Julia↔JAX / adjoint↔autodiff composition the hackathon brief diagrams.
- **Language & dependency boundary:** Julia runtime + SciML stack vs Python + JAX/NumPyro, containerized separately.
- **Two components, one function:** Pit and meat are *separate* Tesseracts coupled in the JAX orchestrator, with two-way coupling (pit temperature → meat boundary condition; meat evaporative load → pit energy balance). Gradients flow through the coupling in both directions.
- **Component reuse:** The identical meat Tesseract is consumed unchanged by two different workflows (inference and control). Infrastructure, not a script.
- **Anti-costume check for Cursor:** at plan review, you must be able to answer "could this whole project have been done in pure Julia with Turing.jl?" with "not without abandoning NumPyro/Optax or reimplementing a solver" — and the architecture must make that answer true.

---

## 3. Physical models

### 3.1 Meat Tesseract (Julia, SciML, continuous adjoint)

- **Geometry:** 1D radial (infinite-cylinder or sphere approximation of a brisket flat/point). **[DECIDE]** cylinder vs sphere vs 1D slab; propose based on identifiability and literature convention. 30–50 spatial nodes (method of lines).
- **State:** temperature field T(r,t) and moisture content W(r,t).
- **Heat:** conduction with temperature/composition-dependent properties from **Choi–Okos correlations** (cite); convective + radiative boundary at the surface driven by pit temperature (input).
- **Moisture:** effective diffusion; surface evaporation providing the evaporative-cooling sink that produces the stall.
- **The stall mechanism (required):** surface energy balance where evaporation flux depends on surface moisture availability and the pit's vapor/wet-bulb conditions. **Critical numerical requirement:** the evaporation switch must be **smoothed** (e.g., sigmoid on moisture availability / wet-bulb condition) rather than a hard discontinuity, specifically to keep adjoint gradients well-behaved and the NUTS likelihood surface smooth. **[DECIDE]** the exact smoothed formulation; propose 1–2 candidates from the food-drying literature with the smoothing applied.
- **Doneness diagnostic (output, not state):** collagen denaturation progress via Arrhenius first-order kinetics integrated along T(r,t) — literature parameters. Used as the control target in Phase C.
- **Differentiation:** SciMLSensitivity continuous adjoint (interpolating checkpointing as needed). Expose VJPs w.r.t. all physical parameters and the boundary-condition input trajectory.
- **Tesseract interface:** inputs = parameter vector, pit-temperature trajectory (discretized), geometry; outputs = surface + core temperature trajectories, evaporative mass-flux trajectory, final doneness field.

### 3.2 Pit Tesseract (Julia, ODE, analytic or ForwardDiff adjoint — deliberately a *different* differentiation strategy than the meat)

- **Lumped-parameter model, 2–3 states:** pit gas/wall temperature(s), remaining fuel mass.
- **Physics:** fuel burn rate as a function of damper setting (input schedule) and remaining fuel; heat release; wall losses to ambient; heat absorbed by the meat (input: evaporative + sensible load from meat Tesseract).
- **Sanity anchor:** parameters must be able to reproduce the informally-documented behavior of a Weber Smokey Mountain-class cooker (holds ~225–275 °F for 8+ h on a full charcoal ring).
- **Tesseract interface:** inputs = parameter vector, damper/fuel schedule, meat heat-load trajectory; outputs = pit temperature trajectory, fuel consumption.

### 3.3 Coupling (JAX orchestrator)

- Two-way: iterate pit ↔ meat to consistency per outer step, or co-integrate on a shared time grid via operator splitting. **[DECIDE]** coupling scheme; must remain differentiable end-to-end through the composed VJPs. Weak coupling (few Picard iterations on a coarse exchange grid) is acceptable if gradients remain correct — plan a gradient-correctness test (finite-difference check of the *composed* pipeline) as a required milestone.

---

## 4. Data

- **Source:** one public cook log scavenged from BBQ forums / ThermoWorks blog / amazingribs.com — need pit temperature and internal meat temperature time series for a single brisket cook exhibiting a clear stall. Document provenance and license/attribution in the repo.
- **Fallback (planned, honest):** synthetic-data validation — simulate with known parameters + realistic noise, show posterior recovery. This is standard SBI/UQ practice and will be labeled as such; the real-log calibration is the capstone if a usable log is found.

---

## 5. Phase A — Infrastructure & forward model (Aug 8–9, spillover to 15)

1. Repo scaffolding from `cookiecutter-tesseract`; CI that builds both Tesseract containers.
2. Meat Tesseract forward solve; verify stall appears for literature parameters. Plot: internal temp trajectory with visible stall.
3. Pit Tesseract forward solve; verify WSM-like hold behavior.
4. JAX orchestrator composing both; forward coupled simulation of a full 12-h cook.
5. **Gate:** composed finite-difference vs adjoint gradient check passes (relative error tolerance to be proposed).

## 6. Phase B — Bayesian calibration (Aug 15–16, 22–23)

- **Latents (≈7–9):** pit — thermal mass, wall-loss coefficient, burn-rate/damper parameters (3–4); meat — effective moisture diffusivity, evaporation coefficient, geometry nuisance (size), possibly conductivity multiplier.
- **Priors:** weakly-informative, literature-centered; document every prior's source.
- **Likelihood:** Gaussian observation noise on pit + internal temperature trajectories; noise scales as additional latents.
- **Machinery:** NumPyro. Stage 1: MAP (validates the full gradient path cheaply, initializes NUTS). Stage 2: full NUTS posterior.
- **Fallback (pre-authorized):** if NUTS is intractable in the final weekend, ship MAP + Laplace with an honest sentence in the writeup. Do not silently degrade.
- **Key figure:** corner plot showing which parameters the stall identifies (expect: moisture diffusivity tight; evaporation coefficient correlated with geometry). Narrative point: *the stall is the informative feature*.

## 7. Phase C — Optimal cook control (Aug 22–23, 29–30)

- **Decision variables:** damper schedule (piecewise-linear or spline knots) and optionally fuel-loading events.
- **Objective:** hit target doneness (collagen conversion in the core) by a deadline, penalties for overshooting surface temperature (bark burning proxy) and fuel use. **[DECIDE]** exact weighting; propose defaults.
- **Optimizer:** Optax (Adam or L-BFGS via jaxopt) through the full coupled pipeline at posterior-mean parameters.
- **Required figure:** objective vs iteration + before/after temperature and doneness trajectories.
- **Stretch (only if Phases A–B are done):** optimize expected objective over N posterior samples — control under calibrated uncertainty.

## 8. Phase D — Writeup, visuals, submission (Aug 29–31)

- 2–4 page technical writeup naming track 04, structured around the six judging criteria in order; the "why Tesseract" section reuses §2 verbatim-in-spirit.
- README with one-command reproduction (docker compose or equivalent); pin everything.
- Visuals: (1) animated cook — temperature/moisture fields evolving with the stall visible, (2) corner plot, (3) control before/after. Optional ≤5-min demo video.
- Submission form + LinkedIn post tagging Pasteur Labs & ISI and Tesseract; forum showcase post.

## 9. Risk register (Cursor: extend this)

| Risk | Mitigation |
|---|---|
| Adjoint fragile through evaporation front | Smoothed switch (mandatory, §3.1); gradient-check gate in Phase A |
| NUTS too slow | Coarse spatial grid; MAP+Laplace fallback (§6) |
| No usable public cook log | Synthetic-data validation labeled honestly (§4) |
| Two-way coupling breaks differentiability | Coupling scheme chosen for VJP-compatibility; composed FD check |
| Tesseract missing a needed feature | Identify in plan review; fork/patch is allowed and separately prized |
| Weekend time slips | Every phase gate leaves a demoable state; Phase C stretch is severable |

## 10. Deliverables checklist

- [ ] Public GitHub repo, Apache 2.0, code authored Aug 3–31
- [ ] Two composed Tesseracts, end-to-end gradients, composed gradient-correctness test
- [ ] Calibration results (NUTS posterior, or MAP+Laplace honestly labeled)
- [ ] Control results (objective improving, recovered schedule)
- [ ] Writeup (track 04 named, why-Tesseract case, criteria-ordered)
- [ ] Visuals + optional video
- [ ] Submission form, LinkedIn post, forum showcase