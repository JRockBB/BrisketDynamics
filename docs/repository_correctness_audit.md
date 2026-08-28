# BrisketDynamics repository correctness audit

> Historical audit of the two-Tesseract pipeline (pre-monolith).

**Date:** 2026-08-24. Historical snapshot — not the current defect list.
Current gaps: `docs/model_gaps.md`. Current equations: `docs/model.md`.

**Scope at audit time:** working tree, including untracked run artifacts  
**Method:** Read-only source review, equation/unit checks, literature cross-checks, test execution, artifact-schema inspection, and targeted numerical diagnostics

## Executive assessment

BrisketDynamics is a technically strong differentiable research prototype. The main heat/moisture finite-volume discretization is dimensionally coherent, the pit energy balance has correct units and signs, the solver choices are appropriate, and the whole-horizon default-state derivative verification is unusually thorough.

It is not currently reliable as:

- A receding-horizon MPC controller.
- A food-safety certifier.
- A source of converged posterior uncertainty for the current model.
- A quantitatively validated cook-duration, wrap-time, bark, juiciness, or doneness recommender.

The audit identified:

- **3 Critical findings**
- **13 High findings**
- **7 Medium findings**

The most important defects are:

1. MPC forward forecasts use the handed-off mid-cook state, but their custom derivatives use a fresh default brisket.
2. Food-safety terms are soft penalties evaluated from the surface rather than cold-spot feasibility checks.
3. Mismatched pit input lengths can silently change how concatenated controls and heat loads are interpreted.
4. Checked-in calibration/control artifacts belong to incompatible model eras.
5. The real-cook iid likelihood is contradicted by extreme residual autocorrelation.

## Validation performed

- Julia package tests: **100/100 passed** in 4m 08s.
- Python application tests: **25/25 passed** in 10m 49s.
- Ruff: two style-only diagnostics:
  - Import ordering in `app/tests/test_mpc.py`.
  - Imperative-mood docstring rule in `scripts/render_math_diagram.py`.
- Independent coupling study:
  - K=2 iteration delta: **6.69 K**
  - K=3 iteration delta: **0.55 K**
  - K=4 iteration delta: **0.058 K**
  - K=2 versus K=4 maximum difference:
    - Pit temperature: **0.51 K**
    - Core temperature: **0.20 K**
- Moisture diagnostic:
  - Reported retained-water fraction: **0.37203**
  - Finite-volume-weighted value: **0.37312**
  - Default purge rate exactly at its stated equilibrium: **0.02795 kg/kg/h**

Passing tests establish that many nominal trajectories and default-state derivatives match the implemented code. They do not establish physical validity, posterior convergence, safety feasibility, or correctness of the state-dependent MPC derivative.

## Equation verification

### Heat equation and spatial discretization

The implemented meat heat equation is

```math
\frac{\partial T}{\partial t}
=
\alpha \frac{\partial^2 T}{\partial x^2},
\qquad
\alpha = \frac{k_{\mathrm{mult}}K_{\mathrm{meat}}}{\rho c_p}.
```

The interior central difference, mirrored center boundary, and surface half-cell finite-volume balance are dimensionally and numerically consistent.

For a center-to-surface coordinate, the code correctly implements

```math
k\frac{\partial T}{\partial x}\bigg|_s = q_{\mathrm{net}}.
```

`docs/model.md:133-147` prints the opposite sign. The code is correct; the documentation is not.

### Moisture diffusion

The diffusion and surface mass-flux signs are correct:

```math
-\rho_dD_{\mathrm{eff}}
\frac{\partial W}{\partial x}\bigg|_s = j.
```

The local finite-volume discretization is conservative when half-cell endpoint weights are used. The reported global averages do not use those weights, however.

### Pit energy balance

The implemented balance

```math
C_{\mathrm{th}}\dot T_p
=
\Delta H_{\mathrm{eff}}\dot m_b
- UA(T_p-T_{\mathrm{amb}})
- q_{\mathrm{meat}}
```

has correct SI units and signs.

Using convective plus radiative input for `q_meat`, rather than net meat heating after latent loss, is physically sensible: the pit supplies surface heat, and latent heat subsequently leaves with vapor.

### Evaporation

The implemented mass-transfer expression is dimensionally correct:

```math
j =
(1-\eta_es)k_m\phi(W_s)
\frac{\operatorname{softplus}(p_{\mathrm{sat}}(T_s)-p_{v,\mathrm{pit}})}
{R_vT_s}.
```

It is not a thermodynamic surface-equilibrium model. A more standard driving force contains approximately

```math
a_w(W_s)p_{\mathrm{sat}}(T_s)-p_{v,\infty}.
```

The code computes water activity for bark but does not use it in evaporation. Condensation is impossible, and the pressure softplus remains positive at zero deficit.

### Collagen, fat, and bark kinetics

The first-order state forms are internally coherent and preserve their nominal bounds for valid parameters:

```math
\dot C = k_C(T)(1-C),
\qquad
\dot F = -k_f(T)F,
```

```math
\dot B = k_B(T_s,a_w)(1-B),
\qquad
\dot S = k_{\mathrm{dep}}\,\mathrm{wet}(a_w)\dot m_b(1-s)(1-S).
```

Their absolute physical scales are not sufficiently validated:

- Collagen `C=0.90` is not calibrated against texture or shear-force data.
- Bark kinetics extrapolate bakery data below its fitted temperature regime.
- `k0_bark` is multiplied by five as an effective adjustment.
- Normalized bark states are mapped directly to absolute CIELAB L* without color data.
- Smoke deposition is proportional to charcoal burn rather than measured smoke or phenolic emissions.
- Juiciness contains a placeholder fat weighting and can exceed one.

### Coupling

The K-pass unrolled map is differentiated correctly for tested whole-horizon default initial conditions.

That means AD is exact for the algorithm. It does not mean the algorithm is a converged physical fixed point.

With K=2:

- The returned pit sees `q^(1)`.
- The returned meat produces `q^(2)`.
- `q^(2)` is never fed back.

The returned pit and meat therefore are not a mutually consistent coupled pair.

### Likelihood

The real-cook likelihood correctly implements the documented total standard deviation:

```math
\sigma_{\mathrm{total}} = 1.2\,\mathrm{K} + \sigma_{\mathrm{excess}}.
```

NumPyro's `Normal` includes the required normalization, and no transform Jacobian is missing.

The iid residual assumption is not credible for the actual data.

## Detailed findings

## Critical

### C1. MPC differentiates a different initial state than its forward model

**Locations**

- `tesseracts/meat/tesseract_api.py:117-131`
- `tesseracts/meat/tesseract_api.py:468-482`
- `src/meat.jl:756-758`
- `app/src/brisket_app/mpc.py:265-280`

The primal meat endpoint receives handed-off:

- Temperature field.
- Moisture field.
- Collagen field.
- Fat field.
- Maillard bark state.
- Smoke-deposition state.

The custom VJP omits all of them and builds its adjoint problem with `_meat_u0(cfg)`.

Every MPC replan after the first segment therefore uses a gradient that is not the derivative of its own forecast.

**Impact**

The current MPC schedule and wrap-retiming results are not trustworthy. The latest artifact provides corroborating evidence:

- Defended reference: `J = 1.0307`
- Open-loop disturbed: `J = 1.2031`
- Nominal MPC: `J = 1.2531`
- Disturbed MPC: `J = 1.2047`

MPC does not improve the current disturbed objective and substantially worsens nominal performance.

**Required remediation**

Construct the VJP problem with the exact same T/W/C/F/B/S initial values used by the primal. Add a composed finite-difference test from a nonuniform mid-cook state before accepting any MPC result.

### C2. The safety formulation does not establish food safety

**Location:** `app/src/brisket_app/control.py:238-260`

Lethality and danger-zone dwell are computed from the surface trace. USDA FSIS time/temperature tables require the stated condition in all parts of the product.

The terms are also soft objective penalties. There is no:

- Hard feasibility constraint.
- Unsafe-plan rejection.
- Post-optimization safety status.
- Cold-spot or minimum-field calculation.

**Impact**

The current result is not a safety certification. It is particularly unsuitable for injected or mechanically tenderized meat, altered objective weights, long come-up times, or an under-converged optimization.

**Required remediation**

- Evaluate cold-spot/core and surface hazards explicitly.
- State the intact-muscle assumption.
- Expose raw lethality and dwell residuals.
- Require post-solve feasibility.
- Do not describe the current soft penalty as an unconditional safety constraint.

### C3. Pit input-length mismatches can silently change parameter slicing

**Locations**

- `src/pit.jl:52-57`
- `src/pit.jl:95-110`
- `src/pit.jl:219-221`
- `tesseracts/pit/tesseract_api.py:141-149`

The RHS slices concatenated vectors using knot-grid lengths. Output assembly uses the control-value-vector length.

The API does not require:

```text
len(u_knots) == len(u_tknots)
len(qmeat_knots) == len(q_tknots)
```

A short control vector can cause early heat-load values to be consumed as controls while later output calculations use a different partition.

**Required remediation**

Validate paired lengths before concatenation at both Python and Julia boundaries.

## High

### H1. “No wrap” optimization actually applies an endpoint wrap

**Locations:** `app/src/brisket_app/control.py:192-203,375-382,431-445`

With `opt_wrap=false`, the code pins `z_w=8`, then computes:

```math
t_{\mathrm{wrap}}=t_f\sigma(8).
```

For a 12-hour horizon:

- Final wrap blend: **0.506**
- Blend 30 minutes before finish: **0.049**

This is not `WRAP_NEVER`.

Reporting scripts often re-simulate the resulting schedule with `t_wrap=1e9`, so optimization and evaluation use different physics.

**Remediation:** use `WRAP_NEVER + 0*z_w` when wrap optimization is disabled.

### H2. Non-aligned horizons simulate the wrong duration

**Locations**

- `app/src/brisket_app/coupled.py:33-40`
- `app/src/brisket_app/mpc.py:98-112`

`np.arange(0, t_end + dt/2, dt)` can end before or after `t_end`.

The ODE stops at the generated endpoint, but MPC advances `CookState.t` by the requested duration.

**Impact**

Physical state, disturbance time, wrap time, and bookkeeping can diverge silently on arbitrary final segments.

**Remediation:** choose an interval count and use an exactly endpoint-aligned uniform `linspace`.

### H3. Purge does not stop at its stated equilibrium

**Location:** `src/meat.jl:271-289,396-399`

The sink is

```math
k_e\delta_e\log\left(1+e^{(W-W_{\mathrm{eq}})/\delta_e}\right).
```

At `W=W_eq`, it remains positive:

```math
k_e\delta_e\ln 2.
```

At defaults this equals **0.02795 kg/kg/h**.

Below equilibrium it remains positive, merely becoming smaller. There is no finite equilibrium or mathematical `W >= 0` invariant.

**Impact**

Purge, stall break, retained water, and calibrated `k_e` inherit a smoothing artifact that contradicts the documented one-sided relaxation.

### H4. Mass summaries use incorrect finite-volume weights

**Locations**

- `src/meat.jl:529-556`
- `src/meat.jl:670-689`
- `test/test_meat.jl:141-151`

Endpoint cells have half volume. The conservative mean is

```math
\bar W =
\frac{\frac12W_1+\sum_{i=2}^{n-1}W_i+\frac12W_n}{n-1}.
```

The code uses `sum(W)/n`. Fat uses the same arithmetic average.

The reported `drip_w` is defined as total water loss minus integrated evaporation. The closure test checks that same definition, so it cannot detect an error in PDE conservation or purge accounting.

**Impact**

`w_retained`, `juiciness`, `fat_rendered`, and drip outputs are mesh-dependent. In segmented solves, cumulative water loss is also combined with only the current segment's evaporative integral.

### H5. K=2 coupling is not converged

**Location:** `app/src/brisket_app/coupled.py:93-142`

Independent default-cook results:

- K=2 delta: 6.69 K
- K=3 delta: 0.55 K
- K=4 delta: 0.058 K

At the real-cook MAP, the two-pass delta is **3.73 K**, larger than the fitted **1.64 K** pit residual scale.

Calling the unrolled map “the model” makes its gradient internally exact. It does not make coupling truncation smaller than observation error.

### H6. Checked-in artifacts mix incompatible model eras

**Locations**

- `app/src/brisket_app/synth.py:19-25`
- `scripts/run_control.py:43-46`
- `scripts/run_timeplan.py:65-83`
- `scripts/run_nuts.py:131-160`

Current source expects:

- 7 meat parameters.
- 7 pit parameters.

Canonical stored synthetic truth contains:

- 5 meat parameters.
- 5 pit parameters.

`map_result.json` lacks `k_e` and `a_myosin`, so current `run_control.py` cannot assemble `theta`.

Stored real posterior files also omit those latents. `run_timeplan.py` indexes them. `run_robust_control.py` defaults to a missing `nuts_samples_real_7a.npz`.

Checkpoint resume compares CLI options but no model/schema hash.

`recovery_table()` can silently misalign old truth values against new parameter names, producing nonsensical truth labels.

**Required remediation**

Every artifact must contain:

- Schema/model version.
- Parameter names and units.
- Commit hash.
- Model/data hash.
- Dependency versions or lock hashes.
- Solver configuration.
- Seed.
- Source artifact hashes.

Incompatible loads and resumes must fail.

### H7. Existing posterior chains are not converged

**Location:** `data/calibration/multichain_diagnostics.json`

Pre-moisture multichain split-Rhat ranges from **1.13 to 4.63**.

Current seven-meat-parameter `7a` and `7b` checkpoints are marked incomplete and have no completed sample artifacts.

Recomputed older-chain diagnostics include:

- Synthetic ESS: 6.6–81.8, split-Rhat up to 1.139.
- Real single-chain ESS: 3.6–17.4, split-Rhat up to 2.567.
- Real three-chain maximum split-Rhat: 4.631.

The ridges are real, but the chains have not explored positions along them adequately.

The runner saves parameter values but discards:

- Divergences.
- Acceptance probabilities.
- Energy.
- Tree depth.
- BFMI inputs.

**Impact**

Credible intervals, posterior predictive bands, and posterior-based control results are not reliable posterior summaries for the current model.

### H8. The iid real-cook likelihood is contradicted by residual autocorrelation

**Locations**

- `app/src/brisket_app/calibrate.py:155-180`
- `app/src/brisket_app/cooklog.py:48-70`
- `data/cook_log/PROVENANCE.md:30-33`

At the checked-in final-physics MAP:

- Pit residual SD: 1.634 K
- Pit lag-1 correlation: **0.849**
- Core residual SD: 2.519 K
- Core lag-1 correlation: **0.968**

The fitted marginal scales match those SDs, but the residuals are smooth model discrepancy, not independent measurement noise.

An AR(1) heuristic reduces 111 samples to approximately:

- 9 independent pit observations.
- 2 independent core observations.

The likelihood also treats the digitized fan envelope as exact despite documented ±3–5 percentage-point uncertainty. Ambient is fixed at 300 K.

**Impact**

The posterior overcounts information, understates uncertainty, and forces physical parameters to absorb correlated model, controller, weather, and input error.

### H9. Evaporation is phenomenological rather than thermodynamic

**Locations**

- `src/meat.jl:95-116`
- `src/meat.jl:183-195`
- `src/thermophysics.jl:20-32`

Problems:

- Water activity does not reduce surface vapor pressure.
- Softplus gives a 69.3 Pa positive drive at zero nominal pressure deficit.
- Condensation cannot occur.
- Pit humidity is fixed.
- The meat's own evaporation does not humidify the pit.

This is a useful differentiable effective model, but it should not be described as equilibrium surface drying physics.

### H10. Absolute quality targets are unvalidated

**Locations**

- `src/meat.jl:28-78`
- `src/meat.jl:181-260`
- `src/meat.jl:322-336`
- `app/src/brisket_app/control.py:54-76`

The 17-hour duration and 3.5–3.8-hour wrap depend on:

- Uncalibrated bark constants.
- An arbitrary fivefold bark-rate scaling.
- A direct normalized-state-to-CIELAB mapping.
- Charcoal burn as a smoke proxy.
- Uncalibrated collagen kinetics and `C=0.90`.
- A placeholder fat contribution to juiciness.

KCBS category weights do not make incomparable proxy penalties equivalent to judging scores.

The results are hypothesis-generating, not quantitatively validated culinary recommendations.

### H11. Optimizers return an unchecked last iterate

**Locations**

- `app/src/brisket_app/control.py:120-130`
- `app/src/brisket_app/control.py:319-337`
- `app/src/brisket_app/control.py:417-449`
- `app/src/brisket_app/mpc.py:265-282`

Loss is evaluated before each Adam update, but the updated parameters are returned.

The best finite iterate is not retained. MPC discards optimizer history and does not verify:

- Finite objective.
- Objective improvement.
- Convergence.
- Safety residuals.
- Picard residual.
- Finite controls.

There is no incumbent or safe-controller fallback.

### H12. NUTS soft-fail can replace a valid-point adjoint failure with zero gradient

**Locations**

- `scripts/run_nuts.py:32-35`
- `tesseracts/meat/tesseract_api.py:392-409`
- `tesseracts/meat/tesseract_api.py:577-587`
- `tesseracts/pit/tesseract_api.py:259-270`
- `tesseracts/pit/tesseract_api.py:377-384`

All Julia exceptions are converted to `RuntimeError`. Soft-fail then converts broad failures to sentinel trajectories or zero gradients.

A forward solve can succeed with finite likelihood while the independent adjoint call fails. In that case, leapfrog integration receives a false zero gradient at a valid finite-potential point.

Shape, indexing, bridge, and implementation errors can also look like rejected proposals.

**Required remediation**

- Never suppress adjoint failures.
- Soft-fail only explicitly classified primal numerical-domain failures.
- Return an explicit invalid potential.
- Count failures and abort above a threshold.
- Never suppress schema or programming errors.

### H13. Initial meat temperature incorporates probe settling and early heating

**Locations**

- `app/src/brisket_app/cooklog.py:63-79`
- `app/src/brisket_app/calibrate.py:184-195`
- `data/cook_log/PROVENANCE.md:15-18`

Provenance marks the first 12 minutes as lid-open/probe-settling. The first usable ten-minute-grid core observation occurs around 20 minutes.

`tcore0_K` selects the first finite resampled meat value without applying the usability mask, then initializes the entire slab to that value at time zero.

**Impact**

Approximately 20 minutes of early heating and sensor settling are absorbed into initial state, thickness, conductivity, convection, and ignition parameters.

**Remediation**

Use a measured/fridge-temperature prior at meat-on, infer initial temperature as a nuisance parameter, or propagate from a justified initial field while starting the likelihood at the first usable observation.

## Medium

### M1. The chance layer is not a 95% certification and can omit wrap

**Locations**

- `app/src/brisket_app/timeplan.py:143-179`
- `scripts/run_timeplan.py:29-32,64-95`

Only 16 posterior draws are compared with a 0.95 empirical threshold. Even 16/16 does not establish a 95% lower confidence bound of 0.95.

`chance_of_done()` also replays the schedule without passing the record's optimized wrap time.

### M2. Dry-basis water/fat bookkeeping is inconsistent

**Locations**

- `src/thermophysics.jl:1-17`
- `src/meat.jl:7-8,61-67`

`RHO_DRY = rho/(1+W0)` treats dry matter as all non-water material. `F0` is subsequently added as another dry-basis component.

Those interpretations cannot both be correct without clarifying that the denominator excludes fat and recomputing `RHO_DRY`.

Applying the cited Choi–Okos mixture at 60 °C also does not reproduce the stated constants. The selected values remain plausible as effective beef properties, but the claimed derivation is inaccurate.

The real MAP `k_mult≈2.09` implies `k≈0.94 W/m/K`, suggesting compensation for geometry or missing physics.

### M3. Physical-domain validation and smoothness claims are incomplete

**Locations**

- `src/meat.jl:632-645`
- `src/pit.jl:59-68,208-224`
- `docs/model.md:13-16`

The solvers do not consistently reject:

- Negative or zero physical parameters.
- Infinity.
- Duplicate or nonuniform grids.
- Too-short grids.
- Invalid parameter combinations.

`max`, `clamp`, and `abs` create derivative kinks in fuel, ignition, smoke, water activity, and interpolation, despite documentation claiming all switches are smooth.

### M4. CI and documented gradient coverage are incomplete

**Locations**

- `.github/workflows/CI.yml:36-55`
- `app/tests/test_composed_grad.py:102-107`
- `docs/verification.md:48-56`

CI runs only `test_local_apply.py` on the Python side.

The composed local gradient gate checks meat parameters 1–5, not `k_e` and `a_myosin`.

There is no:

- Mid-cook state VJP test.
- Pit schema-failure test.
- Exact no-wrap test.
- Arbitrary-horizon test.
- Calibration or checkpoint test.
- End-to-end optimized MPC test.

### M5. The posterior-predictive plot omits observation noise

**Location:** `app/src/brisket_app/plots.py:121-159`

The plot propagates parameter samples through the deterministic simulator but does not sample:

- `sigma_pit`
- `sigma_core`
- Correlated model discrepancy

It is a latent mean-trajectory credible band, not a posterior predictive interval.

### M6. Documentation mixes current equations and historical evidence

**Locations**

- `docs/model.md:133-147`
- `README.md:35-44`
- `data/control/mpc_result.json`

Examples:

- Thermal boundary sign in `model.md` is opposite the code.
- README test counts are stale.
- README reports MPC 0.92 versus 0.68; the current artifact reports open-loop 0.991 versus MPC 0.957.
- README calls the 17-hour plan unwrapped, while current artifacts wrap near 3.5–3.8 hours.
- Current models have 12 synthetic and 14 real physical latents, while older prose reports 10 and 12.

### M7. Known omissions remain decision-relevant

The repository already acknowledges most of these:

- Shrinkage and changing conduction length.
- Temperature/composition-dependent properties.
- Pit humidity feedback and condensation.
- Airflow-dependent convection.
- Carryover/rest.
- 3-D geometry and point/flat differences.
- Coal-bed thermal lag.
- Smoke quality.
- Realistic actuator mapping.
- Internal-state estimation for deployable MPC.

These omissions can move late-cook timing, water-pan/wrap conclusions, edge behavior, fast control response, and real-world recommendations.

## Verified strengths

1. Main finite-volume heat and moisture equations are dimensionally coherent.
2. Boundary half-cell factors and code-level transport signs are correct.
3. Pit energy balance and fuel-rate units are correct.
4. FBDF and Tsit5 are appropriate for the modeled timescales.
5. Whole-horizon default-state derivatives have extensive finite-difference coverage.
6. Solver failures and truncated outputs normally fail loudly.
7. Cotangent normalization addresses a documented adjoint-solver failure mode.
8. Cross-thread JuliaCall serialization and owned array copies address real interoperability hazards.
9. Forward state handoff includes all current meat and pit state classes.
10. Fahrenheit/Kelvin and minute/second conversions are correct.
11. Parameter ordering agrees across Julia, Python, and Tesseract schemas.
12. NumPyro parameter transforms and likelihood normalization are correct.
13. `Manifest.toml` and `uv.lock` provide strong dependency locks.
14. Real-cook provenance is unusually explicit about source and digitization uncertainty.
15. Documentation candidly records many model limitations and previous AD failures.

## Priority sequence

### P0: correctness blockers

1. Make the meat VJP use identical handed-off state values.
2. Replace surface soft penalties with explicit safety feasibility reporting.
3. Validate pit paired-vector lengths.
4. Implement true no-wrap semantics.
5. Build exact endpoint-aligned grids.
6. Stop suppressing adjoint failures during NUTS.

### P1: re-establish evidence

1. Version all datasets, checkpoints, and result artifacts.
2. Regenerate synthetic data and MAP under the current parameter schema.
3. Complete multiple final-physics chains.
4. Require Rhat, ESS, divergence, BFMI, and tree-depth gates.
5. Add AR/state-space residual structure and input uncertainty.
6. Add K/exchange-grid convergence acceptance.
7. Rerun MPC only after a mid-cook finite-difference derivative passes.
8. Run the full application test suite in CI.

### P2: scientific validation

1. Independent water/fat accumulator states and finite-volume-weighted inventories.
2. Activity-based evaporation with condensation and humidity feedback.
3. Mass, separated-drip, and shrinkage experiments.
4. Surface humidity and temperature measurements.
5. Color-card bark photography.
6. Texture or shear-force calibration.
7. Open-loop smoker characterization burn.
8. Smoke/combustion-quality measurements.
9. Sensitivity studies for fixed wrap, bark, fat, geometry, and property constants.

## External references used

- [USDA FSIS Cooking Guideline, Revised Appendix A](https://www.fsis.usda.gov/guidelines/2021-0014)
- [Meat sorption-isotherm comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC9547243/)
- [Beef denaturation and water-holding capacity](https://doi.org/10.1093/ijfood/vvaf076)
- [Connective tissue and water holding](https://doi.org/10.3390/foods11233835)
- [Purlis and Salvadori browning model](https://doi.org/10.1016/j.foodres.2009.03.007)
- [Picard coupling stability analysis](https://arxiv.org/pdf/2301.00289)
- [Modernist Cuisine barbecue stall experiment](https://modernistcuisine.com/mc/barbecue-stall/)

## Final disposition

BrisketDynamics is credible as an exploratory differentiable simulator and as a demonstration of cross-language scientific AD.

The following claims should be withheld until the corresponding blockers are resolved:

- MPC improvement or disturbance rejection.
- Current-model posterior uncertainty.
- 95% completion guarantees.
- Food-safety certification.
- Quantitative optimal duration/wrap recommendations.
- Absolute bark, juiciness, or doneness quality predictions.

No source code or data was modified during the audit. This Markdown report is the only repository file added.
