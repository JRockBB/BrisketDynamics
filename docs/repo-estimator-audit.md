# BrisketDynamics Repository and Estimator Audit

Audited baseline: `master`, commit
`a811008a459f4bf1f0f48c60d6726b97df2f05c4` (2026-09-21).

This was a read-only audit of the Julia physics, Python calibration and
control stack, Julia/RxInfer estimator, tests, CI, documentation, and
published validation artifacts.

## Executive verdict

BrisketDynamics is a sophisticated research prototype with unusually strong
differentiability, physics-unit testing, and candid documentation. The
RxInfer estimator is real: measurement updates are performed through RxInfer
factor graphs rather than through a substitute EKF.

The present evidence does not support claims that the current sequential
estimator is statistically consistent as a standalone loop:

- Current consistency artifacts predate the estimator/physics fingerprint
  and fail several of their own NIS/NEES criteria.
- The estimator suite currently fails and is absent from CI.
- Validation is principally model-matched synthetic validation, so model-form
  error is zero by construction.
- The calibrated water budget is physically implausible.

The appropriate status is: credible modeling and standalone inference
research platform whose current estimator validation remains incomplete.

## What the RxInfer estimator actually implements

The authoritative estimator paths do use RxInfer:

- `cook_graph` and `cook_graph_lm` contain one multivariate latent block, a
  Gaussian/local-quadratic prior, one unit-variance normal factor per
  whitened residual, and an LM factor:
  `estimator/src/BrisketEstimator.jl:390-427`,
  `estimator/src/iterated.jl:109-120`.
- `solve_channels_graph` invokes `infer` for batch and sequential updates:
  `estimator/src/iterated.jl:132-161`.
- The sequential filter calls that path through `solve_map_channels`:
  `estimator/src/sequential.jl:743-813`.
- AR(1) discrepancy estimation uses a separate RxInfer graph:
  `estimator/src/noise.jl:70-124`.
- Volatility uses RxInfer's GCV node.

The complete algorithm is not one joint factor graph. Outside RxInfer are:

- ForwardDiff physics solves and Jacobians.
- Nonlinear relinearization, line search, move acceptance, and damping,
  implemented in `after_iteration` callbacks:
  `estimator/src/iterated.jl:40-100`.
- Sigma-point construction and state propagation.
- Process-noise injection, event inflation, fault handling, and volatility
  adaptation.
- Alternation between physics fitting and AR-noise fitting:
  `estimator/src/stream.jl:68-100`.
- The older `solve_map` outer relinearization loop:
  `estimator/src/BrisketEstimator.jl:430-479`.

Therefore, "the loop lives in the graph" is imprecise. A more exact
description is: RxInfer solves every affine Gaussian update, while a Julia
callback hosted by one `infer` call controls the nonlinear Gauss-Newton loop.

The Python MPC is intentionally outside the estimator loop at this stage.
Its lack of RxInfer integration is therefore not an estimator defect or a
current implementation requirement. The estimator should first converge,
propagate uncertainty, and pass standalone consistency tests on its own.

## Prioritized findings

Resolved findings and their verification evidence are recorded separately in
`docs/estimator-remediation-report.md`.

### 2. Current consistency and coverage claims are not established

**High -- unsupported current claim/documentation drift -- confidence: high**

The stored state-noise fingerprint is `ba928b214b27`
(`data/calibration/state_noise.json:51-55`); the audited tree computes
`3a846a18649e`. `load_state_noise` warns but continues using stale rates:
`estimator/src/sequential.jl:210-231`.

The published fresh-round summary fails several nominal consistency checks:

- NIS/dof mean: `0.520`.
- Only `43.5%` of slices inside the nominal 95% NIS band.
- Latent NEES/dof: `2.243`, above its `[0.782, 1.244]` band.
- Core innovation SD: `0.652`, with lag-one correlation `0.204` outside its
  stated band.
- Several state groups are substantially too wide or too narrow.

Evidence: `data/mock/filter_consistency_summary.json:41-190`.

Low NIS combined with high latent NEES means uncertainty is misallocated:
measurement predictions are broadly inflated while latent covariance remains
too narrow. It is not evidence of a calibrated belief.

Further limitations:

- The plant on this bench is the same physics as the filter, as acknowledged
  in `docs/results.md:360-390`.
- The current normal-score parameterization postdates the tracked artifacts.
- State-group acceptance bands use a heuristic "one independent cook-hour"
  effective sample size:
  `estimator/python/plot_filter_consistency.py:93-102`.
- Synthetic self-consistency cannot test geometry, moisture-model,
  sensor-placement, or calibration discrepancy.

**Remedy:** fail closed on fingerprint mismatch; fix the defects in this
report; rerun a preregistered tuning/fresh-seed protocol on the exact current
tree; report all seeds and failures; then repeat on held-out real cooks.
Use the term "model-matched synthetic consistency" until real validation
exists.

### 5. The calibrated water budget is physically implausible

**Critical -- model-adequacy failure -- confidence: high**

The calibrated real cook reportedly retains about 4% of initial water while
a real flat should retain roughly 50--60%:
`docs/review.md:88-102`, `docs/roadmap.md:357-366`.

Numerical PDE bookkeeping may close while the inferred physical budget is
still wrong. This invalidates physical interpretation of predicted
juiciness, wrap timing, and spritz value even after correcting the scale
bookkeeping.

**Remedy:** make before/during/after mass, purge, tallow, thickness, and
humidity measurements the next calibration experiment. Do not add a more
complex moisture PDE until the present model is tested against those
observations.

### 10. Process noise, discrepancy, and sensor noise remain confounded

**High -- theory and identifiability risk -- confidence: high**

Real calibration estimates 15 physics parameters plus four probe-noise
parameters from one digitized cook and two temperature trajectories:
`app/src/brisket_app/calibrate.py:43-89`, `292-315`.

Online, the same innovations drive:

- Parameter/state process noise.
- AR discrepancy re-identification.
- Volatility levels.
- Sensor fault decisions.

The AR graph is fit separately and its posterior means are plugged into the
physics graph. Its uncertainty is not propagated through the sequential
belief. The AR coefficient keeps the node's unbounded Gaussian (the node's
rules are conjugate in it); its reported moments are now the Gaussian's
truncation to the stationary interval rather than a post-hoc point clamp
(remediation report, issue 10), which bounds the summary, not the node.

Simultaneous identification of process noise, measurement noise, physical
parameters, and discrepancy is not generally identifiable without structural
assumptions and sufficient observability [3]. Adding flexible discrepancy
also risks confounding physical calibration unless its prior and structure
are scientifically constrained [1,2].

**Remedy:** (the bounded summary of the AR correlation is done; a bounded
parameterization inside the node would need a custom AR rule, since a delta
node in front of it does not bound the posterior the node uses under
mean-field) separate instrument characterization from state/process-noise
estimation; use
replicated cooks and independently measured mass, humidity, geometry, and
event data; report sensitivity/Fisher rank, profile posteriors, and posterior
correlations.

### 14. Deterministic soft safety is insufficient for live control

**High operational risk -- documented limitation -- confidence: high**

The code correctly warns that it is not USDA certification:
`app/src/brisket_app/control.py:201-206`. Nevertheless:

- MPC safety is a tradeable soft penalty:
  `control.py:121-124`, `523-535`.
- Appendix A humidity requirements are not modeled.
- Constraints are evaluated on one deterministic state/parameter trajectory.
- There is no uncertainty margin or worst-case lethality constraint.
- Model and state uncertainty are absent from planning.

FSIS Revised Appendix A requires validated critical operating parameters, not
merely a modeled endpoint integral [13].

**Remedy:** retain the disclaimer, but require hard feasibility checks outside
the optimizer; use conservative uncertainty margins; measure or model the
relevant humidity condition; and provide fail-safe fallback schedules.

### 16. Secondary confirmed issues

- Julia and app objective implementations disagree on the tracked MPC case:
  `0.6802` versus `0.6461`, principally appearance:
  `data/mock/objective_crosscheck.json:67-110`. The formula is identical on
  both sides (a term-by-term parity fixture now gates it; see the remediation
  report); the residual disagreement is the two engines' bark, a roadmap item.
- The supported 2-D model omits several 1-D dynamics, including dynamic fat
  rendering, smoke/ring, fuel additions, and lid lifts, by design:
  `src/meats/slab2d.jl:1-25`. It is a useful shadow model, not independent
  verification of the full planner.

The spritz mask past the wrap, the water pan's overshoot below zero, the
cook-log pit gaps marked usable, and the stale module header are resolved
and recorded in `docs/estimator-remediation-report.md`.

## Physical and model-theory gaps

### Moisture transport and shrinkage

The current model combines Fickian effective diffusion, empirical water
activity, and local denaturation-driven expulsion. This is computationally
effective, but long-time meat moisture transport is also pressure-driven
through a shrinking protein matrix. Flory-Rehner/Darcy models explicitly
capture that coupling [9].

This matters because effective diffusion, evaporation, expulsion, geometry,
and discrepancy can reproduce similar temperature stalls while predicting
very different retained mass and juiciness.

The immediate response should be mass-balance experiments, not automatic
replacement of the model.

### Tenderness and collagen

One first-order Arrhenius collagen-conversion state is not a validated
tenderness model. Meat texture reflects competing myosin, collagen, and
actin transformations over different temperature ranges [10], and observed
tenderness can first improve and then worsen with time and temperature [11].

The current `C approximately 0.90` target is anchored by one pitmaster
judgment, not mechanical or sensory data. It should remain labeled a
doneness proxy.

### Water activity and salt

The Oswin relation is empirical and food/range-specific. Sorption models have
restricted applicability and should not be extrapolated across temperature,
salt, composition, and high-water-activity regimes without data [12]. The
current salt correction is a simple multiplicative factor and does not
establish meat-specific sorption behavior.

### Geometry and cold spot

A homogeneous symmetric slab cannot represent flat/point thickness
variation, fat cap, local probe contact, anisotropy, or event-dependent
cold-spot migration. The 2-D cylinder adds useful structure but is not
geometry-matched to a full packer and is not the production inference model.

Use the 2-D model for discrepancy bounds and sensor-placement studies until
imaging or multi-probe data support calibration.

### Model discrepancy

The present AR residual accounts for serial output mismatch but does not
distinguish sensor bias, probe placement, missing physics, and cook-to-cook
effects. Mechanistic calibration without explicit discrepancy can produce
biased, overconfident physical parameters [2]; an unrestricted discrepancy
can instead make those parameters non-identifiable. A restricted discrepancy
informed by conservation laws and replicated experiments is preferable.

## Verification results

Directly verified on the audited tree:

- Root Julia suite: **368/368 passed** in approximately 15.5 minutes.
- RxInfer estimator suite: **14 of 15 test files passed**;
  `test_smoother.jl` errored with an unstable monolith solve. Total suite
  time was approximately 1 hour 57 minutes.
- Python app suite: **32 passed, 2 failed, 42 setup errors**. All
  failures/errors were blocked by Julia/Python integration initialization.
  The first error showed missing `JSON` during `PythonCall`/`CondaPkg`
  precompilation; subsequent attempts resolved `sys.dll` relative to the
  repository instead of the installed Julia runtime. This is a local
  reproducibility failure, not evidence that the blocked functional tests
  are wrong.
- Physics fingerprint: current `3a846a18649e`; stored state-noise fingerprint
  `ba928b214b27`.
- Git status after verification contained only pre-existing/untracked
  artifacts; no tracked file changed during the audit.

CI runs root Julia tests and the Python app on Ubuntu, but has no estimator
job: `.github/workflows/CI.yml:13-58`. Consequently, the reproduced estimator
regression can merge unnoticed.

## Ranked next steps

### Immediate work requiring no new data

1. Add the complete estimator suite to CI and make fingerprint mismatch fatal
   for validation scripts.
2. Bound AR correlation.
3. Rerun the full held-out score-coordinate consistency bench; do not reuse
    current state-noise rates.
4. Stabilize and document the PythonCall environment on Windows.
5. Add term-by-term Julia/app objective parity as a CI gate.

### Designed experiments

1. **Instrumented mass/moisture cook:** calibrated scale, drip/tallow
   collection, wet-bulb or dew-point measurement, surface/core probes,
   thickness measurements, and precise event logs.
2. **Open-loop pit experiment:** excite damper/fan levels while measuring
   fuel mass, ambient conditions, pit temperature, and humidity to separate
   wall loss, thermal mass, burn rate, leak, and ignition.
3. **Replicated geometry experiment:** multiple briskets and probe depths to
   identify between-cook variability and probe-placement effects.
4. **Texture experiment:** Warner-Bratzler or comparable mechanical
   measurements over multiple time-temperature histories, supplemented by
   blinded sensory judgments.
5. **Sensor characterization:** saturation, frozen-probe, bias,
   replacement-probe, and lid-gap runs independent of physics calibration.

### Longer-term research

1. Compare the existing moisture model against a reduced
   Flory-Rehner/Darcy alternative using mass and thickness data.
2. If estimator/controller integration later becomes a project goal,
   separately evaluate robust output-feedback MPC with uncertainty bounds.
3. Use a multi-fidelity 1-D/2-D discrepancy model rather than treating 2-D
   as verification.
4. Introduce mixture/particle diagnostics when event uncertainty or
   bounded-state posteriors are materially non-Gaussian.
5. Build hierarchical calibration across cooks, cuts, pits, and sensor
   instances.

## References

1. Kennedy and O'Hagan, "Bayesian calibration of computer models,"
   [DOI 10.1111/1467-9868.00294](https://doi.org/10.1111/1467-9868.00294).
2. Brynjarsdottir and O'Hagan, "Learning about physical parameters: the
   importance of model discrepancy,"
   [DOI 10.1088/0266-5611/30/11/114007](https://doi.org/10.1088/0266-5611/30/11/114007).
3. Mehra, "On the identification of variances and adaptive Kalman filtering,"
   [DOI 10.1109/TAC.1970.1099422](https://doi.org/10.1109/TAC.1970.1099422).
4. Piche, "Online tests of Kalman filter consistency,"
   [DOI 10.1002/acs.2571](https://doi.org/10.1002/acs.2571).
5. Gustafsson and Hendeby, "Some Relations Between Extended and Unscented
   Kalman Filters,"
   [DOI 10.1109/TSP.2011.2172431](https://doi.org/10.1109/TSP.2011.2172431).
6. Simon, "Kalman filtering with state constraints,"
   [DOI 10.1049/iet-cta.2009.0032](https://doi.org/10.1049/iet-cta.2009.0032).
7. Allik et al., "The Tobit Kalman Filter,"
   [DOI 10.1109/TCST.2015.2432155](https://doi.org/10.1109/TCST.2015.2432155).
8. Koyama et al., "Approximate Methods for State-Space Models,"
   [DOI 10.1198/jasa.2009.tm08326](https://doi.org/10.1198/jasa.2009.tm08326).
9. van der Sman, "Moisture transport during cooking of meat,"
   [DOI 10.1016/j.meatsci.2007.02.014](https://doi.org/10.1016/j.meatsci.2007.02.014).
10. Tornberg, "Effects of heat on meat proteins,"
    [DOI 10.1016/j.meatsci.2004.11.021](https://doi.org/10.1016/j.meatsci.2004.11.021).
11. Bertola et al., "Heat treatment effect on texture changes and thermal
    denaturation of proteins in beef muscle,"
    [DOI 10.1111/j.1745-4549.1994.tb00240.x](https://doi.org/10.1111/j.1745-4549.1994.tb00240.x).
12. Boquet, Chirife, and Iglesias, "Equations for fitting water sorption
    isotherms of foods,"
    [DOI 10.1111/j.1365-2621.1978.tb00792.x](https://doi.org/10.1111/j.1365-2621.1978.tb00792.x).
13. USDA FSIS,
    [Cooking Guideline for Meat and Poultry Products -- Revised Appendix A](https://www.fsis.usda.gov/guidelines/2021-0014).
