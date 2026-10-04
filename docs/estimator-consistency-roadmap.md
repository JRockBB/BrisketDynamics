# Estimator Consistency Roadmap

Establish statistical consistency on the full-order single-meat factor-graph backend (`SystemModel`, `online.jl`)
first, then earn ROM and multi-meat calibration separately. The factor-graph backend is the sole implementation
target. RxInfer, the sequential filter (`sequential.jl`) and the Python bench (`bench_state_noise.py`, `als_fit.py`,
`bench_manifest.py`) are frozen comparison oracles: nothing new is built in or on them. Every new bench, calibration
step, manifest and statistic is Julia. The Python plot scripts remain figure renderers of the record schema only.

## Review of the plan as first written, and what changed (2026-09-26)

1. **The Python-application migration was a phantom.** The Python pilot (`app/src/brisket_app/pilot.py`) runs its
   own pit-snapshot and ambient Kalman estimator and states that "the RxInfer estimator remains separate"; it never
   called the RxInfer filter, so there is nothing to migrate "from RxInfer". The estimator's application path is the
   Julia pilot loop (`estimator/scripts/mock_closed_loop_system.jl`). Removed. Real-pit shadow mode is the hardware
   item C1 of `roadmap.md`, not this plan.
2. **The bench for the new backend does not go through the Python driver.** A backend selector in
   `bench_state_noise.py` would have made the new estimator's calibration logic Python. The bench is a Julia worker
   (`coverage_system.jl`), a Julia round driver (`bench_system.jl`) with the manifest, the pooled statistics, the
   autocovariance fit and the rates file, all in Julia. The record schema is the existing one so the plot scripts
   render it unchanged.
3. **"Matched, static" is defined.** The RxInfer bench told its filter the calibration's noise prior (a core σ of
   1.65 K against the mock's 1.2 K pure AR(1)), and the core's lag-zero innovation gap that no state rate could
   close was that configuration's. Matched means the sensor models carry the mock instruments' own AR sd and
   correlation time (`RealCookNoise`) with the instrument floors of the pilot loop; static means the drift draw is
   zero with the same seeds and truth draws. The drifting plant is the second configuration, not the first.
4. **The NIS was missing from the backend's record.** The smoother recorded each reading's innovation but not the
   predicted innovation covariance. NIS/dof, the standardized innovations (whiteness) and the identification's
   sufficient statistics all need `S = H P⁻ Hᵀ + R` at the predicted belief of the newest node before the arrival's
   observation factors act. That is a change to `online.jl` (one covariance of the graph at the predicted point,
   whose dynamics Jacobian the first iteration needs anyway), not to a script.
5. **Joint noise identification has a mechanism.** "Fit process noise together with sensor white/OU noise where
   identifiable" is realized in the innovation-autocovariance regression (`als.jl`): a sensor's OU discrepancy is a
   state of the graph whose innovation variance is a process-noise group on its own coordinate, and a sensor's
   white variance enters the same-update products and, through the gain, the state error (an R-group). Both are
   linear in the autocovariances. The same admission rules apply (a group enters when its own column pins it or the
   unbounded fit moves it; capped; held when the fit cannot tell its sign). No channel floor is imposed on the
   bench: the white sd starts at the instrument's and may move within the cap.
6. **Feasibility is a measured target.** Before any optimization, one profile of the backend on the real-cook mock
   gives the per-arrival wall, the iteration and Jacobian counts, and the cost of one primal solve and one Jacobian
   at several chunk sizes and tolerances. Only what the profile names is changed, with a regression gate that the
   beliefs are unchanged within the solver's tolerance. The target is stated in the report with the profile.
7. **Latent drift has a design.** The window's latents are static (decision 8 of the multi-meat report). The drifting
   bench walks the leak, the water-loss and the convective coefficients hourly and jumps the charge's sites at the
   add. The design is the smoother's (`smoother.jl`) carried into the window: the latent vector is split into a
   static block (one variable) and a moving block with one copy per window node; a random-walk factor ties
   consecutive copies (the drift rate over the slice's hours, the jump's variance added on the slice crossing the
   add); a slice's dynamics factor reads the copy at its start node, an observation the copy at its node;
   marginalizing a node removes its copy. Drift is never folded into the state's process noise.
8. **The rates file of the new backend is its own.** `data/calibration/state_noise_system.json`, fingerprinted by
   `backend_fingerprint()` (the physics and the backend's files), read strictly by every driver; the RxInfer filter's
   `state_noise.json` is the filter's and stays untouched.
9. **The ROM item names its plant.** A ROM bench's plant is the full-order piece (`Brick3D`), the estimator the slab
   with the artifact attached. The current artifact (one archetype, three draws, negative held-out skill on the core
   and the mass) cannot be benched; retraining on a design that covers the bench's controls and events is the item.
10. **The multi-meat item names its plant.** The two-piece bench's plant is a two-piece `CookSystem` solved by
    `system_solve`, its readings the compiled observables under the instruments' AR noise, the permutation check the
    same seeds with the pieces in the other order: `test_multi.jl`'s setting over whole cooks with events.
11. **No long run without a go.** The bench's rounds take hours. The build is smoke-tested on one short cook; the
    tune and fresh rounds are launched only when the user says so, and nothing is quoted from a smoke or a tune round.

## Implementation TODOs

- [x] **Build the new-backend bench (Julia):** a seed-safe NIS/NEES/coverage worker for the factor-graph backend on
      the existing plants, seeds and record schema, with the innovation-autocovariance accumulator fed from the
      graph; a Julia round driver with the manifest, the pooled statistics and their bands. (Built and smoke-tested;
      the tune round ran, the fresh round was stopped: report §4b.)
- [x] **Make the bench feasible:** profile the backend, optimize only measured bottlenecks, gate the beliefs unchanged.
      (Lag 0 at 5.8 s per arrival on four threads; report §4.)
- [ ] **Fix what the tune round showed before any relaunch** (report §4c): the ignited fraction's saturation is now
      representable (its own coordinate floor, decision 11) and a solve stalled at the ODE floor with a joint step under
      a hundredth of a posterior sd is no longer thrown away (decision 12). Done on one cook's first hours, the factors,
      bench-system and plan-system shards; the field noise's shape amplifier was answered with smooth modes (decision 14) and the answer withdrawn
      after two cooks (the probe's depth drifted to 6.5 sd; report §4c item 9): the bulk alone ships, the shape-error
      structure waits for the ROM bench's mismatched plant; still to do: the long suites (`test_online.jl`, `test_multi.jl`, `test_offline.jl`, `test_rom.jl`) on the
      fixed tree, and the two-minima path dependence at single arrivals. No round without a go.
- [ ] **Calibrate the FOM estimator:** the autocovariance fit in Julia over the state's rates and the sensors' OU and
      white variances; the rates file with the backend fingerprint; tune and fresh rounds on disjoint seeds; the
      held-out gates. (The fit and the file exist; the rounds on the fixed tree have not run; the existing file is
      stale and earned nothing.)
- [ ] **Model latent drift:** the split latent block with per-node moving copies and random-walk factors; validated
      on drifting-plant cooks against the truth in force; drift never folded into state noise.
- [ ] **Recalibrate the ROM:** retrain on a covering design, cross-validate the error model, require positive
      held-out skill on the core and the mass, then the same bench with the artifact attached.
- [ ] **Validate multi-meat:** the two-piece bench with distinct events, heterogeneous sensors, missing channels and
      ordering permutations; plan stability and decision quality across complete cooks.

## 1. Build the new-backend bench

- `estimator/scripts/coverage_system.jl` beside `coverage_filter.jl`: the same plants (the monolith in segments,
  drifting or static by a switch), the same truth draws per seed, the same real-cook instruments and masks, the
  same edges and record schema (`temps` with `nis`, `dof`, `innov` per channel, `nees_z`, `nees_z16`, `nees_T`,
  `nees_g` per state group; `batches` with `err_sd_graphfilter`; `als` per cook), driving
  `SystemModel`/`online_init`/`assimilate!` at the pilot's lag.
- `online.jl` records at every arrival the predicted innovation covariance, NIS with its dof, the standardized
  innovations per sensor, the belief's coordinate state and covariance, the solve's iterations, gradient,
  convergence and commitment; and feeds an `ALSAccumulator` when the bench asks for it.
- `estimator/scripts/bench_system.jl`: the round driver. Workers as Julia processes, one per seed block; the round's
  manifest (`data/mock/bench_manifest_system_<round>.json`, the same fields as the Python manifests) validated
  before pooling; the pooled statistics and their finite-sample bands (the chi-squared band of one sample per
  cook-hour, the cook-cluster t interval) printed and written; the figures rendered by the existing plot scripts
  from the manifest.
- Start with matched, static, one-piece cooks. Do not use state noise to hide missing latent-drift dynamics.

## 2. Make the bench computationally feasible

- Profile one cook's first arrivals (per-arrival wall, iterations, line-search trials, dynamics Jacobians and
  primal solves; one primal propagate and one Jacobian alone at the smoother's tolerances and the filter's, at
  chunk sizes 12, 24 and 48; one window covariance). Optimize only what the profile names, in `factors.jl` and
  `online.jl`, and gate the beliefs unchanged within the solver's tolerance on the arrivals test.
- Require bounded memory, every update converged and committed (no rollback), and the per-arrival wall below the
  two-minute probe cadence by a margin the report states, before launching multi-cook rounds.

## 3. Earn FOM uncertainty calibration

- `als.jl` gains R-groups (a sensor's white variance) beside the process-noise groups; the OU discrepancies enter as
  process-noise groups on their coordinates. The fit (`estimator/src/bench_fit.jl`): the feasible weighting per
  channel pair and kind, the admission and hold rules, the cap, the leave-one-cook-out jackknife, the lag table;
  the same rules as the frozen Python fit, in Julia.
- The tune round on its seeds with the rates as they stand, one fit, the fresh round on seeds never seen with the
  identified rates and sensor noise; the fresh round alone is quoted. The rates file carries the backend
  fingerprint, the commit and both rounds' numbers.
- Accept only if, on the fresh round: NIS/dof, the latents' NEES/16, the temperatures' NEES/3 and every observed
  state group's NEES/dof sit inside their bands; the standardized innovations' mean and lag-one autocorrelation sit
  inside ±2/√N per channel; every update converged and committed; the coverage table shows no site's mean beyond
  two standard errors and spreads within the band. Anything out of band is reported, not tuned away.
- Add the latent drift factors before evaluating drifting-plant cooks; then repeat the calibration rather than
  folding drift into state noise.

## 4. Rebuild ROM evidence separately

- The scripts: `estimator/scripts/train_rom_bench.jl` (the artifact on prior draws over the mock's schedule in two
  regimes, held-out draws in those and a third, raw against corrected, the gate printed) and
  `estimator/scripts/coverage_system_rom.jl` (the bench worker under the artifact, `bench_system.jl --worker rom`).
- Done 2026-10-01: the covering design (a Latin hypercube over every latent's 0.1 %–99.9 % prior quantiles, so the
  runtime envelope is the prior's bulk), the cross-validated error model (a ridge regression of the reduction error
  on the latents per time point, its weight by leave-one-design-draw-out cross-validation, its spread the
  cross-validated residual's; read at the current latents by the observation factors, the dynamics correction
  and the planner), positive held-out skill on every output in both trained regimes and the untrained one
  (`data/calibration/rom_halfbrick_flat.json`; `docs/results.md`).
- Still to do: earn ROM-specific `StateNoise` on the same bench with the artifact attached (the plant the
  full-order piece, the estimator the slab with the artifact); the chamber vapor's spread is narrow (85–88 %
  within 2 sd against 95 %).
- Require corrected held-out RMSE to beat the raw ROM error on the safety and quality outputs, calibrated spread,
  and artifact-backed assimilation and planning that pass the same gates as section 3.

## 5. Multi-meat

- The worker: `estimator/scripts/coverage_system_multi.jl` (`bench_system.jl --worker multi`, with `--order 1` for
  the permutation and `--drop <sensor>` for a missing channel).
- Repeat the consistency bench on two-piece cooks (a two-piece `CookSystem` plant) with distinct events,
  heterogeneous sensors, missing channels and ordering permutations.
- Validate plan stability and decision quality across complete cooks on the Julia pilot loop.
- Then update `roadmap.md` and `multi-meat-inference-implementation-report.md` with earned evidence and remaining
  limits.

## Implementation handoff

- Work through the TODOs in order and leave the working tree uncommitted for review.
- Do not claim consistency from deterministic tests, smoke runs or tune-round results; acceptance evidence must come
  from fresh held-out seeds, and only when the rounds have been run with a go.
- Record every changed file, design decision, exact command and outcome, generated artifact and fingerprint, failed
  experiment, deferred item and unresolved risk in `docs/estimator-consistency-implementation-report.md`.
- End the report with a concise review order and explicitly distinguish passing plumbing from statistically earned
  calibration.
