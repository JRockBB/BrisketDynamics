# Estimator consistency after merging the from-lighting model

Work started 2026-10-04 on `estimator-consistency` at `96553a3`, merging
`Photo-segementation` at `7aace2c`. The user authorized estimator changes
and overnight validation. Model, observation-graph and numerical solver defects
were corrected. The follow-up solver repair is committed as `13c3d4e`, with
the posterior-accuracy policy refined in `25e7c29`. That revision passed
446 targeted assertions but failed the causal runtime target. A subsequent
trial-state closure correction is committed as `d3bedd9`. The curvature-aware
online search is committed as `f5854ac`, with 493 targeted assertions passing
and both frozen slow arrivals meeting the unchanged convergence threshold.
Its separate causal replay is recorded under `stream_lighting_curvature_41002`.
That replay failed the runtime target and was stopped; the passing numerical
regressions did not establish acceptable streaming latency.
The resulting bound-handling correction is committed as `2294f39`. It passes
497 targeted assertions and four frozen slow-arrival controls. Its independent
causal verification is saved separately under `stream_lighting_bounds_41002`.
The merged estimator has **not established statistical
consistency**: full-cook held-out validation has not been completed, and tuning
runs are not held-out evidence.

**Verification 2026-10-05: the bound fix holds through loading.** After the
arrival-evidence fixes (`07fa00b`, backend `95f5c04f2599`), the causal replay
`data/mock/stream_lighting_ws1_41002.json` completed its full 3 h horizon:
470 updates, 657 readings, zero rejected, zero dropped, total wall 6624 s.
Update wall time: median 9.8 s, p95 39.6 s, max 109.6 s (in the loading
window, which previously stalled at 165–485 s and ended every replay).
Reading-to-estimate latency: median 10.9 s, max 178 s. NIS/dof 0.89, latent
NEES/16 0.93, white innovations. Under the corrected belief-time scoring the
meat-temperature bias REMAINS: centre/surface standardized error sits at +1.9
before loading (the prior-drawn initial state, no meat sensors to correct it),
snaps to zero when the core probe arrives, then climbs to +2.8/+3.2 by 3 h
(temperature NEES/3 2.93; state groups T 4.42, W 4.39). The latent trace shows
the structure: `k_m` leaves the band at loading and holds −2.0 to −2.6 for the
rest; `core_depth_fraction` steps to +1.15 near 2.2 h and freezes;
`a_myosin` compensates to −1.5. The scoring fix did not explain the bias —
this is the fixed-lag lock and model-structure question the structure
diagnosis (truth residuals, per-latent sensitivities, `prior_relin`/`lag_max`)
must answer before any calibration round. Figures:
`figures/estimator/stream_lighting_ws1_41002*.png`.

**Does the latent inconsistency cost the planner? 2026-10-13: mostly not.**
`estimator/scripts/predictive_check.jl` was run on four banked 18 h cooks (seeds 41002,
43001 ×2 arms, 43002; 60 checkpoints from 2 to 16 h). At each checkpoint the core
temperature was rolled forward from the plant's true state under the estimator's latents,
the prior's, and the true latents. The true-latent rollouts reproduce the plant to
0.00 K, so the errors below come from the latents alone.

| | 1 h ahead | 2 h ahead | 4 h ahead | 6 h ahead |
|---|---|---|---|---|
| Estimator, mean abs error (bias) | 0.48 K (+0.13) | 0.61 K (+0.30) | 0.66 K (+0.42) | 0.66 K (+0.47) |
| Prior, mean abs error (bias) | 1.91 K | 3.64 K | 5.08 K | 5.63 K |

Median error in the time the core first reaches each temperature:

| | 65 °C | 75 °C | 85 °C |
|---|---|---|---|
| Estimator | 3.7 min | 6.4 min | 10.7 min |
| Prior | 34.1 min | 18.4 min | 27.1 min |

Seed 43001, where the ignited-group NEES reached ~240, forecasts within 1.2 K at 4 h.
Once the fire saturates, the ignition latents stop shaping the trajectory, which is also
why they cannot be learned. Their miscalibration is therefore mainly a reporting problem,
not a planning one.

Not yet established:
- **State-estimation error** is excluded, because the rollouts start from the truth.
- **Model error** is absent: the plant is matched.
- **The predictive bands are too wide.** They cover 100% at 95%, likely because the records
  keep only latent SDs, so the draws ignore correlations.
- **The 65 °C crossing is fragile.** The stall plateaus at that temperature, and the
  estimator's rollout missed the crossing in 8 of 25 cases.

**Pit-only quadrature during lighting, 2026-10-13 (stopped; infrastructure kept, `b43b451`).**
Since the meat is not on during lighting, the ignition quadrature was moved onto a pit-only
system (6 states, which matches the full system's pit to 3.8e-7 K before loading), with an
exact hand-off to the full cook at t_load − 600 s. At that size the cost is fine: a full
phased run took about 15 min. On model-generated lighting data the method still does not
commit:

- With a0 and k_spread integrated, the other five fire latents go non-Gaussian (shape
  defect 0.6–0.9 against the 0.25 guard).
- Refining toward resolution puts rule nodes 4–10 chart-sd out. Their components fail to
  converge, and the layer rejects the whole update for any failed component.

Two further attempts made it worse, and both were reverted:
- accepting failed tail components whose weight share was below 1e-6;
- growing the axis block from the shape defect.

The fire posterior during lighting is non-Gaussian in more directions than feasible tensor
rules can resolve. Kept: the zero-piece `CookSystem`, `pit_only_model`, `handoff_smoother`
(exact posterior carry-over, tested) and the `PhasedSmoother` hand-off mechanism. The
estimator steps back to the calibration round on the locked Gaussian configuration, and
will measure how much the fire's non-Gaussianity moves the planner's predictions before
any further method work.

**Quadrature escalation probes, 2026-10-11 (seed 43001, a0 + k_spread, causal, nx 40).**
The conditional backend now runs under slice attachment (`b414ab7`), and the adaptive one
learns then freezes (`ffae168`). Two shipped-scale probes:

- *Fixed chart* (`ConditionalSmoother`, rules 3/5): the prior-scaled nodes lose resolution
  within ~90 s of lighting (coarse-vs-fine mean error 0.24 vs tolerance 0.05) as the
  ignition posterior contracts; every later update was rejected.
- *Adaptive chart* (`DeterministicSmoother`, learn-then-freeze): on the first 5-reading
  batch the chart re-centred twice and refined to the order-9 rule, giving 115 component
  smoothers at full history. Updates then took 19 min (5 readings) and 52 min (48
  readings); the second was rejected because one tail component (k_spread 4.6 chart-sd
  out) missed its convergence budget, and the layer drops no failed component. The
  process held 11.7 GB private. Stopped after 2 h with the belief still at t = 0.

The mechanism is sound (the chart follows the contracting posterior), but at this
physics resolution and on this machine the faithful full-history quadrature is about two
orders of magnitude from live pace.

**LR2 mitigation bench, stopped 2026-10-11 (runs 1-3 of 6 banked).** Causal
18 h replays on fresh seeds, `locked` (lag_max 3, prior_relin Inf, no
sensitivity refresh) against `refresh` (lag_max 6, prior_relin 0.2,
slice_g_refresh). Run 4 (refresh/43002) was stopped by the user at 5.68 h
simulated after 9 h wall: `rebuild_prior!` re-linearizes the whole departed
history, and its backlog grew faster than it drained (one update took 2.2 h).
Scored on HOLD-TIME weights (each belief counts for the cook time it stood):

| | locked/43001 | refresh/43001 | locked/43002 |
|---|---|---|---|
| ignited-group NEES | 238 | 296 | 0.6 |
| post-load centre z | -0.92 | -0.99 | -0.26 |
| a0 / k_spread at 18 h | +2.0 / -2.1 | +2.7 / -2.8 | +1.9 / -1.3 |

The mitigations did not move the ignition pair toward band; on the paired seed
they moved it away, at the cost of the backlog spiral. An earlier reading
(ignited 248 -> 55, centre |z| 0.86 -> 0.13) was a per-update averaging
artifact: the backlogged arm's 354 updates were 60% pre-load. `stream_bench.jl`
and `scripts/plot_estimator_preheat.py` now report hold-time-weighted
statistics. The meat-temperature bias changes sign across seeds (positive on
41002, negative on 43001/43002), so it is not a one-signed linearization
(Jensen) bias. The ignition blowup is seed-dependent: seed 43001 reaches
ignited NEES in the hundreds from ~3 h in both arms, seed 43002 stays
consistent. Per the plan's pre-agreed trigger the ignition lock escalates to
the conditional (quadrature over a0, k_spread) backend; the Gaussian-path
mitigations and the planned 4 h relinearization horizon are shelved.

**Verification 2026-10-06: the first full cook.** The 18 h causal replay
`data/mock/stream_lighting_ws1_full_41002.json` (same seed and configuration,
backend `95f5c04f2599`) completed lighting → loading → spritz → wrap → fuel
addition → pull → rest: 5002 readings in 2753 updates, ONE rejected (a
3-reading batch at 17.01 h stalled at newton_sd 0.011 against the 0.01
budget; its readings re-entered by the retry queue and committed — zero
permanently lost). Wall 58,153 s; update median 20.9 s, p95 39.5 s, max
160.3 s; reading-to-estimate latency median 31.5 s, max 230.6 s; memory flat
(4 window nodes throughout). Pooled over the cook: NIS/dof 0.92, latent
NEES/16 0.72, temperature NEES/3 0.93; post-loading centre/surface RMSE
1.01/1.07 K with max standardized error 1.70/1.90. The 3 h run's +3σ
temperature climb is a POST-LOAD TRANSIENT that decays; the legacy
latent-NEES explosion (31 by 14 h) does not recur — edges stay ≤ 2.1 sd at
every 2 h checkpoint. What remains is the lock signature on single latents:
`a_myosin` pinned at −1.9 from ~10 h, `core_depth_fraction` frozen at
+1.3…+1.5 from 2.2 h, and the W state group at NEES 3.53 with several groups
far too wide (F 0.04, S 0.14, T_pan 0.04, ignited 0.04). One tuning seed:
diagnostic evidence, not held-out validation. Figures:
`figures/estimator/stream_lighting_ws1_full_41002*.png`.

**Decision 2026-10-05: the 120 s update target is dropped.** It was a
provisional label introduced during this repair (first in `13c3d4e`), never
derived from a requirement and never enforced in code: the replays recorded as
failing it were stopped by hand and annotated afterwards. Wall time and
reading-to-estimate latency remain reported diagnostics; a future runtime
requirement, if one is needed, must be derived from the planner's
belief-staleness needs. The narrative below keeps the stops as they happened —
"exceeded the 120 s update target" describes why a run was stopped at the time,
not an acceptance criterion.

The requested source exists locally at `7aace2c`. A remote fetch on
2026-10-04 returned `Repository not found` for the configured GitHub
remote; this integration therefore uses that local source commit.

## Follow-up repair and estimator figures

The follow-up adds a nonlinear dynamics search path, a roundoff guard at the
state-coordinate floors, stalled-step bookkeeping, and a fresh joint-posterior
accuracy certificate. Numerical regressions have passed; the separate causal
replay measures latency through loading. The result table under "Earlier
integration checks" describes the earlier revision.

The preceding stopped causal replay's estimator-versus-truth figures are
[`stream_lighting_budget_41002.png`](../figures/estimator/stream_lighting_budget_41002.png),
its measurement/processing/node timeline is
[`stream_lighting_budget_41002_timing.png`](../figures/estimator/stream_lighting_budget_41002_timing.png),
and its parameter errors are
[`stream_lighting_budget_41002_parameters.png`](../figures/estimator/stream_lighting_budget_41002_parameters.png).
They come from `data/mock/stream_lighting_budget_41002.json`, before the closure
fix; that run was stopped after runtime failures. The figure titles and JSON
explicitly identify incomplete, failed or completed runs.

The subsequent `f5854ac` run is saved in
[`stream_lighting_curvature_41002.png`](../figures/estimator/stream_lighting_curvature_41002.png),
with its
[arrival/latency timeline](../figures/estimator/stream_lighting_curvature_41002_timing.png)
and [parameter errors](../figures/estimator/stream_lighting_curvature_41002_parameters.png).
It was stopped after 242 committed updates and 270 of 657 readings, through
1.70222 h. The first core reading took 35 iterations and 234.809 s; a subsequent
arrival took 62 iterations over two attempts and 484.327 s. Both converged under
the unchanged accuracy threshold, but both exceeded the 120 s update target.
The next arrival was interrupted. Its prefix scheduling/input audit passed;
the completed-horizon audit is explicitly false. The first 68 updates may
include memory pressure from an idle diagnostic worker; that worker was closed
well before either long stall. No competing Julia diagnostics ran during the
timed replay. Exact snapshots of both slow arrivals were retained for controlled
comparison. These figures are evidence of a runtime failure, not a finished fix.

The causal baseline before the joint-posterior stopping change writes estimator-versus-truth figures to
[`stream_lighting_causal_41002.png`](../figures/estimator/stream_lighting_causal_41002.png),
measurement/processing/node timestamps to
[`stream_lighting_causal_41002_timing.png`](../figures/estimator/stream_lighting_causal_41002_timing.png),
and parameter errors to
[`stream_lighting_causal_41002_parameters.png`](../figures/estimator/stream_lighting_causal_41002_parameters.png).
These are generated from the saved input tape and actual estimator records in
`data/mock/stream_lighting_causal_41002.json`. The figure title and JSON state
whether the run is incomplete, failed, or completed its requested three hours.
The timing audit explicitly distinguishes a verified prefix from a completed
horizon. The loading transition is centred at 1.5 h; three hours do not cover
wrapping or pulling. Both plant and estimator use the existing logistic
`load_blend` with `MeatConfig.δ_load = 120 s`. Its nonzero leading tail explains the
surface-temperature rise before the marked midpoint; this is a model assumption,
not a timestamp shift or anticipation of future observations. This replay does
not test a hard, exactly zero-before-loading event.

Earlier saved estimator-versus-truth figures are in
[`figures/estimator/stream_lighting_guard_41002.png`](../figures/estimator/stream_lighting_guard_41002.png)
and the companion
[`stream_lighting_guard_41002_parameters.png`](../figures/estimator/stream_lighting_guard_41002_parameters.png).
The first includes reported marginal uncertainty, standardized temperature
errors, update wall times, and the remaining coupled Newton step. The second
shows every latent's error divided by its reported standard deviation. Each
figure labels whether its source cook completed, failed, or remains incomplete.
The JSON source is `data/mock/stream_lighting_guard_41002.json`; regenerate with:

```powershell
app/.venv/Scripts/python.exe scripts/plot_estimator_preheat.py data/mock/stream_lighting_guard_41002.json
```

The floor defect was independently reproduced: numerically identical posterior
means could produce standard deviations differing by 70%, because roundoff at
an encoded floor switched the derivative between zero and one. A guard only
around the existing floor makes the derivative stable while preserving the
interior coordinate transform. The nonlinear regression compares both the
posterior mode and covariance with the original search, rather than checking
only that an update reports convergence.

The dynamics search transports state endpoints along their nonlinear
propagation while retaining the proposed first-order process residuals. It
does not change the likelihood or process covariance, and the actual endpoint
must obey the caller's step bounds. The generic solver keeps this option off;
the online smoother enables it. Final convergence and covariance still use a
fresh linearization. The established Jacobian cache radii remain in use.

The search refinement in `f5854ac` estimates the directional curvature of supported
Gaussian residuals along that nonlinear propagation path. It uses two residual
evaluations at plus/minus 0.1 of the proposed step and a quadratic tangent
correction, following the geodesic-acceleration construction in
[Transtrum and Sethna (2012)](https://arxiv.org/abs/1201.5885).
The correction-size guard uses marginal posterior standard deviations to avoid
mixing coordinate units. It excludes non-Gaussian score/curvature
pseudo-residuals and the already affine process defects. Invalid probes revert
to the ordinary retracted direction; interrupts still propagate. Every accepted
candidate must decrease the original objective and obey actual endpoint bounds.
Up to four doubling trials can recover an over-shortened backtrack. Neither
the model noise nor the convergence threshold changes. The generic solver keeps
this option off; the online smoother enables it.

`test_geodesic_search.jl` has 30 assertions covering an analytically soluble
Rosenbrock step attached to a real dynamics graph, endpoint bounds, fresh
stationarity, invalid-probe fallback and interrupts. The expansion-rejection
regression now runs with both search options (21 assertions). All 493 targeted
assertions passed for `f5854ac`; the four new bound assertions bring the corrected
revision to 497.

The loading stalls exposed an invalid extra restriction in that search:
after nonlinear transport, it compared the candidate with `alpha * bound`
instead of the caller's actual bound. A tangent step already on a bound plus
an outward quadratic correction could then be rejected for every positive
`alpha`. The analytic counterexample starts at `(0.5, 0.25)` with limited
tangent `(0.1, 0.1)` and correction `(0, 0.01)`. Its half step satisfies the
original `0.1` bound, but cannot satisfy the incorrectly scaled `0.05` bound.
The old solver made no progress and failed exactly two new assertions; the
corrected solver passes all 30. The
[failed-before log](estimator-bound-regression-before-fix.txt) preserves that
counterexample. The fix changes neither the caller's limit nor the objective
or convergence tolerance.

The exact first-core checkpoint converges in four ordinary-search iterations.
The defective curvature search rejects six descending proposals, raises damping
to 100, and takes 35 iterations. Correct endpoint bounds restore four iterations.
The second slow arrival drops from 62 iterations to four. Their maximum
all-coordinate mean differences from the old converged solutions are 0.004416
and 0.001582 posterior SD; maximum relative SD differences are 0.0368% and
0.2185%. The earlier slow controls retain their 9- and 18-iteration trajectories.
The [control record](../data/mock/online_bound_controls.json) includes the paired
coordinate means and SDs as well as summary comparisons. These checks precede
the separate causal replay; they do not establish that replay's latency.

Microscopic accepted steps no longer indefinitely reset the stalled-search
counter. Three stalled trials still need the existing fresh coupled
stationarity certificate. Caller-bounded progress is excluded. Strict online
optimization requests scale the coordinate-resolution threshold with the requested
tolerance. The posterior-SD exception is now an explicit `newton_tol` keyword on
`assimilate!` and `solve_window!`, recorded with each update. Its default remains
the original 0.01 SD. Path-comparison fixtures request 0.0001 SD, ten times tighter
than their unchanged 0.001-SD comparison assertions. Inferring posterior units
linearly from coordinate `tol` had made those fixtures unnecessarily strict;
driving integration toward machine precision then became slower and still failed.
Integration accuracy remains explicit in `SystemModel`; the original
covariance allowance can be preserved when requesting more accurate integration.
An experimental automatic tolerance rule was removed because a general rule
was not justified by the controlled measurements.

The follow-up's original hard floor guard had a discontinuity. It now joins the
constant floor to the interior inverse through a C1 band only 1e-14 wide. Nine
continuity/derivative checks, 30 restart-roundoff checks, and 32 nonlinear
search/mean/covariance and public-API checks passed in the warm diagnostic process.

A frozen small version of the original delayed-arrival fixture isolated the
numerical failure. With the graph, priors, process weights, statistical floor,
and stopping gates fixed, 1e-12/1e-10 integration rejected after 60 iterations;
1e-14/1e-12 integration converged in 13. Warm times were 2.07 and 0.83 seconds.
The complete small replay accepted all 17 updates; delayed versus chronological
assimilation differed by 4.02e-6 latent posterior SD. At the failed mean, both
integration settings still saw the unfinished Newton step, so this was an
optimization stall rather than merely an inaccurate rejection certificate.
The full 40-node chronological/delayed fixture subsequently accepted all 17
updates. Final latent means differed by 5.67e-6 posterior SD, pit means by
4.07e-8 K, and latent SDs by at most 1.24e-6 relative. The largest state-SD
difference was 0.001295 relative, in collagen; temperature SD differences were
below 1.77e-7. Replacing only the delayed graph's fixed process whiteners with
the chronological graph's anchors reduced the maximum state-SD difference to
2.74e-7. This separates the existing arrival-dependent covariance approximation
from the optimizer and floor-derivative defects. The raw result is retained in
`data/mock/online_arrival_order_40.json`; no production weights were rewritten.

The strict node-versus-slice replay accepted every arrival with explicit
`newton_tol=1e-4` and integration 1e-14/1e-12, preserving the 1e-10/1e-8 noise
allowance. Mean differences were 5.39e-7 posterior SD, relative latent-SD
differences 5.37e-9, and the maximum NIS difference 3.49e-7 against its unchanged
0.001 limit. The fixture now also checks every arrival committed and records
the requested numerical stopping bound.

The saved three-hour guarded run used **node-sized batches** (`pace=-1`): 657
readings in 22 accepted updates. Its plots now label that policy explicitly.
It is numerical diagnostic evidence, not evidence of realistic streaming
latency. The causal benchmark processes each arrival when idle and only the
already-arrived backlog when busy. It preserves measurement timestamps,
records processing start/finish times, and scores only estimates available by
the scoring time. Eight deterministic checks cover the backlog and availability
rules. Graph-node placement is independent of this scheduling.
The multi-cook `stream_round.jl` launcher also no longer hard-codes `pace=-1`.
Its optional final `PACE` argument defaults to 1, and the manifest records the
chosen policy. Offline node batching requires explicitly passing -1. Use one
worker when measuring latency; concurrent cooks compete for the same CPU.

The causal replay uses all 40 spatial nodes, the default process-rate scale
of 1, lag 3, and a 600-second maximum slice. Actual integration is 1e-12/1e-10;
the covariance allowance stays 1e-10/1e-8. Sensors have their own jittered
30/30/120/120-second clocks, with core and mass absent before loading.
`pace=1` uses the measured wall time of the entire `assimilate!` call to determine
the next backlog; it includes chunk-placement overhead. Diagnostic serialization,
truth scoring, and plotting are outside that measured estimator call.
The replacement replay runs in a separate process, warmed on a saved loading
arrival, with four Julia threads and one BLAS thread. It measures warm estimator
latency; a fresh process also pays compilation cost. The record retains its source fingerprint from
startup. The baseline retains the algorithm loaded before the joint-posterior
stopping change. The replacement replay loads the changed solver and writes a
separate record; fingerprints in existing
records are never rewritten to match later source edits. Both runs use the same
saved measurement tape, checked by its input fingerprint.

For a backlog crossing several graph nodes, `assimilate!` performs several
internal chunk solves. The record's wall time covers all of them, while its
iteration count, attempts, and final stationarity metrics describe the last
chunk. New records include `chunks` and label that scope explicitly.

```powershell
$env:STREAM_START='lighting'
$env:STREAM_NOISE='matched'
$env:STREAM_INITIAL='prior'
$env:STREAM_SN_SCALE='1'
$env:STREAM_ABSTOL='1e-12'
$env:STREAM_RELTOL='1e-10'
$env:STREAM_FLOOR_ABSTOL='1e-10'
$env:STREAM_FLOOR_RELTOL='1e-8'
$env:STREAM_MAX_SLICE_S='600'
$env:STREAM_FAIL_FAST='true'
$env:OPENBLAS_NUM_THREADS='1'
julia --threads=4 --project=estimator estimator/scripts/stream_bench.jl 1 41002 stream_lighting_reproduction_41002.json 3 3 slice 1 30 30 120 120 1 3 Inf
app/.venv/Scripts/python.exe scripts/plot_estimator_preheat.py data/mock/stream_lighting_reproduction_41002.json
app/.venv/Scripts/python.exe scripts/validate_stream_timing.py data/mock/stream_lighting_reproduction_41002.json figures/estimator/stream_lighting_reproduction_41002_timing_audit.json
```

The standalone command includes any compilation inside assimilation. The saved
replacement timings used the warmed process described above, so a cold rerun can
produce different processing times and backlog grouping even with the same tape.

Regressions on `2294f39` passed: joint stationarity 14/14; factor solver
263/263; floor/retraction/API 71/71; online transactions 47/47; strict node/slice
agreement 21/21; capped-slice observations 17/17; fixed-lag comparisons against
the saved full-history references 13/13; rejected curved expansions 21/21;
curvature search safeguards 30/30 (497 assertions total). The captured
[regression output](estimator-preheat-regression-output.txt) retains the test
summaries and the backend fingerprint used to release the replay. The additional
API test constructs a reading requiring exactly 0.005 joint posterior SD of
motion: the default budget permits the existing mean, while a strict caller
gets a refined estimate. Disabling and tightening the budget are also checked.

Before removing the hidden joint cap, the complete lag test passed 10/10 and
took 29m10s with two Julia threads. Its delayed versus chronological full-history
means differ by at most 0.0000694 latent posterior SD and 0.00000714 K in the
pit, below the unchanged 0.001 limits. Preheat 25/25 and causal scheduling 8/8
also passed before the joint stopping change; those implementations are unchanged.
These are targeted checks, not a pass of the entire repository suite or a
statistical consistency certificate.

### Joint posterior accuracy and the loading runtime

The causal baseline accepted its loading arrivals, but exceeded the 120-second
update target: updates 197 and 198 took 133.9 and 144.4 seconds, with the oldest
reading waiting 276.6 seconds for an estimate. Those observations precede the
concurrent diagnostic work. From update 200 onward, baseline timing may include
contention with a second process investigating frozen arrivals; those later
times are not isolated latency measurements.

That baseline completed all 657 readings in 242 committed updates, with no
rejections. Its complete-horizon timing/input audit passed. The median update
was 20.90 seconds, p95 131.15 seconds, and maximum 608.75 seconds; the largest
reading-to-estimate latency was 1110.16 seconds. Its raw JSON, summary and plots
retain the concurrent-diagnostic timing caveat. Those timings do not establish
an isolated speed ratio against the replacement replay.

A saved loading arrival isolated the cost. After six iterations its mean was
within 0.000366 posterior SD of the final 28-iteration result, and the maximum
relative SD difference was 0.0000965. Repeated fresh Jacobians were polishing
adaptive-integration noise after the posterior was already accurate.

The online solver now also checks the fresh Gauss--Newton decrement
`norm(J * delta)`, where `delta` is the full undamped coupled step. Under the
local Gaussian covariance `C = inv(J'J)`, this equals
`sqrt(delta' * inv(C) * delta)` and bounds the displacement of every linear
contrast in its own posterior SD. The online target is the caller's
`newton_tol`, default 0.01. The existing coordinate test and three-stall numerical
exception remain available; `newton_tol=0` disables both posterior-relative
checks. The generic solver leaves the new `joint_tol` criterion off by default.
The nonlinear objective, observations, process weights and integration settings
are unchanged by this criterion. Records expose both joint and marginal metrics.

| Frozen arrival | Earlier iterations | Declared 0.01 budget | Maximum mean change / SD | Maximum relative SD change |
|---|---:|---:|---:|---:|
| Before loading, reported latents | 13 | 3 | 0.000782 | 0.0000163 |
| Original loading graph, all coordinates | 28 | 4 | 0.003022 | 0.001051 |
| Follow-on loading graph, all coordinates | 14 | 5 | 0.006417 | 0.002707 |

The preloading pit mean changed by 0.0000221 K. Each loading comparison uses the
same prepared graph, weights, observations and starting values. These controls
measure numerical work and posterior agreement; different thread counts and
concurrent CPU use prevent treating their wall times as an isolated speed ratio.
Fourteen dedicated tests cover the joint criterion, including a correlated
direction with tiny marginal errors but a remaining displacement of twenty
joint SDs. The original factor, slice, transaction and delayed-arrival assertions
are retained and rerun for this change.

The first joint implementation added an extra 0.001 cap below the caller's
declared 0.01 budget. It helped the frozen case but failed the causal runtime
gate: update 256 took 164.7 seconds and 18 iterations. Its following arrival
took 14 iterations, including a restart and repeated rejected searches at the
same objective. A frozen replay of that following graph reproduces the result;
honoring the declared joint budget takes five iterations, with the differences
in the table above. The original strict node/slice and delayed-history fixtures
explicitly request 0.0001, so their accuracy target remains unchanged.

`stream_lighting_joint_41002.json` retains that stopped candidate: 260 committed
updates, 359 of 657 readings, through 1.99389 h. Its prefix timing audit passes,
but it **did not complete** the requested three hours. Update 256's runtime
failure preceded concurrent frozen-arrival diagnostics; later timing may be
contended. The JSON and plots retain both the termination reason and that caveat.
`stream_lighting_budget_41002.json` is the separate replay of the revised policy,
started after the controls and regressions passed. The raw controls are
`online_joint_budget_loading_control.json` and
`online_joint_budget_additional_controls.json`.

The declared-budget replay also failed the runtime target. Update 269 took
127.884 seconds and 12 iterations. A second arrival, after 287 committed
updates, remained in flight for several minutes and the run was stopped.
The saved prefix contains 373 of 657 readings through 2.04583 h, with no rejected
completed updates. Its prefix timing/input audit passes, but its complete-horizon
audit does not. Both exact pre-arrival states were preserved. The first case
reproduces all 12 iterations; the second still has a meaningful remaining
posterior step after 40 iterations. These are unresolved runtime failures,
despite the targeted regression passes. No further accuracy relaxation has been
applied; the line-search trajectory is under controlled investigation.

That investigation exposed a correctness defect in the new retraction path:
the nested trial function assigned `xn`, which Julia captured from the outer
solver. A rejected expanded trial could overwrite the accepted state while
retaining the previous objective value. The fix uses a separate trial-local
variable. Two analytic expansion cases reproduce six failures before the fix
and pass all 11 assertions afterward. Replaying the two real slow arrivals with
only this fix changes their means by at most 0.002474 and 0.001211 posterior SD,
and their SDs by at most 0.1154% and 0.0961% relative. They still take 12 and
44 iterations, so this correctness fix does not by itself solve the runtime
problem.

The production curvature search subsequently passed both frozen-arrival
comparisons with the original graph, starting point, noise, and tolerance.
The controls are saved in
[`online_curvature_controls.json`](../data/mock/online_curvature_controls.json).

| Frozen arrival | Previous search iterations | Curvature search iterations | Final joint step (SD) | Maximum mean difference (SD) | Maximum relative SD difference |
|---|---:|---:|---:|---:|---:|
| First slow arrival | 12 | 9 | 0.004466 | 0.002401 | 0.118% |
| Second slow arrival | 44 | 18 | 0.0009395 | 0.003960 | 0.315% |

Mean and width comparisons use the saved converged reference on each frozen
graph. The joint-step threshold remains 0.01. These two-thread diagnostic
timings overlapped with a separate worker's warmup, so they are not isolated
arrival latency measurements or a speed ratio. The four-thread causal replay
begins only after this diagnostic worker is idle. Adaptive damping and repeated
transported-correction experiments did not pass the bounded controls and were
not adopted.

The earlier rejected 2.39472-hour arrival now commits in the guarded tuning
run. This is evidence of a repaired solver failure, not a statistical
consistency certificate. The plots expose both slow updates and temperature
errors outside the reported 95% intervals; neither is omitted from the record.

## Earlier integration checks

| Gate | Recorded outcome |
|---|---|
| Root physics | 703/703 pass |
| Factor solver | 259/259 pass |
| Probability-model consistency | 104/104 pass |
| Preheat / slice observations / capped-slice regressions | 25/25, 18/18, 17/17 pass |
| Online estimator | 165/165 pass |
| Multi-piece assimilation and event handling | 126/126 pass |
| Multi-piece offline check, with explicit integration accuracy | 3/3 pass |
| Original offline hierarchy, with explicit integration accuracy | 22/22 pass |
| Heterogeneous offline hierarchy, with explicit integration accuracy | 18/18 pass; expensive accuracy setting |
| Planner | Single piece 19/19 pass; multi-piece convergence failure |
| Streaming scorer | 21/21 pass |
| Held-out complete-cook statistical consistency | Not established; no held-out round launched |

These are separate targeted runs, with their timing and source-change order
recorded below. They are not a claim that the entire repository test suite passed.

## Integration and diagnostics

- The propagation conflict combines observation-time saves from the
  streaming branch with latent-dependent ROM exchange corrections from
  the photo branch. Streaming observation corrections now also evaluate
  at the current latents and carry their derivatives.
- The estimator manifest lacked the physics package's new dependencies.
  Resolving it adds SparseArrays and OpenSSL_jll to that package's entry
  and aligns OpenSSL_jll with the root project's 3.0 compatibility bound.
- The legacy streaming fixture starts the pit at 362.43 K with meat already
  loaded and uses the old ignition prior. A separate `STREAM_START=lighting`
  design starts at ambient, loads at the refit's documented 90 minutes,
  and shifts the subsequent controls and events consistently.
- The lighting validation prior uses independent marginal distributions
  centered on the from-lighting MAP, with its Fisher widths. Bounded
  fractions use beta distributions matching the local delta-method width.
  Initial ignition has a declared beta prior with mean 0.15 and sd 0.05;
  this spread was **not inferred** by the refit, which fixed a0. The old
  production prior and calibration chain are preserved.
- `STREAM_NOISE=matched` matches the core interpolation and stationary
  OU plus white instrument noise. The default legacy stress case keeps
  the linear probe interpolation and the post-add pit-noise bump. Neither
  deterministic plant is a draw from the estimator's stochastic process
  model; the process-rate scale is recorded explicitly.
- Segmented handoff now preserves physical ignition endpoints instead of
  pushing a fully lit bed back to `1 - 1e-6`. System solves use the same
  load-transition integration stops as the monolith.
- The monolith's vapor closure included the entire meat source during
  partial loading, while the system included only the loaded fraction.
  The monolith now gates the source inside the closure as well as the
  final flux. Six RHS comparisons across the load cover this regression.
- Asynchronous slices resampled the damper onto a new grid, smoothing
  over original control corners. `damper_knots` now returns the original
  covering knots; integration still starts and ends at the requested
  times. This removes a 0.085 K pit discrepancy at the loading boundary
  in the full-resolution seed-41002 diagnostic. The final 4 h diagnostic
  has every state group's whitened mean-square residual below 0.5 at
  default process rates, and matched sensor predictions agree to rounding
  (`scratchpad/estimator_preheat/truth_exact_controls.log`).
- Batched innovations now retain every reading, whitened in chronological
  order against the joint predicted covariance. Keeping only each
  channel's last marginally standardized residual could not test
  innovation whiteness. Their squared norm remains the same NIS.
- Capped readings on the slice path inserted a state node without closing
  the preceding observation slice. Delayed capped readings also left
  later readings attached to the wrong anchor. Node-backed readings now
  split/rebuild the process and OU bridges, preserving the shared end
  state and the dynamics/slice solve cache. A focused regression failed
  9 of 12 checks before this fix and passes all 12 afterward.
- A capped observation in a mixed batch was included in the covariance
  used to score that batch's uncapped readings. The slice prediction now
  excludes all newly arriving observation factors, matching the node
  path. Five regression checks compare with explicitly removing the
  capped factor, verify restoration of factor order, and demonstrate that
  including it would change the forecast uncertainty.
- The stopping test divided each score coordinate by its own curvature,
  ignoring correlations. For `J = [1e6 -1e6; 1 1]` and data `[0, 2]`, it
  certified `[0, 0]` although the exact mean is `[1, 1]`, two posterior
  standard deviations away. The check now solves for the full undamped
  Gauss--Newton displacement. Its public diagnostic field is still named
  `gradient`; it measures a coupled coordinate displacement. Regression
  checks cover the solution, residual scaling and sparse QR.
- A failed line search could keep using a stale Jacobian for the entire
  iteration budget, only refreshing it at the final certificate. An
  analytic nonlinear regression reached its optimum in one step, then
  spent nine iterations rejecting the same stale direction. Failed
  searches now invalidate stale Jacobians immediately. The same example
  finishes in three iterations with unchanged tolerances. Trial residual
  evaluations also preserve a Jacobian already made at the restored
  accepted point; the primal and derivative cache points are distinct.
- The online solver can request its configured restart early after three
  failed searches if a freshly computed joint step already lies inside
  the existing posterior-width resolution envelope. That request returns
  `converged=false`; only the final attempt may accept under the existing
  relaxed resolution rule. This avoids spending the first attempt's
  entire budget damping an unchanged, numerically stalled point.
- `SystemModel` now separates actual integration accuracy from its
  process-covariance error allowance through `floor_abstol` and
  `floor_reltol`. Defaults preserve the previous relationship. Holding
  those fields fixed allows tighter solves without silently reducing
  state uncertainty. Execution updates preserve both settings; benchmark
  records and pooling checks include them.
- The bench saves its starting source/backend fingerprints, complete
  declared priors, failure details and actual scoring times. Short runs
  stop the plant at the requested horizon. Both plant and estimator use
  1e-10 absolute / 1e-8 relative integration tolerances.

The legacy full-resolution truth diagnostic over 14 h showed a maximum
listed temperature-field dynamics residual of 1.6 times its process
variance per coordinate at 8.5 h, and sub-0.01 K physical pit discrepancies.
Its core-observation discrepancy is about 0.0045 K early in the cook
because the two probe interpolations differ. This is diagnostic evidence,
not statistical consistency. The log is
`scratchpad/estimator_preheat/truth_merged_14h.log`.

## Validation record

Commands use Julia 1.11.9 and one OpenBLAS thread. Direct benchmark runs
use four Julia threads unless their command overrides it. The test shard
driver originally dropped the parent's `--threads=4`, leaving its children
at Julia's default; it now forwards the requested pool explicitly. The
already running shards continue on their original pool. Their timings
also include competition with tuning jobs on this 16 GB machine.
`scratchpad/estimator_preheat/run.py` captures Julia's actual exit
code and combined log; PowerShell's native-stderr formatting otherwise
reports an error merely when `Pkg.activate` writes its normal message.

- Initial merge checks: factors 248/248, consistency 90/90 and slices
  11/11 passed. ROM passed 60/60 after the physics fixes were stable.
  New controls and innovation diagnostics require another affected-suite
  run, which is in progress.
- The root physics suite passed 703/703 after the vapor-closure fix.
- Fixture/preheat/handoff checks passed 19/19. Six new checks for
  preservation of control corners also passed (25 total). Updated
  consistency checks passed 94/94, including chronological whitening.
- The first preheat assertion sampled only 2.5 logistic widths before
  loading, where the model intentionally blends the load. It now samples
  20 widths before loading, consistent with the existing physics tests.
- Seed 41001's initial 3 h smoke used fixed initial conditions and long
  slices (four updates). It is historical tuning evidence only. Subsequent
  lighting runs draw initial coordinates from the estimator's declared
  initial-state prior and truth latents directly in their prior coordinates.
- With process rates scaled by 0.001, the node threshold is scaled by
  its square: `node_ratio=1.5e-7` corresponds to 0.15 at default rates.
  This retains roughly five-minute slices instead of allowing many hours
  of observations to accumulate in one dense factor. Two-hour scoring
  checkpoints also stop batches before the edge. Neither choice changes
  the production defaults or the frozen comparison backends.
- A full 18 h, 40-node truth diagnostic with sampled initial conditions
  passed through all events. At process-rate scale 0.001, the largest
  per-group whitened mean-square residual was 0.0229 (pit temperature).
  Maximum physical errors were 0.000240 K in the pit and 0.0000474 K in
  the meat field; sensor prediction errors were at floating-point rounding.
  The complete record is `data/mock/truth_lighting_41002.json`.
- The optional `max_slice_s` age cap passed four placement checks. Its
  default is `Inf`; a benchmark can use `STREAM_MAX_SLICE_S` to bound
  dense factors while keeping longer nearly deterministic propagation
  slices. This option's statistical and runtime assessment is pending.
- The planning regression's single-slab checks passed 19/19. The two-piece
  replan failed to reach stationarity within its test's 30-iteration budget.
  This remains under investigation; it is not counted as a passing suite.
- Six analytic checks of the scorer's chi-square and Student-t bands and
  independent-cook handling passed. Tuning records are not validation.
- The coupled stopping test passed all 253 factor checks, including the
  Kalman/RTS oracle and marginalization tests (`factors_coupled.log`).
- With the cache-retry fix, factors passed 255/255 and probability-model
  consistency checks passed 96/96 (`cache_factor_consistency.log`).
  The cache tests check that restoring an accepted mean reuses its exact
  Jacobian even when a trial evaluation moved the primal cache.
- Early-restart safeguards passed with all 259 factor checks
  (`restart_factor_tests.log`), including refusal to label a requested
  restart as convergence and refusal outside its posterior-step bound.
- The updated slice-observation suite passed all 18 checks on four Julia
  threads in 496.5 s (`cache_slice_tests.log`), including age-cap placement,
  cached derivatives, node/slice equivalence and conditioned innovations.
- Online regressions passed 165/165 in 3379 s on two Julia threads,
  including transaction rollback, delayed observations, full-history
  comparison and the frozen RxInfer tangent oracle. This run preceded
  the capped-slice fix; the latter's 17 targeted checks passed separately
  in 204.8 s (`censored_slice_after.log`).
- After adding independent covariance-floor controls, probability-model
  consistency passed 104/104 (including eight new covariance/constructor/
  execution checks), preheat passed 25/25, and scoring passed 21/21.
  The original offline hierarchy passed 22/22 at its explicitly tighter
  integration accuracy. The heterogeneous hierarchy initially passed 17
  assertions but failed convergence at 0.0000483 against `tol=2e-5`, so
  that combined run failed (`floor_accuracy_regressions.log`). A subsequent
  run at 1e-15/1e-13 accuracy passed all 18 assertions in eight iterations,
  with final displacement 0.00000771. It cost 12 min 43.6 s within the
  testset, 808.9 s including startup (`semantic_accuracy_probe.log`). The
  final fixture uses that accuracy only for the heterogeneous models;
  production defaults remain unchanged. The two offline testsets passed
  in separate runs, not one final combined invocation.
- The production benchmark exercised the new controls successfully on a
  three-minute lighting prefix: one committed update, four iterations,
  40.2 s for the update, 139.6 s including startup and plant construction.
  `stream_lighting_floor_api_smoke_41002.json` records both sets of
  tolerances. This is an API smoke test on a tuning seed, not a cook-length
  or statistical-consistency result.
- Multi-piece online checks passed: 12 layout, 73 assimilation, 36 piece
  permutation and 5 complete-event-cook checks. Maximum permutation
  differences were 0.000224 K and 0.000792 latent-coordinate units.
  Its separate sparse offline check failed convergence at 15 iterations;
  the two unobserved-group assertions passed. That failure is being
  diagnosed, not counted as a passing file.
- Four concurrent short-slice tuning workers loaded the old stopping
  test. They were stopped after preserving their logs when memory
  contention prevented the corrected-solver diagnostic from starting.
  They are **incomplete, superseded tuning runs**, not completed cooks
  or statistical evidence. The process commands and reason are recorded
  in `scratchpad/estimator_preheat/superseded_tuning.json`. Seed 41001's
  full-horizon worker had already rejected its two-hour arrival.
  Subsequent heavy diagnostics run serially alongside the regression
  shard, with memory headroom checked before starting.
- New lighting runs stop and save diagnostics at the first uncommitted
  arrival (`STREAM_FAIL_FAST=true`). Such a run already fails the
  complete-cook criterion; continuing it cannot recover that criterion.
  The scorer records even a round with zero completed cooks as a failure.
- The first bounded-slice tuning run (seed 41002, 3 h requested,
  `max_slice_s=3600`, `node_ratio=0.00015`, lag 2, rate scale 0.001)
  used the coupled certificate before the cache/restart fixes. It
  committed through 2.39 h, then rejected the 2.43 h update: remaining
  coordinate step 0.0206 and posterior step 0.0782 sd. Its two-hour
  update took 595 s and 53 iterations; this configuration had not met
  the runtime gate. `data/mock/stream_lighting_tune_capped_41002.json`
  retains the failed trajectory; it contains zero completed cooks.

The source branch's fingerprinted ROM and process-noise artifacts are
not re-earned by these full-order checks. The physics changes invalidate
their old fingerprints; strict artifact checks remain enabled. The
lighting fixture's declared validation prior is not a newly calibrated
production prior.

## Earlier failure investigations and remaining validation

The following paragraphs preserve the earlier experiments and their limits.
Statements that a change was still experimental describe that point in the
investigation; the follow-up section above records the current implementation
and validation. The historical runs are not evidence for the current revision.

The updated cache/restart path still needed 50 iterations and 610 s at
the two-hour loading checkpoint of seed 41002 (the earlier path needed
53 iterations and 595 s on this contended machine). It did not remove
that bottleneck. It subsequently rejected the 2.39472 h update, with
coordinate step 0.002263 and posterior step 0.013518 sd, in
`data/mock/stream_lighting_tune_cache_41002.json` (zero completed cooks).
A same-seed diagnostic used 1e-12 / 1e-10 integration tolerances: the
1.496 h update took 21 iterations and 158.4 s, against 5 iterations and
54.1 s at baseline tolerances. It was stopped after missing the declared
120-second runtime gate; only three loading iterations ran, so this is
not a completed cook. These tolerances also enter the existing integrator
covariance floor, so this is a numerical-model sensitivity check, not a
comparison with an identical likelihood. A separate scratch experiment tests a
nonlinear dynamics retraction and labels its backend as experimental;
it is not production code or held-out validation.

With the capped-slice correction, the experimental retraction/accuracy run
committed seven updates through 2.4975 h, including three capped humidity
readings. The formerly failing 2.43056 h update committed with a remaining
joint step of 0.000411 posterior sd. It still took 241.9 s; the 2.39472 h
and 2.4975 h updates took 340.2 s and 222.1 s. The run was stopped after
1326.1 s because it repeatedly missed the 120-second update gate.
`stream_lighting_censor_fixed_41002.json` remains an **incomplete active
cook**, with zero completed cooks; the scorer refuses to certify it.
The retraction remains a scratch experiment. Production solver settings
and acceptance thresholds were not relaxed to adopt it.

The first retraction experiment reduced the two-hour update from 50 to
16 iterations (174.2 s on two Julia threads, versus 610.1 s on four in
the cache baseline; these runs shared the machine with regressions).
It committed the 2.39472 h update in 19 iterations / 241.6 s, but rejected
2.43056 h after 37 iterations / 397.1 s: its posterior step was 0.013493 sd,
outside the existing 0.01 bound. It therefore remains experimental.
`stream_lighting_retraction_probe_41002.json` contains zero completed cooks.
The next scratch diagnostic holds the original integration-error covariance
allowance fixed while tightening the actual solves, to separate numerical
accuracy from changing the factor weights.

The multi-piece planning regression also failed a separate diagnostic
using its production budget of 50 iterations per start. Neither start
reached the new coupled-displacement certificate. The failed result is
preserved in `scratchpad/estimator_preheat/plan_default_budget.log`;
the budget has not been raised and the stopping gate has not been relaxed.
The 30-iteration failure occurred before the coupled convergence check was
added (the first solver probe was at 03:59, after that planning run ended).
An isolated no-adaptive-damping trial also failed at the production budget
of 50 iterations per start (`plan_no_adaptive_probe.log`, 485.9 s).
A Powell-damped BFGS trial also failed both starts at that budget
(`plan_bfgs_probe.log`, 403.1 s): final coupled displacements were 0.842
and 0.830. Neither experimental planning algorithm is in production.
Tightening the planner's integration to 1e-12/1e-10 also failed both
starts at the original test budget of 30 (displacements 0.142 and 0.0849;
`plan_accuracy_probe.log`, 385.2 s). That change was not adopted either.

The offline regressions also expose unresolved stopping failures. The
two-cook hierarchy and the standalone cook failed their convergence
assertions; the other 20 assertions in that testset passed. The two-cook
solve stopped with coupled displacement 0.000683 against `tol=1e-5`.
The multi-piece offline test passed its two state-group assertions but
failed convergence at 0.001601 against `tol=1e-4`. Normalizing humidity
discrepancy coordinates by their sensor sd left that result unchanged:
the largest remaining step belongs to a physical state coordinate.
Tightening integration from 1e-10/1e-8 to 1e-12/1e-10 while keeping
the original covariance allowance reduced the two-cook displacement to
0.0000918, but both convergence assertions still failed. This is an
accuracy diagnostic, not a production tolerance change or a relaxed gate.
The same accuracy change passed all three multi-piece offline assertions
in its original 15-iteration budget (final displacement 0.0000984,
`multi_precision_probe.log`, 191.5 s including startup). This isolates a
numerical-accuracy contribution to that failure; it does not establish
calibration or justify changing every model's default tolerances.
The actual API subsequently reproduced the multi-piece offline pass,
3/3 checks in 116.9 s including startup (`multi_floor_accuracy.log`).
At 1e-14/1e-12 integration accuracy, still holding the allowance at
1e-10/1e-8, the hierarchy converged in nine iterations at 0.00000982;
all 22 legacy offline assertions passed (`offline_precision_tight_probe.log`,
216.2 s including startup). The offline fixtures now explicitly request
the accuracy their existing convergence thresholds need; neither their
iteration budgets nor their thresholds nor covariance allowances changed.
The production API for this choice is:

```julia
SystemModel(sys, log, blk; abstol=1e-14, reltol=1e-12,
            floor_abstol=1e-10, floor_reltol=1e-8)
```

For streaming diagnostics the corresponding controls are
`STREAM_ABSTOL`, `STREAM_RELTOL`, `STREAM_FLOOR_ABSTOL`, and
`STREAM_FLOOR_RELTOL`. This does not make the remaining streaming
convergence or calibration gates pass by construction.

The scorer now accepts the benchmark's nonfinite diagnostic encoding and
writes an explicit failure when there are zero completed cooks. Twenty-one
regression checks verify preserved NaN/Inf failure details, refusal to
score an active worker, analytic chi-square/Student-t values, and
per-site/state-group failures despite passing aggregate NEES. The scorer
now gates every latent's independent-cook bias, variance and coverage,
and every reported state group's normalized NEES. These are pointwise
95% tests; with many comparisons, some failures are expected by chance.
The first rejected capped-reading update had 11 readings but only 10
uncensored innovation degrees of freedom. Filtering out capped readings
selects a truncated subset, so unit-Gaussian innovation tests no longer
have their advertised null distribution. The scorer reports this and
cannot certify all gates when censoring occurred; dedicated censored
predictive-calibration tests remain necessary.
Per-update records now retain latent means, widths and truths even when
a later update fails, so tuning can distinguish a moving mean from a
changing standardized error. The obsolete, duplicate `test_online.jl` worker
was stopped after verifying its process ancestry; the replacement runs
the current solver and forwards the requested Julia thread count.

Remaining acceptance work is the unresolved numerical regression failures and
censored predictive calibration; wall time and reading-to-estimate latency are
reported alongside the statistical gates, not gated themselves (decision
2026-10-05 above). Freeze
a working configuration before running disjoint held-out seeds through
loading, wrapping, fuel addition, pulling and resting. Statistical claims
must distinguish independent cooks from correlated arrivals, count every
failed/uncommitted update, and state the configuration and time interval
actually assessed. The 18-hour truth-trajectory result does not substitute
for posterior coverage or forecast calibration. Existing calibrated-noise
and ROM artifacts still require their normal fingerprint checks and fresh
validation; their fingerprints were not rewritten to bypass those checks.
