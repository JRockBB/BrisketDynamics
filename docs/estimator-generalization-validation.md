# Estimator consistency and deployment validation

The estimator must work with a selected, well-instrumented sensor suite and with customer installations
that provide only one or two temperature probes. The probability model reads sensor specifications;
no covariance correction, inference setting, or fitted noise rate may depend on a test seed or suite name.

## Separation of development and evaluation

- Seeds 3001–3008, 3011–3014 and 10001–10064 have been used for diagnosis. They are development cases, not evidence
  of generalization. A selected extreme case is useful for comparing posterior approximations only.
- Seeds 20001–20004 belong to the frozen independent smoke evaluations. Do not reuse them as untouched
  evidence after inspecting their results. Reserve **21001–21064** per suite for the larger study after
  freezing the inference algorithm. Record the code fingerprint, model configuration, seeds, failures,
  and all attempted cooks; the larger study has not been run.
- Evaluate each sensor configuration separately. A large suite must not hide failures with one probe.
- Do not fit physical/noise parameters against the evaluation truths or discard failed updates/cooks.
- Prior and noise calibration data must be separate from validation cooks. Customer cooks need population
  variation and sensor-specific calibration uncertainty; calibration on the same cooker is not a transfer test.

## Required configurations

| Configuration | What it exercises |
| --- | --- |
| One meat-temperature probe, uncertain insertion depth | State/parameter ambiguity and reliance on priors |
| One pit-temperature probe | Unobserved meat state and probe-depth prior remaining unconstrained |
| Two meat-temperature probes with independent depth uncertainty | Added spatial information without pretending depth is known |
| One pit and one meat probe | Minimal coupled cooker/meat instrumentation |
| Pit, meat, humidity and mass | The selected instrumented suite |

The matched-control script exposes these as `:core`, `:pit`, `:two_core`, `:pit_core`, and `:full`.
These labels only construct sensors; inference must use their ordinary `SensorSpec` declarations.
Use different declared noise levels and sampling cadences, followed by asynchronous arrivals, dropout,
censoring, missing sensor metadata, and uncertain placement. Do not invent known placement or noise when
the installation does not supply it. Unsupported configurations must be identified explicitly.

## Levels of evidence

1. **Declared-model controls:** draw parameters, initial state, process disturbances and sensor noise
   from the model actually fitted. Use posterior rank/quantile coverage, bias and predictive calibration
   across independent cooks at fixed times. Check state and parameter blocks separately. For a non-Gaussian
   posterior, do not assume a chi-square distribution for Mahalanobis errors or use mean ± 1.96 SD as a
   substitute for posterior quantiles. Report Monte Carlo error and sampling diagnostics.
2. **Model-mismatch controls:** vary geometry, cooker dynamics, noise correlation/bias, sensor placement,
   actions and disturbances independently of the fitted model. Report degradation and whether uncertainty
   honestly reflects it. A matched simulation cannot establish robustness to physical model error.
3. **Held-out real cooks:** evaluate instrumented reference cooks, replayed with each deployment subset.
   Withheld reference sensors must not enter the likelihood. Compare predicted temperatures/trajectories
   against reference measurements with their own uncertainty. Many physical parameters lack observable
   ground truth; predictive performance is not proof that every latent parameter is identified.
4. **Closed-loop validation:** replay or simulate sensor selection and control based on past information,
   accounting for the executed actions in the model. Selection based on values (threshold/reporting rules)
   needs its likelihood represented; it is not ordinary missing-at-random data.

Sparse sensors should reduce information. The acceptance criterion is calibrated uncertainty and useful
predictions, not the same narrow intervals or parameter recovery achieved with the instrumented suite.
Parameters left unobserved must retain appropriate prior uncertainty. Additional sensors should contribute
information according to their noise, dependence, placement and timing, not according to their count.

Temperature-only installations may also lack a reliable control/action history. Unknown vent settings,
fuel additions, ambient conditions, probe positions and calibration biases must remain uncertain inputs
or parameters. The current `SystemModel` takes an executed `CookLog`; substituting a fixed damper history
when none was observed is an additional model assumption, not sensor agnosticism. The sampled backend
does not yet infer arbitrary unknown action histories. These deployment cases therefore still require
model support and independent predictive validation beyond removing sensors from a fully known simulation.

This is the validation specification, not a claim that these checks have all passed.

The [sampled online/offline implementation](sampled-estimation.md) now shares a joint posterior
representation. Its four-case comparison uses dedicated seeds 22001–22004 and tests implementation
agreement under asynchronous/dropout sensor schedules; it does not consume or replace the reserved
calibration study. Online and offline agreement can share model errors, so it is not transfer evidence.

## Running the non-Gaussian controls

`consistency_posterior.jl` reports actual marginal quantile coverage, marginal posterior ranks and joint
radial ranks for latent, state and combined blocks. Its family-wise rank-CDF and coverage gates account
for five sensor configurations. A run cannot pass if a cook failed, sampling precision failed, or fewer
than 64 independent cooks were attempted. A small smoke run remains useful but is not a calibration pass.
These statistical checks address the declared model; they do not establish robustness to model mismatch.
The 64-cook minimum is a guard against calling a smoke run a study, not a guarantee of power against
every form of miscalibration. Passing means these predeclared checks did not reject, not proof of exactness.

After freezing the algorithm and settings, run each suite separately. The arguments are cook count,
first seed, suite, output, method, particles per independent system, maximum particles, independent
systems and rejuvenation moves:

```powershell
julia --threads=4 --project=estimator estimator/scripts/consistency_posterior.jl 64 21001 core posterior_core.json smc 2048 4096 8 4
```

Repeat with `pit`, `two_core`, `pit_core` and `full`. This can be expensive; record runtime alongside
precision failures. Failed cooks must remain in the denominator. Do not increase confidence by counting
several timestamps from a cook as independent cooks or by applying Gaussian chi-square gates to the
sampled posterior.

The repository's `data/cook_log/mache_brisket_cook.csv` is already calibration material, digitized from
a published cook trace. Reusing it as a held-out real-cook test would not demonstrate transfer. Independent
reference logs remain necessary, ideally with calibration, sensor metadata, executed actions and enough
instrumentation to replay one- and two-probe subsets while withholding the other readings.

For suite design, score the reduction in uncertainty of the quantities needed for decisions, together
with the ability to detect model error. Parameter information alone can favor a sensor that produces
precise but wrong predictions under missing physics. Evaluate selection policies against changes in
priors and model discrepancy, and retain the time-correlated sensor error in future-reading predictions.
This is supported by the distinction between information gain and discrepancy detection in
[Catanach and Das, Metrics for Bayesian Optimal Experiment Design under Model Misspecification](https://arxiv.org/abs/2304.07949).
