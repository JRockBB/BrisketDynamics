# Posterior validation evidence

These are small declared-model simulation checks. **They do not establish statistical calibration or
transfer to real cooks.** Every per-suite report retains a failed statistical gate because the number
of independent cooks is below the predeclared minimum. Uniform particle weights are not treated as
independent samples; precision diagnostics use independent particle systems.

- `original_*.json`: the first frozen sampler, seeds 20001 and 20002 in each of five sensor suites.
  `original_smoke_summary.json` also retains the initial invalid grid-coverage attempt. All ten cases
  were rerun after correcting that setup, without changing inference or precision requirements.
- `blocked_heldout_*.json`: a separately frozen candidate adding an invariant refresh of weakly informed
  directions. Seed 20003 is used for all five suites; seed 20004 uses the full suite at later times.
  The candidate-kernel, target-backend and validation/prior fingerprints are recorded separately.
- `development_*.json`: selected development precision checks, including failures. They are not held-out
  evidence and must not be pooled with the independent evaluations. The warm-process fingerprints in
  these older development reports are not frozen execution attestations.
- `original_posterior_smc.jl.txt` and `candidate_kernel.jl.txt` preserve the original sampler source and
  exact candidate patch module. The patch's relative source path refers to its original scratch directory;
  for reproduction it must be redirected to the archived original sampler, not the current integrated one.
- `gaussian_covariance_audit.json` compares ordinary, scaled and 256-bit covariance solves for the
  selected development diagnostic. It is a numerical audit, not another independent cook.
- `posterior_geometry.json` and `plot_geometry.ps1` reproduce the corrected exploratory figure using
  Windows .NET Charting. These selected development draws are not a numerically qualified independent
  validation sample; the figure must not be used as a calibration result.

Within a case, Gaussian and sampled state scores use the same already-constructed graph and readings.
Across suites the control harness may construct different adaptive process covariances. Reusing a seed
across suites does not create independent cooks for a pooled significance test.

The 64-cook-per-suite study, model-mismatch studies and independent real-cook validation remain open.
See [the review](../../estimator-consistency-review.md) and
[the validation protocol](../../estimator-generalization-validation.md) for limitations and commands.
