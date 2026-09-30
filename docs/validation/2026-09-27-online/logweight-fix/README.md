# Retained log weights: current implementation

Backend `84201766a16f`, physics `4e4a6800e4a2`. Only `particle_online.jl` changed from the baseline
backend `411c33d5521d`; the original file is retained in `../baseline-source/`. The
[production validation metadata](production_validation.json) records hashes of all backend source
files and the eight focused test files. The four cook comparisons in the parent record were run
**before this correction** and have not been repeated. Their overall failure remains unresolved.

## Correction

Sequential updates previously recovered log weights with `log.(posterior.weights)`. Exponentiating
a finite log weight can round to zero, after which that round trip makes the hypothesis permanently
impossible. Later observations can favor precisely that hypothesis. The smoother now retains
normalized log weights between arrivals and includes them in checkpoints and rollback. A full-history
refresh resets them from its equal weights. Display weights may still underflow, but continuation does
not use them to reconstruct log weights or prune trajectories.

The analytic recovery test constructs two temperature hypotheses. One reading drives the disfavored
weights to zero; a second reading favors those same hypotheses. The production incremental likelihood
recovers their mass and agrees with the closed-form Gaussian likelihood with a shared sensor offset.
This is a numerical correctness result, not a physical calibration result or a performance improvement.

## Completed checks

The final [regression log](regression.log) records **266 passing assertions** across eight files:

| Test | Passed assertions |
| --- | ---: |
| Nonlinear posterior sampling | 32 |
| SMC analytic and measurement oracles | 38 |
| Full-refit sampled transactions | 26 |
| Incremental likelihood, updates, checkpoints and batching | 48 |
| Offline sampled entry point | 10 |
| Sampled histories, planning and forecasts | 63 |
| Log-weight mathematics, continuation, rollback and recovery | 40 |
| Retained-posterior forecast and checkpoint isolation | 9 |

These counts overlap the baseline regression and must not be added to its 708 assertions. No full
repository run or new real-cook calibration study was performed for this correction.

The final run used Julia 1.11.9, four Julia threads and one BLAS thread in a warm worker. That worker
initially loaded the corrected particle source in a fresh module. Two temporary forecast-related method
overrides were subsequently restored from the unchanged production files before the final regression.
The [driver](production_driver.jl) checks byte identity between the loaded particle source and the
promoted production file, and verifies unchanged backend and physics fingerprints throughout the run.
The [restoration driver](restore_forecast_driver.jl) records the original methods. This final run was
not a separate fresh-process load of the promoted module. Drivers retain their original workspace paths.

## Development failures retained

The [complete development log](development.log) includes the following resolved test/driver mistakes:

- The initial two-temperature oracle omitted the decoder's `1e-8` positive-state offset. Its displayed
  weights recovered correctly, but the tight log-odds assertion failed. The expected physical means
  were corrected; the relative tolerance stayed `1e-12`. The [initial oracle](recovery_initial_oracle.jl)
  is retained.
- A suspected retained-forecast alias was a false positive. The transport already deep-copies the
  graph. The initial test wrongly expected that copy to change after an accepted reading; actual
  before/after forecasts already matched. Its [initial assertions](retained_forecast_initial_test.jl)
  and failures are retained. An initial driver also failed to unwrap `include`'s `LoadError`. No
  forecast or transport source change was promoted. The final regression checks the live graph's
  change and the saved posterior's isolation.
- Promoting the particle source during an intermediate test changed the backend fingerprint between
  checkpoint creation and restoration. The compatibility guard correctly rejected that checkpoint.
  The final run occurred after promotion, with unchanged source throughout, and passed.

The worker was stopped after the final regression. The log-weight correction does not address the
94-minute failed refresh in the baseline cook comparison. The online implementation remains unsuitable
for a live-use qualification on the present evidence.
