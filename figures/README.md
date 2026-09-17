# Figures

One subdirectory per topic. Each figure is written by one script (Python scripts run from `app/`, Julia scripts from the repo root with `--project=estimator`); where a Julia script produces the record the Python script plots, both are named. `docs/results.md` is the record that discusses them.

## model/

The physics itself: the model diagrams, the bark model's forward validation, a forward cook, the animated calibrated cook.

| figure | written by | discussed in |
|---|---|---|
| `math_diagram.png` | `scripts/render_math_diagram.py` | `README.md`, `docs/model.md` |
| `composition_diagram.png` | `scripts/render_diagram.py` | `docs/writeup.md` |
| `bark_model.png` | `scripts/run_bark.py` | `README.md`, `docs/writeup.md`, `docs/submission.md` |
| `forward_cook.png` | `scripts/forward_demo.py` | the forward demo's own output |
| `cook_animation.gif` | `scripts/render_animation.py` | `README.md` |

## calibration/

The real-cook calibration and its posterior: the MAP fit, the NUTS and Laplace corners and predictives, the width comparison, the pooled cooks, the wrap step response, the water budget.

| figure | written by | discussed in |
|---|---|---|
| `cooklog_fit_map.png` | `scripts/run_gn_map.py` | `README.md`, `docs/writeup.md`, `docs/submission.md` |
| `cooklog_replay_defaults.png` | `scripts/replay_cooklog.py` | the replay's own output (the cook log at the prior medians) |
| `corner_nuts_real.png` | `scripts/merge_nuts_mono.py` | `README.md`, `docs/writeup.md`, `docs/results.md` |
| `posterior_predictive_nuts_real.png` | `scripts/merge_nuts_mono.py` | `README.md`, `docs/writeup.md` |
| `posterior_widths.png` | `scripts/merge_nuts_mono.py`, `scripts/compare_approx.py` | `README.md`, `docs/writeup.md` |
| `corner_laplace_real.png` | `scripts/run_laplace.py` | `docs/submission.md` |
| `posterior_predictive_laplace_real.png` | `scripts/run_laplace.py` | the Laplace run's own output |
| `tmb_ignition_marginal.png` | `scripts/run_tmb_ignition.py` | the TMB ignition run's own output |
| `pool_real.png`, `pool_wsm.png`, `pool_keg.png` | `scripts/run_pool_real.py` | `docs/pooling.md`, `docs/results.md` |
| `wrap_fit_erickson.png` | `scripts/run_wrap_fit.py` | the wrap fit's own output |
| `water_budget_scan.png` | `scripts/run_water_budget.py` | `docs/results.md` |

## validation/

Transfer claims on other cooks with zero fitted parameters.

| figure | written by | discussed in |
|---|---|---|
| `oven_roast_cet2015.png` | `scripts/run_oven_roast.py` | `docs/results.md` |
| `sous_vide_baldwin.png` | `scripts/run_sous_vide.py` | `docs/results.md` |

## control/

Open-loop planning and the app's MPC.

| figure | written by | discussed in |
|---|---|---|
| `fullplan.png` | `scripts/run_fullplan.py` | `README.md`, `docs/writeup.md`, `docs/submission.md` |
| `fullplan_mock1d.png` | `scripts/run_fullplan.py --out fullplan_result_mock1d.json` (the full plan on the mock brisket, the MPC's seed in the pilot comparison) | `docs/results.md` |
| `duration_opt.png` | `scripts/run_duration_opt.py` | `docs/writeup.md` |
| `start_time.png`, `start_time_compare.png`, `start_time_robust.png` | `scripts/run_start_time.py` | `docs/writeup.md` |
| `control_before_after.png` | `scripts/run_control.py` | the control run's own output |
| `control_robust.png` | `scripts/run_robust_control.py` | `README.md`, `docs/writeup.md`, `docs/submission.md` |
| `mpc_demo.png` | `scripts/run_mpc.py` | `README.md`, `docs/writeup.md`, `docs/submission.md` |
| `pilot_http.png` | `scripts/plot_pilot.py` from the journal `scripts/run_pilot.py` writes | `docs/results.md` |

## estimator/

The online estimator and its pilot: the hourly streaming refits on mock plants, the misfit check, the noise identification, the closed loop on the real-cook mock, the pilot against the MPC.

| figure | written by | discussed in |
|---|---|---|
| `stream_mock_1d.png`, `stream_mock_2d.png` | `estimator/python/plot_stream.py` from `estimator/scripts/mock_stream.jl` | `docs/results.md` |
| `stream_mock_1d_phys2d.png`, `stream_mock_2d_phys2d.png` | `estimator/python/plot_stream.py` from `estimator/scripts/mock_stream_2d.jl` (the 2-D engine reading each plant) | `docs/results.md` |
| `stream_mock_1d_shape.png`, `stream_mock_2d_shape.png` | `estimator/python/plot_stream.py` from `estimator/scripts/mock_stream_shape.jl` | `docs/results.md` |
| `misfit.png`, `misfit_phys2d.png` | `estimator/python/plot_misfit.py` from `estimator/scripts/misfit_stream.jl` | `docs/results.md` |
| `noise_mock_1d_alt.png`, `noise_mock_1d_alt_tightprior.png` | `estimator/python/plot_noise.py` from `estimator/scripts/mock_stream_noise.jl` | `docs/results.md` |
| `realcook_stream.png` | `estimator/python/plot_realcook_stream.py` from `estimator/scripts/mock_stream_realcook.jl` | `docs/results.md` |
| `closed_loop.png` | `estimator/python/plot_closed_loop.py` from `estimator/scripts/mock_closed_loop.jl` | `docs/results.md` |
| `closed_loop_realcook.png` | `estimator/python/plot_closed_loop_realcook.py` from `estimator/scripts/mock_closed_loop_realcook.jl` and `estimator/closed_loop_trace.jl` | `docs/results.md` |
| `pilot_comparison.png` | `estimator/python/plot_pilot_comparison.py` from `estimator/scripts/pilot_comparison_trace.jl` (the pilot and each MPC variant through one plant) | `docs/results.md` |
| `coverage_realcook.png` | `estimator/python/plot_coverage.py` from `estimator/scripts/coverage_realcook.jl` (21 cooks with truths drawn from the pilot's prior, the Gaussian posterior's standardized errors pooled) | `README.md`, `docs/results.md` |
| `coverage_realcook_gauss9.png`, `coverage_realcook_ridge.png` | `estimator/python/plot_coverage.py "coverage_ridge_*.json" err_sd _gauss9` and `... err_sd_ridge _ridge` (the same nine cooks under the Gaussian and under the chain on the exact posterior) | `docs/results.md` |
| `coverage_realcook_nohum.png`, `coverage_realcook_truthnoise30.png` | `estimator/python/plot_coverage.py "coverage_nohum_*.json" err_sd _nohum` and `"coverage_truthnoise_*.json" err_sd _truthnoise30` (the diagnostic variants: without the humidity channel; every channel's noise fixed at the truth, thirty cooks) | `docs/results.md` |
| `coverage_realcook_driftbatch.png`, `coverage_realcook_driftfilter.png`, `coverage_realcook_driftsmoother.png` | `estimator/python/plot_coverage.py "coverage_drift_*.json" err_sd _driftbatch`, `... err_sd_filter _driftfilter`, `... err_sd_smoother _driftsmoother` from `estimator/scripts/coverage_drift.jl` (nine cooks on the DRIFTING plant of `estimator/src/drift.jl`: the leak and the surface coefficients on random walks, the add's charge jumping at the add; the same cooks streamed by the batch estimator, the filter and the smoother, the standardized errors against the truth in force at each hour; the filter's headline includes its a0, a handed-off state, which the results entry excludes) |
| `coverage_realcook_driftgraphfilter.png` | `estimator/python/plot_coverage.py "coverage_filter_*.json" err_sd_graphfilter _driftgraphfilter` from the fresh round of `estimator/python/bench_state_noise.py` (nine drifting cooks on seeds the state's process noise was never tuned on, `estimator/scripts/coverage_filter.jl` seeds 10 to 18, run by the filter on the graph, `estimator/src/sequential.jl`: the belief over the plant's state, the latents and the instruments' discrepancies, updated at every arrival of a sample) |
| `filter_belief_drift.png` | `estimator/python/plot_filter_belief.py` from `estimator/scripts/mock_filter_drift.jl` (the filter on the graph over the drifting mock's whole cook: the pit, surface and centre temperatures as the filter believes them at every update, mean and ±2 sd, against the plant and the probes; below, the latents that move on the plant as the filter follows them, with the truth in force; at the bottom every channel's noise level with the spans where a sensor was distrusted) |
| `filter_belief_fault.png` | `estimator/python/plot_filter_belief.py filter_drift_fault.json filter_belief_fault` from `estimator/scripts/mock_filter_drift.jl 16 prior_next_cook.json 8` (the same cook with the core probe stuck at its last reading from hour 8: the dead-reading rule flags the channel within five samples, its span is shaded, and the belief of the centre keeps heating with the physics and the other channels instead of following the frozen probe) |
| `filter_consistency_drift.png` | `estimator/python/plot_filter_consistency.py` from the fresh round of `estimator/python/bench_state_noise.py` (the filter's consistency tests at every update, pooled over the nine fresh-seed drifting cooks: the normalized innovation squared against its chi-squared band, the normalized estimation error squared of the latents and of the pit, centre and surface jointly against the truth, each channel's innovation mean and lag-one autocorrelation, and the belief's state by group against the plant's, the reading the state's process noise in `data/calibration/state_noise.json` was earned from) |
| `closed_loop_realcook_drift_batch.png`, `closed_loop_realcook_drift_smoother.png`, `..._batch_s2.png`, `..._smoother_s2.png`, `..._batch_s3.png`, `..._smoother_s3.png` | `estimator/python/plot_closed_loop_realcook.py <record>` (the trace step these flights used, `closed_loop_trace.jl`, is retired: the record now carries the belief at every step) from `estimator/scripts/mock_closed_loop_realcook.jl prior_next_cook.json <record> 0 0 300 <batch|smoother> <seed>` (the pilot bench on three drifting plants, both estimators under one planner: the cook as it happened with the pilot's belief of the pit, surface and core at every hourly check carried to the next, the decisions, the outcome with its predicted uncertainty) |
| `closed_loop_realcook_drift_batch_prior10.png`, `closed_loop_realcook_drift_smoother_replay.png`, `closed_loop_realcook_drift_smoother_noguard.png` | the same, the seed-1 flights that diagnosed the plan (the batch pilot under the ten-sd prior with the prior-charged pick; the smoother pilot with the plan re-cooking the history; the smoother pilot with the history but the deferred wrap): `docs/results.md`, the drifting-plant pilot entry |

## slab2d/

The 2-D axisymmetric line: the planned cook re-simulated in 2-D, its fields, salt and cook animations.

| figure | written by | discussed in |
|---|---|---|
| `slab2d_plan.png` | `scripts/plot_slab2d.py` | `README.md`, `docs/review.md` |
| `slab2d_fields.png`, `slab2d_salt_bark.png` | `scripts/plot_slab2d.py` | the 2-D figure set's own output |
| `cook2d.gif`, `salt2d.gif` | `scripts/plot_slab2d.py` | the 2-D figure set's own output |
