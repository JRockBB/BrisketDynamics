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
| `stream_mock_1d.png`, `stream_mock_2d.png` | `estimator/plot_stream.py` from `estimator/mock_stream.jl` | `docs/results.md` |
| `stream_mock_1d_phys2d.png`, `stream_mock_2d_phys2d.png` | `estimator/plot_stream.py` from `estimator/mock_stream_2d.jl` (the 2-D engine reading each plant) | `docs/results.md` |
| `stream_mock_1d_shape.png`, `stream_mock_2d_shape.png` | `estimator/plot_stream.py` from `estimator/mock_stream_shape.jl` | `docs/results.md` |
| `misfit.png`, `misfit_phys2d.png` | `estimator/plot_misfit.py` from `estimator/misfit_stream.jl` | `docs/results.md` |
| `noise_mock_1d_alt.png`, `noise_mock_1d_alt_tightprior.png` | `estimator/plot_noise.py` from `estimator/mock_stream_noise.jl` | `docs/results.md` |
| `realcook_stream.png` | `estimator/plot_realcook_stream.py` from `estimator/mock_stream_realcook.jl` | `docs/results.md` |
| `closed_loop.png` | `estimator/plot_closed_loop.py` from `estimator/mock_closed_loop.jl` | `docs/results.md` |
| `closed_loop_realcook.png` | `estimator/plot_closed_loop_realcook.py` from `estimator/mock_closed_loop_realcook.jl` and `estimator/closed_loop_trace.jl` | `docs/results.md` |
| `pilot_comparison.png` | `estimator/plot_pilot_comparison.py` from `estimator/pilot_comparison_trace.jl` (the pilot and each MPC variant through one plant) | `docs/results.md` |

## slab2d/

The 2-D axisymmetric line: the planned cook re-simulated in 2-D, its fields, salt and cook animations.

| figure | written by | discussed in |
|---|---|---|
| `slab2d_plan.png` | `scripts/plot_slab2d.py` | `README.md`, `docs/review.md` |
| `slab2d_fields.png`, `slab2d_salt_bark.png` | `scripts/plot_slab2d.py` | the 2-D figure set's own output |
| `cook2d.gif`, `salt2d.gif` | `scripts/plot_slab2d.py` | the 2-D figure set's own output |
