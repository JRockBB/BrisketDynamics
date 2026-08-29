# The monolith: one coupled solve, forward-mode derivatives

`src/monolith.jl` integrates the meat PDE, the pit ODE and the quasi-steady
pit humidity balance as a single stiff system (FBDF). The pit sees the
meat's instantaneous heat load and the meat sees the instantaneous pit
temperature, burn rate and vapor pressure: no exchange grid, no Picard
truncation, one solve, one derivative. The meat stencil is shared with the
Tesseract path (`meat_rhs_core!`), so the two pipelines integrate the same
physics and differ only in how the coupling is closed.

State `x = [T(nx) | W(nx) | C(nx) | F(nx) | B | S | T_pit | m_fuel | ignited]`,
parameters `p = [θ(8) | φ(7) | u_knots | t_wrap | t_pull]`.

```julia
out = monolith_solve(θ, φ, u_knots, u_tknots, tsave; pit=PitConfig(T_amb=300.0))
J   = monolith_jacobian(o -> vcat(o.Tpit, o.Tcore), θ, φ, u_knots, u_tknots, tsave; pit)
ref = picard_cook(θ, φ, u_knots, u_tknots, tsave; K=2, pit)   # the composed pipeline, in Julia
```

`monolith_solve` returns everything `coupled_cook` returns plus the pit
trajectory and `pv_base`; it accepts ForwardDiff duals, which is what
`monolith_jacobian` uses.

## Numbers (`scripts/bench_monolith.jl`, real-cook grid: 19 h, 115 save points, MAP-real parameters)

| | wall |
|---|---|
| monolith forward solve | 0.045 s |
| monolith Jacobian, 230 × 15, forward mode | 2.5 s |
| Picard K=2 (pit + meat, twice) | 0.32 s |
| continuous adjoint gradient of the same residuals (Tesseract path, measured 2026-08-27) | ~50 s |

End-to-end, same scripts, one import swapped:

| | two Tesseracts (K = 2 Picard) | monolith Tesseract |
|---|---|---|
| fullplan scan, 13 cook lengths, schedule + wrap + pull each | 4.5 h | 59 min |
| MPC demo, 4 cooks, hourly replans incl. the pull | ~2 h per replan under the disturbance | 43 min total |
| Gauss–Newton MAP from the prior medians | — (Adam + adjoint, hours) | 136 s |
| robust control: E[J] over 8 draws, 30 held-out evaluations | ~2 h | 30 min |
| the whole chain (fullplan → MPC → robust) | days | 2 h 42 min |

| Picard vs monolith | max ΔT_pit | max ΔT_core |
|---|---|---|
| K = 1 | 4.3 K | 6.7 K |
| K = 2 | 0.27 K | 0.11 K |
| K = 4, 8 | 0.04 K | 0.03 K |

The `picard_dT` diagnostic the old scripts reported was |x₂ − x₁| — the
K = 1 error. The K = 2 truncation actually carried by the composed pipeline is
~0.3 K on the real cook; the K ≥ 4 residual is the exchange-grid
interpolation.

## Verification

`test/test_monolith.jl`: Picard converges toward the monolith with K
(K = 8 within 2 K on the default-parameter fixture), and the forward
Jacobian matches central differences to ~1e-4 relative on D_eff, h_c, UA
and u_leak. The existing meat / pit / adjoint suites run unchanged on the
shared stencil.

## Gauss–Newton MAP and Fisher covariance (`scripts/run_gn_map.py`)

Levenberg–Marquardt over the 15 physics latents in the unconstrained
space, noise parameters fixed at the previous MAP's values, whitened AR(1)
residuals (222 of them) from the forward Jacobian. Cold start at the prior
medians:

| | F = ½‖w‖² − log p(z) | wall |
|---|---|---|
| prior medians | 5329 | — |
| after 17 LM iterations | 139.2 | 136 s |

A finite-difference Hessian of the potential at the optimum (measured on the
seven-latent fit that preceded the draft-convection latent) is indefinite
(min eigenvalue −20 against a max of 1.7·10⁵; 8 of 18 directions below a
10⁻⁴ floor): second differences of an adaptive solve are noise along the
ridges. The Fisher information (Jᵀ Σ⁻¹ J + prior) from the exact Jacobian is
the curvature used everywhere — the Laplace draws (`scripts/run_laplace.py`:
`jax.hessian` of the model with the solve replaced by its tangent model, so
the noise and physics × noise blocks are exact and only the solve's second
derivatives are dropped), the NUTS metric, the robust-control draws. `data/calibration/gn_map_real.json` carries the
optimum, the iteration history and the Fisher widths.

## What it leaves open

- Exact Hessians via nested duals (control and calibration).
- The acausal component version (pit, fuel bed, wrap, rest, ambient as
  ports; the slab as a generated component) with this file as its oracle.
