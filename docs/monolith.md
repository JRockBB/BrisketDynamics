# The monolith: one coupled solve, forward-mode derivatives

`src/monolith.jl` integrates the meat PDE, the pit ODE and the quasi-steady
pit humidity balance as a single stiff system (FBDF). The pit sees the
meat's instantaneous heat load and the meat sees the instantaneous pit
temperature, burn rate and vapor pressure: no exchange grid, no Picard
truncation, one solve, one derivative. The meat stencil is shared with the
Tesseract path (`meat_rhs_core!`), so the two pipelines integrate the same
physics and differ only in how the coupling is closed.

State `x = [T(nx) | W(nx) | C(nx) | F(nx) | B | S | T_pit | m_fuel | ignited]`,
parameters `p = [θ(7) | φ(7) | u_knots | t_wrap | t_pull]`.

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
| monolith Jacobian, 230 × 14, forward mode | 2.5 s |
| Picard K=2 (pit + meat, twice) | 0.32 s |
| continuous adjoint gradient of the same residuals (Tesseract path, measured 2026-08-27) | ~50 s |

| Picard vs monolith | max ΔT_pit | max ΔT_core |
|---|---|---|
| K = 1 | 4.3 K | 6.7 K |
| K = 2 | 0.27 K | 0.11 K |
| K = 4, 8 | 0.04 K | 0.03 K |

The `picard_dT` diagnostic the scripts report is |x₂ − x₁| — the K = 1
error. The K = 2 truncation actually carried by the composed pipeline is
~0.3 K on the real cook; the K ≥ 4 residual is the exchange-grid
interpolation.

## Verification

`test/test_monolith.jl`: Picard converges toward the monolith with K
(K = 8 within 2 K on the default-parameter fixture), and the forward
Jacobian matches central differences to ~1e-4 relative on D_eff, h_c, UA
and u_leak. The existing meat / pit / adjoint suites run unchanged on the
shared stencil.

## What it unlocks

- Gauss–Newton / Levenberg–Marquardt MAP from the residual Jacobian; the
  Fisher information Jᵀ Σ⁻¹ J as the Laplace covariance (positive
  semi-definite, no finite differences, no eigenvalue floor).
- log-density gradients Jᵀ Σ⁻¹ r for HMC at the cost of one Jacobian.
- Control Jacobians w.r.t. the damper knots, wrap and pull by the same
  route (`p` carries them), and exact Hessians via nested duals.
- The acausal component version (pit, fuel bed, wrap, rest, ambient as
  ports; the slab as a generated component) with this file as its oracle.
