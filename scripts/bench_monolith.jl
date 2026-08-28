# Monolith vs the composed Picard pipeline on the real-cook grid at the MAP:
# forward cost, forward-mode Jacobian cost, and the K-truncation error.
#
#   julia --project=. scripts/bench_monolith.jl
#
using BrisketDynamics
using Printf

# map_real.json (2026-08-27): θ = [D_eff, k_m, h_c, L, k_mult, k_e, a_myosin],
# φ = [C_th, UA, r_max, ΔH_eff, u_leak, k_spread, a0]
θ = [8.219e-9, 1.298e-3, 6.265, 0.01717, 1.088, 3.208e-4, 0.0971]
φ = [6.558e4, 16.81, 2.842e-4, 1.443e7, 0.2848, 2.141e-4, 0.9154]
t_f = 19 * 3600.0
tsave = collect(0.0:600.0:t_f)                     # 115 points, like the cook log
u = 0.15 .+ 0.04 .* sin.(range(0, 6π; length=length(tsave)))  # a wandering damper
pit = PitConfig(; T_amb=300.0, Tpit0=362.0)
f = o -> vcat(o.Tpit, o.Tcore)

warm(fn) = (fn(); @elapsed fn())
t_fwd = warm(() -> monolith_solve(θ, φ, u, tsave, tsave; pit))
t_jac = warm(() -> monolith_jacobian(f, θ, φ, u, tsave, tsave; pit))
t_pic = warm(() -> picard_cook(θ, φ, u, tsave, tsave; K=2, pit))
o = monolith_solve(θ, φ, u, tsave, tsave; pit)
J = monolith_jacobian(f, θ, φ, u, tsave, tsave; pit)

@printf "grid: %d save points, %d damper knots, %d parameters\n" length(tsave) length(u) size(J, 2)
@printf "monolith forward      %.3f s\n" t_fwd
@printf "monolith Jacobian     %.2f s  (%d x %d, forward mode)\n" t_jac size(J)...
@printf "Picard K=2 (2 solves) %.3f s\n" t_pic
for K in (1, 2, 4, 8)
    pc = picard_cook(θ, φ, u, tsave, tsave; K, pit)
    @printf "|Picard K=%d - monolith|  pit %.2f K   core %.2f K\n" K maximum(
        abs.(pc.pit.Tpit .- o.Tpit)
    ) maximum(abs.(pc.meat.Tcore .- o.Tcore))
end
@printf "core: max %.1f C, end %.1f C; pit mean %.1f C; C_center %.3f\n" (
    maximum(o.Tcore) - 273.15
) (o.Tcore[end] - 273.15) (sum(o.Tpit) / length(o.Tpit) - 273.15) o.C_field_final[1]
