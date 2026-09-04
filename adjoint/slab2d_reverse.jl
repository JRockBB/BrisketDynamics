# M2: Enzyme reverse through the 2-D axisymmetric ADI solver.
#
# meat_solve_2d is already the adjoint-friendly structure the M1 port
# validated — fixed-step, hand-coded Thomas sweeps, pure loops — so here
# Enzyme reverses the WHOLE solver as-is (engine closure, linearized-
# implicit face evaporation, quality states riding along), no rewrite.
# Gradient of the retained-water fraction wrt θ2 validated against
# ForwardDiff on the same solver, with timings.
#
# Run:  julia --project=adjoint adjoint/slab2d_reverse.jl

using Pkg
Pkg.activate(@__DIR__)
using Enzyme
using ForwardDiff
using Printf
using BrisketDynamics

const hr = 3600.0
const TKNOTS = collect(0.0:600.0:6hr)
const TPIT = fill(383.15, length(TKNOTS))
const TSAVE = [0.0, 3hr, 6hr]
const CFG = BrisketDynamics.with_config(MeatConfig(); n_shrink=0.0)

function loss(v::AbstractVector)
    return meat_solve_2d(v, TPIT, TKNOTS, TSAVE;
                         R=0.05, Hz=0.06, nr=16, nz=14, cfg=CFG,
                         h=30.0).w_retained
end

θ2 = [3.8e-9, 0.0018, 10.0, 1.0, 5.6e-4, 0.20]

L0 = loss(θ2)  # compile + value
@printf("w_retained = %.6f\n", L0)
t0 = time(); loss(θ2); tp = time() - t0

g_enz = zero(θ2)
Enzyme.autodiff(Enzyme.Reverse, loss, Enzyme.Active, Enzyme.Duplicated(θ2, g_enz))
t0 = time()
let g = zero(θ2)
    Enzyme.autodiff(Enzyme.Reverse, loss, Enzyme.Active, Enzyme.Duplicated(θ2, g))
end
tr = time() - t0

g_fwd = ForwardDiff.gradient(loss, θ2)
t0 = time(); ForwardDiff.gradient(loss, θ2); tf = time() - t0

rel = maximum(abs.(g_enz .- g_fwd) ./ (abs.(g_fwd) .+ 1e-30))
println("grad (Enzyme):      ", g_enz)
println("grad (ForwardDiff): ", g_fwd)
@printf("max rel |Enzyme - ForwardDiff| = %.3e\n", rel)

# one FD spot check on k_m
h_fd = 0.02 * θ2[2]
fd = (loss([θ2[1], θ2[2] + h_fd, θ2[3:end]...]) -
      loss([θ2[1], θ2[2] - h_fd, θ2[3:end]...])) / (2h_fd)
@printf("d/dk_m: Enzyme %.6e  FD %.6e  rel %.2e\n",
        g_enz[2], fd, abs(g_enz[2] - fd) / (abs(fd) + 1e-30))

@printf("timings: primal %.2fs  reverse %.2fs (%.1fx primal)  forward-6 %.2fs\n",
        tp, tr, tr / tp, tf)
