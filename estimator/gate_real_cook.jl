# Acceptance gate for the RxInfer estimator: reproduce the Gauss-Newton MAP
# and the Fisher covariance of the real cook (scripts/run_gn_map.py) from
# the same ingredients (estimator/export_gate_data.py) by message passing.
#
# Run:  julia --project=estimator estimator/gate_real_cook.jl
# Writes data/calibration/estimator_gate_real_result.json.
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using JSON
using Printf

REPO = normpath(joinpath(@__DIR__, ".."))
d = load_gate(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
println("gate data: $(length(d.t)) grid points, $(size(d.L, 1)) whitened residuals, ",
        "$(length(d.names)) latents; reference F(MAP) = $(round(d.F_map; digits=4))")

# sanity: the objective at the reference MAP from the Julia solve should match
# the Python value (same solver, same tolerances; last-bit jitter only)
t0 = time()
μ_map = physics_mean(d.z_map, d)
F_here = objective_value(d.z_map, μ_map, d)
@printf("F(z_map) recomputed here: %.4f (Python %.4f, diff %.2e)  [%.1f s incl. compile]\n",
        F_here, d.F_map, F_here - d.F_map, time() - t0)
flush(stdout)

t0 = time()
res = solve_map(d; c0=d.z0, iters=30, tol=1e-4, bp_iters=20, verbose=true)
wall = time() - t0
out = compare_gate(res, d; wall)
dst = joinpath(REPO, "data", "calibration", "estimator_gate_real_result.json")
open(dst, "w") do io
    JSON.print(io, out, 1)
end
println("wrote ", dst)
