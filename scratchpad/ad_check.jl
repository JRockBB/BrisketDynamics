# The state hand-off (segments.jl) and the full-covariance filter prior (FilterPrior, prior_quadratic).
#   julia --project=estimator estimator/test_segments.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
x = Float64.(m["truth"]["x"]); θ, φ, A, L = x[1:8], x[9:15], x[16], x[4]
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
keep = t .<= 16 * 3600
tt, uu = t[keep], u[keep]

using Printf, ForwardDiff
edges = [findlast(tt .<= 5 * 3600), findlast(tt .<= 10 * 3600)]
keep10 = tt .<= 10 * 3600 + 1e-9
t10, u10 = tt[keep10], uu[keep10]
f_whole(v) = (θv = vcat(θ[1:2], v[1], θ[4:8]); φv = vcat(φ[1:5], v[2], φ[7]); o = monolith_solve(θv, φv, u10, t10, t10; pit, cfg, A_meat=A, events...); [o.Tcore[end], o.Tpit[end]])
f_seg(v) = (θv = vcat(θ[1:2], v[1], θ[4:8]); φv = vcat(φ[1:5], v[2], φ[7]); o = solve_segments([θv, θv, θv], [φv, φv, φv], uu, tt, edges; pit, cfg, A_meat=A, events...); [o.Tcore[edges[2]], o.Tpit[edges[2]]])
v0 = [θ[3], φ[6]]
for (lab, f) in (("whole to 10 h", f_whole), ("segmented (5, 10 h)", f_seg))
    Jad = ForwardDiff.jacobian(f, v0)
    for rel in (1e-2, 1e-3)
        h = rel .* abs.(v0)
        Jfd = hcat([(f(v0 .+ h[j] .* (1:2 .== j)) .- f(v0 .- h[j] .* (1:2 .== j))) ./ (2h[j]) for j in 1:2]...)
        @printf("%-22s FD rel %.0e:  core/h_c AD %.3f FD %.3f | core/u_leak AD %.1f FD %.1f | pit/h_c AD %.4f FD %.4f | pit/u_leak AD %.1f FD %.1f\n",
                lab, rel, Jad[1, 1], Jfd[1, 1], Jad[1, 2], Jfd[1, 2], Jad[2, 1], Jfd[2, 1], Jad[2, 2], Jfd[2, 2])
    end
end
