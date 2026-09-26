using Pkg; Pkg.activate("estimator")
include(joinpath(@__DIR__, "..", "estimator", "src", "BrisketEstimator.jl")); using .BrisketEstimator, BrisketDynamics, JSON, Random, Printf
m = JSON.parsefile(joinpath(@__DIR__, "..", "data", "mock", "mock_1d.json")); gate = JSON.parsefile(joinpath(@__DIR__, "..", "data", "calibration", "prior_next_cook.json"))
names = vcat(String.(gate["names"]), ["A_meat", "core_depth_fraction"]); idx = Dict(n => i for (i, n) in enumerate(names))
prior_phys = read_sites(gate["prior"])
x = Float64.(m["truth"]["x"]); z0 = vcat([to_z(v, p) for (v, p) in zip(x[1:15], prior_phys)], log(x[16]), to_z(x[17], probe_depth_site()))
dd = DriftDraw(MersenneTwister(2001), idx)
for nm in ("u_leak", "k_e", "h_c"); i = idx[nm]; @printf("%-7s z0 %+.3f  cumulative by hour: %s\n", nm, z0[i], join([@sprintf("%+.2f", sum(dd.inc[i][1:h])) for h in 1:18], " ")); end
for nm in ("r_max", "dH_eff"); i = idx[nm]; @printf("%-7s jump %+.3f (x%.2f); prior sd %.2f\n", nm, dd.jump[i], exp(dd.jump[i]), prior_phys[i].p2); end
