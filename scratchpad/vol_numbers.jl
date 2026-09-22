using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "BrisketEstimator.jl"))
using .BrisketEstimator, JSON, Random, Statistics, Printf
const E = BrisketEstimator
function unit_ar1(rng, n, ρ); a = zeros(n); a[1] = randn(rng); for k in 2:n; a[k] = ρ*a[k-1] + sqrt(1-ρ^2)*randn(rng); end; a; end
rng = MersenneTwister(9); n, ρ, σ0 = 600, 0.9, 1.5
lt = [200 <= k < 320 ? 3.0 : 1.0 for k in 1:n]; e = σ0 .* lt .* unit_ar1(rng, n, ρ)
r = volatility_level(e, ρ, σ0; w=0.05); t0 = time(); r = volatility_level(e, ρ, σ0; w=0.05); w1 = time() - t0
@printf("synthetic: inside median %.2f (true 3), outside %.2f (true 1), sd(z) %.2f; wall %.2f s for %d samples\n", median(r.level[230:300]), median(r.level[vcat(30:170, 360:590)]), median(r.z_sd), w1, n)
ch = E.PitProbe(σ0, ρ, trues(n); data=e)
@printf("whitened power: flat %.2f (inside %.2f), levelled %.2f, true level %.2f\n", mean((E.whitening(ch)*e).^2), mean((E.whitening(ch)*e)[200:320].^2), mean((E.whitening(Levelled(ch, r.level))*e).^2), mean((E.whitening(Levelled(ch, lt))*e).^2))
m = JSON.parsefile(joinpath(@__DIR__, "..", "data", "mock", "mock_realcook.json"))
t = Float64.(m["t"]); dt = t[2]-t[1]; nc = m["noise"]
ep = Float64.(m["readings"]["pit"]) .- Float64.(m["clean"]["pit"]); mask = Bool.(m["masks"]["pit"]); ρp = exp(-dt/nc["pit_tau"])
runs = E.contiguous_runs(mask; maxgap=E.run_maxgap(mask)); t0 = time()
L = level_on_grid(mask, [volatility_level(ep[run], ρp, Float64(nc["pit_sd"]); w=0.05).level for run in runs]); w2 = time() - t0
ta = Float64(m["events"]["t_add"])
true_level = [tk > ta ? 1 + nc["pit_add_bump"]*exp(-(tk-ta)/nc["add_tau"]) : 1.0 for tk in t]
for (lo, hi, lab) in [(ta-5400, ta-600, "before"), (ta+120, ta+900, "0-15 min after"), (ta+900, ta+2700, "15-45 min after"), (ta+2700, ta+7200, "45-120 min after")]
    s = (t .> lo) .& (t .< hi) .& mask
    @printf("real-cook pit level %-16s inferred median %.2f  true median %.2f\n", lab, median(L[s]), median(true_level[s]))
end
@printf("real-cook pit: %d runs, %d samples, wall %.1f s\n", length(runs), count(mask), w2)
