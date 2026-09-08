# The noise level as a process, where the answer is known: a unit AR(1)
# scaled by a level that triples for a stretch is recovered by the
# volatility smoother (the level inside the stretch, the base outside);
# the levelled whitening turns that process into unit white residuals
# while the flat whitening does not; and the level on the real-cook
# mock's pit residual at the truth rises after the fuel add.
#
# Run:  julia --project=estimator estimator/test_volatility.jl
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Statistics, Test
const E = BrisketEstimator

function unit_ar1(rng, n, ρ)
    a = zeros(n); a[1] = randn(rng)
    for k in 2:n
        a[k] = ρ * a[k - 1] + sqrt(1 - ρ^2) * randn(rng)
    end
    return a
end
rng = MersenneTwister(9)
n, ρ, σ0 = 600, 0.9, 1.5
level_true = [200 <= k < 320 ? 3.0 : 1.0 for k in 1:n]
e = σ0 .* level_true .* unit_ar1(rng, n, ρ)

@testset "the volatility smoother recovers a stretch of tripled noise" begin
    r = volatility_level(e, ρ, σ0; w=0.05)
    @test length(r.level) == n
    inside = 230:300; outside = vcat(30:170, 360:590)
    @test 2.0 < median(r.level[inside]) < 4.5
    @test 0.6 < median(r.level[outside]) < 1.5
    @test median(r.level[inside]) > 2 * median(r.level[outside])
end

@testset "the levelled whitening is exact for the scaled process" begin
    mask = trues(n)
    ch = PitProbe(σ0, ρ, mask; data=e)
    r = volatility_level(e, ρ, σ0; w=0.05)
    lv = Levelled(ch, r.level)
    w_flat = E.whitening(ch) * e
    w_lev = E.whitening(lv) * e
    w_true = E.whitening(Levelled(ch, level_true)) * e
    @test abs(mean(w_true .^ 2) - 1) < 0.1                       # the true level whitens exactly
    @test abs(mean(w_lev .^ 2) - 1) < 0.25                        # the inferred one nearly
    @test mean(w_flat[200:320] .^ 2) > 4                           # the flat one is blind to the stretch
    @test lv.σ == σ0 && lv.mask == mask                            # the wrapper forwards the channel's fields
    @test E.channel_label(lv) == "pit, levelled"
    lv2 = E.restrict(lv, [k <= 300 for k in 1:n])
    @test count(lv2.mask) == 300 && lv2.level === r.level
end

@testset "the real-cook mock's pit level rises after the fuel add" begin
    REPO = normpath(joinpath(@__DIR__, ".."))
    m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
    t = Float64.(m["t"]); dt = t[2] - t[1]
    nc = m["noise"]
    e_pit = Float64.(m["readings"]["pit"]) .- Float64.(m["clean"]["pit"])
    mask = Bool.(m["masks"]["pit"])
    ρ = exp(-dt / nc["pit_tau"])
    runs = E.contiguous_runs(mask; maxgap=E.run_maxgap(mask))
    levels = [volatility_level(e_pit[run], ρ, Float64(nc["pit_sd"]); w=0.05).level for run in runs]
    L = level_on_grid(mask, levels)
    t_add = Float64(m["events"]["t_add"])
    after = (t .> t_add + 120) .& (t .< t_add + 900) .& mask
    before = (t .> t_add - 5400) .& (t .< t_add - 600) .& mask
    @test median(L[after]) > 1.6 * median(L[before])
    @test 0.5 < median(L[before]) < 1.6
end
println("test_volatility.jl: done")
