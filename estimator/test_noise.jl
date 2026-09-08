# The AR node on residual processes where the answer is known: a
# synthetic AR(1) series at the real cook's length (115 points) with a
# weak prior recovers its sd and correlation within the posterior's
# width; a run split by a "lid lift" gives the same answer; a strong
# prior at the wrong values is overridden by the data; and the
# posterior narrows with the series.
#
# Run:  julia --project=estimator estimator/test_noise.jl
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using Random, Statistics, Test

function ar1(rng, n, σ, ρ)
    e = zeros(n); e[1] = σ * randn(rng)
    for k in 2:n
        e[k] = ρ * e[k - 1] + σ * sqrt(1 - ρ^2) * randn(rng)
    end
    return e
end

rng = MersenneTwister(5)
weak = NoisePrior(1.5, 1.5, 0.5, 0.4)            # a wide prior around nothing in particular

@testset "AR node: a synthetic AR(1) is recovered within its posterior width" begin
    hits = 0; ntrial = 12; errs_σ = Float64[]; errs_ρ = Float64[]
    for trial in 1:ntrial
        σ, ρ = 1.54, 0.83
        e = ar1(rng, 115, σ, ρ)
        p = ar1_posterior([e], weak)
        push!(errs_σ, (p.σ - σ) / p.σ_sd); push!(errs_ρ, (p.ρ - ρ) / p.ρ_sd)
        hits += abs(p.σ - σ) < 3 * p.σ_sd && abs(p.ρ - ρ) < 3 * p.ρ_sd
    end
    @test hits >= ntrial - 1
    @test abs(mean(errs_σ)) < 1.0 && abs(mean(errs_ρ)) < 1.0
    println("  standardized errors over $ntrial draws: σ mean $(round(mean(errs_σ); digits=2)) sd $(round(std(errs_σ); digits=2)), ρ mean $(round(mean(errs_ρ); digits=2)) sd $(round(std(errs_ρ); digits=2))")
end

@testset "AR node: runs split by a lid lift agree with the unsplit series" begin
    e = ar1(rng, 115, 1.54, 0.83)
    p1 = ar1_posterior([e], weak)
    p2 = ar1_posterior([e[1:60], e[61:115]], weak)      # one chain start more, one transition fewer
    @test abs(p1.σ - p2.σ) < 0.3 * p1.σ_sd
    @test abs(p1.ρ - p2.ρ) < 0.3 * p1.ρ_sd
end

@testset "AR node: a strong wrong prior is overridden by 115 points" begin
    e = ar1(rng, 115, 3.0, 0.6)
    wrong = NoisePrior(1.54, 0.25, 0.83, 0.06)       # the calibration's confidence, at the calibration's values
    p = ar1_posterior([e], wrong)
    @test 2.4 < p.σ < 3.6
    @test 0.45 < p.ρ < 0.75
end

@testset "AR node: the posterior narrows with the series" begin
    e = ar1(rng, 460, 1.54, 0.83)
    p_short = ar1_posterior([e[1:60]], weak)
    p_long = ar1_posterior([e], weak)
    @test p_long.σ_sd < 0.6 * p_short.σ_sd
    @test p_long.ρ_sd < 0.6 * p_short.ρ_sd
end
println("test_noise.jl: done")
