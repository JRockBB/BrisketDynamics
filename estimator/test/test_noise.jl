# The AR node on residual processes where the answer is known: a
# synthetic AR(1) series at the real cook's length (115 points) with a
# weak prior recovers its sd and correlation within the posterior's
# width; a run split by a "lid lift" gives the same answer; a strong
# prior at the wrong values is overridden by the data; and the
# posterior narrows with the series.
#
# Run:  julia --project=estimator estimator/test/test_noise.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using LinearAlgebra, Random, Statistics, Test
const E = BrisketEstimator

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
@testset "AR node: the coefficient's summary lies inside the stationary bound" begin
    # the moments of a truncated normal, in the tails too
    m0, s0 = E._truncated_moments(0.0, 1.0, -1.0, 1.0)
    @test abs(m0) < 1e-12 && abs(s0 - 0.5396) < 1e-3
    m1, s1 = E._truncated_moments(1.03, 1e-3, -1.0, 1.0)
    @test 0.9999 < m1 < 1.0 && 0 < s1 < 1e-4
    m2, s2 = E._truncated_moments(-1.03, 1e-3, -1.0, 1.0)
    @test m2 ≈ -m1 && s2 ≈ s1
    m3, s3 = E._truncated_moments(0.8, 0.05, -1.0, 1.0)
    @test abs(m3 - 0.8) < 1e-4 && abs(s3 - 0.05) < 1e-4
    @test_throws ArgumentError E._truncated_moments(0.0, 0.0, -1.0, 1.0)
    # an explosive residual (a trend the physics did not explain) whose least-squares slope exceeds one: the node's
    # Gaussian sits beyond the bound, the summary inside it with a finite width narrower than the Gaussian's
    rng_x = MersenneTwister(11)
    e = zeros(115); e[1] = 0.3
    for k in 2:115; e[k] = 1.03 * e[k - 1] + 0.05 * randn(rng_x); end
    @test dot(e[1:(end - 1)], e[2:end]) / dot(e[1:(end - 1)], e[1:(end - 1)]) > 1
    p = ar1_posterior([e], weak)
    @test mean(p.θ) > 1
    @test -1 < p.ρ < 1 && 0 < p.ρ_sd < std(p.θ) && isfinite(p.σ) && isfinite(p.σ_sd)
    @test 0 <= ar1_noise([e], 1, weak).ρ < 1
    # a stationary series: the summary is the node's own moments
    e8 = ar1(rng, 115, 1.5, 0.8)
    p8 = ar1_posterior([e8], weak)
    @test abs(p8.ρ - 0.8) < 3 * p8.ρ_sd
    @test abs(p8.ρ - mean(p8.θ)) < 1e-2 * std(p8.θ) && abs(p8.ρ_sd - std(p8.θ)) < 1e-2 * std(p8.θ)
end
println("test_noise.jl: done")
