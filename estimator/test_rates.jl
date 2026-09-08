# Channels at their own rates on one physics grid: the gap-aware
# whitening turns a subsampled AR(1) process into white unit residuals,
# a channel sampled every grid step is whitened as before, a long gap
# breaks a run while a channel's regular spacing does not, and the AR
# node estimated at the channel's spacing returns the grid-step
# correlation.
#
# Run:  julia --project=estimator estimator/test_rates.jl
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
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
rng = MersenneTwister(4)
n = 60000; σ, ρ = 1.5, 0.9         # a 2-minute grid, ρ 0.9 per step; long enough that the power's sd is 0.01

@testset "gap-aware whitening: a channel every 5 steps is white at unit power" begin
    e = ar1(rng, n, σ, ρ)
    mask = falses(n); mask[1:5:n] .= true
    @test E.channel_gap(mask) == 5
    Lw = E.whitening_rows(mask, σ, ρ)
    @test size(Lw, 1) == count(mask)                # one run: no gap exceeds three times the spacing
    w = Lw * e
    @test abs(mean(w .^ 2) - 1) < 0.05
    @test abs(cor(w[1:end-1], w[2:end])) < 0.03
    # the same as whitening the subsampled series itself with its per-sample correlation ρ^5
    sub = e[mask]
    w_manual = E.whitening_rows(trues(length(sub)), σ, ρ^5) * sub
    @test maximum(abs.(w_manual .- w)) < 1e-10
end

@testset "a full mask whitens exactly as before" begin
    e = ar1(rng, 500, σ, ρ)
    mask = trues(500)
    @test E.channel_gap(mask) == 1
    Lw = E.whitening_rows(mask, σ, ρ)
    innov = σ * sqrt(1 - ρ^2)
    @test Lw[1, 1] ≈ 1 / σ && Lw[2, 2] ≈ 1 / innov && Lw[2, 1] ≈ -ρ / innov
    @test abs(mean((Lw * e) .^ 2) - 1) < 0.1
end

@testset "a long gap breaks the run, a regular spacing does not" begin
    mask = falses(200); mask[1:2:100] .= true; mask[151:2:200] .= true
    runs = E.contiguous_runs(mask; maxgap=E.run_maxgap(mask))
    @test length(runs) == 2 && length(runs[1]) == 50 && length(runs[2]) == 25
    @test E.contiguous_runs(mask) |> length == 75    # the strict rule: every sample its own run
end

@testset "the AR node at a channel's spacing returns the grid-step correlation" begin
    e = ar1(rng, 3000, σ, ρ)
    g = 5
    sub = e[1:g:end]                                 # the channel's own regular series, ρ^5 = 0.59 per sample
    weak = NoisePrior(1.5, 1.5, 0.5, 0.4)
    p = ar1_posterior([sub], weak)
    ρ_step = p.ρ^(1 / g)
    @test abs(p.ρ - ρ^g) < 3 * p.ρ_sd
    @test abs(ρ_step - ρ) < 0.03
    @test abs(p.σ - σ) < 3 * p.σ_sd
end
println("test_rates.jl: done")
