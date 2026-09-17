# The misfit statistics on a linear-Gaussian model where the answer is
# known: white unit residuals through a least-squares fit have
# E[Σ r²] = n − tr(H) exactly and a lag-1 autocorrelation of sd 1/√n, so
# the z-scores sit inside ±4 over many draws and their mean near zero;
# a residual with a bias in it lights both.
#
# Run:  julia --project=estimator estimator/test/test_misfit.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using LinearAlgebra, Random, Statistics, Test

# two "channels" of a linear model, no prior (V = (BᵀB)⁻¹), a static row
rng = MersenneTwister(7)
n1, n2, p = 150, 90, 5
B = randn(rng, n1 + n2 + 1, p)
V = inv(B' * B)
mask1 = trues(n1); mask2 = trues(n2)
chs = [PitProbe(1.0, 0.0, mask1), HumiditySensor(1.0, 0.0, mask2), StaticReading(1, 1.0)]

@testset "misfit: white residuals are inside the band" begin
    zc, zr = Float64[], Float64[]
    for _ in 1:400
        e = randn(rng, n1 + n2 + 1)
        r = e - B * (V * (B' * e))          # least-squares residual (I − H) e
        mis = misfit(chs, zeros(length(r)), B, r, V)
        @test length(mis.channels) == 3
        @test mis.channels[1].n == n1 && mis.channels[2].n == n2 && mis.channels[3].n == 1
        @test mis.total.dof ≈ n1 + n2 + 1 - p atol=1e-9   # tr(H) = p exactly without a prior
        push!(zc, mis.total.z_chi2); push!(zr, mis.total.z_rho1)
    end
    @test abs(mean(zc)) < 0.25 && 0.7 < std(zc) < 1.3
    @test abs(mean(zr)) < 0.25 && 0.7 < std(zr) < 1.3
    @test maximum(abs, zc) < 5 && maximum(abs, zr) < 5
end

@testset "misfit: a bias lights both statistics" begin
    e = randn(rng, n1 + n2 + 1)
    bias = vcat(1.2 .* sin.(range(0, 3π, length=n1)), zeros(n2 + 1))   # a slow structured error on channel 1
    r = (e + bias) - B * (V * (B' * (e + bias)))
    mis = misfit(chs, zeros(length(r)), B, r, V)
    @test mis.channels[1].z_chi2 > 3 && mis.channels[1].z_rho1 > 3
    @test abs(mis.channels[2].z_rho1) < 4        # the clean channel stays clean
end

@testset "misfit: rows must cover the residual" begin
    @test_throws ArgumentError misfit(chs[1:2], zeros(n1 + n2 + 1), B, zeros(n1 + n2 + 1), V)
end
println("test_misfit.jl: done")
