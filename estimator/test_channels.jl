# The channel registry's analytic maps against automatic differentiation:
# the wet-bulb map inverts the psychrometer equation and its implicit
# derivatives match ForwardDiff through the Newton solve; the fuel-scale
# and bark rows are the stacked entries; one whitened Jacobian row of the
# wet-bulb channel matches ForwardDiff of the whole channel on a small
# stacked output.
#
# Run:  julia --project=estimator estimator/test_channels.jl
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using ForwardDiff, LinearAlgebra, Random, Test

const E = BrisketEstimator
γ = 6.62e-4 * BrisketDynamics.P_ATM

@testset "wet bulb: the psychrometer equation is solved" begin
    for (T, pv) in ((383.15, 8.0e3), (373.15, 3.0e3), (393.15, 15.0e3), (360.0, 1.0e3))
        Tw, fp = E.wet_bulb(T, pv, γ)
        @test abs(BrisketDynamics.p_sat(Tw) + γ * (Tw - T) - pv) < 1e-6
        @test Tw < T
        @test fp > 0
    end
end

@testset "wet bulb: implicit derivatives match ForwardDiff through the solve" begin
    for (T, pv) in ((383.15, 8.0e3), (373.15, 3.0e3), (393.15, 15.0e3))
        Tw, fp = E.wet_bulb(T, pv, γ)
        dT_ad = ForwardDiff.derivative(t -> E.wet_bulb(t, pv, γ)[1], T)
        dp_ad = ForwardDiff.derivative(p -> E.wet_bulb(T, p, γ)[1], pv)
        @test isapprox(dT_ad, γ / fp; rtol=1e-8)
        @test isapprox(dp_ad, 1 / fp; rtol=1e-8)
    end
end

@testset "fuel, bark and wet-bulb rows on a synthetic stack" begin
    rng = MersenneTwister(3)
    nx, nt, nz = 4, 5, 3
    lay = OutputLayout(nx, nt)
    S = zeros(length(lay))
    S[(lay.oTpit + 1):(lay.oTpit + nt)] .= 380.0 .+ 5 .* rand(rng, nt)
    S[(lay.opv + 1):(lay.opv + nt)] .= 5.0e3 .+ 3.0e3 .* rand(rng, nt)
    S[(lay.oM + 1):(lay.oM + nt)] .= 2.0 .- 0.1 .* (1:nt)
    S[(lay.oL + 1):(lay.oL + nt)] .= 60.0 .- 3.0 .* (1:nt)
    J = randn(rng, length(lay), nz)
    z = zeros(nz)
    prior = PriorSite[]
    mask = trues(nt)
    rF, JF = E.predict(FuelScale(0.05, 0.5, mask), S, J, z, prior, lay)
    @test rF == S[(lay.oM + 1):(lay.oM + nt)] && JF == J[(lay.oM + 1):(lay.oM + nt), :]
    rB, JB = E.predict(BarkLightness(3.0, 0.8, mask), S, J, z, prior, lay)
    @test rB == S[(lay.oL + 1):(lay.oL + nt)] && JB == J[(lay.oL + 1):(lay.oL + nt), :]
    # the wet-bulb channel as a function of the stack, differentiated through the chain S(z) = S0 + J z
    wb = WetBulb(1.0, 0.8, mask)
    rW, JW = E.predict(wb, S, J, z, prior, lay)
    f(zz) = E.predict(wb, S .+ J * zz, J, zz, prior, lay)[1]
    Jad = ForwardDiff.jacobian(f, z)
    @test isapprox(JW, Jad; rtol=1e-7, atol=1e-9)
    @test all(rW .< S[(lay.oTpit + 1):(lay.oTpit + nt)])
end
println("test_channels.jl: done")
