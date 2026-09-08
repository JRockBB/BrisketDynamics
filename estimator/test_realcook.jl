# The real-cook reading model where the answer is known: each channel's
# mask has its own spacing and drops the lid-open windows; the lid
# windows follow every event including each spritz; the pit noise level
# rises after the fuel add and decays; the humidity reading never
# exceeds the cap; the censored channel predicts min(pred, cap) with a
# zero residual and zero slope where the model is above the cap and the
# plain prediction where the reading is below it.
#
# Run:  julia --project=estimator estimator/test_realcook.jl
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using LinearAlgebra, Random, Statistics, Test
const E = BrisketEstimator

t = collect(0.0:120.0:20 * 3600.0)
nc = RealCookNoise()
events = (t_wrap=6.0 * 3600, t_pull=12.0 * 3600, t_add=4.0 * 3600, t_spritz=3.0 * 3600, dt_spritz=1.5 * 3600, m_spritz=0.05)

@testset "lid windows and masks" begin
    open = lid_windows(t, events, nc)
    # the wrap, the pull, the fuel add, and the spritzes at 3, 4.5, 6, 7.5, 9, 10.5, 12 h: each six minutes = 3 grid points
    n_ev = 3 + 7
    @test count(open) <= n_ev * 3 && count(open) >= (n_ev - 2) * 3      # the wrap at 6 h and a spritz at 6 h overlap
    masks = realcook_masks(t, open, nc)
    @test E.channel_gap(masks.pit) == 1 && E.channel_gap(masks.core) == 1
    @test E.channel_gap(masks.humidity) == 5 && E.channel_gap(masks.mass) == 15
    @test !any(masks.pit .& open) && !any(masks.mass .& open)
end

@testset "the pit level follows the fire, the humidity is capped" begin
    nx, nt = 12, length(t)
    o = (Tpit=fill(380.0, nt), T_field=fill(340.0, nx, nt), pv_pit=fill(6400.0, nt) .+ 300 .* sin.(t ./ 7200),
         W_field=fill(0.6, nx, nt), F_field=fill(0.1, nx, nt))
    r, clean, masks, open = realcook_readings(o, t, events, nc, MersenneTwister(1); A=0.18, L=0.04, depth=0.03)
    quiet = (t_wrap=events.t_wrap, t_pull=events.t_pull, t_add=1e9, t_spritz=events.t_spritz, dt_spritz=events.dt_spritz, m_spritz=events.m_spritz)
    r0, _, _, _ = realcook_readings(o, t, quiet, nc, MersenneTwister(1); A=0.18, L=0.04, depth=0.03)
    # the same unit process, scaled: the ratio of the pit noise with and without the fuel add is the level exactly
    e, e0 = r.pit .- clean.pit, r0.pit .- clean.pit
    level = [tk > events.t_add ? 1 + nc.pit_add_bump * exp(-(tk - events.t_add) / nc.add_tau) : 1.0 for tk in t]
    ok = abs.(e0) .> 1e-6
    @test maximum(abs.(e[ok] ./ e0[ok] .- level[ok])) < 1e-9
    @test level[findfirst(t .> events.t_add)] > 2.8 && level[end] < 1.01      # two minutes into the bump: 1 + 2 exp(−120/1800)
    @test all(r.humidity .<= nc.hum_cap + 1e-9)
    @test any(r.humidity[masks.humidity] .>= nc.hum_cap - 1e-9)
    @test all(isfinite, r.mass) && abs(mean(r.core .- clean.core)) < 0.3
end

@testset "the censored humidity channel" begin
    nt = 5
    lay = OutputLayout(3, nt)
    S = zeros(length(lay)); J = zeros(length(lay), 2)
    pv = [6000.0, 6600.0, 7000.0, 6400.0, 6900.0]
    S[(lay.opv + 1):(lay.opv + nt)] .= pv
    J[(lay.opv + 1):(lay.opv + nt), 1] .= 1.0
    data = [6000.0, 6500.0, 6500.0, 6300.0, 6500.0]            # readings 2, 3 and 5 at the cap
    ch = CensoredHumidity(100.0, 0.8, trues(nt); cap=6500.0, data=data)
    r, Jr = E.predict(ch, S, J, zeros(2), PriorSite[], lay)
    @test r[1] == 6000.0 && Jr[1, 1] == 1.0                     # below the cap: the plain prediction
    @test r[4] == 6400.0 && Jr[4, 1] == 1.0
    @test abs(r[3] - 6500.0) < 1.0 && Jr[3, 1] < 0.01           # the model above the cap: residual ≈ 0, slope ≈ 0
    @test abs(r[5] - 6500.0) < 1.0 && Jr[5, 1] < 0.01
    @test 6450.0 < r[2] < 6500.0 && 0.1 < Jr[2, 1] < 0.99         # just above the cap: the smooth corner
    @test E.channel_label(ch) == "humidity (censored)"
    @test size(E.whitening(ch), 1) == nt
end
println("test_realcook.jl: done")
