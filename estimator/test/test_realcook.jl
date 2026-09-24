# The real-cook reading model where the answer is known: each channel's
# mask has its own spacing and drops the lid-open windows; the lid
# windows follow every event including each spritz; the pit noise level
# rises after the fuel add and decays; the humidity reading never
# exceeds the cap; capped samples use the exact normal survival
# likelihood while RxInfer receives its score-and-curvature-matched
# local Gaussian site.
#
# Run:  julia --project=estimator estimator/test/test_realcook.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using LinearAlgebra, Random, Statistics, Test
const E = BrisketEstimator

struct _StaticHumidityPhysics <: E.Physics
    stack::Vector{Float64}
    lay::OutputLayout
end
E.physics_stack(z, d, ph::_StaticHumidityPhysics) = (ph.stack, ph.lay)
E.physics_outputs(z, d, ph::_StaticHumidityPhysics) =
    (ph.stack, zeros(length(ph.stack), length(z)), ph.lay)

t = collect(0.0:120.0:20 * 3600.0)
nc = RealCookNoise()
events = (t_wrap=6.0 * 3600, t_pull=12.0 * 3600, t_add=4.0 * 3600, t_spritz=3.0 * 3600, dt_spritz=1.5 * 3600, m_spritz=0.05)

@testset "lid windows and masks" begin
    open = lid_windows(t, events, nc)
    # the wrap, the pull, the fuel add, and the spritzes at 3, 4.5 and 6 h (the train stops at the wrap): each six minutes = 3 grid points
    n_ev = 3 + 3
    @test count(open) <= n_ev * 3 && count(open) >= (n_ev - 2) * 3      # the wrap at 6 h and a spritz at 6 h overlap
    masks = realcook_masks(t, open, nc)
    @test E.channel_gap(masks.pit) == 1 && E.channel_gap(masks.core) == 1
    @test E.channel_gap(masks.humidity) == 5 && E.channel_gap(masks.mass) == 15
    @test !any(masks.pit .& open) && !any(masks.mass .& open)
    # after the wrap no spritz opens the lid (the plant sprays no wrapped brisket; monolith.jl gates the lift the
    # same way): between the wrap's own window and the pull nothing is open, on the grid and at any time
    after_wrap = (t .>= events.t_wrap + nc.lid_open) .& (t .< events.t_pull)
    @test !any(open[after_wrap]) && !any(E.lid_open_at(tk, events, nc) for tk in t[after_wrap])
    @test E.lid_open_at(4.5 * 3600, events, nc) && !E.lid_open_at(7.5 * 3600, events, nc)
    no_wrap = (t_wrap=1e9, t_pull=events.t_pull, t_add=events.t_add, t_spritz=events.t_spritz, dt_spritz=events.dt_spritz, m_spritz=events.m_spritz)
    @test E.lid_open_at(7.5 * 3600, no_wrap, nc) && count(lid_windows(t, no_wrap, nc)) > count(open)   # unwrapped, the train runs to the pull
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
    @test all(isfinite, r.mass) && abs(mean(r.core .- clean.core)) < 0.8
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
    @test r == pv && all(Jr[:, 1] .== 1.0)                     # prediction remains the latent uncapped reading
    @test E.channel_label(ch) == "humidity (censored)"
    @test size(E.whitening(ch), 1) == nt

    a, B, w = E.assemble([ch], zeros(2), S, J, PriorSite[], lay)
    row = 3                                                     # two uncensored AR rows, then capped sample 2
    ε = 0.1
    f(x) = E._censored_normal_nll(x, ch.cap, ch.σ)
    g_fd = (f(pv[2] + ε) - f(pv[2] - ε)) / (2ε)
    h_fd = (f(pv[2] + ε) - 2f(pv[2]) + f(pv[2] - ε)) / ε^2
    @test B[row, 1] * (a[row] - w[row]) ≈ g_fd rtol=2e-6 atol=1e-10
    @test B[row, 1]^2 ≈ h_fd rtol=2e-5 atol=1e-10

    lay1 = OutputLayout(1, 1)
    S1 = zeros(length(lay1)); S1[lay1.opv + 1] = 6600.0
    ch1 = CensoredHumidity(100.0, 0.0, trues(1); cap=6500.0, data=[6500.0])
    loss = E.likelihood_objective([ch1], zeros(1), S1, PriorSite[], lay1, nothing)
    @test loss ≈ f(6600.0) rtol=1e-12
    @test f(6000.0) > f(6500.0) > f(7000.0)                    # evidence is one-sided, not a squared hinge
    extreme = E._censored_normal_site(0.0, 6500.0, 10.0)
    @test isfinite(E._censored_normal_nll(0.0, 6500.0, 10.0)) &&
          isfinite(extreme.pseudo) && extreme.root_precision > 0

    # The ordinary RxInfer solve conditions the discrepancy state before
    # clipping; this is the graph path used by the sequential estimator.
    stack = zeros(length(lay1)); stack[lay1.opv + 1] = 6400.0
    ph = _StaticHumidityPhysics(stack, lay1)
    d = estimator_setup([0.0, 1.0], [0.0, 0.0], PitConfig(), MeatConfig(), PriorSite[], String[])
    sites = state_sites(1)
    prior = FilterPrior(sites, [0.0], reshape([1e-4], 1, 1), reshape([100.0], 1, 1))
    sq = E.Sequential(ch1, 1, 10.0)
    fit = solve_map_channels([sq], prior, d; c0=[0.0], physics=ph, iters=20, tol=1e-6, ftol=1e-8)
    @test fit.converged && fit.z[1] > 50.0 && sqrt(fit.cov[1, 1]) < 100.0
    # the exact scalar posterior as an oracle: the objective the solve minimizes, the survival term at the row's
    # floor plus the prior's quadratic, its minimizer by Newton with a finite-difference curvature and the local
    # variance the inverse curvature; the graph's centre, its mean and its covariance must be that point, and the
    # centre must be the graph's own fixed point (until 2026-09-24 the iterations paired a fresh site with the
    # previous centre's pseudo-observation and stopped at 127.7 with a variance six times the local one)
    f_ex(z) = E._censored_normal_nll(6400.0 + z, 6500.0, 10.0) + 0.5 * z^2 / 100.0^2
    fd1(z; ε=1e-3) = (f_ex(z + ε) - f_ex(z - ε)) / (2ε)
    fd2(z; ε=1e-2) = (f_ex(z + ε) - 2f_ex(z) + f_ex(z - ε)) / ε^2
    z_map = 100.0
    for _ in 1:60; z_map -= fd1(z_map) / fd2(z_map); end
    var_map = 1 / fd2(z_map)
    println("  the censored scalar oracle: MAP ", round(z_map; digits=2), ", local sd ", round(sqrt(var_map); digits=2),
            "; the graph: centre ", round(fit.z[1]; digits=2), ", mean ", round(fit.mean[1]; digits=2), ", sd ", round(sqrt(fit.cov[1, 1]); digits=2),
            ", fixed-point gap ", round(fit.gap; digits=5), " sd")
    @test abs(fd1(z_map)) < 1e-8
    @test abs(fit.z[1] - z_map) < 5e-3 * abs(z_map)
    @test abs(fit.mean[1] - fit.z[1]) < 1e-2 * 100.0 && fit.gap < 1e-2
    @test abs(fit.cov[1, 1] - var_map) < 0.05 * var_map
    # the same site with its reading under the cap is one linear-Gaussian row: the graph must return the Kalman
    # update exactly, its centre, its mean and its variance alike
    ch2 = CensoredHumidity(100.0, 0.0, trues(1); cap=6500.0, data=[6450.0])
    sq2 = E.Sequential(ch2, 1, 10.0)
    fit2 = solve_map_channels([sq2], prior, d; c0=[0.0], physics=ph, iters=20, tol=1e-8, ftol=1e-10)
    var_kf = 1 / (1 / 100.0^2 + 1 / 10.0^2); mean_kf = var_kf * (6450.0 - 6400.0) / 10.0^2
    @test fit2.converged && isapprox(fit2.z[1], mean_kf; rtol=1e-6) && isapprox(fit2.mean[1], mean_kf; rtol=1e-6)
    @test isapprox(fit2.cov[1, 1], var_kf; rtol=1e-6) && fit2.gap < 1e-6
end
println("test_realcook.jl: done")
