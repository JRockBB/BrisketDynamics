# The receding-horizon pieces where the answer is known: the hinge is
# zero above the floor and quadratic below with the right derivative;
# the free set shrinks as events commit; encode/decode round-trips and
# keeps the ordering wrap < pull < serve and spritz, fuel before the
# pull for random plans; a due event commits at its planned time and is
# fixed after, one not due stays free; the executed damper is preserved
# by the plan's damper; the plan physics runs at the truth with all
# events and its goal residuals are finite with a finite Jacobian.
#
# Run:  julia --project=estimator estimator/test_horizon.jl
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using ForwardDiff, JSON, LinearAlgebra, Random, Test
const E = BrisketEstimator

@testset "hinge: a one-sided floor" begin
    @test hinge(-10.0) < 1e-4
    @test abs(hinge(10.0) - 10.0) < 1e-4
    @test abs(hinge(0.0) - log(2)) < 1e-12
    @test abs(ForwardDiff.derivative(hinge, 3.0) - 1 / (1 + exp(-3.0))) < 1e-12
    @test hinge(100.0) == 100.0                      # no overflow far below the floor
end

GRID = collect(0.0:600.0:30 * 3600.0)
rng = MersenneTwister(2)

@testset "free set, encode/decode, ordering" begin
    s = Schedule(GRID; u0=0.15)
    f = free_set(s)
    @test f.damper && f.wrap && f.pull && f.serve && f.spritz && f.add && length(f) == 10
    m, Λ = plan_prior(f, s)
    @test length(m) == 10 && size(Λ) == (10, 10)
    for trial in 1:50
        ζ = m .+ 0.5 .* randn(rng, 10)          # inside the encoding's clamp band (fractions in 0.02 to 0.98)
        p = decode_plan(ζ, f, s)
        @test s.t_now < p.t_wrap < p.t_pull < p.t_serve
        @test s.t_now < p.spritz.t < p.t_pull && s.t_now < p.t_add < p.t_pull
        @test all(0 .< p.u .< 1) && p.spritz.dt > 0 && p.spritz.m > 0
        ζ2 = encode_plan(p, f, s)
        @test maximum(abs.(ζ2 .- ζ)) < 1e-8
    end
    # after the pull only the serve time is free
    s2 = Schedule(GRID; u0=0.15); s2.t_now = 12 * 3600.0; s2.t_pull = 11.5 * 3600.0
    f2 = free_set(s2)
    @test !f2.damper && !f2.wrap && !f2.pull && f2.serve && !f2.spritz && !f2.add && length(f2) == 1
    p2 = decode_plan([log(2.0)], f2, s2)
    @test p2.t_pull == s2.t_pull && abs(p2.t_serve - (s2.t_pull + 2 * 3600)) < 1e-9
end

@testset "commit: due events execute at their planned time and stay fixed" begin
    s = Schedule(GRID; u0=0.15)
    Δ = 2 * 3600.0
    plan = (u=[0.2, 0.3, 0.25], t_wrap=1.5 * 3600, t_pull=9.0 * 3600, t_serve=11 * 3600,
            spritz=(t=1.0 * 3600, dt=3600.0, m=0.04), t_add=5.0 * 3600)
    done = commit!(s, plan, Δ)
    @test Set(done) == Set([:wrap, :spritz])
    @test s.t_wrap == plan.t_wrap && s.spritz.t == plan.spritz.t && isnan(s.t_pull) && isnan(s.t_add)
    @test s.t_serve == plan.t_serve
    execute_damper!(s, plan, Δ)
    @test s.t_now == Δ
    f = free_set(s)
    @test !f.wrap && !f.spritz && f.pull && f.add && f.damper
    # a later plan cannot move the executed wrap
    plan2 = (u=[0.2, 0.3, 0.25], t_wrap=NaN, t_pull=9.0 * 3600, t_serve=11 * 3600, spritz=nothing, t_add=5.0 * 3600)
    p = decode_plan(encode_plan(plan2, f, s), f, s)
    @test p.t_wrap == plan.t_wrap && p.spritz.t == plan.spritz.t
    ev = schedule_events(s)
    @test ev.t_wrap == plan.t_wrap && ev.t_pull == BrisketDynamics.PULL_NEVER && ev.t_spritz == plan.spritz.t
end

@testset "the plan's damper preserves the executed log and follows the knots" begin
    s = Schedule(GRID; u0=0.15)
    s.u[GRID .<= 4 * 3600] .= 0.33
    s.t_now = 4 * 3600.0
    plan = (u=[0.1, 0.5, 0.2], t_wrap=NaN, t_pull=13.0 * 3600, t_serve=15 * 3600, spritz=nothing, t_add=NaN)
    u = plan_damper(plan, s)
    @test all(u[GRID .<= 4 * 3600] .== 0.33)
    @test u[findfirst(GRID .> 4 * 3600)] ≈ 0.1 atol=0.05      # ten minutes into a three-hour ramp toward 0.5
    k2 = findfirst(GRID .>= 4 * 3600 + (13 - 4) * 3600 / 3)
    @test abs(u[k2] - 0.5) < 0.05
    @test all(u[GRID .>= 13 * 3600] .== 0.2)
end

@testset "the plan physics at the truth, all events on, finite residuals and Jacobian" begin
    REPO = normpath(joinpath(@__DIR__, ".."))
    m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json"))
    gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
    meta = m["meta"]
    pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
    cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
    prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
    s = Schedule(GRID; u0=0.15)
    d = estimator_setup(GRID, fill(0.15, length(GRID)), pit, cfg, prior_phys, String.(gate["names"]))
    x = Float64.(m["truth"]["x"])
    z = vcat([isbeta(p) ? log(v / (1 - v)) : log(v) for (v, p) in zip(x[1:15], prior_phys)], log(x[16]))
    f = free_set(s)
    mζ, _ = plan_prior(f, s)
    y = plan_outputs_rh(z, mζ, f, s, d)
    @test length(y) == 12 && all(isfinite, y)
    @test 0 <= y[1] <= 1 && 0 < y[2] < 1 && 0 < y[3] < 100 && 273.15 < y[4] < 373.15 && 0 <= y[5] <= 1 && y[7] >= 0 && y[11] > 0 && y[12] > 0
    g = PlanGoals()
    r = E.goal_residuals(y, g)
    @test all(isfinite, r) && all(r .>= 0) && length(r) == 13
    # the app's objective, term for term: a good cook scores under one; the terms read as the app's
    y0 = [0.93, 0.48, 31.0, 273.15 + 75, 0.7, 0.55, 0.0, 0.3, 0.0, 0.0, 1e5, 1e5]
    T = objective_terms(y0, g)
    @test T.J < 1.0 && abs(T.J - (T.tenderness + T.appearance + T.taste + T.surf + T.fuel + T.smooth + T.safety)) < 1e-12
    sp(x) = log1p(exp(x))
    @test abs(T.taste - 0.57 * (sp(-1.875)^2 + sp(-2.5)^2 + 4.0 * 0.45^2)) < 1e-9        # the smoke band's tails and the dryness
    @test abs(T.fuel - 0.1 * 0.3) < 1e-9 && T.safety < 1e-6
    @test abs(T.appearance - 0.14 * 2 * sp(-1.5)^2) < 1e-9                               # L* 31, the band's tails
    over = objective_terms(vcat(0.99, y0[2:end]), g).tenderness                       # 0.04 past pot roast
    @test abs(over - 0.29 * ((sp(-1.8) / 20 / 0.05)^2 + (sp(3.2) / 80 / 0.05)^2)) < 1e-9
    pale = objective_terms(vcat(y0[1:2], 38.0, y0[4:end]), g).appearance                  # L* 38, four past the band
    @test abs(pale - 0.14 * (sp(2.0)^2 + sp(-5.0)^2)) < 1e-9
    @test objective_terms(vcat(y0[1:10], 10.0, 10.0), g).safety > 1000                   # ten seconds of lethality: the curve bites
    @test all(isfinite, ForwardDiff.gradient(v -> sum(abs2, E.goal_residuals(v, g)), y0))
    Jr = ForwardDiff.jacobian(v -> E.goal_residuals(plan_outputs_rh(z, v, f, s, d), g), mζ)
    @test size(Jr) == (13, 10) && all(isfinite, Jr)
    @test any(abs.(Jr) .> 0)                # the objective answers to the plan
    # the executed damper's smoothness term: hourly means to the pull
    @test executed_du(GRID, fill(0.2, length(GRID)), 6 * 3600.0) ≈ zeros(5) atol=1e-12
end
@testset "the replan trigger: first, event due, misfit, loss rise, floor, hold" begin
    s = Schedule(GRID; u0=0.15)
    s.t_now = 4 * 3600.0
    pol = ReplanPolicy()
    plan = (u=[0.2, 0.3, 0.25], t_wrap=6.0 * 3600, t_pull=12.0 * 3600, t_serve=14 * 3600, spritz=(t=4.8 * 3600, dt=3600.0, m=0.04), t_add=8.0 * 3600)
    @test replan_due(pol, s, nothing, 0.0, 0.0, 0.0, nothing) == (true, :first)
    @test replan_due(pol, s, plan, 0.5, 0.5, 2 * 3600.0, 0.5) == (true, :event_due)          # the spritz at 4.8 h is due within the hour
    # the commit rule: an event a replan left within the tolerance commits now, however far ahead
    prev = (u=plan.u, t_wrap=6.2 * 3600, t_pull=12.0 * 3600, t_serve=14 * 3600, spritz=(t=4.75 * 3600, dt=3600.0, m=0.04), t_add=9.0 * 3600)
    st = stable_events(plan, prev, pol.stable)
    @test Set(st) == Set([:wrap, :spritz, :pull])                  # the add moved an hour, the others under half an hour
    @test stable_events(plan, nothing, pol.stable) == Symbol[]
    s3 = Schedule(GRID; u0=0.15); s3.t_now = 4 * 3600.0
    done3 = commit!(s3, plan, 3600.0; force=st)
    @test Set(done3) == Set([:wrap, :spritz, :pull]) && s3.t_wrap == plan.t_wrap && s3.t_pull == plan.t_pull && isnan(s3.t_add)
    s.spritz = plan.spritz                                                                   # once committed it no longer triggers
    @test replan_due(pol, s, plan, 0.5, 0.5, 2 * 3600.0, 0.5) == (false, :hold)
    @test replan_due(pol, s, plan, 0.5, 0.5, 2 * 3600.0, 3.5) == (true, :misfit)
    @test replan_due(pol, s, plan, 0.5, 0.5, 2 * 3600.0, -3.5) == (false, :hold)           # too little power is conservative, not wrong
    @test replan_due(pol, s, plan, 0.9, 0.5, 2 * 3600.0, 0.5) == (true, :loss_rise)
    @test replan_due(pol, s, plan, 0.6, 0.5, 2 * 3600.0, 0.5) == (false, :hold)             # a rise under the threshold holds
    @test replan_due(pol, s, plan, 27.0, 134.5, 2 * 3600.0, 0.5) == (true, :loss_high)      # the loss fell, and the plan is still bad
    @test replan_due(pol, s, plan, 0.6, 0.5, 2 * 3600.0, 0.5; kl=5.0) == (true, :posterior_moved)   # the belief moved five nats
    @test replan_due(pol, s, plan, 0.6, 0.5, 2 * 3600.0, 0.5; kl=0.2) == (false, :hold)
    @test replan_due(pol, s, plan, 0.6, 0.5, 2 * 3600.0, 0.5; kl=3.0, n_latent=17) == (false, :hold)             # under a quarter nat per latent
    @test replan_due(pol, s, plan, 0.6, 0.5, 2 * 3600.0, 0.5; kl=5.0, n_latent=17) == (true, :posterior_moved)
    # the KL of a posterior against itself is zero; halving one width and moving one mean by a sd is under a nat
    V0 = Matrix(Diagonal([1.0, 4.0])); m0 = [0.0, 1.0]
    @test abs(posterior_kl(m0, V0, m0, V0)) < 1e-12
    @test 0.3 < posterior_kl([1.0, 1.0], Matrix(Diagonal([0.25, 4.0])), m0, V0) < 1.0
    @test replan_due(pol, s, plan, 0.5, 0.5, 0.0, 0.5) == (true, :floor)                     # four hours since the last replan
    s.t_now = 5.2 * 3600.0
    @test replan_due(pol, s, plan, 0.5, 0.5, 4 * 3600.0, 0.5) == (true, :event_due)          # the wrap at 6 h comes due
    # once the pull has committed, a standing plan's spritz or wrap is no longer a decision and cannot come due
    s4 = Schedule(GRID; u0=0.15); s4.t_now = 11.5 * 3600.0; s4.t_pull = 12.0 * 3600.0; s4.t_wrap = 6.0 * 3600.0
    plan4 = (u=plan.u, t_wrap=6.0 * 3600, t_pull=12.0 * 3600, t_serve=16.0 * 3600, spritz=(t=11.8 * 3600, dt=3600.0, m=0.04), t_add=11.9 * 3600)
    @test replan_due(pol, s4, plan4, 0.5, 0.5, 11.0 * 3600.0, 0.5) == (false, :hold)
    plan5 = (plan4..., t_serve=12.4 * 3600)
    @test replan_due(pol, s4, plan5, 0.5, 0.5, 11.0 * 3600.0, 0.5) == (true, :event_due)      # the serve is still free
end

@testset "the standing plan's expected loss under posterior draws" begin
    REPO = normpath(joinpath(@__DIR__, ".."))
    m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json"))
    gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
    meta = m["meta"]
    pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
    cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
    prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
    s = Schedule(GRID; u0=0.15)
    d = estimator_setup(GRID, fill(0.15, length(GRID)), pit, cfg, prior_phys, String.(gate["names"]))
    x = Float64.(m["truth"]["x"])
    z = vcat([isbeta(p) ? log(v / (1 - v)) : log(v) for (v, p) in zip(x[1:15], prior_phys)], log(x[16]))
    f = free_set(s)
    mζ, _ = plan_prior(f, s)
    plan = decode_plan(mζ, f, s)
    l1 = plan_loss([z], plan, s, d)
    @test length(l1.draws) == 1 && isfinite(l1.mean) && l1.mean >= 0
    @test abs(l1.mean - 0.5 * sum(abs2, E.goal_residuals(plan_outputs_rh(z, mζ, f, s, d), PlanGoals()))) < 1e-9
    zs = [z, z .+ 0.3 .* randn(rng, length(z))]
    l2 = plan_loss(zs, plan, s, d)
    @test length(l2.draws) == 2 && abs(l2.mean - sum(l2.draws) / 2) < 1e-12
    # a later clock with the same plan: re-encoded in the shrunken window, the loss is still finite
    s.t_now = 2 * 3600.0
    @test isfinite(plan_loss([z], plan, s, d).mean)
end
println("test_horizon.jl: done")
