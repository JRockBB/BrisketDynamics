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
    @test length(y) == 4 && all(isfinite, y)
    @test 273.15 < y[1] < 373.15 && 0 < y[2] < 1 && 0 < y[3] < 100 && 273.15 < y[4] < 373.15
    r = E.goal_residuals(y, PlanGoals())
    @test all(isfinite, r) && r[4] >= 0
    Jr = ForwardDiff.jacobian(v -> E.goal_residuals(plan_outputs_rh(z, v, f, s, d), PlanGoals()), mζ)
    @test size(Jr) == (4, 10) && all(isfinite, Jr)
    @test any(abs.(Jr[1, :]) .> 0)          # the pull temperature answers to the plan
end
println("test_horizon.jl: done")
