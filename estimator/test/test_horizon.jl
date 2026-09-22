# The receding-horizon pieces where the answer is known: the hinge is
# zero above the floor and quadratic below with the right derivative;
# the free set shrinks as events commit; encode/decode round-trips and
# keeps the ordering wrap < pull < serve and spritz, fuel before the
# pull for random plans; a due event commits at its planned time and is
# fixed after, one not due stays free; the executed damper is preserved
# by the plan's damper; the plan physics runs at the truth with all
# events and its goal residuals are finite with a finite Jacobian.
#
# Run:  julia --project=estimator estimator/test/test_horizon.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using ForwardDiff, JSON, LinearAlgebra, Random, Test
const E = BrisketEstimator
const knot_f = E.knot_f

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
    @test f.damper && f.wrap && f.pull && f.serve && f.spritz && f.add && length(f) == N_KNOTS + 7
    m, Λ = plan_prior(f, s)
    @test length(m) == N_KNOTS + 7 && size(Λ) == (N_KNOTS + 7, N_KNOTS + 7)
    for trial in 1:50
        ζ = m .+ 0.5 .* randn(rng, N_KNOTS + 7)          # inside the encoding's clamp band (fractions in 0.02 to 0.98)
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
    # a pull committed ahead fixes its own time only: the damper, wrap, spritz and add before it stay decisions
    s2b = Schedule(GRID; u0=0.15); s2b.t_now = 8 * 3600.0; s2b.t_pull = 15.0 * 3600.0
    f2b = free_set(s2b)
    @test f2b.damper && f2b.wrap && !f2b.pull && f2b.serve && f2b.spritz && f2b.add && length(f2b) == N_KNOTS + 6
    p2b = decode_plan(plan_prior(f2b, s2b)[1], f2b, s2b)
    @test p2b.t_pull == s2b.t_pull && s2b.t_now < p2b.t_wrap < p2b.t_pull && s2b.t_now < p2b.spritz.t < p2b.t_pull
    p2 = decode_plan([E.lgt(2 * 3600 / (T_SERVE_MAX - s2.t_pull))], f2, s2)     # the rest: a fraction of the pull to the bound
    @test p2.t_pull == s2.t_pull && abs(p2.t_serve - (s2.t_pull + 2 * 3600)) < 1e-6
    # no plan leaves the app's decision space: the serve stays under the 24-h bound whatever the coordinates
    for ζ in ([fill(10.0, N_KNOTS); 0.0; 10.0; 10.0; 0.0; 0.0; log(0.05); 0.0], [fill(-10.0, N_KNOTS); 0.0; -10.0; -10.0; 0.0; 0.0; log(0.05); 0.0])
        p = decode_plan(ζ, f, s)
        @test s.t_now < p.t_pull < p.t_serve <= T_SERVE_MAX + 1e-9
    end
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
    # a plan with seven knots draws a push: high in the middle knots, low at the ends
    plan7 = (u=[0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1], t_wrap=NaN, t_pull=11.0 * 3600, t_serve=13 * 3600, spritz=nothing, t_add=NaN)
    u7 = plan_damper(plan7, s)
    tk = 4 * 3600 .+ knot_f(7) .* (11 - 4) * 3600
    @test abs(u7[searchsortedlast(GRID, tk[3])] - 0.9) < 0.05 && abs(u7[searchsortedlast(GRID, tk[6])] - 0.1) < 0.05
    @test resample_knots([0.1, 0.5, 0.2], 7)[1] == 0.1 && length(resample_knots([0.1, 0.5, 0.2], 7)) == 7
    @test maximum(abs.(resample_knots(resample_knots([0.1, 0.5, 0.2], 7), 3) .- [0.1, 0.5, 0.2])) < 0.1     # a peak between knots is clipped a little
end

@testset "the plan physics at the truth, all events on, finite residuals and Jacobian" begin
    REPO = normpath(joinpath(@__DIR__, "..", ".."))
    m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json"))
    gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
    meta = m["meta"]
    pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
    cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
    prior_phys = read_sites(gate["prior"])
    s = Schedule(GRID; u0=0.15)
    d = estimator_setup(GRID, fill(0.15, length(GRID)), pit, cfg, prior_phys, String.(gate["names"]))
    x = Float64.(m["truth"]["x"])
    z = vcat([to_z(v, p) for (v, p) in zip(x[1:15], prior_phys)], log(x[16]))
    f = free_set(s)
    mζ, _ = plan_prior(f, s)
    y = plan_outputs_rh(z, mζ, f, s, d)
    @test length(y) == 8 + (N_KNOTS - 1) + 2 && all(isfinite, y)
    @test 0 <= y[1] <= 1 && 0 < y[2] < 1 && 0 < y[3] < 100 && 273.15 < y[4] < 373.15 && 0 <= y[5] <= 1 && y[7] >= 0 && y[end - 1] > 0 && y[end] > 0
    g = PlanGoals()
    r = E.goal_residuals(y, g)
    @test all(isfinite, r) && all(r .>= 0) && length(r) == 9 + (N_KNOTS - 1) + 2
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
    @test size(Jr) == (9 + (N_KNOTS - 1) + 2, N_KNOTS + 7) && all(isfinite, Jr)
    @test any(abs.(Jr) .> 0)                # the objective answers to the plan
    # the plan's predictive: the belief's covariance carried through the plan physics into the temperatures and
    # the outcomes; the mean is the plain solve, the sd finite, zero under a zero covariance, and growing with it
    V = Matrix(Diagonal(fill(0.05^2, length(z))))
    pr = plan_predictive(z, V, mζ, f, s, d)
    n = length(s.t)
    @test length(pr.Tcore.mean) == n && length(pr.Tpit.sd) == n && length(pr.Tsurf.sd) == n && length(pr.y.mean) == length(y)
    @test maximum(abs.(pr.y.mean .- y)) < 1e-9 && all(isfinite, pr.Tcore.sd) && all(pr.Tcore.sd .>= 0)
    kp = searchsortedlast(s.t, decode_plan(mζ, f, s).t_pull)
    @test pr.Tcore.sd[kp] > 0.2 && pr.Tpit.sd[kp] > 0.2 && pr.y.sd[1] > 1e-3          # the belief matters to the cook
    pr0 = plan_predictive(z, zeros(length(z), length(z)), mζ, f, s, d)
    @test maximum(pr0.Tcore.sd) < 1e-9 && maximum(pr0.y.sd) < 1e-9
    pr4 = plan_predictive(z, 4 .* V, mζ, f, s, d)
    @test abs(pr4.Tcore.sd[kp] / pr.Tcore.sd[kp] - 2) < 1e-6                        # linear in the sd
    # the replan's rest starts: never worse than the warm start alone, from a state with the pull committed
    s6 = Schedule(GRID; u0=0.15); s6.t_now = 13 * 3600.0; s6.u[GRID .<= s6.t_now] .= 0.3
    s6.t_wrap = 11.0 * 3600; s6.t_pull = 14.0 * 3600; s6.t_add = 10.0 * 3600
    zz = vcat(z, log(0.03))
    r1 = replan_on_graph([zz], s6, d; iters=6, scan_wrap=(), scan_rest=(), scan_damper=false)
    r3 = replan_on_graph([zz], s6, d; iters=6)
    @test r1.n_starts == 1 && r3.n_starts <= 2 && r3.F <= r1.F + 1e-9
    # the hedged replan over sigma points runs and returns a plan inside the decision space
    zs6 = sigma_draws(zz, Diagonal(fill(0.02, length(zz))); n_dirs=1)
    rh = replan_on_graph(zs6, s6, d; iters=3, scan_wrap=(), scan_rest=(), scan_damper=false)
    @test length(zs6) == 3 && isfinite(rh.F) && s6.t_now < rh.plan.t_serve <= T_SERVE_MAX + 1e-9
    # the executed damper's smoothness term: hourly means to the pull
    @test executed_du(GRID, fill(0.2, length(GRID)), 6 * 3600.0) ≈ zeros(5) atol=1e-12
end
@testset "the replan trigger: first, event due, misfit, loss high, loss rise, hold; never a clock" begin
    s = Schedule(GRID; u0=0.15)
    s.t_now = 4 * 3600.0
    pol = ReplanPolicy()
    @test !hasfield(ReplanPolicy, :floor) && !hasfield(ReplanPolicy, :kl_per_latent) && pol.lead == 15 * 60.0
    plan = (u=[0.2, 0.3, 0.25], t_wrap=6.0 * 3600, t_pull=12.0 * 3600, t_serve=14 * 3600, spritz=(t=4.2 * 3600, dt=3600.0, m=0.04), t_add=8.0 * 3600)
    @test replan_due(pol, s, nothing, 0.0, 0.0, nothing) == (true, :first)
    @test replan_due(pol, s, plan, 0.5, 0.5, 0.5) == (true, :event_due)          # the spritz at 4.2 h is due within the lead
    # the commit rule: an event a replan left within the tolerance commits now, however far ahead
    prev = (u=plan.u, t_wrap=6.2 * 3600, t_pull=12.0 * 3600, t_serve=14 * 3600, spritz=(t=4.15 * 3600, dt=3600.0, m=0.04), t_add=9.0 * 3600)
    st = stable_events(plan, prev, pol.stable)
    @test Set(st) == Set([:wrap, :spritz, :pull])                  # the add moved an hour, the others under half an hour
    @test stable_events(plan, nothing, pol.stable) == Symbol[]
    s3 = Schedule(GRID; u0=0.15); s3.t_now = 4 * 3600.0
    done3 = commit!(s3, plan, 3600.0; force=st)
    @test Set(done3) == Set([:wrap, :spritz, :pull]) && s3.t_wrap == plan.t_wrap && s3.t_pull == plan.t_pull && isnan(s3.t_add)
    s5 = Schedule(GRID; u0=0.15); s5.t_now = 4 * 3600.0
    done5 = commit!(s5, plan, 3600.0; force=st, force_horizon=pol.stable_horizon)
    @test Set(done5) == Set([:wrap, :spritz]) && isnan(s5.t_pull)                   # the pull at 12 h is eight hours out: not forced
    s.spritz = plan.spritz                                                                   # once committed it no longer triggers
    @test replan_due(pol, s, plan, 0.5, 0.5, 0.5) == (false, :hold)
    @test replan_due(pol, s, plan, 0.5, 0.5, 3.5) == (true, :misfit)
    @test replan_due(pol, s, plan, 0.5, 0.5, -3.5) == (false, :hold)           # too little power is conservative, not wrong
    @test replan_due(pol, s, plan, 0.9, 0.5, 0.5) == (true, :loss_rise)
    @test replan_due(pol, s, plan, 0.6, 0.5, 0.5) == (false, :hold)             # a rise under the threshold holds
    @test replan_due(pol, s, plan, 27.0, 134.5, 0.5) == (true, :loss_high)      # the loss fell, and the plan is still bad
    # no clock: a plan that has stood for hours with a steady belief and no event due is held
    s.t_now = 5.0 * 3600.0
    @test replan_due(pol, s, plan, 0.5, 0.5, 0.5) == (false, :hold)
    s.t_now = 5.8 * 3600.0
    @test replan_due(pol, s, plan, 0.5, 0.5, 0.5) == (true, :event_due)          # the wrap at 6 h comes due within the quarter hour
    note_replanned!(s, plan, pol)                                                    # replanned for it once: a later update in the same window holds
    @test replan_due(pol, s, plan, 0.5, 0.5, 0.5) == (false, :hold)
    plan_moved = (; plan..., t_wrap=plan.t_wrap + 120.0)                             # the plan moved the wrap: it comes due again
    @test replan_due(pol, s, plan_moved, 0.5, 0.5, 0.5) == (true, :event_due)
    empty!(s.replanned_for)
    # once the pull has been executed (the meat is out), a standing plan's spritz or add is no decision and cannot come due
    s4 = Schedule(GRID; u0=0.15); s4.t_now = 12.5 * 3600.0; s4.t_pull = 12.0 * 3600.0; s4.t_wrap = 6.0 * 3600.0
    plan4 = (u=plan.u, t_wrap=6.0 * 3600, t_pull=12.0 * 3600, t_serve=16.0 * 3600, spritz=(t=12.6 * 3600, dt=3600.0, m=0.04), t_add=12.7 * 3600)
    @test replan_due(pol, s4, plan4, 0.5, 0.5, 0.5) == (false, :hold)
    plan5 = (plan4..., t_serve=12.7 * 3600)
    @test replan_due(pol, s4, plan5, 0.5, 0.5, 0.5) == (true, :event_due)      # the serve is still free
    # a pull committed ahead but not executed leaves the spritz a decision, and it can come due
    s4b = Schedule(GRID; u0=0.15); s4b.t_now = 11.5 * 3600.0; s4b.t_pull = 12.0 * 3600.0; s4b.t_wrap = 6.0 * 3600.0
    plan4b = (plan4..., spritz=(t=11.7 * 3600, dt=3600.0, m=0.04), t_add=13.0 * 3600)
    @test replan_due(pol, s4b, plan4b, 0.5, 0.5, 0.5) == (true, :event_due)
end

@testset "the standing plan's expected loss under posterior draws" begin
    REPO = normpath(joinpath(@__DIR__, "..", ".."))
    m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json"))
    gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
    meta = m["meta"]
    pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
    cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
    prior_phys = read_sites(gate["prior"])
    s = Schedule(GRID; u0=0.15)
    d = estimator_setup(GRID, fill(0.15, length(GRID)), pit, cfg, prior_phys, String.(gate["names"]))
    x = Float64.(m["truth"]["x"])
    z = vcat([to_z(v, p) for (v, p) in zip(x[1:15], prior_phys)], log(x[16]))
    f = free_set(s)
    mζ, _ = plan_prior(f, s)
    plan = decode_plan(mζ, f, s)
    l1 = plan_loss([z], plan, s, d)
    @test length(l1.draws) == 1 && isfinite(l1.mean) && l1.mean >= 0
    @test abs(l1.mean - 0.5 * sum(abs2, E.goal_residuals(plan_outputs_rh(z, mζ, f, s, d), PlanGoals()))) < 1e-9
    zs = [z, z .+ 0.3 .* randn(rng, length(z))]
    # the hedge: the tails at the mean, the bands averaged over the other draws, one goal length per node
    r0 = E.goal_residuals(plan_outputs_rh(z, mζ, f, s, d), PlanGoals())
    @test E.draw_residual(r0, 1, 1, true) === r0 && E.draw_residual(r0, 2, 3, false) === r0
    rm, r2 = E.draw_residual(r0, 1, 3, true), E.draw_residual(r0, 2, 3, true)
    @test length(rm) == length(r0) == length(r2) && all(rm[1:N_BAND] .== 0) && all(r2[(N_BAND + 1):end] .== 0)
    @test rm[(N_BAND + 1):end] == r0[(N_BAND + 1):end] && r2[1:N_BAND] ≈ r0[1:N_BAND] ./ sqrt(2)
    sd = sigma_draws(z, Diagonal(fill(0.01, length(z))); n_dirs=2)
    @test length(sd) == 5 && sd[1] == z && sd[2] .+ sd[3] ≈ 2 .* z && sd[4] .+ sd[5] ≈ 2 .* z
    lh = plan_loss([z, z, z], plan, s, d; hedge=true)
    @test abs(lh.mean - 0.5 * sum(abs2, r0)) < 1e-9        # identical draws: the hedge is the plain loss
    @test abs(plan_loss([z, z, z], plan, s, d; hedge=false).mean - 0.5 * sum(abs2, r0)) < 1e-9
    l2 = plan_loss(zs, plan, s, d; hedge=false)
    @test length(l2.draws) == 2 && abs(l2.mean - sum(l2.draws) / 2) < 1e-12
    # a later clock with the same plan: re-encoded in the shrunken window, the loss is still finite
    s.t_now = 2 * 3600.0
    @test isfinite(plan_loss([z], plan, s, d).mean)
end
println("test_horizon.jl: done")
