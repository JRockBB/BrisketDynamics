# The multi-meat planner (plan_system.jl):
#   parity     for a system of one slab the plan's cook and its goal residuals are horizon.jl's at the same plan
#   the block  the plan's encode and decode round-trip per piece; the free set shrinks as events commit
#   two pieces a replan from a snapshot of a flat and a thin rack pulls the rack first, improves the goals over
#              the prior's plan, commits per piece as events fall due, and predicts each piece's outcomes with a width
#   julia --project=estimator estimator/test/test_plan_system.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test, ForwardDiff, Printf
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
prior_file = JSON.parsefile(joinpath(REPO, "data", "calibration", "prior_next_cook.json"))
prior_phys = read_sites(prior_file["prior"]); phys_names = String.(prior_file["names"])
pit = PitConfig(; T_amb=300.0, Tpit0=362.4, m0=7.0)
GRID = collect(0.0:600.0:(30 * 3600.0))                          # the plan grid: ten minutes, thirty hours
cfg1 = MeatConfig(; nx=10, T0=299.8)
goals = PlanGoals()
const ONLY = filter(!isempty, split(get(ENV, "TEST_PLAN_SYSTEM_ONLY", ""), ","))
run_set(name) = isempty(ONLY) || name in ONLY

"A snapshot of a system at its prior-centre latents, the state from a solve to `t_now` under a constant damper, the widths diagonal."
function snapshot_at(sys, blk, z, t_now; u0=0.15, events=CookEvent[])
    k = findfirst(τ -> abs(τ - t_now) < 1e-6, GRID)
    θs = [zeros(8) for _ in sys.pieces]; As = [pc.cfg.A_meat for pc in sys.pieces]
    p0, play = pack_params(sys, θs, As, vcat(zeros(6), 0.5), fill(u0, k); events)
    p = apply_latents(blk, z, p0, play)
    o = system_solve(sys, p, GRID[1:k], GRID[1:k])
    bl = belief_layout(sys)
    x = belief_from_system(o.X[:, k], bl); y = belief_encode(x, bl)
    n = length(z) + bl.n
    C = Matrix(Diagonal(vcat(fill(0.05^2, length(z)), fill(1e-4, bl.n))))
    past = Dict{Symbol,Past}()
    for (i, pc) in enumerate(sys.pieces)
        pa = Past(); pa.m0 = o.m_fuel[1]
        pulls = [e.t for e in events if e.target === pc.id && e.kind === :pull]
        t_pull = isempty(pulls) ? BrisketDynamics.PULL_NEVER : only(pulls)
        E._accumulate!(pa, (Tsurf=o.pieces[i].Tsurf, Tcore=o.pieces[i].Tcore), GRID[1:k], t_pull, goals)
        past[pc.id] = pa
    end
    h = belief_handoff(x, bl)
    return BeliefSnapshot(GRID[k], sys, bl, blk, collect(Float64, z), y, C, x, C[(length(z) + 1):end, (length(z) + 1):end], h, past, (n_nodes=1,)), o
end

run_set("one") && @testset "one slab: the system plan's cook and residuals are horizon.jl's at the same plan" begin
    @test_throws ArgumentError SystemControlGoals(w_fuel=-1.0)
    sys = CookSystem(Slab1D(:b; cfg=cfg1); pit)
    blk = system_latents(sys, prior_phys, phys_names)
    z = [z_moments(p)[1] for p in blk.sites]
    t_now = 2 * 3600.0
    snap, o = snapshot_at(sys, blk, z, t_now)
    k = findfirst(==(t_now), GRID)
    # horizon.jl's schedule and the system's, both at t_now with the same executed damper
    s_h = Schedule(GRID; u0=0.15); s_h.u[1:k] .= 0.15; s_h.t_now = t_now
    s_s = SystemSchedule(GRID, [:b]; u0=0.15); s_s.u[1:k] .= 0.15; s_s.t_now = t_now
    f_h = free_set(s_h); f_s = system_free_set(s_s)
    @test length(f_s) == length(f_h) == N_KNOTS + 1 + 1 + 1 + 3 + 1
    mζ_h, _ = plan_prior(f_h, s_h)
    p_h = decode_plan(mζ_h, f_h, s_h)
    # the same decisions in the system's block
    plan_s = (u=p_h.u, pieces=Dict(:b => (t_wrap=p_h.t_wrap, t_pull=p_h.t_pull, spritz=p_h.spritz)), t_add=p_h.t_add, t_serve=p_h.t_serve, t_last_pull=p_h.t_pull)
    ζ_s = encode_system_plan(plan_s, f_s, s_s)
    d_s = decode_system_plan(ζ_s, f_s, s_s)
    @test abs(d_s.pieces[:b].t_pull - p_h.t_pull) < 1e-6 && abs(d_s.pieces[:b].t_wrap - p_h.t_wrap) < 1e-6 && abs(d_s.t_serve - p_h.t_serve) < 1e-6
    @test abs(d_s.t_add - p_h.t_add) < 1e-6 && abs(d_s.pieces[:b].spritz.t - p_h.spritz.t) < 1e-6 && maximum(abs.(d_s.u .- p_h.u)) < 1e-9
    # the plan's cook: horizon.jl from the snapshot's state, the system from the same
    d = estimator_setup(GRID, fill(0.15, length(GRID)), pit, cfg1, prior_phys, phys_names)
    st = to_plan_state(snap, s_h)
    y_h = plan_outputs_rh(z, mζ_h, f_h, s_h, d; goals, state=st)
    ys, plan_d = system_plan_outputs(snap, z, ζ_s, f_s, s_s, Dict(:b => goals))
    y_s = ys[:b]
    @test length(y_s) == length(y_h)
    rel_y = abs.(y_s .- y_h) ./ (abs.(y_h) .+ 1e-3)
    @test maximum(rel_y) < 2e-3
    r_h = E.goal_residuals(y_h, goals); r_s = system_goal_residuals(ys, Dict(:b => goals), [:b])
    @test length(r_s) == length(r_h) && maximum(abs.(r_s .- r_h)) < 2e-3 * (1 + maximum(abs.(r_h)))
    # the damper on the grid agrees
    @test maximum(abs.(system_plan_damper(plan_d, s_s) .- plan_damper(p_h, s_h))) < 1e-9
    # the plan factor's Jacobian (ForwardDiff through the solve) against a difference quotient in the pull direction: the
    # same sign and size (the adaptive solver's response is rough at the 0.1 % level and the outcomes at serve are read by
    # piecewise-linear interpolation, so a difference quotient is no yardstick beyond that; test_segments.jl says the same)
    pf = PlanFactor(VarKey(:plan, :zeta, 0), snap, z, f_s, s_s, Dict(:b => goals), 1, 1, false, length(r_s), nothing)
    J = E.jacobians(pf, [ζ_s])[1]
    ip = N_KNOTS + 2
    ε = 1e-2; ζp = copy(ζ_s); ζp[ip] += ε; ζm = copy(ζ_s); ζm[ip] -= ε
    fd = (E.residual(pf, [ζp]) .- E.residual(pf, [ζm])) ./ (2ε)
    @test size(J) == (length(r_s), length(ζ_s)) && all(isfinite, J)
    imax = argmax(abs.(J[:, ip]))
    @test sign(fd[imax]) == sign(J[imax, ip]) && 0.5 < fd[imax] / J[imax, ip] < 2.0
    # State-only uncertainty is carried into the same plan factors: the latent portions of the draws
    # are identical, while opposite core-temperature state draws change the planning residual.
    nz, ny = length(snap.z), length(snap.y)
    Cstate = zeros(nz + ny, nz + ny)
    iy = first(E.belief_index(snap.bl, :b, :T))
    Cstate[nz + iy, nz + iy] = 0.02^2
    snap_state = BeliefSnapshot(snap.t, snap.sys, snap.bl, snap.blk, snap.z, snap.y, Cstate,
                                snap.x, snap.x_cov, snap.handoff, snap.past, snap.provenance)
    state_draws = belief_plan_draws(snap_state; n_dirs=1)
    @test length(state_draws) == 3
    @test state_draws[2][1:nz] == state_draws[3][1:nz] == snap.z
    pfp = PlanFactor(VarKey(:plan, :state_plus, 0), snap_state, state_draws[2], f_s, s_s,
                     Dict(:b => goals), 2, 3, true, length(r_s), nothing)
    pfm = PlanFactor(VarKey(:plan, :state_minus, 0), snap_state, state_draws[3], f_s, s_s,
                     Dict(:b => goals), 3, 3, true, length(r_s), nothing)
    @test norm(E.residual(pfp, [ζ_s]) - E.residual(pfm, [ζ_s])) > 1e-6
    pr_state = system_plan_predictive(snap_state, ζ_s, f_s, s_s, Dict(:b => goals))
    yp = system_plan_outputs(snap_state, state_draws[2][1:nz], ζ_s, f_s, s_s, Dict(:b => goals);
                             y=state_draws[2][(nz + 1):end])[1][:b]
    ym = system_plan_outputs(snap_state, state_draws[3][1:nz], ζ_s, f_s, s_s, Dict(:b => goals);
                             y=state_draws[3][(nz + 1):end])[1][:b]
    fd_sd = abs.(yp .- ym) ./ 2
    im = argmax(fd_sd)
    @test maximum(pr_state[:b].sd) > 1e-6 && fd_sd[im] > 1e-6
    @test isapprox(pr_state[:b].sd[im], fd_sd[im]; rtol=0.15, atol=1e-6)
end

run_set("multi") && @testset "two pieces: the thin rack is pulled first, the goals improve, the events commit per piece" begin
    flat = Slab1D(:flat; cfg=cfg1)
    rack = Slab1D(:rack; cfg=MeatConfig(; nx=8, geom_m=1.0, T0=301.0, A_meat=0.05))
    sys = CookSystem([flat, rack]; pit)
    blk = system_latents(sys, prior_phys, phys_names)
    z = [z_moments(p)[1] for p in blk.sites]
    z[latent_index(blk, "rack.L")] -= 0.6                                 # the rack about half the flat's thickness
    t_now = 3600.0
    snap, _ = snapshot_at(sys, blk, z, t_now)
    s = SystemSchedule(GRID, [:flat, :rack]; u0=0.15); s.u[1:7] .= 0.15; s.t_now = t_now
    f = system_free_set(s)
    @test f.damper && f.add && f.serve && all(f.wrap[p] && f.pull[p] && f.spritz[p] for p in (:flat, :rack))
    @test length(f) == N_KNOTS + 2 * 5 + 2
    gl = Dict(:flat => goals, :rack => goals)
    mζ, _ = system_plan_prior(f, s)
    L0 = system_plan_loss(snap, [z], decode_system_plan(mζ, f, s), s, gl; hedge=false).mean
    t0 = time()
    r = replan_system(snap, [z], s, gl; iters=30, scan_rest=(1.5, 3.0), scan_wrap=(0.55, 0.85), verbose=false)
    println("  the two-piece replan: ", length(r.hist), " iterations from ", r.n_starts, " starts, ", round(Int, time() - t0), " s; goals ", round(r.F_goals; digits=3), " from the prior's ", round(L0; digits=3))
    pl = r.plan
    @printf("  flat: wrap %.2f h pull %.2f h; rack: wrap %.2f h pull %.2f h; add %.2f h; serve %.2f h\n", pl.pieces[:flat].t_wrap / 3600, pl.pieces[:flat].t_pull / 3600,
            pl.pieces[:rack].t_wrap / 3600, pl.pieces[:rack].t_pull / 3600, pl.t_add / 3600, pl.t_serve / 3600)
    @test r.converged && r.F_goals < L0
    @test pl.pieces[:rack].t_pull < pl.pieces[:flat].t_pull
    @test pl.t_serve >= pl.t_last_pull && pl.t_last_pull == max(pl.pieces[:flat].t_pull, pl.pieces[:rack].t_pull)
    @test all(isfinite, r.ζ) && size(r.cov) == (length(f), length(f))
    # the standing plan's loss under the belief and its encode round trip
    ζ2 = encode_system_plan(pl, f, s)
    d2 = decode_system_plan(ζ2, f, s)
    @test abs(d2.pieces[:rack].t_pull - pl.pieces[:rack].t_pull) < 1.0 && abs(d2.t_serve - pl.t_serve) < 1.0
    L1 = system_plan_loss(snap, [z], pl, s, gl; hedge=false).mean
    @test abs(L1 - r.F_goals) < 1e-3 * (1 + L1)
    # Unequal family goals (including deliberately incompatible legacy control weights) remain invariant
    # to piece ordering because every piece is hedged independently and shared controls are system-level.
    gu = Dict(:flat => PlanGoals(c_target=0.93, w_tenderness=0.55, w_fuel=100.0, w_smooth=10.0),
              :rack => PlanGoals(c_target=0.82, w_tenderness=0.10, w_fuel=1e-4, w_smooth=1e-4))
    sys_rev = CookSystem([rack, flat]; pit)
    blk_rev = system_latents(sys_rev, prior_phys, phys_names)
    z_by_key = Dict(latent_key(blk, i) => z[i] for i in eachindex(z))
    z_rev = [z_by_key[latent_key(blk_rev, i)] for i in 1:length(blk_rev)]
    snap_rev, _ = snapshot_at(sys_rev, blk_rev, z_rev, t_now)
    s_rev = SystemSchedule(GRID, [:rack, :flat]; u0=0.15); s_rev.u[1:7] .= 0.15; s_rev.t_now = t_now
    L_order = system_plan_loss(snap, [z], pl, s, gu; hedge=false).mean
    L_reverse = system_plan_loss(snap_rev, [z_rev], pl, s_rev, gu; hedge=false).mean
    @test isapprox(L_order, L_reverse; rtol=2e-4, atol=2e-5)
    ys_shared = system_plan_outputs(snap, z, ζ2, f, s, gu)[1]
    r_legacy = system_goal_residuals(ys_shared, gu, [:flat, :rack])
    gu_control_changed = Dict(:flat => PlanGoals(c_target=0.93, w_tenderness=0.55, w_fuel=1e-8, w_smooth=1e8),
                              :rack => PlanGoals(c_target=0.82, w_tenderness=0.10, w_fuel=1e8, w_smooth=1e-8))
    @test r_legacy ≈ system_goal_residuals(ys_shared, gu_control_changed, [:flat, :rack])
    @test_throws ArgumentError SystemControlGoals(m0_ref=0.0)
    # commits: the interval to the rack's pull commits the rack's pull (and any earlier event), not the flat's
    Δ = pl.pieces[:rack].t_pull - s.t_now + 1.0
    done = commit_system!(s, pl, Δ)
    @test (:rack, :pull) in done && !((:flat, :pull) in done) && !isnan(s.t_pull[:rack]) && isnan(s.t_pull[:flat])
    f2 = system_free_set(s)
    @test !f2.pull[:rack] && f2.pull[:flat] && f2.damper
    ev = system_schedule_events(s)
    @test any(e -> e.kind === :pull && e.target === :rack, ev) && !any(e -> e.kind === :pull && e.target === :flat, ev)
    # A replan after the rack is already out uses a physically matching snapshot: its past wrap/pull
    # were present in the propagation and in the accumulated quality history. Its events stay fixed
    # while the flat, shared damper, add and serve decisions remain finite and free.
    partial_now = 11 * 3600.0
    rack_wrap = 7 * 3600.0
    rack_pull = 10 * 3600.0
    partial_events = [CookEvent(:wrap, :rack, rack_wrap), CookEvent(:pull, :rack, rack_pull)]
    snap_partial, _ = snapshot_at(sys, blk, z, partial_now; events=partial_events)
    sp = SystemSchedule(GRID, [:flat, :rack]; u0=0.15)
    sp.u[1:findfirst(==(partial_now), GRID)] .= 0.15
    sp.t_now = partial_now
    sp.t_pull[:rack] = rack_pull
    sp.t_wrap[:rack] = rack_wrap
    fp = system_free_set(sp)
    mp, _ = system_plan_prior(fp, sp)
    @test all(isfinite, mp) && !fp.pull[:rack] && fp.pull[:flat] && fp.damper
    warm_partial = encode_system_plan(pl, fp, sp)
    rp = replan_system(snap_partial, [z], sp, gl; ζ_init=warm_partial, iters=50,
                       scan_wrap=(), scan_rest=(), scan_damper=false, cold_start=false)
    @test rp.converged && all(isfinite, rp.ζ) && rp.plan.pieces[:rack].t_pull == sp.t_pull[:rack] && !isempty(rp.hist)
    @test_throws PlanConvergenceError replan_system(snap, [z], sp, gl; iters=0, scan_wrap=(), scan_rest=(),
                                                    scan_damper=false, cold_start=false)
    # With every piece already pulled and no fuel add, only the rest remains free and shared bookkeeping stays finite.
    sall = SystemSchedule(GRID, [:flat, :rack]; u0=0.15); sall.u[1:7] .= 0.15; sall.t_now = t_now
    sall.t_pull[:flat] = t_now - 60.0; sall.t_pull[:rack] = t_now - 120.0
    fall = system_free_set(sall)
    mall, _ = system_plan_prior(fall, sall)
    yall = system_plan_outputs(snap, z, mall, fall, sall, gl)[1]
    @test !fall.damper && !fall.add && all(!fall.pull[p] for p in (:flat, :rack))
    @test all(isfinite, mall) && all(isfinite, system_goal_residuals(yall, gl, [:flat, :rack]))
    # the replan policy over the pieces' events
    pol = ReplanPolicy()
    s3 = SystemSchedule(GRID, [:flat, :rack]; u0=0.15); s3.u[1:7] .= 0.15; s3.t_now = pl.pieces[:rack].t_pull - 600.0
    due, why = system_replan_due(pol, s3, pl, 0.1, 0.1, nothing)
    @test due && why === :event_due
    note_system_replanned!(s3, pl, pol)
    @test system_replan_due(pol, s3, pl, 0.1, 0.1, nothing) == (false, :hold)
    @test system_replan_due(pol, s3, pl, 2.0, 0.1, nothing) == (true, :loss_high)
    @test system_replan_due(pol, s3, nothing, 0.1, 0.1, nothing) == (true, :first)
    @test (:rack, :pull) in stable_system_events(pl, pl, 1.0)
    # the damper executed and the clock advanced
    s4 = SystemSchedule(GRID, [:flat, :rack]; u0=0.15); s4.u[1:7] .= 0.15; s4.t_now = t_now
    execute_system_damper!(s4, pl, 600.0)
    @test s4.t_now == t_now + 600.0 && !isnan(s4.u[8]) && isnan(s4.u[9])
    # the predictive: every piece's outcomes with a width from the snapshot's covariance
    pr = system_plan_predictive(snap, r.ζ, f, s4, gl)
    @test Set(keys(pr)) == Set([:flat, :rack]) && all(isfinite, pr[:flat].sd) && all(pr[:rack].sd .>= 0) && pr[:rack].sd[1] > 0
end
println("test_plan_system.jl: done")
