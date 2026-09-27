# The multi-meat planner (plan_system.jl):
#   parity     for a system of one slab the plan's cook and its goal residuals are horizon.jl's at the same plan
#   the block  the plan's encode and decode round-trip per piece; the free set shrinks as events commit
#   two pieces a replan from a snapshot of a flat and a thin rack pulls the rack first, improves the goals over
#              the prior's plan, commits per piece as events fall due, and predicts each piece's outcomes with a width
#   julia --project=estimator estimator/test/test_plan_system.jl
using Pkg
Pkg.activate(joinpath("C:/Users/jmurp/.julia/dev/BrisketDynamics/estimator/test", ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath("C:/Users/jmurp/.julia/dev/BrisketDynamics/estimator/test", "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test, ForwardDiff, Printf
const E = BrisketEstimator

REPO = normpath(joinpath("C:/Users/jmurp/.julia/dev/BrisketDynamics/estimator/test", "..", ".."))
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

begin
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
    # --- the gate's two numbers, per output, under the ignited fraction's own floor (the tree) and under the unit floor (before)
    ys0, _ = system_plan_outputs(snap_state, snap.z, ζ_s, f_s, s_s, Dict(:b => goals))
    names = collect(keys(ys0[:b]))
    iign = first(E.belief_index(snap.bl, :pit, :ignited))
    function gate(label)
        pr = system_plan_predictive(snap_state, ζ_s, f_s, s_s, Dict(:b => goals))
        yp = system_plan_outputs(snap_state, state_draws[2][1:nz], ζ_s, f_s, s_s, Dict(:b => goals); y=state_draws[2][(nz + 1):end])[1][:b]
        ym = system_plan_outputs(snap_state, state_draws[3][1:nz], ζ_s, f_s, s_s, Dict(:b => goals); y=state_draws[3][(nz + 1):end])[1][:b]
        fd = abs.(collect(yp) .- collect(ym)) ./ 2
        sdl = collect(pr[:b].sd)
        i = argmax(fd)
        println(label, ": the snapshot's ignited coordinate ", round(snap_state.y[iign]; digits=3), " (a = ", E._belief_dec(snap_state.y[iign], snap.bl.kinds[iign]), ")")
        println("  the gate's output: ", names[i], " linearized sd ", round(sdl[i]; digits=1), " central-difference sd ", round(fd[i]; digits=1), " ratio ", round(sdl[i] / fd[i]; digits=3))
        for j in eachindex(names)
            (fd[j] > 1e-6 || sdl[j] > 1e-6) && println("    ", rpad(string(names[j]), 16), " lin ", lpad(round(sdl[j]; sigdigits=4), 10), "  fd ", lpad(round(fd[j]; sigdigits=4), 10), "  ratio ", round(sdl[j] / max(fd[j], 1e-12); digits=3))
        end
    end
    gate("the tree (kind :ignited)")
    # the same snapshot with the unit floor: re-encode the physical state under the old kind and redo the draws
    snap.bl.kinds[iign] = :unit
    y_old = belief_encode(belief_decode(snap.y, snap.bl), snap.bl)
    snap_state = BeliefSnapshot(snap.t, snap.sys, snap.bl, snap.blk, snap.z, y_old, Cstate, snap.x, snap.x_cov, snap.handoff, snap.past, snap.provenance)
    state_draws = belief_plan_draws(snap_state; n_dirs=1)
    gate("before (kind :unit)")
end
