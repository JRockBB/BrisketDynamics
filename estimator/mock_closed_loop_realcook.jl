# The pilot on the real-cook plant, flying the filter on the graph and deciding rather than ticking. The plant is
# the monolith at the mock's true latents (constant, or drifting from a seed: drift.jl), advanced one grid step
# at a time with its state handed off exactly, read by instruments at their own rates on a two-minute grid (the
# pit and core probes every two minutes, the humidity every ten and capped, the scale every half hour, the lid
# lifts dropped, the pit noise level following the fire: realcook.jl). The estimator is the filter
# (sequential.jl): the belief over the plant's state, the latents and the instruments' discrepancies, updated
# at every grid point where a sample arrives and only propagated where none does; every channel's noise
# identified on the AR node when its series has grown by a correlation time, its level at every sample. After
# every update the standing plan is re-scored from the filtered state (one plant run, no optimisation) and a
# replan fires only when its value has changed (its loss high, or risen since it was made), when a planned
# event is about to commit, or when the misfit check fires (horizon.jl's ReplanPolicy: no floor, no clock). The
# plan solves the FUTURE from the filtered state; the past enters through the filter's integrals. The record
# holds every step with the belief and every decision with its reason, and the plant's actual outcome.
#
# Run:  julia --project=estimator estimator/mock_closed_loop_realcook.jl [prior.json] [out.json] [hedge dirs] [drift seed] [fault_h] [stop_h]
#   prior.json (data/calibration): the prior the pilot starts from; the gate's literature prior by default,
#   prior_next_cook.json for the calibration chain's posterior by the pool's roles (export_chain_prior.py)
#   hedge dirs: 0 = the plan at the posterior mean (the default, docs/results.md); n = the hedge over sigma points
#   drift seed > 0: the plant's leak, surface coefficients and the add's charge move (drift.jl, drawn once from the seed)
#   fault_h > 0: the core probe sticks at its last reading from that hour on (the level isolates it)
#   stop_h > 0: stop the loop at that hour with a partial record (the end-to-end check; pilot_outcome.jl scores it)
# Writes data/mock/<out.json> (closed_loop_realcook.json by default).
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf, Statistics
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json"))
prior_file = length(ARGS) >= 1 ? ARGS[1] : "estimator_gate_real.json"
out_name = length(ARGS) >= 2 ? ARGS[2] : "closed_loop_realcook.json"
HEDGE_DIRS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0
DRIFT_SEED = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 0
FAULT_H = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.0
STOP_H = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 0.0
N_DRAWS = 2 * HEDGE_DIRS
println("prior: ", prior_file, "; record: ", out_name, "; hedge directions: ", HEDGE_DIRS, "; plant: ", DRIFT_SEED > 0 ? "drifting (seed $DRIFT_SEED)" : "constant",
        FAULT_H > 0 ? "; the core probe stuck from $(FAULT_H) h" : "")
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
blk = latent_block(joinpath(REPO, "data", "calibration", prior_file), cfg)
sites, allnames, prior_phys, names, idx = blk.sites, blk.names, blk.prior_phys, blk.phys_names, blk.idx
AREA, DEPTH = blk.area, blk.depth
n_z = length(sites)
GRID = collect(0.0:120.0:30 * 3600.0)          # one grid: the plant, the filter and the plan
DT = GRID[2] - GRID[1]
goals = PlanGoals()
pol = ReplanPolicy()
nc = RealCookNoise()
sn = load_state_noise()                            # the state's process noise as the bench earned it (data/calibration/state_noise.json)

# --- the plant: the mock's truth, stepped with its state handed off, read like a real cook ------------
x_true = Float64.(m["truth"]["x"])
θ_true, φ_true, A_true, depth_true = x_true[1:8], x_true[9:15], x_true[16], x_true[17]
L_true = θ_true[4]
z_true0 = vcat(to_z(x_true[1:15], prior_phys), log(A_true), log(depth_true))
dd = DRIFT_SEED > 0 ? DriftDraw(MersenneTwister(2000 + DRIFT_SEED), idx) : nothing     # the plant's drift, drawn once
"The plant's latent block in force just before `te` (what a belief at `te` is scored against)."
truth_now(te, t_add) = dd === nothing ? z_true0 : truth_at(z_true0, dd, te, t_add)
"The plant's drift for the record: the seed, the rates and the block in force from every hour under the schedule's add."
drift_record(s::Schedule) = dd === nothing ? nothing :
    Dict("seed" => DRIFT_SEED, "sd_per_hour" => DRIFT_SD_PER_HOUR, "jump_log_sd" => JUMP_LOG_SD,
         "z_by_hour" => [truth_at(z_true0, dd, h * 3600.0, schedule_events(s).t_add; from=true) for h in 0:29])
nn(x) = isnan(x) ? nothing : x
inst = Instruments(GRID, nc, MersenneTwister(23))
mutable struct Plant
    ics::NamedTuple
    a::Float64
    k::Int
    Tpit::Vector{Float64}; Tcore::Vector{Float64}; Tsurf::Vector{Float64}
    clean::Vector{Any}
end
"The block the plant runs on from grid point `k` (the drift's hourly walk, the add's jump from the grid point at or before the executed add)."
function plant_block(k::Int, s::Schedule)
    dd === nothing && return z_true0
    ta = s.t_add
    t_add_grid = isnan(ta) ? BrisketDynamics.ADD_NEVER : GRID[findlast(GRID .<= ta + 1e-9)]
    return truth_at(z_true0, dd, GRID[k], t_add_grid; from=true)
end
function plant_init(s::Schedule)
    x = from_z(plant_block(1, s), sites)
    o = monolith_solve(x[1:8], x[9:15], [s.u[1], s.u[1]], GRID[1:2], GRID[1:2]; pit, cfg, A_meat=A_true, schedule_events(s)...)
    ics, a = handoff(o, 1)
    c = clean_at(o, 1; A=A_true, L=L_true, depth=depth_true)
    return Plant(ics, a, 1, [o.Tpit[1]], [o.Tcore[1]], [o.Tsurf[1]], Any[c])
end
"The plant advanced one grid step under the executed damper and events, its state handed off exactly."
function plant_step!(pl::Plant, s::Schedule)
    k = pl.k
    x = from_z(plant_block(k, s), sites)
    φ = vcat(x[9:14], k == 1 ? x[15] : pl.a)
    o = monolith_solve(x[1:8], φ, s.u[k:(k + 1)], GRID[k:(k + 1)], GRID[k:(k + 1)]; pit, cfg, A_meat=A_true, pl.ics..., schedule_events(s)...)
    pl.ics, pl.a = handoff(o, 2)
    pl.k = k + 1
    push!(pl.Tpit, o.Tpit[2]); push!(pl.Tcore, o.Tcore[2]); push!(pl.Tsurf, o.Tsurf[2])
    push!(pl.clean, clean_at(o, 2; A=A_true, L=L_true, depth=depth_true))
    return pl
end
"The whole cook under the executed schedule in one solve (the plant's outcome, and the stepped plant's check)."
function plant_whole(s::Schedule)
    ev = schedule_events(s)
    ug = [isnan(x) ? s.u[findlast(!isnan, s.u)] : x for x in s.u]
    dd === nothing && return monolith_solve(θ_true, φ_true, ug, GRID, GRID; pit, cfg, A_meat=A_true, ev...)
    return solve_drifting(z_true0, dd, prior_phys, ug, GRID; pit, cfg, A_meat=A_true, ev...)
end

# --- the estimator: the filter on the registry's channels, filled as the samples arrive ----------------
np0 = realcook_noise_priors(joinpath(REPO, "data", "calibration", "noise_prior_real.json"), DT)
noise = [np0[i] for i in 1:4]
chnames = ["pit", "core", "humidity", "mass"]
empty = (pit=fill(NaN, length(GRID)), core=fill(NaN, length(GRID)), humidity=fill(NaN, length(GRID)), mass=fill(NaN, length(GRID)))
nomask = (pit=falses(length(GRID)), core=falses(length(GRID)), humidity=falses(length(GRID)), mass=falses(length(GRID)))
chs = realcook_channels(empty, nomask, np0; cap=nc.hum_cap, depth_latent=DEPTH)
dr = drift_rates(blk.file)     # the moving sites' rates the filter is told: earlier cooks' (next_cook.py) or the modelling choice
q = zeros(n_z); for (nm, sd) in dr.sd_per_hour; q[idx[nm]] = sd^2; end
"The filter's jumps under the executed schedule: the add's charge at the add, the probe's depth at the wrap (a re-seat)."
function filter_jumps(s::Schedule)
    j = Tuple{Int,Float64,Float64}[]
    isnan(s.t_add) || append!(j, [(idx[nm], s.t_add, sd^2) for (nm, sd) in dr.jump_log_sd])
    isnan(s.t_wrap) || push!(j, (DEPTH, s.t_wrap, 0.3^2))
    return j
end

# --- the loop ------------------------------------------------------------------------------------
s = Schedule(GRID; u0=0.15, t_serve=14 * 3600.0)
d_plan = estimator_setup(GRID, fill(0.15, length(GRID)), pit, cfg, prior_phys, names)   # the plan physics reads its prior, pit, cfg, tolerances
st = filter_init(chs, GRID, s.u, pit, cfg, sites, allnames, schedule_events(s); noise, area_latent=AREA, sn, q, jumps=filter_jumps(s), goals)
pl = plant_init(s)
draws(z, V) = HEDGE_DIRS == 0 ? [copy(z)] : sigma_draws(z, V; n_dirs=HEDGE_DIRS)     # the mean first
belief() = (st.m[1:n_z], Matrix(Symmetric(st.P[1:n_z, 1:n_z])))
"The plan's predicted outcomes at serve under the joint belief of the latents and the state."
function predicted(ζ, f, stt)
    z, V = belief()
    pr = plan_predictive(z, V, ζ, f, s, d_plan; goals, state=stt)
    return (mean=pr.y.mean, sd=pr.y.sd, t_h=s.t_now / 3600)
end
plan_dict(p) = Dict("u" => p.u, "t_wrap_h" => nn(p.t_wrap / 3600), "t_pull_h" => p.t_pull / 3600, "t_serve_h" => p.t_serve / 3600,
                    "spritz" => p.spritz === nothing ? nothing : Dict("t_h" => p.spritz.t / 3600, "dt_h" => p.spritz.dt / 3600, "m" => p.spritz.m),
                    "t_add_h" => nn(p.t_add / 3600))
plan = nothing
L_plan = Inf
pred = nothing
frecs = NamedTuple[]           # the filter's records (the misfit reads the trailing window)
record = []
n_replans = 0
t_start = time()
# the first plan: from the prior's belief at the recipe's state, a warm start for the cook, never a commit
let
    z, V = belief()
    stt = state_now(st)
    r = replan_on_graph(draws(z, V), s, d_plan; goals, ζ_init=nothing, verbose=false, state=stt)
    global plan = r.plan
    global L_plan = plan_loss(draws(z, V), plan, s, d_plan; goals, state=stt).mean
    global pred = predicted(r.ζ, r.free, stt)
    global n_replans += 1
    @printf("the first plan, from the prior: wrap %.2f / pull %.2f / serve %.2f h, loss %.3f (%.0f s)\n", plan.t_wrap / 3600, plan.t_pull / 3600, plan.t_serve / 3600, L_plan, time() - t_start)
    flush(stdout)
end
println("t[h]  reason (* = the replan was worse than the standing plan, which was kept)  committed        L_now  L_plan  plan: wrap / pull / serve / spritz / add   predicted at the last replan (collagen at serve, water, L*, serve C)   misfit z   levels   wall")
flush(stdout)
save_record(partial::Bool, extra::Dict) = open(joinpath(REPO, "data", "mock", out_name), "w") do io
    JSON.print(io, merge(Dict("checks" => record, "n_draws" => N_DRAWS, "n_replans" => n_replans, "prior_file" => prior_file, "partial" => partial,
                              "estimator" => "the filter on the graph (sequential.jl): updates on arrival", "drift" => drift_record(s), "fault_h" => FAULT_H,
                              "grid_dt_s" => DT, "state_noise" => Dict(string(k) => getfield(sn, k) for k in fieldnames(StateNoise))), extra))
end
for k in 2:length(GRID)
    t0 = time()
    execute_damper!(s, plan, DT)                                  # the damper for this step, the clock to it
    plant_step!(pl, s)
    ev = schedule_events(s)
    rd, present = read_at(inst, k, pl.clean[k], ev)
    if FAULT_H > 0 && GRID[k] >= FAULT_H * 3600 && present.core
        kl = findlast(j -> chs[2].mask[j], 1:(k - 1))
        kl === nothing || (rd = (rd..., core=chs[2].data[kl]))         # the core probe stuck at its last reading
    end
    for (c, key) in enumerate((:pit, :core, :humidity, :mass))
        chs[c].mask[k] = present[key]
        present[key] && (chs[c].data[k] = rd[key])
    end
    rec = filter_update!(st, k; diagnostics=true)
    push!(frecs, rec)
    why, due, kept, done, stable, L_now, mis = :propagated, false, false, Symbol[], Symbol[], NaN, nothing
    if rec.updated
        z, V = belief()
        stt = state_now(st)
        zs = draws(z, V)
        L_now = plan_loss(zs, plan, s, d_plan; goals, state=stt).mean
        mis = innovation_misfit(frecs; window_s=3000.0, t_now=s.t_now)
        due, why = replan_due(pol, s, plan, L_now, L_plan, mis === nothing ? nothing : mis.z)
        if due
            ζ_init = encode_plan(plan, free_set(s), s)
            r = replan_on_graph(zs, s, d_plan; goals, ζ_init, verbose=false, state=stt)
            L_new = plan_loss(zs, r.plan, s, d_plan; goals, state=stt).mean
            # a plan the pilot holds is never replaced by a worse one under the same belief
            kept = L_new > L_now + 1e-6
            why == :event_due && (stable = stable_events(kept ? plan : r.plan, plan, pol.stable))   # left where it stood: commit it
            kept || (global plan = r.plan)
            global L_plan = kept ? L_now : L_new
            global n_replans += 1
            global pred = predicted(kept ? ζ_init : r.ζ, r.free, stt)
        end
        done = commit!(s, plan, DT; force=stable, force_horizon=pol.stable_horizon)
        if !isempty(done)
            st.events = schedule_events(s)
            st.jumps = filter_jumps(s)
        end
    end
    p = plan
    sdz = rec.z_sd
    zt = truth_now(s.t_now, ev.t_add)
    push!(record, Dict("t_h" => s.t_now / 3600, "k" => k, "updated" => rec.updated, "channels" => rec.channels, "reason" => string(why), "replanned" => due, "kept_standing" => kept,
                       "committed" => string.(done), "stable" => string.(stable), "L_now" => nn(L_now), "L_plan" => L_plan, "wall_s" => time() - t0,
                       "u" => s.u[k],
                       "T_pit" => [rec.T_pit[1] - 273.15, rec.T_pit[2]], "T_centre" => [rec.T_centre[1] - 273.15, rec.T_centre[2]], "T_surface" => [rec.T_surface[1] - 273.15, rec.T_surface[2]],
                       "posterior_z" => rec.z, "posterior_sd" => sdz, "z_true" => zt, "err_sd" => (rec.z .- zt) ./ sdz,
                       "disc" => rec.disc, "level" => rec.level, "faulty" => rec.faulty,
                       "noise" => Dict(chnames[c] => Dict("sigma" => v.σ, "rho" => v.ρ, "sigma_sd" => v.σ_sd, "rho_sd" => v.ρ_sd) for (c, v) in enumerate(rec.noise)),
                       "nis" => isfinite(rec.nis) ? rec.nis : nothing, "dof" => rec.dof, "misfit_z" => mis === nothing ? nothing : mis.z,
                       "readings" => Dict(chnames[c] => (present[key] ? (c <= 2 ? rd[key] - 273.15 : rd[key]) : nothing) for (c, key) in enumerate((:pit, :core, :humidity, :mass))),
                       "plan" => plan_dict(p),
                       "predicted" => due ? Dict("mean" => pred.mean, "sd" => [nn(x) for x in pred.sd], "t_h" => pred.t_h) : nothing,
                       "converged" => rec.converged, "iters" => rec.iters, "fit_wall_s" => rec.wall))
    if due || !isempty(done) || floor(s.t_now / 3600 + 1e-9) > floor((s.t_now - DT) / 3600 + 1e-9)
        @printf("%5.2f  %-15s  %-15s  %5.2f  %5.2f  %5.2f / %5.2f / %5.2f / %s / %5.2f   %5.3f±%5.3f %5.3f±%5.3f %5.1f±%4.1f %6.1f±%4.1f   %s   %s   %3.0f s\n",
                s.t_now / 3600, string(why) * (kept ? "*" : ""), isempty(done) ? "-" : join(string.(done), ","),
                L_now, L_plan,
                (isnan(p.t_wrap) ? NaN : p.t_wrap / 3600), p.t_pull / 3600, p.t_serve / 3600,
                p.spritz === nothing ? "-" : @sprintf("%.1f", p.spritz.t / 3600), (isnan(p.t_add) ? NaN : p.t_add / 3600),
                pred.mean[1], pred.sd[1], pred.mean[2], pred.sd[2], pred.mean[3], pred.sd[3], pred.mean[4] - 273.15, pred.sd[4],
                mis === nothing ? "  -  " : @sprintf("%5.2f", mis.z), join([@sprintf("%.1f", l) for l in rec.level], " ") * (any(rec.faulty) ? " FAULT" : ""), time() - t0)
        flush(stdout)
    end
    # the steps so far, after every one: a crash or a kill later never loses them (pilot_outcome.jl recovers the
    # outcome from a partial record); the record is built before the file is opened
    (due || !isempty(done) || k % 15 == 0) && save_record(true, Dict())
    !isnan(s.t_pull) && s.t_now + 1e-9 >= s.t_serve && break
    if STOP_H > 0 && s.t_now + 1e-9 >= STOP_H * 3600
        save_record(true, Dict())
        @printf("stopped at %.1f h as asked: a partial record (%d steps, %d updates, %d replans); pilot_outcome.jl scores it\n",
                s.t_now / 3600, length(record), count(r["updated"] for r in record), n_replans)
        exit(0)
    end
end

# --- the plant's actual outcome under the executed schedule --------------------------------------
o = plant_whole(s)
kend = pl.k
step_err = maximum(abs.(pl.Tcore[1:kend] .- o.Tcore[1:kend]))
@printf("the stepped plant against the whole-cook solve: core within %.3f K over %d steps\n", step_err, kend)
nx = size(o.W_field, 1); w = E.fv_weights(nx)
at(y, tq) = (k = clamp(searchsortedlast(GRID, tq), 1, length(GRID) - 1); f = (tq - GRID[k]) / (GRID[k + 1] - GRID[k]); (1 - f) * y[k] + f * y[k + 1])
ug = [isnan(x) ? s.u[findlast(!isnan, s.u)] : x for x in s.u]
y_act = cook_outputs(o, GRID, executed_du(GRID, ug, s.t_pull), s.t_pull, isnan(s.t_add) ? BrisketDynamics.ADD_NEVER : s.t_add, s.t_serve, cfg, goals)
terms = objective_terms(y_act, goals)
actual = (J=terms.J, tenderness=terms.tenderness, appearance=terms.appearance, taste=terms.taste, surf=terms.surf, fuel=terms.fuel, smooth=terms.smooth, safety=terms.safety,
          C_serve=y_act[1], C_pull=at(view(o.C_field, 1, :), s.t_pull), T_pull=at(o.Tcore, s.t_pull) - 273.15, W=y_act[2], L=y_act[3], T_serve=y_act[4] - 273.15,
          S=y_act[5], juiciness=y_act[6])
reasons = Dict(r => count(rec -> rec["reason"] == r, record) for r in unique(rec["reason"] for rec in record))
@printf("\nexecuted: wrap %.2f h, pull %.2f h, serve %.2f h, spritz %s, fuel add %.2f h; %d steps, %d updates, %d replans (%s); %.0f s total\n",
        s.t_wrap / 3600, s.t_pull / 3600, s.t_serve / 3600,
        s.spritz === nothing ? "-" : @sprintf("(%.1f h, every %.2f h, %.3f kg)", s.spritz.t / 3600, s.spritz.dt / 3600, s.spritz.m),
        s.t_add / 3600, length(record), count(r["updated"] for r in record), n_replans, join(["$k $v" for (k, v) in sort(collect(reasons))], ", "), time() - t_start)
@printf("the plant's actual outcome: J %.3f (tenderness %.3f, appearance %.3f, taste %.3f, surf %.3f, fuel %.3f, smooth %.3f, safety %.3f); collagen at serve %.3f (%.3f at the pull, core %.1f C), water %.3f, juiciness %.3f, L* %.1f, smoke %.2f, core at serve %.1f C\n",
        actual.J, actual.tenderness, actual.appearance, actual.taste, actual.surf, actual.fuel, actual.smooth, actual.safety,
        actual.C_serve, actual.C_pull, actual.T_pull, actual.W, actual.juiciness, actual.L, actual.S, actual.T_serve)
save_record(false, Dict("belief" => Dict("z" => st.m[1:n_z], "sd" => sqrt.(diag(st.P)[1:n_z]), "cov" => [st.P[i, 1:n_z] for i in 1:n_z], "t_h" => s.t_now / 3600,
                                         "noise" => Dict(chnames[c] => Dict("sigma" => v.σ_mean, "rho" => v.ρ_mean, "sigma_sd" => v.σ_sd, "rho_sd" => v.ρ_sd) for (c, v) in enumerate(st.np)),
                                         "drift_rates_used" => dr.sd_per_hour),
                        "policy" => Dict(string(k) => getfield(pol, k) for k in fieldnames(ReplanPolicy)),
                        "goals" => Dict(string(k) => getfield(goals, k) for k in fieldnames(PlanGoals)),
                        "executed" => Dict("t_wrap_h" => s.t_wrap / 3600, "t_pull_h" => s.t_pull / 3600, "t_serve_h" => s.t_serve / 3600,
                                           "t_add_h" => nn(s.t_add / 3600),
                                           "spritz" => s.spritz === nothing ? nothing : Dict("t_h" => s.spritz.t / 3600, "dt_h" => s.spritz.dt / 3600, "m" => s.spritz.m),
                                           "u" => [nn(x) for x in s.u], "t_h" => s.t ./ 3600),
                        "actual" => Dict(string(k) => getfield(actual, k) for k in fieldnames(typeof(actual))),
                        "plant" => Dict("t_h" => GRID ./ 3600, "pit" => collect(o.Tpit) .- 273.15, "centre" => collect(o.Tcore) .- 273.15, "surface" => collect(o.Tsurf) .- 273.15,
                                        "probe" => [c.core - 273.15 for c in pl.clean], "stepped_core_err_K" => step_err,
                                        "description" => dd === nothing ? "monolith (1-D) at the mock's truth, real-cook instruments (2-min grid)" :
                                                         "monolith (1-D) at the mock's truth drifting (drift.jl, seed $DRIFT_SEED), real-cook instruments (2-min grid)"),
                        "noise_truth" => Dict(String(k) => getfield(nc, k) for k in fieldnames(RealCookNoise)),
                        "z_true" => z_true0, "names" => allnames))
println("wrote data/mock/", out_name)
