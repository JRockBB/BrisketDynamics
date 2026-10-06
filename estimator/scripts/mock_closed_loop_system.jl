# The pilot on the real-cook plant, flying the NEW backend and the multi-meat planner (docs/multi-meat-inference-plan.md,
# the application's migration): the plant is the monolith at the mock's true latents, stepped one grid point at a
# time with its state handed off exactly and read by the real-cook instruments (realcook.jl); the estimator is the
# square-root fixed-lag smoother over a `CookSystem` of one slab (online.jl), assimilating every arrival at its own
# time and handing the planner a `BeliefSnapshot`; the planner is `replan_system` (plan_system.jl) on the same
# factor infrastructure, deciding rather than ticking under `ReplanPolicy` (a replan on a raised or high loss, an
# event about to commit, or the misfit check; never on a clock). The record holds every step with the belief,
# every decision with its reason, the snapshot's provenance, and the plant's actual outcome, in the shape of
# `mock_closed_loop_realcook.jl`'s record so `pilot_outcome.jl` and the plots read it.
#
# Run:  julia --project=estimator estimator/scripts/mock_closed_loop_system.jl [prior.json] [out.json] [lag] [stop_h]
#   prior.json (data/calibration): the prior the pilot starts from (estimator_gate_real.json by default)
#   lag: the smoother's window behind the newest node (3 by default)
#   stop_h > 0: stop the loop at that hour with a partial record
# Writes data/mock/<out.json> (closed_loop_system.json by default).
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf, Statistics
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json"))
prior_file = length(ARGS) >= 1 ? ARGS[1] : "estimator_gate_real.json"
out_name = length(ARGS) >= 2 ? ARGS[2] : "closed_loop_system.json"
LAG = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
STOP_H = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.0
println("prior: ", prior_file, "; record: ", out_name, "; lag: ", LAG, STOP_H > 0 ? "; stop at $(STOP_H) h" : "")
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
blk_old = latent_block(joinpath(REPO, "data", "calibration", prior_file), cfg)
GRID = collect(0.0:120.0:30 * 3600.0)
DT = GRID[2] - GRID[1]
goals = PlanGoals()
pol = ReplanPolicy()
nc = RealCookNoise()
PIECE = :brisket
sys = CookSystem(Slab1D(PIECE; cfg); pit)
blk = system_latents(sys, blk_old.prior_phys, blk_old.phys_names; probes=[PIECE])
n_z = length(blk)
AREA, DEPTH = latent_index(blk, "A_meat"), latent_index(blk, "core_depth_fraction")

# --- the plant: the mock's truth, stepped with its state handed off, read like a real cook ------------
x_true = Float64.(m["truth"]["x"])
θ_true, φ_true, A_true, depth_fraction_true = x_true[1:8], x_true[9:15], x_true[16], x_true[17]
L_true = θ_true[4]
depth_true = probe_depth(L_true, depth_fraction_true)
z_true = vcat(to_z(x_true[1:15], blk_old.prior_phys), log(A_true), to_z(depth_fraction_true, blk.sites[DEPTH]))
nn(x) = isnan(x) ? nothing : x
inst = Instruments(GRID, nc, MersenneTwister(23))
mutable struct Plant
    ics::NamedTuple
    a::Float64
    k::Int
    Tpit::Vector{Float64}; Tcore::Vector{Float64}; Tsurf::Vector{Float64}
    clean::Vector{Any}
end
"The executed events of the system schedule as the monolith's keywords."
function mono_events(s::SystemSchedule)
    sp = s.spritz[PIECE]
    return (t_wrap=isnan(s.t_wrap[PIECE]) ? BrisketDynamics.WRAP_NEVER : s.t_wrap[PIECE], t_pull=isnan(s.t_pull[PIECE]) ? BrisketDynamics.PULL_NEVER : s.t_pull[PIECE],
            t_add=isnan(s.t_add) ? BrisketDynamics.ADD_NEVER : s.t_add, t_spritz=sp === nothing ? BrisketDynamics.SPRITZ_NEVER : sp.t,
            dt_spritz=sp === nothing ? 3600.0 : sp.dt, m_spritz=sp === nothing ? 0.05 : sp.m)
end
function plant_init(s::SystemSchedule)
    o = monolith_solve(θ_true, φ_true, [s.u[1], s.u[1]], GRID[1:2], GRID[1:2]; pit, cfg, A_meat=A_true, mono_events(s)...)
    ics, a = handoff(o, 1)
    c = clean_at(o, 1; A=A_true, L=L_true, depth=depth_true)
    return Plant(ics, a, 1, [o.Tpit[1]], [o.Tcore[1]], [o.Tsurf[1]], Any[c])
end
function plant_step!(pl::Plant, s::SystemSchedule)
    k = pl.k
    φ = vcat(φ_true[1:6], k == 1 ? φ_true[7] : pl.a)
    o = monolith_solve(θ_true, φ, s.u[k:(k + 1)], GRID[k:(k + 1)], GRID[k:(k + 1)]; pit, cfg, A_meat=A_true, pl.ics..., mono_events(s)...)
    pl.ics, pl.a = handoff(o, 2)
    pl.k = k + 1
    push!(pl.Tpit, o.Tpit[2]); push!(pl.Tcore, o.Tcore[2]); push!(pl.Tsurf, o.Tsurf[2])
    push!(pl.clean, clean_at(o, 2; A=A_true, L=L_true, depth=depth_true))
    return pl
end
function plant_whole(s::SystemSchedule)
    ug = [isnan(x) ? s.u[findlast(!isnan, s.u)] : x for x in s.u]
    return monolith_solve(θ_true, φ_true, ug, GRID, GRID; pit, cfg, A_meat=A_true, mono_events(s)...)
end

# --- the estimator: the smoother over the system, its log the executed schedule ---------------------------
np0 = realcook_noise_priors(joinpath(REPO, "data", "calibration", "noise_prior_real.json"), DT)
tau(npr) = -DT / log(clamp(npr.ρ_mean, 1e-6, 0.999))
sensors = [SensorSpec(:pit, PitTemperature(), NoiseModel(white_sd=0.1, ar_sd=np0[1].σ_mean, tau=tau(np0[1]))),
           SensorSpec(:core, TemperatureAt(PIECE; depth_latent=:core_depth_fraction), NoiseModel(white_sd=0.1, ar_sd=np0[2].σ_mean, tau=tau(np0[2]))),
           SensorSpec(:humidity, ChamberVapor(), NoiseModel(white_sd=10.0, ar_sd=np0[3].σ_mean, tau=tau(np0[3]), cap=nc.hum_cap)),
           SensorSpec(:mass, TotalMass(), NoiseModel(white_sd=0.005, ar_sd=np0[4].σ_mean, tau=tau(np0[4])))]
s = SystemSchedule(GRID, [PIECE]; u0=0.15, t_serve=14 * 3600.0)
clog = CookLog(GRID, s.u; sensors)                 # the log's damper follows the schedule's executed damper step by step, its events as they commit
# The SHIPPED estimator configuration, passed explicitly (the StateNoise defaults remain: the bench has
# not earned this backend's rates — the calibration round replaces this with the fingerprinted file).
SHIPPED = E.SHIPPED_STREAM_CONFIG
model = SystemModel(sys, clog, blk; sn=StateNoise(), abstol=SHIPPED.abstol, reltol=SHIPPED.reltol,
                    floor_abstol=SHIPPED.floor_abstol, floor_reltol=SHIPPED.floor_reltol)
sm = online_init(model; lag=LAG, goals, attach=SHIPPED.attach, node_ratio=SHIPPED.node_ratio,
                 max_slice_s=SHIPPED.max_slice_s, lag_max=max(LAG, SHIPPED.lag_max), prior_relin=SHIPPED.prior_relin)
pl = plant_init(s)
gl = Dict(PIECE => goals)

plan = nothing
L_plan = Inf
record = []
n_replans = 0
t_start = time()
snap = belief_snapshot(sm)
let
    r = replan_system(snap, s, gl; n_dirs=1, ζ_init=nothing, verbose=false)
    global plan = r.plan
    global L_plan = system_plan_loss(snap, plan, s, gl; n_dirs=1, hedge=false).mean
    global n_replans += 1
    @printf("the first plan, from the prior: wrap %.2f / pull %.2f / serve %.2f h, goals %.3f (%.0f s)\n", plan.pieces[PIECE].t_wrap / 3600, plan.pieces[PIECE].t_pull / 3600,
            plan.t_serve / 3600, L_plan, time() - t_start)
    flush(stdout)
end
plan_dict(p) = Dict("u" => p.u, "t_wrap_h" => nn(p.pieces[PIECE].t_wrap / 3600), "t_pull_h" => p.pieces[PIECE].t_pull / 3600, "t_serve_h" => p.t_serve / 3600,
                    "spritz" => p.pieces[PIECE].spritz === nothing ? nothing : Dict("t_h" => p.pieces[PIECE].spritz.t / 3600, "dt_h" => p.pieces[PIECE].spritz.dt / 3600, "m" => p.pieces[PIECE].spritz.m),
                    "t_add_h" => nn(p.t_add / 3600))
save_record(partial::Bool, extra::Dict) = open(joinpath(REPO, "data", "mock", out_name), "w") do io
    JSON.print(io, merge(Dict("checks" => record, "n_draws" => 1, "n_replans" => n_replans, "prior_file" => prior_file, "partial" => partial,
                              "estimator" => "the square-root fixed-lag smoother over a CookSystem (online.jl), lag $LAG; the planner replan_system (plan_system.jl)",
                              "grid_dt_s" => DT, "state_noise" => Dict(string(k) => getfield(StateNoise(), k) for k in fieldnames(StateNoise))), extra))
end
println("t[h]  reason (* = the replan was worse than the standing plan, which was kept)  committed        L_now  L_plan  plan: wrap / pull / serve / spritz / add   nodes  wall")
flush(stdout)
for k in 2:length(GRID)
    t0 = time()
    execute_system_damper!(s, plan, DT)
    u_exec = collect(sm.model.log.u); u_exec[k] = s.u[k]
    update_execution!(sm; u=u_exec)                                   # future factors use a new immutable execution record
    plant_step!(pl, s)
    rd, present = read_at(inst, k, pl.clean[k], mono_events(s))
    recs = MeasurementRecord[]
    for (id, key) in ((:pit, :pit), (:core, :core), (:humidity, :humidity), (:mass, :mass))
        present[key] && push!(recs, MeasurementRecord(id, GRID[k], rd[key]))
    end
    rec = isempty(recs) ? nothing : assimilate!(sm, recs)
    updated = rec !== nothing && rec.converged
    why, due, kept, done, stable, L_now = :propagated, false, false, Tuple{Symbol,Symbol}[], Tuple{Symbol,Symbol}[], NaN
    if updated
        global snap = belief_snapshot(sm)
        L_now = system_plan_loss(snap, plan, s, gl; n_dirs=1, hedge=false).mean
        due, why = system_replan_due(pol, s, plan, L_now, L_plan, nothing)
        if due
            ζ_init = encode_system_plan(plan, system_free_set(s), s)
            r = try
                replan_system(snap, s, gl; n_dirs=1, ζ_init, verbose=false)
            catch err
                err isa PlanConvergenceError || rethrow()
                nothing
            end
            if r === nothing
                kept = true
                why = :planner_failed
            else
                L_new = system_plan_loss(snap, r.plan, s, gl; n_dirs=1, hedge=false).mean
                kept = L_new > L_now + 1e-6
                why == :event_due && (stable = stable_system_events(kept ? plan : r.plan, plan, pol.stable))
                kept || (global plan = r.plan)
                why == :event_due && note_system_replanned!(s, plan, pol)
                global L_plan = kept ? L_now : L_new
            end
            global n_replans += 1
        end
        done = commit_system!(s, plan, DT; force=stable, force_horizon=pol.stable_horizon)
        if !isempty(done)
            update_execution!(sm; events=system_schedule_events(s))              # new immutable event record for future factors
        end
    end
    p = plan
    push!(record, Dict("t_h" => s.t_now / 3600, "k" => k, "updated" => updated, "channels" => [string(r.sensor) for r in recs], "reason" => string(why), "replanned" => due,
                       "kept_standing" => kept, "committed" => [string(d[1]) * ":" * string(d[2]) for d in done], "L_now" => nn(L_now), "L_plan" => L_plan, "wall_s" => time() - t0,
                       "u" => s.u[k],
                       "T_pit" => rec === nothing ? nothing : [rec.T_pit[1] - 273.15, rec.T_pit[2]],
                       "T_centre" => rec === nothing ? nothing : [rec.pieces[PIECE].T_centre[1] - 273.15, rec.pieces[PIECE].T_centre[2]],
                       "T_surface" => rec === nothing ? nothing : [rec.pieces[PIECE].T_surface[1] - 273.15, rec.pieces[PIECE].T_surface[2]],
                       "posterior_z" => rec === nothing ? nothing : rec.z, "posterior_sd" => rec === nothing ? nothing : sqrt.(diag(rec.z_cov)),
                       "z_true" => z_true, "err_sd" => rec === nothing ? nothing : (rec.z .- z_true) ./ sqrt.(diag(rec.z_cov)),
                       "innovation" => rec === nothing ? nothing : Dict(string(k2) => v for (k2, v) in rec.innovation),
                       "readings" => Dict(string(key) => (present[key] ? (key in (:pit, :core) ? rd[key] - 273.15 : rd[key]) : nothing) for key in (:pit, :core, :humidity, :mass)),
                       "plan" => plan_dict(p), "provenance" => Dict(string(k2) => v for (k2, v) in pairs(snap.provenance)),
                       "converged" => rec === nothing ? nothing : rec.converged, "iters" => rec === nothing ? 0 : rec.iters, "fit_wall_s" => rec === nothing ? 0.0 : rec.wall))
    if due || !isempty(done) || floor(s.t_now / 3600 + 1e-9) > floor((s.t_now - DT) / 3600 + 1e-9)
        @printf("%5.2f  %-15s  %-15s  %5.2f  %5.2f  %5.2f / %5.2f / %5.2f / %s / %5.2f   %d   %3.0f s\n",
                s.t_now / 3600, string(why) * (kept ? "*" : ""), isempty(done) ? "-" : join([string(d[1]) * ":" * string(d[2]) for d in done], ","),
                L_now, L_plan, (isnan(p.pieces[PIECE].t_wrap) ? NaN : p.pieces[PIECE].t_wrap / 3600), p.pieces[PIECE].t_pull / 3600, p.t_serve / 3600,
                p.pieces[PIECE].spritz === nothing ? "-" : @sprintf("%.1f", p.pieces[PIECE].spritz.t / 3600), (isnan(p.t_add) ? NaN : p.t_add / 3600),
                length(sm.nodes), time() - t0)
        flush(stdout)
    end
    (due || !isempty(done) || k % 15 == 0) && save_record(true, Dict())
    !isnan(s.t_pull[PIECE]) && s.t_now + 1e-9 >= s.t_serve && break
    if STOP_H > 0 && s.t_now + 1e-9 >= STOP_H * 3600
        save_record(true, Dict())
        @printf("stopped at %.1f h as asked: a partial record (%d steps, %d updates, %d replans)\n", s.t_now / 3600, length(record), count(r["updated"] for r in record), n_replans)
        exit(0)
    end
end

# --- the plant's actual outcome under the executed schedule --------------------------------------
o = plant_whole(s)
kend = pl.k
step_err = maximum(abs.(pl.Tcore[1:kend] .- o.Tcore[1:kend]))
ug = [isnan(x) ? s.u[findlast(!isnan, s.u)] : x for x in s.u]
ev = mono_events(s)
y_act = cook_outputs(o, GRID, executed_du(GRID, ug, ev.t_pull), ev.t_pull, ev.t_add, s.t_serve, cfg, goals)
terms = objective_terms(y_act, goals)
@printf("\nexecuted: wrap %.2f h, pull %.2f h, serve %.2f h, fuel add %.2f h; %d steps, %d updates, %d replans; %.0f s total\n",
        s.t_wrap[PIECE] / 3600, s.t_pull[PIECE] / 3600, s.t_serve / 3600, s.t_add / 3600, length(record), count(r["updated"] for r in record), n_replans, time() - t_start)
@printf("the plant's actual outcome: J %.3f (tenderness %.3f, appearance %.3f, taste %.3f, surf %.3f, fuel %.3f, smooth %.3f, safety %.3f)\n",
        terms.J, terms.tenderness, terms.appearance, terms.taste, terms.surf, terms.fuel, terms.smooth, terms.safety)
save_record(false, Dict("belief" => Dict("z" => snap.z, "sd" => sqrt.(diag(snap.zy_cov)[1:n_z]), "t_h" => snap.t / 3600),
                        "policy" => Dict(string(k) => getfield(pol, k) for k in fieldnames(ReplanPolicy)),
                        "goals" => Dict(string(k) => getfield(goals, k) for k in fieldnames(PlanGoals)),
                        "executed" => Dict("t_wrap_h" => nn(s.t_wrap[PIECE] / 3600), "t_pull_h" => nn(s.t_pull[PIECE] / 3600), "t_serve_h" => s.t_serve / 3600, "t_add_h" => nn(s.t_add / 3600),
                                           "u" => [nn(x) for x in s.u], "t_h" => s.t ./ 3600),
                        "actual" => Dict("J" => terms.J, "tenderness" => terms.tenderness, "appearance" => terms.appearance, "taste" => terms.taste, "surf" => terms.surf,
                                         "fuel" => terms.fuel, "smooth" => terms.smooth, "safety" => terms.safety, "C_serve" => y_act[1], "W" => y_act[2], "L" => y_act[3], "T_serve" => y_act[4] - 273.15),
                        "plant" => Dict("t_h" => GRID ./ 3600, "pit" => collect(o.Tpit) .- 273.15, "centre" => collect(o.Tcore) .- 273.15, "surface" => collect(o.Tsurf) .- 273.15,
                                        "stepped_core_err_K" => step_err, "description" => "monolith (1-D) at the mock's truth, real-cook instruments (2-min grid)"),
                        "z_true" => z_true, "names" => blk.names))
println("wrote data/mock/", out_name)
