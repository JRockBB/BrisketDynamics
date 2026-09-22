# Recover a pilot record's outcome from its steps. The loop (mock_closed_loop_realcook.jl) writes its steps
# before the plant's outcome, so a failure in the outcome block leaves a record marked "partial" with every
# step but no "actual". This replays the executed schedule from the steps — the events each step committed and
# the damper each step executed — on the same plant (the mock's truth, drifting from the record's seed when it
# says so), and writes the record the loop would have written. The outcome block mirrors the loop's.
#
# Run:  julia --project=estimator estimator/scripts/pilot_outcome.jl <record.json>
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
length(ARGS) >= 1 || error("which record? julia --project=estimator estimator/scripts/pilot_outcome.jl <record.json>")
out_name = ARGS[1]
dst = joinpath(REPO, "data", "mock", out_name)
rec = JSON.parsefile(dst)
get(rec, "partial", false) == true || (println(out_name, " is not partial; nothing to recover"); exit(0))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json"))
prior_file = rec["prior_file"]
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
blk = latent_block(joinpath(REPO, "data", "calibration", prior_file), cfg)
sites, prior_phys, allnames, idx = blk.sites, blk.prior_phys, blk.names, blk.idx
GRID = collect(0.0:120.0:30 * 3600.0)
goals = PlanGoals()
pol = ReplanPolicy()
nc = RealCookNoise()
x_true = Float64.(m["truth"]["x"])
θ_true, φ_true, A_true, depth_fraction_true = x_true[1:8], x_true[9:15], x_true[16], x_true[17]
depth_true = probe_depth(θ_true[4], depth_fraction_true)
z_true0 = vcat(to_z(x_true[1:15], prior_phys), log(A_true), to_z(depth_fraction_true, sites[blk.depth]))
drift = rec["drift"]
DRIFT_SEED = drift === nothing ? 0 : Int(drift["seed"])
dd = DRIFT_SEED > 0 ? DriftDraw(MersenneTwister(2000 + DRIFT_SEED), idx) : nothing
nn(x) = isnan(x) ? nothing : x

# --- the schedule replayed from the steps ---------------------------------------------------------
s = Schedule(GRID; u0=0.15, t_serve=14 * 3600.0)
h(x) = x === nothing ? NaN : Float64(x) * 3600
for c in rec["checks"]
    global s
    k = Int(c["k"])
    s.u[k] = Float64(c["u"])
    s.t_now = GRID[k]
    p = c["plan"]
    sp = p["spritz"]
    plan = (u=Float64.(p["u"]), t_wrap=h(p["t_wrap_h"]), t_pull=h(p["t_pull_h"]), t_serve=h(p["t_serve_h"]), t_add=h(p["t_add_h"]),
            spritz=sp === nothing ? nothing : (t=h(sp["t_h"]), dt=h(sp["dt_h"]), m=Float64(sp["m"])))
    for ev in c["committed"]
        ev == "wrap" && (s.t_wrap = plan.t_wrap)
        ev == "add" && (s.t_add = plan.t_add)
        ev == "pull" && (s.t_pull = plan.t_pull)
        ev == "spritz" && (s.spritz = plan.spritz)
    end
    Bool(c["updated"]) && (s.t_serve = plan.t_serve)         # commit! sets the serve at every update
end
record = rec["checks"]
n_replans = rec["n_replans"]

# --- the plant's actual outcome under the executed schedule (the loop's block) -----------------------
ev = schedule_events(s)
ug = [isnan(x) ? s.u[findlast(!isnan, s.u)] : x for x in s.u]
o = dd === nothing ? monolith_solve(θ_true, φ_true, ug, GRID, GRID; pit, cfg, A_meat=A_true, ev...) :
    solve_drifting(z_true0, dd, prior_phys, ug, GRID; pit, cfg, A_meat=A_true, ev...)
nx = size(o.W_field, 1); w = E.fv_weights(nx)
at(y, tq) = (k = clamp(searchsortedlast(GRID, tq), 1, length(GRID) - 1); f = (tq - GRID[k]) / (GRID[k + 1] - GRID[k]); (1 - f) * y[k] + f * y[k + 1])
y_act = cook_outputs(o, GRID, executed_du(GRID, ug, s.t_pull), s.t_pull, isnan(s.t_add) ? BrisketDynamics.ADD_NEVER : s.t_add, s.t_serve, cfg, goals)
terms = objective_terms(y_act, goals)
actual = (J=terms.J, tenderness=terms.tenderness, appearance=terms.appearance, taste=terms.taste, surf=terms.surf, fuel=terms.fuel, smooth=terms.smooth, safety=terms.safety,
          C_serve=y_act[1], C_pull=at(view(o.C_field, 1, :), s.t_pull), T_pull=at(o.Tcore, s.t_pull) - 273.15, W=y_act[2], L=y_act[3], T_serve=y_act[4] - 273.15,
          S=y_act[5], juiciness=y_act[6])
reasons = Dict(r => count(rec -> rec["reason"] == r, record) for r in unique(rec["reason"] for rec in record))
@printf("executed: wrap %.2f h, pull %.2f h, serve %.2f h, spritz %s, fuel add %.2f h; %d steps, %d replans (%s)\n",
        s.t_wrap / 3600, s.t_pull / 3600, s.t_serve / 3600,
        s.spritz === nothing ? "-" : @sprintf("(%.1f h, every %.2f h, %.3f kg)", s.spritz.t / 3600, s.spritz.dt / 3600, s.spritz.m),
        s.t_add / 3600, length(record), n_replans, join(["$k $v" for (k, v) in sort(collect(reasons))], ", "))
@printf("the plant's actual outcome: J %.3f (tenderness %.3f, appearance %.3f, taste %.3f, surf %.3f, fuel %.3f, smooth %.3f, safety %.3f); collagen at serve %.3f (%.3f at the pull, core %.1f C), water %.3f, juiciness %.3f, L* %.1f, smoke %.2f, core at serve %.1f C\n",
        actual.J, actual.tenderness, actual.appearance, actual.taste, actual.surf, actual.fuel, actual.smooth, actual.safety,
        actual.C_serve, actual.C_pull, actual.T_pull, actual.W, actual.juiciness, actual.L, actual.S, actual.T_serve)
probe = [E.clean_at(o, k; A=A_true, L=x_true[4], depth=depth_true).core - 273.15 for k in eachindex(GRID)]
open(dst, "w") do io
    JSON.print(io, merge(Dict(k => v for (k, v) in rec if k != "partial"),
                         Dict("partial" => false, "recovered" => true,
                              "policy" => Dict(string(k) => getfield(pol, k) for k in fieldnames(ReplanPolicy)),
                              "goals" => Dict(string(k) => getfield(goals, k) for k in fieldnames(PlanGoals)),
                              "executed" => Dict("t_wrap_h" => s.t_wrap / 3600, "t_pull_h" => s.t_pull / 3600, "t_serve_h" => s.t_serve / 3600,
                                                 "t_add_h" => nn(s.t_add / 3600),
                                                 "spritz" => s.spritz === nothing ? nothing : Dict("t_h" => s.spritz.t / 3600, "dt_h" => s.spritz.dt / 3600, "m" => s.spritz.m),
                                                 "u" => [nn(x) for x in s.u], "t_h" => s.t ./ 3600),
                              "actual" => Dict(string(k) => getfield(actual, k) for k in fieldnames(typeof(actual))),
                              "plant" => Dict("t_h" => GRID ./ 3600, "pit" => collect(o.Tpit) .- 273.15, "centre" => collect(o.Tcore) .- 273.15, "surface" => collect(o.Tsurf) .- 273.15,
                                              "probe" => probe, "stepped_core_err_K" => nothing,
                                              "description" => dd === nothing ? "monolith (1-D) at the mock's truth, real-cook instruments (2-min grid)" :
                                                               "monolith (1-D) at the mock's truth drifting (drift.jl, seed $DRIFT_SEED), real-cook instruments (2-min grid)"),
                              "noise_truth" => Dict(String(k) => getfield(nc, k) for k in fieldnames(RealCookNoise)),
                              "z_true" => z_true0, "names" => allnames)))
end
println("wrote data/mock/", out_name, " (recovered from its steps)")
