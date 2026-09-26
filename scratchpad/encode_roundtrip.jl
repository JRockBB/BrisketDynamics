# Does the standing plan survive its re-encoding as the warm start? At a record's check: the state replayed to
# that hour, the standing plan encoded with encode_plan and decoded back, side by side, with the planner's
# objective (goals + the plan prior) at the encoded point and the plan loss the trigger uses.
#   julia --project=estimator scratchpad/encode_roundtrip.jl <record.json> <hour>
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, JSON, LinearAlgebra, Printf, Random
const E = BrisketEstimator
REPO = normpath(joinpath(@__DIR__, ".."))
rec_name = ARGS[1]; hour = parse(Int, ARGS[2])
rec = JSON.parsefile(joinpath(REPO, "data", "mock", rec_name))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json")); gate = JSON.parsefile(joinpath(REPO, "data", "calibration", rec["prior_file"]))
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"])); cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
blk = latent_block(joinpath(REPO, "data", "calibration", rec["prior_file"]), cfg)
prior, prior_phys, names, allnames, idx = blk.sites, blk.prior_phys, blk.phys_names, blk.names, blk.idx
GRID = collect(0.0:120.0:30 * 3600.0); PLAN_GRID = collect(0.0:600.0:30 * 3600.0)
CHECK = Float64(rec["check_h"]) * 3600
d = estimator_setup(PLAN_GRID, fill(0.15, length(PLAN_GRID)), pit, cfg, prior_phys, names)
goals = PlanGoals()
h(x) = x === nothing ? NaN : Float64(x) * 3600
function plan_of(c)
    p = c["plan"]; sp = p["spritz"]
    (u=Float64.(p["u"]), t_wrap=h(p["t_wrap_h"]), t_pull=h(p["t_pull_h"]), t_serve=h(p["t_serve_h"]), t_add=h(p["t_add_h"]),
     spritz=sp === nothing ? nothing : (t=h(sp["t_h"]), dt=h(sp["dt_h"]), m=Float64(sp["m"])))
end
s = Schedule(PLAN_GRID; u0=0.15, t_serve=14 * 3600.0)
for c in rec["checks"]
    c["t_h"] >= hour && break
    global s
    s.t_now = Float64(c["t_h"]) * 3600
    plan = plan_of(c)
    for ev in c["committed"]
        ev == "wrap" && (s.t_wrap = plan.t_wrap); ev == "add" && (s.t_add = plan.t_add); ev == "pull" && (s.t_pull = plan.t_pull); ev == "spritz" && (s.spritz = plan.spritz)
    end
    s.t_now > 0 && (s.t_serve = plan.t_serve)
    E.execute_damper!(s, plan, CHECK)
end
kc = findfirst(c -> c["t_h"] == hour, rec["checks"])
ck = rec["checks"][kc]; prev = rec["checks"][kc - 1]
z_hat = Float64.(ck["posterior_z"])
f = free_set(s)
t_add_s = isnan(s.t_add) ? BrisketDynamics.ADD_NEVER : s.t_add
info = get(ck, "estimator", nothing)
hist_b = nothing
if info !== nothing && haskey(info, "z_stack")
    moving = smoother_moving(idx, t_add_s)
    specs = [(sidx, [findlast(GRID .<= hh * 3600 + 1e-9) for hh in Float64.(info["starts_h"][allnames[sidx]])]) for (sidx, _, _) in moving]
    global hist_b = plan_history(Float64.(info["z_stack"]), DriftLayout(length(prior), specs), GRID, PLAN_GRID, prior)
end
p0 = plan_of(prev)
ζ = encode_plan(p0, f, s)
p1 = decode_plan(ζ, f, s)
show_plan(lab, p) = @printf("  %-14s wrap %.3f pull %.3f serve %.3f add %s spritz %s; knots %s\n", lab, p.t_wrap / 3600, p.t_pull / 3600, p.t_serve / 3600,
                            isnan(p.t_add) ? "-" : @sprintf("%.3f", p.t_add / 3600), p.spritz === nothing ? "-" : @sprintf("%.2f h", p.spritz.t / 3600),
                            join([@sprintf("%.3f", v) for v in p.u], " "))
@printf("state at %d h: t_now %.1f h, wrap %s add %s pull %s serve %.2f; free damper %s wrap %s pull %s serve %s spritz %s add %s\n", hour, s.t_now / 3600,
        isnan(s.t_wrap) ? "-" : string(s.t_wrap / 3600), isnan(s.t_add) ? "-" : string(s.t_add / 3600), isnan(s.t_pull) ? "-" : string(s.t_pull / 3600), s.t_serve / 3600,
        f.damper, f.wrap, f.pull, f.serve, f.spritz, f.add)
show_plan("standing", p0)
show_plan("re-encoded", p1)
mζ, Λζ = plan_prior(f, s)
Lg(p) = 0.5 * sum(abs2, E.goal_residuals(plan_outputs_rh(z_hat, encode_plan(p, f, s), f, s, d; goals, history=hist_b), goals))
@printf("  plan loss (goals only): standing %.4f, re-encoded %.4f; the plan prior at the encoded point %.4f; ζ = %s\n", Lg(p0), Lg(p1),
        0.5 * sum(abs2, sqrt(Λζ) * (ζ .- mζ)), string(round.(ζ; digits=2)))
# the executed damper on the plan grid around now, and what plan_damper gives for the standing plan
u_std = plan_damper(p0, s)
k = searchsortedlast(PLAN_GRID, s.t_now)
@printf("  damper from plan_damper(standing) at now-1h..now+3h: %s\n", join([@sprintf("%.2f", v) for v in u_std[max(1, k - 6):min(end, k + 18)]], " "))
