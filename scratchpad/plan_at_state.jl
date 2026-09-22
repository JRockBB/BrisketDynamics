# From the closed loop's own mid-cook states (the executed damper and events to then): the planner at the posterior
# mean versus the planner at the truth, each plan scored at the truth. If the truth's plan scores much better on
# the plant, the estimate limits the cook; if not, the optimiser or the landscape from that state does.
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, JSON, LinearAlgebra, Printf
const E = BrisketEstimator
REPO = normpath(joinpath(@__DIR__, ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json")); gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
rec = JSON.parsefile(joinpath(REPO, "data", "mock", "closed_loop_realcook.json"))
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"])); cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = read_sites(gate["prior"])
PLAN_GRID = collect(0.0:600.0:30 * 3600.0)
d = estimator_setup(PLAN_GRID, fill(0.15, length(PLAN_GRID)), pit, cfg, prior_phys, String.(gate["names"]))
z_true = Float64.(rec["z_true"])
goals = PlanGoals()
ex = rec["executed"]
ev_h(k) = ex[k] === nothing ? NaN : Float64(ex[k]) * 3600
function state_at(hour)
    s = Schedule(PLAN_GRID; u0=0.15)
    s.u .= [v === nothing ? NaN : Float64(v) for v in ex["u"]]; s.u[PLAN_GRID .> hour * 3600] .= NaN; s.t_now = hour * 3600.0
    t_wrap, t_add = ev_h("t_wrap_h"), ev_h("t_add_h")
    !isnan(t_wrap) && t_wrap <= s.t_now && (s.t_wrap = t_wrap)
    !isnan(t_add) && t_add <= s.t_now && (s.t_add = t_add)
    sp = ex["spritz"]
    sp !== nothing && Float64(sp["t_h"]) * 3600 <= s.t_now && (s.spritz = (t=Float64(sp["t_h"]) * 3600, dt=Float64(sp["dt_h"]) * 3600, m=Float64(sp["m"])))
    return s
end
function score(lab, plan, s, f)
    ζ = encode_plan(plan, f, s)
    out = String[]
    for (zl, z) in (("truth", z_true), ("belief", nothing))
        z === nothing && continue
        y = plan_outputs_rh(z, ζ, f, s, d; goals); T = objective_terms(y, goals)
        push!(out, @sprintf("J at the %s %.3f (tend %.3f app %.3f taste %.3f burn %.3f fuel %.3f) collagen %.3f juic %.3f L* %.1f smoke %.2f", zl, T.J, T.tenderness, T.appearance, T.taste, T.surf, T.fuel, y[1], y[6], y[3], y[5]))
    end
    @printf("  %-26s %s\n  %-26s knots %s; wrap %.1f pull %.1f serve %.1f spritz %s add %s\n", lab, join(out, " | "), "",
            join([@sprintf("%.2f", v) for v in plan.u], " "), plan.t_wrap / 3600, plan.t_pull / 3600, plan.t_serve / 3600,
            plan.spritz === nothing ? "-" : @sprintf("%.1f h %.3f kg", plan.spritz.t / 3600, plan.spritz.m), isnan(plan.t_add) ? "-" : @sprintf("%.1f", plan.t_add / 3600))
end
for hour in (4, 8, 11)
    ck = rec["checks"][hour + 1]
    z_hat = Float64.(ck["posterior_z"])
    s = state_at(hour); f = free_set(s)
    e = [abs(x) for x in ck["err_sd"]]
    @printf("\nfrom the state at %d h (posterior median |err|/sd %.2f, max %.2f; free: damper %s wrap %s pull %s spritz %s add %s):\n", hour,
            sort(e)[div(end + 1, 2)], maximum(e), f.damper, f.wrap, f.pull, f.spritz, f.add)
    t0 = time(); r_b = replan_on_graph([z_hat], s, d; goals, iters=30); tb = time() - t0
    t0 = time(); r_t = replan_on_graph([z_true], s, d; goals, iters=30); tt = time() - t0
    yb = plan_outputs_rh(z_hat, r_b.ζ, f, s, d; goals)
    @printf("  planner at the belief: predicted J %.3f (%.0f s); planner at the truth (%.0f s):\n", objective_terms(yb, goals).J, tb, tt)
    score("the belief's plan", r_b.plan, s, f)
    score("the truth's plan", r_t.plan, s, f)
    flush(stdout)
end
