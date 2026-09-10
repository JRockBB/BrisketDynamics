# Is the pilot's gap to the MPC the estimate or the optimiser? At the TRUTH, from a fresh start (t = 0):
#   (a) J of the MPC's executed schedule as a seven-knot plan  (b) J of the pilot's executed schedule as a plan
#   (c) the planner from its default start  (d) the planner from the MPC-like warm start
# If (d) ends well below (c), the objective is multi-modal in the damper and the scan needs damper candidates.
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, JSON, LinearAlgebra, Printf
const E = BrisketEstimator
REPO = normpath(joinpath(@__DIR__, ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json")); gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
rec = JSON.parsefile(joinpath(REPO, "data", "mock", "closed_loop_realcook.json"))
mp = JSON.parsefile(joinpath(REPO, "data", "control", "mpc_mock1d_traces_mpc_nominal_trace.json"))
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"])); cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
PLAN_GRID = collect(0.0:600.0:30 * 3600.0)
d = estimator_setup(PLAN_GRID, fill(0.15, length(PLAN_GRID)), pit, cfg, prior_phys, String.(gate["names"]))
z_true = Float64.(rec["z_true"])
goals = PlanGoals()
interp(tt, uu, ti) = ti >= tt[end] ? uu[end] : (k = searchsortedlast(tt, ti); k < 1 ? uu[1] : k >= length(tt) ? uu[end] :
                      uu[k] + (uu[k + 1] - uu[k]) * (ti - tt[k]) / (tt[k + 1] - tt[k]))
knots_from(tt, uu, t_pull) = [clamp(interp(tt, uu, fk * t_pull), 0.02, 0.98) for fk in E.knot_f(N_KNOTS)]

# the MPC's executed cook as a plan
es = mp["events_s"]
t_pull_m = Float64(es["pull"]); tm = Float64.(mp["t_h"]) .* 3600; um = Float64.(mp["u"])
sp_m = (es["spritz"] === nothing || es["spritz_dose"] === nothing || es["spritz_dose"] <= 0) ? (t=0.5 * t_pull_m, dt=3600.0, m=0.005) :
       (t=Float64(es["spritz"]), dt=es["dt_spritz"] === nothing ? 3600.0 : Float64(es["dt_spritz"]), m=Float64(es["spritz_dose"]))
plan_mpc = (u=knots_from(tm, um, t_pull_m), t_wrap=Float64(es["wrap"]), t_pull=t_pull_m, t_serve=Float64(es["serve"]), spritz=sp_m,
            t_add=es["add"] === nothing ? 0.98 * t_pull_m : Float64(es["add"]))
# the pilot's executed cook as a plan
ex = rec["executed"]
u10 = [v === nothing ? NaN : Float64(v) for v in ex["u"]]; u10 = [isnan(v) ? u10[findlast(!isnan, u10)] : v for v in u10]
t10 = Float64.(ex["t_h"]) .* 3600
t_pull_r = Float64(ex["t_pull_h"]) * 3600; spr = ex["spritz"]
plan_rx = (u=knots_from(t10, u10, t_pull_r), t_wrap=Float64(ex["t_wrap_h"]) * 3600, t_pull=t_pull_r, t_serve=Float64(ex["t_serve_h"]) * 3600,
           spritz=spr === nothing ? (t=0.5 * t_pull_r, dt=3600.0, m=0.005) : (t=Float64(spr["t_h"]) * 3600, dt=Float64(spr["dt_h"]) * 3600, m=Float64(spr["m"])),
           t_add=ex["t_add_h"] === nothing ? 0.98 * t_pull_r : Float64(ex["t_add_h"]) * 3600)

s = Schedule(PLAN_GRID; u0=0.15)
f = free_set(s)
function report(lab, plan)
    ζ = encode_plan(plan, f, s)
    y = plan_outputs_rh(z_true, ζ, f, s, d; goals); T = objective_terms(y, goals)
    @printf("%-34s J %.3f (tenderness %.3f appearance %.3f taste %.3f burn %.3f fuel %.3f smooth %.3f safety %.3f)  collagen %.3f juiciness %.3f L* %.1f smoke %.2f serve %.0f C\n",
            lab, T.J, T.tenderness, T.appearance, T.taste, T.surf, T.fuel, T.smooth, T.safety, y[1], y[6], y[3], y[5], y[4] - 273.15)
    @printf("%-34s knots %s; wrap %.1f h pull %.1f h serve %.1f h spritz %.1f h every %.1f h %.3f kg add %.1f h\n", "",
            join([@sprintf("%.2f", v) for v in plan.u], " "), plan.t_wrap / 3600, plan.t_pull / 3600, plan.t_serve / 3600, plan.spritz.t / 3600, plan.spritz.dt / 3600, plan.spritz.m, plan.t_add / 3600)
    return ζ
end
println("at the truth, everything free (t = 0):")
ζ_mpc = report("(a) the MPC's cook as a plan", plan_mpc)
ζ_rx = report("(b) the pilot's cook as a plan", plan_rx)
flush(stdout)
t0 = time(); r_c = replan_on_graph([z_true], s, d; goals, iters=30); @printf("(c) planner, default start: F %.3f (half-J with prior), %d starts, %.0f s\n", r_c.F, r_c.n_starts, time() - t0)
report("(c) the planner's own plan", r_c.plan); flush(stdout)
t0 = time(); r_d = replan_on_graph([z_true], s, d; goals, ζ_init=ζ_mpc, iters=30); @printf("(d) planner, MPC warm start: F %.3f, %d starts, %.0f s\n", r_d.F, r_d.n_starts, time() - t0)
report("(d) from the MPC warm start", r_d.plan)
