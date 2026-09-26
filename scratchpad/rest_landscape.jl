using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "src", "BrisketEstimator.jl"))
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
z = Float64.(rec["checks"][end]["posterior_z"])
ex = rec["executed"]
s = Schedule(PLAN_GRID; u0=0.15)
s.u .= [v === nothing ? NaN : Float64(v) for v in ex["u"]]; s.t_now = 13 * 3600.0     # the state at the pull's commit
s.t_wrap = ex["t_wrap_h"] * 3600; s.t_add = ex["t_add_h"] * 3600; sp = ex["spritz"]; s.spritz = (t=sp["t_h"] * 3600, dt=sp["dt_h"] * 3600, m=sp["m"])
goals = PlanGoals()
f = free_set(s)
base = (u=[0.16, 0.16, 0.16], t_wrap=s.t_wrap, t_pull=ex["t_pull_h"] * 3600, t_serve=ex["t_serve_h"] * 3600, spritz=s.spritz, t_add=s.t_add)
println("J on the final posterior mean, from the state at 13 h (wrap, add, spritz executed); pull x rest grid:")
@printf("%12s", "rest [h] ->"); for r in (1.0, 1.5, 2.5, 3.5, 4.5, 6.0); @printf("%9.1f", r); end; println()
for tp in (12.0, 13.0, 13.5, 14.25, 15.0)
    @printf("pull %5.2f h ", tp)
    for r in (1.0, 1.5, 2.5, 3.5, 4.5, 6.0)
        plan = (base..., t_pull=tp * 3600, t_serve=(tp + r) * 3600)
        ζ = encode_plan(plan, f, s)
        y = plan_outputs_rh(z, ζ, f, s, d; goals)
        @printf("%9.3f", objective_terms(y, goals).J)
    end
    println()
end
plan = (base..., t_pull=14.25 * 3600, t_serve=15.75 * 3600); y = plan_outputs_rh(z, encode_plan(plan, f, s), f, s, d; goals); T = objective_terms(y, goals)
@printf("executed (14.25, 1.5): J %.3f: tenderness %.3f taste %.3f (juiciness %.3f, collagen %.3f) appearance %.3f\n", T.J, T.tenderness, T.taste, y[6], y[1], T.appearance)
plan = (base..., t_pull=12.0 * 3600, t_serve=16.5 * 3600); y = plan_outputs_rh(z, encode_plan(plan, f, s), f, s, d; goals); T = objective_terms(y, goals)
@printf("the MPC's shape (12.0, 4.5): J %.3f: tenderness %.3f taste %.3f (juiciness %.3f, collagen %.3f) appearance %.3f\n", T.J, T.tenderness, T.taste, y[6], y[1], T.appearance)
