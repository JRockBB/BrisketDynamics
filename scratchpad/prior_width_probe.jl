# The planner at the opening state (the prior-mean belief) under two widths of the plan prior: the basins it
# lands in and the goals-only loss, verbose.  julia --project=estimator scratchpad/prior_width_probe.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "src", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, JSON, LinearAlgebra, Printf
const E = BrisketEstimator
REPO = normpath(joinpath(@__DIR__, ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json")); gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "prior_next_cook.json"))
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"])); cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = read_sites(gate["prior"])
PLAN_GRID = collect(0.0:600.0:30 * 3600.0)
d = estimator_setup(PLAN_GRID, fill(0.15, length(PLAN_GRID)), pit, cfg, prior_phys, String.(gate["names"]))
z = vcat(d.z0, [log(cfg.A_meat), z_center(probe_depth_site())])
goals = PlanGoals()
for sd in (10.0, 100.0)
    s = Schedule(PLAN_GRID; u0=0.15, t_serve=14 * 3600.0)
    f = free_set(s)
    t0 = time()
    r = replan_on_graph([z], s, d; goals, verbose=true, plan_sd=sd)
    Lg = 0.5 * sum(abs2, E.goal_residuals(plan_outputs_rh(z, r.ζ, f, s, d; goals), goals))
    p = r.plan
    @printf("\nPLAN SD %.0f: F %.4f, goals-only %.4f, %d starts, converged %s, %.0f s; wrap %.2f pull %.2f serve %.2f add %s spritz %s; knots %s\n\n", sd, r.F, Lg, r.n_starts, r.converged, time() - t0,
            p.t_wrap / 3600, p.t_pull / 3600, p.t_serve / 3600, isnan(p.t_add) ? "-" : @sprintf("%.2f", p.t_add / 3600),
            p.spritz === nothing ? "-" : @sprintf("%.1f h", p.spritz.t / 3600), join([@sprintf("%.2f", v) for v in p.u], " "))
    flush(stdout)
end
