# The objective's parity fixture: one cook's trajectories and the terms Julia's port of the app's objective
# (horizon.jl: cook_outputs, objective_terms) gives them, written to data/mock/objective_parity.json, so that
# estimator/test/test_horizon.jl and app/tests/test_objective_parity.py score the SAME stored cook term by term,
# neither running any physics. The cook is mock_realcook.json at its truth under its executed damper and events,
# served two hours after the pull; the fixture stores the surface, core and fuel trajectories, the executed damper
# as hourly means held on the grid (the app's executed score reads the damper's steps), the scalars the objective
# reads at serve, and Julia's terms. Regenerate it when the objective or the goals change, never when the physics
# does (the tests read trajectories, not parameters).
#
# Run:  julia --project=estimator estimator/scripts/objective_parity.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Statistics
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]),
          t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
x = Float64.(m["truth"]["x"])
θ, φ, A = x[1:N_THETA], x[(N_THETA + 1):N_PHYS], x[16]
t_serve = events.t_pull + 2 * 3600.0                                  # a two-hour rest
keep = t .<= t_serve + 1e-9
tg, ug = t[keep], u[keep]
o = monolith_solve(θ, φ, ug, tg, tg; pit, cfg, A_meat=A, events...)
g = PlanGoals()
Δu = executed_du(tg, ug, events.t_pull)                               # the executed damper's hourly means to the pull
y = cook_outputs(o, tg, Δu, events.t_pull, events.t_add, t_serve, cfg, g)
terms = objective_terms(y, g)
# the same damper as the app's executed score reads it: the hourly means as a step function on the grid, the
# last mean held through the pull's partial hour (so the steps are exactly the means Δu differences)
hours = length(Δu) + 1
means = [mean(ug[(tg .>= (h - 1) * 3600) .& (tg .< h * 3600)]) for h in 1:hours]
u_steps = [means[clamp(floor(Int, tk / 3600) + 1, 1, hours)] for tk in tg]
wfv = E.fv_weights(size(o.F_field, 1))
fmean = [dot(wfv, view(o.F_field, :, k)) for k in axes(o.F_field, 2)]
fx = Dict("source" => "mock_realcook.json at its truth, the executed damper and events, served two hours after the pull",
          "t" => tg, "tsurf" => o.Tsurf, "tcore" => o.Tcore, "m_fuel" => o.m_fuel, "u_steps" => u_steps, "du" => Δu,
          "t_wrap" => events.t_wrap, "t_pull" => events.t_pull, "t_add" => events.t_add, "t_serve" => t_serve,
          "doneness" => y[1], "w_serve" => y[2], "l_star_end" => y[3], "tcore_serve" => y[4], "smoke_end" => y[5],
          "juiciness" => y[6], "surf_term" => y[7], "fuel_frac" => y[8], "leth_s" => y[end - 1], "leth_c" => y[end],
          "fmean_serve" => E._interp_t(fmean, tg, t_serve),
          "cfg" => Dict("W0" => cfg.W0, "beta_fat" => cfg.β_fat, "eta_drain" => cfg.η_drain, "F0" => cfg.F0),
          "goals" => Dict(string(f) => getfield(g, f) for f in fieldnames(PlanGoals)),
          "terms" => Dict(string(k) => v for (k, v) in pairs(terms)))
dst = joinpath(REPO, "data", "mock", "objective_parity.json")
open(dst, "w") do io; JSON.print(io, fx); end
println("wrote ", dst, "; the terms: ", join(["$k $(round(v; digits=6))" for (k, v) in pairs(terms)], ", "))
