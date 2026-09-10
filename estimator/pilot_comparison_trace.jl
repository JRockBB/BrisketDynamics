# Two pilots' executed cooks through the same plant: the RxInfer pilot's
# schedule from data/mock/closed_loop_realcook.json and the app's MPC
# schedule from an exported trace (estimator/export_mpc_trace.py), each
# re-run in the Julia monolith on the plant it was flown on, and scored
# under the one objective, the app's J (cook_outputs / objective_terms in
# horizon.jl). Also checks the port: the Julia core against the
# Tesseract's own trace.
#
# Run:  julia --project=estimator estimator/pilot_comparison_trace.jl [mpc_trace.json] [out.json] [label]
# Writes data/mock/<out.json> (default pilot_comparison.json); one record per MPC variant compared,
# e.g. pilot_comparison_calmodel.json for the MPC flown with the calibration's parameters.
using Pkg
Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Printf, Statistics
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, ".."))
mpc_file = length(ARGS) >= 1 ? ARGS[1] : "mpc_mock1d_traces_mpc_nominal_trace.json"
out_name = length(ARGS) >= 2 ? ARGS[2] : "pilot_comparison.json"          # one record per MPC variant compared
mpc_label = length(ARGS) >= 3 ? ARGS[3] : "MPC"
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json"))
rec = JSON.parsefile(joinpath(REPO, "data", "mock", "closed_loop_realcook.json"))
mp = JSON.parsefile(joinpath(REPO, "data", "control", mpc_file))
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
GRID = collect(0.0:120.0:30 * 3600.0)
NAMES = haskey(m, "names") ? String.(m["names"]) : String.(m["truth"]["names"])
interp(tt, uu, ti) = ti >= tt[end] ? uu[end] : (k = searchsortedlast(tt, ti); k < 1 ? uu[1] : k >= length(tt) ? uu[end] :
                      uu[k] + (uu[k + 1] - uu[k]) * (ti - tt[k]) / (tt[k + 1] - tt[k]))
NEVER = (wrap=BrisketDynamics.WRAP_NEVER, pull=BrisketDynamics.PULL_NEVER, add=BrisketDynamics.ADD_NEVER, spritz=BrisketDynamics.SPRITZ_NEVER)
ev_or(v, never) = v === nothing ? never : Float64(v)

"Run the plant at parameters x (15 physics + A) under a schedule; the pilot's goals and the MPC's terms at the events."
function cook(x, A, u, ev; label)
    θ, φ = x[1:8], x[9:15]
    o = monolith_solve(θ, φ, u, GRID, GRID; pit, cfg, A_meat=A, t_wrap=ev.wrap, t_pull=ev.pull, t_add=ev.add,
                       t_spritz=ev.spritz, dt_spritz=ev.dt_spritz, m_spritz=ev.m_spritz)
    nx = size(o.W_field, 1); w = E.fv_weights(nx)
    wret = [dot(w, o.W_field[:, k]) / cfg.W0 for k in axes(o.W_field, 2)]
    th = GRID ./ 3600
    at(y, tq) = (k = clamp(searchsortedlast(GRID, tq), 1, length(GRID) - 1); f = (tq - GRID[k]) / (GRID[k + 1] - GRID[k]); (1 - f) * y[k] + f * y[k + 1])
    kp = searchsortedlast(GRID, ev.pull); ks = searchsortedlast(GRID, ev.serve)
    Tc = o.Tcore .- 273.15; Tp = o.Tpit .- 273.15
    first_at(v) = (k = findfirst(Tc .>= v); k === nothing ? nothing : th[k])
    y = cook_outputs(o, GRID, executed_du(GRID, u, ev.pull), ev.pull, ev.add, ev.serve, cfg, PlanGoals())
    terms = objective_terms(y, PlanGoals())
    metrics = Dict("J" => terms.J, "J_terms" => Dict(string(k) => getfield(terms, k) for k in propertynames(terms)),
                   "S_serve" => y[5], "juiciness_serve" => y[6], "surf_term" => y[7], "fuel_frac" => y[8],
                   "T_pull_C" => at(Tc, ev.pull), "T_serve_C" => at(Tc, ev.serve), "W_serve" => at(wret, ev.serve), "L_serve" => at(o.l_star, ev.serve),
                   "doneness_serve" => o.C_field[1, ks], "doneness_pull" => o.C_field[1, kp],
                   "juiciness_serve_approx" => at(wret, ev.serve) + (o.juiciness - o.w_retained),   # the fat term at the grid end stands in for serve
                   "pit_mean_C" => mean(Tp[1:kp]), "pit_max_C" => maximum(Tp[1:kp]), "cook_h" => ev.serve / 3600,
                   "wrap_h" => ev.wrap > 1e8 ? nothing : ev.wrap / 3600, "pull_h" => ev.pull / 3600, "rest_h" => (ev.serve - ev.pull) / 3600,
                   "core_70_h" => first_at(70), "core_80_h" => first_at(80), "core_90_h" => first_at(90),
                   "wet_basis_water_serve" => (W = at(wret, ev.serve) * cfg.W0; W / (1 + W)))
    @printf("%s: J %.3f (tenderness %.3f, appearance %.3f, taste %.3f, surf %.3f, fuel %.3f, smooth %.3f, safety %.3f); pit mean %.0f C (%.0f F) max %.0f; wrap %s, pull %.2f h at %.1f C, serve %.2f h at %.1f C; W/W0 %.3f, juiciness %.3f, L* %.1f, smoke %.2f, collagen %.3f at the pull / %.3f at serve\n",
            label, terms.J, terms.tenderness, terms.appearance, terms.taste, terms.surf, terms.fuel, terms.smooth, terms.safety,
            metrics["pit_mean_C"], metrics["pit_mean_C"] * 9 / 5 + 32, metrics["pit_max_C"],
            metrics["wrap_h"] === nothing ? "-" : @sprintf("%.2f h", metrics["wrap_h"]), metrics["pull_h"], metrics["T_pull_C"], metrics["cook_h"], metrics["T_serve_C"],
            metrics["W_serve"], y[6], metrics["L_serve"], y[5], metrics["doneness_pull"], metrics["doneness_serve"])
    keep = GRID .<= ev.serve + 1e-9
    return Dict("t_h" => th[keep], "T_pit_C" => Tp[keep], "T_core_C" => Tc[keep], "W_ret" => wret[keep], "L_star" => collect(o.l_star)[keep],
                "C_center" => o.C_field[1, keep], "u" => u[keep],
                "events_h" => Dict("wrap" => metrics["wrap_h"], "pull" => metrics["pull_h"], "serve" => metrics["cook_h"],
                                   "add" => ev.add > 1e8 ? nothing : ev.add / 3600, "spritz" => ev.spritz > 1e8 ? nothing : ev.spritz / 3600,
                                   "spritz_dt" => ev.dt_spritz / 3600),
                "metrics" => metrics), o
end

# --- the RxInfer pilot: the mock's truth, its executed schedule --------------------------------
x_true = Float64.(m["truth"]["x"])
ex = rec["executed"]
u10 = [v === nothing ? NaN : Float64(v) for v in ex["u"]]; u10 = [isnan(v) ? u10[findlast(!isnan, u10)] : v for v in u10]
t10 = Float64.(ex["t_h"]) .* 3600
u_rx = [interp(t10, u10, ti) for ti in GRID]
sp = ex["spritz"]
ev_rx = (wrap=Float64(ex["t_wrap_h"]) * 3600, pull=Float64(ex["t_pull_h"]) * 3600, serve=Float64(ex["t_serve_h"]) * 3600,
         add=ev_or(ex["t_add_h"], NEVER.add / 3600) * 3600, spritz=sp === nothing ? NEVER.spritz : Float64(sp["t_h"]) * 3600,
         dt_spritz=sp === nothing ? 3600.0 : Float64(sp["dt_h"]) * 3600, m_spritz=sp === nothing ? 0.05 : Float64(sp["m"]))
rx, _ = cook(x_true, x_true[16], u_rx, ev_rx; label="RxInfer pilot")

# --- the MPC: its own parameters, its executed damper and events -------------------------------
x_mpc = [Float64(mp["params"][n]) for n in NAMES[1:15]]
tm = Float64.(mp["t_h"]) .* 3600; um = Float64.(mp["u"])
u_mpc = [interp(tm, um, ti) for ti in GRID]
es = mp["events_s"]
ev_mpc = (wrap=ev_or(es["wrap"], NEVER.wrap), pull=ev_or(es["pull"], NEVER.pull), serve=Float64(es["serve"]),
          add=ev_or(es["add"], NEVER.add), spritz=(es["spritz"] === nothing || es["spritz_dose"] === nothing || es["spritz_dose"] <= 0) ? NEVER.spritz : Float64(es["spritz"]),
          dt_spritz=es["dt_spritz"] === nothing ? 3600.0 : Float64(es["dt_spritz"]), m_spritz=es["spritz_dose"] === nothing ? 0.05 : Float64(es["spritz_dose"]))
mc, o_mpc = cook(x_mpc, cfg.A_meat, u_mpc, ev_mpc; label=mpc_label * " (" * mp["variant"] * ")")
# the port check: the Julia core against the Tesseract's own trace at its grid points
tc_tess = Float64.(mp["tcore_C"])
tc_julia = [interp(GRID, o_mpc.Tcore .- 273.15, ti) for ti in tm]
port = maximum(abs.(tc_julia .- tc_tess))
@printf("port check: max |core Julia - Tesseract| %.2f C over %d points; Tesseract doneness %.3f, juiciness %.3f, L* %.1f\n",
        port, length(tm), mp["metrics"]["doneness"], mp["metrics"]["juiciness"], mp["metrics"]["l_star_end"])
same_plant = all(abs(x_mpc[i] - x_true[i]) <= 1e-6 * max(1.0, abs(x_true[i])) for i in 1:15)
println("same plant as the RxInfer pilot: ", same_plant)

dst = joinpath(REPO, "data", "mock", out_name)
open(dst, "w") do io
    JSON.print(io, Dict("rxinfer" => rx, "mpc" => mc, "mpc_source" => mpc_file, "mpc_variant" => mp["variant"], "mpc_params_file" => mp["params_file"], "mpc_label" => mpc_label,
                        "same_plant" => same_plant, "port_check_max_core_C" => port, "mpc_tesseract_metrics" => mp["metrics"],
                        "goals" => rec["goals"], "rxinfer_checks" => length(rec["checks"]), "rxinfer_replans" => rec["n_replans"],
                        "rxinfer_wall_s" => sum(c["wall_s"] for c in rec["checks"])))
end
println("wrote ", dst)
