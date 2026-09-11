# The plant's history under the schedule the real-cook closed loop
# executed: pit and core temperatures, retained water and bark
# lightness on the plant's grid, with the milestones a pitmaster would
# name (the stall, the wrap, the pull). Reads the closed-loop record,
# writes data/mock/closed_loop_realcook_trace.json for the figure.
#
# Run:  julia --project=estimator estimator/closed_loop_trace.jl
using Pkg
Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Printf, Random, Statistics
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json"))
rec_name = length(ARGS) >= 1 ? ARGS[1] : "closed_loop_realcook.json"     # the loop record to trace; its trace is <stem>_trace.json
rec = JSON.parsefile(joinpath(REPO, "data", "mock", rec_name))
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
x_true = Float64.(m["truth"]["x"])
θ_true, φ_true, A_true = x_true[1:8], x_true[9:15], x_true[16]
# the drifting plant, when the record says so: the same draw from the same seed (drift.jl)
drift = get(rec, "drift", nothing)
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", rec["prior_file"]))
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
idx = Dict(n => i for (i, n) in enumerate(vcat(String.(gate["names"]), ["A_meat", "core_depth"])))
z_true0 = [isbeta(p) ? log(v / (1 - v)) : log(v) for (v, p) in zip(x_true[1:15], prior_phys)]
dd = drift === nothing ? nothing : DriftDraw(MersenneTwister(2000 + Int(drift["seed"])), idx)
ex = rec["executed"]
t10 = Float64.(ex["t_h"]) .* 3600
u10 = [v === nothing ? NaN : Float64(v) for v in ex["u"]]
last = u10[findlast(!isnan, u10)]
u10 = [isnan(v) ? last : v for v in u10]
GRID = collect(0.0:120.0:30 * 3600.0)
u = [ti >= t10[end] ? u10[end] : (k = searchsortedlast(t10, ti); k >= length(t10) ? u10[end] :
     u10[k] + (u10[k + 1] - u10[k]) * (ti - t10[k]) / (t10[k + 1] - t10[k])) for ti in GRID]
sp = ex["spritz"]
events = (t_wrap=Float64(ex["t_wrap_h"]) * 3600, t_pull=Float64(ex["t_pull_h"]) * 3600,
          t_add=ex["t_add_h"] === nothing ? BrisketDynamics.ADD_NEVER : Float64(ex["t_add_h"]) * 3600,
          t_spritz=sp === nothing ? BrisketDynamics.SPRITZ_NEVER : Float64(sp["t_h"]) * 3600,
          dt_spritz=sp === nothing ? 3600.0 : Float64(sp["dt_h"]) * 3600, m_spritz=sp === nothing ? 0.05 : Float64(sp["m"]))
o = dd === nothing ? monolith_solve(θ_true, φ_true, u, GRID, GRID; pit, cfg, A_meat=A_true, events...) :
                     solve_drifting(z_true0, dd, prior_phys, u, GRID; pit, cfg, A_meat=A_true, events...)
nx = size(o.W_field, 1); w = E.fv_weights(nx)
wret = [dot(w, o.W_field[:, k]) / cfg.W0 for k in axes(o.W_field, 2)]
th = GRID ./ 3600
Tc = o.Tcore .- 273.15; Tp = o.Tpit .- 273.15
first_at(y, v) = (k = findfirst(y .>= v); k === nothing ? NaN : th[k])
kpull = searchsortedlast(GRID, events.t_pull); kwrap = searchsortedlast(GRID, events.t_wrap)
@printf("pit: mean %.0f C to the pull (%.0f F), min %.0f, max %.0f; damper knots at the end %s\n",
        mean(Tp[1:kpull]), mean(Tp[1:kpull]) * 9 / 5 + 32, minimum(Tp[1:kpull]), maximum(Tp[1:kpull]), string(round.(rec["checks"][end]["plan"]["u"]; digits=2)))
@printf("core: 60 C at %.1f h, 70 C at %.1f h, 75 C at %.1f h, 80 C at %.1f h, 85 C at %.1f h; at the wrap (%.1f h) %.1f C; at the pull (%.1f h) %.1f C; max %.1f C\n",
        first_at(Tc, 60), first_at(Tc, 70), first_at(Tc, 75), first_at(Tc, 80), first_at(Tc, 85), events.t_wrap / 3600, Tc[kwrap], events.t_pull / 3600, Tc[kpull], maximum(Tc[1:kpull]))
rate = diff(Tc) ./ diff(th)
@printf("core rise rate: %.1f C/h in hours 2-4, %.1f C/h in hours 10-14 (the stall), %.1f C/h after the wrap to the pull\n",
        mean(rate[(th[2:end] .> 2) .& (th[2:end] .<= 4)]), mean(rate[(th[2:end] .> 10) .& (th[2:end] .<= 14)]),
        mean(rate[(th[2:end] .> events.t_wrap / 3600) .& (th[2:end] .<= events.t_pull / 3600)]))
@printf("retained water W/W0: %.2f at the wrap, %.2f at the pull, %.2f at serve (W0 %.2f dry basis: wet-basis water %.0f%% -> %.0f%%)\n",
        wret[kwrap], wret[kpull], wret[searchsortedlast(GRID, Float64(ex["t_serve_h"]) * 3600)], cfg.W0,
        100 * cfg.W0 / (1 + cfg.W0), 100 * (wret[kpull] * cfg.W0) / (1 + wret[kpull] * cfg.W0))
@printf("bark L*: %.1f at the wrap, %.1f at the pull\n", o.l_star[kwrap], o.l_star[kpull])
dst = joinpath(REPO, "data", "mock", replace(rec_name, ".json" => "_trace.json"))
open(dst, "w") do io
    JSON.print(io, Dict("t_h" => th, "T_pit_C" => Tp, "T_core_C" => Tc, "W_ret" => wret, "L_star" => collect(o.l_star), "u" => u,
                        "events_h" => Dict("wrap" => events.t_wrap / 3600, "pull" => events.t_pull / 3600, "serve" => Float64(ex["t_serve_h"]),
                                           "add" => ex["t_add_h"], "spritz" => sp === nothing ? nothing : sp["t_h"], "spritz_dt" => sp === nothing ? nothing : sp["dt_h"])))
end
println("wrote ", dst)
