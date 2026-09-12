# The plant's history under the schedule the real-cook closed loop executed (pit, core and surface temperatures,
# retained water and bark lightness on the plant's grid, with the milestones a pitmaster would name), and what
# the pilot BELIEVED the cook would do at several of its checks: the plan's predicted pit, core and surface
# trajectories and outcomes with the belief's covariance carried through the plan physics (plan_predictive),
# so the figure can draw the temperatures the pilot planned on as the uncertain quantities they are. Reads the
# closed-loop record (a drifting plant is rebuilt from the record's seed), writes data/mock/<stem>_trace.json.
#
# Run:  julia --project=estimator estimator/closed_loop_trace.jl [record.json]
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
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
names = String.(gate["names"]); allnames = vcat(names, ["A_meat", "core_depth"])
idx = Dict(n => i for (i, n) in enumerate(allnames))
L_med = exp(prior_phys[4].p1)
prior = vcat(prior_phys, [PriorSite("lognormal", log(cfg.A_meat), 0.3), PriorSite("lognormal", log(0.8 * L_med), 0.5)])
z_true0 = [isbeta(p) ? log(v / (1 - v)) : log(v) for (v, p) in zip(x_true[1:15], prior_phys)]
dd = drift === nothing ? nothing : DriftDraw(MersenneTwister(2000 + Int(drift["seed"])), idx)
ex = rec["executed"]
t10 = Float64.(ex["t_h"]) .* 3600
u10 = [v === nothing ? NaN : Float64(v) for v in ex["u"]]
last = u10[findlast(!isnan, u10)]
u10 = [isnan(v) ? last : v for v in u10]
GRID = collect(0.0:120.0:30 * 3600.0)
PLAN_GRID = collect(0.0:600.0:30 * 3600.0)
CHECK = Float64(rec["check_h"]) * 3600
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
Tc = o.Tcore .- 273.15; Tp = o.Tpit .- 273.15; Ts = o.Tsurf .- 273.15
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

# --- what the pilot believed at every check: the plan's predictive under the posterior ------------------------
# the state at a check replayed from the checks before it (their commits and their hour of damper), the check's
# own plan and belief, the smoother's history when the record carries it; the predictive at EVERY check, so the
# figure can draw the belief as the pilot held it: updated at each hourly check and carried from there to the
# next (the temperatures at now and the plan's forecast over the coming hour), one track per temperature
d_plan = estimator_setup(PLAN_GRID, fill(0.15, length(PLAN_GRID)), pit, cfg, prior_phys, names)
goals = PlanGoals()
h(x) = x === nothing ? NaN : Float64(x) * 3600
function plan_of(c)
    p = c["plan"]; sp = p["spritz"]
    (u=Float64.(p["u"]), t_wrap=h(p["t_wrap_h"]), t_pull=h(p["t_pull_h"]), t_serve=h(p["t_serve_h"]), t_add=h(p["t_add_h"]),
     spritz=sp === nothing ? nothing : (t=h(sp["t_h"]), dt=h(sp["dt_h"]), m=Float64(sp["m"])))
end
function state_at(hour)
    s = Schedule(PLAN_GRID; u0=0.15, t_serve=14 * 3600.0)
    for c in rec["checks"]
        c["t_h"] >= hour && break
        s.t_now = Float64(c["t_h"]) * 3600
        plan = plan_of(c)
        for ev in c["committed"]
            ev == "wrap" && (s.t_wrap = plan.t_wrap); ev == "add" && (s.t_add = plan.t_add); ev == "pull" && (s.t_pull = plan.t_pull); ev == "spritz" && (s.spritz = plan.spritz)
        end
        s.t_now > 0 && (s.t_serve = plan.t_serve)
        E.execute_damper!(s, plan, CHECK)
    end
    s.t_now = hour * 3600.0
    return s
end
function history_of(c, s)
    info = get(c, "estimator", nothing)
    (info === nothing || !haskey(info, "z_stack")) && return nothing
    t_add_s = isnan(s.t_add) ? BrisketDynamics.ADD_NEVER : s.t_add
    moving = smoother_moving(idx, t_add_s)
    specs = [(sidx, [findlast(GRID .<= hh * 3600 + 1e-9) for hh in Float64.(info["starts_h"][allnames[sidx]])]) for (sidx, _, _) in moving]
    return plan_history(Float64.(info["z_stack"]), DriftLayout(length(prior), specs), GRID, PLAN_GRID, prior)
end
chosen = [c for c in rec["checks"] if c["t_h"] > 0]                # every check after the opening plan
predictive = Any[]
for c in chosen
    t0 = time()
    s = state_at(c["t_h"])
    f = free_set(s)
    ζ = encode_plan(plan_of(c), f, s)
    z = Float64.(c["posterior_z"])
    V = reduce(hcat, [Float64.(row) for row in c["posterior_cov"]])'
    pr = plan_predictive(z, Matrix(V), ζ, f, s, d_plan; goals, history=history_of(c, s))
    k_now = searchsortedlast(PLAN_GRID, s.t_now)
    @printf("the belief at %2.0f h: core at the pull %.1f ± %.1f C, pit now %.0f ± %.0f C, surface at the pull %.0f ± %.0f C; collagen at serve %.3f ± %.3f, L* %.1f ± %.1f, serve %.1f ± %.1f C (%.0f s)\n",
            c["t_h"], pr.Tcore.mean[searchsortedlast(PLAN_GRID, plan_of(c).t_pull)] - 273.15, pr.Tcore.sd[searchsortedlast(PLAN_GRID, plan_of(c).t_pull)],
            pr.Tpit.mean[k_now] - 273.15, pr.Tpit.sd[k_now], pr.Tsurf.mean[searchsortedlast(PLAN_GRID, plan_of(c).t_pull)] - 273.15, pr.Tsurf.sd[searchsortedlast(PLAN_GRID, plan_of(c).t_pull)],
            pr.y.mean[1], pr.y.sd[1], pr.y.mean[3], pr.y.sd[3], pr.y.mean[4] - 273.15, pr.y.sd[4], time() - t0)
    push!(predictive, Dict("t_h" => c["t_h"], "grid_h" => PLAN_GRID ./ 3600, "last" => c === chosen[end], "replanned" => c["replanned"],
                           "Tpit" => Dict("mean" => pr.Tpit.mean .- 273.15, "sd" => pr.Tpit.sd),
                           "Tcore" => Dict("mean" => pr.Tcore.mean .- 273.15, "sd" => pr.Tcore.sd),
                           "Tsurf" => Dict("mean" => pr.Tsurf.mean .- 273.15, "sd" => pr.Tsurf.sd),
                           "y" => Dict("mean" => pr.y.mean, "sd" => pr.y.sd),
                           "plan" => Dict("t_wrap_h" => c["plan"]["t_wrap_h"], "t_pull_h" => c["plan"]["t_pull_h"], "t_serve_h" => c["plan"]["t_serve_h"], "t_add_h" => c["plan"]["t_add_h"])))
    flush(stdout)
end
dst = joinpath(REPO, "data", "mock", replace(rec_name, ".json" => "_trace.json"))
open(dst, "w") do io
    JSON.print(io, Dict("t_h" => th, "T_pit_C" => Tp, "T_core_C" => Tc, "T_surf_C" => Ts, "W_ret" => wret, "L_star" => collect(o.l_star), "u" => u,
                        "events_h" => Dict("wrap" => events.t_wrap / 3600, "pull" => events.t_pull / 3600, "serve" => Float64(ex["t_serve_h"]),
                                           "add" => ex["t_add_h"], "spritz" => sp === nothing ? nothing : sp["t_h"], "spritz_dt" => sp === nothing ? nothing : sp["dt_h"]),
                        "predictive" => predictive))
end
println("wrote ", dst)
