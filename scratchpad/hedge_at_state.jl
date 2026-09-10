# Does the hedge help while the belief is young? From the closed loop's own early states (the executed damper and
# events to then, the posterior at that check), the plan at the posterior mean against the hedged plan (the band
# terms averaged over sigma points ±1 sd along the widest posterior directions, the tails at the mean), both
# scored at the truth. Usage: julia --project=estimator scratchpad/hedge_at_state.jl [record.json] [hours...]
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, JSON, LinearAlgebra, Printf
const E = BrisketEstimator
REPO = normpath(joinpath(@__DIR__, ".."))
rec_name = length(ARGS) >= 1 ? ARGS[1] : "closed_loop_realcook.json"
hours = length(ARGS) >= 2 ? parse.(Int, ARGS[2:end]) : [1, 2, 4]
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json"))
rec = JSON.parsefile(joinpath(REPO, "data", "mock", rec_name))
prior_file = get(rec, "prior_file", "estimator_gate_real.json")
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", prior_file))
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"])); cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
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
function score(lab, plan, s, f, z_hat)
    ζ = encode_plan(plan, f, s)
    yt = plan_outputs_rh(z_true, ζ, f, s, d; goals); Tt = objective_terms(yt, goals)
    yb = plan_outputs_rh(z_hat, ζ, f, s, d; goals); Tb = objective_terms(yb, goals)
    @printf("  %-22s J at the truth %.3f (tend %.3f app %.3f taste %.3f burn %.3f fuel %.3f) collagen %.3f juic %.3f L* %.1f | believed %.3f\n",
            lab, Tt.J, Tt.tenderness, Tt.appearance, Tt.taste, Tt.surf, Tt.fuel, yt[1], yt[6], yt[3], Tb.J)
    @printf("  %-22s knots %s; wrap %.1f pull %.1f serve %.1f spritz %s add %s\n", "",
            join([@sprintf("%.2f", v) for v in plan.u], " "), plan.t_wrap / 3600, plan.t_pull / 3600, plan.t_serve / 3600,
            plan.spritz === nothing ? "-" : @sprintf("%.1f h %.3f kg", plan.spritz.t / 3600, plan.spritz.m), isnan(plan.t_add) ? "-" : @sprintf("%.1f", plan.t_add / 3600))
end
println("record ", rec_name, " (prior ", prior_file, ")")
for hour in hours
    ck = rec["checks"][hour + 1]
    z_hat = Float64.(ck["posterior_z"])
    V = haskey(ck, "posterior_cov") ? Matrix{Float64}(reduce(hcat, [Float64.(r) for r in ck["posterior_cov"]])') : Matrix(Diagonal(Float64.(ck["posterior_sd"]) .^ 2))
    s = state_at(hour); f = free_set(s)
    e = [abs(x) for x in ck["err_sd"]]
    zs = sigma_draws(z_hat, V; n_dirs=2)
    @printf("\nfrom the state at %d h (posterior median |err|/sd %.2f, max %.2f; sigma spread in J-relevant sd: %s):\n", hour,
            sort(e)[div(end + 1, 2)], maximum(e), haskey(ck, "posterior_cov") ? "full covariance" : "diagonal")
    t0 = time(); r_m = replan_on_graph([z_hat], s, d; goals, iters=30); tm = time() - t0
    t0 = time(); r_h = replan_on_graph(zs, s, d; goals, iters=30); th = time() - t0
    @printf("  the plan at the mean (%.0f s) and the hedged plan over %d sigma points (%.0f s):\n", tm, length(zs) - 1, th)
    score("at the mean", r_m.plan, s, f, z_hat)
    score("hedged", r_h.plan, s, f, z_hat)
    flush(stdout)
end
