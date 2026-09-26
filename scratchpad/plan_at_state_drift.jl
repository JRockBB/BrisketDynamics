# From a pilot record's own check (partial or final) on the DRIFTING plant: the state replayed from the checks to
# that hour; the plan that stood before the check and the plan the check made, each scored at the belief and at
# the truth; then the planner at the truth from the same state. Two physics for each: the plan's whole-cook solve
# at constants ("constant": the history re-cooked with the values in force now) and the segmented solve from the
# history that was in force ("history": the smoother's stacked block for the belief, the plant's own walk for the
# truth). Was the replan warranted, and does the plan's picture of the state at now matter?
#   julia --project=estimator scratchpad/plan_at_state_drift.jl <record.json> <hour>
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "src", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, JSON, LinearAlgebra, Printf, Random
const E = BrisketEstimator
REPO = normpath(joinpath(@__DIR__, ".."))
rec_name = ARGS[1]; hour = parse(Int, ARGS[2])
rec = JSON.parsefile(joinpath(REPO, "data", "mock", rec_name))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json")); gate = JSON.parsefile(joinpath(REPO, "data", "calibration", rec["prior_file"]))
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"])); cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = read_sites(gate["prior"])
names = String.(gate["names"]); allnames = vcat(names, ["A_meat", "core_depth_fraction"]); idx = Dict(n => i for (i, n) in enumerate(allnames))
prior = vcat(prior_phys, [PriorSite("lognormal", log(cfg.A_meat), 0.3), probe_depth_site()])
GRID = collect(0.0:120.0:30 * 3600.0); PLAN_GRID = collect(0.0:600.0:30 * 3600.0)
CHECK = Float64(rec["check_h"]) * 3600
d = estimator_setup(PLAN_GRID, fill(0.15, length(PLAN_GRID)), pit, cfg, prior_phys, names)
goals = PlanGoals()
x_true = Float64.(m["truth"]["x"])
z_true0 = vcat([to_z(v, p) for (v, p) in zip(x_true[1:15], prior_phys)], log(x_true[16]), to_z(x_true[17], probe_depth_site()))
drift = rec["drift"]; dd = drift === nothing ? nothing : DriftDraw(MersenneTwister(2000 + Int(drift["seed"])), idx)
h(x) = x === nothing ? NaN : Float64(x) * 3600
function plan_of(c)
    p = c["plan"]; sp = p["spritz"]
    (u=Float64.(p["u"]), t_wrap=h(p["t_wrap_h"]), t_pull=h(p["t_pull_h"]), t_serve=h(p["t_serve_h"]), t_add=h(p["t_add_h"]),
     spritz=sp === nothing ? nothing : (t=h(sp["t_h"]), dt=h(sp["dt_h"]), m=Float64(sp["m"])))
end
# the state at the check: the checks before it replayed (their commits and their hour of damper)
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
t_add_s = isnan(s.t_add) ? BrisketDynamics.ADD_NEVER : s.t_add
z_hat = Float64.(ck["posterior_z"])
z_true = dd === nothing ? z_true0 : truth_at(z_true0, dd, hour * 3600.0, t_add_s; from=true)     # in force from now
f = free_set(s)
# the belief's history: the smoother's stacked block and its layout, when the record carries them
info = get(ck, "estimator", nothing)
hist_b = nothing
if info !== nothing && haskey(info, "z_stack")
    moving = smoother_moving(idx, t_add_s)
    specs = [(sidx, [findlast(GRID .<= hh * 3600 + 1e-9) for hh in Float64.(info["starts_h"][allnames[sidx]])]) for (sidx, _, _) in moving]
    lay = DriftLayout(length(prior), specs)
    global hist_b = plan_history(Float64.(info["z_stack"]), lay, GRID, PLAN_GRID, prior)
end
# the truth's history: the plant's own walk to now (segments on the hour and at the add), the block from now on = z_true
hist_t = nothing
if dd !== nothing
    e_all, b_all = drift_segments(GRID, z_true0, dd, t_add_s)
    keep = [k for (k, e) in enumerate(e_all) if GRID[e] <= hour * 3600 + 1e-9]
    if !isempty(keep)
        edges = Int[]; θs = Vector{Float64}[]; φs = Vector{Float64}[]
        for k in keep
            ie = searchsortedlast(PLAN_GRID, GRID[e_all[k]] + 1e-9)
            x = from_z(b_all[k][1:N_PHYS], prior_phys)                   # the block in force before this edge
            if !isempty(edges) && ie == edges[end]
                θs[end], φs[end] = x[1:N_THETA], x[(N_THETA + 1):N_PHYS]; continue
            end
            push!(edges, ie); push!(θs, x[1:N_THETA]); push!(φs, x[(N_THETA + 1):N_PHYS])
        end
        global hist_t = (edges=edges, θs=θs, φs=φs)
    end
end
function score(lab, plan)
    ζ = encode_plan(plan, f, s)
    @printf("  %-24s wrap %.1f pull %.1f serve %.1f add %s; knots %s\n", lab, plan.t_wrap / 3600, plan.t_pull / 3600, plan.t_serve / 3600,
            isnan(plan.t_add) ? "-" : @sprintf("%.1f", plan.t_add / 3600), join([@sprintf("%.2f", v) for v in plan.u], " "))
    for (zl, z, hist) in (("belief, constant", z_hat, nothing), ("belief, history", z_hat, hist_b), ("truth, constant", z_true, nothing), ("truth, history", z_true, hist_t))
        (zl == "belief, history" && hist_b === nothing) && continue
        (zl == "truth, history" && hist_t === nothing) && continue
        y = plan_outputs_rh(z, ζ, f, s, d; goals, history=hist); T = objective_terms(y, goals)
        @printf("     %-18s J %.3f (tend %.3f app %.3f taste %.3f burn %.3f fuel %.3f) collagen %.3f L* %.1f serve %.0f C\n", zl, T.J, T.tenderness, T.appearance, T.taste, T.surf, T.fuel, y[1], y[3], y[4] - 273.15)
    end
end
e = ck["err_sd"]
@printf("record %s at %d h (%s, KL %.1f, L_now %.2f): belief |err|/sd median %.2f max %.2f; the state: wrap %.1f add %.1f pull %.1f; free: damper %s wrap %s pull %s spritz %s add %s; history edges (h): belief %s, truth %s\n",
        rec_name, hour, ck["reason"], ck["kl"] === nothing ? NaN : ck["kl"], ck["L_now"] === nothing ? NaN : ck["L_now"], sort(abs.(e))[div(end + 1, 2)], maximum(abs.(e)),
        s.t_wrap / 3600, s.t_add / 3600, s.t_pull / 3600, f.damper, f.wrap, f.pull, f.spritz, f.add,
        hist_b === nothing ? "-" : string(round.(PLAN_GRID[hist_b.edges] ./ 3600; digits=2)), hist_t === nothing ? "-" : string(round.(PLAN_GRID[hist_t.edges] ./ 3600; digits=2)))
score("the plan that stood", plan_of(prev))
score("the plan the check made", plan_of(ck))
t0 = time(); r_t = replan_on_graph([z_true], s, d; goals, iters=30, history=hist_t); tt = time() - t0
@printf("  the planner at the truth from the true state (%.0f s):\n", tt)
score("the truth's plan", r_t.plan)
if hist_b !== nothing
    t0 = time(); r_b = replan_on_graph([z_hat], s, d; goals, iters=30, history=hist_b); tb = time() - t0
    @printf("  the planner at the belief from the smoother's state, cold (%.0f s):\n", tb)
    score("the belief's plan, history", r_b.plan)
    ζw = encode_plan(plan_of(prev), f, s)
    t0 = time(); r_w = replan_on_graph([z_hat], s, d; goals, iters=30, history=hist_b, ζ_init=ζw); tw = time() - t0
    @printf("  the same with the standing plan as the warm start, as the loop ran it (%.0f s; %d starts, F %.4f; the warm start itself scores %.4f):\n", tw, r_w.n_starts, r_w.F,
            0.5 * sum(abs2, E.goal_residuals(plan_outputs_rh(z_hat, ζw, f, s, d; goals, history=hist_b), goals)))
    score("the belief's plan, warm", r_w.plan)
end
flush(stdout)
