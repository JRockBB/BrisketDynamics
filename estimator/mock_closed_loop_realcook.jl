# The pilot on the real-cook plant, replanning when it should rather
# than on a clock. The plant is the monolith at the mock's true latents
# read by instruments at their own rates on a two-minute grid (the pit
# and core probes every two minutes, the humidity every ten and capped,
# the scale every half hour, the lid lifts dropped, the pit noise level
# following the fire: realcook.jl). Every hour the estimator refits on
# the graph engine with every channel's noise identified at its own
# spacing and its level as a process; the standing plan is then
# re-evaluated under the new posterior draws (one plant run per draw, no
# optimisation) and a replan fires only when its expected loss has
# risen, a planned event is about to commit, the misfit check fires, or
# a floor interval has passed (horizon.jl's ReplanPolicy). The plan's
# physics runs on a ten-minute grid; the executed damper is interpolated
# onto the plant's grid. The record holds every check with its reason,
# and the plant's actual outcome.
#
# Run:  julia --project=estimator estimator/mock_closed_loop_realcook.jl [prior.json] [out.json] [hedge dirs] [ridge k] [ridge steps] [estimator] [drift seed]
#   estimator: "batch" (every latent a constant on the history) or "smoother" (smoother.jl: the leak, the surface
#   coefficients and the add's charge piecewise constant over the history with a random-walk prior between the
#   pieces at drift.jl's rates; the plan, the trigger and the record read the latest values)
#   drift seed > 0: the plant's leak, surface coefficients and the add's charge move (drift.jl, drawn once from the seed)
#   ridge k > 0: after every refit the belief along the k widest posterior directions is sampled on the exact
#   posterior (ridge.jl) and the plan, the replan trigger and the record read the corrected belief
#   prior.json (data/calibration): the prior the pilot starts from; the gate's literature prior by default,
#   prior_next_cook.json for the calibration chain's posterior by the pool's roles (export_chain_prior.py)
# Writes data/mock/<out.json> (closed_loop_realcook.json by default).
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf, Statistics
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json"))
prior_file = length(ARGS) >= 1 ? ARGS[1] : "estimator_gate_real.json"
out_name = length(ARGS) >= 2 ? ARGS[2] : "closed_loop_realcook.json"
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", prior_file))
println("prior: ", prior_file, "; record: ", out_name)
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
names = String.(gate["names"])
L_med = exp(prior_phys[4].p1)
prior = vcat(prior_phys, [PriorSite("lognormal", log(cfg.A_meat), 0.3), PriorSite("lognormal", log(0.8 * L_med), 0.5)])
AREA, DEPTH = 16, 17
GRID = collect(0.0:120.0:30 * 3600.0)          # the plant and the estimator
PLAN_GRID = collect(0.0:600.0:30 * 3600.0)     # the plan's physics
CHECK = 3600.0                                  # the posterior updates hourly; the replan decides for itself
HEDGE_DIRS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0     # off by default: side by side it lost to the plan at the mean (docs/results.md)
                     # 0: the plan at the posterior mean, as the MPC plans at its point estimate; n: the hedge, the band terms
                     # averaged over sigma points ±1 sd along the n widest posterior directions, the tails at the mean
N_DRAWS = 2 * HEDGE_DIRS
RIDGE_K = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 0
RIDGE_STEPS = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 300
println("hedge directions: ", HEDGE_DIRS, " (", N_DRAWS, " sigma points); ridge chain: k ", RIDGE_K, RIDGE_K > 0 ? " with $RIDGE_STEPS steps" : " (off)")
ESTIMATOR = length(ARGS) >= 6 ? ARGS[6] : "batch"
ESTIMATOR in ("batch", "smoother") || error("the estimator is batch or smoother")
DRIFT_SEED = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 0
COPY_EVERY = 2 * 3600.0          # the smoother gives a drifting site a new value every two hours of history (the coverage test's spacing)
println("estimator: ", ESTIMATOR, ESTIMATOR == "smoother" ? " (a new value every $(COPY_EVERY / 3600) h at the rates " * string(DRIFT_SD_PER_HOUR) * "; the add's charge " * string(JUMP_LOG_SD) * ")" : "",
        "; plant: ", DRIFT_SEED > 0 ? "drifting (seed $DRIFT_SEED)" : "constant")
goals = PlanGoals()
pol = ReplanPolicy()
nc = RealCookNoise()

# --- the plant: the mock's truth, read like a real cook ----------------------------------------
x_true = Float64.(m["truth"]["x"])
θ_true, φ_true, A_true, depth_true = x_true[1:8], x_true[9:15], x_true[16], x_true[17]
L_true = θ_true[4]
idx = Dict(n => i for (i, n) in enumerate(vcat(names, ["A_meat", "core_depth"])))
z_true0 = vcat([isbeta(p) ? log(v / (1 - v)) : log(v) for (v, p) in zip(x_true[1:15], prior_phys)], log(A_true), log(depth_true))
dd = DRIFT_SEED > 0 ? DriftDraw(MersenneTwister(2000 + DRIFT_SEED), idx) : nothing     # the plant's drift, drawn once
"The plant's latent block in force just before `te` (what a belief at `te` is scored against)."
truth_now(te, t_add) = dd === nothing ? z_true0 : truth_at(z_true0, dd, te, t_add)
from_z_all(z) = [isbeta(p) ? 1 / (1 + exp(-v)) : exp(v) for (v, p) in zip(z, prior)]
"The plant's drift for the record: the seed, the rates and the block in force from every hour under the schedule's add."
drift_record(s::Schedule) = dd === nothing ? nothing :
    Dict("seed" => DRIFT_SEED, "sd_per_hour" => DRIFT_SD_PER_HOUR, "jump_log_sd" => JUMP_LOG_SD,
         "z_by_hour" => [truth_at(z_true0, dd, h * 3600.0, schedule_events(s).t_add; from=true) for h in 0:29])
fill_future(u) = (last = u[findlast(!isnan, u)]; [isnan(x) ? last : x for x in u])
nn(x) = isnan(x) ? nothing : x
"The executed damper from the schedule's grid onto the plant's (linear, held after the schedule's end)."
function damper_on_grid(s::Schedule)
    u10 = fill_future(s.u)
    return [ti >= s.t[end] ? u10[end] : (k = searchsortedlast(s.t, ti); k >= length(s.t) ? u10[end] :
            u10[k] + (u10[k + 1] - u10[k]) * (ti - s.t[k]) / (s.t[k + 1] - s.t[k])) for ti in GRID]
end
"The plant's readings under the executed schedule: the same unit noise every call, scaled by the executed fire and lid lifts."
function plant_readings(s::Schedule)
    ev = schedule_events(s)
    ug = damper_on_grid(s)
    o = if dd === nothing
        monolith_solve(θ_true, φ_true, ug, GRID, GRID; pit, cfg, A_meat=A_true, ev...)
    else                    # the drifting plant: segments on the hour and at the add, the state handed off exactly
        edges, blocks = drift_segments(GRID, z_true0, dd, ev.t_add)
        xs = from_z_all.(blocks)
        solve_segments([x[1:8] for x in xs], [x[9:15] for x in xs], ug, GRID, edges; pit, cfg, A_meat=A_true, ev...)
    end
    r, _, masks, _ = realcook_readings(o, GRID, ev, nc, MersenneTwister(23); A=A_true, L=L_true, depth=depth_true)
    return r, masks, o
end

# --- the estimator: channels at their rates, noise identified, the level as a process ----------
cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json"))
dt = GRID[2] - GRID[1]
tostep(np, from_dt) = (a = dt / from_dt; NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean^a, np.ρ_sd * a * np.ρ_mean^(a - 1); floor=np.floor))
np = Dict(1 => tostep(cal["pit"], 600.0), 2 => tostep(cal["core"], 600.0),
          3 => NoisePrior(100.0, 100.0, 0.8^(dt / 600), 0.3; floor=60.0), 4 => NoisePrior(0.02, 0.02, 0.5^(dt / 1800), 0.3; floor=0.015))   # floors from the instruments' spec: the sparse channels' sd reads low from a few samples
chnames = Dict(1 => "pit", 2 => "core", 3 => "humidity", 4 => "mass")
noise = Dict(i => (v.σ_mean, v.ρ_mean) for (i, v) in np)                  # carried from check to check
levels = Dict{Int,Union{Nothing,Vector{Float64}}}(i => nothing for i in keys(np))
z_stack, lay_prev = nothing, nothing        # the smoother's stacked belief and its layout, carried between checks
function estimate!(s::Schedule, z_prev)
    readings, masks, _ = plant_readings(s)
    keep = GRID .<= s.t_now + 1e-9
    d = estimator_setup(GRID, damper_on_grid(s), pit, cfg, prior_phys, names)
    ev = schedule_events(s)
    base = [PitProbe(noise[1]..., masks.pit .& keep; data=readings.pit),
            MeatProbe(noise[2]..., masks.core .& keep; depth_latent=DEPTH, data=readings.core),
            CensoredHumidity(noise[3]..., masks.humidity .& keep; cap=nc.hum_cap, data=readings.humidity),
            MassScale(noise[4]..., masks.mass .& keep; data=readings.mass)]
    chs = [levels[i] === nothing ? ch : Levelled(ch, levels[i]) for (i, ch) in enumerate(base)]
    if ESTIMATOR == "smoother"
        # the history to now over the stacked block: a new value of each drifting site every COPY_EVERY of history,
        # the add's charge from the add; warm-started from the last check's stacked belief
        starts = filter(e -> e < s.t_now - 1e-9, COPY_EVERY .* (1:ceil(Int, GRID[end] / COPY_EVERY)))
        batch_edges = vcat(starts, s.t_now)
        rec = smoother_step(chs, prior, d, batch_edges, length(batch_edges), z_stack === nothing ? z_prev : z_stack, lay_prev;
                            moving=smoother_moving(idx, ev.t_add), area_latent=AREA, events=ev, engine=:graph, noise=np, noise_rounds=1,
                            levelled=true, verbose=false)
        global z_stack, lay_prev = rec.z, rec.layout
        for (i, v) in rec.noise
            noise[i] = (v.σ, v.ρ)
            levels[i] = rec.channels[i] isa Levelled ? rec.channels[i].level : nothing
        end
        info = Dict("block" => length(rec.z), "iters" => rec.iters, "wall_s" => rec.wall, "F" => rec.F,
                    "copies" => Dict(allnames[rec.layout.sites[m]] => length(rec.layout.starts[m]) for m in eachindex(rec.layout.sites)))
        return rec.z_now, Matrix(Symmetric(rec.cov_now)), rec.misfit.total, rec.noise, rec.converged, rec.z_now, nothing, info
    end
    ph = Physics1D(area_latent=AREA, events=ev)
    res = solve_map_channels(chs, prior, d; c0=z_prev, physics=ph, engine=:graph, noise=np, noise_rounds=1,
                             levelled=true, verbose=false)
    for (i, v) in res.noise
        noise[i] = (v.σ, v.ρ)
        levels[i] = res.channels[i] isa Levelled ? res.channels[i].level : nothing
    end
    ridge = nothing
    z_b, V_b = res.z, res.cov
    if RIDGE_K > 0 && res.converged
        t1 = time()
        rc = ridge_chain(res.channels, prior, d, ph, res.z, res.cov; k=RIDGE_K, steps=RIDGE_STEPS, rng=MersenneTwister(4000 + round(Int, s.t_now / 3600)))
        z_b, V_b = rc.z, rc.cov
        ridge = Dict("accept" => rc.accept, "ess" => rc.ess, "wall_s" => time() - t1, "sd_ratio" => rc.sd ./ res.sd, "mean_shift_sd" => (rc.z .- res.z) ./ res.sd)
    end
    return z_b, V_b, res.misfit.total, res.noise, res.converged, res.z, ridge, nothing
end
draws(z, V, rng) = HEDGE_DIRS == 0 ? [copy(z)] : sigma_draws(z, V; n_dirs=HEDGE_DIRS)     # the mean first
"The pit level at the latest sample (1 when none is inferred)."
pit_level_now(s) = (lv = levels[1]; lv === nothing ? 1.0 : (k = findlast(GRID .<= s.t_now + 1e-9); lv[k]))

# --- the loop ------------------------------------------------------------------------------------
s = Schedule(PLAN_GRID; u0=0.15, t_serve=14 * 3600.0)
d_plan = estimator_setup(PLAN_GRID, fill(0.15, length(PLAN_GRID)), pit, cfg, prior_phys, names)   # the plan physics reads its prior, pit, cfg, tolerances
z = vcat(d_plan.z0, [log(cfg.A_meat), log(0.8 * L_med)])
z_map = copy(z)          # the refit's warm start: the MAP, not the ridge-corrected belief
ridge = nothing
est_info = nothing
allnames = vcat(names, ["A_meat", "core_depth"])
_, _, h0 = E.prior_terms(z, prior)
V = Matrix(Diagonal(1 ./ h0))
plan = nothing
L_plan = Inf
t_last = -Inf
z_at_plan, V_at_plan = copy(z), copy(V)          # the belief the standing plan was made on
pred = nothing
record = []
n_replans = 0
rng_draws = MersenneTwister(7)
t_start = time()
println("t[h]  reason           committed        L_now  L_plan  KL     plan: wrap / pull / serve / spritz / add   outcomes at the last replan (collagen at serve, water, L*, serve C)   misfit z   pit σ, level   wall")
flush(stdout)
while s.t_now < 28 * 3600
    t0 = time()
    mis = nothing; noise_post = Dict{Int,NamedTuple}(); conv = true
    if s.t_now > 0
        global z, V, mis, noise_post, conv, z_map, ridge, est_info = estimate!(s, z_map)
    end
    zs = draws(z, V, rng_draws)
    L_now = plan === nothing ? NaN : plan_loss(zs, plan, s, d_plan; goals).mean
    kl = plan === nothing ? NaN : posterior_kl(z, V, z_at_plan, V_at_plan)
    due, why = replan_due(pol, s, plan, L_now, L_plan, t_last, mis === nothing ? nothing : mis.z_chi2; kl=isnan(kl) ? nothing : kl, n_latent=length(z))
    stable = Symbol[]
    if due
        ζ_init = plan === nothing ? nothing : encode_plan(plan, free_set(s), s)
        r = replan_on_graph(zs, s, d_plan; goals, ζ_init, verbose=false)
        why == :event_due && (stable = stable_events(r.plan, plan, pol.stable))   # left where it stood: commit it
        global plan = r.plan
        global L_plan = plan_loss(zs, plan, s, d_plan; goals).mean
        global t_last = s.t_now
        global n_replans += 1
        global z_at_plan, V_at_plan = copy(z), copy(V)
        ys = reduce(hcat, [plan_outputs_rh(zz, r.ζ, r.free, s, d_plan; goals) for zz in zs])
        global pred = (mean=vec(mean(ys; dims=2)), sd=vec(std(ys; dims=2)), t_h=s.t_now / 3600)
    end
    done = s.t_now > 0 ? commit!(s, plan, CHECK; force=stable, force_horizon=pol.stable_horizon) : Symbol[]     # the prior's plan is a warm start, not a decision
    p = plan
    sdz = sqrt.(diag(V))
    @printf("%4.0f  %-15s  %-15s  %5.2f  %5.2f  %5.1f  %5.2f / %5.2f / %5.2f / %s / %5.2f   %5.3f±%5.3f %5.3f±%5.3f %5.1f±%4.1f %6.1f±%4.1f   %s   %.2f %.2f   %3.0f s\n",
            s.t_now / 3600, string(why), isempty(done) ? "-" : join(string.(done), ","),
            L_now, L_plan, kl,
            (isnan(p.t_wrap) ? NaN : p.t_wrap / 3600), p.t_pull / 3600, p.t_serve / 3600,
            p.spritz === nothing ? "-" : @sprintf("%.1f", p.spritz.t / 3600), (isnan(p.t_add) ? NaN : p.t_add / 3600),
            pred.mean[1], pred.sd[1], pred.mean[2], pred.sd[2], pred.mean[3], pred.sd[3], pred.mean[4] - 273.15, pred.sd[4],
            mis === nothing ? "  -  " : @sprintf("%5.2f", mis.z_chi2), noise[1][1], pit_level_now(s), time() - t0)
    flush(stdout)
    push!(record, Dict("t_h" => s.t_now / 3600, "reason" => string(why), "replanned" => due, "committed" => string.(done),
                       "L_now" => nn(L_now), "L_plan" => L_plan, "kl" => nn(kl), "stable" => string.(stable), "converged" => conv, "wall_s" => time() - t0,
                       "plan" => Dict("u" => p.u, "t_wrap_h" => nn(p.t_wrap / 3600), "t_pull_h" => p.t_pull / 3600, "t_serve_h" => p.t_serve / 3600,
                                      "spritz" => p.spritz === nothing ? nothing : Dict("t_h" => p.spritz.t / 3600, "dt_h" => p.spritz.dt / 3600, "m" => p.spritz.m),
                                      "t_add_h" => nn(p.t_add / 3600)),
                       "predicted" => Dict("mean" => pred.mean, "sd" => [nn(x) for x in pred.sd], "t_h" => pred.t_h),
                       "posterior_z" => z, "posterior_sd" => sdz, "posterior_cov" => [V[i, :] for i in axes(V, 1)],
                       "z_true" => truth_now(s.t_now, schedule_events(s).t_add), "err_sd" => (z .- truth_now(s.t_now, schedule_events(s).t_add)) ./ sdz,
                       "ridge" => ridge, "estimator" => est_info,
                       "misfit_z" => mis === nothing ? nothing : mis.z_chi2,
                       "noise" => Dict(chnames[i] => Dict("sigma" => v.σ, "rho" => v.ρ, "sigma_sd" => v.σ_sd, "rho_sd" => v.ρ_sd) for (i, v) in noise_post),
                       "pit_level_now" => pit_level_now(s)))
    # the checks so far, after every check: a crash or a kill later never loses them (pilot_outcome.jl recovers the
    # outcome from a partial record); the record is built before the file is opened, so a failure in building it
    # cannot truncate what was written
    partial = Dict("checks" => record, "check_h" => CHECK / 3600, "n_draws" => N_DRAWS, "n_replans" => n_replans, "prior_file" => prior_file, "partial" => true,
                   "estimator" => ESTIMATOR, "drift" => drift_record(s))
    open(io -> JSON.print(io, partial), joinpath(REPO, "data", "mock", out_name), "w")
    execute_damper!(s, plan, CHECK)
    !isnan(s.t_pull) && s.t_now + 1e-9 >= s.t_serve && break
end

# --- the plant's actual outcome under the executed schedule --------------------------------------
_, _, o = plant_readings(s)
nx = size(o.W_field, 1); w = E.fv_weights(nx)
wret = [dot(w, o.W_field[:, k]) / cfg.W0 for k in axes(o.W_field, 2)]
at(y, tq) = (k = clamp(searchsortedlast(GRID, tq), 1, length(GRID) - 1); f = (tq - GRID[k]) / (GRID[k + 1] - GRID[k]); (1 - f) * y[k] + f * y[k + 1])
y_act = cook_outputs(o, GRID, executed_du(GRID, damper_on_grid(s), s.t_pull), s.t_pull, isnan(s.t_add) ? BrisketDynamics.ADD_NEVER : s.t_add, s.t_serve, cfg, goals)
terms = objective_terms(y_act, goals)
actual = (J=terms.J, tenderness=terms.tenderness, appearance=terms.appearance, taste=terms.taste, surf=terms.surf, fuel=terms.fuel, smooth=terms.smooth, safety=terms.safety,
          C_serve=y_act[1], C_pull=at(view(o.C_field, 1, :), s.t_pull), T_pull=at(o.Tcore, s.t_pull) - 273.15, W=y_act[2], L=y_act[3], T_serve=y_act[4] - 273.15,
          S=y_act[5], juiciness=y_act[6])
reasons = Dict(r => count(rec -> rec["reason"] == r, record) for r in unique(rec["reason"] for rec in record))
@printf("\nexecuted: wrap %.2f h, pull %.2f h, serve %.2f h, spritz %s, fuel add %.2f h; %d checks, %d replans (%s); %.0f s total\n",
        s.t_wrap / 3600, s.t_pull / 3600, s.t_serve / 3600,
        s.spritz === nothing ? "-" : @sprintf("(%.1f h, every %.2f h, %.3f kg)", s.spritz.t / 3600, s.spritz.dt / 3600, s.spritz.m),
        s.t_add / 3600, length(record), n_replans, join(["$k $v" for (k, v) in sort(collect(reasons))], ", "), time() - t_start)
@printf("the plant's actual outcome: J %.3f (tenderness %.3f, appearance %.3f, taste %.3f, surf %.3f, fuel %.3f, smooth %.3f, safety %.3f); collagen at serve %.3f (%.3f at the pull, core %.1f C), water %.3f, juiciness %.3f, L* %.1f, smoke %.2f, core at serve %.1f C\n",
        actual.J, actual.tenderness, actual.appearance, actual.taste, actual.surf, actual.fuel, actual.smooth, actual.safety,
        actual.C_serve, actual.C_pull, actual.T_pull, actual.W, actual.juiciness, actual.L, actual.S, actual.T_serve)
final = Dict("checks" => record, "check_h" => CHECK / 3600, "n_draws" => N_DRAWS, "n_replans" => n_replans, "prior_file" => prior_file,
                        "estimator" => ESTIMATOR, "drift" => drift_record(s),
                        "policy" => Dict(string(k) => getfield(pol, k) for k in fieldnames(ReplanPolicy)),
                        "goals" => Dict(string(k) => getfield(goals, k) for k in fieldnames(PlanGoals)),
                        "executed" => Dict("t_wrap_h" => s.t_wrap / 3600, "t_pull_h" => s.t_pull / 3600, "t_serve_h" => s.t_serve / 3600,
                                           "t_add_h" => nn(s.t_add / 3600),
                                           "spritz" => s.spritz === nothing ? nothing : Dict("t_h" => s.spritz.t / 3600, "dt_h" => s.spritz.dt / 3600, "m" => s.spritz.m),
                                           "u" => [nn(x) for x in s.u], "t_h" => s.t ./ 3600),
                        "actual" => Dict(string(k) => getfield(actual, k) for k in fieldnames(typeof(actual))),
                        "plant" => dd === nothing ? "monolith (1-D) at the mock's truth, real-cook instruments (2-min grid)" :
                                   "monolith (1-D) at the mock's truth drifting (drift.jl, seed $DRIFT_SEED), real-cook instruments (2-min grid)",
                        "noise_truth" => Dict(String(k) => getfield(nc, k) for k in fieldnames(RealCookNoise)),
                        "z_true" => z_true0, "names" => allnames)
open(io -> JSON.print(io, final), joinpath(REPO, "data", "mock", out_name), "w")
println("wrote data/mock/", out_name)
