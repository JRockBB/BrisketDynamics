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
# Run:  julia --project=estimator estimator/mock_closed_loop_realcook.jl
# Writes data/mock/closed_loop_realcook.json.
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
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
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
N_DRAWS = 6
goals = PlanGoals()
pol = ReplanPolicy()
nc = RealCookNoise()

# --- the plant: the mock's truth, read like a real cook ----------------------------------------
x_true = Float64.(m["truth"]["x"])
θ_true, φ_true, A_true, depth_true = x_true[1:8], x_true[9:15], x_true[16], x_true[17]
L_true = θ_true[4]
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
    o = monolith_solve(θ_true, φ_true, damper_on_grid(s), GRID, GRID; pit, cfg, A_meat=A_true, ev...)
    r, _, masks, _ = realcook_readings(o, GRID, ev, nc, MersenneTwister(23); A=A_true, L=L_true, depth=depth_true)
    return r, masks, o
end

# --- the estimator: channels at their rates, noise identified, the level as a process ----------
cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json"))
dt = GRID[2] - GRID[1]
tostep(np, from_dt) = (a = dt / from_dt; NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean^a, np.ρ_sd * a * np.ρ_mean^(a - 1); floor=np.floor))
np = Dict(1 => tostep(cal["pit"], 600.0), 2 => tostep(cal["core"], 600.0),
          3 => NoisePrior(100.0, 100.0, 0.8^(dt / 600), 0.3), 4 => NoisePrior(0.02, 0.02, 0.5^(dt / 1800), 0.3))
chnames = Dict(1 => "pit", 2 => "core", 3 => "humidity", 4 => "mass")
noise = Dict(i => (v.σ_mean, v.ρ_mean) for (i, v) in np)                  # carried from check to check
levels = Dict{Int,Union{Nothing,Vector{Float64}}}(i => nothing for i in keys(np))
function estimate!(s::Schedule, z_prev)
    readings, masks, _ = plant_readings(s)
    keep = GRID .<= s.t_now + 1e-9
    d = estimator_setup(GRID, damper_on_grid(s), pit, cfg, prior_phys, names)
    base = [PitProbe(noise[1]..., masks.pit .& keep; data=readings.pit),
            MeatProbe(noise[2]..., masks.core .& keep; depth_latent=DEPTH, data=readings.core),
            CensoredHumidity(noise[3]..., masks.humidity .& keep; cap=nc.hum_cap, data=readings.humidity),
            MassScale(noise[4]..., masks.mass .& keep; data=readings.mass)]
    chs = [levels[i] === nothing ? ch : Levelled(ch, levels[i]) for (i, ch) in enumerate(base)]
    ph = Physics1D(area_latent=AREA, events=schedule_events(s))
    res = solve_map_channels(chs, prior, d; c0=z_prev, physics=ph, engine=:graph, noise=np, noise_rounds=1,
                             levelled=true, verbose=false)
    for (i, v) in res.noise
        noise[i] = (v.σ, v.ρ)
        levels[i] = res.channels[i] isa Levelled ? res.channels[i].level : nothing
    end
    return res.z, res.cov, res.misfit.total, res.noise, res.converged
end
draws(z, V, rng) = (Lc = cholesky(Symmetric(V)).L; [z .+ Lc * randn(rng, length(z)) for _ in 1:N_DRAWS])
"The pit level at the latest sample (1 when none is inferred)."
pit_level_now(s) = (lv = levels[1]; lv === nothing ? 1.0 : (k = findlast(GRID .<= s.t_now + 1e-9); lv[k]))

# --- the loop ------------------------------------------------------------------------------------
s = Schedule(PLAN_GRID; u0=0.15, t_serve=14 * 3600.0)
d_plan = estimator_setup(PLAN_GRID, fill(0.15, length(PLAN_GRID)), pit, cfg, prior_phys, names)   # the plan physics reads its prior, pit, cfg, tolerances
z = vcat(d_plan.z0, [log(cfg.A_meat), log(0.8 * L_med)])
_, _, h0 = E.prior_terms(z, prior)
V = Matrix(Diagonal(1 ./ h0))
z_true = vcat([isbeta(p) ? log(v / (1 - v)) : log(v) for (v, p) in zip(x_true[1:15], prior_phys)], log(A_true), log(depth_true))
plan = nothing
L_plan = Inf
t_last = -Inf
z_at_plan, V_at_plan = copy(z), copy(V)          # the belief the standing plan was made on
pred = nothing
record = []
n_replans = 0
rng_draws = MersenneTwister(7)
t_start = time()
println("t[h]  reason           committed        L_now  L_plan  KL     plan: wrap / pull / serve / spritz / add   outcomes at the last replan (collagen at the pull, water, L*, serve C)   misfit z   pit σ, level   wall")
flush(stdout)
while s.t_now < 28 * 3600
    t0 = time()
    mis = nothing; noise_post = Dict{Int,NamedTuple}(); conv = true
    if s.t_now > 0
        global z, V, mis, noise_post, conv = estimate!(s, z)
    end
    zs = draws(z, V, rng_draws)
    L_now = plan === nothing ? NaN : plan_loss(zs, plan, s, d_plan; goals).mean
    kl = plan === nothing ? NaN : posterior_kl(z, V, z_at_plan, V_at_plan)
    due, why = replan_due(pol, s, plan, L_now, L_plan, t_last, mis === nothing ? nothing : mis.z_chi2; kl=isnan(kl) ? nothing : kl)
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
        ys = reduce(hcat, [plan_outputs_rh(zz, r.ζ, r.free, s, d_plan) for zz in zs])
        global pred = (mean=vec(mean(ys; dims=2)), sd=vec(std(ys; dims=2)), t_h=s.t_now / 3600)
    end
    done = commit!(s, plan, CHECK; force=stable)
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
                       "predicted" => Dict("mean" => pred.mean, "sd" => pred.sd, "t_h" => pred.t_h),
                       "posterior_z" => z, "posterior_sd" => sdz, "err_sd" => (z .- z_true) ./ sdz,
                       "misfit_z" => mis === nothing ? nothing : mis.z_chi2,
                       "noise" => Dict(chnames[i] => Dict("sigma" => v.σ, "rho" => v.ρ, "sigma_sd" => v.σ_sd, "rho_sd" => v.ρ_sd) for (i, v) in noise_post),
                       "pit_level_now" => pit_level_now(s)))
    execute_damper!(s, plan, CHECK)
    !isnan(s.t_pull) && s.t_now + 1e-9 >= s.t_serve && break
end

# --- the plant's actual outcome under the executed schedule --------------------------------------
_, _, o = plant_readings(s)
nx = size(o.W_field, 1); w = E.fv_weights(nx)
wret = [dot(w, o.W_field[:, k]) / cfg.W0 for k in axes(o.W_field, 2)]
at(y, tq) = (k = clamp(searchsortedlast(GRID, tq), 1, length(GRID) - 1); f = (tq - GRID[k]) / (GRID[k + 1] - GRID[k]); (1 - f) * y[k] + f * y[k + 1])
actual = (C_pull=at(view(o.C_field, 1, :), s.t_pull), T_pull=at(o.Tcore, s.t_pull) - 273.15, W=at(wret, s.t_serve), L=at(o.l_star, s.t_serve), T_serve=at(o.Tcore, s.t_serve) - 273.15)
reasons = Dict(r => count(rec -> rec["reason"] == r, record) for r in unique(rec["reason"] for rec in record))
@printf("\nexecuted: wrap %.2f h, pull %.2f h, serve %.2f h, spritz %s, fuel add %.2f h; %d checks, %d replans (%s); %.0f s total\n",
        s.t_wrap / 3600, s.t_pull / 3600, s.t_serve / 3600,
        s.spritz === nothing ? "-" : @sprintf("(%.1f h, every %.2f h, %.3f kg)", s.spritz.t / 3600, s.spritz.dt / 3600, s.spritz.m),
        s.t_add / 3600, length(record), n_replans, join(["$k $v" for (k, v) in sort(collect(reasons))], ", "), time() - t_start)
@printf("the plant's actual outcome: collagen conversion at the pull %.3f (band %.2f to %.2f) with the core at %.1f C, retained water %.3f (floor %.2f), bark L* %.1f (band %.0f to %.0f), core at serve %.1f C (floor %.0f)\n",
        actual.C_pull, goals.C_lo, goals.C_hi, actual.T_pull, actual.W, goals.W_lo, actual.L, goals.L_lo, goals.L_hi, actual.T_serve, goals.T_serve_floor - 273.15)
open(joinpath(REPO, "data", "mock", "closed_loop_realcook.json"), "w") do io
    JSON.print(io, Dict("checks" => record, "check_h" => CHECK / 3600, "n_draws" => N_DRAWS, "n_replans" => n_replans,
                        "policy" => Dict(string(k) => getfield(pol, k) for k in fieldnames(ReplanPolicy)),
                        "goals" => Dict(string(k) => getfield(goals, k) for k in fieldnames(PlanGoals)),
                        "executed" => Dict("t_wrap_h" => s.t_wrap / 3600, "t_pull_h" => s.t_pull / 3600, "t_serve_h" => s.t_serve / 3600,
                                           "t_add_h" => nn(s.t_add / 3600),
                                           "spritz" => s.spritz === nothing ? nothing : Dict("t_h" => s.spritz.t / 3600, "dt_h" => s.spritz.dt / 3600, "m" => s.spritz.m),
                                           "u" => [nn(x) for x in s.u], "t_h" => s.t ./ 3600),
                        "actual" => Dict(string(k) => getfield(actual, k) for k in fieldnames(typeof(actual))),
                        "plant" => "monolith (1-D) at the mock's truth, real-cook instruments (2-min grid)",
                        "noise_truth" => Dict(String(k) => getfield(nc, k) for k in fieldnames(RealCookNoise)),
                        "z_true" => z_true, "names" => vcat(names, ["A_meat", "core_depth"])))
end
println("wrote data/mock/closed_loop_realcook.json")
