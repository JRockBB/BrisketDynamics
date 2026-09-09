# The pilot without hardware: the receding horizon closed on the matched
# mock plant. Every two hours the plant (the monolith at the mock's true
# latents) has run the executed schedule and produced its readings (pit,
# core at the true depth, humidity, scale, the mock's AR(1) noise
# pre-drawn on the grid); the estimator refits on the graph engine with
# the executed events as inputs; the posterior's draws feed the replan
# from the current state on the graph; the events that fall due commit
# and become irreversible; the damper executes for the interval. After
# the pull only the serve time is replanned; the cook ends at serve.
# The record holds every replan and the plant's actual outcome.
#
# Run:  julia --project=estimator estimator/mock_closed_loop.jl
# Writes data/mock/closed_loop.json.
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
GRID = collect(0.0:600.0:30 * 3600.0)
Δ = 2 * 3600.0
N_DRAWS = 6
goals = PlanGoals()

# --- the plant: the mock's truth --------------------------------------------------------------
x_true = Float64.(m["truth"]["x"])
θ_true, φ_true, A_true, depth_true = x_true[1:8], x_true[9:15], x_true[16], x_true[17]
L_true = θ_true[4]
nz = m["noise"]
function ar1_noise(rng, n, σ, ρ)
    e = zeros(n); e[1] = σ * randn(rng)
    for k in 2:n
        e[k] = ρ * e[k - 1] + σ * sqrt(1 - ρ^2) * randn(rng)
    end
    return e
end
rng_noise = MersenneTwister(11)
NOISE = Dict(ch => ar1_noise(rng_noise, length(GRID), Float64(nz[ch]["sigma"]), Float64(nz[ch]["rho"]))
             for ch in ("pit", "core", "humidity", "mass"))
interp1(v, pos) = (k = clamp(floor(Int, pos), 1, length(v) - 1); f = pos - k; (1 - f) * v[k] + f * v[k + 1])
fill_future(u) = (last = u[findlast(!isnan, u)]; [isnan(x) ? last : x for x in u])
nn(x) = isnan(x) ? nothing : x                     # an event never planned is null in the record, not NaN

"The plant's readings on the grid under the executed schedule (only those up to now are used)."
function plant_readings(s::Schedule)
    o = monolith_solve(θ_true, φ_true, fill_future(s.u), GRID, GRID; pit, cfg, A_meat=A_true, schedule_events(s)...)
    nx = size(o.T_field, 1)
    w = E.fv_weights(nx)
    pos = 1 + (1 - depth_true / L_true) * (nx - 1)
    clean = Dict("pit" => collect(o.Tpit), "core" => [interp1(o.T_field[:, k], pos) for k in axes(o.T_field, 2)],
                 "humidity" => collect(o.pv_pit),
                 "mass" => [E.RHO_DRY_EST * A_true * L_true * (1 + dot(w, o.W_field[:, k]) + dot(w, o.F_field[:, k])) for k in axes(o.W_field, 2)])
    return Dict(ch => clean[ch] .+ NOISE[ch] for ch in keys(clean)), o
end

# --- the estimator on the graph engine ------------------------------------------------------
function estimate(s::Schedule, readings, z_prev)
    d = estimator_setup(GRID, fill_future(s.u), pit, cfg, prior_phys, names)
    keep = GRID .<= s.t_now + 1e-9
    chs = [PitProbe(nz["pit"]["sigma"], nz["pit"]["rho"], keep; data=readings["pit"]),
           MeatProbe(nz["core"]["sigma"], nz["core"]["rho"], keep; depth_latent=DEPTH, data=readings["core"]),
           HumiditySensor(nz["humidity"]["sigma"], nz["humidity"]["rho"], keep; data=readings["humidity"]),
           MassScale(nz["mass"]["sigma"], nz["mass"]["rho"], keep; data=readings["mass"])]
    ph = Physics1D(area_latent=AREA, events=schedule_events(s))
    res = solve_map_channels(chs, prior, d; c0=z_prev, physics=ph, engine=:graph, verbose=false)
    return res.z, res.cov, res.misfit.total, d
end
draws(z, V, rng) = (Lc = cholesky(Symmetric(V)).L; [z .+ Lc * randn(rng, length(z)) for _ in 1:N_DRAWS])

# --- the loop ------------------------------------------------------------------------------------
s = Schedule(GRID; u0=0.15, t_serve=14 * 3600.0)
d0 = estimator_setup(GRID, fill(0.15, length(GRID)), pit, cfg, prior_phys, names)
z = vcat(d0.z0, [log(cfg.A_meat), log(0.8 * L_med)])
_, _, h0 = E.prior_terms(z, prior)
V = Matrix(Diagonal(1 ./ h0))
z_true = vcat([isbeta(p) ? log(v / (1 - v)) : log(v) for (v, p) in zip(x_true[1:15], prior_phys)], log(A_true), log(depth_true))
plan_prev = nothing
record = []
rng_draws = MersenneTwister(7)
t_start = time()
println("t[h]  events committed             plan: dampers / wrap / pull / serve / spritz(t,dt,m) / add    outcomes predicted (collagen at the pull, water, L*, serve C)   post sd L,h_c,A   misfit z")
flush(stdout)
while s.t_now < 28 * 3600
    t0 = time()
    mis = nothing
    if s.t_now > 0
        readings, _ = plant_readings(s)
        global z, V, mis, d0 = estimate(s, readings, z)
    end
    zs = draws(z, V, rng_draws)
    ζ_init = plan_prev === nothing ? nothing : encode_plan(plan_prev, free_set(s), s)
    r = replan_on_graph(zs, s, d0; goals, ζ_init, verbose=false)
    ys = reduce(hcat, [plan_outputs_rh(zz, r.ζ, r.free, s, d0) for zz in zs])
    pred = (mean=vec(mean(ys; dims=2)), sd=vec(std(ys; dims=2)))
    done = commit!(s, r.plan, Δ)
    p = r.plan
    sdz = sqrt.(diag(V))
    @printf("%4.0f  %-26s  %s / %5.2f / %5.2f / %5.2f / %s / %5.2f   %6.1f±%4.1f %5.3f±%5.3f %5.1f±%4.1f %6.1f±%4.1f   %.3f %.3f %.3f   %s\n",
            s.t_now / 3600, isempty(done) ? "-" : join(string.(done), ","),
            string(round.(p.u; digits=2)), (isnan(p.t_wrap) ? NaN : p.t_wrap / 3600), p.t_pull / 3600, p.t_serve / 3600,
            p.spritz === nothing ? "-" : @sprintf("(%.1f,%.2f,%.3f)", p.spritz.t / 3600, p.spritz.dt / 3600, p.spritz.m),
            (isnan(p.t_add) ? NaN : p.t_add / 3600),
            pred.mean[1], pred.sd[1], pred.mean[2], pred.sd[2], pred.mean[3], pred.sd[3], pred.mean[4] - 273.15, pred.sd[4],
            sdz[4], sdz[3], sdz[AREA], mis === nothing ? "-" : @sprintf("%.2f", mis.z_chi2))
    flush(stdout)
    push!(record, Dict("t_h" => s.t_now / 3600, "committed" => string.(done), "free" => length(r.free), "F" => r.F,
                       "iterations" => length(r.hist), "converged" => r.converged, "wall_s" => time() - t0,
                       "plan" => Dict("u" => p.u, "t_wrap_h" => nn(p.t_wrap / 3600), "t_pull_h" => p.t_pull / 3600, "t_serve_h" => p.t_serve / 3600,
                                      "spritz" => p.spritz === nothing ? nothing : Dict("t_h" => p.spritz.t / 3600, "dt_h" => p.spritz.dt / 3600, "m" => p.spritz.m),
                                      "t_add_h" => nn(p.t_add / 3600)),
                       "predicted" => Dict("mean" => pred.mean, "sd" => pred.sd),
                       "posterior_z" => z, "posterior_sd" => sdz, "err_sd" => (z .- z_true) ./ sdz,
                       "misfit_z" => mis === nothing ? nothing : mis.z_chi2))
    execute_damper!(s, r.plan, Δ)
    global plan_prev = r.plan
    !isnan(s.t_pull) && s.t_now + 1e-9 >= s.t_serve && break
end

# --- the plant's actual outcome under the executed schedule --------------------------------------
_, o = plant_readings(s)
nx = size(o.W_field, 1); w = E.fv_weights(nx)
wret = [dot(w, o.W_field[:, k]) / cfg.W0 for k in axes(o.W_field, 2)]
at(y, tq) = (k = clamp(searchsortedlast(GRID, tq), 1, length(GRID) - 1); f = (tq - GRID[k]) / 600.0; (1 - f) * y[k] + f * y[k + 1])
actual = (C_pull=at(view(o.C_field, 1, :), s.t_pull), T_pull=at(o.Tcore, s.t_pull) - 273.15, W=at(wret, s.t_serve), L=at(o.l_star, s.t_serve), T_serve=at(o.Tcore, s.t_serve) - 273.15)
@printf("\nexecuted: wrap %.2f h, pull %.2f h, serve %.2f h, spritz %s, fuel add %.2f h; %.0f s total\n",
        s.t_wrap / 3600, s.t_pull / 3600, s.t_serve / 3600,
        s.spritz === nothing ? "-" : @sprintf("(%.1f h, every %.2f h, %.3f kg)", s.spritz.t / 3600, s.spritz.dt / 3600, s.spritz.m),
        s.t_add / 3600, time() - t_start)
@printf("the plant's actual outcome: collagen conversion at the pull %.3f (goal %.2f ± %.2f) with the core at %.1f C, retained water %.3f (goal %.2f), bark L* %.1f (goal %.0f), core at serve %.1f C (floor %.0f)\n",
        actual.C_pull, goals.C_pull, goals.sd_C, actual.T_pull, actual.W, goals.W, actual.L, goals.L, actual.T_serve, goals.T_serve_floor - 273.15)
open(joinpath(REPO, "data", "mock", "closed_loop.json"), "w") do io
    JSON.print(io, Dict("replans" => record, "delta_h" => Δ / 3600, "n_draws" => N_DRAWS,
                        "executed" => Dict("u" => [isnan(x) ? nothing : x for x in s.u], "t_wrap_h" => nn(s.t_wrap / 3600), "t_pull_h" => nn(s.t_pull / 3600), "t_serve_h" => s.t_serve / 3600,
                                           "spritz" => s.spritz === nothing ? nothing : Dict("t_h" => s.spritz.t / 3600, "dt_h" => s.spritz.dt / 3600, "m" => s.spritz.m),
                                           "t_add_h" => nn(s.t_add / 3600)),
                        "actual" => Dict(pairs(actual)), "goals" => Dict(String(k) => getfield(goals, k) for k in fieldnames(PlanGoals)), "z_true" => z_true, "names" => vcat(names, ["A_meat", "core_depth"])))
end
println("wrote data/mock/closed_loop.json")
