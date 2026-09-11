# The real-cook mock with parameters that MOVE during the cook: the plant solved in segments with the state
# handed from one to the next (segments.jl), the leak and the surface coefficients on random walks stepping
# on the hour, a new charge of fuel at the add with its own burn rate and enthalpy (drift.jl). The readings
# are the real-cook instruments' (realcook.jl), the schedule the real-cook mock's. A constant-parameter
# estimator on this plant averages what moved; the smoother (smoother.jl) is judged on plants like it by
# coverage against the truth in force at each hour (coverage_drift.jl).
#
# Run:  julia --project=estimator estimator/mock_drift_realcook.jl [seed]
# Writes data/mock/mock_drift.json (the layout of mock_realcook.json plus "truth_by_hour" and "drift").
using Pkg
Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, ".."))
seed = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 23
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
t = Float64.(m["t"]); u = Float64.(m["u"])
names = String.(m["truth"]["names"])
x0 = Float64.(m["truth"]["x"])
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]),
          dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
nc = RealCookNoise()
idx = Dict(n => i for (i, n) in enumerate(names))
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]   # the families give the unconstrained coordinate
tz(x, p) = isbeta(p) ? log(x / (1 - x)) : log(x)
fz(v, p) = isbeta(p) ? 1 / (1 + exp(-v)) : exp(v)
z0 = [tz(x0[i], prior_phys[i]) for i in 1:N_PHYS]

# the drift: random walks stepping on the hour on the sites that move on a real pit, the add's charge at the add (drift.jl)
dd = DriftDraw(MersenneTwister(seed), idx)
edges, blocks = drift_segments(t, z0, dd, events.t_add)
xs = [[fz(z[i], prior_phys[i]) for i in 1:N_PHYS] for z in blocks]
θs = [xx[1:8] for xx in xs]; φs = [xx[9:15] for xx in xs]
A_true, depth_true, L_true = x0[16], x0[17], x0[4]
o = solve_segments(θs, φs, u, t, edges; pit, cfg, A_meat=A_true, events...)
rng_r = MersenneTwister(1000 + seed)
readings, clean, masks, lid = realcook_readings(o, t, events, nc, rng_r; A=A_true, L=L_true, depth=depth_true)
hours = collect(0.0:1.0:29.0)
x_by_hour = [[fz(z[i], prior_phys[i]) for i in 1:N_PHYS] for z in [truth_at(z0, dd, h * 3600, events.t_add; from=true) for h in hours]]
@printf("drift mock (seed %d): %d segments; u_leak %.3f -> %.3f, k_e %.2e -> %.2e, h_c %.2f -> %.2f; the add's charge r_max %.2e -> %.2e, dH_eff %.3e -> %.3e; core at the pull %.1f C\n",
        seed, length(edges) + 1, xs[1][idx["u_leak"]], xs[end][idx["u_leak"]], xs[1][idx["k_e"]], xs[end][idx["k_e"]], xs[1][idx["h_c"]], xs[end][idx["h_c"]],
        xs[1][idx["r_max"]], xs[end][idx["r_max"]], xs[1][idx["dH_eff"]], xs[end][idx["dH_eff"]], clean.core[searchsortedlast(t, events.t_pull)] - 273.15)
dst = joinpath(REPO, "data", "mock", "mock_drift.json")
open(dst, "w") do io
    JSON.print(io, Dict("plant" => "monolith (1-D) in segments on the hour and at the add, parameters drifting; real-cook instruments", "t" => t, "u" => u, "meta" => meta,
                        "truth" => m["truth"], "truth_by_hour" => Dict("hours" => hours, "x" => x_by_hour, "names" => names[1:N_PHYS]),
                        "drift" => Dict("sd_per_hour" => DRIFT_SD_PER_HOUR, "jump_log_sd" => JUMP_LOG_SD, "seed" => seed, "segment_edges_h" => t[edges] ./ 3600),
                        "events" => Dict(String(k) => v for (k, v) in pairs(events)),
                        "noise" => Dict(String(k) => getfield(nc, k) for k in fieldnames(RealCookNoise)),
                        "readings" => Dict(String(k) => v for (k, v) in pairs(readings)),
                        "clean" => Dict(String(k) => v for (k, v) in pairs(clean)),
                        "masks" => Dict(String(k) => v for (k, v) in pairs(masks)), "lid_open" => lid))
end
println("wrote ", dst)
