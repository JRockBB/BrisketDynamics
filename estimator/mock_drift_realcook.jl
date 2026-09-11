# The real-cook mock with parameters that MOVE during the cook: the plant solved in hourly segments with the
# state handed from one to the next (segments.jl), the leak and the surface coefficients on random walks, a
# new charge of fuel at the add with its own burn rate and enthalpy. The readings are the real-cook
# instruments' (realcook.jl), the schedule the real-cook mock's. A constant-parameter estimator on this plant
# averages what moved; the filter (filter.jl) is judged on it by coverage against the truth AT EACH HOUR.
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

# the drift: log (or logit) random walks per hour on the sites that move on a real pit, a jump at the add
DRIFT = Dict("u_leak" => 0.05, "k_e" => 0.05, "h_c" => 0.03)          # sd per hour, in the unconstrained coordinate
JUMP = Dict("r_max" => 0.15, "dH_eff" => 0.10)                         # log-sd of the new charge's values at the add
hours = collect(1.0:1.0:30.0)
edges = [findlast(t .<= h * 3600) for h in hours[1:(end - 1)]]
rng = MersenneTwister(seed)
xs = Vector{Vector{Float64}}()
x = copy(x0)
for (k, h) in enumerate(hours)
    global x
    if k > 1
        for (nm, sd) in DRIFT
            i = idx[nm]
            x[i] = nm == "u_leak" ? (v = log(x[i] / (1 - x[i])) + sd * randn(rng); 1 / (1 + exp(-v))) : x[i] * exp(sd * randn(rng))
        end
        if hours[k - 1] < events.t_add / 3600 <= h        # the segment that starts at the add burns the new charge
            for (nm, sd) in JUMP
                x[idx[nm]] = x0[idx[nm]] * exp(sd * randn(rng))
            end
        end
    end
    push!(xs, copy(x))
end
θs = [xx[1:8] for xx in xs]; φs = [xx[9:15] for xx in xs]
A_true, depth_true, L_true = x0[16], x0[17], x0[4]
o = solve_segments(θs, φs, u, t, edges; pit, cfg, A_meat=A_true, events...)
rng_r = MersenneTwister(1000 + seed)
readings, clean, masks, lid = realcook_readings(o, t, events, nc, rng_r; A=A_true, L=L_true, depth=depth_true)
@printf("drift mock (seed %d): %d hourly segments; u_leak %.3f -> %.3f, k_e %.2e -> %.2e, h_c %.2f -> %.2f; the add's charge r_max %.2e -> %.2e, dH_eff %.3e -> %.3e; core at the pull %.1f C\n",
        seed, length(hours), xs[1][idx["u_leak"]], xs[end][idx["u_leak"]], xs[1][idx["k_e"]], xs[end][idx["k_e"]], xs[1][idx["h_c"]], xs[end][idx["h_c"]],
        xs[1][idx["r_max"]], xs[end][idx["r_max"]], xs[1][idx["dH_eff"]], xs[end][idx["dH_eff"]], clean.core[searchsortedlast(t, events.t_pull)] - 273.15)
dst = joinpath(REPO, "data", "mock", "mock_drift.json")
open(dst, "w") do io
    JSON.print(io, Dict("plant" => "monolith (1-D) in hourly segments, parameters drifting; real-cook instruments", "t" => t, "u" => u, "meta" => meta,
                        "truth" => m["truth"], "truth_by_hour" => Dict("hours" => hours, "x" => xs, "names" => names),
                        "drift" => Dict("sd_per_hour" => DRIFT, "jump_log_sd" => JUMP, "seed" => seed),
                        "events" => Dict(String(k) => v for (k, v) in pairs(events)),
                        "noise" => Dict(String(k) => getfield(nc, k) for k in fieldnames(RealCookNoise)),
                        "readings" => Dict(String(k) => v for (k, v) in pairs(readings)),
                        "clean" => Dict(String(k) => v for (k, v) in pairs(clean)),
                        "masks" => Dict(String(k) => v for (k, v) in pairs(masks)), "lid_open" => lid))
end
println("wrote ", dst)
