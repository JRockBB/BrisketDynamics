# The real-cook mock: the matched plant on a two-minute grid over 30 h
# with the executed schedule the full plan block found (wrap 11 h,
# spritz from 8.5 h every 1.6 h at 46 g, fuel add 12 h, pull 15.7 h),
# read by instruments at their own rates with fire-regime noise, lid
# lifts and a saturating humidity sensor (realcook.jl).
#
# Run:  julia --project=estimator estimator/scripts/mock_realcook.jl
# Writes data/mock/mock_realcook.json.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json"))
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
x_true = Float64.(m["truth"]["x"])
θ_true, φ_true, A_true, depth_true = x_true[1:8], x_true[9:15], x_true[16], x_true[17]
L_true = θ_true[4]

GRID = collect(0.0:120.0:30 * 3600.0)
# the executed damper: the mock's 10-minute log, linearly interpolated onto the fine grid, held after its end
t_log = Float64.(m["t"]); u_log = Float64.(m["u"])
u = [ti >= t_log[end] ? u_log[end] : (k = searchsortedlast(t_log, ti); k >= length(t_log) ? u_log[end] :
      u_log[k] + (u_log[k + 1] - u_log[k]) * (ti - t_log[k]) / (t_log[k + 1] - t_log[k])) for ti in GRID]
events = (t_wrap=11.0 * 3600, t_pull=15.7 * 3600, t_add=12.0 * 3600, t_spritz=8.5 * 3600, dt_spritz=1.6 * 3600, m_spritz=0.046)
nc = RealCookNoise()

o = monolith_solve(θ_true, φ_true, u, GRID, GRID; pit, cfg, A_meat=A_true, events...)
rng = MersenneTwister(23)
readings, clean, masks, lid = realcook_readings(o, GRID, events, nc, rng; A=A_true, L=L_true, depth=depth_true)
@printf("grid %d points at %.0f s; samples: pit %d, core %d, humidity %d (%d at the cap), scale %d; lid-open points %d\n",
        length(GRID), GRID[2], count(masks.pit), count(masks.core), count(masks.humidity),
        count(readings.humidity[masks.humidity] .>= nc.hum_cap - 1e-9), count(masks.mass), count(lid))
@printf("core at the pull %.1f C; pit sd bump after the fuel add at %.0f h\n", clean.core[searchsortedlast(GRID, events.t_pull)] - 273.15, events.t_add / 3600)

dst = joinpath(REPO, "data", "mock", "mock_realcook.json")
open(dst, "w") do io
    JSON.print(io, Dict("plant" => "monolith (1-D), real-cook instruments", "t" => GRID, "u" => u, "meta" => meta,
                        "truth" => m["truth"], "events" => Dict(String(k) => v for (k, v) in pairs(events)),
                        "noise" => Dict(String(k) => getfield(nc, k) for k in fieldnames(RealCookNoise)),
                        "readings" => Dict(String(k) => v for (k, v) in pairs(readings)),
                        "clean" => Dict(String(k) => v for (k, v) in pairs(clean)),
                        "masks" => Dict(String(k) => v for (k, v) in pairs(masks)), "lid_open" => lid))
end
println("wrote ", dst)
