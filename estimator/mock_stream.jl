# The streaming estimator on a mock cook: hourly full-history refits for
# three channel sets, the posterior per hour and its standardized error
# against the plant's truth — the "watch it work, watch it fail" run.
#
# Run:  julia --project=estimator estimator/mock_stream.jl data/mock/mock_2d.json
# Writes data/mock/<name>_stream.json.
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf

REPO = normpath(joinpath(@__DIR__, ".."))
path = length(ARGS) >= 1 ? ARGS[1] : joinpath(REPO, "data", "mock", "mock_2d.json")
m = JSON.parsefile(path)
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
t = Float64.(m["t"])
u = Float64.(m["u"])
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
names = String.(gate["names"])
d = estimator_setup(t, u, pit, cfg, prior_phys, names)

# estimator-only latents: the exposed area and the core probe's depth
L_med = exp(prior_phys[4].p1)
extra = [LatentSpec("A_meat", "lognormal", log(cfg.A_meat), 0.3),
         LatentSpec("core_depth", "lognormal", log(0.8 * L_med), 0.5)]
prior = vcat(prior_phys, [e.site for e in extra])
allnames = vcat(names, [e.name for e in extra])
AREA, DEPTH = length(prior) - 1, length(prior)

x_true = Float64.(m["truth"]["x"])
z_true = [isbeta(p) ? log(x / (1 - x)) : log(x) for (x, p) in zip(x_true, prior)]

rd = m["readings"]
nz = m["noise"]
full = trues(length(t))
rng = MersenneTwister(21)
pit_ch = PitProbe(nz["pit"]["sigma"], nz["pit"]["rho"], full; data=Float64.(rd["pit"]))
core_ch = MeatProbe(nz["core"]["sigma"], nz["core"]["rho"], full; depth_latent=DEPTH,
                    data=Float64.(rd["core"]))
hum_ch = HumiditySensor(nz["humidity"]["sigma"], nz["humidity"]["rho"], full; data=Float64.(rd["humidity"]))
scale_ch = MassScale(nz["mass"]["sigma"], nz["mass"]["rho"], full; area_latent=AREA,
                     data=Float64.(rd["mass"]))
ruler_L = StaticReading(4, 0.003; data=x_true[4] + 0.003 * randn(rng))
ruler_A = StaticReading(AREA, 0.01; data=x_true[AREA] + 0.01 * randn(rng))

sets = [
    ("pit + core", [pit_ch, core_ch]),
    ("pit + core + humidity + scale", [pit_ch, core_ch, hum_ch, scale_ch]),
    ("+ rulers (thickness, footprint)", [pit_ch, core_ch, hum_ch, scale_ch, ruler_L, ruler_A]),
]
edges = 3600.0 .* collect(1:floor(Int, t[end] / 3600 + 1e-9))

println("plant: ", m["plant"], "  (", length(edges), " hourly batches)")
out_sets = []
for (label, chs) in sets
    println("set: ", label)
    flush(stdout)
    t0 = time()
    recs = stream_fit(chs, prior, d, edges; area_latent=AREA, verbose=true)
    batches = [Dict("t_h" => r.t / 3600, "z" => r.z, "sd" => r.sd,
                    "err_sd" => (r.z .- z_true) ./ r.sd, "F" => r.F, "iters" => r.iters,
                    "wall_s" => r.wall) for r in recs]
    last = recs[end]
    @printf("  final |err|/sd: max %.2f, median %.2f; wall %.0f s\n",
            maximum(abs.((last.z .- z_true) ./ last.sd)), sort(abs.((last.z .- z_true) ./ last.sd))[div(end + 1, 2)],
            time() - t0)
    flush(stdout)
    push!(out_sets, Dict("label" => label, "batches" => batches))
end

stem = splitext(basename(path))[1]
dst = joinpath(REPO, "data", "mock", stem * "_stream.json")
open(dst, "w") do io
    JSON.print(io, Dict("plant" => m["plant"], "names" => allnames, "z_true" => z_true,
                        "x_true" => x_true, "sets" => out_sets, "edges_h" => edges ./ 3600))
end
println("wrote ", dst)
