# The streaming estimator with the noise identified as it goes: the
# alternative-noise mock (mock_plant.jl --altnoise: a noisier pit probe,
# a quieter and less correlated core probe, a worse humidity sensor, a
# better scale) streamed hourly with every AR channel's sd and
# correlation estimated by the AR node from the calibration's prior
# (pit, core) or an assumed one (humidity, scale), all of them wrong for
# this plant. The record carries the noise posterior per batch.
#
# Run:  julia --project=estimator estimator/scripts/mock_stream_noise.jl data/mock/mock_1d_alt.json
# Writes data/mock/<name>_noise_stream.json.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf

REPO = normpath(joinpath(@__DIR__, "..", ".."))
path = length(ARGS) >= 1 ? ARGS[1] : joinpath(REPO, "data", "mock", "mock_1d_alt.json")
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

L_med = exp(prior_phys[4].p1)
extra = [LatentSpec("A_meat", "lognormal", log(cfg.A_meat), 0.3),
         LatentSpec("core_depth", "lognormal", log(0.8 * L_med), 0.5)]
prior = vcat(prior_phys, [e.site for e in extra])
allnames = vcat(names, [e.name for e in extra])
AREA, DEPTH = length(prior) - 1, length(prior)
x_true = Float64.(m["truth"]["x"])
z_true = [isbeta(p) ? log(x / (1 - x)) : log(x) for (x, p) in zip(x_true, prior)]

cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json"))
priors = Dict("pit" => cal["pit"], "core" => cal["core"],
              "humidity" => NoisePrior(100.0, 100.0, 0.8, 0.3), "mass" => NoisePrior(0.02, 0.02, 0.5, 0.3))
truth = Dict(String(k) => (Float64(v["sigma"]), Float64(v["rho"])) for (k, v) in m["noise"])
println("plant: ", m["plant"], "; true noise ", truth)

rd = m["readings"]
full = trues(length(t))
start(np) = (np.σ_mean, np.ρ_mean)                 # every channel starts at its prior's means
pit_ch = PitProbe(start(priors["pit"])..., full; data=Float64.(rd["pit"]))
core_ch = MeatProbe(start(priors["core"])..., full; depth_latent=DEPTH, data=Float64.(rd["core"]))
hum_ch = HumiditySensor(start(priors["humidity"])..., full; data=Float64.(rd["humidity"]))
scale_ch = MassScale(start(priors["mass"])..., full; data=Float64.(rd["mass"]))

sets = [
    ("pit + core", [pit_ch, core_ch], Dict(1 => priors["pit"], 2 => priors["core"]), ["pit", "core"]),
    ("pit + core + humidity + scale", [pit_ch, core_ch, hum_ch, scale_ch],
     Dict(1 => priors["pit"], 2 => priors["core"], 3 => priors["humidity"], 4 => priors["mass"]),
     ["pit", "core", "humidity", "mass"]),
]
edges = 3600.0 .* collect(1:floor(Int, t[end] / 3600 + 1e-9))

out_sets = []
for (label, chs, np, chnames) in sets
    println("set: ", label)
    flush(stdout)
    t0 = time()
    recs = stream_fit(chs, prior, d, edges; area_latent=AREA, noise=np, verbose=true)
    batches = [Dict("t_h" => r.t / 3600, "z" => r.z, "sd" => r.sd,
                    "err_sd" => (r.z .- z_true) ./ r.sd, "F" => r.F, "iters" => r.iters,
                    "wall_s" => r.wall, "misfit" => r.misfit,
                    "noise" => Dict(chnames[i] => Dict(pairs(v)) for (i, v) in r.noise)) for r in recs]
    last = recs[end]
    e = abs.((last.z .- z_true) ./ last.sd)
    mt = last.misfit.total
    @printf("  final |err|/sd: max %.2f (%s), median %.2f; residual power/row %.3f, z_chi2 %.2f; wall %.0f s\n",
            maximum(e), allnames[argmax(e)], sort(e)[div(end + 1, 2)], mt.ss / mt.n, mt.z_chi2, time() - t0)
    for (i, v) in sort(collect(last.noise); by=first)
        tr = truth[chnames[i]]
        @printf("    %-8s σ %.4g ± %.2g (truth %.4g, prior %.4g)   ρ %.3f ± %.3f (truth %.2f, prior %.2f)\n",
                chnames[i], v.σ, v.σ_sd, tr[1], np[i].σ_mean, v.ρ, v.ρ_sd, tr[2], np[i].ρ_mean)
    end
    flush(stdout)
    push!(out_sets, Dict("label" => label, "batches" => batches, "channels" => chnames))
end

stem = splitext(basename(path))[1]
dst = joinpath(REPO, "data", "mock", stem * "_noise_stream.json")
open(dst, "w") do io
    JSON.print(io, Dict("plant" => m["plant"] * ", noise identified", "names" => allnames,
                        "z_true" => z_true, "x_true" => x_true, "sets" => out_sets,
                        "noise_truth" => Dict(k => Dict("sigma" => v[1], "rho" => v[2]) for (k, v) in truth),
                        "noise_prior" => Dict(k => Dict("sigma" => v.σ_mean, "rho" => v.ρ_mean) for (k, v) in priors),
                        "edges_h" => edges ./ 3600))
end
println("wrote ", dst)
