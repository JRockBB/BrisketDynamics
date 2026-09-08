# The streaming estimator with a shape-factor latent on the scale: the
# thing the misfit check asks for on the cylinder plant. Same plants,
# readings, hourly batches and channel sets (the two with a scale) as
# mock_stream.jl, one more latent — the log shape factor s of the mass
# bookkeeping m = s ρ_d A L (1 + W̄ + F̄), prior LogNormal(0, 0.3), truth 1
# for the slab plant and πR²H / (A L) for the cylinder — and the misfit
# check recorded per batch.
#
# Run:  julia --project=estimator estimator/mock_stream_shape.jl data/mock/mock_2d.json
# Writes data/mock/<name>_shape_stream.json.
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

# estimator-only latents: the exposed area, the core probe's depth, the shape factor
L_med = exp(prior_phys[4].p1)
extra = [LatentSpec("A_meat", "lognormal", log(cfg.A_meat), 0.3),
         LatentSpec("core_depth", "lognormal", log(0.8 * L_med), 0.5),
         LatentSpec("shape", "lognormal", 0.0, 0.3)]
prior = vcat(prior_phys, [e.site for e in extra])
allnames = vcat(names, [e.name for e in extra])
AREA, DEPTH, SHAPE = length(prior) - 2, length(prior) - 1, length(prior)

x_true = Float64.(m["truth"]["x"])            # 15 physics, the area, the depth
is2d = occursin("2-D", String(m["plant"]))
R_CYL, H_CYL = 0.135, 0.078                   # mock_plant.jl's cylinder
s_true = is2d ? π * R_CYL^2 * H_CYL / (x_true[AREA] * x_true[4]) : 1.0
x_true = vcat(x_true, s_true)
z_true = [isbeta(p) ? log(x / (1 - x)) : log(x) for (x, p) in zip(x_true, prior)]
@printf("plant: %s  shape factor truth %.3f\n", m["plant"], s_true)

rd = m["readings"]
nz = m["noise"]
full = trues(length(t))
rng = MersenneTwister(21)
pit_ch = PitProbe(nz["pit"]["sigma"], nz["pit"]["rho"], full; data=Float64.(rd["pit"]))
core_ch = MeatProbe(nz["core"]["sigma"], nz["core"]["rho"], full; depth_latent=DEPTH,
                    data=Float64.(rd["core"]))
hum_ch = HumiditySensor(nz["humidity"]["sigma"], nz["humidity"]["rho"], full; data=Float64.(rd["humidity"]))
scale_ch = MassScale(nz["mass"]["sigma"], nz["mass"]["rho"], full; area_latent=AREA,
                     shape_latent=SHAPE, data=Float64.(rd["mass"]))
ruler_L = StaticReading(4, 0.003; data=x_true[4] + 0.003 * randn(rng))       # the same draws as mock_stream.jl
ruler_A = StaticReading(AREA, 0.01; data=x_true[AREA] + 0.01 * randn(rng))

sets = [
    ("pit + core + humidity + scale", [pit_ch, core_ch, hum_ch, scale_ch]),
    ("+ rulers (thickness, footprint)", [pit_ch, core_ch, hum_ch, scale_ch, ruler_L, ruler_A]),
]
edges = 3600.0 .* collect(1:floor(Int, t[end] / 3600 + 1e-9))

out_sets = []
for (label, chs) in sets
    println("set: ", label)
    flush(stdout)
    t0 = time()
    recs = stream_fit(chs, prior, d, edges; area_latent=AREA, verbose=true)
    batches = [Dict("t_h" => r.t / 3600, "z" => r.z, "sd" => r.sd,
                    "err_sd" => (r.z .- z_true) ./ r.sd, "F" => r.F, "iters" => r.iters,
                    "wall_s" => r.wall, "misfit" => r.misfit) for r in recs]
    last = recs[end]
    e = abs.((last.z .- z_true) ./ last.sd)
    @printf("  final |err|/sd: max %.2f (%s), median %.2f; shape %.3f ± %.3f (truth %.3f); 2F/row %.3f; wall %.0f s\n",
            maximum(e), allnames[argmax(e)], sort(e)[div(end + 1, 2)],
            exp(last.z[SHAPE]), last.sd[SHAPE], s_true, last.misfit.total.ss / last.misfit.total.n, time() - t0)
    flush(stdout)
    push!(out_sets, Dict("label" => label, "batches" => batches))
end

stem = splitext(basename(path))[1]
dst = joinpath(REPO, "data", "mock", stem * "_shape_stream.json")
open(dst, "w") do io
    JSON.print(io, Dict("plant" => m["plant"] * ", shape-factor latent", "names" => allnames,
                        "z_true" => z_true, "x_true" => x_true, "sets" => out_sets,
                        "edges_h" => edges ./ 3600))
end
println("wrote ", dst)
