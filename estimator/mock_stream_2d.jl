# The streaming estimator with the 2-D physics behind the channel registry:
# the cylinder plant read by its own engine (matched: the misfit check
# should go quiet and the latents land), and the slab plant read by the
# cylinder engine (mismatched the other way: the check should light).
# Latents: the fifteen physics (m_conv unused by the 2-D engine, so it
# keeps its prior), the radius (LogNormal, 30 %) and the core probe's
# depth; the channel sets of mock_stream.jl, the rulers now on the
# thickness and the radius.
#
# Run:  julia --project=estimator estimator/mock_stream_2d.jl data/mock/mock_2d.json
# Writes data/mock/<name>_phys2d_stream.json.
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

x_phys = Float64.(m["truth"]["x"])            # 15 physics, the area, the depth
A_true, depth_true, L_true = x_phys[16], x_phys[17], x_phys[4]
has_geom = haskey(m, "geometry")
# the radius: the plant's when it is a cylinder, else the cylinder of height 2L with the slab's exposed area
R_true = has_geom ? Float64(m["geometry"]["R"]) : (-2L_true + sqrt((2L_true)^2 + 2 * A_true / π)) / 2
nr, nz = has_geom ? (Int(m["geometry"]["nr"]), Int(m["geometry"]["nz"])) : (16, 14)
ph = Physics2D(radius_latent=16, nr=nr, nz=nz, h=30.0, n_meat=1.0, nx_line=nz)

L_med = exp(prior_phys[4].p1)
extra = [LatentSpec("R", "lognormal", log(0.10), 0.3),                  # a prior radius of 10 cm
         LatentSpec("core_depth", "lognormal", log(0.8 * L_med), 0.5)]
prior = vcat(prior_phys, [e.site for e in extra])
allnames = vcat(names, [e.name for e in extra])
RAD, DEPTH = 16, 17
x_true = vcat(x_phys[1:15], R_true, depth_true)
z_true = [isbeta(p) ? log(x / (1 - x)) : log(x) for (x, p) in zip(x_true, prior)]
@printf("plant: %s  read by the 2-D engine (%dx%d); radius truth %.4f m%s\n", m["plant"], nr, nz, R_true,
        has_geom ? "" : " (the equivalent cylinder's)")

rd = m["readings"]
nz_ = m["noise"]
full = trues(length(t))
rng = MersenneTwister(21)
pit_ch = PitProbe(nz_["pit"]["sigma"], nz_["pit"]["rho"], full; data=Float64.(rd["pit"]))
core_ch = MeatProbe(nz_["core"]["sigma"], nz_["core"]["rho"], full; depth_latent=DEPTH, data=Float64.(rd["core"]))
hum_ch = HumiditySensor(nz_["humidity"]["sigma"], nz_["humidity"]["rho"], full; data=Float64.(rd["humidity"]))
scale_ch = MassScale(nz_["mass"]["sigma"], nz_["mass"]["rho"], full; data=Float64.(rd["mass"]))
ruler_L = StaticReading(4, 0.003; data=x_true[4] + 0.003 * randn(rng))         # the same draw as mock_stream.jl
ruler_R = StaticReading(RAD, 0.003; data=R_true + 0.003 * randn(rng))          # a caliper on the radius

sets = [
    ("pit + core", [pit_ch, core_ch]),
    ("pit + core + humidity + scale", [pit_ch, core_ch, hum_ch, scale_ch]),
]
has_geom && push!(sets, ("+ rulers (thickness, radius)", [pit_ch, core_ch, hum_ch, scale_ch, ruler_L, ruler_R]))
edges = 3600.0 .* collect(1:floor(Int, t[end] / 3600 + 1e-9))

out_sets = []
for (label, chs) in sets
    println("set: ", label)
    flush(stdout)
    t0 = time()
    recs = stream_fit(chs, prior, d, edges; physics=ph, verbose=true)
    batches = [Dict("t_h" => r.t / 3600, "z" => r.z, "sd" => r.sd,
                    "err_sd" => (r.z .- z_true) ./ r.sd, "F" => r.F, "iters" => r.iters,
                    "wall_s" => r.wall, "misfit" => r.misfit) for r in recs]
    last = recs[end]
    e = abs.((last.z .- z_true) ./ last.sd)
    keep = [i for i in eachindex(e) if i != 8]                                  # m_conv: not a 2-D latent
    mt = last.misfit.total
    @printf("  final |err|/sd (m_conv aside): max %.2f (%s), median %.2f; R %.4f ± %.1f%% (truth %.4f); residual power/row %.3f, z_chi2 %.2f, z_rho1 %.2f; wall %.0f s\n",
            maximum(e[keep]), allnames[keep[argmax(e[keep])]], sort(e[keep])[div(end + 1, 2)],
            exp(last.z[RAD]), 100 * last.sd[RAD], R_true, mt.ss / mt.n, mt.z_chi2, mt.z_rho1, time() - t0)
    flush(stdout)
    push!(out_sets, Dict("label" => label, "batches" => batches))
end

stem = splitext(basename(path))[1]
dst = joinpath(REPO, "data", "mock", stem * "_phys2d_stream.json")
open(dst, "w") do io
    JSON.print(io, Dict("plant" => m["plant"] * ", read by the 2-D engine", "names" => allnames,
                        "z_true" => z_true, "x_true" => x_true, "sets" => out_sets,
                        "edges_h" => edges ./ 3600))
end
println("wrote ", dst)
