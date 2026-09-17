# The misfit check scored post hoc on a stored streaming run: for every
# batch of every channel set, the physics and its Jacobian at the stored
# posterior mode, the graph's covariance, and the per-channel statistics.
#
# Run:  julia --project=estimator estimator/scripts/misfit_stream.jl data/mock/mock_2d.json
# Reads data/mock/<name>_stream.json, writes data/mock/<name>_misfit.json.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf

REPO = normpath(joinpath(@__DIR__, "..", ".."))
path = length(ARGS) >= 1 ? ARGS[1] : joinpath(REPO, "data", "mock", "mock_2d.json")
stem = splitext(basename(path))[1]
m = JSON.parsefile(path)
S = JSON.parsefile(joinpath(REPO, "data", "mock", stem * "_stream.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
t = Float64.(m["t"])
u = Float64.(m["u"])
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
names = String.(gate["names"])
d = estimator_setup(t, u, pit, cfg, prior_phys, names)

# the same latents and channels as mock_stream.jl (the rulers' draws included)
L_med = exp(prior_phys[4].p1)
extra = [LatentSpec("A_meat", "lognormal", log(cfg.A_meat), 0.3),
         LatentSpec("core_depth", "lognormal", log(0.8 * L_med), 0.5)]
prior = vcat(prior_phys, [e.site for e in extra])
AREA, DEPTH = length(prior) - 1, length(prior)
x_true = Float64.(m["truth"]["x"])
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
sets = Dict(
    "pit + core" => [pit_ch, core_ch],
    "pit + core + humidity + scale" => [pit_ch, core_ch, hum_ch, scale_ch],
    "+ rulers (thickness, footprint)" => [pit_ch, core_ch, hum_ch, scale_ch, ruler_L, ruler_A],
)

_fmt(x) = x === nothing ? "-" : @sprintf("%.1f", x)     # a one-row channel has no autocorrelation
println("plant: ", m["plant"])
out_sets = []
for st in S["sets"]
    label = String(st["label"])
    chs = sets[label]
    println("set: ", label)
    flush(stdout)
    batches = []
    for b in st["batches"]
        te = 3600.0 * Float64(b["t_h"])
        keep = t .<= te + 1e-9
        chs_b = [restrict(ch, keep) for ch in chs]
        z = Float64.(b["z"])
        t0 = time()
        Sx, J, lay = physics_outputs(z, d; area_latent=AREA)
        a, B, w = assemble(chs_b, z, Sx, J, prior, lay)
        _, V = graph_posterior_affine(z, a, B, w, prior)
        mis = misfit(chs_b, a, B, w, V)
        tot = mis.total
        @printf("  t = %5.1f h  rows %4d  2F/row %.3f  z_chi2 %6.2f  z_rho1 %6.2f   per channel:",
                te / 3600, tot.n, tot.ss / tot.n, tot.z_chi2, tot.z_rho1)
        for c in mis.channels
            @printf("  %s %s/%s", c.name, _fmt(c.z_chi2), _fmt(c.z_rho1))
        end
        @printf("   (%.0f s)\n", time() - t0)
        flush(stdout)
        push!(batches, Dict("t_h" => b["t_h"], "channels" => [Dict(pairs(c)) for c in mis.channels],
                            "total" => Dict(pairs(tot))))
    end
    push!(out_sets, Dict("label" => label, "batches" => batches))
end

dst = joinpath(REPO, "data", "mock", stem * "_misfit.json")
open(dst, "w") do io
    JSON.print(io, Dict("plant" => m["plant"], "sets" => out_sets))
end
println("wrote ", dst)
