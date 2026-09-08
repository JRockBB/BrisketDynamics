# The graph engine against the outer-loop engine on the streaming path:
# the first three hourly batches of the matched mock's humidity-plus-scale
# set, both engines from the same start, must agree on the posterior
# (means within a few hundredths of a width, widths within a percent),
# and the graph engine must match the stored outer-loop record.
#
# Run:  julia --project=estimator estimator/test_engine.jl
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Printf, Test

REPO = normpath(joinpath(@__DIR__, ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json"))
S = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d_stream.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
d = estimator_setup(t, u, pit, cfg, prior_phys, String.(gate["names"]))
L_med = exp(prior_phys[4].p1)
prior = vcat(prior_phys, [PriorSite("lognormal", log(cfg.A_meat), 0.3), PriorSite("lognormal", log(0.8 * L_med), 0.5)])
AREA, DEPTH = 16, 17
rd = m["readings"]; nz = m["noise"]; full = trues(length(t))
chs = [PitProbe(nz["pit"]["sigma"], nz["pit"]["rho"], full; data=Float64.(rd["pit"])),
       MeatProbe(nz["core"]["sigma"], nz["core"]["rho"], full; depth_latent=DEPTH, data=Float64.(rd["core"])),
       HumiditySensor(nz["humidity"]["sigma"], nz["humidity"]["rho"], full; data=Float64.(rd["humidity"])),
       MassScale(nz["mass"]["sigma"], nz["mass"]["rho"], full; data=Float64.(rd["mass"]))]
edges = 3600.0 .* [1.0, 2.0, 3.0]
stored = first(s for s in S["sets"] if s["label"] == "pit + core + humidity + scale")["batches"]

@testset "graph engine against the outer loop on the stream" begin
    t0 = time()
    rg = stream_fit(chs, prior, d, edges; area_latent=AREA, engine=:graph, verbose=false)
    tg = time() - t0
    t0 = time()
    ro = stream_fit(chs, prior, d, edges; area_latent=AREA, engine=:outer, verbose=false)
    to = time() - t0
    for (b, (g, o)) in enumerate(zip(rg, ro))
        sd = o.sd
        dz = maximum(abs.((g.z .- o.z) ./ sd))
        dsd = maximum(abs.(g.sd ./ o.sd .- 1))
        st = stored[b]
        dz_rec = maximum(abs.((g.z .- Float64.(st["z"])) ./ Float64.(st["sd"])))
        @printf("  batch %d: |Δz|/sd graph vs outer %.4f, widths %.4f; graph vs the stored record %.4f; F %.4f vs %.4f (stored %.4f); graph %d it, outer %d it\n",
                b, dz, dsd, dz_rec, g.F, o.F, Float64(st["F"]), g.iters, o.iters)
        @test dz < 0.05
        @test dsd < 0.01
        @test dz_rec < 0.1
        @test g.F <= o.F + 1e-3
    end
    @printf("  wall: graph %.0f s, outer %.0f s\n", tg, to)
end
println("test_engine.jl: done")
