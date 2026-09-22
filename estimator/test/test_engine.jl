# The graph engine against the stored streaming record: the first three
# hourly batches of the matched mock's humidity-plus-scale set must land
# on the record's posterior (means within a tenth of a width). The record
# was first made by the retired outer-loop engine, whose fixed point is the
# same; since 2026-09-22 it is the graph engine's own run on the regenerated
# mock (mock_stream.jl), so this is the regression that the solve, the
# channels and the mock have not moved.
#
# Run:  julia --project=estimator estimator/test/test_engine.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Printf, Test

REPO = normpath(joinpath(@__DIR__, "..", ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json"))
S = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d_stream.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
blk = latent_block(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"), cfg)
prior_phys, prior = blk.prior_phys, blk.sites
d = estimator_setup(t, u, pit, cfg, prior_phys, blk.phys_names)
AREA, DEPTH = blk.area, blk.depth
rd = m["readings"]; nz = m["noise"]; full = trues(length(t))
chs = [PitProbe(nz["pit"]["sigma"], nz["pit"]["rho"], full; data=Float64.(rd["pit"])),
       MeatProbe(nz["core"]["sigma"], nz["core"]["rho"], full; depth_latent=DEPTH, data=Float64.(rd["core"])),
       HumiditySensor(nz["humidity"]["sigma"], nz["humidity"]["rho"], full; data=Float64.(rd["humidity"])),
       MassScale(nz["mass"]["sigma"], nz["mass"]["rho"], full; data=Float64.(rd["mass"]))]
edges = 3600.0 .* [1.0, 2.0, 3.0]
stored = first(s for s in S["sets"] if s["label"] == "pit + core + humidity + scale")["batches"]

@testset "graph engine against the stored streaming record" begin
    t0 = time()
    rg = stream_fit(chs, prior, d, edges; area_latent=AREA, verbose=false)
    tg = time() - t0
    for (b, g) in enumerate(rg)
        st = stored[b]
        dz_rec = maximum(abs.((g.z .- Float64.(st["z"])) ./ Float64.(st["sd"])))
        dsd_rec = maximum(abs.(g.sd ./ Float64.(st["sd"]) .- 1))
        @printf("  batch %d: |dz|/sd graph vs the stored record %.4f, widths %.4f; F %.4f (stored %.4f); %d it\n",
                b, dz_rec, dsd_rec, g.F, Float64(st["F"]), g.iters)
        @test dz_rec < 0.1
        @test dsd_rec < 0.02
        @test g.F <= Float64(st["F"]) + 1e-2
    end
    @printf("  wall: graph %.0f s\n", tg)
end
println("test_engine.jl: done")
