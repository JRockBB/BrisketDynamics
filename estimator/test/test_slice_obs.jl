# Readings through the physics from their slice's node (slice_obs.jl, online.jl attach=:slice): exactness checks on
# the real-cook mock at a coarse slab (these check the algebra, not consistency).
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test, ForwardDiff
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; nx=6, T0=Float64(meta["tcore0_K"]))
blk_old = latent_block(joinpath(REPO, "data", "calibration", "prior_next_cook.json"), cfg)
sys = CookSystem(Slab1D(:brisket; cfg); pit)
blk = system_latents(sys, blk_old.prior_phys, blk_old.phys_names; probes=[:brisket])
nc = RealCookNoise()
sensors = [SensorSpec(:pit, PitTemperature(), NoiseModel(white_sd=0.1, ar_sd=nc.pit_sd, tau=nc.pit_tau)),
           SensorSpec(:core, TemperatureAt(:brisket; depth_latent=:core_depth_fraction), NoiseModel(white_sd=0.1, ar_sd=nc.core_sd, tau=nc.core_tau))]
rd = m["readings"]
# readings off the grid, each sensor on its own clock: the pit every 37 s, the core every 53 s, over the first 20 min
function stream(tmax)
    recs = MeasurementRecord[]
    interp(v, τ) = (k = clamp(searchsortedlast(t, τ), 1, length(t) - 1); w = (τ - t[k]) / (t[k+1] - t[k]); (1 - w) * v[k] + w * v[k+1])
    for τ in 37.0:37.0:tmax; push!(recs, MeasurementRecord(:pit, τ, interp(Float64.(rd["pit"]), τ))); end
    for τ in 53.0:53.0:tmax; push!(recs, MeasurementRecord(:core, τ, interp(Float64.(rd["core"]), τ))); end
    sort!(recs; by=r -> r.t)
    return recs
end
model = SystemModel(sys, CookLog(t, u; sensors), blk; sn=StateNoise())
run(recs; kw...) = (sm = online_init(model; lag=typemax(Int), kw...); out = [assimilate!(sm, [r]) for r in recs]; (sm, out))

@testset "an optional slice-age cap bounds factors even at zero process variance" begin
    recs = stream(600.0)
    sm = online_init(model; attach=:slice, node_ratio=Inf, max_slice_s=180.0)
    chunks = E._slice_chunks(sm, recs)
    @test length(chunks) > 1
    E._add_slice_readings!(sm, recs)
    @test first.(sm.nodes)[2:end] == [first(ch).t for ch in chunks[2:end]]
    @test maximum(diff(first.(sm.nodes))) <= 180 + 37
    @test_throws ArgumentError online_init(model; max_slice_s=0.0)
end

@testset "a slice factor's Jacobian is its predictions' derivative" begin
    recs = stream(600.0)
    sm, out = run(recs; attach=:slice)
    @test all(r -> r.committed, out)
    @test all(r -> length(r.innovation_whitened) == r.dof &&
        isapprox(sum(abs2(w.value) for w in r.innovation_whitened), r.nis; rtol=1e-10), out)
    @test length(sm.nodes) < length(recs)                            # readings shared nodes
    f = first(values(sm.slices))
    vals = [copy(variable_value(sm.graph, k)) for k in vars(f)]
    μ, Jμ = E._slice_predict(f, vals; jac=true)
    # the reference: forward mode straight through the whole prediction (the solve included), on a throwaway copy whose
    # cache the Duals may fill; the assembled chain rule must match it to rounding
    lens = length.(vals); offs = cumsum(vcat(0, lens))
    unflat(w) = [w[(offs[i] + 1):offs[i + 1]] for i in eachindex(lens)]
    fr = deepcopy(f); fr.cache = nothing
    Jref = ForwardDiff.jacobian(w -> first(E._slice_predict(fr, unflat(w))), reduce(vcat, vals))
    Jmine = reduce(hcat, Jμ)
    @test maximum(abs.(Jref .- Jmine)) <= 1e-6 * max(1.0, maximum(abs.(Jref)))
    @test isapprox(first(E._slice_predict(fr, vals)), μ; rtol=1e-10)
    nj = f.n_jacobians
    trial = deepcopy(vals); trial[1] .+= 1e-6
    residual(f, trial)
    E._ensure_current!(f, sm.graph)
    μrestored, Jrestored = E._slice_predict(f, vals; jac=true)
    @test f.n_jacobians == nj
    @test μrestored ≈ μ && Jrestored == Jμ
end

@testset "with a node at every reading the slice path is the node path" begin
    recs = stream(420.0)
    sn_, on_ = run(recs; attach=:node, tol=1e-7, ftol=1e-9)
    ss_, os_ = run(recs; attach=:slice, node_ratio=0.0, tol=1e-7, ftol=1e-9)
    @test length(ss_.nodes) == length(sn_.nodes)
    a, b = on_[end], os_[end]
    @test maximum(abs.(a.z .- b.z) ./ sqrt.(diag(a.z_cov))) < 1e-3
    @test maximum(abs.(sqrt.(diag(a.z_cov)) ./ sqrt.(diag(b.z_cov)) .- 1)) < 1e-3
    @test abs(a.T_pit[1] - b.T_pit[1]) < 1e-3 && abs(a.pieces[:brisket].T_centre[1] - b.pieces[:brisket].T_centre[1]) < 1e-3
    # the per-arrival consistency statistics agree too
    @test maximum(abs(x.nis - y.nis) for (x, y) in zip(on_, os_) if isfinite(x.nis)) < 1e-3 * max(1.0, maximum(x.nis for x in on_ if isfinite(x.nis)))
end

@testset "the arrival's innovation is conditioned on the slice's earlier readings" begin
    recs = stream(400.0)
    sm, _ = run(recs[1:end-1]; attach=:slice)
    r = recs[end]
    E._add_slice_readings!(sm, [r])
    ka = sm.nodes[findlast(x -> x[1] <= r.t + 1e-9, sm.nodes)][2]
    pc = E._predicted_consistency_slice(sm, [(ka, [r])])
    # brute force: the full factor's noise and rows, the earlier readings' residual conditioned out
    f = sm.slices[ka]
    vals = [variable_value(sm.graph, k) for k in vars(f)]
    μ, _ = E._slice_predict(f, vals)
    v = [x.value for x in f.readings]
    N = findall(x -> x.t == r.t && x.spec.id === r.sensor, f.readings); O = setdiff(1:length(v), N)
    ν = (v[N] .- μ[N]) .- f.Σ[N, O] * (f.Σ[O, O] \ (v[O] .- μ[O]))
    @test pc.dof == 1 && isapprox(pc.nu, ν; rtol=1e-10, atol=1e-10)
    @test isfinite(pc.nis) && pc.nis >= 0
end
