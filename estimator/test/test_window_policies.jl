# The fixed-lag window's lock mitigations and memory bounds (online.jl): lag_max/settle_tol holding nodes
# while the latents move, prior_relin/rebuild_prior! re-linearizing departed information, the slice caches
# pruned with the window, and the solver's damping bookkeeping on accepted full and doubled steps.
# Algebra on the real-cook mock at a coarse slab; these check the mechanisms, not statistical consistency.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test
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
interp(v, τ) = (k = clamp(searchsortedlast(t, τ), 1, length(t) - 1); w = (τ - t[k]) / (t[k + 1] - t[k]); (1 - w) * v[k] + w * v[k + 1])
function stream(tmax; pit_s=37.0, core_s=53.0)
    recs = MeasurementRecord[]
    for τ in pit_s:pit_s:tmax; push!(recs, MeasurementRecord(:pit, τ, interp(Float64.(rd["pit"]), τ))); end
    for τ in core_s:core_s:tmax; push!(recs, MeasurementRecord(:core, τ, interp(Float64.(rd["core"]), τ))); end
    sort!(recs; by=r -> r.t)
    return recs
end
model = SystemModel(sys, CookLog(t, u; sensors), blk; sn=StateNoise())

@testset "lag_max holds the window while the latents move; settle_tol releases it" begin
    recs = stream(1200.0)
    # never settled (settle_tol = 0): the window grows to lag_max + 1 nodes and no further
    sm_hold = online_init(model; attach=:slice, lag=1, lag_max=3, settle_tol=0.0, node_ratio=Inf, max_slice_s=120.0)
    peaks = Int[]
    for r in recs
        assimilate!(sm_hold, r)
        push!(peaks, length(sm_hold.nodes))
    end
    @test maximum(peaks) == 3 + 1
    @test length(sm_hold.nodes) == 4
    # always settled (settle_tol = Inf): plain lag-1 behaviour, never more than lag + 1 behind-plus-newest
    sm_go = online_init(model; attach=:slice, lag=1, lag_max=3, settle_tol=Inf, node_ratio=Inf, max_slice_s=120.0)
    for r in recs; assimilate!(sm_go, r); end
    @test length(sm_go.nodes) <= 2
    @test sm_go.committed_t > sm_hold.committed_t - 1e-9   # the holding window marginalizes later, never earlier
end

@testset "rebuild_prior! re-linearizes departed information where the latents are now" begin
    recs = stream(2400.0)
    run(; kw...) = (sm = online_init(model; attach=:slice, node_ratio=Inf, max_slice_s=120.0, kw...);
                    out = [assimilate!(sm, r) for r in recs]; (sm, out))
    sm_full, out_full = run(; lag=typemax(Int) ÷ 2)          # the reference: nothing marginalized
    sm_inf, out_inf = run(; lag=1, prior_relin=Inf)           # the locked window
    sm_rel, out_rel = run(; lag=1, prior_relin=0.2)           # the mitigation
    # an arrival may fail and its readings commit on a later retry; what matters is zero permanent loss
    for (nm, sm_, out_) in (("full", sm_full, out_full), ("locked", sm_inf, out_inf), ("relin", sm_rel, out_rel))
        n_un = count(r -> !r.committed, out_)
        n_un == 0 || println("  prior_relin fixture [", nm, "]: ", n_un, " of ", length(out_),
                             " arrivals uncommitted (retried); dropped ", sm_.n_dropped)
        @test sm_.n_dropped == 0 && isempty(sm_.retry)
    end
    @test count(r -> !r.committed, vcat(out_full, out_inf, out_rel)) <= 3
    @test sm_rel.n_prior_relin >= 1                           # the trigger fired on this fixture
    @test !isempty(sm_rel.hist_factors)
    @test count(f -> f isa E.LinearFactor && startswith(f.source, "marginal"), sm_rel.graph.factors) >= 1
    zf, zi, zr = out_full[end].z, out_inf[end].z, out_rel[end].z
    sdf = sqrt.(max.(LinearAlgebra.diag(out_full[end].z_cov), 1e-300))
    gap_inf = maximum(abs.(zi .- zf) ./ sdf)
    gap_rel = maximum(abs.(zr .- zf) ./ sdf)
    # the rebuilt prior must not sit farther from the full-history answer than the locked one
    @test gap_rel <= gap_inf + 0.05
    println("  prior_relin: worst gap to full history ", round(gap_rel; digits=3), " sd (locked ",
            round(gap_inf; digits=3), " sd), ", sm_rel.n_prior_relin, " rebuilds")
end

@testset "the slice caches and the duplicate memory leave with the window" begin
    recs = stream(1800.0; pit_s=37.0, core_s=53.0)
    sm = online_init(model; attach=:slice, lag=1, node_ratio=Inf, max_slice_s=60.0)
    counts = Int[]
    for r in recs
        assimilate!(sm, r)
        push!(counts, length(sm.slice_rd))
    end
    t_old = sm.nodes[1][1]
    @test all(k[2] >= t_old - 1e-9 for k in keys(sm.slice_G))
    @test all(k[2] >= t_old - 1e-9 for k in keys(sm.slice_rd))
    @test all(k[2] >= t_old - 1e-9 for k in keys(sm.seen))
    # bounded by the window: the lag-1 window spans at most two 60 s slices plus the open one
    window_s = 3 * 60.0 + 60.0
    cap = ceil(Int, window_s / 37.0) + ceil(Int, window_s / 53.0) + 4
    @test maximum(counts[20:end]) <= cap                     # after the window fills, the caches stop growing
    @test length(sm.slice_G) <= cap && length(sm.seen) <= cap
end

@testset "a slice sensitivity refreshes when the belief leaves its reference point" begin
    for (refresh, expect_move) in ((true, true), (false, false))
        sm = online_init(model; attach=:slice, node_ratio=Inf, max_slice_s=Inf, slice_g_refresh=refresh)
        for r in stream(300.0); assimilate!(sm, r); end
        # a core reading: its sensitivity is the depth interpolation's weights, which move with the
        # latents (a pit reading's is a constant unit selector and cannot show the retake)
        ka = first(k for (k, f) in sm.slices if any(r -> r.sensor === :core, get(sm.slice_recs, k, MeasurementRecord[])))
        key = first((r.sensor, r.t) for r in sm.slice_recs[ka] if r.sensor === :core)
        before = sm.slice_G[key]
        v = E.variable_value(sm.graph, sm.key_z)
        v .+= 50 * sm.relin_tol                                # far outside the relinearization tolerance
        E._rebuild_slice!(sm, ka)
        after = sm.slice_G[key]
        if expect_move
            @test after.z != before.z                          # the reference followed the belief
            @test after.g != before.g                          # and the sensitivity was retaken there
        else
            @test after.z == before.z && after.g == before.g   # frozen at first sight (the baseline arm)
        end
        # a move inside the tolerance never retakes it
        v .+= 0.1 * sm.relin_tol
        E._rebuild_slice!(sm, ka)
        @test sm.slice_G[key].z == after.z && sm.slice_G[key].g == after.g
    end
end

@testset "accepted full and doubled steps relax damping; stalled trials escalate it" begin
    recs = stream(600.0)
    sm = online_init(model; attach=:slice, node_ratio=Inf, max_slice_s=Inf)
    for r in recs; assimilate!(sm, r); end
    # push the window off its optimum so the next solve takes several accepted iterations
    v = E.variable_value(sm.graph, sm.key_z)
    v .+= 0.15 .* sqrt.([E.z_moments(p)[2] for p in sm.model.blk.sites])
    rc = E.gauss_newton!(sm.graph; iters=20, tol=1e-6, ftol=1e-9,
                         retract_dynamics=true, geodesic_search=true, expand=E.ONLINE_EXPAND[])
    @test any(h -> get(h, :accepted, false), rc.hist)
    # a row's `damping` is recorded AFTER the bookkeeping. Rejected trials between acceptances escalate
    # damping, so the invariant is against the lambda in force IMMEDIATELY before the acceptance (the
    # preceding history row): an accepted full step OR doubling whose gain beat 0.75 must leave damping
    # at or below it (the regression of 2026-10-05: an accepted doubling, alpha > 1, fell between the
    # escalation and relaxation branches and kept it).
    for i in 2:length(rc.hist)
        h = rc.hist[i]
        if get(h, :accepted, false) && isfinite(get(h, :gain, NaN)) && h.alpha >= 1.0 && h.gain > 0.75
            @test h.damping <= rc.hist[i - 1].damping + 1e-12
        end
    end
    @test rc.converged || !isempty(rc.hist)
end
