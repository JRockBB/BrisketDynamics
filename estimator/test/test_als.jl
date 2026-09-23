# The innovation autocovariances' accumulator (als.jl), which the bench's identification of the state's rates
# reads: the sigma prediction's equivalent transition reproduces its covariance, the per-group sensitivity of
# the process covariance sums to the mapped covariance, the accumulator's analytic autocovariances agree with a
# simulation of the same error dynamics so that the fit recovers a planted change of a rate, the filter's belief
# is the same with or without the accumulator, and the summary is what the driver pools.
#   julia --project=estimator estimator/test/test_als.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test, Statistics
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
blk = latent_block(joinpath(REPO, "data", "calibration", "prior_next_cook.json"), cfg)
sites, names = blk.sites, blk.names
AREA, DEPTH = blk.area, blk.depth
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
nc = RealCookNoise()
np = realcook_noise_priors(joinpath(REPO, "data", "calibration", "noise_prior_real.json"), dt)
rd = m["readings"]; mk = m["masks"]
readings = (pit=Float64.(rd["pit"]), core=Float64.(rd["core"]), humidity=Float64.(rd["humidity"]), mass=Float64.(rd["mass"]))
masks = (pit=Bool.(mk["pit"]), core=Bool.(mk["core"]), humidity=Bool.(mk["humidity"]), mass=Bool.(mk["mass"]))
channels() = realcook_channels(readings, masks, np; cap=nc.hum_cap, depth_latent=DEPTH)
noise = [np[i] for i in 1:4]
q = zeros(length(sites))
T_END = 1800.0

@testset "the sigma prediction's transition and the process covariance's sensitivity" begin
    st = filter_init(channels(), t, u, pit, cfg, sites, names, events; noise, area_latent=AREA, q, n_sigma=64)
    filter_update!(st, 2)                                                  # one update, so the belief's latents and state are coupled
    lay = st.lay; zr = 1:lay.n_z; ξr = (lay.n_z + 1):lay.n; xr = state_range(lay)
    tw = t[2:3]; uw = u[2:3]
    dtr = estimator_setup(tw, uw, pit, cfg, sites[1:N_PHYS], names[1:N_PHYS])
    ph = PhysicsState(AREA, events, lay, false)
    L = E._tangent(st.m, dtr, ph)
    m2, P2, F = E._sigma_prediction(lay, st.sigma, st.m, st.P, L.J_end, dtr, ph)
    FPF = F * st.P * F'
    @test size(F) == (lay.n, lay.n) && F[zr, zr] == I && all(F[zr, ξr] .== 0)
    @test maximum(abs.(P2[zr, zr] .- FPF[zr, zr])) <= 1e-12 * maximum(abs.(P2[zr, zr]))
    @test maximum(abs.(P2[ξr, zr] .- FPF[ξr, zr])) <= 1e-8 * maximum(abs.(P2[ξr, zr]))           # the cross block is the transition's exactly
    Ω = Symmetric(P2[ξr, ξr] .- FPF[ξr, ξr])                                                       # the rest is the regression's residual, a covariance
    @test minimum(eigvals(Ω)) >= -1e-8 * maximum(abs.(P2[ξr, ξr]))
    # the per-group sensitivity sums to the interval's mapped process covariance wherever the coordinate cap is not holding
    Δh = (tw[end] - tw[1]) / 3600
    Qm = E._state_process_cov(st, tw, L.x_end)
    dQ = E._state_process_sens(st, tw, L.x_end)
    θ = [getfield(st.sn, Symbol(g))^2 for g in STATE_GROUPS]
    Qsum = sum(θ[g] .* dQ[g] for g in eachindex(θ))[xr, xr]
    _, scale = E._coord_scales(state_noise_cov(lay, st.sn, E.state_rates(st.sn, lay) .* sqrt(Δh)), L.x_end, lay)
    free = findall(scale .== 1)
    @test length(dQ) == length(STATE_GROUPS) && all(size(D) == (lay.n, lay.n) for D in dQ)
    @test !isempty(free) && maximum(abs.(Qsum[free, free] .- Qm[free, free])) <= 1e-10 * maximum(abs.(Qm))
    @test all(all(D[zr, :] .== 0) && all(D[:, zr] .== 0) for D in dQ)                             # the rates are the state's alone
end

@testset "the accumulator against a simulation of its error dynamics: a planted change of a rate is recovered" begin
    st = filter_init(channels(), t, u, pit, cfg, sites, names, events; noise, area_latent=AREA, q, n_sigma=64)
    st.als = ALSAccumulator(st.sn, st.lay; lag_s=3600.0, bin_s=dt, keep=true)
    recs = filter_advance!(st; t_to=T_END, diagnostics=true)
    acc = st.als
    st2 = filter_init(channels(), t, u, pit, cfg, sites, names, events; noise, area_latent=AREA, q, n_sigma=64)
    recs2 = filter_advance!(st2; t_to=T_END, diagnostics=true)
    @test acc.n_updates == length(recs) == length(recs2) && length(acc.pieces) == length(recs)
    @test isapprox(st.m, st2.m; rtol=1e-10, atol=1e-12) && isapprox(st.P, st2.P; rtol=1e-10, atol=1e-12)   # the accumulator is a reading, not a change
    @test acc.n_rows > 0 && issymmetric(acc.G) && minimum(eigvals(Symmetric(acc.G))) >= -1e-12 * maximum(abs.(acc.G))
    @test all(pc.S ≈ pc.S' && all(abs.(diag(pc.S) .- 1) .< 1e-9) for pc in acc.pieces if size(pc.S, 1) > 0)   # standardized rows
    # the simulation: the linear error dynamics the accumulator's model describes, under the run's own transitions,
    # rows and gains, driven by a planted change of the pit's variance rate (its rate by √3) with R replicates at
    # once; the base innovations white with the predicted covariance; the fit on the accumulated statistics
    n_g = length(acc.groups); θ = acc.theta
    g_pit = findfirst(==("T_pit"), acc.groups)
    δθ = zeros(n_g); δθ[g_pit] = 2.0 * θ[g_pit]
    R = 400
    rng = MersenneTwister(11)
    sim = ALSAccumulator(acc.groups, θ, acc.n, acc.n_ch; lag_s=acc.lag_s, bin_s=acc.bin_s)
    ε = zeros(acc.n, R)
    for pc in acc.pieces
        Lw = E._psd_factor(sum(δθ[g] .* pc.dQ[g] for g in 1:n_g))
        ε = pc.F * ε + Lw * randn(rng, size(Lw, 2), R)
        p = size(pc.H, 1)
        if p > 0
            e = pc.H * ε + cholesky(Symmetric(pc.S)).L * randn(rng, p, R)
            als_update!(sim, pc.F, pc.dQ, pc.H, pc.S, pc.K, e, pc.rowch, pc.t)
            ε = ε - pc.K * (pc.H * ε)
        else
            als_update!(sim, pc.F, pc.dQ, pc.H, pc.S, pc.K, zeros(0, R), Int[], pc.t)
        end
    end
    @test sim.n_rows == acc.n_rows * R
    # the groups the simulated innovations pin (their own column's relative standard error under a half), the
    # least squares on them, and the planted change recovered within three standard errors, the rest at zero
    σ2 = sim.rss / (sim.n_rows - 1)
    rel = [sim.G[g, g] > 0 && θ[g] > 0 ? sqrt(σ2 / sim.G[g, g]) / θ[g] : Inf for g in 1:n_g]
    a = findall(<(0.5), rel)
    @test g_pit in a
    Ga = Symmetric(sim.G[a, a]); ha = sim.h[a]
    d = Ga \ ha
    rss1 = sim.rss - 2 * dot(ha, d) + dot(d, Ga * d)
    se = sqrt.(diag(inv(Ga)) .* (rss1 / (sim.n_rows - length(a))))
    println("  identified groups: ", join(["$(acc.groups[g]) (rel se $(round(rel[g]; digits=2)))" for g in a], ", "))
    println("  the pit's variance rate: planted +", δθ[g_pit], ", fitted +", round(d[findfirst(==(g_pit), a)]; digits=3), " ± ", round(se[findfirst(==(g_pit), a)]; digits=3))
    @test all(abs(d[j] - δθ[a[j]]) <= 3 * se[j] for j in eachindex(a))
    @test rss1 < sim.rss
    # the summary: what the driver pools, through JSON, its pairs binned by the lag in grid steps
    s = JSON.parse(JSON.json(als_summary(acc)))
    @test s["groups"] == collect(STATE_GROUPS) && length(s["theta"]) == n_g && s["n_rows"] == acc.n_rows && length(s["G"]) == n_g
    @test all(0 <= p["bin"] <= round(Int, acc.lag_s / dt) && 1 <= p["a"] <= 4 && 1 <= p["b"] <= 4 && p["n"] >= 1 && length(p["sum_b"]) == n_g && p["sum_product2"] >= 0 for p in s["pairs"])
    @test sum(p["n"] for p in s["pairs"]) == acc.n_rows
    # the blocks: every row in one of them, their normal equations summing to the whole, the squares in the blocks at one update
    @test sum(b["n"] for b in s["blocks"]) == acc.n_rows && all(b["a"] == b["b"] for b in s["blocks"] if b["same"] && b["n"] > 0 && b["a"] == b["b"])
    @test isapprox(sum(hcat(b["G"]...)' for b in s["blocks"]), acc.G; rtol=1e-10) && isapprox(sum(b["h"] for b in s["blocks"]), acc.h; rtol=1e-10, atol=1e-12)
    @test isapprox(sum(b["rss"] for b in s["blocks"]), acc.rss; rtol=1e-10)
end
println("test_als.jl: done")
