# The smoother (smoother.jl): the stacked layout, the random-walk prior, the segmented physics against the
# 1-D physics, and the fit with no moving site against the batch fit.
#   julia --project=estimator estimator/test/test_smoother.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test, ForwardDiff
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "prior_next_cook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
names = String.(gate["names"])
L_med = exp(prior_phys[4].p1)
extra = [LatentSpec("A_meat", "lognormal", log(cfg.A_meat), 0.3), LatentSpec("core_depth", "lognormal", log(0.8 * L_med), 0.5)]
sites = vcat(prior_phys, [e.site for e in extra]); allnames = vcat(names, [e.name for e in extra])
AREA, DEPTH = length(sites) - 1, length(sites)
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
keep6 = t .<= 6 * 3600 + 1e-9
t6, u6 = t[keep6], u[keep6]
d6 = estimator_setup(t6, u6, pit, cfg, prior_phys, names)
iu = findfirst(==("u_leak"), allnames); ik = findfirst(==("k_e"), allnames); ir = findfirst(==("r_max"), allnames)
z0 = vcat(d6.z0, [log(cfg.A_meat), log(0.8 * L_med)])
i2, i4 = findlast(t6 .<= 2 * 3600), findlast(t6 .<= 4 * 3600)

@testset "the stacked layout" begin
    lay = DriftLayout(17, [(iu, [i2, i4]), (ir, [i4])])
    @test lay.n == 20 && lay.offsets == [17, 19] && segment_edges(lay) == [i2, i4]
    @test index_at(lay, 1, 1) == iu && index_at(lay, 1, i2) == 18 && index_at(lay, 1, i4) == 19 && index_at(lay, 1, i4 + 5) == 19
    @test index_at(lay, 2, i4 - 1) == ir && index_at(lay, 2, i4) == 20
    ci = current_index(lay)
    @test ci[iu] == 19 && ci[ir] == 20 && ci[ik] == ik && length(ci) == 17
    z = vcat(z0, [0.1, 0.2, -0.3])
    @test block_at(z, lay, 1)[iu] == z0[iu] && block_at(z, lay, i2)[iu] == 0.1 && block_at(z, lay, i4)[iu] == 0.2 && block_at(z, lay, i4)[ir] == -0.3
end

@testset "the random-walk prior: value, gradient, Hessian" begin
    lay = DriftLayout(17, [(iu, [i2, i4]), (ir, [i4])])
    p = DriftPrior(sites, lay, [[0.01, 0.02], [0.05]])
    @test length(p) == 20 && p[18].family == sites[iu].family && p[20].family == sites[ir].family
    rng = MersenneTwister(3)
    z = vcat(z0, [z0[iu] + 0.1, z0[iu] + 0.25, z0[ir] - 0.2]) .+ 0.05 .* randn(rng, 20)
    f, g, H = prior_terms(z, p)
    fb, gb, hb = prior_terms(z[1:17], sites)
    @test f ≈ fb + 0.5 * (z[18] - z[iu])^2 / 0.01 + 0.5 * (z[19] - z[18])^2 / 0.02 + 0.5 * (z[20] - z[ir])^2 / 0.05
    ε = 1e-6
    gfd = [(prior_terms(z .+ ε .* (1:20 .== i), p)[1] - prior_terms(z .- ε .* (1:20 .== i), p)[1]) / 2ε for i in 1:20]
    @test maximum(abs.(gfd .- g)) < 1e-5
    Hfd = hcat([(prior_terms(z .+ ε .* (1:20 .== i), p)[2] .- prior_terms(z .- ε .* (1:20 .== i), p)[2]) ./ 2ε for i in 1:20]...)
    @test maximum(abs.(Hfd .- H)) < 1e-3 && H ≈ H'
    @test H[18, 17 + 2] ≈ -1 / 0.02 && H[18, iu] ≈ -1 / 0.01
end

@testset "the segmented physics reads the stacked block" begin
    ph1 = Physics1D(area_latent=AREA, events=events)
    S1, J1, lay1 = physics_outputs(z0, d6, ph1)
    # no moving site: the same outputs and Jacobian as the 1-D physics
    ph0 = PhysicsDrift(AREA, events, DriftLayout(17, []))
    S0, J0, lay0 = physics_outputs(z0, d6, ph0)
    # per output block, at the instruments' scale (the stack holds kelvin, pascals and kilograms side by side)
    nt, nx = lay1.nt, lay1.nx
    blocks = [(lay1.oTpit, nt, 0.05), (lay1.oT, nx * nt, 0.05), (lay1.oW, nx * nt, 1e-3), (lay1.opv, nt, 5.0), (lay1.oMass, nt, 1e-3)]
    close(Sa, Sb) = all(maximum(abs.(Sa[(o + 1):(o + len)] .- Sb[(o + 1):(o + len)])) < tol for (o, len, tol) in blocks)
    @test lay0.nt == lay1.nt && close(S0, S1)
    # the Jacobians agree block by block, relative to each block's own scale
    jclose(Ja, Jb) = all((r = (o + 1):(o + len); maximum(abs.(Ja[r, :] .- Jb[r, :])) < 0.02 * maximum(abs.(Jb[r, :])) + tol) for (o, len, tol) in blocks)
    @test jclose(J0, J1)
    # a moving leak: the copies act only after their start
    layd = DriftLayout(17, [(iu, [i2, i4])])
    phd = PhysicsDrift(AREA, events, layd)
    zc = vcat(z0, [z0[iu], z0[iu]])
    Sc, _ = physics_stack(zc, d6, phd)
    @test close(Sc, S1)                                        # equal copies: the constant cook, in segments
    zd = vcat(z0, [z0[iu] + 0.3, z0[iu] + 0.3])
    Sd, layl = physics_stack(zd, d6, phd)
    pit_c, pit_d = Sc[(layl.oTpit + 1):(layl.oTpit + layl.nt)], Sd[(layl.oTpit + 1):(layl.oTpit + layl.nt)]
    @test maximum(abs.(pit_d[1:(i2 - 1)] .- pit_c[1:(i2 - 1)])) < 0.05     # untouched before the first start
    @test maximum(abs.(pit_d[(i2 + 10):end] .- pit_c[(i2 + 10):end])) > 0.5   # a larger leak shows after it
    Sd2, Jd2, _ = physics_outputs(zd, d6, phd)
    @test size(Jd2) == (length(Sd2), 19) && all(isfinite, Jd2)
    @test maximum(abs.(Jd2[1:(layl.oTpit + i2 - 1), 18:19])) < 1e-6     # no sensitivity to a copy before it starts
end

@testset "the drifting plant and the jump at the add" begin
    idx = Dict(n => i for (i, n) in enumerate(allnames))
    dd = DriftDraw(MersenneTwister(5), idx)
    @test length(dd.inc[iu]) == 30 && haskey(dd.jump, ir)
    t_add = events.t_add
    zb = truth_at(z0, dd, 2 * 3600.0, t_add)                    # before 2 h: one hour's increment
    zf = truth_at(z0, dd, 2 * 3600.0, t_add; from=true)         # from 2 h on: two
    @test zb[iu] ≈ z0[iu] + dd.inc[iu][1] && zf[iu] ≈ z0[iu] + sum(dd.inc[iu][1:2])
    @test truth_at(z0, dd, 2.5 * 3600, t_add)[ik] ≈ z0[ik] + sum(dd.inc[ik][1:2])
    @test truth_at(z0, dd, t_add, t_add)[ir] ≈ z0[ir] && truth_at(z0, dd, t_add, t_add; from=true)[ir] ≈ z0[ir] + dd.jump[ir]
    @test truth_at(z0, dd, 0.0, t_add) == z0
    edges, blocks = drift_segments(t, z0, dd, t_add)
    @test length(blocks) == length(edges) + 1 && all(1 .< edges .< length(t)) && issorted(edges) && allunique(edges)
    ia = findlast(t .<= t_add + 1e-9)
    @test ia in edges
    ka = findfirst(==(ia), edges)
    @test blocks[ka + 1][ir] ≈ z0[ir] + dd.jump[ir] && blocks[ka][ir] ≈ z0[ir]
    # the plant solved on the walk: finite, equal to the constant cook before the first step, different after it
    o_c = monolith_solve(from_z(z0[1:N_PHYS], prior_phys)[1:8], from_z(z0[1:N_PHYS], prior_phys)[9:15], u6, t6, t6; pit, cfg, A_meat=exp(z0[AREA]), events...)
    o_d = solve_drifting(z0, dd, sites, u6, t6; pit, cfg, A_meat=exp(z0[AREA]), events...)
    i1 = findlast(t6 .<= 3600)
    @test all(isfinite, o_d.Tpit) && length(o_d.t) == length(t6)
    @test maximum(abs.(o_d.Tpit[1:i1] .- o_c.Tpit[1:i1])) < 0.05 && maximum(abs.(o_d.Tpit .- o_c.Tpit)) > 0.5
    # the smoother's jump copy starts at the add's grid point once the history runs past it
    d = estimator_setup(t, u, pit, cfg, prior_phys, names)
    mv = smoother_moving(idx, t_add)
    @test length(mv) == 5 && length(smoother_moving(idx, BrisketDynamics.ADD_NEVER)) == 3 && length(smoother_moving(idx, NaN)) == 3
    be = 3600.0 .* [2, 4, 6, 8, 10, 12, 14]
    lay6, _, _ = drift_layout(sites, d, be, 6, mv; area_latent=AREA, events)      # to 12 h: no data past the add yet, no copy
    lay7, pr7, _ = drift_layout(sites, d, be, 7, mv; area_latent=AREA, events)    # to 14 h: the copy from the add
    m_r = findfirst(==(ir), lay7.sites); m_u = findfirst(==(iu), lay7.sites)
    @test isempty(lay6.starts[findfirst(==(ir), lay6.sites)]) && lay7.starts[m_r] == [ia]
    @test pr7.incvar[m_r] == [JUMP_LOG_SD["r_max"]^2]
    @test length(lay7.starts[m_u]) == 6 && pr7.incvar[m_u] ≈ fill(2 * DRIFT_SD_PER_HOUR["u_leak"]^2, 6)
    @test lay6.n == 17 + 3 * 5 && lay7.n == 17 + 3 * 6 + 2
    # the warm start carries the old copies over and continues the last value into a new one
    c = warm_start(lay7, vcat(z0, 0.1 .* (1:(lay6.n - 17))), lay6)
    @test length(c) == lay7.n && c[lay7.offsets[m_u] + 5] == 0.1 * (lay6.offsets[findfirst(==(iu), lay6.sites)] - 17 + 5) && c[lay7.offsets[m_u] + 6] == c[lay7.offsets[m_u] + 5]
    @test c[lay7.offsets[m_r] + 1] == z0[ir]
end

@testset "the smoother with no moving site is the batch fit" begin
    cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json"))
    tostep(np, from_dt) = (a = dt / from_dt; NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean^a, np.ρ_sd * a * np.ρ_mean^(a - 1); floor=np.floor))
    np = Dict(1 => tostep(cal["pit"], 600.0), 2 => tostep(cal["core"], 600.0))
    rd = m["readings"]; mk = m["masks"]
    chs = [PitProbe(np[1].σ_mean, np[1].ρ_mean, Bool.(mk["pit"]); data=Float64.(rd["pit"])),
           MeatProbe(np[2].σ_mean, np[2].ρ_mean, Bool.(mk["core"]); depth_latent=DEPTH, data=Float64.(rd["core"]))]
    d = estimator_setup(t, u, pit, cfg, prior_phys, names)
    edges = [2 * 3600.0, 4 * 3600.0]
    br = stream_fit(chs, sites, d, edges; physics=Physics1D(area_latent=AREA, events=events), noise=np, noise_rounds=1, levelled=true, verbose=false)
    sr = smoother_fit(chs, sites, d, edges; moving=[], area_latent=AREA, events=events, noise=np, noise_rounds=1, levelled=true, verbose=false)
    for (b, s) in zip(br, sr)
        @test length(s.z) == 17 && s.converged
        @test maximum(abs.(s.z_now .- b.z) ./ b.sd) < 0.1
        @test maximum(abs.(s.sd_now ./ b.sd .- 1)) < 0.1
    end
    # a drifting leak with a small rate: the copies stay near the base and the block grows by one per edge
    sr2 = smoother_fit(chs, sites, d, edges; moving=[(iu, :drift, 0.05^2)], area_latent=AREA, events=events, noise=np, noise_rounds=1, levelled=true, verbose=false)
    @test length(sr2[1].z) == 17 && length(sr2[2].z) == 18 && sr2[2].layout.starts[1] == [findlast(t .<= 2 * 3600 + 1e-9)]
    @test abs(sr2[2].z_now[iu] - sr2[2].z[iu]) < 3 * 0.05 * sqrt(2)      # within three walk-sd of the base value
    @test sr2[2].sd_now[iu] >= 0.9 * sr2[2].sd[iu]                       # the latest copy is not tighter than the base
end
