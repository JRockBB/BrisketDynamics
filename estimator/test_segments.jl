# The state hand-off (segments.jl) and the full-covariance filter prior (FilterPrior, prior_quadratic).
#   julia --project=estimator estimator/test_segments.jl
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test, ForwardDiff
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
x = Float64.(m["truth"]["x"]); θ, φ, A, L = x[1:8], x[9:15], x[16], x[4]
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
keep = t .<= 16 * 3600
tt, uu = t[keep], u[keep]

@testset "a cook solved in segments equals the cook solved whole" begin
    o1 = monolith_solve(θ, φ, uu, tt, tt; pit, cfg, A_meat=A, events...)
    edges = [findlast(tt .<= 5 * 3600), findlast(tt .<= 10 * 3600), findlast(tt .<= 14 * 3600)]     # away from the add's pulse at 12 h
    o3 = solve_segments([θ, θ, θ, θ], [φ, φ, φ, φ], uu, tt, edges; pit, cfg, A_meat=A, events...)
    @test length(o3.t) == length(tt) && size(o3.T_field) == size(o1.T_field)
    nx = size(o1.W_field, 1); w = E.fv_weights(nx)
    mass(o) = [E.RHO_DRY_EST * A * L * (1 + dot(w, view(o.W_field, :, k)) + dot(w, view(o.F_field, :, k))) for k in axes(o.W_field, 2)]
    dcore = maximum(abs.(o3.Tcore .- o1.Tcore)); dpit = maximum(abs.(o3.Tpit .- o1.Tpit))
    dmass = maximum(abs.(mass(o3) .- mass(o1))); dpv = maximum(abs.(o3.pv_pit .- o1.pv_pit)); dbark = maximum(abs.(o3.l_star .- o1.l_star))
    println("segmented vs whole: core $(round(dcore; digits=4)) K, pit $(round(dpit; digits=4)) K, mass $(round(dmass * 1000; digits=3)) g, vapor $(round(dpv; digits=2)) Pa, bark $(round(dbark; digits=3)) L*")
    @test dcore < 0.05 && dpit < 0.2 && dmass < 1e-3 && dpv < 5.0 && dbark < 0.05     # the noise is 1.2 K, 1.5 K, 20 g, 100 Pa
    # the handed-off ignited fraction is what the whole solve had at the boundary
    ics, a0 = handoff(o1, edges[1])
    @test a0 ≈ clamp(o1.ignited[edges[1]], 1e-6, 1 - 1e-6) && ics.Tpit0 ≈ o1.Tpit[edges[1]] && ics.mfuel0 ≈ o1.m_fuel[edges[1]] && ics.Tfb0 ≈ o1.T_firebox[edges[1]]
    @test length(ics.T0f) == nx
    # the segmented cook differentiates through the hand-off: its ForwardDiff Jacobian equals the whole solve's
    # (finite differences are no yardstick here: the adaptive solver's response is rough at the 0.1 % level)
    e2 = edges[1:2]; k10 = e2[2]
    t10, u10 = tt[1:k10], uu[1:k10]
    f_whole(v) = (θv = vcat(θ[1:2], v[1], θ[4:8]); φv = vcat(φ[1:5], v[2], φ[7]);
                  o = monolith_solve(θv, φv, u10, t10, t10; pit, cfg, A_meat=A, events...); [o.Tcore[end], o.Tpit[end]])
    f_seg(v) = (θv = vcat(θ[1:2], v[1], θ[4:8]); φv = vcat(φ[1:5], v[2], φ[7]);
                o = solve_segments([θv, θv, θv], [φv, φv, φv], uu, tt, e2; pit, cfg, A_meat=A, events...); [o.Tcore[k10], o.Tpit[k10]])
    v0 = [θ[3], φ[6]]
    Jw = ForwardDiff.jacobian(f_whole, v0); Js = ForwardDiff.jacobian(f_seg, v0)
    println("Jacobian at 10 h, whole ", round.(Jw; sigdigits=4), " segmented ", round.(Js; sigdigits=4))
    @test maximum(abs.(Js .- Jw) ./ (abs.(Jw) .+ 1e-2)) < 0.02
end

@testset "the filter prior: a full Gaussian the graph reads like the families" begin
    prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
    names = String.(gate["names"])
    d = estimator_setup(t, u, pit, cfg, prior_phys, names)
    L_med = exp(prior_phys[4].p1)
    extra = [LatentSpec("A_meat", "lognormal", log(cfg.A_meat), 0.3), LatentSpec("core_depth", "lognormal", log(0.8 * L_med), 0.5)]
    sites = vcat(prior_phys, [e.site for e in extra])
    n = length(sites)
    rng = MersenneTwister(4)
    Q = randn(rng, n, n); Λ = Q * Q' + 0.5 * I
    mz = randn(rng, n)
    fp = FilterPrior(sites, mz, Λ)
    @test length(fp) == n && fp[4].family == "lognormal" && isbeta(fp[8]) == isbeta(sites[8]) && collect(fp) == sites
    @test from_z(mz, fp) == from_z(mz, sites)
    c = mz .+ 0.3 .* randn(rng, n)
    f, g, H = prior_terms(c, fp)
    @test f ≈ 0.5 * dot(c - mz, Λ * (c - mz)) && g ≈ Λ * (c - mz) && H ≈ Λ
    # finite-difference gradient
    ε = 1e-6; gfd = [(prior_terms(c .+ ε .* (1:n .== i), fp)[1] - prior_terms(c .- ε .* (1:n .== i), fp)[1]) / 2ε for i in 1:n]
    @test maximum(abs.(gfd .- g)) < 1e-5
    # the graph's prior node sees exactly the Gaussian, at any linearization point
    mp, Λp = prior_quadratic(c, fp)
    @test maximum(abs.(mp .- mz)) < 1e-9
    @test Λp ≈ Λ
    mp2, Λp2 = prior_quadratic(c, fp; damping=2.0)
    @test Λp2 ≈ Λ + 2.0 * I
    # the families still go through the same helper
    mp3, Λp3 = prior_quadratic(c, sites)
    _, g3, h3 = prior_terms(c, sites)
    @test Λp3 ≈ Matrix(Diagonal(h3)) && mp3 ≈ c .- g3 ./ h3
    # a fit under a filter prior centred on a fit's own MAP stays there: the bench's first two hours
    cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json"))
    tostep(np, from_dt) = (a = dt / from_dt; NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean^a, np.ρ_sd * a * np.ρ_mean^(a - 1); floor=np.floor))
    nc = m["noise"]; rd = m["readings"]; mk = m["masks"]
    sp = tostep(cal["pit"], 600.0); sc = tostep(cal["core"], 600.0)
    keep2 = t .<= 2 * 3600
    chs = [PitProbe(sp.σ_mean, sp.ρ_mean, Bool.(mk["pit"]) .& keep2; data=Float64.(rd["pit"])),
           MeatProbe(sc.σ_mean, sc.ρ_mean, Bool.(mk["core"]) .& keep2; depth_latent=n, data=Float64.(rd["core"]))]
    ph = Physics1D(area_latent=n - 1, events=events)
    z0 = vcat(d.z0, [log(cfg.A_meat), log(0.8 * L_med)])
    r1 = solve_map_channels(chs, sites, d; c0=z0, physics=ph, engine=:graph, verbose=false)
    sd1 = sqrt.(diag(r1.cov))
    # the families replaced by their own local quadratic at the MAP: the same objective to second order there,
    # so the fit stays put and reports the same widths
    mq, Λq = prior_quadratic(r1.z, sites)
    fp2 = FilterPrior(sites, mq, Λq)
    r2 = solve_map_channels(chs, fp2, d; c0=r1.z, physics=ph, engine=:graph, verbose=false)
    sd2 = sqrt.(diag(r2.cov))
    @test r2.converged && maximum(abs.(r2.z .- r1.z) ./ sd1) < 0.05
    @test maximum(abs.(sd2 ./ sd1 .- 1)) < 0.05
end
