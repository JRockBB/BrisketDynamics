# The state hand-off (segments.jl) and the full-covariance filter prior (FilterPrior, prior_quadratic).
#   julia --project=estimator estimator/test_segments.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test
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
    ics, pit_k, a0 = handoff(o1, edges[1], pit)
    @test a0 ≈ clamp(o1.ignited[edges[1]], 1e-6, 1 - 1e-6) && pit_k.Tpit0 ≈ o1.Tpit[edges[1]] && pit_k.m0 ≈ o1.m_fuel[edges[1]]
    @test pit_k.T_amb == pit.T_amb && length(ics.T0f) == nx
end

prior_phys = read_sites(gate["prior"])
names = String.(gate["names"])
d = estimator_setup(t, u, pit, cfg, prior_phys, names)
extra = [LatentSpec("A_meat", "lognormal", log(cfg.A_meat), 0.3), LatentSpec("core_depth_fraction", probe_depth_site())]
sites = vcat(prior_phys, [e.site for e in extra]); n = length(sites)
cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json"))
tostep(np, from_dt) = (a = dt / from_dt; NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean^a, np.ρ_sd * a * np.ρ_mean^(a - 1); floor=np.floor))
nc = m["noise"]; rd = m["readings"]; mk = m["masks"]
sp = tostep(cal["pit"], 600.0); sc = tostep(cal["core"], 600.0)
keep2 = t .<= 2 * 3600
chs = [PitProbe(sp.σ_mean, sp.ρ_mean, Bool.(mk["pit"]) .& keep2; data=Float64.(rd["pit"])),
       MeatProbe(sc.σ_mean, sc.ρ_mean, Bool.(mk["core"]) .& keep2; depth_latent=n, data=Float64.(rd["core"]))]
ph = Physics1D(area_latent=n - 1, events=events)
z0 = vcat(d.z0, [log(cfg.A_meat), z_center(sites[end])])
r1 = solve_map_channels(chs, sites, d; c0=z0, physics=ph, engine=:graph, verbose=false)
sd1 = sqrt.(diag(r1.cov))
println("fit 1: converged ", r1.converged, ", F ", round(r1.F; digits=3), ", iters ", length(r1.hist))
fp2 = FilterPrior(sites, r1.z, inv(Symmetric(Matrix(r1.cov))))
r2 = solve_map_channels(chs, fp2, d; c0=r1.z, physics=ph, engine=:graph, verbose=true)
sd2 = sqrt.(diag(r2.cov))
println("fit 2: converged ", r2.converged, ", F ", round(r2.F; digits=3), ", iters ", length(r2.hist))
println("max |z2 - z1| / sd1 = ", round(maximum(abs.(r2.z .- r1.z) ./ sd1); digits=3), "; sd2/sd1 max ", round(maximum(sd2 ./ sd1); digits=3), " min ", round(minimum(sd2 ./ sd1); digits=3))
println("the objective at z1 under the filter prior: ", round(E._objective(assemble(chs, r1.z, physics_outputs(r1.z, d, ph)[1], zeros(0,0), fp2, physics_outputs(r1.z, d, ph)[3])[1], assemble(chs, r1.z, physics_outputs(r1.z, d, ph)[1], zeros(0,0), fp2, physics_outputs(r1.z, d, ph)[3])[3], r1.z, fp2); digits=3))
