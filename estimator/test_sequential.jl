# The sequential filter on the graph (sequential.jl): the state block's packing, the linear family, the slice
# physics against the whole-cook physics at the recipe's state, and one slice's fit on the graph.
#   julia --project=estimator estimator/test_sequential.jl
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
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "prior_next_cook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
names = String.(gate["names"])
L_med = exp(prior_phys[4].p1)
sites = vcat(prior_phys, [PriorSite("lognormal", log(cfg.A_meat), 0.3), PriorSite("lognormal", log(0.8 * L_med), 0.5)])
AREA, DEPTH = 16, 17
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
z0 = [isbeta(p) ? log(p.p1 / p.p2) : p.p1 for p in sites]
x0 = from_z(z0[1:N_PHYS], prior_phys)

@testset "the state block: packing, the linear family" begin
    o0 = monolith_solve(x0[1:N_THETA], x0[(N_THETA + 1):N_PHYS], u[1:2], t[1:2], t[1:2]; pit, cfg, A_meat=exp(z0[AREA]), events...)
    ics, a = handoff(o0, 1)
    lay = FilterLayout(length(ics.T0f), length(sites))
    @test lay.n_state == 5 * lay.nx + 9 && lay.n == 17 + lay.n_state
    xs = pack_state(ics, a, lay)
    ics2, a2 = unpack_state(xs, lay)
    @test a2 == a && ics2.T0f == ics.T0f && ics2.Ssalt0f == ics.Ssalt0f && ics2.Tpit0 == ics.Tpit0 && ics2.Tfb0 == ics.Tfb0 && ics2.mfuel0 == ics.mfuel0
    @test xs[pit_index(lay) - lay.n_z] == ics.Tpit0 && xs[centre_index(lay) - lay.n_z] == ics.T0f[1] && xs[surface_index(lay) - lay.n_z] == ics.T0f[end]
    ss = state_sites(lay.n_state)
    v = randn(MersenneTwister(1), lay.n_state)
    @test from_z(v, ss) == v && all(E.dx_dz(v, ss) .== 1)
    joint = vcat(sites, ss)
    f, g, h = prior_terms(vcat(z0, v), joint)
    @test length(g) == lay.n && all(isfinite, h)
end

@testset "the slice physics from the recipe's state is the whole-cook physics" begin
    keep = t .<= 1200.0 + 1e-9                          # the first two ten-minute slices as one window
    tw, uw = t[keep], u[keep]
    d = estimator_setup(tw, uw, pit, cfg, prior_phys, names)
    o0 = monolith_solve(x0[1:N_THETA], x0[(N_THETA + 1):N_PHYS], u[1:2], t[1:2], t[1:2]; pit, cfg, A_meat=exp(z0[AREA]), events...)
    ics, a = handoff(o0, 1)
    lay = FilterLayout(length(ics.T0f), length(sites))
    v = vcat(z0, pack_state(ics, a, lay))
    ph = PhysicsState(AREA, events, lay, true)
    S1, lay1 = physics_stack(z0, d, Physics1D(area_latent=AREA, events=events))
    S2, lay2 = physics_stack(v, d, ph)
    nt, nx = lay1.nt, lay1.nx
    blocks = [(lay1.oTpit, nt, 0.05), (lay1.oT, nx * nt, 0.05), (lay1.oW, nx * nt, 1e-3), (lay1.opv, nt, 5.0), (lay1.oMass, nt, 1e-3)]
    @test lay2.nt == nt && all(maximum(abs.(S2[(o + 1):(o + len)] .- S1[(o + 1):(o + len)])) < tol for (o, len, tol) in blocks)
    S3, J3, _ = physics_outputs(v, d, ph)
    @test size(J3) == (length(S3), lay.n) && all(isfinite, J3)
    L = ph.last[]
    @test length(L.x_end) == lay.n_state && size(L.J_end) == (lay.n_state, lay.n) && all(isfinite, L.J_end)
    @test abs(L.x_end[5 * lay.nx + 4] - S3[lay2.oTpit + nt]) < 1e-6           # the end state's pit temperature is the stack's last pit value
    @test maximum(abs.(L.J_end[:, (lay.n_z + 1):end] .- Matrix(1.0 * I, lay.n_state, lay.n_state))) > 0.01   # the state moves the end state (not the identity)
end

@testset "a slice fit on the graph moves the belief toward the readings" begin
    nc = RealCookNoise()
    cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json"))
    tostep(np, from_dt) = (a = dt / from_dt; NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean^a, np.ρ_sd * a * np.ρ_mean^(a - 1); floor=np.floor))
    noise = Dict(1 => tostep(cal["pit"], 600.0), 2 => tostep(cal["core"], 600.0),
                 3 => NoisePrior(100.0, 100.0, 0.8^(dt / 600), 0.3; floor=60.0), 4 => NoisePrior(0.02, 0.02, 0.5^(dt / 1800), 0.3; floor=0.015))
    rd = m["readings"]; mk = m["masks"]
    readings = (pit=Float64.(rd["pit"]), core=Float64.(rd["core"]), humidity=Float64.(rd["humidity"]), mass=Float64.(rd["mass"]))
    masks = (pit=Bool.(mk["pit"]), core=Bool.(mk["core"]), humidity=Bool.(mk["humidity"]), mass=Bool.(mk["mass"]))
    q = zeros(length(sites))
    recs = sequential_filter(readings, masks, t, u, pit, cfg, sites, names, events; step=600.0, noise, cap=nc.hum_cap,
                             depth_latent=DEPTH, area_latent=AREA, q, verbose=false, t_end=1800.0)
    @test length(recs) == 3 && all(all(isfinite, r.z) && isfinite(r.misfit.total.z_chi2) for r in recs)   # a slice is one or two linearizations, not a converged fit
    r = recs[end]
    @test all(isfinite, r.z) && all(r.z_sd .> 0) && r.T_pit[2] > 0 && r.T_centre[2] >= 0 && r.T_surface[2] > 0
    _, _, h0 = prior_terms(z0, sites)
    @test all(r.z_sd .<= sqrt.(1 ./ h0) .* (1 + 1e-6))                                  # no latent wider than its prior (no process noise here)
    # the belief's pit temperature sits with the pit readings of the last slice (the probe's noise is 1.5 K)
    k = searchsortedlast(t, r.t)
    @test abs(r.T_pit[1] - readings.pit[k]) < 6.0
end
