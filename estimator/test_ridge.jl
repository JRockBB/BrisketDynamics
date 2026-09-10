# The ridge chain: a Metropolis chain on the exact posterior along the widest Gaussian directions
# (ridge.jl), on the real-cook mock's first two hours. Guards the acceptance test (a closure assigning to the
# chain's variable once accepted every proposal), the corrected covariance, and the draws' log-posteriors.
#   julia --project=estimator estimator/test_ridge.jl
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
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
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
names = String.(gate["names"])
d = estimator_setup(t, u, pit, cfg, prior_phys, names)
L_med = exp(prior_phys[4].p1)
extra = [LatentSpec("A_meat", "lognormal", log(cfg.A_meat), 0.3), LatentSpec("core_depth", "lognormal", log(0.8 * L_med), 0.5)]
prior = vcat(prior_phys, [e.site for e in extra])
AREA, DEPTH = length(prior) - 1, length(prior)
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
ph = Physics1D(area_latent=AREA, events=events)
cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json"))
tostep(np, from_dt) = (a = dt / from_dt; NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean^a, np.ρ_sd * a * np.ρ_mean^(a - 1); floor=np.floor))
priors = Dict("pit" => tostep(cal["pit"], 600.0), "core" => tostep(cal["core"], 600.0),
              "humidity" => NoisePrior(100.0, 100.0, 0.8^(dt / 600), 0.3; floor=60.0), "mass" => NoisePrior(0.02, 0.02, 0.5^(dt / 1800), 0.3; floor=0.015))
nc = m["noise"]; rd = m["readings"]; mk = m["masks"]
start(np) = (np.σ_mean, np.ρ_mean)
chs = [PitProbe(start(priors["pit"])..., Bool.(mk["pit"]); data=Float64.(rd["pit"])),
       MeatProbe(start(priors["core"])..., Bool.(mk["core"]); depth_latent=DEPTH, data=Float64.(rd["core"])),
       CensoredHumidity(start(priors["humidity"])..., Bool.(mk["humidity"]); cap=Float64(nc["hum_cap"]), data=Float64.(rd["humidity"])),
       MassScale(start(priors["mass"])..., Bool.(mk["mass"]); data=Float64.(rd["mass"]))]
np = Dict(1 => priors["pit"], 2 => priors["core"], 3 => priors["humidity"], 4 => priors["mass"])
recs = stream_fit(chs, prior, d, [2 * 3600.0]; physics=ph, noise=np, noise_rounds=1, levelled=true, verbose=false)
r = recs[end]

@testset "the ridge chain samples the exact posterior along the widest directions" begin
    rc = ridge_chain(r.channels, prior, d, ph, r.z, r.cov; k=2, steps=80, burn=20, rng=MersenneTwister(11), thin=5)
    @test 0.02 < rc.accept < 0.95                          # a chain that accepts everything has lost its acceptance test
    @test length(rc.z) == length(r.z) && all(isfinite, rc.sd) && all(rc.sd .> 0)
    @test size(rc.cov) == (length(r.z), length(r.z)) && eigmin(Symmetric(rc.cov)) > -1e-10
    @test size(rc.directions) == (length(r.z), 2) && length(rc.ess) == 2 && all(rc.ess .>= 1)
    # the draws sit where the posterior has mass: within a few dozen nats of the MAP, never thousands away
    S0, J0, lay = physics_outputs(r.z, d, ph)
    _, _, w = assemble(r.channels, r.z, S0, J0, prior, lay); ok = .!isnan.(w)
    lp(zz) = (S = physics_stack(zz, d, ph)[1]; a = assemble(r.channels, zz, S, J0, prior, lay)[1]; -0.5 * sum(abs2, (w .- a)[ok]) - prior_terms(zz, prior)[1])
    lp0 = lp(r.z)
    rel = [lp(zz) - lp0 for zz in rc.draws]
    @test all(rel .> -40) && maximum(rel) < 5
    # off the ridge directions the belief is the Gaussian's: the covariance agrees there
    P = I - rc.directions * rc.directions'
    @test maximum(abs.(P * (rc.cov - Matrix(r.cov)) * P)) < 1e-8
    # the corrected mean moves only along the ridge directions
    @test norm(P * (rc.z - r.z)) < 1e-8
end
