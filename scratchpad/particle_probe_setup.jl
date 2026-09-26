# Feasibility of a particle correction to the Gaussian belief: on one coverage cook (seed 1, C_th 6 sd off at
# 14 h under the Laplace), fit a batch, then draw particles from the Gaussian widened along its widest
# directions, weight them by the EXACT posterior (a plant solve each, the identified noise as fixed), and
# report: the cost of a solve, the effective sample size, the particle mean/sd against the Laplace's, and the
# truth's standardized error under each.  julia --project=estimator scratchpad/particle_probe.jl [seed] [N] [widen]
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "src", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, JSON, LinearAlgebra, Random, Printf, Statistics
const E = BrisketEstimator
REPO = normpath(joinpath(@__DIR__, ".."))
seed = 1
NP = 0
WIDEN = 1.0
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "prior_next_cook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = read_sites(gate["prior"])
names = String.(gate["names"])
d = estimator_setup(t, u, pit, cfg, prior_phys, names)
extra = [LatentSpec("A_meat", "lognormal", log(cfg.A_meat), 0.3), LatentSpec("core_depth_fraction", probe_depth_site())]
prior = vcat(prior_phys, [e.site for e in extra]); allnames = vcat(names, [e.name for e in extra])
AREA, DEPTH = length(prior) - 1, length(prior)
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
ph = Physics1D(area_latent=AREA, events=events)
nc = RealCookNoise()
cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json"))
tostep(np, from_dt) = (a = dt / from_dt; NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean^a, np.ρ_sd * a * np.ρ_mean^(a - 1); floor=np.floor))
priors = Dict("pit" => tostep(cal["pit"], 600.0), "core" => tostep(cal["core"], 600.0), "humidity" => NoisePrior(100.0, 100.0, 0.8^(dt / 600), 0.3; floor=60.0), "mass" => NoisePrior(0.02, 0.02, 0.5^(dt / 1800), 0.3; floor=0.015))
np = Dict(1 => priors["pit"], 2 => priors["core"], 3 => priors["humidity"], 4 => priors["mass"])
start(p) = (p.σ_mean, p.ρ_mean)
rng = MersenneTwister(1000 + seed)
z_true = draw_truth(rng, prior); x_true = from_z(z_true, prior)     # the same truth and readings as the coverage worker's seed
θ, φ, A, depth = x_true[1:8], x_true[9:15], x_true[16], probe_depth(x_true[4], x_true[17])
o = monolith_solve(θ, φ, u, t, t; pit, cfg, A_meat=A, events...)
readings, clean, masks, lid = realcook_readings(o, t, events, nc, rng; A=A, L=θ[4], depth=depth, cfg)
chs = [PitProbe(start(priors["pit"])..., Bool.(masks.pit); data=Float64.(readings.pit)),
       MeatProbe(start(priors["core"])..., Bool.(masks.core); depth_latent=DEPTH, data=Float64.(readings.core)),
       CensoredHumidity(start(priors["humidity"])..., Bool.(masks.humidity); cap=nc.hum_cap, data=Float64.(readings.humidity)),
       MassScale(start(priors["mass"])..., Bool.(masks.mass); data=Float64.(readings.mass))]
edges = 3600.0 .* [4.0]
recs = stream_fit(chs, prior, d, edges; physics=ph, noise=np, noise_rounds=1, levelled=true, verbose=false)
