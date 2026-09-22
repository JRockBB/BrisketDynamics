# the exact log-posterior along the a_myosin coordinate at the 14-h MAP of three coverage cooks (the truth's noise,
# so the recorded MAP is exact for these channels): its skew, and the conditional mean against the mode
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, JSON, LinearAlgebra, Random, Printf
const E = BrisketEstimator
REPO = normpath(joinpath(@__DIR__, ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "prior_next_cook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"])); cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = read_sites(gate["prior"])
names = String.(gate["names"])
d = estimator_setup(t, u, pit, cfg, prior_phys, names)
extra = [LatentSpec("A_meat", "lognormal", log(cfg.A_meat), 0.3),
         LatentSpec("core_depth_fraction", probe_depth_site())]
prior = vcat(prior_phys, [e.site for e in extra]); allnames = vcat(names, [e.name for e in extra])
AREA, DEPTH = length(prior) - 1, length(prior)
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
ph = Physics1D(area_latent=AREA, events=events)
nc = RealCookNoise()
truth_noise = Dict("pit" => (nc.pit_sd, exp(-dt / nc.pit_tau)), "core" => (nc.core_sd, exp(-dt / nc.core_tau)), "humidity" => (nc.hum_sd, exp(-dt / nc.hum_tau)), "mass" => (nc.mass_sd, exp(-dt / nc.mass_tau)))
draw_amyosin_truth(rng) = E.draw_truth(rng, prior)
physical_amyosin(z) = E.from_z(z, prior)
ia = findfirst(==("a_myosin"), allnames)
recs = Dict(c["seed"] => c for f in ("coverage_truthnoise_1.json", "coverage_truthnoise_4.json", "coverage_truthnoise_7.json") for c in JSON.parsefile(joinpath(REPO, "data", "mock", f))["cooks"])
