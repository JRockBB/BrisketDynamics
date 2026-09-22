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
draw_profile_truth(rng) = E.draw_truth(rng, prior)
physical_profile(z) = E.from_z(z, prior)
ia = findfirst(==("a_myosin"), allnames)
recs = Dict(c["seed"] => c for f in ("coverage_truthnoise_1.json", "coverage_truthnoise_4.json", "coverage_truthnoise_7.json") for c in JSON.parsefile(joinpath(REPO, "data", "mock", f))["cooks"])
for seed in (1, 2, 3, 5)
    c = recs[seed]; b = c["batches"][end]
    z = Float64.(b["z"]); sd = Float64.(b["sd"]); z_true = Float64.(c["z_true"])
    rng = MersenneTwister(1000 + seed); zt = draw_profile_truth(rng); xt = physical_profile(zt)
    θ, φ, A, depth_fraction = xt[1:8], xt[9:15], xt[16], xt[17]
    depth = probe_depth(θ[4], depth_fraction)
    o = monolith_solve(θ, φ, u, t, t; pit, cfg, A_meat=A, events...)
    readings, clean, masks, lid = realcook_readings(o, t, events, nc, rng; A=A, L=θ[4], depth=depth)
    keep = t .<= 14 * 3600
    chs = [PitProbe(truth_noise["pit"]..., Bool.(masks.pit .& keep); data=Float64.(readings.pit)),
           MeatProbe(truth_noise["core"]..., Bool.(masks.core .& keep); depth_latent=DEPTH, data=Float64.(readings.core)),
           CensoredHumidity(truth_noise["humidity"]..., Bool.(masks.humidity .& keep); cap=nc.hum_cap, data=Float64.(readings.humidity)),
           MassScale(truth_noise["mass"]..., Bool.(masks.mass .& keep); data=Float64.(readings.mass))]
    S0, J0, lay = physics_outputs(z, d, ph); _, _, w = assemble(chs, z, S0, J0, prior, lay); ok = .!isnan.(w)
    lp(zz) = (S = physics_stack(zz, d, ph)[1]; a = assemble(chs, zz, S, J0, prior, lay)[1]; -0.5 * sum(abs2, (w .- a)[ok]) - prior_terms(zz, prior)[1])
    lp0 = lp(z)
    ss = collect(-3.0:0.5:3.0)
    vals = [ (zz = copy(z); zz[ia] += s * sd[ia]; lp(zz) - lp0) for s in ss ]
    p = exp.(vals .- maximum(vals)); mean_s = sum(ss .* p) / sum(p)
    @printf("seed %d: a_myosin MAP %.3f (sd %.3f), truth %.3f -> err/sd %+.2f; conditional profile (log-post rel. to the MAP) at s = -3..3:\n  %s\n  conditional mean at s = %+.2f sd (the mode is 0): mode - mean = %+.2f sd; Gaussian would be symmetric\n",
            seed, z[ia], sd[ia], z_true[ia], (z[ia] - z_true[ia]) / sd[ia], join([@sprintf("%+.1f", v) for v in vals], " "), mean_s, -mean_s)
    flush(stdout)
end
