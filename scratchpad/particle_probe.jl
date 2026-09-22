# Feasibility of a particle correction to the Gaussian belief: on one coverage cook (seed 1, C_th 6 sd off at
# 14 h under the Laplace), fit a batch, then draw particles from the Gaussian widened along its widest
# directions, weight them by the EXACT posterior (a plant solve each, the identified noise as fixed), and
# report: the cost of a solve, the effective sample size, the particle mean/sd against the Laplace's, and the
# truth's standardized error under each.  julia --project=estimator scratchpad/particle_probe.jl [seed] [N] [widen]
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, JSON, LinearAlgebra, Random, Printf, Statistics
const E = BrisketEstimator
REPO = normpath(joinpath(@__DIR__, ".."))
seed = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
NP = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100
WIDEN = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 3.0
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "prior_next_cook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
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
cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json"))
tostep(np, from_dt) = (a = dt / from_dt; NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean^a, np.ρ_sd * a * np.ρ_mean^(a - 1); floor=np.floor))
priors = Dict("pit" => tostep(cal["pit"], 600.0), "core" => tostep(cal["core"], 600.0), "humidity" => NoisePrior(100.0, 100.0, 0.8^(dt / 600), 0.3; floor=60.0), "mass" => NoisePrior(0.02, 0.02, 0.5^(dt / 1800), 0.3; floor=0.015))
np = Dict(1 => priors["pit"], 2 => priors["core"], 3 => priors["humidity"], 4 => priors["mass"])
start(p) = (p.σ_mean, p.ρ_mean)
# the same truth and readings as the coverage worker's seed
rng = MersenneTwister(1000 + seed)
z_true = E.draw_truth(rng, prior); x_true = E.from_z(z_true, prior)
θ, φ, A, depth_fraction = x_true[1:8], x_true[9:15], x_true[16], x_true[17]
depth = probe_depth(θ[4], depth_fraction)
o = monolith_solve(θ, φ, u, t, t; pit, cfg, A_meat=A, events...)
readings, clean, masks, lid = realcook_readings(o, t, events, nc, rng; A=A, L=θ[4], depth=depth)
chs = [PitProbe(start(priors["pit"])..., Bool.(masks.pit); data=Float64.(readings.pit)),
       MeatProbe(start(priors["core"])..., Bool.(masks.core); depth_latent=DEPTH, data=Float64.(readings.core)),
       CensoredHumidity(start(priors["humidity"])..., Bool.(masks.humidity); cap=nc.hum_cap, data=Float64.(readings.humidity)),
       MassScale(start(priors["mass"])..., Bool.(masks.mass); data=Float64.(readings.mass))]
edges = 3600.0 .* [6.0, 14.0]
recs = stream_fit(chs, prior, d, edges; physics=ph, noise=np, noise_rounds=1, levelled=true, verbose=false)
println("seed ", seed, ": fits at ", [r.t / 3600 for r in recs], " h; N ", NP, " particles, widen ", WIDEN)
for r in recs
    z, V = r.z, Matrix(Symmetric(r.cov))
    sd = sqrt.(diag(V))
    channels = r.channels
    S0, J0, lay = physics_outputs(z, d, ph)
    a0, _, w = assemble(channels, z, S0, J0, prior, lay)
    ok = .!isnan.(w)
    logpost(zz) = (S = try physics_stack(zz, d, ph)[1] catch err; err isa InterruptException ? rethrow() : nothing end;
                   S === nothing ? -Inf : (a = assemble(channels, zz, S, J0, prior, lay)[1]; -0.5 * sum(abs2, (w .- a)[ok]) - prior_terms(zz, prior)[1]))
    t0 = time(); for _ in 1:5; physics_stack(z, d, ph); end; tsolve = (time() - t0) / 5
    # the proposal: the Gaussian, its widest k directions widened
    F = eigen(Symmetric(V)); idx = sortperm(F.values; rev=true)
    scale = fill(1.2, length(z)); for i in idx[1:4]; scale[i] = WIDEN; end
    Lq = F.vectors * Diagonal(sqrt.(max.(F.values, 0.0)) .* scale[invperm(idx)] )   # widen along eigen-directions
    # (scale is indexed by eigen-order; map back)
    Lq = F.vectors * Diagonal([sqrt(max(F.values[i], 0.0)) * (i in idx[1:4] ? WIDEN : 1.2) for i in eachindex(F.values)])
    Q = Lq * Lq'; Qi = inv(Symmetric(Q + 1e-12I)); logdetQ = logdet(Symmetric(Q + 1e-12I))
    rngp = MersenneTwister(7)
    zs = [z .+ Lq * randn(rngp, length(z)) for _ in 1:NP]
    t1 = time()
    lp = [logpost(zz) for zz in zs]
    lq = [-0.5 * dot(zz .- z, Qi * (zz .- z)) - 0.5 * logdetQ for zz in zs]
    lw = lp .- lq; lw .-= maximum(lw); wgt = exp.(lw); wgt ./= sum(wgt)
    ess = 1 / sum(abs2, wgt)
    zm = sum(wgt .* zs); Vp = sum(wgt[i] .* (zs[i] .- zm) * (zs[i] .- zm)' for i in eachindex(zs)); sdp = sqrt.(diag(Vp))
    lp_map = logpost(z)
    @printf("\nat %.0f h: a solve %.2f s; %d particles in %.0f s; ESS %.1f; log-posterior at the MAP %.1f, best particle %.1f (%d particles above the MAP)\n",
            r.t / 3600, tsolve, NP, time() - t1, ess, lp_map, maximum(lp), count(lp .> lp_map))
    @printf("%-11s %9s %9s %9s %9s %9s\n", "site", "Laplace sd", "particle sd", "ratio", "err/sd L", "err/sd P")
    for (i, n) in enumerate(allnames)
        @printf("%-11s %9.3f %9.3f %9.2f %9.2f %9.2f\n", n, sd[i], sdp[i], sdp[i] / sd[i], (z[i] - z_true[i]) / sd[i], (zm[i] - z_true[i]) / sdp[i])
    end
    flush(stdout)
end
