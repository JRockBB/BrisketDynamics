# The coverage test of the streaming estimator on the honest bench. One
# mock cook shows the truth inside the band on one hard case; it cannot
# show the posterior is calibrated, because its truth was chosen off the
# prior's center and its hourly refits share one noise draw. Here the
# truths are DRAWN from the prior the pilot flies on, each cook gets its
# own noise draw, the plant (the monolith on the two-minute grid under
# the real-cook schedule) is read by the real-cook instruments at their
# rates, and the estimator streams it exactly as mock_stream_realcook.jl
# does (the noise identified, the level a process, the humidity censored,
# two-hourly refits to the pull). The standardized errors
# (estimate − truth) / sd pooled over cooks, per site and per hour, are
# what plot_coverage.py reads: a calibrated posterior has them at mean
# zero and spread one everywhere.
#
# Run:  julia --project=estimator estimator/coverage_realcook.jl [prior.json] [n_cooks] [seed0] [out.json] [edges_to_h] [ridge_k] [ridge_steps]
#   ridge_k > 0: after every batch the belief along the k widest posterior directions is sampled on the exact
#   posterior (ridge.jl) and the corrected mean and sd are recorded beside the Gaussian's (err_sd_ridge)
#   several workers with different seed0 write different files; plot_coverage.py pools them all.
# Writes data/mock/<out.json> (coverage_realcook_<seed0>.json by default), after every cook.
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, ".."))
prior_file = length(ARGS) >= 1 ? ARGS[1] : "prior_next_cook.json"
n_cooks = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
seed0 = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
out_name = length(ARGS) >= 4 ? ARGS[4] : "coverage_realcook_$(seed0).json"
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", prior_file))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
names = String.(gate["names"])
d = estimator_setup(t, u, pit, cfg, prior_phys, names)
L_med = exp(prior_phys[4].p1)
extra = [LatentSpec("A_meat", "lognormal", log(cfg.A_meat), 0.3),
         LatentSpec("core_depth", "lognormal", log(0.8 * L_med), 0.5)]
prior = vcat(prior_phys, [e.site for e in extra])
allnames = vcat(names, [e.name for e in extra])
AREA, DEPTH = length(prior) - 1, length(prior)
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]),
          t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
edges_to_h = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : floor(events.t_pull / 3600)
ridge_k = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 0
ridge_steps = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 500
ph = Physics1D(area_latent=AREA, events=events)
nc = RealCookNoise()

# the instruments' noise as the estimator starts it, as on the bench
cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json"))
tostep(np, from_dt) = (a = dt / from_dt; NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean^a, np.ρ_sd * a * np.ρ_mean^(a - 1); floor=np.floor))
priors = Dict("pit" => tostep(cal["pit"], 600.0), "core" => tostep(cal["core"], 600.0),
              "humidity" => NoisePrior(100.0, 100.0, 0.8^(dt / 600), 0.3; floor=60.0), "mass" => NoisePrior(0.02, 0.02, 0.5^(dt / 1800), 0.3; floor=0.015))
np = Dict(1 => priors["pit"], 2 => priors["core"], 3 => priors["humidity"], 4 => priors["mass"])
start(p) = (p.σ_mean, p.ρ_mean)
edges = 3600.0 .* collect(2:2:floor(Int, edges_to_h))

"A draw from Beta(a, b) by the inverse of its distribution function on a grid (no sampling dependency)."
function beta_rand(rng, a, b)
    x = range(1e-6, 1 - 1e-6; length=4001)
    lp = (a - 1) .* log.(x) .+ (b - 1) .* log.(1 .- x)
    p = exp.(lp .- maximum(lp))
    c = cumsum(p); c ./= c[end]
    r = rand(rng)
    k = clamp(searchsortedfirst(c, r), 2, length(c))
    return x[k - 1] + (x[k] - x[k - 1]) * (r - c[k - 1]) / max(c[k] - c[k - 1], 1e-300)
end
"A truth in z drawn from the prior: Gaussian in log for the lognormal sites, a beta draw in logit for the bounded ones."
function draw_truth(rng, prior)
    z = [isbeta(p) ? (x = beta_rand(rng, p.p1, p.p2); log(x / (1 - x))) : p.p1 + p.p2 * randn(rng) for p in prior]
    z[DEPTH] = min(z[DEPTH], log(0.9 * exp(z[4])))          # the probe sits inside the meat
    return z
end
from_z(z) = [isbeta(p) ? 1 / (1 + exp(-v)) : exp(v) for (v, p) in zip(z, prior)]

cooks = Any[]
skipped = Int[]
dst = joinpath(REPO, "data", "mock", out_name)
save() = open(dst, "w") do io
    JSON.print(io, Dict("plant" => "monolith (1-D) at a truth drawn from the prior, real-cook instruments (2-min grid)", "names" => allnames,
                        "prior_file" => prior_file, "edges_h" => edges ./ 3600, "cooks" => cooks, "skipped_seeds" => skipped,
                        "ridge_k" => ridge_k, "ridge_steps" => ridge_steps,
                        "noise" => Dict(String(k) => getfield(nc, k) for k in fieldnames(RealCookNoise))))
end
println("prior ", prior_file, "; ", n_cooks, " cooks from seed ", seed0, "; batches to ", edges_to_h, " h; record ", out_name)
flush(stdout)
for k in 1:n_cooks
    seed = seed0 + k - 1
    rng = MersenneTwister(1000 + seed)
    z_true = draw_truth(rng, prior)
    x_true = from_z(z_true)
    θ, φ, A, depth = x_true[1:8], x_true[9:15], x_true[16], x_true[17]
    t0 = time()
    o = try
        monolith_solve(θ, φ, u, t, t; pit, cfg, A_meat=A, events...)
    catch err
        err isa InterruptException && rethrow()
        @printf("cook %d (seed %d): the plant failed (%s); skipped\n", k, seed, sprint(showerror, err)[1:min(end, 80)]); push!(skipped, seed); save(); continue
    end
    if !all(isfinite, o.Tcore)
        @printf("cook %d (seed %d): the plant is not finite; skipped\n", k, seed); push!(skipped, seed); save(); continue
    end
    readings, clean, masks, lid = realcook_readings(o, t, events, nc, rng; A=A, L=θ[4], depth=depth)
    chs = [PitProbe(start(priors["pit"])..., Bool.(masks.pit); data=Float64.(readings.pit)),
           MeatProbe(start(priors["core"])..., Bool.(masks.core); depth_latent=DEPTH, data=Float64.(readings.core)),
           CensoredHumidity(start(priors["humidity"])..., Bool.(masks.humidity); cap=nc.hum_cap, data=Float64.(readings.humidity)),
           MassScale(start(priors["mass"])..., Bool.(masks.mass); data=Float64.(readings.mass))]
    recs = try
        stream_fit(chs, prior, d, edges; physics=ph, noise=np, noise_rounds=1, levelled=true, verbose=false)
    catch err
        err isa InterruptException && rethrow()
        @printf("cook %d (seed %d): the estimator failed (%s); skipped\n", k, seed, sprint(showerror, err)[1:min(end, 80)]); push!(skipped, seed); save(); continue
    end
    batches = [Dict("t_h" => r.t / 3600, "z" => r.z, "sd" => r.sd, "err_sd" => (r.z .- z_true) ./ r.sd, "F" => r.F, "iters" => r.iters,
                    "wall_s" => r.wall, "misfit_power" => r.misfit.total.ss / r.misfit.total.n, "misfit_z" => r.misfit.total.z_chi2) for r in recs]
    if ridge_k > 0
        for (b, r) in zip(batches, recs)
            t1 = time()
            rc = ridge_chain(r.channels, prior, d, ph, r.z, r.cov; k=ridge_k, steps=ridge_steps, rng=MersenneTwister(5000 + seed + 97 * round(Int, b["t_h"])))
            b["z_ridge"] = rc.z; b["sd_ridge"] = rc.sd; b["err_sd_ridge"] = (rc.z .- z_true) ./ rc.sd
            b["ridge_accept"] = rc.accept; b["ridge_ess"] = rc.ess; b["ridge_wall_s"] = time() - t1
            b["ridge_sd_ratio"] = rc.sd ./ r.sd
        end
    end
    push!(cooks, Dict("seed" => seed, "z_true" => z_true, "x_true" => x_true, "batches" => batches, "wall_s" => time() - t0,
                      "core_at_pull_C" => clean.core[searchsortedlast(t, events.t_pull)] - 273.15))
    e = (recs[end].z .- z_true) ./ recs[end].sd
    @printf("cook %d (seed %d): core at the pull %.1f C; at %.0f h |err|/sd max %.2f (%s) median %.2f, mean err/sd %+.2f; residual power %.2f; %.0f s\n",
            k, seed, clean.core[searchsortedlast(t, events.t_pull)] - 273.15, recs[end].t / 3600, maximum(abs.(e)), allnames[argmax(abs.(e))],
            sort(abs.(e))[div(end + 1, 2)], sum(e) / length(e), recs[end].misfit.total.ss / recs[end].misfit.total.n, time() - t0)
    if ridge_k > 0
        b = batches[end]; er = b["err_sd_ridge"]
        @printf("    with the ridge chain (k %d, %d steps): |err|/sd max %.2f (%s) median %.2f; sd ratio max %.2f (%s); accept %.2f, ESS %s; %.0f s per batch
",
                ridge_k, ridge_steps, maximum(abs.(er)), allnames[argmax(abs.(er))], sort(abs.(er))[div(end + 1, 2)],
                maximum(b["ridge_sd_ratio"]), allnames[argmax(b["ridge_sd_ratio"])], b["ridge_accept"], join([@sprintf("%.0f", v) for v in b["ridge_ess"]], "/"), b["ridge_wall_s"])
    end
    flush(stdout)
    save()
end
println("wrote ", dst, " (", length(cooks), " cooks, ", length(skipped), " skipped)")
