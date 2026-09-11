# The coverage test on a DRIFTING plant: truths drawn from the pilot's prior, then walked through the cook
# (the leak and the surface coefficients on random walks, a new charge at the add), each cook read by the
# real-cook instruments with its own noise draw, and streamed two ways: the batch estimator (every latent a
# constant on the full history) and the filter (filter.jl: predict with the same process noise the plant
# walks with, update on the window). The standardized errors are against the truth AT THAT HOUR. A calibrated
# belief has them at mean zero and spread one; the batch estimator on a moving parameter does not.
#
# Run:  julia --project=estimator estimator/coverage_drift.jl [prior.json] [n_cooks] [seed0] [out.json] [edges_to_h]
# Writes data/mock/<out.json> (coverage_drift_<seed0>.json by default), after every cook; plot_coverage.py
# reads it with the keys err_sd (batch), err_sd_filter and err_sd_smoother.
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
out_name = length(ARGS) >= 4 ? ARGS[4] : "coverage_drift_$(seed0).json"
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", prior_file))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
dt = t[2] - t[1]
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
edges_to_h = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : floor(events.t_pull / 3600)
edges = 3600.0 .* collect(2:2:floor(Int, edges_to_h))
nc = RealCookNoise()
cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json"))
tostep(np, from_dt) = (a = dt / from_dt; NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean^a, np.ρ_sd * a * np.ρ_mean^(a - 1); floor=np.floor))
np = Dict(1 => tostep(cal["pit"], 600.0), 2 => tostep(cal["core"], 600.0), 3 => NoisePrior(100.0, 100.0, 0.8^(dt / 600), 0.3; floor=60.0), 4 => NoisePrior(0.02, 0.02, 0.5^(dt / 1800), 0.3; floor=0.015))
idx = Dict(n => i for (i, n) in enumerate(allnames))
# the drift the plant walks with, and the process noise the filter is told (the same: the test is of the filter, not of guessing q)
DRIFT = Dict("u_leak" => 0.05, "k_e" => 0.05, "h_c" => 0.03)      # sd per hour in z
JUMP = Dict("r_max" => 0.15, "dH_eff" => 0.10)                     # the new charge at the add: log-sd about the first charge
q = zeros(length(sites)); for (nm, sd) in DRIFT; q[idx[nm]] = sd^2; end
moving = vcat([(idx[nm], :drift, sd^2) for (nm, sd) in DRIFT], [(idx[nm], :jump, (events.t_add, sd^2)) for (nm, sd) in JUMP])
hours = collect(1.0:1.0:30.0)
seg_edges = [findlast(t .<= h * 3600) for h in hours[1:(end - 1)]]

function beta_rand(rng, a, b)
    x = range(1e-6, 1 - 1e-6; length=4001); lp = (a - 1) .* log.(x) .+ (b - 1) .* log.(1 .- x); p = exp.(lp .- maximum(lp)); c = cumsum(p); c ./= c[end]
    r = rand(rng); k = clamp(searchsortedfirst(c, r), 2, length(c)); x[k - 1] + (x[k] - x[k - 1]) * (r - c[k - 1]) / max(c[k] - c[k - 1], 1e-300)
end
function draw_truth(rng)
    z = [isbeta(p) ? (x = beta_rand(rng, p.p1, p.p2); log(x / (1 - x))) : p.p1 + p.p2 * randn(rng) for p in sites]
    z[DEPTH] = min(z[DEPTH], log(0.9 * exp(z[4]))); z
end
from_z_all(z) = [isbeta(p) ? 1 / (1 + exp(-v)) : exp(v) for (v, p) in zip(z, sites)]
to_z_all(x) = [isbeta(p) ? log(v / (1 - v)) : log(v) for (v, p) in zip(x, sites)]
"The truth walked through the cook: z per hour (17 latents; the area and the depth constant)."
function walk_truth(rng, z0)
    zs = Vector{Vector{Float64}}(); z = copy(z0)
    for (k, h) in enumerate(hours)
        if k > 1
            for (nm, sd) in DRIFT; z[idx[nm]] += sd * randn(rng); end
            if hours[k - 1] < events.t_add / 3600 <= h
                for (nm, sd) in JUMP; z[idx[nm]] = z0[idx[nm]] + sd * randn(rng); end
            end
        end
        push!(zs, copy(z))
    end
    return zs
end
truth_at(zs, te) = zs[max(1, findlast(hours .<= te / 3600 + 1e-9))]
# the filter's a0 after its first window is the ignited fraction at the window's START (a state handed off from
# the last window's solve), not the initial fraction: its truth is the plant's ignited fraction there
function truth_filter(zs, o, k)
    zt = copy(truth_at(zs, edges[k]))
    if k > 1
        i0 = findlast(t .<= edges[k - 1] + 1e-9)
        ig = clamp(o.ignited[i0], 1e-6, 1 - 1e-6)
        zt[A0_INDEX] = log(ig / (1 - ig))
    end
    return zt
end

cooks = Any[]; skipped = Int[]
dst = joinpath(REPO, "data", "mock", out_name)
save() = open(dst, "w") do io
    JSON.print(io, Dict("plant" => "monolith (1-D) in hourly segments, parameters drifting; real-cook instruments", "names" => allnames,
                        "prior_file" => prior_file, "edges_h" => edges ./ 3600, "cooks" => cooks, "skipped_seeds" => skipped,
                        "drift" => Dict("sd_per_hour" => DRIFT, "jump_log_sd" => JUMP), "q" => q))
end
println("prior ", prior_file, "; ", n_cooks, " cooks from seed ", seed0, "; batches to ", edges_to_h, " h; drift ", DRIFT, "; jump ", JUMP, "; record ", out_name)
flush(stdout)
for k in 1:n_cooks
    seed = seed0 + k - 1
    rng = MersenneTwister(2000 + seed)
    z0 = draw_truth(rng)
    zs = walk_truth(rng, z0)
    xs = [from_z_all(z) for z in zs]
    A, depth = xs[1][AREA], xs[1][DEPTH]
    t0 = time()
    o = try
        solve_segments([x[1:8] for x in xs], [x[9:15] for x in xs], u, t, seg_edges; pit, cfg, A_meat=A, events...)
    catch err
        err isa InterruptException && rethrow()
        @printf("cook %d (seed %d): the plant failed (%s); skipped\n", k, seed, sprint(showerror, err)[1:min(end, 80)]); push!(skipped, seed); save(); continue
    end
    all(isfinite, o.Tcore) || (@printf("cook %d (seed %d): the plant is not finite; skipped\n", k, seed); push!(skipped, seed); save(); continue)
    readings, clean, masks, lid = realcook_readings(o, t, events, nc, rng; A=A, L=xs[1][4], depth=depth)
    # the batch estimator
    chs = [PitProbe(np[1].σ_mean, np[1].ρ_mean, Bool.(masks.pit); data=Float64.(readings.pit)),
           MeatProbe(np[2].σ_mean, np[2].ρ_mean, Bool.(masks.core); depth_latent=DEPTH, data=Float64.(readings.core)),
           CensoredHumidity(np[3].σ_mean, np[3].ρ_mean, Bool.(masks.humidity); cap=nc.hum_cap, data=Float64.(readings.humidity)),
           MassScale(np[4].σ_mean, np[4].ρ_mean, Bool.(masks.mass); data=Float64.(readings.mass))]
    d = estimator_setup(t, u, pit, cfg, prior_phys, names)
    br = try
        stream_fit(chs, sites, d, edges; physics=Physics1D(area_latent=AREA, events=events), noise=np, noise_rounds=1, levelled=true, verbose=false)
    catch err
        err isa InterruptException && rethrow()
        @printf("cook %d (seed %d): the batch estimator failed (%s); skipped\n", k, seed, sprint(showerror, err)[1:min(end, 80)]); push!(skipped, seed); save(); continue
    end
    tb = time() - t0; t1 = time()
    # the filter
    rd = (pit=Float64.(readings.pit), core=Float64.(readings.core), humidity=Float64.(readings.humidity), mass=Float64.(readings.mass))
    mk = (pit=Bool.(masks.pit), core=Bool.(masks.core), humidity=Bool.(masks.humidity), mass=Bool.(masks.mass))
    fr = try
        filter_fit(rd, mk, t, u, pit, cfg, sites, names, events, edges; q, noise=np, cap=nc.hum_cap, depth_latent=DEPTH, area_latent=AREA, verbose=false)
    catch err
        err isa InterruptException && rethrow()
        @printf("cook %d (seed %d): the filter failed (%s); skipped\n", k, seed, sprint(showerror, err)[1:min(end, 80)]); push!(skipped, seed); save(); continue
    end
    tf = time() - t1; t2 = time()
    sr = try
        smoother_fit(chs, sites, d, edges; moving, area_latent=AREA, events=events, noise=np, noise_rounds=1, levelled=true, verbose=false)
    catch err
        err isa InterruptException && rethrow()
        @printf("cook %d (seed %d): the smoother failed (%s); skipped
", k, seed, sprint(showerror, err)[1:min(end, 80)]); push!(skipped, seed); save(); continue
    end
    ts = time() - t2
    batches = [Dict("t_h" => b.t / 3600, "z" => b.z, "sd" => b.sd, "err_sd" => (b.z .- truth_at(zs, b.t)) ./ b.sd, "F" => b.F,
                    "misfit_power" => b.misfit.total.ss / b.misfit.total.n,
                    "z_filter" => f.z, "sd_filter" => f.sd, "err_sd_filter" => (f.z .- truth_filter(zs, o, kk)) ./ f.sd, "F_filter" => f.F,
                    "z_true_filter" => truth_filter(zs, o, kk), "z_true" => truth_at(zs, b.t),
                    "filter_prior_sd" => [isnan(x) ? nothing : x for x in f.prior_sd], "filter_converged" => f.converged, "filter_iters" => f.iters,
                    "z_smoother" => sm.z_now, "sd_smoother" => sm.sd_now, "err_sd_smoother" => (sm.z_now .- truth_at(zs, sm.t)) ./ sm.sd_now,
                    "smoother_block" => length(sm.z), "smoother_converged" => sm.converged, "smoother_iters" => sm.iters, "F_smoother" => sm.F) for (kk, (b, f, sm)) in enumerate(zip(br, fr, sr))]
    push!(cooks, Dict("seed" => seed, "z_by_hour" => zs, "batches" => batches, "wall_batch_s" => tb, "wall_filter_s" => tf, "wall_smoother_s" => ts))
    eb = abs.(batches[end]["err_sd"]); ef = abs.(batches[end]["err_sd_filter"]); es = abs.(batches[end]["err_sd_smoother"])
    @printf("cook %d (seed %d): at %.0f h  batch |err|/sd max %.2f (%s) median %.2f | filter max %.2f (%s) median %.2f | smoother max %.2f (%s) median %.2f; batch %.0f s, filter %.0f s, smoother %.0f s
",
            k, seed, br[end].t / 3600, maximum(eb), allnames[argmax(eb)], sort(eb)[div(end + 1, 2)], maximum(ef), allnames[argmax(ef)], sort(ef)[div(end + 1, 2)],
            maximum(es), allnames[argmax(es)], sort(es)[div(end + 1, 2)], tb, tf, ts)
    for nm in ("u_leak", "k_e", "h_c", "r_max")
        i = idx[nm]
        @printf("    %-7s truth %s | batch %s | filter %s | smoother %s
", nm, join([@sprintf("%+.2f", truth_at(zs, b.t)[i]) for b in br], " "),
                join([@sprintf("%+.2f", b["err_sd"][i]) for b in batches], " "), join([@sprintf("%+.2f", b["err_sd_filter"][i]) for b in batches], " "),
                join([@sprintf("%+.2f", b["err_sd_smoother"][i]) for b in batches], " "))
    end
    flush(stdout)
    save()
end
println("wrote ", dst, " (", length(cooks), " cooks, ", length(skipped), " skipped)")
