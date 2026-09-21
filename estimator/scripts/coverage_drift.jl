# The coverage test on a DRIFTING plant: truths drawn from the pilot's prior, then walked through the cook
# (drift.jl: the leak and the surface coefficients on random walks stepping on the hour, a new charge at the
# add), each cook read by the real-cook instruments with its own noise draw, and streamed two ways as the bench's
# baselines: the batch estimator (every latent a constant on the full history) and the smoother (smoother.jl: the
# moving sites piecewise constant over the full history, a random-walk prior between the pieces, the jump at the
# add). The filter on the graph, the estimator the pilot flies, is judged on the same cooks by coverage_filter.jl.
# The standardized errors are against the truth in force at that hour. A calibrated belief has them at mean zero
# and spread one; the batch estimator on a moving parameter does not.
#
# Run:  julia --project=estimator estimator/scripts/coverage_drift.jl [prior.json] [n_cooks] [seed0] [out.json] [edges_to_h]
# Writes data/mock/<out.json> (coverage_drift_<seed0>.json by default), after every cook; plot_coverage.py
# reads it with the keys err_sd (batch) and err_sd_smoother.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
prior_file = length(ARGS) >= 1 ? ARGS[1] : "prior_next_cook.json"
n_cooks = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
seed0 = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
out_name = length(ARGS) >= 4 ? ARGS[4] : "coverage_drift_$(seed0).json"
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
blk = latent_block(joinpath(REPO, "data", "calibration", prior_file), cfg)
sites, allnames, prior_phys, names = blk.sites, blk.names, blk.prior_phys, blk.phys_names
AREA, DEPTH = blk.area, blk.depth
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
edges_to_h = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : floor(events.t_pull / 3600)
edges = 3600.0 .* collect(2:2:floor(Int, edges_to_h))
nc = RealCookNoise()
np = realcook_noise_priors(joinpath(REPO, "data", "calibration", "noise_prior_real.json"), dt)
idx = blk.idx
# the drift the plant walks with (drift.jl), and the process noise the smoother is told (the same: the test is of
# the estimators' mechanism, not of guessing the rates)
q = zeros(length(sites)); for (nm, sd) in DRIFT_SD_PER_HOUR; q[idx[nm]] = sd^2; end
moving = smoother_moving(idx, events.t_add; sites)

cooks = Any[]; skipped = Int[]
dst = joinpath(REPO, "data", "mock", out_name)
save() = open(dst, "w") do io
    JSON.print(io, Dict("plant" => "monolith (1-D) in hourly segments, parameters drifting; real-cook instruments", "names" => allnames,
                        "prior_file" => prior_file, "edges_h" => edges ./ 3600, "cooks" => cooks, "skipped_seeds" => skipped,
                        "drift" => Dict("sd_per_hour" => DRIFT_SD_PER_HOUR, "jump_log_sd" => JUMP_LOG_SD), "q" => q))
end
println("prior ", prior_file, "; ", n_cooks, " cooks from seed ", seed0, "; batches to ", edges_to_h, " h; drift ", DRIFT_SD_PER_HOUR, "; jump ", JUMP_LOG_SD, "; record ", out_name)
flush(stdout)
for k in 1:n_cooks
    seed = seed0 + k - 1
    rng = MersenneTwister(2000 + seed)
    z0 = draw_truth(rng, sites; depth=DEPTH)
    dd = DriftDraw(rng, idx; sites)
    seg_edges, blocks = drift_segments(t, z0, dd, events.t_add)
    xs = [from_z(z, sites) for z in blocks]
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
    chs = realcook_channels(readings, masks, np; cap=nc.hum_cap, depth_latent=DEPTH)
    d = estimator_setup(t, u, pit, cfg, prior_phys, names)
    br = try
        stream_fit(chs, sites, d, edges; physics=Physics1D(area_latent=AREA, events=events), noise=np, noise_rounds=1, levelled=true, verbose=false)
    catch err
        err isa InterruptException && rethrow()
        @printf("cook %d (seed %d): the batch estimator failed (%s); skipped\n", k, seed, sprint(showerror, err)[1:min(end, 80)]); push!(skipped, seed); save(); continue
    end
    tb = time() - t0; t2 = time()
    sr = try
        smoother_fit(chs, sites, d, edges; moving, area_latent=AREA, events=events, noise=np, noise_rounds=1, levelled=true, verbose=false)
    catch err
        err isa InterruptException && rethrow()
        @printf("cook %d (seed %d): the smoother failed (%s); skipped
", k, seed, sprint(showerror, err)[1:min(end, 80)]); push!(skipped, seed); save(); continue
    end
    ts = time() - t2
    batches = [Dict("t_h" => b.t / 3600, "z" => b.z, "sd" => b.sd, "err_sd" => (b.z .- truth_at(z0, dd, b.t, events.t_add)) ./ b.sd, "F" => b.F,
                    "misfit_power" => b.misfit.total.ss / b.misfit.total.n,
                    "z_true" => truth_at(z0, dd, b.t, events.t_add),
                    "z_smoother" => sm.z_now, "sd_smoother" => sm.sd_now, "err_sd_smoother" => (sm.z_now .- truth_at(z0, dd, sm.t, events.t_add)) ./ sm.sd_now,
                    "smoother_block" => length(sm.z), "smoother_converged" => sm.converged, "smoother_iters" => sm.iters, "F_smoother" => sm.F) for (b, sm) in zip(br, sr)]
    push!(cooks, Dict("seed" => seed, "z_by_hour" => [truth_at(z0, dd, h * 3600.0, events.t_add; from=true) for h in 0:29], "segment_edges_h" => t[seg_edges] ./ 3600, "batches" => batches, "wall_batch_s" => tb, "wall_smoother_s" => ts))
    eb = abs.(batches[end]["err_sd"]); es = abs.(batches[end]["err_sd_smoother"])
    @printf("cook %d (seed %d): at %.0f h  batch |err|/sd max %.2f (%s) median %.2f | smoother max %.2f (%s) median %.2f; batch %.0f s, smoother %.0f s\n",
            k, seed, br[end].t / 3600, maximum(eb), allnames[argmax(eb)], sort(eb)[div(end + 1, 2)],
            maximum(es), allnames[argmax(es)], sort(es)[div(end + 1, 2)], tb, ts)
    for nm in ("u_leak", "k_e", "h_c", "r_max")
        i = idx[nm]
        @printf("    %-7s truth %s | batch %s | smoother %s\n", nm, join([@sprintf("%+.2f", truth_at(z0, dd, b.t, events.t_add)[i]) for b in br], " "),
                join([@sprintf("%+.2f", b["err_sd"][i]) for b in batches], " "), join([@sprintf("%+.2f", b["err_sd_smoother"][i]) for b in batches], " "))
    end
    flush(stdout)
    save()
end
println("wrote ", dst, " (", length(cooks), " cooks, ", length(skipped), " skipped)")
