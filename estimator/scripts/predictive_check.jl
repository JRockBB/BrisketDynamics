# Does the estimator's latent error cost the planner? From a banked stream_bench record, at hourly checkpoints
# through the cook: start from the plant's TRUE state, roll the model forward 6 h under (a) the estimator's latents
# at that moment, (b) the prior's, (c) the true latents (the harness's floor: model-vs-plant integration only), and
# (d) draws from the estimator's reported latent uncertainty (diagonal: records keep sds, not the joint). Score the
# core temperature at 1, 2, 4, 6 h and the time the core first crosses the planner's thresholds. State-estimation
# error is excluded by construction (the rollouts start from the truth); this isolates the LATENTS' cost.
#
# Run: julia --threads=4 --project=estimator estimator/scripts/predictive_check.jl <out.json> <record.json>...
#      (records from the lighting fixture, matched noise: the plant is regenerated from each record's seed)
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
out_path = ARGS[1]; record_files = ARGS[2:end]
# the bench's fixture, plant and truth decoder without running a cook (zero cooks, the full 18 h horizon)
ENV["STREAM_START"] = "lighting"
empty!(ARGS); append!(ARGS, ["0", "1", "predictive_check_fixture.json", "18.0"])
include(joinpath(@__DIR__, "stream_bench.jl"))
rm(joinpath(REPO, "data", "mock", "predictive_check_fixture.json"); force=true)
using Statistics

const HORIZONS_H = [1.0, 2.0, 4.0, 6.0]
const THRESHOLDS_C = [65.0, 75.0, 85.0]          # the stall region, the wrap's neighbourhood, the pull's approach
const CHECKPOINTS_H = collect(2.0:1.0:16.0)
const N_DRAWS = 12
const DT_ROLL = 120.0
model = SystemModel(sys, CookLog(t, u; events=cook_events, sensors), blk; sn, abstol=BENCH_ABSTOL, reltol=BENCH_RELTOL,
                    floor_abstol=BENCH_FLOOR_ABSTOL, floor_reltol=BENCH_FLOOR_RELTOL)
z_prior = [z_moments(s)[1] for s in sites]

"The first time a series crosses `thr` upward (linear between samples), or NaN."
function crossing(ts, vs, thr)
    for i in 2:length(vs)
        vs[i - 1] < thr <= vs[i] && return ts[i - 1] + (thr - vs[i - 1]) / (vs[i] - vs[i - 1]) * (ts[i] - ts[i - 1])
    end
    return NaN
end

results = Any[]
for f in record_files
    d = parse_json_file(f)
    # completed cooks, and an in-progress one (its snapshots so far), so a long run can be checked early
    for c in vcat(d["cooks"], get(d, "active_cook", nothing) === nothing ? Any[] : Any[d["active_cook"]])
        seed = Int(c["seed"])
        rng = MersenneTwister(2000 + seed)
        z0 = draw_stream_truth(rng)
        dd0 = DriftDraw(rng, idx; sites)
        dd = DriftDraw(Dict(k => zero(v) for (k, v) in dd0.inc), Dict(k => 0.0 for (k, v) in dd0.jump))
        seg_edges, blocks = drift_segments(t, z0, dd, events.t_add)
        xs = [from_z(z, sites) for z in blocks]
        Am = xs[1][AREA]
        dense = collect(0.0:30.0:t_end)
        o = plant(xs, seg_edges, dense, Am; initial_ics=draw_initial_ics(MersenneTwister(19000 + seed)))
        Tcore = o.Tcore .- 273.15
        snaps = get(c, "snapshots", nothing)
        if snaps !== nothing && !isempty(snaps)
            # The estimator's FULL belief: roll forward from its own state, and draw jointly from its covariance.
            @printf("seed %d (%s): %d full-belief snapshots\n", seed, basename(f), length(snaps)); flush(stdout)
            nz = length(sites)
            for s in snaps
                t0 = Float64(s["t_belief_s"]); t0 / 3600 < 2.0 && continue
                times = collect((t0 + DT_ROLL):DT_ROLL:min(t0 + 6 * 3600, t_end))
                length(times) < 2 && continue
                zy = vcat(Float64.(s["z"]), Float64.(s["y"]))
                C = E._stable_cov(reduce(vcat, permutedims.([Float64.(row) for row in s["zy_cov"]])))
                L = cholesky(Symmetric(C)).L
                k = argmin(abs.(o.t .- t0)); _, y_true = truth_belief(o, k)
                ztrue = truth_at(z0, dd, o.t[k], events.t_add)
                rollfrom(y, z, ts0) = try
                    S = E.propagate_states(model, y, z, ts0, times; first_slice=false)
                    [E.belief_decode(S[:, m], bl)[ic] - 273.15 for m in axes(S, 2)]
                catch err
                    err isa InterruptException && rethrow(); fill(NaN, length(times))
                end
                draws = [zy .+ L * randn(MersenneTwister(97 * seed + round(Int, t0 / 60) + q), length(zy)) for q in 1:N_DRAWS]
                cases = vcat([(zy[(nz + 1):end], zy[1:nz], t0), (y_true, zy[1:nz], o.t[k]), (y_true, ztrue, o.t[k])],
                             [(v[(nz + 1):end], v[1:nz], t0) for v in draws])
                runs = Vector{Vector{Float64}}(undef, length(cases))
                Threads.@threads for q in eachindex(cases); runs[q] = rollfrom(cases[q]...); end
                truth = [Tcore[argmin(abs.(o.t .- τ))] for τ in times]
                ens = reduce(hcat, runs[4:end])
                row = Dict{String,Any}("seed" => seed, "t_h" => t0 / 3600, "record" => basename(f), "mode" => "full_belief",
                                       "core_now_true" => Tcore[k], "core_now_est" => E.belief_decode(zy[(nz + 1):end], bl)[ic] - 273.15)
                for h in HORIZONS_H
                    m = findfirst(>=(t0 + h * 3600 - 1e-6), times); m === nothing && continue
                    v = filter(isfinite, ens[m, :]); length(v) < 3 && continue
                    lo, hi = quantile(v, [0.025, 0.975])
                    row["h$(Int(h))"] = Dict("truth" => truth[m], "est" => runs[1][m], "truestate_estlat" => runs[2][m],
                                              "truelat" => runs[3][m], "ens_lo" => lo, "ens_hi" => hi, "ens_sd" => std(v),
                                              "covered" => lo <= truth[m] <= hi)
                end
                for thr in THRESHOLDS_C
                    tt = crossing(times, truth, thr); isfinite(tt) || continue
                    row["cross$(Int(thr))"] = Dict("truth_min" => (tt - t0) / 60,
                        "est_err_min" => (crossing(times, runs[1], thr) - tt) / 60,
                        "truestate_estlat_err_min" => (crossing(times, runs[2], thr) - tt) / 60)
                end
                push!(results, row)
                e = get(row, "h4", nothing)
                e === nothing || @printf("  t %4.1f h  core now %5.1f C (est %+5.2f)  4 h ahead: truth %5.1f, full belief %+5.2f, true state + est latents %+5.2f K; joint 95%% band +-%4.2f %s\n",
                                         t0 / 3600, Tcore[k], row["core_now_est"] - Tcore[k], e["truth"], e["est"] - e["truth"],
                                         e["truestate_estlat"] - e["truth"], (e["ens_hi"] - e["ens_lo"]) / 2, e["covered"] ? "covers" : "MISSES")
                flush(stdout)
            end
            continue
        end
        rows = [r for r in c["temps"] if r["committed"]]
        tb = [get(r, "t_belief_h", r["t_h"]) for r in rows]
        @printf("seed %d (%s): %d committed beliefs\n", seed, basename(f), length(rows)); flush(stdout)
        for tc in CHECKPOINTS_H
            j = findlast(<=(tc), tb); j === nothing && continue
            r = rows[j]; tk = tb[j] * 3600
            k = argmin(abs.(o.t .- tk)); t0 = o.t[k]
            _, y0 = truth_belief(o, k)
            times = collect((t0 + DT_ROLL):DT_ROLL:min(t0 + 6 * 3600, t_end))
            length(times) < 2 && continue
            ztrue = truth_at(z0, dd, t0, events.t_add)
            zhat = Float64.(r["z"]); zsd = Float64.(r["z_sd"])
            roll(z) = try
                S = E.propagate_states(model, y0, z, t0, times; first_slice=false)
                [E.belief_decode(S[:, m], bl)[ic] - 273.15 for m in axes(S, 2)]
            catch err
                err isa InterruptException && rethrow(); fill(NaN, length(times))
            end
            draws = [zhat .+ zsd .* randn(MersenneTwister(31 * seed + round(Int, 10tc) + q), length(zhat)) for q in 1:N_DRAWS]
            runs = Vector{Vector{Float64}}(undef, 3 + N_DRAWS)
            zs = vcat([zhat, z_prior, ztrue], draws)
            Threads.@threads for q in eachindex(zs); runs[q] = roll(zs[q]); end
            truth = [Tcore[argmin(abs.(o.t .- τ))] for τ in times]
            ens = reduce(hcat, runs[4:end])
            row = Dict{String,Any}("seed" => seed, "t_h" => t0 / 3600, "record" => basename(f),
                                   "core_now_true" => Tcore[k], "core_now_est" => r["centre"][1] - 273.15)
            for h in HORIZONS_H
                m = findfirst(>=(t0 + h * 3600 - 1e-6), times); m === nothing && continue
                lo, hi = quantile(filter(isfinite, ens[m, :]), [0.025, 0.975])
                row["h$(Int(h))"] = Dict("truth" => truth[m], "est" => runs[1][m], "prior" => runs[2][m], "truelat" => runs[3][m],
                                          "ens_lo" => lo, "ens_hi" => hi, "covered" => lo <= truth[m] <= hi)
            end
            for thr in THRESHOLDS_C
                tt = crossing(times, truth, thr); isfinite(tt) || continue
                row["cross$(Int(thr))"] = Dict("truth_min" => (tt - t0) / 60,
                    "est_err_min" => (crossing(times, runs[1], thr) - tt) / 60,
                    "prior_err_min" => (crossing(times, runs[2], thr) - tt) / 60,
                    "truelat_err_min" => (crossing(times, runs[3], thr) - tt) / 60)
            end
            push!(results, row)
            e = get(row, "h4", nothing)
            e === nothing || @printf("  t %4.1f h  core %5.1f C  4 h ahead: truth %5.1f, est %+5.1f, prior %+5.1f, true-latent %+5.2f K; 95%% band %s\n",
                                     t0 / 3600, Tcore[k], e["truth"], e["est"] - e["truth"], e["prior"] - e["truth"], e["truelat"] - e["truth"],
                                     e["covered"] ? "covers" : "MISSES")
            flush(stdout)
        end
    end
end
open(out_path, "w") do io; write(io, json_document(Dict("results" => results, "horizons_h" => HORIZONS_H,
    "thresholds_c" => THRESHOLDS_C, "n_draws" => N_DRAWS,
    "note" => "rollouts start from the plant's true state; estimator draws are diagonal in the latents"))); end

# the summary, per mode: forecast error by horizon and source, coverage, time-to-threshold error
function summarize(results, label, sources)
    isempty(results) && return
    println("\n== $label: forecast error of the core temperature (K), mean |err| (bias) by horizon")
    for h in HORIZONS_H
        key = "h$(Int(h))"; rs = [r[key] for r in results if haskey(r, key)]
        isempty(rs) && continue
        s(f) = (v = [x[f] - x["truth"] for x in rs if isfinite(x[f])]; @sprintf("%5.2f (%+5.2f)", mean(abs.(v)), mean(v)))
        extra = haskey(rs[1], "ens_sd") ? @sprintf("; band half-width mean %.2f K", mean((x["ens_hi"] - x["ens_lo"]) / 2 for x in rs)) : ""
        @printf("  %d h  n %3d  %s | 95%% band covers %d%%%s\n", Int(h), length(rs),
                join(["$(nm) $(s(f))" for (nm, f) in sources], "  "), round(Int, 100 * mean(x["covered"] for x in rs)), extra)
    end
    println("  time-to-threshold error (min): median |err| (median signed) [found/crossings]")
    for thr in THRESHOLDS_C
        key = "cross$(Int(thr))"; rs = [r[key] for r in results if haskey(r, key)]
        isempty(rs) && continue
        s(f) = (v = filter(isfinite, [x[f] for x in rs if haskey(x, f)]); isempty(v) ? "  n/a" : @sprintf("%5.1f (%+5.1f) [%d/%d]", median(abs.(v)), median(v), length(v), length(rs)))
        @printf("  %2d C  n %2d  %s\n", Int(thr), length(rs), join(["$(nm) $(s(f * "_err_min"))" for (nm, f) in sources if f != "truth"], "  "))
    end
end
summarize([r for r in results if get(r, "mode", "") == "full_belief"], "from the estimator's full belief",
          [("full belief", "est"), ("true state + est latents", "truestate_estlat"), ("true latents", "truelat")])
summarize([r for r in results if get(r, "mode", "") != "full_belief"], "from the true state (latents only)",
          [("estimator", "est"), ("prior", "prior"), ("true latents", "truelat")])
println("wrote $out_path")
