# The factor-graph backend's calibration bench, end to end and in Julia: the state's process-noise rates and the
# sensors' noise identified from the smoother's own innovations on the bench's cooks, checked on cooks it has never
# seen, written to data/calibration/state_noise_system.json with the backend's fingerprint. The RxInfer filter's bench
# (estimator/python/bench_state_noise.py) is the frozen oracle of this protocol; nothing here touches it or its file.
#
#   1. the tuning round: coverage_system.jl on its seeds with the calibration as it stands (the file's, or the code's
#      defaults and the matched sensor noise when there is none); each cook's smoother carries the accumulator of the
#      innovation autocovariances' sensitivity to every group's variance (the state's twelve groups, every discrepancy
#      sensor's OU variance, every sensor's white variance; online.jl `ALSFeed`, als.jl), and the belief is scored against
#      the plant's state group by group;
#   2. one autocovariance least-squares step (bench_fit.jl: Odelson, Rajamani, Rawlings 2006 in the fixed-gain
#      time-varying form; the products of the standardized innovations at every pair of arrivals inside the lag window,
#      less the smoother's own white model, regressed on the change of the variances, pooled over the cooks, weighted per
#      channel pair and kind by the residual variance, bounded by the cap, a group the innovations cannot see kept and
#      reported, the initial widths kept; a leave-one-cook-out jackknife beside the model's standard errors);
#   3. the fresh round: the bench again on seeds it has never seen with the identified calibration; the numbers quoted are
#      this round's alone, its consistency and coverage figures rendered from its manifest. Anything out of band is
#      reported, not tuned away.
#
# Each round's records are named in a manifest (data/mock/bench_manifest_system_<round><tag>.json: the records, their seed
# blocks, the cook count, the backend fingerprint and the state noise they ran with) that is validated before pooling;
# the cook count is whole blocks (workers × cooks per worker); the fresh seeds follow the tuning seeds unless given and
# may not overlap them. The figures are drawn by the existing plot scripts (estimator/python/plot_filter_consistency.py,
# plot_coverage.py), which read the manifest and render the record schema; they are renderers, not the bench.
#
# Run:  julia --project=estimator estimator/scripts/bench_system.jl [--cooks 10] [--workers 2] [--hours 14] [--prior prior_next_cook.json]
#           [--tune-seed0 1] [--fresh-seed0 <after the tuning seeds>] [--cap 4] [--lag-h 2] [--lag 0] [--drift 0|1] [--noise matched|prior]
#           [--threads 4] [--start file|default] [--rounds tune,fresh] [--smoke] [--reuse] [--variant NAME] [--no-figures]
#   --lag: the smoother's window (0, the filter's, by default: the RxInfer bench's like-for-like and the cheapest; the pilot flies 3)
#   --threads: Julia threads per worker, for the dynamics Jacobians' chunks (the same numbers on any count)
#   --smoke: one worker, one cook, at most two hours, into scratch names (the driver's own check; nothing quoted from it)
#   --rounds fresh: the fresh round only, with the file's calibration as it is (the check without the correction)
#   --reuse: aggregate a round's records when they all exist and are the round's instead of running its workers
#   --variant NAME: a named variant (its own records, manifests and figures; the calibration file untouched)
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Printf, Statistics
using SpecialFunctions: gamma_inc_inv, beta_inc_inv
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
MOCK = joinpath(REPO, "data", "mock")
# the records name their own noise groups (the filter's twelve for one piece, `piece.field` beside the pit's for two), their
# channels and their sensors; the one-piece worker's channels are the RxInfer bench's four
CHANNELS_DEFAULT = Dict(1 => "pit probe", 2 => "core probe", 3 => "humidity", 4 => "scale")
LAG_BINS = [0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60]
WORKERS = Dict("one" => (script="coverage_system.jl", stem_tune="coverage_system_tune", stem_fresh="coverage_system", smoke_tune="bench_system_smoke_tune",
                         smoke_fresh="bench_system_smoke_fresh", tag="_system", cal="state_noise_system.json"),
               "multi" => (script="coverage_system_multi.jl", stem_tune="coverage_system_multi_tune", stem_fresh="coverage_system_multi", smoke_tune="bench_system_multi_smoke_tune",
                           smoke_fresh="bench_system_multi_smoke_fresh", tag="_system_multi", cal="state_noise_system_multi.json"),
               "rom" => (script="coverage_system_rom.jl", stem_tune="coverage_system_rom_tune", stem_fresh="coverage_system_rom", smoke_tune="bench_system_rom_smoke_tune",
                         smoke_fresh="bench_system_rom_smoke_fresh", tag="_system_rom", cal="state_noise_system_rom.json"))

# --- the arguments --------------------------------------------------------------------------------------------------
function parse_args(args)
    a = Dict{String,Any}("cooks" => 10, "workers" => 2, "hours" => 14.0, "prior" => "prior_next_cook.json", "tune-seed0" => 1, "fresh-seed0" => nothing,
                         "cap" => 4.0, "lag-h" => 2.0, "lag" => 0, "drift" => 0, "noise" => "matched", "start" => "file", "rounds" => "tune,fresh",
                         "threads" => 4, "worker" => "one", "nx" => 20, "order" => 0, "drop" => "none", "rom" => "rom_halfbrick_flat.json",
                         "smoke" => false, "reuse" => false, "variant" => "", "no-figures" => false)
    i = 1
    while i <= length(args)
        k = args[i]
        startswith(k, "--") || error("unexpected argument $k")
        k = k[3:end]
        if k in ("smoke", "reuse", "no-figures")
            a[k] = true; i += 1; continue
        end
        haskey(a, k) || error("unknown option --$k")
        i + 1 <= length(args) || error("--$k needs a value")
        v = args[i + 1]
        a[k] = k in ("cooks", "workers", "tune-seed0", "fresh-seed0", "lag", "drift", "threads", "nx", "order") ? parse(Int, v) :
               k in ("hours", "cap", "lag-h") ? parse(Float64, v) : v
        i += 2
    end
    a["noise"] in ("matched", "prior") || error("--noise must be matched or prior")
    a["start"] in ("file", "default") || error("--start must be file or default")
    haskey(WORKERS, a["worker"]) || error("--worker must be one (the slab on the RxInfer bench's plant), multi (two pieces on a CookSystem plant) or rom (the artifact's slab on its brick)")
    return a
end

# --- the bands ------------------------------------------------------------------------------------------------------
"The quantile of a chi-squared distribution with `k` degrees of freedom."
chi2_quantile(p::Real, k::Real) = 2 * gamma_inc_inv(k / 2, p, 1 - p)
"The two-sided 95 % quantile of Student's t with `ν` degrees of freedom."
function t_quantile_95(ν::Real)
    x = beta_inc_inv(ν / 2, 0.5, 0.05, 0.95)[1]
    return sqrt(ν * (1 - x) / x)
end
"The 95 % band of a mean of chi-squared(dof)/dof variables over `n_eff` independent samples (one per cook-hour: indicative, it orders the groups)."
function band(n_eff::Real, dof::Real)
    k = max(round(Int, n_eff * dof), 1)
    return chi2_quantile(0.025, k) / k, chi2_quantile(0.975, k) / k
end
"The 95 % interval of the mean over cooks from the cooks' own spread (a t interval on the per-cook values)."
function cluster_interval(per_cook)
    v = [x for x in per_cook if isfinite(x)]
    length(v) < 3 && return NaN, NaN
    half = t_quantile_95(length(v) - 1) * std(v) / sqrt(length(v))
    return mean(v) - half, mean(v) + half
end

# --- the manifest ---------------------------------------------------------------------------------------------------
function record_summary(path)
    d = parse_json_file(path)
    return (name=basename(path), seeds=[Int(c["seed"]) for c in d["cooks"]], n_cooks=length(d["cooks"]), physics=get(d, "physics", nothing),
            state_noise=get(d, "state_noise", nothing), skipped=Int.(get(d, "skipped_seeds", [])), backend=get(d, "backend", nothing))
end
function same_noise(a, b)
    (a === nothing || b === nothing) && return a === b
    Set(keys(a)) == Set(keys(b)) || return false
    for (k, x) in a
        y = b[k]
        if x isa Number && y isa Number
            isapprox(Float64(x), Float64(y); rtol=1e-9, atol=0.0) || return false
        else
            x == y || return false
        end
    end
    return true
end
"The records against the round's seed blocks: each carries exactly its block's seeds (a skipped seed allowed when it says so), one backend fingerprint, one state noise, the expected ones when given."
function validate_records(files, blocks; physics=nothing, state_noise=nothing)
    length(files) == length(blocks) || error("$(length(files)) records for $(length(blocks)) seed blocks")
    ss = [record_summary(f) for f in files]
    for (s, block) in zip(ss, blocks)
        expected = [x for x in block if !(x in s.skipped)]
        s.seeds == expected || error("$(s.name) carries seeds $(s.seeds), the round's block is $block (skipped $(s.skipped))")
    end
    all(s.backend == "system" for s in ss) || error("a record is not the factor-graph backend's")
    phys = unique([s.physics for s in ss])
    length(phys) == 1 || error("the records ran on $(length(phys)) different backends: $(phys)")
    physics === nothing || ss[1].physics == physics || error("the records ran on backend $(ss[1].physics), expected $physics")
    for s in ss[2:end]
        same_noise(s.state_noise, ss[1].state_noise) || error("$(s.name) ran on other state noise than $(ss[1].name)")
    end
    state_noise === nothing || same_noise(ss[1].state_noise, state_noise) || error("the records ran on other state noise than expected ($(ss[1].name))")
    seeds = sort(reduce(vcat, [s.seeds for s in ss]))
    length(unique(seeds)) == length(seeds) || error("a seed appears twice across the records: $seeds")
    return Dict{String,Any}("seeds" => seeds, "n_cooks" => length(seeds), "physics" => ss[1].physics, "state_noise" => ss[1].state_noise,
                            "records" => [s.name for s in ss], "blocks" => blocks, "skipped" => sort(reduce(vcat, [s.skipped for s in ss]; init=Int[])),
                            "backend" => "system")
end
function write_manifest(path, round_name, files, blocks; physics=nothing, state_noise=nothing, extra=Dict{String,Any}())
    m = validate_records(files, blocks; physics, state_noise)
    m["round"] = round_name; m["date"] = E._today_string()
    merge!(m, extra)
    open(path, "w") do io; write(io, json_document(m; pretty=true)); end
    return m
end

# --- the workers ----------------------------------------------------------------------------------------------------
"The workers in parallel, one Julia process per seed block (`--threads` Julia threads each for the dynamics Jacobians' chunks, one BLAS thread), their logs in `log_dir`."
function run_workers(label, seed0s, n_per, a, cal_file, out_names, log_dir; stale_ok::Bool=false)
    procs = []
    julia = Base.julia_cmd().exec[1]
    w = WORKERS[a["worker"]]
    for (s, out) in zip(seed0s, out_names)
        script = joinpath(REPO, "estimator", "scripts", w.script)
        cmd = a["worker"] == "one" ?
              `$julia --startup-file=no -t $(a["threads"]) --project=$(joinpath(REPO, "estimator")) $script $(a["prior"]) $n_per $s $out $(a["hours"]) $(a["drift"]) $cal_file $(a["lag"]) $(stale_ok ? 1 : 0) $(a["lag-h"]) $(a["noise"])` :
              a["worker"] == "multi" ?
              `$julia --startup-file=no -t $(a["threads"]) --project=$(joinpath(REPO, "estimator")) $script $(a["prior"]) $n_per $s $out $(a["hours"]) $(a["nx"]) $cal_file $(a["lag"]) $(stale_ok ? 1 : 0) $(a["lag-h"]) $(a["order"]) $(a["drop"])` :
              `$julia --startup-file=no -t $(a["threads"]) --project=$(joinpath(REPO, "estimator")) $script $(a["prior"]) $n_per $s $out $(a["hours"]) $(a["rom"]) $cal_file $(a["lag"]) $(stale_ok ? 1 : 0) $(a["lag-h"])`
        log = joinpath(log_dir, "$(label)_$(s).log")
        println("  $label: seeds $s..$(s + n_per - 1) -> data/mock/$out  ($(join(cmd.exec, " ")))")
        io = open(log, "w")
        p = run(pipeline(setenv(cmd, "OPENBLAS_NUM_THREADS" => (a["threads"] > 1 ? "1" : "2"); dir=REPO); stdout=io, stderr=io); wait=false)
        push!(procs, (s, out, p, io))
    end
    flush(stdout)
    t0 = time()
    failed = Tuple{Int,Int}[]
    for (s, out, p, io) in procs
        wait(p); close(io)
        success(p) || push!(failed, (s, p.exitcode))
    end
    @printf("  %s: done in %.0f min\n", label, (time() - t0) / 60)
    isempty(failed) || error("$label: workers failed (seed0, code): $failed; see $log_dir")
    return [joinpath(MOCK, out) for out in out_names]
end

# --- the round's numbers --------------------------------------------------------------------------------------------
nanmean(v) = (w = [x for x in v if isfinite(x)]; isempty(w) ? NaN : mean(w))
nanmedian(v) = (w = [x for x in v if isfinite(x)]; isempty(w) ? NaN : median(w))
"The pooled consistency numbers of a round: NEES/dof per state group, of the latents and the temperatures, NIS/dof, the whiteness per channel, the solves' convergence and cost, the rates the round ran with, the accumulators."
function load_round(files)
    cooks = Any[]; sn = nothing; fp = nothing; gd = nothing; sensors_used = nothing; seeds = Int[]; skipped = Int[]; lag = nothing; GROUPS = String[]; channels = CHANNELS_DEFAULT
    for f in files
        d = parse_json_file(f)
        append!(cooks, d["cooks"])
        sn = d["state_noise"]; fp = d["physics"]; gd = d["group_dof"]; sensors_used = d["sensors"]; lag = d["lag"]; GROUPS = String.(d["groups"])
        haskey(d, "channels") && (channels = Dict(parse(Int, k) => String(v) for (k, v) in d["channels"]))
        append!(seeds, [Int(c["seed"]) for c in d["cooks"]]); append!(skipped, Int.(get(d, "skipped_seeds", [])))
    end
    isempty(cooks) && error("the round produced no cooks")
    temps = [r for c in cooks for r in c["temps"]]
    dof_T = haskey(temps[1], "dof_T") ? Int(temps[1]["dof_T"]) : 3
    out = Dict{String,Any}("n_cooks" => length(cooks), "seeds" => seeds, "skipped" => skipped, "n_updates" => length(temps), "physics" => fp, "group_dof" => gd, "lag" => lag,
                           "group_names" => GROUPS, "channels" => channels, "state_noise" => sn, "dof_T" => dof_T,
                           "rates" => Dict(g => sn[g] for g in GROUPS), "init_sd" => Dict(g => sn[g * "0"] for g in GROUPS), "sensors" => sensors_used,
                           "fields" => get(sn, "fields", "nodes"), "node_jitter" => get(sn, "node_jitter", 0.0), "field_modes" => get(sn, "field_modes", 0),
                           "events" => Dict(k => sn[k] for k in ("T_event", "T_spritz", "m_add") if haskey(sn, k)), "groups" => Dict{String,Any}())
    for g in GROUPS
        v = [Float64(r["nees_g"][g]) for r in temps if r["nees_g"][g] !== nothing]
        v = v[isfinite.(v)]
        per_cook = [nanmean([Float64(r["nees_g"][g]) for r in c["temps"] if r["nees_g"][g] !== nothing]) for c in cooks]
        out["groups"][g] = Dict("nees_dof" => isempty(v) ? NaN : mean(v), "n" => length(v), "cook_median" => nanmedian(per_cook), "cook_max" => maximum(per_cook))
    end
    fnum(x) = x === nothing ? NaN : Float64(x)
    n16 = length(first(cooks)["batches"][1]["z"]) - 1                                   # the latents less a0
    out["dof_z"] = n16
    out["latents_per_cook"] = [nanmean([fnum(r["nees_z16"]) for r in c["temps"]]) / n16 for c in cooks]
    out["temps_per_cook"] = [nanmean([fnum(r["nees_T"]) for r in c["temps"]]) / dof_T for c in cooks]
    out["latents_nees16"] = nanmean([fnum(r["nees_z16"]) for r in temps]) / n16
    out["temps_nees3"] = nanmean([fnum(r["nees_T"]) for r in temps]) / dof_T
    nis = [fnum(r["nis"]) for r in temps if r["dof"] > 0]; dof = [Float64(r["dof"]) for r in temps if r["dof"] > 0]
    keep = isfinite.(nis)
    out["nis_dof"] = isempty(dof) ? NaN : sum(nis[keep]) / sum(dof[keep])
    grads = [fnum(r["gradient"]) for r in temps]; grads = grads[isfinite.(grads)]
    out["gradient"] = Dict("n" => length(grads), "max" => isempty(grads) ? nothing : maximum(grads), "median" => isempty(grads) ? nothing : median(grads))
    out["uncommitted"] = count(r -> !Bool(r["committed"]), temps)
    out["iters"] = Dict("mean" => mean([Float64(r["iters"]) for r in temps]), "max" => maximum([Int(r["iters"]) for r in temps]))
    walls = [Float64(r["wall_s"]) for r in temps]
    out["wall_s"] = Dict("mean" => mean(walls), "max" => maximum(walls), "cook_mean" => mean([Float64(c["wall_filter_s"]) for c in cooks]))
    white = Dict{String,Any}()
    for (ch, lab) in channels
        seqs = [Float64[v for r in c["temps"] for v in get(r["innov"], string(ch), [])] for c in cooks]
        seqs = [s for s in seqs if length(s) > 3]
        allv = isempty(seqs) ? Float64[] : reduce(vcat, seqs)
        r1 = [cor(s[1:end - 1], s[2:end]) for s in seqs]
        white[lab] = Dict("n" => length(allv), "mean" => isempty(allv) ? NaN : mean(allv), "sd" => length(allv) > 1 ? std(allv) : NaN,
                          "lag1" => isempty(r1) ? NaN : mean(r1), "band" => isempty(seqs) ? NaN : 2 / sqrt(mean([length(s) for s in seqs])))
    end
    out["whiteness"] = white
    # the graph's own reads: the whitened residual power per state group (the newest slice's dynamics residual, and the
    # window's), per sensor the observation and OU residual squares, the least-squares goodness of fit 2F/(rows − cols);
    # and the smoothed past's NEES where the records carry it (lag above zero)
    with_resid = [r for r in temps if get(r, "resid", nothing) !== nothing]
    if !isempty(with_resid)
        function pool_dyn(key)
            d = Dict{String,Any}()
            for g in GROUPS
                ss = 0.0; n = 0.0
                for r in with_resid
                    v = get(r["resid"][key], g, nothing); v === nothing && continue
                    ss += Float64(v[1]); n += Float64(v[2])
                end
                d[g] = n > 0 ? ss / n : NaN
            end
            return d
        end
        pool_sensor(key) = (d = Dict{String,Any}(); for r in with_resid, (id, v) in r["resid"][key]; a = get!(d, id, [0.0, 0]); a[1] += Float64(v); a[2] += 1; end;
                            Dict(id => a[1] / a[2] for (id, a) in d))
        gof = [2 * Float64(r["F"]) / max(Float64(r["resid"]["rows"]) - Float64(r["resid"]["cols"]), 1.0) for r in with_resid if haskey(r, "F") && r["F"] !== nothing]
        out["resid"] = Dict("dyn" => pool_dyn("dyn"), "dyn_window" => pool_dyn("dyn_window"), "obs" => pool_sensor("obs"), "ou" => pool_sensor("ou"),
                            "gof" => isempty(gof) ? NaN : mean(gof), "n" => length(with_resid))
    end
    with_past = [r["past"] for r in temps if get(r, "past", nothing) !== nothing]
    if !isempty(with_past)
        out["past"] = Dict("n" => length(with_past), "nees_T" => nanmean([fnum(p["nees_T"]) for p in with_past]) / dof_T,
                           "nees_z16" => nanmean([fnum(p["nees_z16"]) for p in with_past]) / n16,
                           "groups" => Dict(g => nanmean([fnum(p["nees_g"][g]) for p in with_past]) for g in GROUPS))
    end
    als = [c["als"] for c in cooks if c["als"] !== nothing]
    out["als_summaries"] = als
    out["als"] = isempty(als) ? nothing : als_pool(als)
    return out
end

function print_round(name, r, hours)
    gd = r["group_dof"]; n_eff = r["n_cooks"] * hours; GROUPS = r["group_names"]
    println("$name: $(r["n_cooks"]) cooks (seeds $(r["seeds"])), $(r["n_updates"]) arrivals at lag $(Int(r["lag"])); backend $(r["physics"]); the fields' noise by $(r["fields"])")
    lo16, hi16 = band(r["n_cooks"], r["dof_z"]); lo3, hi3 = band(r["n_cooks"], r["dof_T"])
    c16 = cluster_interval(r["latents_per_cook"]); c3 = cluster_interval(r["temps_per_cook"])
    @printf("  NIS/dof %.2f; NEES/%d of the latents %.2f (per cook: %s; the mean's interval from the cooks' spread %.2f to %.2f; a consistent filter's band for %d cooks %.2f to %.2f)\n",
            r["nis_dof"], r["dof_z"], r["latents_nees16"], join([@sprintf("%.2f", v) for v in r["latents_per_cook"]], " "), c16[1], c16[2], r["n_cooks"], lo16, hi16)
    @printf("  NEES/%d of the temperatures %.2f (per cook: %s; the mean's interval %.2f to %.2f; the band %.2f to %.2f)\n",
            r["dof_T"], r["temps_nees3"], join([@sprintf("%.2f", v) for v in r["temps_per_cook"]], " "), c3[1], c3[2], lo3, hi3)
    g = r["gradient"]
    g["n"] > 0 && @printf("  the solves: iterations mean %.1f max %d; the accepted point's scaled gradient max %.2e median %.2e; %d of %d arrivals not committed; %.1f s per arrival (max %.0f), %.0f s per cook\n",
                          r["iters"]["mean"], r["iters"]["max"], g["max"], g["median"], r["uncommitted"], r["n_updates"], r["wall_s"]["mean"], r["wall_s"]["max"], r["wall_s"]["cook_mean"])
    @printf("  %-9s %10s %10s %9s %12s %9s %32s\n", "group", "rate", "init sd", "NEES/dof", "cook median", "cook max", "band (one sample per cook-hour)")
    for gname in GROUPS
        v = r["groups"][gname]; lo, hi = band(n_eff, gd[gname])
        flag = lo <= v["nees_dof"] <= hi ? "" : (v["nees_dof"] > hi ? "  too narrow" : "  too wide")
        @printf("  %-9s %10.4g %10.4g %9.2f %12.2f %9.2f %18.2f to %.2f%s\n", gname, r["rates"][gname], r["init_sd"][gname], v["nees_dof"], v["cook_median"], v["cook_max"], lo, hi, flag)
    end
    println("  whiteness: " * join([@sprintf("%s mean %+.2f lag-1 %+.2f (band +-%.2f)", lab, w["mean"], w["lag1"], w["band"]) for (lab, w) in sort(collect(r["whiteness"]); by=first)], "; "))
    println("  sensors: " * join([@sprintf("%s white %.3g ou %.3g tau %.0f s", id, s["white_sd"], s["ar_sd"], s["tau"]) for (id, s) in sort(collect(r["sensors"]); by=first)], "; "))
    if haskey(r, "resid")
        rs = r["resid"]
        @printf("  the graph's whitened residuals at the accepted point (%d arrivals): the newest slice's dynamics residual power per coordinate by group (one when the rate is right, less by the group's leverage):\n    ", rs["n"])
        println(join([@sprintf("%s %.2f", g, rs["dyn"][g]) for g in GROUPS], ", "))
        println("    the window's: " * join([@sprintf("%s %.2f", g, rs["dyn_window"][g]) for g in GROUPS], ", "))
        println("    the observation residual squared per sensor: " * join([@sprintf("%s %.3f", id, v) for (id, v) in sort(collect(rs["obs"]); by=first)], ", ") *
                "; the OU residual squared per sensor: " * join([@sprintf("%s %.3f", id, v) for (id, v) in sort(collect(rs["ou"]); by=first)], ", ") *
                @sprintf("; the least-squares goodness of fit 2F/(rows − cols) %.3f", rs["gof"]))
    end
    if haskey(r, "past")
        p = r["past"]
        @printf("  the smoothed past (a node's belief as it left the window, against the plant at its time; %d nodes): NEES/%d of the temperatures %.2f, NEES/%d of the latents %.2f; by group ", p["n"], r["dof_T"], p["nees_T"], r["dof_z"], p["nees_z16"])
        println(join([@sprintf("%s %.2f", g, p["groups"][g]) for g in GROUPS], ", "))
    end
    flush(stdout)
end

"The identification on the round's pooled statistics: the state noise (the record's dictionary with the fitted rates written in) and the sensors' noise after it, the fit's record."
function identify(r, cap)
    r["als"] === nothing && error("the round's records carry no innovation-autocovariance statistics: rerun the workers with the accumulator (--lag-h > 0)")
    pooled = r["als"]; GROUPS = r["group_names"]
    pooled.groups[1:length(GROUPS)] == GROUPS || error("the accumulator's first groups $(pooled.groups[1:min(end, length(GROUPS))]) are not the record's $(GROUPS)")
    fit = als_fit(pooled; cap, summaries=r["als_summaries"])
    fit["lags"] = als_lag_table(pooled, fit, r["channels"], LAG_BINS)
    noise_new = Dict{String,Any}(String(k) => v for (k, v) in r["state_noise"])
    for (i, g) in enumerate(GROUPS); noise_new[g] = fit["rates_new"][i]; end
    sensors_new = Dict{Symbol,NamedTuple}()
    for (id, s) in r["sensors"]
        iou = findfirst(==("ou." * id), pooled.groups); iw = findfirst(==("white." * id), pooled.groups)
        sensors_new[Symbol(id)] = (white_sd=iw === nothing ? Float64(s["white_sd"]) : fit["rates_new"][iw],
                                   ar_sd=iou === nothing ? Float64(s["ar_sd"]) : fit["rates_new"][iou], tau=Float64(s["tau"]))
    end
    return noise_new, sensors_new, fit
end

function bench_numbers(r)
    GROUPS = r["group_names"]
    Dict{String,Any}("seeds" => r["seeds"], "skipped" => r["skipped"], "physics" => r["physics"], "lag" => r["lag"], "rates" => r["rates"], "init_sd" => r["init_sd"], "sensors" => r["sensors"],
                     "nees_dof" => Dict(g => r["groups"][g]["nees_dof"] for g in GROUPS), "nees_dof_cook_median" => Dict(g => r["groups"][g]["cook_median"] for g in GROUPS),
                     "nees_dof_cook_max" => Dict(g => r["groups"][g]["cook_max"] for g in GROUPS), "nis_dof" => r["nis_dof"], "latents_nees16" => r["latents_nees16"],
                     "latents_per_cook" => r["latents_per_cook"], "temps_nees3" => r["temps_nees3"], "temps_per_cook" => r["temps_per_cook"], "whiteness" => r["whiteness"],
                     "gradient" => r["gradient"], "uncommitted" => r["uncommitted"], "iters" => r["iters"], "wall_s" => r["wall_s"], "n_updates" => r["n_updates"])
end

"The consistency figure and the coverage figure from a round's manifest, by the existing renderers (run from app/ with uv)."
function render(manifest, tag, log_dir, drift_on::Bool)
    key = drift_on ? "err_sd_graphfilter" : "err_sd_system"
    for (script, args) in (("plot_filter_consistency.py", [manifest, tag]), ("plot_coverage.py", [manifest, key, tag]))
        cmd = `uv run python $(joinpath(REPO, "estimator", "python", script)) $args`
        log = joinpath(log_dir, script * ".log")
        io = open(log, "w")
        ok = success(pipeline(setenv(cmd; dir=joinpath(REPO, "app")); stdout=io, stderr=io))
        close(io)
        print(read(log, String))
        ok || error("$script failed; see $log")
    end
end

function main()
    a = parse_args(ARGS)
    rounds = split(a["rounds"], ",")
    if a["smoke"]
        a["cooks"], a["workers"], a["hours"] = 1, 1, min(a["hours"], 2.0)
    end
    n_per = cld(a["cooks"], a["workers"]); n_cooks = n_per * a["workers"]; a["cooks"] = n_cooks
    a["fresh-seed0"] === nothing && (a["fresh-seed0"] = a["tune-seed0"] + n_cooks)
    tune_seeds = [a["tune-seed0"] + i * n_per for i in 0:(a["workers"] - 1)]
    fresh_seeds = [a["fresh-seed0"] + i * n_per for i in 0:(a["workers"] - 1)]
    tune_blocks = [collect(s:(s + n_per - 1)) for s in tune_seeds]; fresh_blocks = [collect(s:(s + n_per - 1)) for s in fresh_seeds]
    overlap = intersect(reduce(vcat, tune_blocks), reduce(vcat, fresh_blocks))
    isempty(overlap) || error("the tuning seeds $tune_blocks and the fresh seeds $fresh_blocks share $overlap: the fresh round would not be held out")
    variant = a["variant"]
    w = WORKERS[a["worker"]]
    cal_file = joinpath(REPO, "data", "calibration", w.cal)
    stem_tune, stem_fresh = a["smoke"] ? (w.smoke_tune, w.smoke_fresh) : (w.stem_tune, w.stem_fresh)
    if !isempty(variant)
        stem_tune = "variant$(w.tag)_$(variant)_tune"; stem_fresh = "variant$(w.tag)_$(variant)_fresh"
    end
    tag = a["smoke"] ? w.tag * "_smoke" : (isempty(variant) ? w.tag : w.tag * "_$(variant)")
    cal_out = a["smoke"] ? joinpath(REPO, "scratchpad", "state_noise$(w.tag)_smoke.json") :
              (isempty(variant) ? cal_file : joinpath(REPO, "scratchpad", "state_noise$(w.tag)_$(variant).json"))
    manifest_of(rnd) = joinpath(MOCK, "bench_manifest_system_$(rnd)$(tag).json")
    log_dir = joinpath(REPO, "scratchpad", "bench" * tag)
    mkpath(log_dir); mkpath(dirname(cal_out))
    start_file = (a["start"] == "default" || !isfile(cal_file)) ? "default" : basename(cal_file)
    what = a["worker"] == "one" ? "one slab on the RxInfer bench's plant" :
           a["worker"] == "multi" ? "two pieces on a CookSystem plant, nx $(a["nx"]), the pieces $(a["order"] == 1 ? "flipped" : "in order"), dropped $(a["drop"])" :
           "the artifact $(a["rom"])'s slab on its brick"
    println("the factor-graph backend's bench ($what): ",
            "$n_cooks cooks per round in $(a["workers"]) workers ($n_per each; tuning seeds $tune_blocks, fresh seeds $fresh_blocks), $(a["hours"]) h each, prior $(a["prior"]); ",
            "lag $(a["lag"]); the plant $(a["drift"] == 1 ? "drifting" : "static"); the sensor noise $(a["noise"]); rounds $rounds; the tuning round starts from ",
            start_file == "default" ? "the code defaults" : "data/calibration/$start_file", "; logs in $log_dir")
    flush(stdout)
    record = nothing
    if "tune" in rounds
        names = ["$(stem_tune)_$(s).json" for s in tune_seeds]
        files = [joinpath(MOCK, nm) for nm in names]
        if a["reuse"] && all(isfile, files) && (try; validate_records(files, tune_blocks); true; catch e; println("  the existing records are not the round's ($(sprint(showerror, e))): the workers run"); false; end)
            println("  tune: reusing the records " * join(names, ", "))
        else
            files = run_workers("tune", tune_seeds, n_per, a, start_file, names, log_dir; stale_ok=start_file != "default")
        end
        write_manifest(manifest_of("tune"), "tune", files, tune_blocks; extra=Dict{String,Any}("state_noise_file" => start_file, "hours" => a["hours"], "prior" => a["prior"], "workers" => a["workers"], "lag" => a["lag"], "drift" => a["drift"], "noise" => a["noise"]))
        println("  wrote $(manifest_of("tune"))")
        tune = load_round(files)
        print_round("the tuning round (before the identification)", tune, a["hours"])
        noise_new, sensors_new, fit = identify(tune, a["cap"])
        println(format_als_fit(fit, Dict(g => E._group_unit(g) for g in fit["groups"]), tune["channels"]))
        println(format_als_lags(fit["lags"]))
        println("  the identification, each group's factor on its rate: " * join([@sprintf("%s x%.2f", g, fit["factors"][i]) for (i, g) in enumerate(fit["groups"])], ", "))
        method = "one autocovariance least-squares step on the tuning round (Odelson et al. 2006, fixed-gain time-varying form) over the state's twelve variance rates, " *
                 "every discrepancy sensor's OU variance and every sensor's white variance: the products of the standardized innovations at every pair of arrivals inside the lag " *
                 "window, less the smoother's own white model, regressed on the change of each variance under the linear gains at the predicted points, pooled over the cooks, " *
                 "weighted per channel pair and kind, bounded by the cap; a group the innovations cannot see keeps its numbers, the initial widths are kept; the fresh round is " *
                 "the check and the round the numbers are quoted from; the bench's plant is the model's own physics, so the rates are a floor"
        bench = Dict{String,Any}("cooks_per_round" => n_cooks, "cooks_per_worker" => n_per, "workers" => a["workers"], "hours" => a["hours"], "prior" => a["prior"], "cap" => a["cap"],
                                 "lag_h" => a["lag-h"], "lag" => a["lag"], "drift" => a["drift"], "noise" => a["noise"],
                                 "tune" => merge(bench_numbers(tune), Dict{String,Any}("records" => [basename(f) for f in files], "manifest" => basename(manifest_of("tune")), "started_from" => start_file,
                                                                                       "stale_ok" => start_file != "default", "factors" => Dict(g => fit["factors"][i] for (i, g) in enumerate(fit["groups"])))))
        record = system_calibration_record(noise_new, sensors_new; fit, bench, method)
        write_system_calibration(cal_out, record)
        println("  wrote $cal_out")
        flush(stdout)
    end
    if "fresh" in rounds
        cal_path = cal_out
        if !isempty(variant) && !isfile(cal_path)
            cal_path = cal_file                                                            # a variant's fresh round runs on the calibration as earned
        end
        isfile(cal_path) || error("no calibration file at $cal_path: run the tuning round first")
        record = parse_json_file(cal_path)
        expected_noise = merge(Dict{String,Any}(String(g) => v for (g, v) in record["rates"]), Dict{String,Any}(String(g) * "0" => v for (g, v) in record["init_sd"]),
                               Dict{String,Any}("fields" => record["fields"], "node_jitter" => record["node_jitter"], "field_modes" => get(record, "field_modes", 0)),
                               Dict{String,Any}(String(k) => v for (k, v) in record["events"]))
        cal_arg = cal_path == cal_file ? basename(cal_path) : cal_path
        names = ["$(stem_fresh)_$(s).json" for s in fresh_seeds]
        files = [joinpath(MOCK, nm) for nm in names]
        if a["reuse"] && all(isfile, files) && (try; validate_records(files, fresh_blocks; physics=record["physics"]["fingerprint"], state_noise=expected_noise); true; catch e; println("  the existing records are not the round's ($(sprint(showerror, e))): the workers run"); false; end)
            println("  fresh: reusing the records " * join(names, ", "))
        else
            files = run_workers("fresh", fresh_seeds, n_per, a, cal_arg, names, log_dir)     # strict: the calibration must be this tree's
        end
        write_manifest(manifest_of("fresh"), "fresh", files, fresh_blocks; physics=record["physics"]["fingerprint"], state_noise=expected_noise,
                       extra=Dict{String,Any}("state_noise_file" => cal_arg, "hours" => a["hours"], "prior" => a["prior"], "workers" => a["workers"], "lag" => a["lag"], "drift" => a["drift"], "noise" => a["noise"]))
        println("  wrote $(manifest_of("fresh"))")
        fresh = load_round(files)
        print_round("the fresh round (the calibration as written, seeds never seen)", fresh, a["hours"])
        out_path = isempty(variant) ? cal_path : joinpath(REPO, "scratchpad", "state_noise$(w.tag)_$(variant).json")
        record["bench"]["fresh"] = merge(bench_numbers(fresh), Dict{String,Any}("records" => [basename(f) for f in files], "manifest" => basename(manifest_of("fresh"))))
        write_system_calibration(out_path, record)
        println("  wrote $out_path")
        a["no-figures"] || render(basename(manifest_of("fresh")), tag, log_dir, a["drift"] == 1)
    end
    println("done")
end
main()
