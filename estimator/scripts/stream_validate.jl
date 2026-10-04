# Independent-cook scoring for stream_bench records. Correlated arrival rows
# are descriptive only; uncertainty bands use one checkpoint per cook.
# Run: julia --project=estimator estimator/scripts/stream_validate.jl output.json records.json ...
using JSON, Statistics, Printf
using SpecialFunctions: gamma_inc, beta_inc

length(ARGS) >= 2 || error("usage: stream_validate.jl output.json records.json ...")
dst, files = ARGS[1], ARGS[2:end]
records = [JSON.parsefile(path; allownan=true) for path in files]
all(d -> get(d, "active_cook", nothing) === nothing, records) || error("a worker is still running; do not score partial cooks as complete")
const CONFIG_KEYS = ("backend_fingerprint", "bench_fingerprint", "start_mode", "noise_mode", "initial_mode",
    "process_rate_scale", "node_ratio", "lag", "pace", "batch_s", "max_slice_s", "lag_max", "prior_relin", "steps_s", "prior", "abstol", "reltol", "floor_abstol", "floor_reltol", "fail_fast", "names", "groups")
signature(d) = [get(d, k, nothing) for k in CONFIG_KEYS]
all(d -> signature(d) == signature(first(records)), records) || error("cannot pool different benchmark configurations")
cooks = reduce(vcat, [d["cooks"] for d in records])
skipped = reduce(vcat, [d["skipped_seeds"] for d in records])
failures = reduce(vcat, [get(d, "failures", []) for d in records])
seeds = [c["seed"] for c in cooks]
length(unique(vcat(seeds, skipped))) == length(seeds) + length(skipped) || error("duplicate seeds in validation inputs")
if isempty(cooks)
    result = Dict("configuration" => Dict(k => get(first(records), k, nothing) for k in CONFIG_KEYS),
        "files" => files, "seeds" => seeds, "skipped_seeds" => skipped, "failures" => failures,
        "n_cooks" => 0, "checkpoint_gates_pass" => false, "all_reported_gates_pass" => false,
        "error" => "No completed cooks; no statistical consistency claim is possible.")
    mkpath(dirname(abspath(dst)))
    open(dst, "w") do io; write(io, JSON.json(result; allownan=true, pretty=true)); end
    println("no completed cooks; failures recorded in $dst")
    exit()
end
names = first(records)["names"]
nz = length(names)

function chi_quantile(p, df)
    lo, hi = 0.0, Float64(max(df, 1))
    while gamma_inc(df / 2, hi / 2)[1] < p; hi *= 2; end
    for _ in 1:80
        mid = (lo + hi) / 2
        gamma_inc(df / 2, mid / 2)[1] < p ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end
function joint_score(v, dim)
    n = length(v)
    all(x -> x isa Real && isfinite(x), v) || return Dict("pass" => false, "error" => "nonfinite statistic")
    df = n * dim
    lo, hi = chi_quantile(0.025, df) / df, chi_quantile(0.975, df) / df
    value = sum(v) / df
    return Dict("normalized_nees" => value, "band95" => [lo, hi], "n_cooks" => n, "dimension" => dim, "pass" => lo <= value <= hi)
end

function t_critical(df)
    lo, hi = 0.0, 2.0
    tail(x) = beta_inc(df / 2, 0.5, df / (df + x^2))[1] / 2
    while tail(hi) > 0.025; hi *= 2; end
    for _ in 1:80
        mid = (lo + hi) / 2
        tail(mid) > 0.025 ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end
"A cook, not an arrival, is the independent unit of this interval."
function cluster_score(v, target)
    n = length(v)
    n >= 2 && all(x -> x isa Real && isfinite(x), v) || return Dict("pass" => false, "n_cooks" => n, "error" => "need two finite independent cooks")
    m = mean(v); width = t_critical(n - 1) * std(v) / sqrt(n)
    return Dict("mean" => m, "interval95" => [m - width, m + width], "target" => target,
                "n_cooks" => n, "pass" => m - width <= target <= m + width)
end

function site_score(v)
    n = length(v)
    n >= 2 && all(isfinite, v) || return Dict("pass" => false, "n_cooks" => n, "error" => "need two finite independent cooks")
    bias = cluster_score(v, 0.0)
    variance_band = [chi_quantile(0.025, n - 1), chi_quantile(0.975, n - 1)] ./ (n - 1)
    spread = var(v)
    spread_pass = variance_band[1] <= spread <= variance_band[2]
    # Wilson interval for the fraction inside the nominal two-sided 95% interval.
    coverage = mean(abs.(v) .<= 1.96)
    q = 1.959963984540054
    denominator = 1 + q^2 / n
    centre = (coverage + q^2 / (2n)) / denominator
    halfwidth = q * sqrt(coverage * (1 - coverage) / n + q^2 / (4n^2)) / denominator
    coverage_band = [centre - halfwidth, centre + halfwidth]
    coverage_pass = coverage_band[1] <= 0.95 <= coverage_band[2]
    return Dict("mean_z" => mean(v), "rms_z" => sqrt(mean(abs2, v)), "bias" => bias,
        "variance_z" => spread, "variance_band95" => variance_band, "spread_pass" => spread_pass,
        "coverage95" => coverage, "coverage_interval95" => coverage_band, "coverage_pass" => coverage_pass,
        "pass" => bias["pass"] && spread_pass && coverage_pass)
end

edge_values = [b["t_h"] for b in first(cooks)["batches"]]
all(c -> [b["t_h"] for b in c["batches"]] == edge_values, cooks) || error("checkpoints differ")
edges = Any[]
for (j, time_h) in enumerate(edge_values)
    rows = [c["batches"][j] for c in cooks]
    Z = reduce(hcat, [Float64.(b["err_sd"]) for b in rows])
    per_site = Dict(names[i] => site_score(Z[i, :]) for i in 1:nz)
    groups = Dict(g => cluster_score([get(get(b, "nees_g", Dict()), g, NaN) for b in rows], 1.0)
                  for g in get(first(records), "groups", String[]))
    latents = joint_score([b["nees_z"] for b in rows], nz)
    temperatures = joint_score([b["nees_T"] for b in rows], 3)
    push!(edges, Dict("time_h" => time_h, "actual_times_h" => [b["state_t_h"] for b in rows],
        "latents" => latents, "temperatures" => temperatures, "per_site" => per_site, "state_groups" => groups,
        "pass" => latents["pass"] && temperatures["pass"] && all(s["pass"] for s in values(per_site)) &&
                  !isempty(groups) && all(s["pass"] for s in values(groups))))
    @printf("%5.1f h: latent NEES/%d %.3f %s; temperature NEES/3 %.3f %s (%d independent cooks)\n",
        time_h, nz, get(latents, "normalized_nees", NaN), latents["pass"] ? "inside band" : "OUTSIDE band",
        get(temperatures, "normalized_nees", NaN), temperatures["pass"] ? "inside band" : "OUTSIDE band", length(cooks))
end
rows = [r for c in cooks for r in c["temps"]]
uncommitted = count(r -> !r["committed"], rows)
unconverged = count(r -> !r["converged"], rows)
censored = sum(get(r, "n_censored", max(0, get(r, "n_readings", r["dof"]) - r["dof"])) for r in rows)
finite_nis = [r for r in rows if r["dof"] > 0 && r["nis"] isa Real && isfinite(r["nis"])]
total_dof = sum(r["dof"] for r in finite_nis)
pooled_nis = total_dof > 0 ? sum(r["nis"] for r in finite_nis) / total_dof : nothing
per_cook_nis = [sum(r["nis"] for r in c["temps"] if r["dof"] > 0) /
                sum(r["dof"] for r in c["temps"] if r["dof"] > 0) for c in cooks]
nis_cluster = cluster_score(per_cook_nis, 1.0)
innovation = Dict()
for id in ("pit", "core", "humidity", "mass")
    series = [[w for r in c["temps"] for w in get(r, "innovation_whitened", []) if w["sensor"] == id] for c in cooks]
    for s in series; sort!(s; by=w -> w["t"]); end
    series_values = [Float64[w["value"] for w in s] for s in series]
    enough = all(v -> length(v) >= 3 && all(isfinite, v), series_values)
    if !enough
        innovation[id] = Dict("pass" => false, "error" => "missing or invalid chronological innovations")
        continue
    end
    lag1(v) = (w = v .- mean(v); sum(w[1:end-1] .* w[2:end]) / sqrt(sum(abs2, w[1:end-1]) * sum(abs2, w[2:end])))
    means = cluster_score(mean.(series_values), 0.0)
    second_moment = cluster_score([mean(abs2, v) for v in series_values], 1.0)
    correlations = cluster_score(lag1.(series_values), 0.0)
    innovation[id] = Dict("mean" => means, "second_moment" => second_moment, "lag1" => correlations,
        "n_per_cook" => length.(series_values), "pass" => means["pass"] && second_moment["pass"] && correlations["pass"])
end
result = Dict("configuration" => Dict(k => get(first(records), k, nothing) for k in CONFIG_KEYS),
    "files" => files, "seeds" => seeds, "skipped_seeds" => skipped, "failures" => failures, "n_cooks" => length(cooks),
    "n_updates" => length(rows), "uncommitted" => uncommitted, "unconverged" => unconverged,
    "n_censored" => censored, "innovation_gaussian_null_valid" => censored == 0,
    "nis_per_dof_descriptive" => pooled_nis, "nis_cook_cluster" => nis_cluster, "innovations" => innovation, "checkpoints" => edges,
    "checkpoint_gates_pass" => isempty(skipped) && isempty(failures) && uncommitted == 0 && unconverged == 0 &&
        !isempty(edges) && all(e -> e["pass"], edges),
    "scope" => "One checkpoint per independent cook. Gates include joint NEES, every latent's bias, variance and coverage, and every reported state group's normalized NEES. NIS and chronological whitened-innovation intervals use independent cook summaries. Pointwise 95% intervals are not simultaneous bands; some failures are expected by chance across many comparisons. When readings are capped, selecting only uncensored innovations invalidates the unit-Gaussian null: those innovation scores are descriptive and cannot certify consistency. These tests do not establish real-pit calibration or a matched stochastic process model.")
result["all_reported_gates_pass"] = result["checkpoint_gates_pass"] && censored == 0 && nis_cluster["pass"] && all(v["pass"] for v in values(innovation))
mkpath(dirname(abspath(dst)))
open(dst, "w") do io; write(io, JSON.json(result; allownan=true, pretty=true)); end
println("skipped=$(skipped), uncommitted=$uncommitted, unconverged=$unconverged; wrote $dst")
