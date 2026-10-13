# The calibration on the SHIPPED cook: one autocovariance least-squares step from the innovation accumulators of a
# tuning round of stream_bench.jl records (lighting start, the shipped lag, the sensors' own clocks), written as a
# calibration file. The same identification bench_system.jl makes on its lockstep plant (bench_fit.jl's als_pool /
# als_fit), on the cook the estimator ships for.
#
# The accumulator is fed in NODE attachment (stream_bench.jl STREAM_ALS_LAG_H with attach=node, pace 0): inside a
# slice the process noise is correlated across arrivals, which the accumulator's Markov frame does not represent,
# while the node path is the exact representation of the same model. The rates are the model's; the check is the
# fresh round on unseen seeds in the shipped SLICE configuration (stream_bench.jl STREAM_CALIBRATION=<this file>).
#
# Run: julia --project=estimator estimator/scripts/stream_calibrate.jl <out.json> <cap> <record.json>...
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using JSON, Printf
const E = BrisketEstimator

length(ARGS) >= 3 || error("usage: stream_calibrate.jl <out.json> <cap> <record.json>...")
out_path = ARGS[1]; cap = parse(Float64, ARGS[2]); files = ARGS[3:end]
const LAG_BINS = [0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60]

summaries = Any[]; seeds = Int[]; groups = nothing; sn = nothing; sensors = nothing; channels = nothing; lags = Set{Any}()
for f in files
    global groups, sn, sensors, channels
    d = parse_json_file(f)                       # the bench writes NaN/Inf (empty innovation statistics)
    get(d, "als_lag_h", 0) > 0 || error("$f carries no accumulator (run stream_bench.jl with STREAM_ALS_LAG_H > 0)")
    g = String.(d["groups"])
    groups === nothing ? (groups = g) : (g == groups || error("$f has groups $g, the others $groups"))
    sn === nothing ? (sn = d["state_noise"]) : (d["state_noise"] == sn || error("$f ran with another state noise"))
    sensors === nothing && (sensors = d["sensors"]); channels === nothing && (channels = d["channels"])
    push!(lags, d["lag"])
    for c in d["cooks"]
        c["als"] === nothing && error("$f: cook $(c["seed"]) has no accumulator")
        get(c, "n_uncommitted", 0) == 0 || @warn "cook $(c["seed"]) had $(c["n_uncommitted"]) uncommitted updates (their readings left the accumulator)"
        push!(summaries, c["als"]); push!(seeds, Int(c["seed"]))
    end
end
length(lags) == 1 || error("the records ran at different lags: $(collect(lags))")
length(unique(seeds)) == length(seeds) || error("a seed appears twice: $seeds")
isempty(summaries) && error("no completed cooks to pool")
chan = Dict(parse(Int, k) => String(v) for (k, v) in channels)

pooled = als_pool(summaries)
pooled.groups[1:length(groups)] == groups || error("the accumulator's groups $(pooled.groups) do not lead with the record's $groups")
fit = als_fit(pooled; cap, summaries)
fit["lags"] = als_lag_table(pooled, fit, chan, LAG_BINS)
println(format_als_fit(fit, Dict(g => E._group_unit(g) for g in fit["groups"]), chan))
println(format_als_lags(fit["lags"]))

noise_new = Dict{String,Any}(String(k) => v for (k, v) in sn)
for (i, g) in enumerate(groups); noise_new[g] = fit["rates_new"][i]; end
sensors_new = Dict{Symbol,NamedTuple}()
for (id, s) in sensors
    iou = findfirst(==("ou." * id), pooled.groups); iw = findfirst(==("white." * id), pooled.groups)
    sensors_new[Symbol(id)] = (white_sd=iw === nothing ? Float64(s["white_sd"]) : fit["rates_new"][iw],
                               ar_sd=iou === nothing ? Float64(s["ar_sd"]) : fit["rates_new"][iou], tau=Float64(s["tau"]))
end
method = "one autocovariance least-squares step (Odelson et al. 2006, fixed-gain time-varying form) on a tuning round of the " *
         "shipped cook (stream_bench.jl: lighting start, the sensors' own clocks, lag $(first(lags))), the accumulator fed in node " *
         "attachment (the exact representation; slice noise is correlated across arrivals), pooled over $(length(seeds)) cooks, " *
         "bounded by the cap $cap; the check is the fresh round in the shipped slice configuration; the plant is the model's own " *
         "physics, so the rates are a floor"
bench = Dict{String,Any}("tune_records" => basename.(files), "tune_seeds" => seeds, "cap" => cap, "lag" => first(lags),
                         "factors" => Dict(g => fit["factors"][i] for (i, g) in enumerate(fit["groups"])))
record = system_calibration_record(noise_new, sensors_new; fit, bench, method)
write_system_calibration(out_path, record)
println("wrote $out_path from $(length(seeds)) cooks (seeds $(join(seeds, ", ")))")
