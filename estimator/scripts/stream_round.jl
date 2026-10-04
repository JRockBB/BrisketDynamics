# Reproducible process-level rounds of the full-resolution streaming bench.
# Usage: julia --project=estimator estimator/scripts/stream_round.jl TAG SEED N WORKERS HOURS SCALE RATIO [LAG] [THREADS] [PACE]
# PACE defaults to 1 (causal processing backlog); 0 processes each timestamp
# without compute pacing; -1 explicitly requests offline node-sized batches.
# Concurrent workers contend for resources, so use WORKERS=1 for latency measurements.
# Additional STREAM_* controls (integration tolerances, batch duration) are inherited and recorded by every worker.
using JSON, Dates

length(ARGS) >= 7 || error("usage: stream_round.jl TAG SEED N WORKERS HOURS SCALE RATIO [LAG] [THREADS] [PACE]")
tag = ARGS[1]
occursin(r"^[A-Za-z0-9_]+$", tag) || error("tag must contain only letters, digits and underscores")
seed0, n, workers = parse.(Int, ARGS[2:4])
hours, scale, ratio = parse.(Float64, ARGS[5:7])
lag = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 2
nthreads = length(ARGS) >= 9 ? parse(Int, ARGS[9]) : 4
pace = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : 1.0
isfinite(pace) && (pace == -1 || pace >= 0) || error("pace must be -1 (offline node batches), 0, or positive")
update_policy = pace < 0 ? "node_batches" : pace == 0 ? "every_arrival" : "processing_backlog"
n > 0 && workers > 0 && nthreads > 0 || error("counts must be positive")
all(x -> isfinite(x) && x > 0, (hours, scale, ratio)) || error("hours, scale and ratio must be finite and positive")
lag >= 0 || error("lag cannot be negative")
workers = min(workers, n)
repo = normpath(joinpath(@__DIR__, "..", ".."))
env = merge(Dict(ENV), Dict("STREAM_START" => "lighting", "STREAM_NOISE" => "matched", "STREAM_INITIAL" => "prior",
    "STREAM_SN_SCALE" => string(scale), "OPENBLAS_NUM_THREADS" => "1"))
logdir = joinpath(repo, "scratchpad", "estimator_preheat", tag)
mkpath(logdir)
manifest_path = joinpath(repo, "data", "mock", "stream_manifest_$(tag).json")
isfile(manifest_path) && error("round already exists: $manifest_path")
for i in 1:workers
    output = joinpath(repo, "data", "mock", "stream_$(tag)_$(seed0 + fld((i - 1) * n, workers)).json")
    isfile(output) && error("record already exists: $output")
end
jobs = Any[]; entries = Any[]
manifest = Dict("tag" => tag, "started" => string(now()), "seed0" => seed0, "n_cooks" => n,
    "hours" => hours, "scale" => scale, "node_ratio" => ratio, "lag" => lag, "threads_per_worker" => nthreads,
    "pace" => pace, "update_policy" => update_policy,
    "workers" => entries, "state" => "running")
save_manifest() = open(io -> JSON.print(io, manifest, 2), manifest_path, "w")
for i in 1:workers
    first_seed = seed0 + fld((i - 1) * n, workers)
    count = fld(i * n, workers) - fld((i - 1) * n, workers)
    output = "stream_$(tag)_$(first_seed).json"
    isfile(joinpath(repo, "data", "mock", output)) && error("record already exists: $output")
    logfile = joinpath(logdir, "worker_$(first_seed).log")
    io = open(logfile, "w")
    cmd = `$(Base.julia_cmd()) --threads=$nthreads --project=$(joinpath(repo, "estimator")) $(joinpath(@__DIR__, "stream_bench.jl")) $count $first_seed $output $hours $lag slice $pace 30 30 120 120 $ratio`
    process = run(pipeline(setenv(cmd, env); stdout=io, stderr=io); wait=false)
    push!(jobs, (; process, io))
    push!(entries, Dict("first_seed" => first_seed, "n_cooks" => count, "record" => "data/mock/$output",
        "log" => relpath(logfile, repo), "command" => string(cmd), "state" => "running"))
    println("started seeds $first_seed:$(first_seed + count - 1), log $logfile"); flush(stdout)
    save_manifest()
end
while any(e -> e["state"] == "running", entries)
    sleep(5)
    for (job, entry) in zip(jobs, entries)
        if entry["state"] == "running" && process_exited(job.process)
            wait(job.process); close(job.io)
            entry["exit_code"] = job.process.exitcode
            entry["state"] = success(job.process) ? "complete" : "failed"
            println("worker $(entry["first_seed"]): $(entry["state"])"); flush(stdout)
            save_manifest()
        end
    end
end
manifest["finished"] = string(now())
manifest["state"] = all(e -> e["state"] == "complete", entries) ? "complete" : "failed"
save_manifest()
println("round $(manifest["state"]): $manifest_path")
manifest["state"] == "complete" || exit(1)
