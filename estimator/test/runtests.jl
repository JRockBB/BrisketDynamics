# Every test file of the estimator, each in its own process under the estimator's environment (each file
# activates the environment and includes the module itself, so they run alone as well), with the wall time of
# each so the CI shards can be balanced.
#
# Run:  julia estimator/test/runtests.jl                       (every test_*.jl)
#       julia estimator/test/runtests.jl test_noise.jl test_channels.jl   (a shard: these files only)
#       (or any single file: julia --project=estimator estimator/test/test_x.jl)
const HERE = @__DIR__
const ALL = sort(filter(f -> startswith(f, "test_") && endswith(f, ".jl"), readdir(HERE)))
const FILES = isempty(ARGS) ? ALL : ARGS
for f in FILES
    f in ALL || error("no such test file: $f (the files are $(join(ALL, ", ")))")
end
failed = String[]
for f in FILES
    println("== ", f, " (", Threads.nthreads(), " Julia threads)"); flush(stdout)
    t0 = time()
    # julia_cmd preserves compiler options, but not --threads. Forward the
    # requested pool explicitly so expensive derivative tests use that pool.
    ok = success(run(ignorestatus(`$(Base.julia_cmd()) --threads=$(Threads.nthreads()) --project=$(joinpath(HERE, "..")) $(joinpath(HERE, f))`)))
    println("== ", f, ": ", ok ? "passed" : "FAILED", " in ", round(Int, time() - t0), " s")
    flush(stdout)
    ok || push!(failed, f)
end
if isempty(failed)
    println("all ", length(FILES), " test files passed")
else
    println(length(failed), " of ", length(FILES), " test files failed: ", join(failed, ", "))
    exit(1)
end
