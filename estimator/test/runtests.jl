# Every test file of the estimator, each in its own process under the estimator's environment (each file
# activates the environment and includes the module itself, so they run alone as well).
#
# Run:  julia estimator/test/runtests.jl            (or any single file: julia --project=estimator estimator/test/test_x.jl)
const HERE = @__DIR__
const FILES = sort(filter(f -> startswith(f, "test_") && endswith(f, ".jl"), readdir(HERE)))
failed = String[]
for f in FILES
    println("== ", f)
    ok = success(run(ignorestatus(`$(Base.julia_cmd()) --project=$(joinpath(HERE, "..")) $(joinpath(HERE, f))`)))
    ok || push!(failed, f)
end
if isempty(failed)
    println("all ", length(FILES), " test files passed")
else
    println(length(failed), " of ", length(FILES), " test files failed: ", join(failed, ", "))
    exit(1)
end
