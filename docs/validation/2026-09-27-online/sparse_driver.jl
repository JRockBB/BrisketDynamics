# Restore the two overwritten methods from the exact frozen production source before validation.
for (file,first_marker,next_marker) in (
    ("particle_online.jl","function _incremental_posterior","function _posterior_predictions"),
    ("posterior_smc.jl","function _weak_moves!","function _systematic_resample"))
    path=joinpath(@__DIR__,"../../../estimator/src",file);source=read(path,String)
    lo=first(findfirst(first_marker,source));hi=first(findfirst(next_marker,source))
    Base.include_string(E,source[lo:prevind(source,hi)],path)
end
BASE_BACKEND==backend_fingerprint() || error("frozen production source changed")
println("BASELINE METHODS RESTORED ",BASE_BACKEND," threads=",Threads.nthreads());flush(stdout)
# Run only the last two cases of the unchanged prespecified comparison; the first process owns 1-2.
saved_cases=copy(ONLINE_COMPARISON_CASES)
try
    empty!(ONLINE_COMPARISON_CASES);append!(ONLINE_COMPARISON_CASES,saved_cases[3:4])
    sparse_records=validate_sampled_online(output=joinpath(@__DIR__,"baseline_sparse_comparison.json"))
    all(r.completed && r.comparison.passed for r in sparse_records) || error("sparse baseline comparison failed")
finally
    empty!(ONLINE_COMPARISON_CASES);append!(ONLINE_COMPARISON_CASES,saved_cases)
end
BASE_BACKEND==backend_fingerprint() || error("source changed during sparse comparison")
