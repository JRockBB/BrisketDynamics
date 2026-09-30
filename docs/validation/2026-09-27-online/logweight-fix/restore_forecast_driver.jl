# The suspected forecast alias was a false positive: _posterior_transport already deep-copies
# the graph. Restore both original methods; no forecast production change is needed.
for (file,marker,nextmarker) in (
    ("posterior_sampling.jl","function _posterior_transport", "\"The sampled posterior's means/covariances"),
    ("sampled_planning.jl","function posterior_predict(p::SampledPosterior", "posterior_predict(sm::Union"))
    path=joinpath(@__DIR__,"../../../estimator/src",file);original=read(path,String)
    lo=first(findfirst(marker,original));hi=first(findnext(nextmarker,original,lo))
    Base.include_string(E,original[lo:prevind(original,hi)],path)
    println("RESTORED PRODUCTION METHOD ",file," sha1=",bytes2hex(sha1(read(path))));flush(stdout)
end
include(joinpath(@__DIR__,"retained_forecast_test.jl"))
