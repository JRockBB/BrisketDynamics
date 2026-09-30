using RxInfer,BrisketDynamics,ForwardDiff,LinearAlgebra,Random,JSON,Printf,SHA,SparseArrays,SpecialFunctions,Test,Statistics
println("DEPENDENCIES READY");flush(stdout)
while !isfile(joinpath(@__DIR__,"source_ready"));sleep(.5);end
include(joinpath(@__DIR__,"../../../estimator/src/BrisketEstimator.jl"))
using .BrisketEstimator
const E=BrisketEstimator
println("READY ",E.backend_fingerprint());flush(stdout)
while !isfile(joinpath(@__DIR__,"stop"))
    jobs=sort(filter(f->endswith(f,".job.jl"),readdir(@__DIR__)))
    if isempty(jobs);sleep(.5);continue;end
    path=joinpath(@__DIR__,first(jobs));println("START ",basename(path));flush(stdout)
    try
        include(path);println("PASS ",basename(path))
    catch err
        showerror(stdout,err,catch_backtrace());println();println("FAIL ",basename(path))
    end
    mv(path,path*".done";force=true);flush(stdout)
end