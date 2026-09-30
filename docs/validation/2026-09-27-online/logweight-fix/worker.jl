using RxInfer,BrisketDynamics,ForwardDiff,LinearAlgebra,Random,JSON,Printf,SHA,SparseArrays,SpecialFunctions,Test,Statistics
module_path=joinpath(@__DIR__,"../../../estimator/src/BrisketEstimator.jl")
candidate_path=joinpath(@__DIR__,"particle_online.jl")
source=replace(read(module_path,String),"include(joinpath(@__DIR__, \"particle_online.jl\"))"=>"include("*repr(candidate_path)*")")
occursin("include("*repr(candidate_path)*")",source) || error("candidate include was not substituted")
Base.include_string(Main,source,module_path)
using .BrisketEstimator
const E=BrisketEstimator
const BASE_BACKEND=backend_fingerprint()
println("LOGWEIGHT CANDIDATE base=",BASE_BACKEND," candidate=",bytes2hex(sha1(read(candidate_path))));flush(stdout)
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
