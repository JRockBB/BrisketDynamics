using RxInfer,BrisketDynamics,ForwardDiff,LinearAlgebra,Random,JSON,Printf,SHA,SparseArrays,SpecialFunctions,Test,Statistics
include(joinpath(@__DIR__,"../../estimator/src/BrisketEstimator.jl"))
using .BrisketEstimator
const E=BrisketEstimator
const START_BACKEND=backend_fingerprint()
println("FRESH COMPARISON BACKEND ",START_BACKEND," PHYSICS ",physics_fingerprint());flush(stdout)
include(joinpath(@__DIR__,"../../estimator/scripts/validate_sampled_online.jl"))
println("COMPARISON SPECIFICATION ",online_comparison_fingerprint());flush(stdout)
records=validate_sampled_online(output=joinpath(@__DIR__,"online_comparison.json"))
START_BACKEND==backend_fingerprint() || error("source changed during comparison")
all(r.completed && r.comparison.passed for r in records) || error("online comparison failed")
println("COMPARISON FINISHED ",START_BACKEND);flush(stdout)
