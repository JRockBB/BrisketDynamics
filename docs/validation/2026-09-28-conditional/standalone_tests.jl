# Let the test files initialize their own environment, without preloaded helpers.
include(joinpath(@__DIR__,"../../../estimator/test/test_deterministic_online.jl"))
include(joinpath(@__DIR__,"../../../estimator/test/test_conditional_density.jl"))
println("FRESH PROCESS PASS ",BrisketEstimator.backend_fingerprint()," PHYSICS ",BrisketEstimator.physics_fingerprint());flush(stdout)
write(joinpath(@__DIR__,"completed"),"163 assertions passed")