module FinalConditionalValidation
    include(joinpath(@__DIR__,"../../../estimator/src/BrisketEstimator.jl"))
    using .BrisketEstimator, BrisketDynamics, Test, LinearAlgebra
    const E=BrisketEstimator
    println("VALIDATION BACKEND ",E.backend_fingerprint()," PHYSICS ",E.physics_fingerprint());flush(stdout)
    include(joinpath(@__DIR__,"../../../estimator/scripts/consistency_control.jl"))
    for name in ("test_conditional.jl","test_conditional_online.jl","test_conditional_density.jl","test_deterministic_online.jl")
        println("VALIDATION ",name);flush(stdout)
        include(joinpath(@__DIR__,"../../../estimator/test",name))
    end
    for name in ("test_factors.jl","test_consistency.jl")
        println("REGRESSION ",name);flush(stdout)
        path=joinpath(@__DIR__,"../../../estimator/test",name)
        src=read(path,String)
        src=replace(src,r"(?m)^include\(joinpath\(@__DIR__,\s*\"\.\.\",\s*\"src\",\s*\"BrisketEstimator.jl\"\)\)\s*$"=>"")
        src=replace(src,r"(?m)^const E\s*=\s*BrisketEstimator\s*$"=>"")
        Base.include_string(@__MODULE__,src,path)
    end
end