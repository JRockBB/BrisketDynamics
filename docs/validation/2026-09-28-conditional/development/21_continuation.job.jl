for name in ("conditional_online.jl","deterministic_online.jl")
    src=read(joinpath(@__DIR__,"../../../estimator/src",name),String)
    for ex in Meta.parseall(src).args;isstructexpr(ex) || Core.eval(R.E,ex);end
end
Base.include(R,joinpath(@__DIR__,"../../../estimator/scripts/validate_deterministic.jl"))
R.validate_deterministic(output=joinpath(@__DIR__,"continuation_development.json"))