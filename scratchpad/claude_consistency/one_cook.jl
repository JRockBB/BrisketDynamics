include(joinpath(@__DIR__, "..", "..", "estimator", "scripts", "consistency_control.jl"))
seed = parse(Int, ARGS[1]); t_a = parse(Float64, ARGS[2])
model = consistency_model(; nx=4, suite=:full, stop=t_a)
try
    control_cook(model, seed; times=collect(0.0:120.0:t_a), lag=typemax(Int))
    println("seed $seed committed every arrival")
catch err
    println(sprint(showerror, err)[1:min(end, 3000)])
end
