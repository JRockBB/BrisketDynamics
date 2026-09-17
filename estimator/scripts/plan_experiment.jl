# Planning as inference in RxInfer, the bounded experiment: the problem and the three
# solvers are in plan_experiment_setup.jl; this runs them at the posterior's width and
# at a tenth of it, then the naive graph.
#
# Run:  julia --project=estimator estimator/scripts/plan_experiment.jl
# Writes data/mock/plan_experiment.json.
include(joinpath(@__DIR__, "plan_experiment_setup.jl"))

results = Dict{String,Any}("targets" => Dict("T_C" => T_TARGET - 273.15, "W" => W_TARGET),
                           "prior_plan" => Dict("u1" => 0.15, "u2" => 0.15, "pull_h" => 12.0), "N" => N,
                           "sd6" => sd6, "z6" => z6)
for (label, scale) in (("posterior width", 1.0), ("width shrunk tenfold", 0.1))
    zs = draws(scale)
    println("=== draws at ", label)
    rR = solve_reference(zs)
    rB = solve_graph_B(zs)
    oR = outcomes(rR.ζ, zs); oB = outcomes(rB.ζ, zs); o0 = outcomes(ζ0, zs)
    @printf("  R  plan u1 %.3f u2 %.3f pull %.2f h  F %.3f   outcomes: T %.1f ± %.1f C, W %.3f ± %.3f\n",
            plan_phys(rR.ζ)..., rR.F, oR.T_mean, oR.T_sd, oR.W_mean, oR.W_sd)
    @printf("  B  plan u1 %.3f u2 %.3f pull %.2f h  F %.3f   outcomes: T %.1f ± %.1f C, W %.3f ± %.3f\n",
            plan_phys(rB.ζ)..., rB.F, oB.T_mean, oB.T_sd, oB.W_mean, oB.W_sd)
    @printf("  prior plan: outcomes T %.1f ± %.1f C, W %.3f ± %.3f\n", o0.T_mean, o0.T_sd, o0.W_mean, o0.W_sd)
    @printf("  B - R in the plan's own sd: %s\n", string(round.((rB.ζ .- rR.ζ) ./ rR.sd; digits=2)))
    flush(stdout)
    results[label] = Dict("R" => Dict("zeta" => rR.ζ, "F" => rR.F, "sd" => rR.sd, "plan" => collect(plan_phys(rR.ζ)), "outcomes" => Dict(pairs(oR))),
                          "B" => Dict("zeta" => rB.ζ, "F" => rB.F, "sd" => rB.sd, "plan" => collect(plan_phys(rB.ζ)), "hist" => rB.hist, "outcomes" => Dict(pairs(oB))),
                          "prior_outcomes" => Dict(pairs(o0)))
end
println("=== A: the naive graph at the posterior width")
rA = solve_graph_A(V6)
results["A"] = Dict("zeta" => rA.ζ, "plan" => collect(plan_phys(rA.ζ)), "dz_sd" => rA.dz_sd, "sd" => rA.sd)

dst = joinpath(REPO, "data", "mock", "plan_experiment.json")
open(dst, "w") do io
    JSON.print(io, results)
end
println("wrote ", dst)
