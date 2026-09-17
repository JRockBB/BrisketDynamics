# The planning graph as a step generator: the per-draw Unscented graph of
# plan_experiment.jl inside the estimator's outer loop (the graph's
# posterior mean as a step from the current plan, line-searched on the
# expected loss), with the graph initialized at the current plan. Two
# questions: does the outer loop reach the expected-loss optimum the
# reference found, and does the graph's evaluation point follow the
# current plan at all (a step taken with the initialization at c against
# the same step with the initialization at the prior mean).
#
# Run:  julia --project=estimator estimator/scripts/plan_experiment_outer.jl
# Appends to data/mock/plan_experiment.json under "B_outer".
include(joinpath(@__DIR__, "plan_experiment_setup.jl"))

function graph_step(c, zs; iters=1, init_at=c)
    ZS[] = zs
    g = [[T_TARGET, W_TARGET] for _ in 1:length(zs)]
    Σg = Diagonal([σ_T^2, σ_W^2]) |> Matrix
    Λi = Matrix(Λ0) .* 4.0                       # a tight start at init_at so the sigma points sit there
    res = infer(model = plan_graph_B(mζ = ζ0, Λζ = Matrix(Λ0), N = length(zs), Σg = Σg), data = (g = g,),
                meta = META_B, initialization = init_ζ(init_at, Λi), iterations = iters,
                returnvars = (ζ = KeepLast(),))
    return collect(mean(res.posteriors[:ζ]))
end

expected_loss(ζ, zs) = try 0.5 * sum(abs2, residuals(ζ, zs)) catch; Inf end

function solve_outer(zs; iters=12, inner=1, verbose=true)
    c = copy(ζ0)
    F = expected_loss(c, zs)
    nsolve = 0
    for k in 1:iters
        m = graph_step(c, zs; iters=inner)
        nsolve += inner * length(zs) * 7
        step = m - c
        α = 1.0; cn = c + step; Fn = expected_loss(cn, zs); nsolve += length(zs)
        while (!isfinite(Fn) || Fn > F) && α > 1e-3
            α /= 2; cn = c + α * step; Fn = expected_loss(cn, zs); nsolve += length(zs)
        end
        verbose && @printf("    outer %2d  F %.4f -> %.4f  |step| %.3g  alpha %.3g  plan (u1 %.3f, u2 %.3f, pull %.2f h)\n",
                           k, F, Fn, norm(step), α, plan_phys(cn)...)
        flush(stdout)
        (isfinite(Fn) && Fn <= F) || break
        done = α * norm(step) < 1e-3 || F - Fn < 1e-3
        c, F = cn, Fn
        done && break
    end
    return (ζ=c, F=F, nsolve=nsolve)
end

zs = draws(1.0)
println("=== does the graph's evaluation point follow the current plan?")
c_far = [log(0.10 / 0.90), log(0.55 / 0.45), log(11.5)]     # near the reference optimum
s_prior = graph_step(c_far, zs; init_at=ζ0) - c_far
s_here = graph_step(c_far, zs; init_at=c_far) - c_far
@printf("  step from c with the graph initialized at the prior mean: %s\n  step from c with the graph initialized at c:              %s\n",
        string(round.(s_prior; digits=3)), string(round.(s_here; digits=3)))
flush(stdout)

println("=== B with the outer loop, posterior width")
rBo = solve_outer(zs)
oBo = outcomes(rBo.ζ, zs)
@printf("  B_outer plan u1 %.3f u2 %.3f pull %.2f h  F %.3f  (%d solves)  outcomes: T %.1f ± %.1f C, W %.3f ± %.3f\n",
        plan_phys(rBo.ζ)..., rBo.F, rBo.nsolve, oBo.T_mean, oBo.T_sd, oBo.W_mean, oBo.W_sd)
flush(stdout)

path = joinpath(REPO, "data", "mock", "plan_experiment.json")
results = isfile(path) ? JSON.parsefile(path) : Dict{String,Any}()
results["B_outer"] = Dict("zeta" => rBo.ζ, "F" => rBo.F, "nsolve" => rBo.nsolve, "plan" => collect(plan_phys(rBo.ζ)),
                          "outcomes" => Dict(pairs(oBo)),
                          "step_init_prior" => s_prior, "step_init_here" => s_here, "c_far" => c_far)
open(path, "w") do io
    JSON.print(io, results)
end
println("wrote ", path)
