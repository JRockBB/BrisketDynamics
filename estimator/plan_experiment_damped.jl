# The planning graph as a step generator inside the estimator's damped
# outer loop: the per-draw Unscented graph of plan_experiment_setup.jl,
# the prior node carrying a Levenberg-Marquardt damping around the
# current plan (the local quadratic of the Gaussian prior at c with λ
# added to its precision, exactly as graph_posterior_affine does for the
# estimator), the graph initialized at c, the step line-searched on the
# expected loss, λ raised on a rejected step and lowered on an accepted
# full step. The question left open by plan_experiment_outer.jl: with
# the damping the estimator needed, does the graph reach the
# expected-loss optimum the reference found?
#
# Run:  julia --project=estimator estimator/plan_experiment_damped.jl
# Appends to data/mock/plan_experiment.json under "B_damped".
include(joinpath(@__DIR__, "plan_experiment_setup.jl"))

# the Gaussian prior N(ζ0, Λ0⁻¹) re-expressed as the damped local quadratic around c
function damped_prior(c, λ)
    Λd = Matrix(Λ0) + λ * I
    g = Matrix(Λ0) * (c .- ζ0)                  # the prior's gradient at c
    return c .- Λd \ g, Λd
end

function graph_step_damped(c, zs, λ; iters=2)
    ZS[] = zs
    g = [[T_TARGET, W_TARGET] for _ in 1:length(zs)]
    Σg = Diagonal([σ_T^2, σ_W^2]) |> Matrix
    mp, Λp = damped_prior(c, λ)
    res = infer(model = plan_graph_B(mζ = mp, Λζ = Λp, N = length(zs), Σg = Σg), data = (g = g,),
                meta = META_B, initialization = init_ζ(c, Λp), iterations = iters,
                returnvars = (ζ = KeepLast(),))
    return collect(mean(res.posteriors[:ζ]))
end

expected_loss(ζ, zs) = try 0.5 * sum(abs2, residuals(ζ, zs)) catch; Inf end

function solve_damped(zs; iters=25, verbose=true)
    c = copy(ζ0)
    F = expected_loss(c, zs)
    λ = 0.0; λ0 = 0.5
    nsolve = length(zs)
    for k in 1:iters
        accepted = false; cn = c; Fn = F; step = zeros(3); α = 1.0
        for attempt in 1:8
            m = graph_step_damped(c, zs, λ)
            nsolve += 2 * length(zs) * 7
            step = m - c
            α = 1.0; cn = c + step; Fn = expected_loss(cn, zs); nsolve += length(zs)
            while (!isfinite(Fn) || Fn > F) && α > 1e-2
                α /= 2; cn = c + α * step; Fn = expected_loss(cn, zs); nsolve += length(zs)
            end
            if isfinite(Fn) && Fn <= F
                accepted = true
                α == 1.0 && (λ = λ / 3 < λ0 / 10 ? 0.0 : λ / 3)
                break
            end
            norm(step) < 1e-3 && break
            λ = λ == 0.0 ? λ0 : 10λ
        end
        verbose && @printf("    outer %2d  F %.4f -> %.4f  |step| %.3g  alpha %.3g  damping %.2g  plan (u1 %.3f, u2 %.3f, pull %.2f h)\n",
                           k, F, Fn, norm(step), α, λ, plan_phys(cn)...)
        flush(stdout)
        accepted || break
        done = α * norm(step) < 1e-3 || F - Fn < 1e-3
        c, F = cn, Fn
        done && break
    end
    return (ζ=c, F=F, nsolve=nsolve)
end

path = joinpath(REPO, "data", "mock", "plan_experiment.json")
results = isfile(path) ? JSON.parsefile(path) : Dict{String,Any}()
for (label, scale) in (("posterior width", 1.0), ("width shrunk tenfold", 0.1))
    zs = draws(scale)
    println("=== B, damped outer loop, ", label)
    r = solve_damped(zs)
    o = outcomes(r.ζ, zs)
    ref = results[label]["R"]
    @printf("  B_damped plan u1 %.3f u2 %.3f pull %.2f h  F %.3f  (%d solves)  outcomes: T %.1f ± %.1f C, W %.3f ± %.3f\n",
            plan_phys(r.ζ)..., r.F, r.nsolve, o.T_mean, o.T_sd, o.W_mean, o.W_sd)
    @printf("  reference R: plan %s  F %.3f;  B_damped - R in the plan's own sd: %s\n",
            string(round.(Float64.(ref["plan"]); digits=3)), Float64(ref["F"]),
            string(round.((r.ζ .- Float64.(ref["zeta"])) ./ Float64.(ref["sd"]); digits=2)))
    flush(stdout)
    results[label]["B_damped"] = Dict("zeta" => r.ζ, "F" => r.F, "nsolve" => r.nsolve, "plan" => collect(plan_phys(r.ζ)),
                                      "outcomes" => Dict(pairs(o)))
end
open(path, "w") do io
    JSON.print(io, results)
end
println("wrote ", path)
