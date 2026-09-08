# The sample-based node: the per-draw planning graph of
# plan_experiment_setup.jl with the CVI method on the delta nodes
# (samples through the physics, natural-gradient updates of the message;
# the installed ReactiveMP's `CVI`, deprecated upstream in favor of
# CVIProjection, which needs a package this environment lacks), inside
# the damped outer loop of plan_experiment_damped.jl. The question: does
# a sample-based message reach the expected-loss optimum where the
# linearizing one did or did not, and at what cost in solves.
#
# Run:  julia --project=estimator estimator/plan_experiment_cvi.jl
# Appends to data/mock/plan_experiment.json under "B_cvi".
include(joinpath(@__DIR__, "plan_experiment_setup.jl"))
using Random

const N_SAMPLES = 20
cvi_opt(new_λ, λ, ∇) = λ .+ 0.1 .* ∇                       # a plain natural-gradient step
const META_CVI = @meta begin
    plan_node() -> CVI(MersenneTwister(11), N_SAMPLES, 20, cvi_opt)
end

function damped_prior(c, λ)
    Λd = Matrix(Λ0) + λ * I
    g = Matrix(Λ0) * (c .- ζ0)
    return c .- Λd \ g, Λd
end

function graph_step_cvi(c, zs, λ; iters=2)
    ZS[] = zs
    g = [[T_TARGET, W_TARGET] for _ in 1:length(zs)]
    Σg = Diagonal([σ_T^2, σ_W^2]) |> Matrix
    mp, Λp = damped_prior(c, λ)
    res = infer(model = plan_graph_B(mζ = mp, Λζ = Λp, N = length(zs), Σg = Σg), data = (g = g,),
                meta = META_CVI, initialization = init_ζ(c, Λp), iterations = iters,
                returnvars = (ζ = KeepLast(),))
    return collect(mean(res.posteriors[:ζ]))
end

expected_loss(ζ, zs) = try 0.5 * sum(abs2, residuals(ζ, zs)) catch; Inf end

function solve_cvi(zs; iters=20, verbose=true)
    c = copy(ζ0)
    F = expected_loss(c, zs)
    λ = 0.0; λ0 = 0.5
    nsolve = length(zs)
    for k in 1:iters
        accepted = false; cn = c; Fn = F; step = zeros(3); α = 1.0
        for attempt in 1:6
            m = graph_step_cvi(c, zs, λ)
            nsolve += 2 * length(zs) * N_SAMPLES
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
zs = draws(1.0)
println("=== B with CVI nodes, damped outer loop, posterior width")
t0 = time()
r = solve_cvi(zs)
o = outcomes(r.ζ, zs)
ref = results["posterior width"]["R"]
@printf("  B_cvi plan u1 %.3f u2 %.3f pull %.2f h  F %.3f  (%d solves, %.0f s)  outcomes: T %.1f ± %.1f C, W %.3f ± %.3f\n",
        plan_phys(r.ζ)..., r.F, r.nsolve, time() - t0, o.T_mean, o.T_sd, o.W_mean, o.W_sd)
@printf("  reference R: plan %s  F %.3f;  B_cvi - R in the plan's own sd: %s\n",
        string(round.(Float64.(ref["plan"]); digits=3)), Float64(ref["F"]),
        string(round.((r.ζ .- Float64.(ref["zeta"])) ./ Float64.(ref["sd"]); digits=2)))
results["posterior width"]["B_cvi"] = Dict("zeta" => r.ζ, "F" => r.F, "nsolve" => r.nsolve, "plan" => collect(plan_phys(r.ζ)),
                                           "outcomes" => Dict(pairs(o)), "n_samples" => N_SAMPLES)
open(path, "w") do io
    JSON.print(io, results)
end
println("wrote ", path)
