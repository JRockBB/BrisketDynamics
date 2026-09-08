# The planning graph the estimator's way: per-draw AFFINE nodes built at
# the current plan from a ForwardDiff Jacobian (three columns per draw),
# inside the damped outer loop. The delta node's rules expand at the
# inbound message's mean and offer no expansion point, so iterated
# linearization has to be supplied from outside; this is that supply. If
# it reproduces the reference optimum, the planning graph's shortfall
# was the expansion point and nothing else.
#
# Run:  julia --project=estimator estimator/plan_experiment_affine.jl
# Appends to data/mock/plan_experiment.json under "B_affine".
include(joinpath(@__DIR__, "plan_experiment_setup.jl"))

# the affine model of draw i around c: y ≈ a_i + B_i (ζ − c)
const AFFS = Ref{Vector{Tuple{Vector{Float64},Matrix{Float64}}}}(Tuple{Vector{Float64},Matrix{Float64}}[])
const CENTER = Ref{Vector{Float64}}(zeros(3))
affine_node(ζ, i) = AFFS[][i][1] .+ AFFS[][i][2] * (ζ .- CENTER[])

@model function plan_graph_affine(g, mζ, Λζ, N, Σg)
    ζ ~ MvNormalMeanPrecision(mζ, Λζ)
    for i in 1:N
        y[i] := affine_node(ζ, i)
        g[i] ~ MvNormalMeanCovariance(y[i], Σg)
    end
end
const META_AFF = @meta begin
    affine_node() -> Linearization()
end

function build_affine!(c, zs)
    AFFS[] = [(plan_outputs(z, c), ForwardDiff.jacobian(v -> plan_outputs(z, v), c)) for z in zs]
    CENTER[] = copy(c)
    return nothing
end

function damped_prior(c, λ)
    Λd = Matrix(Λ0) + λ * I
    g = Matrix(Λ0) * (c .- ζ0)
    return c .- Λd \ g, Λd
end

function graph_step_affine(c, zs, λ)
    g = [[T_TARGET, W_TARGET] for _ in 1:length(zs)]
    Σg = Diagonal([σ_T^2, σ_W^2]) |> Matrix
    mp, Λp = damped_prior(c, λ)
    res = infer(model = plan_graph_affine(mζ = mp, Λζ = Λp, N = length(zs), Σg = Σg), data = (g = g,),
                meta = META_AFF, initialization = init_ζ(c, Λp), iterations = 5,
                returnvars = (ζ = KeepLast(),))
    return collect(mean(res.posteriors[:ζ]))
end

expected_loss(ζ, zs) = try 0.5 * sum(abs2, residuals(ζ, zs)) catch; Inf end

function solve_affine(zs; iters=25, verbose=true)
    c = copy(ζ0)
    F = expected_loss(c, zs)
    λ = 0.0; λ0 = 0.5
    nsolve = length(zs)
    for k in 1:iters
        build_affine!(c, zs)                     # one dual solve per draw: the Jacobian at c
        nsolve += 3 * length(zs)
        accepted = false; cn = c; Fn = F; step = zeros(3); α = 1.0
        for attempt in 1:8
            m = graph_step_affine(c, zs, λ)      # the graph: a linear-Gaussian solve, no physics
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
    println("=== B with affine nodes built at the current plan, damped outer loop, ", label)
    t0 = time()
    r = solve_affine(zs)
    o = outcomes(r.ζ, zs)
    ref = results[label]["R"]
    @printf("  B_affine plan u1 %.3f u2 %.3f pull %.2f h  F %.3f  (%d solve-equivalents, %.0f s)  outcomes: T %.1f ± %.1f C, W %.3f ± %.3f\n",
            plan_phys(r.ζ)..., r.F, r.nsolve, time() - t0, o.T_mean, o.T_sd, o.W_mean, o.W_sd)
    @printf("  reference R: plan %s  F %.3f;  B_affine - R in the plan's own sd: %s\n",
            string(round.(Float64.(ref["plan"]); digits=3)), Float64(ref["F"]),
            string(round.((r.ζ .- Float64.(ref["zeta"])) ./ Float64.(ref["sd"]); digits=2)))
    flush(stdout)
    results[label]["B_affine"] = Dict("zeta" => r.ζ, "F" => r.F, "nsolve" => r.nsolve, "plan" => collect(plan_phys(r.ζ)),
                                      "outcomes" => Dict(pairs(o)))
end
open(path, "w") do io
    JSON.print(io, results)
end
println("wrote ", path)
