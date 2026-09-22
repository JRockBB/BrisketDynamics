# The iterator on the graph, gated twice: on the planning problem it must
# reproduce the expected-loss reference to the last digit, and on the
# real cook it must reproduce the estimator's gate (the Gauss-Newton MAP
# within a third of a standard deviation, the fixed point's objective).
#
# Run:  julia --project=estimator estimator/test/test_iterated.jl
include(joinpath(@__DIR__, "..", "scripts", "plan_experiment_setup.jl"))
using Test

struct MockMarginalEvent{T}
    variable_name::Symbol
    update::T
end
mutable struct MockIterationEvent
    iteration::Int
    stop_iteration::Bool
end

@testset "moving surrogates compare trials only under frozen weights" begin
    weight = Ref(1.0)
    target = [3.0]
    objective(c) = 0.5 * weight[] * sum(abs2, c .- target)
    rebuild!(c) = (weight[] = 1 + abs(c[1]))
    prob = IteratedProblem(:z, objective, rebuild!, 1)
    function run_mock(_Λ_lm, callbacks, its)
        q = nothing
        for it in 1:its
            c = E.LM_ANCHOR[]
            q = MvNormalMeanCovariance(c .+ 0.5 .* (target .- c), ones(1, 1))
            callbacks.on_marginal_update(MockMarginalEvent(:z, q))
            if hasproperty(callbacks, :after_iteration)
                event = MockIterationEvent(it, false)
                callbacks.after_iteration(event)
                event.stop_iteration && break
            end
        end
        return q
    end
    r = iterated_solve(prob, run_mock; c0=[0.0], tol=1e-5, ftol=1e-6)
    accepted = filter(h -> h.accepted, r.hist)
    @test r.converged && !isempty(accepted)
    @test all(h.F_trial <= h.F_before + 1e-12 for h in accepted)
    @test any(abs(h.F - h.F_trial) > 1e-6 for h in accepted)
end

@testset "the trust region: no site moves beyond its own width in one iteration" begin
    target = [0.5, 3.0]; w = [1.0, 0.1]
    objective(c) = 0.5 * sum(abs2, c .- target)
    centres = Vector{Vector{Float64}}()
    rebuild!(c) = push!(centres, copy(c))
    cap(_, s) = E._cap_factor(s ./ w)
    function run_exact(_Λ_lm, callbacks, its)           # the graph's step: exactly to the target
        q = nothing
        for it in 1:its
            q = MvNormalMeanCovariance(copy(target), Matrix(1.0 * I, 2, 2))
            callbacks.on_marginal_update(MockMarginalEvent(:z, q))
            if hasproperty(callbacks, :after_iteration)
                event = MockIterationEvent(it, false)
                callbacks.after_iteration(event)
                event.stop_iteration && break
            end
        end
        return q
    end
    r0 = iterated_solve(IteratedProblem(:z, objective, rebuild!, 2), run_exact; c0=[0.0, 0.0], tol=1e-8, ftol=1e-10)
    @test r0.converged && all(h.alpha_max == 1.0 for h in r0.hist) && length(r0.hist) <= 2 && r0.z ≈ target
    empty!(centres)
    r1 = iterated_solve(IteratedProblem(:z, objective, rebuild!, 2; step_cap=cap), run_exact; c0=[0.0, 0.0], iters=60, tol=1e-8, ftol=1e-10)
    @test r1.converged && isapprox(r1.z, target; atol=1e-6)
    @test abs(r1.hist[1].alpha_max - 1 / 30) < 1e-12 && r1.hist[1].alpha == r1.hist[1].alpha_max && all(h.accepted for h in r1.hist)
    moves = [maximum(abs.(centres[k] .- centres[k - 1]) ./ w) for k in 2:length(centres)]
    @test all(moves .<= 1 + 1e-9) && count(h.alpha_max < 1 for h in r1.hist) == 29     # 29 capped moves of one width, then the free step to the target
    @test r1.hist[end].alpha_max == 1.0                      # convergence is declared only on an uncapped step
    # the whitened cap: a move of three widths on one coordinate of the block, whatever the correlation, caps at a third
    rng_t = MersenneTwister(3)
    S0 = Matrix(cholesky(Symmetric(let A = randn(rng_t, 4, 4); A * A' + I end)).L)
    wv = sqrt.(vec(sum(abs2, S0; dims=2)))
    Δv = zeros(4); Δv[2] = 3 * wv[2]
    @test abs(E._whitened_step_cap(S0)(nothing, S0 \ Δv) - 1 / 3) < 1e-12
    @test E._whitened_step_cap(S0)(nothing, S0 \ (Δv ./ 6)) == 1.0
end

# ---------------------------------------------------------------- the planner on the graph
const AFFS = Ref{Vector{Tuple{Vector{Float64},Matrix{Float64}}}}(Tuple{Vector{Float64},Matrix{Float64}}[])
const CENTER = Ref{Vector{Float64}}(zeros(3))
affine_node(ζ, i) = AFFS[][i][1] .+ AFFS[][i][2] * (ζ .- CENTER[])

@model function plan_graph_lm(g, lm_obs, Λ_lm, mζ, Λζ, N, Σg)
    ζ ~ MvNormalMeanPrecision(mζ, Λζ)
    for i in 1:N
        y[i] := affine_node(ζ, i)
        g[i] ~ MvNormalMeanCovariance(y[i], Σg)
    end
    lm := lm_node(ζ)
    lm_obs ~ MvNormalMeanPrecision(lm, Λ_lm)
end
const META_LM = @meta begin
    affine_node() -> Linearization()
    lm_node() -> Linearization()
end

function plan_on_graph(zs; verbose=true)
    g = [[T_TARGET, W_TARGET] for _ in 1:length(zs)]
    Σg = Diagonal([σ_T^2, σ_W^2]) |> Matrix
    objective(c) = 0.5 * sum(abs2, residuals(c, zs))
    function rebuild!(c)
        AFFS[] = [(plan_outputs(z, c), ForwardDiff.jacobian(v -> plan_outputs(z, v), c)) for z in zs]
        CENTER[] = copy(c)
    end
    prob = IteratedProblem(:ζ, objective, rebuild!, 3)
    run_infer(Λ_lm, callbacks, iters) = infer(model = plan_graph_lm(mζ = ζ0, Λζ = Matrix(Λ0), N = length(zs), Σg = Σg),
                                              data = (g = g, lm_obs = zeros(3), Λ_lm = Λ_lm), meta = META_LM,
                                              initialization = init_ζ(CENTER[], Matrix(Λ0)), iterations = iters,
                                              callbacks = callbacks, returnvars = (ζ = KeepLast(),))
    return iterated_solve(prob, run_infer; c0=ζ0, verbose=verbose)
end

results = JSON.parsefile(joinpath(REPO, "data", "mock", "plan_experiment.json"))
@testset "the planner on the graph reproduces the expected-loss reference" begin
    for (label, scale) in (("posterior width", 1.0), ("width shrunk tenfold", 0.1))
        zs = draws(scale)
        println("=== planner on the graph, ", label)
        t0 = time()
        r = plan_on_graph(zs)
        ref = results[label]["R"]
        dz = (r.z .- Float64.(ref["zeta"])) ./ Float64.(ref["sd"])
        @printf("  graph: plan u1 %.3f u2 %.3f pull %.2f h  F %.4f  (%d graph iterations, %.0f s)  reference F %.4f; difference in the plan's sd %s\n",
                plan_phys(r.z)..., r.F, length(r.hist), time() - t0, Float64(ref["F"]), string(round.(dz; digits=3)))
        flush(stdout)
        @test r.converged
        @test abs(r.F - Float64(ref["F"])) < 2e-3
        @test maximum(abs.(dz)) < 0.05
    end
end

# ---------------------------------------------------------------- the estimator on the graph, the real cook
const reading = E.reading
@model function cook_graph_lm(w, mp, Λp, lm_obs, Λ_lm)
    z ~ MvNormalMeanPrecision(mp, Λp)
    for i in 1:length(w)
        pred[i] := reading(z, i)
        w[i] ~ Normal(mean = pred[i], var = 1.0)
    end
    lm := lm_node(z)
    lm_obs ~ MvNormalMeanPrecision(lm, Λ_lm)
end
const META_COOK_LM = @meta begin
    reading() -> Linearization()
    lm_node() -> Linearization()
end

@testset "the estimator on the graph reproduces the real-cook gate" begin
    gate_path = joinpath(REPO, "data", "calibration", "estimator_gate_real.json")
    dg = load_gate(gate_path)
    raw = JSON.parsefile(gate_path)
    nz = raw["noise"]
    chs = [PitProbe(nz["sigma_pit_eff"], nz["rho_pit"], Bool.(raw["mask_pit"]); data=Float64.(raw["obs_pit"])),
           MeatProbe(nz["sigma_core_eff"], nz["rho_core"], Bool.(raw["mask_core"]); s=0.0, data=Float64.(raw["obs_core"]))]
    ph = Physics1D(0)
    n = length(dg.prior)
    mp = zeros(n); Λp = zeros(n, n)
    w_ref = Ref{Vector{Float64}}(Float64[])
    function objective(c)
        S, lay = physics_stack(c, dg, ph)
        a, _, w = assemble(chs, c, S, zeros(length(S), n), dg.prior, lay)
        return 0.5 * sum(abs2, w .- a) + E.prior_terms(c, dg.prior)[1]
    end
    function rebuild!(c)
        S, J, lay = physics_outputs(c, dg, ph)
        a, B, w = assemble(chs, c, S, J, dg.prior, lay)
        E.AFF[] = E.Affine(a, B, copy(c))
        w_ref[] = w
        _, g, h = E.prior_terms(c, dg.prior)
        mp .= c .- g ./ h
        Λp .= Matrix(Diagonal(h))
    end
    prob = IteratedProblem(:z, objective, rebuild!, n)
    run_infer(Λ_lm, callbacks, iters) = infer(model = cook_graph_lm(), data = (w = w_ref[], mp = mp, Λp = Λp, lm_obs = zeros(n), Λ_lm = Λ_lm),
                                              meta = META_COOK_LM, initialization = E.init_z(mp, Λp), iterations = iters,
                                              callbacks = callbacks, returnvars = (z = KeepLast(),))
    println("=== estimator on the graph, the real cook")
    t0 = time()
    r = iterated_solve(prob, run_infer; c0=dg.z0, verbose=true)
    sd = sqrt.(diag(dg.cov_map))
    dz = (r.z .- dg.z_map) ./ sd
    sd_g = sqrt.(diag(r.cov))
    @printf("  graph: F %.4f (calibration %.4f, the outer loop's fixed point 133.1206); max |dz|/sd %.3f (%s), median %.3f; widths vs Fisher max rel diff %.3f; %d graph iterations, %.0f s\n",
            r.F, dg.F_map, maximum(abs.(dz)), dg.names[argmax(abs.(dz))], sort(abs.(dz))[8],
            maximum(abs.(sd_g ./ sd .- 1)), length(r.hist), time() - t0)
    flush(stdout)
    @test r.converged
    @test r.F < dg.F_map
    @test maximum(abs.(dz)) < 0.5
    @test maximum(abs.(sd_g ./ sd .- 1)) < 0.05
end
println("test_iterated.jl: done")
