# The iterator on the graph: iterated Gauss-Newton with the graph owning
# the loop. Included by BrisketEstimator after stream.jl.
#
# The delta node's rules expand at the inbound messages' means and offer
# no expansion point, so the nonlinear model is kept exactly AFFINE (the
# built-in Linearization rule is then exact wherever it expands) and
# re-centered from inside `infer`: an `after_iteration` callback reads the
# current marginal, evaluates the true objective there, accepts or
# rejects the move, rebuilds the affine model at the new center, and
# re-centers the prior's local quadratic (both are data arrays the loop
# re-feeds every iteration). Damping is a Levenberg-Marquardt node on
# the graph, an identity delta node anchored at the current center whose
# precision is a data matrix the callback scales: a rejected move raises
# it and keeps the center, an accepted full move lowers it; the callback
# also halves the move when the full one does not lower the objective
# (the estimator's line search). One `infer`
# call is the whole solve; its last marginal is the posterior at the
# converged center once the damping is released.

export iterated_solve, IteratedProblem, lm_node, solve_channels_graph

"""
    IteratedProblem(varname; objective, rebuild!, n)

What the graph loop needs from a problem: the latent's name in the
model, `objective(c)` (the true objective at a point, solves only),
`rebuild!(c)` (the affine model and any moving data at the new center,
one Jacobian), and the latent's dimension.
"""
struct IteratedProblem{O,R}
    varname::Symbol
    objective::O
    rebuild!::R
    n::Int
end

const LM_ANCHOR = Ref{Vector{Float64}}(Float64[])
lm_node(x) = x .- LM_ANCHOR[]                      # the Levenberg-Marquardt node: affine, exact

"""
    iterated_solve(problem, run_infer; c0, iters=40, tol=1e-4, ftol=1e-3, λ0=0.5, verbose=false)

`run_infer(Λ_lm, callbacks)` must call `infer` on the problem's model
with the LM precision matrix `Λ_lm` passed as DATA and the callbacks
attached, and return the result. The loop lives in the callbacks; this
function only sets the center, evaluates, and reads the posterior back.
Returns (z, F, cov, hist, converged).
"""
function iterated_solve(p::IteratedProblem, run_infer; c0, iters::Int=40, tol::Float64=1e-4,
                        ftol::Float64=1e-3, λ0::Float64=0.5, verbose::Bool=false)
    c = copy(c0)
    LM_ANCHOR[] = copy(c)
    p.rebuild!(c)
    Fc = p.objective(c)
    λ = 0.0
    Λ_lm = Matrix(1e-9 * I, p.n, p.n)                 # the data matrix the callback scales
    last = Ref{Any}(nothing)
    hist = NamedTuple[]
    converged = Ref(false)
    on_update = (event) -> (event.variable_name === p.varname && (last[] = event.update); nothing)
    after = (event) -> begin
        t0 = time()
        m = collect(mean(last[]))
        step = m .- c
        # the graph gives the direction; the callback picks the length (the estimator's line search)
        α = 1.0
        trial(v) = try p.objective(v) catch err; err isa InterruptException ? rethrow() : Inf end
        Fn = norm(step) > 20 ? Inf : trial(c .+ step)
        while (!isfinite(Fn) || Fn > Fc) && α > 1e-3
            α /= 2
            Fn = trial(c .+ α .* step)
        end
        accepted = isfinite(Fn) && Fn <= Fc
        if accepted
            c .= c .+ α .* step
            LM_ANCHOR[] = copy(c)
            p.rebuild!(c)
            dropped = Fc - Fn
            Fc = Fn
            α == 1.0 && (λ = λ / 3 < λ0 / 10 ? 0.0 : λ / 3)
            if α * norm(step) < tol || dropped < ftol
                converged[] = true
                event.stop_iteration = true
            end
        else
            λ = λ == 0.0 ? λ0 : 10λ
            λ > 1e9 && (event.stop_iteration = true)
        end
        Λ_lm .= Matrix((λ + 1e-9) * I, p.n, p.n)
        push!(hist, (it=event.iteration, F=Fc, step=norm(step), alpha=α, damping=λ, accepted=accepted, wall=time() - t0))
        verbose && @printf("    graph it %2d  F %10.4f  |step| %.3g  alpha %.3g  damping %.2g  %s\n", event.iteration, Fc, norm(step), α, λ,
                           accepted ? "accepted" : "rejected")
        verbose && flush(stdout)
        nothing
    end
    run_infer(Λ_lm, (on_marginal_update=on_update, after_iteration=after), iters)
    # the posterior at the converged center, damping released: one pass
    Λ_lm .= Matrix(1e-9 * I, p.n, p.n)
    res = run_infer(Λ_lm, (on_marginal_update=on_update,), 1)
    q = last[]
    return (z=c, F=Fc, mean=collect(mean(q)), cov=Matrix(cov(q)), hist=hist, converged=converged[])
end

# --- the production solve on the graph -------------------------------------------------

@model function cook_graph_lm(w, mp, Λp, lm_obs, Λ_lm)
    z ~ MvNormalMeanPrecision(mp, Λp)
    for i in 1:length(w)
        pred[i] := reading(z, i)
        w[i] ~ Normal(mean = pred[i], var = 1.0)
    end
    lm := lm_node(z)
    lm_obs ~ MvNormalMeanPrecision(lm, Λ_lm)
end
const GRAPH_META_LM = @meta begin
    reading() -> Linearization()
    lm_node() -> Linearization()
end

"""
    solve_channels_graph(channels, prior, d, ph; c0, iters, tol, ftol, bp_iters, verbose)

The fixed-noise solve of `solve_map_channels` with the graph owning the
loop (`iterated_solve`): the same affine node per whitened residual, the
prior's local quadratic and the LM precision as data the loop re-feeds,
the objective and the line search in the callback. Returns what
`_solve_fixed_noise` returns.
"""
function solve_channels_graph(channels, prior, d::GateData, ph::Physics; c0, iters::Int=40, tol::Float64=1e-4,
                              ftol::Float64=1e-3, bp_iters::Int=20, verbose::Bool=false)
    n = length(prior)
    mp = zeros(n); Λp = zeros(n, n)
    w_ref = Ref{Vector{Float64}}(Float64[])
    lay_ref = Ref{Any}(nothing)
    function objective(c)
        S, lay = physics_stack(c, d, ph)
        a, _, w = assemble(channels, c, S, zeros(length(S), n), prior, lay)
        return _objective(a, w, c, prior)
    end
    function rebuild!(c)
        S, J, lay = physics_outputs(c, d, ph)
        a, B, w = assemble(channels, c, S, J, prior, lay)
        any(isnan, w) && throw(ArgumentError("every channel in a fit must carry data"))
        AFF[] = Affine(a, B, copy(c))
        w_ref[] = w
        lay_ref[] = lay
        _, g, h = prior_terms(c, prior)
        mp .= c .- g ./ h
        Λp .= Matrix(Diagonal(h))
    end
    prob = IteratedProblem(:z, objective, rebuild!, n)
    run_infer(Λ_lm, callbacks, its) = infer(model = cook_graph_lm(), data = (w = w_ref[], mp = mp, Λp = Λp, lm_obs = zeros(n), Λ_lm = Λ_lm),
                                            meta = GRAPH_META_LM, initialization = init_z(mp, Λp), iterations = its,
                                            callbacks = callbacks, returnvars = (z = KeepLast(),))
    r = iterated_solve(prob, run_infer; c0, iters, tol, ftol, verbose)
    a, B = AFF[].a, AFF[].B
    mis = misfit(channels, a, B, w_ref[], r.cov)
    return (z=r.z, mean=r.mean, cov=r.cov, F=r.F, hist=r.hist, misfit=mis, converged=r.converged)
end
