# --- the belief where the Gaussian is wrong: a chain on the exact posterior --------------------
#
# The coverage test (docs/results.md, coverage) found the Gauss-Newton posterior calibrated on the
# identified meat sites and two to three times too narrow on the fire's ridge (a0, C_th, u_leak, the
# ignition spread), with C_th's width collapsing at the fuel add: the curvature at a point is not the
# posterior's width. Importance sampling around the Gaussian cannot repair that in 17 dimensions (a proposal
# widened threefold along four directions keeps one or two effective particles in a hundred), and sampling
# only the Gaussian's widest directions misses the failures, which sit in directions the Gaussian believes
# are narrow. So: plain random-walk Metropolis on the EXACT posterior (a plant solve per step, 0.2 s, the
# identified noise held at its estimate), preconditioned by the Gaussian's covariance, over every direction.
# Slower to mix than a chain on a few directions, and free of any rule about which directions matter.

"""
    ridge_chain(channels, prior, d, ph, z, V; k=length(z), steps=2000, burn=steps ÷ 4, scale=2.4 / sqrt(k), rng, thin=20)

The belief sampled by random-walk Metropolis on the exact posterior of the registry's readings under
`channels` (whitened as they stand: the identified noise and levels are fixed), the proposal shaped by the
Gaussian posterior `(z, V)` and its step adapted to a 0.3 acceptance during the burn-in. With `k` below
the dimension only the Gaussian's `k` widest directions move (a cheap truncation for tests); by default
every direction does. Returns the chain's mean and covariance (the Gaussian's where a direction did not
move), the draws in the full space (thinned), the acceptance rate and an effective sample size per
direction.
"""
function ridge_chain(channels, prior, d::GateData, ph::Physics, z, V; k::Int=length(z), steps::Int=2000, burn::Int=steps ÷ 4,
                     scale::Float64=2.4 / sqrt(k), rng=MersenneTwister(1), thin::Int=20)
    S0, J0, lay = physics_outputs(z, d, ph)
    _, _, w = assemble(channels, z, S0, J0, prior, lay)
    ok = .!isnan.(w)
    function logpost(zz)
        S = try
            physics_stack(zz, d, ph)[1]
        catch err
            err isa InterruptException && rethrow()
            return -Inf
        end
        a = assemble(channels, zz, S, J0, prior, lay)[1]
        val = -0.5 * sum(abs2, (w .- a)[ok]) - prior_terms(zz, prior)[1]     # not `lp`: a closure assigning to the
        return isfinite(val) ? val : -Inf                                     # chain's `lp` would overwrite it
    end
    F = eigen(Symmetric(Matrix(V)))
    idx = sortperm(F.values; rev=true)[1:min(k, length(z))]
    Vk = F.vectors[:, idx]
    λ = max.(F.values[idx], 0.0)
    kk = length(idx)
    s = zeros(kk)
    lp = logpost(z)
    draws = zeros(kk, steps)
    acc = 0
    acc_win = 0
    sc = scale
    for it in 1:steps
        s_new = s .+ sc .* sqrt.(λ) .* randn(rng, kk)
        lp_new = logpost(z .+ Vk * s_new)
        if log(rand(rng)) < lp_new - lp
            s, lp = s_new, lp_new
            acc += 1
            acc_win += 1
        end
        draws[:, it] = s
        # the step adapts during the burn-in toward a 0.3 acceptance (the posterior is narrower than the
        # Gaussian on some directions and wider on others; one fixed scale fits neither), then freezes
        if it <= burn && it % 25 == 0
            sc *= exp(2 * (acc_win / 25 - 0.3))
            acc_win = 0
        end
    end
    D = draws[:, (burn + 1):end]
    nD = size(D, 2)
    ms = vec(sum(D; dims=2)) ./ nD
    Dc = D .- ms
    Cs = Dc * Dc' ./ max(nD - 1, 1)
    V_rest = Symmetric(Matrix(V) - Vk * Diagonal(λ) * Vk')      # the Gaussian on the directions that did not move
    V_new = Matrix(Symmetric(V_rest + Vk * Cs * Vk'))
    z_new = z .+ Vk * ms
    ess = [_ess(D[i, :]) for i in 1:kk]
    kept = [z .+ Vk * D[:, j] for j in 1:thin:size(D, 2)]
    return (z=z_new, cov=V_new, sd=sqrt.(max.(diag(V_new), 0.0)), draws=kept, accept=acc / steps, ess=ess,
            directions=Vk, ridge_mean=ms, ridge_cov=Cs, ridge_sd_gauss=sqrt.(λ), step=sc)
end

"Effective sample size of a chain by its initial positive autocorrelation sequence."
function _ess(x::AbstractVector)
    n = length(x)
    n < 4 && return 1.0
    xc = x .- sum(x) / n
    v = dot(xc, xc) / n
    v <= 0 && return 1.0
    τ = 1.0
    for l in 1:(n ÷ 2)
        ρ = dot(xc[1:(n - l)], xc[(l + 1):n]) / n / v
        ρ <= 0 && break
        τ += 2ρ
    end
    return n / τ
end
