# --- the belief along the ridge: a chain where the Gaussian is wrong -----------------------------
#
# The coverage test (docs/results.md, coverage) found the Gauss-Newton posterior calibrated on the
# identified meat sites and two to three times too narrow on the fire's ridge (a0, C_th, u_leak, the
# ignition spread): the curvature at a point on a ridge is not the posterior's width along it. Importance
# sampling around the Gaussian cannot repair that in 17 dimensions (a proposal widened threefold along four
# directions keeps one or two effective particles in a hundred). What can: a random-walk Metropolis chain on
# the EXACT posterior (a plant solve per step, the identified noise held at its estimate) restricted to the
# k widest directions of the Gaussian, the other directions kept at the Gaussian conditional. The chain's
# mean and covariance along those directions replace the Gaussian's there; everything else stays. The
# result is a belief whose fire block has the ridge's real width and skew, at k-dimensional cost.

"""
    ridge_chain(channels, prior, d, ph, z, V; k=5, steps=500, burn=steps ÷ 4, scale=2.4 / sqrt(k), rng, thin=10)

The belief along the `k` widest directions of the Gaussian posterior `(z, V)`, sampled by random-walk
Metropolis on the exact posterior of the registry's readings under `channels` (whitened as they stand: the
identified noise and levels are fixed), the other directions at the Gaussian conditional. Returns the
corrected mean and covariance (the Gaussian with its ridge block replaced by the chain's), the chain's
draws in the full space (thinned), the acceptance rate and an effective sample size per direction.
"""
function ridge_chain(channels, prior, d::GateData, ph::Physics, z, V; k::Int=5, steps::Int=500, burn::Int=steps ÷ 4,
                     scale::Float64=2.4 / sqrt(k), rng=MersenneTwister(1), thin::Int=10)
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
        # the step adapts during the burn-in toward a 0.3 acceptance (the posterior along a ridge is narrower
        # than the Gaussian on some directions and wider on others; one fixed scale fits neither), then freezes
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
    V_rest = Symmetric(Matrix(V) - Vk * Diagonal(λ) * Vk')      # the Gaussian without its ridge directions
    V_new = Matrix(Symmetric(V_rest + Vk * Cs * Vk'))
    z_new = z .+ Vk * ms
    ess = [_ess(D[i, :]) for i in 1:kk]
    kept = [z .+ Vk * D[:, j] for j in 1:thin:size(D, 2)]
    return (z=z_new, cov=V_new, sd=sqrt.(max.(diag(V_new), 0.0)), draws=kept, accept=acc / steps, ess=ess,
            directions=Vk, ridge_mean=ms, ridge_cov=Cs, ridge_sd_gauss=sqrt.(λ))
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
