# --- the noise parameters' uncertainty in the physics widths ---------------------------------------
#
# The coverage test (docs/results.md, 2026-09-11) found the Gauss-Newton widths a quarter too narrow on the
# bench and calibrated when every channel's noise is known: the identified σ, ρ and level are plugged in with
# no uncertainty carried. The law of total variance repairs that at the MAP without a new solve: draw the noise
# parameters from the AR node's posterior, re-whiten, and take the Gauss-Newton mean and covariance under each
# draw; the belief's covariance is the mean of the covariances plus the covariance of the means.

"""
    noise_marginal_cov(channels, prior, d, ph, z, noise_post; K=8, rng) -> (cov, sd, cov_mean, spread, draws)

The posterior covariance at the MAP `z` with the noise parameters marginalised by Monte Carlo: `K` draws of
each identified channel's (σ, ρ) from `noise_post` (the AR node's posterior: `σ`, `σ_sd`, `ρ`, `ρ_sd` per
channel index, as `solve_map_channels` returns), each re-whitening the channel (its level kept), a Gauss-Newton
step and covariance at `z` under each, and the law of total variance across them. One physics solve and
Jacobian, `K` linear solves.
"""
function noise_marginal_cov(channels, prior, d::GateData, ph::Physics, z, noise_post; K::Int=8, rng=MersenneTwister(1))
    S, J, lay = physics_outputs(z, d, ph)
    _, g, h = prior_terms(z, prior)
    chs0 = collect(Channel, channels)
    covs = Matrix{Float64}[]
    means = Vector{Float64}[]
    for k in 1:K
        chs = copy(chs0)
        for (i, np) in noise_post
            σ = max(np.σ + np.σ_sd * randn(rng), 0.25 * np.σ)
            ρ = clamp(np.ρ + np.ρ_sd * randn(rng), 0.0, 0.999)
            chs[i] = with_noise(chs[i], σ, ρ)              # a Levelled channel keeps its level
        end
        a, B, w = assemble(chs, z, S, J, prior, lay)
        ok = .!isnan.(w)
        H = Symmetric(B[ok, :]' * B[ok, :] + _prec(h))
        δ = H \ (B[ok, :]' * (w[ok] .- a[ok]) .- g)      # the Gauss-Newton step from the MAP under this whitening
        push!(covs, Matrix(inv(H)))
        push!(means, z .+ δ)
    end
    cov_mean = sum(covs) ./ K
    m = sum(means) ./ K
    spread = sum((mk .- m) * (mk .- m)' for mk in means) ./ max(K - 1, 1)
    cov = Matrix(Symmetric(cov_mean + spread))
    return (cov=cov, sd=sqrt.(max.(diag(cov), 0.0)), cov_mean=cov_mean, spread=spread, draws=means)
end
