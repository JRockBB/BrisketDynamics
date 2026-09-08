# Noise as graph structure: the AR(1) discrepancy of a channel and its
# scale as RxInfer's AR node, with the calibration's posterior as the
# prior in data form, instead of a fixed whitening. Included by
# BrisketEstimator after channels.jl.
#
# The graph for one channel's residual process r_k = y_k − pred_k(ẑ):
#   x_k ~ AR(x_{k−1}, θ, γ)        θ the AR coefficient (ρ), γ the innovation precision
#   r_k ~ Normal(x_k, ε)            ε a small observation variance (the residual IS the process)
# with q(x) q(θ) q(γ) (structured over the chain, mean-field over the
# parameters, the factorization RxInfer's own AR tests use), one chain
# per contiguous run of the mask (a lid lift breaks the chain), θ and γ
# shared across runs. The whitening's parameters are the marginal sd
# σ = 1 / sqrt(γ (1 − ρ²)) and ρ.
#
# The physics block does not change: it sees the noise through the
# whitening, and the two are alternated (fit z at the current noise,
# estimate the noise on the residuals at ẑ, refit) until the noise stops
# moving — a structured mean-field between the physics latents and the
# noise parameters, the residuals taken at the posterior mode (z's
# width is second order for a 115-point noise series; the misfit check's
# hat-matrix trace says how much of a channel the fit absorbs).

export NoisePrior, noise_priors_from_file, ar1_posterior, estimate_noise, with_noise, residual_runs

"""
    NoisePrior(σ_mean, σ_sd, ρ_mean, ρ_sd; floor=0.0)

The prior on a channel's AR(1) discrepancy, in the whitening's
parameterization (marginal sd, correlation), as moments; `floor` is the
calibration's sd floor for that channel (the applied sd is max(σ̂, floor)).
"""
struct NoisePrior
    σ_mean::Float64
    σ_sd::Float64
    ρ_mean::Float64
    ρ_sd::Float64
    floor::Float64
end
NoisePrior(σ_mean, σ_sd, ρ_mean, ρ_sd; floor=0.0) = NoisePrior(σ_mean, σ_sd, ρ_mean, ρ_sd, floor)

"""
    noise_priors_from_file(path; ρ_sd_min=0.05, σ_rel_min=0.2) -> Dict

The calibration's noise posterior (export_noise_prior.py) as priors,
keyed by channel name, with a between-cook floor on the widths: a
single cook's posterior on ρ (0.005 on the core probe) is a within-cook
precision, a hundred times what 115 points of another cook can say, and
another probe or thermal contact has another ρ; as a prior across cooks
it must be at least as wide as cooks differ.
"""
function noise_priors_from_file(path; ρ_sd_min::Float64=0.05, σ_rel_min::Float64=0.2)
    j = JSON.parsefile(path)
    fl = Float64(j["sigma_floor_K"])
    return Dict(String(k) => NoisePrior(Float64(v["sigma_mean"]),
                                        max(Float64(v["sigma_sd"]), σ_rel_min * Float64(v["sigma_mean"])),
                                        Float64(v["rho_mean"]), max(Float64(v["rho_sd"]), ρ_sd_min); floor=fl)
                for (k, v) in j["channels"])
end

# the AR node's parameters from the prior's moments: θ ~ Normal(ρ), γ ~ Gamma matched to
# γ = 1 / (σ² (1 − ρ²)) with the delta method on σ
function _ar_prior(p::NoisePrior)
    γm = 1 / (p.σ_mean^2 * (1 - p.ρ_mean^2))
    γsd = 2 * γm * p.σ_sd / p.σ_mean
    shape = (γm / γsd)^2
    rate = γm / γsd^2
    return (θ_mean=p.ρ_mean, θ_var=p.ρ_sd^2, γ_shape=shape, γ_rate=rate)
end

@model function ar1_chain(r, starts, θ_mean, θ_var, γ_shape, γ_rate, x0_var, ε)
    θ ~ Normal(mean = θ_mean, var = θ_var)
    γ ~ Gamma(shape = γ_shape, rate = γ_rate)
    x_prev = nothing
    for k in eachindex(r)
        if starts[k]
            x[k] ~ Normal(mean = 0.0, var = x0_var)          # a run starts from the stationary spread
        else
            x[k] ~ AR(x_prev, θ, γ) where {meta = ARMeta(Univariate, 1, ARsafe())}
        end
        r[k] ~ Normal(mean = x[k], var = ε)
        x_prev = x[k]
    end
end

@constraints function ar1_constraints()
    q(x, θ, γ) = q(x)q(θ)q(γ)
end

@initialization function ar1_init(θ_mean, θ_var, γ_shape, γ_rate)
    q(θ) = NormalMeanVariance(θ_mean, θ_var)
    q(γ) = GammaShapeRate(γ_shape, γ_rate)
end

"""
    ar1_posterior(runs, prior; iters=30, ε_rel=0.05) -> (σ, ρ, σ_sd, ρ_sd, θ, γ)

The AR node's posterior on one channel's residual runs (a vector of
vectors, raw units): the marginal sd and correlation with their delta-method
widths, and the raw posteriors of θ and γ.
"""
function ar1_posterior(runs::AbstractVector{<:AbstractVector}, prior::NoisePrior; iters::Int=30, ε_rel::Float64=0.05)
    r = reduce(vcat, [collect(Float64, run) for run in runs])
    starts = reduce(vcat, [[i == 1 for i in eachindex(run)] for run in runs])
    ap = _ar_prior(prior)
    res = infer(model = ar1_chain(starts = starts, θ_mean = ap.θ_mean, θ_var = ap.θ_var,
                                  γ_shape = ap.γ_shape, γ_rate = ap.γ_rate,
                                  x0_var = prior.σ_mean^2, ε = (ε_rel * prior.σ_mean)^2),
                data = (r = r,), constraints = ar1_constraints(),
                initialization = ar1_init(ap.θ_mean, ap.θ_var, ap.γ_shape, ap.γ_rate),
                iterations = iters, returnvars = (θ = KeepLast(), γ = KeepLast()))
    qθ, qγ = res.posteriors[:θ], res.posteriors[:γ]
    ρ, ρ_sd = mean(qθ), std(qθ)
    γm, γsd = mean(qγ), std(qγ)
    ρc = clamp(ρ, -0.999, 0.999)
    σ = 1 / sqrt(γm * (1 - ρc^2))
    # delta method: d log σ = −½ d log γ + ρ dρ / (1 − ρ²)
    σ_sd = σ * sqrt((0.5 * γsd / γm)^2 + (ρc * ρ_sd / (1 - ρc^2))^2)
    return (σ=σ, ρ=ρ, σ_sd=σ_sd, ρ_sd=ρ_sd, θ=qθ, γ=qγ)
end

"A channel's raw residuals y − pred at its sampled rows, split into runs (a run breaks at a gap longer than three times the channel's spacing)."
function residual_runs(ch::Channel, S, J, z, prior, lay)
    r, _ = predict(ch, S, J, z, prior, lay)
    y = rawdata(ch)
    y === nothing && throw(ArgumentError("the channel carries no data"))
    e = y .- r
    nm = noise_mask(ch)
    return [e[run] for run in contiguous_runs(nm; maxgap=run_maxgap(nm))]
end

"""
    estimate_noise(ch, S, J, z, prior, lay, np::NoisePrior; kw...) -> (σ, ρ, post)

The channel's AR(1) noise from its residuals at `z`, with the prior;
the applied sd respects the prior's floor.
"""
function estimate_noise(ch::Channel, S, J, z, prior, lay, np::NoisePrior; kw...)
    g = channel_gap(noise_mask(ch))
    # the AR node sees the channel's own spacing; its prior and posterior on ρ are per grid step
    np_g = NoisePrior(np.σ_mean, np.σ_sd, clamp(np.ρ_mean, 0.0, 0.999)^g, np.ρ_sd * g * max(np.ρ_mean, 1e-3)^(g - 1); floor=np.floor)
    post = ar1_posterior(residual_runs(ch, S, J, z, prior, lay), np_g; kw...)     # de-levelled for a Levelled channel
    ρ_step = clamp(post.ρ, 0.0, 0.999)^(1 / g)      # back to the grid step
    return (σ=max(post.σ, np.floor), ρ=ρ_step, post=post)
end

# the same channel with a new whitening
with_noise(ch::PitProbe, σ, ρ) = PitProbe(σ, ρ, ch.mask, ch.data)
with_noise(ch::SurfaceIR, σ, ρ) = SurfaceIR(σ, ρ, ch.mask, ch.data)
with_noise(ch::HumiditySensor, σ, ρ) = HumiditySensor(σ, ρ, ch.mask, ch.data)
with_noise(ch::MassScale, σ, ρ) = MassScale(σ, ρ, ch.mask, ch.shape_latent, ch.data)
with_noise(ch::MeatProbe, σ, ρ) = MeatProbe(ch.s_fixed, ch.depth_fixed, ch.depth_latent, σ, ρ, ch.mask, ch.data)
with_noise(ch::FuelScale, σ, ρ) = FuelScale(σ, ρ, ch.mask, ch.data)
with_noise(ch::BarkLightness, σ, ρ) = BarkLightness(σ, ρ, ch.mask, ch.data)
with_noise(ch::WetBulb, σ, ρ) = WetBulb(σ, ρ, ch.mask, ch.p_atm, ch.a_psy, ch.data)
with_noise(ch::StaticReading, σ, ρ) = ch
