# The noise level as a process: a channel's AR(1) discrepancy whose
# marginal sd follows the cook (the pit probe's fire regime), inferred
# on the graph as a stochastic-volatility model. Included by
# BrisketEstimator after noise.jl.
#
# The residual e_k of a channel is a unit AR(1) a_k times a level s_k
# (realcook.jl's reading model; the whitening below is exact for it).
# Given the AR node's ρ, the innovations η_k = e_k − ρ e_{k−1} are white
# with variance s_k² (1 − ρ²) when the level moves slowly, and their
# cumulative sum is a random walk whose increments have that variance:
# exactly RxInfer's GCV node, x_k ~ N(x_{k−1}, exp(z_k)), with the
# log-variance z_k itself a random walk (the hierarchical Gaussian
# filter's two layers, Gauss–Hermite messages, mean-field between the
# layers). The smoother's posterior on z_k is the level.

export volatility_level, Levelled, level_on_grid, unlevelled

@model function volatility_smoother(x, w, z0_mean, z0_var, ε)
    z[1] ~ Normal(mean = z0_mean, var = z0_var)
    xk[1] ~ Normal(mean = x[1], var = ε)
    for k in 2:length(x)
        z[k] ~ Normal(mean = z[k - 1], var = w)
        xk[k] ~ GCV(xk[k - 1], z[k], 1.0, 0.0)
        x[k] ~ Normal(mean = xk[k], var = ε)
    end
end
@constraints function volatility_constraints()
    q(xk, z) = q(xk)q(z)
end
const VOLATILITY_META = @meta begin
    GCV() -> GCVMetadata(GaussHermiteCubature(21))
end
@initialization function volatility_init(z0_mean, z0_var, x1)
    q(z) = NormalMeanVariance(z0_mean, z0_var)
    q(xk) = NormalMeanVariance(x1, 1.0)
end

"""
    volatility_level(e, ρ, σ0; w=0.05, iters=20) -> (level, z_mean, z_sd)

The level s_k (the marginal sd relative to `σ0`, so 1 means the base
noise) along one residual run `e` with AR(1) correlation `ρ` per
sample: the innovations' log-variance from the volatility smoother,
with `w` the log-variance's random-walk variance per sample (how fast a
regime can change) and the prior centered on the base noise.
"""
function volatility_level(e::AbstractVector, ρ::Real, σ0::Real; w::Float64=0.05, iters::Int=20)
    n = length(e)
    n < 4 && return (level=ones(n), z_mean=fill(log(σ0^2 * (1 - ρ^2)), n), z_sd=fill(1.0, n))
    η = [e[k] - ρ * e[k - 1] for k in 2:n]                  # the innovations
    x = vcat(0.0, cumsum(η))                                 # a random walk with those increments
    z0 = log(σ0^2 * (1 - ρ^2))
    res = infer(model = volatility_smoother(w = w, z0_mean = z0, z0_var = 1.0, ε = 1e-6 * σ0^2),
                data = (x = x,), constraints = volatility_constraints(), meta = VOLATILITY_META,
                initialization = volatility_init(z0, 1.0, x[1]), iterations = iters,
                returnvars = (z = KeepLast(),), options = (limit_stack_depth = 100,))
    qz = res.posteriors[:z]
    zm = [mean(q) for q in qz]
    zs = [std(q) for q in qz]
    level = sqrt.(exp.(zm) ./ (σ0^2 * (1 - ρ^2)))
    return (level=level, z_mean=zm, z_sd=zs)
end

"""
    Levelled(channel, level)

A channel whose noise level varies along the cook: `level[k]` multiplies
the channel's sd at grid point k (1 where the level is unknown). The
whitening is exact for a unit AR(1) scaled by the level.
"""
struct Levelled{C<:Channel} <: Channel
    ch::C
    level::Vector{Float64}
end
Base.getproperty(l::Levelled, f::Symbol) = f in (:ch, :level) ? getfield(l, f) : getproperty(getfield(l, :ch), f)
predict(l::Levelled, S, J, z, prior, lay) = predict(l.ch, S, J, z, prior, lay)
rawdata(l::Levelled) = rawdata(l.ch)
channel_label(l::Levelled) = channel_label(l.ch) * ", levelled"
restrict(l::Levelled, keep) = Levelled(restrict(l.ch, keep), l.level)
with_noise(l::Levelled, σ, ρ) = Levelled(with_noise(l.ch, σ, ρ), l.level)
unlevelled(ch::Channel) = ch
unlevelled(l::Levelled) = l.ch
function residual_runs(l::Levelled, S, J, z, prior, lay)
    nm = noise_mask(l)
    runs = contiguous_runs(nm; maxgap=run_maxgap(nm))
    return [e ./ l.level[run] for (e, run) in zip(residual_runs(l.ch, S, J, z, prior, lay), runs)]
end
noise_mask(l::Levelled) = noise_mask(l.ch)
extra_rows(l::Levelled, level) = extra_rows(l.ch, level)
misfit_rows(l::Levelled) = misfit_rows(l.ch)
"The whitening rows of a unit AR(1) scaled by `level`, gap-aware as `whitening_rows`."
function whitening_rows(mask::AbstractVector{Bool}, σ::Real, ρ::Real, level::AbstractVector)
    n = length(mask)
    rows = Vector{Vector{Float64}}()
    for run in contiguous_runs(mask; maxgap=run_maxgap(mask))
        r = zeros(n); r[run[1]] = 1 / (σ * level[run[1]]); push!(rows, r)
        for j in 2:length(run)
            k, kprev = run[j], run[j - 1]
            ρg = ρ^(k - kprev)
            innov = sqrt(max(1 - ρg * ρg, 1e-8))
            r = zeros(n)
            r[k] = 1 / (σ * level[k] * innov)
            r[kprev] = -ρg / (σ * level[kprev] * innov)
            push!(rows, r)
        end
    end
    return isempty(rows) ? zeros(0, n) : reduce(vcat, permutedims.(rows))
end
whitening(l::Levelled) = vcat(whitening_rows(noise_mask(l), l.σ, l.ρ, l.level), extra_rows(l.ch, l.level))

"The level on the grid from the levels along a channel's runs (1 off the channel's samples)."
function level_on_grid(mask::AbstractVector{Bool}, levels_by_run::AbstractVector)
    L = ones(length(mask))
    for (run, lv) in zip(contiguous_runs(mask; maxgap=run_maxgap(mask)), levels_by_run)
        L[run] .= lv
    end
    return L
end
