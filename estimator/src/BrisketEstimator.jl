# The online estimator as a factor graph (RxInfer) over the BrisketDynamics
# monolith: the calibration's Gauss-Newton / Laplace posterior reproduced by
# message passing, then the channel registry, the noise on the graph, and
# the filter that carries the belief through the physics as the cook goes.
# `sampled_init` provides incremental joint nonlinear inference; `sampled_fit` is the offline reference.
# Its uncertainty comes from joint samples rather than the optimizer's local Gaussian covariance;
# sensor declarations are shared, and numerical precision is checked before accepting updates.
#
# Structure, fixed by three measured RxInfer behaviours (test_rules.jl):
#   * ONE multivariate latent block z, every site in an unconstrained
#     coordinate: the log for a lognormal site; the logit or the normal
#     score for a beta or logbeta site (coord = :score, the pilot's prior
#     file: the family is exactly N(0, 1) there, tails included); the value
#     itself for a linear site (the filter's state block, itself in
#     softplus / logit coordinates) — separate blocks feeding one node are
#     mean-field in RxInfer and return conditional, not marginal, variances;
#   * one node PER WHITENED RESIDUAL (or per local censored-normal survival
#     site), each with a scalar output — a node with more outputs than
#     latent inputs has a singular forward output covariance and RxInfer
#     silently returns the prior;
#   * the physics enters each node AFFINE around a centre c (f(c) + J(c)(z − c),
#     J by ForwardDiff through the FBDF solve, or the regression over sigma
#     points in the filter), so the Linearization rule is exact; the centre
#     moves INSIDE one `infer` call (iterated.jl: the after-iteration callback
#     re-centres the affine model, runs a damped line search on the regression's
#     own cost, and scales the Levenberg-Marquardt node);
#   * the prior node is the local quadratic of −log p(z) at c (exact for a
#     Gaussian or a score site, the logit-Beta curvature for a logit site; a
#     filter's predicted belief as a full Gaussian, solved in its whitened
#     coordinates), so the batch fixed point and covariance are those of
#     scripts/run_gn_map.py.
# The files: channels.jl (the registry: PitProbe, MeatProbe, CensoredHumidity,
# MassScale and the rest, each with predict / noise_mask / extra_rows),
# noise.jl (the AR node on each channel's residuals), volatility.jl (the
# level as a GCV process), stream.jl (the batch fit alternating physics and
# noise), iterated.jl (the loop on the graph), segments.jl / drift.jl /
# smoother.jl (the plant in segments, the drifting truths, the moving sites),
# sequential.jl (the filter: the state in constrained coordinates, sigma-point
# prediction, the whitened update, the process noise earned on the bench).
module BrisketEstimator

using RxInfer
using ForwardDiff
using LinearAlgebra
using Random
using JSON
using Printf
using SHA
using SparseArrays
using SpecialFunctions: erfc, erfcx, erfcinv, beta_inc, beta_inc_inv, logbeta
using BrisketDynamics

export ridge_chain, noise_marginal_cov, FilterPrior, prior_quadratic, handoff, solve_segments
export DriftLayout, PhysicsDrift, DriftPrior, smoother_fit, smoother_step, drift_layout, warm_start, segment_edges, index_at, current_index, block_at
export DriftDraw, truth_at, drift_segments, solve_drifting, smoother_moving, DRIFT_SD_PER_HOUR, JUMP_LOG_SD, drift_rates
export FilterLayout, PhysicsState, pack_state, unpack_state, state_to_coord, state_from_coord, state_physical_moments,
       state_sites, disc_sites, sequential_filter, pit_index, centre_index, surface_index
export FilterState, StateNoise, Past, filter_init, filter_update!, filter_advance!, state_range, disc_range, disc_index, state_now, innovation_misfit
export STATE_GROUPS, STATE_NOISE_FILE, state_groups, group_dofs, state_noise_cov, physics_fingerprint, state_noise_dict,
       state_noise_from_dict, load_state_noise, sigma_design
export ALSAccumulator, als_update!, als_summary
export GateData, load_gate, from_z, to_z, dx_dz, z_center, z_moments, prior_terms, physics_mean, physics_mean_jac,
       objective_value, solve_map, compare_gate, PriorSite, N_PHYS, N_THETA, isbeta, islogbeta, islinear, isscore, drift_scale, logistic,
       latent_block, probe_depth_site, probe_depth, read_sites, draw_truth, FAMILIES

const N_THETA = 8
const N_PHYS = 15

"""
    PriorSite(family, p1, p2[, p3, p4]; coord=:logit)

A site of the latent block in its family, the transform z -> x and the prior in one:
- `"lognormal"`: x = exp(z), z ~ N(p1, p2²) (a positive site);
- `"beta"`: the fraction x ~ Beta(p1, p2) (a bounded site);
- `"logbeta"`: x = exp(p3 + (p4 − p3) f), the fraction f ~ Beta(p1, p2) (a positive site whose
  calibration chain is skewed: bounded in the log to [p3, p4], the skew in the Beta);
- `"linear"`: x = z, z ~ N(p1, p2²) (a plant state in the filter's block, in its own units).

A bounded site's coordinate is `coord`: `:logit` (f = logistic(z), the prior the Beta's own density in the
logit: the calibration's coordinate, the gate's) or `:score` (f = F⁻¹(Φ(z)) with F the Beta's distribution
function, the normal score, under which the prior is exactly N(0, 1); the pilot's prior file carries its
bounded sites so, `export_chain_prior.py`). A Gaussian belief in the logit holds a Beta's exponential tail as
a quadratic and leaves a truth two prior sd into the tail three to five posterior sd out; in the score the
family's tails are the Gaussian's by construction (`docs/results.md`, the entry of 2026-09-17, night).
"""
struct PriorSite
    family::String
    p1::Float64
    p2::Float64
    p3::Float64
    p4::Float64
    coord::Symbol
    function PriorSite(family::AbstractString, p1::Real, p2::Real, p3::Real=0.0, p4::Real=1.0; coord::Symbol=:logit)
        family in FAMILIES || throw(ArgumentError("unknown prior family \"$family\" (one of $(join(FAMILIES, ", ")))"))
        coord in (:logit, :score) || throw(ArgumentError("a site's coordinate is :logit or :score, not :$coord"))
        coord == :score && !(family in ("beta", "logbeta")) && throw(ArgumentError("the normal score is a bounded family's coordinate, not a $family site's"))
        return new(String(family), Float64(p1), Float64(p2), Float64(p3), Float64(p4), coord)
    end
end
const FAMILIES = ("lognormal", "beta", "logbeta", "linear")

struct GateData
    t::Vector{Float64}
    u::Vector{Float64}
    L::Matrix{Float64}        # whitening of the masked AR(1) likelihood (n_obs × 2nt)
    w_obs::Vector{Float64}    # whitened observations L·y
    names::Vector{String}
    prior::Vector{PriorSite}
    z0::Vector{Float64}       # cold start (prior medians / means)
    z_map::Vector{Float64}    # reference: Gauss-Newton MAP
    cov_map::Matrix{Float64}  # reference: (Jᵀ Σ⁻¹ J + prior curvature)⁻¹ at the MAP
    F_map::Float64
    pit::PitConfig
    cfg::MeatConfig
    abstol::Float64
    reltol::Float64
end

_mat(rows) = reduce(vcat, [permutedims(Float64.(r)) for r in rows])

"""
    load_gate(path)

Read `data/calibration/estimator_gate_real.json` (written by
`estimator/python/export_gate_data.py`).
"""
function load_gate(path::AbstractString)
    d = JSON.parsefile(path)
    meta = d["meta"]
    L = _mat(d["L"])
    y = Float64.(d["y"])
    prior = read_sites(d["prior"])
    return GateData(
        Float64.(d["t"]), Float64.(d["u"]), L, L * y, String.(d["names"]), prior,
        Float64.(d["z0"]), Float64.(d["z_map"]), _mat(d["cov_map"]), Float64(d["F_map"]),
        PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]),
                  m0=Float64(meta["m0_kg"])),
        MeatConfig(; T0=Float64(meta["tcore0_K"])),
        Float64(meta["abstol"]), Float64(meta["reltol"]),
    )
end

# --- unconstrained parameterization and the prior ---------------------------

isbeta(p::PriorSite) = p.family == "beta"
islogbeta(p::PriorSite) = p.family == "logbeta"
islinear(p::PriorSite) = p.family == "linear"          # a state entry in a filter's block: Gaussian in its own coordinate, no transform
_betalike(p::PriorSite) = isbeta(p) || islogbeta(p)    # the prior is a Beta on a fraction f of the site's coordinate
isscore(p::PriorSite) = p.coord == :score             # a bounded site carried in its normal score (see `PriorSite`)

"The prior's centre in z: the Gaussian's mean, the logit of a Beta's a/(a+b), or zero for a site in its score."
z_center(p::PriorSite) = _betalike(p) ? (isscore(p) ? 0.0 : log(p.p1 / p.p2)) : p.p1

"The digamma function by the recurrence to x ≥ 6 and the asymptotic series (to 1e-10)."
function _digamma(x::Real)
    r = 0.0
    while x < 6; r -= 1 / x; x += 1; end
    f = 1 / (x * x)
    return r + log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))
end
"The trigamma function, the same way."
function _trigamma(x::Real)
    r = 0.0
    while x < 6; r += 1 / (x * x); x += 1; end
    f = 1 / (x * x)
    return r + 1 / x + 0.5 * f + f / x * (1 / 6 - f * (1 / 30 - f * (1 / 42 - f / 30)))
end

"""
    z_moments(p::PriorSite) -> (mean, var)

The family's mean and variance in the transformed coordinate: a lognormal or linear site's own; a beta or
logbeta site's logit moments, ψ(a) − ψ(b) and ψ'(a) + ψ'(b). The filter's belief starts from these, the
Gaussian closest to the family in its coordinate, rather than from the curvature at the mode: a beta with b
near one has an upper tail that is exponential in the logit (the ignition fraction's toward a fully lit
start), which the curvature at the mode understates (the Laplace width 0.99 against the draws' 1.19 for a0)
and which no local quadratic can carry (its curvature there is zero, and an exact prior in the update blew
the posterior width up); matching the moments is what a Gaussian can do, and it scores unit NEES against the
family's own draws by construction.
"""
function z_moments(p::PriorSite)
    _betalike(p) || return (p.p1, p.p2^2)
    isscore(p) && return (0.0, 1.0)                                                  # the score: standard normal by construction
    return (_digamma(p.p1) - _digamma(p.p2), _trigamma(p.p1) + _trigamma(p.p2))
end

"One fully specified site from a serialized prior."
function _read_site(p)
    family = String(p["family"])
    if family == "logbeta" && !(haskey(p, "p3") && haskey(p, "p4"))
        throw(ArgumentError("a serialized logbeta prior requires p3 and p4 bounds"))
    end
    return PriorSite(family, Float64(p["p1"]), Float64(p["p2"]),
                     Float64(get(p, "p3", 0.0)), Float64(get(p, "p4", 1.0));
                     coord=Symbol(get(p, "coord", "logit")))
end

"The sites of a prior file's `prior` list (`prior_next_cook.json`, the gate file), including bounds and coordinate."
read_sites(list) = _read_site.(list)

"""
    probe_depth_site(; mean=0.8, concentration=10.0)

The bounded prior for the meat-probe insertion depth as a fraction of the
current half-thickness. Its score coordinate is standard normal, while its
physical value is always in `(0, 1)`.
"""
function probe_depth_site(; mean::Float64=0.8, concentration::Float64=10.0)
    0 < mean < 1 || throw(ArgumentError("probe-depth fraction mean must lie in (0, 1)"))
    concentration > 0 || throw(ArgumentError("probe-depth concentration must be positive"))
    return PriorSite("beta", mean * concentration, (1 - mean) * concentration; coord=:score)
end

"Physical insertion depth from half-thickness and its bounded fraction."
probe_depth(L, fraction) = L * fraction

"""
    latent_block(prior_path, cfg; area_sd=0.3, depth_frac=0.8, depth_concentration=10.0)

The estimator's latent block, declared once: the fifteen physics sites of the prior file in their families and
names, then the two estimator-only sites, the exposed area (`A_meat`, lognormal about the config's) and the meat
probe's insertion-depth fraction (bounded Beta with mean `depth_frac`). Physical depth is `L * fraction`, so
thickness and depth share one joint parameterization. Returns `(sites, names, prior_phys, phys_names, area,
depth, idx, file)`; every driver reads its block from here.
"""
function latent_block(prior_path::AbstractString, cfg::MeatConfig; area_sd::Float64=0.3,
                      depth_frac::Float64=0.8, depth_concentration::Float64=10.0)
    file = JSON.parsefile(prior_path)
    prior_phys = read_sites(file["prior"])
    length(prior_phys) == N_PHYS || throw(ArgumentError("the prior file carries $(length(prior_phys)) physics sites, not $N_PHYS"))
    phys_names = String.(file["names"])
    sites = vcat(prior_phys, [PriorSite("lognormal", log(cfg.A_meat), area_sd),
                             probe_depth_site(mean=depth_frac, concentration=depth_concentration)])
    names = vcat(phys_names, ["A_meat", "core_depth_fraction"])
    return (sites=sites, names=names, prior_phys=prior_phys, phys_names=phys_names, area=length(sites) - 1, depth=length(sites),
            idx=Dict(n => i for (i, n) in enumerate(names)), file=file)
end

"An inverse-CDF draw of Beta(a, b) on a fine grid (no Distributions dependency)."
function _beta_rand(rng, a, b)
    x = range(1e-6, 1 - 1e-6; length=4001); lp = (a - 1) .* log.(x) .+ (b - 1) .* log.(1 .- x); p = exp.(lp .- maximum(lp)); c = cumsum(p); c ./= c[end]
    r = rand(rng); k = clamp(searchsortedfirst(c, r), 2, length(c)); x[k - 1] + (x[k] - x[k - 1]) * (r - c[k - 1]) / max(c[k] - c[k - 1], 1e-300)
end
"""
    draw_truth(rng, sites) -> z

A truth drawn from every site's declared prior, in z (Gaussian for the
lognormal and linear sites, a Beta draw in the site's logit or score
coordinate for beta and logbeta sites). Probe depth needs no special clipping:
its declared site is already a bounded fraction and physical depth is formed
jointly with thickness by `probe_depth`.
"""
function draw_truth(rng::AbstractRNG, sites::AbstractVector{PriorSite}; depth::Int=0, thickness::Int=4)
    depth == 0 || throw(ArgumentError("probe-depth truth clipping was removed; declare a bounded fraction with probe_depth_site and call draw_truth without `depth`"))
    z = [_betalike(p) ? (f = _beta_rand(rng, p.p1, p.p2); isscore(p) ? _score_of_fraction(f, p.p1, p.p2) : log(f / (1 - f))) : p.p1 + p.p2 * randn(rng) for p in sites]
    return z
end

"""
    FilterPrior(sites, m, Λ)
    FilterPrior(sites, m, Λ, S)

A full-covariance Gaussian prior on the latent block in z (a filter's predicted belief: the last posterior
inflated by process noise), carrying the per-site families `sites` for the transforms z -> x. Indexes and
iterates as its sites, so every consumer of a `Vector{PriorSite}` (from_z, the channels, the layouts) reads
it unchanged; `prior_terms` returns the Gaussian's value, gradient and FULL Hessian. `S`, when present, is the
lower square root of the covariance (`P = S*S'`); the sequential graph is then solved in standardized
coordinates instead of forming or messaging the ill-conditioned precision.
"""
struct FilterPrior
    sites::Vector{PriorSite}
    m::Vector{Float64}
    Λ::Matrix{Float64}
    S::Union{Nothing,Matrix{Float64}}
end
FilterPrior(sites::Vector{PriorSite}, m::AbstractVector, Λ::AbstractMatrix) =
    FilterPrior(sites, collect(Float64, m), Matrix{Float64}(Λ), nothing)
FilterPrior(sites::Vector{PriorSite}, m::AbstractVector, Λ::AbstractMatrix, S::AbstractMatrix) =
    FilterPrior(sites, collect(Float64, m), Matrix{Float64}(Λ), Matrix{Float64}(S))
Base.getindex(p::FilterPrior, i) = p.sites[i]
Base.length(p::FilterPrior) = length(p.sites)
Base.lastindex(p::FilterPrior) = lastindex(p.sites)
Base.iterate(p::FilterPrior, st...) = iterate(p.sites, st...)
Base.eltype(::Type{FilterPrior}) = PriorSite
function prior_terms(z, p::FilterPrior)
    r = z .- p.m
    if p.S === nothing
        return 0.5 * dot(r, p.Λ * r), p.Λ * r, Matrix(p.Λ)
    end
    u = p.S \ r
    g = p.S' \ u
    H = p.S' \ (p.S \ Matrix{Float64}(I, length(r), length(r)))
    return 0.5 * dot(u, u), g, Matrix(Symmetric(H))
end
"The prior's precision as a matrix: a diagonal from the per-site curvature, or the Gaussian's own."
_prec(h::AbstractVector) = Matrix(Diagonal(h))
_prec(H::AbstractMatrix) = Matrix(H)
"""
    prior_quadratic(c, prior; damping=0.0) -> (mp, Λp)

The prior's local quadratic at `c` as the graph's prior node takes it (mean and precision), with an optional
Levenberg-Marquardt penalty added to the precision.
"""
function prior_quadratic(c, prior; damping::Float64=0.0)
    _, g, h = prior_terms(c, prior)
    Λp = _prec(h) + damping * I
    mp = c .- Λp \ g
    return mp, Λp
end
logistic(v) = v >= 0 ? inv(1 + exp(-v)) : (e = exp(v); e / (1 + e))

# --- the normal score of a Beta: f = F⁻¹(Φ(z)), under which the prior is exactly N(0, 1) ------------------------
const SCORE_MAX = 8.0     # beyond eight score sd the fraction is held at its value there (Φ(±8) sits 6e-16 from 0 and 1)
"The fraction and its complement at score `z` under Beta(a, b): the inverse regularized incomplete beta at Φ(z), Φ and 1 − Φ both carried for the tails."
function _score_pair(z::Real, a, b)
    zc = clamp(z, -SCORE_MAX, SCORE_MAX)
    p = 0.5 * erfc(-zc / sqrt(2.0)); q = 0.5 * erfc(zc / sqrt(2.0))
    return beta_inc_inv(a, b, p, q)
end
"df/dz at score `z`: φ(z) over the Beta's density at f, in logs (zero beyond ±SCORE_MAX, where the fraction is held)."
function _score_slope(z, a, b, f, y)
    (abs(z) > SCORE_MAX || f <= 0 || y <= 0) && return zero(f)
    return exp(-0.5 * z^2 - 0.5 * log(2π) - ((a - 1) * log(f) + (b - 1) * log(y) - logbeta(a, b)))
end
# under ForwardDiff: the value from the inverse beta, the derivative by the slope, through any nesting
function _score_pair(z::ForwardDiff.Dual{T}, a, b) where {T}
    zv = ForwardDiff.value(z)
    f, y = _score_pair(zv, a, b)
    s = _score_slope(zv, a, b, f, y); pz = ForwardDiff.partials(z)
    return (ForwardDiff.Dual{T}(f, pz * s), ForwardDiff.Dual{T}(y, pz * (-s)))
end
_score_fraction(z, a, b) = _score_pair(z, a, b)[1]
"The score under Beta(a, b) of a fraction `f`: Φ⁻¹ of the regularized incomplete beta, from whichever tail is the smaller."
function _score_of_fraction(f::Real, a, b)
    fc = clamp(f, 1e-12, 1 - 1e-12)
    p, q = beta_inc(a, b, fc, 1 - fc)
    z = p <= 0.5 ? -sqrt(2.0) * erfcinv(2p) : sqrt(2.0) * erfcinv(2q)
    return clamp(z, -SCORE_MAX, SCORE_MAX)
end
"df/dz of a bounded site's fraction at `v`: the logistic's slope, or the score's."
function _frac_slope(v, p::PriorSite)
    if isscore(p)
        f, y = _score_pair(v, p.p1, p.p2)
        return _score_slope(v, p.p1, p.p2, f, y)
    end
    s = logistic(v); return s * (1 - s)
end
_fraction(v, p::PriorSite) = isscore(p) ? _score_fraction(v, p.p1, p.p2) : logistic(v)

"The constrained value of one site at `v` in z (see `PriorSite`)."
function from_z(v, p::PriorSite)
    isbeta(p) && return _fraction(v, p)
    islogbeta(p) && return exp(p.p3 + (p.p4 - p.p3) * _fraction(v, p))
    islinear(p) && return v
    return exp(v)
end
"dx/dz of one site at `v`."
function dx_dz(v, p::PriorSite)
    isbeta(p) && return _frac_slope(v, p)
    islogbeta(p) && return from_z(v, p) * (p.p4 - p.p3) * _frac_slope(v, p)
    islinear(p) && return one(v)
    return exp(v)
end
"The unconstrained coordinate of one site's value `x` (the inverse of `from_z`)."
function to_z(x, p::PriorSite)
    isbeta(p) && return isscore(p) ? _score_of_fraction(x, p.p1, p.p2) : log(x / (1 - x))
    islogbeta(p) && (f = clamp((log(x) - p.p3) / (p.p4 - p.p3), 1e-9, 1 - 1e-9); return isscore(p) ? _score_of_fraction(f, p.p1, p.p2) : log(f / (1 - f)))
    islinear(p) && return x
    return log(x)
end
"""
    drift_scale(p::PriorSite) -> Float64

The factor that carries a rate stated in the logit of a bounded site's fraction (drift.jl's constants, a prior
file's `drift` field) into the site's own coordinate: one for a site carried in the logit or the log, and for a
site in its score the score's slope on the logit at the prior's centre, so that the plant walks and jumps by the
same amount in x whichever coordinate the belief carries the site in.
"""
function drift_scale(p::PriorSite)
    (_betalike(p) && isscore(p)) || return 1.0
    f, y = _score_pair(0.0, p.p1, p.p2)
    return f * y / _score_slope(0.0, p.p1, p.p2, f, y)      # dz/dlogit = (dz/df)(df/dlogit) = f(1 − f) / (df/dz)
end
from_z(z, prior) = [from_z(v, p) for (v, p) in zip(z, prior)]
dx_dz(z, prior) = [dx_dz(v, p) for (v, p) in zip(z, prior)]
to_z(x::AbstractVector, prior::AbstractVector) = [to_z(v, p) for (v, p) in zip(x, prior)]

"""
    prior_terms(z, prior) -> (f, g, h)

−log p(z), its gradient and diagonal Hessian in the unconstrained space:
Gaussian for the LogNormal and linear sites; for Beta(a, b) sites (and the
logbeta's fraction) the logit-Jacobian form −a·log σ(z) − b·log(1 − σ(z)) in the logit, ½z² in the score.
Identical to `brisket_app.monolith.prior_terms` on the calibration's families.
"""
function prior_terms(z, prior)
    T = float(eltype(z))
    f = zero(T)
    g = zeros(T, length(z))
    h = zeros(T, length(z))
    for (i, (v, p)) in enumerate(zip(z, prior))
        if _betalike(p) && !isscore(p)
            a, b = p.p1, p.p2
            s = logistic(v)
            f += -a * log(s) - b * log(1 - s)
            g[i] = -a * (1 - s) + b * s
            h[i] = (a + b) * s * (1 - s)
        elseif _betalike(p)                                          # the score: standard normal, exactly
            f += 0.5 * v^2
            g[i] = v
            h[i] = one(T)
        else
            m, sd = p.p1, p.p2
            f += 0.5 * ((v - m) / sd)^2
            g[i] = (v - m) / sd^2
            h[i] = 1 / sd^2
        end
    end
    return f, g, h
end

# --- the physics: μ(z) = [T_pit; T_core] on the log grid, and dμ/dz --------

function _split(z, d::GateData)
    x = from_z(z, d.prior)
    return x[1:N_THETA], x[(N_THETA + 1):end]
end

function physics_mean(z, d::GateData)
    θ, φ = _split(z, d)
    o = monolith_solve(θ, φ, d.u, d.t, d.t; pit=d.pit, cfg=d.cfg, abstol=d.abstol, reltol=d.reltol)
    return vcat(o.Tpit, o.Tcore)
end

function physics_mean_jac(z, d::GateData)
    θ, φ = _split(z, d)
    kw = (pit=d.pit, cfg=d.cfg, abstol=d.abstol, reltol=d.reltol)
    o = monolith_solve(θ, φ, d.u, d.t, d.t; kw...)
    Jx = monolith_jacobian(o -> vcat(o.Tpit, o.Tcore), θ, φ, d.u, d.t, d.t; kw...)
    return vcat(o.Tpit, o.Tcore), Jx .* permutedims(dx_dz(z, d.prior))
end

"F(z) = ½‖L(y − μ(z))‖² − log p(z), the objective `run_gn_map.py` minimizes."
objective_value(z, μ, d::GateData) = 0.5 * sum(abs2, d.w_obs - d.L * μ) + prior_terms(z, d.prior)[1]

# --- the graph -----------------------------------------------------------------

mutable struct Affine
    a::Vector{Float64}   # L·μ(c)
    B::Matrix{Float64}   # L·J_z(c)
    c::Vector{Float64}
end
const AFF = Ref{Affine}(Affine(Float64[], zeros(0, 0), Float64[]))

# whitened residual i: L_i μ(c) + (L_i J_z(c)) (z − c) — affine in z, so the
# Linearization rule is exact; i enters as a constant
function reading(z, i)
    A = AFF[]
    return A.a[i] + dot(view(A.B, i, :), z .- A.c)
end

@model function cook_graph(w, mp, Λp)
    z ~ MvNormalMeanPrecision(mp, Λp)
    for i in 1:length(w)
        pred[i] := reading(z, i)
        w[i] ~ Normal(mean = pred[i], var = 1.0)
    end
end

const GRAPH_META = @meta begin
    reading() -> Linearization()
end

@initialization function init_z(mp, Λp)
    μ(z) = MvNormalMeanPrecision(mp, Λp)
end

"""
    graph_posterior(c, μ, Jz, d; bp_iters) -> (mean, cov)

One message-passing solve of the graph linearized at `c`: the prior node is
the local quadratic of −log p(z) at `c`, every whitened residual is a node.
Returns the Gaussian posterior of z under that linearization — the
Gauss-Newton step and covariance from `c`.
"""
function graph_posterior(c, μ, Jz, d::GateData; bp_iters::Int=20)
    AFF[] = Affine(d.L * μ, d.L * Jz, copy(c))
    mp, Λp = prior_quadratic(c, d.prior)
    r = infer(model = cook_graph(mp = mp, Λp = Λp), data = (w = d.w_obs,), meta = GRAPH_META,
              initialization = init_z(mp, Λp), iterations = bp_iters,
              returnvars = (z = KeepLast(),))
    m, V = mean_cov(r.posteriors[:z])
    return collect(m), Matrix(V)
end

"""
    solve_map(d; c0, iters, tol, bp_iters, verbose)

Outer relinearization loop: each pass solves the graph at `c`, takes the
posterior mean as the Gauss-Newton step, and halves the step while the
objective does not decrease (the same safeguard Levenberg-Marquardt gives
`run_gn_map.py`; the fixed point is unchanged). Returns the converged
point, the posterior mean and covariance of the graph solved at it, the
objective, and the iteration history.
"""
function solve_map(d::GateData; c0=d.z0, iters::Int=30, tol::Float64=1e-4,
                   bp_iters::Int=20, verbose::Bool=true)
    c = copy(c0)
    μ, Jz = physics_mean_jac(c, d)
    Fc = objective_value(c, μ, d)
    hist = NamedTuple[]
    verbose && @printf("  it  0  F=%10.4f\n", Fc)
    verbose && flush(stdout)
    m, V = c, zeros(N_PHYS, N_PHYS)
    for k in 1:iters
        t0 = time()
        m, V = graph_posterior(c, μ, Jz, d; bp_iters)
        step = m - c
        α = 1.0
        c_new = c + step
        μn = physics_mean(c_new, d)
        Fn = objective_value(c_new, μn, d)
        while Fn > Fc && α > 1e-3
            α /= 2
            c_new = c + α * step
            μn = physics_mean(c_new, d)
            Fn = objective_value(c_new, μn, d)
        end
        push!(hist, (it=k, F=Fc, step=norm(step), alpha=α, wall=time() - t0))
        verbose && @printf("  it %2d  F=%10.4f -> %10.4f  |step|=%.3g  alpha=%.3g  (%.1f s)\n",
                           k, Fc, Fn, norm(step), α, time() - t0)
        verbose && flush(stdout)
        if Fn > Fc
            # no descent left along the Gauss-Newton direction: the objective's
            # noise floor (last-bit Jacobian jitter) — keep the best point
            break
        end
        c = c_new
        Fc = Fn
        μ, Jz = physics_mean_jac(c, d)
        α * norm(step) < tol && break
    end
    # the posterior of the graph solved AT the converged point: mean ≈ c,
    # covariance = (Jᵀ Σ⁻¹ J + prior curvature)⁻¹ there
    m, V = graph_posterior(c, μ, Jz, d; bp_iters)
    return (z=c, mean=m, cov=V, F=Fc, hist=hist)
end

"""
    compare_gate(res, d; wall) -> Dict

Print and return the gate table: per site the reference MAP, the graph's
fixed point, |Δ| in reference-sd units, and the width ratio; plus the
covariance and objective agreement.
"""
function compare_gate(res, d::GateData; wall=NaN)
    sd_ref = sqrt.(diag(d.cov_map))
    sd_rx = sqrt.(diag(res.cov))
    println()
    @printf("%-10s %10s %10s %9s %9s %9s %9s\n", "site", "z_map(GN)", "z_rx", "|dz|/sd", "sd_GN", "sd_rx", "ratio")
    for i in 1:N_PHYS
        @printf("%-10s %10.4f %10.4f %9.4f %9.4f %9.4f %9.4f\n", d.names[i], d.z_map[i], res.z[i],
                abs(res.z[i] - d.z_map[i]) / sd_ref[i], sd_ref[i], sd_rx[i], sd_rx[i] / sd_ref[i])
    end
    dz = maximum(abs.(res.z .- d.z_map) ./ sd_ref)
    dcov = maximum(abs.(res.cov .- d.cov_map)) / maximum(abs.(d.cov_map))
    # correlation structure, not just the diagonal
    corr(V) = V ./ (sqrt.(diag(V)) * sqrt.(diag(V))')
    dcorr = maximum(abs.(corr(res.cov) .- corr(d.cov_map)))
    println()
    @printf("max |dz| / sd            : %.4f\n", dz)
    @printf("max |dCov| / max |Cov|   : %.2e\n", dcov)
    @printf("max |dCorr|              : %.2e\n", dcorr)
    @printf("F at fixed point / F_map : %.4f / %.4f\n", res.F, d.F_map)
    @printf("outer iterations / wall  : %d / %.0f s\n", length(res.hist), wall)
    return Dict(
        "names" => d.names, "z_map" => d.z_map, "z_rx" => res.z, "sd_gn" => sd_ref,
        "sd_rx" => sd_rx, "cov_rx" => res.cov, "max_dz_over_sd" => dz,
        "max_dcov_rel" => dcov, "max_dcorr" => dcorr, "F_rx" => res.F, "F_map" => d.F_map,
        "iterations" => length(res.hist), "wall_s" => wall,
        "history" => [Dict(pairs(h)) for h in res.hist],
    )
end

# --- the channel registry (sensors as nodes on the same latent) ---------------
include(joinpath(@__DIR__, "channels.jl"))
include(joinpath(@__DIR__, "physics2d.jl"))
include(joinpath(@__DIR__, "noise.jl"))
include(joinpath(@__DIR__, "volatility.jl"))
include(joinpath(@__DIR__, "misfit.jl"))
# --- streaming: full-history refits per batch over the registry ---------------
include(joinpath(@__DIR__, "stream.jl"))
include(joinpath(@__DIR__, "iterated.jl"))
include(joinpath(@__DIR__, "horizon.jl"))
include(joinpath(@__DIR__, "realcook.jl"))
include(joinpath(@__DIR__, "ridge.jl"))
include(joinpath(@__DIR__, "noise_marginal.jl"))
include(joinpath(@__DIR__, "segments.jl"))
include(joinpath(@__DIR__, "smoother.jl"))
include(joinpath(@__DIR__, "drift.jl"))
include(joinpath(@__DIR__, "als.jl"))
include(joinpath(@__DIR__, "sequential.jl"))
# --- the multi-meat inference architecture (docs/multi-meat-inference-plan.md): the belief over a CookSystem, the
# --- semantic cook log, the backend-neutral factors, the online square-root fixed-lag backend, the offline
# --- full-history and hierarchical backend
include(joinpath(@__DIR__, "system_state.jl"))
include(joinpath(@__DIR__, "semantic.jl"))
include(joinpath(@__DIR__, "factors.jl"))
include(joinpath(@__DIR__, "slice_obs.jl"))
include(joinpath(@__DIR__, "online.jl"))
include(joinpath(@__DIR__, "posterior_sampling.jl"))
include(joinpath(@__DIR__, "posterior_smc.jl"))
include(joinpath(@__DIR__, "sampled_online.jl"))
include(joinpath(@__DIR__, "particle_online.jl"))
include(joinpath(@__DIR__, "sampled_offline.jl"))
include(joinpath(@__DIR__, "bench_fit.jl"))
include(joinpath(@__DIR__, "offline.jl"))
include(joinpath(@__DIR__, "conditional.jl"))
include(joinpath(@__DIR__, "conditional_transport.jl"))
include(joinpath(@__DIR__, "conditional_density.jl"))
include(joinpath(@__DIR__, "conditional_online.jl"))
include(joinpath(@__DIR__, "deterministic_online.jl"))
include(joinpath(@__DIR__, "plan_system.jl"))
include(joinpath(@__DIR__, "sampled_planning.jl"))
include(joinpath(@__DIR__, "rom.jl"))

end # module
