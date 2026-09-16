# The online estimator as a factor graph (RxInfer) over the BrisketDynamics
# monolith — step 2 of the review's item 8: the Gauss-Newton / Laplace
# calibration of the real cook, reproduced by message passing.
#
# Structure (what step 1 established, docs/results.md "RxInfer step 1"):
#   * ONE multivariate latent block z (the 15 physics latents in the
#     unconstrained space: log for LogNormal sites, logit for Beta sites) —
#     separate blocks feeding one node are mean-field in RxInfer and return
#     conditional, not marginal, variances;
#   * one node PER WHITENED RESIDUAL, each with a scalar output — a node
#     with more outputs than latent inputs has a singular forward output
#     covariance and RxInfer silently returns the prior;
#   * the physics enters each node AFFINE around a linearization point c
#     (f(c) + J(c)(z − c), J by ForwardDiff through the FBDF solve), so the
#     Linearization rule is exact and an OUTER loop moves c: Gauss-Newton
#     with the prior intact, expressed as message passing;
#   * the prior node is the local quadratic of −log p(z) at c (exact for the
#     Gaussian sites, the logit-Beta curvature for the Beta sites), so the
#     fixed point and the covariance are those of scripts/run_gn_map.py.
# Sensors are then nodes on the same latent: this file carries the two
# probe channels of the real cook (as the whitened AR(1) residuals the
# calibration uses); the channel registry comes next.
module BrisketEstimator

using RxInfer
using ForwardDiff
using LinearAlgebra
using Random
using JSON
using Printf
using BrisketDynamics

export ridge_chain, noise_marginal_cov, FilterPrior, prior_quadratic, handoff, solve_segments
export DriftLayout, PhysicsDrift, DriftPrior, smoother_fit, smoother_step, drift_layout, warm_start, segment_edges, index_at, current_index, block_at
export DriftDraw, truth_at, drift_segments, solve_drifting, smoother_moving, DRIFT_SD_PER_HOUR, JUMP_LOG_SD, drift_rates
export FilterLayout, PhysicsState, pack_state, unpack_state, state_sites, disc_sites, sequential_filter, pit_index, centre_index, surface_index
export FilterState, StateNoise, Past, filter_init, filter_update!, filter_advance!, state_range, disc_range, disc_index, state_now, innovation_misfit
export GateData, load_gate, from_z, to_z, dx_dz, z_center, prior_terms, physics_mean, physics_mean_jac,
       objective_value, solve_map, compare_gate, PriorSite, N_PHYS, N_THETA, isbeta, islogbeta, islinear, logistic,
       latent_block, read_sites, draw_truth, FAMILIES

const N_THETA = 8
const N_PHYS = 15

"""
    PriorSite(family, p1, p2[, p3, p4])

A site of the latent block in its family, the transform z -> x and the prior in one:
- `"lognormal"`: x = exp(z), z ~ N(p1, p2²) (a positive site);
- `"beta"`: x = logistic(z), x ~ Beta(p1, p2) (a bounded site, the prior exact in the logit);
- `"logbeta"`: x = exp(p3 + (p4 − p3) logistic(z)), the fraction ~ Beta(p1, p2) (a positive site whose
  calibration chain is skewed: bounded in the log to [p3, p4], the skew in the Beta);
- `"linear"`: x = z, z ~ N(p1, p2²) (a plant state in the filter's block, in its own units).
"""
struct PriorSite
    family::String
    p1::Float64
    p2::Float64
    p3::Float64
    p4::Float64
    function PriorSite(family::AbstractString, p1::Real, p2::Real, p3::Real=0.0, p4::Real=1.0)
        family in FAMILIES || throw(ArgumentError("unknown prior family \"$family\" (one of $(join(FAMILIES, ", ")))"))
        return new(String(family), Float64(p1), Float64(p2), Float64(p3), Float64(p4))
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
`estimator/export_gate_data.py`).
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
_betalike(p::PriorSite) = isbeta(p) || islogbeta(p)    # the prior is a Beta on a fraction f = logistic(z)

"The prior's centre in z: the Gaussian's mean, or the logit of a Beta's mean (a/(a+b))."
z_center(p::PriorSite) = _betalike(p) ? log(p.p1 / p.p2) : p.p1

"The sites of a prior file's `prior` list (`prior_next_cook.json`, the gate file)."
read_sites(list) = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"]), Float64(get(p, "p3", 0.0)), Float64(get(p, "p4", 1.0))) for p in list]

"""
    latent_block(prior_path, cfg; area_sd=0.3, depth_frac=0.8, depth_sd=0.5)

The estimator's latent block, declared once: the fifteen physics sites of the prior file in their families and
names, then the two estimator-only sites, the exposed area (`A_meat`, lognormal about the config's) and the meat
probe's depth (lognormal about `depth_frac` of the thickness's median). Returns `(sites, names, prior_phys,
phys_names, area, depth, idx, file)`; every driver reads its block from here.
"""
function latent_block(prior_path::AbstractString, cfg::MeatConfig; area_sd::Float64=0.3, depth_frac::Float64=0.8, depth_sd::Float64=0.5)
    file = JSON.parsefile(prior_path)
    prior_phys = read_sites(file["prior"])
    length(prior_phys) == N_PHYS || throw(ArgumentError("the prior file carries $(length(prior_phys)) physics sites, not $N_PHYS"))
    phys_names = String.(file["names"])
    L_med = from_z(z_center(prior_phys[4]), prior_phys[4])
    sites = vcat(prior_phys, [PriorSite("lognormal", log(cfg.A_meat), area_sd), PriorSite("lognormal", log(depth_frac * L_med), depth_sd)])
    names = vcat(phys_names, ["A_meat", "core_depth"])
    return (sites=sites, names=names, prior_phys=prior_phys, phys_names=phys_names, area=length(sites) - 1, depth=length(sites),
            idx=Dict(n => i for (i, n) in enumerate(names)), file=file)
end

"An inverse-CDF draw of Beta(a, b) on a fine grid (no Distributions dependency)."
function _beta_rand(rng, a, b)
    x = range(1e-6, 1 - 1e-6; length=4001); lp = (a - 1) .* log.(x) .+ (b - 1) .* log.(1 .- x); p = exp.(lp .- maximum(lp)); c = cumsum(p); c ./= c[end]
    r = rand(rng); k = clamp(searchsortedfirst(c, r), 2, length(c)); x[k - 1] + (x[k] - x[k - 1]) * (r - c[k - 1]) / max(c[k] - c[k - 1], 1e-300)
end
"""
    draw_truth(rng, sites; depth=0, thickness=4) -> z

A truth drawn from the sites' prior, in z (Gaussian for the lognormal and linear sites, a Beta draw in the logit for
the beta and logbeta sites); with `depth` > 0 the probe's depth is kept inside the meat (under 0.9 of the
thickness at site `thickness`).
"""
function draw_truth(rng::AbstractRNG, sites::AbstractVector{PriorSite}; depth::Int=0, thickness::Int=4)
    z = [_betalike(p) ? (f = _beta_rand(rng, p.p1, p.p2); log(f / (1 - f))) : p.p1 + p.p2 * randn(rng) for p in sites]
    depth > 0 && (z[depth] = min(z[depth], log(0.9 * from_z(z[thickness], sites[thickness]))))
    return z
end

"""
    FilterPrior(sites, m, Λ)

A full-covariance Gaussian prior on the latent block in z (a filter's predicted belief: the last posterior
inflated by process noise), carrying the per-site families `sites` for the transforms z -> x. Indexes and
iterates as its sites, so every consumer of a `Vector{PriorSite}` (from_z, the channels, the layouts) reads
it unchanged; `prior_terms` returns the Gaussian's value, gradient and FULL Hessian.
"""
struct FilterPrior
    sites::Vector{PriorSite}
    m::Vector{Float64}
    Λ::Matrix{Float64}
end
Base.getindex(p::FilterPrior, i) = p.sites[i]
Base.length(p::FilterPrior) = length(p.sites)
Base.lastindex(p::FilterPrior) = lastindex(p.sites)
Base.iterate(p::FilterPrior, st...) = iterate(p.sites, st...)
Base.eltype(::Type{FilterPrior}) = PriorSite
function prior_terms(z, p::FilterPrior)
    r = z .- p.m
    return 0.5 * dot(r, p.Λ * r), p.Λ * r, Matrix(p.Λ)
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
logistic(v) = 1 / (1 + exp(-v))

"The constrained value of one site at `v` in z (see `PriorSite`)."
function from_z(v, p::PriorSite)
    isbeta(p) && return logistic(v)
    islogbeta(p) && return exp(p.p3 + (p.p4 - p.p3) * logistic(v))
    islinear(p) && return v
    return exp(v)
end
"dx/dz of one site at `v`."
function dx_dz(v, p::PriorSite)
    isbeta(p) && (s = logistic(v); return s * (1 - s))
    islogbeta(p) && (s = logistic(v); return from_z(v, p) * (p.p4 - p.p3) * s * (1 - s))
    islinear(p) && return one(v)
    return exp(v)
end
"The unconstrained coordinate of one site's value `x` (the inverse of `from_z`)."
function to_z(x, p::PriorSite)
    isbeta(p) && return log(x / (1 - x))
    islogbeta(p) && (f = clamp((log(x) - p.p3) / (p.p4 - p.p3), 1e-9, 1 - 1e-9); return log(f / (1 - f)))
    islinear(p) && return x
    return log(x)
end
from_z(z, prior) = [from_z(v, p) for (v, p) in zip(z, prior)]
dx_dz(z, prior) = [dx_dz(v, p) for (v, p) in zip(z, prior)]
to_z(x::AbstractVector, prior::AbstractVector) = [to_z(v, p) for (v, p) in zip(x, prior)]

"""
    prior_terms(z, prior) -> (f, g, h)

−log p(z), its gradient and diagonal Hessian in the unconstrained space:
Gaussian for the LogNormal and linear sites; for Beta(a, b) sites (and the
logbeta's fraction) the logit-Jacobian form −a·log σ(z) − b·log(1 − σ(z)).
Identical to `brisket_app.monolith.prior_terms` on the calibration's families.
"""
function prior_terms(z, prior)
    T = float(eltype(z))
    f = zero(T)
    g = zeros(T, length(z))
    h = zeros(T, length(z))
    for (i, (v, p)) in enumerate(zip(z, prior))
        if _betalike(p)
            a, b = p.p1, p.p2
            s = logistic(v)
            f += -a * log(s) - b * log(1 - s)
            g[i] = -a * (1 - s) + b * s
            h[i] = (a + b) * s * (1 - s)
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
include(joinpath(@__DIR__, "sequential.jl"))

end # module
