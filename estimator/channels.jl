# The channel registry: sensors as measurement operators on the monolith
# trajectory, and static readings on latents. Included by BrisketEstimator.
#
# Every channel is a linear map of ONE stacked monolith output vector
#   S(z) = [T_pit(t); T_surf(t); vec(T_field); vec(W_field); vec(F_field); p_v,pit(t)]
# (plus, for a probe, the interpolation position and, for a scale, the
# thickness — both latents the chain rule handles analytically), so the
# whole registry needs one solve and one forward-mode Jacobian of S per
# outer step, whatever the sensor set. A channel provides its predicted
# readings, their Jacobian w.r.t. the full latent vector, its whitening
# (AR(1) per contiguous run of its mask, or iid), and optionally data.
# A missing sensor is an absent channel; an unmeasured quantity is a
# wider marginal. No fallback branches.
#
# Latent vector: the 15 physics latents (unconstrained, as in the gate)
# followed by estimator-only latents (a probe's depth, ...), each with a
# PriorSite; channels address extras by index.

using LinearAlgebra

export Channel, PitProbe, MeatProbe, SurfaceIR, HumiditySensor, MassScale, StaticReading,
       LatentSpec, physics_outputs, assemble, fisher_cov, contiguous_runs, whitening_rows,
       OutputLayout

struct OutputLayout
    nx::Int
    nt::Int
    oTpit::Int
    oTsurf::Int
    oT::Int
    oW::Int
    oF::Int
    opv::Int
end
function OutputLayout(nx::Int, nt::Int)
    b = nx * nt
    return OutputLayout(nx, nt, 0, nt, 2nt, 2nt + b, 2nt + 2b, 2nt + 3b)
end
Base.length(lay::OutputLayout) = 3 * lay.nt + 3 * lay.nx * lay.nt

"""
    physics_outputs(z, d; area_latent=0) -> (S, J_S, lay)

The stacked monolith outputs on the log grid at the latents `z` (the
first 15 entries are the physics latents), and their Jacobian w.r.t. the
FULL latent vector. With `area_latent > 0` the exposed area is the
LogNormal latent at that index and enters the solve as the monolith's
runtime `A_meat` (one more ForwardDiff column); other estimator-only
latents get zero columns here and the chain rule in their channels.
"""
function physics_outputs(z, d::GateData; area_latent::Int=0)
    zp = z[1:N_PHYS]
    x = from_z(zp, d.prior[1:N_PHYS])
    kw = (pit=d.pit, cfg=d.cfg, abstol=d.abstol, reltol=d.reltol)
    stack(o) = vcat(o.Tpit, o.Tsurf, vec(o.T_field), vec(o.W_field), vec(o.F_field), o.pv_pit)
    Jz = nothing
    if area_latent > 0
        A = exp(z[area_latent])
        v = vcat(x, A)
        g(v) = stack(monolith_solve(v[1:N_THETA], v[(N_THETA + 1):N_PHYS], d.u, d.t, d.t;
                                    kw..., A_meat=v[end]))
        o = monolith_solve(x[1:N_THETA], x[(N_THETA + 1):end], d.u, d.t, d.t; kw..., A_meat=A)
        S = stack(o)
        cfgj = ForwardDiff.JacobianConfig(g, v, ForwardDiff.Chunk{length(v)}())
        Jv = ForwardDiff.jacobian(g, v, cfgj)
        Jz = zeros(length(S), length(z))
        Jz[:, 1:N_PHYS] = Jv[:, 1:N_PHYS] .* permutedims(dx_dz(zp, d.prior[1:N_PHYS]))
        Jz[:, area_latent] = Jv[:, end] .* A
    else
        θ, φ = x[1:N_THETA], x[(N_THETA + 1):end]
        o = monolith_solve(θ, φ, d.u, d.t, d.t; kw...)
        S = stack(o)
        Jx = monolith_jacobian(stack, θ, φ, d.u, d.t, d.t; kw...)
        Jz = zeros(length(S), length(z))
        Jz[:, 1:N_PHYS] = Jx .* permutedims(dx_dz(zp, d.prior[1:N_PHYS]))
    end
    lay = OutputLayout(size(o.T_field, 1), length(d.t))
    return S, Jz, lay
end

# --- whitening -----------------------------------------------------------------

"Index runs of contiguous `true` in a mask (lid events break the AR(1) chain)."
function contiguous_runs(mask::AbstractVector{Bool})
    runs = Vector{UnitRange{Int}}()
    i = 1
    n = length(mask)
    while i <= n
        if mask[i]
            j = i
            while j < n && mask[j + 1]
                j += 1
            end
            push!(runs, i:j)
            i = j + 1
        else
            i += 1
        end
    end
    return runs
end

"""
    whitening_rows(mask, σ, ρ) -> L (n_w × n)

The masked AR(1) whitening of a series of length `n`: the first point of
each run carries the marginal scale σ, the rest the innovations
σ√(1 − ρ²) with the lag-1 term — `Whitened.build` of the Python stack.
ρ = 0 is iid.
"""
function whitening_rows(mask::AbstractVector{Bool}, σ::Real, ρ::Real)
    n = length(mask)
    innov = σ * sqrt(max(1 - ρ * ρ, 1e-8))
    rows = Vector{Vector{Float64}}()
    for run in contiguous_runs(mask)
        r = zeros(n); r[first(run)] = 1 / σ; push!(rows, r)
        for k in (first(run) + 1):last(run)
            r = zeros(n); r[k] = 1 / innov; r[k - 1] = -ρ / innov; push!(rows, r)
        end
    end
    return isempty(rows) ? zeros(0, n) : reduce(vcat, permutedims.(rows))
end

# --- channels ---------------------------------------------------------------

abstract type Channel end

"Pit gas temperature [K] on the log grid."
struct PitProbe <: Channel
    σ::Float64
    ρ::Float64
    mask::Vector{Bool}
    data::Union{Nothing,Vector{Float64}}
end
PitProbe(σ, ρ, mask; data=nothing) = PitProbe(σ, ρ, collect(Bool, mask), data)

"""
Meat probe at a position inside the half-slab, read from the temperature
field by linear interpolation in the material coordinate. `s` is the
fraction of the half-thickness from the CENTER (0 = center, 1 = surface);
either fixed, or `1 − depth/L` with `depth` [m from the surface] fixed
or the latent `depth_latent` (LogNormal site, so depth = exp(z)).
"""
struct MeatProbe <: Channel
    s_fixed::Float64            # NaN when depth-parametrized
    depth_fixed::Float64        # NaN when the depth is a latent
    depth_latent::Int           # 0 when not a latent
    σ::Float64
    ρ::Float64
    mask::Vector{Bool}
    data::Union{Nothing,Vector{Float64}}
end
MeatProbe(σ, ρ, mask; s=NaN, depth=NaN, depth_latent=0, data=nothing) =
    MeatProbe(s, depth, depth_latent, σ, ρ, collect(Bool, mask), data)

"Meat surface temperature [K] (an IR reading of the exposed face)."
struct SurfaceIR <: Channel
    σ::Float64
    ρ::Float64
    mask::Vector{Bool}
    data::Union{Nothing,Vector{Float64}}
end
SurfaceIR(σ, ρ, mask; data=nothing) = SurfaceIR(σ, ρ, collect(Bool, mask), data)

"Pit vapor pressure [Pa] (a humidity sensor; the wet-bulb is a psychrometric map of it)."
struct HumiditySensor <: Channel
    σ::Float64
    ρ::Float64
    mask::Vector{Bool}
    data::Union{Nothing,Vector{Float64}}
end
HumiditySensor(σ, ρ, mask; data=nothing) = HumiditySensor(σ, ρ, collect(Bool, mask), data)

"""
Meat mass on a scale [kg] at the log times in `mask`:
m = ρ_d A_meat L (1 + W̄ + F̄), the model's own bookkeeping (purge and
rendered fat leave). The exposed area is the latent `area_latent` when
given (its column in J then carries the pit coupling as well), else the
constant `A_meat`; L is the thickness latent.
"""
struct MassScale <: Channel
    σ::Float64
    ρ::Float64
    mask::Vector{Bool}
    A_meat::Float64
    area_latent::Int
    data::Union{Nothing,Vector{Float64}}
end
MassScale(σ, ρ, mask; A_meat=0.12, area_latent=0, data=nothing) =
    MassScale(σ, ρ, collect(Bool, mask), A_meat, area_latent, data)

"A direct reading of one latent's constrained value (a ruler on L, an inserted depth, ...)."
struct StaticReading <: Channel
    latent::Int
    σ::Float64
    data::Union{Nothing,Float64}
end
StaticReading(latent, σ; data=nothing) = StaticReading(latent, σ, data)

const RHO_DRY_EST = 320.0     # thermophysics.RHO_DRY
const L_INDEX = 4             # θ[4] = L in the physics block

# material-coordinate mean weights (half cells at the ends): fv_mean
fv_weights(nx) = (w = ones(nx); w[1] = 0.5; w[end] = 0.5; w ./ (nx - 1))

_col(lay, o, k) = (o + (k - 1) * lay.nx + 1):(o + k * lay.nx)   # rows of field column k in S

# predicted readings and their Jacobian (n_ch × n_z), unwhitened
function predict(ch::PitProbe, S, J, z, prior, lay)
    idx = lay.oTpit .+ (1:lay.nt)
    return S[idx], J[idx, :]
end
function predict(ch::SurfaceIR, S, J, z, prior, lay)
    idx = lay.oTsurf .+ (1:lay.nt)
    return S[idx], J[idx, :]
end
function predict(ch::HumiditySensor, S, J, z, prior, lay)
    idx = lay.opv .+ (1:lay.nt)
    return S[idx], J[idx, :]
end
function predict(ch::MeatProbe, S, J, z, prior, lay)
    nx, nt = lay.nx, lay.nt
    L = exp(z[L_INDEX])
    if !isnan(ch.s_fixed)
        s, ds_dzL, ds_dzd = ch.s_fixed, 0.0, 0.0
    else
        depth = ch.depth_latent > 0 ? exp(z[ch.depth_latent]) : ch.depth_fixed
        s = 1 - depth / L
        ds_dzL = depth / L                       # d s / d log L
        ds_dzd = ch.depth_latent > 0 ? -depth / L : 0.0   # d s / d log depth
    end
    sc = clamp(s, 0.0, 1.0)
    pos = 1 + sc * (nx - 1)
    i = clamp(floor(Int, pos), 1, nx - 1)
    λ = pos - i
    r = zeros(nt)
    Jr = zeros(nt, length(z))
    for k in 1:nt
        c = _col(lay, lay.oT, k)
        Ti, Tj = S[c[i]], S[c[i + 1]]
        r[k] = (1 - λ) * Ti + λ * Tj
        Jr[k, :] = (1 - λ) .* J[c[i], :] .+ λ .* J[c[i + 1], :]
        dT_ds = (Tj - Ti) * (nx - 1) * (0.0 <= s <= 1.0)   # zero when clamped
        Jr[k, L_INDEX] += dT_ds * ds_dzL
        ch.depth_latent > 0 && (Jr[k, ch.depth_latent] += dT_ds * ds_dzd)
    end
    return r, Jr
end
function predict(ch::MassScale, S, J, z, prior, lay)
    nx, nt = lay.nx, lay.nt
    L = exp(z[L_INDEX])
    A = ch.area_latent > 0 ? exp(z[ch.area_latent]) : ch.A_meat
    w = fv_weights(nx)
    r = zeros(nt)
    Jr = zeros(nt, length(z))
    base = RHO_DRY_EST * A * L
    for k in 1:nt
        cW = _col(lay, lay.oW, k)
        cF = _col(lay, lay.oF, k)
        Wm = dot(w, S[cW]); Fm = dot(w, S[cF])
        r[k] = base * (1 + Wm + Fm)
        Jr[k, :] = base .* (permutedims(w) * J[cW, :] .+ permutedims(w) * J[cF, :])[1, :]
        Jr[k, L_INDEX] += base * (1 + Wm + Fm)          # d/d log L of the explicit factor
        ch.area_latent > 0 && (Jr[k, ch.area_latent] += base * (1 + Wm + Fm))  # d/d log A, explicit
    end
    return r, Jr
end
function predict(ch::StaticReading, S, J, z, prior, lay)
    p = prior[ch.latent]
    x = isbeta(p) ? logistic(z[ch.latent]) : exp(z[ch.latent])
    dx = isbeta(p) ? x * (1 - x) : x
    Jr = zeros(1, length(z)); Jr[1, ch.latent] = dx
    return [x], Jr
end

whitening(ch::PitProbe) = whitening_rows(ch.mask, ch.σ, ch.ρ)
whitening(ch::SurfaceIR) = whitening_rows(ch.mask, ch.σ, ch.ρ)
whitening(ch::HumiditySensor) = whitening_rows(ch.mask, ch.σ, ch.ρ)
whitening(ch::MeatProbe) = whitening_rows(ch.mask, ch.σ, ch.ρ)
whitening(ch::MassScale) = whitening_rows(ch.mask, ch.σ, ch.ρ)
whitening(ch::StaticReading) = fill(1 / ch.σ, 1, 1)

rawdata(ch::StaticReading) = ch.data === nothing ? nothing : [ch.data]
rawdata(ch::Channel) = ch.data

"""
    assemble(channels, z, S, J, prior, lay) -> (a, B, w)

Whitened predicted readings `a`, their Jacobian `B` (n_w × n_z) and the
whitened observations `w` (NaN where a channel carries no data) across
the registry — the affine node data of the graph at the linearization
point `z`.
"""
function assemble(channels, z, S, J, prior, lay)
    as, Bs, ws = Vector{Float64}[], Matrix{Float64}[], Vector{Float64}[]
    for ch in channels
        r, Jr = predict(ch, S, J, z, prior, lay)
        Lw = whitening(ch)
        size(Lw, 1) == 0 && continue
        push!(as, Lw * r)
        push!(Bs, Lw * Jr)
        y = rawdata(ch)
        push!(ws, y === nothing ? fill(NaN, size(Lw, 1)) : Lw * y)
    end
    return reduce(vcat, as), reduce(vcat, Bs), reduce(vcat, ws)
end

"""
    fisher_cov(channels, z, S, J, prior, lay) -> (cov, sd)

The Gauss-Newton (Fisher) posterior covariance of the latents under the
registry at `z`: (Σ Bᵀ B + prior curvature)⁻¹. Needs no data — the value
of a sensor before it exists.
"""
function fisher_cov(channels, z, S, J, prior, lay; fix::AbstractVector{<:Integer}=Int[])
    _, B, _ = assemble(channels, z, S, J, prior, lay)
    _, _, h = prior_terms(z, prior)
    H = B' * B + Diagonal(h)
    n = length(z)
    keep = setdiff(1:n, fix)          # latents in `fix` are treated as known (conditioned on)
    cov = fill(NaN, n, n)
    cov[keep, keep] = inv(Symmetric(H[keep, keep]))
    sd = fill(NaN, n)
    sd[keep] = sqrt.(diag(cov[keep, keep]))
    return cov, sd
end

"A prior site for an estimator-only latent, appended after the physics block."
LatentSpec(name::AbstractString, family::AbstractString, p1::Real, p2::Real) =
    (name=String(name), site=PriorSite(String(family), Float64(p1), Float64(p2)))
