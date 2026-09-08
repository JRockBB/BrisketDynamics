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
       FuelScale, BarkLightness, WetBulb,
       LatentSpec, physics_outputs, physics_stack, assemble, fisher_cov, contiguous_runs,
       whitening_rows, OutputLayout

struct OutputLayout
    nx::Int
    nt::Int
    oTpit::Int
    oTsurf::Int
    oT::Int
    oW::Int
    oF::Int
    opv::Int
    oM::Int      # fuel mass [kg]
    oL::Int      # bark lightness L*
    oMass::Int   # meat mass on a scale [kg], the physics' own bookkeeping
end
function OutputLayout(nx::Int, nt::Int)
    b = nx * nt
    return OutputLayout(nx, nt, 0, nt, 2nt, 2nt + b, 2nt + 2b, 2nt + 3b, 3nt + 3b, 4nt + 3b,
                        5nt + 3b)
end
Base.length(lay::OutputLayout) = 6 * lay.nt + 3 * lay.nx * lay.nt

# ONE stacked output for every channel, in the order OutputLayout indexes
# it: pit and surface temperature, the temperature, water and fat lines
# (center to surface), the pit vapor pressure, the fuel mass, the bark
# lightness, the meat mass. physics2d.jl builds it for the slab and for
# the cylinder.

"""
    physics_outputs(z, d, physics) -> (S, J_S, lay)
    physics_outputs(z, d; area_latent=0)

The stacked outputs on the log grid at the latents `z` (the first 15
entries are the physics latents) and their Jacobian w.r.t. the FULL
latent vector, for the slab (`Physics1D`, the keyword form) or the
cylinder (`Physics2D`); physics2d.jl. Estimator-only latents that the
physics does not see get zero columns here and the chain rule in their
channels.

    physics_stack(z, d, physics) -> (S, lay)

The stacked outputs alone (one solve, no Jacobian), for objective
evaluations in a line search.
"""
function physics_outputs end
function physics_stack end

# --- whitening -----------------------------------------------------------------

"""
    contiguous_runs(mask; maxgap=1) -> Vector{Vector{Int}}

Index runs of the sampled points of a mask: consecutive samples closer
than or equal to `maxgap` grid steps belong to one run, a longer gap
(a lid lift, a dropout) breaks it. `maxgap = 1` is the old contiguous
rule; [`channel_gap`](@ref) gives a channel's own spacing.
"""
function contiguous_runs(mask::AbstractVector{Bool}; maxgap::Int=1)
    idx = findall(mask)
    runs = Vector{Vector{Int}}()
    isempty(idx) && return runs
    cur = [idx[1]]
    for k in 2:length(idx)
        if idx[k] - idx[k - 1] <= maxgap
            push!(cur, idx[k])
        else
            push!(runs, cur)
            cur = [idx[k]]
        end
    end
    push!(runs, cur)
    return runs
end

"A channel's typical spacing in grid steps (the median gap between its samples; 1 for a full mask)."
function channel_gap(mask::AbstractVector{Bool})
    idx = findall(mask)
    length(idx) < 2 && return 1
    gaps = diff(idx)
    return max(1, Int(round(sort(gaps)[div(length(gaps) + 1, 2)])))
end

"A run breaks at a gap longer than three times the channel's own spacing; a full-rate channel keeps the calibration's rule that any gap is an event."
run_maxgap(mask::AbstractVector{Bool}) = (g = channel_gap(mask); g == 1 ? 1 : 3g)

function _contiguous_runs_old(mask::AbstractVector{Bool})
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

An Ornstein–Uhlenbeck discrepancy on the physics grid: `ρ` is the
correlation over ONE grid step, and two samples `g` steps apart are
correlated `ρ^g`, so a channel at any rate carries the same physical
correlation time; a run starts fresh (row 1/σ) and breaks only at a
gap longer than three times the channel's own spacing.

The masked AR(1) whitening of a series of length `n`: the first point of
each run carries the marginal scale σ, the rest the innovations
σ√(1 − ρ²) with the lag-1 term — `Whitened.build` of the Python stack.
ρ = 0 is iid.
"""
function whitening_rows(mask::AbstractVector{Bool}, σ::Real, ρ::Real)
    n = length(mask)
    rows = Vector{Vector{Float64}}()
    for run in contiguous_runs(mask; maxgap=run_maxgap(mask))
        r = zeros(n); r[run[1]] = 1 / σ; push!(rows, r)
        for j in 2:length(run)
            k, kprev = run[j], run[j - 1]
            ρg = ρ^(k - kprev)                       # the correlation over this gap
            innov = σ * sqrt(max(1 - ρg * ρg, 1e-8))
            r = zeros(n); r[k] = 1 / innov; r[kprev] = -ρg / innov; push!(rows, r)
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
Meat mass on a scale [kg] at the log times in `mask`: the physics' own
mass bookkeeping (the stack's mass block: ρ_d A L (1 + W̄ + F̄) for the
slab, the volume integral for the cylinder; purge and rendered fat
leave), times a shape factor s = exp(z[shape_latent]) when
`shape_latent` is given (s = 1 for a slab; for a cylinder read with the
slab physics s = πR²H / (A L)), so a plant that is not a slab can keep
its mass bookkeeping honest without bending the physics. The area and
thickness columns come through the physics' Jacobian.
"""
struct MassScale <: Channel
    σ::Float64
    ρ::Float64
    mask::Vector{Bool}
    shape_latent::Int
    data::Union{Nothing,Vector{Float64}}
end
MassScale(σ, ρ, mask; shape_latent=0, data=nothing, area_latent=0, A_meat=0.12) =   # area_latent, A_meat: accepted for call compatibility, the area column comes through the physics
    MassScale(σ, ρ, collect(Bool, mask), shape_latent, data)

"Fuel mass on a scale under the firebox [kg]: the pit model's fuel state."
struct FuelScale <: Channel
    σ::Float64
    ρ::Float64
    mask::Vector{Bool}
    data::Union{Nothing,Vector{Float64}}
end
FuelScale(σ, ρ, mask; data=nothing) = FuelScale(σ, ρ, collect(Bool, mask), data)

"Bark lightness L* from a colorimeter or a calibrated camera: the model's L* map of its bark states."
struct BarkLightness <: Channel
    σ::Float64
    ρ::Float64
    mask::Vector{Bool}
    data::Union{Nothing,Vector{Float64}}
end
BarkLightness(σ, ρ, mask; data=nothing) = BarkLightness(σ, ρ, collect(Bool, mask), data)

"""
    WetBulb(σ, ρ, mask; p_atm=101325, a_psy=6.62e-4, data=nothing)

A wetted thermocouple in the pit (smokehouse practice): the wet-bulb
temperature T_wb solving the psychrometer equation
p_sat(T_wb) − a_psy p_atm (T_pit − T_wb) = p_v,pit for a ventilated wick,
an analytic map of two stacked outputs with its Jacobian by implicit
differentiation. It reads the vapor pressure through the vapor balance's
own constants, which is what makes it the C2 validation of that balance.
"""
struct WetBulb <: Channel
    σ::Float64
    ρ::Float64
    mask::Vector{Bool}
    p_atm::Float64
    a_psy::Float64
    data::Union{Nothing,Vector{Float64}}
end
WetBulb(σ, ρ, mask; p_atm=BrisketDynamics.P_ATM, a_psy=6.62e-4, data=nothing) =
    WetBulb(σ, ρ, collect(Bool, mask), p_atm, a_psy, data)

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
    idx = (lay.oMass + 1):(lay.oMass + lay.nt)
    s = ch.shape_latent > 0 ? exp(z[ch.shape_latent]) : 1.0
    r = s .* S[idx]
    Jr = s .* J[idx, :]
    ch.shape_latent > 0 && (Jr[:, ch.shape_latent] .+= r)      # d/d log s
    return r, Jr
end
function predict(ch::FuelScale, S, J, z, prior, lay)
    idx = (lay.oM + 1):(lay.oM + lay.nt)
    return S[idx], J[idx, :]
end
function predict(ch::BarkLightness, S, J, z, prior, lay)
    idx = (lay.oL + 1):(lay.oL + lay.nt)
    return S[idx], J[idx, :]
end

# d p_sat / dT of the Magnus form in thermophysics.jl
_dpsat(T) = BrisketDynamics.p_sat(T) * 17.625 * (273.15 - 30.11) / (T - 30.11)^2

"""
    wet_bulb(T, pv, γ) -> (T_wb, dF/dT_wb)

The root of F(T_wb) = p_sat(T_wb) + γ (T_wb − T) − pv by Newton from T
(F is increasing and convex in T_wb, so the iteration from the right
converges monotonically); returns the root and F' there, the
denominator of the implicit derivatives dT_wb/dT = γ/F', dT_wb/dpv = 1/F'.
"""
function wet_bulb(T, pv, γ)
    Tw = T
    fp = _dpsat(Tw) + γ
    for _ in 1:60
        f = BrisketDynamics.p_sat(Tw) + γ * (Tw - T) - pv
        fp = _dpsat(Tw) + γ
        step = f / fp
        Tw -= step
        abs(step) < 1e-10 && break
    end
    return Tw, _dpsat(Tw) + γ
end
function predict(ch::WetBulb, S, J, z, prior, lay)
    nt = lay.nt
    γ = ch.a_psy * ch.p_atm
    Tel = promote_type(eltype(S), eltype(J))      # dual-capable, for tests through the map
    r = zeros(Tel, nt)
    Jr = zeros(Tel, nt, length(z))
    for k in 1:nt
        iT, ip = lay.oTpit + k, lay.opv + k
        Tw, fp = wet_bulb(S[iT], S[ip], γ)
        r[k] = Tw
        Jr[k, :] = (γ / fp) .* J[iT, :] .+ (1 / fp) .* J[ip, :]
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

# the samples whose residuals are the channel's noise process (a censored reading is not one)
noise_mask(ch::Channel) = ch.mask
# independent rows a channel appends after its AR rows (none by default); level[k] scales the sd at k
extra_rows(ch::Channel, level) = zeros(0, length(level))
# the leading rows the misfit check scores (the AR rows; the extra rows are not unit-power residuals)
misfit_rows(ch::Channel) = size(whitening(ch), 1)
whitening(ch::PitProbe) = whitening_rows(ch.mask, ch.σ, ch.ρ)
whitening(ch::SurfaceIR) = whitening_rows(ch.mask, ch.σ, ch.ρ)
whitening(ch::HumiditySensor) = whitening_rows(ch.mask, ch.σ, ch.ρ)
whitening(ch::MeatProbe) = whitening_rows(ch.mask, ch.σ, ch.ρ)
whitening(ch::MassScale) = whitening_rows(ch.mask, ch.σ, ch.ρ)
whitening(ch::FuelScale) = whitening_rows(ch.mask, ch.σ, ch.ρ)
whitening(ch::BarkLightness) = whitening_rows(ch.mask, ch.σ, ch.ρ)
whitening(ch::WetBulb) = whitening_rows(ch.mask, ch.σ, ch.ρ)
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
