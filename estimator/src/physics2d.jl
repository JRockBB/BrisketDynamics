# The physics behind the channel registry, selectable: the 1-D monolith
# (the slab) or the 2-D axisymmetric engine (a finite cylinder), both
# producing the SAME stacked output so every channel works unchanged.
# Included by BrisketEstimator after channels.jl.
#
# The 2-D stack maps the cylinder onto the slab's conventions: the
# "field" lines are the axis (r = 0) sampled at uniform depths from the
# top face to the mid-plane (index 1 the center, index nx the surface,
# the slab's order), the surface temperature is the top face's center,
# the mass is the volume integral, the bark lightness is not available
# (the 2-D bark is a distribution at the end of the cook, not a series)
# and its block is zero. The latents are the physics' fifteen, with the
# thickness L as the half-height (Hz = 2 L) and the draft exponent
# m_conv unused (a zero column: the 2-D engine takes a fixed convective
# coefficient `h`), plus the radius as a LogNormal latent.

export Physics, Physics1D, Physics2D, geometry_latents

abstract type Physics end

"""
    Physics1D(; area_latent=0, events=NamedTuple())

The monolith slab; `area_latent` > 0 makes the exposed area a latent
(runtime `A_meat`); `events` are the executed cook events as
`monolith_solve` keywords (`t_wrap`, `t_pull`, `t_add`, `t_spritz`,
`dt_spritz`, `m_spritz`), inputs of the physics like the damper log.
"""
struct Physics1D <: Physics
    area_latent::Int
    events::NamedTuple
end
Physics1D(area_latent::Int) = Physics1D(area_latent, NamedTuple())
Physics1D(; area_latent::Int=0, events::NamedTuple=NamedTuple()) = Physics1D(area_latent, events)

"""
    Physics2D(; radius_latent, nr=16, nz=14, h=30.0, n_meat=1.0, nx_line=nz)

The 2-D axisymmetric engine as the estimator's forward model: a cylinder
of radius `exp(z[radius_latent])` and height 2 L on an `nr` × `nz` grid,
the pit coupled with a fixed convective coefficient `h`; the axis line
sampled at `nx_line` uniform depths for the probe channels.
"""
struct Physics2D <: Physics
    radius_latent::Int
    nr::Int
    nz::Int
    h::Float64
    n_meat::Float64
    nx_line::Int
end
Physics2D(; radius_latent::Int, nr::Int=16, nz::Int=14, h::Real=30.0, n_meat::Real=1.0,
          nx_line::Int=nz) = Physics2D(radius_latent, nr, nz, float(h), float(n_meat), nx_line)

"The indices of the geometry latents a physics adds beyond the fifteen (area, or radius)."
geometry_latents(ph::Physics1D) = ph.area_latent > 0 ? [ph.area_latent] : Int[]
geometry_latents(ph::Physics2D) = [ph.radius_latent]

# --- the 1-D stack ---------------------------------------------------------------

# mass on the scale from the slab's own bookkeeping: ρ_d A L (1 + W̄ + F̄)
function _mass_1d(o, A, L)
    nx = size(o.T_field, 1)
    w = fv_weights(nx)
    return [RHO_DRY_EST * A * L * (1 + dot(w, view(o.W_field, :, k)) + dot(w, view(o.F_field, :, k)))
            for k in axes(o.T_field, 2)]
end

# the stacked output of a monolith solve at exposed area A and thickness L
_stack_1d(o, A, L) = vcat(o.Tpit, o.Tsurf, vec(o.T_field), vec(o.W_field), vec(o.F_field), o.pv_pit,
                          o.m_fuel, o.l_star, _mass_1d(o, A, L))

function physics_outputs(z, d::GateData, ph::Physics1D)
    zp = z[1:N_PHYS]
    x = from_z(zp, d.prior[1:N_PHYS])
    kw = (pit=d.pit, cfg=d.cfg, abstol=d.abstol, reltol=d.reltol, ph.events...)
    Jz = zeros(0, 0)
    if ph.area_latent > 0
        A = exp(z[ph.area_latent])
        v = vcat(x, A)
        g(v) = _stack_1d(monolith_solve(v[1:N_THETA], v[(N_THETA + 1):N_PHYS], d.u, d.t, d.t;
                                        kw..., A_meat=v[end]), v[end], v[4])
        o = monolith_solve(x[1:N_THETA], x[(N_THETA + 1):end], d.u, d.t, d.t; kw..., A_meat=A)
        S = _stack_1d(o, A, x[4])
        cfgj = ForwardDiff.JacobianConfig(g, v, ForwardDiff.Chunk{length(v)}())
        Jv = ForwardDiff.jacobian(g, v, cfgj)
        Jz = zeros(length(S), length(z))
        Jz[:, 1:N_PHYS] = Jv[:, 1:N_PHYS] .* permutedims(dx_dz(zp, d.prior[1:N_PHYS]))
        Jz[:, ph.area_latent] = Jv[:, end] .* A
    else
        A = d.cfg.A_meat
        v = x
        g0(v) = _stack_1d(monolith_solve(v[1:N_THETA], v[(N_THETA + 1):N_PHYS], d.u, d.t, d.t; kw...),
                          A, v[4])
        o = monolith_solve(x[1:N_THETA], x[(N_THETA + 1):end], d.u, d.t, d.t; kw...)
        S = _stack_1d(o, A, x[4])
        cfgj = ForwardDiff.JacobianConfig(g0, v, ForwardDiff.Chunk{length(v)}())
        Jv = ForwardDiff.jacobian(g0, v, cfgj)
        Jz = zeros(length(S), length(z))
        Jz[:, 1:N_PHYS] = Jv .* permutedims(dx_dz(zp, d.prior[1:N_PHYS]))
    end
    return S, Jz, OutputLayout(size(o.T_field, 1), length(d.t))
end

function physics_stack(z, d::GateData, ph::Physics1D)
    zp = z[1:N_PHYS]
    x = from_z(zp, d.prior[1:N_PHYS])
    kw = (pit=d.pit, cfg=d.cfg, abstol=d.abstol, reltol=d.reltol, ph.events...)
    A = ph.area_latent > 0 ? exp(z[ph.area_latent]) : d.cfg.A_meat
    o = ph.area_latent > 0 ?
        monolith_solve(x[1:N_THETA], x[(N_THETA + 1):end], d.u, d.t, d.t; kw..., A_meat=A) :
        monolith_solve(x[1:N_THETA], x[(N_THETA + 1):end], d.u, d.t, d.t; kw...)
    return _stack_1d(o, A, x[4]), OutputLayout(size(o.T_field, 1), length(d.t))
end

# --- the 2-D stack ---------------------------------------------------------------

const THETA2_INDEX = [1, 2, 3, 5, 6, 7]     # θ2 = θ without L (4) and m_conv (8)

# the axis line of a (nr, nz, nt) field at nx uniform depths from the top
# face (index nx, depth 0) to the mid-plane (index 1, depth L = Hz/2),
# linear in z between grid nodes; the weights are duals when Hz is
function _axis_line(F3, Hz, nx)
    nz, nt = size(F3, 2), size(F3, 3)
    Δz = Hz / (nz - 1)
    L = Hz / 2
    Tv = promote_type(eltype(F3), typeof(Hz))
    out = zeros(Tv, nx, nt)
    for i in 1:nx
        depth = L * (1 - (i - 1) / (nx - 1))
        pos = 1 + (Hz - depth) / Δz                 # fractional z index, 1-based
        pv = ForwardDiff.value(pos)
        j = isfinite(pv) ? clamp(floor(Int, pv), 1, nz - 1) : 1   # a blown-up solve propagates NaN, no throw
        f = pos - j
        for k in 1:nt
            out[i, k] = (1 - f) * F3[1, j, k] + f * F3[1, j + 1, k]
        end
    end
    return out
end

# the cylinder's volume weights (nr, nz), z ends half cells
function _cyl_volume(R, Hz, nr, nz)
    Δr, Δz = R / (nr - 1), Hz / (nz - 1)
    rV = [i == 1 ? (Δr / 2)^2 / 2 : i == nr ? (R^2 - (R - Δr / 2)^2) / 2 : (i - 1) * Δr * Δr for i in 1:nr]
    zw = [j == 1 || j == nz ? 0.5 : 1.0 for j in 1:nz]
    return [2π * rV[i] * zw[j] * Δz for i in 1:nr, j in 1:nz]
end

function _stack_2d(o, R, Hz, nx, F0)
    nr, nz, nt = size(o.T_field)
    vol = _cyl_volume(R, Hz, nr, nz)
    Tline = _axis_line(o.T_field, Hz, nx)
    Wline = _axis_line(o.W_field, Hz, nx)
    Tv = eltype(Tline)
    Fline = fill(Tv(F0), nx, nt)
    mass = [RHO_DRY_EST * sum(vol .* (1 .+ view(o.W_field, :, :, k) .+ F0)) for k in 1:nt]
    Tsurf = o.T_field[1, nz, :]
    return vcat(o.Tpit, Tsurf, vec(Tline), vec(Wline), vec(Fline), o.pv_pit, o.m_fuel,
                zeros(Tv, nt), mass)
end

function _solve_2d(x, R, d::GateData, ph::Physics2D)
    θ2 = x[THETA2_INDEX]
    φ = x[(N_THETA + 1):N_PHYS]
    return coupled_solve_2d(θ2, φ, d.u, d.t, d.t; R=R, Hz=2 * x[4], nr=ph.nr, nz=ph.nz,
                            cfg=d.cfg, pit=d.pit, hum=Humidity(), h=ph.h, n_meat=ph.n_meat)
end

function physics_outputs(z, d::GateData, ph::Physics2D)
    zp = z[1:N_PHYS]
    x = from_z(zp, d.prior[1:N_PHYS])
    R = exp(z[ph.radius_latent])
    v = vcat(x, R)
    g(v) = _stack_2d(_solve_2d(v[1:N_PHYS], v[end], d, ph), v[end], 2 * v[4], ph.nx_line, d.cfg.F0)
    o = _solve_2d(x, R, d, ph)
    S = _stack_2d(o, R, 2 * x[4], ph.nx_line, d.cfg.F0)
    cfgj = ForwardDiff.JacobianConfig(g, v, ForwardDiff.Chunk{length(v)}())
    Jv = ForwardDiff.jacobian(g, v, cfgj)
    Jz = zeros(length(S), length(z))
    Jz[:, 1:N_PHYS] = Jv[:, 1:N_PHYS] .* permutedims(dx_dz(zp, d.prior[1:N_PHYS]))
    Jz[:, ph.radius_latent] = Jv[:, end] .* R
    return S, Jz, OutputLayout(ph.nx_line, length(d.t))
end

function physics_stack(z, d::GateData, ph::Physics2D)
    zp = z[1:N_PHYS]
    x = from_z(zp, d.prior[1:N_PHYS])
    R = exp(z[ph.radius_latent])
    o = _solve_2d(x, R, d, ph)
    return _stack_2d(o, R, 2 * x[4], ph.nx_line, d.cfg.F0), OutputLayout(ph.nx_line, length(d.t))
end

# the keyword forms keep working: the slab, with or without the area latent
physics_outputs(z, d::GateData; area_latent::Int=0) = physics_outputs(z, d, Physics1D(area_latent))
physics_stack(z, d::GateData; area_latent::Int=0) = physics_stack(z, d, Physics1D(area_latent))
