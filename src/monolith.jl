# Monolithic coupled cook: the meat PDE, the pit ODE and the quasi-steady pit
# humidity balance as ONE stiff system. The pit sees the meat's instantaneous
# heat load and the meat sees the instantaneous pit temperature, burn rate
# and vapor pressure — no exchange grid, no Picard truncation, one solve,
# one derivative. States x = [T(nx) | W(nx) | C(nx) | F(nx) | B | S |
# T_pit | m_fuel | ignited]; parameters p = [θ(7) | φ(7) | u_knots(nu) |
# t_wrap | t_pull].
#
# Differentiation: forward mode. The parameter count is small (14 physics
# latents, a few dozen control knots) and the forward solve is ~100x cheaper
# than the continuous adjoint, so ForwardDiff through the solve gives a full
# Jacobian for the cost of a few forward solves — Gauss-Newton, Fisher
# information and Newton-type control follow from it.

"""
    Humidity(; afr, y_w, pan_kg_s, omega_amb, p_atm)

Pit vapor balance, quasi-steady. Air mass flow follows the draft,
`afr · r_max · g(u)`; the base vapor pressure is ambient plus combustion
water plus the water pan, diluted by that flow; the meat's own vapor is
added at the surface through the mixing coefficient.
"""
Base.@kwdef struct Humidity
    afr::Float64 = 20.0         # air-to-fuel ratio of the draft [kg air / kg fuel at r_max]
    y_w::Float64 = 0.28         # combustion water yield [kg H2O / kg charcoal]
    pan_kg_s::Float64 = 0.0     # water-pan evaporation [kg/s]
    omega_amb::Float64 = 0.007  # ambient absolute humidity [kg/kg]
    p_atm::Float64 = P_ATM
end

const MONO_NP = MEAT_NP + PIT_NP

"""
    _pit_algebra(ud, m, a, r_max, u_leak, pit, hum)

The pit's algebraic quantities at one instant from the damper fraction and
the fuel/ignition states: `(airflow, ṁ, mair, pv_base)` — the same
expressions in the RHS and in the observables, so the burn rate and vapor
pressure reported at the save times are exact.
"""
function _pit_algebra(ud, m, a, r_max, u_leak, pit::PitConfig, hum::Humidity)
    airflow = u_leak + (1 - u_leak) * ud
    a_c = clamp(a, zero(a), one(a))
    ṁ = r_max * airflow * a_c * tanh(max(m, zero(m)) / pit.m_taper)
    mair = hum.afr * r_max * airflow
    pv_base =
        (hum.p_atm / EPS_W) * (hum.omega_amb + (hum.y_w * ṁ + hum.pan_kg_s) / mair)
    return airflow, ṁ, mair, pv_base
end

"""
    make_monolith_rhs(cfg, pit, hum, u_t0, u_dt, nu)

Build the in-place coupled RHS over `x = [meat(4·nx + 2) | T_pit, m_fuel,
ignited]` with `p = [θ | φ | u_knots(nu) | t_wrap | t_pull]`; the damper
schedule is interpolated from `nu` knots on the uniform grid
`u_t0 + (i-1)·u_dt`.
"""
function make_monolith_rhs(
    cfg::MeatConfig, pit::PitConfig, hum::Humidity, u_t0::Float64, u_dt::Float64, nu::Int
)
    nm = 4 * cfg.nx + 2
    n_s, n_a, v0 = shrink_exponents(cfg)
    return function mono_rhs!(dx, x, p, t)
        θ = (p[1], p[2], p[3], p[4], p[5], p[6], p[7])
        C_th, UA, r_max, ΔH_eff, u_leak, k_spread = p[8], p[9], p[10], p[11], p[12], p[13]
        u_knots = @view p[(MONO_NP + 1):(MONO_NP + nu)]
        t_wrap = p[MONO_NP + nu + 1]
        t_pull = p[MONO_NP + nu + 2]

        Tp, m, a = x[nm + 1], x[nm + 2], x[nm + 3]
        ud = lininterp(t, u_t0, u_dt, u_knots)
        ud = clamp(ud, zero(ud), one(ud))
        airflow, ṁ, mair, pv_base = _pit_algebra(ud, m, a, r_max, u_leak, pit, hum)

        um = @view x[1:nm]
        dum = @view dx[1:nm]
        q_meat = meat_rhs_core!(
            dum, um, t, θ, Tp, ṁ, pv_base, 1 / mair, t_wrap, t_pull, cfg, n_s, n_a, v0
        )
        a_c = clamp(a, zero(a), one(a))
        dx[nm + 1] = (ΔH_eff * ṁ - UA * (Tp - pit.T_amb) - q_meat) / C_th
        dx[nm + 2] = -ṁ
        dx[nm + 3] = k_spread * a_c * (1 - a_c) * airflow
        return nothing
    end
end

function _monolith_solve_arr(
    p,
    cfg::MeatConfig,
    pit::PitConfig,
    hum::Humidity,
    u_tknots,
    tsave,
    nu::Int;
    abstol=1e-6,
    reltol=1e-6,
    T0f=Float64[],
    W0f=Float64[],
    C0f=Float64[],
    F0f=Float64[],
    B0=NaN,
    S0=NaN,
)
    u_t0, u_dt = float(u_tknots[1]), float(u_tknots[2] - u_tknots[1])
    rhs! = make_monolith_rhs(cfg, pit, hum, u_t0, u_dt, nu)
    Tp = promote_type(eltype(p), Float64)
    # a0 = φ[7] participates in x0 so forward mode carries its sensitivity
    x0 = vcat(
        Tp.(_meat_u0(cfg, T0f, W0f, C0f; F0f, B0, S0)),
        [Tp(pit.Tpit0), Tp(pit.m0), p[MEAT_NP + PIT_NP]],
    )
    prob = ODEProblem{true,SciMLBase.FullSpecialize}(
        rhs!, x0, (float(tsave[1]), float(tsave[end])), p
    )
    sol = solve(prob, FBDF(); saveat=tsave, abstol=abstol, reltol=reltol)
    A = Array(sol)
    (SciMLBase.successful_retcode(sol) && size(A, 2) == length(tsave)) || throw(
        ErrorException(
            "monolith solve failed: retcode=$(sol.retcode), " *
            "saved $(size(A, 2))/$(length(tsave)) points",
        ),
    )
    return A
end

function _monolith_outputs(
    A, p, cfg::MeatConfig, pit::PitConfig, hum::Humidity, u_tknots, tsave, nu::Int
)
    nx = cfg.nx
    nm = 4nx + 2
    nt = length(tsave)
    θ = p[1:MEAT_NP]
    r_max, u_leak = p[MEAT_NP + 3], p[MEAT_NP + 5]
    u_t0, u_dt = float(u_tknots[1]), float(u_tknots[2] - u_tknots[1])
    uk = @view p[(MONO_NP + 1):(MONO_NP + nu)]
    t_wrap, t_pull = p[MONO_NP + nu + 1], p[MONO_NP + nu + 2]

    Tpit = A[nm + 1, :]
    m_fuel = A[nm + 2, :]
    ignited = A[nm + 3, :]
    alg = map(1:nt) do i
        ud = lininterp(float(tsave[i]), u_t0, u_dt, uk)
        _pit_algebra(clamp(ud, zero(ud), one(ud)), m_fuel[i], ignited[i], r_max, u_leak, pit, hum)
    end
    mdot = getindex.(alg, 2)
    imair_ts = 1 ./ getindex.(alg, 3)
    pv_base_ts = getindex.(alg, 4)

    Tsurf = A[nx, :]
    Wsurf = A[2nx, :]
    s_ts = wrap_blend.(tsave, t_wrap, Ref(cfg))
    pl_ts = pull_blend.(tsave, t_pull, Ref(cfg))
    n_s, n_a, v0 = shrink_exponents(cfg)
    wfv = fv_weights(nx)
    Vmat = specific_volume.(A[(nx + 1):2nx, :], A[(3nx + 1):4nx, :], Ref(cfg))
    l_thick = (θ[4] / (nx - 1)) .* vec(wfv' * ((Vmat ./ v0) .^ n_s))
    a_ts = (vec(wfv' * Vmat) ./ ((nx - 1) * v0)) .^ n_a
    c_ts = mixing_coefficient.(a_ts, imair_ts, Ref(cfg))
    j_evap = surface_j_blended.(
        Tsurf, Wsurf, θ[3], θ[2], s_ts, pl_ts, pv_base_ts, c_ts, Ref(cfg)
    )
    pv_pit =
        pv_base_ts .+
        c_ts .* evap_flux_fb.(
            Tsurf, Wsurf, (1 .- cfg.η_evap .* s_ts) .* θ[2], pv_base_ts, c_ts, Ref(cfg)
        )
    q_meat =
        cfg.A_meat .* a_ts .*
        surface_q_pit_blended.(Tpit, Tsurf, θ[3], s_ts, pl_ts, Ref(cfg))
    bark_b = A[4nx + 1, :]
    bark_s = A[4nx + 2, :]
    w_retained = fv_mean(A[(nx + 1):2nx, end]) / cfg.W0
    F_mean = fv_mean(A[(3nx + 1):4nx, end])
    R_ret = (1 - cfg.η_drain) * (cfg.F0 - F_mean)
    dts = diff(collect(float.(tsave)))
    ja = j_evap .* a_ts
    evap_w = sum((ja[1:(end - 1)] .+ ja[2:end]) ./ 2 .* dts) / (RHO_DRY * θ[4])
    return (;
        Tpit,
        m_fuel,
        ignited,
        mdot,
        Tsurf,
        Tcore=A[1, :],
        bark_b,
        bark_s,
        l_star=l_star_bark.(bark_b, bark_s, Ref(cfg)),
        j_evap,
        q_meat,
        C_field_final=A[(2nx + 1):3nx, end],
        w_retained,
        juiciness=w_retained + cfg.β_fat * R_ret / cfg.W0,
        fat_rendered=(cfg.F0 - F_mean) / cfg.F0,
        evap_w,
        drip_w=(cfg.W0 - w_retained * cfg.W0) - evap_w,
        drip_f=cfg.η_drain * (cfg.F0 - F_mean),
        l_thick,
        pv_pit,
        pv_base=pv_base_ts,
        T_field=A[1:nx, :],
        W_field=A[(nx + 1):2nx, :],
        C_field=A[(2nx + 1):3nx, :],
        F_field=A[(3nx + 1):4nx, :],
        t=collect(float.(tsave)),
    )
end

"""
    monolith_solve(θ, φ, u_knots, u_tknots, tsave; cfg, pit, hum, t_wrap,
                   t_pull, T0f, W0f, C0f, F0f, B0, S0, abstol, reltol)

One coupled solve of pit + meat + humidity. Same inputs as the composed
Tesseract pipeline (`coupled_cook`), same outputs plus the pit trajectory
and `pv_base`; accepts ForwardDiff-dual `θ`/`φ`/`u_knots`.

# Arguments
- `θ::AbstractVector`: Meat latents (length `MEAT_NP`; see [`meat_solve`](@ref)).
- `φ::AbstractVector`: Pit latents (length `PIT_NP`; see [`pit_solve`](@ref)).
- `u_knots::AbstractVector`: Damper schedule values ∈ [0, 1] on `u_tknots`.
- `u_tknots::AbstractVector`: Damper knot times [s] (uniform grid).
- `tsave::AbstractVector`: Output save times [s].

# Keyword Arguments
- `cfg::MeatConfig`, `pit::PitConfig`, `hum::Humidity`: Configurations.
- `t_wrap, t_pull::Real`: Wrap and pull times [s]. [Default=never]
- `T0f, W0f, C0f, F0f, B0, S0`: Meat field / bark ICs for a state handoff
  (see [`meat_solve`](@ref)). [Default=uniform]
- `abstol, reltol::Float64`: Solver tolerances. [Default=1e-6]

# Returns
- `out::NamedTuple`: Trajectories on `tsave` and terminal quality observables.
"""
function monolith_solve(
    θ::AbstractVector,
    φ::AbstractVector,
    u_knots::AbstractVector,
    u_tknots::AbstractVector,
    tsave::AbstractVector;
    cfg::MeatConfig=MeatConfig(),
    pit::PitConfig=PitConfig(),
    hum::Humidity=Humidity(),
    t_wrap::Real=WRAP_NEVER,
    t_pull::Real=PULL_NEVER,
    T0f::AbstractVector=Float64[],
    W0f::AbstractVector=Float64[],
    C0f::AbstractVector=Float64[],
    F0f::AbstractVector=Float64[],
    B0::Real=NaN,
    S0::Real=NaN,
    abstol=1e-6,
    reltol=1e-6,
)
    length(θ) == MEAT_NP ||
        throw(ArgumentError("θ must have length $MEAT_NP; got $(length(θ))"))
    length(φ) == PIT_NP ||
        throw(ArgumentError("φ must have length $PIT_NP; got $(length(φ))"))
    length(u_knots) == length(u_tknots) || throw(
        ArgumentError(
            "u_knots length $(length(u_knots)) != u_tknots length $(length(u_tknots))",
        ),
    )
    p = vcat(θ, φ, u_knots, float(t_wrap), float(t_pull))
    any(isnan, p) && throw(
        ArgumentError(
            "monolith solve got NaN inputs at indices $(findall(isnan, p)) of p " *
            "(θ|φ|u|t_wrap|t_pull)",
        ),
    )
    nu = length(u_knots)
    A = _monolith_solve_arr(
        p, cfg, pit, hum, u_tknots, tsave, nu; abstol, reltol, T0f, W0f, C0f, F0f, B0, S0
    )
    return _monolith_outputs(A, p, cfg, pit, hum, u_tknots, tsave, nu)
end

"""
    monolith_jacobian(f, θ, φ, u_knots, u_tknots, tsave; kw...)

Forward-mode Jacobian of `f(out)` — any function of the
[`monolith_solve`](@ref) output returning a vector, e.g.
`o -> vcat(o.Tpit, o.Tcore)` for the calibration residuals — w.r.t.
`vcat(θ, φ)` (one chunked ForwardDiff pass through the solve). Keyword
arguments are forwarded to `monolith_solve`.

# Returns
- `J::Matrix`: size `length(f(out)) × (MEAT_NP + PIT_NP)`.
"""
function monolith_jacobian(f, θ, φ, u_knots, u_tknots, tsave; kw...)
    x = vcat(float.(θ), float.(φ))
    g = v -> f(
        monolith_solve(v[1:MEAT_NP], v[(MEAT_NP + 1):end], u_knots, u_tknots, tsave; kw...)
    )
    cfgj = ForwardDiff.JacobianConfig(g, x, ForwardDiff.Chunk{length(x)}())
    return ForwardDiff.jacobian(g, x, cfgj)
end

"""
    picard_cook(θ, φ, u_knots, u_tknots, tsave; K, cfg, pit, hum, t_wrap,
                t_pull, abstol, reltol)

The exchange-grid Picard coupling of the composed Tesseract pipeline
(`coupled_cook`), in Julia, for comparison against the monolith: `K`
alternating pit / meat solves with the pit temperature, burn rate, base
vapor pressure and inverse air flow exchanged as knots on `tsave` and the
meat heat load fed back.

# Returns
- `(; pit, meat)`: The last pass's [`pit_solve`](@ref) and
  [`meat_solve`](@ref) outputs.
"""
function picard_cook(
    θ,
    φ,
    u_knots,
    u_tknots,
    tsave;
    K::Int=2,
    cfg::MeatConfig=MeatConfig(),
    pit::PitConfig=PitConfig(),
    hum::Humidity=Humidity(),
    t_wrap::Real=WRAP_NEVER,
    t_pull::Real=PULL_NEVER,
    abstol=1e-6,
    reltol=1e-6,
)
    nt = length(tsave)
    q = zeros(nt)
    u_t0, u_dt = float(u_tknots[1]), float(u_tknots[2] - u_tknots[1])
    u_tk = [lininterp(float(t), u_t0, u_dt, u_knots) for t in tsave]
    local pito, meato
    for _ in 1:K
        pito = pit_solve(
            φ, u_knots, u_tknots, q, tsave, tsave;
            cfg=pit, abstol=min(abstol, 1e-8), reltol=min(reltol, 1e-8),
        )
        mair = hum.afr .* φ[3] .* (φ[5] .+ (1 - φ[5]) .* u_tk)
        pv_base =
            (hum.p_atm / EPS_W) .*
            (hum.omega_amb .+ (hum.y_w .* pito.mdot .+ hum.pan_kg_s) ./ mair)
        meato = meat_solve(
            θ, pito.Tpit, tsave, tsave;
            cfg, abstol, reltol, t_wrap, t_pull,
            smoke_knots=pito.mdot, pv_base_knots=pv_base, imair_knots=1 ./ mair,
        )
        q = meato.q_meat
    end
    return (; pit=pito, meat=meato)
end
