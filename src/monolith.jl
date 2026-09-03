# Monolithic coupled cook: the meat PDE, the pit ODE and the quasi-steady pit
# humidity balance as ONE stiff system. The pit sees the meat's instantaneous
# heat load and the meat sees the instantaneous pit temperature, burn rate
# and vapor pressure — one solve, one derivative. States x = [T(nx) | W(nx)
# | C(nx) | F(nx) | B | S | R | T_pit | m_fuel | ignited]; parameters
# p = [θ(8) | φ(7) | u_knots(nu) | t_wrap | t_pull | t_add | t_spritz |
# dt_spritz | m_spritz].
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
    taper = tanh(max(m, zero(m)) / pit.m_taper)
    # controlled-feed (pellet auger) source replaces the Minion bed law
    ṁ = pit.m_auger > 0 ? pit.m_auger * ud * taper : r_max * airflow * a_c * taper
    # electric source: nothing burns — the fuel, smoke and combustion-water
    # pathways all carry the zeroed burn rate (heat enters via p_heater·ud
    # in the pit energy balance instead)
    if pit.p_heater > 0
        ṁ = zero(ṁ)
    end
    # floored: a sealed source (earth pit, r_max = 0) still exchanges a
    # trace of air — and the humidity algebra must never divide by zero
    mair = max(hum.afr * r_max * airflow, 1.0e-6)
    ṁ_gas_w = pit.y_gas_w * pit.p_heater * ud  # burner combustion water
    pv_base =
        (hum.p_atm / EPS_W) *
        (hum.omega_amb + (hum.y_w * ṁ + hum.pan_kg_s + ṁ_gas_w) / mair)
    return airflow, ṁ, mair, pv_base
end

"""
    _pan_algebra(Tp, mp, Tpan, mair, pv_nopan, pit, hum, cfg)

Water-pan fluxes from the pan states: heat drawn from the pit gas
`q_pan = UA_pan·(Tp − T_pan)` and evaporation
`ṁ_pan = g_pan·(p_sat(T_pan) − pv_nopan)·taper(m_pan)` (two-sided: a
cold pan under a humid pit condenses), both gated off entirely when the
config carries no pan. Returns `(q_pan, ṁ_pan, pv_base)` with the pan
vapor added onto the pre-pan base.
"""
function _pan_algebra(Tp, mp, Tpan, mair, pv_nopan, pit::PitConfig, hum::Humidity)
    if pit.m_pan0 <= 0
        return zero(Tp), zero(Tp), pv_nopan
    end
    taper = tanh(max(mp, zero(mp)) / pit.m_pan_taper)
    ṁ_pan = pit.g_pan * (p_sat(Tpan) - pv_nopan) * taper
    q_pan = pit.UA_pan * (Tp - Tpan)
    return q_pan, ṁ_pan, pv_nopan + (hum.p_atm / EPS_W) * ṁ_pan / mair
end

"""
    make_monolith_rhs(cfg, pit, hum, u_t0, u_dt, nu)

Build the in-place coupled RHS over `x = [meat(5·nx + 4) | T_pit, m_fuel,
ignited, m_pan, T_pan, T_firebox]` with `p = [θ | φ | u_knots(nu) | t_wrap | t_pull]`; the damper
schedule is interpolated from `nu` knots on the uniform grid
`u_t0 + (i-1)·u_dt`.
"""
function make_monolith_rhs(
    cfg::MeatConfig, pit::PitConfig, hum::Humidity, u_t0::Real, u_dt::Real, nu::Int
)
    nm = 5 * cfg.nx + 4  # + the wrap-latch scalar
    n_s, n_a, v0 = shrink_exponents(cfg)
    wb = bone_weights(cfg)
    return function mono_rhs!(dx, x, p, t)
        θ = (p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8])
        C_th, UA, r_max, ΔH_eff, u_leak, k_spread = p[9], p[10], p[11], p[12], p[13], p[14]
        u_knots = @view p[(MONO_NP + 1):(MONO_NP + nu)]
        t_wrap = p[MONO_NP + nu + 1]
        t_pull = p[MONO_NP + nu + 2]
        t_add = p[MONO_NP + nu + 3]
        t_spritz = p[MONO_NP + nu + 4]
        dt_spritz = p[MONO_NP + nu + 5]
        m_spritz = p[MONO_NP + nu + 6]

        Tp, m, a = x[nm + 1], x[nm + 2], x[nm + 3]
        mp, Tpan = x[nm + 4], x[nm + 5]
        Tfb = x[nm + 6]
        ud = lininterp(t, u_t0, u_dt, u_knots)
        ud = clamp(ud, zero(ud), one(ud))
        airflow, ṁ, mair, pv_nopan = _pit_algebra(ud, m, a, r_max, u_leak, pit, hum)
        q_pan, ṁ_pan, pv_base = _pan_algebra(Tp, mp, Tpan, mair, pv_nopan, pit, hum)

        um = @view x[1:nm]
        dum = @view dx[1:nm]
        q_meat = meat_rhs_core!(
            dum, um, t, θ, Tp, ṁ, pv_base, 1 / mair, airflow, t_wrap, t_pull, t_spritz,
            dt_spritz, m_spritz, cfg, n_s, n_a, v0, wb,
        )
        a_c = clamp(a, zero(a), one(a))
        ṁ_add = fuel_add_rate(t, t_add, pit)
        # the lid comes off to dump the chimney and for every spritz while the
        # policy runs (in the pit, unwrapped)
        lid = lid_open(t, t_add, pit.τ_open_add) +
              lid_open(t, spritz_pulse_time(t, t_spritz, dt_spritz), pit.τ_open_spritz) *
              (1 - wrap_blend(t, t_wrap, cfg)) * (1 - pull_blend(t, t_pull, cfg))
        q_lid = pit.UA_open * (Tp - pit.T_amb) * lid
        # two-zone: the burn heats the FIREBOX and the chamber receives
        # draft-advected duct heat; single-zone (c_duct = 0): direct, as ever
        if pit.c_duct > 0
            q_duct = pit.c_duct * airflow * (Tfb - Tp)
            q_burn = q_duct
            dx[nm + 6] =
                (ΔH_eff * ṁ - pit.UA_fb * (Tfb - pit.T_amb) - q_duct) / pit.C_fb
        else
            q_burn = ΔH_eff * ṁ
            dx[nm + 6] = zero(Tp)
        end
        dx[nm + 1] =
            (q_burn + pit.p_heater * ud - UA * (Tp - pit.T_amb) - q_meat - q_lid -
             q_pan) / C_th
        dx[nm + 2] = -ṁ + ṁ_add
        dx[nm + 3] =
            k_spread * a_c * (1 - a_c) * airflow + (1 - a_c) * ṁ_add / max(m, pit.m_taper)
        # pan: water leaves by evaporation (or arrives by condensation);
        # its temperature follows the drawn heat minus the latent load,
        # with the thermal mass shrinking as it dries (metal floor)
        dx[nm + 4] = -ṁ_pan
        dx[nm + 5] =
            pit.m_pan0 > 0 ?
            (q_pan - cfg.λ_vap * ṁ_pan) /
            (CP_WATER * max(mp, zero(mp)) + pit.C_pan_dry) : zero(Tp)
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
    R0=NaN,
)
    u_t0, u_dt = float(u_tknots[1]), float(u_tknots[2] - u_tknots[1])
    rhs! = make_monolith_rhs(cfg, pit, hum, u_t0, u_dt, nu)
    Tp = promote_type(eltype(p), Float64)
    # a0 = φ[7] participates in x0 so forward mode carries its sensitivity
    x0 = vcat(
        Tp.(_meat_u0(cfg, T0f, W0f, C0f; F0f, B0, S0, R0,
                     s0=salt_profile(p[4], cfg))),
        [Tp(pit.Tpit0), Tp(pit.m0), p[MEAT_NP + PIT_NP], Tp(pit.m_pan0),
         Tp(pit.T_pan0), Tp(pit.Tpit0)],
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
    nm = 5nx + 4  # + the wrap-latch scalar
    nt = length(tsave)
    θ = p[1:MEAT_NP]
    r_max, u_leak = p[MEAT_NP + 3], p[MEAT_NP + 5]
    u_t0, u_dt = float(u_tknots[1]), float(u_tknots[2] - u_tknots[1])
    uk = @view p[(MONO_NP + 1):(MONO_NP + nu)]
    t_wrap, t_pull = p[MONO_NP + nu + 1], p[MONO_NP + nu + 2]
    t_spritz, dt_spritz = p[MONO_NP + nu + 4], p[MONO_NP + nu + 5]
    m_spritz = p[MONO_NP + nu + 6]

    Tpit = A[nm + 1, :]
    m_fuel = A[nm + 2, :]
    ignited = A[nm + 3, :]
    m_pan = A[nm + 4, :]
    T_pan = A[nm + 5, :]
    T_firebox = A[nm + 6, :]
    alg = map(1:nt) do i
        ud = lininterp(float(tsave[i]), u_t0, u_dt, uk)
        _pit_algebra(clamp(ud, zero(ud), one(ud)), m_fuel[i], ignited[i], r_max, u_leak, pit, hum)
    end
    mdot = getindex.(alg, 2)
    imair_ts = 1 ./ getindex.(alg, 3)
    pan = map(1:nt) do i
        _pan_algebra(Tpit[i], m_pan[i], T_pan[i], alg[i][3], alg[i][4], pit, hum)
    end
    pan_evap = getindex.(pan, 2)
    pv_base_ts = getindex.(pan, 3)
    g_fac = (getindex.(alg, 1) ./ cfg.g_ref) .^ θ[8]
    h_ts = θ[3] .* g_fac
    km_ts = θ[2] .* g_fac

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
    f_aw_ts = 1 ./ (1 .+ cfg.k_aw .* max.(A[5nx, :], 0.0))
    j_evap = surface_j_blended.(
        Tsurf, Wsurf, h_ts, km_ts, s_ts, pl_ts, pv_base_ts, c_ts, Ref(cfg), f_aw_ts
    )
    pv_pit =
        pv_base_ts .+
        c_ts .* evap_flux_fb.(
            Tsurf, Wsurf, (1 .- cfg.η_evap .* s_ts) .* km_ts, pv_base_ts, c_ts,
            Ref(cfg), f_aw_ts,
        )
    q_meat =
        cfg.A_meat * cfg.n_meat .* a_ts .*
        surface_q_pit_blended.(Tpit, Tsurf, h_ts, s_ts, pl_ts, Ref(cfg))
    bark_b = A[5nx + 1, :]
    bark_s = A[5nx + 2, :]
    ring = A[5nx + 3, :]
    # smoke quality: cleanliness of the burn at each save point, and the
    # DIRTY share of the same deposition integrand that drives bark_s —
    # the two agree by construction on the total deposited mass
    smoke_clean = burn_cleanliness.(getindex.(alg, 1), ignited, m_fuel, Ref(pit))
    dep_dirty =
        cfg.k_dep .* wet_blend.(a_w_oswin.(Wsurf, Ref(cfg)), Ref(cfg)) .*
        max.(mdot, 0.0) .* (1 .- smoke_clean) .* (1 .- s_ts) .* (1 .- pl_ts) .*
        (1 .- bark_s)
    w_retained = fv_mean(A[(nx + 1):2nx, end]) / cfg.W0
    F_mean = fv_mean(A[(3nx + 1):4nx, end])
    R_ret = (1 - cfg.η_drain) * (cfg.F0 - F_mean)
    dts = diff(collect(float.(tsave)))
    ja = j_evap .* a_ts
    evap_w = sum((ja[1:(end - 1)] .+ ja[2:end]) ./ 2 .* dts) / (RHO_DRY * θ[4])
    bitterness = sum((dep_dirty[1:(end - 1)] .+ dep_dirty[2:end]) ./ 2 .* dts)
    jsp = spritz_rate.(tsave, t_spritz, dt_spritz, m_spritz, Ref(cfg)) .* (1 .- s_ts) .* (1 .- pl_ts) .* a_ts
    spritz_w = cfg.f_abs * sum((jsp[1:(end - 1)] .+ jsp[2:end]) ./ 2 .* dts) / (RHO_DRY * θ[4])
    return (;
        Tpit,
        m_fuel,
        ignited,
        mdot,
        Tsurf,
        Tcore=A[1, :],
        bark_b,
        bark_s,
        ring,
        l_star=l_star_bark.(bark_b, bark_s, Ref(cfg)),
        j_evap,
        q_meat,
        C_field_final=A[(2nx + 1):3nx, end],
        w_retained,
        juiciness=w_retained + cfg.β_fat * R_ret / cfg.W0,
        fat_rendered=(cfg.F0 - F_mean) / cfg.F0,
        evap_w,
        spritz_w,
        drip_w=(cfg.W0 - w_retained * cfg.W0) + spritz_w - evap_w,
        drip_f=cfg.η_drain * (cfg.F0 - F_mean),
        l_thick,
        pv_pit,
        pv_base=pv_base_ts,
        T_field=A[1:nx, :],
        W_field=A[(nx + 1):2nx, :],
        C_field=A[(2nx + 1):3nx, :],
        F_field=A[(3nx + 1):4nx, :],
        salt_field=A[(4nx + 1):5nx, :],
        smoke_clean,
        bitterness,
        m_pan,
        T_pan,
        pan_evap,
        T_firebox,
        t=collect(float.(tsave)),
    )
end

"""
    monolith_solve(θ, φ, u_knots, u_tknots, tsave; cfg, pit, hum, t_wrap,
                   t_pull, t_add, t_spritz, dt_spritz, m_spritz, T0f, W0f, C0f, F0f,
                   B0, S0, R0, abstol, reltol)

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
- `t_add::Real`: Mid-cook fuel addition (one chimney) [s]. [Default=never]
- `t_spritz, dt_spritz::Real`: Spritz policy — first spritz and interval [s].
  [Default=never, 3600]
- `m_spritz::Real`: Spritz dose [kg/m²]; NaN ⇒ `cfg.m_spritz`. [Default=NaN]
- `T0f, W0f, C0f, F0f, B0, S0, R0`: Meat field / bark / ring ICs for a state handoff
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
    t_add::Real=ADD_NEVER,
    t_spritz::Real=SPRITZ_NEVER,
    dt_spritz::Real=3600.0,
    m_spritz::Real=NaN,
    T0f::AbstractVector=Float64[],
    W0f::AbstractVector=Float64[],
    C0f::AbstractVector=Float64[],
    F0f::AbstractVector=Float64[],
    B0::Real=NaN,
    S0::Real=NaN,
    R0::Real=NaN,
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
    msp = isnan(m_spritz) ? cfg.m_spritz : float(m_spritz)
    p = vcat(
        θ, φ, u_knots, float(t_wrap), float(t_pull), float(t_add), float(t_spritz),
        float(dt_spritz), msp,
    )
    any(isnan, p) && throw(
        ArgumentError(
            "monolith solve got NaN inputs at indices $(findall(isnan, p)) of p " *
            "(θ|φ|u|t_wrap|t_pull|t_add|t_spritz|dt_spritz|m_spritz)",
        ),
    )
    nu = length(u_knots)
    A = _monolith_solve_arr(
        p, cfg, pit, hum, u_tknots, tsave, nu; abstol, reltol, T0f, W0f, C0f, F0f, B0, S0,
        R0,
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

# --- stacked differentiable outputs (the Tesseract's derivative endpoints) ----

"""
    monolith_stack(o)

The differentiable outputs of [`monolith_solve`](@ref) as one vector, in
the order `Tpit, Tcore, Tsurf, m_fuel, j_evap, q_meat, bark_b, bark_s,
l_star, l_thick` (each nt), `C_field_final` (nx), then the scalars
`w_retained, juiciness, fat_rendered`.
"""
function monolith_stack(o)
    return vcat(
        o.Tpit, o.Tcore, o.Tsurf, o.m_fuel, o.j_evap, o.q_meat, o.bark_b, o.bark_s,
        o.l_star, o.l_thick, o.C_field_final, [o.w_retained, o.juiciness, o.fat_rendered],
    )
end

"""
    _mono_stack_fn(θ, φ, u_knots, u_tknots, tsave, t_wrap, t_pull, t_add,
                   t_spritz, dt_spritz, m_spritz; kw...)

`(g, v0)`: the stacked outputs as a function of the full input vector
`v = [θ; φ; u_knots; t_wrap; t_pull]`, and the vector's value.
"""
function _mono_stack_fn(
    θ, φ, u_knots, u_tknots, tsave, t_wrap, t_pull, t_add, t_spritz, dt_spritz, m_spritz;
    kw...,
)
    nθ, nφ, nu = length(θ), length(φ), length(u_knots)
    v0 = vcat(
        float.(θ), float.(φ), float.(u_knots), float(t_wrap), float(t_pull), float(t_add),
        float(t_spritz), float(dt_spritz), float(m_spritz),
    )
    g(v) = monolith_stack(
        monolith_solve(
            v[1:nθ], v[(nθ + 1):(nθ + nφ)], v[(nθ + nφ + 1):(nθ + nφ + nu)], u_tknots,
            tsave; t_wrap=v[end - 5], t_pull=v[end - 4], t_add=v[end - 3],
            t_spritz=v[end - 2], dt_spritz=v[end - 1], m_spritz=v[end], kw...,
        ),
    )
    return g, v0
end

"""
    monolith_stack_jacobian(θ, φ, u_knots, u_tknots, tsave, t_wrap, t_pull, t_add,
                            t_spritz, dt_spritz, m_spritz, sel; kw...)

Forward-mode Jacobian of [`monolith_stack`](@ref) w.r.t. the entries `sel`
(1-based indices into `[θ; φ; u_knots; t_wrap; t_pull]`): one chunked
ForwardDiff pass per ~12 selected inputs. Keyword arguments go to
[`monolith_solve`](@ref).
"""
function monolith_stack_jacobian(
    θ, φ, u_knots, u_tknots, tsave, t_wrap, t_pull, t_add, t_spritz, dt_spritz, m_spritz,
    sel::AbstractVector{<:Integer}; kw...,
)
    g, v0 = _mono_stack_fn(
        θ, φ, u_knots, u_tknots, tsave, t_wrap, t_pull, t_add, t_spritz, dt_spritz,
        m_spritz; kw...,
    )
    sel = collect(Int, sel)
    function gs(vs)
        v = convert(Vector{eltype(vs)}, copy(v0))
        v[sel] .= vs
        return g(v)
    end
    return ForwardDiff.jacobian(gs, v0[sel])
end

"""
    monolith_stack_jvp(θ, φ, u_knots, u_tknots, tsave, t_wrap, t_pull, t_add,
                       t_spritz, dt_spritz, m_spritz, dv; kw...)

Directional derivative of [`monolith_stack`](@ref) along the full-length
tangent `dv` — one Dual solve.
"""
function monolith_stack_jvp(
    θ, φ, u_knots, u_tknots, tsave, t_wrap, t_pull, t_add, t_spritz, dt_spritz, m_spritz,
    dv; kw...,
)
    g, v0 = _mono_stack_fn(
        θ, φ, u_knots, u_tknots, tsave, t_wrap, t_pull, t_add, t_spritz, dt_spritz,
        m_spritz; kw...,
    )
    dv = collect(Float64, dv)
    return ForwardDiff.derivative(t -> g(v0 .+ t .* dv), 0.0)
end

# --- differentiable duration: s = t / t_f ------------------------------------

"""Fractional event position treated as "never" (t = FRAC_NEVER·t_f ≫ horizon)."""
const FRAC_NEVER = 1.0e3

function _monolith_solve_arr_s(
    p,
    cfg::MeatConfig,
    pit::PitConfig,
    hum::Humidity,
    u_tknots,
    s_save,
    t_f,
    nu::Int;
    abstol=1e-6,
    reltol=1e-6,
    T0f=Float64[],
    W0f=Float64[],
    C0f=Float64[],
    F0f=Float64[],
    B0=NaN,
    S0=NaN,
    R0=NaN,
)
    # u_t0/u_dt in seconds may carry Duals of t_f; the interp ratio
    # (s·t_f − u_t0)/u_dt cancels t_f exactly, value and derivative
    u_t0, u_dt = float(u_tknots[1]), float(u_tknots[2] - u_tknots[1])
    rhs! = make_monolith_rhs(cfg, pit, hum, u_t0, u_dt, nu)
    # integrate over s ∈ [0, 1]: time type stays Float64, the duration
    # enters only through the RHS scaling and the seconds-valued p entries
    rhs_s! = function (dx, x, q, s)
        rhs!(dx, x, q, s * t_f)
        @inbounds for i in eachindex(dx)
            dx[i] *= t_f
        end
        return nothing
    end
    Tp = promote_type(eltype(p), typeof(float(t_f)), Float64)
    x0 = vcat(
        Tp.(_meat_u0(cfg, T0f, W0f, C0f; F0f, B0, S0, R0,
                     s0=salt_profile(p[4], cfg))),
        [Tp(pit.Tpit0), Tp(pit.m0), p[MEAT_NP + PIT_NP], Tp(pit.m_pan0),
         Tp(pit.T_pan0), Tp(pit.Tpit0)],
    )
    s_grid = collect(Float64, s_save)
    prob = ODEProblem{true,SciMLBase.FullSpecialize}(rhs_s!, x0, (0.0, 1.0), p)
    sol = solve(prob, FBDF(); saveat=s_grid, abstol=abstol, reltol=reltol)
    A = Array(sol)
    (SciMLBase.successful_retcode(sol) && size(A, 2) == length(s_grid)) || throw(
        ErrorException(
            "monolith s-solve failed: retcode=$(sol.retcode), " *
            "saved $(size(A, 2))/$(length(s_grid)) points",
        ),
    )
    return A
end

"""
    monolith_solve_tf(θ, φ, u_knots, t_f, s_u, s_save; frac_wrap, frac_pull,
                      frac_add, frac_spritz, frac_dt_spritz, m_spritz, kw...)

[`monolith_solve`](@ref) with the cook **duration `t_f` [s] as a smooth
input** (roadmap: differentiable duration via s = t/t_f). The schedule
shape is fixed in fractional time: damper knots sit on `s_u ⊂ [0, 1]`,
outputs on `s_save ⊂ [0, 1]`, and every event is a fraction of the cook
(`t_wrap = frac_wrap·t_f`, …; `frac ≥ FRAC_NEVER` never happens). The
integration runs over s ∈ [0, 1] with the RHS scaled by `t_f`, so
`d(outputs)/d(t_f)` — including the quality integrals' stretching terms —
is one ForwardDiff direction like any other parameter.

Physical time constants (blend widths, lid-open decay, chimney taper) stay
in seconds: a longer cook does not smear its events.
"""
function monolith_solve_tf(
    θ::AbstractVector,
    φ::AbstractVector,
    u_knots::AbstractVector,
    t_f::Real,
    s_u::AbstractVector,
    s_save::AbstractVector;
    cfg::MeatConfig=MeatConfig(),
    pit::PitConfig=PitConfig(),
    hum::Humidity=Humidity(),
    frac_wrap::Real=FRAC_NEVER,
    frac_pull::Real=FRAC_NEVER,
    frac_add::Real=FRAC_NEVER,
    frac_spritz::Real=FRAC_NEVER,
    frac_dt_spritz::Real=0.5,
    m_spritz::Real=NaN,
    T0f::AbstractVector=Float64[],
    W0f::AbstractVector=Float64[],
    C0f::AbstractVector=Float64[],
    F0f::AbstractVector=Float64[],
    B0::Real=NaN,
    S0::Real=NaN,
    R0::Real=NaN,
    abstol=1e-6,
    reltol=1e-6,
)
    length(θ) == MEAT_NP ||
        throw(ArgumentError("θ must have length $MEAT_NP; got $(length(θ))"))
    length(φ) == PIT_NP ||
        throw(ArgumentError("φ must have length $PIT_NP; got $(length(φ))"))
    length(u_knots) == length(s_u) || throw(
        ArgumentError(
            "u_knots length $(length(u_knots)) != s_u length $(length(s_u))",
        ),
    )
    value_of(x) = x isa ForwardDiff.Dual ? ForwardDiff.value(x) : x
    value_of(t_f) > 0 || throw(ArgumentError("t_f must be positive; got $t_f"))
    msp = isnan(value_of(m_spritz)) ? cfg.m_spritz : m_spritz
    p = vcat(
        float.(θ), float.(φ), float.(u_knots), frac_wrap * t_f, frac_pull * t_f,
        frac_add * t_f, frac_spritz * t_f, frac_dt_spritz * t_f, float(msp),
    )
    any(x -> isnan(value_of(x)), p) && throw(
        ArgumentError("monolith_solve_tf got NaN inputs in p"),
    )
    nu = length(u_knots)
    u_tk = float.(s_u) .* t_f
    tsave = float.(s_save) .* t_f
    A = _monolith_solve_arr_s(
        p, cfg, pit, hum, u_tk, s_save, t_f, nu; abstol, reltol, T0f, W0f, C0f, F0f, B0,
        S0, R0,
    )
    return _monolith_outputs(A, p, cfg, pit, hum, u_tk, tsave, nu)
end

"""
    _mono_stack_fn_tf(θ, φ, u_knots, t_f, s_u, s_save, frac_wrap, frac_pull,
                      frac_add, frac_spritz, frac_dt_spritz, m_spritz; kw...)

`(g, v0)`: the stacked outputs as a function of
`v = [θ; φ; u_knots; t_f; frac_wrap; frac_pull; frac_add; frac_spritz;
frac_dt_spritz; m_spritz]`, and that vector's value.
"""
function _mono_stack_fn_tf(
    θ, φ, u_knots, t_f, s_u, s_save, frac_wrap, frac_pull, frac_add, frac_spritz,
    frac_dt_spritz, m_spritz; kw...,
)
    nθ, nφ, nu = length(θ), length(φ), length(u_knots)
    v0 = vcat(
        float.(θ), float.(φ), float.(u_knots), float(t_f), float(frac_wrap),
        float(frac_pull), float(frac_add), float(frac_spritz), float(frac_dt_spritz),
        float(m_spritz),
    )
    g(v) = monolith_stack(
        monolith_solve_tf(
            v[1:nθ], v[(nθ + 1):(nθ + nφ)], v[(nθ + nφ + 1):(nθ + nφ + nu)],
            v[end - 6], s_u, s_save; frac_wrap=v[end - 5], frac_pull=v[end - 4],
            frac_add=v[end - 3], frac_spritz=v[end - 2], frac_dt_spritz=v[end - 1],
            m_spritz=v[end], kw...,
        ),
    )
    return g, v0
end

"""
    monolith_stack_jacobian_tf(θ, φ, u_knots, t_f, s_u, s_save, frac_wrap,
                               frac_pull, frac_add, frac_spritz, frac_dt_spritz,
                               m_spritz, sel; kw...)

Forward-mode Jacobian of [`monolith_stack`](@ref) in duration mode w.r.t.
the entries `sel` (1-based indices into `[θ; φ; u_knots; t_f; frac_wrap;
frac_pull; frac_add; frac_spritz; frac_dt_spritz; m_spritz]`).
"""
function monolith_stack_jacobian_tf(
    θ, φ, u_knots, t_f, s_u, s_save, frac_wrap, frac_pull, frac_add, frac_spritz,
    frac_dt_spritz, m_spritz, sel::AbstractVector{<:Integer}; kw...,
)
    g, v0 = _mono_stack_fn_tf(
        θ, φ, u_knots, t_f, s_u, s_save, frac_wrap, frac_pull, frac_add, frac_spritz,
        frac_dt_spritz, m_spritz; kw...,
    )
    sel = collect(Int, sel)
    function gs(vs)
        v = convert(Vector{eltype(vs)}, copy(v0))
        v[sel] .= vs
        return g(v)
    end
    return ForwardDiff.jacobian(gs, v0[sel])
end

"""
    monolith_stack_jvp_tf(θ, φ, u_knots, t_f, s_u, s_save, frac_wrap, frac_pull,
                          frac_add, frac_spritz, frac_dt_spritz, m_spritz, dv; kw...)

Directional derivative of [`monolith_stack`](@ref) in duration mode along
the full-length tangent `dv` — one Dual solve.
"""
function monolith_stack_jvp_tf(
    θ, φ, u_knots, t_f, s_u, s_save, frac_wrap, frac_pull, frac_add, frac_spritz,
    frac_dt_spritz, m_spritz, dv; kw...,
)
    g, v0 = _mono_stack_fn_tf(
        θ, φ, u_knots, t_f, s_u, s_save, frac_wrap, frac_pull, frac_add, frac_spritz,
        frac_dt_spritz, m_spritz; kw...,
    )
    dv = collect(Float64, dv)
    return ForwardDiff.derivative(t -> g(v0 .+ t .* dv), 0.0)
end

"""
    picard_cook(θ, φ, u_knots, u_tknots, tsave; K, cfg, pit, hum, t_wrap,
                t_pull, abstol, reltol)

The exchange-grid Picard coupling of the component solvers, kept as the
oracle for the monolith's coupling: `K`
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
    t_add::Real=ADD_NEVER,
    t_spritz::Real=SPRITZ_NEVER,
    dt_spritz::Real=3600.0,
    m_spritz::Real=NaN,
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
            cfg=pit, abstol=min(abstol, 1e-8), reltol=min(reltol, 1e-8), t_add,
            t_spritz, dt_spritz, t_spritz_stop=min(float(t_wrap), float(t_pull)),
            δ_spritz_stop=(t_wrap <= t_pull ? cfg.δ_wrap : cfg.δ_pull),
        )
        airflow = φ[5] .+ (1 - φ[5]) .* u_tk
        mair = hum.afr .* φ[3] .* airflow
        pv_base =
            (hum.p_atm / EPS_W) .*
            (hum.omega_amb .+ (hum.y_w .* pito.mdot .+ hum.pan_kg_s) ./ mair)
        meato = meat_solve(
            θ, pito.Tpit, tsave, tsave;
            cfg, abstol, reltol, t_wrap, t_pull,
            smoke_knots=pito.mdot, pv_base_knots=pv_base, imair_knots=1 ./ mair,
            airflow_knots=airflow, t_spritz, dt_spritz, m_spritz,
        )
        q = meato.q_meat
    end
    return (; pit=pito, meat=meato)
end
