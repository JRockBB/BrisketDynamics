# Lumped-parameter smoker ("pit") model: 3 states — pit gas temperature,
# remaining fuel mass, and ignited bed fraction. Fuel burn rate responds to
# the damper schedule u(t) (with a leak floor, so closing the damper never
# fully kills the fire), scales with the ignited fraction (Minion-method
# fires spread across the unlit bed over hours), and tapers smoothly to zero
# as the fuel runs out.
#
# The ignited fraction a ∈ (0,1] follows airflow-assisted logistic spread,
#   da/dt = k_spread · a(1−a) · (u_leak + (1−u_leak)u),
# with a(0) = a0. At a0 = 1 the spread term vanishes and the model reduces
# EXACTLY to the fully-lit two-state model — synthetic results predating this
# state remain valid by construction.
#
# Differentiation: forward mode through the coupled monolith (src/monolith.jl).
# a0 lives in φ (not the config) so calibration can sample it with gradients.

Base.@kwdef struct PitConfig
    T_amb::Float64 = 293.15    # ambient temperature [K]
    Tpit0::Float64 = 293.15    # initial pit temperature [K] (cold start)
    m0::Float64 = 7.0          # initial charcoal load [kg] (full WSM ring)
    m_taper::Float64 = 0.5     # fuel mass scale for the smooth burn-out taper [kg]
    m_chimney::Float64 = 1.4   # fuel added per mid-cook addition (one lit chimney) [kg]
    τ_add::Float64 = 300.0     # duration of dumping a chimney [s]
    # lid-open loss: what a lid lift costs the lumped thermal mass (convective
    # exchange + open-top radiation from the bed, ~0.3 MJ per 30 s at a 90 K
    # pit–ambient difference) — not the dome-gas transient a probe shows
    UA_open::Float64 = 110.0     # lid-open loss coefficient [W/K]
    τ_open_add::Float64 = 120.0  # lid open to dump a chimney [s]
    τ_open_spritz::Float64 = 30.0  # lid open per spritz [s]
end

"""
    lid_open(t, t0, τ)

Smooth lid-open window of unit height and duration `τ` centred at `t0`
(`4σ′(4(t − t0)/τ)`: integrates to `τ`, exactly 0 far away).
"""
function lid_open(t, t0, τ)
    s = logistic(4 * (t - t0) / τ)
    return 4 * s * (1 - s)
end

# Default fuel-addition time: beyond any horizon, the pulse is exactly 0.
const ADD_NEVER = 1.0e9

"""
    fuel_add_rate(t, t_add, cfg::PitConfig)

Mass rate of a mid-cook fuel addition [kg/s]: one chimney `cfg.m_chimney`
dumped over `cfg.τ_add` as a smooth unit-area pulse (a logistic derivative)
centred at `t_add`; `t_add = ADD_NEVER` is exactly zero everywhere.
"""
function fuel_add_rate(t, t_add, cfg::PitConfig)
    s = logistic((t - t_add) / cfg.τ_add)
    return cfg.m_chimney * s * (1 - s) / cfg.τ_add
end

# φ = [C_th, UA, r_max, ΔH_eff, u_leak, k_spread, a0]
const PIT_NP = 7

"""
    make_pit_rhs(cfg::PitConfig, u_t0, u_dt, u_nk::Int, q_t0, q_dt, q_nk::Int)

Build the in-place pit RHS closure. The parameter vector is
`p = vcat(φ, u_knots, qmeat_knots)`: the damper schedule (`u_nk` knots) and
the meat heat-load trajectory (`q_nk` knots) both live in `p`, exposing VJPs
w.r.t. the control and the coupling input.

# Arguments
- `cfg::PitConfig`: Fixed pit configuration (ambient, initial load, taper).
- `u_t0::Number`: First damper knot time [s].
- `u_dt::Number`: Damper knot spacing [s].
- `u_nk::Int`: Number of damper knots.
- `q_t0::Number`: First heat-load knot time [s].
- `q_dt::Number`: Heat-load knot spacing [s].
- `q_nk::Int`: Number of heat-load knots.

# Returns
- `pit_rhs!::Function`: In-place RHS `pit_rhs!(dx, x, p, t)` over states
  `x = [T_pit K, m_fuel kg, ignited fraction]`.
"""
function make_pit_rhs(cfg::PitConfig, u_t0, u_dt, u_nk::Int, q_t0, q_dt, q_nk::Int)
    return function pit_rhs!(dx, x, p, t)
        C_th, UA, r_max, ΔH_eff, u_leak = p[1], p[2], p[3], p[4], p[5]
        k_spread = p[6]
        u_knots = @view p[(PIT_NP + 1):(PIT_NP + u_nk)]
        q_knots = @view p[(PIT_NP + u_nk + 1):(PIT_NP + u_nk + q_nk)]
        t_add = p[PIT_NP + u_nk + q_nk + 1]
        t_spritz, dt_spritz = p[PIT_NP + u_nk + q_nk + 2], p[PIT_NP + u_nk + q_nk + 3]
        t_spritz_stop = p[PIT_NP + u_nk + q_nk + 4]

        u = clamp(lininterp(t, u_t0, u_dt, u_knots), 0, 1)
        q_meat = lininterp(t, q_t0, q_dt, q_knots)
        airflow = u_leak + (1 - u_leak) * u

        Tp, m, a = x[1], x[2], x[3]
        a = clamp(a, zero(a), one(a))
        ṁ = r_max * airflow * a * tanh(max(m, zero(m)) / cfg.m_taper)
        # a lit chimney adds fuel AND ignited bed: the ignited fraction of
        # the (larger) bed moves toward 1 at the rate the mass arrives
        ṁ_add = fuel_add_rate(t, t_add, cfg)
        # the lid comes off to dump the chimney and for every spritz (until the
        # spritz policy stops); the oracle gates the spritz lifts by a smooth
        # stop time where the monolith uses the wrap and pull blends
        lid = lid_open(t, t_add, cfg.τ_open_add) +
              lid_open(t, spritz_pulse_time(t, t_spritz, dt_spritz), cfg.τ_open_spritz) *
              logistic((t_spritz_stop - t) / 30.0)
        q_lid = cfg.UA_open * (Tp - cfg.T_amb) * lid
        dx[1] = (ΔH_eff * ṁ - UA * (Tp - cfg.T_amb) - q_meat - q_lid) / C_th
        dx[2] = -ṁ + ṁ_add
        dx[3] = k_spread * a * (1 - a) * airflow + (1 - a) * ṁ_add / max(m, cfg.m_taper)
        return nothing
    end
end

"""
    _pit_solve_arr(p, cfg::PitConfig, u_tknots, q_tknots, tsave; abstol, reltol)

Solve the pit ODE and return the raw state array. Fails loudly on an
unsuccessful retcode or a truncated save — a silently short array would
propagate wrong-shaped outputs through the composed pipeline.

# Arguments
- `p::AbstractVector`: Full parameter vector `vcat(φ, u_knots, qmeat_knots)`
  (may be ForwardDiff-dual-valued).
- `cfg::PitConfig`: Pit configuration.
- `u_tknots::AbstractVector`: Damper knot times [s] (uniform grid).
- `q_tknots::AbstractVector`: Heat-load knot times [s] (uniform grid).
- `tsave::AbstractVector`: Output save times [s].

# Keyword Arguments
- `abstol::Float64`: ODE absolute tolerance. [Default=1e-8]
- `reltol::Float64`: ODE relative tolerance. [Default=1e-8]

# Returns
- `A::Matrix`: States at `tsave`, size 3 × nt (rows: T_pit, m_fuel, ignited).
"""
function _pit_solve_arr(
    p, cfg::PitConfig, u_tknots, q_tknots, tsave; abstol=1e-8, reltol=1e-8
)
    u_t0, u_dt = float(u_tknots[1]), float(u_tknots[2] - u_tknots[1])
    q_t0, q_dt = float(q_tknots[1]), float(q_tknots[2] - q_tknots[1])
    rhs! = make_pit_rhs(cfg, u_t0, u_dt, length(u_tknots), q_t0, q_dt, length(q_tknots))
    # a0 = p[7] participates in x0 so ForwardDiff carries its sensitivity
    x0 = [convert(eltype(p), cfg.Tpit0), convert(eltype(p), cfg.m0), p[7]]
    any(isnan, p) && throw(
        ArgumentError(
            "pit solve got NaN inputs at indices $(findall(isnan, p)) of p (φ|u|q)"
        ),
    )
    prob = ODEProblem(rhs!, x0, (float(tsave[1]), float(tsave[end])), p)
    sol = solve(prob, Tsit5(); saveat=tsave, abstol=abstol, reltol=reltol)
    A = Array(sol)                            # 3 × nt
    # A failed solve must fail LOUDLY: a truncated solution array silently
    # propagates wrong-shaped outputs through the composed pipeline.
    (SciMLBase.successful_retcode(sol) && size(A, 2) == length(tsave)) || throw(
        ErrorException(
            "pit solve failed: retcode=$(sol.retcode), " *
            "saved $(size(A, 2))/$(length(tsave)) points",
        ),
    )
    return A
end

"""
    _pit_mdot_row(A, p, cfg::PitConfig, u_tknots, tsave, nu::Int)

Exact burn rate at the save times, recomputed algebraically from the saved
states and the damper interpolant — the same expression as the RHS, so it is
exact (no numerical differencing of `m_fuel`) and ForwardDiff carries its
sensitivity when called with dual-valued `p`.

# Arguments
- `A::AbstractMatrix`: Saved state array (3 × nt) from [`_pit_solve_arr`](@ref).
- `p::AbstractVector`: Full parameter vector (φ | u_knots | qmeat_knots).
- `cfg::PitConfig`: Pit configuration.
- `u_tknots::AbstractVector`: Damper knot times [s].
- `tsave::AbstractVector`: Save times [s].
- `nu::Int`: Number of damper knots.

# Returns
- `mdot::Vector`: Instantaneous fuel burn rate at each save time [kg/s].
"""
function _pit_mdot_row(A, p, cfg::PitConfig, u_tknots, tsave, nu::Int)
    u_t0, u_dt = float(u_tknots[1]), float(u_tknots[2] - u_tknots[1])
    r_max, u_leak = p[3], p[5]
    uk = @view p[(PIT_NP + 1):(PIT_NP + nu)]
    return [
        begin
            u = clamp(lininterp(float(tsave[i]), u_t0, u_dt, uk), 0, 1)
            airflow = u_leak + (1 - u_leak) * u
            m = A[2, i]
            a = clamp(A[3, i], zero(A[3, i]), one(A[3, i]))
            r_max * airflow * a * tanh(max(m, zero(m)) / cfg.m_taper)
        end for i in eachindex(tsave)
    ]
end

"""
    pit_solve(φ, u_knots, u_tknots, qmeat_knots, q_tknots, tsave; cfg, abstol, reltol)

Forward pit solve over the damper schedule and meat heat-load forcing.

# Arguments
- `φ::AbstractVector`: Pit latents `[C_th, UA, r_max, ΔH_eff, u_leak, k_spread, a0]`.
- `u_knots::AbstractVector`: Damper schedule values ∈ [0, 1] on `u_tknots`.
- `u_tknots::AbstractVector`: Damper knot times [s] (uniform grid).
- `qmeat_knots::AbstractVector`: Meat heat-load trajectory [W] on `q_tknots`.
- `q_tknots::AbstractVector`: Heat-load knot times [s] (uniform grid).
- `tsave::AbstractVector`: Output save times [s].

# Keyword Arguments
- `cfg::PitConfig`: Pit configuration. [Default=PitConfig()]
- `abstol::Float64`: ODE absolute tolerance. [Default=1e-8]
- `reltol::Float64`: ODE relative tolerance. [Default=1e-8]
- `t_add::Real`: Mid-cook fuel addition time [s] (one chimney, see
  [`fuel_add_rate`](@ref)); `ADD_NEVER` = none. [Default=ADD_NEVER]
- `t_spritz, dt_spritz, t_spritz_stop::Real`: Spritz policy (for the lid
  lifts, [`lid_open`](@ref)); the lifts stop at `t_spritz_stop`.
  [Default=SPRITZ_NEVER, 3600, Inf]

# Returns
- `out::NamedTuple`: `(; Tpit [K], m_fuel [kg], ignited, mdot [kg/s], t [s])`
  trajectories at `tsave` (`mdot` = exact instantaneous burn rate, the
  smoke-flux proxy for the bark model).
"""
function _pit_check_lengths(u_knots, u_tknots, qmeat_knots, q_tknots)
    length(u_knots) == length(u_tknots) || throw(
        ArgumentError(
            "u_knots length $(length(u_knots)) != u_tknots length $(length(u_tknots))",
        ),
    )
    length(qmeat_knots) == length(q_tknots) || throw(
        ArgumentError(
            "qmeat_knots length $(length(qmeat_knots)) != q_tknots length $(length(q_tknots))",
        ),
    )
    return nothing
end

function pit_solve(
    φ::AbstractVector,
    u_knots::AbstractVector,
    u_tknots::AbstractVector,
    qmeat_knots::AbstractVector,
    q_tknots::AbstractVector,
    tsave::AbstractVector;
    cfg::PitConfig=PitConfig(),
    abstol=1e-8,
    reltol=1e-8,
    t_add::Real=ADD_NEVER,
    t_spritz::Real=SPRITZ_NEVER,
    dt_spritz::Real=3600.0,
    t_spritz_stop::Real=Inf,
)
    _pit_check_lengths(u_knots, u_tknots, qmeat_knots, q_tknots)
    p = vcat(
        φ, u_knots, qmeat_knots, float(t_add), float(t_spritz), float(dt_spritz),
        float(t_spritz_stop),
    )
    A = _pit_solve_arr(p, cfg, u_tknots, q_tknots, tsave; abstol, reltol)
    mdot = _pit_mdot_row(A, p, cfg, u_tknots, tsave, length(u_knots))
    return (
        Tpit=A[1, :], m_fuel=A[2, :], ignited=A[3, :], mdot=mdot, t=collect(float.(tsave))
    )
end
