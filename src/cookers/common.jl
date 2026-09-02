# Shared cooker pieces: the vessel configuration (PitConfig — thermal mass,
# lid, fuel-addition and electric-element constants), the lid-open window,
# the chimney pulse, and the φ parameter contract. The charcoal burn ODE
# lives in smoker.jl; the monolith couples through these same pieces.

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
    # Electric source mode: when > 0 the heat input is p_heater·u (the
    # damper input becomes the element duty) and NOTHING burns — no fuel
    # loss, no combustion water, no smoke, so bark deposition and the
    # smoke ring are inert, which is what an electric cooker does. Pass
    # a0 = 1 (nothing to ignite). 0 = charcoal (bitwise default).
    p_heater::Float64 = 0.0      # max heater power [W]
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
