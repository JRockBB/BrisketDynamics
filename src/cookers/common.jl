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
    # Gas mode: a burner is an electric-like source that also makes
    # WATER (CH4 + 2 O2 -> CO2 + 2 H2O: ~4.5e-8 kg H2O per J at LHV).
    # y_gas_w > 0 with p_heater > 0 adds that vapor to the pit humidity;
    # smoke stays exactly zero (nothing chars). 0 = dry element.
    y_gas_w::Float64 = 0.0       # combustion water yield [kg H2O / J]
    # Smoke QUALITY ("thin blue" vs smoldering; the bitterness variable):
    # combustion runs clean when the draft is large relative to the lit
    # bed and smolders when it is starved — cleanliness is a logistic in
    # the draft-to-lit-bed ratio airflow/(a·taper). The same deposited
    # smoke MASS then splits into a clean and a dirty share; the dirty
    # share integrates to the `bitterness` output. Constants are
    # placeholder scale (roadmap: standardized-photo colorimetry + a
    # wrapped/unwrapped pair calibrate the bark family; bitterness needs
    # a tasting panel or a smolder-forced designed cook).
    # Two-zone offset smoker (+1 monolith state: firebox temperature).
    # In two-zone mode (c_duct > 0) the fuel burns in the FIREBOX — its
    # own energy balance with mass C_fb and loss UA_fb — and the cook
    # chamber (the existing pit state) receives heat by draft-driven
    # advection through the duct, q_duct = c_duct·airflow·(T_fb − T_pit).
    # Smoke and combustion water ride the same gas into the chamber, so
    # the humidity/bark pathways are unchanged. c_duct = 0 is the
    # single-zone (bitwise) default: the firebox state rides inert and
    # the burn heats the chamber directly, exactly as before.
    C_fb::Float64 = 2.0e4       # firebox thermal mass [J/K]
    UA_fb::Float64 = 12.0       # firebox loss to ambient [W/K]
    c_duct::Float64 = 0.0       # duct advective conductance at full draft [W/K]
    # Wood splits: a finite dt_add turns the single mid-cook fuel
    # addition into a PULSE TRAIN (one split of m_chimney every dt_add
    # from t_add on — how an offset is actually fed). Inf = one addition
    # (bitwise default). Config for now; differentiable split timing is a
    # planner step.
    dt_add::Float64 = Inf       # interval between fuel additions [s]
    # Water pan (+2 monolith states: pan water mass and temperature).
    # The pan draws pit heat through UA_pan, buffers near boiling (the
    # latent sink behind a WSM's steadiness), FEEDS THE PIT HUMIDITY with
    # its evaporation — replacing the constant Humidity.pan_kg_s when
    # present — and can DRY OUT mid-cook (tanh taper on the remaining
    # water). Evaporation drives against the pre-pan vapor pressure
    # (combustion + ambient); the pan's own self-feedback through the pit
    # vapor is second-order and not closed. m_pan0 = 0 is the no-pan
    # default: both couplings are gated off on the CONFIG (AD-safe), the
    # two states ride along inert.
    m_pan0::Float64 = 0.0       # initial pan water [kg] (0 = no pan)
    T_pan0::Float64 = 293.15    # initial pan water temperature [K]
    UA_pan::Float64 = 25.0      # pit gas -> pan heat conductance [W/K]
    g_pan::Float64 = 1.5e-9     # pan evaporation conductance [kg/(s Pa)] (~0.5 kg/h at boiling: a foiled WSM pan lasts ~7 h)
    C_pan_dry::Float64 = 2000.0 # dry pan + frame heat capacity [J/K]
    m_pan_taper::Float64 = 0.1  # dry-out taper scale [kg]
    # Pellet-auger source: when > 0 the fuel feed is CONTROLLED — the
    # burn rate is m_auger·u·taper(hopper) instead of the Minion bed law
    # (u becomes the auger duty; the fan is implicit in the same duty).
    # Combustion water and smoke stay ON (pellets burn), nothing spreads
    # (pass a0 = 1), and the hopper empties through the same fuel state.
    # 0 = charcoal (bitwise default).
    m_auger::Float64 = 0.0       # full-duty pellet feed rate [kg/s]
    r_smolder::Float64 = 0.25    # draft/lit-bed ratio at the clean-dirty midpoint
    δ_smolder::Float64 = 0.08    # logistic width of the transition
end

"""
    burn_cleanliness(airflow, a, m, pit::PitConfig)

Combustion cleanliness q ∈ (0, 1): a logistic in the draft-to-lit-bed
ratio `airflow / (a·tanh(m/m_taper))`. A choked damper over a large lit
bed smolders (q → 0, acrid smoke); an open draft burns clean (q → 1).
With nothing lit or no fuel the ratio diverges to clean — and the burn
rate is zero there, so no smoke deposits either way.
"""
function burn_cleanliness(airflow, a, m, pit::PitConfig)
    bed = clamp(a, zero(a), one(a)) * tanh(max(m, zero(m)) / pit.m_taper)
    return logistic((airflow / (bed + 1.0e-3) - pit.r_smolder) / pit.δ_smolder)
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
    tc = isfinite(cfg.dt_add) ? spritz_pulse_time(t, t_add, cfg.dt_add) : t_add
    s = logistic((t - tc) / cfg.τ_add)
    return cfg.m_chimney * s * (1 - s) / cfg.τ_add
end

# φ = [C_th, UA, r_max, ΔH_eff, u_leak, k_spread, a0]
const PIT_NP = 7
