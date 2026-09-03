"""
    pibil_pv(T_env, n)

The sealed earth pit (pibil / imu) as a saturated-steam environment:
`pv_base_knots = p_sat(T_env)` makes the existing two-sided surface flux
condense onto the cold meat and the stall never forms. Pure
configuration — pass the returned knots to `meat_solve`. The mechanism is
tested (test/meats); no external validation data yet.
"""
pibil_pv(T_env::Real, n::Integer) = fill(p_sat(float(T_env)), n)

"""
    earth_pit(; m_rock=200.0, c_rock=840.0, T_rock0=700.0, UA_seal=8.0)

The sealed earth pit (pibil / imu) as a COMPOSITION of existing pieces —
no new physics: the rock reservoir is the pit's lumped thermal mass,
precharged and discharging (C_th = m_rock·c_rock, Tpit0 = T_rock0), with
NO fuel (m0 = 0, r_max = 0 in φ → nothing burns, no smoke), a tight seal
(u_leak small, u ≡ 0), and the saturated-steam environment from
[`pibil_pv`](@ref) killing the stall. "Control" becomes design: rock
mass and seal time. Returns `(pit, φ_hint)` where φ_hint is the
[C_th, UA, r_max, ΔH_eff, u_leak, k_spread, a0] vector to use —
placeholder scale, no instrumented earth-pit data exists.
"""
function earth_pit(;
    m_rock::Real=200.0, c_rock::Real=840.0, T_rock0::Real=700.0, UA_seal::Real=8.0,
)
    pit = PitConfig(; Tpit0=float(T_rock0), m0=0.0)
    φ_hint = [m_rock * c_rock, float(UA_seal), 0.0, 1.0e7, 0.02, 1.0e-4, 1.0]
    return (pit=pit, φ_hint=φ_hint)
end
