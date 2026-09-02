"""
    pibil_pv(T_env, n)

The sealed earth pit (pibil / imu) as a saturated-steam environment:
`pv_base_knots = p_sat(T_env)` makes the existing two-sided surface flux
condense onto the cold meat and the stall never forms. Pure
configuration — pass the returned knots to `meat_solve`. The mechanism is
tested (test/meats); no external validation data yet.
"""
pibil_pv(T_env::Real, n::Integer) = fill(p_sat(float(T_env)), n)
