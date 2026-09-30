"""
    pellet(; m_auger=2.0e-4, kw...)

Pellet-cooker preset: a controlled-feed source (`PitConfig.m_auger`) —
the damper input becomes the auger duty, the burn rate is
`m_auger·u·tanh(m/m_taper)` (the hopper empties through the same fuel
state), combustion water and smoke stay on, and nothing spreads (pass
a0 = 1 in φ). Full-duty feed default ~0.7 kg/h of pellets. Constants are
placeholder scale.
"""
pellet(; m_auger::Real=2.0e-4, kw...) = PitConfig(; m_auger=float(m_auger), kw...)
