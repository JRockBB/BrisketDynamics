"""
    offset_smoker(; C_fb=2.0e4, UA_fb=12.0, c_duct=60.0, dt_add=3600.0, kw...)

Stick-burner preset: the two-zone source (firebox state + draft-driven
duct advection into the chamber) fed by hourly wood splits (`dt_add`
pulse train on the fuel-addition machinery — pass a `t_add` for the
first split). Constants are placeholder scale; a firebox/chamber probe
pair is the designed experiment that calibrates C_fb/UA_fb/c_duct.
"""
offset_smoker(;
    C_fb::Real=2.0e4, UA_fb::Real=12.0, c_duct::Real=60.0, dt_add::Real=3600.0,
    kw...,
) = PitConfig(;
    C_fb=float(C_fb), UA_fb=float(UA_fb), c_duct=float(c_duct),
    dt_add=float(dt_add), kw...,
)
