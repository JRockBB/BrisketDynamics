# --- a cook in segments: the state handed from one solve to the next ------------------------------
#
# The monolith solves one cook with one parameter vector. A plant whose parameters move during the cook
# (a new charge of fuel at the add with its own burn rate and enthalpy, a leak that drifts as the vents
# are nudged, surface coefficients that change as the bark sets) and an estimator that carries the state
# forward between refits both need the same thing: a solve that starts where the last one ended. The
# monolith takes the meat fields, the bark and ring states and the salt field as initial conditions
# (`T0f`, `W0f`, `C0f`, `F0f`, `B0`, `S0`, `R0`, `Ssalt0f`) and the pit's state through its config
# and the pit's state as solver keywords (`Tpit0`, `mfuel0`, `mpan0`, `Tpan0`, `Tfb0`), the ignited fraction
# through `a0` (φ[7], the only place a0 acts); the state-triggered wrap latch is not carried (the wrap here is
# a time). Every handed-off quantity may be a ForwardDiff Dual, so a segmented cook differentiates. Event times stay absolute, so a
# segment after the wrap is wrapped from its first step and an add before it is already in `m0`; a
# boundary must not fall inside the add's pulse.

"The state of an output `o` at save index `k`: the initial-condition keywords a solve from there takes (the meat fields, bark, ring and salt; the pit's temperature, fuel, pan and firebox as solver keywords, so a Dual state flows through) and the ignited fraction."
function handoff(o, k::Int)
    ics = (T0f=o.T_field[:, k], W0f=o.W_field[:, k], C0f=o.C_field[:, k], F0f=o.F_field[:, k], Ssalt0f=o.salt_field[:, k],
           B0=o.bark_b[k], S0=o.bark_s[k], R0=o.ring[k],
           Tpit0=o.Tpit[k], mfuel0=o.m_fuel[k], mpan0=o.m_pan[k], Tpan0=o.T_pan[k], Tfb0=o.T_firebox[k])
    return ics, clamp(o.ignited[k], 1e-6, 1 - 1e-6)
end

"""
    solve_segments(θs, φs, u, t, edges; pit, cfg, A_meat, events...) -> o

The cook on the grid `t` solved in segments split at the grid indices `edges` (the last index of each
segment but the final one), segment `i` at parameters `(θs[i], φs[i])`, each started from the state the
previous one ended in; `φs[i][7]` (a0) is replaced by the handed-off ignited fraction for `i > 1`. The
outputs are concatenated on `t` (time series and fields); the terminal integrals (`w_retained`,
`evap_w`, ...) are the LAST segment's only and must not be read from a segmented cook.
"""
function solve_segments(θs, φs, u, t, edges; pit::PitConfig, cfg::MeatConfig, A_meat, kw...)
    n = length(t)
    bounds = vcat(1, collect(Int, edges), n)
    outs = NamedTuple[]
    ics = (;)
    for i in 1:(length(bounds) - 1)
        ia, ib = bounds[i], bounds[i + 1]
        tt = t[ia:ib]; uu = u[ia:ib]
        φ = φs[i]
        if i > 1
            ics, a0 = handoff(outs[end], size(outs[end].T_field, 2))
            φ = vcat(φ[1:6], a0)                     # a0 is the handed-off ignited fraction (keeps a Dual φ)
        end
        push!(outs, monolith_solve(θs[i], φ, uu, tt, tt; pit, cfg, A_meat, ics..., kw...))
    end
    length(outs) == 1 && return outs[1]
    # concatenate: the first segment whole, later ones without their first point (the boundary is shared)
    names = keys(outs[1])
    vals = map(names) do nm
        parts = [i == 1 ? getfield(o, nm) : _drop_first(getfield(o, nm)) for (i, o) in enumerate(outs)]
        v1 = getfield(outs[1], nm)
        if v1 isa AbstractMatrix && size(v1, 2) == length(outs[1].t)
            reduce(hcat, parts)
        elseif v1 isa AbstractVector && length(v1) == length(outs[1].t)
            reduce(vcat, parts)
        else
            getfield(outs[end], nm)
        end
    end
    return NamedTuple{names}(vals)
end
_drop_first(v::AbstractMatrix) = v[:, 2:end]
_drop_first(v::AbstractVector) = v[2:end]
_drop_first(v) = v
