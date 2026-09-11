# --- a cook in segments: the state handed from one solve to the next ------------------------------
#
# The monolith solves one cook with one parameter vector. A plant whose parameters move during the cook
# (a new charge of fuel at the add with its own burn rate and enthalpy, a leak that drifts as the vents
# are nudged, surface coefficients that change as the bark sets) and an estimator that carries the state
# forward between refits both need the same thing: a solve that starts where the last one ended. The
# monolith takes the meat fields, the bark and ring states and the salt field as initial conditions
# (`T0f`, `W0f`, `C0f`, `F0f`, `B0`, `S0`, `R0`, `Ssalt0f`) and the pit's state through its config
# (`Tpit0`, `m0`, `m_pan0`, `T_pan0`) and the ignited fraction `a0` (φ[7], the only place a0 acts). The
# firebox temperature restarts at the pit's (exact for a single-zone pit, `c_duct = 0`), and the
# state-triggered wrap latch is not carried (the wrap here is a time). Event times stay absolute, so a
# segment after the wrap is wrapped from its first step and an add before it is already in `m0`; a
# boundary must not fall inside the add's pulse.

"The state of an output `o` at save index `k`: the initial-condition keywords, the pit config and the ignited fraction a solve from there takes."
function handoff(o, k::Int, pit::PitConfig)
    ics = (T0f=o.T_field[:, k], W0f=o.W_field[:, k], C0f=o.C_field[:, k], F0f=o.F_field[:, k], Ssalt0f=o.salt_field[:, k],
           B0=o.bark_b[k], S0=o.bark_s[k], R0=o.ring[k])
    fields = Dict(f => getfield(pit, f) for f in fieldnames(PitConfig))
    fields[:Tpit0] = o.Tpit[k]; fields[:m0] = o.m_fuel[k]; fields[:m_pan0] = o.m_pan[k]; fields[:T_pan0] = o.T_pan[k]
    pit_k = PitConfig(; (f => fields[f] for f in fieldnames(PitConfig))...)
    return ics, pit_k, clamp(o.ignited[k], 1e-6, 1 - 1e-6)
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
    ics = (;); pit_i = pit
    for i in 1:(length(bounds) - 1)
        ia, ib = bounds[i], bounds[i + 1]
        tt = t[ia:ib]; uu = u[ia:ib]
        φ = copy(φs[i])
        if i > 1
            ics, pit_i, a0 = handoff(outs[end], size(outs[end].T_field, 2), pit)
            φ[7] = a0
        end
        push!(outs, monolith_solve(θs[i], φ, uu, tt, tt; pit=pit_i, cfg, A_meat, ics..., kw...))
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
