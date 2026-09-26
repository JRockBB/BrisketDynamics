# The belief's coordinates over a CookSystem's state (src/system.jl): the constrained coordinates of the
# sequential filter (softplus for a positive entry, the logit for a unit-interval one) generalized from the
# monolith's fixed layout to a system of any pieces, with the field groups the process noise acts on and the
# hand-off contract per piece. For a system of one slab the belief layout is the filter's `FilterLayout`
# entry for entry (test_online.jl gates it), so the state-noise groups and their rates carry over by name.
#
# The belief carries every piece's fields, bark, smoke and ring, and the pit's chamber temperature, fuel,
# pan water and temperature, firebox and ignited fraction; not the wrap latch (the wrap is a time to the
# estimator, as in `handoff`), which is zero on every restart. Included by BrisketEstimator after sequential.jl.

using BrisketDynamics: MEAT_NP, PIT_NP, PIECE_PARAM_SCALARS, _primal, fv_mean, l_star_bark, piece

export BeliefLayout, belief_layout, belief_from_system, system_from_belief, belief_encode, belief_decode, belief_decode_jacobian,
       belief_groups, belief_process_cov, belief_init_cov, belief_cov_to_coord, belief_physical_moments, belief_handoff

"""
    BeliefLayout

The belief's state over a `SystemLayout`: `n` entries, each a system-state index (`sys_index`) with a
coordinate kind; per piece the range of its entries and the pit's; the noise groups (`"flat.T"`, ...,
`"T_pit"`, ...) as name => range.
"""
struct BeliefLayout
    sys::SystemLayout
    n::Int
    kinds::Vector{Symbol}
    sys_index::Vector{Int}
    piece_range::Dict{Symbol,UnitRange{Int}}
    pit_range::UnitRange{Int}
    groups::Vector{Pair{String,UnitRange{Int}}}
end
const PIECE_BELIEF_FIELDS = (:T, :W, :C, :F, :S, :bark_b, :bark_s, :ring)
const PIT_BELIEF_FIELDS = (:T_pit, :m_fuel, :m_pan, :T_pan, :T_fb, :ignited)
const PIT_GROUP_NAMES = ("T_pit", "m_fuel", "m_pan", "T_pan", "T_fb", "ignited")

"""
    belief_layout(sys::CookSystem) -> BeliefLayout

The belief's layout over the system: per piece its fields in `PIECE_BELIEF_FIELDS` order (the latch left
out), then the pit's in `PIT_BELIEF_FIELDS` order; one noise group per field of every piece (the bark, smoke
and ring scalars one group, `"id.bark"`) and one per pit entry.
"""
function belief_layout(sys::CookSystem)
    lay = sys.layout
    kinds = Symbol[]; idx = Int[]; pr = Dict{Symbol,UnitRange{Int}}(); groups = Pair{String,UnitRange{Int}}[]
    for id in lay.pieces
        b = lay.piece[id]; start = length(idx) + 1
        bark0 = 0
        for (f, r0, kind) in zip(b.names, b.ranges, b.kinds)                # every field of the piece's own layout but the latch
            f === :latch && continue
            r = lay.piece_range[id][r0]
            g0 = length(idx) + 1
            append!(idx, collect(r)); append!(kinds, fill(kind, length(r)))
            if f in (:bark_b, :bark_s, :ring)                                  # the bark, smoke and ring one group
                bark0 == 0 && (bark0 = g0)
            else
                push!(groups, "$id.$f" => g0:length(idx))
            end
        end
        bark0 > 0 && push!(groups, "$id.bark" => bark0:length(idx))
        pr[id] = start:length(idx)
    end
    p0 = length(idx) + 1
    for (f, g) in zip(PIT_BELIEF_FIELDS, PIT_GROUP_NAMES)
        r = lay.pit_range[lay.pit[f]]
        append!(idx, collect(r)); append!(kinds, fill(lay.pit.kinds[findfirst(==(f), lay.pit.names)], length(r)))
        push!(groups, g => length(idx):length(idx))
    end
    return BeliefLayout(lay, length(idx), kinds, idx, pr, p0:length(idx), groups)
end
belief_groups(bl::BeliefLayout) = bl.groups
"The belief's range of `field` of piece `id` (or of the pit)."
function belief_index(bl::BeliefLayout, id::Symbol, field::Symbol)
    r = state_index(bl.sys, id, field)
    out = [findfirst(==(i), bl.sys_index) for i in r]
    any(isnothing, out) && throw(ArgumentError("$field of $id is not carried by the belief"))
    return first(out):last(out)
end

"The belief's physical state from the system's."
belief_from_system(x_sys::AbstractVector, bl::BeliefLayout) = x_sys[bl.sys_index]
"The system's state from the belief's physical state: the latch of every piece at `latch` (zero: the wrap is a time)."
function system_from_belief(x_b::AbstractVector, bl::BeliefLayout; latch::Real=0.0)
    length(x_b) == bl.n || throw(ArgumentError("the belief state has $(length(x_b)) entries; the layout $(bl.n)"))
    x = zeros(promote_type(eltype(x_b), typeof(float(latch))), bl.sys.n)
    x[bl.sys_index] = x_b
    for id in bl.sys.pieces
        haskey(bl.sys.piece[id], :latch) && (x[state_index(bl.sys, id, :latch)] .= latch)
    end
    return x
end

# the constrained coordinates: the filter's decode (sequential.jl) by kind, and an encode of bounded slope. The filter's
# encode takes the log of the excess over the floor down to a round-off (a salt field of an unsalted cut sits at the
# floor and its coordinate swings by tens of units on 1e-24 noise); a filter never evaluates a dynamics residual, a
# smoother does, and whitened by the field's shape precision such swings are thousands of nats of roughness. The
# belief's encode floors the excess at `BELIEF_ENC_MIN`: `_soft_floor` is δ itself up to d = δ (so a state at its floor
# decodes to floor + δ and encodes back to the same coordinate: a fixed point, no drift slice after slice), rises from
# there with a continuous slope (no kink for the collagen nodes crossing 1e-6 one after another in the first twenty
# minutes of a cook), and is d itself beyond a few δ (the correction is under 1e-20 past 50 δ), so every state farther
# than 5e-5 from its floor keeps the filter's coordinate exactly.
const BELIEF_ENC_MIN = 1e-6
# a unit-interval field's floor is a thousandth: its coordinate is the logit, whose slope is 1/x, and the transition from
# the floor's zero slope to it is a band as wide as the floor; the collagen field's interior nodes sit inside a 1e-6 band
# for a quarter hour early in a cook, where the Gauss-Newton model of the dynamics residual changed by a million between
# iterations (the two-piece bench of 2026-09-25 capped its iterations there); a conversion under a thousandth is nothing to
# a band that starts at 0.9, and the ignited fraction lives far from either end
const BELIEF_ENC_MIN_UNIT = 1e-3
function _soft_floor(d, δ)
    d <= δ && return δ
    u = (d - δ) / δ
    return d - (d - δ) * exp(-u)
end
function _belief_enc(x, kind::Symbol)
    kind === :free && return x
    if kind === :positive
        δ = oftype(x, BELIEF_ENC_MIN)
        return _state_invsoftplus(_soft_floor(x - oftype(x, STATE_POSITIVE_FLOOR), δ))
    end
    δ = oftype(x, BELIEF_ENC_MIN_UNIT)
    ε = oftype(x, STATE_UNIT_EPS)
    q = (x - ε) / (one(x) - 2ε)
    qs = one(q) - _soft_floor(one(q) - _soft_floor(q, δ), δ)          # floored at both ends
    return log(qs / (one(qs) - qs))
end
_belief_dec(y, kind::Symbol) = kind === :free ? y : _state_decode(y, kind)
function _belief_dec_jac(y, kind::Symbol)
    kind === :free && return one(y)
    kind === :positive && return logistic(y)
    return (one(y) - 2oftype(y, STATE_UNIT_EPS)) * logistic(y) * (one(y) - logistic(y))
end
"The belief coordinates of a physical belief state."
belief_encode(x_b::AbstractVector, bl::BeliefLayout) = [_belief_enc(v, k) for (v, k) in zip(x_b, bl.kinds)]
"The physical belief state of a coordinate vector, strictly inside its domain."
belief_decode(y::AbstractVector, bl::BeliefLayout) = [_belief_dec(v, k) for (v, k) in zip(y, bl.kinds)]
"The diagonal of ∂ physical / ∂ coordinate at `y`."
belief_decode_jacobian(y::AbstractVector, bl::BeliefLayout) = [_belief_dec_jac(v, k) for (v, k) in zip(y, bl.kinds)]
"Decode a coordinate Gaussian by first-order moment propagation, for physical diagnostics."
function belief_physical_moments(y::AbstractVector, P::AbstractMatrix, bl::BeliefLayout)
    x = belief_decode(y, bl)
    d = belief_decode_jacobian(y, bl)
    return x, Matrix(Symmetric((d .* P) .* d'))
end

"""
    belief_cov_to_coord(Q, y, bl; max_sd=3.0) -> Matrix

A physical covariance mapped into the coordinates at `y`, any one entry's coordinate sd capped at `max_sd`
(the filter's own rule next to a physical boundary), the cap a sharp smooth minimum `(1 + (sd/max)⁸)^(−1/8)`
rather than a hard one: a smoother's process whitening depends on the state it is taken at, and a hard cap
put a kink in the objective wherever a collagen node's width crossed the cap as it converted (the two-piece
bench of 2026-09-25). Under an eighth of the cap the two agree to 1e-13.
"""
function belief_cov_to_coord(Q::AbstractMatrix, y::AbstractVector, bl::BeliefLayout; max_sd::Float64=3.0)
    d = inv.(max.(belief_decode_jacobian(y, bl), eps(Float64)))
    sd = sqrt.(max.(diag(Q) .* d .^ 2, 0.0))
    scale = (1 .+ (sd ./ max_sd) .^ 8) .^ (-1 / 8)
    e = scale .* d
    return Matrix(Symmetric((e .* Q) .* e'))
end

# the rate and the initial sd of a group from a StateNoise (per piece, `sn[id]` when a Dict is given)
_sn_of(sn::StateNoise, ::Symbol) = sn
function _sn_of(sn::AbstractDict, id::Symbol)
    haskey(sn, id) && return sn[id]
    haskey(sn, :default) && return sn[:default]
    throw(ArgumentError("no StateNoise for piece $id (give one per piece, or a :default)"))
end
_pit_sn(sn::StateNoise) = sn
function _pit_sn(sn::AbstractDict)
    haskey(sn, :pit) && return sn[:pit]
    haskey(sn, :default) && return sn[:default]
    throw(ArgumentError("a per-piece StateNoise dictionary needs an explicit :pit or :default entry"))
end
function _group_rate(bl::BeliefLayout, sn, g::String, init::Bool)
    if occursin('.', g)
        id, f = rsplit(g, '.'; limit=2); s = _sn_of(sn, Symbol(id))
        return init ? getfield(s, Symbol(f * "0")) : getfield(s, Symbol(f))
    end
    s = _pit_sn(sn)
    return init ? getfield(s, Symbol(g * "0")) : getfield(s, Symbol(g))
end

"""
    belief_process_cov(bl, sn, sd_of_group) -> Matrix

The physical covariance of the belief state from per-group widths: under the noise's `:offset` structure a
field's nodes share one draw (rank one plus the jitter), the scalars are independent; under `:nodes` a
diagonal. `sd_of_group(g)` gives the width of group `g`.
"""
function belief_process_cov(bl::BeliefLayout, sn, sd_of_group)
    Q = zeros(bl.n, bl.n)
    for (g, rg) in bl.groups
        s = sd_of_group(g)
        field = occursin('.', g) && !endswith(g, ".bark")
        sng = occursin('.', g) ? _sn_of(sn, Symbol(first(rsplit(g, '.'; limit=2)))) : _pit_sn(sn)
        sng.fields in (:offset, :nodes) || throw(ArgumentError("StateNoise.fields must be :offset or :nodes, not $(sng.fields)"))
        if field && sng.fields === :offset && length(rg) > 1
            Q[rg, rg] .= s^2
            for i in rg; Q[i, i] = s^2 * (1 + sng.node_jitter^2); end
        else
            for i in rg; Q[i, i] = s^2; end
        end
    end
    return Q
end
"The physical process covariance over `Δh` hours (the rates times √Δh) with the events' disturbances (`kinds` the lid-open events crossed: :add, :wrap, :pull, :spritz)."
function belief_process_cov(bl::BeliefLayout, sn, Δh::Real; events::AbstractVector{Symbol}=Symbol[])
    Q = belief_process_cov(bl, sn, g -> _group_rate(bl, sn, g, false) * sqrt(max(Δh, 0.0)))
    ps = _pit_sn(sn)
    ip = first(bl.groups[findfirst(x -> first(x) == "T_pit", bl.groups)][2])
    ifb = first(bl.groups[findfirst(x -> first(x) == "T_fb", bl.groups)][2])
    ifuel = first(bl.groups[findfirst(x -> first(x) == "m_fuel", bl.groups)][2])
    for kind in events
        σ = kind === :spritz ? ps.T_spritz : ps.T_event
        Q[ip, ip] += σ^2; Q[ifb, ifb] += σ^2
        kind === :add && (Q[ifuel, ifuel] += ps.m_add^2)
    end
    return Q
end
"The physical covariance of the belief at the start: the initial widths by group."
belief_init_cov(bl::BeliefLayout, sn) = belief_process_cov(bl, sn, g -> _group_rate(bl, sn, g, true))

"""
    belief_handoff(x_b, bl) -> (pieces, pit, ignited)

The restart keywords of a physical belief state: per piece `handoff`'s contract (the latch at zero), the
pit's keywords and the ignited fraction (`system_solve`'s `ics`, `pit_ics`).
"""
function belief_handoff(x_b::AbstractVector, bl::BeliefLayout)
    pieces = map(bl.sys.pieces) do id
        g(f) = x_b[belief_index(bl, id, f)]
        haskey(bl.sys.piece[id], :ring) || return (T0f=g(:T), W0f=g(:W), C0f=g(:C), F0f=g(:F), B0f=g(:bark_b), S0f=g(:bark_s))   # a grid piece
        (T0f=g(:T), W0f=g(:W), C0f=g(:C), F0f=g(:F), Ssalt0f=g(:S), B0=g(:bark_b)[1], S0=g(:bark_s)[1], R0=g(:ring)[1], LATCH0=0.0)
    end
    p(f) = x_b[belief_index(bl, :pit, f)][1]
    return (pieces=collect(pieces), pit=(Tpit0=p(:T_pit), mfuel0=p(:m_fuel), mpan0=p(:m_pan), Tpan0=p(:T_pan), Tfb0=p(:T_fb)),
            ignited=clamp(p(:ignited), 1e-6, 1 - 1e-6))
end
