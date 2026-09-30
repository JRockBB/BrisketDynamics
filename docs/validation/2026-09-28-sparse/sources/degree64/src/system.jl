# The cook as a SYSTEM: one shared pit and explicit, heterogeneous pieces of meat, each with its own model,
# state, parameters, geometry, events and exchange port, coupled through the pit's energy balance (the heat
# loads sum) and ONE chamber-humidity closure (every piece's vapor dilutes into the same air flow). This is
# the semantic core the multi-meat inference plan asks for (docs/multi-meat-inference-plan.md): named
# state and parameter layouts generated from the system, model-owned initialization, restart, propagation,
# event application, observables and exchange fluxes, and forward-mode derivatives through the solve.
#
# The 1-D slab of meats/common.jl is the first piece model (`Slab1D`); its RHS, constitutive kernels and
# quality states are reused unchanged (`meat_rhs_core!`). `MeatConfig.n_meat` survives as a piece's
# MULTIPLICITY (identical copies sharing the pit), so a system of one piece with n_meat = n is the old
# monolith with n identical briskets, and a system of n pieces with n_meat = 1 each is the same cook
# with the copies made explicit (test/test_system.jl gates both parities).
#
# The chamber humidity: piece i's pit-entering flux is j_i = (1 − pl_i) a_i (e_i − p_v), with a_i its
# surface conductance k/(R_v T_s) under wrap and skin, e_i = f_aw a_w p_sat(T_s) its equilibrium vapor
# pressure and pl_i its pull blend, and the chamber vapor pressure is the base vapor pressure plus every
# piece's vapor diluted by the air mass flow, p_v = pv_base + K Σ_i A_i n_i α_i j_i (K = p_atm/(0.622 ṁ_air),
# α_i the exposed-area factor). Substitution gives
# p_v* = [pv_base + K Σ_i (1 − pl_i) A_i n_i α_i a_i e_i] /
#        [1 + K Σ_i (1 − pl_i) A_i n_i α_i a_i],
# a scalar closed form over all pieces. It matches the monolith while a piece is in the pit and after it is
# fully out; during the monolith's artificial finite pull blend this system closure is the conservative law
# (the monolith leaves the blend out of its denominator). It is what the RHS and the
# observables both use, so the reported chamber vapor pressure is the one the pieces saw.

using LinearAlgebra: dot

"""
    PieceModel

A piece of meat in a `CookSystem`: the model owns its state layout, its initial state, its exchange
port toward the pit and its right-hand side. `Slab1D` is the 1-D slab; other geometries implement the
same interface (`ode_layout`, `initial_state`, `_piece_exchange`, `_piece_rhs!`, `piece_outputs`).
"""
abstract type PieceModel end

"The ordered, model-owned physical parameter schema. Existing slab and grid adapters use the legacy eight names."
piece_parameter_schema(::PieceModel) = (:D_eff, :k_m, :h_c, :L, :k_mult, :k_e, :a_myosin, :m_conv)
"Indices of that schema which are estimable for this model."
piece_parameter_indices(pc::PieceModel) = eachindex(piece_parameter_schema(pc))
"Events a piece model accepts."
piece_event_kinds(::PieceModel) = (:wrap, :pull, :spritz)
"Semantic observables a piece model can compile through the estimator adapter."
piece_observable_schema(::PieceModel) = (:temperature, :surface_temperature, :mass, :bark)
"Restart fields owned by a piece model."
function piece_restart_fields end
"Multiplicity represented by one explicit piece node."
piece_multiplicity(pc::PieceModel) = pc.cfg.n_meat

"""
    Slab1D(id; cfg=MeatConfig())

The 1-D slab piece (meats/common.jl): `id` names it in the system's layouts and in the cook log; `cfg`
is its own configuration (geometry mode, nodes, constitutive constants), `cfg.n_meat` its multiplicity.
"""
struct Slab1D <: PieceModel
    id::Symbol
    cfg::MeatConfig
end
Slab1D(id::Symbol; cfg::MeatConfig=MeatConfig()) = Slab1D(id, cfg)
piece_id(pc::PieceModel) = pc.id
piece_restart_fields(::Slab1D) = (:T0f, :W0f, :C0f, :F0f, :Ssalt0f, :B0, :S0, :R0, :LATCH0)

"""
    BlockLayout(entries)

A named layout of one block of a state vector from `(name, length, kind)` entries: each entry's index
range within the block and the KIND of its coordinate (`:positive`, `:unit` or `:free`), which the
estimator's constrained belief coordinates read. Index by name: `lay[:T]`.
"""
struct BlockLayout
    n::Int
    names::Vector{Symbol}
    ranges::Vector{UnitRange{Int}}
    kinds::Vector{Symbol}
end
function BlockLayout(entries::AbstractVector{<:Tuple{Symbol,Int,Symbol}})
    names = Symbol[]; ranges = UnitRange{Int}[]; kinds = Symbol[]
    o = 0
    for (nm, len, kind) in entries
        kind in (:positive, :unit, :free) || throw(ArgumentError("a coordinate kind is :positive, :unit or :free, not :$kind"))
        push!(names, nm); push!(ranges, (o + 1):(o + len)); push!(kinds, kind)
        o += len
    end
    return BlockLayout(o, names, ranges, kinds)
end
function Base.getindex(b::BlockLayout, name::Symbol)
    i = findfirst(==(name), b.names)
    i === nothing && throw(ArgumentError("no entry named $name in the layout (its entries are $(join(b.names, ", ")))"))
    return b.ranges[i]
end
Base.haskey(b::BlockLayout, name::Symbol) = name in b.names
"The coordinate kind of every entry of the block, expanded to its length."
expand_kinds(b::BlockLayout) = reduce(vcat, [fill(k, length(r)) for (k, r) in zip(b.kinds, b.ranges)]; init=Symbol[])

"The ODE state layout of a piece: what the system integrates for it (the coordinate kinds are the filter's: the ring's depth, under a metre, as a unit fraction)."
ode_layout(pc::Slab1D) = (nx = pc.cfg.nx;
    BlockLayout([(:T, nx, :positive), (:W, nx, :positive), (:C, nx, :unit), (:F, nx, :positive), (:S, nx, :positive),
                 (:bark_b, 1, :positive), (:bark_s, 1, :positive), (:ring, 1, :unit), (:latch, 1, :unit)]))

"The pit's ODE state layout: the monolith's order (chamber temperature, fuel, ignited fraction, pan water, pan temperature, firebox)."
const PIT_LAYOUT = BlockLayout([(:T_pit, 1, :positive), (:m_fuel, 1, :positive), (:ignited, 1, :unit),
                                (:m_pan, 1, :positive), (:T_pan, 1, :positive), (:T_fb, 1, :positive)])

"""
    SystemLayout

The named layout of a system's state: every piece's block (`piece[id]`, a `BlockLayout` in local indices,
at `piece_range[id]` in the state), then the pit's. `state_index(lay, id, field)` is the global range.
"""
struct SystemLayout
    pieces::Vector{Symbol}
    piece::Dict{Symbol,BlockLayout}
    piece_range::Dict{Symbol,UnitRange{Int}}
    pit::BlockLayout
    pit_range::UnitRange{Int}
    n::Int
end
function SystemLayout(pieces::AbstractVector{<:PieceModel})
    ids = Symbol[piece_id(pc) for pc in pieces]
    length(unique(ids)) == length(ids) || throw(ArgumentError("piece ids must be distinct (got $(ids))"))
    pl = Dict{Symbol,BlockLayout}(); pr = Dict{Symbol,UnitRange{Int}}()
    o = 0
    for pc in pieces
        b = ode_layout(pc)
        pl[pc.id] = b; pr[pc.id] = (o + 1):(o + b.n)
        o += b.n
    end
    pit_r = (o + 1):(o + PIT_LAYOUT.n)
    return SystemLayout(ids, pl, pr, PIT_LAYOUT, pit_r, o + PIT_LAYOUT.n)
end
"The global index range of `field` of piece `id` (or of the pit, `id = :pit`) in the system state."
function state_index(lay::SystemLayout, id::Symbol, field::Symbol)
    id === :pit && return lay.pit_range[lay.pit[field]]
    haskey(lay.piece_range, id) || throw(ArgumentError("no piece named $id (the pieces are $(join(lay.pieces, ", ")))"))
    return lay.piece_range[id][lay.piece[id][field]]
end
"The coordinate kind of every entry of the system state."
function coordinate_kinds(lay::SystemLayout)
    ks = Symbol[]
    for id in lay.pieces; append!(ks, expand_kinds(lay.piece[id])); end
    append!(ks, expand_kinds(lay.pit))
    return ks
end

"""
    CookSystem(pieces; pit=PitConfig(), hum=Humidity())

One pit, its humidity constants and the explicit pieces cooking in it. The state layout is generated.
"""
struct CookSystem{P<:PieceModel}
    pieces::Vector{P}
    pit::PitConfig
    hum::Humidity
    layout::SystemLayout
end
function CookSystem(pieces::AbstractVector{<:PieceModel}; pit::PitConfig=PitConfig(), hum::Humidity=Humidity())
    isempty(pieces) && throw(ArgumentError("a cook system needs at least one piece"))
    any(pc -> piece_id(pc) === :pit, pieces) &&
        throw(ArgumentError(":pit is reserved for the shared cooker and cannot be a piece id"))
    any(pc -> isempty(String(piece_id(pc))), pieces) &&
        throw(ArgumentError("piece ids cannot be empty"))
    return CookSystem(collect(pieces), pit, hum, SystemLayout(pieces))
end
"One piece in the pit: the monolith's own cook as a system."
CookSystem(pc::PieceModel; kw...) = CookSystem([pc]; kw...)
n_state(sys::CookSystem) = sys.layout.n
n_pieces(sys::CookSystem) = length(sys.pieces)
function piece(sys::CookSystem, id::Symbol)
    i = findfirst(pc -> pc.id === id, sys.pieces)
    i === nothing && throw(ArgumentError("no piece named $id (the pieces are $(join(sys.layout.pieces, ", ")))"))
    return sys.pieces[i]
end

# --- events ------------------------------------------------------------------------------------------------

"""
    CookEvent(kind, target, t; params...)

An event of the cook: `kind` one of `:wrap`, `:pull`, `:spritz` (on a piece, `params` `dt` and `m`: the
train's finite positive interval [s] and dose [kg/m²]) and `:add` (on the pit, `params` `dt_add`: the split
interval, Inf for one addition). `t` [s] is absolute cook time. Events are validated data; both positional
and keyword construction enforce the same contract.
"""
struct CookEvent
    kind::Symbol
    target::Symbol
    t::Float64
    params::NamedTuple
    function CookEvent(kind::Symbol, target::Symbol, t::Real, params::NamedTuple)
        _validate_event_fields(kind, target, t, params)
        return new(kind, target, Float64(t), params)
    end
end
function _validate_event_fields(kind::Symbol, target::Symbol, t::Real, nt::NamedTuple)
    kind in (:wrap, :pull, :spritz, :add) || throw(ArgumentError("an event is :wrap, :pull, :spritz or :add, not :$kind"))
    kind === :add && target !== :pit && throw(ArgumentError("a fuel addition is the pit's event (target :pit), not $(target)'s"))
    kind !== :add && target === :pit && throw(ArgumentError("a $kind is a piece's event, not the pit's"))
    isfinite(t) && t >= 0 || throw(ArgumentError("an event time must be finite and nonnegative"))
    allowed = kind === :spritz ? (:dt, :m) : kind === :add ? (:dt_add,) : ()
    all(k -> k in allowed, keys(nt)) || throw(ArgumentError("unsupported parameters for a $kind event: $(collect(keys(nt)))"))
    if kind === :spritz
        dt = get(nt, :dt, 3600.0); m = get(nt, :m, 0.0)
        dt isa Real && isfinite(dt) && dt > 0 || throw(ArgumentError("a spritz interval must be finite and positive"))
        m isa Real && isfinite(m) && m >= 0 || throw(ArgumentError("a spritz dose must be finite and nonnegative"))
    elseif kind === :add
        dt = get(nt, :dt_add, Inf)
        dt isa Real && ((isfinite(dt) || dt == Inf) && dt > 0) ||
            throw(ArgumentError("a fuel-addition interval must be positive"))
    end
    return nt
end
function CookEvent(kind::Symbol, target::Symbol, t::Real; params...)
    nt = NamedTuple(params)
    return CookEvent(kind, target, Float64(t), nt)
end

"Validate event identity and uniqueness against a concrete system."
function validate_events(sys::CookSystem, events::AbstractVector{CookEvent})
    seen = Set{Tuple{Symbol,Symbol}}()
    ids = Set(sys.layout.pieces)
    for e in events
        _validate_event_fields(e.kind, e.target, e.t, e.params)
        if e.kind === :add
            e.target === :pit || throw(ArgumentError("a fuel addition targets :pit"))
        else
            e.target in ids || throw(ArgumentError("event $(e.kind) targets unknown piece $(e.target)"))
            e.kind in piece_event_kinds(piece(sys, e.target)) ||
                throw(ArgumentError("piece $(e.target) does not support event $(e.kind)"))
        end
        key = (e.target, e.kind)
        key in seen && throw(ArgumentError("duplicate $(e.kind) event for $(e.target); use one train/event per kind"))
        push!(seen, key)
    end
    return events
end

"The named event parameters of piece `id` from a list of events (`NEVER` where an event is absent)."
function piece_events(events::AbstractVector{CookEvent}, id::Symbol; cfg::MeatConfig=MeatConfig())
    t_wrap, t_pull, t_spritz, dt_spritz, m_spritz = WRAP_NEVER, PULL_NEVER, SPRITZ_NEVER, 3600.0, cfg.m_spritz
    for e in events
        e.target === id || continue
        e.kind === :wrap && (t_wrap = e.t)
        e.kind === :pull && (t_pull = e.t)
        if e.kind === :spritz
            t_spritz = e.t
            dt_spritz = Float64(get(e.params, :dt, 3600.0))
            m_spritz = Float64(get(e.params, :m, cfg.m_spritz))
        end
    end
    return (t_wrap=t_wrap, t_pull=t_pull, t_spritz=t_spritz, dt_spritz=dt_spritz, m_spritz=m_spritz)
end
"The pit's event parameters from a list of events."
function pit_events(events::AbstractVector{CookEvent}; pit::PitConfig=PitConfig())
    t_add, dt_add = ADD_NEVER, pit.dt_add
    for e in events
        e.kind === :add || continue
        t_add = e.t; dt_add = Float64(get(e.params, :dt_add, pit.dt_add))
    end
    return (t_add=t_add, dt_add=dt_add)
end

# --- the parameter vector: one flat vector with a generated named layout ----------------------------------

"""
    ParamLayout(sys, nu)

The named layout of the system's parameter vector `p` for `nu` damper knots: per piece its model-owned θ block
(`(id, :theta)`), its exposed area `(id, :A_meat)` and its event parameters (`(id, :t_wrap)`,
`(id, :t_pull)`, `(id, :t_spritz)`, `(id, :dt_spritz)`, `(id, :m_spritz)`, `(id, :T_trig)`); then the
pit's φ block `(:pit, :phi)` (`PIT_NP`), the damper knots `(:pit, :u_knots)`, `(:pit, :t_add)` and
`(:pit, :dt_add)`. Index by key: `play[(:brisket, :theta)]`.
"""
struct ParamLayout
    keys::Vector{Tuple{Symbol,Symbol}}
    ranges::Dict{Tuple{Symbol,Symbol},UnitRange{Int}}
    n::Int
end
const PIECE_PARAM_SCALARS = (:A_meat, :t_wrap, :t_pull, :t_spritz, :dt_spritz, :m_spritz, :T_trig)
function ParamLayout(sys::CookSystem, nu::Int)
    keys = Tuple{Symbol,Symbol}[]; ranges = Dict{Tuple{Symbol,Symbol},UnitRange{Int}}()
    o = 0
    for pc in sys.pieces
        nθ = length(piece_parameter_schema(pc))
        push!(keys, (pc.id, :theta)); ranges[(pc.id, :theta)] = (o + 1):(o + nθ); o += nθ
        for s in PIECE_PARAM_SCALARS
            push!(keys, (pc.id, s)); ranges[(pc.id, s)] = (o + 1):(o + 1); o += 1
        end
    end
    push!(keys, (:pit, :phi)); ranges[(:pit, :phi)] = (o + 1):(o + PIT_NP); o += PIT_NP
    push!(keys, (:pit, :u_knots)); ranges[(:pit, :u_knots)] = (o + 1):(o + nu); o += nu
    push!(keys, (:pit, :t_add)); ranges[(:pit, :t_add)] = (o + 1):(o + 1); o += 1
    push!(keys, (:pit, :dt_add)); ranges[(:pit, :dt_add)] = (o + 1):(o + 1); o += 1
    return ParamLayout(keys, ranges, o)
end
function Base.getindex(pl::ParamLayout, key::Tuple{Symbol,Symbol})
    haskey(pl.ranges, key) || throw(ArgumentError("no parameter $(key) in the layout"))
    return pl.ranges[key]
end
param_index(pl::ParamLayout, id::Symbol, name::Symbol) = pl[(id, name)]
n_params(pl::ParamLayout) = pl.n

# the index tables the RHS and the observables read (built once per closure, no lookups in the hot path)
struct _SysIndex
    θ::Vector{UnitRange{Int}}
    A::Vector{Int}
    t_wrap::Vector{Int}
    t_pull::Vector{Int}
    t_spritz::Vector{Int}
    dt_spritz::Vector{Int}
    m_spritz::Vector{Int}
    T_trig::Vector{Int}
    x::Vector{UnitRange{Int}}
    φ::UnitRange{Int}
    u::UnitRange{Int}
    t_add::Int
    dt_add::Int
    pit::UnitRange{Int}
end
function _SysIndex(sys::CookSystem, play::ParamLayout)
    f(s) = [first(play[(pc.id, s)]) for pc in sys.pieces]
    return _SysIndex([play[(pc.id, :theta)] for pc in sys.pieces], f(:A_meat), f(:t_wrap), f(:t_pull), f(:t_spritz),
                     f(:dt_spritz), f(:m_spritz), f(:T_trig), [sys.layout.piece_range[pc.id] for pc in sys.pieces],
                     play[(:pit, :phi)], play[(:pit, :u_knots)], first(play[(:pit, :t_add)]), first(play[(:pit, :dt_add)]),
                     sys.layout.pit_range)
end

"""
    pack_params(sys, θs, As, φ, u_knots; events=CookEvent[], T_trig=Inf, t_add=NaN, dt_add=NaN) -> (p, play)

The system's parameter vector from per-piece θ vectors and exposed areas, the pit's φ, the damper knots and
the events (each piece reads its own; `T_trig` the wrap trigger temperature per piece or one for all;
`t_add`/`dt_add` override the events' fuel addition when given). Duals pass through.
"""
function pack_params(sys::CookSystem, θs::AbstractVector, As::AbstractVector, φ::AbstractVector, u_knots::AbstractVector;
                     events::AbstractVector{CookEvent}=CookEvent[], T_trig=Inf, t_add::Real=NaN, dt_add::Real=NaN)
    np = n_pieces(sys)
    length(θs) == np || throw(ArgumentError("one θ per piece ($(length(θs)) for $np pieces)"))
    length(As) == np || throw(ArgumentError("one exposed area per piece ($(length(As)) for $np pieces)"))
    length(φ) == PIT_NP || throw(ArgumentError("φ must have length $PIT_NP; got $(length(φ))"))
    for (pc, θ) in zip(sys.pieces, θs)
        nθ = length(piece_parameter_schema(pc))
        length(θ) == nθ || throw(ArgumentError("piece $(pc.id)'s θ must follow $(piece_parameter_schema(pc)) (got $(length(θ)) values)"))
    end
    validate_events(sys, events)
    trig = T_trig isa AbstractVector ? T_trig : fill(T_trig, np)
    length(trig) == np || throw(ArgumentError("one wrap trigger per piece"))
    play = ParamLayout(sys, length(u_knots))
    Tel = promote_type(eltype(reduce(vcat, θs)), eltype(As), eltype(φ), eltype(u_knots), eltype(trig), Float64)
    p = zeros(Tel, play.n)
    for (i, pc) in enumerate(sys.pieces)
        ev = piece_events(events, pc.id; cfg=pc.cfg)
        p[play[(pc.id, :theta)]] = θs[i]
        p[first(play[(pc.id, :A_meat)])] = As[i]
        p[first(play[(pc.id, :t_wrap)])] = ev.t_wrap
        p[first(play[(pc.id, :t_pull)])] = ev.t_pull
        p[first(play[(pc.id, :t_spritz)])] = ev.t_spritz
        p[first(play[(pc.id, :dt_spritz)])] = ev.dt_spritz
        p[first(play[(pc.id, :m_spritz)])] = ev.m_spritz
        p[first(play[(pc.id, :T_trig)])] = trig[i]
    end
    pe = pit_events(events; pit=sys.pit)
    p[play[(:pit, :phi)]] = φ
    p[play[(:pit, :u_knots)]] = u_knots
    p[first(play[(:pit, :t_add)])] = isnan(t_add) ? pe.t_add : t_add
    p[first(play[(:pit, :dt_add)])] = isnan(dt_add) ? pe.dt_add : dt_add
    bad = findall(x -> isnan(_primal(x)), p)
    isempty(bad) || throw(ArgumentError("the system's parameter vector has NaN entries at $bad"))
    return p, play
end
"The named pieces of a parameter vector: per piece `(theta, A_meat, events...)`, the pit's `(phi, u_knots, t_add, dt_add)`, keyed as the layout."
function unpack_params(play::ParamLayout, p::AbstractVector)
    length(p) == play.n || throw(ArgumentError("the parameter vector has $(length(p)) entries; the layout $(play.n)"))
    out = Dict{Tuple{Symbol,Symbol},Any}()
    for k in play.keys
        r = play[k]
        out[k] = length(r) == 1 && k[2] !== :u_knots ? p[first(r)] : p[r]
    end
    return out
end

# --- the piece's exchange port and right-hand side ------------------------------------------------------

"""
    _piece_exchange(pc::Slab1D, u, t, θ, airflow, t_wrap, t_pull, A_meat, hoisted)

The slab's exchange port at one instant: its surface conductance times its exposed area and multiplicity
(`cond = A n α a`, with a = k/(R_v T_s) under the wrap's and the skin's throttling), that conductance times
its equilibrium vapor pressure (`num = cond · f_aw a_w p_sat(T_s)`; a piece with many exposed faces sums
both over them), the exposed-area factor, the wrap blend (time trigger OR latch), the time-only wrap blend
the lid term reads, and the pull blend. The same expressions `meat_rhs_core!` and `surface_fluxes_blended`
use, so the closure and the RHS agree.
"""
function _piece_exchange(pc::Slab1D, u, t, θ, airflow, t_wrap, t_pull, A_meat, hoisted)
    n_s, n_a, v0 = hoisted[1], hoisted[2], hoisted[3]
    cfg = pc.cfg; nx = cfg.nx
    k_m, m_conv = θ[2], θ[8]
    k_m = k_m * (airflow / cfg.g_ref)^m_conv
    W = @view u[(nx + 1):2nx]; F = @view u[(3nx + 1):4nx]
    vsum = specific_volume(W[1], F[1], cfg) / 2 + specific_volume(W[nx], F[nx], cfg) / 2
    @inbounds for i in 2:(nx - 1)
        vsum += specific_volume(W[i], F[i], cfg)
    end
    a_area = (vsum / (nx - 1) / v0)^n_a
    Ts, Ws = u[nx], u[2nx]
    latch = clamp(u[5nx + 4], zero(u[5nx + 4]), one(u[5nx + 4]))
    s_time = wrap_blend(t, t_wrap, cfg)
    s = 1 - (1 - s_time) * (1 - latch)
    pl = pull_blend(t, t_pull, cfg)
    f_aw = 1 / (1 + cfg.k_aw * max(u[5nx], zero(u[5nx])))
    k_eff = isinf(cfg.k_skin) ? k_m : 1 / (1 / k_m + 1 / cfg.k_skin)
    a = (1 - cfg.η_evap * s) * k_eff / (R_V * Ts)
    e = f_aw * a_w_oswin(Ws, cfg) * p_sat(Ts)
    cond = A_meat * cfg.n_meat * a_area * a
    return cond, cond * e, a_area, s, s_time, pl
end

"The piece's right-hand side under the resolved chamber vapor pressure `pv_star` (the mixing term is then zero: the closure already holds the piece's own vapor); returns the heat drawn from the pit [W]. Every piece model implements it."
function _piece_rhs!(pc::Slab1D, du, u, t, θ, Tp, ṁ, pv_star, airflow, t_wrap, t_pull, t_spritz, dt_spritz, m_spritz, T_trig, hoisted, A_meat)
    n_s, n_a, v0, wb = hoisted
    return meat_rhs_core!(du, u, t, θ, Tp, ṁ, pv_star, zero(pv_star), airflow, t_wrap, t_pull, t_spritz, dt_spritz, m_spritz,
                          pc.cfg, n_s, n_a, v0, wb, T_trig; A_meat=A_meat)
end
_piece_hoisted(pc::Slab1D) = (shrink_exponents(pc.cfg)..., bone_weights(pc.cfg))

# the pieces grouped by their PRIMAL spritz schedule (a train shared by several pieces is one lid lift per pulse,
# the gate the widest of the group's): the grouping compares values, so a
# Dual schedule keeps its partials inside its group
function _spritz_groups(p, ix::_SysIndex)
    np = length(ix.t_spritz)
    groups = Vector{Vector{Int}}()
    for i in 1:np
        ti, di = _primal(p[ix.t_spritz[i]]), _primal(p[ix.dt_spritz[i]])
        g = findfirst(gr -> _primal(p[ix.t_spritz[gr[1]]]) == ti && _primal(p[ix.dt_spritz[gr[1]]]) == di, groups)
        g === nothing ? push!(groups, [i]) : push!(groups[g], i)
    end
    return groups
end

"""
    _system_instant(sys, x, p, t, ud, ix, hoisted)

The system's algebraic quantities at one instant from its state and parameters under the damper fraction
`ud`: the pit algebra (airflow, burn rate, air mass flow), the pan's fluxes, the base vapor pressure, every
piece's exchange port (vectors over the pieces: conductance, equilibrium vapor pressure, area factor, wrap,
time-only wrap, pull), the resolved chamber vapor pressure `pv_star` and the mixing constant `K`. The RHS
and the observables read the same function.
"""
function _system_instant(sys::CookSystem, x, p, t, ud, ix::_SysIndex, hoisted; pv_bias=0.0)
    pit = sys.pit; hum = sys.hum
    pr = ix.pit
    Tp, m, a, mp, Tpan, Tfb = x[pr[1]], x[pr[2]], x[pr[3]], x[pr[4]], x[pr[5]], x[pr[6]]
    r_max, u_leak = p[ix.φ[3]], p[ix.φ[5]]
    airflow, ṁ, mair, pv_nopan = _pit_algebra(ud, m, a, r_max, u_leak, pit, hum)
    q_pan, ṁ_pan, pv_base = _pan_algebra(Tp, mp, Tpan, mair, pv_nopan, pit, hum)
    K = (hum.p_atm / EPS_W) / mair                 # mixing constant per unit exposed area and flux
    Tel = promote_type(eltype(x), eltype(p), typeof(float(t)))
    np = length(sys.pieces)
    cond = Vector{Tel}(undef, np); nums = Vector{Tel}(undef, np); area = Vector{Tel}(undef, np)
    s = Vector{Tel}(undef, np); s_time = Vector{Tel}(undef, np); pl = Vector{Tel}(undef, np)
    num = zero(Tel); den = zero(Tel)
    for i in 1:np
        pc = sys.pieces[i]
        θ = Tuple(p[ix.θ[i]])
        c_i, n_i, a_i, s_i, st_i, pl_i = _piece_exchange(pc, view(x, ix.x[i]), t, θ, airflow, p[ix.t_wrap[i]], p[ix.t_pull[i]],
                                                         p[ix.A[i]], hoisted[i])
        cond[i] = c_i; nums[i] = n_i; area[i] = a_i; s[i] = s_i; s_time[i] = st_i; pl[i] = pl_i
        gate = 1 - pl_i
        num += gate * n_i
        den += gate * c_i
    end
    pv_uncorrected = (pv_base + K * num) / (1 + K * den)
    pv_star = pv_uncorrected - pv_bias
    # Equivalent omitted vapor-load source [kg/s] that makes the corrected pressure satisfy the same
    # closed balance, rather than shifting the pressure only after the pieces have used it.
    vapor_correction = -pv_bias * (1 + K * den) / K
    return (airflow=airflow, mdot=ṁ, mair=mair, K=K, q_pan=q_pan, mdot_pan=ṁ_pan, pv_base=pv_base, pv_star=pv_star,
            pv_uncorrected, vapor_correction, cond=cond, num=nums, area=area, s=s, s_time=s_time, pl=pl,
            Tp=Tp, m=m, a=a, mp=mp, Tpan=Tpan, Tfb=Tfb)
end

"""
    make_system_rhs(sys, u_t0, u_dt, nu) -> (rhs!, play)

The in-place coupled RHS over the system state (`sys.layout`) with the parameter vector laid out by
`ParamLayout(sys, nu)`; the damper schedule is interpolated from `nu` knots on the uniform grid
`u_t0 + (i − 1) u_dt`.
"""
function make_system_rhs(sys::CookSystem, u_t0::Real, u_dt::Real, nu::Int; correction=nothing)
    pit = sys.pit
    play = ParamLayout(sys, nu)
    ix = _SysIndex(sys, play)
    hoisted = [_piece_hoisted(pc) for pc in sys.pieces]
    λ_pan = _pan_lambda(sys)
    pr = ix.pit
    np = length(sys.pieces)
    rhs! = function sys_rhs!(dx, x, p, t)
        C_th, UA, ΔH_eff, k_spread = p[ix.φ[1]], p[ix.φ[2]], p[ix.φ[4]], p[ix.φ[6]]
        u_knots = @view p[ix.u]
        ud = clamp(lininterp(t, u_t0, u_dt, u_knots), zero(eltype(u_knots)), one(eltype(u_knots)))
        if correction === nothing
            inst = _system_instant(sys, x, p, t, ud, ix, hoisted)
            corr = (pv=zero(inst.pv_star), q=zero(inst.pv_star))
        else
            corr = correction(t)
            inst = _system_instant(sys, x, p, t, ud, ix, hoisted; pv_bias=corr.pv)
        end
        Tp, m, a, mp, Tpan, Tfb = inst.Tp, inst.m, inst.a, inst.mp, inst.Tpan, inst.Tfb
        pv_piece = inst.pv_star
        q_meat = zero(inst.pv_star)
        for i in 1:np
            θ = Tuple(p[ix.θ[i]])
            q_meat += _piece_rhs!(sys.pieces[i], view(dx, ix.x[i]), view(x, ix.x[i]), t, θ, Tp, inst.mdot, pv_piece, inst.airflow,
                                  p[ix.t_wrap[i]], p[ix.t_pull[i]], p[ix.t_spritz[i]], p[ix.dt_spritz[i]], p[ix.m_spritz[i]],
                                  p[ix.T_trig[i]], hoisted[i], p[ix.A[i]])
        end
        q_meat -= corr.q
        # The lid is one physical opening. Shared trains and coincident/overlapping pulses therefore take the
        # widest opening rather than adding several copies of the same heat-loss path.
        t_add = p[ix.t_add]; dt_add = p[ix.dt_add]
        lid = lid_open(t, t_add, pit.τ_open_add)
        for g in _spritz_groups(p, ix)
            gate = zero(Tp)
            for i in g
                gate = max(gate, (1 - inst.s_time[i]) * (1 - inst.pl[i]))
            end
            lid = max(lid, lid_open(t, spritz_pulse_time(t, p[ix.t_spritz[g[1]]], p[ix.dt_spritz[g[1]]]), pit.τ_open_spritz) * gate)
        end
        q_lid = pit.UA_open * (Tp - pit.T_amb) * lid
        a_c = clamp(a, zero(a), one(a))
        ṁ_add = fuel_add_rate(t, t_add, dt_add, pit)
        if pit.c_duct > 0
            q_duct = pit.c_duct * inst.airflow * (Tfb - Tp)
            q_burn = q_duct
            dx[pr[6]] = (ΔH_eff * inst.mdot - pit.UA_fb * (Tfb - pit.T_amb) - q_duct) / pit.C_fb
        else
            q_burn = ΔH_eff * inst.mdot
            dx[pr[6]] = zero(Tp)
        end
        dx[pr[1]] = (q_burn + pit.p_heater * ud - UA * (Tp - pit.T_amb) - q_meat - q_lid - inst.q_pan) / C_th
        dx[pr[2]] = -inst.mdot + ṁ_add
        dx[pr[3]] = k_spread * a_c * (1 - a_c) * inst.airflow + (1 - a_c) * ṁ_add / max(m, pit.m_taper)
        dx[pr[4]] = -inst.mdot_pan
        dx[pr[5]] = pit.m_pan0 > 0 ?
                    (inst.q_pan - λ_pan * inst.mdot_pan) / (CP_WATER * max(mp, zero(mp)) + pit.C_pan_dry) : zero(Tp)
        return nothing
    end
    return rhs!, play
end
# Water-pan latent heat is a cooker property, independent of piece order.
_pan_lambda(sys::CookSystem) = sys.pit.λ_pan

# --- initialization and restart -------------------------------------------------------------------------

"""
    initial_state(pc::Slab1D, θ; ics=(;), t_salt=NaN) -> Vector

The piece's initial ODE state: uniform fields at the config's raw state, the brine profile from `θ[4]` (L)
over `t_salt` (NaN: the config's), the bark, ring and latch at zero; the keywords of `ics` (`T0f`, `W0f`,
`C0f`, `F0f`, `Ssalt0f`, `B0`, `S0`, `R0`, `LATCH0`, the hand-off contract) replace them for a restart.
"""
function initial_state(pc::Slab1D, θ; ics::NamedTuple=(;), t_salt::Real=NaN)
    cfg = pc.cfg
    all(k -> k in piece_restart_fields(pc), keys(ics)) ||
        throw(ArgumentError("unsupported restart field for $(pc.id); expected $(piece_restart_fields(pc))"))
    g(k, default) = haskey(ics, k) ? getfield(ics, k) : default
    u0 = _meat_u0(cfg, g(:T0f, Float64[]), g(:W0f, Float64[]), g(:C0f, Float64[]);
                  F0f=g(:F0f, Float64[]), B0=g(:B0, NaN), S0=g(:S0, NaN), R0=g(:R0, NaN), LATCH0=g(:LATCH0, NaN),
                  Ssalt0f=g(:Ssalt0f, Float64[]), s0=salt_profile(θ[4], cfg, isnan(_primal(t_salt)) ? cfg.t_salt : t_salt))
    all(v -> isfinite(_primal(v)), u0) || throw(ArgumentError("restart state for $(pc.id) must be finite"))
    return u0
end

"""
    initial_state(sys, p, play; ics=[(;), ...], pit_ics=(;), t_salt=NaN) -> x0

The system's initial state from its parameter vector: every piece's `initial_state` (with its restart
keywords from `ics[i]` and its brine duration from `t_salt`, a scalar or one per piece), the pit at its
config's initial values with the ignited fraction from φ[7] (the keywords of `pit_ics`, `Tpit0`, `mfuel0`,
`mpan0`, `Tpan0`, `Tfb0`, replace them; `ignited` there replaces φ[7]). Dual parameters make a Dual state.
"""
function initial_state(sys::CookSystem, p::AbstractVector, play::ParamLayout; ics::AbstractVector=[(;) for _ in sys.pieces],
                       pit_ics::NamedTuple=(;), t_salt=NaN)
    pit = sys.pit
    length(ics) == n_pieces(sys) || throw(ArgumentError("one restart tuple per piece ($(length(ics)) for $(n_pieces(sys)) pieces)"))
    allowed_pit = (:Tpit0, :mfuel0, :ignited, :mpan0, :Tpan0, :Tfb0)
    all(k -> k in allowed_pit, keys(pit_ics)) || throw(ArgumentError("unsupported pit restart field; expected $allowed_pit"))
    ts = t_salt isa AbstractVector ? t_salt : fill(t_salt, n_pieces(sys))
    blocks = [initial_state(pc, p[play[(pc.id, :theta)]]; ics=ics[i], t_salt=ts[i]) for (i, pc) in enumerate(sys.pieces)]
    g(k, default) = haskey(pit_ics, k) ? getfield(pit_ics, k) : default
    pit_block = [g(:Tpit0, pit.Tpit0), g(:mfuel0, pit.m0), g(:ignited, p[play[(:pit, :phi)][7]]), g(:mpan0, pit.m_pan0),
                 g(:Tpan0, pit.T_pan0), g(:Tfb0, g(:Tpit0, pit.Tpit0))]
    all(v -> isfinite(_primal(v)), pit_block) || throw(ArgumentError("the pit restart state must be finite"))
    (_primal(pit_block[1]) > 0 && _primal(pit_block[2]) >= 0 && 0 <= _primal(pit_block[3]) <= 1 &&
     _primal(pit_block[4]) >= 0 && _primal(pit_block[5]) > 0 && _primal(pit_block[6]) > 0) ||
        throw(ArgumentError("the pit restart state is outside its physical domain"))
    return vcat(blocks..., pit_block)
end

_system_pan_oob(sys::CookSystem) = (i = sys.layout.pit_range[4]; (x, p, t) -> _primal(x[i]) < -1e-6 * sys.pit.m_pan_taper)

function _uniform_control_grid(u_tknots::AbstractVector)
    isempty(u_tknots) && throw(ArgumentError("a system solve needs at least one control knot"))
    all(isfinite, u_tknots) || throw(ArgumentError("control-knot times must be finite"))
    length(u_tknots) == 1 && return (float(u_tknots[1]), 1.0)
    du = diff(collect(float.(u_tknots))); δ = first(du)
    δ > 0 && all(d -> d > 0 && abs(d - δ) <= 1e-9 * max(1.0, abs(δ)), du) ||
        throw(ArgumentError("control-knot times must be strictly increasing and uniform"))
    return float(u_tknots[1]), δ
end

"""
    system_solve_raw(sys, p, x0, u_tknots, tsave; abstol=1e-6, reltol=1e-6) -> A

The system integrated from `x0` under the parameter vector `p` (`ParamLayout(sys, length(u_tknots))`):
the state at every save time, `n × nt`. Fails loudly on an unsuccessful solve.
"""
function system_solve_raw(sys::CookSystem, p::AbstractVector, x0::AbstractVector, u_tknots::AbstractVector, tsave::AbstractVector;
                          correction=nothing, abstol=1e-6, reltol=1e-6)
    nu = length(u_tknots)
    u_t0, u_dt = _uniform_control_grid(u_tknots)
    rhs!, play = make_system_rhs(sys, u_t0, u_dt, nu; correction)
    length(p) == play.n || throw(ArgumentError("the parameter vector has $(length(p)) entries; the system with $nu knots takes $(play.n)"))
    length(x0) == sys.layout.n || throw(ArgumentError("the state has $(length(x0)) entries; the system $(sys.layout.n)"))
    Tel = promote_type(eltype(p), eltype(x0), Float64)
    prob = ODEProblem{true,SciMLBase.FullSpecialize}(rhs!, Tel.(x0), (float(tsave[1]), float(tsave[end])), p)
    # FBDF preserves the monolith's numerical parity and is normally faster, but can hit a
    # tolerance-dependent dt-epsilon failure when several moisture nodes cross a constitutive kink
    # together. QNDF is the same implicit BDF family and crosses that case robustly. Retry only an
    # unsuccessful/truncated solve so ordinary and differentiated trajectories keep the established
    # algorithm whenever it is valid.
    solve_kw = (; saveat=tsave, abstol, reltol, isoutofdomain=_system_pan_oob(sys))
    sol = solve(prob, FBDF(); solve_kw...)
    if !(SciMLBase.successful_retcode(sol) && length(sol.u) == length(tsave))
        sol = solve(prob, QNDF(); solve_kw...)
    end
    A = Array(sol)
    (SciMLBase.successful_retcode(sol) && size(A, 2) == length(tsave)) || throw(
        ErrorException("system solve failed: retcode=$(sol.retcode), saved $(size(A, 2))/$(length(tsave)) points"))
    return A
end

# --- observables ------------------------------------------------------------------------------------------

"Mass on a scale of one slab (per copy) from its dry-matter reference volume: non-fat solids, unrendered fat, retained rendered fat and water."
meat_mass_slab(W, F, A, L, cfg::MeatConfig) = RHO_DRY * A * L * ((1 - cfg.F0) + F + (1 - cfg.η_drain) * (cfg.F0 - F) + W)

"""
    piece_outputs(pc::Slab1D, Ap, θ, A_meat, ps, tsave, ev) -> NamedTuple

The slab's trajectories and terminal quality observables from its state block `Ap` (rows the piece's ODE
state, columns the save times), under the pit series `ps` (`airflow`, `mdot`, `smoke_clean`, `pv_star`,
`Tpit`) and its events: what `_monolith_outputs` reports for the meat, per piece, plus its pit-entering flux
`j_pit`, its vapor load `A n α j_pit`, its exposed-area factor and its mass on a scale (per copy).
"""
function piece_outputs(pc::Slab1D, Ap::AbstractMatrix, θ, A_m, ps::NamedTuple, tsave, ev::NamedTuple)
    cfg = pc.cfg; nx = cfg.nx; nt = length(tsave)
    Tsurf = Ap[nx, :]; Wsurf = Ap[2nx, :]
    latch_ts = clamp.(Ap[5nx + 4, :], 0.0, 1.0)
    s_ts = 1 .- (1 .- wrap_blend.(tsave, ev.t_wrap, Ref(cfg))) .* (1 .- latch_ts)
    pl_ts = pull_blend.(tsave, ev.t_pull, Ref(cfg))
    n_s, n_a, v0 = shrink_exponents(cfg)
    wfv = fv_weights(nx)
    Vmat = specific_volume.(Ap[(nx + 1):2nx, :], Ap[(3nx + 1):4nx, :], Ref(cfg))
    l_thick = (θ[4] / (nx - 1)) .* vec(wfv' * ((Vmat ./ v0) .^ n_s))
    a_ts = (vec(wfv' * Vmat) ./ ((nx - 1) * v0)) .^ n_a
    g_fac = (ps.airflow ./ cfg.g_ref) .^ θ[8]
    h_ts = θ[3] .* g_fac; km_ts = θ[2] .* g_fac
    f_aw_ts = 1 ./ (1 .+ cfg.k_aw .* max.(Ap[5nx, :], 0.0))
    # the fluxes under the resolved chamber vapor pressure: the mixing term is inside pv_star already
    fluxes = surface_fluxes_blended.(Tsurf, Wsurf, h_ts, km_ts, s_ts, pl_ts, ps.pv_star, 0.0, Ref(cfg), f_aw_ts)
    j_evap = getproperty.(fluxes, :j)
    j_pit = getproperty.(fluxes, :j_pit)
    q_meat = A_m * cfg.n_meat .* a_ts .* surface_q_pit_blended.(ps.Tpit, Tsurf, h_ts, s_ts, pl_ts, Ref(cfg))
    bark_b = Ap[5nx + 1, :]; bark_s = Ap[5nx + 2, :]; ring = Ap[5nx + 3, :]
    dep_dirty = cfg.k_dep .* wet_blend.(a_w_oswin.(Wsurf, Ref(cfg)), Ref(cfg)) .* max.(ps.mdot, 0.0) .* (1 .- ps.smoke_clean) .*
                (1 .- s_ts) .* (1 .- pl_ts) .* (1 .- bark_s)
    w_retained = fv_mean(Ap[(nx + 1):2nx, end]) / cfg.W0
    F_mean = fv_mean(Ap[(3nx + 1):4nx, end])
    R_ret = (1 - cfg.η_drain) * (cfg.F0 - F_mean)
    dts = diff(collect(float.(tsave)))
    ja = j_evap .* a_ts
    cumtrap(v) = vcat(zero(eltype(v)), cumsum((v[1:(end - 1)] .+ v[2:end]) ./ 2 .* dts))
    evap_w_cum = cumtrap(ja) ./ (RHO_DRY * θ[4])
    bitterness_cum = cumtrap(dep_dirty)
    jsp = spritz_rate.(tsave, ev.t_spritz, ev.dt_spritz, ev.m_spritz, Ref(cfg)) .* (1 .- s_ts) .* (1 .- pl_ts) .* a_ts
    spritz_w_cum = cfg.f_abs .* cumtrap(jsp) ./ (RHO_DRY * θ[4])
    evap_w, spritz_w, bitterness = evap_w_cum[end], spritz_w_cum[end], bitterness_cum[end]
    mass = [meat_mass_slab(dot(wfv, view(Ap, (nx + 1):2nx, k)) / (nx - 1), dot(wfv, view(Ap, (3nx + 1):4nx, k)) / (nx - 1), A_m, θ[4], cfg)
            for k in 1:nt]
    return (; id=pc.id, Tsurf, Tcore=Ap[1, :], bark_b, bark_s, ring, l_star=l_star_bark.(bark_b, bark_s, Ref(cfg)),
            j_evap, j_pit, q_meat, area_factor=a_ts, vapor_load=A_m * cfg.n_meat .* a_ts .* j_pit,
            C_field_final=Ap[(2nx + 1):3nx, end], w_retained, juiciness=w_retained + cfg.β_fat * R_ret / cfg.W0,
            fat_rendered=(cfg.F0 - F_mean) / cfg.F0, evap_w, spritz_w,
            drip_w=(cfg.W0 - w_retained * cfg.W0) + spritz_w - evap_w, drip_f=cfg.η_drain * (cfg.F0 - F_mean),
            evap_w_cum, spritz_w_cum, bitterness_cum,
            l_thick, T_field=Ap[1:nx, :], W_field=Ap[(nx + 1):2nx, :], C_field=Ap[(2nx + 1):3nx, :],
            F_field=Ap[(3nx + 1):4nx, :], salt_field=Ap[(4nx + 1):5nx, :], wrap_latch=latch_ts, wrap_effective=s_ts,
            bitterness, mass)
end

"""
    system_outputs(sys, A, p, u_tknots, tsave) -> NamedTuple

The system's observables from its state array: the pit series (`Tpit`, `m_fuel`, `ignited`, `mdot`, `m_pan`,
`T_pan`, `T_firebox`, `pan_evap`, `pv_base`, `pv_pit` the chamber vapor pressure the pieces saw, `q_meat` the
total heat load, `smoke_clean`, `airflow`), every piece's `piece_outputs` in `pieces` (by index; `piece[id]`
by name), the total scale mass `mass` (every copy of every piece), the state array `X` (for a restart at any
save index) and `t`.
"""
function system_outputs(sys::CookSystem, A::AbstractMatrix, p::AbstractVector, u_tknots::AbstractVector, tsave::AbstractVector;
                        history=nothing, correction=nothing)
    pit = sys.pit
    nu = length(u_tknots)
    play = ParamLayout(sys, nu)
    ix = _SysIndex(sys, play)
    hoisted = [_piece_hoisted(pc) for pc in sys.pieces]
    u_t0, u_dt = _uniform_control_grid(u_tknots)
    uk = @view p[ix.u]
    nt = length(tsave)
    corr = correction === nothing ? [(pv=0.0, q=0.0) for _ in 1:nt] : [correction(t) for t in tsave]
    inst = map(1:nt) do k
        ud = lininterp(float(tsave[k]), u_t0, u_dt, uk)
        _system_instant(sys, view(A, :, k), p, float(tsave[k]), clamp(ud, zero(ud), one(ud)), ix, hoisted;
                        pv_bias=corr[k].pv)
    end
    pr = ix.pit
    Tpit = A[pr[1], :]; m_fuel = A[pr[2], :]; ignited = A[pr[3], :]; m_pan = A[pr[4], :]; T_pan = A[pr[5], :]; T_firebox = A[pr[6], :]
    airflow = getproperty.(inst, :airflow); mdot = getproperty.(inst, :mdot)
    pv_base = getproperty.(inst, :pv_base); pv_star = getproperty.(inst, :pv_star)
    smoke_clean = burn_cleanliness.(airflow, ignited, m_fuel, Ref(pit))
    ps = (airflow=airflow, mdot=mdot, smoke_clean=smoke_clean, pv_star=pv_star, Tpit=Tpit)
    histories = history === nothing ? [(evap_w=0.0, spritz_w=0.0, bitterness=0.0) for _ in sys.pieces] : history
    length(histories) == length(sys.pieces) || throw(ArgumentError("one cumulative-history offset is required per piece"))
    raw_pieces = map(enumerate(sys.pieces)) do (i, pc)
        θ = p[ix.θ[i]]
        ev = (t_wrap=p[ix.t_wrap[i]], t_pull=p[ix.t_pull[i]], t_spritz=p[ix.t_spritz[i]], dt_spritz=p[ix.dt_spritz[i]], m_spritz=p[ix.m_spritz[i]])
        po = piece_outputs(pc, view(A, ix.x[i], :), θ, p[ix.A[i]], ps, tsave, ev)
        h = histories[i]
        ew = get(h, :evap_w, 0.0); sw = get(h, :spritz_w, 0.0); bt = get(h, :bitterness, 0.0)
        evap_w_cum = po.evap_w_cum .+ ew
        spritz_w_cum = po.spritz_w_cum .+ sw
        bitterness_cum = po.bitterness_cum .+ bt
        merge(po, (evap_w=evap_w_cum[end], spritz_w=spritz_w_cum[end], bitterness=bitterness_cum[end],
                   evap_w_cum, spritz_w_cum, bitterness_cum,
                   drip_w=(pc.cfg.W0 - po.w_retained * pc.cfg.W0) + spritz_w_cum[end] - evap_w_cum[end]))
    end
    raw_q = sum(po.q_meat for po in raw_pieces)
    pieces = map(enumerate(raw_pieces)) do (i, po)
        q_correction = map(eachindex(corr)) do k
            c = corr[k]
            if hasproperty(c, :q_piece)
                get(c.q_piece, po.id, zero(c.q))
            elseif length(raw_pieces) == 1
                c.q
            elseif abs(raw_q[k]) > eps(Float64)
                c.q * po.q_meat[k] / raw_q[k]
            else
                c.q / length(raw_pieces)
            end
        end
        merge(po, (q_meat=po.q_meat .- q_correction,))
    end
    pv_pit = pv_star
    q_meat = sum(po.q_meat for po in pieces)
    mass = sum(sys.pieces[i].cfg.n_meat .* pieces[i].mass for i in eachindex(pieces))
    vapor_correction = getproperty.(inst, :vapor_correction)
    vapor_load = sum(po.vapor_load for po in pieces) .+ vapor_correction
    return (; Tpit, m_fuel, ignited, mdot, m_pan, T_pan, T_firebox, pan_evap=getproperty.(inst, :mdot_pan), pv_base, pv_pit,
            q_meat, vapor_load, vapor_correction, smoke_clean, airflow,
            pieces=collect(pieces), piece=Dict(po.id => po for po in pieces), mass,
            X=Matrix(A), t=collect(float.(tsave)))
end

"""
    system_solve(sys, θs, As, φ, u_knots, u_tknots, tsave; events, T_trig, t_add, dt_add, ics, pit_ics, t_salt, abstol, reltol)

The coupled cook of a system: every piece's θ and exposed area, the pit's φ, the damper schedule, the events
(`CookEvent`s; a piece reads its own, the pit its fuel addition), and the restart keywords (`ics` one
NamedTuple per piece in `handoff`'s contract, `pit_ics` the pit's, `ignited` in it replacing φ[7]). Returns
`system_outputs`. Duals in any parameter differentiate through the solve.

    system_solve(sys, p, u_tknots, tsave; x0, ...)

The same from the flat parameter vector (`pack_params`) and, optionally, a full initial state `x0`.
"""
function system_solve(sys::CookSystem, θs::AbstractVector, As::AbstractVector, φ::AbstractVector, u_knots::AbstractVector,
                      u_tknots::AbstractVector, tsave::AbstractVector; events::AbstractVector{CookEvent}=CookEvent[], T_trig=Inf,
                      t_add::Real=NaN, dt_add::Real=NaN, ics::AbstractVector=[(;) for _ in sys.pieces], pit_ics::NamedTuple=(;),
                      history=nothing, correction=nothing, t_salt=NaN, abstol=1e-6, reltol=1e-6)
    length(u_knots) == length(u_tknots) || throw(ArgumentError("u_knots length $(length(u_knots)) != u_tknots length $(length(u_tknots))"))
    p, play = pack_params(sys, θs, As, φ, u_knots; events, T_trig, t_add, dt_add)
    x0 = initial_state(sys, p, play; ics, pit_ics, t_salt)
    A = system_solve_raw(sys, p, x0, u_tknots, tsave; correction, abstol, reltol)
    return system_outputs(sys, A, p, u_tknots, tsave; history, correction)
end
function system_solve(sys::CookSystem, p::AbstractVector, u_tknots::AbstractVector, tsave::AbstractVector; x0=nothing,
                      ics::AbstractVector=[(;) for _ in sys.pieces], pit_ics::NamedTuple=(;), history=nothing,
                      correction=nothing, t_salt=NaN, abstol=1e-6, reltol=1e-6)
    play = ParamLayout(sys, length(u_tknots))
    x0v = x0 === nothing ? initial_state(sys, p, play; ics, pit_ics, t_salt) : x0
    A = system_solve_raw(sys, p, x0v, u_tknots, tsave; correction, abstol, reltol)
    return system_outputs(sys, A, p, u_tknots, tsave; history, correction)
end

"""
    system_handoff(o, k) -> (pieces, pit, ignited, x)

The system's state at save index `k` of an output `o` as restart keywords: one NamedTuple per piece
(`handoff`'s meat contract with the latch, `LATCH0`), the pit's (`Tpit0`, `mfuel0`, `mpan0`, `Tpan0`, `Tfb0`),
the ignited fraction and the raw state vector.
"""
function system_handoff(o, k::Int)
    pieces = map(o.pieces) do po
        haskey(po, :bark_dist) && return (T0f=po.T_field[:, k], W0f=po.W_field[:, k], C0f=po.C_field[:, k], F0f=po.F_field[:, k],
                                          B0f=po.bark_dist[:, k], S0f=po.smoke_dist[:, k])                      # a grid piece: its bark per exposed node
        (T0f=po.T_field[:, k], W0f=po.W_field[:, k], C0f=po.C_field[:, k], F0f=po.F_field[:, k], Ssalt0f=po.salt_field[:, k],
         B0=po.bark_b[k], S0=po.bark_s[k], R0=po.ring[k], LATCH0=po.wrap_latch[k])
    end
    pit = (Tpit0=o.Tpit[k], mfuel0=o.m_fuel[k], mpan0=o.m_pan[k], Tpan0=o.T_pan[k], Tfb0=o.T_firebox[k])
    history = [(evap_w=po.evap_w_cum[k], spritz_w=po.spritz_w_cum[k], bitterness=po.bitterness_cum[k]) for po in o.pieces]
    return (pieces=collect(pieces), pit=pit, ignited=o.ignited[k], history, x=o.X[:, k])
end

"""
    system_stack(o) -> Vector

The differentiable outputs of a system solve as one vector: the pit's `Tpit`, `m_fuel`, `pv_pit`, `q_meat`
(each nt), then per piece `Tcore`, `Tsurf`, `bark_b`, `bark_s`, `l_star`, `j_evap`, `mass` (each nt) and the
scalars `w_retained`, `juiciness`, `fat_rendered`.
"""
function system_stack(o)
    parts = Any[o.Tpit, o.m_fuel, o.pv_pit, o.q_meat]
    for po in o.pieces
        push!(parts, po.Tcore, po.Tsurf, po.bark_b, po.bark_s, po.l_star, po.j_evap, po.mass, [po.w_retained, po.juiciness, po.fat_rendered])
    end
    return reduce(vcat, parts)
end

"""
    system_jacobian(f, sys, θs, As, φ, u_knots, u_tknots, tsave; kw...) -> J

Forward-mode Jacobian of `f(o)` (a vector function of the `system_solve` output) w.r.t.
`vcat(θs..., As, φ)`: one chunked ForwardDiff pass through the solve. Keywords go to `system_solve`.
"""
function system_jacobian(f, sys::CookSystem, θs, As, φ, u_knots, u_tknots, tsave; kw...)
    np = n_pieces(sys)
    nθ = [length(piece_parameter_schema(pc)) for pc in sys.pieces]
    offsets = cumsum(vcat(0, nθ))
    v0 = vcat([float.(θ) for θ in θs]..., float.(As), float.(φ))
    function g(v)
        θv = [v[(offsets[i] + 1):offsets[i + 1]] for i in 1:np]
        Av = v[(offsets[end] + 1):(offsets[end] + np)]
        φv = v[(offsets[end] + np + 1):end]
        return f(system_solve(sys, θv, Av, φv, u_knots, u_tknots, tsave; kw...))
    end
    cfgj = ForwardDiff.JacobianConfig(g, v0, ForwardDiff.Chunk{min(length(v0), 12)}())
    return ForwardDiff.jacobian(g, v0, cfgj)
end

"""
    system_jvp(sys, p, x0, u_tknots, tsave, dp, dx0; abstol, reltol) -> dS

Directional derivative of `system_stack` of the raw solve along a tangent on the parameters (`dp`) and the
initial state (`dx0`): one Dual solve. The model's JVP; a VJP is not provided (forward mode throughout).
"""
function system_jvp(sys::CookSystem, p::AbstractVector, x0::AbstractVector, u_tknots, tsave, dp, dx0; abstol=1e-6, reltol=1e-6)
    g(s) = system_stack(system_solve(sys, p .+ s .* dp, u_tknots, tsave; x0=x0 .+ s .* dx0, abstol, reltol))
    return ForwardDiff.derivative(g, 0.0)
end

"""
    chamber_vapor(sys, x, p, u_tknots, t) -> pv_pit

The chamber vapor pressure at one instant from the system state and parameters: the pit algebra, the pan
and every piece's exchange port through the shared closure, with each piece's pull-gated flux added back
(the humidity observable, identical to the trajectory's `pv_pit`).
"""
function chamber_vapor(sys::CookSystem, x::AbstractVector, p::AbstractVector, u_tknots::AbstractVector, t::Real)
    nu = length(u_tknots); play = ParamLayout(sys, nu); ix = _SysIndex(sys, play)
    u_t0, u_dt = _uniform_control_grid(u_tknots)
    ud = lininterp(float(t), u_t0, u_dt, view(p, ix.u))
    inst = _system_instant(sys, x, p, float(t), clamp(ud, zero(ud), one(ud)), ix, [_piece_hoisted(pc) for pc in sys.pieces])
    return inst.pv_star
end
