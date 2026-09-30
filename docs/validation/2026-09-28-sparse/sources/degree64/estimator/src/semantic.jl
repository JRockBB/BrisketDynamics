# The semantic sensor layer and the immutable cook log (docs/multi-meat-inference-plan.md, sensor-suite
# agnosticism): a sensor is a specification (what it reads, of which entity, with what noise), a measurement is
# an immutable record (which sensor, when, what value), and an observable is addressed by entity and meaning
# (`TemperatureAt(:flat, ...)`, the chamber's vapor pressure, the aggregate scale mass), compiled against a
# `CookSystem` into a function of the belief's state and the latents, failing explicitly when the system
# cannot supply it. The censoring and noise likelihoods of channels.jl are reused (the survival site, the
# OU discrepancy); nothing here needs a common mask or a time grid: a sensor samples when it samples.
#
# The latents of a system: every piece's θ sites and exposed area, the pit's φ sites, and the estimator's own
# sites (a probe's insertion-depth fraction), declared once in a `LatentBlock` that binds each site to the
# entry of the system's parameter vector it sets. Included by BrisketEstimator after system_state.jl.

export Observable, TemperatureAt, SurfaceTemperature, PitTemperature, ChamberVapor, WetBulbTemperature, PieceMass, TotalMass,
       FuelMass, BarkL, LatentValue, NoiseModel, SensorSpec, MeasurementRecord, CookLog, with_measurements, sensor, measurements_of,
       with_execution, ObservableUnavailable, compile_observable, LatentBlock, system_latents, apply_latents, latent_index, damper_knots,
       censored_at, sensor_events, LatentKey, latent_key

abstract type Observable end
"A temperature probe in `piece`: at fraction `s` from centre to surface, at an inferred insertion depth, or at a physical grid location `(x,y,z)`/`(r,z,0)`."
struct TemperatureAt <: Observable
    piece::Symbol
    s::Float64
    depth_latent::Symbol
    location::Union{Nothing,NTuple{3,Float64}}
end
function TemperatureAt(piece::Symbol; s::Real=NaN, depth_latent::Symbol=:none, location=nothing)
    sf = Float64(s)
    loc = location === nothing ? nothing : Tuple(Float64.(location))
    loc === nothing || (length(loc) == 3 && all(isfinite, loc)) ||
        throw(ArgumentError("a grid probe location must have three finite coordinates"))
    modes = Int(isfinite(sf)) + Int(depth_latent !== :none) + Int(loc !== nothing)
    modes <= 1 || throw(ArgumentError("a temperature probe accepts only one of s, depth_latent, or location"))
    isfinite(sf) && !(0.0 <= sf <= 1.0) &&
        throw(ArgumentError("a temperature probe fraction must lie in [0, 1]"))
    return TemperatureAt(piece, sf, depth_latent, loc)
end
TemperatureAt(piece::Symbol, s::Real, depth_latent::Symbol) =
    TemperatureAt(piece; s, depth_latent)
"The exposed surface's temperature of a piece (an IR reading)."
struct SurfaceTemperature <: Observable; piece::Symbol; end
"The chamber's gas temperature."
struct PitTemperature <: Observable end
"The chamber's vapor pressure [Pa] (a humidity sensor)."
struct ChamberVapor <: Observable end
"A wetted thermocouple in the chamber. With no explicit pressure, compilation uses the system humidity model's atmospheric pressure."
struct WetBulbTemperature <: Observable
    p_atm::Union{Nothing,Float64}
    a_psy::Float64
    function WetBulbTemperature(p_atm::Union{Nothing,Real}, a_psy::Real)
        p_atm === nothing || (isfinite(p_atm) && p_atm > 0) ||
            throw(ArgumentError("wet-bulb atmospheric pressure must be finite and positive"))
        isfinite(a_psy) && a_psy > 0 ||
            throw(ArgumentError("wet-bulb psychrometric coefficient must be finite and positive"))
        return new(p_atm === nothing ? nothing : Float64(p_atm), Float64(a_psy))
    end
end
WetBulbTemperature(; p_atm::Union{Nothing,Real}=nothing, a_psy::Real=6.62e-4) =
    WetBulbTemperature(p_atm, a_psy)
"The mass on a scale of every copy of piece `piece` [kg]."
struct PieceMass <: Observable; piece::Symbol; end
"The mass on a scale of everything in the pit [kg]: every copy of every piece."
struct TotalMass <: Observable end
"The fuel mass on a scale under the firebox [kg]."
struct FuelMass <: Observable end
"The bark lightness L* of a piece (a colorimeter)."
struct BarkL <: Observable; piece::Symbol; end
"A direct reading of one latent's constrained value (a ruler on a thickness, a measured depth)."
struct LatentValue <: Observable; latent::Symbol; end

"""
    NoiseModel(; white_sd, ar_sd=0.0, tau=Inf, cap=Inf)

A sensor's error model: the instrument's white floor, an OU discrepancy with stationary sd `ar_sd` and
correlation time `tau` [s] (none when `ar_sd = 0`), and a saturation cap (a reading at the cap says only that
the latent reading was at least that; `Inf` for none).
"""
struct NoiseModel
    white_sd::Float64
    ar_sd::Float64
    tau::Float64
    cap::Float64
    function NoiseModel(white_sd, ar_sd, tau, cap)
        white_sd > 0 || throw(ArgumentError("a sensor's white floor must be positive"))
        ar_sd >= 0 || throw(ArgumentError("the discrepancy's sd cannot be negative"))
        (ar_sd == 0 || tau > 0) || throw(ArgumentError("an OU discrepancy needs a positive correlation time"))
        return new(Float64(white_sd), Float64(ar_sd), Float64(tau), Float64(cap))
    end
end
NoiseModel(; white_sd::Real, ar_sd::Real=0.0, tau::Real=Inf, cap::Real=Inf) = NoiseModel(white_sd, ar_sd, tau, cap)
has_discrepancy(nm::NoiseModel) = nm.ar_sd > 0
"The OU correlation over `Δt` seconds and the innovation sd it leaves."
ou_step(nm::NoiseModel, Δt::Real) = (ρ = has_discrepancy(nm) ? exp(-max(Δt, 0.0) / nm.tau) : 0.0; (ρ, nm.ar_sd * sqrt(max(1 - ρ^2, 1e-12))))

"A sensor: its id, what it observes and its noise."
struct SensorSpec
    id::Symbol
    observable::Observable
    noise::NoiseModel
end
"One reading: which sensor, when [s], what value (a capped sensor's reading at its cap is censored)."
struct MeasurementRecord
    sensor::Symbol
    t::Float64
    value::Float64
end
MeasurementRecord(sensor::Symbol, t::Real, value::Real) = MeasurementRecord(sensor, Float64(t), Float64(value))
"Whether a record of sensor `sp` is censored (at or above the cap)."
censored_at(sp::SensorSpec, r::MeasurementRecord) = isfinite(sp.noise.cap) && r.value >= sp.noise.cap - 1e-9

"""
    CookLog(t_grid, u, events, sensors, measurements)

The immutable record of a cook: the executed damper `u` on the uniform grid `t_grid` [s], the executed
events (`CookEvent`s), the sensors present and the measurements in arrival order. Nothing here is mutated:
`with_measurements` returns a new log with readings appended, `measurements_of` a sensor's readings.
"""
struct ImmutableVector{T} <: AbstractVector{T}
    data::Tuple{Vararg{T}}
end
ImmutableVector(::Type{T}, xs) where {T} = ImmutableVector{T}(Tuple(collect(T, xs)))
Base.IndexStyle(::Type{<:ImmutableVector}) = IndexLinear()
Base.size(v::ImmutableVector) = (length(v.data),)
Base.length(v::ImmutableVector) = length(v.data)
Base.getindex(v::ImmutableVector, i::Int) = v.data[i]
Base.getindex(v::ImmutableVector, r::AbstractVector{<:Integer}) = [v.data[i] for i in r]
Base.iterate(v::ImmutableVector, st...) = iterate(v.data, st...)

struct CookLog
    t_grid::ImmutableVector{Float64}
    u::ImmutableVector{Float64}
    events::ImmutableVector{CookEvent}
    sensors::ImmutableVector{SensorSpec}
    measurements::ImmutableVector{MeasurementRecord}
    function CookLog(t_grid, u, events, sensors, measurements)
        length(t_grid) == length(u) || throw(ArgumentError("the damper needs one value per grid point ($(length(u)) for $(length(t_grid)))"))
        length(t_grid) >= 2 || throw(ArgumentError("the damper grid needs at least two points"))
        all(isfinite, t_grid) && all(diff(collect(float.(t_grid))) .> 0) || throw(ArgumentError("the damper grid must be finite and strictly increasing"))
        dt = float(t_grid[2] - t_grid[1])
        all(abs((t_grid[i + 1] - t_grid[i]) - dt) <= 1e-9 * max(1.0, abs(dt)) for i in 2:(length(t_grid) - 1)) ||
            throw(ArgumentError("the damper grid must be uniform"))
        all(x -> isfinite(x) || isnan(x), u) || throw(ArgumentError("damper values must be finite or NaN beyond execution"))
        ids = [s.id for s in sensors]
        length(unique(ids)) == length(ids) || throw(ArgumentError("sensor ids must be distinct"))
        for m in measurements
            m.sensor in ids || throw(ArgumentError("a measurement of an unknown sensor $(m.sensor)"))
            isfinite(m.t) && first(t_grid) <= m.t <= last(t_grid) ||
                throw(ArgumentError("a measurement of $(m.sensor) at $(m.t) s is outside the cook grid"))
            isfinite(m.value) || throw(ArgumentError("a measurement of $(m.sensor) at $(m.t) s is not finite"))
        end
        return new(ImmutableVector(Float64, t_grid), ImmutableVector(Float64, u), ImmutableVector(CookEvent, events),
                   ImmutableVector(SensorSpec, sensors), ImmutableVector(MeasurementRecord, measurements))
    end
end
CookLog(t_grid, u; events=CookEvent[], sensors=SensorSpec[], measurements=MeasurementRecord[]) = CookLog(t_grid, u, events, sensors, measurements)
with_measurements(log::CookLog, ms::AbstractVector{MeasurementRecord}) =
    CookLog(log.t_grid, log.u, log.events, log.sensors, [log.measurements...; ms...])
with_measurements(log::CookLog, m::MeasurementRecord) = with_measurements(log, [m])
"Return a log with updated executed controls/events; prior immutable log instances remain valid for old factors."
with_execution(log::CookLog; u=log.u, events=log.events) =
    CookLog(log.t_grid, u, events, log.sensors, log.measurements)
function sensor(log::CookLog, id::Symbol)
    i = findfirst(s -> s.id === id, log.sensors)
    i === nothing && throw(ArgumentError("no sensor $id in the log (the sensors are $(join([s.id for s in log.sensors], ", ")))"))
    return log.sensors[i]
end
measurements_of(log::CookLog, id::Symbol) = filter(m -> m.sensor === id, log.measurements)
"The lid-open events of the log inside `(t0, t1]` by kind (the add, the wraps, the pulls, every spritz pulse of every train to its piece's wrap or pull)."
function sensor_events(log::CookLog, t0::Real, t1::Real)
    out = Symbol[]
    for e in log.events
        e.kind === :spritz && continue
        t0 < e.t <= t1 && push!(out, e.kind)
    end
    spritz_times = Float64[]
    for e in log.events
        e.kind === :spritz || continue
        stop = min(t1, minimum(vcat([x.t for x in log.events if x.target === e.target && x.kind in (:wrap, :pull)], Inf)))
        dt = Float64(get(e.params, :dt, 3600.0))
        isfinite(dt) && dt > 0 || throw(ArgumentError("a spritz interval must be finite and positive"))
        ts = e.t
        while ts <= stop
            t0 < ts <= t1 && push!(spritz_times, ts)
            ts += dt
        end
    end
    sort!(spritz_times)
    unique!(spritz_times)
    append!(out, fill(:spritz, length(spritz_times))) # coincident pulses require one physical lid opening
    return out
end

"""
    damper_knots(log, t0, t1) -> (u_tknots, u_knots)

The damper over `[t0, t1]` on a uniform knot grid no coarser than the log's, aligned to `t0`: the log's own
points when `t0` and `t1` are on its grid, the log's damper interpolated onto the slice's grid otherwise
(a piecewise-linear damper resampled between its knots moves by less than the pit can answer in a step).
"""
function damper_knots(log::CookLog, t0::Real, t1::Real)
    t1 > t0 || throw(ArgumentError("a slice needs t1 > t0 (got $t0, $t1)"))
    tg = log.t_grid; dt = tg[2] - tg[1]; n = length(tg)
    (t0 >= tg[1] - 1e-9 && t1 <= tg[end] + 1e-9) || throw(ArgumentError("the slice [$t0, $t1] leaves the log's grid [$(tg[1]), $(tg[end])]"))
    # the log's points covering the slice (a filled-ahead log may hold NaN beyond the executed part: only the covering points are read)
    i0 = clamp(floor(Int, (t0 - tg[1]) / dt + 1e-9) + 1, 1, n - 1)
    i1 = clamp(ceil(Int, (t1 - tg[1]) / dt - 1e-9) + 1, i0 + 1, n)
    useg = log.u[i0:i1]
    any(isnan, useg) && throw(ArgumentError("the damper is not set on the whole slice [$t0, $t1] (NaN in the log there)"))
    nk = max(2, Int(ceil((t1 - t0) / dt - 1e-9)) + 1)
    tk = collect(range(Float64(t0), Float64(t1); length=nk))
    uk = [lininterp(τ, tg[i0], dt, useg) for τ in tk]
    return tk, uk
end

# --- the latents of a system ---------------------------------------------------------------------------------

"A latent's semantic identity, independent of its display name or position in a cook-specific vector."
struct LatentKey
    scope::Symbol
    entity::Symbol
    parameter::Symbol
    function LatentKey(scope::Symbol, entity::Symbol, parameter::Symbol)
        scope in (:population, :family, :cooker, :cook, :piece, :sensor) ||
            throw(ArgumentError("a latent scope must be population, family, cooker, cook, piece or sensor"))
        return new(scope, entity, parameter)
    end
end
Base.show(io::IO, k::LatentKey) = print(io, k.scope, "/", k.entity, "/", k.parameter)

"""
    LatentBlock

The estimator's latents over a system, declared once: the sites (families and priors), display names, typed
semantic keys, and for
every site bound to the system's parameter vector the parameter key and offset it sets (`bindings`); the
unbound sites (a probe's depth fraction) are read by observables. `latent_index(blk, name)` finds a site.
"""
struct LatentBlock
    sites::Vector{PriorSite}
    names::Vector{String}
    keys::Vector{LatentKey}
    bindings::Dict{Int,Tuple{Tuple{Symbol,Symbol},Int}}
    index::Dict{String,Int}
    function LatentBlock(sites, names, semantic_keys, bindings, index)
        length(sites) == length(names) == length(semantic_keys) || throw(ArgumentError("a latent needs one display name and one semantic key"))
        length(unique(names)) == length(names) || throw(ArgumentError("latent names must be unique (got $(names))"))
        length(unique(semantic_keys)) == length(semantic_keys) || throw(ArgumentError("latent semantic keys must be unique (got $(semantic_keys))"))
        all(i -> 1 <= i <= length(sites), keys(bindings)) || throw(ArgumentError("a latent binding index is out of range"))
        Dict{String,Int}(index) == Dict(n => i for (i, n) in enumerate(names)) ||
            throw(ArgumentError("the latent name index does not match the names"))
        return new(collect(PriorSite, sites), collect(String, names), collect(LatentKey, semantic_keys),
                   Dict{Int,Tuple{Tuple{Symbol,Symbol},Int}}(bindings), Dict{String,Int}(index))
    end
end
Base.length(b::LatentBlock) = length(b.sites)

"Compatibility constructor for externally assembled blocks; generated system blocks always pass explicit keys."
function LatentBlock(sites, names, bindings, index)
    semantic = LatentKey[]
    for (i, name) in enumerate(names)
        parameter = Symbol(last(split(name, '.')))
        if haskey(bindings, i)
            (entity, block), _ = bindings[i]
            push!(semantic, block === :phi ? LatentKey(:cooker, entity, parameter) : LatentKey(:piece, entity, parameter))
        else
            occursin('.', name) &&
                throw(ArgumentError("an unbound dotted latent name is ambiguous; construct LatentBlock with explicit LatentKeys"))
            push!(semantic, LatentKey(:sensor, :unspecified, parameter))
        end
    end
    return LatentBlock(sites, names, semantic, bindings, index)
end

"The semantic key of latent `i`."
function latent_key(b::LatentBlock, i::Int)
    1 <= i <= length(b) || throw(BoundsError(b.sites, i))
    return b.keys[i]
end
function latent_index(b::LatentBlock, key::LatentKey)
    hits = findall(i -> latent_key(b, i) == key, 1:length(b))
    length(hits) == 1 || throw(ArgumentError(isempty(hits) ? "no latent with semantic key $key" : "semantic key $key is ambiguous"))
    return only(hits)
end

function latent_index(b::LatentBlock, name::AbstractString)
    haskey(b.index, name) || throw(ArgumentError("no latent named $name (the latents are $(join(b.names, ", ")))"))
    return b.index[name]
end
latent_index(b::LatentBlock, name::Symbol) = latent_index(b, String(name))

"""
    system_latents(sys, prior_phys, phys_names; area_sd=0.3, probes=Symbol[], depth_frac=0.8, depth_concentration=10.0)

The latent block of a system from the physics prior (the fifteen sites of a prior file, in the calibration's
order: the eight θ then the seven φ): per piece its eight θ sites named `id.D_eff` ..., bound to its θ block,
and its exposed area `id.A_meat` (lognormal about the config's, `area_sd` wide); the pit's seven φ sites
named as the file, bound to φ; then one bounded insertion-depth fraction site `id.core_depth_fraction` per
piece in `probes` (unbound; `TemperatureAt(id; depth_latent=...)` reads it). For a system of one slab the
block is `latent_block`'s, names aside, in the same order (θ, φ, area, depth).
"""
function system_latents(sys::CookSystem, prior_phys::AbstractVector{PriorSite}, phys_names::AbstractVector{<:AbstractString};
                        area_sd::Float64=0.3, probes::AbstractVector{Symbol}=Symbol[], depth_frac::Float64=0.8, depth_concentration::Float64=10.0,
                        area::Bool=true)
    length(prior_phys) == N_PHYS || throw(ArgumentError("the physics prior carries $(length(prior_phys)) sites, not $N_PHYS"))
    sites = PriorSite[]; names = String[]; keys = LatentKey[]
    bindings = Dict{Int,Tuple{Tuple{Symbol,Symbol},Int}}()
    np = n_pieces(sys)
    for pc in sys.pieces
        schema = piece_parameter_schema(pc)
        all(j -> 1 <= j <= length(schema), piece_parameter_indices(pc)) ||
            throw(ArgumentError("piece $(pc.id) declares an active parameter outside its schema"))
        for j in piece_parameter_indices(pc)
            j <= length(prior_phys) && schema[j] == Symbol(phys_names[j]) ||
                throw(ArgumentError("piece $(pc.id)'s parameter $(schema[j]) has no matching semantic prior"))
            push!(sites, prior_phys[j]); push!(names, (np == 1 ? "" : "$(pc.id).") * String(phys_names[j]))
            push!(keys, LatentKey(:piece, pc.id, Symbol(phys_names[j])))
            bindings[length(sites)] = ((pc.id, :theta), j)
        end
    end
    for j in 1:PIT_NP
        push!(sites, prior_phys[N_THETA + j]); push!(names, String(phys_names[N_THETA + j]))
        push!(keys, LatentKey(:cooker, :pit, Symbol(phys_names[N_THETA + j])))
        bindings[length(sites)] = ((:pit, :phi), j)
    end
    for pc in sys.pieces
        area || break                                                     # no area latent: every system keeps its own exposed area (a reduced model's training)
        push!(sites, PriorSite("lognormal", log(pc.cfg.A_meat), area_sd)); push!(names, (np == 1 ? "" : "$(pc.id).") * "A_meat")
        push!(keys, LatentKey(:piece, pc.id, :A_meat))
        bindings[length(sites)] = ((pc.id, :A_meat), 1)
    end
    for id in probes
        piece(sys, id)
        push!(sites, probe_depth_site(mean=depth_frac, concentration=depth_concentration))
        push!(names, (np == 1 ? "" : "$id.") * "core_depth_fraction")
        push!(keys, LatentKey(:sensor, id, :core_depth_fraction))
    end
    length(unique(probes)) == length(probes) || throw(ArgumentError("a piece can have only one unnamed probe-depth latent; use sensor-scoped latent names for multiple probes"))
    return LatentBlock(sites, names, keys, bindings, Dict(n => i for (i, n) in enumerate(names)))
end

"""
    apply_latents(blk, z, p, play) -> p

The system's parameter vector with every bound site's constrained value written in (`from_z`): the physics
parameters the belief carries, over the events and damper `p` already holds. Duals in `z` flow through.
"""
function apply_latents(blk::LatentBlock, z::AbstractVector, p::AbstractVector, play::ParamLayout)
    length(z) == length(blk) || throw(ArgumentError("the latent vector has $(length(z)) entries; the block $(length(blk))"))
    q = convert(Vector{promote_type(eltype(z), eltype(p))}, copy(p))
    for (i, (key, off)) in blk.bindings
        q[play[key][off]] = from_z(z[i], blk.sites[i])
    end
    return q
end

# --- observables compiled against a system ---------------------------------------------------------------------

"An observable the system cannot supply (no such piece, a quantity its model does not carry)."
struct ObservableUnavailable <: Exception
    msg::String
end
Base.showerror(io::IO, e::ObservableUnavailable) = print(io, "ObservableUnavailable: ", e.msg)

# the temperature at a fraction s of the half-thickness from the centre of a slab's field: a cubic Hermite interpolant
# between nodes with central-difference tangents (one-sided at the ends), continuous in its slope. The channel
# registry's probe interpolates linearly; its reading has a kink at every node, and a smoother whose depth latent
# crosses nodes as it moves stalls on those kinks (the two-piece bench of 2026-09-25: steps of a tenth rejected in a
# three-iteration cycle). On a uniform field the two agree; elsewhere they differ by the field's curvature over a cell.
function _slab_temperature_at(T::AbstractVector, s)
    nx = length(T)
    sc = clamp(s, zero(s), one(s))
    pos = 1 + sc * (nx - 1)
    i = clamp(floor(Int, _primal(pos)), 1, nx - 1)
    u = pos - i
    m0 = i == 1 ? T[2] - T[1] : (T[i + 1] - T[i - 1]) / 2
    m1 = i + 1 == nx ? T[nx] - T[nx - 1] : (T[i + 2] - T[i]) / 2
    h00 = (1 + 2u) * (1 - u)^2; h10 = u * (1 - u)^2; h01 = u^2 * (3 - 2u); h11 = u^2 * (u - 1)
    return h00 * T[i] + h10 * m0 + h01 * T[i + 1] + h11 * m1
end

function _grid_axis_weights(q::Float64, extent::Float64, n::Int, label::AbstractString)
    -1e-12 <= q <= extent + 1e-12 ||
        throw(ObservableUnavailable("grid probe coordinate $label=$q lies outside [0, $extent]"))
    n == 1 && return ((1, 1.0),)
    pos = clamp(q / extent, 0.0, 1.0) * (n - 1) + 1
    i = min(floor(Int, pos), n - 1)
    w = pos - i
    return ((i, 1 - w), (i + 1, w))
end

"Node indices and multilinear weights for a physical probe location."
function _grid_temperature_weights(pc::GridPiece, loc::NTuple{3,Float64})
    mesh = pc.mesh
    if pc.kind === :brick3d
        wx = _grid_axis_weights(loc[1], mesh.extents[1], mesh.dims[1], "x")
        wy = _grid_axis_weights(loc[2], mesh.extents[2], mesh.dims[2], "y")
        wz = _grid_axis_weights(loc[3], mesh.extents[3], mesh.dims[3], "z")
        nx, ny, _ = mesh.dims
        idx(i, j, k) = i + (j - 1) * nx + (k - 1) * nx * ny
        return [(idx(i, j, k), wi * wj * wk) for (i, wi) in wx for (j, wj) in wy for (k, wk) in wz if wi * wj * wk != 0]
    elseif pc.kind === :cylinder2d
        abs(loc[3]) <= 1e-12 ||
            throw(ObservableUnavailable("an axisymmetric grid probe uses coordinates (r, z, 0)"))
        wr = _grid_axis_weights(loc[1], mesh.extents[1], mesh.dims[1], "r")
        wz = _grid_axis_weights(loc[2], mesh.extents[2], mesh.dims[2], "z")
        nr = mesh.dims[1]
        idx(i, j) = i + (j - 1) * nr
        return [(idx(i, j), wi * wj) for (i, wi) in wr for (j, wj) in wz if wi * wj != 0]
    end
    throw(ObservableUnavailable("piece $(pc.id) has unsupported grid kind $(pc.kind)"))
end

"""
    compile_observable(obs, sys, bl, blk, log) -> h

The observable as a function `h(x_sys, z, p, tk, t)` of the system's physical state, the latents, the parameter
vector (with the latents applied, over the damper knots `tk`) and the time, differentiable by ForwardDiff in
`x_sys` and `z`. Throws `ObservableUnavailable` when the system cannot supply it.
"""
function compile_observable(obs::TemperatureAt, sys::CookSystem, bl::BeliefLayout, blk::LatentBlock, log::CookLog)
    haskey(sys.layout.piece_range, obs.piece) || throw(ObservableUnavailable("no piece $(obs.piece) in the system for a temperature probe"))
    pc = piece(sys, obs.piece)
    :temperature in piece_observable_schema(pc) ||
        throw(ObservableUnavailable("piece $(obs.piece) does not expose a temperature observable"))
    rT = state_index(sys.layout, obs.piece, :T)
    at(T, s) = pc isa GridPiece ? _slab_temperature_at(view(T, pc.mesh.line), s) : _slab_temperature_at(T, s)
    if obs.location !== nothing
        :temperature_location in piece_observable_schema(pc) ||
            throw(ObservableUnavailable("piece $(obs.piece) does not expose physical probe locations"))
        pc isa GridPiece || throw(ObservableUnavailable("the current physical-location adapter requires a grid piece"))
        nw = _grid_temperature_weights(pc, obs.location)
        return (x, z, p, tk, t) -> sum(w * x[first(rT) + n - 1] for (n, w) in nw)
    end
    if obs.depth_latent === :none
        isfinite(obs.s) || throw(ObservableUnavailable("a temperature probe in $(obs.piece) needs a fixed fraction or a depth latent"))
        s = obs.s
        return (x, z, p, tk, t) -> at(view(x, rT), s)
    end
    i = latent_index(blk, obs.depth_latent)
    isbeta(blk.sites[i]) || throw(ObservableUnavailable("the depth latent $(obs.depth_latent) must be a bounded beta fraction"))
    site = blk.sites[i]
    return (x, z, p, tk, t) -> at(view(x, rT), 1 - from_z(z[i], site))
end
function compile_observable(obs::SurfaceTemperature, sys::CookSystem, bl::BeliefLayout, blk::LatentBlock, log::CookLog)
    haskey(sys.layout.piece_range, obs.piece) || throw(ObservableUnavailable("no piece $(obs.piece) in the system for a surface reading"))
    pc = piece(sys, obs.piece)
    :surface_temperature in piece_observable_schema(pc) ||
        throw(ObservableUnavailable("piece $(obs.piece) does not expose a surface-temperature observable"))
    rT = state_index(sys.layout, obs.piece, :T)
    if pc isa GridPiece
        nodes = first.(pc.mesh.boundary); w = last.(pc.mesh.boundary) ./ pc.mesh.area
        return (x, z, p, tk, t) -> dot(w, view(x, rT)[nodes])
    end
    i = last(rT)
    return (x, z, p, tk, t) -> x[i]
end
function compile_observable(::PitTemperature, sys::CookSystem, bl::BeliefLayout, blk::LatentBlock, log::CookLog)
    i = first(state_index(sys.layout, :pit, :T_pit))
    return (x, z, p, tk, t) -> x[i]
end
function compile_observable(::FuelMass, sys::CookSystem, bl::BeliefLayout, blk::LatentBlock, log::CookLog)
    i = first(state_index(sys.layout, :pit, :m_fuel))
    return (x, z, p, tk, t) -> x[i]
end
function compile_observable(::ChamberVapor, sys::CookSystem, bl::BeliefLayout, blk::LatentBlock, log::CookLog)
    return (x, z, p, tk, t) -> chamber_vapor(sys, x, p, tk, t)
end
function compile_observable(obs::WetBulbTemperature, sys::CookSystem, bl::BeliefLayout, blk::LatentBlock, log::CookLog)
    iT = first(state_index(sys.layout, :pit, :T_pit))
    γ = obs.a_psy * something(obs.p_atm, sys.hum.p_atm)
    return (x, z, p, tk, t) -> _wet_bulb_value(x[iT], chamber_vapor(sys, x, p, tk, t), γ)
end
# the wet bulb by Newton (channels.jl's `wet_bulb`), with its derivative through the implicit function for Duals
function _wet_bulb_value(T, pv, γ)
    Tw, fp = wet_bulb(_primal(T), _primal(pv), γ)
    (T isa ForwardDiff.Dual || pv isa ForwardDiff.Dual) || return Tw
    return Tw + (γ / fp) * (T - _primal(T)) + (1 / fp) * (pv - _primal(pv))
end
function compile_observable(obs::PieceMass, sys::CookSystem, bl::BeliefLayout, blk::LatentBlock, log::CookLog)
    pc = piece(sys, obs.piece)
    :mass in piece_observable_schema(pc) ||
        throw(ObservableUnavailable("piece $(obs.piece) does not expose a mass observable"))
    return (x, z, p, tk, t) -> _piece_mass(pc, x, p, sys) * piece_multiplicity(pc)
end
function compile_observable(::TotalMass, sys::CookSystem, bl::BeliefLayout, blk::LatentBlock, log::CookLog)
    all(pc -> :mass in piece_observable_schema(pc), sys.pieces) ||
        throw(ObservableUnavailable("not every piece in the system exposes a mass observable"))
    return (x, z, p, tk, t) -> sum(_piece_mass(pc, x, p, sys) * piece_multiplicity(pc) for pc in sys.pieces)
end
function _piece_mass(pc::Slab1D, x, p, sys::CookSystem)
    W = view(x, state_index(sys.layout, pc.id, :W)); F = view(x, state_index(sys.layout, pc.id, :F))
    return meat_mass_slab(fv_mean(W), fv_mean(F), _param_of(p, sys, pc.id, :A_meat), _param_of(p, sys, pc.id, :theta, 4), pc.cfg)
end
function _piece_mass(pc::GridPiece, x, p, sys::CookSystem)
    W = view(x, state_index(sys.layout, pc.id, :W)); F = view(x, state_index(sys.layout, pc.id, :F))
    A = _param_of(p, sys, pc.id, :A_meat); cfg = pc.cfg
    return BrisketDynamics.RHO_DRY * A * sum(pc.mesh.vol[n] * ((1 - cfg.F0) + F[n] + (1 - cfg.η_drain) * (cfg.F0 - F[n]) + W[n]) for n in 1:pc.mesh.N)
end
# A parameter's entry by semantic key. Infer the control-knot count, then ask the generated model-owned layout.
function _param_of(p, sys::CookSystem, id::Symbol, name::Symbol, off::Int=1)
    fixed = sum(length(piece_parameter_schema(pc)) + length(PIECE_PARAM_SCALARS) for pc in sys.pieces) + PIT_NP + 2
    nu = length(p) - fixed
    nu >= 1 || throw(ArgumentError("the parameter vector has no damper knots"))
    play = ParamLayout(sys, nu)
    r = play[(id, name)]
    1 <= off <= length(r) || throw(ArgumentError("parameter offset $off is outside $(id).$(name)"))
    return p[r[off]]
end
function compile_observable(obs::BarkL, sys::CookSystem, bl::BeliefLayout, blk::LatentBlock, log::CookLog)
    pc = piece(sys, obs.piece)
    :bark in piece_observable_schema(pc) ||
        throw(ObservableUnavailable("piece $(obs.piece) does not expose a bark observable"))
    cfg = pc.cfg
    if pc isa GridPiece
        rb = state_index(sys.layout, obs.piece, :bark_b); rs = state_index(sys.layout, obs.piece, :bark_s)
        w = last.(pc.mesh.boundary) ./ pc.mesh.area
        return (x, z, p, tk, t) -> l_star_bark(dot(w, view(x, rb)), dot(w, view(x, rs)), cfg)
    end
    ib = first(state_index(sys.layout, obs.piece, :bark_b)); is = first(state_index(sys.layout, obs.piece, :bark_s))
    return (x, z, p, tk, t) -> l_star_bark(x[ib], x[is], cfg)
end
function compile_observable(obs::LatentValue, sys::CookSystem, bl::BeliefLayout, blk::LatentBlock, log::CookLog)
    i = latent_index(blk, obs.latent); site = blk.sites[i]
    return (x, z, p, tk, t) -> from_z(z[i], site)
end
compile_observable(obs::Observable, sys::CookSystem, bl::BeliefLayout, blk::LatentBlock, log::CookLog) =
    throw(ObservableUnavailable("no compilation of $(typeof(obs)) for this system"))
