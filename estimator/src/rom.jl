# Reduced-order models as versioned artifacts (docs/multi-meat-inference-plan.md, 3-D and reduced-order models).
# The full-order model of a cut is a grid piece (`Brick3D`, `Cylinder2D`: src/meats/grid3d.jl), the offline
# oracle; the online model is the slab (`Slab1D`), the reduced order, whose state is the full order's
# through-thickness line at the core (the projection) and whose fields broadcast back along the transverse
# directions (the reconstruction). A `ReducedOrderModel` packages that pairing with the full order's
# fingerprint (the physics sources and the archetype), the geometry and archetype, the projection and
# reconstruction operators, the validation envelope (the parameter ranges of the training design) and the
# reduction-error model: per observable the bias and the spread of `ROM − FOM` along the cook, fitted on the
# design. The online backend reads the error model as extra observation noise (bias removed, spread added in
# quadrature) on the observables the reduced model predicts; the process noise of a reduced model is its own
# (`state_noise`), to be earned by the bench per ROM, never the slab's rates carried over.
#
# The training design is the user's: `train_rom` takes the pairs of systems (full, reduced) it should run and
# the parameter draws; `validate_rom` scores held-out draws by the fraction of their errors inside the model's
# bands. Included by BrisketEstimator after plan_system.jl.

export ReducedOrderModel, ROMError, train_rom, validate_rom, rom_error, save_rom, load_rom, rom_fingerprint, rom_observables,
       project_to_line, reconstruct_from_line, ROM_OBSERVABLES

"The observables a reduction-error model carries: the core and surface temperatures, the mass on a scale, the chamber vapor pressure and the heat load."
const ROM_OBSERVABLES = (:Tcore, :Tsurf, :mass, :pv_pit, :q_meat)
const ROM_ARTIFACT_SCHEMA = "BrisketDynamics.ReducedOrderModel"
const ROM_ARTIFACT_VERSION = 2

function _meat_config_fingerprint(cfg::MeatConfig)
    record = join((string(nm, "=", repr(getfield(cfg, nm))) for nm in fieldnames(MeatConfig)), "\n")
    return bytes2hex(SHA.sha1(codeunits(record)))[1:16]
end

"The bias and spread of `ROM − FOM` for one observable on a time grid (linear between its points, held beyond)."
struct ROMError
    t::Vector{Float64}
    bias::Vector{Float64}
    sd::Vector{Float64}
end
function _interp(t::AbstractVector, v::AbstractVector, tq::Real)
    tq <= t[1] && return v[1]
    tq >= t[end] && return v[end]
    k = searchsortedlast(t, tq)
    w = (tq - t[k]) / (t[k + 1] - t[k])
    return (1 - w) * v[k] + w * v[k + 1]
end
"The bias and sd of the reduction error at time `t`."
rom_error(e::ROMError, t::Real) = (bias=_interp(e.t, e.bias, t), sd=_interp(e.t, e.sd, t))

"""
    ReducedOrderModel

The artifact: the archetype (a name), the full order's description (kind, mesh dimensions, geometry), the
reduced order's (the slab's node count), the projection (the full order's line nodes), the envelope (per
latent name the minimum and maximum of the training design), the error model per observable, the state noise
the reduced model is to run with (the defaults until the bench earns them), the fingerprint the artifact was
trained under and the training's record.
"""
struct ReducedOrderModel
    archetype::String
    fom::Dict{String,Any}
    rom::Dict{String,Any}
    line::Vector{Int}
    envelope::Dict{String,Tuple{Float64,Float64}}
    errors::Dict{Symbol,ROMError}
    state_noise::Dict{String,Any}
    fingerprint::String
    training::Dict{String,Any}
end
rom_observables(r::ReducedOrderModel) = collect(keys(r.errors))
"The bias and sd of the artifact's error on `obs` at `t` (zero bias, zero sd for an observable it does not carry)."
rom_error(r::ReducedOrderModel, obs::Symbol, t::Real) = haskey(r.errors, obs) ? rom_error(r.errors[obs], t) : (bias=0.0, sd=0.0)

function _validate_rom_contents(r::ReducedOrderModel)
    isempty(r.archetype) && throw(ArgumentError("a ROM artifact needs a nonempty archetype"))
    Set(keys(r.fom)) == Set(("kind", "dims", "N", "area", "volume", "line_length", "config_fingerprint")) ||
        throw(ArgumentError("the ROM artifact has an incomplete or unknown full-order geometry schema"))
    Set(keys(r.rom)) == Set(("kind", "nx", "geom_m", "n_shrink", "config_fingerprint")) ||
        throw(ArgumentError("the ROM artifact has an incomplete or unknown reduced-model schema"))
    N = Int(r.fom["N"])
    dims = Int.(r.fom["dims"])
    N > 0 && !isempty(dims) && prod(dims) == N && all(>(0), dims) &&
        isfinite(Float64(r.fom["area"])) && Float64(r.fom["area"]) > 0 &&
        isfinite(Float64(r.fom["volume"])) && Float64(r.fom["volume"]) > 0 ||
        throw(ArgumentError("the ROM artifact has invalid full-order geometry"))
    String(r.fom["kind"]) in ("brick3d", "cylinder2d") ||
        throw(ArgumentError("the ROM artifact has unsupported full-order kind $(r.fom["kind"])"))
    !isempty(String(r.fom["config_fingerprint"])) ||
        throw(ArgumentError("the ROM artifact has no full-order configuration fingerprint"))
    String(r.rom["kind"]) == "slab1d" && Int(r.rom["nx"]) >= 2 &&
        isfinite(Float64(r.rom["geom_m"])) && 0 <= Float64(r.rom["geom_m"]) <= 2 &&
        isfinite(Float64(r.rom["n_shrink"])) && !isempty(String(r.rom["config_fingerprint"])) ||
        throw(ArgumentError("the ROM artifact has invalid reduced-model geometry or configuration"))
    !isempty(r.line) && all(i -> 1 <= i <= N, r.line) && length(unique(r.line)) == length(r.line) ||
        throw(ArgumentError("the ROM artifact has an invalid projection line"))
    haskey(r.fom, "line_length") && Int(r.fom["line_length"]) == length(r.line) ||
        throw(ArgumentError("the ROM artifact's projection line does not match its geometry record"))
    Set(keys(r.errors)) == Set(ROM_OBSERVABLES) ||
        throw(ArgumentError("the ROM artifact's reduction-error schema must contain exactly $(collect(ROM_OBSERVABLES))"))
    for (name, bounds) in r.envelope
        lo, hi = bounds
        isfinite(lo) && isfinite(hi) && lo <= hi ||
            throw(ArgumentError("the ROM envelope for $name is not a finite ordered interval"))
    end
    for (obs, e) in r.errors
        !isempty(e.t) && length(e.t) == length(e.bias) == length(e.sd) ||
            throw(ArgumentError("the ROM error series for $obs has inconsistent lengths"))
        all(isfinite, e.t) && all(diff(e.t) .> 0) && all(isfinite, e.bias) &&
            all(s -> isfinite(s) && s >= 0, e.sd) ||
            throw(ArgumentError("the ROM error series for $obs is invalid"))
    end
    first_error = r.errors[first(ROM_OBSERVABLES)]
    all(r.errors[obs].t == first_error.t for obs in ROM_OBSERVABLES) ||
        throw(ArgumentError("the ROM observables must share one error-model time grid"))
    String(get(r.training, "artifact_schema", "")) == ROM_ARTIFACT_SCHEMA &&
        Int(get(r.training, "artifact_version", 0)) == ROM_ARTIFACT_VERSION ||
        throw(ArgumentError("the ROM artifact has an unsupported in-memory schema or version"))
    schema = get(r.training, "latent_schema", nothing)
    schema isa AbstractDict && Set(String.(keys(schema))) == Set(keys(r.envelope)) ||
        throw(ArgumentError("the ROM artifact's latent schema must match its validation envelope exactly"))
    for name in keys(r.envelope)
        saved = schema[name]
        saved isa AbstractDict && Set(String.(keys(saved))) == Set(("family", "coord", "p1", "p2", "p3", "p4")) ||
            throw(ArgumentError("the ROM artifact has an invalid latent schema for $name"))
        String(saved["family"]) != "" && Symbol(saved["coord"]) in (:logit, :score) &&
            all(isfinite, Float64[saved["p1"], saved["p2"], saved["p3"], saved["p4"]]) ||
            throw(ArgumentError("the ROM artifact has invalid latent shape, units or transform for $name"))
    end
    Set(String.(keys(get(r.state_noise, "rates", Dict())))) == Set(STATE_GROUPS) &&
        Set(String.(keys(get(r.state_noise, "init_sd", Dict())))) == Set(STATE_GROUPS) &&
        Set(String.(keys(get(r.state_noise, "events", Dict())))) == Set(("T_event", "T_spritz", "m_add")) ||
        throw(ArgumentError("the ROM artifact's process-noise schema is incomplete or unknown"))
    noise_values = vcat(collect(values(r.state_noise["rates"])), collect(values(r.state_noise["init_sd"])),
                        collect(values(r.state_noise["events"])), [get(r.state_noise, "node_jitter", NaN)])
    all(v -> v isa Real && isfinite(v) && v >= 0, noise_values) &&
        Symbol(get(r.state_noise, "fields", "")) in (:offset, :nodes) ||
        throw(ArgumentError("the ROM artifact's process-noise values are invalid"))
    try
        state_noise_from_dict(r.state_noise)
    catch err
        throw(ArgumentError("the ROM artifact has invalid process noise: $(sprint(showerror, err))"))
    end
    return nothing
end

_site_schema(s::PriorSite) = Dict{String,Any}(
    "family" => s.family, "coord" => String(s.coord),
    "p1" => s.p1, "p2" => s.p2, "p3" => s.p3, "p4" => s.p4)
function _validate_rom_latents(r::ReducedOrderModel, blk::LatentBlock, name_map::Function=identity)
    schema = get(r.training, "latent_schema", nothing)
    for name in keys(r.envelope)
        attached = name_map(name)
        haskey(blk.index, attached) ||
            throw(ArgumentError("the attached latent block does not carry ROM latent $attached"))
        schema === nothing && continue
        haskey(schema, name) || throw(ArgumentError("the ROM artifact has no schema for envelope latent $name"))
        saved = schema[name]; site = blk.sites[blk.index[attached]]
        String(saved["family"]) == site.family && Symbol(saved["coord"]) == site.coord &&
            Float64(saved["p1"]) == site.p1 && Float64(saved["p2"]) == site.p2 &&
            Float64(saved["p3"]) == site.p3 && Float64(saved["p4"]) == site.p4 ||
            throw(ArgumentError("ROM latent $attached has an incompatible prior family, shape, transform or units"))
        μ, _ = z_moments(site)
        lo, hi = r.envelope[name]
        lo - 1e-9 <= μ <= hi + 1e-9 ||
            throw(ArgumentError("ROM latent $attached has prior centre $μ outside its validated coordinate envelope [$lo, $hi]"))
    end
    return nothing
end
function _validate_rom_bound_envelope(r::ReducedOrderModel, blk::LatentBlock, piece_id::Symbol, name_map::Function=identity)
    covered = Set(name_map(name) for name in keys(r.envelope))
    for (i, binding) in blk.bindings
        target = first(binding)
        target[1] in (piece_id, :pit) || continue
        blk.names[i] in covered ||
            throw(ArgumentError("ROM artifact for $piece_id has no validation envelope for dynamics latent $(blk.names[i])"))
    end
    return nothing
end

function _validate_rom_use(r::ReducedOrderModel, blk::LatentBlock, z, t0, t1, name_map::Function=identity)
    length(z) == length(blk) ||
        throw(ArgumentError("the ROM use point has $(length(z)) latents, but the attached block has $(length(blk))"))
    for (name, (lo, hi)) in r.envelope
        attached = name_map(name)
        i = latent_index(blk, attached)
        zi = _primal(z[i])
        isfinite(zi) && lo - 1e-9 <= zi <= hi + 1e-9 ||
            throw(DomainError(zi, "ROM latent $attached is outside its validated coordinate envelope [$lo, $hi]"))
    end
    e = r.errors[first(ROM_OBSERVABLES)]
    a, b = _primal(t0), _primal(t1)
    isfinite(a) && isfinite(b) && first(e.t) - 1e-9 <= a <= b <= last(e.t) + 1e-9 ||
        throw(DomainError((a, b), "ROM use interval lies outside its validated time horizon [$(first(e.t)), $(last(e.t))] s"))
    return nothing
end
function _validate_rom_use(rs::AbstractDict{Symbol,<:ReducedOrderModel}, blk::LatentBlock, z, t0, t1)
    for id in sort!(collect(keys(rs)); by=string)
        name_map = name -> haskey(blk.index, name) ? name : "$id.$name"
        _validate_rom_use(rs[id], blk, z, t0, t1, name_map)
    end
    return nothing
end

function _validate_rom_fom(r::ReducedOrderModel, pc::GridPiece)
    String(r.fom["kind"]) == string(pc.kind) && Int.(r.fom["dims"]) == collect(pc.mesh.dims) &&
        Int(r.fom["N"]) == pc.mesh.N && Int.(r.line) == pc.mesh.line &&
        isapprox(Float64(r.fom["area"]), pc.mesh.area; rtol=1e-12, atol=0.0) &&
        isapprox(Float64(r.fom["volume"]), sum(pc.mesh.vol); rtol=1e-12, atol=0.0) &&
        String(r.fom["config_fingerprint"]) == _meat_config_fingerprint(pc.cfg) ||
        throw(ArgumentError("the full-order model does not match the ROM artifact's geometry and configuration"))
    return nothing
end

function _validate_rom_attachment(r::ReducedOrderModel, sys::CookSystem, blk::LatentBlock)
    _validate_rom_contents(r)
    n_pieces(sys) == 1 || throw(ArgumentError("attach one ROM artifact only to a one-piece system; use a piece-id registry for multiple pieces"))
    pc = sys.pieces[1]
    pc isa Slab1D || throw(ArgumentError("a reduced-order artifact attaches to its Slab1D online model"))
    r.fingerprint == rom_fingerprint() || throw(ArgumentError("the ROM artifact was trained under physics $(r.fingerprint), not $(rom_fingerprint())"))
    Int(r.rom["nx"]) == pc.cfg.nx || throw(ArgumentError("the ROM artifact has $(r.rom["nx"]) nodes, the attached slab $(pc.cfg.nx)"))
    Float64(r.rom["geom_m"]) == pc.cfg.geom_m || throw(ArgumentError("the ROM artifact's geometry mode does not match the slab"))
    String(r.rom["config_fingerprint"]) == _meat_config_fingerprint(pc.cfg) ||
        throw(ArgumentError("the ROM artifact's meat configuration does not match the attached slab"))
    _validate_rom_latents(r, blk)
    _validate_rom_bound_envelope(r, blk, pc.id)
    return nothing
end
_rom_state_noise(r::ReducedOrderModel) = state_noise_from_dict(r.state_noise)
function _rom_dynamics_correction(r::ReducedOrderModel, t)
    pv = rom_error(r, :pv_pit, t).bias
    q = rom_error(r, :q_meat, t).bias
    return (pv=pv, q=q)
end

function _validate_rom_attachment(rs::AbstractDict{Symbol,<:ReducedOrderModel}, sys::CookSystem, blk::LatentBlock)
    piece_ids = Set(pc.id for pc in sys.pieces)
    artifact_ids = Set(keys(rs))
    missing = setdiff(piece_ids, artifact_ids)
    extra = setdiff(artifact_ids, piece_ids)
    isempty(missing) || throw(ArgumentError("the ROM registry is missing artifacts for $(sort!(collect(missing); by=string))"))
    isempty(extra) || throw(ArgumentError("the ROM registry has artifacts for pieces not in the system: $(sort!(collect(extra); by=string))"))
    for pc in sys.pieces
        pc isa Slab1D || throw(ArgumentError("ROM artifact $(pc.id) must attach to a Slab1D online piece"))
        r = rs[pc.id]
        _validate_rom_contents(r)
        r.fingerprint == rom_fingerprint() || throw(ArgumentError("ROM artifact $(pc.id) has a stale physics fingerprint"))
        Int(r.rom["nx"]) == pc.cfg.nx || throw(ArgumentError("ROM artifact $(pc.id)'s node count does not match its slab"))
        Float64(r.rom["geom_m"]) == pc.cfg.geom_m || throw(ArgumentError("ROM artifact $(pc.id)'s geometry mode does not match its slab"))
        String(r.rom["config_fingerprint"]) == _meat_config_fingerprint(pc.cfg) ||
            throw(ArgumentError("ROM artifact $(pc.id)'s meat configuration does not match its slab"))
        name_map = name -> haskey(blk.index, name) ? name : "$(pc.id).$name"
        _validate_rom_latents(r, blk, name_map)
        _validate_rom_bound_envelope(r, blk, pc.id, name_map)
    end
    return nothing
end
function _rom_state_noise(rs::AbstractDict{Symbol,<:ReducedOrderModel})
    ids = sort(collect(keys(rs)); by=string)
    isempty(ids) && throw(ArgumentError("a ROM registry cannot be empty"))
    out = Dict{Symbol,StateNoise}(id => state_noise_from_dict(rs[id].state_noise) for id in ids)
    pit_fields = (:T_pit, :m_fuel, :m_pan, :T_pan, :T_fb, :ignited,
                  :T_pit0, :m_fuel0, :m_pan0, :T_pan0, :T_fb0, :ignited0,
                  :T_event, :T_spritz, :m_add)
    pit_sn = out[first(ids)]
    all(id -> all(f -> getfield(out[id], f) == getfield(pit_sn, f), pit_fields), ids) ||
        throw(ArgumentError("all artifacts in a ROM registry must agree on shared-pit process noise"))
    out[:pit] = pit_sn
    return out
end
function _rom_dynamics_correction(rs::AbstractDict{Symbol,<:ReducedOrderModel}, t)
    ids = sort!(collect(keys(rs)); by=string)
    es = Dict(id => _rom_dynamics_correction(rs[id], t) for id in ids)
    q_piece = Dict(id => e.q for (id, e) in es)
    return (pv=sum(es[id].pv for id in ids; init=0.0),
            q=sum(q_piece[id] for id in ids; init=0.0), q_piece)
end
_observable_piece(obs::TemperatureAt) = obs.piece
_observable_piece(obs::SurfaceTemperature) = obs.piece
_observable_piece(obs::PieceMass) = obs.piece
_observable_piece(obs::BarkL) = obs.piece
_observable_piece(::Observable) = nothing
_rom_piece_error(r::ReducedOrderModel, piece::Symbol, key::Symbol, t) = rom_error(r, key, t)
function _rom_piece_error(rs::AbstractDict{Symbol,<:ReducedOrderModel}, piece::Symbol, key::Symbol, t)
    haskey(rs, piece) || throw(ArgumentError("the ROM registry has no artifact for piece $piece"))
    return rom_error(rs[piece], key, t)
end
function _rom_observation_error(rs::AbstractDict{Symbol,<:ReducedOrderModel}, obs, key, t)
    id = _observable_piece(obs)
    if id !== nothing
        haskey(rs, id) || return (bias=0.0, sd=0.0)
        return rom_error(rs[id], key, t)
    end
    es = [rom_error(rs[id], key, t) for id in sort!(collect(keys(rs)); by=string)]
    return (bias=sum(e.bias for e in es; init=0.0), sd=sqrt(sum(e.sd^2 for e in es; init=0.0)))
end

"""
    rom_fingerprint(; repo=REPO_ROOT) -> String

The first twelve hex digits of the SHA-1 over the physics sources the full and reduced orders run on
(`src/**/*.jl`), each as its path and bytes with carriage returns dropped: an artifact trained under another
tree is refused by `load_rom` unless `strict=false`.
"""
function rom_fingerprint(; repo::AbstractString=REPO_ROOT)
    files = String[]
    for (root, _, fs) in walkdir(joinpath(repo, "src")), f in fs
        endswith(f, ".jl") && push!(files, replace(relpath(joinpath(root, f), repo), "\\" => "/"))
    end
    sort!(files)
    ctx = SHA.SHA1_CTX()
    for f in files
        SHA.update!(ctx, codeunits(f))
        SHA.update!(ctx, filter(!=(0x0d), read(joinpath(repo, f))))
    end
    return bytes2hex(SHA.digest!(ctx))[1:12]
end

"The full order's field resampled on its core-to-surface line."
function project_to_line(pc::GridPiece, field::AbstractVector, n::Int=length(pc.mesh.line))
    length(field) == pc.mesh.N || throw(ArgumentError("the full-order field has $(length(field)) nodes, expected $(pc.mesh.N)"))
    n >= 2 || throw(ArgumentError("a projected line needs at least two nodes"))
    raw = field[pc.mesh.line]
    n == length(raw) && return collect(raw)
    mesh = pc.mesh
    axis = argmax(abs.(mesh.coords[mesh.line[end], :] .- mesh.coords[mesh.line[1], :]))
    c0 = mesh.coords[mesh.core, axis]
    xs = [abs(mesh.coords[i, axis] - c0) for i in mesh.line]
    return [_interp(xs, raw, x) for x in range(first(xs), last(xs); length=n)]
end
function project_to_line(r::ReducedOrderModel, pc::GridPiece, field::AbstractVector)
    _validate_rom_fom(r, pc)
    return project_to_line(pc, field, Int(r.rom["nx"]))
end
"""
    reconstruct_from_line(pc::GridPiece, line_values) -> field

The full order's field from the reduced order's line: every node takes the line value at its own distance
from the core along the line's axis (the transverse directions uniform), the reconstruction operator.
"""
function reconstruct_from_line(pc::GridPiece, line_values::AbstractVector)
    length(line_values) >= 2 || throw(ArgumentError("a reconstructed line needs at least two nodes"))
    mesh = pc.mesh
    axis = argmax(abs.(mesh.coords[mesh.line[end], :] .- mesh.coords[mesh.line[1], :]))
    c0 = mesh.coords[mesh.core, axis]
    dmax = abs(mesh.coords[mesh.line[end], axis] - c0)
    xs = collect(range(0.0, dmax; length=length(line_values)))
    out = zeros(promote_type(eltype(line_values), Float64), mesh.N)
    for n in 1:mesh.N
        d = abs(mesh.coords[n, axis] - c0)
        out[n] = _interp(xs, line_values, d)
    end
    return out
end
function reconstruct_from_line(r::ReducedOrderModel, pc::GridPiece, line_values::AbstractVector)
    _validate_rom_fom(r, pc)
    length(line_values) == Int(r.rom["nx"]) ||
        throw(ArgumentError("the reduced line has $(length(line_values)) nodes, expected $(r.rom["nx"])"))
    return reconstruct_from_line(pc, line_values)
end

# the observables of a system solve's first piece, on the solve's grid
function _rom_series(o)
    po = o.pieces[1]
    return Dict(:Tcore => po.Tcore, :Tsurf => po.Tsurf, :mass => o.mass, :pv_pit => o.pv_pit, :q_meat => o.q_meat)
end

function _rom_scenarios(u_knots, u_tknots, tsave, events, scenarios)
    raw = scenarios === nothing ? [(u_knots=u_knots, u_tknots=u_tknots, tsave=tsave, events=events)] : collect(scenarios)
    isempty(raw) && throw(ArgumentError("ROM training or validation needs at least one control/event scenario"))
    return map(raw) do sc
        uk = collect(Float64, get(sc, :u_knots, u_knots))
        utk = collect(Float64, get(sc, :u_tknots, u_tknots))
        ts = collect(Float64, get(sc, :tsave, tsave))
        ev = collect(CookEvent, get(sc, :events, events))
        length(uk) == length(utk) || throw(ArgumentError("a ROM scenario needs one time per control knot"))
        length(uk) >= 2 && length(ts) >= 2 && all(isfinite, uk) && all(isfinite, utk) && all(isfinite, ts) &&
            all(diff(utk) .> 0) && all(diff(ts) .> 0) ||
            throw(ArgumentError("a ROM scenario needs finite controls and finite, strictly increasing time grids"))
        du = diff(utk); δ = first(du)
        all(d -> abs(d - δ) <= 1e-9 * max(1.0, abs(δ)), du) ||
            throw(ArgumentError("a ROM scenario's control grid must be uniform"))
        first(ts) <= first(utk) && last(utk) <= last(ts) ||
            throw(ArgumentError("a ROM scenario's solve grid must cover its control grid"))
        (u_knots=uk, u_tknots=utk, tsave=ts, events=ev)
    end
end

"""
    train_rom(archetype, sys_fom, sys_rom, blk, design, u_knots, u_tknots, tsave;
              t_grid, events, scenarios, state_noise, kw...) -> ReducedOrderModel

The artifact from a training design: for every draw `z` of the latents (the block `blk` binds them to both
systems, which share a layout), the full-order system and the reduced-order system are solved under the same
damper and events and the observables' differences `ROM − FOM` are taken; on `t_grid` (the solve's grid by
default) the bias is their mean over the design and the sd their spread (at least the sd's floor of a tenth
of the largest bias, so a design of one draw still leaves a width). Optional `scenarios` supplies additional
held-in control/event regimes as named tuples with `u_knots`, `u_tknots`, `tsave` and `events`; every design
draw runs in every scenario. The envelope records each latent's range over the design. `sys_fom` holds one
grid piece, `sys_rom` one slab.
"""
function train_rom(archetype::AbstractString, sys_fom::CookSystem, sys_rom::CookSystem, blk::LatentBlock, design::AbstractVector,
                   u_knots::AbstractVector, u_tknots::AbstractVector, tsave::AbstractVector; t_grid::AbstractVector=tsave,
                   events::AbstractVector{CookEvent}=CookEvent[], scenarios=nothing,
                   state_noise::StateNoise=StateNoise(), abstol=1e-8, reltol=1e-8)
    (n_pieces(sys_fom) == 1 && sys_fom.pieces[1] isa GridPiece) || throw(ArgumentError("the full-order system must hold one grid piece"))
    (n_pieces(sys_rom) == 1 && sys_rom.pieces[1] isa Slab1D) || throw(ArgumentError("the reduced-order system must hold one slab"))
    isempty(design) && throw(ArgumentError("a training design needs at least one draw"))
    all(z -> length(z) == length(blk) && all(isfinite, z), design) ||
        throw(ArgumentError("every ROM design point must contain one finite coordinate per attached latent"))
    scens = _rom_scenarios(u_knots, u_tknots, tsave, events, scenarios)
    all(sc -> first(sc.tsave) <= first(t_grid) && last(sc.tsave) >= last(t_grid), scens) ||
        throw(ArgumentError("every ROM training scenario must cover the artifact error grid"))
    env = Dict{String,Tuple{Float64,Float64}}()
    for (i, nm) in enumerate(blk.names)
        vals = [z[i] for z in design]
        env[nm] = (minimum(vals), maximum(vals))
        μ, _ = z_moments(blk.sites[i])
        env[nm][1] - 1e-9 <= μ <= env[nm][2] + 1e-9 ||
            throw(ArgumentError("the ROM design for $nm must bracket the attached prior centre $μ"))
    end
    nruns = length(design) * length(scens)
    diffs = Dict(obs => Matrix{Float64}(undef, length(t_grid), nruns) for obs in ROM_OBSERVABLES)
    col = 0
    for z in design, sc in scens
        col += 1
        of = _rom_solve(sys_fom, blk, z, sc.u_knots, sc.u_tknots, sc.tsave; events=sc.events, abstol, reltol)
        orr = _rom_solve(sys_rom, blk, z, sc.u_knots, sc.u_tknots, sc.tsave; events=sc.events, abstol, reltol)
        sf, sr = _rom_series(of), _rom_series(orr)
        for obs in ROM_OBSERVABLES
            d = sr[obs] .- sf[obs]
            diffs[obs][:, col] = [_interp(sc.tsave, d, tq) for tq in t_grid]
        end
    end
    errors = Dict{Symbol,ROMError}()
    for obs in ROM_OBSERVABLES
        D = diffs[obs]
        bias = vec(sum(D; dims=2) ./ size(D, 2))
        sd = size(D, 2) > 1 ? vec(sqrt.(sum((D .- bias) .^ 2; dims=2) ./ (size(D, 2) - 1))) : zeros(length(bias))
        floor = 0.1 * maximum(abs.(bias); init=0.0)
        errors[obs] = ROMError(collect(float.(t_grid)), bias, max.(sd, floor))
    end
    pf = sys_fom.pieces[1]; pr = sys_rom.pieces[1]
    fom = Dict{String,Any}("kind" => string(pf.kind), "dims" => collect(pf.mesh.dims), "N" => pf.mesh.N, "area" => pf.mesh.area,
                           "volume" => sum(pf.mesh.vol), "line_length" => length(pf.mesh.line),
                           "config_fingerprint" => _meat_config_fingerprint(pf.cfg))
    rom = Dict{String,Any}("kind" => "slab1d", "nx" => pr.cfg.nx, "geom_m" => pr.cfg.geom_m,
                           "n_shrink" => pr.cfg.n_shrink, "config_fingerprint" => _meat_config_fingerprint(pr.cfg))
    schema = Dict{String,Any}(nm => _site_schema(blk.sites[i]) for (i, nm) in enumerate(blk.names))
    return ReducedOrderModel(String(archetype), fom, rom, copy(pf.mesh.line), env, errors, state_noise_dict(state_noise), rom_fingerprint(),
                             Dict{String,Any}("n_design" => length(design), "t_end_s" => float(tsave[end]), "n_t_grid" => length(t_grid),
                                              "n_scenarios" => length(scens), "events" => sum(length(sc.events) for sc in scens),
                                              "latent_schema" => schema, "trained" => string(Dates_now()),
                                              "artifact_schema" => ROM_ARTIFACT_SCHEMA,
                                              "artifact_version" => ROM_ARTIFACT_VERSION))
end
Dates_now() = Base.Libc.strftime("%Y-%m-%dT%H:%M:%S", time())
function _rom_solve(sys::CookSystem, blk::LatentBlock, z, u_knots, u_tknots, tsave; events, abstol, reltol)
    θs = [zeros(length(piece_parameter_schema(pc))) for pc in sys.pieces]; As = [pc.cfg.A_meat for pc in sys.pieces]
    p0, play = pack_params(sys, θs, As, vcat(zeros(6), 0.5), u_knots; events)
    p = apply_latents(blk, z, p0, play)
    return system_solve(sys, p, u_tknots, tsave; abstol, reltol)
end

"""
    validate_rom(rom, sys_fom, sys_rom, blk, heldout, u_knots, u_tknots, tsave;
                 events, scenarios, width=2.0) -> Dict obs => (coverage, rms, raw_rms, max)

Held-out draws scored against the error model: per observable the fraction of `(ROM − FOM − bias)` within
`width` sds, the corrected root-mean-square, the uncorrected `raw_rms`, and the largest corrected absolute
error over the draws and the solve's grid; and whether every held-out draw sits inside the envelope.
`scenarios` evaluates controls/events not used by the fit with the same named-tuple contract as `train_rom`.
"""
function validate_rom(rom::ReducedOrderModel, sys_fom::CookSystem, sys_rom::CookSystem, blk::LatentBlock, heldout::AbstractVector,
                      u_knots::AbstractVector, u_tknots::AbstractVector, tsave::AbstractVector; events::AbstractVector{CookEvent}=CookEvent[],
                      scenarios=nothing, width::Float64=2.0, abstol=1e-8, reltol=1e-8)
    isempty(heldout) && throw(ArgumentError("ROM validation needs at least one held-out draw"))
    all(z -> length(z) == length(blk) && all(isfinite, z), heldout) ||
        throw(ArgumentError("every held-out point must contain one finite coordinate per attached latent"))
    _validate_rom_attachment(rom, sys_rom, blk)
    (n_pieces(sys_fom) == 1 && sys_fom.pieces[1] isa GridPiece) ||
        throw(ArgumentError("ROM validation needs the artifact's one-piece full-order grid system"))
    pf = sys_fom.pieces[1]
    _validate_rom_fom(rom, pf)
    out = Dict{Symbol,NamedTuple}()
    inside = Dict{Symbol,Vector{Bool}}(obs => Bool[] for obs in ROM_OBSERVABLES)
    errs = Dict{Symbol,Vector{Float64}}(obs => Float64[] for obs in ROM_OBSERVABLES)
    raw_errs = Dict{Symbol,Vector{Float64}}(obs => Float64[] for obs in ROM_OBSERVABLES)
    scens = _rom_scenarios(u_knots, u_tknots, tsave, events, scenarios)
    for z in heldout, sc in scens
        of = _rom_solve(sys_fom, blk, z, sc.u_knots, sc.u_tknots, sc.tsave; events=sc.events, abstol, reltol)
        orr = _rom_solve(sys_rom, blk, z, sc.u_knots, sc.u_tknots, sc.tsave; events=sc.events, abstol, reltol)
        sf, sr = _rom_series(of), _rom_series(orr)
        for obs in ROM_OBSERVABLES, (k, tq) in enumerate(sc.tsave)
            e = rom_error(rom, obs, tq)
            raw = sr[obs][k] - sf[obs][k]
            d = raw - e.bias
            push!(raw_errs[obs], raw)
            push!(errs[obs], d); push!(inside[obs], abs(d) <= width * max(e.sd, 1e-12))
        end
    end
    for obs in ROM_OBSERVABLES
        out[obs] = (coverage=sum(inside[obs]) / max(length(inside[obs]), 1),
                    rms=sqrt(sum(abs2, errs[obs]) / max(length(errs[obs]), 1)),
                    raw_rms=sqrt(sum(abs2, raw_errs[obs]) / max(length(raw_errs[obs]), 1)),
                    max=maximum(abs.(errs[obs]); init=0.0))
    end
    in_env = all(all(rom.envelope[nm][1] - 1e-9 <= z[i] <= rom.envelope[nm][2] + 1e-9 for (i, nm) in enumerate(blk.names)) for z in heldout)
    return merge(out, Dict(:in_envelope => (coverage=in_env ? 1.0 : 0.0, rms=NaN, raw_rms=NaN, max=NaN)))
end

"The artifact to JSON (and back), the fingerprint with it."
function save_rom(path::AbstractString, r::ReducedOrderModel)
    _validate_rom_contents(r)
    d = Dict("schema" => ROM_ARTIFACT_SCHEMA, "version" => ROM_ARTIFACT_VERSION,
             "archetype" => r.archetype, "fom" => r.fom, "rom" => r.rom, "line" => r.line,
             "envelope" => Dict(k => [v[1], v[2]] for (k, v) in r.envelope),
             "errors" => Dict(string(k) => Dict("t" => e.t, "bias" => e.bias, "sd" => e.sd) for (k, e) in r.errors),
             "state_noise" => r.state_noise, "fingerprint" => r.fingerprint, "training" => r.training)
    open(path, "w") do io
        JSON.print(io, d)
    end
    return path
end
"""
    load_rom(path; strict=true) -> ReducedOrderModel

The artifact from its file; under `strict` a fingerprint that is not the tree's is an error (a reduced model
trained under other physics is not a result).
"""
function load_rom(path::AbstractString; strict::Bool=true)
    d = JSON.parsefile(path)
    required = Set(("schema", "version", "archetype", "fom", "rom", "line", "envelope", "errors",
                    "state_noise", "fingerprint", "training"))
    Set(keys(d)) == required ||
        throw(ArgumentError("the reduced-order artifact in $path has missing or unknown top-level fields"))
    String(d["schema"]) == ROM_ARTIFACT_SCHEMA ||
        throw(ArgumentError("the reduced-order artifact in $path has unsupported schema $(d["schema"])"))
    d["version"] isa Integer && Int(d["version"]) == ROM_ARTIFACT_VERSION ||
        throw(ArgumentError("the reduced-order artifact in $path has unsupported version $(d["version"])"))
    fp = string(d["fingerprint"]); here = rom_fingerprint()
    fp == here || (strict ? error("the reduced-order model in $path was trained under physics $fp; the tree's is $here: retrain it (strict=false to read it anyway)") :
                            @warn "the reduced-order model in $path was trained under physics $fp; the tree's is $here")
    errors = Dict{Symbol,ROMError}(Symbol(k) => ROMError(Float64.(v["t"]), Float64.(v["bias"]), Float64.(v["sd"])) for (k, v) in d["errors"])
    env = Dict{String,Tuple{Float64,Float64}}(String(k) => (Float64(v[1]), Float64(v[2])) for (k, v) in d["envelope"])
    r = ReducedOrderModel(String(d["archetype"]), Dict{String,Any}(d["fom"]), Dict{String,Any}(d["rom"]), Int.(d["line"]), env, errors,
                          Dict{String,Any}(d["state_noise"]), fp, Dict{String,Any}(d["training"]))
    _validate_rom_contents(r)
    return r
end

# the reduction-error key of a semantic observable (nothing for one the artifact does not carry)
_rom_key(o::TemperatureAt) = o.depth_latent === :none && isfinite(o.s) && abs(o.s) <= 1e-12 ? :Tcore : nothing
_rom_key(::SurfaceTemperature) = :Tsurf
_rom_key(::TotalMass) = :mass
_rom_key(::PieceMass) = :mass
_rom_key(::ChamberVapor) = :pv_pit
_rom_key(::Observable) = nothing

function _require_rom_observable_error(obs::Observable)
    if obs isa WetBulbTemperature
        throw(ObservableUnavailable("a ROM artifact carries chamber-vapor, not wet-bulb, reduction error; calibrate the psychrometric transform before attaching this sensor"))
    elseif obs isa BarkL
        throw(ObservableUnavailable("the ROM artifact has no bark-lightness reduction-error model"))
    elseif obs isa TemperatureAt && _rom_key(obs) === nothing
        throw(ObservableUnavailable("the ROM artifact validates temperature only at the core and exposed surface, not this probe location"))
    end
    return nothing
end
_validate_rom_observable(::ReducedOrderModel, obs::Observable) = _require_rom_observable_error(obs)
_validate_rom_observable(::AbstractDict{Symbol,<:ReducedOrderModel}, obs::Observable) = _require_rom_observable_error(obs)
