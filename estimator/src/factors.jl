# The shared factor infrastructure (docs/multi-meat-inference-plan.md): a backend-neutral temporal factor
# representation. A factor touches named variables (`VarKey`s), exposes its exact loss, its whitened
# residual, its Jacobian blocks (dense, one per variable) and its provenance; a `FactorGraph` holds the
# variables with their current values and the factors, linearizes them into one least-squares system,
# takes damped Gauss-Newton steps on the exact loss, marginalizes variables in square-root information
# form (a QR with the leaving variables ordered first, the trailing triangle kept as a linear factor on the
# rest) and reports marginal covariances. The online backend (online.jl) and the offline one (offline.jl)
# are the same factors under different numerics: a dense window with marginalization, a sparse full history.
#
# Factors here: the Gaussian prior on a variable; a linear factor (a marginal prior, a linear test); the
# dynamics of a CookSystem between two nodes with the process noise of the belief's coordinates; a
# semantic observation at a sensor's own timestamp (with the survival site of a censored reading, whose
# pseudo-observation is a function of the current point, so no iteration can pair a stale one with a fresh
# factor); the OU transition of a sensor's discrepancy between its own sample times; the hierarchy tying
# a cook's parameters to a population's; the drift of a parameter between segments. Every residual is
# whitened, so the loss is ½‖r‖² unless a factor says otherwise. Included by BrisketEstimator after semantic.jl.

export VarKey, Factor, GaussianPrior, SitePriorFactor, LinearFactor, DynamicsFactor, ObservationFactor, DiscrepancyFactor, HierarchyFactor,
       FactorGraph, add_variable!, add_factor!, remove_factor!, has_variable, variable_value, set_value!, factors_of, objective,
       linearize, gauss_newton!, marginalize!, marginal_cov, joint_cov, residual, jacobians, loss, vars, provenance, ndim,
       factor_linearization

"""
    VarKey(entity, name, k)

A variable's key: `entity` what it is (`:params`, `:state`, `:disc`, `:hyper`), `name` which one (a cook's
id, a sensor's id, a population's name) and `k` its index in time (a node's number, a sample's number).
"""
struct VarKey
    entity::Symbol
    name::Symbol
    scope::Tuple{Vararg{Symbol}}
    k::Int
end
VarKey(entity::Symbol, name::Symbol, k::Int) = VarKey(entity, name, (name,), k)
VarKey(entity::Symbol, scope::Tuple{Vararg{Symbol}}, k::Int) =
    isempty(scope) ? throw(ArgumentError("a variable key's scope cannot be empty")) : VarKey(entity, last(scope), scope, k)
function Base.show(io::IO, v::VarKey)
    print(io, "(", v.entity, ":", join(string.(v.scope), "/"), "#", v.k, ")")
end

abstract type Factor end
"The keys of the variables a factor touches, in the order its Jacobian blocks come."
function vars end
"The residual's length."
function ndim end
"The whitened residual at the values `vals` (one vector per touched variable, in `vars` order)."
function residual end
"The Jacobian blocks at `vals`: one dense matrix per touched variable, in `vars` order."
function jacobians end
"The exact loss at `vals`: ½‖r‖² unless the factor's likelihood is not Gaussian."
loss(f::Factor, vals) = 0.5 * sum(abs2, residual(f, vals))
"A factor's residual and Jacobian blocks from one internally consistent linearization."
factor_linearization(f::Factor, vals) = (r=residual(f, vals), Js=jacobians(f, vals))
"Where the factor came from, for a record."
provenance(f::Factor) = string(typeof(f).name.name)

# --- the Gaussian prior and the linear factor ----------------------------------------------------------------------

"""
    GaussianPrior(key, m, R)

The prior N(m, (RᵀR)⁻¹) on one variable in square-root information form: r = R (x − m). `R` any matrix
whose Gram is the precision (the upper Cholesky factor of a precision, the inverse of a covariance's lower
square root, a diagonal of inverse sds).
"""
struct GaussianPrior <: Factor
    key::VarKey
    m::Vector{Float64}
    R::Matrix{Float64}
    source::String
end
GaussianPrior(key::VarKey, m::AbstractVector, R::AbstractMatrix; source::AbstractString="prior") =
    GaussianPrior(key, collect(Float64, m), Matrix{Float64}(R), String(source))
"A Gaussian prior from a covariance: R the inverse of its lower Cholesky factor."
function GaussianPrior(key::VarKey, m::AbstractVector, P::Symmetric; source::AbstractString="prior")
    L = cholesky(P).L
    return GaussianPrior(key, m, Matrix(inv(L)); source)
end
vars(f::GaussianPrior) = [f.key]
ndim(f::GaussianPrior) = size(f.R, 1)
residual(f::GaussianPrior, vals) = f.R * (vals[1] .- f.m)
jacobians(f::GaussianPrior, vals) = [copy(f.R)]
provenance(f::GaussianPrior) = f.source

"""
    SitePriorFactor(key, sites; indices=eachindex(sites))

The exact transformed-family priors of selected entries of a latent vector. Its residual/Jacobian are the
local score/curvature site (`J'r = ∇loss`, `J'J = ∇²loss`), while `loss` remains the exact logit-Beta,
normal-score, lognormal-coordinate or linear-coordinate negative log prior.
"""
struct SitePriorFactor <: Factor
    key::VarKey
    sites::Vector{PriorSite}
    indices::Vector{Int}
    source::String
end
SitePriorFactor(key::VarKey, sites::AbstractVector{PriorSite}; indices=collect(eachindex(sites)), source::AbstractString="site prior") =
    SitePriorFactor(key, collect(PriorSite, sites), collect(Int, indices), String(source))
vars(f::SitePriorFactor) = [f.key]
ndim(f::SitePriorFactor) = length(f.indices)
function _site_terms(f::SitePriorFactor, x)
    z = x[f.indices]
    _, g, h = prior_terms(z, f.sites[f.indices])
    q = sqrt.(max.(h, eps(Float64)))
    return g ./ q, q
end
residual(f::SitePriorFactor, vals) = _site_terms(f, vals[1])[1]
function jacobians(f::SitePriorFactor, vals)
    _, q = _site_terms(f, vals[1])
    J = zeros(length(f.indices), length(vals[1]))
    for (r, i) in enumerate(f.indices); J[r, i] = q[r]; end
    return [J]
end
loss(f::SitePriorFactor, vals) = prior_terms(vals[1][f.indices], f.sites[f.indices])[1]
provenance(f::SitePriorFactor) = f.source

"""
    LinearFactor(keys, A, b)

r = Σ_i A_i x_i − b over the variables `keys` (a marginal prior after marginalization; a linear test's
measurement). `A` one block per key.
"""
struct LinearFactor <: Factor
    keys::Vector{VarKey}
    A::Vector{Matrix{Float64}}
    b::Vector{Float64}
    source::String
end
LinearFactor(keys::AbstractVector{VarKey}, A::AbstractVector{<:AbstractMatrix}, b::AbstractVector; source::AbstractString="linear") =
    LinearFactor(collect(keys), [Matrix{Float64}(a) for a in A], collect(Float64, b), String(source))
vars(f::LinearFactor) = f.keys
ndim(f::LinearFactor) = length(f.b)
function residual(f::LinearFactor, vals)
    r = -copy(f.b)
    for (A, x) in zip(f.A, vals); r .+= A * x; end
    return r
end
jacobians(f::LinearFactor, vals) = [copy(A) for A in f.A]
provenance(f::LinearFactor) = f.source

# --- the model a dynamics factor and an observation factor read ----------------------------------------------------

"""
    SystemModel(sys, log, blk; sn=StateNoise(), abstol=1e-10, reltol=1e-8,
                floor_abstol=abstol, floor_reltol=reltol)

A CookSystem with its cook log (damper, events), its latent block and its belief layout, the process noise
the belief's state receives and the solver tolerances: what the dynamics and the observation factors need to
propagate a belief state and to read an observable from one. The tolerances are tighter than the filter's
(1e-6): a smoother evaluates the dynamics residual `y1 − f(y0)` whitened by the process noise, and in the
coordinates of a field near zero (the salt of an unsalted cut, the collagen at the start) that noise is far
below the integrator's error at 1e-6, which would make the objective rough at the level of whole nats.
`floor_abstol` and `floor_reltol` set the numerical-error allowance in the process covariance. They default
to the solve tolerances, preserving the original model. Hold them fixed when tightening `abstol`/`reltol`
to improve optimization accuracy without simultaneously narrowing the statistical model's state noise.
"""
struct SystemModel
    sys::CookSystem
    log::CookLog
    blk::LatentBlock
    bl::BeliefLayout
    sn::Any
    abstol::Float64
    reltol::Float64
    rom::Any                      # a ReducedOrderModel (rom.jl) whose reduction error the observation factors add, or nothing
    floor_abstol::Float64
    floor_reltol::Float64
end
# Preserve the positional constructor used by existing diagnostic callers.
SystemModel(sys::CookSystem, log::CookLog, blk::LatentBlock, bl::BeliefLayout, sn,
            abstol::Float64, reltol::Float64, rom) =
    SystemModel(sys, log, blk, bl, sn, abstol, reltol, rom, abstol, reltol)
_validate_rom_attachment(rom, sys, blk) = nothing
_validate_rom_use(rom, blk, z, t0, t1) = nothing
_validate_rom_observable(rom, obs) = nothing
_rom_state_noise(rom) = state_noise_from_dict(getproperty(rom, :state_noise))
_rom_dynamics_correction(rom, t) = (pv=0.0, q=0.0)
_rom_observation_error(rom, obs, key, t) = rom_error(rom, key, t)
_rom_piece_error(rom, piece::Symbol, key::Symbol, t) = (bias=0.0, sd=0.0)
_rom_dynamics_correction(rom, t, z, blk) = (pv=0.0, q=0.0)
_rom_piece_error(rom, piece::Symbol, key::Symbol, t, z, blk) = (bias=0.0, sd=0.0)
_rom_observation_bias(rom, obs, key, t, z, blk) = 0.0
function SystemModel(sys::CookSystem, log::CookLog, blk::LatentBlock; sn=nothing, abstol::Float64=1e-10, reltol::Float64=1e-8, rom=nothing,
                     floor_abstol::Float64=abstol, floor_reltol::Float64=reltol)
    all(x -> isfinite(x) && x > 0, (abstol, reltol, floor_abstol, floor_reltol)) ||
        throw(ArgumentError("integration and covariance-floor tolerances must be finite and positive"))
    validate_events(sys, collect(log.events))
    _validate_rom_attachment(rom, sys, blk)
    actual_sn = sn === nothing ? (rom === nothing ? StateNoise() : _rom_state_noise(rom)) : sn
    bl = belief_layout(sys)
    for sp in log.sensors
        _validate_rom_observable(rom, sp.observable)
        compile_observable(sp.observable, sys, bl, blk, log)
    end
    return SystemModel(sys, log, blk, bl, actual_sn, abstol, reltol, rom, floor_abstol, floor_reltol)
end
export SystemModel

"The system's parameter vector over the slice's damper knots with the latents applied."
function _slice_params(m::SystemModel, z, t0, t1)
    tk, uk = damper_knots(m.log, t0, t1)
    θs = Vector{Float64}[zeros(length(piece_parameter_schema(pc))) for pc in m.sys.pieces]; As = Float64[pc.cfg.A_meat for pc in m.sys.pieces]
    p0, play = pack_params(m.sys, θs, As, vcat(zeros(6), 0.5), uk; events=m.log.events)
    return apply_latents(m.blk, z, p0, play), play, tk
end

"The log's damper at the instant `t`: interpolated between its grid points when both are set, the last executed one held beyond it (a filled-ahead log ends in NaN)."
function _damper_at(log::CookLog, t)
    tg = log.t_grid; dt = tg[2] - tg[1]; n = length(tg)
    k = clamp(floor(Int, (t - tg[1]) / dt + 1e-9) + 1, 1, n)
    isnan(log.u[k]) && throw(ArgumentError("the damper is not set at $t s"))
    (k == n || isnan(log.u[k + 1])) && return log.u[k]
    w = (t - tg[k]) / dt
    return (1 - w) * log.u[k] + w * log.u[k + 1]
end
"The system's parameter vector at one instant `t` (an observation): the damper at `t` held on a two-knot grid around it, clipped to the log's."
function _point_params(m::SystemModel, z, t)
    tg = m.log.t_grid
    t0 = max(tg[1], t - 1.0); t1 = min(tg[end], t + 1.0)
    t1 > t0 || throw(ArgumentError("an observation at $t s falls outside the log's grid"))
    ud = _damper_at(m.log, t)
    θs = Vector{Float64}[zeros(length(piece_parameter_schema(pc))) for pc in m.sys.pieces]; As = Float64[pc.cfg.A_meat for pc in m.sys.pieces]
    p0, play = pack_params(m.sys, θs, As, vcat(zeros(6), 0.5), [ud, ud]; events=m.log.events)
    return apply_latents(m.blk, z, p0, play), play, [Float64(t0), Float64(t1)]
end

"""
    propagate(m::SystemModel, y0, z, t0, t1; first_slice=false) -> (y1, o)

The belief state at `t1` from the belief state `y0` (coordinates) at `t0` under the latents `z`: decoded,
lifted to the system's state, integrated over the slice, encoded again; `first_slice` reads
the ignited fraction from the latent a0 rather than from the state (the cook's first slice, as the filter).
Duals in `y0` and `z` flow through the solve (the factor's Jacobian). Also returns the slice's outputs.
"""
function propagate(m::SystemModel, y0::AbstractVector, z::AbstractVector, t0::Real, t1::Real; first_slice::Bool=false, output_times=nothing)
    _validate_rom_use(m.rom, m.blk, z, t0, t1)
    p, play, tk = _slice_params(m, z, t0, t1)
    x_b = belief_decode(y0, m.bl)
    x0 = system_from_belief(x_b, m.bl)
    if first_slice
        x0 = convert(Vector{promote_type(eltype(x0), eltype(p))}, x0)
        x0[first(state_index(m.sys.layout, :pit, :ignited))] = p[play[(:pit, :phi)][7]]
    end
    correction = m.rom === nothing ? nothing : (t -> _rom_dynamics_correction(m.rom, t, z, m.blk))
    save = output_times===nothing ? [Float64(t0),Float64(t1)] : collect(Float64,output_times)
    output_times===nothing || (length(save)>=2 && first(save)==t0 && last(save)==t1 && all(isfinite,save) && all(>(0),diff(save))) ||
        throw(ArgumentError("propagation output times must increase from t0 through t1"))
    A = system_solve_raw(m.sys, p, x0, tk, save; correction, abstol=m.abstol, reltol=m.reltol)
    o = system_outputs(m.sys, A, p, tk, save; correction)
    y1 = belief_encode(belief_from_system(A[:, end], m.bl), m.bl)
    return y1, o
end

"""
    propagate_states(m::SystemModel, y0, z, t0, times; first_slice=false) -> Matrix

The belief state (coordinates) at every time in `times` (increasing, after `t0`) from `y0` at `t0` under the latents
`z`: ONE solve saving at all of them, each column the propagation `propagate` would give to that time. Duals flow
through, so one forward-mode pass differentiates every column at once (a slice's readings cost one Jacobian however many
there are).
"""
function propagate_states(m::SystemModel, y0::AbstractVector, z::AbstractVector, t0::Real, times::AbstractVector; first_slice::Bool=false)
    isempty(times) && return zeros(promote_type(eltype(y0), eltype(z)), m.bl.n, 0)
    all(>(t0), times) && issorted(times) && allunique(times) || throw(ArgumentError("propagation times must increase after t0"))
    t1 = last(times)
    _validate_rom_use(m.rom, m.blk, z, t0, t1)
    p, play, tk = _slice_params(m, z, t0, t1)
    x0 = system_from_belief(belief_decode(y0, m.bl), m.bl)
    if first_slice
        x0 = convert(Vector{promote_type(eltype(x0), eltype(p))}, x0)
        x0[first(state_index(m.sys.layout, :pit, :ignited))] = p[play[(:pit, :phi)][7]]
    end
    correction = m.rom === nothing ? nothing : (t -> _rom_dynamics_correction(m.rom, t, z, m.blk))
    A = system_solve_raw(m.sys, p, x0, tk, vcat(Float64(t0), collect(Float64, times)); correction, abstol=m.abstol, reltol=m.reltol)
    return reduce(hcat, [belief_encode(belief_from_system(A[:, j + 1], m.bl), m.bl) for j in eachindex(times)])
end

"""
    LatentSplit(static, moving)

The latent vector held in two graph variables: the entries `static` in one (the same for the whole cook) and
the entries `moving` in a copy per window node (the latents that drift; online.jl ties consecutive copies by a
random-walk factor). A factor that reads the latents assembles the full vector from the two. Without a split
one variable holds every entry.
"""
struct LatentSplit
    static::Vector{Int}
    moving::Vector{Int}
    n::Int
    function LatentSplit(static::AbstractVector{Int}, moving::AbstractVector{Int})
        s = collect(Int, static); mv = collect(Int, moving)
        isempty(intersect(s, mv)) || throw(ArgumentError("a latent cannot be both static and moving"))
        n = length(s) + length(mv)
        sort(vcat(s, mv)) == 1:n || throw(ArgumentError("the static and moving latents must partition 1:n"))
        return new(s, mv, n)
    end
end
export LatentSplit
"The full latent vector from its static and moving parts."
function assemble_latents(sp::LatentSplit, zs::AbstractVector, zm::AbstractVector)
    z = zeros(promote_type(eltype(zs), eltype(zm)), sp.n)
    z[sp.static] = zs; z[sp.moving] = zm
    return z
end
"The columns of a Jacobian in the full latents split by the static and moving parts."
_split_cols(sp::LatentSplit, Jz::AbstractMatrix) = [Jz[:, sp.static], Jz[:, sp.moving]]
"The position, in the stacked `[static; moving]` vector, of every entry of the full latent vector."
function split_permutation(sp::LatentSplit)
    perm = zeros(Int, sp.n)
    for (j, i) in enumerate(sp.static); perm[i] = j; end
    for (j, i) in enumerate(sp.moving); perm[i] = length(sp.static) + j; end
    return perm
end
export assemble_latents, split_permutation

"""
    DynamicsFactor(model, key_z, key_y0, key_y1, t0, t1; first=false, events=Symbol[], key_m=nothing, split=nothing)

x_{k+1} = f(x_k, z) + w over the slice `[t0, t1]`: r = W (y1 − f(y0, z)), W the inverse lower square root
of the process covariance of the interval in the belief's coordinates at the initial prediction (the rates of
the model's `StateNoise` over the interval's hours plus the disturbances of the lid-open `events` crossed).
That covariance is fixed for the lifetime of the factor. Relinearization changes the dynamics Jacobian, not
the probability model. A state-dependent covariance would instead require its derivatives and log determinant.
The Jacobian is ForwardDiff through the solve, in chunks, over the state and the latents; the factor keeps
its last solve (the slice's outputs, for the past's integrals) and its last linearization by point. With a
`split`, the latents are `key_z` (the static entries) and `key_m` (the moving copy in force over the slice).
"""
mutable struct DynamicsFactor <: Factor
    model::SystemModel
    key_z::VarKey
    key_m::Union{Nothing,VarKey}
    split::Union{Nothing,LatentSplit}
    key_y0::VarKey
    key_y1::VarKey
    t0::Float64
    t1::Float64
    first::Bool
    events::Vector{Symbol}
    chunk::Int
    relin_tol::Float64          # the latents' move (∞-norm, their coordinates) beyond which the linearization is retaken
    relin_tol_state::Float64    # the start node's move (∞-norm, the belief's coordinates) beyond which it is retaken
    cache::Any          # (z, y0, f, o) at the last solve; fixed (W, wy); (jz, jy0, Jy, Jz) at the last linearization
    n_solves::Int
    n_jacobians::Int
    peer::Base.RefValue{Any}   # a closed slice factor over the same interval (slice_obs.jl) whose one solve and Jacobian this factor reads, or nothing
end
function DynamicsFactor(model::SystemModel, key_z::VarKey, key_y0::VarKey, key_y1::VarKey, t0::Real, t1::Real; first::Bool=false,
                        events::AbstractVector{Symbol}=Symbol[], chunk::Int=12, relin_tol::Float64=1e-3, relin_tol_state::Float64=relin_tol,
                        key_m::Union{Nothing,VarKey}=nothing, split::Union{Nothing,LatentSplit}=nothing)
    (key_m === nothing) == (split === nothing) || throw(ArgumentError("a moving latent copy needs its split, and a split its copy"))
    return DynamicsFactor(model, key_z, key_m, split, key_y0, key_y1, Float64(t0), Float64(t1), first, collect(Symbol, events), chunk, relin_tol, relin_tol_state, nothing, 0, 0, Ref{Any}(nothing))
end
vars(f::DynamicsFactor) = f.key_m === nothing ? [f.key_z, f.key_y0, f.key_y1] : [f.key_z, f.key_m, f.key_y0, f.key_y1]
ndim(f::DynamicsFactor) = f.model.bl.n
provenance(f::DynamicsFactor) = "dynamics $(f.key_y0) → $(f.key_y1) over [$(f.t0), $(f.t1)] s"
# the full latents and the two node values from a factor's values (in `vars` order)
_dyn_unpack(f::DynamicsFactor, vals) = f.key_m === nothing ? (vals[1], vals[2], vals[3]) : (assemble_latents(f.split, vals[1], vals[2]), vals[3], vals[4])
"The full latent vector a dynamics or observation factor reads from the graph's current values."
_factor_latents(f, g) = f.key_m === nothing ? g.values[f.key_z] : assemble_latents(f.split, g.values[f.key_z], g.values[f.key_m])
# the process noise in the belief's coordinates over the slice, floored at `INTEGRATOR_FLOOR` times the integrator's
# error mapped into each coordinate (a state near zero under a log-type coordinate turns the solver's absolute
# tolerance into 1e-3 of coordinate; whitened by a field's shape precision that was tenths of a nat of roughness per
# node): no residual is whitened beyond what the solver resolves
const INTEGRATOR_FLOOR = 100.0
# a physical process covariance of the slice mapped into the belief's coordinates at `y1` with the integrator's floor
# on the diagonal: what the whitener inverts (`to_coord` the coordinate map: the filter's own form for the whitener,
# the variance form for the rates' sensitivity below, Dual-capable in `Q`)
#
# Next to a physical boundary the positive kinds are boundary-damped (system_state.jl `_boundary_damping`, the map that
# draws the process), the process noise and the integrator's floor each by its own physical width: a salt or a pan at
# its floor otherwise took the coordinate cap of noise every slice (nine nats of variance, plus one of floor), and off
# the floor's guard that random walk accumulated without bound. The unit kinds stay undamped (the 2026-09-30 stall).
const WHITENER_DAMPED_KINDS = (:positive,)
function _process_cov_coord(f::DynamicsFactor, y1, Q::AbstractMatrix; to_coord=belief_cov_to_coord)
    m = f.model; bl = m.bl
    Qc = to_coord(Q, y1, bl)
    w = _boundary_damping(diag(Q), y1, bl; kinds=WHITENER_DAMPED_KINDS)
    Qc = (w .* Qc) .* w'
    x = belief_decode(y1, bl)
    d = inv.(max.(belief_decode_jacobian(y1, bl), eps(Float64)))
    err = INTEGRATOR_FLOOR .* (m.floor_abstol .+ m.floor_reltol .* abs.(x))              # physical
    wf = _boundary_damping(err .^ 2, y1, bl; kinds=WHITENER_DAMPED_KINDS)
    errc = err .* d .* wf
    for i in eachindex(errc); Qc[i, i] += errc[i]^2; end
    return Qc
end
function _whitener(f::DynamicsFactor, y1)
    m = f.model
    Δh = (f.t1 - f.t0) / 3600
    Qc = _process_cov_coord(f, y1, belief_process_cov(m.bl, m.sn, Δh; events=f.events))
    Qc = _stable_cov(Qc)
    return Matrix(inv(cholesky(Symmetric(Qc)).L))
end
# a belief noise group's name in a bench accumulator: the piece's prefix dropped when the system has one piece
# ("brisket.T" is "T", the filter's group), kept otherwise
_als_group_name(bl::BeliefLayout, g::AbstractString) = length(bl.sys.pieces) == 1 && occursin('.', g) ? String(last(rsplit(g, '.'; limit=2))) : String(g)
"""
    _process_cov_sens(f::DynamicsFactor, y1, names) -> Vector{Matrix}

Per name in `names`, the derivative of the slice's coordinate process covariance (as the factor whitens it, the
events' disturbances and the integrator's floor aside since neither moves with a rate) with respect to that
group's VARIANCE rate (the square of its `StateNoise` rate): the bench accumulator's `dQ` for the state's groups.
A name no belief group carries gets a zero matrix.
"""
function _process_cov_sens(f::DynamicsFactor, y1, names::AbstractVector{<:AbstractString})
    m = f.model; bl = m.bl
    Δh = max((f.t1 - f.t0) / 3600, 0.0)
    idx = Dict(String(nm) => i for (i, nm) in enumerate(names))
    θ0 = [(_group_rate(bl, m.sn, g, false))^2 for (g, _) in bl.groups]           # the model's own variance rates by belief group
    function cov_of(θ)
        function var_of(g)
            g in bl.inert && return zero(eltype(θ))                                      # no rate moves a state the physics does not
            i = get(idx, _als_group_name(bl, g), 0)
            return (i == 0 ? θ0[findfirst(x -> first(x) == g, bl.groups)] : θ[i]) * Δh
        end
        Q = _belief_process_var(bl, m.sn, var_of)
        # The coordinate cap is nonlinear in total variance: fixed event disturbances must be present
        # when differentiating, even though their own derivative with respect to a rate is zero.
        _add_event_disturbances!(Q, bl, m.sn, f.events)
        return vec(_process_cov_coord(f, y1, Q; to_coord=_cov_to_coord_var))
    end
    θ = zeros(length(names))
    for (g, _) in bl.groups
        i = get(idx, _als_group_name(bl, g), 0)
        i == 0 || (θ[i] = θ0[findfirst(x -> first(x) == g, bl.groups)])
    end
    J = ForwardDiff.jacobian(cov_of, θ)
    n = bl.n
    return [Matrix(Symmetric(reshape(J[:, i], n, n))) for i in eachindex(names)]
end
# the solve at (y0, z), cached by point; the Jacobian when asked, kept while the point stays within `relin_tol` of the
# one it was taken at (a node whose neighbours have settled is not re-solved nineteen times per iteration). The
# whitener W is fixed at the factor's initial prediction, including its coordinate cap and integrator floor.
# Refreshing it with the Jacobian changed the loss at an unchanged graph value, while omitting the covariance
# derivatives and normalization of a state-dependent Gaussian. Keep the reference `wy` for the calibration's dQ.
function _dyn_eval!(f::DynamicsFactor, z, y0; jac::Bool)
    p = f.peer[]
    p === nothing || return _dyn_eval_peer!(f, p, z, y0; jac)
    c = f.cache
    fresh = c !== nothing && c.z == z && c.y0 == y0
    if !fresh
        tq = time(); fy, o = propagate(f.model, y0, z, f.t0, f.t1; first_slice=f.first); _tick!(:dyn_solve, time() - tq)
        f.n_solves += 1
        W = c === nothing ? _whitener(f, fy) : c.W
        c = c === nothing ? (z=copy(z), y0=copy(y0), f=fy, W=W, wy=copy(fy), o=o, jz=nothing, jy0=nothing, Jy=nothing, Jz=nothing) :
                            (z=copy(z), y0=copy(y0), f=fy, W=W, wy=c.wy, o=o, jz=c.jz, jy0=c.jy0, Jy=c.Jy, Jz=c.Jz)
    end
    if jac && (c.Jy === nothing || norm(z .- c.jz, Inf) > f.relin_tol || norm(y0 .- c.jy0, Inf) > f.relin_tol_state)
        ny = length(y0)
        g(v) = propagate(f.model, v[1:ny], v[(ny + 1):end], f.t0, f.t1; first_slice=f.first)[1]
        tq = time()
        J = _chunked_jacobian(g, vcat(y0, z), ndim(f), f.chunk)
        _tick!(:dyn_jac, time() - tq)
        f.n_jacobians += 1
        c = (z=c.z, y0=c.y0, f=c.f, W=c.W, wy=c.wy, o=c.o, jz=copy(z), jy0=copy(y0), Jy=J[:, 1:ny], Jz=J[:, (ny + 1):end])
    end
    f.cache = c
    return c
end
"""
    _chunked_jacobian(g, v0, m, chunk) -> J (m × length(v0))

The Jacobian of `g` at `v0` in chunks of `chunk` inputs: every chunk one forward-mode pass over a Dual vector
of the chunk's width seeded with unit partials on the chunk's inputs and zeros elsewhere, the last (shorter)
chunk in the same width with its trailing partials unseeded, exactly as `ForwardDiff.jacobian` proceeds in
sequence, so the numbers are the sequential call's bit for bit on any thread count; the chunks on the
process's threads, since a dynamics factor's nineteen slice solves are independent and each is a second of
stiff integration (measured 2026-09-26 on the real-cook mock: 1.54 s sequential, 0.85 s on four threads).
"""
function _chunked_jacobian(g, v0::AbstractVector, m::Int, chunk::Int)
    n = length(v0)
    N = min(chunk, n)
    if Threads.nthreads() == 1                                       # one thread: the sequential call itself
        v0f = collect(Float64, v0)
        return ForwardDiff.jacobian(g, v0f, ForwardDiff.JacobianConfig(g, v0f, ForwardDiff.Chunk{N}()))
    end
    nchunks = cld(n, N)
    J = Matrix{Float64}(undef, m, n)
    v0f = collect(Float64, v0)
    tag = ForwardDiff.Tag(g, Float64)                       # registered through the constructor: the tag's rank against the integrator's own Duals depends on it
    D = ForwardDiff.Dual{typeof(tag),Float64,N}
    Threads.@threads for c in 1:nchunks
        c0 = (c - 1) * N
        cols = (c0 + 1):min(c0 + N, n)
        v = Vector{D}(undef, n)
        for i in 1:n
            v[i] = D(v0f[i], ForwardDiff.Partials(ntuple(j -> (i - c0 == j && i in cols) ? 1.0 : 0.0, N)))
        end
        y = g(v)
        for (j, ci) in enumerate(cols), r in 1:m
            J[r, ci] = ForwardDiff.partials(y[r], j)
        end
    end
    return J
end
"Discard only a dynamics factor's secant linearization; its current primal solve may still be reused."
function _invalidate_linearization!(f::DynamicsFactor)
    c = f.cache
    c === nothing && return f
    f.cache = (z=c.z, y0=c.y0, f=c.f, W=c.W, wy=c.wy, o=c.o,
               jz=nothing, jy0=nothing, Jy=nothing, Jz=nothing)
    return f
end
function residual(f::DynamicsFactor, vals)
    z, y0, y1 = _dyn_unpack(f, vals)
    c = _dyn_eval!(f, z, y0; jac=false)
    return c.W * (y1 .- c.f)
end
_dyn_jac_blocks(f::DynamicsFactor, c) = f.key_m === nothing ? [-c.W * c.Jz, -c.W * c.Jy, copy(c.W)] :
                                        vcat([-c.W * J for J in _split_cols(f.split, c.Jz)], [-c.W * c.Jy, copy(c.W)])
function jacobians(f::DynamicsFactor, vals)
    z, y0, _ = _dyn_unpack(f, vals)
    c = _dyn_eval!(f, z, y0; jac=true)
    return _dyn_jac_blocks(f, c)
end
function factor_linearization(f::DynamicsFactor, vals)
    z, y0, y1 = _dyn_unpack(f, vals)
    c = _dyn_eval!(f, z, y0; jac=true)
    return (r=c.W * (y1 .- c.f), Js=_dyn_jac_blocks(f, c))
end
"The slice's outputs at the last evaluation (for the past's integrals), or nothing; taken from a solve of their own when the factor reads a peer's."
function last_solve(f::DynamicsFactor)
    c = f.cache
    c === nothing && return nothing
    c.o === nothing || return c.o
    _, o = propagate(f.model, c.y0, c.z, f.t0, f.t1; first_slice=f.first)
    f.cache = merge(c, (o=o,))
    return o
end

"""
    ObservationFactor(model, spec, record, key_z, key_y, key_d; key_m=nothing, split=nothing)

A sensor's reading at its own time: r = (y − h(x, z) − d)/σ_w with `h` the compiled observable, `d` the
sensor's discrepancy at that sample (`key_d` nothing for a sensor without one) and σ_w the instrument's
floor. A censored reading (at the cap) contributes the survival likelihood −log Φ((h + d − cap)/σ_w) as its
loss and, as its residual, the local Gaussian site whose score and curvature match it at the current point
(`_censored_normal_site`): the pseudo-observation moves with the point, never lagging a linearization. With a
`split` the latents are `key_z` (static) and `key_m` (the moving copy at the reading's node).
"""
struct ObservationFactor <: Factor
    model::SystemModel
    spec::SensorSpec
    value::Float64
    censored::Bool
    key_z::VarKey
    key_m::Union{Nothing,VarKey}
    split::Union{Nothing,LatentSplit}
    key_y::VarKey
    key_d::Union{Nothing,VarKey}
    t::Float64
    h::Function
    rom_key::Union{Nothing,Symbol} # the reduction-error key of the observable under an artifact (its bias read at the latents), or nothing
    σ::Float64                    # the instrument's floor with the reduction error's spread in quadrature
end
function ObservationFactor(model::SystemModel, spec::SensorSpec, rec::MeasurementRecord, key_z::VarKey, key_y::VarKey, key_d::Union{Nothing,VarKey};
                           key_m::Union{Nothing,VarKey}=nothing, split::Union{Nothing,LatentSplit}=nothing)
    rec.sensor === spec.id || throw(ArgumentError("the record is $(rec.sensor)'s, the sensor $(spec.id)"))
    (key_m === nothing) == (split === nothing) || throw(ArgumentError("a moving latent copy needs its split, and a split its copy"))
    h = compile_observable(spec.observable, model.sys, model.bl, model.blk, model.log)
    rom_key, σ = nothing, spec.noise.white_sd
    if model.rom !== nothing
        key = _rom_key(spec.observable)
        if key !== nothing
            e = _rom_observation_error(model.rom, spec.observable, key, rec.t)
            rom_key = key; σ = sqrt(σ^2 + e.sd^2)
        end
    end
    return ObservationFactor(model, spec, rec.value, censored_at(spec, rec), key_z, key_m, split, key_y, key_d, rec.t, h, rom_key, σ)
end
function vars(f::ObservationFactor)
    ks = f.key_m === nothing ? [f.key_z, f.key_y] : [f.key_z, f.key_m, f.key_y]
    f.key_d === nothing || push!(ks, f.key_d)
    return ks
end
ndim(::ObservationFactor) = 1
provenance(f::ObservationFactor) = "$(f.spec.id) at $(f.t) s" * (f.censored ? " (censored)" : "")
# the prediction h(x, z) + d as a function of the stacked (z[static, moving], y[, d])
function _obs_pred(f::ObservationFactor, v)
    nz = length(f.model.blk)
    if f.key_m === nothing
        z = view(v, 1:nz)
    else
        ns = length(f.split.static)
        z = assemble_latents(f.split, view(v, 1:ns), view(v, (ns + 1):nz))
    end
    y = view(v, (nz + 1):(nz + f.model.bl.n))
    _validate_rom_use(f.model.rom, f.model.blk, z, f.t, f.t)
    p, _, tk = _point_params(f.model, z, f.t)
    x = system_from_belief(belief_decode(y, f.model.bl), f.model.bl)
    # ROMError stores ROM − FOM, so the full-order-equivalent prediction is ROM − bias, the bias at these latents.
    pred = f.h(x, z, p, tk, f.t)
    f.rom_key === nothing || (pred -= _rom_observation_bias(f.model.rom, f.spec.observable, f.rom_key, f.t, z, f.model.blk))
    f.key_d === nothing || (pred += v[end])
    return pred
end
_obs_stack(f::ObservationFactor, vals) = reduce(vcat, vals)
function residual(f::ObservationFactor, vals)
    μ = _obs_pred(f, _obs_stack(f, vals))
    σ = f.σ
    f.censored || return [(f.value - μ) / σ]
    site = _censored_normal_site(μ, f.spec.noise.cap, σ)
    return [site.root_precision * (μ - site.pseudo)]
end
function jacobians(f::ObservationFactor, vals)
    v = _obs_stack(f, vals)
    g = ForwardDiff.gradient(w -> _obs_pred(f, w), v)
    σ = f.σ
    if f.censored
        μ = _obs_pred(f, v)
        q = _censored_normal_site(μ, f.spec.noise.cap, σ).root_precision
        g = q .* g
    else
        g = -g ./ σ
    end
    out = Matrix{Float64}[]
    o = 0
    for x in vals
        push!(out, reshape(g[(o + 1):(o + length(x))], 1, length(x))); o += length(x)
    end
    return out
end
function loss(f::ObservationFactor, vals)
    μ = _obs_pred(f, _obs_stack(f, vals))
    f.censored || return 0.5 * ((f.value - μ) / f.σ)^2
    return _censored_normal_nll(μ, f.spec.noise.cap, f.σ)
end
"The sensor's innovation at the current values: the reading less the prediction (physics plus discrepancy), NaN when censored."
function innovation(f::ObservationFactor, vals)
    f.censored && return NaN
    return f.value - _obs_pred(f, _obs_stack(f, vals))
end
export innovation

"""
    DiscrepancyFactor(key_prev, key_next, ρ, σ_inn)

The OU transition of a sensor's discrepancy between two of its samples: r = (d_next − ρ d_prev)/σ_inn.
"""
struct DiscrepancyFactor <: Factor
    key_prev::VarKey
    key_next::VarKey
    ρ::Float64
    σ_inn::Float64
end
vars(f::DiscrepancyFactor) = [f.key_prev, f.key_next]
ndim(::DiscrepancyFactor) = 1
residual(f::DiscrepancyFactor, vals) = [(vals[2][1] - f.ρ * vals[1][1]) / f.σ_inn]
jacobians(f::DiscrepancyFactor, vals) = [fill(-f.ρ / f.σ_inn, 1, 1), fill(1 / f.σ_inn, 1, 1)]
provenance(f::DiscrepancyFactor) = "OU $(f.key_prev) → $(f.key_next)"

"""
    HierarchyFactor(child, parent, idx_child, idx_parent, τ)

A cook's parameters around the population's: r = (z_child[idx_child] − z_parent[idx_parent]) ./ τ (entrywise
sds); the same factor ties a parameter's copy to its predecessor for a drift between segments.
"""
struct HierarchyFactor <: Factor
    child::VarKey
    parent::VarKey
    idx_child::Vector{Int}
    idx_parent::Vector{Int}
    τ::Vector{Float64}
    function HierarchyFactor(child, parent, idx_child, idx_parent, τ)
        length(idx_child) == length(idx_parent) == length(τ) || throw(ArgumentError("a hierarchy factor needs one sd per tied entry"))
        all(x -> isfinite(x) && x > 0, τ) || throw(ArgumentError("the between-cook sds must be finite and positive"))
        length(unique(idx_child)) == length(idx_child) || throw(ArgumentError("hierarchy child indices must be unique"))
        length(unique(idx_parent)) == length(idx_parent) || throw(ArgumentError("hierarchy parent indices must be unique"))
        return new(child, parent, collect(Int, idx_child), collect(Int, idx_parent), collect(Float64, τ))
    end
end
HierarchyFactor(child::VarKey, parent::VarKey, τ::AbstractVector) = HierarchyFactor(child, parent, 1:length(τ), 1:length(τ), τ)
vars(f::HierarchyFactor) = [f.child, f.parent]
ndim(f::HierarchyFactor) = length(f.τ)
residual(f::HierarchyFactor, vals) = (vals[1][f.idx_child] .- vals[2][f.idx_parent]) ./ f.τ
function jacobians(f::HierarchyFactor, vals)
    Jc = zeros(length(f.τ), length(vals[1])); Jp = zeros(length(f.τ), length(vals[2]))
    for (r, (ic, ip, τ)) in enumerate(zip(f.idx_child, f.idx_parent, f.τ))
        Jc[r, ic] = 1 / τ; Jp[r, ip] = -1 / τ
    end
    return [Jc, Jp]
end
provenance(f::HierarchyFactor) = "hierarchy $(f.child) ~ $(f.parent)"

# --- the graph ------------------------------------------------------------------------------------------------------

"""
    FactorGraph()

Variables (keyed, with dimensions and current values, in insertion order) and factors. `linearize` stacks
every factor's Jacobian at the current values, `gauss_newton!` iterates, `marginalize!` removes variables in
square-root information form.
"""
mutable struct FactorGraph
    keys::Vector{VarKey}
    dims::Dict{VarKey,Int}
    values::Dict{VarKey,Vector{Float64}}
    factors::Vector{Factor}
end
FactorGraph() = FactorGraph(VarKey[], Dict{VarKey,Int}(), Dict{VarKey,Vector{Float64}}(), Factor[])
has_variable(g::FactorGraph, k::VarKey) = haskey(g.dims, k)
"Ensure cached dynamics Jacobians were made at the graph's exact current point, preserving their noise model."
function _ensure_current_dynamics!(g::FactorGraph)
    for f in g.factors
        _ensure_current!(f, g)
    end
    return g
end
# a factor with a cached secant linearization of a physics solve drops it unless it was made at the exact current point
_ensure_current!(::Factor, ::FactorGraph) = nothing
function _ensure_current!(f::DynamicsFactor, g::FactorGraph)
    c = f.cache
    c === nothing && return nothing
    z = _factor_latents(f, g); y0 = g.values[f.key_y0]
    # Trial residuals can move the primal cache without moving the Jacobian's
    # reference point. _dyn_eval! refreshes that primal independently.
    if c.Jy === nothing || c.jz != z || c.jy0 != y0
        _invalidate_linearization!(f)
    end
    return nothing
end
function add_variable!(g::FactorGraph, k::VarKey, x0::AbstractVector)
    has_variable(g, k) && throw(ArgumentError("the variable $k is already in the graph"))
    push!(g.keys, k); g.dims[k] = length(x0); g.values[k] = collect(Float64, x0)
    return g
end
function add_factor!(g::FactorGraph, f::Factor)
    for k in vars(f)
        has_variable(g, k) || throw(ArgumentError("the factor $(provenance(f)) touches $k, which is not in the graph"))
    end
    if f isa HierarchyFactor
        all(i -> 1 <= i <= g.dims[f.child], f.idx_child) || throw(ArgumentError("a hierarchy child index is out of range"))
        all(i -> 1 <= i <= g.dims[f.parent], f.idx_parent) || throw(ArgumentError("a hierarchy parent index is out of range"))
    end
    push!(g.factors, f)
    return g
end
function remove_factor!(g::FactorGraph, f::Factor)
    i = findfirst(x -> x === f, g.factors)
    i === nothing && throw(ArgumentError("the factor $(provenance(f)) is not in the graph"))
    deleteat!(g.factors, i)
    return g
end
variable_value(g::FactorGraph, k::VarKey) = g.values[k]
set_value!(g::FactorGraph, k::VarKey, x::AbstractVector) = (g.values[k] .= x; g)
"The factors touching the variable `k`."
factors_of(g::FactorGraph, k::VarKey) = [f for f in g.factors if k in vars(f)]
_vals(g::FactorGraph, f::Factor) = [g.values[k] for k in vars(f)]
"The exact objective: the sum of every factor's loss at the current values."
objective(g::FactorGraph) = sum(loss(f, _vals(g, f)) for f in g.factors; init=0.0)
function _objective_at(g::FactorGraph, x::AbstractVector, offsets)
    vals(f) = [x[offsets[k]] for k in vars(f)]
    return sum(loss(f, vals(f)) for f in g.factors; init=0.0)
end
"The stacked residual at `x`, in `linearize`'s row order (the factors in the graph's order)."
function _residuals_at(g::FactorGraph, x::AbstractVector, offsets)
    vals(f) = [x[offsets[k]] for k in vars(f)]
    return reduce(vcat, (residual(f, vals(f)) for f in g.factors); init=Float64[])
end
"The offset range of every variable in the stacked vector, in the graph's order (or the given order)."
function _offsets(g::FactorGraph, order::AbstractVector{VarKey}=g.keys)
    off = Dict{VarKey,UnitRange{Int}}(); o = 0
    for k in order
        off[k] = (o + 1):(o + g.dims[k]); o += g.dims[k]
    end
    return off, o
end
_stacked(g::FactorGraph, order, offsets, n) = (x = zeros(n); for k in order; x[offsets[k]] = g.values[k]; end; x)

"""
    linearize(g; order=g.keys, sparse=false) -> (J, r, offsets, n)

Every factor linearized at the current values, stacked: r(x̂ + δ) ≈ r + J δ. Dense, or a sparse matrix
for a large graph.
"""
function linearize(g::FactorGraph; order::AbstractVector{VarKey}=g.keys, sparse::Bool=false, factors::AbstractVector=g.factors)
    offsets, n = _offsets(g, order)
    m = sum(ndim(f) for f in factors; init=0)
    r = zeros(m)
    if sparse
        I_ = Int[]; J_ = Int[]; V_ = Float64[]
        row = 0
        for f in factors
            vals = _vals(g, f)
            lin = factor_linearization(f, vals); rf, Js = lin.r, lin.Js
            r[(row + 1):(row + length(rf))] = rf
            for (k, Jk) in zip(vars(f), Js)
                cols = offsets[k]
                for (jj, cj) in enumerate(cols), ii in 1:size(Jk, 1)
                    v = Jk[ii, jj]
                    v == 0 && continue
                    push!(I_, row + ii); push!(J_, cj); push!(V_, v)
                end
            end
            row += length(rf)
        end
        return SparseArrays.sparse(I_, J_, V_, m, n), r, offsets, n
    end
    J = zeros(m, n)
    row = 0
    for f in factors
        vals = _vals(g, f)
        lin = factor_linearization(f, vals); rf, Js = lin.r, lin.Js
        r[(row + 1):(row + length(rf))] = rf
        for (k, Jk) in zip(vars(f), Js)
            J[(row + 1):(row + length(rf)), offsets[k]] .+= Jk
        end
        row += length(rf)
    end
    return J, r, offsets, n
end

# the least-squares step −argmin ‖J δ + r‖² + λ‖D δ‖² with Marquardt's scaling (D the columns' norms, so the damping
# is the same fraction of every coordinate's own curvature whatever its units); dense by QR, sparse by SPQR
function _ls_step(J::AbstractMatrix, r::AbstractVector, λ::Float64)
    n = size(J, 2)
    if λ > 0
        d = [sqrt(sum(abs2, view(J, :, j))) for j in 1:n]
        d = max.(d, 1e-8 * max(maximum(d), 1e-300))
        D = J isa SparseArrays.SparseMatrixCSC ? SparseArrays.spdiagm(0 => sqrt(λ) .* d) : Matrix(Diagonal(sqrt(λ) .* d))
        A = vcat(J, D); b = vcat(-r, zeros(n))
    else
        A = J; b = -r
    end
    if A isa SparseArrays.SparseMatrixCSC
        F = SparseArrays.SPQR.qr(A; ordering=SparseArrays.SPQR.ORDERING_FIXED, tol=-2.0)
        return F \ b
    end
    return qr(A, ColumnNorm()) \ b
end

# Limit a total displacement, not merely the first Gauss--Newton proposal.  This is also applied after
# nonlinear corrections, so no correction can bypass a caller's trust bound.
function _limit_step!(d::AbstractVector, offsets, max_step::Float64, step_bounds::AbstractDict)
    bounded = false
    s = norm(d, Inf)
    if s > max_step
        d .*= max_step / s
        bounded = true
    end
    for (k, b) in step_bounds
        haskey(offsets, k) || continue
        b > 0 || throw(ArgumentError("a step bound must be positive"))
        s = norm(d[offsets[k]], Inf)
        if s > b
            d .*= b / s
            bounded = true
        end
    end
    return bounded
end

# A straight step through a nearly deterministic dynamics constraint can have a
# huge second-order defect even when its linearized residual is tiny. Transport
# the state endpoints with the nonlinear propagation, retaining the proposed
# first-order process innovations. Only the search path changes: the objective,
# acceptance, stationarity and covariance still use the original graph.
function _dynamics_retraction_data(g::FactorGraph)
    fs = sort([f for f in g.factors if f isa DynamicsFactor]; by=f -> f.t1)
    allunique(f.key_y1 for f in fs) ||
        throw(ArgumentError("dynamics retraction requires one incoming propagation per state node"))
    return map(fs) do f
        z, y0, y1 = _dyn_unpack(f, _vals(g, f))
        c = _dyn_eval!(f, z, y0; jac=true)
        (f=f, z=copy(z), y0=copy(y0), y1=copy(y1), defect=y1 .- c.f, Jy=c.Jy, Jz=c.Jz)
    end
end

function _retract_dynamics(xtrial, offsets, data)
    # Proposed defects use the original tangent step. Endpoints are then
    # transported in time order, including chains and split latent blocks.
    xn = copy(xtrial)
    for d in data
        f = d.f
        z, y0, y1 = _dyn_unpack(f, [xtrial[offsets[k]] for k in vars(f)])
        defect = d.defect .+ (y1 .- d.y1) .- d.Jy * (y0 .- d.y0) .- d.Jz * (z .- d.z)
        zr, yr, _ = _dyn_unpack(f, [xn[offsets[k]] for k in vars(f)])
        xn[offsets[f.key_y1]] .= _dyn_eval!(f, zr, yr; jac=false).f .+ defect
    end
    return xn
end

# Only genuine, fixed-weight least-squares residuals supply directional
# curvature. Local score/curvature pseudo-residuals (e.g. censored likelihoods
# and non-Gaussian sites) must not be differentiated as if they were residuals
# of the original objective. Dynamics defects are affine in the retraction
# chart, so their curvature is zero without finite-difference cancellation.
_search_residual_curvature(::Factor) = false
_search_residual_curvature(::GaussianPrior) = true
_search_residual_curvature(::LinearFactor) = true

function _geodesic_search_data(g, x, d, J, r, offsets, transport)
    fallback = (zeros(length(d)), 0.0)
    isempty(transport) && return fallback
    any(_search_residual_curvature, g.factors) || return fallback
    # A substantial directional difference avoids amplifying integration
    # roundoff. This affects the search only; exact objective and fresh
    # stationarity checks still decide acceptance and convergence.
    h = 0.1
    try
        xp = _retract_dynamics(x .+ h .* d, offsets, transport)
        xm = _retract_dynamics(x .- h .* d, offsets, transport)
        curvature = zeros(length(r)); row = 0
        for f in g.factors
            rows = (row + 1):(row + ndim(f))
            if _search_residual_curvature(f)
                rp = residual(f, [xp[offsets[k]] for k in vars(f)])
                rm = residual(f, [xm[offsets[k]] for k in vars(f)])
                curvature[rows] .= (rp .- 2 .* r[rows] .+ rm) ./ h^2
            end
            row += ndim(f)
        end
        all(isfinite, curvature) || return fallback
        q = qr(Matrix(J), ColumnNorm()); n = size(J, 2)
        a = -0.5 .* (q \ curvature)
        Ri = UpperTriangular(Matrix(q.R)[1:n, 1:n]) \ Matrix{Float64}(I, n, n)
        sd = max.(sqrt.(vec(sum(abs2, Ri; dims=2))[invperm(q.p)]), eps())
        # Express both vectors in marginal posterior SDs to avoid mixing
        # temperature, humidity, and latent-coordinate units in this guard.
        ratio = 2norm(a ./ sd) / max(norm(d ./ sd), eps())
        all(isfinite, a) && isfinite(ratio) || return fallback
        return a, ratio
    catch err
        # Any probe error falls back to the plain step: the fallback skips acceleration and cannot
        # change the problem (test_geodesic_search pins this), unlike a swallowed line-search trial.
        err isa InterruptException && rethrow()
        return fallback
    end
end

function _within_step_bounds(d, offsets, max_step, step_bounds; origin=zeros(length(d)))
    # Subtracting the accepted point from a trial can lose a few ulps of the
    # point itself, even when the tangent step was exactly on the bound.
    slack(r, b) = 16eps(Float64) * max(1.0, b, norm(origin[r], Inf))
    norm(d, Inf) <= max_step + slack(eachindex(d), max_step) || return false
    return all(!haskey(offsets, k) || norm(d[offsets[k]], Inf) <= b + slack(offsets[k], b) for (k, b) in step_bounds)
end

# The full undamped Gauss--Newton displacement in local coordinate-step units. Solving the coupled
# least-squares problem is essential: dividing the score by diagonal curvature can report convergence
# arbitrarily far along a correlated valley. The coordinate metric remains invariant to uniform residual scaling;
# the joint decrement uses the declared noise scale, through the whitened Jacobian.
# A positive coordinate scale makes the comparison dimensionless when one graph contains,
# for example, latent coordinates of order one and a humidity discrepancy measured in pascals. It also
# prevents a tiny damped move from being called stationary when an undamped relative move is still large.
function _stationarity_metrics(J::AbstractMatrix, r::AbstractVector, scales::AbstractVector)
    isempty(r) && return (0.0, 0.0)
    length(scales) == size(J, 2) || throw(DimensionMismatch("the stationarity scales do not match the graph coordinates"))
    all(isfinite, scales) && all(>(0), scales) || throw(ArgumentError("stationarity scales must be finite and positive"))
    δ = _ls_step(J, r, 0.0)
    return maximum(abs.(δ) ./ scales; init=0.0), norm(J * δ)
end
_scaled_gradient(J::AbstractMatrix, r::AbstractVector, scales::AbstractVector=ones(size(J, 2))) =
    first(_stationarity_metrics(J, r, scales))
# The undamped Gauss-Newton step at a point, in two metrics: its largest scaled coordinate move (the solver's
# tolerance is stated in these) and its largest move in units of the point's own Gauss-Newton posterior standard
# deviation, from the same pivoted QR ((JᵀJ)⁻¹ = P R⁻¹ R⁻ᵀ Pᵀ). The first metric is the same coupled
# displacement used by the convergence check; the second also accounts for the posterior's local width.
# The solver reports both at its accepted point; a dense graph only (the offline sparse problem is too large to invert).
function _newton_step_metrics(J::AbstractMatrix, r::AbstractVector, scales::AbstractVector)
    isempty(r) && return (0.0, 0.0)
    Jd = Matrix(J)
    n = size(Jd, 2)
    size(Jd, 1) >= n || return (NaN, NaN)                            # fewer rows than columns: no posterior width to speak of
    F = qr(Jd, ColumnNorm())
    δ = F \ (-r)
    Ri = UpperTriangular(Matrix(F.R)[1:n, 1:n]) \ Matrix{Float64}(I, n, n)
    v = vec(sum(abs2, Ri; dims=2))                                   # diag(R⁻¹R⁻ᵀ) in the pivoted order
    sd = sqrt.(max.(v[invperm(F.p)], 0.0))
    return (maximum(abs.(δ) ./ scales; init=0.0), maximum(abs.(δ) ./ max.(sd, eps(Float64)); init=0.0))
end
function _stationarity_scale_vector(order, offsets, n::Int, stationarity_scales::AbstractDict)
    scales = ones(n)
    for (k, sk) in stationarity_scales
        haskey(offsets, k) || continue
        dest = offsets[k]
        if sk isa Real
            isfinite(sk) && sk > 0 || throw(ArgumentError("the stationarity scale for $k must be finite and positive"))
            scales[dest] .= sk
        else
            length(sk) == length(dest) ||
                throw(DimensionMismatch("the stationarity scale for $k has length $(length(sk)), expected $(length(dest))"))
            all(isfinite, sk) && all(>(0), sk) ||
                throw(ArgumentError("the stationarity scales for $k must be finite and positive"))
            scales[dest] .= sk
        end
    end
    return scales
end

"""
    _armijo_search(F0, d0, try_step; c1=1e-4, max_trials=8) -> (α, Fn, trials)

The step length along a descent direction by backtracking with interpolation (Nocedal & Wright, §3.5): the
full step first; when it fails Armijo's sufficient decrease `F(α) ≤ F0 + c1 α F'(0)`, the next length is
the minimiser of the quadratic through `F0`, `F'(0)` and the failed trial, and after that of the cubic
through the last two trials, each safeguarded into [α/10, α/2]; a trial that does not evaluate (a failed
solve) quarters the length. `try_step(α)` returns the objective at `x + α δ` (Inf when it cannot). Returns the
accepted length and objective, or `α = 0` with `Fn = Inf` when no trial passed.
"""
function _armijo_search(F0::Float64, d0::Float64, try_step; c1::Float64=1e-4, max_trials::Int=8)
    α = 1.0; α_prev = NaN; F_prev = NaN
    for trial in 1:max_trials
        Fn = try_step(α)
        # Armijo's right-hand side can round back to F0 for a heavily damped microscopic step.
        # Equality is not progress: accepting it resets the consecutive-rejection counter and can
        # strand the solver forever at a nonstationary point.
        (isfinite(Fn) && Fn < F0 && Fn <= F0 + c1 * α * d0) && return (α, Fn, trial)
        if !isfinite(Fn)
            α_new = α / 4
        elseif isnan(α_prev)
            denom = 2 * (Fn - F0 - d0 * α)                                  # the quadratic through F0, F'(0) and (α, Fn)
            α_new = denom > 0 ? -d0 * α^2 / denom : α / 2
        else
            r1 = Fn - F0 - d0 * α; r0 = F_prev - F0 - d0 * α_prev              # the cubic through both trials (N&W eq. 3.59)
            det = α_prev^2 * α^2 * (α - α_prev)
            ca = (α_prev^2 * r1 - α^2 * r0) / det
            cb = (-α_prev^3 * r1 + α^3 * r0) / det
            disc = cb^2 - 3 * ca * d0
            α_new = (abs(ca) < 1e-30 || disc < 0) ? (cb > 0 ? -d0 / (2cb) : α / 2) : (-cb + sqrt(disc)) / (3ca)
        end
        isfinite(α_new) || (α_new = α / 2)
        α_prev, F_prev = α, Fn
        α = clamp(α_new, α / 10, α / 2)
    end
    return (0.0, Inf, max_trials)
end

"A full step's actual over predicted decrease above which `gauss_newton!`'s `expand` tries a longer step."
const EXPAND_GAIN = 1.25

"""
    gauss_newton!(g; iters=30, tol=1e-6, ftol=1e-8, gtol=tol,
                  resolution_gtol=2gtol, λ0=0.0, sparse=false,
                  verbose=false, step_bounds=Dict(), stationarity_scales=Dict(),
                  adaptive_damping=false) -> (F, hist, converged, gradient, newton_step, newton_sd)

Damped Gauss-Newton on the exact objective: the least-squares step of the linearized factors (with the
Marquardt-scaled damping λ when it is on), its length by the interpolating Armijo backtracking of
`_armijo_search` along the step (the directional derivative rᵀJδ is known from the linearization, so the
search fits the step length rather than halving blindly); a step rejected at every trial raises the damping
tenfold. With `adaptive_damping=true`, an accepted step whose actual reduction is poor against the frozen
quadratic model, or whose line-search fraction is below a tenth, also raises the damping; a well-predicted
full step eases it threefold. This planner-oriented option remains line-search Gauss-Newton rather than a
pure Levenberg-Marquardt gain-ratio update: a useful fraction of an undamped direction is still accepted
before damping is changed for the next iteration. Without it, every accepted full step eases damping and a
partial step leaves it unchanged, which is more robust to the online dynamics factors' local kinks. Stops
when an accepted move is under `tol` in the stacked coordinates, or when an undamped full step drops the
objective by less than `ftol`; a damping past 1e8 stops it unconverged. `step_bounds` maps a variable to the
largest move (∞-norm) one iteration may make in it: the whole step is scaled down to honour it, keeping the
direction. It is the bound on a log-parameter's move that keeps
the physics inside its linear regime: a diffusion time scales with the thickness squared, so the linearization
of the propagation in log-thickness holds to a tenth over a move of a quarter and not at all over one of 0.6,
which is what the unbounded step asked for at every arrival of the two-piece bench (2026-09-25: the true
objective's curvature along the step was a hundred times the model's, the line search settled at a hundredth
of the step and thirty iterations moved a tenth of the way). A bounded step's small drop does not declare
convergence.

`soc` requests safeguarded nonlinear corrections at a trial point. They are fixed-J chord corrections, not
an SQP second-order correction. Each correction solves for the COMPLETE displacement from the accepted
point, with the same Marquardt penalty as the base step; caller bounds and `max_step` are applied to that
complete displacement, scaled by the line-search fraction. Thus a correction can move unbounded state nodes
enough to absorb a dynamics factor's nonlinear response, but cannot evade damping, the line search, or a
latent trust bound. A correction is retained only when it lowers the trial objective, while Armijo still
requires the decrease predicted by the base descent direction. A failed line search refreshes any stale
physics Jacobian at the accepted mean before retrying with increased damping. Convergence additionally requires the scaled
full undamped Gauss--Newton displacement to be below `gtol`; a small damped, bounded, or backtracked move alone is never
convergence. The numerical-resolution exception requires three stalled trials and a fresh scaled
gradient no larger than `resolution_gtol` (which must be at least `gtol`). Failed searches count as stalled;
unbounded accepted moves count only while their total scaled displacement remains within `tol` of the
stall anchor, so roundoff-sized decreases cannot indefinitely reset this check. Caller-bounded moves do
not qualify. This covers objectives quantized by an ODE solve without accepting the large-gradient damped-step failure
the stationarity gate was added to reject. `stationarity_scales` maps variables to positive scalar or
per-coordinate characteristic scales; the reported gradient and both thresholds are relative to those
scales. The default `2gtol` is deliberately narrow; an ODE-backed caller may supply a separately measured
resolution envelope. `newton_step` and `newton_sd` are the undamped Gauss-Newton step at the accepted point, its largest
scaled coordinate move and its largest move in the point's own posterior standard deviations (`_newton_step_metrics`;
NaN for a sparse solve). The historically named `gradient` field also reports the scaled full displacement.
With `newton_tol` > 0 (off by
default) a solve stalled three trials is also accepted when its `newton_sd`, at an
exact relinearization, is under `newton_tol`. This resolution exception measures the move against the posterior
width as well as the declared coordinate units; it never fires on a sparse solve. `expand` > 0 lets an accepted full step
whose decrease beat the model's by more than `EXPAND_GAIN` double (up to `expand` times, within the bounds) while the
objective keeps falling: a path change for under-stepping on curved valleys, never a certificate.
`restart_newton_tol` (off by default) permits the same three-stall check to request a fresh solver attempt:
when the posterior step is below this bound but no acceptance rule passes, return `converged=false` early,
with `restart_requested=true` in the last history entry. It changes the retry budget, never the acceptance gate.

`retract_dynamics=true` follows the nonlinear propagation along the proposed
step while retaining its linearized process innovations. It requires one incoming
dynamics factor per state node. The existing secant-cache radii govern the search
linearization; rejected searches and final certificates refresh stale derivatives.
All caller bounds also apply to the transported endpoint. On graphs
with dynamics this replaces chord corrections; expansion follows the same
nonlinear path and still respects the original bounds.
Backtracking reduces the candidate along that path; it does not shrink the
caller's bound again after transport. Otherwise an outward quadratic correction
at a saturated tangent bound can make every positive step infeasible.
The original graph objective, stationarity tests and posterior covariance remain
authoritative. Graphs without dynamics retain the ordinary search.

`geodesic_search=true` adds a quadratic tangent correction from the directional
curvature of supported Gaussian residuals along that retraction. It requires
`retract_dynamics=true` and a dense solve. The correction is bounded relative
to the first-order displacement, and every candidate must satisfy the original
objective decrease and actual endpoint bounds. Invalid curvature probes fall
back to the ordinary retracted direction. After backtracking, up to four
doublings recover an over-shortened step only while the objective improves.
It is off by default and never changes the stopping criterion.

`joint_tol` (off by default) additionally accepts a fresh linearization whose
undamped Gauss--Newton decrement `norm(J * δ)` is below this bound. With local
covariance `C = inv(J'J)`, this is `sqrt(δ' * inv(C) * δ)`: it accounts for
correlations and bounds the displacement of every linear contrast in its own
posterior SD. It does not require numerical stalling. Cached derivatives may
only prompt the check; acceptance always uses fresh derivatives. The returned
`newton_joint_sd` reports this metric, while `newton_sd` remains the largest
component displacement divided by its marginal SD. This is a local Gaussian
accuracy criterion; the original nonlinear objective still governs the search.
"""
# A trial point may fail legitimately only through what it evaluates: a trajectory the integrator cannot
# carry (the system solve's own failure), a coordinate boundary (DomainError), or the factorization of a
# trial-dependent matrix (Singular/PosDef). Those score Inf and the line search moves on. Anything else is
# a defect in the code, not in the trial, and must surface instead of masquerading as a failed step that
# escalates damping. Threaded jacobian chunks arrive wrapped; classify what they carry.
function _trial_failure(err)
    err isa CompositeException && return !isempty(err.exceptions) && all(_trial_failure, err.exceptions)
    err isa TaskFailedException && return _trial_failure(err.task.exception)
    return (err isa ErrorException && startswith(err.msg, "system solve failed")) ||
           err isa DomainError || err isa SingularException || err isa PosDefException
end
function gauss_newton!(g::FactorGraph; iters::Int=30, tol::Float64=1e-6, ftol::Float64=1e-8, gtol::Float64=tol,
                       resolution_gtol::Float64=2gtol, newton_tol::Float64=0.0, restart_newton_tol::Float64=0.0,
                       joint_tol::Float64=0.0,
                       λ0::Float64=0.0, sparse::Bool=false,
                       verbose::Bool=false, max_step::Float64=Inf, step_bounds::AbstractDict=Dict{VarKey,Float64}(), soc::Int=10,
                       stationarity_scales::AbstractDict=Dict(),
                       adaptive_damping::Bool=false, expand::Int=0, retract_dynamics::Bool=false,
                       geodesic_search::Bool=false)
    geodesic_search && (!retract_dynamics || sparse) &&
        throw(ArgumentError("geodesic search requires dense dynamics retraction"))
    isfinite(resolution_gtol) && resolution_gtol >= gtol ||
        throw(ArgumentError("resolution_gtol must be finite and no smaller than gtol"))
    isfinite(joint_tol) && joint_tol >= 0 || throw(ArgumentError("joint_tol must be finite and non-negative"))
    newton_tol >= 0 || throw(ArgumentError("newton_tol must be non-negative"))
    isfinite(restart_newton_tol) && restart_newton_tol >= 0 ||
        throw(ArgumentError("restart_newton_tol must be finite and non-negative"))
    order = g.keys
    offsets, n = _offsets(g, order)
    gradient_scales = _stationarity_scale_vector(order, offsets, n, stationarity_scales)
    # the joint read: the undamped step at an exactly relinearized point, in the point's own posterior sds (NaN on a
    # sparse solve, so the stall rescue below never fires there)
    newton_sd_at(J, r) = sparse ? NaN : _newton_step_metrics(J, r, gradient_scales)[2]
    x = _stacked(g, order, offsets, n)
    F = _objective_at(g, x, offsets)
    isfinite(F) || error("the objective is not finite at the start of the solve")
    λ = λ0
    hist = NamedTuple[]; converged = false; rejected_at_point = 0
    stall_anchor = copy(x)
    for it in 1:iters
        t0 = time()
        # Reuse the established secant cache within its declared radii. Failed
        # searches and every final certificate still refresh stale derivatives.
        J, r, _, _ = linearize(g; order, sparse)
        F = _objective_at(g, x, offsets)                               # the baseline under the linearization's own whitening (a refresh is cached solves, no new ones)
        grad, joint_sd = _stationarity_metrics(J, r, gradient_scales)
        if grad <= gtol || (joint_tol > 0 && joint_sd <= joint_tol)
            # A cached secant within `relin_tol` is sufficient for a trial direction, but not for a
            # stationarity certificate. Refresh any dynamics Jacobian not made at this exact mean.
            _ensure_current_dynamics!(g)
            J, r, _, _ = linearize(g; order, sparse)
            F = _objective_at(g, x, offsets)
            grad, joint_sd = _stationarity_metrics(J, r, gradient_scales)
            if grad <= gtol || (joint_tol > 0 && joint_sd <= joint_tol)
                converged = true
                push!(hist, (it=it, F=F, F_trial=F, step=0.0, alpha=0.0, damping=λ, trials=0, soc=0,
                             accepted=true, bounded=false, gradient=grad, newton_joint_sd=joint_sd,
                             joint_converged=grad > gtol, wall=time() - t0))
                break
            end
        end
        δ = _ls_step(J, r, λ)
        bounded = _limit_step!(δ, offsets, max_step, step_bounds)
        d0 = dot(r, J * δ)                                             # the directional derivative of ½‖r‖² along δ
        transport = retract_dynamics ? _dynamics_retraction_data(g) : []
        curved = !isempty(transport)
        acceleration, curvature_ratio = geodesic_search && curved ?
            _geodesic_search_data(g, x, δ, J, r, offsets, transport) : (zeros(length(δ)), 0.0)
        cand = Ref((F=Inf, x=x, soc=0, bounded=bounded))
        function try_point(xn)
            for k in order; g.values[k] .= xn[offsets[k]]; end
            return try _objective_at(g, xn, offsets) catch err; _trial_failure(err) || rethrow(); Inf end
        end
        function try_step(α)
            α * curvature_ratio <= 0.75 || return Inf
            base = α .* δ .+ α^2 .* acceleration
            # This must be local to the trial: a rejected expansion must not
            # overwrite the outer solver's accepted xn through a closure.
            trial_x = x .+ base
            if curved
                trial_x = try _retract_dynamics(trial_x, offsets, transport) catch err
                    _trial_failure(err) || rethrow()
                    return Inf
                end
                # These are bounds on the actual endpoint. Multiplying them
                # by alpha again would make an outward O(alpha^2) correction
                # infeasible at every alpha when the tangent saturates a bound.
                _within_step_bounds(trial_x .- x, offsets, max_step, step_bounds; origin=x) || return Inf
            end
            best = (F=try_point(trial_x), x=trial_x, soc=0, bounded=bounded)
            disp = base
            # The correction is a new COMPLETE damped displacement, not an increment with its own
            # damping penalty. At a trial displacement d with residual r(d), the fixed-J model referred
            # back to the accepted point is J*d + (r(d) - J*d). Solving that model with `_ls_step`
            # therefore penalizes the total d. Bounds are line-search-scaled so a correction cannot use
            # a full latent trust radius after Armijo shortened the base step.
            trial_max_step = isfinite(max_step) ? α * max_step : max_step
            trial_bounds = Dict(k => α * b for (k, b) in step_bounds)
            for sc in 1:(curved ? 0 : soc)
                isfinite(best.F) || break
                rn = try _residuals_at(g, best.x, offsets) catch err; _trial_failure(err) || rethrow(); nothing end
                rn === nothing && break
                dtry = _ls_step(J, rn .- J * disp, λ)
                corrected_bounded = _limit_step!(dtry, offsets, trial_max_step, trial_bounds)
                xc = x .+ dtry
                norm(xc .- best.x, Inf) > eps(Float64) || break
                Fc = try_point(xc)
                (isfinite(Fc) && Fc < best.F) || break
                disp = dtry
                best = (F=Fc, x=xc, soc=sc, bounded=bounded || corrected_bounded)
            end
            cand[] = best
            return best.F
        end
        α, Fn, trials = d0 < 0 ? _armijo_search(F, d0, try_step) : (0.0, Inf, 0)
        accepted = α > 0 && isfinite(Fn) && Fn <= F
        if !accepted
            for k in order; g.values[k] .= x[offsets[k]]; end
            # A secant Jacobian can point uphill at the accepted mean. Retrying
            # it with more damping cannot repair that direction. Refresh stale
            # Jacobians before the next attempt, even outside the stopping gate.
            _ensure_current_dynamics!(g)
            rejected_at_point += 1
            λ = λ == 0 ? 1e-3 : 10λ
            push!(hist, (it=it, F=F, F_trial=Fn, step=norm(δ), alpha=0.0, damping=λ, trials=trials, soc=0,
                         accepted=false, bounded=bounded, gradient=grad, wall=time() - t0))
            verbose && @printf("    GN it %2d rejected  F %.6g  |step| %.3g  slope %.3g  damping %.2g\n", it, F, norm(δ), d0, λ)
            verbose && flush(stdout)
            # Near the ODE solver's resolution floor an exact descent direction can fail Armijo at every
            # trial because the requested objective drop is below repeatable integration noise. Do not
            # resurrect the old "small damped step means convergence" rule: require three stalled
            # trials within tol of the stall anchor, then refresh every dynamics
            # linearization and accept only a KKT displacement within the caller's explicit numerical
            # resolution envelope. A large gradient still fails and remains transactional in the online backend.
            # With `newton_tol` > 0 the same three stalls also qualify a point whose undamped joint step is under
            # `newton_tol` of a posterior sd. This uses the posterior width instead of characteristic coordinate
            # units, at the exactly relinearized point, like the coordinate-resolution check.
            if rejected_at_point >= 3 && (grad <= resolution_gtol || newton_tol > 0 || restart_newton_tol > 0)
                _ensure_current_dynamics!(g)
                Jc, rc, _, _ = linearize(g; order, sparse)
                Fc = _objective_at(g, x, offsets)
                gradc = _scaled_gradient(Jc, rc, gradient_scales)
                nsdc = (gradc <= resolution_gtol || max(newton_tol, restart_newton_tol) <= 0) ? NaN : newton_sd_at(Jc, rc)
                if gradc <= resolution_gtol || (newton_tol > 0 && isfinite(nsdc) && nsdc <= newton_tol)
                    F = Fc
                    converged = true
                    hist[end] = merge(hist[end], (F=F, gradient_final=gradc, newton_sd=nsdc, resolution_limited=true))
                    break
                elseif restart_newton_tol > 0 && isfinite(nsdc) && nsdc <= restart_newton_tol
                    F = Fc
                    hist[end] = merge(hist[end], (F=F, gradient_final=gradc, newton_sd=nsdc, restart_requested=true))
                    break
                end
            end
            λ > 1e8 && break
            continue
        end
        xn = cand[].x; nsoc = cand[].soc; bounded = cand[].bounded
        if geodesic_search && curved && α < 1.0
            for _ in 1:4
                next_alpha = min(1.0, 2α)
                next_alpha > α || break
                next_F = try_step(next_alpha)
                (isfinite(next_F) && next_F < Fn && next_F <= F + 1e-4 * next_alpha * d0) || break
                α = next_alpha; Fn = next_F; xn = cand[].x; bounded = cand[].bounded
            end
        end
        # Expansion (`expand` > 0 doublings, off by default): a full step whose decrease beat the Gauss-Newton model's
        # prediction by more than EXPAND_GAIN is under-stepping along a curved valley (the two-probe control of
        # 2026-09-30 crawled 80 accepted full steps at gain ≈ 2, each a fiftieth of a posterior sd, and was refused at
        # the iteration cap). The same accepted displacement is doubled while the objective keeps falling, never past
        # the caller's bounds. Only the path changes: convergence is still the stationarity certificate.
        if expand > 0 && α == 1.0
            d1 = curved ? δ .+ acceleration : xn .- x
            Jd = J * d1
            pred1 = -(dot(r, Jd) + 0.5 * sum(abs2, Jd))
            if pred1 > 0 && (F - Fn) / pred1 > EXPAND_GAIN
                room = isfinite(max_step) ? max_step / max(norm(d1, Inf), eps()) : Inf
                for (k, b) in step_bounds
                    haskey(offsets, k) || continue
                    room = min(room, b / max(norm(d1[offsets[k]], Inf), eps()))
                end
                scale = 1.0
                for _ in 1:expand
                    2scale <= room || break
                    xt = x .+ (2scale) .* d1
                    Ft = if curved
                        trial_F = try_step(2scale)
                        xt = cand[].x
                        trial_F <= F + 1e-4 * (2scale) * d0 ? trial_F : Inf
                    else
                        try_point(xt)
                    end
                    (isfinite(Ft) && Ft < Fn) || break
                    scale *= 2; xn = xt; Fn = Ft
                    curved && (bounded = bounded || cand[].bounded)   # a soc-corrected trial's bound must survive into stall accounting
                end
                scale > 1 && (bounded = bounded || 2scale > room)
                curved && (α = scale)
            end
        end
        dropped = F - Fn
        stepn = norm(xn .- x); moved = norm(xn .- x, Inf)
        Jstep = J * (curved ? α .* δ .+ α^2 .* acceleration : xn .- x)
        predicted_drop = -(dot(r, Jstep) + 0.5 * sum(abs2, Jstep))
        gain = predicted_drop > 0 ? dropped / predicted_drop : -Inf
        for k in order; g.values[k] .= xn[offsets[k]]; end
        x = xn; F = Fn
        # ODE roundoff can accept microscopic decreases between rejected
        # searches. They must not reset the stall counter indefinitely. Count
        # only unbounded moves whose TOTAL displacement since the anchor is
        # within tol; a fresh coupled certificate below remains mandatory.
        if !bounded && maximum(abs.(x .- stall_anchor) ./ gradient_scales) <= tol
            rejected_at_point += 1
        else
            rejected_at_point = 0
            stall_anchor = copy(x)
        end
        # α > 1 is an accepted curved doubling: the model under-predicted the drop, the strongest
        # case for relaxing damping. It must not fall between the escalation and relaxation branches.
        if adaptive_damping
            if α < 0.1 || gain < 0.25
                λ = λ == 0 ? 1e-3 : 10λ
            elseif α >= 1.0 && gain > 0.75
                λ = λ / 3 < 1e-6 ? 0.0 : λ / 3
            end
        elseif α >= 1.0
            λ = λ / 3 < 1e-6 ? 0.0 : λ / 3
        end
        push!(hist, (it=it, F=F, F_trial=Fn, step=stepn, alpha=α, damping=λ, trials=trials, soc=nsoc,
                     accepted=true, bounded=bounded, gradient=grad, gain=gain, wall=time() - t0))
        verbose && @printf("    GN it %2d  F %.6g  |step| %.3g  alpha %.3g (%d trials, %d soc)  gain %.3g  damping %.2g%s\n", it, F, stepn, α, trials, nsoc, gain, λ, bounded ? "  bounded" : "")
        verbose && flush(stdout)
        # A small move/drop only asks for a stationarity check at the accepted point. Refresh the dynamics
        # Jacobians so the returned covariance and mean use the same local model of the fixed objective.
        if moved < tol || rejected_at_point >= 3 || (dropped < ftol && α == 1.0 && λ < 1e-2 && !bounded)
            _ensure_current_dynamics!(g)
            Jf, rf, _, _ = linearize(g; order, sparse)
            F = _objective_at(g, x, offsets)
            gradf, joint_sd = _stationarity_metrics(Jf, rf, gradient_scales)
            if gradf <= gtol || (joint_tol > 0 && joint_sd <= joint_tol)
                converged = true
                hist[end] = merge(hist[end], (F=F, gradient_final=gradf,
                                             newton_joint_sd=joint_sd, joint_converged=gradf > gtol))
                break
            elseif rejected_at_point >= 3
                nsd = (gradf <= resolution_gtol || max(newton_tol, restart_newton_tol) <= 0) ? NaN : newton_sd_at(Jf, rf)
                if gradf <= resolution_gtol || (newton_tol > 0 && isfinite(nsd) && nsd <= newton_tol)
                    converged = true
                    hist[end] = merge(hist[end], (F=F, gradient_final=gradf, newton_sd=nsd, resolution_limited=true))
                    break
                elseif restart_newton_tol > 0 && isfinite(nsd) && nsd <= restart_newton_tol
                    hist[end] = merge(hist[end], (F=F, gradient_final=gradf, newton_sd=nsd, restart_requested=true))
                    break
                end
            end
        end
    end
    for k in order; g.values[k] .= x[offsets[k]]; end
    _ensure_current_dynamics!(g)
    Jf, rf, _, _ = linearize(g; order, sparse)
    F = _objective_at(g, x, offsets)
    grad, joint_sd = _stationarity_metrics(Jf, rf, gradient_scales)
    # The last permitted iteration may land exactly at a stationary point. Its refreshed KKT check is
    # authoritative even when no earlier in-loop stopping branch had set `converged`.
    converged = converged || grad <= gtol || (joint_tol > 0 && joint_sd <= joint_tol)
    # Also report the joint step in posterior standard deviations at the accepted point.
    newton_step, newton_sd = sparse ? (NaN, NaN) : _newton_step_metrics(Jf, rf, gradient_scales)
    verbose && @printf("    GN final  F %.6g  gradient %.3g  Newton step %.3g (%.2f posterior sd)  converged %s\n", F, grad, newton_step, newton_sd, converged)
    verbose && flush(stdout)
    return (F=F, hist=hist, converged=converged, gradient=grad, newton_step=newton_step,
            newton_sd=newton_sd, newton_joint_sd=joint_sd, joint_tol=joint_tol)
end

"""
    marginalize!(g, out::AbstractVector{VarKey}) -> LinearFactor

The variables `out` removed in square-root information form: every factor touching them, linearized at the
current values, stacked with the leaving variables' columns first and QR-factored; the trailing triangle
is the marginal prior on the staying variables those factors touched, added as a `LinearFactor` (in
absolute coordinates: R₂₂ x_stay = z₂ + R₂₂ x̂_stay); the factors and the variables are removed.
"""
function marginalize!(g::FactorGraph, out::AbstractVector{VarKey})
    isempty(out) && return nothing
    for k in out; has_variable(g, k) || throw(ArgumentError("cannot marginalize $k: not in the graph")); end
    outset = Set(out)
    touching = [f for f in g.factors if any(k in outset for k in vars(f))]
    stay = VarKey[]
    for f in touching, k in vars(f)
        (k in outset || k in stay) || push!(stay, k)
    end
    order = vcat(collect(out), stay)
    J, r, offsets, n = linearize(g; order, factors=touching)
    xhat = _stacked(g, order, offsets, n)
    c = J * xhat .- r                                     # the factors as ‖J x − c‖²
    n_out = sum(g.dims[k] for k in out)
    rank(view(J, :, 1:n_out)) == n_out || error("cannot marginalize $(out): their outgoing Jacobian is rank deficient")
    F = qr(J)
    R = Matrix(F.R)                                       # min(m, n) × n
    Qc = F.Q' * c
    m = size(R, 1)
    m > n_out || error("marginalizing $(out) leaves no information on the rest (the leaving variables have $n_out entries and the factors $m rows)")
    R22 = R[(n_out + 1):m, (n_out + 1):n]
    z2 = Qc[(n_out + 1):m]
    blocks = [R22[:, offsets[k] .- n_out] for k in stay]
    lf = LinearFactor(stay, blocks, z2; source="marginal of $(join(string.(out), ", "))")
    for f in touching; remove_factor!(g, f); end
    for k in out
        deleteat!(g.keys, findfirst(==(k), g.keys)); delete!(g.dims, k); delete!(g.values, k)
    end
    isempty(stay) || add_factor!(g, lf)
    return lf
end

"""
    joint_cov(g; keys=g.keys, sparse=false) -> (cov, offsets)

The Gauss-Newton covariance (JᵀJ)⁻¹ of the graph at the current values, over `keys` (their marginal block).
"""
function joint_cov(g::FactorGraph; keys::AbstractVector{VarKey}=g.keys, sparse::Bool=false)
    order = vcat(collect(keys), [k for k in g.keys if !(k in keys)])
    _ensure_current_dynamics!(g)
    J, r, offsets, n = linearize(g; order, sparse)
    nk = sum(g.dims[k] for k in keys)
    E = Matrix(1.0I, n, nk)
    # Normalize columns before both the numerical-rank test and the triangular solve. Process/state rows can
    # differ from prior rows by ten orders of magnitude; testing an unscaled R against its largest diagonal
    # falsely calls a structurally full-rank graph singular. With q = Sx and Js = J*S⁻¹, form Cov(q) and map
    # the selected block back by S⁻¹.
    scales = collect(Float64, vec(sqrt.(sum(abs2, J; dims=1))))
    all(s -> isfinite(s) && s > 0, scales) || error("the graph is rank deficient: at least one variable column has no finite information")
    invscales = 1.0 ./ scales
    Js = J isa SparseArrays.SparseMatrixCSC ? J * SparseArrays.spdiagm(0 => invscales) : J .* transpose(invscales)
    # (JsᵀJs)⁻¹ = R⁻¹ R⁻ᵀ, its leading block YᵀY with Y = R⁻ᵀE.
    if J isa SparseArrays.SparseMatrixCSC
        F = SparseArrays.SPQR.qr(Js; ordering=SparseArrays.SPQR.ORDERING_FIXED, tol=-2.0)
        R = SparseArrays.sparse(F.R)
        d = abs.(diag(R)); dtol = eps(Float64) * max(size(Js)...) * maximum(d; init=0.0)
        (size(R, 1) == n && length(d) >= n && all(>(dtol), d[1:n])) || error("the graph is rank deficient: no covariance")
        Y = LowerTriangular(SparseArrays.sparse(R')) \ E
    else
        F = qr(Js)
        R = Matrix(F.R)
        d = abs.(diag(R)); dtol = eps(Float64) * max(size(Js)...) * maximum(d; init=0.0)
        (size(R, 1) >= n && length(d) >= n && all(>(dtol), d[1:n])) || error("the graph is rank deficient: no covariance")
        Y = LowerTriangular(R[1:n, :]') \ E
    end
    Cq = Y' * Y
    ds = invscales[1:nk]
    return Matrix(Symmetric((ds .* Cq) .* transpose(ds))), Dict(k => offsets[k] for k in keys)
end
"The marginal covariance of one variable."
marginal_cov(g::FactorGraph, k::VarKey; sparse::Bool=false) = joint_cov(g; keys=[k], sparse)[1]
