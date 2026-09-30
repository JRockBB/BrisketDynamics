# The online backend (docs/multi-meat-inference-plan.md): a square-root information fixed-lag smoother on
# the shared factors over a CookSystem. The belief is a window of state nodes at the times measurements
# arrived (no grid: a node is made when a sample is), the latents once, and one discrepancy variable per
# sensor sample at its own time; each new arrival adds its observation factor (and, when its time falls inside
# the window, a node between two others, splitting the dynamics factor there: the delayed measurement the
# plan asks for), the window is re-solved by damped Gauss-Newton on the exact objective, and the nodes older
# than the lag are marginalized in square-root form (factors.jl `marginalize!`), the trailing triangle kept
# as the linear prior on what stays. With the lag infinite nothing is marginalized and the smoother is the
# full-history solve (test_factors.jl: the two agree on a linear system to roundoff).
#
# A sensor's newest discrepancy variable is never marginalized: the next sample's OU factor needs it, and a
# scalar costs nothing. A delayed sample may not precede the sensor's oldest live discrepancy when older
# ones have been committed (the chain's correlation with a committed variable is gone), nor the window's
# oldest node: both are errors, not approximations.
#
# The memory is bounded by the lag; every solve's cost is the window's dynamics Jacobians (ForwardDiff through
# the slice solves, cached by point, so a node whose neighbours did not move is not re-solved) and one dense
# QR of the window. The statistical linearization of sequential.jl is not carried over: this backend
# linearizes at the current estimate and iterates. Included by BrisketEstimator after factors.jl.

export FixedLagSmoother, online_init, add_measurement!, solve_window!, marginalize_old!, assimilate!, belief_at,
       BeliefSnapshot, belief_snapshot, to_plan_state, window_nodes, key_params, key_node, key_disc, key_drift,
       SmootherCheckpoint, smoother_checkpoint, restore_checkpoint, restore_checkpoint!, backend_fingerprint,
       update_execution!, ALSFeed

"""
    ALSFeed(model; lag_s=7200.0, bin_s=120.0, ou=true, white=true, keep=false)

The bench's innovation-autocovariance accumulator (als.jl) over the online backend's newest-node frame: the
latents, the newest node's belief coordinates, then the newest live discrepancy of every sensor that has one
(at its stationary prior while it has none). Its groups: the state's noise groups (the filter's twelve names for
a one-piece system, the layout's names otherwise), then with `ou` the stationary variance of every discrepancy
sensor (`ou.<id>`, a process group on its own coordinate), then with `white` the white variance of every sensor
(`white.<id>`, a measurement group). The variance rates it starts from are the model's own. Feed arrivals
in strictly increasing time order, with all readings at one timestamp batched together. Delayed and mixed-time
batches remain supported by the smoother without an ALS feed; they are not a single Markov-frame transition.
"""
struct ALSFeed
    acc::ALSAccumulator
    disc_sensors::Vector{Symbol}
    state_names::Vector{String}
    ou_group::Dict{Symbol,Int}
    white_group::Dict{Symbol,Int}
end
function ALSFeed(model::SystemModel; lag_s::Float64=7200.0, bin_s::Float64=120.0, ou::Bool=true, white::Bool=true, keep::Bool=false)
    bl = model.bl
    state_names = String[]; θ = Float64[]
    for (g, _) in bl.groups
        nm = _als_group_name(bl, g)
        nm in state_names && continue
        push!(state_names, nm); push!(θ, _group_rate(bl, model.sn, g, false)^2)
    end
    names = copy(state_names)
    disc_sensors = [s.id for s in model.log.sensors if has_discrepancy(s.noise)]
    ou_group = Dict{Symbol,Int}(); white_group = Dict{Symbol,Int}()
    if ou
        for id in disc_sensors
            push!(names, "ou." * string(id)); push!(θ, sensor(model.log, id).noise.ar_sd^2); ou_group[id] = length(names)
        end
    end
    if white
        for s in model.log.sensors
            push!(names, "white." * string(s.id)); push!(θ, s.noise.white_sd^2); white_group[s.id] = length(names)
        end
    end
    n = length(model.blk) + bl.n + length(disc_sensors)
    acc = ALSAccumulator(names, θ, n, length(model.log.sensors); lag_s, bin_s, keep)
    return ALSFeed(acc, disc_sensors, state_names, ou_group, white_group)
end

# Adaptive ODE solves cease to give a repeatable descent signal below roughly five parts in ten thousand
# in the solver's diagonal Gauss--Newton coordinate. This is not a general convergence tolerance: it is
# consulted only after three complete Armijo failures at one point, and the stricter `tol` still governs
# ordinary stationary solves. The two-piece adversarial gate leaves a resolvable 7.8e-4 displacement at
# arrival 12 (which must restart) and a repeatably unresolvable 4.63e-4 displacement at arrival 13.
const ONLINE_ODE_RESOLUTION_GTOL = 5e-4
# the joint stationarity rescue of a stalled solve (gauss_newton!'s `newton_tol`), on the final attempt only: a point three
# complete line searches could not leave, whose undamped step at an exact relinearization is under this many posterior sds,
# is accepted (2026-09-26: the smoother lost one arrival in thirty at a diagonal read of 5e-4 with the joint step at 1e-4 sd)
const ONLINE_NEWTON_TOL = 1e-2

# A dynamics factor spanning several execution-grid intervals accumulates more adaptive-step quantization
# than the one-grid-step factors on which ONLINE_ODE_RESOLUTION_GTOL was measured. Scale that measured
# envelope only as sqrt(number of grid intervals), and never beyond the dynamics factors' own relinearization
# radius: a point outside that radius has a resolvable fresh local model rather than a numerical-floor excuse.
function _online_resolution_gtol(sm)
    isempty(sm.dyn) && return ONLINE_ODE_RESOLUTION_GTOL
    Δgrid = minimum(diff(sm.model.log.t_grid))
    Δfactor = maximum(f.t1 - f.t0 for f in values(sm.dyn))
    relin_cap = minimum(f.relin_tol for f in values(sm.dyn))
    return min(relin_cap, ONLINE_ODE_RESOLUTION_GTOL * sqrt(max(1.0, Δfactor / Δgrid)))
end

"""
    FixedLagSmoother

The online backend's state: the model, the graph, the window's nodes `(t, key)` ascending, the dynamics
factors by `(from, to)`, per sensor its live discrepancy nodes and the time of its newest committed one, the
observation factors, the lag (nodes kept behind the newest; `typemax(Int)` for the full history), the
solver's settings, the committed past per piece and the records of every assimilation.
"""
mutable struct FixedLagSmoother
    model::SystemModel
    graph::FactorGraph
    cook::Symbol
    key_z::VarKey
    nodes::Vector{Tuple{Float64,VarKey}}
    node_count::Int
    dyn::Dict{Tuple{VarKey,VarKey},DynamicsFactor}
    disc::Dict{Symbol,Vector{Tuple{Float64,VarKey}}}
    disc_count::Dict{Symbol,Int}
    disc_committed_t::Dict{Symbol,Float64}
    disc_fac::Dict{Tuple{VarKey,VarKey},DiscrepancyFactor}
    obs::Vector{ObservationFactor}
    lag::Int
    iters::Int
    restarts::Int                  # fresh-damping retries at the same uncommitted graph point
    tol::Float64
    ftol::Float64
    latent_step::Float64           # the largest move of the latents one Gauss-Newton iteration may make (gauss_newton!'s step_bounds)
    relin_tol::Float64             # the dynamics factors' relinearization tolerances (DynamicsFactor): the latents' move
    relin_tol_state::Float64       # and the start node's, beyond which a factor's linearization is retaken during a solve
    goals::PlanGoals
    past::Dict{Symbol,Past}
    committed_t::Float64
    records::Vector{NamedTuple}
    record_limit::Int
    record_count::Int
    n_solves::Int
    als::Union{Nothing,ALSFeed}    # the bench's accumulator, fed after every committed arrival when the bench asks for it
    split::Union{Nothing,LatentSplit}          # the latents that drift: a copy per window node beside the static block (nothing: every latent static)
    drift_q::Vector{Float64}                   # per moving entry its random-walk variance per hour (at least DRIFT_Q_FLOOR)
    drift_jumps::Vector{Tuple{Int,Float64,Float64}}   # (moving entry, time [s], variance) added to the copy-to-copy factor of the slice crossing the time
    drift_fac::Dict{Tuple{VarKey,VarKey},HierarchyFactor}   # the random-walk factors between consecutive copies, by (earlier, later)
end
key_params(sm::FixedLagSmoother) = sm.key_z
key_node(sm::FixedLagSmoother, k::Int) = VarKey(:state, sm.cook, k)
key_disc(sm::FixedLagSmoother, id::Symbol, j::Int) = VarKey(:disc, (sm.cook, id), j)
"The moving latents' copy at node number `k` (the node's own number: `key_node(sm, k).k`)."
key_drift(sm::FixedLagSmoother, k::Int) = VarKey(:drift, sm.cook, k)
_key_m(sm::FixedLagSmoother, ky::VarKey) = sm.split === nothing ? nothing : key_drift(sm, ky.k)
window_nodes(sm::FixedLagSmoother) = copy(sm.nodes)
# a moving latent whose plant only jumps has no walk between jumps: its copies are tied at this floor variance per hour
# (1e-3 per √h in the latent's coordinate: over a day 5e-3, nothing against a posterior width), since a factor cannot
# tie two variables exactly and a variable tied exactly is the static block
const DRIFT_Q_FLOOR = 1e-6
"The random-walk sds between the copies at `tp` and `t`: the walk over the hours, the jumps the slice crosses."
function _drift_tau(sm::FixedLagSmoother, tp::Float64, t::Float64)
    Δh = max(t - tp, 0.0) / 3600
    v = sm.drift_q .* Δh
    for (i, tj, var) in sm.drift_jumps
        tp < tj <= t && (v[i] += var)
    end
    return sqrt.(max.(v, DRIFT_Q_FLOOR * max(Δh, 1e-9)))
end
"The full latent vector at node `ky` (the static block with the copy at the node), and the keys it comes from."
function _latents_at(sm::FixedLagSmoother, ky::VarKey)
    sm.split === nothing && return variable_value(sm.graph, sm.key_z), [sm.key_z]
    km = key_drift(sm, ky.k)
    return assemble_latents(sm.split, variable_value(sm.graph, sm.key_z), variable_value(sm.graph, km)), [sm.key_z, km]
end

"""
    update_execution!(sm; u=sm.model.log.u, events=sm.model.log.events)

Advance the immutable execution record used by future factors. Existing factors retain their prior immutable
`SystemModel` and log, while subsequently inserted dynamics/observation factors use the validated replacement.
"""
function update_execution!(sm::FixedLagSmoother; u=sm.model.log.u, events=sm.model.log.events)
    old = sm.model
    log = with_execution(old.log; u, events)
    sm.model = SystemModel(old.sys, log, old.blk; sn=old.sn, abstol=old.abstol, reltol=old.reltol, rom=old.rom)
    return sm
end

"""
    online_init(model; t0=model.log.t_grid[1], lag=3, iters=40, restarts=1, tol=1e-4, ftol=1e-2, goals=PlanGoals(), graph=FactorGraph(), cook=:cook, hierarchy=nothing,
                als=nothing, relin_tol=1e-3, relin_tol_state=relin_tol, drift=nothing) -> FixedLagSmoother

The belief before any sample: the latents at each family's moments in its coordinate (the filter's start), the
first node at `t0` with the recipe's state (the system's initial state at the prior centre, the ignited
fraction from a0) at the `StateNoise` initial widths mapped into the belief's coordinates. `ftol`, the objective's
drop under which a solve stops, is a hundredth of a nat (the RxInfer filter stops at a twentieth). `graph` and `cook`
let several cooks share one graph under distinct keys (the offline backend); `hierarchy = (idx, key_pop, tau)`
ties the latents `idx` of this cook to the population variable `key_pop` with the between-cook sds `tau`
instead of giving them the family's prior. `als` is the bench's accumulator (`ALSFeed`), fed after every committed
arrival. `relin_tol` and `relin_tol_state` are the dynamics factors' relinearization tolerances (the latents' move,
the node's move). `drift = (moving, sd_per_hour, jumps)` makes the latents `moving` a copy per window node tied to
the previous node's by a random walk of `sd_per_hour` (each in its coordinate) with `jumps = [(index, t, sd)]` added
on the slice crossing `t`; the rest stay one static block (`LatentSplit`). A cook with a hierarchy cannot drift.
"""
function online_init(model::SystemModel; t0::Real=model.log.t_grid[1], lag::Int=3, iters::Int=40, restarts::Int=1, tol::Float64=1e-4,
                     ftol::Float64=1e-2, latent_step::Float64=0.25, goals::PlanGoals=PlanGoals(), graph::FactorGraph=FactorGraph(),
                     cook::Symbol=:cook, hierarchy=nothing, record_limit::Int=256, als::Union{Nothing,ALSFeed}=nothing,
                     relin_tol::Float64=1e-3, relin_tol_state::Float64=relin_tol, drift=nothing)
    lag >= 0 || throw(ArgumentError("the lag counts nodes kept behind the newest; it cannot be negative"))
    iters >= 0 || throw(ArgumentError("iters cannot be negative"))
    restarts >= 0 || throw(ArgumentError("restarts cannot be negative"))
    record_limit >= 0 || throw(ArgumentError("record_limit cannot be negative"))
    g = graph
    blk = model.blk; bl = model.bl
    z0 = [z_moments(p)[1] for p in blk.sites]
    key_z = VarKey(:params, cook, 0)
    # the drifting latents: `drift = (moving, sd_per_hour, jumps)`: the indices of the latents that move, each one's
    # random-walk sd per hour in its coordinate, and the jumps `(index, t, sd)`; the static block is the rest
    split = nothing; drift_q = Float64[]; drift_jumps = Tuple{Int,Float64,Float64}[]
    if drift !== nothing
        hierarchy === nothing || throw(ArgumentError("drifting latents and a hierarchy on one cook are not built together"))
        moving = collect(Int, drift.moving)
        all(i -> 1 <= i <= length(blk), moving) || throw(ArgumentError("a drifting latent index is out of range"))
        length(unique(moving)) == length(moving) || throw(ArgumentError("a drifting latent is listed twice"))
        length(drift.sd_per_hour) == length(moving) || throw(ArgumentError("one drift sd per moving latent"))
        all(x -> isfinite(x) && x >= 0, drift.sd_per_hour) || throw(ArgumentError("a drift sd must be finite and not negative"))
        split = LatentSplit(setdiff(1:length(blk), moving), moving)
        drift_q = max.(Float64.(drift.sd_per_hour) .^ 2, DRIFT_Q_FLOOR)
        for (i, tj, sd) in drift.jumps
            j = findfirst(==(Int(i)), moving)
            j === nothing && throw(ArgumentError("a jump on latent $i, which does not move: list it among the moving latents"))
            isfinite(tj) && isfinite(sd) && sd >= 0 || throw(ArgumentError("a jump needs a finite time and sd"))
            push!(drift_jumps, (j, Float64(tj), Float64(sd)^2))
        end
        isempty(split.static) && throw(ArgumentError("every latent drifting leaves no static block; keep at least one static"))
    end
    add_variable!(g, key_z, split === nothing ? z0 : z0[split.static])
    if split !== nothing
        km1 = VarKey(:drift, cook, 1)
        add_variable!(g, km1, z0[split.moving])
        add_factor!(g, SitePriorFactor(km1, blk.sites[split.moving]; source="$(cook): the moving latents' exact priors at the first node"))
    end
    if hierarchy === nothing
        if split === nothing
            add_factor!(g, SitePriorFactor(key_z, blk.sites; source="$(cook): the latents' exact transformed-family priors"))
        else
            add_factor!(g, SitePriorFactor(key_z, blk.sites[split.static]; source="$(cook): the static latents' exact transformed-family priors"))
        end
    else
        links = if hierarchy isa Tuple
            idx, key_pop, tau = hierarchy
            [(idx_child=Int(ic), key_parent=key_pop, idx_parent=j, tau=Float64(tau[j])) for (j, ic) in enumerate(idx)]
        else
            collect(hierarchy)
        end
        tied = [l.idx_child for l in links]
        length(unique(tied)) == length(tied) || throw(ArgumentError("a cook latent cannot be tied to two hierarchy parents"))
        all(i -> 1 <= i <= length(blk), tied) || throw(ArgumentError("a hierarchy child index is out of range"))
        own = setdiff(1:length(blk), tied)
        isempty(own) || add_factor!(g, SitePriorFactor(key_z, blk.sites; indices=own, source="$(cook): the cook's own exact latent priors"))
        for l in links
            isfinite(l.tau) && l.tau > 0 || throw(ArgumentError("a hierarchy sd must be finite and positive"))
            add_factor!(g, HierarchyFactor(key_z, l.key_parent, [l.idx_child], [l.idx_parent], [l.tau]))
        end
    end
    # the first node: the recipe's state at the prior centre
    p, play, _ = _point_params(model, z0, Float64(t0))
    x0 = initial_state(model.sys, p, play)
    y0 = belief_encode(belief_from_system(x0, bl), bl)
    Q0 = belief_cov_to_coord(belief_init_cov(bl, model.sn), y0, bl)
    k1 = VarKey(:state, cook, 1)
    add_variable!(g, k1, y0)
    add_factor!(g, GaussianPrior(k1, y0, Symmetric(_stable_cov(Q0)); source="$(cook): the state's initial widths"))
    past = Dict(pc.id => Past() for pc in model.sys.pieces)
    for pc in model.sys.pieces; past[pc.id].m0 = x0[first(state_index(model.sys.layout, :pit, :m_fuel))]; past[pc.id].t = Float64(t0); end
    return FixedLagSmoother(model, g, cook, key_z, [(Float64(t0), k1)], 1, Dict{Tuple{VarKey,VarKey},DynamicsFactor}(),
                            Dict(s.id => Tuple{Float64,VarKey}[] for s in model.log.sensors), Dict(s.id => 0 for s in model.log.sensors),
                            Dict(s.id => -Inf for s in model.log.sensors), Dict{Tuple{VarKey,VarKey},DiscrepancyFactor}(),
                            ObservationFactor[], lag, iters, restarts, tol, ftol, latent_step, relin_tol, relin_tol_state, goals, past, Float64(t0), NamedTuple[],
                            record_limit, 0, 0, als, split, drift_q, drift_jumps, Dict{Tuple{VarKey,VarKey},HierarchyFactor}())
end

"Append one diagnostic to the bounded in-memory ring."
function _push_record!(sm::FixedLagSmoother, rec::NamedTuple)
    sm.record_count += 1
    sm.record_limit == 0 && return rec
    push!(sm.records, rec)
    while length(sm.records) > sm.record_limit
        popfirst!(sm.records)
    end
    return rec
end

"Replace a smoother's state in place; used to roll back an unsuccessful transaction."
function _restore_smoother!(sm::FixedLagSmoother, saved::FixedLagSmoother)
    for i in 1:fieldcount(FixedLagSmoother)
        setfield!(sm, i, getfield(saved, i))
    end
    return sm
end

# A transaction snapshot preserves the graph object's and all pre-existing factors' identities. This matters
# when several offline smoothers share one graph: replacing `sm.graph` with a deep copy would leave the
# OfflineProblem and its other smoothers pointing at a mutated graph, while replacing its factors with copied
# objects would invalidate their dictionaries. Only DynamicsFactor carries mutable nonlinear cache state.
function _transaction_snapshot(sm::FixedLagSmoother)
    dyn_state = IdDict{DynamicsFactor,NamedTuple}()
    for f in sm.graph.factors
        f isa DynamicsFactor || continue
        dyn_state[f] = (cache=deepcopy(f.cache), n_solves=f.n_solves, n_jacobians=f.n_jacobians)
    end
    return (
        graph_keys=copy(sm.graph.keys),
        graph_dims=copy(sm.graph.dims),
        graph_values=Dict(k => copy(v) for (k, v) in sm.graph.values),
        graph_factors=copy(sm.graph.factors),
        dyn_state,
        nodes=copy(sm.nodes),
        node_count=sm.node_count,
        dyn=copy(sm.dyn),
        disc=Dict(id => copy(chain) for (id, chain) in sm.disc),
        disc_count=copy(sm.disc_count),
        disc_committed_t=copy(sm.disc_committed_t),
        disc_fac=copy(sm.disc_fac),
        drift_fac=copy(sm.drift_fac),
        obs=copy(sm.obs),
        past=deepcopy(sm.past),
        committed_t=sm.committed_t,
        records=copy(sm.records),
        record_count=sm.record_count,
        n_solves=sm.n_solves,
    )
end
function _restore_transaction!(sm::FixedLagSmoother, saved)
    g = sm.graph
    g.keys = saved.graph_keys
    g.dims = saved.graph_dims
    g.values = saved.graph_values
    g.factors = saved.graph_factors
    for (f, state) in saved.dyn_state
        f.cache = state.cache
        f.n_solves = state.n_solves
        f.n_jacobians = state.n_jacobians
    end
    sm.nodes = saved.nodes
    sm.node_count = saved.node_count
    sm.dyn = saved.dyn
    sm.disc = saved.disc
    sm.disc_count = saved.disc_count
    sm.disc_committed_t = saved.disc_committed_t
    sm.disc_fac = saved.disc_fac
    sm.drift_fac = saved.drift_fac
    sm.obs = saved.obs
    sm.past = saved.past
    sm.committed_t = saved.committed_t
    sm.records = saved.records
    sm.record_count = saved.record_count
    sm.n_solves = saved.n_solves
    return sm
end

"""
    SmootherCheckpoint

An in-memory, versioned exact continuation point. It owns a deep copy of the nonlinear window, boundary
square-root factors, discrepancy chains, committed integrals, counters and bounded diagnostics. The physics
fingerprint prevents restoration under different source physics.
"""
struct SmootherCheckpoint
    version::Int
    physics::String
    backend::String
    smoother::FixedLagSmoother
end
const BACKEND_FILES = ("factors.jl", "system_state.jl", "semantic.jl", "online.jl", "offline.jl",
                       "plan_system.jl", "rom.jl", "als.jl", "bench_fit.jl", "posterior_sampling.jl", "posterior_smc.jl", "sampled_online.jl", "particle_online.jl", "sampled_planning.jl", "sampled_offline.jl", "conditional.jl", "conditional_transport.jl", "conditional_density.jl", "conditional_online.jl", "deterministic_online.jl")
"The implementation fingerprint for checkpoint compatibility, including physics and every new backend layer."
function backend_fingerprint(; repo::AbstractString=REPO_ROOT)
    ctx = SHA.SHA1_CTX()
    SHA.update!(ctx, codeunits(physics_fingerprint(; repo)))
    for name in sort(collect(BACKEND_FILES))
        path = "estimator/src/" * name
        SHA.update!(ctx, codeunits(path))
        SHA.update!(ctx, filter(!=(0x0d), read(joinpath(repo, path))))
    end
    return bytes2hex(SHA.digest!(ctx))[1:12]
end
smoother_checkpoint(sm::FixedLagSmoother) =
    SmootherCheckpoint(2, physics_fingerprint(), backend_fingerprint(), deepcopy(sm))
function restore_checkpoint(cp::SmootherCheckpoint)
    cp.version == 2 || throw(ArgumentError("unsupported smoother checkpoint version $(cp.version)"))
    cp.physics == physics_fingerprint() || throw(ArgumentError("the checkpoint was made under physics $(cp.physics), not $(physics_fingerprint())"))
    cp.backend == backend_fingerprint() ||
        throw(ArgumentError("the checkpoint was made under backend $(cp.backend), not $(backend_fingerprint())"))
    return deepcopy(cp.smoother)
end
restore_checkpoint!(sm::FixedLagSmoother, cp::SmootherCheckpoint) = _restore_smoother!(sm, restore_checkpoint(cp))

# the dynamics factor between two nodes, added to the graph and the table
function _add_dyn!(sm::FixedLagSmoother, ka::VarKey, ta::Float64, kb::VarKey, tb::Float64; first::Bool)
    f = DynamicsFactor(sm.model, sm.key_z, ka, kb, ta, tb; first, events=sensor_events(sm.model.log, ta, tb),
                       relin_tol=sm.relin_tol, relin_tol_state=sm.relin_tol_state, key_m=_key_m(sm, ka), split=sm.split)   # the copy in force from the slice's start
    # Anchor the noise before any measurements can move this slice's prediction. This also makes an
    # offline graph's covariance independent of which factor happens to be evaluated first in a solve.
    _dyn_eval!(f, _factor_latents(f, sm.graph), variable_value(sm.graph, ka); jac=false)
    add_factor!(sm.graph, f); sm.dyn[(ka, kb)] = f
    return f
end
function _remove_dyn!(sm::FixedLagSmoother, ka::VarKey, kb::VarKey)
    f = pop!(sm.dyn, (ka, kb))
    remove_factor!(sm.graph, f)
    return f
end
# the random-walk factor between the copies at nodes `ka` (at `ta`) and `kb` (at `tb`)
function _add_drift!(sm::FixedLagSmoother, ka::VarKey, ta::Float64, kb::VarKey, tb::Float64)
    sm.split === nothing && return nothing
    kma, kmb = key_drift(sm, ka.k), key_drift(sm, kb.k)
    nm = length(sm.split.moving)
    f = HierarchyFactor(kmb, kma, 1:nm, 1:nm, _drift_tau(sm, ta, tb))
    add_factor!(sm.graph, f); sm.drift_fac[(kma, kmb)] = f
    return f
end
function _remove_drift!(sm::FixedLagSmoother, ka::VarKey, kb::VarKey)
    sm.split === nothing && return nothing
    f = pop!(sm.drift_fac, (key_drift(sm, ka.k), key_drift(sm, kb.k)))
    remove_factor!(sm.graph, f)
    return f
end

"""
    _node_at!(sm, t) -> key

The state node at time `t`: the existing one, or a new one inserted in time order (between two nodes the
dynamics factor is split; after the newest a new slice is appended), its value the propagation from the
node before it; with drifting latents its moving copy beside it, starting at the previous copy's value and
tied to its neighbours' copies by the random walk. An error before the window's oldest node.
"""
function _node_at!(sm::FixedLagSmoother, t::Float64)
    for (tn, k) in sm.nodes
        abs(tn - t) < 1e-6 && return k
    end
    t > sm.nodes[1][1] || throw(ArgumentError("a sample at $t s precedes the window's oldest node at $(sm.nodes[1][1]) s: outside the lag"))
    i = findlast(x -> x[1] < t, sm.nodes)
    tp, kp = sm.nodes[i]
    sm.node_count += 1
    kn = key_node(sm, sm.node_count)
    z_prev, _ = _latents_at(sm, kp)
    if sm.split !== nothing
        add_variable!(sm.graph, key_drift(sm, kn.k), copy(variable_value(sm.graph, key_drift(sm, kp.k))))
    end
    if i < length(sm.nodes)
        tn, kn_next = sm.nodes[i + 1]
        old = _remove_dyn!(sm, kp, kn_next)
        _remove_drift!(sm, kp, kn_next)
        y, _ = propagate(sm.model, variable_value(sm.graph, kp), z_prev, tp, t; first_slice=old.first)
        add_variable!(sm.graph, kn, y)
        _add_dyn!(sm, kp, tp, kn, t; first=old.first)
        _add_dyn!(sm, kn, t, kn_next, tn; first=false)
        _add_drift!(sm, kp, tp, kn, t); _add_drift!(sm, kn, t, kn_next, tn)
    else
        fs = kp.k == 1                                               # the slice from the initial node reads a0
        y, _ = propagate(sm.model, variable_value(sm.graph, kp), z_prev, tp, t; first_slice=fs)
        add_variable!(sm.graph, kn, y)
        _add_dyn!(sm, kp, tp, kn, t; first=fs)
        _add_drift!(sm, kp, tp, kn, t)
    end
    insert!(sm.nodes, i + 1, (t, kn))
    return kn
end

"""
    _disc_at!(sm, spec, t) -> key or nothing

The sensor's discrepancy variable at `t` (a sensor without a discrepancy has none): inserted into the sensor's
chain with its OU factors (the neighbours' factor split), a stationary prior when it is the chain's first;
an error when it would precede a committed one.
"""
function _disc_at!(sm::FixedLagSmoother, spec::SensorSpec, t::Float64)
    has_discrepancy(spec.noise) || return nothing
    chain = sm.disc[spec.id]
    for (td, k) in chain
        abs(td - t) < 1e-6 && return k
    end
    t > sm.disc_committed_t[spec.id] || throw(ArgumentError("a sample of $(spec.id) at $t s precedes its committed discrepancy at $(sm.disc_committed_t[spec.id]) s: outside the lag"))
    (isempty(chain) || t > chain[1][1] || sm.disc_committed_t[spec.id] == -Inf) ||
        throw(ArgumentError("a sample of $(spec.id) at $t s precedes its oldest live discrepancy at $(chain[1][1]) s while older ones are committed: outside the lag"))
    sm.disc_count[spec.id] += 1
    kd = key_disc(sm, spec.id, sm.disc_count[spec.id])
    i = findlast(x -> x[1] < t, chain)
    if i === nothing
        add_variable!(sm.graph, kd, [0.0])
        add_factor!(sm.graph, GaussianPrior(kd, [0.0], fill(1 / spec.noise.ar_sd, 1, 1); source="$(spec.id)'s stationary discrepancy"))
        if !isempty(chain)                                          # before every live one: the old first loses its stationary prior for an OU link
            tn, kn = chain[1]
            for f in factors_of(sm.graph, kn)
                f isa GaussianPrior && remove_factor!(sm.graph, f)
            end
            ρ, σ = ou_step(spec.noise, tn - t)
            fac = DiscrepancyFactor(kd, kn, ρ, σ); add_factor!(sm.graph, fac); sm.disc_fac[(kd, kn)] = fac
        end
        insert!(chain, 1, (t, kd))
    else
        tp, kp = chain[i]
        ρ, σ = ou_step(spec.noise, t - tp)
        add_variable!(sm.graph, kd, [ρ * variable_value(sm.graph, kp)[1]])
        if i < length(chain)
            tn, kn = chain[i + 1]
            remove_factor!(sm.graph, pop!(sm.disc_fac, (kp, kn)))
            ρn, σn = ou_step(spec.noise, tn - t)
            fac2 = DiscrepancyFactor(kd, kn, ρn, σn); add_factor!(sm.graph, fac2); sm.disc_fac[(kd, kn)] = fac2
        end
        fac = DiscrepancyFactor(kp, kd, ρ, σ); add_factor!(sm.graph, fac); sm.disc_fac[(kp, kd)] = fac
        insert!(chain, i + 1, (t, kd))
    end
    return kd
end

"""
    add_measurement!(sm, rec) -> ObservationFactor

One reading into the window: its node (made or found), its discrepancy variable, its observation factor.
Nothing is solved here.
"""
function add_measurement!(sm::FixedLagSmoother, rec::MeasurementRecord)
    spec = sensor(sm.model.log, rec.sensor)
    rec.t >= sm.nodes[1][1] - 1e-9 || throw(ArgumentError("a sample at $(rec.t) s precedes the window's oldest node"))
    ky = _node_at!(sm, rec.t)
    kd = _disc_at!(sm, spec, rec.t)
    f = ObservationFactor(sm.model, spec, rec, sm.key_z, ky, kd; key_m=_key_m(sm, ky), split=sm.split)
    add_factor!(sm.graph, f); push!(sm.obs, f)
    return f
end

"The window solved by damped Gauss-Newton on the exact objective."
function solve_window!(sm::FixedLagSmoother; verbose::Bool=false)
    all_hist = NamedTuple[]
    r = nothing
    attempts = 0
    # Stationarity is a relative coordinate displacement. State and latent coordinates keep their established
    # solver tolerances; sensor discrepancies retain sensor-specific physical units, so normalize each live
    # discrepancy by its stationary AR width (for example, pascals for chamber humidity).
    stationarity_scales = Dict{VarKey,Vector{Float64}}()
    for (id, chain) in sm.disc
        σd = sensor(sm.model.log, id).noise.ar_sd
        for (_, k) in chain
            stationarity_scales[k] = [σd]
        end
    end
    measured_resolution_gtol = _online_resolution_gtol(sm)
    for attempt in 1:(sm.restarts + 1)
        attempts = attempt
        # Preserve permutation accuracy by exhausting fresh-damping solves under the narrow generic
        # resolution gate. Only the final configured attempt may use the measured adaptive-ODE
        # envelope; this keeps a resolvable 4e-4 stall from becoming a different accepted posterior
        # merely because a system's pieces changed order.
        resolution_gtol = attempt == sm.restarts + 1 ?
                          max(2 * sm.tol, measured_resolution_gtol) : 2 * sm.tol
        bounds = Dict{VarKey,Float64}(sm.key_z => sm.latent_step)
        if sm.split !== nothing
            for (_, k) in sm.nodes; bounds[key_drift(sm, k.k)] = sm.latent_step; end
        end
        rc = gauss_newton!(sm.graph; iters=sm.iters, tol=sm.tol, ftol=sm.ftol, verbose,
                           resolution_gtol, newton_tol=attempt == sm.restarts + 1 ? ONLINE_NEWTON_TOL : 0.0,
                           step_bounds=bounds, stationarity_scales)
        append!(all_hist, (merge(h, (attempt=attempt,)) for h in rc.hist))
        r = rc
        rc.converged && break
    end
    sm.n_solves += 1
    return merge(r, (hist=all_hist, attempts=attempts))
end

# the past's integrals of one slice's outputs, per piece
function _accumulate_slice!(sm::FixedLagSmoother, f::DynamicsFactor)
    o = last_solve(f)
    o === nothing && return nothing
    tw = [f.t0, f.t1]
    for (i, pc) in enumerate(sm.model.sys.pieces)
        ev = piece_events(sm.model.log.events, pc.id; cfg=pc.cfg)
        _accumulate!(sm.past[pc.id], (Tsurf=o.pieces[i].Tsurf, Tcore=o.pieces[i].Tcore), tw, ev.t_pull, sm.goals)
    end
    return nothing
end

"""
    marginalize_old!(sm) -> Vector{VarKey}

The nodes older than the lag marginalized, oldest first, each with the discrepancy variables at or before its
time that are not their sensor's newest; the slice leaving the window is added to the committed past. Returns
the keys removed.
"""
function marginalize_old!(sm::FixedLagSmoother)
    removed = VarKey[]
    while length(sm.nodes) - 1 > sm.lag                                # (not `lag + 1`: an infinite lag is typemax)
        to, ko = sm.nodes[1]
        tn, kn = sm.nodes[2]
        f = sm.dyn[(ko, kn)]
        residual(f, [variable_value(sm.graph, k) for k in vars(f)])            # the slice at the current values, for the past
        _accumulate_slice!(sm, f)
        out = [ko]
        if sm.split !== nothing                                                  # the node's moving copy leaves with it
            push!(out, key_drift(sm, ko.k))
            delete!(sm.drift_fac, (key_drift(sm, ko.k), key_drift(sm, kn.k)))
        end
        for (id, chain) in sm.disc
            while length(chain) > 1 && chain[1][1] <= to + 1e-9
                td, kd = chain[1]
                push!(out, kd); sm.disc_committed_t[id] = td
                popfirst!(chain)
                delete!(sm.disc_fac, (kd, chain[1][2]))
            end
        end
        marginalize!(sm.graph, out)
        delete!(sm.dyn, (ko, kn))
        filter!(o -> !(o.key_y in out), sm.obs)
        popfirst!(sm.nodes)
        sm.committed_t = to
        append!(removed, out)
    end
    return removed
end

"""
    _predicted_consistency(sm, fs) -> NamedTuple or nothing

The arrival's readings against the belief BEFORE their observation factors act: the joint covariance of the
variables the readings touch (the latents, the newest node, their discrepancies) from the graph with the new
factors set aside, the whitened prediction rows `H = ∂h/∂x / σ` and whitened innovations `ν = (y − h)/σ` at the
predicted point, the predicted innovation covariance `S = H C Hᵀ + I` and the normalized innovation squared
`νᵀ S⁻¹ ν` with its degrees of freedom (the uncensored readings). The linearization it takes of the newest
dynamics factor is the one the solve's first iteration uses. `nothing` when every reading is censored and
there is no ALS feed; with a feed an empty observation block still carries the frame's transition.
"""
function _predicted_consistency(sm::FixedLagSmoother, fs::AbstractVector{ObservationFactor})
    g = sm.graph
    live = [f for f in fs if !f.censored]
    isempty(live) && sm.als === nothing && return nothing
    keys = VarKey[]
    for f in live, k in vars(f)
        k in keys || push!(keys, k)
    end
    # The calibration needs the gain on the ENTIRE current frame, including an idle sensor's latest
    # discrepancy. Its cross-covariance with a reporting sensor is generally nonzero. A marginal over
    # just the observed variables gives the right NIS but silently zeroes those gain rows in _als_pieces.
    if sm.als !== nothing
        ky = sm.nodes[end][2]
        _, zkeys = _latents_at(sm, ky)
        for k in vcat(zkeys, [ky], [last(sm.disc[id])[2] for id in sm.als.disc_sensors if !isempty(sm.disc[id])])
            k in keys || push!(keys, k)
        end
    end
    # the observation factors set aside and put back at their own positions: the graph's factor order is the stacked
    # rows' order, and a solve's path follows its rounding
    pos = [findfirst(x -> x === f, g.factors) for f in fs]
    any(isnothing, pos) && throw(ArgumentError("an arrival's observation factor is not in the graph"))
    for i in sort(pos; rev=true); deleteat!(g.factors, i); end
    C, off = try
        joint_cov(g; keys)
    finally
        for i in sortperm(pos); insert!(g.factors, pos[i], fs[i]); end
    end
    n = size(C, 1); p = length(live)
    H = zeros(p, n); ν = zeros(p)
    for (i, f) in enumerate(live)
        vals = [variable_value(g, k) for k in vars(f)]
        for (k, Jk) in zip(vars(f), jacobians(f, vals))
            H[i, off[k]] .-= vec(Jk)                               # the whitened residual's Jacobian is −∂h/∂x / σ
        end
        ν[i] = residual(f, vals)[1]
    end
    S = Matrix(Symmetric(H * C * H' + I))
    sdev = sqrt.(max.(diag(S), 1e-12))
    return (nis=p == 0 ? NaN : dot(ν, S \ ν), dof=p, S=S, nu=ν, sdev=sdev, H=H, C=C, off=off, live=live)
end

# the accumulator's reading of one arrival in the feed's frame [latents; newest node; each discrepancy sensor's
# newest]: the transition into the frame (the latents still, the newest node from its predecessor through the
# newest dynamics factor's Jacobian at the predicted point, a discrepancy by its OU coefficient when its sensor
# sampled now, a first sample a fresh draw, an idle sensor's newest standing), the process covariance's
# derivative per group, the predicted covariance embedded (a sensor with no live discrepancy at its stationary
# prior), the standardized rows, innovations and covariance, the linear gain at the predicted point, and the
# white variances' derivative per measurement group
function _als_pieces(sm::FixedLagSmoother, feed::ALSFeed, pc, t_arr::Float64)
    acc = feed.acc; g = sm.graph
    n_z = length(sm.model.blk); n_y = sm.model.bl.n; n = acc.n
    rz = 1:n_z; ry = (n_z + 1):(n_z + n_y)
    kz = sm.key_z; ty, ky = sm.nodes[end]
    frame = Dict{VarKey,Vector{Int}}(ky => collect(ry))                       # each graph variable's columns in the frame (the latents in the block's full order)
    if sm.split === nothing
        frame[kz] = collect(rz)
    else
        frame[kz] = sm.split.static; frame[key_drift(sm, ky.k)] = sm.split.moving
    end
    for (j, id) in enumerate(feed.disc_sensors)
        chain = sm.disc[id]
        isempty(chain) || (frame[chain[end][2]] = [n_z + n_y + j])
    end
    F = Matrix(1.0I, n, n)
    dQ = [zeros(n, n) for _ in acc.groups]
    if length(sm.nodes) >= 2 && abs(ty - t_arr) < 1e-6
        kp = sm.nodes[end - 1][2]
        f = sm.dyn[(kp, ky)]
        c = _dyn_eval!(f, _factor_latents(f, g), variable_value(g, kp); jac=true)
        F[ry, rz] = c.Jz; F[ry, ry] = c.Jy
        for (i, D) in enumerate(_process_cov_sens(f, c.wy, feed.state_names))
            dQ[i][ry, ry] = D
        end
    end
    for (j, id) in enumerate(feed.disc_sensors)
        i = n_z + n_y + j
        chain = sm.disc[id]
        (isempty(chain) || abs(chain[end][1] - t_arr) > 1e-6) && continue
        gi = get(feed.ou_group, id, 0)
        if length(chain) == 1
            F[i, i] = 0.0
            # The first discrepancy is drawn from N(0, theta_ou), not a known initial condition.
            gi == 0 || (dQ[gi][i, i] = 1.0)
        else
            fac = sm.disc_fac[(chain[end - 1][2], chain[end][2])]
            F[i, i] = fac.ρ
            gi == 0 || (dQ[gi][i, i] = 1 - fac.ρ^2)
        end
    end
    P = zeros(n, n)
    for (ka, ra) in frame, (kb, rb) in frame
        (haskey(pc.off, ka) && haskey(pc.off, kb)) || continue
        P[ra, rb] = pc.C[pc.off[ka], pc.off[kb]]
    end
    for (j, id) in enumerate(feed.disc_sensors)
        isempty(sm.disc[id]) || continue
        i = n_z + n_y + j
        P[i, i] = sensor(sm.model.log, id).noise.ar_sd^2
    end
    p = pc.dof
    H = zeros(p, n)
    for (k, r) in frame
        haskey(pc.off, k) || continue
        H[:, r] = pc.H[:, pc.off[k]]
    end
    sdev = pc.sdev
    Hs = H ./ sdev; Ss = pc.S ./ (sdev * sdev'); es = pc.nu ./ sdev
    K = P * Hs' / Symmetric(Ss)
    ids = [s.id for s in sm.model.log.sensors]
    rowch = Int[findfirst(==(f.spec.id), ids) for f in pc.live]
    dR = [zeros(p, p) for _ in acc.groups]
    for (i, f) in enumerate(pc.live)
        gi = get(feed.white_group, f.spec.id, 0)
        gi == 0 || (dR[gi][i, i] = 1 / (f.σ^2 * sdev[i]^2))
    end
    return (F=F, dQ=dQ, H=Hs, S=Ss, K=K, e=es, rowch=rowch, dR=dR)
end

"""
    _residual_power(sm, fs, t_arr) -> NamedTuple

The whitened residuals at the accepted point, by kind, before the old nodes leave: the newest slice's dynamics
residual per state group as its sum of squares and its count (unit power per coordinate when the process noise
is right; the fit shrinks it by the group's leverage), every window slice's pooled likewise, this arrival's
observation residual squared per sensor (censored readings aside) and its OU residual squared per sensor, and
the graph's row and column counts (with the objective, the least-squares goodness of fit `2F/(rows − cols)`).
Every solve is cached at the accepted point: no new integration.
"""
function _residual_power(sm::FixedLagSmoother, fs::AbstractVector{ObservationFactor}, t_arr::Float64)
    g = sm.graph; bl = sm.model.bl
    groups = [(_als_group_name(bl, gname), rg) for (gname, rg) in bl.groups]
    dyn_new = Dict{String,Vector{Float64}}(); dyn_win = Dict{String,Vector{Float64}}()
    acc!(d, nm, ss, n) = (v = get!(d, nm, [0.0, 0.0]); v[1] += ss; v[2] += n)
    newest = sm.nodes[end][2]
    for f in values(sm.dyn)
        r = residual(f, [variable_value(g, k) for k in vars(f)])
        for (nm, rg) in groups
            ss = sum(abs2, view(r, rg)); n = length(rg)
            acc!(dyn_win, nm, ss, n)
            f.key_y1 == newest && acc!(dyn_new, nm, ss, n)
        end
    end
    obs = Dict{Symbol,Float64}()
    for f in fs
        f.censored && continue
        obs[f.spec.id] = residual(f, [variable_value(g, k) for k in vars(f)])[1]^2
    end
    ou = Dict{Symbol,Float64}()
    for (id, chain) in sm.disc
        (length(chain) >= 2 && abs(chain[end][1] - t_arr) < 1e-6) || continue
        fac = get(sm.disc_fac, (chain[end - 1][2], chain[end][2]), nothing)
        fac === nothing && continue
        ou[id] = residual(fac, [variable_value(g, fac.key_prev), variable_value(g, fac.key_next)])[1]^2
    end
    rows = sum(ndim(f) for f in g.factors; init=0)
    cols = sum(g.dims[k] for k in g.keys; init=0)
    return (dyn=dyn_new, dyn_window=dyn_win, obs=obs, ou=ou, rows=rows, cols=cols)
end

"""
    assimilate!(sm, recs; verbose=false) -> record

Readings into the window, the window solved, the old nodes marginalized: one arrival step. The record
carries the time, the sensors, each reading's innovation at the belief before the solve, the normalized
innovation squared of the uncensored readings against the predicted innovation covariance with its degrees of
freedom and the standardized innovations per sensor (`_predicted_consistency`), the solve's objective,
iterations and convergence, the newest node's belief (per piece the centre and surface temperatures with their
sds, the pit's), the latents' posterior and the wall time; on a committed arrival also the whitened residuals
at the accepted point by kind (`_residual_power`) and, when a node leaves the window, that node's belief as it
leaves (`past`: the smoothed past at the lag, `belief_at`'s fields without the joint covariance; at lag 0 it is
the previous node, re-solved once with the new readings before it goes). A committed arrival feeds the bench's
accumulator when the smoother carries one.
"""
function assimilate!(sm::FixedLagSmoother, recs::AbstractVector{MeasurementRecord}; verbose::Bool=false)
    t0 = time()
    isempty(recs) && throw(ArgumentError("nothing to assimilate"))
    # Validate the whole batch before the first graph mutation. The deep snapshot then makes solver failure
    # and any later exception transactional, including node/factor splitting for a delayed reading.
    for rec in recs
        sensor(sm.model.log, rec.sensor)
        isfinite(rec.t) && isfinite(rec.value) || throw(ArgumentError("a measurement must have a finite time and value"))
        rec.t >= sm.nodes[1][1] - 1e-9 || throw(ArgumentError("a sample at $(rec.t) s precedes the window's oldest node"))
    end
    if sm.als !== nothing
        ta = first(recs).t
        all(r -> r.t == ta, recs) || throw(ArgumentError("ALS requires one timestamp per arrival batch"))
        ta >= sm.nodes[end][1] && (sm.als.acc.n_updates == 0 || ta > sm.nodes[end][1]) ||
            throw(ArgumentError("ALS requires strictly increasing arrivals; batch simultaneous readings and disable ALS for delayed data"))
    end
    saved = _transaction_snapshot(sm)
    try
        fs = [add_measurement!(sm, r) for r in recs]
        innov = Dict(r.sensor => innovation(f, [variable_value(sm.graph, k) for k in vars(f)]) for (r, f) in zip(recs, fs))
        t_arr = maximum(x.t for x in recs)
        pc = _predicted_consistency(sm, fs)
        nis = pc === nothing ? NaN : pc.nis
        dof = pc === nothing ? 0 : pc.dof
        innov_std = pc === nothing ? Dict{Symbol,Float64}() : Dict(f.spec.id => pc.nu[i] / pc.sdev[i] for (i, f) in enumerate(pc.live))
        als = (sm.als !== nothing && pc !== nothing) ? _als_pieces(sm, sm.als, pc, t_arr) : nothing
        r = solve_window!(sm; verbose)
        if !r.converged
            _restore_transaction!(sm, saved)
            sm.n_solves += 1
            b = belief_at(sm)                                            # the belief as it stands (the previous node's); the record's time is the arrival's
            rec = merge(b, (t=t_arr, sensors=[x.sensor for x in recs], innovation=innov, nis=nis, dof=dof, innov_std=innov_std, F=r.F,
                            iters=length(r.hist), attempts=r.attempts, converged=false, committed=false, gradient=r.gradient, newton=r.newton_step, newton_sd=r.newton_sd, wall=time() - t0,
                            n_nodes=length(sm.nodes), resid=nothing, past=nothing))
            return _push_record!(sm, rec)
        end
        resid = _residual_power(sm, fs, t_arr)
        past = nothing
        if length(sm.nodes) - 1 > sm.lag                                  # a node leaves: its smoothed belief at the lag, before it goes
            bp = belief_at(sm; node=1)
            past = (t=bp.t, z=bp.z, z_cov=bp.z_cov, y=bp.y, y_cov=bp.y_cov, x=bp.x, x_cov=bp.x_cov, T_pit=bp.T_pit, pieces=bp.pieces)
        end
        marginalize_old!(sm)
        b = belief_at(sm)
        rec = merge(b, (t=t_arr, sensors=[x.sensor for x in recs], innovation=innov, nis=nis, dof=dof, innov_std=innov_std, F=r.F,
                        iters=length(r.hist), attempts=r.attempts, converged=true, committed=true, gradient=r.gradient, newton=r.newton_step, newton_sd=r.newton_sd, wall=time() - t0,
                        n_nodes=length(sm.nodes), resid=resid, past=past))
        # Finish graph/covariance work before mutating the accumulator: a failed belief extraction must
        # not leave calibration statistics for an arrival whose graph is rolled back.
        als === nothing || als_update!(sm.als.acc, als.F, als.dQ, als.H, als.S, als.K, als.e, als.rowch, t_arr; dR=als.dR)
        return _push_record!(sm, rec)
    catch
        _restore_transaction!(sm, saved)
        rethrow()
    end
end
assimilate!(sm::FixedLagSmoother, rec::MeasurementRecord; kw...) = assimilate!(sm, [rec]; kw...)

"""
    belief_at(sm; node=last) -> NamedTuple

The belief at a window node (the newest by default): the latents' mean and covariance, the node's belief
coordinates with their covariance, the joint covariance of the two, the physical state with its first-order
covariance, and per piece its centre and surface temperatures and the pit's, each `(mean, sd)`.
"""
function belief_at(sm::FixedLagSmoother; node::Int=length(sm.nodes))
    t, ky = sm.nodes[node]
    z, zkeys = _latents_at(sm, ky)
    C, off = joint_cov(sm.graph; keys=vcat(zkeys, [ky]))
    n_z = length(z)
    if sm.split !== nothing                                                     # the covariance in the full latents' order: static and moving interleaved as the block lists them
        perm = vcat(split_permutation(sm.split), (n_z + 1):(n_z + sm.model.bl.n))
        C = C[perm, perm]
    end
    rz, ry = 1:n_z, (n_z + 1):(n_z + sm.model.bl.n)
    y = variable_value(sm.graph, ky)
    bl = sm.model.bl
    x, Px = belief_physical_moments(y, C[ry, ry], bl)
    pieces = Dict{Symbol,NamedTuple}()
    for id in bl.sys.pieces
        pc = piece(sm.model.sys, id); rT = belief_index(bl, id, :T)
        if pc isa GridPiece
            ic = rT[pc.mesh.core]
            w = zeros(length(rT))
            for (node, area) in pc.mesh.boundary; w[node] += area / pc.mesh.area; end
            μs = dot(w, view(x, rT)); σs = sqrt(max(dot(w, view(Px, rT, rT) * w), 0.0))
            pieces[id] = (T_centre=(x[ic], sqrt(max(Px[ic, ic], 0.0))), T_surface=(μs, σs))
        else
            ic, is = first(rT), last(rT)
            pieces[id] = (T_centre=(x[ic], sqrt(max(Px[ic, ic], 0.0))), T_surface=(x[is], sqrt(max(Px[is, is], 0.0))))
        end
    end
    ip = first(belief_index(bl, :pit, :T_pit))
    return (t=t, z=copy(z), z_cov=C[rz, rz], y=copy(y), y_cov=C[ry, ry], zy_cov=C, x=x, x_cov=Px,
            T_pit=(x[ip], sqrt(max(Px[ip, ip], 0.0))), pieces=pieces)
end

"""
    BeliefSnapshot

The belief handed to a planner, model-independent: the time, the system and its layouts, the latents and the
belief state with their joint covariance, the physical state, the restart keywords (`belief_handoff`), the
past's integrals per piece to the snapshot's time (the committed past plus the window's slices) and the
provenance (the solves, the lag, the nodes, the records' count).
"""
struct BeliefSnapshot
    t::Float64
    sys::CookSystem
    bl::BeliefLayout
    blk::LatentBlock
    z::Vector{Float64}
    y::Vector{Float64}
    zy_cov::Matrix{Float64}
    x::Vector{Float64}
    x_cov::Matrix{Float64}
    handoff::NamedTuple
    past::Dict{Symbol,Past}
    provenance::NamedTuple
    rom::Any
end
BeliefSnapshot(t, sys, bl, blk, z, y, zy_cov, x, x_cov, handoff, past, provenance) =
    BeliefSnapshot(t, sys, bl, blk, z, y, zy_cov, x, x_cov, handoff, past, provenance, nothing)
function belief_snapshot(sm::FixedLagSmoother)
    b = belief_at(sm)
    past = Dict(id => Past(p.leth_s, p.leth_c, p.surf_num, p.surf_den, p.m0, p.t) for (id, p) in sm.past)
    for j in 1:(length(sm.nodes) - 1)                                     # the window's slices, at the current values
        f = sm.dyn[(sm.nodes[j][2], sm.nodes[j + 1][2])]
        residual(f, [variable_value(sm.graph, k) for k in vars(f)])
        o = last_solve(f)
        for (i, pc) in enumerate(sm.model.sys.pieces)
            ev = piece_events(sm.model.log.events, pc.id; cfg=pc.cfg)
            _accumulate!(past[pc.id], (Tsurf=o.pieces[i].Tsurf, Tcore=o.pieces[i].Tcore), [f.t0, f.t1], ev.t_pull, sm.goals)
        end
    end
    rom_provenance = sm.model.rom === nothing ? nothing :
                     sm.model.rom isa AbstractDict ? Dict(string(k) => getproperty(v, :fingerprint) for (k, v) in sm.model.rom) :
                     getproperty(sm.model.rom, :fingerprint)
    return BeliefSnapshot(b.t, sm.model.sys, sm.model.bl, sm.model.blk, b.z, b.y, b.zy_cov, b.x, b.x_cov, belief_handoff(b.x, sm.model.bl),
                          past, (n_solves=sm.n_solves, lag=sm.lag, n_nodes=length(sm.nodes), n_records=sm.record_count,
                                 retained_records=length(sm.records), committed_t=sm.committed_t,
                                 physics=physics_fingerprint(), rom=rom_provenance), sm.model.rom)
end

"""
    to_plan_state(snap, s::Schedule; piece=first) -> NamedTuple

The snapshot as the planner's `state` (`plan_solve`, `plan_predictive`: `k`, `t`, `ics`, `ignited`, `x`, `past`,
`lay`, `zx_cov`) for a system of one slab, on the schedule's grid (the snapshot's time must be one of its
points). The planner's latent vector is the block's first `N_PHYS + 1` entries (the physics and the area),
as `latent_block` orders them.
"""
function to_plan_state(snap::BeliefSnapshot, s::Schedule)
    n_pieces(snap.sys) == 1 || throw(ArgumentError("the planner's state is a one-slab contract; the system has $(n_pieces(snap.sys)) pieces (the multi-piece planner is not built)"))
    k = findfirst(τ -> abs(τ - snap.t) < 1e-6, s.t)
    k === nothing && throw(ArgumentError("the snapshot's time $(snap.t) s is not a point of the schedule's grid"))
    id = snap.sys.pieces[1].id
    h = snap.handoff
    ics = merge(h.pieces[1], h.pit)
    lay = FilterLayout(snap.sys.pieces[1].cfg.nx, length(snap.blk), 0)
    return (k=k, t=snap.t, ics=ics, ignited=h.ignited, x=snap.y, past=snap.past[id], lay=lay, zx_cov=snap.zy_cov)
end
