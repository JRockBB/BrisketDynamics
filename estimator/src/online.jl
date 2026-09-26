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
       BeliefSnapshot, belief_snapshot, to_plan_state, window_nodes, key_params, key_node, key_disc,
       SmootherCheckpoint, smoother_checkpoint, restore_checkpoint, restore_checkpoint!, backend_fingerprint,
       update_execution!

# Adaptive ODE solves cease to give a repeatable descent signal below roughly five parts in ten thousand
# in the solver's diagonal Gauss--Newton coordinate. This is not a general convergence tolerance: it is
# consulted only after three complete Armijo failures at one point, and the stricter `tol` still governs
# ordinary stationary solves. The two-piece adversarial gate leaves a resolvable 7.8e-4 displacement at
# arrival 12 (which must restart) and a repeatably unresolvable 4.63e-4 displacement at arrival 13.
const ONLINE_ODE_RESOLUTION_GTOL = 5e-4

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
    goals::PlanGoals
    past::Dict{Symbol,Past}
    committed_t::Float64
    records::Vector{NamedTuple}
    record_limit::Int
    record_count::Int
    n_solves::Int
end
key_params(sm::FixedLagSmoother) = sm.key_z
key_node(sm::FixedLagSmoother, k::Int) = VarKey(:state, sm.cook, k)
key_disc(sm::FixedLagSmoother, id::Symbol, j::Int) = VarKey(:disc, (sm.cook, id), j)
window_nodes(sm::FixedLagSmoother) = copy(sm.nodes)

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
    online_init(model; t0=model.log.t_grid[1], lag=3, iters=40, restarts=1, tol=1e-4, ftol=1e-2, goals=PlanGoals(), graph=FactorGraph(), cook=:cook, hierarchy=nothing) -> FixedLagSmoother

The belief before any sample: the latents at each family's moments in its coordinate (the filter's start), the
first node at `t0` with the recipe's state (the system's initial state at the prior centre, the ignited
fraction from a0) at the `StateNoise` initial widths mapped into the belief's coordinates. `ftol`, the objective's
drop under which a solve stops, is a hundredth of a nat (the RxInfer filter stops at a twentieth). `graph` and `cook`
let several cooks share one graph under distinct keys (the offline backend); `hierarchy = (idx, key_pop, tau)`
ties the latents `idx` of this cook to the population variable `key_pop` with the between-cook sds `tau`
instead of giving them the family's prior.
"""
function online_init(model::SystemModel; t0::Real=model.log.t_grid[1], lag::Int=3, iters::Int=40, restarts::Int=1, tol::Float64=1e-4,
                     ftol::Float64=1e-2, latent_step::Float64=0.25, goals::PlanGoals=PlanGoals(), graph::FactorGraph=FactorGraph(),
                     cook::Symbol=:cook, hierarchy=nothing, record_limit::Int=256)
    lag >= 0 || throw(ArgumentError("the lag counts nodes kept behind the newest; it cannot be negative"))
    iters >= 0 || throw(ArgumentError("iters cannot be negative"))
    restarts >= 0 || throw(ArgumentError("restarts cannot be negative"))
    record_limit >= 0 || throw(ArgumentError("record_limit cannot be negative"))
    g = graph
    blk = model.blk; bl = model.bl
    z0 = [z_moments(p)[1] for p in blk.sites]
    sd0 = [sqrt(z_moments(p)[2]) for p in blk.sites]
    key_z = VarKey(:params, cook, 0)
    add_variable!(g, key_z, z0)
    if hierarchy === nothing
        add_factor!(g, SitePriorFactor(key_z, blk.sites; source="$(cook): the latents' exact transformed-family priors"))
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
                            ObservationFactor[], lag, iters, restarts, tol, ftol, latent_step, goals, past, Float64(t0), NamedTuple[],
                            record_limit, 0, 0)
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
                       "plan_system.jl", "rom.jl")
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
    f = DynamicsFactor(sm.model, sm.key_z, ka, kb, ta, tb; first, events=sensor_events(sm.model.log, ta, tb))
    add_factor!(sm.graph, f); sm.dyn[(ka, kb)] = f
    return f
end
function _remove_dyn!(sm::FixedLagSmoother, ka::VarKey, kb::VarKey)
    f = pop!(sm.dyn, (ka, kb))
    remove_factor!(sm.graph, f)
    return f
end

"""
    _node_at!(sm, t) -> key

The state node at time `t`: the existing one, or a new one inserted in time order (between two nodes the
dynamics factor is split; after the newest a new slice is appended), its value the propagation from the
node before it. An error before the window's oldest node.
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
    if i < length(sm.nodes)
        tn, kn_next = sm.nodes[i + 1]
        old = _remove_dyn!(sm, kp, kn_next)
        y, _ = propagate(sm.model, variable_value(sm.graph, kp), variable_value(sm.graph, sm.key_z), tp, t; first_slice=old.first)
        add_variable!(sm.graph, kn, y)
        _add_dyn!(sm, kp, tp, kn, t; first=old.first)
        _add_dyn!(sm, kn, t, kn_next, tn; first=false)
    else
        fs = kp.k == 1                                               # the slice from the initial node reads a0
        y, _ = propagate(sm.model, variable_value(sm.graph, kp), variable_value(sm.graph, sm.key_z), tp, t; first_slice=fs)
        add_variable!(sm.graph, kn, y)
        _add_dyn!(sm, kp, tp, kn, t; first=fs)
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
    f = ObservationFactor(sm.model, spec, rec, sm.key_z, ky, kd)
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
        rc = gauss_newton!(sm.graph; iters=sm.iters, tol=sm.tol, ftol=sm.ftol, verbose,
                           resolution_gtol,
                           step_bounds=Dict(sm.key_z => sm.latent_step), stationarity_scales)
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
    assimilate!(sm, recs; verbose=false) -> record

Readings into the window, the window solved, the old nodes marginalized: one arrival step. The record
carries the time, the sensors, each reading's innovation at the belief before the solve, the solve's
objective, iterations and convergence, the newest node's belief (per piece the centre and surface
temperatures with their sds, the pit's), the latents' posterior and the wall time.
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
    saved = _transaction_snapshot(sm)
    try
        fs = [add_measurement!(sm, r) for r in recs]
        innov = Dict(r.sensor => innovation(f, [variable_value(sm.graph, k) for k in vars(f)]) for (r, f) in zip(recs, fs))
        r = solve_window!(sm; verbose)
        if !r.converged
            _restore_transaction!(sm, saved)
            sm.n_solves += 1
            b = belief_at(sm)
            rec = (t=maximum(x.t for x in recs), sensors=[x.sensor for x in recs], innovation=innov, F=r.F,
                   iters=length(r.hist), attempts=r.attempts, converged=false, committed=false, gradient=r.gradient, wall=time() - t0,
                   n_nodes=length(sm.nodes), b...)
            return _push_record!(sm, rec)
        end
        marginalize_old!(sm)
        b = belief_at(sm)
        rec = (t=maximum(x.t for x in recs), sensors=[x.sensor for x in recs], innovation=innov, F=r.F,
               iters=length(r.hist), attempts=r.attempts, converged=true, committed=true, gradient=r.gradient, wall=time() - t0,
               n_nodes=length(sm.nodes), b...)
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
    C, off = joint_cov(sm.graph; keys=[sm.key_z, ky])
    rz, ry = off[sm.key_z], off[ky]
    z = variable_value(sm.graph, sm.key_z); y = variable_value(sm.graph, ky)
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
