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
# in the solver's scaled Gauss--Newton coordinate. This is not a general convergence tolerance: it is
# consulted only after three complete Armijo failures at one point, and the stricter `tol` still governs
# ordinary stationary solves. This envelope was originally measured with diagonal scaling;
# the coupled displacement check is stricter in correlated directions and must pass the regression gates.
const ONLINE_ODE_RESOLUTION_GTOL = 5e-4
# the joint stationarity rescue of a stalled solve (gauss_newton!'s `newton_tol`), on the final attempt only: a point three
# complete line searches could not leave, whose undamped step at an exact relinearization is under this many posterior sds,
# is accepted (2026-09-26: the smoother lost one arrival in thirty at a diagonal read of 5e-4 with the joint step at 1e-4 sd)
const ONLINE_NEWTON_TOL = 1e-2
# The fresh joint certificate uses the caller's same posterior-accuracy budget.
# A hidden 1e-3 cap spent 18 iterations approaching the integration noise floor
# despite the declared 1e-2 budget. The joint norm includes correlations and
# bounds every linear contrast's local displacement in its posterior SD.
# the doublings an accepted full step may take when it beat the model (gauss_newton!'s `expand`): the two-probe control's
# seed 3012 (2026-09-30) crawled its 80 iterations at gain 2 along the probe depths' valley and was refused
const ONLINE_EXPAND = Ref(4)
"""
    SHIPPED_STREAM_CONFIG

The one shipped streaming configuration — the real problem the estimator's evidence is earned on: slice
attachment at lag 3, variance-ruled node placement with the 600 s age cap, and the replays' integration
accuracies. Drivers pass these EXPLICITLY (`online_init`'s own defaults stay neutral) so no bench, replay
or pilot silently drifts from the benchmarked configuration. When the window-lock mitigations are adopted,
`lag_max`/`prior_relin` change HERE and every driver follows.
"""
const SHIPPED_STREAM_CONFIG = (attach=:slice, lag=3, lag_max=3, node_ratio=1.0, max_slice_s=600.0,
                               prior_relin=Inf, slice_g_refresh=true,
                               abstol=1e-12, reltol=1e-10, floor_abstol=1e-10, floor_reltol=1e-8)
# profiling: seconds accumulated per phase of an update (diagnostic only; reset by the caller)
const SLICE_TIMERS = Dict{Symbol,Float64}()
_tick!(k::Symbol, dt) = (SLICE_TIMERS[k] = get(SLICE_TIMERS, k, 0.0) + dt)

# A dynamics factor spanning several execution-grid intervals accumulates more adaptive-step quantization
# than the one-grid-step factors on which ONLINE_ODE_RESOLUTION_GTOL was measured. Scale that measured
# envelope only as sqrt(number of grid intervals), and never beyond the dynamics factors' own relinearization
# radius: a point outside that radius has a resolvable fresh local model rather than a numerical-floor excuse.
function _online_resolution_gtol(sm)
    requested = min(1.0, sm.tol / 1e-4)
    isempty(sm.dyn) && return requested * ONLINE_ODE_RESOLUTION_GTOL
    Δgrid = minimum(diff(sm.model.log.t_grid))
    Δfactor = maximum(f.t1 - f.t0 for f in values(sm.dyn))
    relin_cap = minimum(f.relin_tol for f in values(sm.dyn))
    return requested * min(relin_cap, ONLINE_ODE_RESOLUTION_GTOL * sqrt(max(1.0, Δfactor / Δgrid)))
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
    attach::Symbol                                          # :node (a state node per reading time) or :slice (readings through the physics from their slice's node)
    slices::Dict{VarKey,SliceObservationFactor}             # :slice: per anchor node its readings' factor
    slice_recs::Dict{VarKey,Vector{MeasurementRecord}}      # :slice: per anchor node its uncensored readings
    slice_G::Dict{Tuple{Symbol,Float64},NamedTuple{(:g, :z, :y0),Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}}}   # :slice: per reading (sensor, time) its sensitivity with the (latents, anchor) it was taken at; refreshed at the relin tolerances when `slice_g_refresh`
    node_ratio::Float64                                     # :slice: a new node when the process variance a reading sees exceeds this times its measurement variance
    max_slice_s::Float64                                    # :slice: an optional age cap; Inf preserves variance-only placement
    slice_rd::Dict{Tuple{Symbol,Float64},SliceReading}      # :slice: per reading its compiled form
    lag_max::Int                                            # the window may hold up to this many nodes behind the newest while the latents still move
    settle_tol::Float64                                     # a node may leave once the latents moved less than this many posterior sds since its first solve
    z_sd::Vector{Float64}                                   # the latents' posterior sds at the last committed update (for the settle test)
    z_snap::Dict{VarKey,Vector{Float64}}                    # per window node the latents' estimate at its first committed solve
    prior_relin::Float64                                    # rebuild the marginal prior when the latents moved this many sds since it was built (Inf: never)
    hist_factors::Vector{Factor}                            # every nonlinear factor marginalized out of the window, kept for rebuilding the prior
    hist_vals::Dict{VarKey,Vector{Float64}}                 # the marginalized variables' values when they left
    z_prior_lin::Vector{Float64}                            # the latents (the graph's static block) the marginal prior was last built at
    n_prior_relin::Int                                      # how many times the prior was rebuilt
    seen::Dict{Tuple{Symbol,Float64},Float64}               # committed readings (sensor, time) -> value: the duplicate policy's memory, pruned as the window moves
    retry::Vector{Tuple{MeasurementRecord,Int}}             # readings whose solve did not commit, with the attempts spent: re-attempted on the next arrival
    n_dropped::Int                                          # readings given up on: a conflicting duplicate, retries exhausted, or fallen behind the window
    slice_g_refresh::Bool                                   # recompute a slice reading's sensitivity (and so its slice's Σ) when the belief left its reference by the relin tolerances
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
    sm.model = SystemModel(old.sys, log, old.blk; sn=old.sn, abstol=old.abstol, reltol=old.reltol, rom=old.rom,
                           floor_abstol=old.floor_abstol, floor_reltol=old.floor_reltol)
    return sm
end

"""
    online_init(model; t0=model.log.t_grid[1], lag=3, iters=40, restarts=1, tol=1e-4, ftol=1e-2, goals=PlanGoals(), graph=FactorGraph(), cook=:cook, hierarchy=nothing,
                als=nothing, relin_tol=1e-3, relin_tol_state=relin_tol, drift=nothing) -> FixedLagSmoother

The belief before any sample: the latents at each family's moments in its coordinate (the filter's start), the
first node at `t0` with the recipe's state (the system's initial state at the prior centre, the ignited
fraction from a0) at the `StateNoise` initial widths mapped into the belief's coordinates. `ftol`, the objective
drop prompting a fresh stationarity check, is a hundredth of a nat. `graph` and `cook`
let several cooks share one graph under distinct keys (the offline backend); `hierarchy = (idx, key_pop, tau)`
ties the latents `idx` of this cook to the population variable `key_pop` with the between-cook sds `tau`
instead of giving them the family's prior. `als` is the bench's accumulator (`ALSFeed`), fed after every committed
arrival. `relin_tol` and `relin_tol_state` are the dynamics factors' relinearization tolerances (the latents' move,
the node's move). `drift = (moving, sd_per_hour, jumps)` makes the latents `moving` a copy per window node tied to
the previous node's by a random walk of `sd_per_hour` (each in its coordinate) with `jumps = [(index, t, sd)]` added
on the slice crossing `t`; the rest stay one static block (`LatentSplit`). A cook with a hierarchy cannot drift.
For `attach=:slice`, `max_slice_s` optionally creates a node at the first reading at least that long after
the current anchor. It bounds dense slice factors when process rates are tiny; `Inf` (the default) disables it.
`tol` also scales the coordinate-resolution acceptance limit. It does not change the model's integration
settings. For stricter optimization, request more accurate `abstol`/`reltol` explicitly in `SystemModel`,
holding `floor_abstol`/`floor_reltol` fixed to preserve the statistical model.
The separate `newton_tol` keyword of `assimilate!` or `solve_window!` sets the numerical-resolution
exception in posterior standard deviations; it is not inferred from coordinate units.
"""
function online_init(model::SystemModel; t0::Real=model.log.t_grid[1], lag::Int=3, iters::Int=40, restarts::Int=1, tol::Float64=1e-4,
                     ftol::Float64=1e-2, latent_step::Float64=0.25, goals::PlanGoals=PlanGoals(), graph::FactorGraph=FactorGraph(),
                     cook::Symbol=:cook, hierarchy=nothing, record_limit::Int=256, als::Union{Nothing,ALSFeed}=nothing,
                     relin_tol::Float64=1e-3, relin_tol_state::Float64=relin_tol, drift=nothing, attach::Symbol=:node, node_ratio::Float64=1.0,
                     lag_max::Int=lag, settle_tol::Float64=0.1, prior_relin::Float64=Inf, max_slice_s::Float64=Inf,
                     slice_g_refresh::Bool=true)
    attach in (:node, :slice) || throw(ArgumentError("attach must be :node or :slice, not $attach"))
    attach === :slice && als !== nothing && throw(ArgumentError("the ALS accumulator reads one node per arrival; it is not built for :slice"))
    lag >= 0 || throw(ArgumentError("the lag counts nodes kept behind the newest; it cannot be negative"))
    iters >= 0 || throw(ArgumentError("iters cannot be negative"))
    restarts >= 0 || throw(ArgumentError("restarts cannot be negative"))
    record_limit >= 0 || throw(ArgumentError("record_limit cannot be negative"))
    max_slice_s > 0 || throw(ArgumentError("max_slice_s must be positive (Inf disables the age cap)"))
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
                            record_limit, 0, 0, als, split, drift_q, drift_jumps, Dict{Tuple{VarKey,VarKey},HierarchyFactor}(),
                            attach, Dict{VarKey,SliceObservationFactor}(), Dict{VarKey,Vector{MeasurementRecord}}(),
                            Dict{Tuple{Symbol,Float64},NamedTuple{(:g, :z, :y0),Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}}}(),
                            node_ratio, max_slice_s, Dict{Tuple{Symbol,Float64},SliceReading}(),
                            max(lag_max, lag), settle_tol, Float64[], Dict{VarKey,Vector{Float64}}(),
                            prior_relin, Factor[], Dict{VarKey,Vector{Float64}}(), Float64[], 0,
                            Dict{Tuple{Symbol,Float64},Float64}(), Tuple{MeasurementRecord,Int}[], 0, slice_g_refresh)
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
    slice_state = IdDict{SliceObservationFactor,NamedTuple}()
    for f in values(sm.slices)
        slice_state[f] = (cache=deepcopy(f.cache), n_solves=f.n_solves, n_jacobians=f.n_jacobians)
    end
    return (
        graph_keys=copy(sm.graph.keys),
        graph_dims=copy(sm.graph.dims),
        graph_values=Dict(k => copy(v) for (k, v) in sm.graph.values),
        graph_factors=copy(sm.graph.factors),
        dyn_state,
        slice_state,
        slices=copy(sm.slices),
        slice_recs=Dict(k => copy(v) for (k, v) in sm.slice_recs),
        slice_G=copy(sm.slice_G),
        slice_rd=copy(sm.slice_rd),
        z_sd=copy(sm.z_sd), z_snap=copy(sm.z_snap),
        hist_factors=copy(sm.hist_factors), hist_vals=copy(sm.hist_vals), z_prior_lin=copy(sm.z_prior_lin), n_prior_relin=sm.n_prior_relin,
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
    for (f, state) in saved.slice_state
        f.cache = state.cache
        f.n_solves = state.n_solves
        f.n_jacobians = state.n_jacobians
    end
    sm.slices = saved.slices
    sm.slice_recs = saved.slice_recs
    sm.slice_G = saved.slice_G
    sm.slice_rd = saved.slice_rd
    sm.z_sd = saved.z_sd; sm.z_snap = saved.z_snap
    sm.hist_factors = saved.hist_factors; sm.hist_vals = saved.hist_vals; sm.z_prior_lin = saved.z_prior_lin; sm.n_prior_relin = saved.n_prior_relin
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
    for (_, ka) in sm.nodes
        _link_peer!(sm, ka)
    end
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
const BACKEND_FILES = ("factors.jl", "slice_obs.jl", "system_state.jl", "semantic.jl", "online.jl", "offline.jl",
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
    touched = Set{VarKey}()
    ky = if sm.attach === :slice
        # A node-backed reading (in particular a censored one) cuts the same
        # process and OU bridges as a variance/event node. Inserting only the
        # state left the preceding factor open and lost its endpoint correlation.
        for te in _event_times(sm.model.log, sm.nodes[end][1], rec.t)
            _slice_new_node!(sm, te, touched)
        end
        any(abs(tn - rec.t) < 1e-6 for (tn, _) in sm.nodes) ?
            _node_at!(sm, rec.t) : _slice_new_node!(sm, rec.t, touched)
    else
        _node_at!(sm, rec.t)
    end
    kd = _disc_at!(sm, spec, rec.t)
    f = ObservationFactor(sm.model, spec, rec, sm.key_z, ky, kd; key_m=_key_m(sm, ky), split=sm.split)
    add_factor!(sm.graph, f); push!(sm.obs, f)
    if !isempty(touched)
        for (_, ka) in sm.nodes
            ka in touched && _rebuild_slice!(sm, ka)
        end
    end
    return f
end

# ---- readings through the physics from their slice's node (attach = :slice; slice_obs.jl) --------------------------------

"The times of the log's lid-open events in (ta, tb]: each one a state node, since an event is information in itself."
function _event_times(log::CookLog, ta::Float64, tb::Float64)
    ts = Float64[]
    for e in log.events
        if e.kind === :spritz
            for tp in _spritz_pulse_times(log, e)
                ta < tp <= tb && push!(ts, tp)
            end
        else
            ta < e.t <= tb && push!(ts, Float64(e.t))
        end
    end
    return sort(unique(ts))
end
# a spritz train's pulses, as `sensor_events` counts them: every `dt` from its start until its piece's wrap or pull
function _spritz_pulse_times(log::CookLog, e)
    stop = minimum((x.t for x in log.events if x.target === e.target && x.kind in (:wrap, :pull)); init=Inf)
    dt = Float64(get(e.params, :dt, 3600.0))
    ts = Float64[]; τ = Float64(e.t)
    while τ <= stop && isfinite(τ)
        push!(ts, τ); τ += dt
    end
    return ts
end

"The belief state carried from node `ka` (at its current value) to each of `times`, one solve."
function _carried(sm::FixedLagSmoother, ka::VarKey, ta::Float64, times::AbstractVector{Float64})
    z, _ = _latents_at(sm, ka)
    y0 = variable_value(sm.graph, ka)
    ts = sort(unique([t for t in times if t > ta + 1e-9]))
    S = propagate_states(sm.model, y0, z, ta, ts; first_slice=ka.k == 1)
    return Dict(t => (t > ta + 1e-9 ? S[:, findfirst(==(t), ts)] : copy(y0)) for t in times), z
end

"A reading's compiled form, made once per reading (the observable compiled once, not on every rebuild of its slice)."
function _slice_reading(sm::FixedLagSmoother, r::MeasurementRecord)
    get!(sm.slice_rd, (r.sensor, r.t)) do
        SliceReading(sm.model, sensor(sm.model.log, r.sensor), r)
    end
end

"Rebuild the slice factor at anchor node `ka` from its readings (closed when a later node exists), replacing the old one."
function _rebuild_slice!(sm::FixedLagSmoother, ka::VarKey)
    old = pop!(sm.slices, ka, nothing)
    old === nothing || remove_factor!(sm.graph, old)
    recs = get(sm.slice_recs, ka, MeasurementRecord[])
    isempty(recs) && (_link_peer!(sm, ka); return nothing)
    i = findfirst(x -> x[2] == ka, sm.nodes)
    ta = sm.nodes[i][1]
    closed = i < length(sm.nodes)
    kb, tb = closed ? (sm.nodes[i + 1][2], sm.nodes[i + 1][1]) : (nothing, NaN)
    readings = [_slice_reading(sm, r) for r in recs]
    # A sensitivity is recomputed when the belief has left the point it was taken at by the same
    # tolerances that retake a slice's Jacobian (`_slice_eval!`): a G frozen at the prior's first
    # prediction misstates the slice's Σ for the whole window once the latents move.
    zka, _ = _latents_at(sm, ka)
    y0ka = variable_value(sm.graph, ka)
    stale(e) = sm.slice_g_refresh &&
               (norm(zka .- e.z, Inf) > sm.relin_tol || norm(y0ka .- e.y0, Inf) > sm.relin_tol_state)
    need = [r for r in readings if !haskey(sm.slice_G, (r.spec.id, r.t)) || stale(sm.slice_G[(r.spec.id, r.t)])]
    if !isempty(need)
        Y, z = _carried(sm, ka, ta, unique([r.t for r in need]))
        for r in need
            sm.slice_G[(r.spec.id, r.t)] = (g=slice_sensitivity(sm.model, r, Y[r.t], z), z=copy(zka), y0=copy(y0ka))
        end
    end
    G = reduce(vcat, [permutedims(sm.slice_G[(r.spec.id, r.t)].g) for r in readings])
    kd0 = Union{Nothing,VarKey}[_disc_at!(sm, r.spec, ta) for r in readings]
    kd1 = Union{Nothing,VarKey}[closed ? _disc_at!(sm, r.spec, tb) : nothing for r in readings]
    f = SliceObservationFactor(sm.model, readings, sm.key_z, ka, ta, ka.k == 1; key_y1=kb, t1=tb, kd0, kd1,
                               key_m=_key_m(sm, ka), split=sm.split, G, relin_tol=sm.relin_tol, relin_tol_state=sm.relin_tol_state)
    add_factor!(sm.graph, f); sm.slices[ka] = f
    _link_peer!(sm, ka)
    return f
end
# the dynamics factor leaving node `ka` reads the closed slice at `ka` (one solve, one Jacobian for both), or its own
function _link_peer!(sm::FixedLagSmoother, ka::VarKey)
    i = findfirst(x -> x[2] == ka, sm.nodes)
    (i === nothing || i == length(sm.nodes)) && return nothing
    kb = sm.nodes[i + 1][2]
    d = get(sm.dyn, (ka, kb), nothing)
    d === nothing && return nothing
    p = get(sm.slices, ka, nothing)
    d.peer[] = (p !== nothing && p.key_y1 == kb) ? p : nothing
    return nothing
end

"""
    _add_slice_readings!(sm, recs) -> Set of anchor keys touched

Uncensored readings into the window through the physics, in time order: a node at every lid-open event since the newest
node; a new node at a reading when the process noise it would see from the newest node (its channel's sensitivity through
the state's process covariance over the gap) exceeds `node_ratio` times its own measurement variance (the white floor and
the OU discrepancy's), or its age reaches `max_slice_s`; otherwise it joins the newest node's open slice. A delayed reading joins the closed slice it falls
in. Every slice the batch touched is rebuilt once at the end. The node rule's sensitivities come from ONE solve from the
newest node over the batch's times.

`node_times` MIRRORS a structure decided elsewhere instead: exactly those nodes are made (each once) and the rule is not
consulted. The conditional backend passes its anchor's new node times so every quadrature component holds the same
nodes, slices and factor keys; the rule reads the smoother's own estimate and would place each component's nodes apart.
"""
function _add_slice_readings!(sm::FixedLagSmoother, recs::AbstractVector{MeasurementRecord}; node_times=nothing)
    touched = Set{VarKey}()
    recs = sort(collect(recs); by=r -> r.t)
    if node_times !== nothing
        for te in sort(collect(Float64, node_times))
            any(x -> abs(x[1] - te) <= 1e-9, sm.nodes) || _slice_new_node!(sm, te, touched)
        end
    else
        tn0, kn0 = sm.nodes[end]
        ahead = unique([r.t for r in recs if r.t > tn0 + 1e-9])
        Y, z = isempty(ahead) ? (Dict{Float64,Vector{Float64}}(), nothing) : _carried(sm, kn0, tn0, ahead)
        for rec in recs
            spec = sensor(sm.model.log, rec.sensor)
            tn, _ = sm.nodes[end]
            rec.t > tn + 1e-9 || continue
            for te in _event_times(sm.model.log, tn, rec.t)
                _slice_new_node!(sm, te, touched)
            end
            tn, _ = sm.nodes[end]
            if rec.t > tn + 1e-9
                g = slice_sensitivity(sm.model, _slice_reading(sm, rec), Y[rec.t], z)
                Q = belief_process_cov(sm.model.bl, sm.model.sn, (rec.t - tn) / 3600)
                v_meas = spec.noise.white_sd^2 + (has_discrepancy(spec.noise) ? spec.noise.ar_sd^2 : 0.0)
                (rec.t - tn >= sm.max_slice_s || dot(g, Q * g) > sm.node_ratio * v_meas) && _slice_new_node!(sm, rec.t, touched)
            end
        end
    end
    for rec in recs
        ka = sm.nodes[findlast(x -> x[1] <= rec.t + 1e-9, sm.nodes)][2]
        push!(get!(sm.slice_recs, ka, MeasurementRecord[]), rec)
        push!(touched, ka)
    end
    for ka in touched
        any(x -> x[2] == ka, sm.nodes) && _rebuild_slice!(sm, ka)
    end
    return touched
end
# a new node after the newest (or inside the window): the slice it cuts marked for rebuilding, its readings after the node moved
function _slice_new_node!(sm::FixedLagSmoother, t::Float64, touched::Set{VarKey})
    i = findlast(x -> x[1] < t - 1e-9, sm.nodes)
    ka = sm.nodes[i][2]
    kn = _node_at!(sm, t)
    moved = [r for r in get(sm.slice_recs, ka, MeasurementRecord[]) if r.t >= t - 1e-9]
    if !isempty(moved)
        filter!(r -> r.t < t - 1e-9, sm.slice_recs[ka])
        append!(get!(sm.slice_recs, kn, MeasurementRecord[]), moved)
        push!(touched, kn)
    end
    push!(touched, ka)
    return kn
end

function _whitened_innovations(ν, S, ids, times)
    isempty(ν) && return NamedTuple[]
    # A batch's marginally standardized rows are correlated. Whiten in
    # chronological order to expose sequential conditional innovations.
    order = sortperm(eachindex(ν); by=i -> (times[i], string(ids[i])))
    w = cholesky(Symmetric(S[order, order])).L \ ν[order]
    return [(t=Float64(times[i]), sensor=ids[i], value=w[j]) for (j, i) in enumerate(order)]
end

"""
    _predicted_consistency_slice(sm, added; exclude=Factor[]) -> NamedTuple

The arrival's readings against the belief before them, in the slice factors they joined: per slice, the new readings'
innovation conditioned on the slice's earlier readings' (their noise is joint), `ν = v_new − μ_new − K (v_old − μ_old)`
with `K = Σ_no Σ_oo⁻¹`, its rows `H = ∂μ_new − K ∂μ_old` over the factor's variables, and `S = H C Hᵀ + Σ_nn − K Σ_on`
with `C` the joint covariance of those variables in the graph holding the slices WITHOUT the new readings. Summed over the
slices the arrival touched. `exclude` holds other observation factors from this same arrival, including capped
readings: they must not inform the pre-update covariance. `white` retains every innovation, whitened in
chronological order within the batch.
"""
function _predicted_consistency_slice(sm::FixedLagSmoother, added::Vector{Tuple{VarKey,Vector{MeasurementRecord}}};
                                      exclude::AbstractVector{<:Factor}=Factor[])
    g = sm.graph
    parts = NamedTuple[]
    for (ka, newrecs) in added
        f = sm.slices[ka]
        isnew = [any(r -> r.sensor === rd.spec.id && abs(r.t - rd.t) < 1e-9 && r.value == rd.value, newrecs) for rd in f.readings]
        N = findall(isnew); O = findall(.!isnew)
        push!(parts, (f=f, N=N, O=O, fo=_slice_subset(f, O)))
    end
    swapped = Tuple{Int,Factor,Union{Nothing,Factor}}[
        (something(findfirst(x -> x === pt.f, g.factors)), pt.f, pt.fo) for pt in parts]
    for f in exclude
        pos = findfirst(x -> x === f, g.factors)
        pos === nothing && throw(ArgumentError("an excluded arrival factor is not in the graph"))
        any(s -> s[1] == pos, swapped) && throw(ArgumentError("an arrival factor cannot be excluded twice"))
        push!(swapped, (pos, f, nothing))
    end
    keys = VarKey[]
    for pt in parts, k in vars(pt.f)
        k in keys || push!(keys, k)
    end
    for (pos, f, fo) in sort(swapped; by=first, rev=true)
        fo === nothing ? deleteat!(g.factors, pos) : (g.factors[pos] = fo)
    end
    C, off = try
        joint_cov(g; keys)
    finally
        for (pos, f, fo) in sort(swapped; by=first)
            fo === nothing ? insert!(g.factors, pos, f) : (g.factors[pos] = f)
        end
    end
    ν = Float64[]; Hs = Matrix{Float64}[]; Ns = Matrix{Float64}[]; ids = Symbol[]; times = Float64[]
    for pt in parts
        f = pt.f; N, O = pt.N, pt.O
        vals = [variable_value(g, k) for k in vars(f)]
        μ, Jμ = _slice_predict(f, vals; jac=true)
        v = _slice_values(f)
        Jfull = zeros(length(f.readings), size(C, 1))
        for (k, J) in zip(vars(f), Jμ)
            Jfull[:, off[k]] .+= J
        end
        if isempty(O)
            append!(ν, v[N] .- μ[N]); push!(Hs, Jfull[N, :]); push!(Ns, f.Σ[N, N])
        else
            Kg = f.Σ[N, O] / Symmetric(f.Σ[O, O])
            append!(ν, (v[N] .- μ[N]) .- Kg * (v[O] .- μ[O]))
            push!(Hs, Jfull[N, :] .- Kg * Jfull[O, :])
            push!(Ns, f.Σ[N, N] .- Kg * f.Σ[O, N])
        end
        append!(ids, [f.readings[i].spec.id for i in N])
        append!(times, [f.readings[i].t for i in N])
    end
    H = reduce(vcat, Hs)
    Rn = zeros(length(ν), length(ν)); o = 0
    for Nb in Ns
        n = size(Nb, 1); Rn[(o + 1):(o + n), (o + 1):(o + n)] = Nb; o += n
    end
    S = Matrix(Symmetric(H * C * H' + Rn))
    sdev = sqrt.(max.(diag(S), 1e-300))
    white = _whitened_innovations(ν, S, ids, times)
    return (nis=isempty(ν) ? NaN : sum(abs2(r.value) for r in white), dof=length(ν), nu=ν, sdev=sdev, ids=ids, times=times, white=white)
end

"""
    belief_now(sm, t) -> NamedTuple

The belief at time `t` at or after the newest node: the newest node's belief carried through the physics to `t` with the
process noise over the gap, conditioned on the open slice's readings (their noise shares that path):
`y(t) = Φ_t + K ε`, `K = Cov(w_t, ε) Σ⁻¹`, `ε = v − μ`; its linear map over the slice factor's variables and the conditional
noise `Q_t − K Cov(ε, w_t)` give the covariance. `belief_at`'s fields.
"""
function belief_now(sm::FixedLagSmoother, t::Float64)
    tn, kn = sm.nodes[end]
    f = get(sm.slices, kn, nothing)
    (t <= tn + 1e-9 || f === nothing) && return belief_at(sm)
    g = sm.graph; bl = sm.model.bl
    z, zkeys = _latents_at(sm, kn)
    keys = copy(zkeys)
    for k in vars(f); k in keys || push!(keys, k); end
    C, off = joint_cov(g; keys)
    vals = [variable_value(g, k) for k in vars(f)]
    μ, Jμ = _slice_predict(f, vals; jac=true)
    ε = _slice_values(f) .- μ
    y0 = variable_value(g, kn); ny = length(y0)
    j = _slice_col(f, t)
    if j !== nothing && j > 0
        c = f.cache
        Φ = c.S[:, j]; Jy = c.Jy[((j - 1) * ny + 1):(j * ny), :]; Jz = c.Jz[((j - 1) * ny + 1):(j * ny), :]
    else
        gfun(v) = propagate_states(sm.model, v[1:ny], v[(ny + 1):end], tn, [t]; first_slice=kn.k == 1)[:, 1]
        Φ = gfun(vcat(y0, z))
        J = _chunked_jacobian(gfun, vcat(y0, z), ny, 12)
        Jy = J[:, 1:ny]; Jz = J[:, (ny + 1):end]
    end
    Q1 = belief_process_cov(bl, sm.model.sn, 1.0)
    τt = (t - tn) / 3600
    # the filter's own coordinate map (`belief_cov_to_coord`'s capped factors): the raw decode Jacobian
    # inverts to ~1e8 next to a physical boundary (ignited ≈ 1, an empty pan), and an uncapped D·Q1·D'
    # swamped the reported covariance there (group NEES near zero)
    D = belief_coord_factors(Q1 .* max(τt, 0.0), Φ, bl)
    kτ = [min(τt, (r.t - tn) / 3600) for r in f.readings]
    Cwε = (D .* (Q1 * f.G')) .* kτ'                                          # Cov(w_t in coordinates, ε)
    K = Cwε / Symmetric(f.Σ)
    y = Φ .+ K * ε
    A = zeros(ny, size(C, 1))                                                 # ∂y/∂(the variables) = ∂Φ − K ∂μ
    A[:, off[kn]] .+= Jy
    if sm.split === nothing
        A[:, off[sm.key_z]] .+= Jz
    else
        A[:, off[sm.key_z]] .+= Jz[:, sm.split.static]; A[:, off[key_drift(sm, kn.k)]] .+= Jz[:, sm.split.moving]
    end
    for (k, J) in zip(vars(f), Jμ)
        A[:, off[k]] .-= K * J
    end
    Qt = (D .* Q1 .* D') .* τt
    Cyy = Matrix(Symmetric(A * C * A' + Qt - K * Cwε'))
    zix = reduce(vcat, [collect(off[k]) for k in zkeys])
    Czz = C[zix, zix]; Czy = C[zix, :] * A'
    if sm.split !== nothing
        perm = split_permutation(sm.split)
        Czz = Czz[perm, perm]; Czy = Czy[perm, :]
    end
    return _belief_fields(sm, t, z, y, Czz, Czy, Cyy)
end

"""
    rebuild_prior!(sm) -> LinearFactor or nothing

The marginal prior rebuilt from every factor that left the window: those factors at the variables' values when they left
and the window's boundary (the latents, the oldest node and its companions) at their CURRENT values, all the left
variables eliminated in one linearization there. A fixed-lag window freezes a leaving node's information about the latents
at the latents' estimate of that moment; while the latents still travel (the fire lighting) the stacked frozen pieces claim
information along directions the data cannot see, and the window locks. Rebuilding re-linearizes them where the latents are
now. Replaces the window's marginal factor(s).
"""
function rebuild_prior!(sm::FixedLagSmoother)
    isempty(sm.hist_factors) && return nothing
    H = FactorGraph()
    for f in sm.hist_factors, k in vars(f)
        has_variable(H, k) && continue
        add_variable!(H, k, haskey(sm.hist_vals, k) ? sm.hist_vals[k] : copy(variable_value(sm.graph, k)))
    end
    for f in sm.hist_factors; add_factor!(H, f); end
    out = [k for k in H.keys if haskey(sm.hist_vals, k)]
    lf = marginalize!(H, out)
    for f in [f for f in sm.graph.factors if f isa LinearFactor && startswith(f.source, "marginal")]
        remove_factor!(sm.graph, f)
    end
    lf === nothing || add_factor!(sm.graph, LinearFactor(lf.keys, lf.A, lf.b; source="marginal (rebuilt)"))
    sm.z_prior_lin = copy(variable_value(sm.graph, sm.key_z))
    sm.n_prior_relin += 1
    return lf
end

"The window solved on the graph objective; `newton_tol` tightens the joint and stalled marginal posterior-SD checks (zero disables both)."
function solve_window!(sm::FixedLagSmoother; verbose::Bool=false, newton_tol::Float64=ONLINE_NEWTON_TOL)
    isfinite(newton_tol) && newton_tol >= 0 || throw(ArgumentError("newton_tol must be finite and non-negative"))
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
    # Coordinate accuracy and posterior-relative numerical resolution have
    # different units. Let callers state the latter directly instead of
    # inferring it from the coordinate tolerance.
    posterior_resolution = newton_tol
    for attempt in 1:(sm.restarts + 1)
        attempts = attempt
        # Preserve permutation accuracy with fresh-damping solves under the narrow generic
        # resolution gate. A stalled point already inside the posterior-step envelope can request
        # its restart early instead of exhausting the budget at the same mean. Only the final
        # configured attempt may accept under the measured adaptive-ODE
        # envelope; this keeps a resolvable 4e-4 stall from becoming a different accepted posterior
        # merely because a system's pieces changed order.
        resolution_gtol = attempt == sm.restarts + 1 ?
                          max(2 * sm.tol, measured_resolution_gtol) : 2 * sm.tol
        bounds = Dict{VarKey,Float64}(sm.key_z => sm.latent_step)
        if sm.split !== nothing
            for (_, k) in sm.nodes; bounds[key_drift(sm, k.k)] = sm.latent_step; end
        end
        rc = gauss_newton!(sm.graph; iters=sm.iters, tol=sm.tol, ftol=sm.ftol, verbose,
                           resolution_gtol, newton_tol=attempt == sm.restarts + 1 ? posterior_resolution : 0.0,
                           restart_newton_tol=attempt < sm.restarts + 1 ? posterior_resolution : 0.0,
                           step_bounds=bounds, stationarity_scales, expand=ONLINE_EXPAND[], retract_dynamics=true, geodesic_search=true,
                           joint_tol=posterior_resolution)
        append!(all_hist, (merge(h, (attempt=attempt,)) for h in rc.hist))
        r = rc
        rc.converged && break
    end
    sm.n_solves += 1
    return merge(r, (hist=all_hist, attempts=attempts, newton_tol=newton_tol))
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
    # A node's information about the latents is frozen at their current estimate when it leaves. Beyond the lag it leaves
    # only once the latents moved less than `settle_tol` posterior sds since its first solve (they are not still travelling
    # along what it linearized); never more than `lag_max` stay.
    function settled(ko)
        sm.lag_max == sm.lag && return true
        zn = variable_value(sm.graph, sm.key_z)
        (haskey(sm.z_snap, ko) && length(sm.z_sd) == length(zn)) || return false
        return maximum(abs.(zn .- sm.z_snap[ko]) ./ max.(sm.z_sd, eps())) < sm.settle_tol
    end
    while length(sm.nodes) - 1 > sm.lag && (length(sm.nodes) - 1 > sm.lag_max || settled(sm.nodes[1][2]))   # (not `lag + 1`: an infinite lag is typemax)
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
        if isfinite(sm.prior_relin)
            outset = Set(out)
            for f in sm.graph.factors
                f isa LinearFactor && continue                                   # a previous marginal: rebuilt from the factors themselves
                any(k in outset for k in vars(f)) && !any(x -> x === f, sm.hist_factors) && push!(sm.hist_factors, f)
            end
            for k in out; sm.hist_vals[k] = copy(variable_value(sm.graph, k)); end
            isempty(sm.z_prior_lin) && (sm.z_prior_lin = copy(variable_value(sm.graph, sm.key_z)))
        end
        marginalize!(sm.graph, out)
        delete!(sm.dyn, (ko, kn))
        delete!(sm.slices, ko); delete!(sm.slice_recs, ko)
        filter!(o -> !(o.key_y in out), sm.obs)
        popfirst!(sm.nodes)
        delete!(sm.z_snap, ko)
        sm.committed_t = to
        append!(removed, out)
    end
    # The per-reading compiled forms and sensitivities leave with their window: a reading older than the
    # oldest node cannot rejoin a slice, so memory stays bounded by the lag as the header promises.
    if !isempty(removed)
        t_old = sm.nodes[1][1]
        for cache in (sm.slice_G, sm.slice_rd)
            for k in collect(keys(cache))
                k[2] < t_old - 1e-9 && delete!(cache, k)
            end
        end
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
    white = _whitened_innovations(ν, S, [f.spec.id for f in live], [f.t for f in live])
    return (nis=p == 0 ? NaN : sum(abs2(r.value) for r in white), dof=p, S=S, nu=ν, sdev=sdev, H=H, C=C, off=off, live=live, white=white)
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
    _slice_chunks(sm, recs) -> Vector of reading vectors

A batch split at the nodes it will make (the lid-open events it crosses and the node rule's firings, predicted from the
newest node through one solve), so each solve carries at most one new node and the window never holds more than the lag
plus the newest: readings that waited while the last update ran are still taken in order, a node's worth at a time.
"""
function _slice_chunks(sm::FixedLagSmoother, recs::AbstractVector{MeasurementRecord})
    rs = sort(collect(recs); by=r -> r.t)
    tn, kn = sm.nodes[end]
    ahead = unique([r.t for r in rs if r.t > tn + 1e-9])
    isempty(ahead) && return [rs]
    Y, z = _carried(sm, kn, tn, ahead)
    bounds = Float64[]; tcur = tn
    for r in rs
        r.t > tcur + 1e-9 || continue
        for te in _event_times(sm.model.log, tcur, r.t)
            push!(bounds, te); tcur = te
        end
        r.t > tcur + 1e-9 || continue
        spec = sensor(sm.model.log, r.sensor)
        censored_at(spec, r) && (push!(bounds, r.t); tcur = r.t; continue)
        g = slice_sensitivity(sm.model, _slice_reading(sm, r), Y[r.t], z)
        Q = belief_process_cov(sm.model.bl, sm.model.sn, (r.t - tcur) / 3600)
        v_meas = spec.noise.white_sd^2 + (has_discrepancy(spec.noise) ? spec.noise.ar_sd^2 : 0.0)
        (r.t - tcur >= sm.max_slice_s || dot(g, Q * g) > sm.node_ratio * v_meas) && (push!(bounds, r.t); tcur = r.t)
    end
    chunks = [MeasurementRecord[] for _ in 0:length(bounds)]
    for r in rs
        push!(chunks[1 + count(b -> b <= r.t + 1e-9, bounds)], r)
    end
    return filter(!isempty, chunks)
end

"""
    assimilate!(sm, recs; verbose=false, newton_tol=0.01) -> record

Readings into the window, the window solved, the old nodes marginalized: one arrival step (`_assimilate_one!`). Under
`attach = :slice` a batch that crosses several nodes is taken a node's worth at a time (`_slice_chunks`): each chunk its
own committed step. Every chunk is attempted: one that does not commit leaves no trace in the graph and its readings
join the retry queue, re-attempted on the next arrival up to `RETRY_MAX` times before they are given up on
(`sm.n_dropped`); the ALS accumulator requires strictly increasing arrivals, so under ALS they are dropped and counted
instead. The record returned is the last chunk's, with the batch's readings, its summed NIS and degrees of freedom,
every reading's standardized innovation (`innov_rows`), the per-sensor dicts (the last per sensor), its whole wall time
and `committed` true only when every chunk committed. A duplicate of an assimilated reading with the same value is
ignored; one at the same (sensor, time) with a different value is dropped with a warning — the first reading stays. A
batch of nothing but repeats raises an `ArgumentError`. On a thrown exception the current chunk rolls back and the
batch's remaining readings are not queued.
A fresh joint Gauss--Newton posterior distance below `newton_tol`
also establishes numerical convergence. This accounts for correlations across
all graph coordinates. `newton_tol` additionally bounds the largest marginal-SD
step when three stalled trials invoke the numerical-resolution exception.
The default remains 0.01; zero disables both posterior-relative checks.
These bounds are independent of coordinate `tol`. Both use fresh derivatives.
"""
function assimilate!(sm::FixedLagSmoother, recs::AbstractVector{MeasurementRecord}; verbose::Bool=false, newton_tol::Float64=ONLINE_NEWTON_TOL)
    isfinite(newton_tol) && newton_tol >= 0 || throw(ArgumentError("newton_tol must be finite and non-negative"))
    isempty(recs) && throw(ArgumentError("nothing to assimilate"))
    batch = _apply_duplicate_policy!(sm, recs)
    pending = _drain_retries!(sm, batch)
    isempty(batch) && throw(ArgumentError("nothing to assimilate: the batch repeats readings already assimilated"))
    chunks = sm.attach === :slice ? _slice_chunks(sm, batch) : [batch]
    t0 = time()
    nis = 0.0; dof = 0; innov = Dict{Symbol,Float64}(); innov_std = Dict{Symbol,Float64}()
    rows = NamedTuple[]; white = NamedTuple[]; last = nothing; all_committed = true
    for ch in chunks
        r = _assimilate_one!(sm, ch; verbose, newton_tol)
        isfinite(r.nis) && (nis += r.nis; dof += r.dof)
        merge!(innov, r.innovation); merge!(innov_std, r.innov_std)
        append!(rows, r.innov_rows); append!(white, r.innovation_whitened)
        if r.committed
            for rec in ch; sm.seen[(rec.sensor, rec.t)] = rec.value; end
        else
            all_committed = false
            _queue_retries!(sm, ch, pending)
        end
        last = r
    end
    _prune_seen!(sm)
    return merge(last, (sensors=[x.sensor for x in batch], committed=all_committed, nis=dof > 0 ? nis : NaN, dof=dof,
                        innovation=innov, innov_std=innov_std, innov_rows=rows,
                        innovation_whitened=white, wall=time() - t0, chunks=length(chunks)))
end

"An uncommitted solve's readings may be re-attempted this many times before they are given up on."
const RETRY_MAX = 2

# The duplicate policy: a reading already assimilated (or earlier in this batch) with the same value is an
# exact repeat and is ignored; one at the same (sensor, time) with a DIFFERENT value conflicts and is dropped
# with a warning — the first reading stays. Returns the readings to attempt.
function _apply_duplicate_policy!(sm::FixedLagSmoother, recs::AbstractVector{MeasurementRecord})
    batch = MeasurementRecord[]
    inbatch = Dict{Tuple{Symbol,Float64},Float64}()
    for rec in recs
        key = (rec.sensor, rec.t)
        prev = haskey(sm.seen, key) ? sm.seen[key] : get(inbatch, key, nothing)
        if prev === nothing
            inbatch[key] = rec.value
            push!(batch, rec)
        elseif prev != rec.value
            sm.n_dropped += 1
            @warn "a conflicting duplicate reading was dropped (the first reading stays)" sensor = rec.sensor t = rec.t kept = prev dropped = rec.value
        end
    end
    return batch
end

# Readings queued by earlier uncommitted solves join the batch (re-sorted by time); one that fell behind the
# window, or that this batch already carries, leaves the queue. Returns each queued reading's attempts so far.
function _drain_retries!(sm::FixedLagSmoother, batch::Vector{MeasurementRecord})
    pending = Dict{Tuple{Symbol,Float64},Int}()
    isempty(sm.retry) && return pending
    t_old = sm.nodes[1][1]
    for (rec, n) in sm.retry
        key = (rec.sensor, rec.t)
        if rec.t < t_old - 1e-9
            sm.n_dropped += 1
            @warn "a queued reading fell behind the window and was dropped" sensor = rec.sensor t = rec.t
        elseif haskey(sm.seen, key) || any(r -> r.sensor === rec.sensor && r.t == rec.t, batch)
            # assimilated meanwhile, or this batch already carries it
        else
            pending[key] = n
            push!(batch, rec)
        end
    end
    empty!(sm.retry)
    sort!(batch; by=r -> r.t)
    return pending
end

# An uncommitted chunk's readings are queued for the next arrival, each up to `RETRY_MAX` re-attempts. The
# ALS accumulator requires strictly increasing arrivals, so under ALS they are dropped and counted instead.
function _queue_retries!(sm::FixedLagSmoother, ch::AbstractVector{MeasurementRecord}, pending::Dict{Tuple{Symbol,Float64},Int})
    for rec in ch
        n = get(pending, (rec.sensor, rec.t), 0) + 1
        if sm.als !== nothing || n > RETRY_MAX
            sm.n_dropped += 1
        else
            push!(sm.retry, (rec, n))
        end
    end
    return nothing
end

"The duplicate policy's memory shrinks with the window: a reading older than the oldest node cannot arrive as valid input."
function _prune_seen!(sm::FixedLagSmoother)
    t_old = sm.nodes[1][1]
    for k in collect(keys(sm.seen))
        k[2] < t_old - 1e-9 && delete!(sm.seen, k)
    end
    return nothing
end

"""
    _assimilate_one!(sm, recs; verbose=false) -> record

Readings into the window, the window solved, the old nodes marginalized: one arrival step. The record
carries the arrival time `t` and the belief's own time `t_belief` (the propagation target in an open slice,
the newest node otherwise; on an uncommitted arrival the pre-arrival node), the sensors, each reading's raw
innovation at the belief before the solve (`innovation`, the last per sensor) and every reading's
standardized innovation (`innov_rows`: `(t, sensor, std)` per uncensored reading), the normalized
innovation squared of the uncensored readings against the predicted innovation covariance with its degrees of
freedom and the standardized innovations per sensor (`_predicted_consistency`), the solve's objective,
iterations and convergence, the newest node's belief (per piece the centre and surface temperatures with their
sds, the pit's), the latents' posterior and the wall time; on a committed arrival also the whitened residuals
at the accepted point by kind (`_residual_power`) and, when a node leaves the window, that node's belief as it
leaves (`past`: the smoothed past at the lag, `belief_at`'s fields without the joint covariance; at lag 0 it is
the previous node, re-solved once with the new readings before it goes). A committed arrival feeds the bench's
accumulator when the smoother carries one.
"""
function _assimilate_one!(sm::FixedLagSmoother, recs::AbstractVector{MeasurementRecord}; verbose::Bool=false, newton_tol::Float64=ONLINE_NEWTON_TOL)
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
        t_arr = maximum(x.t for x in recs)
        if sm.attach === :slice
            # a censored reading keeps its own node and observation factor (its likelihood is not Gaussian); the rest go
            # through the physics from their slice's node
            cens = [r for r in recs if censored_at(sensor(sm.model.log, r.sensor), r)]
            fs = [add_measurement!(sm, r) for r in cens]
            _tick!(:place, @elapsed _add_slice_readings!(sm, [r for r in recs if !censored_at(sensor(sm.model.log, r.sensor), r)]))
            added = Dict{VarKey,Vector{MeasurementRecord}}()
            for r in recs
                censored_at(sensor(sm.model.log, r.sensor), r) && continue
                ka = sm.nodes[findlast(x -> x[1] <= r.t + 1e-9, sm.nodes)][2]
                push!(get!(added, ka, MeasurementRecord[]), r)
            end
            tq = time(); pcs = isempty(added) ? nothing : _predicted_consistency_slice(sm, [(k, v) for (k, v) in added]; exclude=fs); _tick!(:nis, time() - tq)
            pco = isempty(fs) ? nothing : _predicted_consistency(sm, fs)
            nis = (pcs === nothing ? 0.0 : pcs.nis) + (pco === nothing ? 0.0 : pco.nis)
            dof = (pcs === nothing ? 0 : pcs.dof) + (pco === nothing ? 0 : pco.dof)
            dof == 0 && (nis = NaN)
            # one row per uncensored reading, `(t, sensor, std)` with `std` the marginally standardized
            # innovation: the complete per-reading evidence (the per-sensor dicts keep only the last)
            rows = NamedTuple[]
            pcs === nothing || for (i, id) in enumerate(pcs.ids)
                push!(rows, (t=pcs.times[i], sensor=id, std=pcs.nu[i] / pcs.sdev[i]))
            end
            pco === nothing || for (i, f) in enumerate(pco.live)
                push!(rows, (t=f.t, sensor=f.spec.id, std=pco.nu[i] / pco.sdev[i]))
            end
            sort!(rows; by=rr -> (rr.t, string(rr.sensor)))
            innov_std = Dict{Symbol,Float64}(rr.sensor => rr.std for rr in rows)
            # `innovation` is RAW (the reading less its prediction, measurement units) in every attach mode:
            # the slice path's conditioned innovations, a censored/node factor's at the pre-update point
            innov = Dict{Symbol,Float64}()
            pcs === nothing || for (i, id) in enumerate(pcs.ids); innov[id] = pcs.nu[i]; end
            for f in fs
                innov[f.spec.id] = innovation(f, [variable_value(sm.graph, k) for k in vars(f)])
            end
            white = vcat(pcs === nothing ? NamedTuple[] : pcs.white, pco === nothing ? NamedTuple[] : pco.white)
            pc = nothing
        else
            fs = [add_measurement!(sm, r) for r in recs]
            innov = Dict(r.sensor => innovation(f, [variable_value(sm.graph, k) for k in vars(f)]) for (r, f) in zip(recs, fs))
            pc = _predicted_consistency(sm, fs)
            nis = pc === nothing ? NaN : pc.nis
            dof = pc === nothing ? 0 : pc.dof
            innov_std = pc === nothing ? Dict{Symbol,Float64}() : Dict(f.spec.id => pc.nu[i] / pc.sdev[i] for (i, f) in enumerate(pc.live))
            rows = pc === nothing ? NamedTuple[] :
                   sort!([(t=f.t, sensor=f.spec.id, std=pc.nu[i] / pc.sdev[i]) for (i, f) in enumerate(pc.live)]; by=rr -> (rr.t, string(rr.sensor)))
            white = pc === nothing ? NamedTuple[] : pc.white
        end
        als = (sm.als !== nothing && pc !== nothing) ? _als_pieces(sm, sm.als, pc, t_arr) : nothing
        tq = time(); r = solve_window!(sm; verbose, newton_tol); _tick!(:solve, time() - tq)
        if !r.converged
            _restore_transaction!(sm, saved)
            sm.n_solves += 1
            b = belief_at(sm)                                            # the belief as it stands (the previous node's); `t` is the arrival's, `t_belief` the belief's own
            rec = merge(b, (t=t_arr, t_belief=b.t, sensors=[x.sensor for x in recs], innovation=innov, nis=nis, dof=dof, innov_std=innov_std, innov_rows=rows, F=r.F,
                            iters=length(r.hist), attempts=r.attempts, converged=false, committed=false, gradient=r.gradient, newton=r.newton_step, newton_sd=r.newton_sd, newton_tol=newton_tol,
                            newton_joint_sd=r.newton_joint_sd, joint_tol=r.joint_tol, wall=time() - t0,
                            n_nodes=length(sm.nodes), innovation_whitened=white, resid=nothing, past=nothing))
            return _push_record!(sm, rec)
        end
        resid = _residual_power(sm, fs, t_arr)
        past = nothing
        if length(sm.nodes) - 1 > sm.lag                                  # a node leaves: its smoothed belief at the lag, before it goes
            bp = belief_at(sm; node=1)
            past = (t=bp.t, z=bp.z, z_cov=bp.z_cov, y=bp.y, y_cov=bp.y_cov, x=bp.x, x_cov=bp.x_cov, T_pit=bp.T_pit, pieces=bp.pieces)
        end
        if sm.lag_max > sm.lag
            zn = copy(variable_value(sm.graph, sm.key_z))
            for (_, k) in sm.nodes
                haskey(sm.z_snap, k) || (sm.z_snap[k] = zn)
            end
        end
        tq = time(); marginalize_old!(sm); _tick!(:marginalize, time() - tq)
        if isfinite(sm.prior_relin) && !isempty(sm.z_prior_lin) && length(sm.z_sd) == length(sm.z_prior_lin)
            zn = variable_value(sm.graph, sm.key_z)
            if maximum(abs.(zn .- sm.z_prior_lin) ./ max.(sm.z_sd, eps())) > sm.prior_relin
                tq = time(); rebuild_prior!(sm); _tick!(:prior_rebuild, time() - tq)
            end
        end
        tq = time(); b = sm.attach === :slice ? belief_now(sm, t_arr) : belief_at(sm); _tick!(:belief, time() - tq)
        if sm.lag_max > sm.lag || isfinite(sm.prior_relin)
            sdz = sqrt.(max.(diag(b.z_cov), 0.0))
            sm.z_sd = sm.split === nothing ? sdz : sdz[sm.split.static]
        end
        rec = merge(b, (t=t_arr, t_belief=b.t, sensors=[x.sensor for x in recs], innovation=innov, nis=nis, dof=dof, innov_std=innov_std, innov_rows=rows, F=r.F,
                        iters=length(r.hist), attempts=r.attempts, converged=true, committed=true, gradient=r.gradient, newton=r.newton_step, newton_sd=r.newton_sd, newton_tol=newton_tol,
                        newton_joint_sd=r.newton_joint_sd, joint_tol=r.joint_tol, wall=time() - t0,
                        n_nodes=length(sm.nodes), innovation_whitened=white, resid=resid, past=past))
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
    return _belief_fields(sm, t, z, variable_value(sm.graph, ky), C[rz, rz], C[rz, ry], C[ry, ry])
end
"The belief's record fields from the latents and the state with their covariance blocks."
function _belief_fields(sm::FixedLagSmoother, t, z, y, Czz, Czy, Cyy)
    n_z = length(z); rz, ry = 1:n_z, (n_z + 1):(n_z + length(y))
    C = [Czz Czy; Czy' Cyy]
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
