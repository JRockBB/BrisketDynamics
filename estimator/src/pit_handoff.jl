# Learn the fire on the pit alone, then hand off to the whole cook.
#
# Before the meat goes on, the full system's pit evolves independently of the meat (up to the load blend's
# logistic tail, ~3e-7 thirty minutes before the load), and everything the cook says about the ignition
# latents arrives then. A CookSystem with NO pieces is that pit: 6 states instead of ~200, not stiff. So the
# adaptive quadrature over the ignition axes runs there at full history, its chart following the contracting
# posterior, and shortly before the load every component is handed to the full system: its conditional
# posterior over the cooker's latents and the pit's state becomes the full smoother's prior on those, the
# meat's latents keep their exact priors and its state its initial widths, and the quadrature weights carry
# over. The chart freezes at the hand-off; the components then run at the hold lag for the rest of the cook,
# each conditioned on its fixed point of the ignition axes.
#
# Known simplification: the sensors' discrepancy chains restart at stationarity in the full graph (the pit
# phase's last discrepancy states are not carried across).

export pit_only_model, handoff_smoother, PhasedSmoother, phased_init

"The cooker's latents in a block, in the cooker's own order (the φ schema)."
_phi_indices(blk::LatentBlock) =
    sort([i for (i, (b, _)) in blk.bindings if b == (:pit, :phi)]; by=i -> blk.bindings[i][2])

# A sensor the pit alone can read: its observable compiles against the pit-only system. Observables of a
# piece (a probe, the scale) refuse there; nothing is decided by sensor name.
function _pit_readable(s::SensorSpec, sys0, bl0, blk0, log0)
    try
        compile_observable(s.observable, sys0, bl0, blk0, log0); return true
    catch err
        (err isa ObservableUnavailable || err isa ArgumentError) || rethrow()
        return false
    end
end

"""
    pit_only_model(model) -> SystemModel

The same cook without its pieces: the pit and firebox alone, the cooker's latents only (cut from the full
block by their bindings, so names and priors match exactly), the pit's own events, and the sensors the pit
alone can read with their readings. No ROM: a reduced model corrects a piece.
"""
function pit_only_model(model::SystemModel)
    sys0 = CookSystem(PieceModel[]; pit=model.sys.pit, hum=model.sys.hum)
    blk = model.blk; iφ = _phi_indices(blk)
    blk0 = LatentBlock(blk.sites[iφ], blk.names[iφ], blk.keys[iφ],
                       Dict(k => blk.bindings[i] for (k, i) in enumerate(iφ)),
                       Dict(n => k for (k, n) in enumerate(blk.names[iφ])))
    evs = CookEvent[e for e in model.log.events if e.target === :pit]
    bl0 = belief_layout(sys0)
    probe = CookLog(model.log.t_grid, model.log.u; events=evs)
    sensors = SensorSpec[s for s in model.log.sensors if _pit_readable(s, sys0, bl0, blk0, probe)]
    ids = Set(s.id for s in sensors)
    log0 = CookLog(model.log.t_grid, model.log.u; events=evs, sensors,
                   measurements=MeasurementRecord[m for m in model.log.measurements if m.sensor in ids])
    return SystemModel(sys0, log0, blk0; sn=model.sn, abstol=model.abstol, reltol=model.reltol, rom=nothing,
                       floor_abstol=model.floor_abstol, floor_reltol=model.floor_reltol)
end

# A Gaussian N(m, C) on selected coordinates of several variables, as a whitened linear factor:
# residual W (S x - m), W = L^-1 with C = L L'.
function _selection_prior(sel::Vector{Tuple{VarKey,Vector{Int},Int}}, m::AbstractVector, C::AbstractMatrix; source)
    W = Matrix(inv(cholesky(Symmetric(_stable_cov(Matrix(C)))).L))
    n = length(m); keys = VarKey[]; As = Matrix{Float64}[]; row = 0
    for (k, idx, dim) in sel
        S = zeros(n, dim)
        for (j, i) in enumerate(idx); S[row + j, i] = 1.0; end
        push!(keys, k); push!(As, W * S); row += length(idx)
    end
    row == n || throw(ArgumentError("the selection does not cover the prior's dimension"))
    return LinearFactor(keys, As, W * m; source)
end

"""
    handoff_smoother(model, zφ, y_pit, C, free; t0, kwargs...) -> FixedLagSmoother

A full smoother whose first node, at `t0`, carries a pit phase's posterior: the cooker's latents `zφ` (their
`free` entries jointly Gaussian with the pit state `y_pit` under `C`, ordered [free φ; pit state]; the rest
are held by conditioning), the meat latents' exact site priors and the meat state's initial widths. The first
slice does NOT reset the ignited fraction from a0: the node already holds the learned one.
"""
function handoff_smoother(model::SystemModel, zφ::AbstractVector, y_pit::AbstractVector, C::AbstractMatrix,
                          free::AbstractVector{Bool}; t0::Real, kwargs...)
    sm = online_init(model; t0, ignite_first=false, kwargs...)
    g = sm.graph; kz = sm.key_z; k1 = sm.nodes[1][2]; bl = model.bl; blk = model.blk
    iφ = _phi_indices(blk); pr = bl.pit_range
    length(zφ) == length(iφ) == length(free) || throw(ArgumentError("one value and one free flag per cooker latent"))
    length(y_pit) == length(pr) || throw(ArgumentError("the pit state has $(length(pr)) coordinates"))
    nf = count(free); size(C) == (nf + length(pr), nf + length(pr)) ||
        throw(ArgumentError("the hand-off covariance spans the free cooker latents and the pit state"))
    # drop online_init's own priors on these two variables; lay down the hand-off's
    filter!(f -> !((f isa SitePriorFactor && f.key == kz) || (f isa GaussianPrior && f.key == k1)), g.factors)
    z = copy(variable_value(g, kz)); z[iφ] .= zφ; g.values[kz] .= z
    y = copy(variable_value(g, k1)); y[pr] .= y_pit; g.values[k1] .= y
    meat_z = setdiff(eachindex(blk.sites), iφ)
    isempty(meat_z) || add_factor!(g, SitePriorFactor(kz, blk.sites; indices=meat_z,
        source="$(sm.cook): the meat's exact latent priors (hand-off)"))
    add_factor!(g, _selection_prior([(kz, iφ[free], length(z)), (k1, collect(pr), length(y))],
        vcat(zφ[free], y_pit), C; source="$(sm.cook): the pit phase's posterior on the cooker and its state"))
    meat_y = setdiff(1:bl.n, pr)
    if !isempty(meat_y)
        Q0 = belief_cov_to_coord(belief_init_cov(bl, model.sn), y, bl)
        add_factor!(g, _selection_prior([(k1, meat_y, length(y))], y[meat_y], Q0[meat_y, meat_y];
            source="$(sm.cook): the meat state's initial widths (hand-off)"))
    end
    return sm
end

# A pit smoother's posterior over (its latents, its newest node) from a stacked mean and square-root
# covariance over its graph's keys.
function _pit_marginal(g::FactorGraph, mean::AbstractVector, root::AbstractMatrix, kz::VarKey, kn::VarKey)
    off, _ = _offsets(g)
    rows = vcat(collect(off[kz]), collect(off[kn]))
    return mean[off[kz]], mean[off[kn]], root[rows, :] * root[rows, :]'
end

"""
    PhasedSmoother

The cook's estimator in two phases: the adaptive quadrature over the ignition axes on the pit alone (full
history) until the first reading at or after `t_handoff`, then the same quadrature components handed to the
full system with the chart frozen and every smoother at `hold_lag`. Records come in the full system's
coordinates in both phases (in the pit phase the meat sits at its prior and initial state, unobserved).
"""
mutable struct PhasedSmoother
    pit::DeterministicSmoother
    full::Union{Nothing,DeterministicSmoother}
    full_model::SystemModel
    template::FixedLagSmoother          # the full system's belief layout and its prior moments, for pit-phase records
    t_handoff::Float64
    hold_lag::Int
    kwargs::NamedTuple                  # the full smoothers' online_init options (attach, node_ratio, ...)
    t_handed::Float64                   # the hand-off node's time (NaN before)
    n_dropped_pre::Int                  # pre-hand-off readings of sensors the pit cannot read
end

"""
    phased_init(model; axes, orders=(3,5), hold_lag=3, handoff_margin=600.0, kwargs...) -> PhasedSmoother

The hand-off time is `handoff_margin` seconds before the earliest load event in `model`'s log (a model
without a load event never hands off). `kwargs` reach both phases' smoothers (attach, node_ratio,
max_slice_s, ...); the pit phase keeps full history and adapts its chart throughout. `shape_tol` is the
pit phase's guard on how non-Gaussian the cooker's remaining latents may grow (the density layer's own).
"""
function phased_init(model::SystemModel; axes, orders=(3, 5), hold_lag::Int=3, handoff_margin::Float64=600.0,
                     shape_tol::Float64=0.25, kwargs...)
    loads = [e.t for e in model.log.events if e.kind === :load]
    t_h = isempty(loads) ? Inf : minimum(loads) - handoff_margin
    pm = pit_only_model(model)
    pit = deterministic_init(pm; axes, orders, freeze_stable=nothing, lag=typemax(Int), shape_tol, kwargs...)
    template = online_init(model; kwargs...)
    return PhasedSmoother(pit, nothing, model, template, t_h, hold_lag, NamedTuple(kwargs), NaN, 0)
end

"The smoother that currently owns the graph structure (the bench reads its counters)."
structure_owner(ps::PhasedSmoother) = ps.full === nothing ? ps.pit.bank.anchor : ps.full.bank.anchor

# Hand every quadrature component to the full system at the pit's newest node; freeze the chart there.
function _handoff!(ps::PhasedSmoother)
    pit = ps.pit; model = ps.full_model
    pa = pit.bank.anchor; t0, kn = last(pa.nodes)
    iφ = _phi_indices(model.blk)
    axes_pit = [i for (_, i) in pit.bank.axes]
    kw = merge(ps.kwargs, (lag=ps.hold_lag, lag_max=ps.hold_lag, slice_g_refresh=false))
    # the anchor: the pit anchor's Gaussian posterior over all of its latents and its state
    C, off = joint_cov(pa.graph; keys=[pa.key_z, kn])
    zφ = copy(variable_value(pa.graph, pa.key_z)); yp = copy(variable_value(pa.graph, kn))
    anchor = handoff_smoother(model, zφ, yp, C, trues(length(zφ)); t0, kw...)
    axes_full = [(anchor.key_z, iφ[i]) for i in axes_pit]
    banks = _ConditionalBank[]
    for b in pit.bank.banks
        smoothers = Vector{FixedLagSmoother}(undef, length(b.nodes)); nodes = Vector{ConditionalNode}(undef, length(b.nodes))
        Threads.@threads for j in eachindex(b.nodes)
            c = b.nodes[j]; inner = b.smoothers[j]
            zc, yc, Cc = _pit_marginal(inner.graph, c.mean, c.root, inner.key_z, last(inner.nodes)[2])
            free = trues(length(zc)); free[axes_pit] .= false
            keep = vcat(findall(free), length(zc) .+ (1:length(yc)))
            s = handoff_smoother(model, zc, yc, Cc[keep, keep], free; t0, kw...)
            nodes[j] = _conditional_node!(s.graph, axes_full, c.coordinates; iters=s.iters, tol=s.tol, ftol=s.ftol,
                step_bounds=Dict(s.key_z => s.latent_step), newton_tol=ONLINE_NEWTON_TOL)
            smoothers[j] = s
        end
        all(c -> c.solve.converged, nodes) || error("a handed-off component's prior solve did not converge")
        # the accumulated evidence stays in the weights; each node's new local integral is the next denominator
        push!(banks, _ConditionalBank(b.order, smoothers, nodes, copy(b.logweights)))
    end
    posterior = _conditional_current(banks[end], axes_full)
    cs = ConditionalSmoother(anchor, axes_full, banks, posterior, pit.bank.mean_tol, pit.bank.covariance_tol)
    d = _density_from_bank(cs, banks[end], pit.center, pit.scale)
    ps.full = DeterministicSmoother(cs, d, copy(pit.center), copy(pit.scale), pit.density_tol, pit.shape_tol,
                                    pit.max_order, pit.max_nodes, false, pit.max_axes,
                                    merge(pit.diagnostics, (handoff_t=t0,)), nothing, ps.hold_lag, true, 0,
                                    pit.n_recentres, pit.n_refines)
    ps.t_handed = t0
    return ps
end

# A pit-phase record in the full system's coordinates: the cooker's latents and the pit state from the
# pit phase's mixture, the meat's latents at their prior moments and its state at its initial widths (the
# meat is not on yet, nothing observes it).
function _embed_pit_record(ps::PhasedSmoother, r::NamedTuple)
    tm = ps.template; model = ps.full_model; bl = model.bl; blk = model.blk
    iφ = _phi_indices(blk); pr = bl.pit_range; nz = length(blk); ny = bl.n
    z = copy(variable_value(tm.graph, tm.key_z)); z[iφ] .= r.z
    y = copy(variable_value(tm.graph, tm.nodes[1][2])); y[pr] .= r.y
    Czz = zeros(nz, nz)
    for i in setdiff(1:nz, iφ); Czz[i, i] = z_moments(blk.sites[i])[2]; end
    Czz[iφ, iφ] .= r.z_cov
    Q0 = belief_cov_to_coord(belief_init_cov(bl, model.sn), y, bl)
    meat = setdiff(1:ny, pr)
    Cyy = zeros(ny, ny); Cyy[meat, meat] .= Q0[meat, meat]; Cyy[pr, pr] .= r.y_cov
    Czy = zeros(nz, ny); np = length(iφ)
    Czy[iφ, pr] .= r.zy_cov[1:np, (np + 1):end]
    b = _belief_fields(tm, r.t, z, y, Czz, Czy, Cyy)
    return merge(r, b, (t_belief=r.t, phase=:pit, handoff_t=ps.t_handed))
end

function assimilate!(ps::PhasedSmoother, recs::AbstractVector{MeasurementRecord}; verbose=false)
    isempty(recs) && throw(ArgumentError("nothing to assimilate"))
    pre = MeasurementRecord[r for r in recs if r.t < ps.t_handoff]
    post = MeasurementRecord[r for r in recs if r.t >= ps.t_handoff]
    rec = nothing
    if ps.full === nothing && !isempty(pre)
        ids = Set(s.id for s in ps.pit.bank.anchor.model.log.sensors)
        readable = MeasurementRecord[r for r in pre if r.sensor in ids]
        ps.n_dropped_pre += length(pre) - length(readable)
        isempty(readable) || (rec = _embed_pit_record(ps, assimilate!(ps.pit, readable; verbose)))
    end
    if !isempty(post)
        ps.full === nothing && _handoff!(ps)
        r = assimilate!(ps.full, post; verbose)
        rec = merge(r, (phase=r.phase === :hold ? :full : r.phase, handoff_t=ps.t_handed))
    end
    rec === nothing && throw(ArgumentError("nothing the estimator can read in this batch"))
    return rec
end
assimilate!(ps::PhasedSmoother, rec::MeasurementRecord; kwargs...) = assimilate!(ps, [rec]; kwargs...)
