# --- the smoother: moving parameters as piecewise constants over the full history ------------------
#
# The batch estimator refits every latent as a constant on the full history; the window filter (retired)
# predicts and updates on a window from a handed-off state and pays for taking that state as known (the
# leak and the drip read a state error as a parameter error, and it compounds). This is the version with
# no such approximation: the full history is solved in segments (segments.jl, exact and differentiable
# through the hand-off), the sites that move carry one value per segment, and the values are tied by a
# random-walk prior between segments (a variance rate per hour) or by a jump at a known event (a new charge
# at the add). The fit is the same Gauss-Newton graph over the stacked block; the belief at an hour is the
# latest value of each site with its marginal width. What it costs: the batch estimator's solves over a
# larger latent block. What it gives: a parameter that moves is estimated where it is, not averaged.

"""
    DriftLayout(n_base, moving)

The stacked latent block: the `n_base` base latents (the first segment's values of every site), then, for
each moving site, one copy per later segment start. `moving` is a vector of `(site, starts)`: the base index
of the site and the sorted grid indices at which a new value starts (each > 1). Value `j` of a moving site
is in force from `starts[j]` until the next start; before `starts[1]` the base value is.
"""
struct DriftLayout
    n_base::Int
    sites::Vector{Int}
    starts::Vector{Vector{Int}}
    offsets::Vector{Int}          # the index before the first copy of each moving site
    n::Int
end
function DriftLayout(n_base::Int, moving::AbstractVector)
    sites = Int[m[1] for m in moving]
    starts = Vector{Int}[sort(collect(Int, m[2])) for m in moving]
    offsets = Int[]; n = n_base
    for st in starts
        push!(offsets, n); n += length(st)
    end
    return DriftLayout(n_base, sites, starts, offsets, n)
end
"The grid indices at which any moving site changes, sorted and unique."
segment_edges(lay::DriftLayout) = sort(unique(reduce(vcat, lay.starts; init=Int[])))
"The stacked index holding moving site `m`'s value in force at grid index `i` (the base index before its first start)."
function index_at(lay::DriftLayout, m::Int, i::Int)
    j = searchsortedlast(lay.starts[m], i)
    return j == 0 ? lay.sites[m] : lay.offsets[m] + j
end
"The stacked index of each base site's LATEST value (at the end of the history)."
current_index(lay::DriftLayout) = [(m = findfirst(==(s), lay.sites); m === nothing || isempty(lay.starts[m]) ? s : lay.offsets[m] + length(lay.starts[m])) for s in 1:lay.n_base]
"The physics block (length N_PHYS) in force from grid index `i`, from the stacked `z`."
function block_at(z, lay::DriftLayout, i::Int)
    zb = z[1:N_PHYS]
    for m in eachindex(lay.sites)
        s = lay.sites[m]
        s <= N_PHYS && (zb[s] = z[index_at(lay, m, i)])
    end
    return zb
end

"""
    PhysicsDrift(area_latent, events, lay)

The 1-D physics with moving sites: the cook solved in segments at the change points of `lay`, each
segment at the parameters in force there, the state handed off exactly. Reads the stacked z directly
(the Jacobian is taken in z, through the transforms, by ForwardDiff).
"""
struct PhysicsDrift <: Physics
    area_latent::Int
    events::NamedTuple
    lay::DriftLayout
end
function physics_stack(z, d::GateData, ph::PhysicsDrift)
    n = length(d.t)
    edges = filter(e -> 1 < e < n, segment_edges(ph.lay))
    bounds = vcat(1, edges)
    xs = [from_z(block_at(z, ph.lay, b), d.prior[1:N_PHYS]) for b in bounds]
    A = ph.area_latent > 0 ? exp(z[ph.area_latent]) : d.cfg.A_meat
    o = solve_segments([x[1:N_THETA] for x in xs], [x[(N_THETA + 1):N_PHYS] for x in xs], d.u, d.t, edges;
                       pit=d.pit, cfg=d.cfg, A_meat=A, abstol=d.abstol, reltol=d.reltol, ph.events...)
    return _stack_1d(o, A, xs[1][4], d.cfg), OutputLayout(size(o.T_field, 1), n)
end
function physics_outputs(z, d::GateData, ph::PhysicsDrift)
    g(v) = physics_stack(v, d, ph)[1]
    S, lay = physics_stack(z, d, ph)
    cfgj = ForwardDiff.JacobianConfig(g, z, ForwardDiff.Chunk{min(length(z), 12)}())
    J = ForwardDiff.jacobian(g, z, cfgj)
    return S, J, lay
end

"""
    DriftPrior(base, lay, incvar)

The prior on the stacked block: the per-site families on the base block, and on each moving site's copies a
Gaussian on the increment from the value before it (a random walk: `incvar[m][j]` is the variance of copy `j`
minus its predecessor; a jump at a known event is one large increment). Indexes and iterates as the sites
of the stacked block (a copy carries its site's family), so the channels and the transforms read it
unchanged; `prior_terms` returns the value, the gradient and the full Hessian (the walk couples neighbours).
"""
struct DriftPrior
    sites::Vector{PriorSite}
    base::Vector{PriorSite}
    lay::DriftLayout
    incvar::Vector{Vector{Float64}}
end
function DriftPrior(base::Vector{PriorSite}, lay::DriftLayout, incvar::Vector{Vector{Float64}})
    sites = copy(base)
    for m in eachindex(lay.sites), _ in lay.starts[m]
        push!(sites, base[lay.sites[m]])
    end
    length(sites) == lay.n || throw(ArgumentError("the layout and the base block disagree"))
    return DriftPrior(sites, base, lay, incvar)
end
Base.getindex(p::DriftPrior, i) = p.sites[i]
Base.length(p::DriftPrior) = length(p.sites)
Base.lastindex(p::DriftPrior) = lastindex(p.sites)
Base.iterate(p::DriftPrior, st...) = iterate(p.sites, st...)
Base.eltype(::Type{DriftPrior}) = PriorSite
function prior_terms(z, p::DriftPrior)
    nb = p.lay.n_base; n = p.lay.n
    f, gb, hb = prior_terms(z[1:nb], p.base)
    g = zeros(n); H = zeros(n, n)
    g[1:nb] = gb
    for i in 1:nb; H[i, i] = hb[i]; end
    for m in eachindex(p.lay.sites)
        prev = p.lay.sites[m]
        for j in eachindex(p.lay.starts[m])
            cur = p.lay.offsets[m] + j
            v = p.incvar[m][j]
            r = z[cur] - z[prev]
            f += 0.5 * r^2 / v
            g[cur] += r / v; g[prev] -= r / v
            H[cur, cur] += 1 / v; H[prev, prev] += 1 / v; H[cur, prev] -= 1 / v; H[prev, cur] -= 1 / v
            prev = cur
        end
    end
    return f, g, H
end

"""
    drift_layout(sites, d, batch_edges, b, moving) -> (lay, prior, physics)

The stacked layout, the prior and the physics for a fit at batch edge `b` of `batch_edges`: a drift site
starts a new value at every earlier batch edge (increment variance `q` per hour times the hours since its
last value), a jump site at its event's grid point once the history runs past it (one increment of
variance `var`).
"""
function drift_layout(sites::Vector{PriorSite}, d::GateData, batch_edges, b::Int, moving; area_latent::Int=0, events::NamedTuple=NamedTuple())
    ie = findlast(d.t .<= batch_edges[b] + 1e-9)
    dth = (d.t[2] - d.t[1]) / 3600
    specs = Tuple{Int,Vector{Int}}[]; incvar = Vector{Float64}[]
    for (s, kind, param) in moving
        if kind === :drift
            prev_edges = [findlast(d.t .<= e + 1e-9) for e in batch_edges[1:(b - 1)]]
            starts = filter(i -> 1 < i < ie, prev_edges)
            push!(specs, (s, starts)); push!(incvar, param .* (diff(vcat(1, starts)) .* dth))
        elseif kind === :jump
            t_jump, var = param
            ij = isfinite(t_jump) ? findlast(d.t .<= t_jump + 1e-9) : nothing       # the copy starts at the event's grid point
            if ij === nothing || !(1 < ij < ie)
                push!(specs, (s, Int[])); push!(incvar, Float64[])
            else
                push!(specs, (s, [ij])); push!(incvar, [var])
            end
        else
            throw(ArgumentError("a moving site is :drift or :jump"))
        end
    end
    lay = DriftLayout(length(sites), specs)
    return lay, DriftPrior(sites, lay, incvar), PhysicsDrift(area_latent, events, lay)
end

"The warm start for a layout from the last stacked `z` and its layout: every copy continues the value before it."
function warm_start(lay::DriftLayout, z, old::Union{Nothing,DriftLayout})
    nb = lay.n_base
    c = zeros(lay.n); c[1:nb] = z[1:nb]
    for m in eachindex(lay.sites), j in eachindex(lay.starts[m])
        c[lay.offsets[m] + j] = (j == 1 ? z[lay.sites[m]] : c[lay.offsets[m] + j - 1])
    end
    old === nothing && return c
    old.n == lay.n && old.starts == lay.starts && return copy(z)
    for m in eachindex(lay.sites), j in eachindex(lay.starts[m])
        mo = findfirst(==(lay.sites[m]), old.sites)
        mo === nothing && continue
        if j <= length(old.starts[mo])
            c[lay.offsets[m] + j] = z[old.offsets[mo] + j]
        elseif !isempty(old.starts[mo])
            c[lay.offsets[m] + j] = z[old.offsets[mo] + length(old.starts[mo])]
        end
    end
    return c
end

"""
    smoother_step(channels, sites, d, batch_edges, b, z_prev, lay_prev; moving, area_latent, events, kw...) -> record

One refit of the smoother at batch edge `b`: the history to that edge over the stacked block laid out by
`drift_layout`, warm-started from the last fit. Returns the record `smoother_fit` collects, plus the fitted
channels (their noise carried by the caller).
"""
function smoother_step(channels, sites::Vector{PriorSite}, d::GateData, batch_edges, b::Int, z_prev, lay_prev;
                       moving::AbstractVector=[], area_latent::Int=0, events::NamedTuple=NamedTuple(), verbose::Bool=false, kw...)
    lay, prior, ph = drift_layout(sites, d, batch_edges, b, moving; area_latent, events)
    c = warm_start(lay, z_prev, lay_prev)
    keep = d.t .<= batch_edges[b] + 1e-9
    chs = [restrict(ch, keep) for ch in channels]
    t0 = time()
    res = solve_map_channels(chs, prior, d; c0=c, physics=ph, verbose, kw...)
    ci = current_index(lay)
    sd = sqrt.(max.(diag(res.cov), 0.0))
    return (batch=b, t=batch_edges[b], z=res.z, sd=sd, cov=res.cov, z_now=res.z[ci], sd_now=sd[ci], cov_now=res.cov[ci, ci], layout=lay,
            F=res.F, channels=res.channels, iters=length(res.hist), wall=time() - t0, misfit=res.misfit, noise=res.noise, converged=res.converged)
end

"""
    smoother_fit(channels, sites, d, batch_edges; moving, area_latent, noise, ...) -> Vector{NamedTuple}

The streaming estimator with moving sites: at every batch edge the full history to the edge is refit over
the stacked block (`smoother_step`). `moving` is a vector of `(site, kind, param)`: `(s, :drift, q)` gives
site `s` a new value at every batch edge with increment variance `q` (z² per hour) times the hours since
its last value; `(s, :jump, (t_jump, var))` gives it one new value from the first batch edge at or after
`t_jump`, with increment variance `var`. Each record carries the stacked `z` and `cov`, and the belief at
that hour as `z_now`/`sd_now`/`cov_now` (the latest value of every base site).
"""
function smoother_fit(channels, sites::Vector{PriorSite}, d::GateData, batch_edges; moving::AbstractVector=[],
                      area_latent::Int=0, verbose::Bool=false, events::NamedTuple=NamedTuple(), kw...)
    z = vcat(d.z0, z_center.(sites[(N_PHYS + 1):end]))
    lay = nothing
    channels = collect(Channel, channels)
    out = NamedTuple[]
    for b in eachindex(batch_edges)
        rec = smoother_step(channels, sites, d, batch_edges, b, z, lay; moving, area_latent, events, verbose, kw...)
        z, lay = rec.z, rec.layout
        for (i, np) in rec.noise
            channels[i] = with_noise(unlevelled(channels[i]), np.σ, np.ρ)
            rec.channels[i] isa Levelled && (channels[i] = Levelled(channels[i], rec.channels[i].level))
        end
        push!(out, rec)
        verbose && @printf("  smoother batch %2d  t = %5.1f h  block %d  F = %9.3f  outer %2d  (%.0f s)\n", b, batch_edges[b] / 3600, lay.n, rec.F, rec.iters, rec.wall)
        verbose && flush(stdout)
    end
    return out
end
