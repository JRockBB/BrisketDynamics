# The offline backend (docs/multi-meat-inference-plan.md): the full history of one or several cooks on the
# same factors as the online window, with nothing marginalized (multiple shooting: a state node at every
# measurement time, the dynamics factors between them, every reading at its own time, every sensor's
# discrepancy chain), the cooks sharing one graph, and a hierarchy: the latents named hierarchical are one
# population variable plus per-cook offsets with a between-cook sd, so several cooks pool what they share
# (a cut's diffusivity, a cooker's thermal mass) while each keeps its own (its fire, its probe's depth). The
# solve is the same damped Gauss-Newton, sparse (SuiteSparse QR in the natural order, the population's and
# the cooks' parameters last so the state chains stay banded), and the marginals selected by key.
#
# What the available suite cannot identify is stated here too: `unobserved_groups` compares graph information
# with and without a cook's observations. A state need not have a direct sensor row to be informed indirectly
# through coupled dynamics (collagen, fat and bark can affect temperature or mass), while an inert field or
# disconnected shared state remains at its propagated prior. Included by BrisketEstimator after online.jl.

export OfflineProblem, HierarchySpec, offline_problem, solve!, population, hyperposterior, cook_latents, cook_belief, unobserved_groups

"""
    HierarchySpec(parent, members, tau; prior=nothing, ancestor=nothing, ancestor_tau=nothing)

One semantic random-effect parent. `parent` names its population/family/cooker/piece/sensor scope;
`members[cook_id]` names one child latent or a vector of child latents in that cook, so heterogeneous latent
layouts, multi-piece cooks, and subsets are explicit rather than positionally aligned. `prior` defaults to the
first compatible child's prior.
When compatible children carry different prior parameters, the parent prior must be explicit.
For a multilevel hierarchy, `ancestor` names the next broader semantic parent and `ancestor_tau` is the
standard deviation between this parent and that ancestor. Only root nodes receive the transformed-family
prior; intermediate nodes are regularized by their ancestor link.
"""
struct HierarchySpec
    parent::LatentKey
    members::Dict{Symbol,Vector{LatentKey}}
    tau::Float64
    prior::Union{Nothing,PriorSite}
    ancestor::Union{Nothing,LatentKey}
    ancestor_tau::Union{Nothing,Float64}
    function HierarchySpec(parent::LatentKey, members::AbstractDict, tau::Real;
                           prior=nothing, ancestor=nothing, ancestor_tau=nothing)
        parent.scope in (:population, :family, :cooker, :piece, :sensor) ||
            throw(ArgumentError("a hierarchy parent needs a shared semantic scope"))
        isempty(members) && throw(ArgumentError("a hierarchy needs at least one member"))
        isfinite(tau) && tau > 0 || throw(ArgumentError("a hierarchy sd must be finite and positive"))
        if ancestor === nothing
            ancestor_tau === nothing || throw(ArgumentError("ancestor_tau needs an ancestor"))
        else
            ancestor isa LatentKey || throw(ArgumentError("a hierarchy ancestor must be a LatentKey"))
            ancestor == parent && throw(ArgumentError("a hierarchy node cannot be its own ancestor"))
            _scope_rank(ancestor.scope) < _scope_rank(parent.scope) ||
                throw(ArgumentError("a hierarchy ancestor must have a broader semantic scope than its child"))
            ancestor_tau === nothing && throw(ArgumentError("a hierarchy ancestor needs ancestor_tau"))
            isfinite(ancestor_tau) && ancestor_tau > 0 ||
                throw(ArgumentError("an ancestor sd must be finite and positive"))
        end
        normalized = Dict{Symbol,Vector{LatentKey}}()
        for (id, raw) in members
            children = raw isa LatentKey ? [raw] : collect(LatentKey, raw)
            isempty(children) && throw(ArgumentError("cook $id has no hierarchy members"))
            length(unique(children)) == length(children) ||
                throw(ArgumentError("cook $id repeats a hierarchy member"))
            all(child -> child.parameter == parent.parameter, children) ||
                throw(ArgumentError("hierarchy parent $(parent.parameter) cannot pool a different semantic parameter"))
            normalized[Symbol(id)] = children
        end
        ancestor === nothing || ancestor.parameter == parent.parameter ||
            throw(ArgumentError("a hierarchy ancestor and child must name the same semantic parameter"))
        return new(parent, normalized, Float64(tau), prior,
                   ancestor, ancestor_tau === nothing ? nothing : Float64(ancestor_tau))
    end
end

_scope_rank(s::Symbol) = s === :population ? 1 : s === :family ? 2 :
                         s in (:cooker, :cook) ? 3 : s === :piece ? 4 : s === :sensor ? 5 :
                         throw(ArgumentError("unknown hierarchy scope $s"))
_compatible_site(a::PriorSite, b::PriorSite) =
    a.family == b.family && a.coord == b.coord && a.p3 == b.p3 && a.p4 == b.p4 &&
    (!isscore(a) || (a.p1 == b.p1 && a.p2 == b.p2))
_same_site_prior(a::PriorSite, b::PriorSite) =
    _compatible_site(a, b) && a.p1 == b.p1 && a.p2 == b.p2

"""
    OfflineProblem

Several cooks' full histories on one graph: their smoothers (one per cook, the lag infinite), the population
variable of the hierarchical latents (or nothing), the indices of those latents and their between-cook sds.
"""
struct OfflineProblem
    graph::FactorGraph
    cooks::Vector{FixedLagSmoother}
    key_pop::Union{Nothing,VarKey}
    hier::Vector{Int}
    tau::Vector{Float64}
    hyper::Dict{LatentKey,VarKey}
    specs::Vector{HierarchySpec}
end

"""
    offline_problem(models; ids, hierarchy=Int[], tau=Float64[], node_times=nothing, iters=30, tol=1e-6, ftol=1e-8) -> OfflineProblem

The full-history problem over the cooks' models (each a `SystemModel` with its log carrying the readings):
every cook's nodes at its measurement times (or at `node_times[i]`, extra shooting nodes without a reading),
its readings, its discrepancy chains. The latents `hierarchy` (indices into every cook's block, which must
share one layout) are tied to one population variable with the between-cook sds `tau`; the population's prior
is the family's, every cook's other latents keep theirs.
"""
function offline_problem(models::AbstractVector{SystemModel}; ids::AbstractVector{Symbol}=[Symbol("cook", i) for i in eachindex(models)],
                         hierarchy::AbstractVector{<:Integer}=Int[], tau::AbstractVector=Float64[], node_times=nothing,
                         hierarchies::AbstractVector{HierarchySpec}=HierarchySpec[],
                         iters::Int=30, tol::Float64=1e-6, ftol::Float64=1e-8)
    isempty(models) && throw(ArgumentError("an offline problem needs at least one cook"))
    length(ids) == length(models) || throw(ArgumentError("one id per cook"))
    length(unique(ids)) == length(ids) || throw(ArgumentError("cook ids must be distinct"))
    length(hierarchy) == length(tau) || throw(ArgumentError("one between-cook sd per hierarchical latent"))
    isempty(hierarchy) || isempty(hierarchies) || throw(ArgumentError("use semantic hierarchies or the legacy positional hierarchy, not both"))
    g = FactorGraph()
    key_pop = nothing
    hier = collect(Int, hierarchy)
    specs = collect(HierarchySpec, hierarchies)
    if !isempty(hier)
        blk = models[1].blk
        all(i -> 1 <= i <= length(blk), hier) || throw(ArgumentError("a hierarchy index is out of range"))
        length(unique(hier)) == length(hier) || throw(ArgumentError("hierarchy indices must be unique"))
        all(x -> isfinite(x) && x > 0, tau) || throw(ArgumentError("between-cook sds must be finite and positive"))
        for m in models
            length(m.blk) == length(blk) || throw(ArgumentError("legacy positional hierarchy requires equal latent dimensions"))
            for j in hier
                latent_key(m.blk, j) == latent_key(blk, j) ||
                    throw(ArgumentError("legacy hierarchy index $j has different semantic keys; use HierarchySpec"))
                m.blk.sites[j] == blk.sites[j] ||
                    throw(ArgumentError("legacy hierarchy index $j has incompatible prior transforms or units"))
            end
        end
        key_pop = VarKey(:hyper, :population, 0)
        z0 = [z_moments(blk.sites[i])[1] for i in hier]; sd0 = [sqrt(z_moments(blk.sites[i])[2]) for i in hier]
        add_variable!(g, key_pop, z0)
        add_factor!(g, SitePriorFactor(key_pop, blk.sites[hier]; source="the population's exact transformed-family priors"))
    end
    hyper = Dict{LatentKey,VarKey}()
    links = Dict(id => NamedTuple[] for id in ids)
    node_sites = Dict{LatentKey,PriorSite}()
    node_order = LatentKey[]
    ancestry = Dict{LatentKey,Tuple{LatentKey,Float64}}()
    parent_seen = Set{LatentKey}()
    member_seen = Set{Tuple{Symbol,LatentKey}}()
    function register_node!(key, site)
        if haskey(node_sites, key)
            _same_site_prior(node_sites[key], site) ||
                throw(ArgumentError("hierarchy node $key has conflicting prior definitions"))
        else
            node_sites[key] = site
            push!(node_order, key)
        end
    end
    for spec in specs
        spec.parent in parent_seen && throw(ArgumentError("duplicate hierarchy parent $(spec.parent)"))
        push!(parent_seen, spec.parent)
        all(id -> id in ids, keys(spec.members)) || throw(ArgumentError("a hierarchy member names an unknown cook"))
        sites = PriorSite[]
        for (id, child_keys) in spec.members, child_key in child_keys
            i = findfirst(==(id), ids)
            ic = latent_index(models[i].blk, child_key)
            _scope_rank(spec.parent.scope) < _scope_rank(child_key.scope) ||
                throw(ArgumentError("hierarchy parent $(spec.parent) is not broader than child $child_key"))
            push!(sites, models[i].blk.sites[ic])
        end
        if spec.prior === nothing && !all(s -> _same_site_prior(s, first(sites)), sites)
            throw(ArgumentError("children of $(spec.parent) have different priors; provide the parent's prior explicitly"))
        end
        prior = spec.prior === nothing ? first(sites) : spec.prior
        all(s -> _compatible_site(s, prior), sites) ||
            throw(ArgumentError("children of $(spec.parent) have incompatible transforms or units"))
        register_node!(spec.parent, prior)
        if spec.ancestor !== nothing
            _scope_rank(spec.ancestor.scope) < _scope_rank(spec.parent.scope) ||
                throw(ArgumentError("ancestor $(spec.ancestor) is not broader than $(spec.parent)"))
            register_node!(spec.ancestor, prior)
            haskey(ancestry, spec.parent) &&
                throw(ArgumentError("hierarchy node $(spec.parent) has more than one ancestor"))
            ancestry[spec.parent] = (spec.ancestor, something(spec.ancestor_tau))
        end
        for (id, child_keys) in spec.members, child_key in child_keys
            token = (id, child_key)
            token in member_seen &&
                throw(ArgumentError("cook $id latent $child_key belongs to more than one hierarchy parent"))
            push!(member_seen, token)
            i = findfirst(==(id), ids)
            ic = latent_index(models[i].blk, child_key)
            push!(links[id], (idx_child=ic, parent=spec.parent, idx_parent=1, tau=spec.tau))
        end
    end
    for key in node_order
        kp = VarKey(:hyper, (key.scope, key.entity, key.parameter), 0)
        μ, _ = z_moments(node_sites[key])
        add_variable!(g, kp, [μ])
        hyper[key] = kp
    end
    for key in node_order
        haskey(ancestry, key) && continue
        add_factor!(g, SitePriorFactor(hyper[key], [node_sites[key]]; source="$(key)'s exact transformed-family prior"))
    end
    for (child, (ancestor, τ_parent)) in ancestry
        add_factor!(g, HierarchyFactor(hyper[child], hyper[ancestor], [τ_parent]))
    end
    for id in ids
        links[id] = [(idx_child=l.idx_child, key_parent=hyper[l.parent], idx_parent=l.idx_parent, tau=l.tau) for l in links[id]]
    end
    cooks = FixedLagSmoother[]
    for (i, (m, id)) in enumerate(zip(models, ids))
        hierarchy_arg = key_pop === nothing ? (isempty(links[id]) ? nothing : links[id]) :
                        (hier, key_pop, collect(Float64, tau))
        sm = online_init(m; graph=g, cook=id, lag=typemax(Int), iters, tol, ftol,
                         hierarchy=hierarchy_arg)
        if node_times !== nothing
            for t in sort(collect(Float64, node_times[i]))
                t > sm.nodes[1][1] && _node_at!(sm, t)
            end
        end
        for rec in sort(m.log.measurements; by=r -> r.t)
            add_measurement!(sm, rec)
        end
        push!(cooks, sm)
    end
    # the parameters last: the state chains lead the ordering, so the sparse triangle stays banded
    hyper_keys = collect(values(hyper))
    plast = vcat([sm.key_z for sm in cooks], key_pop === nothing ? VarKey[] : [key_pop], hyper_keys)
    g.keys = vcat([k for k in g.keys if !(k in plast)], plast)
    return OfflineProblem(g, cooks, key_pop, hier, collect(Float64, tau), hyper, specs)
end

"""
    solve!(prob; sparse=true, verbose=false, kw...) -> (F, hist, converged)

The whole problem by damped Gauss-Newton (`gauss_newton!`), sparse by default.
"""
function solve!(prob::OfflineProblem; sparse::Bool=true, verbose::Bool=false, kw...)
    sm = prob.cooks[1]
    # The full-history solve uses the same physical-latent trust region as each online window. Without it,
    # a large joint parameter proposal is merely shortened by line search, which is both slower and a poorer
    # local model than constructing the proposal inside every cook's validated latent radius.
    if haskey(kw, :step_bounds)
        r = gauss_newton!(prob.graph; iters=sm.iters, tol=sm.tol, ftol=sm.ftol, sparse, verbose, kw...)
    else
        bounds = Dict(c.key_z => c.latent_step for c in prob.cooks)
        r = gauss_newton!(prob.graph; iters=sm.iters, tol=sm.tol, ftol=sm.ftol, sparse, verbose,
                          step_bounds=bounds, kw...)
    end
    for c in prob.cooks; c.n_solves += 1; end
    return r
end

"The population's posterior: mean and covariance over the hierarchical latents (their indices in `prob.hier`)."
function population(prob::OfflineProblem; sparse::Bool=true)
    if prob.key_pop !== nothing
        return (mean=copy(variable_value(prob.graph, prob.key_pop)), cov=marginal_cov(prob.graph, prob.key_pop; sparse),
                idx=copy(prob.hier), keys=[latent_key(prob.cooks[1].model.blk, i) for i in prob.hier])
    end
    length(prob.hyper) == 1 || throw(ArgumentError("select a semantic parent with hyperposterior; the problem has $(length(prob.hyper))"))
    return hyperposterior(prob, only(keys(prob.hyper)); sparse)
end
"A selected semantic hierarchy parent's posterior."
function hyperposterior(prob::OfflineProblem, parent::LatentKey; sparse::Bool=true)
    haskey(prob.hyper, parent) || throw(ArgumentError("no hierarchy parent $parent"))
    k = prob.hyper[parent]
    return (mean=copy(variable_value(prob.graph, k)), cov=marginal_cov(prob.graph, k; sparse), key=parent)
end
"A cook's latents' posterior: mean and covariance."
function cook_latents(prob::OfflineProblem, i::Int; sparse::Bool=true)
    sm = prob.cooks[i]
    return (mean=copy(variable_value(prob.graph, sm.key_z)), cov=marginal_cov(prob.graph, sm.key_z; sparse),
            names=copy(sm.model.blk.names), keys=[latent_key(sm.model.blk, j) for j in 1:length(sm.model.blk)])
end
"A cook's belief at one of its nodes (`belief_at`), the newest by default."
cook_belief(prob::OfflineProblem, i::Int; node::Int=length(prob.cooks[i].nodes)) = belief_at(prob.cooks[i]; node)

"""
    unobserved_groups(prob, i; tol=1e-4) -> Vector{String}

The newest belief groups whose marginal variance is not reduced by this cook's observations through the
whole factor graph. The comparison removes this cook's observation factors from a copy of the graph while
retaining priors, dynamics, discrepancies and hierarchy, so indirect information through transitions and
other hierarchy members counts; a direct row touching one node no longer marks a whole field observed.
`tol` is the minimum fractional trace reduction.
"""
function unobserved_groups(prob::OfflineProblem, i::Int; tol::Float64=1e-4)
    sm = prob.cooks[i]; bl = sm.model.bl; ky = last(sm.nodes)[2]
    Cpost = marginal_cov(prob.graph, ky; sparse=true)
    g0 = deepcopy(prob.graph)
    filter!(f -> !(f isa ObservationFactor && f.key_z == sm.key_z), g0.factors)
    Cbase = marginal_cov(g0, ky; sparse=true)
    out = String[]
    for (name, rg) in bl.groups
        base = sum(diag(view(Cbase, rg, rg)))
        post = sum(diag(view(Cpost, rg, rg)))
        gain = base > eps(Float64) ? max(0.0, 1 - post / base) : 0.0
        gain <= tol && push!(out, name)
    end
    return out
end
