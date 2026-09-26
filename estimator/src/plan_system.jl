# The multi-meat planner (docs/multi-meat-inference-plan.md, planning): the receding horizon of horizon.jl
# generalized to a `CookSystem` and a `BeliefSnapshot`. The decisions are the pit's (the damper knots, the fuel
# addition, the serve time: everyone eats together) and every piece's own (its wrap, its pull, its spritz train),
# the goals are per piece (a family's quality bands and its safety requirement: a brisket's collagen band, a
# bird's lethality), and the objective is the sum over the pieces of their goal residuals with the shared terms
# (fuel, damper smoothness) counted once. The plan is solved as inference on the factor graph of factors.jl:
# one variable ζ, the plan prior as a Gaussian factor, one plan factor per belief draw whose residual is the
# goal residuals through the system's physics from the snapshot's state (ForwardDiff through the solve), damped
# Gauss-Newton with the line search; a coordinate-wise scan over the wraps, the rest and the damper's shapes
# picks a second start, and only stationary solves can be selected under the graph's full hedged objective.
#
# For a system of one slab the decoded plan, the cook and the residuals are horizon.jl's (test_plan_system.jl
# gates the objective's parity at the same plan). Included by BrisketEstimator after offline.jl.

export SystemSchedule, SystemFreeSet, system_free_set, system_plan_prior, decode_system_plan, encode_system_plan, system_plan_damper,
       system_schedule_events, system_plan_solve, system_plan_outputs, system_goal_residuals, system_plan_loss, replan_system,
       commit_system!, execute_system_damper!, system_replan_due, note_system_replanned!, system_plan_predictive, PlanFactor,
       SystemControlGoals, belief_plan_draws, PlanConvergenceError

struct PlanConvergenceError <: Exception
    message::String
end
Base.showerror(io::IO, e::PlanConvergenceError) = print(io, e.message)

"System-level penalties shared by all pieces; these never depend on piece ordering or a family's goals."
struct SystemControlGoals
    w_fuel::Float64
    w_smooth::Float64
    m_chimney::Float64
    m0_ref::Float64
    function SystemControlGoals(w_fuel::Real, w_smooth::Real, m_chimney::Real, m0_ref::Real)
        isfinite(w_fuel) && w_fuel >= 0 || throw(ArgumentError("the shared fuel weight must be finite and nonnegative"))
        isfinite(w_smooth) && w_smooth >= 0 || throw(ArgumentError("the shared smoothness weight must be finite and nonnegative"))
        isfinite(m_chimney) && m_chimney >= 0 || throw(ArgumentError("the chimney fuel mass must be finite and nonnegative"))
        isfinite(m0_ref) && m0_ref > 0 || throw(ArgumentError("the fuel reference mass must be finite and positive"))
        new(Float64(w_fuel), Float64(w_smooth), Float64(m_chimney), Float64(m0_ref))
    end
end
SystemControlGoals(w_fuel::Real, w_smooth::Real) = SystemControlGoals(w_fuel, w_smooth, 1.4, 7.0)
SystemControlGoals(; w_fuel::Real=0.1, w_smooth::Real=0.01, m_chimney::Real=1.4, m0_ref::Real=7.0) =
    SystemControlGoals(w_fuel, w_smooth, m_chimney, m0_ref)

"""
    SystemSchedule(t, pieces; u0=0.15, t_serve=14 h)

The executed and committed schedule of a system's cook on the grid `t`: the damper (NaN beyond `t_now`), per
piece its wrap, pull (NaN unexecuted) and spritz train (nothing until started), the pit's fuel addition, and the
serve time everyone eats at.
"""
mutable struct SystemSchedule
    t::Vector{Float64}
    u::Vector{Float64}
    t_now::Float64
    pieces::Vector{Symbol}
    t_wrap::Dict{Symbol,Float64}
    t_pull::Dict{Symbol,Float64}
    spritz::Dict{Symbol,Union{Nothing,NamedTuple{(:t, :dt, :m),NTuple{3,Float64}}}}
    t_add::Float64
    t_serve::Float64
    replanned_for::Dict{Tuple{Symbol,Symbol},Float64}
end
SystemSchedule(t, pieces::AbstractVector{Symbol}; u0::Float64=0.15, t_serve::Float64=14 * 3600.0) =
    SystemSchedule(collect(Float64, t), [k == 1 ? u0 : NaN for k in eachindex(t)], 0.0, collect(pieces),
                   Dict(p => NaN for p in pieces), Dict(p => NaN for p in pieces),
                   Dict{Symbol,Union{Nothing,NamedTuple{(:t, :dt, :m),NTuple{3,Float64}}}}(p => nothing for p in pieces),
                   NaN, t_serve, Dict{Tuple{Symbol,Symbol},Float64}())

"Which decisions are free: per piece its wrap (until wrapped or pulled), pull (until committed) and spritz (until started or pulled); the damper and the add until every piece is out; the serve always."
struct SystemFreeSet
    damper::Bool
    wrap::Dict{Symbol,Bool}
    pull::Dict{Symbol,Bool}
    spritz::Dict{Symbol,Bool}
    add::Bool
    serve::Bool
    pieces::Vector{Symbol}
end
function system_free_set(s::SystemSchedule)
    executed(p) = !isnan(s.t_pull[p]) && s.t_now >= s.t_pull[p]
    all_out = all(executed(p) for p in s.pieces)
    return SystemFreeSet(!all_out, Dict(p => isnan(s.t_wrap[p]) && !executed(p) for p in s.pieces),
                         Dict(p => isnan(s.t_pull[p]) for p in s.pieces), Dict(p => s.spritz[p] === nothing && !executed(p) for p in s.pieces),
                         isnan(s.t_add) && !all_out, true, copy(s.pieces))
end
Base.length(f::SystemFreeSet) = N_KNOTS * f.damper + sum(f.wrap[p] + f.pull[p] + 3 * f.spritz[p] for p in f.pieces; init=0) + f.add + f.serve

# the plan block ζ: damper knots | per piece [wrap logit f (of now→its pull) | pull logit (of now→the bound) | spritz logit f, log dt h, log m] | add logit f (of now→the last pull) | serve logit (of the last pull→the bound)
"The prior on the free decisions: means and precisions in ζ's order (horizon.jl's widths: a regulariser, not a preference)."
function system_plan_prior(f::SystemFreeSet, s::SystemSchedule; plan_sd::Float64=PLAN_SD)
    m = Float64[]; sd = Float64[]
    fp = clamp((12 * 3600 - s.t_now) / (T_SERVE_MAX - s.t_now), 0.02, 0.98)
    if f.damper; append!(m, fill(lgt(0.15), N_KNOTS)); append!(sd, fill(plan_sd, N_KNOTS)); end
    for p in f.pieces
        if f.wrap[p]; push!(m, lgt(0.7)); push!(sd, plan_sd); end
        if f.pull[p]; push!(m, lgt(fp)); push!(sd, plan_sd); end
        if f.spritz[p]; append!(m, [lgt(0.3), log(1.0), log(0.05)]); append!(sd, [plan_sd, 1.0, 1.0]); end
    end
    if f.add; push!(m, lgt(0.5)); push!(sd, plan_sd); end
    if f.serve
        nominal_pull(p) = f.pull[p] ? s.t_now + fp * (T_SERVE_MAX - s.t_now) : s.t_pull[p]
        tp = maximum(nominal_pull(p) for p in f.pieces)
        push!(m, lgt(clamp(1.5 * 3600 / max(T_SERVE_MAX - tp, 600.0), 0.02, 0.98))); push!(sd, plan_sd)
    end
    return m, Diagonal(1 ./ sd .^ 2)
end

"The decisions in seconds from ζ: the damper knots, per piece `(t_wrap, t_pull, spritz)`, the add and the serve."
function decode_system_plan(ζ, f::SystemFreeSet, s::SystemSchedule)
    k = 0
    take() = (k += 1; ζ[k])
    u = f.damper ? [lgs(take()) for _ in 1:N_KNOTS] : fill(isnan(s.u[findlast(!isnan, s.u)]) ? 0.15 : s.u[findlast(!isnan, s.u)], N_KNOTS)
    Tel = eltype(u)
    pieces = Dict{Symbol,Any}()
    for p in f.pieces
        fw = f.wrap[p] ? lgs(take()) : NaN
        t_pull = f.pull[p] ? s.t_now + (T_SERVE_MAX - s.t_now) * lgs(take()) : Tel(s.t_pull[p])
        if f.spritz[p]
            fs = lgs(take()); dts = 3600 * exp(take()); ms = exp(take())
            spritz = (t=s.t_now + fs * (t_pull - s.t_now), dt=dts, m=ms)
        else
            spritz = s.spritz[p] === nothing ? nothing : (t=Tel(s.spritz[p].t), dt=Tel(s.spritz[p].dt), m=Tel(s.spritz[p].m))
        end
        t_wrap = f.wrap[p] ? s.t_now + fw * (t_pull - s.t_now) : Tel(s.t_wrap[p])
        pieces[p] = (t_wrap=t_wrap, t_pull=t_pull, spritz=spritz)
    end
    t_last = maximum(pieces[p].t_pull for p in f.pieces)
    fa = f.add ? lgs(take()) : NaN
    t_add = f.add ? s.t_now + fa * (t_last - s.t_now) : Tel(s.t_add)
    t_serve = f.serve ? t_last + max(T_SERVE_MAX - t_last, zero(t_last)) * lgs(take()) : Tel(s.t_serve)
    return (u=u, pieces=pieces, t_add=t_add, t_serve=t_serve, t_last_pull=t_last)
end

"ζ from a plan in seconds (the warm start of a replan); decisions outside the free set are ignored."
function encode_system_plan(plan, f::SystemFreeSet, s::SystemSchedule)
    clampf(x) = clamp(x, 0.02, 0.98)
    ζ = Float64[]
    f.damper && append!(ζ, lgt.(clamp.(resample_knots(plan.u, N_KNOTS), 0.02, 0.98)))
    for p in f.pieces
        pp = plan.pieces[p]
        rem = max(pp.t_pull - s.t_now, 600.0)
        f.wrap[p] && push!(ζ, lgt(clampf((pp.t_wrap - s.t_now) / rem)))
        f.pull[p] && push!(ζ, lgt(clampf(rem / (T_SERVE_MAX - s.t_now))))
        if f.spritz[p]
            sp = pp.spritz === nothing ? (t=s.t_now + 0.3 * rem, dt=3600.0, m=0.05) : pp.spritz
            append!(ζ, [lgt(clampf((sp.t - s.t_now) / rem)), log(sp.dt / 3600), log(sp.m)])
        end
    end
    t_last = max(maximum(plan.pieces[p].t_pull for p in f.pieces), s.t_now + 600.0)
    f.add && push!(ζ, lgt(clampf((plan.t_add - s.t_now) / (t_last - s.t_now))))
    f.serve && push!(ζ, lgt(clampf((plan.t_serve - t_last) / max(T_SERVE_MAX - t_last, 600.0))))
    return ζ
end

"The damper on the grid: executed where it is, the plan's knots (piecewise linear over now→the last pull, held after) beyond now."
function system_plan_damper(plan, s::SystemSchedule)
    Tel = promote_type(eltype(plan.u), typeof(plan.t_last_pull), Float64)
    u = Vector{Tel}(undef, length(s.t))
    nk = length(plan.u)
    tk = s.t_now .+ knot_f(nk) .* (plan.t_last_pull - s.t_now)
    for (i, ti) in enumerate(s.t)
        if ti <= s.t_now && !isnan(s.u[i])
            u[i] = s.u[i]
        elseif ti <= tk[1]
            u[i] = plan.u[1]
        elseif ti >= tk[end]
            u[i] = plan.u[end]
        else
            j = min(searchsortedlast(tk, ForwardDiff.value(ti)), nk - 1)
            w = (ti - tk[j]) / (tk[j + 1] - tk[j])
            u[i] = (1 - w) * plan.u[j] + w * plan.u[j + 1]
        end
    end
    return u
end

"The executed events of the schedule as `CookEvent`s (nothing for an unexecuted one)."
function system_schedule_events(s::SystemSchedule)
    ev = CookEvent[]
    for p in s.pieces
        isnan(s.t_wrap[p]) || push!(ev, CookEvent(:wrap, p, s.t_wrap[p]))
        isnan(s.t_pull[p]) || push!(ev, CookEvent(:pull, p, s.t_pull[p]))
        s.spritz[p] === nothing || push!(ev, CookEvent(:spritz, p, s.spritz[p].t; dt=s.spritz[p].dt, m=s.spritz[p].m))
    end
    isnan(s.t_add) || push!(ev, CookEvent(:add, :pit, s.t_add))
    return ev
end
"The plan's events as `CookEvent`s (a `NEVER` decision leaves no event); Dual times are carried as values in the event and as Duals in the parameter vector by the solve."
function _plan_events(plan, s::SystemSchedule)
    ev = CookEvent[]
    for p in s.pieces
        pp = plan.pieces[p]
        isnan(_primal(pp.t_wrap)) || push!(ev, CookEvent(:wrap, p, _primal(pp.t_wrap)))
        push!(ev, CookEvent(:pull, p, _primal(pp.t_pull)))
        pp.spritz === nothing || push!(ev, CookEvent(:spritz, p, _primal(pp.spritz.t); dt=_primal(pp.spritz.dt), m=_primal(pp.spritz.m)))
    end
    isnan(_primal(plan.t_add)) || push!(ev, CookEvent(:add, :pit, _primal(plan.t_add)))
    return ev
end

"""
    system_plan_solve(snap, z, ζ, f, s) -> (o, plan, tg)

The system's cook under the plan `ζ` from the snapshot's state at the latents `z` (the block's, `apply_latents`),
on the schedule's grid from the snapshot's time on: the executed events and the plan's future ones, the plan's
damper, the pieces restarted from the snapshot's hand-off. Duals in `ζ` (and in `z`) differentiate through.
"""
function system_plan_solve(snap::BeliefSnapshot, z, ζ, f::SystemFreeSet, s::SystemSchedule; y=snap.y)
    sys = snap.sys
    plan = decode_system_plan(ζ, f, s)
    k = findfirst(τ -> abs(τ - snap.t) < 1e-6, s.t)
    k === nothing && throw(ArgumentError("the snapshot's time $(snap.t) s is not a point of the schedule's grid"))
    tg = s.t[k:end]
    _validate_rom_use(snap.rom, snap.blk, z, first(tg), last(tg))
    u = system_plan_damper(plan, s)[k:end]
    # the parameter vector: the plan's decisions as parameters (Duals flow), the latents applied on top
    θs = [zeros(length(piece_parameter_schema(pc))) for pc in sys.pieces]; As = [pc.cfg.A_meat for pc in sys.pieces]
    p0, play = pack_params(sys, θs, As, vcat(zeros(6), 0.5), u; events=_plan_events(plan, s))
    p = apply_latents(snap.blk, z, p0, play)
    Tel = promote_type(eltype(p), eltype(u), eltype(ζ))
    p = convert(Vector{Tel}, p)
    for pc in sys.pieces
        pp = plan.pieces[pc.id]
        isnan(_primal(pp.t_wrap)) || (p[first(play[(pc.id, :t_wrap)])] = pp.t_wrap)
        p[first(play[(pc.id, :t_pull)])] = pp.t_pull
        if pp.spritz !== nothing
            p[first(play[(pc.id, :t_spritz)])] = pp.spritz.t
            p[first(play[(pc.id, :dt_spritz)])] = pp.spritz.dt
            p[first(play[(pc.id, :m_spritz)])] = pp.spritz.m
        end
    end
    isnan(_primal(plan.t_add)) || (p[first(play[(:pit, :t_add)])] = plan.t_add)
    xb = belief_decode(y, snap.bl)
    h0 = belief_handoff(xb, snap.bl)
    h = merge(h0, (history=get(snap.handoff, :history, nothing),))
    x0 = initial_state(sys, p, play; ics=h.pieces, pit_ics=(; h.pit..., ignited=h.ignited))
    correction = snap.rom === nothing ? nothing : (t -> _rom_dynamics_correction(snap.rom, t))
    # Plan factors differentiate this solve and then demand stationarity. The physics default (1e-6)
    # is adequate for a forward trajectory, but its adaptive-step jitter is larger than the objective
    # changes near a plan optimum and makes valid descent directions fail every line-search trial.
    # Match the multiple-shooting factor's smooth evaluation envelope. The planner's separate
    # resolution gate bounds any adaptive-ODE floor by a fraction of its output grid.
    A = system_solve_raw(sys, p, x0, tg, tg; correction, abstol=1e-10, reltol=1e-8)
    return system_outputs(sys, A, p, tg, tg; history=get(h, :history, nothing), correction), plan, tg
end

"""
    system_plan_outputs(snap, z, ζ, f, s, goals) -> Dict piece => y

Every piece's `cook_outputs` under the plan (its own pull, the shared add and serve, the pit's fuel, its past's
integrals from the snapshot), the damper differences shared.
"""
function _planning_piece_output(snap::BeliefSnapshot, pc::PieceModel, po, tg, η=(0.0, 0.0))
    snap.rom === nothing && return po
    ec = [_rom_piece_error(snap.rom, pc.id, :Tcore, t) for t in tg]
    es = [_rom_piece_error(snap.rom, pc.id, :Tsurf, t) for t in tg]
    Tcore = [po.Tcore[k] - ec[k].bias - η[1] * ec[k].sd for k in eachindex(tg)]
    Tsurf = [po.Tsurf[k] - es[k].bias - η[2] * es[k].sd for k in eachindex(tg)]
    return merge(po, (Tcore=Tcore, Tsurf=Tsurf))
end

function _planning_piece_quality(snap::BeliefSnapshot, pc::PieceModel, po, o, plan, tg, goals, η=(0.0, 0.0))
    corrected = _planning_piece_output(snap, pc, po, tg, η)
    oi = (; corrected..., m_fuel=o.m_fuel)
    return cook_outputs(oi, tg, diff(plan.u), plan.pieces[pc.id].t_pull, plan.t_add, plan.t_serve,
                        pc.cfg, goals[pc.id]; past=snap.past[pc.id])
end

function system_plan_outputs(snap::BeliefSnapshot, z, ζ, f::SystemFreeSet, s::SystemSchedule, goals::AbstractDict{Symbol,PlanGoals}; y=snap.y)
    o, plan, tg = system_plan_solve(snap, z, ζ, f, s; y)
    out = Dict{Symbol,Any}()
    for pc in snap.sys.pieces
        out[pc.id] = _planning_piece_quality(snap, pc, o.piece[pc.id], o, plan, tg, goals)
    end
    m0s = [p.m0 for p in values(snap.past) if isfinite(p.m0)]
    m0 = isempty(m0s) ? o.m_fuel[1] : first(m0s)
    all(v -> isapprox(v, m0; atol=1e-9, rtol=1e-9), m0s) ||
        throw(ArgumentError("piece histories disagree on the shared pit's initial fuel mass"))
    out[:_system] = (m_fuel=o.m_fuel, t=tg, m0=m0, t_add=plan.t_add,
                     t_last_pull=plan.t_last_pull, Δu=diff(plan.u))
    return out, plan
end

"""
    system_goal_residuals(ys, goals, pieces) -> Vector

The residual vector over the pieces: each piece's `goal_residuals` in the system's piece order, the shared
terms (the fuel, the damper's smoothness) kept in the first piece's block and zeroed in the others'. J = Σ r².
"""
function system_goal_residuals(ys::AbstractDict, goals::AbstractDict{Symbol,PlanGoals}, pieces::AbstractVector{Symbol};
                               control::SystemControlGoals=SystemControlGoals())
    haskey(ys, :_system) || throw(ArgumentError("system outcomes are missing shared control bookkeeping"))
    shared_out = ys[:_system]
    at(y, tq) = _interp_t(y, shared_out.t, tq)
    lg(x) = 1 / (1 + exp(-x))
    m_added = isfinite(_primal(shared_out.t_add)) ?
              control.m_chimney * lg((shared_out.t_last_pull - shared_out.t_add) / TAU_ADD_S) :
              zero(shared_out.t_last_pull)
    fuel_time = max(shared_out.t_last_pull, first(shared_out.t))
    fuel_frac = (shared_out.m0 + m_added - at(shared_out.m_fuel, fuel_time)) / control.m0_ref
    fuel = sqrt(control.w_fuel * max(fuel_frac, zero(fuel_frac)) + 1e-12)
    if length(pieces) == 1
        p = only(pieces)
        r = copy(goal_residuals(ys[p], goals[p]))
        r[N_BAND + 2] = fuel
        r[(N_BAND + 3):(end - 2)] .= sqrt(control.w_smooth) .* ys[p][9:(end - 2)]
        return r
    end
    parts = Any[]
    for p in pieces
        r = copy(goal_residuals(ys[p], goals[p]))
        r[N_BAND + 2] = zero(r[N_BAND + 2])                            # shared fuel is appended once
        r[(N_BAND + 3):(end - 2)] .= zero(r[1])                         # shared damper smoothness is appended once
        push!(parts, r)
    end
    Δu = shared_out.Δu
    shared = vcat([fuel], sqrt(control.w_smooth) .* Δu)
    return vcat(reduce(vcat, parts), shared)
end

function _split_plan_draw(snap::BeliefSnapshot, v)
    nz = length(snap.z)
    length(v) == nz && return (z=v, y=snap.y)
    length(v) == nz + length(snap.y) || throw(ArgumentError("a planning draw must contain z or [z; y]"))
    return (z=view(v, 1:nz), y=view(v, (nz + 1):length(v)))
end

"Joint latent/state sigma draws from a belief snapshot, mean first."
belief_plan_draws(snap::BeliefSnapshot; n_dirs::Int=2) = sigma_draws(vcat(snap.z, snap.y), snap.zy_cov; n_dirs)

function _plan_residual(snap, v, ζ, f, s, goals, control, k, n_draws, hedge)
    d = _split_plan_draw(snap, v)
    ys = system_plan_outputs(snap, d.z, ζ, f, s, goals; y=d.y)[1]
    pieces = snap.sys.layout.pieces
    length(pieces) == 1 &&
        return draw_residual(system_goal_residuals(ys, goals, pieces; control), k, n_draws, hedge)
    parts = Any[]
    for p in pieces
        rp = copy(goal_residuals(ys[p], goals[p]))
        rp[N_BAND + 2] = zero(rp[N_BAND + 2])
        rp[(N_BAND + 3):(end - 2)] .= zero(rp[1])
        push!(parts, draw_residual(rp, k, n_draws, hedge))
    end
    raw = system_goal_residuals(ys, goals, pieces; control)
    n_piece = sum(length, parts)
    shared = raw[(n_piece + 1):end]
    shared_draw = (!hedge || n_draws == 1 || k == 1) ? shared : zeros(eltype(shared), length(shared))
    return vcat(reduce(vcat, parts), shared_draw)
end

"""
    system_plan_loss(snap, zs, plan, s, goals; hedge=true) -> (mean, draws)

The loss of a standing plan under the belief draws `zs` (the objective a replan minimises, without the plan
prior), as `plan_loss`.
"""
function system_plan_loss(snap::BeliefSnapshot, zs, plan, s::SystemSchedule, goals::AbstractDict{Symbol,PlanGoals};
                          hedge::Bool=true, control::SystemControlGoals=SystemControlGoals())
    f = system_free_set(s)
    ζ = encode_system_plan(plan, f, s)
    N = length(zs)
    ls = [0.5 * sum(abs2, _plan_residual(snap, z, ζ, f, s, goals, control, k, N, hedge)) for (k, z) in enumerate(zs)]
    return (mean=(hedge && N > 1) ? sum(ls) : sum(ls) / N, draws=ls)
end
system_plan_loss(snap::BeliefSnapshot, plan, s::SystemSchedule, goals::AbstractDict{Symbol,PlanGoals};
                 n_dirs::Int=2, kw...) =
    system_plan_loss(snap, belief_plan_draws(snap; n_dirs), plan, s, goals; kw...)

"""
    PlanFactor

The goal residuals of one belief draw as a factor on ζ: r(ζ) through the system's physics from the snapshot,
its Jacobian by ForwardDiff, cached by point.
"""
mutable struct PlanFactor <: Factor
    key::VarKey
    snap::BeliefSnapshot
    draw::Vector{Float64}
    f::SystemFreeSet
    s::SystemSchedule
    goals::Dict{Symbol,PlanGoals}
    control::SystemControlGoals
    k::Int
    n_draws::Int
    hedge::Bool
    n::Int
    cache::Any
end
PlanFactor(key, snap, draw, f, s, goals, k, n_draws, hedge, n, cache) =
    PlanFactor(key, snap, draw, f, s, goals, SystemControlGoals(), k, n_draws, hedge, n, cache)
vars(f::PlanFactor) = [f.key]
ndim(f::PlanFactor) = f.n
provenance(f::PlanFactor) = "the plan's goals under belief draw $(f.k) of $(f.n_draws)"
function _plan_eval!(f::PlanFactor, ζ; jac::Bool)
    c = f.cache
    c !== nothing && c.ζ == ζ && (!jac || c.J !== nothing) && return c
    g(v) = _plan_residual(f.snap, f.draw, v, f.f, f.s, f.goals, f.control, f.k, f.n_draws, f.hedge)
    if jac
        J = ForwardDiff.jacobian(g, ζ)
        r = c !== nothing && c.ζ == ζ ? c.r : g(ζ)
        f.cache = (ζ=copy(ζ), r=r, J=J)
    else
        f.cache = (ζ=copy(ζ), r=g(ζ), J=nothing)
    end
    return f.cache
end
residual(f::PlanFactor, vals) = _plan_eval!(f, vals[1]; jac=false).r
jacobians(f::PlanFactor, vals) = [_plan_eval!(f, vals[1]; jac=true).J]

"""
    replan_system(snap, zs, s, goals; ζ_init=nothing, iters=50, scan_wrap=(0.4, 0.55, 0.7, 0.85), scan_rest=(1.5, 3.0, 5.0), scan_damper=true, hedge=true, cold_start=true, plan_sd=PLAN_SD, verbose=false)

The plan from the snapshot on the factor graph: the expected loss over the draws `zs` (one plan factor per
draw), the plan prior, damped Gauss-Newton with the line search; a coordinate-wise scan (each piece's wrap
fraction, the rest, the damper's shapes with pull horizons) picks a second start from the warm start and, when
there is one, from the cold start; every distinct start is solved and the pick is by the same hedged,
prior-regularized objective represented by the graph. Nonstationary starts are never executable candidates.
Returns the decoded plan, ζ, the objective, the goals' objective, the covariance, the history, convergence, the
free set and the starts' count, as `replan_on_graph`.
"""
function replan_system(snap::BeliefSnapshot, zs, s::SystemSchedule, goals::AbstractDict{Symbol,PlanGoals}; ζ_init=nothing, iters::Int=50,
                       scan_wrap=(0.4, 0.55, 0.7, 0.85), scan_rest=(1.5, 3.0, 5.0), scan_damper::Bool=true, hedge::Bool=true,
                       cold_start::Bool=true, plan_sd::Float64=PLAN_SD, control::SystemControlGoals=SystemControlGoals(),
                       verbose::Bool=false, tol::Float64=1e-4, ftol::Float64=1e-4, gtol::Float64=2e-3, soc::Int=10)
    f = system_free_set(s)
    n = length(f); N = length(zs)
    goals_d = Dict{Symbol,PlanGoals}(goals)
    mζ, Λζ = system_plan_prior(f, s; plan_sd)
    c0 = ζ_init === nothing ? copy(mζ) : copy(ζ_init)
    dmean = _split_plan_draw(snap, zs[1])
    objective_scan(c) = 0.5 * sum(abs2, system_goal_residuals(
        system_plan_outputs(snap, dmean.z, c, f, s, goals_d; y=dmean.y)[1], goals_d, snap.sys.layout.pieces; control))
    objective_goals(c) = sum(0.5 * sum(abs2, _plan_residual(snap, draw, c, f, s, goals_d, control, k, N, hedge))
                             for (k, draw) in enumerate(zs))
    # the coordinates of the scan
    pos = N_KNOTS * f.damper
    k_wrap = Dict{Symbol,Int}(); k_pull = Dict{Symbol,Int}()
    for p in f.pieces
        f.wrap[p] && (k_wrap[p] = pos + 1; pos += 1)
        f.pull[p] && (k_pull[p] = pos + 1; pos += 1)
        f.spritz[p] && (pos += 3)
    end
    f.add && (pos += 1)
    k_serve = f.serve ? pos + 1 : 0
    rest_logit(c, h) = (pl = decode_system_plan(c, f, s); lgt(clamp(h * 3600 / max(T_SERVE_MAX - pl.t_last_pull, 600.0), 0.02, 0.98)))
    function scan_from(cs)
        best = (objective_scan(cs), cs)
        try_candidate!(c) = (Fc = try objective_scan(c) catch err; err isa InterruptException ? rethrow() : Inf end; Fc < best[1] && (best = (Fc, c)))
        for p in f.pieces                                            # each piece's wrap, holding the rest
            haskey(k_wrap, p) || continue
            for fw in scan_wrap
                c = copy(best[2]); c[k_wrap[p]] = lgt(fw); try_candidate!(c)
            end
        end
        if k_serve > 0
            for h in scan_rest
                c = copy(best[2]); c[k_serve] = rest_logit(c, h); try_candidate!(c)
            end
        end
        if scan_damper && f.damper
            shapes = ([0.3], [0.5], [0.15, 0.15, 0.5, 0.9, 0.9, 0.3, 0.15], [0.5, 0.9, 0.9, 0.3, 0.15, 0.15, 0.15], [0.15, 0.15, 0.15, 0.4, 0.9, 0.9, 0.5])
            for sh in shapes, hp in (isempty(k_pull) ? (nothing,) : (nothing, 8.0, 12.0))
                c = copy(best[2])
                c[1:N_KNOTS] .= lgt.(resample_knots(sh, N_KNOTS))
                if hp !== nothing
                    for p in f.pieces
                        haskey(k_pull, p) && (c[k_pull[p]] = lgt(clamp(hp * 3600 / (T_SERVE_MAX - s.t_now), 0.02, 0.98)))
                    end
                end
                try_candidate!(c)
            end
        end
        return best
    end
    starts = [c0]
    distinct(c) = all(maximum(abs.(c .- st)) > 1e-9 for st in starts)
    b1 = scan_from(c0)
    distinct(b1[2]) && push!(starts, b1[2])
    if cold_start && ζ_init !== nothing
        b2 = scan_from(copy(mζ))
        distinct(b2[2]) && push!(starts, b2[2])
    end
    n_out = length(system_goal_residuals(system_plan_outputs(snap, dmean.z, c0, f, s, goals_d; y=dmean.y)[1],
                                         goals_d, snap.sys.layout.pieces; control))
    key = VarKey(:plan, :zeta, 0)
    r = nothing; F_goals = Inf
    for c in starts
        g = FactorGraph()
        add_variable!(g, key, c)
        add_factor!(g, GaussianPrior(key, mζ, Matrix(sqrt.(Λζ)); source="the plan prior"))
        for (k, draw) in enumerate(zs)
            add_factor!(g, PlanFactor(key, snap, collect(Float64, draw), f, s, goals_d, control, k, N, hedge, n_out, nothing))
        end
        # A plan coordinate changes an event time by at most about 21_600 seconds per unit.
        # gtol=2e-3 is therefore a <=43-second local KKT displacement, below both this planner's
        # ten-minute trajectory grid and the physics' five-minute event transitions. An adaptive
        # ODE can stop resolving objective descent before that strict gate; only after three full
        # Armijo failures, permit a displacement no larger than the finest output-grid
        # interval under the largest logit-to-time sensitivity. Estimator graphs retain their
        # tighter coordinate tolerance.
        grid_step = minimum(diff(s.t))
        event_sensitivity = max(0.25 * (T_SERVE_MAX - s.t_now), grid_step)
        resolution_gtol = max(gtol, grid_step / event_sensitivity)
        rc = gauss_newton!(g; iters, tol, ftol, gtol, resolution_gtol, verbose, soc, adaptive_damping=true)
        ζc = copy(variable_value(g, key))
        rc.converged || continue
        Fg = try objective_goals(ζc) catch err; err isa InterruptException ? rethrow() : Inf end
        isfinite(Fg) || continue
        cov = try marginal_cov(g, key) catch; fill(NaN, n, n) end
        if r === nothing || rc.F < r.F
            r = (ζ=ζc, F=rc.F, hist=rc.hist, converged=rc.converged, cov=cov); F_goals = Fg
        end
    end
    r === nothing && throw(PlanConvergenceError("no planning start reached a stationary solution; keep the standing plan"))
    return (plan=decode_system_plan(r.ζ, f, s), ζ=r.ζ, F=r.F, F_goals=F_goals, cov=r.cov, hist=r.hist, converged=r.converged, free=f, n_starts=length(starts))
end

"Replan directly from the snapshot's joint latent/state uncertainty."
function replan_system(snap::BeliefSnapshot, s::SystemSchedule, goals::AbstractDict{Symbol,PlanGoals}; n_dirs::Int=2, kw...)
    return replan_system(snap, belief_plan_draws(snap; n_dirs), s, goals; kw...)
end

# --- when to replan, the commits, the execution -----------------------------------------------------------------

"The plan's events by (piece, kind) with their times and whether each is still free."
function _system_due_events(s::SystemSchedule, plan)
    f = system_free_set(s)
    out = Tuple{Tuple{Symbol,Symbol},Float64,Bool}[]
    for p in s.pieces
        pp = plan.pieces[p]
        push!(out, ((p, :wrap), pp.t_wrap, f.wrap[p]))
        push!(out, ((p, :pull), pp.t_pull, f.pull[p]))
        push!(out, ((p, :spritz), pp.spritz === nothing ? NaN : pp.spritz.t, f.spritz[p]))
    end
    push!(out, ((:pit, :add), plan.t_add, f.add))
    push!(out, ((:pit, :serve), plan.t_serve, f.serve))
    return out
end
"Whether to replan now (`replan_due`'s reasons over every piece's events)."
function system_replan_due(pol::ReplanPolicy, s::SystemSchedule, plan, L_now::Real, L_plan::Real, misfit_z)
    plan === nothing && return (true, :first)
    for (name, t, fr) in _system_due_events(s, plan)
        !isnan(t) && fr && s.t_now < t <= s.t_now + pol.lead && get(s.replanned_for, name, NaN) != t && return (true, :event_due)
    end
    misfit_z !== nothing && misfit_z > pol.misfit_z && return (true, :misfit)
    L_now > pol.loss_high && return (true, :loss_high)
    L_now - L_plan > pol.loss_rise && return (true, :loss_rise)
    return (false, :hold)
end
function note_system_replanned!(s::SystemSchedule, plan, pol::ReplanPolicy)
    for (name, t, fr) in _system_due_events(s, plan)
        !isnan(t) && fr && s.t_now < t <= s.t_now + pol.lead && (s.replanned_for[name] = t)
    end
    return s
end
"The events of a system plan left where the previous plan had them (within `tol` seconds), as `(piece, kind)`."
function stable_system_events(plan, prev, tol)
    prev === nothing && return Tuple{Symbol,Symbol}[]
    out = Tuple{Symbol,Symbol}[]
    same(a, b) = !isnan(a) && !isnan(b) && abs(a - b) <= tol
    for p in keys(plan.pieces)
        haskey(prev.pieces, p) || continue
        same(plan.pieces[p].t_wrap, prev.pieces[p].t_wrap) && push!(out, (p, :wrap))
        plan.pieces[p].spritz !== nothing && prev.pieces[p].spritz !== nothing && same(plan.pieces[p].spritz.t, prev.pieces[p].spritz.t) && push!(out, (p, :spritz))
        same(plan.pieces[p].t_pull, prev.pieces[p].t_pull) && push!(out, (p, :pull))
    end
    same(plan.t_add, prev.t_add) && push!(out, (:pit, :add))
    return out
end
export stable_system_events
"""
    commit_system!(s, plan, Δ; force=[], force_horizon=Inf) -> committed

The plan's events inside the next interval executed at their planned times and fixed forever, per piece; the
serve carried as the current plan.
"""
function commit_system!(s::SystemSchedule, plan, Δ; force=Tuple{Symbol,Symbol}[], force_horizon=Inf)
    done = Tuple{Symbol,Symbol}[]
    due(t, name) = !isnan(t) && s.t_now < t && (t <= s.t_now + Δ || (name in force && t <= s.t_now + force_horizon))
    for p in s.pieces
        pp = plan.pieces[p]
        if isnan(s.t_wrap[p]) && due(pp.t_wrap, (p, :wrap)); s.t_wrap[p] = pp.t_wrap; push!(done, (p, :wrap)); end
        if s.spritz[p] === nothing && pp.spritz !== nothing && due(pp.spritz.t, (p, :spritz))
            s.spritz[p] = (t=pp.spritz.t, dt=pp.spritz.dt, m=pp.spritz.m); push!(done, (p, :spritz))
        end
        if isnan(s.t_pull[p]) && due(pp.t_pull, (p, :pull)); s.t_pull[p] = pp.t_pull; push!(done, (p, :pull)); end
    end
    if isnan(s.t_add) && due(plan.t_add, (:pit, :add)); s.t_add = plan.t_add; push!(done, (:pit, :add)); end
    s.t_serve = plan.t_serve
    return done
end
"Execute the plan's damper over the next interval and advance the clock."
function execute_system_damper!(s::SystemSchedule, plan, Δ)
    u = system_plan_damper(plan, s)
    for (i, ti) in enumerate(s.t)
        s.t_now < ti <= s.t_now + Δ + 1e-9 && (s.u[i] = Float64(u[i]))
    end
    s.t_now += Δ
    return s
end

"""
    system_plan_predictive(snap, ζ, f, s, goals) -> Dict piece => (y_mean, y_sd), t, Tpit, per piece Tcore/Tsurf

What the belief predicts under the plan: every piece's outcomes with the sd the snapshot's joint covariance of
the latents and the state carries through the plan's physics (the Jacobian at the mean).
"""
function system_plan_predictive(snap::BeliefSnapshot, ζ, f::SystemFreeSet, s::SystemSchedule, goals::AbstractDict{Symbol,PlanGoals})
    pieces = snap.sys.layout.pieces
    nz = length(snap.z); bl = snap.bl
    function traj(v)
        z = v[1:nz]; y = v[(nz + 1):end]
        ys, plan = system_plan_outputs(snap, z, ζ, f, s, Dict{Symbol,PlanGoals}(goals); y)
        return reduce(vcat, [ys[p] for p in pieces])
    end
    v0 = vcat(snap.z, snap.y)
    m = traj(v0)
    J = ForwardDiff.jacobian(traj, v0, ForwardDiff.JacobianConfig(traj, v0, ForwardDiff.Chunk{min(length(v0), 12)}()))
    variance = max.(vec(sum((J * Matrix(Symmetric(snap.zy_cov))) .* J; dims=2)), 0.0)
    out = Dict{Symbol,NamedTuple}(); o = 0
    n_each = length(m) ÷ length(pieces)
    if snap.rom !== nothing
        os, plan, tg = system_plan_solve(snap, snap.z, ζ, f, s; y=snap.y)
        for (j, pc) in enumerate(snap.sys.pieces)
            q(η) = _planning_piece_quality(snap, pc, os.piece[pc.id], os, plan, tg, goals, η)
            Jrom = ForwardDiff.jacobian(q, zeros(2))
            rg = ((j - 1) * n_each + 1):(j * n_each)
            variance[rg] .+= vec(sum(abs2, Jrom; dims=2))
        end
    end
    sd = sqrt.(variance)
    for p in pieces
        out[p] = (mean=m[(o + 1):(o + n_each)], sd=sd[(o + 1):(o + n_each)]); o += n_each
    end
    return out
end
