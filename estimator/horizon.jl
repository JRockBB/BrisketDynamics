# The receding horizon on the graph: the plan replanned from the current
# state with the executed schedule fixed, the events committed when due
# and irreversible after, a one-sided floor at serve in place of a
# Gaussian goal. Included by BrisketEstimator after iterated.jl.
#
# The past is data. Every replan solves the whole cook from the start
# with the executed damper log and the executed events as inputs, and
# only the future carries decisions; the decisions live in seconds on
# one long grid, and the duration is the serve time at which the
# outcomes are read (interpolated on the grid). The free set shrinks as
# events commit: after the pull only the serve time is free.

export plan_loss, ReplanPolicy, replan_due, posterior_kl, stable_events
export hinge, Schedule, FreeSet, free_set, plan_prior, decode_plan, encode_plan, plan_damper,
       schedule_events, plan_outputs_rh, replan_on_graph, commit!, execute_damper!, PlanGoals

"A smooth one-sided penalty: softplus with width `s`, ≈ max(x, 0) — zero above the floor, linear below."
function hinge(x, s=1.0)
    y = x / s
    return s * (y > 30 ? y : log1p(exp(y)))
end

"""
    Schedule(t; ...)

The executed and committed cook schedule on the grid `t` [s]: the
damper `u` (NaN beyond `t_now`), the single events (`NaN` = not
executed), the spritz train once started, and the current planned
serve time.
"""
mutable struct Schedule
    t::Vector{Float64}
    u::Vector{Float64}
    t_now::Float64
    t_wrap::Float64
    t_pull::Float64
    t_add::Float64
    spritz::Union{Nothing,NamedTuple{(:t, :dt, :m),NTuple{3,Float64}}}
    t_serve::Float64
end
Schedule(t; u0::Float64=0.15, t_serve::Float64=14 * 3600.0) =
    Schedule(collect(Float64, t), [k == 1 ? u0 : NaN for k in eachindex(t)], 0.0, NaN, NaN, NaN, nothing, t_serve)

"Which decisions are free at the schedule's current time."
struct FreeSet
    damper::Bool
    wrap::Bool
    pull::Bool
    serve::Bool
    spritz::Bool
    add::Bool
end
function free_set(s::Schedule)
    pulled = !isnan(s.t_pull)
    return FreeSet(!pulled, isnan(s.t_wrap) && !pulled, !pulled, true, s.spritz === nothing && !pulled,
                   isnan(s.t_add) && !pulled)
end
Base.length(f::FreeSet) = 3 * f.damper + f.wrap + f.pull + f.serve + 3 * f.spritz + f.add

"The executed events as `monolith_solve` keywords (`NEVER` where not executed)."
function schedule_events(s::Schedule)
    sp = s.spritz
    return (t_wrap=isnan(s.t_wrap) ? BrisketDynamics.WRAP_NEVER : s.t_wrap,
            t_pull=isnan(s.t_pull) ? BrisketDynamics.PULL_NEVER : s.t_pull,
            t_add=isnan(s.t_add) ? BrisketDynamics.ADD_NEVER : s.t_add,
            t_spritz=sp === nothing ? BrisketDynamics.SPRITZ_NEVER : sp.t,
            dt_spritz=sp === nothing ? 3600.0 : sp.dt,
            m_spritz=sp === nothing ? 0.05 : sp.m)
end

# --- the plan block ------------------------------------------------------------------------
# ζ, in the free set's order: damper 3 logits | wrap logit f (of now→pull) | pull log h from now |
#   serve log h after the pull | spritz logit f (of now→pull), log dt h, log m kg | add logit f
const KNOT_F = [0.0, 1 / 3, 2 / 3]
lgt(x) = log(x / (1 - x))
lgs(x) = 1 / (1 + exp(-x))

"The prior on the free decisions: means and precisions in ζ's order."
function plan_prior(f::FreeSet, s::Schedule)
    m = Float64[]; sd = Float64[]
    # the damper's prior is a regulariser for the graph, not a fuel preference: three logit-sd wide, so a
    # half-open damper costs a sixth of a unit per knot (at one sd it cost 1.5 per knot, a cool cook by fiat)
    if f.damper; append!(m, fill(lgt(0.15), 3)); append!(sd, fill(3.0, 3)); end
    if f.wrap;   push!(m, lgt(0.7));  push!(sd, 1.0); end
    if f.pull;   push!(m, log(max((12 * 3600 - s.t_now) / 3600, 1.0))); push!(sd, 1.0); end
    if f.serve;  push!(m, log(1.5));  push!(sd, 0.5); end
    if f.spritz; append!(m, [lgt(0.3), log(1.0), log(0.05)]); append!(sd, [1.0, 0.5, 0.5]); end
    if f.add;    push!(m, lgt(0.5));  push!(sd, 1.0); end
    return m, Diagonal(1 ./ sd .^ 2)
end

"The decisions in seconds (executed ones from the schedule) from ζ."
function decode_plan(ζ, f::FreeSet, s::Schedule)
    k = 0
    take() = (k += 1; ζ[k])
    u = f.damper ? [lgs(take()) for _ in 1:3] : fill(isnan(s.u[findlast(!isnan, s.u)]) ? 0.15 : s.u[findlast(!isnan, s.u)], 3)
    fw = f.wrap ? lgs(take()) : NaN
    t_pull = f.pull ? s.t_now + 3600 * exp(take()) : s.t_pull
    t_serve = f.serve ? t_pull + 3600 * exp(take()) : s.t_serve
    if f.spritz
        fs = lgs(take()); dts = 3600 * exp(take()); ms = exp(take())
        spritz = (t=s.t_now + fs * (t_pull - s.t_now), dt=dts, m=ms)
    else
        spritz = s.spritz === nothing ? nothing : (t=s.spritz.t, dt=s.spritz.dt, m=s.spritz.m)
    end
    fa = f.add ? lgs(take()) : NaN
    t_wrap = f.wrap ? s.t_now + fw * (t_pull - s.t_now) : s.t_wrap
    t_add = f.add ? s.t_now + fa * (t_pull - s.t_now) : s.t_add
    return (u=u, t_wrap=t_wrap, t_pull=t_pull, t_serve=t_serve, spritz=spritz, t_add=t_add)
end

"ζ from decisions in seconds (the warm start of a replan from the previous plan); values outside the free set are ignored."
function encode_plan(plan, f::FreeSet, s::Schedule)
    clampf(x) = clamp(x, 0.02, 0.98)
    ζ = Float64[]
    f.damper && append!(ζ, lgt.(clamp.(plan.u, 0.02, 0.98)))
    rem = max(plan.t_pull - s.t_now, 600.0)
    f.wrap && push!(ζ, lgt(clampf((plan.t_wrap - s.t_now) / rem)))
    f.pull && push!(ζ, log(rem / 3600))
    f.serve && push!(ζ, log(max(plan.t_serve - plan.t_pull, 600.0) / 3600))
    if f.spritz
        sp = plan.spritz === nothing ? (t=s.t_now + 0.3 * rem, dt=3600.0, m=0.05) : plan.spritz
        append!(ζ, [lgt(clampf((sp.t - s.t_now) / rem)), log(sp.dt / 3600), log(sp.m)])
    end
    f.add && push!(ζ, lgt(clampf((plan.t_add - s.t_now) / rem)))
    return ζ
end

"The damper on the grid: executed where it is, the plan's knots (piecewise linear, held after the last) beyond now."
function plan_damper(plan, s::Schedule)
    Tel = promote_type(eltype(plan.u), typeof(plan.t_pull), Float64)
    u = Vector{Tel}(undef, length(s.t))
    tk = s.t_now .+ KNOT_F .* (plan.t_pull - s.t_now)
    for (i, ti) in enumerate(s.t)
        if ti <= s.t_now && !isnan(s.u[i])
            u[i] = s.u[i]
        elseif ti <= tk[1]
            u[i] = plan.u[1]
        elseif ti >= tk[end]
            u[i] = plan.u[end]
        else
            j = ti <= tk[2] ? 1 : 2
            w = (ti - tk[j]) / (tk[j + 1] - tk[j])
            u[i] = (1 - w) * plan.u[j] + w * plan.u[j + 1]
        end
    end
    return u
end

function _interp_t(y, tt, tq)
    k = clamp(searchsortedlast(tt, ForwardDiff.value(tq)), 1, length(tt) - 1)
    w = (tq - tt[k]) / (tt[k + 1] - tt[k])
    return (1 - w) * y[k] + w * y[k + 1]
end

"""
    plan_outputs_rh(z, ζ, f, s, d) -> [core at the pull, retained water at serve, bark L* at serve, core at serve]

The plan's outcomes at latents `z` (physics, area) and plan `ζ`, the
whole cook solved from the start with the executed schedule and the
plan's future.
"""
function plan_outputs_rh(z, ζ, f::FreeSet, s::Schedule, d::GateData)
    x = from_z(z[1:N_PHYS], d.prior[1:N_PHYS])
    A = length(z) >= 16 ? exp(z[16]) : d.cfg.A_meat
    p = decode_plan(ζ, f, s)
    u = plan_damper(p, s)
    sp = p.spritz
    o = monolith_solve(x[1:N_THETA], x[(N_THETA + 1):N_PHYS], u, s.t, s.t; pit=d.pit, cfg=d.cfg,
                       abstol=d.abstol, reltol=d.reltol, A_meat=A,
                       t_wrap=isnan(p.t_wrap) ? BrisketDynamics.WRAP_NEVER : p.t_wrap, t_pull=p.t_pull,
                       t_add=isnan(p.t_add) ? BrisketDynamics.ADD_NEVER : p.t_add,
                       t_spritz=sp === nothing ? BrisketDynamics.SPRITZ_NEVER : sp.t,
                       dt_spritz=sp === nothing ? 3600.0 : sp.dt, m_spritz=sp === nothing ? 0.05 : sp.m)
    wfv = fv_weights(size(o.W_field, 1))
    wret = [dot(wfv, view(o.W_field, :, k)) / d.cfg.W0 for k in axes(o.W_field, 2)]
    return [_interp_t(view(o.C_field, 1, :), s.t, p.t_pull), _interp_t(wret, s.t, p.t_serve),
            _interp_t(o.l_star, s.t, p.t_serve), _interp_t(o.Tcore, s.t, p.t_serve)]
end

"""
The goals, in a pitmaster's terms, as bands: a goal costs nothing
inside its band and half the squared distance outside it, in units of
its sd (a softplus corner a quarter-sd wide keeps the graph's affine
nodes honest). The pull is probe-tender: collagen conversion at the
center node in the band the app's quality objective uses, 0.90 (done)
to 0.95 (pot roast), sd 0.02. The water at serve is a floor at what
this plant's physics retains at doneness on a cook of normal length,
0.45 of the initial dry-basis water (the app's own optimum lands at
0.45 to 0.48; the model's water budget is a roadmap residue), sd 0.04.
The bark is the mahogany band, L* 28 to 34, sd 3. The floor at serve
is the slicing temperature. A Gaussian at each band's center made the
pilot cook six hours longer than the app's MPC on the same brisket
(`docs/results.md`, the pilot in a pitmaster's terms).
"""
Base.@kwdef struct PlanGoals
    C_lo::Float64 = 0.90
    C_hi::Float64 = 0.95
    sd_C::Float64 = 0.02
    W_lo::Float64 = 0.45
    sd_W::Float64 = 0.04
    L_lo::Float64 = 28.0
    L_hi::Float64 = 34.0
    sd_L::Float64 = 3.0
    T_serve_floor::Float64 = 273.15 + 65.0
    sd_serve::Float64 = 1.0          # the hinge's scale [K]: a draw ΔT under the floor costs ½ (ΔT/sd)²
end

"A two-sided band residual: zero inside, the distance outside in sd; the corners a quarter-sd wide."
band_residual(y, lo, hi, sd) = (hinge(lo - y, sd / 4) + hinge(y - hi, sd / 4)) / sd
"A one-sided floor residual: zero above, the shortfall in sd."
floor_residual(y, lo, sd) = hinge(lo - y, sd / 4) / sd

"The goal residuals of one draw's outputs: the collagen band at the pull, the water floor, the bark band and the serve floor."
goal_residuals(y, g::PlanGoals) = [band_residual(y[1], g.C_lo, g.C_hi, g.sd_C), floor_residual(y[2], g.W_lo, g.sd_W),
                                   band_residual(y[3], g.L_lo, g.L_hi, g.sd_L), hinge(g.T_serve_floor - y[4]) / g.sd_serve]

# the graph: per-draw affine nodes of the four outputs, the LM node, the plan prior
const RH_AFFS = Ref{Vector{Tuple{Vector{Float64},Matrix{Float64}}}}(Tuple{Vector{Float64},Matrix{Float64}}[])
const RH_CENTER = Ref{Vector{Float64}}(Float64[])
rh_node(ζ, i) = RH_AFFS[][i][1] .+ RH_AFFS[][i][2] * (ζ .- RH_CENTER[])

@model function rh_graph(g, lm_obs, Λ_lm, mζ, Λζ, N, Σg)
    ζ ~ MvNormalMeanPrecision(mζ, Λζ)
    for i in 1:N
        y[i] := rh_node(ζ, i)
        g[i] ~ MvNormalMeanCovariance(y[i], Σg)
    end
    lm := lm_node(ζ)
    lm_obs ~ MvNormalMeanPrecision(lm, Λ_lm)
end
const RH_META = @meta begin
    rh_node() -> Linearization()
    lm_node() -> Linearization()
end
@initialization function rh_init(mζ, Λζ)
    μ(ζ) = MvNormalMeanPrecision(mζ, Λζ)
end

"""
    replan_on_graph(zs, s, d; goals, ζ_init, verbose) -> (plan, ζ, F, cov, hist, converged, free)

The plan from the current state on the graph: the expected loss over the
draws `zs` (the goal residuals through the physics, one affine node per
draw of the four outputs, the tail as a residual like any other), the
plan prior, the graph iterator; warm-started at `ζ_init` when given.
"""
function replan_on_graph(zs, s::Schedule, d::GateData; goals::PlanGoals=PlanGoals(), ζ_init=nothing,
                         verbose::Bool=false, iters::Int=30)
    f = free_set(s)
    n = length(f)
    mζ, Λζ = plan_prior(f, s)
    c0 = ζ_init === nothing ? copy(mζ) : copy(ζ_init)
    # the residual vector of a plan: goals per draw, then the prior
    resid(ζ) = vcat([goal_residuals(plan_outputs_rh(z, ζ, f, s, d), goals) for z in zs]..., sqrt(Λζ) * (ζ .- mζ))
    objective(c) = 0.5 * sum(abs2, resid(c))
    # the per-draw affine model of the GOAL RESIDUALS (the hinge included) at c: goal ~ N(res, 1)
    function rebuild!(c)
        RH_AFFS[] = [(goal_residuals(plan_outputs_rh(z, c, f, s, d), goals),
                      ForwardDiff.jacobian(v -> goal_residuals(plan_outputs_rh(z, v, f, s, d), goals), c)) for z in zs]
        RH_CENTER[] = copy(c)
    end
    prob = IteratedProblem(:ζ, objective, rebuild!, n)
    g = [zeros(4) for _ in 1:length(zs)]
    Σg = Matrix(1.0 * I, 4, 4)
    run_infer(Λ_lm, callbacks, its) = infer(model = rh_graph(mζ = mζ, Λζ = Matrix(Λζ), N = length(zs), Σg = Σg),
                                            data = (g = g, lm_obs = zeros(n), Λ_lm = Λ_lm), meta = RH_META,
                                            initialization = rh_init(RH_CENTER[], Matrix(Λζ)), iterations = its,
                                            callbacks = callbacks, returnvars = (ζ = KeepLast(),))
    r = iterated_solve(prob, run_infer; c0, iters, verbose)
    return (plan=decode_plan(r.z, f, s), ζ=r.z, F=r.F, cov=r.cov, hist=r.hist, converged=r.converged, free=f)
end

# --- when to replan ------------------------------------------------------------------------
# A replan is a decision, not a cadence. After each posterior update the
# standing plan is re-evaluated under the new draws (a plant run per
# draw, no optimisation): its expected loss is what a replan would start
# from. The replan fires when that loss has risen since the plan was
# made (the new information changed the plan's value), when a planned
# event falls due within the lead (it is about to commit and become
# irreversible: decide it on the freshest posterior), when the misfit
# check fires (the model is wrong somewhere: the plan's outlook is
# unreliable), or when a floor interval has passed without one.

"""
    plan_loss(zs, plan, s, d; goals) -> (mean, draws)

The expected loss of a standing `plan` under the posterior draws `zs`:
the mean over draws of half the squared goal residuals, the objective a
replan minimises without the plan prior. The plan is re-encoded in the
current free set (its committed events are fixed in `s`).
"""
function plan_loss(zs, plan, s::Schedule, d::GateData; goals::PlanGoals=PlanGoals())
    f = free_set(s)
    ζ = encode_plan(plan, f, s)
    ls = [0.5 * sum(abs2, goal_residuals(plan_outputs_rh(z, ζ, f, s, d), goals)) for z in zs]
    return (mean=sum(ls) / length(ls), draws=ls)
end

"""
    ReplanPolicy(; loss_rise=1.0, misfit_z=3.0, lead=3600.0, floor=4*3600.0)

When a replan is due: the standing plan's expected loss has risen by
`loss_rise` (half-squared-sd units) since it was made; a planned,
uncommitted event falls due within `lead` seconds; the misfit z-score
exceeds `misfit_z` (excess residual power only: too little power means
the noise model is conservative, which does not threaten the plan); or
`floor` seconds have passed since the last replan.
"""
Base.@kwdef struct ReplanPolicy
    loss_rise::Float64 = 1.0
    loss_high::Float64 = 3.0
    kl::Float64 = 2.0
    misfit_z::Float64 = 3.0
    lead::Float64 = 3600.0
    floor::Float64 = 4 * 3600.0
    stable::Float64 = 1800.0         # an event a replan moved by less than this commits now, however far ahead
end

"""
    stable_events(plan, prev, tol) -> Vector{Symbol}

The events a replan left where they stood: those of `plan` within `tol`
seconds of their time in the previous plan `prev` (wrap, add, spritz,
pull). An event about to commit forces a replan on the freshest
posterior; when that replan does not move it, deferring it another hour
buys nothing, and it should commit.
"""
function stable_events(plan, prev, tol)
    prev === nothing && return Symbol[]
    out = Symbol[]
    same(a, b) = !isnan(a) && !isnan(b) && abs(a - b) <= tol
    same(plan.t_wrap, prev.t_wrap) && push!(out, :wrap)
    same(plan.t_add, prev.t_add) && push!(out, :add)
    plan.spritz !== nothing && prev.spritz !== nothing && same(plan.spritz.t, prev.spritz.t) && push!(out, :spritz)
    same(plan.t_pull, prev.t_pull) && push!(out, :pull)
    return out
end

"""
    posterior_kl(m_new, V_new, m_old, V_old) -> nats

KL(new ‖ old) between two Gaussian posteriors: how much the belief has
moved since the plan was made, in nats (one full-sd shift of one latent
is half a nat; halving every width of a 17-latent posterior is five).
"""
function posterior_kl(m_new, V_new, m_old, V_old)
    n = length(m_new)
    Vo = Symmetric(Matrix(V_old)); Vn = Symmetric(Matrix(V_new))
    d = m_old .- m_new
    return 0.5 * (tr(Vo \ Vn) + dot(d, Vo \ d) - n + logdet(Vo) - logdet(Vn))
end

"""
    replan_due(pol, s, plan, L_now, L_plan, t_last, misfit_z; kl=nothing) -> (due::Bool, reason::Symbol)

Whether to replan now: `plan` is the standing plan (`nothing` before the
first), `L_now` its expected loss under the current posterior, `L_plan`
its loss when it was made, `t_last` the time of the last replan,
`misfit_z` the latest misfit z-score (or `nothing`), `kl` the posterior's
move since the plan was made (or `nothing`). The reason is one of
`:first`, `:event_due`, `:misfit`, `:loss_high` (the standing plan is
bad on its own terms), `:loss_rise` (worse than when made),
`:posterior_moved` (the belief the plan was made on is gone), `:floor`,
`:hold`. A standing plan is scored against itself by `loss_rise`; when
the posterior tightens that score can fall while a far better plan has
become available, which `loss_high` and `posterior_moved` catch.
"""
function replan_due(pol::ReplanPolicy, s::Schedule, plan, L_now::Real, L_plan::Real, t_last::Real, misfit_z; kl=nothing)
    plan === nothing && return (true, :first)
    # only an event still free to move can come due: a standing plan's spritz after the pull has committed is no decision
    f = free_set(s)
    events = [plan.t_wrap, plan.t_pull, plan.t_add, plan.spritz === nothing ? NaN : plan.spritz.t, plan.t_serve]
    free = [f.wrap, f.pull, f.add, f.spritz, f.serve]
    for (t, fr) in zip(events, free)
        !isnan(t) && fr && s.t_now < t <= s.t_now + pol.lead && return (true, :event_due)
    end
    # one-sided: excess residual power says the model is wrong somewhere; too little says it is conservative
    misfit_z !== nothing && misfit_z > pol.misfit_z && return (true, :misfit)
    L_now > pol.loss_high && return (true, :loss_high)
    L_now - L_plan > pol.loss_rise && return (true, :loss_rise)
    kl !== nothing && kl > pol.kl && return (true, :posterior_moved)
    s.t_now - t_last >= pol.floor - 1e-9 && return (true, :floor)
    return (false, :hold)
end

"""
    commit!(s, plan, Δ; force=Symbol[]) -> committed::Vector{Symbol}

The candidate events of `plan` that fall inside the next interval
(now, now + Δ] are executed at their planned times and fixed forever;
the spritz train starts when its first spritz is due; the serve time is
carried as the current plan. Events named in `force` (see
`stable_events`) commit at any future time.
"""
function commit!(s::Schedule, plan, Δ; force=Symbol[])
    done = Symbol[]
    due(t, name) = !isnan(t) && s.t_now < t && (t <= s.t_now + Δ || name in force)
    if isnan(s.t_wrap) && due(plan.t_wrap, :wrap); s.t_wrap = plan.t_wrap; push!(done, :wrap); end
    if isnan(s.t_add) && due(plan.t_add, :add); s.t_add = plan.t_add; push!(done, :add); end
    if s.spritz === nothing && plan.spritz !== nothing && due(plan.spritz.t, :spritz)
        s.spritz = (t=plan.spritz.t, dt=plan.spritz.dt, m=plan.spritz.m); push!(done, :spritz)
    end
    if isnan(s.t_pull) && due(plan.t_pull, :pull); s.t_pull = plan.t_pull; push!(done, :pull); end
    s.t_serve = plan.t_serve
    return done
end

"Execute the plan's damper over the next interval and advance the clock."
function execute_damper!(s::Schedule, plan, Δ)
    u = plan_damper(plan, s)
    for (i, ti) in enumerate(s.t)
        if s.t_now < ti <= s.t_now + Δ + 1e-9
            s.u[i] = Float64(u[i])
        end
    end
    s.t_now += Δ
    return s
end
