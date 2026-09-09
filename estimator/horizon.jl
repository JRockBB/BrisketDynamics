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

export plan_loss, ReplanPolicy, replan_due, posterior_kl, stable_events, cook_outputs, objective_terms, executed_du
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
    if f.serve;  push!(m, log(1.5));  push!(sd, 1.5); end     # the rest: a regulariser, not a rule (a 4 h rest costs 0.2)
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
The objective: the app's KCBS-weighted quality objective
(`brisket_app/control.py`, `ControlConfig` and `score_outputs`), ported
term for term so the pilot and the MPC optimise one thing. Tenderness
is a two-sided band on collagen conversion at the center at serve
(under 0.90 tough, over 0.95 pot roast, each side normalised to its
band width); appearance the mahogany band on L*; taste the smoke
deposition band plus dryness on the perceptual juiciness; then the
bark-burn cap on the surface in the pit, the fuel burned to the pull,
damper smoothness, and the lethality curve on the surface and the core
(the come-up time is reported, never binding, for brisket). The
defaults are the app's. `goal_residuals` returns the residuals `r` with
J = Σ r², so the graph's unit-variance nodes minimise ½ J.
"""
Base.@kwdef struct PlanGoals
    w_appearance::Float64 = 0.14
    w_taste::Float64 = 0.57
    w_tenderness::Float64 = 0.29
    c_target::Float64 = 0.90
    c_overdone::Float64 = 0.95
    over_rel::Float64 = 1.0
    l_lo::Float64 = 28.0
    l_hi::Float64 = 34.0
    dl_soft::Float64 = 2.0
    s_lo::Float64 = 0.55
    s_hi::Float64 = 0.90
    ds_soft::Float64 = 0.08
    w_dry::Float64 = 4.0
    t_surf_max::Float64 = 368.15
    dT_soft::Float64 = 2.0
    w_surf::Float64 = 1.0
    w_fuel::Float64 = 0.1
    w_smooth::Float64 = 0.01
    w_lethality::Float64 = 500.0
    leth_t_ref::Float64 = 333.15
    leth_z::Float64 = 6.0
    leth_f_req::Float64 = 720.0
    m_chimney::Float64 = 1.4
    m0_ref::Float64 = 7.0
end

"""
    cook_outputs(o, t, Δu, t_pull, t_add, t_serve, cfg, g) -> y

What the app's quality objective reads from one cook, from a monolith
output `o` on the grid `t` [s]: collagen conversion at the center at
serve, retained water at serve, bark L* at serve, the core at serve [K],
the smoke deposition at serve, the perceptual juiciness at serve (water
plus the retained rendered fat, as `brisket_app/mpc.py`), the
surface-burn term (the mean squared excess over the bark-burn cap while
in the pit), the fuel fraction burned to the pull (the load plus a
chimney, priced at the pull), the damper differences `Δu`, and the
lethality integrals on the surface and the core to serve
[60 °C-equivalent s]. Every quantity at serve is read at `t_serve` on
the grid, the integrals with a smooth horizon at `t_serve`, so the plan
can move the serve.
"""
function cook_outputs(o, t, Δu, t_pull, t_add, t_serve, cfg, g)
    at(y, tq) = _interp_t(y, t, tq)
    wfv = fv_weights(size(o.W_field, 1))
    wret = [dot(wfv, view(o.W_field, :, k)) / cfg.W0 for k in axes(o.W_field, 2)]
    fmean = [dot(wfv, view(o.F_field, :, k)) for k in axes(o.F_field, 2)]
    W = at(wret, t_serve)
    juiciness = W + cfg.β_fat * (1 - cfg.η_drain) * (cfg.F0 - at(fmean, t_serve)) / cfg.W0
    lg(x) = 1 / (1 + exp(-x))
    sp(x) = x > 30 ? x : log1p(exp(x))
    horizon = [lg((t_serve - tk) / 600.0) for tk in t]                              # inside the cook, smoothly
    in_pit = [horizon[k] * (1 - lg((t[k] - t_pull) / DELTA_PULL_S)) for k in eachindex(t)]
    surf_term = sum(in_pit[k] * sp((o.Tsurf[k] - g.t_surf_max) / g.dT_soft)^2 for k in eachindex(t)) / sum(in_pit)
    m_added = g.m_chimney * lg((t_pull - t_add) / TAU_ADD_S)
    fuel_frac = (o.m_fuel[1] + m_added - at(o.m_fuel, t_pull)) / g.m0_ref
    leth(T) = sum((t[k + 1] - t[k]) * (horizon[k] * 10^((T[k] - g.leth_t_ref) / g.leth_z) +
                                        horizon[k + 1] * 10^((T[k + 1] - g.leth_t_ref) / g.leth_z)) / 2 for k in 1:(length(t) - 1))
    return vcat([at(view(o.C_field, 1, :), t_serve), W, at(o.l_star, t_serve), at(o.Tcore, t_serve), at(o.bark_s, t_serve),
                 juiciness, surf_term, fuel_frac], Δu, [leth(o.Tsurf), leth(o.Tcore)])
end
const DELTA_PULL_S = 300.0     # the pull blend, as brisket_app/control.py
const TAU_ADD_S = 300.0        # the chimney dump duration, as brisket_app/control.py

"""
    plan_outputs_rh(z, ζ, f, s, d; goals) -> y (see `cook_outputs`)

The plan's outcomes at latents `z` (physics, area) and plan `ζ`, the
whole cook solved from the start with the executed schedule and the
plan's future.
"""
function plan_outputs_rh(z, ζ, f::FreeSet, s::Schedule, d::GateData; goals::PlanGoals=PlanGoals())
    x = from_z(z[1:N_PHYS], d.prior[1:N_PHYS])
    A = length(z) >= 16 ? exp(z[16]) : d.cfg.A_meat
    p = decode_plan(ζ, f, s)
    u = plan_damper(p, s)
    sp = p.spritz
    t_add = isnan(p.t_add) ? BrisketDynamics.ADD_NEVER : p.t_add
    o = monolith_solve(x[1:N_THETA], x[(N_THETA + 1):N_PHYS], u, s.t, s.t; pit=d.pit, cfg=d.cfg,
                       abstol=d.abstol, reltol=d.reltol, A_meat=A,
                       t_wrap=isnan(p.t_wrap) ? BrisketDynamics.WRAP_NEVER : p.t_wrap, t_pull=p.t_pull, t_add=t_add,
                       t_spritz=sp === nothing ? BrisketDynamics.SPRITZ_NEVER : sp.t,
                       dt_spritz=sp === nothing ? 3600.0 : sp.dt, m_spritz=sp === nothing ? 0.05 : sp.m)
    return cook_outputs(o, s.t, diff(p.u), p.t_pull, t_add, p.t_serve, d.cfg, goals)
end


"The residuals of one cook's outputs (`cook_outputs`) under the objective: J = Σ r²."
function goal_residuals(y, g::PlanGoals)
    sp(x) = x > 30 ? x : log1p(exp(x))
    c, L, S, juiciness, surf, fuel = y[1], y[3], y[5], y[6], y[7], y[8]
    Δu = y[9:(end - 2)]
    Fs, Fc = y[end - 1], y[end]
    under = sp(20 * (g.c_target - c)) / 20 / 0.05
    band_o = 1 - g.c_overdone
    k_ov = 4 / band_o
    over = sp(k_ov * (c - g.c_overdone)) / k_ov / band_o
    leth(F) = sp((log10(g.leth_f_req) - log10(F + 1e-9)) / 0.2)
    return vcat([sqrt(g.w_tenderness) * under, sqrt(g.w_tenderness * g.over_rel) * over,
                 sqrt(g.w_appearance) * sp((L - g.l_hi) / g.dl_soft), sqrt(g.w_appearance) * sp((g.l_lo - L) / g.dl_soft),
                 sqrt(g.w_taste) * sp((g.s_lo - S) / g.ds_soft), sqrt(g.w_taste) * sp((S - g.s_hi) / g.ds_soft),
                 sqrt(g.w_taste * g.w_dry) * (1 - juiciness),
                 sqrt(g.w_surf * surf + 1e-12), sqrt(g.w_fuel * max(fuel, zero(fuel)) + 1e-12)],
                sqrt(g.w_smooth) .* Δu,
                [sqrt(g.w_lethality) * leth(Fs), sqrt(g.w_lethality) * leth(Fc)])
end

"The objective's value and its named terms, as the app reports them (J = Σ r²)."
function objective_terms(y, g::PlanGoals)
    r = goal_residuals(y, g)
    return (tenderness=r[1]^2 + r[2]^2, appearance=r[3]^2 + r[4]^2, taste=r[5]^2 + r[6]^2 + r[7]^2, surf=r[8]^2, fuel=r[9]^2,
            smooth=sum(abs2, r[10:(end - 2)]), safety=r[end - 1]^2 + r[end]^2, J=sum(abs2, r))
end

"The executed damper's differences between hourly means to the pull: the smoothness term of an executed cook (as the app's `executed_objective`)."
function executed_du(t, u, t_pull)
    hours = floor(Int, min(t_pull, t[end]) / 3600)
    means = [mean(u[(t .>= (h - 1) * 3600) .& (t .< h * 3600)]) for h in 1:hours]
    return length(means) >= 2 ? diff(means) : Float64[]
end

# the graph: per-draw affine nodes of the residuals, the LM node, the plan prior
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
    resid(ζ) = vcat([goal_residuals(plan_outputs_rh(z, ζ, f, s, d; goals), goals) for z in zs]..., sqrt(Λζ) * (ζ .- mζ))
    objective(c) = 0.5 * sum(abs2, resid(c))
    # the per-draw affine model of the GOAL RESIDUALS (the hinge included) at c: goal ~ N(res, 1)
    function rebuild!(c)
        RH_AFFS[] = [(goal_residuals(plan_outputs_rh(z, c, f, s, d; goals), goals),
                      ForwardDiff.jacobian(v -> goal_residuals(plan_outputs_rh(z, v, f, s, d; goals), goals), c)) for z in zs]
        RH_CENTER[] = copy(c)
    end
    prob = IteratedProblem(:ζ, objective, rebuild!, n)
    n_out = length(goal_residuals(plan_outputs_rh(zs[1], c0, f, s, d; goals), goals))
    g = [zeros(n_out) for _ in 1:length(zs)]
    Σg = Matrix(1.0 * I, n_out, n_out)
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
    ls = [0.5 * sum(abs2, goal_residuals(plan_outputs_rh(z, ζ, f, s, d; goals), goals)) for z in zs]
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
    loss_rise::Float64 = 0.25        # in half-J units: the app's J is 0.6 for a good cook, 14 for a ruined one
    loss_high::Float64 = 1.0
    kl_per_latent::Float64 = 0.25    # the posterior's move since the plan, in nats per latent (a quarter-sd shift of every latent)
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
    replan_due(pol, s, plan, L_now, L_plan, t_last, misfit_z; kl=nothing, n_latent=1) -> (due::Bool, reason::Symbol)

Whether to replan now: `plan` is the standing plan (`nothing` before the
first), `L_now` its expected loss under the current posterior, `L_plan`
its loss when it was made, `t_last` the time of the last replan,
`misfit_z` the latest misfit z-score (or `nothing`), `kl` the posterior's
move since the plan was made in nats (or `nothing`) against a threshold
of `kl_per_latent` × `n_latent`. The reason is one of
`:first`, `:event_due`, `:misfit`, `:loss_high` (the standing plan is
bad on its own terms), `:loss_rise` (worse than when made),
`:posterior_moved` (the belief the plan was made on is gone), `:floor`,
`:hold`. A standing plan is scored against itself by `loss_rise`; when
the posterior tightens that score can fall while a far better plan has
become available, which `loss_high` and `posterior_moved` catch.
"""
function replan_due(pol::ReplanPolicy, s::Schedule, plan, L_now::Real, L_plan::Real, t_last::Real, misfit_z; kl=nothing, n_latent::Int=1)
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
    kl !== nothing && kl > pol.kl_per_latent * n_latent && return (true, :posterior_moved)
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
