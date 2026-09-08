# The full decision set on the graph: damper knots, the wrap, the pull,
# the duration (the time-scaling solve), the spritz (start, interval,
# dose) and a fuel addition, every one an unconstrained latent of one
# plan block with the ordering built into the parameterization (the wrap
# and the spritz start as fractions of the pull, the pull and the fuel
# add as fractions of the duration), the physics `monolith_solve_tf` at
# the estimator's posterior draws (area latent included), Gaussian goals
# on the core temperature at the pull and at serve, the retained water
# and the bark lightness, fuel-economy and cook-time priors. Solved two
# ways on the same expected loss: the gradient reference (damped
# Gauss-Newton, ForwardDiff through the physics) and the graph
# (per-draw affine nodes inside the graph iterator). The gate: the graph
# reproduces the reference.
#
# Run (each solve writes its record as soon as it finishes; the two can run in parallel):
#   julia --project=estimator estimator/plan_block.jl reference
#   julia --project=estimator estimator/plan_block.jl graph
#   julia --project=estimator estimator/plan_block.jl summary
# Writes data/mock/plan_block_reference.json, plan_block_graph.json, plan_block.json.
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using RxInfer, ForwardDiff, JSON, LinearAlgebra, Random, Printf, Statistics, Test
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json"))
S = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d_stream.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
t = Float64.(m["t"]); u_log = Float64.(m["u"]); meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
d = estimator_setup(t, u_log, pit, cfg, prior_phys, String.(gate["names"]))
L_med = exp(prior_phys[4].p1)
prior = vcat(prior_phys, [PriorSite("lognormal", log(cfg.A_meat), 0.3), PriorSite("lognormal", log(0.8 * L_med), 0.5)])
AREA, DEPTH = 16, 17

# --- the posterior at 6 h and its draws (as in the planning experiment) ------------------
st = first(s for s in S["sets"] if s["label"] == "pit + core + humidity + scale")
b6 = first(b for b in st["batches"] if b["t_h"] == 6)
z6 = Float64.(b6["z"])
rd = m["readings"]; nz_ = m["noise"]
keep = t .<= 6 * 3600 + 1e-9
chs = [PitProbe(nz_["pit"]["sigma"], nz_["pit"]["rho"], keep; data=Float64.(rd["pit"])),
       MeatProbe(nz_["core"]["sigma"], nz_["core"]["rho"], keep; depth_latent=DEPTH, data=Float64.(rd["core"])),
       HumiditySensor(nz_["humidity"]["sigma"], nz_["humidity"]["rho"], keep; data=Float64.(rd["humidity"])),
       MassScale(nz_["mass"]["sigma"], nz_["mass"]["rho"], keep; data=Float64.(rd["mass"]))]
Sx, J, lay = physics_outputs(z6, d; area_latent=AREA)
a, B, w = assemble(chs, z6, Sx, J, prior, lay)
_, V6 = graph_posterior_affine(z6, a, B, w, prior)
N = 8
rng = MersenneTwister(3)
Lc = cholesky(Symmetric(V6)).L
zs = [z6 .+ Lc * randn(rng, length(z6)) for _ in 1:N]

# --- the plan block ----------------------------------------------------------------------
# ζ = [logit u1, logit u2, logit u3, logit f_wrap, logit f_pull, log t_f[h], logit f_spritz,
#      log frac_dt_spritz, log m_spritz[kg], logit f_add]
const S_U = [0.0, 0.35, 0.70]                     # damper knots in fractional time
const S_SAVE = collect(range(0.0, 1.0; length=115))
const PLAN_NAMES = ["damper 1", "damper 2", "damper 3", "wrap (of pull)", "pull (of cook)", "duration [h]",
                    "spritz start (of pull)", "spritz interval (of cook)", "spritz dose [kg]", "fuel add (of pull)"]
lg(x) = log(x / (1 - x))
ζ0 = [lg(0.15), lg(0.15), lg(0.15), lg(0.6), lg(0.8), log(14.0), lg(0.3), log(0.07), log(0.05), lg(0.5)]
Λ0 = Diagonal(1 ./ [0.5, 0.5, 0.5, 1.0, 0.5, 0.3, 1.0, 0.5, 0.5, 1.0] .^ 2)
const NP = length(ζ0)

# goals: core at the pull, retained water, bark lightness at the end, core at serve
const GOALS = [273.15 + 93.0, 0.35, 32.0, 273.15 + 65.0]
const GOAL_SD = [2.0, 0.05, 4.0, 3.0]

function plan_decode(ζ)
    u = logistic.(ζ[1:3])
    f_pull = logistic(ζ[5])
    return (u=u, f_wrap=logistic(ζ[4]) * f_pull, f_pull=f_pull, t_f=3600.0 * exp(ζ[6]),
            f_spritz=logistic(ζ[7]) * f_pull, f_dts=exp(ζ[8]), m_spritz=exp(ζ[9]), f_add=logistic(ζ[10]) * f_pull)
end
function plan_phys(ζ)
    p = plan_decode(ζ)
    return (u=p.u, wrap_h=p.f_wrap * p.t_f / 3600, pull_h=p.f_pull * p.t_f / 3600, t_f_h=p.t_f / 3600,
            spritz_h=p.f_spritz * p.t_f / 3600, dt_spritz_h=p.f_dts * p.t_f / 3600, dose=p.m_spritz,
            add_h=p.f_add * p.t_f / 3600)
end

function interp_s(y, sq)
    k = clamp(searchsortedlast(S_SAVE, ForwardDiff.value(sq)), 1, length(S_SAVE) - 1)
    f = (sq - S_SAVE[k]) / (S_SAVE[k + 1] - S_SAVE[k])
    return (1 - f) * y[k] + f * y[k + 1]
end

"The plan's outputs [core at the pull, retained water, bark L* at the end, core at serve] at latents z and plan ζ."
function plan_outputs(z, ζ)
    x = from_z(z[1:15], prior[1:15])
    A = exp(z[AREA])
    p = plan_decode(ζ)
    o = monolith_solve_tf(x[1:8], x[9:15], p.u, p.t_f, S_U, S_SAVE; cfg=d.cfg, pit=d.pit,
                          frac_wrap=p.f_wrap, frac_pull=p.f_pull, frac_spritz=p.f_spritz,
                          frac_dt_spritz=p.f_dts, m_spritz=p.m_spritz, frac_add=p.f_add,
                          A_meat=A, abstol=d.abstol, reltol=d.reltol)
    return [interp_s(o.Tcore, p.f_pull), o.w_retained, o.l_star[end], o.Tcore[end]]
end

function residuals(ζ, zs)
    r = eltype(ζ)[]
    for z in zs
        y = plan_outputs(z, ζ)
        append!(r, (y .- GOALS) ./ GOAL_SD)
    end
    return vcat(r, sqrt(Λ0) * (ζ .- ζ0))
end
expected_loss(ζ, zs) = try 0.5 * sum(abs2, residuals(ζ, zs)) catch err; err isa InterruptException ? rethrow() : Inf end

# --- the reference: damped Gauss-Newton -------------------------------------------------------
function solve_reference(zs; iters=25, verbose=true)
    ζ = copy(ζ0); F = expected_loss(ζ, zs); λ = 0.0
    for k in 1:iters
        r = residuals(ζ, zs)
        Jr = ForwardDiff.jacobian(v -> residuals(v, zs), ζ)
        accepted = false; ζn = ζ; Fn = F; α = 1.0; step = zeros(NP)
        for attempt in 1:8
            step = -((Jr' * Jr + λ * I) \ (Jr' * r))
            α = 1.0; ζn = ζ + step; Fn = expected_loss(ζn, zs)
            while (!isfinite(Fn) || Fn > F) && α > 1e-3
                α /= 2; ζn = ζ + α * step; Fn = expected_loss(ζn, zs)
            end
            if isfinite(Fn) && Fn <= F
                accepted = true; α == 1.0 && (λ = λ / 3 < 0.05 ? 0.0 : λ / 3); break
            end
            λ = λ == 0 ? 0.5 : 10λ
        end
        verbose && @printf("    R it %2d  F %.4f -> %.4f  |step| %.3g  alpha %.3g  damping %.2g\n", k, F, Fn, norm(step), α, λ)
        flush(stdout)
        accepted || break
        done = α * norm(step) < 1e-3 || F - Fn < 1e-3
        ζ, F = ζn, Fn
        done && break
    end
    Jr = ForwardDiff.jacobian(v -> residuals(v, zs), ζ)
    return (ζ=ζ, F=F, sd=sqrt.(diag(inv(Jr' * Jr))))
end

# --- the graph: per-draw affine nodes inside the graph iterator ----------------------------
const AFFS = Ref{Vector{Tuple{Vector{Float64},Matrix{Float64}}}}(Tuple{Vector{Float64},Matrix{Float64}}[])
const CENTER = Ref{Vector{Float64}}(zeros(NP))
affine_node(ζ, i) = AFFS[][i][1] .+ AFFS[][i][2] * (ζ .- CENTER[])

@model function plan_block_graph(g, lm_obs, Λ_lm, mζ, Λζ, N, Σg)
    ζ ~ MvNormalMeanPrecision(mζ, Λζ)
    for i in 1:N
        y[i] := affine_node(ζ, i)
        g[i] ~ MvNormalMeanCovariance(y[i], Σg)
    end
    lm := lm_node(ζ)
    lm_obs ~ MvNormalMeanPrecision(lm, Λ_lm)
end
const META_PB = @meta begin
    affine_node() -> Linearization()
    lm_node() -> Linearization()
end
@initialization function init_ζ(mζ, Λζ)
    μ(ζ) = MvNormalMeanPrecision(mζ, Λζ)
end

function solve_graph(zs; verbose=true)
    g = [GOALS for _ in 1:length(zs)]
    Σg = Matrix(Diagonal(GOAL_SD .^ 2))
    objective(c) = 0.5 * sum(abs2, residuals(c, zs))
    function rebuild!(c)
        AFFS[] = [(plan_outputs(z, c), ForwardDiff.jacobian(v -> plan_outputs(z, v), c)) for z in zs]
        CENTER[] = copy(c)
    end
    prob = IteratedProblem(:ζ, objective, rebuild!, NP)
    run_infer(Λ_lm, callbacks, iters) = infer(model = plan_block_graph(mζ = ζ0, Λζ = Matrix(Λ0), N = length(zs), Σg = Σg),
                                              data = (g = g, lm_obs = zeros(NP), Λ_lm = Λ_lm), meta = META_PB,
                                              initialization = init_ζ(CENTER[], Matrix(Λ0)), iterations = iters,
                                              callbacks = callbacks, returnvars = (ζ = KeepLast(),))
    return iterated_solve(prob, run_infer; c0=ζ0, verbose=verbose)
end

function outcomes(ζ, zs)
    ys = [plan_outputs(z, ζ) for z in zs]
    M = reduce(hcat, ys)
    return (mean=vec(mean(M; dims=2)), sd=vec(std(M; dims=2)))
end


MODE = length(ARGS) >= 1 ? ARGS[1] : "summary"
rec_ref = joinpath(REPO, "data", "mock", "plan_block_reference.json")
rec_graph = joinpath(REPO, "data", "mock", "plan_block_graph.json")

if MODE == "reference"
    println("=== the full plan block, reference")
    t0 = time(); r = solve_reference(zs); tR = time() - t0
    o = outcomes(r.ζ, zs)
    open(rec_ref, "w") do io
        JSON.print(io, Dict("zeta" => r.ζ, "F" => r.F, "sd" => r.sd, "plan" => Dict(pairs(plan_phys(r.ζ))),
                            "outcomes" => Dict(pairs(o)), "wall_s" => tR))
    end
    @printf("reference F %.4f (%.0f s); wrote %s\n", r.F, tR, rec_ref)
elseif MODE == "graph"
    println("=== the full plan block, on the graph")
    t0 = time(); r = solve_graph(zs); tG = time() - t0
    o = outcomes(r.z, zs)
    open(rec_graph, "w") do io
        JSON.print(io, Dict("zeta" => r.z, "F" => r.F, "iterations" => length(r.hist), "converged" => r.converged,
                            "plan" => Dict(pairs(plan_phys(r.z))), "outcomes" => Dict(pairs(o)), "wall_s" => tG))
    end
    @printf("graph F %.4f (%d iterations, %.0f s); wrote %s\n", r.F, length(r.hist), tG, rec_graph)
else
    rR = JSON.parsefile(rec_ref); rG = JSON.parsefile(rec_graph)
    ζR, ζG = Float64.(rR["zeta"]), Float64.(rG["zeta"])
    sdR = Float64.(rR["sd"])
    oR, oG, o0 = outcomes(ζR, zs), outcomes(ζG, zs), outcomes(ζ0, zs)
    dz = (ζG .- ζR) ./ sdR
    @printf("\nreference F %.4f (%.0f s), graph F %.4f (%d iterations, %.0f s); graph - reference in the plan's sd: max %.3f\n",
            rR["F"], rR["wall_s"], rG["F"], rG["iterations"], rG["wall_s"], maximum(abs.(dz)))
    println("\ndecision                      prior      reference   graph")
    PHYS_NAMES = ["damper", "wrap [h]", "pull [h]", "duration [h]", "spritz start [h]", "spritz interval [h]", "spritz dose [kg]", "fuel add [h]"]
    for (name, p0, pr, pg) in zip(PHYS_NAMES, collect(values(plan_phys(ζ0))), collect(values(plan_phys(ζR))), collect(values(plan_phys(ζG))))
        if p0 isa AbstractVector
            for k in eachindex(p0)
                @printf("%-28s %9.3f %11.3f %9.3f\n", "$name knot $k", p0[k], pr[k], pg[k])
            end
        else
            @printf("%-28s %9.3f %11.3f %9.3f\n", name, p0, pr, pg)
        end
    end
    println("\noutcome (goal ± sd)            prior plan        reference         graph")
    for (k, nm) in enumerate(("core at pull [C]", "retained water", "bark L*", "core at serve [C]"))
        off = k in (1, 4) ? 273.15 : 0.0
        @printf("%-30s %7.2f ± %5.2f   %7.2f ± %5.2f   %7.2f ± %5.2f   (goal %.2f ± %.2f)\n", nm,
                o0.mean[k] - off, o0.sd[k], oR.mean[k] - off, oR.sd[k], oG.mean[k] - off, oG.sd[k], GOALS[k] - off, GOAL_SD[k])
    end
    flush(stdout)
    @testset "the full plan block on the graph reproduces the reference" begin
        @test rG["converged"]
        @test abs(rG["F"] - rR["F"]) < 5e-3 * max(1.0, rR["F"])
        @test maximum(abs.(dz)) < 0.2      # two optimizers on a flat valley: 0.07 sd apart at 0.009 in the objective, the graph's the lower
        pg = plan_phys(ζG)
        @test 0 < pg.wrap_h < pg.pull_h < pg.t_f_h            # the ordering the parameterization promises
        @test 0 < pg.spritz_h < pg.pull_h && 0 < pg.add_h < pg.pull_h
    end
    open(joinpath(REPO, "data", "mock", "plan_block.json"), "w") do io
        JSON.print(io, Dict("names" => PLAN_NAMES, "zeta0" => ζ0, "goals" => GOALS, "goal_sd" => GOAL_SD, "N" => N,
                            "reference" => rR, "graph" => rG, "prior_outcomes" => Dict(pairs(o0)),
                            "difference_sd" => dz))
    end
    println("wrote data/mock/plan_block.json")
end
