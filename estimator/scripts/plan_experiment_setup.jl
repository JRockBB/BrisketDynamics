# Planning as inference in RxInfer, the bounded experiment: does a factor
# graph plan a cook under the estimator's parameter uncertainty, does it
# reproduce the expected-loss optimum, and does the plan respond to the
# posterior width (the certainty-equivalence test)?
#
# The plan ζ: two damper knots (before and after 6 h, logit) and the pull
# time (log hours). The physics: the monolith from the start of the
# matched mock cook, solved per parameter draw. The goals: the core
# temperature at the pull near 93 °C, the retained water near 0.35, each a
# Gaussian goal; the prior on ζ prefers a low damper and a 12 h pull.
# The parameter posterior: the estimator's at 6 h of the matched mock
# (pit + core + humidity + scale), its full covariance recomputed.
#
# Three solvers of the same problem:
#   B  the graph with the posterior held fixed as N draws inside per-draw
#      Unscented delta nodes (2 outputs, 3 inputs each): message passing
#      minimizes the expected loss over the draws
#   R  the same expected loss minimized by damped Gauss-Newton with
#      ForwardDiff through the physics (the reference optimum)
#   A  the naive graph: the physics latents in one block with the
#      controls, one Unscented node — the goal updates the belief about
#      the world (the known flaw), measured
# and the covariance test: B and R again with the posterior's width
# shrunk tenfold (the same standard normals, scaled).
#
# plan_experiment_setup.jl: the shared setup (the posterior, the draws, the plan physics,
# the three solvers), included by plan_experiment.jl and plan_experiment_outer.jl.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using RxInfer, ForwardDiff, JSON, LinearAlgebra, Random, Printf, Statistics
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json"))
S = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d_stream.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
t = Float64.(m["t"]); u_log = Float64.(m["u"]); meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = read_sites(gate["prior"])
d = estimator_setup(t, u_log, pit, cfg, prior_phys, String.(gate["names"]))
prior = vcat(prior_phys, [PriorSite("lognormal", log(cfg.A_meat), 0.3), probe_depth_site()])
AREA, DEPTH = 16, 17

# --- the posterior at 6 h, full covariance ------------------------------------
st = first(s for s in S["sets"] if s["label"] == "pit + core + humidity + scale")
b6 = first(b for b in st["batches"] if b["t_h"] == 6)
z6 = Float64.(b6["z"])
stored_names = String.(S["names"])
if stored_names[end] == "core_depth"
    # This historical fixture stores log physical depth. Translate its MAP
    # into the bounded fraction coordinate; covariance is rebuilt below.
    fraction = clamp(exp(z6[DEPTH] - z6[4]), eps(Float64), 1 - eps(Float64))
    z6[DEPTH] = to_z(fraction, prior[DEPTH])
elseif stored_names[end] != "core_depth_fraction"
    throw(ArgumentError("unknown stored probe-depth coordinate $(stored_names[end])"))
end
rd = m["readings"]; nz_ = m["noise"]
keep = t .<= 6 * 3600 + 1e-9
chs = [PitProbe(nz_["pit"]["sigma"], nz_["pit"]["rho"], keep; data=Float64.(rd["pit"])),
       MeatProbe(nz_["core"]["sigma"], nz_["core"]["rho"], keep; depth_latent=DEPTH, data=Float64.(rd["core"])),
       HumiditySensor(nz_["humidity"]["sigma"], nz_["humidity"]["rho"], keep; data=Float64.(rd["humidity"])),
       MassScale(nz_["mass"]["sigma"], nz_["mass"]["rho"], keep; data=Float64.(rd["mass"]))]
Sx, J, lay = physics_outputs(z6, d; area_latent=AREA)
a, B, w = assemble(chs, z6, Sx, J, prior, lay)
_, V6 = graph_posterior_affine(z6, a, B, w, prior)
sd6 = sqrt.(diag(V6))
@printf("posterior at 6 h: sd of L %.3f, h_c %.3f, A_meat %.3f, a0 %.3f\n", sd6[4], sd6[3], sd6[AREA], sd6[15])
flush(stdout)

# --- the plan physics -------------------------------------------------------------
const T_TARGET, W_TARGET = 273.15 + 93.0, 0.35
const σ_T, σ_W = 2.0, 0.05
ζ0 = [log(0.15 / 0.85), log(0.15 / 0.85), log(12.0)]
Λ0 = Diagonal([1 / 0.5^2, 1 / 0.5^2, 1 / 0.3^2])
const T_SPLIT = 6 * 3600.0

function interp_at(y, tt, tq)
    k = clamp(searchsortedlast(tt, ForwardDiff.value(tq)), 1, length(tt) - 1)
    f = (tq - tt[k]) / (tt[k + 1] - tt[k])
    return (1 - f) * y[k] + f * y[k + 1]
end

"The plan's outputs [core temperature at the pull, retained water] at latents z and plan ζ."
function plan_outputs(z, ζ)
    x = from_z(z[1:15], prior[1:15])
    A = exp(z[AREA])
    u1, u2 = logistic(ζ[1]), logistic(ζ[2])
    tp = 3600.0 * exp(ζ[3])
    Tel = promote_type(eltype(z), eltype(ζ))
    ukn = Tel[tk < T_SPLIT ? u1 : u2 for tk in t]
    o = monolith_solve(x[1:8], x[9:15], ukn, t, t; pit=d.pit, cfg=d.cfg, abstol=d.abstol, reltol=d.reltol,
                       A_meat=A, t_pull=tp)
    return [interp_at(o.Tcore, t, tp), o.w_retained]
end
plan_phys(ζ) = (logistic(ζ[1]), logistic(ζ[2]), exp(ζ[3]))

# --- the draws ---------------------------------------------------------------------
N = 8
rng = MersenneTwister(3)
ξ = [randn(rng, length(z6)) for _ in 1:N]
Lc = cholesky(Symmetric(V6)).L
draws(scale) = [z6 .+ scale .* (Lc * ξi) for ξi in ξ]

# --- R: damped Gauss-Newton on the expected loss ------------------------------------
function residuals(ζ, zs)
    r = eltype(ζ)[]
    for z in zs
        y = plan_outputs(z, ζ)
        push!(r, (y[1] - T_TARGET) / σ_T)
        push!(r, (y[2] - W_TARGET) / σ_W)
    end
    return vcat(r, sqrt(Λ0) * (ζ .- ζ0))
end
function solve_reference(zs; iters=15, verbose=true)
    ζ = copy(ζ0)
    r = residuals(ζ, zs); F = 0.5 * sum(abs2, r)
    λ = 0.0
    for k in 1:iters
        Jr = ForwardDiff.jacobian(v -> residuals(v, zs), ζ)
        step = nothing; Fn = Inf; ζn = ζ
        for attempt in 1:8
            step = -((Jr' * Jr + λ * I) \ (Jr' * r))
            α = 1.0
            ζn = ζ + step
            Fn = try 0.5 * sum(abs2, residuals(ζn, zs)) catch; Inf end
            while (!isfinite(Fn) || Fn > F) && α > 1e-3
                α /= 2; ζn = ζ + α * step
                Fn = try 0.5 * sum(abs2, residuals(ζn, zs)) catch; Inf end
            end
            (isfinite(Fn) && Fn <= F) && break
            λ = λ == 0 ? 0.1 : 10λ
        end
        (isfinite(Fn) && Fn <= F) || break
        verbose && @printf("    R it %2d  F %.4f -> %.4f  |step| %.3g  plan (u1 %.3f, u2 %.3f, pull %.2f h)\n",
                           k, F, Fn, norm(ζn - ζ), plan_phys(ζn)...)
        flush(stdout)
        done = norm(ζn - ζ) < 1e-3 || F - Fn < 1e-4
        ζ, F = ζn, Fn
        r = residuals(ζ, zs)
        done && break
    end
    Jr = ForwardDiff.jacobian(v -> residuals(v, zs), ζ)
    Vζ = inv(Jr' * Jr)
    return (ζ=ζ, F=F, sd=sqrt.(diag(Vζ)))
end

# --- B: the graph with per-draw Unscented nodes -----------------------------------------
const ZS = Ref{Vector{Vector{Float64}}}(Vector{Float64}[])
plan_node(ζ, i) = plan_outputs(ZS[][i], ζ)

@model function plan_graph_B(g, mζ, Λζ, N, Σg)
    ζ ~ MvNormalMeanPrecision(mζ, Λζ)
    for i in 1:N
        y[i] := plan_node(ζ, i)
        g[i] ~ MvNormalMeanCovariance(y[i], Σg)
    end
end
const META_B = @meta begin
    plan_node() -> Unscented()
end
@initialization function init_ζ(mζ, Λζ)
    μ(ζ) = MvNormalMeanPrecision(mζ, Λζ)
end

function solve_graph_B(zs; iters=12, verbose=true)
    ZS[] = zs
    g = [[T_TARGET, W_TARGET] for _ in 1:length(zs)]
    Σg = Diagonal([σ_T^2, σ_W^2]) |> Matrix
    t0 = time()
    res = infer(model = plan_graph_B(mζ = ζ0, Λζ = Matrix(Λ0), N = length(zs), Σg = Σg), data = (g = g,),
                meta = META_B, initialization = init_ζ(ζ0, Matrix(Λ0)), iterations = iters,
                returnvars = (ζ = KeepEach(),))
    hist = [collect(mean(q)) for q in res.posteriors[:ζ]]
    q = res.posteriors[:ζ][end]
    ζ = collect(mean(q)); Vζ = Matrix(cov(q))
    F = 0.5 * sum(abs2, residuals(ζ, zs))
    verbose && for (k, h) in enumerate(hist)
        @printf("    B it %2d  plan (u1 %.3f, u2 %.3f, pull %.2f h)\n", k, plan_phys(h)...)
    end
    verbose && @printf("    B: expected-loss objective at the graph's mean %.4f, %.0f s\n", F, time() - t0)
    flush(stdout)
    return (ζ=ζ, F=F, sd=sqrt.(diag(Vζ)), hist=hist)
end

# --- A: the naive graph, z in the block --------------------------------------------------
naive_node(v) = plan_outputs(v[1:17], v[18:20])
@model function plan_graph_A(g, mv, Λv, Σg)
    v ~ MvNormalMeanPrecision(mv, Λv)
    y := naive_node(v)
    g ~ MvNormalMeanCovariance(y, Σg)
end
const META_A = @meta begin
    naive_node() -> Unscented()
end
@initialization function init_v(mv, Λv)
    μ(v) = MvNormalMeanPrecision(mv, Λv)
end
function solve_graph_A(V; iters=6)
    mv = vcat(z6, ζ0)
    Λv = zeros(20, 20); Λv[1:17, 1:17] = inv(Symmetric(V)); Λv[18:20, 18:20] = Matrix(Λ0)
    Σg = Diagonal([σ_T^2, σ_W^2]) |> Matrix
    t0 = time()
    res = infer(model = plan_graph_A(mv = mv, Λv = Λv, Σg = Σg), data = (g = [T_TARGET, W_TARGET],),
                meta = META_A, initialization = init_v(mv, Λv), iterations = iters, returnvars = (v = KeepLast(),))
    q = res.posteriors[:v]
    mq = collect(mean(q)); Vq = Matrix(cov(q))
    dz_sd = (mq[1:17] .- z6) ./ sqrt.(diag(V))
    @printf("    A: plan (u1 %.3f, u2 %.3f, pull %.2f h); the goal moved z by max %.2f sd (%s), %.0f s\n",
            plan_phys(mq[18:20])..., maximum(abs.(dz_sd)), d.names[min(argmax(abs.(dz_sd)), 15)], time() - t0)
    flush(stdout)
    return (ζ=mq[18:20], dz_sd=dz_sd, sd=sqrt.(diag(Vq))[18:20])
end

# --- outcomes under the posterior: the spread of the goals at a plan ------------------------
function outcomes(ζ, zs)
    ys = [plan_outputs(z, ζ) for z in zs]
    Tc = [y[1] for y in ys]; W = [y[2] for y in ys]
    return (T_mean=mean(Tc) - 273.15, T_sd=std(Tc), W_mean=mean(W), W_sd=std(W))
end

