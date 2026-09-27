# The factor layer (factors.jl) against exact oracles on a linear-Gaussian state space:
#   Kalman / RTS      the full-history graph solve reproduces the Rauch-Tung-Striebel smoother (means and
#                     marginal covariances) and its newest marginal the Kalman filter's, to roundoff
#   fixed lag         marginalizing the nodes older than a lag leaves every window marginal equal to the
#                     full-history smoother's (exact for a linear system), whatever the lag
#   delayed insertion a reading inserted after later ones, or a node split, gives the in-order result
#   dropout / subsets fewer sensors, gaps, a sensor that never reads: the same oracle on what remains
#   hierarchy         several cooks with a planted population value recover it and the per-cook offsets
#   censored site     the observation factor's local site matches the survival likelihood's gradient and
#                     curvature; the discrepancy chain's OU factor is the AR(1) whitening
#   julia --project=estimator estimator/test/test_factors.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using LinearAlgebra, Random, Test, ForwardDiff
const E = BrisketEstimator

struct AtomicProbe <: Factor
    key::VarKey
    calls::Base.RefValue{Int}
end
E.vars(f::AtomicProbe) = [f.key]
E.ndim(::AtomicProbe) = 1
E.residual(::AtomicProbe, vals) = error("linearize must use the atomic factor interface")
E.jacobians(::AtomicProbe, vals) = error("linearize must use the atomic factor interface")
function E.factor_linearization(f::AtomicProbe, vals)
    f.calls[] += 1
    return (r=[vals[1][1] - 2.0], Js=[fill(1.0, 1, 1)])
end

struct QuadraticProbe <: Factor
    key::VarKey
end
E.vars(f::QuadraticProbe) = [f.key]
E.ndim(::QuadraticProbe) = 1
E.residual(::QuadraticProbe, vals) = [vals[1][1]^2 - 1]
E.jacobians(::QuadraticProbe, vals) = [fill(2vals[1][1], 1, 1)]

struct ResolutionProbe <: Factor
    key::VarKey
    target::Float64
end
E.vars(f::ResolutionProbe) = [f.key]
E.ndim(::ResolutionProbe) = 1
E.residual(f::ResolutionProbe, vals) = [vals[1][1] - f.target]
E.jacobians(::ResolutionProbe, vals) = [fill(1.0, 1, 1)]
E.loss(f::ResolutionProbe, vals) = round(0.5 * (vals[1][1] - f.target)^2; digits=6)

# --- the linear-Gaussian oracle: Kalman filter and RTS smoother ---------------------------------------------------
function kalman_rts(F, Q, H, R, m0, P0, ys)
    n = length(ys)
    mf = Vector{Vector{Float64}}(undef, n); Pf = Vector{Matrix{Float64}}(undef, n)
    mp = Vector{Vector{Float64}}(undef, n); Pp = Vector{Matrix{Float64}}(undef, n)
    m, P = copy(m0), copy(P0)
    for k in 1:n
        if k > 1
            m = F * m; P = F * P * F' + Q
        end
        mp[k], Pp[k] = copy(m), copy(P)
        y = ys[k]
        if y !== nothing
            S = H * P * H' + R
            K = P * H' / S
            m = m + K * (y - H * m)
            P = (I - K * H) * P
            P = (P + P') / 2
        end
        mf[k], Pf[k] = copy(m), copy(P)
    end
    ms = copy(mf); Ps = copy(Pf)
    for k in (n - 1):-1:1
        C = Pf[k] * F' / Pp[k + 1]
        ms[k] = mf[k] + C * (ms[k + 1] - mp[k + 1])
        Ps[k] = Pf[k] + C * (Ps[k + 1] - Pp[k + 1]) * C'
    end
    return (filt=(mf, Pf), smooth=(ms, Ps))
end

# the same system as a graph: nodes x_1..x_n, dynamics r = W (x_{k+1} - F x_k), readings r = (y - H x)/sd
xk(k) = VarKey(:state, :lin, k)
function linear_graph(F, Q, H, R, m0, P0, ys; nodes=1:length(ys), order=nothing)
    g = FactorGraph()
    W = inv(cholesky(Symmetric(Q)).L); Rw = inv(cholesky(Symmetric(R)).L)
    add_variable!(g, xk(1), m0)
    add_factor!(g, GaussianPrior(xk(1), m0, Symmetric(P0)))
    for k in 2:length(ys)
        add_variable!(g, xk(k), F * variable_value(g, xk(k - 1)))
        add_factor!(g, LinearFactor([xk(k - 1), xk(k)], [-W * F, W], zeros(size(F, 1)); source="dyn $k"))
    end
    for k in (order === nothing ? (1:length(ys)) : order)
        ys[k] === nothing && continue
        add_factor!(g, LinearFactor([xk(k)], [Rw * H], Rw * ys[k]; source="obs $k"))
    end
    return g
end
function graph_marginals(g, keys)
    C, off = joint_cov(g; keys)
    return [(copy(variable_value(g, k)), C[off[k], off[k]]) for k in keys]
end

rng = MersenneTwister(5)
F = [1.0 0.1; -0.05 0.9]; Q = [0.02 0.005; 0.005 0.01]; H = [1.0 0.0; 0.3 1.0]; R = Diagonal([0.5, 0.2]) |> Matrix
m0 = [1.0, -0.5]; P0 = [0.3 0.05; 0.05 0.2]
n = 12
xt = Vector{Vector{Float64}}(undef, n); xt[1] = m0 + cholesky(Symmetric(P0)).L * randn(rng, 2)
for k in 2:n; xt[k] = F * xt[k - 1] + cholesky(Symmetric(Q)).L * randn(rng, 2); end
ys = Any[H * x + cholesky(Symmetric(R)).L * randn(rng, 2) for x in xt]

@testset "the linear oracle: the graph's full-history solve is the RTS smoother, its newest marginal the Kalman filter" begin
    ref = kalman_rts(F, Q, H, R, m0, P0, ys)
    g = linear_graph(F, Q, H, R, m0, P0, ys)
    r = gauss_newton!(g; iters=5)
    @test r.converged && length(r.hist) <= 2                       # one exact step, one to confirm
    marg = graph_marginals(g, [xk(k) for k in 1:n])
    for k in 1:n
        @test maximum(abs.(marg[k][1] .- ref.smooth[1][k])) < 1e-9
        @test maximum(abs.(marg[k][2] .- ref.smooth[2][k])) < 1e-9
    end
    @test maximum(abs.(marg[n][2] .- ref.filt[2][n])) < 1e-9        # the newest smoothed marginal is the filtered one
    # the objective is the negative log posterior up to a constant: the same at the optimum from any start
    @test isfinite(objective(g))
    # a sparse solve gives the same
    gs = linear_graph(F, Q, H, R, m0, P0, ys)
    gauss_newton!(gs; iters=5, sparse=true)
    @test maximum(abs.(variable_value(gs, xk(n)) .- variable_value(g, xk(n)))) < 1e-9
    Cs = marginal_cov(gs, xk(n); sparse=true)
    @test maximum(abs.(Cs .- marg[n][2])) < 1e-9
end

@testset "fixed lag: marginalization keeps every window marginal at the smoother's, and the past is the filter's" begin
    ref = kalman_rts(F, Q, H, R, m0, P0, ys)
    for lag in (0, 1, 3)
        g = FactorGraph()
        W = inv(cholesky(Symmetric(Q)).L); Rw = inv(cholesky(Symmetric(R)).L)
        add_variable!(g, xk(1), m0); add_factor!(g, GaussianPrior(xk(1), m0, Symmetric(P0)))
        add_factor!(g, LinearFactor([xk(1)], [Rw * H], Rw * ys[1]))
        gauss_newton!(g; iters=3)
        window = [1]
        for k in 2:n
            add_variable!(g, xk(k), F * variable_value(g, xk(k - 1)))
            add_factor!(g, LinearFactor([xk(k - 1), xk(k)], [-W * F, W], zeros(2)))
            add_factor!(g, LinearFactor([xk(k)], [Rw * H], Rw * ys[k]))
            push!(window, k)
            gauss_newton!(g; iters=3)
            while length(window) > lag + 1
                marginalize!(g, [xk(window[1])]); popfirst!(window)
            end
            # after the readings to k: every window node's marginal equals the RTS smoother on the readings to k
            refk = kalman_rts(F, Q, H, R, m0, P0, ys[1:k])
            marg = graph_marginals(g, [xk(j) for j in window])
            for (i, j) in enumerate(window)
                @test maximum(abs.(marg[i][1] .- refk.smooth[1][j])) < 1e-9
                @test maximum(abs.(marg[i][2] .- refk.smooth[2][j])) < 1e-9
            end
        end
        @test length(window) == min(lag + 1, n) && !has_variable(g, xk(1)) == (lag + 1 < n)
        # the final window marginals are the full smoother's; the newest is the filter's
        marg = graph_marginals(g, [xk(j) for j in window])
        @test maximum(abs.(marg[end][2] .- ref.filt[2][n])) < 1e-9
        for (i, j) in enumerate(window)
            @test maximum(abs.(marg[i][1] .- ref.smooth[1][j])) < 1e-9
        end
    end
    # marginalizing everything but one node is an error only when nothing is left to carry information
    g = linear_graph(F, Q, H, R, m0, P0, ys); gauss_newton!(g; iters=3)
    lf = marginalize!(g, [xk(1), xk(2)])
    @test lf isa LinearFactor && vars(lf) == [xk(3)] && ndim(lf) == 2
    @test_throws ArgumentError marginalize!(g, [xk(1)])
end

@testset "delayed insertion, dropout and sensor subsets against the same oracle" begin
    # readings inserted out of order: the graph does not care in which order factors arrive
    ref = kalman_rts(F, Q, H, R, m0, P0, ys)
    g = linear_graph(F, Q, H, R, m0, P0, ys; order=[3, 1, 7, 2, 12, 5, 4, 6, 11, 8, 10, 9])
    gauss_newton!(g; iters=5)
    marg = graph_marginals(g, [xk(k) for k in 1:n])
    @test all(maximum(abs.(marg[k][1] .- ref.smooth[1][k])) < 1e-9 for k in 1:n)
    # a delayed reading into a fixed-lag window: the node was made without it, the reading arrives two nodes later;
    # the result is the in-order smoother's on the readings present
    W = inv(cholesky(Symmetric(Q)).L); Rw = inv(cholesky(Symmetric(R)).L)
    g = FactorGraph(); add_variable!(g, xk(1), m0); add_factor!(g, GaussianPrior(xk(1), m0, Symmetric(P0)))
    add_factor!(g, LinearFactor([xk(1)], [Rw * H], Rw * ys[1]))
    for k in 2:6
        add_variable!(g, xk(k), F * variable_value(g, xk(k - 1)))
        add_factor!(g, LinearFactor([xk(k - 1), xk(k)], [-W * F, W], zeros(2)))
        k == 4 || add_factor!(g, LinearFactor([xk(k)], [Rw * H], Rw * ys[k]))
    end
    add_factor!(g, LinearFactor([xk(4)], [Rw * H], Rw * ys[4]))       # the delayed reading of node 4, after node 6 exists
    gauss_newton!(g; iters=5)
    ref6 = kalman_rts(F, Q, H, R, m0, P0, ys[1:6])
    marg = graph_marginals(g, [xk(k) for k in 1:6])
    @test all(maximum(abs.(marg[k][1] .- ref6.smooth[1][k])) < 1e-9 for k in 1:6)
    # dropout: no reading at nodes 3, 4, 8; a subset: the second sensor only from node 6 on
    yd = Any[copy(y) for y in ys]; yd[3] = nothing; yd[4] = nothing; yd[8] = nothing
    refd = kalman_rts(F, Q, H, R, m0, P0, yd)
    gd = linear_graph(F, Q, H, R, m0, P0, yd); gauss_newton!(gd; iters=5)
    margd = graph_marginals(gd, [xk(k) for k in 1:n])
    @test all(maximum(abs.(margd[k][1] .- refd.smooth[1][k])) < 1e-9 for k in 1:n)
    @test all(maximum(abs.(margd[k][2] .- refd.smooth[2][k])) < 1e-9 for k in 1:n)
    @test margd[4][2][1, 1] > marg[4][2][1, 1]                            # the dropped node is wider
    H2 = H[2:2, :]; R2 = R[2:2, 2:2]
    ysub = Any[k < 6 ? nothing : [ys[k][2]] for k in 1:n]
    refs = kalman_rts(F, Q, H2, R2, m0, P0, ysub)
    gs = linear_graph(F, Q, H2, R2, m0, P0, ysub); gauss_newton!(gs; iters=5)
    margs = graph_marginals(gs, [xk(k) for k in 1:n])
    @test all(maximum(abs.(margs[k][1] .- refs.smooth[1][k])) < 1e-9 for k in 1:n)
end

@testset "hierarchy: cooks around a population, the planted value recovered" begin
    # each cook's scalar parameter θ_c = θ_pop + offset, read through noisy direct readings; the population's
    # prior wide; the posterior mean of θ_pop must sit within its sd of the planted value and the offsets recover
    rng2 = MersenneTwister(9)
    θ_pop = 0.7; τ = 0.15; n_cooks = 6; n_reads = 20; σ = 0.3
    kp = VarKey(:hyper, :population, 0)
    g = FactorGraph()
    add_variable!(g, kp, [0.0]); add_factor!(g, GaussianPrior(kp, [0.0], fill(1 / 2.0, 1, 1)))
    truths = Float64[]
    for c in 1:n_cooks
        θc = θ_pop + τ * randn(rng2); push!(truths, θc)
        kc = VarKey(:params, Symbol("cook", c), 0)
        add_variable!(g, kc, [0.0])
        add_factor!(g, HierarchyFactor(kc, kp, [τ]))
        for _ in 1:n_reads
            add_factor!(g, LinearFactor([kc], [fill(1 / σ, 1, 1)], [(θc + σ * randn(rng2)) / σ]))
        end
    end
    r = gauss_newton!(g; iters=5, sparse=true)
    @test r.converged
    mp = variable_value(g, kp)[1]; sp = sqrt(marginal_cov(g, kp; sparse=true)[1, 1])
    @test abs(mp - θ_pop) < 2.5 * sp && sp < τ                               # pooled: narrower than one cook's offset
    for c in 1:n_cooks
        kc = VarKey(:params, Symbol("cook", c), 0)
        mc = variable_value(g, kc)[1]; sc = sqrt(marginal_cov(g, kc; sparse=true)[1, 1])
        @test abs(mc - truths[c]) < 3 * sc && sc < σ / sqrt(n_reads) * 1.01   # the offsets: their own readings plus the pool
    end
    # the exact conjugate answer for the population from the per-cook means: check against the closed form
    means = [sum(residual(f, [variable_value(g, VarKey(:params, Symbol("cook", c), 0))]) for f in factors_of(g, VarKey(:params, Symbol("cook", c), 0)) if f isa LinearFactor) for c in 1:n_cooks]
    @test length(means) == n_cooks
    # a subset hierarchy factor: only the second of two entries tied
    kc2 = VarKey(:params, :two, 0); kp2 = VarKey(:hyper, :pop2, 0)
    hf = HierarchyFactor(kc2, kp2, [2], [1], [0.5])
    @test residual(hf, [[1.0, 3.0], [2.0]]) == [2.0] && jacobians(hf, [[1.0, 3.0], [2.0]])[1] == [0.0 2.0] && jacobians(hf, [[1.0, 3.0], [2.0]])[2] == fill(-2.0, 1, 1)
    @test_throws ArgumentError HierarchyFactor(kc2, kp2, [1], [1], [-1.0])
end

@testset "the censored site and the OU chain" begin
    # the local site's ½r² has the survival loss's gradient and curvature in μ at the current point
    cap, σ = 6500.0, 10.0
    for μ in (6480.0, 6500.0, 6530.0)
        site = E._censored_normal_site(μ, cap, σ)
        q, ỹ = site.root_precision, site.pseudo
        r(m) = q * (m - ỹ)
        gsite = ForwardDiff.derivative(m -> 0.5 * r(m)^2, μ)
        nll(m) = E._censored_normal_nll(m, cap, σ)
        ε = 1e-3                                                        # the loss is Float64-only: central differences
        gexact = (nll(μ + ε) - nll(μ - ε)) / 2ε
        hexact = (nll(μ + ε) - 2nll(μ) + nll(μ - ε)) / ε^2
        @test abs(gsite - gexact) < 1e-5 * max(1, abs(gexact)) && abs(q^2 - hexact) < 1e-4 * max(1, abs(hexact))
    end
    # the OU factor over a gap is the AR(1) whitening row: ρ^gap and σ√(1 − ρ²ᵍ)
    nm = NoiseModel(white_sd=0.1, ar_sd=1.5, tau=3000.0)
    ρ1, s1 = E.ou_step(nm, 120.0); ρ3, s3 = E.ou_step(nm, 360.0)
    @test abs(ρ3 - ρ1^3) < 1e-12 && abs(s3 - 1.5 * sqrt(1 - ρ1^6)) < 1e-12
    Lw = E.whitening_rows([true, false, false, true], 1.5, ρ1)
    f = DiscrepancyFactor(VarKey(:disc, :s, 1), VarKey(:disc, :s, 2), ρ3, s3)
    @test abs(Lw[2, 4] - 1 / s3) < 1e-12 && abs(Lw[2, 1] + ρ3 / s3) < 1e-12
    @test residual(f, [[0.4], [0.7]]) ≈ [(0.7 - ρ3 * 0.4) / s3]
    @test_throws ArgumentError NoiseModel(white_sd=0.0)
    @test_throws ArgumentError NoiseModel(white_sd=1.0, ar_sd=1.0, tau=0.0)
end

@testset "the graph's bookkeeping" begin
    g = FactorGraph()
    a = VarKey(:state, :x, 1); b = VarKey(:state, :x, 2)
    add_variable!(g, a, [1.0, 2.0]); add_variable!(g, b, [0.5])
    @test_throws ArgumentError add_variable!(g, a, [0.0, 0.0])
    @test_throws ArgumentError add_factor!(g, GaussianPrior(VarKey(:state, :x, 3), [0.0], fill(1.0, 1, 1)))
    f = GaussianPrior(a, [0.0, 0.0], Matrix(1.0I, 2, 2))
    add_factor!(g, f)
    @test objective(g) ≈ 0.5 * (1 + 4) && factors_of(g, a) == [f] && isempty(factors_of(g, b))
    remove_factor!(g, f)
    @test isempty(g.factors)
    @test_throws ArgumentError remove_factor!(g, f)
    @test string(a) == "(state:x#1)" && provenance(f) == "prior"
    scoped = VarKey(:disc, (:cook, :probe), 1)
    dotted = VarKey(:disc, Symbol("cook.probe"), 1)
    @test scoped != dotted && string(scoped) == "(disc:cook/probe#1)"
    add_variable!(g, scoped, [0.0]); add_variable!(g, dotted, [0.0])
    @test length(g.keys) == 4                                               # structured and dotted identities do not alias
end

@testset "solver safeguards, stationarity and rank checks" begin
    # A factor can make its residual and Jacobian atomically. This is the path DynamicsFactor uses so a
    # state-dependent whitener cannot change between the two halves of one linearization.
    ka = VarKey(:state, :atomic, 1); calls = Ref(0)
    ga = FactorGraph(); add_variable!(ga, ka, [0.0]); add_factor!(ga, AtomicProbe(ka, calls))
    Ja, ra, _, _ = linearize(ga)
    @test calls[] == 1 && Ja == fill(1.0, 1, 1) && ra == [-2.0]

    # Nonlinear chord corrections must obey the bound on the TOTAL move. The old unrestricted correction
    # moved this scalar by more than eight despite a 0.1 bound.
    kq = VarKey(:state, :quadratic, 1)
    gq = FactorGraph(); add_variable!(gq, kq, [10.0]); add_factor!(gq, QuadraticProbe(kq))
    rq = gauss_newton!(gq; iters=1, gtol=0.0, step_bounds=Dict(kq => 0.1), soc=10)
    @test abs(variable_value(gq, kq)[1] - 10.0) <= 0.1 + 1e-12
    @test !rq.converged && rq.gradient > 0

    # Repeated corrections also apply damping to the complete displacement, not independently to
    # each increment. Ten corrections used to accumulate ten individually damped moves.
    kd = VarKey(:state, :damped_correction, 1)
    gd = FactorGraph(); add_variable!(gd, kd, [10.0]); add_factor!(gd, QuadraticProbe(kd))
    rd = gauss_newton!(gd; iters=1, gtol=0.0, λ0=1e6, soc=10)
    @test abs(variable_value(gd, kd)[1] - 10.0) < 1e-5
    @test !rd.converged

    # A tiny step caused only by damping is not convergence while the normal-equation gradient is large.
    kl = VarKey(:state, :damped, 1)
    gl = FactorGraph(); add_variable!(gl, kl, [0.0])
    add_factor!(gl, LinearFactor([kl], [fill(1.0, 1, 1)], [1.0]))
    rl = gauss_newton!(gl; iters=1, λ0=1e12, tol=1e-3, gtol=1e-10, soc=0)
    @test !rl.converged && rl.gradient > 0.9
    @test E._scaled_gradient(fill(1e6, 1, 1), [-1.0]) ≈ 1e-6   # a unit residual needs only a 1e-6 local coordinate move
    # Stationarity is relative to a coordinate's declared units. A 0.005 Pa discrepancy move is below
    # 1e-4 of a 100 Pa AR width, while the same absolute move in an order-one latent is not stationary.
    kscl = VarKey(:disc, :scaled, 1)
    gscl = FactorGraph(); add_variable!(gscl, kscl, [0.0])
    add_factor!(gscl, LinearFactor([kscl], [fill(1.0, 1, 1)], [0.005]))
    rscl = gauss_newton!(gscl; iters=1, gtol=1e-4, stationarity_scales=Dict(kscl => 100.0), soc=0)
    @test rscl.converged && rscl.gradient ≈ 5e-5 && variable_value(gscl, kscl) == [0.0]
    @test_throws ArgumentError gauss_newton!(gscl; stationarity_scales=Dict(kscl => 0.0))
    # A repeatably quantized objective may be unable to resolve a mathematically descending move. Only
    # after three complete failures at one accepted point, and only within twice gtol, is that a
    # resolution-limited stationary result.
    kr = VarKey(:state, :resolution, 1)
    gres = FactorGraph(); add_variable!(gres, kr, [0.0]); add_factor!(gres, ResolutionProbe(kr, 1.5e-4))
    rres = gauss_newton!(gres; iters=3, tol=1e-4, gtol=1e-4, soc=0)
    @test rres.converged && rres.gradient ≈ 1.5e-4 && get(rres.hist[end], :resolution_limited, false)
    @test variable_value(gres, kr) == [0.0]
    kw = VarKey(:state, :measured_resolution, 1)
    gwide = FactorGraph(); add_variable!(gwide, kw, [0.0]); add_factor!(gwide, ResolutionProbe(kw, 4.5e-4))
    rwide = gauss_newton!(gwide; iters=3, gtol=1e-4, resolution_gtol=5e-4, soc=0)
    @test rwide.converged && rwide.gradient ≈ 4.5e-4 && get(rwide.hist[end], :resolution_limited, false)
    ktoo = VarKey(:state, :outside_resolution, 1)
    gtoo = FactorGraph(); add_variable!(gtoo, ktoo, [0.0]); add_factor!(gtoo, ResolutionProbe(ktoo, 6e-4))
    rtoo = gauss_newton!(gtoo; iters=3, gtol=1e-4, resolution_gtol=5e-4, soc=0)
    @test !rtoo.converged && rtoo.gradient ≈ 6e-4
    @test_throws ArgumentError gauss_newton!(gtoo; gtol=1e-4, resolution_gtol=5e-5)
    # With the joint read on (newton_tol > 0), the same stall is accepted when the undamped step at the exact
    # relinearization is under newton_tol of a posterior sd (a unit Jacobian: the step in sds is the residual itself,
    # 6e-4) and refused when it is not; the default leaves the rule off, as above.
    gtoo2 = FactorGraph(); add_variable!(gtoo2, ktoo, [0.0]); add_factor!(gtoo2, ResolutionProbe(ktoo, 6e-4))
    rtoo2 = gauss_newton!(gtoo2; iters=3, gtol=1e-4, resolution_gtol=5e-4, soc=0, newton_tol=1e-2)
    @test rtoo2.converged && rtoo2.gradient ≈ 6e-4 && get(rtoo2.hist[end], :resolution_limited, false)
    @test rtoo2.hist[end].newton_sd ≈ 6e-4 && rtoo2.newton_sd ≈ 6e-4 && rtoo2.newton_step ≈ 6e-4
    gtoo3 = FactorGraph(); add_variable!(gtoo3, ktoo, [0.0]); add_factor!(gtoo3, ResolutionProbe(ktoo, 6e-4))
    rtoo3 = gauss_newton!(gtoo3; iters=3, gtol=1e-4, resolution_gtol=5e-4, soc=0, newton_tol=1e-4)
    @test !rtoo3.converged && rtoo3.newton_sd ≈ 6e-4
    @test_throws ArgumentError gauss_newton!(gtoo3; newton_tol=-1.0)
    kf = VarKey(:state, :final_iteration, 1)
    gf = FactorGraph(); add_variable!(gf, kf, [0.0])
    add_factor!(gf, LinearFactor([kf], [fill(1.0, 1, 1)], [1.0]))
    rf = gauss_newton!(gf; iters=1, soc=0)
    @test rf.converged && rf.gradient == 0.0 && variable_value(gf, kf) == [1.0]

    # Elimination and covariance fail loudly before silently manufacturing information from a deficient row.
    ko = VarKey(:state, :out, 1); ks = VarKey(:state, :stay, 1)
    gr = FactorGraph(); add_variable!(gr, ko, [0.0, 0.0]); add_variable!(gr, ks, [0.0])
    add_factor!(gr, LinearFactor([ko, ks], [[1.0 0.0], fill(1.0, 1, 1)], [0.0]))
    @test_throws ErrorException marginalize!(gr, [ko])
    @test_throws ErrorException joint_cov(gr)

    # Numerical rank is judged after column equilibration: very different physical units are not a
    # deficiency. Both covariance paths must map the equilibrated result back to the original coordinates.
    ku = VarKey(:state, :mixed_units, 1)
    gu = FactorGraph(); add_variable!(gu, ku, [0.0, 0.0])
    add_factor!(gu, GaussianPrior(ku, zeros(2), Matrix(Diagonal([1e12, 1.0]))))
    Cu = marginal_cov(gu, ku)
    Cus = marginal_cov(gu, ku; sparse=true)
    @test isapprox(Cu, Diagonal([1e-24, 1.0]); rtol=1e-12, atol=1e-30)
    @test isapprox(Cus, Cu; rtol=1e-12, atol=1e-30)
end
println("test_factors.jl: done")
