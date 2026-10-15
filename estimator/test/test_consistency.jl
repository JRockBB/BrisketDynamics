# Fast regressions for probability-model consistency, independent of a long coverage run.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics
using LinearAlgebra, Test, ForwardDiff
const E = BrisketEstimator

cfg = MeatConfig(nx=4)
sys = CookSystem(Slab1D(:brisket; cfg); pit=PitConfig())
prior = latent_block(joinpath(@__DIR__, "..", "..", "data", "calibration", "prior_next_cook.json"), cfg)
blk = system_latents(sys, prior.prior_phys, prior.phys_names; probes=[:brisket])
sensors = [SensorSpec(:fast, PitTemperature(), NoiseModel(white_sd=0.2, ar_sd=1.5, tau=3000.0)),
           SensorSpec(:slow, PitTemperature(), NoiseModel(white_sd=0.3, ar_sd=2.0, tau=4000.0))]
model = SystemModel(sys, CookLog([0.0, 120.0], [0.5, 0.5]; sensors), blk)

@testset "integration accuracy can change without narrowing process covariance" begin
    accurate = SystemModel(sys, model.log, blk; abstol=1e-14, reltol=1e-12,
                           floor_abstol=model.abstol, floor_reltol=model.reltol)
    coupled = SystemModel(sys, model.log, blk; abstol=1e-14, reltol=1e-12)
    sm = online_init(accurate)
    ky = first(sm.nodes)[2]
    y = variable_value(sm.graph, ky)
    factor(m) = DynamicsFactor(m, sm.key_z, ky, VarKey(:state, :floor_test, 1), 0.0, 120.0)
    Q = zeros(model.bl.n, model.bl.n) # isolate the numerical allowance
    reference = E._process_cov_coord(factor(model), y, Q)
    @test E._process_cov_coord(factor(accurate), y, Q) == reference
    # the allowance scales with the tolerances away from any floor; at a floored positive state it is boundary-damped
    # (factors.jl `_process_cov_coord`), so the tighter solver's allowance is smaller there but not by the tolerances' ratio
    x = belief_decode(y, model.bl)
    open = [i for i in 1:model.bl.n if model.bl.kinds[i] !== :positive || x[i] > 1e-3]
    @test length(open) < model.bl.n                                          # the salt and the pan sit at their floors here
    cpl = E._process_cov_coord(factor(coupled), y, Q)
    @test cpl[open, open] ≈ 1e-8 .* reference[open, open]
    @test all(cpl[i, i] < reference[i, i] for i in 1:model.bl.n)
    @test accurate.abstol == 1e-14 && accurate.reltol == 1e-12
    update_execution!(sm; u=[0.4, 0.4])
    @test sm.model.floor_abstol == model.abstol && sm.model.floor_reltol == model.reltol
    @test sm.model.abstol == accurate.abstol && sm.model.reltol == accurate.reltol
    old_constructor = SystemModel(sys, model.log, blk, model.bl, model.sn, model.abstol, model.reltol, nothing)
    @test E._process_cov_coord(factor(old_constructor), y, Q) == reference
    @test_throws ArgumentError SystemModel(sys, model.log, blk; floor_abstol=NaN)
    @test_throws ArgumentError SystemModel(sys, model.log, blk; floor_reltol=-1.0)
end

@testset "factor graph consistency regressions" begin
@testset "batched innovations use chronological conditional variances" begin
    # The second observation has conditional mean 0.5 times the first
    # and conditional variance 8. Supply it first to check ordering too.
    ν = [4.0, 2.0]; S = [9.0 2.0; 2.0 4.0]
    w = E._whitened_innovations(ν, S, [:late, :early], [20.0, 10.0])
    @test getproperty.(w, :sensor) == [:early, :late]
    @test getproperty.(w, :value) ≈ [1.0, 3 / sqrt(8)]
    @test sum(abs2(r.value) for r in w) ≈ dot(ν, S \ ν)
end
@testset "state coordinates preserve the physical interior through repeated handoffs" begin
    for kind in (:positive, :unit, :ignited)
        values = kind === :positive ? [3e-8, 1e-7, 2e-6, 0.002, 1.0, 300.0] :
                                     [3e-8, 1e-7, 2e-6, 0.002, 0.01, 0.5, 0.998, 1-3e-8]
        roundtrip(x) = E._belief_dec(E._belief_enc(x, kind), kind)
        for x in values
            @test roundtrip(x) ≈ x rtol=1e-9 atol=2e-16
            @test ForwardDiff.derivative(roundtrip, x) ≈ 1.0 rtol=1e-8
            repeated = x
            for _ in 1:100; repeated = roundtrip(repeated); end
            @test repeated ≈ x rtol=1e-9 atol=2e-16
        end
        @test abs(roundtrip(0.0)) <= 2.1e-8
        kind === :positive || @test abs(roundtrip(1.0)-1.0) <= 2.1e-8
    end
end

@testset "an idle sensor's discrepancy receives the correlated Kalman correction" begin
    feed = ALSFeed(model)
    sm = online_init(model; als=feed)
    ip = first(E.belief_index(model.bl, :pit, :T_pit))
    T0 = belief_decode(variable_value(sm.graph, sm.nodes[1][2]), model.bl)[ip]
    # An earlier, zero-innovation reading correlates the pit with the slow sensor's discrepancy.
    add_measurement!(sm, MeasurementRecord(:slow, 0.0, T0))
    f = add_measurement!(sm, MeasurementRecord(:fast, 0.0, T0 + 1.0))
    pc = E._predicted_consistency(sm, [f])
    ap = E._als_pieces(sm, feed, pc, 0.0)
    # Independent three-state Kalman oracle: [pit, fast discrepancy, slow discrepancy].
    # The declared initial covariance applies the smooth coordinate-sd cap even to temperatures.
    s0 = model.sn.T_pit0
    v0 = s0^2 / (1 + (s0 / 3)^8)^(1 / 4)
    P0 = Matrix(Diagonal([v0, 1.5^2, 2.0^2]))
    hslow = [1.0, 0.0, 1.0]
    P = P0 - (P0 * hslow) * (P0 * hslow)' / (dot(hslow, P0 * hslow) + 0.3^2)
    hfast = [1.0, 1.0, 0.0]
    S = dot(hfast, P * hfast) + 0.2^2
    correction = P * hfast / S
    nz, ny = length(blk), model.bl.n
    rows = [nz + ip, nz + ny + 1, nz + ny + 2]
    @test pc.nis ≈ 1 / S
    @test length(pc.white) == 1 && sum(abs2(w.value) for w in pc.white) ≈ pc.nis
    @test abs(correction[3]) > 0.1
    @test ap.K[rows, :] * ap.e ≈ correction rtol=1e-10
    @test haskey(pc.off, sm.disc[:slow][end][2])
    @test ap.dQ[feed.ou_group[:fast]][nz + ny + 1, nz + ny + 1] == 1.0
    @test ap.dQ[feed.ou_group[:slow]][nz + ny + 2, nz + ny + 2] == 1.0
end

@testset "ALS rejects arrival orders its single-frame recursion cannot represent" begin
    sm = online_init(model; als=ALSFeed(model))
    before = length(sm.graph.factors)
    @test_throws ArgumentError assimilate!(sm, [MeasurementRecord(:fast, 0.0, 300.0), MeasurementRecord(:slow, 120.0, 300.0)])
    @test length(sm.graph.factors) == before && sm.als.acc.n_updates == 0
    # Model the bookkeeping after one committed feed without paying for a nonlinear solve.
    sm.als.acc.n_updates = 1
    @test_throws ArgumentError assimilate!(sm, MeasurementRecord(:fast, 0.0, 300.0))
    @test length(sm.graph.factors) == before
end

@testset "a censored-only arrival still advances the calibration's process sensitivities" begin
    censored = [SensorSpec(:cap, PitTemperature(), NoiseModel(white_sd=0.2, ar_sd=1.5, tau=3000.0, cap=280.0))]
    cm = SystemModel(sys, CookLog([0.0, 120.0], [0.5, 0.5]; sensors=censored), blk)
    feed = ALSFeed(cm)
    sm = online_init(cm; als=feed)
    f = add_measurement!(sm, MeasurementRecord(:cap, 0.0, 280.0))
    pc = E._predicted_consistency(sm, [f])
    @test pc !== nothing && pc.dof == 0 && isnan(pc.nis)
    if pc !== nothing
        ap = E._als_pieces(sm, feed, pc, 0.0)
        als_update!(feed.acc, ap.F, ap.dQ, ap.H, ap.S, ap.K, ap.e, ap.rowch, 0.0; dR=ap.dR)
        @test feed.acc.n_updates == 1 && feed.acc.n_rows == 0
        @test feed.acc.Pi[feed.ou_group[:cap]][end, end] == 1.0
    end
end

@testset "a dynamics factor has one objective across relinearization" begin
    sm = online_init(model)
    z = copy(variable_value(sm.graph, sm.key_z))
    y0 = copy(variable_value(sm.graph, sm.nodes[1][2]))
    rC = E.belief_index(model.bl, :brisket, :C)
    xb = belief_decode(y0, model.bl); xb[rC] .= 0.02
    y0 = belief_encode(xb, model.bl)
    f = DynamicsFactor(model, sm.key_z, sm.nodes[1][2], VarKey(:state, :cook, 2), 0.0, 1.0)
    y1, _ = E.propagate(model, y0, z, 0.0, 1.0)
    factor_linearization(f, [z, y0, y1])
    W0 = copy(f.cache.W)
    ym = copy(y0); ym[rC] .+= 0.2
    vals = [z, ym, y1]
    before = loss(f, vals)
    lin = factor_linearization(f, vals)
    @test loss(f, vals) ≈ before rtol=1e-13
    @test f.cache.W == W0
    # A cache invalidation, including the exact refresh used before covariance/marginalization,
    # must also preserve the objective and its finite-difference derivative.
    E._invalidate_linearization!(f)
    lin = factor_linearization(f, vals)
    @test loss(f, vals) ≈ before rtol=1e-13
    v = zeros(length(ym)); v[rC] .= 1.0
    h = 1e-4
    fd = (loss(f, [z, ym + h*v, y1]) - loss(f, [z, ym - h*v, y1])) / (2h)
    @test dot(lin.Js[2]' * lin.r, v) ≈ fd rtol=2e-3
    # The finite-difference trials moved only the primal cache. Its Jacobian
    # is still exact at vals and must survive restoration of that accepted mean.
    sm.graph.values[f.key_y0] .= ym
    add_variable!(sm.graph, f.key_y1, y1)
    nj = f.n_jacobians
    E._ensure_current!(f, sm.graph)
    restored = factor_linearization(f, vals)
    @test f.n_jacobians == nj
    @test restored.Js == lin.Js
end

@testset "process variance derivatives include events inside the nonlinear coordinate cap" begin
    sm = online_init(model)
    y = copy(variable_value(sm.graph, sm.nodes[1][2]))
    im = first(E.belief_index(model.bl, :pit, :m_fuel))
    xb = belief_decode(y, model.bl); xb[im] = 0.01
    y = belief_encode(xb, model.bl)
    f = DynamicsFactor(model, sm.key_z, sm.nodes[1][2], VarKey(:state, :cook, 2), 0.0, 120.0; events=[:add])
    names = [E._als_group_name(model.bl, g) for (g, _) in model.bl.groups]
    gi = findfirst(==("m_fuel"), names)
    D = E._process_cov_sens(f, y, names)[gi]
    theta = model.sn.m_fuel^2
    function at_variance(q)
        sn = StateNoise(m_fuel=sqrt(q))
        Q = belief_process_cov(model.bl, sn, 120 / 3600; events=[:add])
        return E._process_cov_coord(f, y, Q)
    end
    h = theta * 0.01
    fd = (at_variance(theta + h) - at_variance(theta - h)) / (2h)
    @test isapprox(D, fd; rtol=1e-3, atol=1e-8)
    # the event saturates this coordinate's variance cap, and the whitener's boundary damping (the fuel 0.01 kg from its
    # floor) leaves a small, finite-difference-verified dependence on the rate: under a thousandth of the raw Δh·d² derivative
    @test abs(D[im, im]) < 1e-3 * (120 / 3600) / 0.01^2
end
end
