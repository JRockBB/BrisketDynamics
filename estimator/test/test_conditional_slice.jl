# The conditional (quadrature) backend under attach=:slice: the anchor decides the graph structure
# (chunks, node times, slice membership) and every component mirrors it with the anchor's slice
# sensitivities, so component evidences compare under one noise model. Algebra on the consistency
# fixture; these check the mechanism, not statistical consistency.
if !isdefined(Main,:BrisketEstimator)
    include(joinpath(@__DIR__,"..","src","BrisketEstimator.jl"))
end
using .BrisketEstimator, BrisketDynamics, LinearAlgebra, Test
isdefined(Main,:E) || (const E=BrisketEstimator)
isdefined(Main,:consistency_model) || include(joinpath(@__DIR__,"..","scripts","consistency_control.jl"))

components(sm)=[inner for bank in sm.banks for inner in bank.smoothers]

@testset "slice components mirror the anchor's structure and noise" begin
    model=consistency_model(nx=4,suite=:pit,stop=600.0)
    # Pit data contain exactly no information about insertion depth: the analytic invariant.
    sm=conditional_init(model;axes=[:core_depth_fraction],orders=(3,5),lag=2,
                        attach=:slice,node_ratio=Inf,max_slice_s=90.0)
    depth=E.latent_index(model.blk,:core_depth_fraction)
    recs=[MeasurementRecord(:pit,t,294.0+0.02t) for t in 30.0:30.0:300.0]
    made=0
    for rec in recs
        n0=sm.anchor.node_count
        r=assimilate!(sm,rec)
        @test r.committed
        made+=sm.anchor.node_count-n0
        times=first.(sm.anchor.nodes)
        @test all(first.(inner.nodes)==times for inner in components(sm))
        # one noise model: every component's slice carries the anchor's Σ
        for (ka,f) in sm.anchor.slices
            @test all(inner.slices[ka].Σ==f.Σ for inner in components(sm))
        end
        @test r.z[depth]≈0.0 atol=1e-8
        @test r.z_cov[depth,depth]≈1.0 atol=1e-8
    end
    @test made>=2                                       # the age cap created nodes: the mirror was exercised
    @test length(sm.anchor.nodes)<=sm.anchor.lag+1+1    # compaction kept the window bounded
    @test all(length(inner.nodes)==length(sm.anchor.nodes) for inner in components(sm))
    @test_throws ArgumentError conditional_init(model;axes=[:core_depth_fraction],orders=(3,5),
                                                attach=:slice,slice_g_refresh=true)
end

@testset "a batch crossing several nodes advances a node at a time, mirrored" begin
    model=consistency_model(nx=4,suite=:pit,stop=600.0)
    one_by_one=conditional_init(model;axes=[:core_depth_fraction],orders=(3,5),lag=typemax(Int)÷2,
                                attach=:slice,node_ratio=Inf,max_slice_s=90.0)
    batched=conditional_init(model;axes=[:core_depth_fraction],orders=(3,5),lag=typemax(Int)÷2,
                             attach=:slice,node_ratio=Inf,max_slice_s=90.0)
    recs=[MeasurementRecord(:pit,t,294.0+0.02t) for t in 30.0:30.0:300.0]
    for rec in recs;@test assimilate!(one_by_one,rec).committed;end
    @test assimilate!(batched,recs).committed
    @test first.(batched.anchor.nodes)==first.(one_by_one.anchor.nodes)
    @test all(first.(inner.nodes)==first.(batched.anchor.nodes) for inner in components(batched))
    # Two arrival paths agree to the solver's declared accuracy (ONLINE_NEWTON_TOL, posterior sds),
    # not to roundoff: an iterated solve's endpoint depends on its path within that tolerance.
    a,b=belief_at(batched),belief_at(one_by_one)
    sd=sqrt.(max.(diag(b.z_cov),eps()))
    @test maximum(abs.(a.z.-b.z)./sd)<=2E.ONLINE_NEWTON_TOL
end
