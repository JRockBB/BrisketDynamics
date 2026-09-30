if !isdefined(Main,:BrisketEstimator)
    include(joinpath(@__DIR__,"..","src","BrisketEstimator.jl"))
end
using .BrisketEstimator, BrisketDynamics, LinearAlgebra, Test
isdefined(Main,:E) || (const E=BrisketEstimator)
isdefined(Main,:consistency_model) || include(joinpath(@__DIR__,"..","scripts","consistency_control.jl"))

@testset "conditional online transactions and bounded memory" begin
    model=consistency_model(nx=4,suite=:pit,stop=60.0)
    # Pit data contain exactly no information about insertion depth. This gives
    # an analytic invariant independent of either inference implementation.
    sm=conditional_init(model;axes=[:core_depth_fraction],orders=(3,5),lag=0,record_limit=2)
    depth=E.latent_index(model.blk,:core_depth_fraction)
    @test belief_at(sm).z[depth]≈0.0 atol=1e-12
    @test belief_at(sm).z_cov[depth,depth]≈1.0 atol=1e-12
    recs=[MeasurementRecord(:pit,0.0,294.0),MeasurementRecord(:pit,10.0,294.4),
          MeasurementRecord(:pit,20.0,295.0)]
    snapshot=sm.posterior;original=copy(snapshot.mean)
    for rec in recs
        result=assimilate!(sm,rec)
        @test result.committed
        @test result.z[depth]≈0.0 atol=1e-8
        @test result.z_cov[depth,depth]≈1.0 atol=1e-8
        @test length(sm.anchor.nodes)==1
        @test all(length(inner.nodes)==1 for bank in sm.banks for inner in bank.smoothers)
        @test size(result.zy_cov)==(length(model.blk)+model.bl.n,length(model.blk)+model.bl.n)
        @test haskey(result.pieces,:brisket)
        @test result.T_pit[2]>0
    end
    @test snapshot.mean==original
    @test length(sm.anchor.records)==2
    batched=conditional_init(model;axes=[:core_depth_fraction],orders=(3,5),lag=0)
    @test assimilate!(batched,recs).committed
    @test belief_at(batched).zy_cov≈belief_at(sm).zy_cov rtol=1e-8 atol=1e-8
    @test belief_at(batched).y≈belief_at(sm).y atol=1e-8
    before=deepcopy(sm)
    @test_throws ArgumentError assimilate!(sm,MeasurementRecord(:pit,19.0,300.0))
    @test_throws ArgumentError assimilate!(sm,MeasurementRecord(:pit,21.0,NaN))
    @test sm.posterior.mean==before.posterior.mean
    @test sm.anchor.graph.values==before.anchor.graph.values
    # Force a conditional (not anchor) optimizer failure after other components
    # have already updated, and require rollback of every bank and anchor.
    sm.banks[end].smoothers[end].iters=0
    before=deepcopy(sm)
    failure=assimilate!(sm,MeasurementRecord(:pit,20.0,305.0))
    @test !failure.committed
    @test failure.reason==:conditional_solve
    @test sm.posterior.mean==before.posterior.mean
    @test sm.anchor.graph.values==before.anchor.graph.values
    for (bank,old) in zip(sm.banks,before.banks)
        @test bank.logweights==old.logweights
        @test all(a.graph.values==b.graph.values for (a,b) in zip(bank.smoothers,old.smoothers))
    end
end

@testset "nonlinear innovation transport preserves the generative prior" begin
    model=consistency_model(nx=4,suite=:core,stop=120.0)
    inner=online_init(model;lag=typemax(Int))
    E._node_at!(inner,120.0)
    kz=inner.key_z;a0=E.latent_index(model.blk,:a0)
    node=E._conditional_node!(deepcopy(inner.graph),[(kz,a0)],[0.0];tol=1e-6)
    @test node.solve.converged
    p=E._conditional_posterior([node],[0.0],[(kz,a0)])
    tp=conditional_transport(p);ref=only(tp.components)
    @test E._conditional_decode(ref,ref.mean)≈node.mean atol=1e-10
    # No observations: the prior is EXACTLY Gaussian in parameter, initial-state
    # and independent process-noise coordinates. Several deterministic joint
    # displacements must agree with the independent Gaussian log density.
    g=deepcopy(node.graph);off,_=E._offsets(g);F=objective(g)
    for scale in (.2,.7,1.0)
        v=scale.*sin.(collect(1:size(ref.root,2)))
        x=E._conditional_decode(ref,ref.mean+ref.root*v)
        @test E._objective_at(g,x,off)-F≈sum(abs2,v)/2 atol=2e-5 rtol=2e-5
        @test x[off[kz][a0]]==0.0
    end
    physical=conditional_shape_diagnostic(p;key=kz,max_components=1)
    transported=conditional_shape_diagnostic(tp;key=kz,max_components=1)
    @test physical.maximum_defect>1.0
    @test transported.maximum_defect<1e-3
    @test transported.maximum_defect<physical.maximum_defect/1000
    @test !transported.validated
    @test_throws ArgumentError conditional_points(tp;max_points=1)
    @test_throws ArgumentError conditional_shape_diagnostic(tp;key=kz,max_components=0)
    # Extraction owns its historical physics and anchors.
    old=E._conditional_decode(ref,ref.mean+ref.root[:,1])
    node.graph.values[kz].+=.1
    @test E._conditional_decode(ref,ref.mean+ref.root[:,1])==old
end
