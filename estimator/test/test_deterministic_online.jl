if !isdefined(Main,:BrisketEstimator)
    include(joinpath(@__DIR__,"..","src","BrisketEstimator.jl"))
end
using .BrisketEstimator, BrisketDynamics, LinearAlgebra, Test
isdefined(Main,:E) || (const E=BrisketEstimator)
isdefined(Main,:consistency_model) || include(joinpath(@__DIR__,"..","scripts","consistency_control.jl"))

@testset "adaptive charts preserve an unobserved prior and model ownership" begin
    model=consistency_model(nx=4,suite=:pit,stop=30.)
    sm=deterministic_init(model;orders=(3,5))
    key=sm.bank.anchor.key_z;depth=E.latent_index(model.blk,:core_depth_fraction)
    @test sm.adaptive_axes
    @test sm.bank.anchor.lag==typemax(Int)
    @test E._rebuild_density_axes!(sm,[1,depth])
    @test length(sm.bank.axes)==2
    @test assimilate!(sm,MeasurementRecord(:pit,0.,294.)).committed
    @test conditional_cdf(sm.posterior,key,depth,.4)≈E._conditional_phi(.4) atol=1e-8
    oldbanks=sm.bank.banks;oldchart=copy(sm.scale);sm.max_nodes=1
    @test !E._rebuild_density_axes!(sm,[depth])
    @test sm.bank.banks===oldbanks
    @test sm.scale==oldchart
    offline=deterministic_fit(model,MeasurementRecord[];axes=[:core_depth_fraction],orders=(3,5))
    @test offline.committed && offline.converged
    @test conditional_cdf(offline.posterior,key,depth,.4)≈E._conditional_phi(.4) atol=1e-8
    readings=[MeasurementRecord(:pit,0.,294.),MeasurementRecord(:pit,10.,294.4)]
    offline=deterministic_fit(model,readings;axes=[:core_depth_fraction],orders=(3,5))
    online=deterministic_init(model;axes=[:core_depth_fraction],orders=(3,5))
    @test offline.committed && assimilate!(online,readings).committed
    @test conditional_cdf(offline.posterior,key,depth,.4)≈E._conditional_phi(.4) atol=1e-8
    @test conditional_density_distance(offline.posterior,online.posterior).tv_bound_estimate<1e-8
end
@testset "continuous online posterior: invariance, snapshots and transactions" begin
    model=consistency_model(nx=4,suite=:pit,stop=60.0)
    kw=(axes=[:core_depth_fraction],orders=(3,5),lag=0,record_limit=2)
    sm=deterministic_init(model;kw...)
    key=sm.bank.anchor.key_z;depth=E.latent_index(model.blk,:core_depth_fraction)
    original=belief_at(sm);oldcdf=conditional_cdf(original,key,depth,.3)
    @test original isa ConditionalDensity
    @test oldcdf≈E._conditional_phi(.3) atol=1e-9
    readings=[MeasurementRecord(:pit,0.,294.),MeasurementRecord(:pit,10.,294.4),MeasurementRecord(:pit,20.,295.)]
    for rec in readings
        result=assimilate!(sm,rec)
        @test result.committed
        @test result.posterior===belief_at(sm)
        @test conditional_cdf(result.posterior,key,depth,.3)≈oldcdf atol=1e-8
        @test length(window_nodes(sm))==1
    end
    @test conditional_cdf(original,key,depth,.3)==oldcdf
    @test length(sm.bank.anchor.records)==2
    batched=deterministic_init(model;kw...)
    @test assimilate!(batched,readings).committed
    @test conditional_density_distance(sm.posterior,batched.posterior).tv_bound_estimate<1e-6
    before=deepcopy(sm)
    @test_throws ArgumentError assimilate!(sm,MeasurementRecord(:pit,19.,300.))
    @test_throws ArgumentError assimilate!(sm,MeasurementRecord(:pit,21.,NaN))
    @test sm.bank.anchor.graph.values==before.bank.anchor.graph.values
    sm.bank.banks[end].smoothers[end].iters=0
    before=deepcopy(sm)
    failed=assimilate!(sm,MeasurementRecord(:pit,20.,305.))
    @test !failed.committed
    @test failed.reason==:conditional_solve
    @test sm.bank.anchor.graph.values==before.bank.anchor.graph.values
    @test all(a.logweights==b.logweights for (a,b) in zip(sm.bank.banks,before.bank.banks))
    @test conditional_cdf(sm.posterior,key,depth,.3)==conditional_cdf(before.posterior,key,depth,.3)
    @test_throws ArgumentError deterministic_init(model;max_axes=4)
    @test_throws ArgumentError deterministic_fit(model,readings;lag=1)
end

@testset "density comparison requires the same physics decoder" begin
    model=consistency_model(nx=4,suite=:pit,stop=30.)
    sm=deterministic_init(model;axes=[:core_depth_fraction],orders=(3,5))
    @test assimilate!(sm,[MeasurementRecord(:pit,0.,294.),MeasurementRecord(:pit,10.,294.4)]).committed
    d=belief_at(sm);other=deepcopy(d)
    @test conditional_density_distance(d,other).tv_bound_estimate<1e-10
    f=first(first(other.transported.components).dynamics)
    f.model=SystemModel(f.model.sys,f.model.log,f.model.blk;sn=f.model.sn,abstol=2f.model.abstol,reltol=f.model.reltol)
    @test_throws ArgumentError conditional_density_distance(d,other)
    other=deepcopy(d);first(first(other.transported.components).dynamics).cache.W[1,1]*=2
    @test_throws ArgumentError conditional_density_distance(d,other)
    ky=last(sm.bank.anchor.nodes)[2]
    @test_throws ArgumentError conditional_cdf(d,ky,1,300.)
end
