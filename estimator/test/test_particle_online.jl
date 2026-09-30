using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
include(joinpath(@__DIR__,"..","src","BrisketEstimator.jl"))
using .BrisketEstimator,BrisketDynamics,LinearAlgebra,Random,Statistics,Test
const E=BrisketEstimator
include(joinpath(@__DIR__,"..","scripts","consistency_control.jl"))

@testset "incremental likelihood equals full correlated likelihood" begin
    model=consistency_model(nx=4,suite=:pit_core,stop=600.)
    sm=online_init(model;lag=typemax(Int),record_limit=0)
    @test assimilate!(sm,[MeasurementRecord(:pit,0.,294.),MeasurementRecord(:core,0.,281.)]).committed
    old=E._posterior_transport(sm);rng=MersenneTwister(401)
    old_u=old.center+randn(rng,old.n)*.1;r0=E._transport_draw(old,old_u)
    # Repeating an old-time reading, then asynchronously reporting only one probe, preserves the
    # old covariance block and conditions the new likelihood on all retained sensor residuals.
    for rec in (MeasurementRecord(:pit,0.,294.2),MeasurementRecord(:core,60.,281.3),MeasurementRecord(:pit,120.,300.))
        E.add_measurement!(sm,rec)
        ref=E._posterior_transport(sm);E._validate_particle_extension(old,ref)
        u=vcat(old_u,randn(rng,ref.n-old.n));full=E._transport_draw(ref,u)
        incremental=E._transport_extend(old,ref,u,r0.draw,r0.predictions)
        @test incremental.draw≈full.draw atol=1e-9
        @test incremental.predictions≈full.predictions atol=1e-9
        @test incremental.log_increment≈-full.potential+r0.potential+sum(abs2,u[old.n+1:end])/2 atol=1e-7
        old=ref;old_u=u;r0=full
    end
end

@testset "online joint particles, repeat readings and continuation" begin
    model=consistency_model(nx=4,suite=:pit,stop=600.)
    sm=sampled_init(model;rng=MersenneTwister(411))
    @test sm isa ParticleSmoother
    @test sm.posterior.history!==nothing
    inner=sm.graph_backend;ky=inner.nodes[1][2];ip=first(E.belief_index(model.bl,:pit,:T_pit));nz=length(model.blk)
    prior=only(f for f in factors_of(inner.graph,ky) if f isa GaussianPrior)
    C0=inv(prior.R'*prior.R);v0=C0[ip,ip];mean0=prior.m[ip]
    spec=only(model.log.sensors);value=mean0+.5
    initial_coordinates=copy(sm.posterior.history.coordinates)
    result=assimilate!(sm,MeasurementRecord(:pit,0.,value))
    @test result.committed
    @test result.diagnostics.update===:incremental
    @test sm.posterior.history.coordinates==initial_coordinates
    checkpoint=smoother_checkpoint(sm);copy_sm=restore_checkpoint(checkpoint)
    @test copy_sm.posterior.draws==sm.posterior.draws
    @test copy_sm.posterior!==sm.posterior
    readings=fill(MeasurementRecord(:pit,0.,value),7)
    result=assimilate!(sm,readings);copy_result=assimilate!(copy_sm,readings)
    @test result.committed && copy_result.committed
    @test result.posterior.draws==copy_result.posterior.draws
    @test result.posterior.weights==copy_result.posterior.weights
    effective_R=spec.noise.ar_sd^2+spec.noise.white_sd^2/8
    expected_var=v0*effective_R/(v0+effective_R)
    expected_mean=mean0+v0/(v0+effective_R)*(value-mean0)
    @test abs(result.T_pit[1]-expected_mean)<6result.diagnostics.mean_mcse[nz+ip]
    @test abs(result.y_cov[ip,ip]/expected_var-1)<.08
    depth=E.latent_index(model.blk,:core_depth_fraction)
    @test abs(result.z[depth])<6result.diagnostics.mean_mcse[depth]
    @test abs(result.z_cov[depth,depth]-1)<.1
    @test result.diagnostics.propagated_intervals==0
    before=sm.posterior;values=deepcopy(inner.graph.values);count=inner.record_count
    @test_throws ArgumentError assimilate!(sm,MeasurementRecord(:pit,-1.,value))
    @test_throws ArgumentError assimilate!(sm,[MeasurementRecord(:pit,20.,value),MeasurementRecord(:pit,10.,value)])
    @test_throws ArgumentError assimilate!(sm,MeasurementRecord(:pit,NaN,value))
    @test sm.posterior===before && inner.graph.values==values && inner.record_count==count
    # A deliberately insufficient refresh budget must refuse the update and retain the accepted past.
    sm.particles=2;sm.max_particles=2
    refused=assimilate!(sm,MeasurementRecord(:pit,0.,value);refresh=true)
    @test !refused.committed && refused.reason===:monte_carlo_precision
    @test sm.posterior===before && inner.graph.values==values && inner.record_count==count
    # A later record outside the execution log fails after the first record has modified the graph.
    @test_throws Exception assimilate!(sm,[MeasurementRecord(:pit,0.,value),MeasurementRecord(:pit,1200.,value)])
    @test sm.posterior===before && inner.graph.values==values && inner.record_count==count
    restore_checkpoint!(sm,checkpoint)
    @test sm.posterior.weights==checkpoint.smoother.posterior.weights
    @test sm.maxstages==checkpoint.smoother.maxstages
    # Execution records change future factors without rewriting the accepted past.
    old_model=sm.graph_backend.model
    update_execution!(sm;u=fill(.4,length(model.log.u)))
    @test sm.graph_backend.model!==old_model
    @test sm.posterior.model===sm.graph_backend.model
    @test sm.posterior.history.reference.model===old_model
end

@testset "online precision budgets and interfaces" begin
    model=consistency_model(nx=4,suite=:core,stop=600.)
    @test_throws ArgumentError sampled_init(model;mode=:unknown)
    @test_throws ArgumentError particle_init(model;particles=1)
    @test_throws ArgumentError particle_init(model;replicates=2)
    @test_throws ArgumentError particle_init(model;lag=3)
    @test_throws ArgumentError particle_init(model;als=nothing)
    @test_throws ArgumentError particle_init(model;particles=2,max_particles=2,rng=MersenneTwister(412))
end

@testset "arrival batching preserves the chronological process model" begin
    model=consistency_model(nx=4,suite=:pit_core,stop=600.)
    recs=[MeasurementRecord(:pit,0.,294.),MeasurementRecord(:core,0.,281.),
          MeasurementRecord(:core,60.,281.3),MeasurementRecord(:pit,120.,300.)]
    batched=online_init(model;lag=typemax(Int));separate=online_init(model;lag=typemax(Int))
    E._append_sampled_measurements!(batched,recs)
    for group in (recs[1:2],recs[3:3],recs[4:4]);E._append_sampled_measurements!(separate,group);end
    a=E._posterior_transport(batched);b=E._posterior_transport(separate)
    @test a.roots==b.roots
    @test a.center==b.center
    @test a.W==b.W
    u=randn(MersenneTwister(421),a.n)
    @test E._transport_draw(a,u).potential==E._transport_draw(b,u).potential
end
