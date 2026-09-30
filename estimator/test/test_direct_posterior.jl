if !isdefined(@__MODULE__,:E)
    include(joinpath(@__DIR__,"../src/BrisketEstimator.jl"))
    using .BrisketEstimator
    const E=BrisketEstimator
end
using Test, LinearAlgebra, BrisketDynamics
isdefined(@__MODULE__,:consistency_model) || include(joinpath(@__DIR__,"../scripts/consistency_control.jl"))
isdefined(@__MODULE__,:DirectPosterior) || include(joinpath(@__DIR__,"../scripts/direct_online.jl"))

@testset "direct prior, CDFs and parallel moment isolation" begin
    model=consistency_model(nx=4,stop=120.,suite=:core)
    anchor=E.online_init(model;lag=typemax(Int),record_limit=0)
    original=deepcopy(anchor)
    p=direct_posterior_fit(anchor;max_evaluations=15000,max_seconds=120.)
    @test p.diagnostics.resolved
    @test anchor.graph.values==original.graph.values
    @test isempty(anchor.obs)
    @test p.normalizer≈1 atol=1e-8
    nz=length(model.blk)
    @test p.mean[1:nz]≈[E.z_moments(s)[1] for s in model.blk.sites] atol=1e-8
    @test p.covariance[1:nz,1:nz]≈Matrix(Diagonal([E.z_moments(s)[2] for s in model.blk.sites])) atol=1e-8
    # Repetition detects shared closure temporaries in threaded target moments.
    repeated=direct_posterior_fit(anchor;max_evaluations=15000,max_seconds=120.)
    @test repeated.mean==p.mean
    @test repeated.covariance==p.covariance
    for i in (1,nz),z in (-1.,.3,1.7)
        mu,variance=E.z_moments(model.blk.sites[i])
        c=direct_cdf(p,i,mu+sqrt(variance)*z;max_evaluations=15000,max_seconds=120.)
        @test c.resolved
        @test c.value≈.5E.erfc(-z/sqrt(2)) atol=1e-8
    end
    state=first(anchor.nodes)[2]
    prior=only(f for f in E.factors_of(anchor.graph,state) if f isa E.GaussianPrior)
    i=first(E.belief_index(model.bl,:brisket,:T))
    sd=norm(inv(prior.R)[i,:]);value=prior.m[i]+.4sd
    c=direct_cdf(p,nz+i,value;max_evaluations=15000,max_seconds=120.)
    @test c.resolved
    @test c.value≈.5E.erfc(-.4/sqrt(2)) atol=2e-8
    @test_throws ArgumentError direct_cdf(p,0,1.)
    @test_throws ArgumentError direct_cdf(p,1,NaN)
end

@testset "direct online/offline and rollback" begin
    model=consistency_model(nx=4,stop=120.,suite=:core)
    stream=direct_init(model;max_evaluations=15000,max_seconds=120.,record_limit=0)
    rec=E.MeasurementRecord(:core,0.,294.)
    result=E.assimilate!(stream,rec)
    @test result.committed
    if result.committed
        offline=direct_fit(model,[rec];max_evaluations=15000,max_seconds=120.,record_limit=0)
        @test offline.diagnostics.resolved
        @test offline.mean≈stream.posterior.mean atol=1e-10
        @test offline.covariance≈stream.posterior.covariance atol=1e-10
    end
    before=deepcopy(stream)
    stream.settings=merge(stream.settings,(max_evaluations=1,))
    rejected=E.assimilate!(stream,E.MeasurementRecord(:core,10.,294.3))
    @test !rejected.committed
    @test stream.posterior.mean==before.posterior.mean
    @test stream.anchor.graph.values==before.anchor.graph.values
    @test length(stream.anchor.obs)==length(before.anchor.obs)
    @test_throws ArgumentError E.assimilate!(stream,E.MeasurementRecord(:core,-1.,294.))
    @test_throws ArgumentError direct_init(model;lag=0)
end

@testset "analytic correlated-noise predictive distribution" begin
    model=consistency_model(nx=4,stop=120.,suite=:pit)
    sm=direct_init(model;max_evaluations=15000,max_seconds=120.,record_limit=0)
    key=first(sm.anchor.nodes)[2]
    prior=only(f for f in E.factors_of(sm.anchor.graph,key) if f isa E.GaussianPrior)
    i=first(E.belief_index(model.bl,:pit,:T_pit))
    initial_mean=prior.m[i];state_variance=sum(abs2,inv(prior.R)[i,:])
    noise=E.sensor(model.log,:pit).noise
    persistent_variance=state_variance+noise.ar_sd^2
    observation=initial_mean+1.3sqrt(persistent_variance+noise.white_sd^2)
    predicted=direct_predictive_cdf(sm.posterior,:pit,observation)
    @test predicted.resolved
    @test predicted.value≈.5E.erfc(-1.3/sqrt(2)) atol=3e-5
    result=E.assimilate!(sm,E.MeasurementRecord(:pit,0.,observation))
    @test result.committed
    if result.committed
        posterior_mean=initial_mean+persistent_variance/(persistent_variance+noise.white_sd^2)*(observation-initial_mean)
        posterior_variance=persistent_variance*noise.white_sd^2/(persistent_variance+noise.white_sd^2)+noise.white_sd^2
        predicted=direct_predictive_cdf(sm.posterior,:pit,posterior_mean+.6sqrt(posterior_variance))
        @test predicted.resolved
        @test predicted.value≈.5E.erfc(-.6/sqrt(2)) atol=3e-5
        before=deepcopy(sm.anchor)
        forecast=direct_forecast(sm,0.)
        @test forecast.diagnostics.resolved
        @test forecast.mean≈sm.posterior.mean atol=1e-10
        @test sm.anchor.graph.values==before.graph.values
        @test length(sm.anchor.obs)==length(before.obs)
        @test_throws ArgumentError direct_forecast(sm,-1.)
    end
end

@testset "unresolved densities cannot be consumed as normalized" begin
    model=consistency_model(nx=4,stop=120.,suite=:core)
    anchor=E.online_init(model;lag=typemax(Int),record_limit=0)
    p=direct_posterior_fit(anchor;max_evaluations=1)
    @test !p.diagnostics.resolved
    @test_throws ArgumentError direct_logpdf(p,p.center)
    @test_throws ArgumentError direct_radius(p,p.mean)
    @test isfinite(direct_logpdf(p,p.center;allow_unresolved=true))
end