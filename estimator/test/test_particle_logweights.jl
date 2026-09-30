using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
include(joinpath(@__DIR__,"..","src","BrisketEstimator.jl"))
using .BrisketEstimator,BrisketDynamics,LinearAlgebra,Random,Statistics,Test
const E=BrisketEstimator
include(joinpath(@__DIR__,"..","scripts","consistency_control.jl"))

@testset "log weights preserve temporarily negligible hypotheses" begin
    w,logs=E._normalize_particle_logweights(repeat([-1000.,0.],4),4)
    @test all(iszero,w[1:2:end])
    @test all(isfinite,logs)
    @test sum(w)≈1
    @test logs[1:2:end]≈fill(-1000-log(4),4)
    recovered,nextlogs=E._normalize_particle_logweights(logs+repeat([0.,-1200.],4),4)
    @test recovered[1:2:end]≈fill(.25,4)
    @test recovered[2:2:end]≈fill(.25exp(-200),4) rtol=1e-12
    @test nextlogs[1:2:end]≈fill(-log(4),4) atol=1e-12
    @test nextlogs[2:2:end]≈fill(-200-log(4),4) atol=1e-12
    # The former round trip through display weights provably gives the opposite answer.
    lost,_=E._normalize_particle_logweights(log.(w)+repeat([0.,-1200.],4),4)
    @test all(iszero,lost[1:2:end])
    @test lost[2:2:end]≈fill(.25,4)
    huge,huge_logs=E._normalize_particle_logweights(fill(-1e300,8),4)
    @test huge≈fill(.125,8)
    @test huge_logs≈fill(-log(8),8)
    hard_zero,hard_logs=E._normalize_particle_logweights(repeat([-Inf,0.],4),4)
    @test all(iszero,hard_zero[1:2:end])
    @test all(==(-Inf),hard_logs[1:2:end])
    @test_throws ArgumentError E._normalize_particle_logweights(Float64[],4)
    @test_throws ArgumentError E._normalize_particle_logweights(zeros(8),0)
    @test_throws ArgumentError E._normalize_particle_logweights(zeros(9),4)
    @test_throws ArgumentError E._normalize_particle_logweights(fill(-Inf,8),4)
    @test_throws ArgumentError E._normalize_particle_logweights(fill(Inf,8),4)
    @test_throws ArgumentError E._normalize_particle_logweights(fill(NaN,8),4)
end
@testset "continuation checkpoints and rollback preserve log weights" begin
    model=consistency_model(nx=4,suite=:pit,stop=600.)
    sm=sampled_init(model;rng=MersenneTwister(5601))
    @test sm.log_weights==log.(sm.posterior.weights)
    @test assimilate!(sm,MeasurementRecord(:pit,0.,294.2)).committed
    @test exp.(sm.log_weights)≈sm.posterior.weights
    checkpoint=smoother_checkpoint(sm);resumed=restore_checkpoint(checkpoint)
    @test resumed.log_weights==sm.log_weights
    @test resumed.log_weights!==sm.log_weights
    a=assimilate!(sm,MeasurementRecord(:pit,0.,294.3))
    b=assimilate!(resumed,MeasurementRecord(:pit,0.,294.3))
    @test a.committed && b.committed
    @test sm.log_weights==resumed.log_weights
    @test sm.posterior.weights==resumed.posterior.weights
    saved=copy(sm.log_weights);posterior=sm.posterior;values=deepcopy(sm.graph_backend.graph.values)
    sm.particles=2;sm.max_particles=2
    @test !assimilate!(sm,MeasurementRecord(:pit,0.,294.2);refresh=true).committed
    @test sm.log_weights==saved && sm.posterior===posterior
    @test_throws Exception assimilate!(sm,[MeasurementRecord(:pit,0.,294.2),MeasurementRecord(:pit,1200.,294.2)])
    @test sm.log_weights==saved && sm.graph_backend.graph.values==values
    sm.replicates=4
    @test_throws ArgumentError assimilate!(sm,MeasurementRecord(:pit,0.,294.2))
    @test sm.log_weights==saved && sm.posterior===posterior
end

@testset "actual sequential likelihood recovers underflowed hypotheses" begin
    model=consistency_model(nx=4,suite=:pit,stop=600.)
    sm=sampled_init(model;rng=MersenneTwister(5602))
    ref=sm.posterior.history.reference;ky=last(ref.ks);nz=length(model.blk)
    ip=first(E.belief_index(model.bl,:pit,:T_pit));prior=ref.generators[ky]
    spec=only(model.log.sensors);R=spec.noise.ar_sd^2+spec.noise.white_sd^2
    center=prior.m[ip];distance=50sqrt(R);n=length(sm.posterior.weights)
    X=copy(sm.posterior.draws);U=copy(sm.posterior.history.coordinates)
    # A deliberately bimodal discrete prior over pit temperature has an analytic update. Other
    # coordinates retain their draws, so the production joint diagnostics also see a full-rank cloud.
    for j in 1:n
        X[nz+ip,j]=center+(isodd(j) ? -distance : distance)
        U[ref.off[ky],j]=ref.roots[ky]\(X[nz+1:end,j]-prior.m)
    end
    sm.posterior=E._replicate_posterior(ref,X,fill(1/n,n),sm.replicates;
        history=E.PosteriorHistory(ref,U))
    for reading in (center+distance,center-2distance)
        E._append_sampled_measurements!(sm.graph_backend,[MeasurementRecord(:pit,0.,reading)])
        nextref=E._posterior_transport(sm.graph_backend)
        p,predictions,logs=E._incremental_posterior(sm,nextref)
        sm.posterior=p;sm.predictions=predictions;sm.log_weights=logs
        if reading>center
            @test all(iszero,p.weights[1:2:end])
            @test all(isfinite,logs)
        end
    end
    # With two simultaneous readings, the shared OU error remains: only white noise halves.
    average_reading=center-distance/2
    effective_R=spec.noise.ar_sd^2+spec.noise.white_sd^2/2
    # Positive temperature coordinates decode to y + 1e-8 in this (>30) branch.
    lower=-(center-distance+1e-8-average_reading)^2/(2effective_R)
    upper=-(center+distance+1e-8-average_reading)^2/(2effective_R)
    odds=exp(upper-lower)
    @test sm.posterior.weights[1:2:end]≈fill(2/n/(1+odds),n÷2)
    @test sm.posterior.weights[2:2:end]≈fill(2/n*odds/(1+odds),n÷2)
    @test sm.log_weights[2:2:end]-sm.log_weights[1:2:end]≈fill(upper-lower,n÷2) rtol=1e-12
    @test all(isfinite,sm.log_weights)
end
