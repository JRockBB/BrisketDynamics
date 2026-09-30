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
    lower=-(center-distance-average_reading)^2/(2effective_R)
    upper=-(center+distance-average_reading)^2/(2effective_R)
    odds=exp(upper-lower)
    @test sm.posterior.weights[1:2:end]≈fill(2/n/(1+odds),n÷2)
    @test sm.posterior.weights[2:2:end]≈fill(2/n*odds/(1+odds),n÷2)
    @test sm.log_weights[2:2:end]-sm.log_weights[1:2:end]≈fill(upper-lower,n÷2) rtol=1e-12
    @test all(isfinite,sm.log_weights)
end
