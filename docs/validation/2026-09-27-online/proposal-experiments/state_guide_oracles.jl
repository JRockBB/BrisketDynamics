include(joinpath(@__DIR__,"../../../estimator/scripts/consistency_control.jl"))

@testset "guided initial Gaussian conditional and exact marginal weight" begin
    model=consistency_model(nx=4,suite=:pit,stop=600.)
    sm=online_init(model;lag=typemax(Int));old=E._posterior_transport(sm)
    k=last(old.ks);prior=old.generators[k];nz=length(old.zm)
    ip=first(E.belief_index(model.bl,:pit,:T_pit));C=old.roots[k]*old.roots[k]'
    spec=only(model.log.sensors);R=spec.noise.ar_sd^2+spec.noise.white_sd^2
    value=prior.m[ip]+.7
    E._append_sampled_measurements!(sm,[MeasurementRecord(:pit,0.,value)])
    ref=E._posterior_transport(sm);rng=MersenneTwister(5101)
    physical_mean=belief_decode(prior.m,model.bl)[ip]
    expected_weight=-log1p(C[ip,ip]/R)/2-(value-physical_mean)^2/(2*(R+C[ip,ip]))
    endpoints=Float64[]
    for j in 1:2048
        eta=randn(rng,old.n);before=E._transport_draw(old,eta);u=copy(eta)
        result=E._guided_transport_extend(old,ref,u,before.draw,Float64[])
        j<=8 && @test result.log_increment≈expected_weight atol=1e-9
        if j==1
            global guide_example=(model=model,sm=sm,old=old,ref=ref,u=u,result=result,value=value,ip=ip,logZ=expected_weight)
            @test u[old.off[old.kz]]==eta[old.off[old.kz]]
            @test result.draw≈E._transport_draw(ref,u).draw atol=1e-10
        end
        push!(endpoints,result.draw[nz+ip])
    end
    v=C[ip,ip]*R/(C[ip,ip]+R);mu=prior.m[ip]+C[ip,ip]/(C[ip,ip]+R)*(value-physical_mean)
    @test abs(mean(endpoints)-mu)<6sqrt(v/length(endpoints))
    @test abs(var(endpoints)/v-1)<.1
end

@testset "defensive guide evaluates the complete mixture density" begin
    ex=guide_example;old=ex.old;ref=ex.ref;rng=MersenneTwister(5102);lambda=.2
    spec=only(ex.model.log.sensors);R=spec.noise.ar_sd^2+spec.noise.white_sd^2;nz=length(old.zm)
    for i in 1:16
        eta=randn(rng,old.n);before=E._transport_draw(old,eta);u=copy(eta)
        r=E._guided_transport_extend(old,ref,u,before.draw,Float64[];prior_weight=lambda,prior_choices=[isodd(i)])
        # For a Gaussian observation, q_conditional / q_prior = likelihood / marginal likelihood.
        logL=-(ex.value-belief_decode(r.draw[nz+1:end],ex.model.bl)[ex.ip])^2/(2R)
        a=log(lambda);b=log1p(-lambda)+logL-ex.logZ;hi=max(a,b)
        expected=logL-hi-log(exp(a-hi)+exp(b-hi))
        @test r.log_increment≈expected atol=1e-9
        @test exp(r.log_increment)<=inv(lambda)*(1+1e-12)
    end
end

@testset "guided transition conditions on retained OU residuals" begin
    ex=guide_example;model=ex.model;sm=ex.sm;old=ex.ref;nz=length(old.zm);ip=ex.ip
    value=301.;E._append_sampled_measurements!(sm,[MeasurementRecord(:pit,120.,value)])
    ref=E._posterior_transport(sm);k=last(ref.ks);factor=ref.generators[k]
    predicted=E.propagate(model,ex.result.draw[nz+1:end],ex.result.draw[1:nz],0.,120.;first_slice=factor.first)[1]
    Q=ref.roots[k]*ref.roots[k]';spec=only(model.log.sensors)
    v=spec.noise.ar_sd^2+spec.noise.white_sd^2;cross=exp(-120/spec.noise.tau)*spec.noise.ar_sd^2
    bias=cross/v*(ex.value-only(ex.result.predictions));R=v-cross^2/v
    expected=-log1p(Q[ip,ip]/R)/2-(value-belief_decode(predicted,model.bl)[ip]-bias)^2/(2*(R+Q[ip,ip]))
    for seed in 5111:5118
        u=vcat(ex.u,randn(MersenneTwister(seed),ref.n-old.n));before=copy(u[1:old.n])
        result=E._guided_transport_extend(old,ref,u,ex.result.draw,ex.result.predictions)
        @test result.log_increment≈expected atol=1e-8
        @test u[1:old.n]==before
        @test result.draw≈E._transport_draw(ref,u).draw atol=1e-9
    end
end

@testset "nonlinear guide uses the exact change of variables" begin
    model=consistency_model(nx=4,suite=:full,stop=600.)
    sm=online_init(model;lag=typemax(Int));old=E._posterior_transport(sm)
    k=last(old.ks);rng=MersenneTwister(5121);eta=.2randn(rng,old.n)
    before=E._transport_draw(old,eta)
    z=before.draw[1:length(old.zm)];y=before.draw[length(old.zm)+1:end]
    p,_,tk=E._point_params(model,z,0.);x=system_from_belief(belief_decode(y,model.bl),model.bl)
    readings=[MeasurementRecord(s.id,0.,compile_observable(s.observable,model.sys,model.bl,model.blk,model.log)(x,z,p,tk,0.)+s.noise.white_sd)
              for s in model.log.sensors]
    E._append_sampled_measurements!(sm,readings);ref=E._posterior_transport(sm)
    u=copy(eta);result=E._guided_transport_extend(old,ref,u,before.draw,Float64[])
    function mapping(v)
        candidate=copy(eta);candidate[old.off[k]]=v
        E._guided_transport_extend(old,ref,candidate,before.draw,Float64[])
        return candidate[old.off[k]]
    end
    v=eta[old.off[k]];h=1e-4;J=zeros(length(v),length(v))
    for i in eachindex(v)
        up=copy(v);down=copy(v);up[i]+=h;down[i]-=h
        J[:,i]=(mapping(up)-mapping(down))/(2h)
    end
    logjac,sign=logabsdet(J)
    @test sign>0
    @test result.log_increment≈-E._transport_draw(ref,u).potential+before.potential+logjac atol=1e-6
end

