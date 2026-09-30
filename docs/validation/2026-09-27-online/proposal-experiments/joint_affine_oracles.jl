Core.eval(E,:(include($(joinpath(@__DIR__,"../affine_candidate.jl")))))
include(joinpath(@__DIR__,"../../../estimator/scripts/consistency_control.jl"))
@testset "joint affine proposal change of variables" begin
    model=consistency_model(nx=4,suite=:pit,stop=600.)
    sm=online_init(model;lag=typemax(Int));old=E._posterior_transport(sm)
    k=last(old.ks);prior=old.generators[k];nz=length(old.zm)
    ip=first(E.belief_index(model.bl,:pit,:T_pit));C=old.roots[k]*old.roots[k]'
    spec=only(model.log.sensors);R=spec.noise.ar_sd^2+spec.noise.white_sd^2
    physical_mean=belief_decode(prior.m,model.bl)[ip];value=physical_mean+.7
    E._append_sampled_measurements!(sm,[MeasurementRecord(:pit,0.,value)])
    ref=E._posterior_transport(sm);rng=MersenneTwister(5201)
    logZ=-log1p(C[ip,ip]/R)/2-(value-physical_mean)^2/(2*(R+C[ip,ip]))
    for i in 1:8
        u=randn(rng,old.n);result=E._affine_transport_extend(old,ref,u,Float64[])
        @test result.log_increment≈logZ atol=1e-7
        recovered=old.center+E._reference_root(old,E._reference_invroot(ref,result.coordinates-ref.center))
        @test recovered≈u atol=1e-10
        @test result.draw≈E._transport_draw(ref,result.coordinates).draw atol=1e-10
    end
    # A nonlinear observation system exercises the exact density correction and new dimensions.
    model=consistency_model(nx=4,suite=:full,stop=600.)
    sm=online_init(model;lag=typemax(Int));old=E._posterior_transport(sm)
    E._append_sampled_measurements!(sm,[MeasurementRecord(:pit,0.,294.),MeasurementRecord(:core,0.,281.),MeasurementRecord(:mass,60.,2.5)])
    ref=E._posterior_transport(sm);u=.1randn(MersenneTwister(5202),ref.n)
    result=E._affine_transport_extend(old,ref,u,Float64[])
    J=ForwardDiff.jacobian(v->E._affine_coordinates(old,ref,v),u)
    logjac,sign=logabsdet(J)
    @test sign>0
    @test logjac≈sum(log,ref.a;init=0.)-sum(log,old.a;init=0.) atol=1e-9
    @test result.log_increment≈sum(abs2,u)/2-E._transport_draw(ref,result.coordinates).potential+logjac atol=1e-8
end
