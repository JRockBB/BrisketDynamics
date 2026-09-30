using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
include(joinpath(@__DIR__,"..","src","BrisketEstimator.jl"))
using .BrisketEstimator,BrisketDynamics,LinearAlgebra,Random,Statistics,Test,ForwardDiff
const E=BrisketEstimator
include(joinpath(@__DIR__,"..","scripts","consistency_control.jl"))

# A known finite distribution is an independent oracle for nonlinear averaging and weighted losses.
# It is deliberately not represented by a Gaussian, and is not a claim about Monte Carlo precision.
function discrete_history_fixture()
    model=consistency_model(nx=4,suite=:core,stop=1800.)
    sm=online_init(model;lag=typemax(Int));E._node_at!(sm,600.)
    ref=E._posterior_transport(sm);nz=length(model.blk)
    U=zeros(ref.n,4);U[1,:]=[-1.,-.25,.5,1.2]
    it=first(E.belief_index(model.bl,:brisket,:T));fuel=first(E.belief_index(model.bl,:pit,:m_fuel))
    U[nz+it,:]=[-1.,0.,1.,2.];U[nz+fuel,:]=[-.5,0.,.5,1.]
    X=hcat([E._transport_draw(ref,U[:,j]).draw for j in 1:4]...)
    physical=hcat([belief_decode(X[nz+1:end,j],model.bl) for j in 1:4]...)
    w=[.1,.2,.3,.4];mu=X*w;delta=X.-mu;C=(delta.*w')*delta'
    return SampledPosterior(600.,model,X,physical,w,mu,C,(mc_precision_ok=true,known_discrete_fixture=true),E.PosteriorHistory(ref,U))
end

@testset "joint sampled histories reach prediction and weighted planning" begin
    p=discrete_history_fixture();goals=Dict(:brisket=>PlanGoals(w_lethality=0.))
    snap=belief_snapshot(p;goals,history_dt=60.)
    @test snap isa SampledBeliefSnapshot
    @test length(snap.scenarios)==4
    @test [s.weight for s in snap.scenarios]==p.weights
    @test !snap.provenance.subsampled
    @test belief_plan_draws(snap)===snap.scenarios
    nz=length(p.model.blk)
    for (j,scenario) in enumerate(snap.scenarios)
        @test scenario.snapshot.z==p.draws[1:nz,j]
        @test scenario.snapshot.y==p.draws[nz+1:end,j]
        @test scenario.snapshot.past[:brisket].t==p.t
    end
    @test maximum(s.snapshot.past[:brisket].leth_c for s in snap.scenarios)>
          minimum(s.snapshot.past[:brisket].leth_c for s in snap.scenarios)
    @test maximum(s.snapshot.past[:brisket].m0 for s in snap.scenarios)>
          minimum(s.snapshot.past[:brisket].m0 for s in snap.scenarios)
    fine=belief_snapshot(p;goals,history_dt=30.)
    @test maximum(abs(a.snapshot.past[:brisket].leth_c/b.snapshot.past[:brisket].leth_c-1)
        for (a,b) in zip(snap.scenarios,fine.scenarios))<.02
    reduced=belief_snapshot(p;goals,n=2,rng=MersenneTwister(451))
    @test reduced.provenance.subsampled
    @test sum(s.weight for s in reduced.scenarios)≈1
    @test all(s.snapshot.y==p.draws[nz+1:end,s.source_index] for s in reduced.scenarios)
    @test_throws ArgumentError belief_snapshot(p;n=1)
    @test_throws ArgumentError belief_snapshot(p;history_dt=0.)

    schedule=SystemSchedule(collect(0.:600.:108000.),[:brisket];u0=.5)
    schedule.t_now=600.;schedule.u[1:2].=.5;schedule.t_pull[:brisket]=600.
    free=system_free_set(schedule);zeta,_=system_plan_prior(free,schedule)
    # Differentiate within an interpolation cell: the derivative is discontinuous at a grid knot.
    zeta .+= .013
    plan=decode_system_plan(zeta,free,schedule)
    @test minimum(abs.(schedule.t.-plan.t_serve))>20
    @test encode_system_plan(plan,free,schedule)≈zeta atol=1e-12
    @test length(free)==1
    manual=[system_plan_outputs(s.snapshot,s.snapshot.z,zeta,free,schedule,goals;y=s.snapshot.y)[1][:brisket] for s in snap.scenarios]
    prediction=system_plan_predictive(snap,zeta,free,schedule,goals)[:brisket]
    Y=hcat(manual...);mu=Y*p.weights
    @test all(isfinite,Y)
    @test prediction.mean≈mu
    @test prediction.sd≈sqrt.(vec(sum((Y.-mu).^2 .* p.weights';dims=2)))
    @test prediction.quantiles[1,:]==E._weighted_quantiles(Y[1,:],p.weights,[.025,.5,.975])
    @test !prediction.includes_future_process_noise
    expected=[.5sum(abs2,system_goal_residuals(system_plan_outputs(s.snapshot,s.snapshot.z,zeta,free,schedule,goals;y=s.snapshot.y)[1],
        goals,[:brisket])) for s in snap.scenarios]
    loss=system_plan_loss(snap,plan,schedule,goals)
    @test loss.mean≈dot(p.weights,expected) rtol=1e-6
    @test_throws ArgumentError system_plan_predictive(snap,zeta,free,schedule,Dict(:brisket=>PlanGoals(leth_z=7.)))

    nout=length(E._plan_residual(snap,first(snap.scenarios),zeta,free,schedule,goals,SystemControlGoals(),1,4,true))
    key=VarKey(:plan,:sample_test,0)
    factors=[E._planning_factor(snap,d,key,free,schedule,goals,SystemControlGoals(),k,4,true,nout) for (k,d) in enumerate(snap.scenarios)]
    @test sum(.5sum(abs2,E.residual(f,[zeta])) for f in factors)≈dot(p.weights,expected)
    J=E.jacobians(first(factors),[zeta])[1];eps_fd=1e-3
    fd=(E.residual(first(factors),[zeta.+eps_fd])-E.residual(first(factors),[zeta.-eps_fd]))/(2eps_fd)
    @test norm(J[:,1]-fd)<.02max(norm(fd),1e-5)
    result=replan_system(snap,schedule,goals;scan_wrap=(),scan_rest=(),scan_damper=false,cold_start=false,iters=40)
    @test result.converged
    @test result.F_goals≈system_plan_loss(snap,result.plan,schedule,goals).mean rtol=1e-5
end

@testset "plan encoding preserves interior decisions and committed pulls" begin
    schedule=SystemSchedule(collect(0.:600.:108000.),[:brisket];u0=.5)
    schedule.t_now=600.;schedule.u[1:2].=.5
    free=system_free_set(schedule);zeta,_=system_plan_prior(free,schedule)
    # A short remaining cook and fractions outside the prior's 2% initialization bounds.
    zeta[1]=log(.005/.995);zeta[N_KNOTS+1]=log(.995/.005)
    zeta[N_KNOTS+2]=log(.001/.999)
    plan=decode_system_plan(zeta,free,schedule)
    @test 0<plan.pieces[:brisket].t_pull-schedule.t_now<600
    @test encode_system_plan(plan,free,schedule)≈zeta atol=1e-10
    for pull in (300.,600.,900.)
        schedule.t_pull[:brisket]=pull
        free=system_free_set(schedule);zeta,_=system_plan_prior(free,schedule)
        plan=decode_system_plan(zeta,free,schedule)
        @test encode_system_plan(plan,free,schedule)≈zeta atol=1e-10
    end
end

@testset "posterior forecasts preserve draws and add declared process uncertainty" begin
    p=discrete_history_fixture();nz=length(p.model.blk)
    forecast=posterior_predict(p,[660.,720.];process_noise=false)
    @test forecast.weights==p.weights
    @test !forecast.provenance.includes_process_noise
    @test forecast.draws[1][1:nz,:]==p.draws[1:nz,:]
    for j in 1:4
        y=E.propagate(p.model,p.draws[nz+1:end,j],p.draws[1:nz,j],600.,660.)[1]
        @test forecast.draws[1][nz+1:end,j]≈y atol=1e-9
    end
    @test size(first(posterior_quantile(forecast)))==(p.model.bl.n,3)
    @test_throws ArgumentError posterior_predict(p,[600.])
    @test_throws ArgumentError posterior_predict(p,[660.,650.])
    @test_throws ArgumentError posterior_quantile(forecast,[-.1])
    # A forecast can use newly chosen controls without replacing the accepted historical model.
    log=E.with_execution(p.model.log;u=fill(.9,length(p.model.log.u)))
    controlled=SystemModel(p.model.sys,log,p.model.blk;sn=p.model.sn)
    changed=posterior_predict(p,[660.];model=controlled,process_noise=false)
    @test changed.model===controlled
    @test p.history.reference.model===p.model
    @test changed.draws[1][1:nz,:]==p.draws[1:nz,:]
    @test changed.draws[1][nz+1:end,1]≈E.propagate(controlled,p.draws[nz+1:end,1],p.draws[1:nz,1],600.,660.)[1]
    @test norm(changed.draws[1]-forecast.draws[1])>1e-6
    # A known point posterior has zero source uncertainty. Its one-step forecast must still have
    # the declared process covariance, in every whitened state direction.
    N=4096;X=repeat(p.draws[:,1:1],1,N);physical=repeat(p.physical[:,1:1],1,N)
    point=SampledPosterior(p.t,p.model,X,physical,fill(1/N,N),X[:,1],zeros(size(X,1),size(X,1)),
        (mc_precision_ok=true,known_point_fixture=true),E.PosteriorHistory(p.history.reference,repeat(p.history.coordinates[:,1:1],1,N)))
    stochastic=posterior_predict(point,[601.];rng=MersenneTwister(461))
    deterministic=E.propagate(p.model,X[nz+1:end,1],X[1:nz,1],600.,601.)[1]
    ref=p.history.reference
    factor=DynamicsFactor(p.model,ref.kz,VarKey(:state,:oracle,1),VarKey(:state,:oracle,2),600.,601.;first=false,
        events=E.sensor_events(p.model.log,600.,601.))
    cache=E._dyn_eval!(factor,ref.g.values[ref.kz],ref.g.values[last(ref.ks)];jac=false)
    whitened=cache.W*(stochastic.draws[1][nz+1:end,:].-deterministic)
    @test maximum(abs.(vec(mean(whitened;dims=2))))<6/sqrt(N)
    @test maximum(abs.(cov(whitened;dims=2)-I))<.09
    @test stochastic.provenance.includes_process_noise
end
