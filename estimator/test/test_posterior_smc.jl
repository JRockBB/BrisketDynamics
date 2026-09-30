using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
include(joinpath(@__DIR__,"..","src","BrisketEstimator.jl"))
using .BrisketEstimator,BrisketDynamics,LinearAlgebra,Random,Statistics,Test
const E=BrisketEstimator
include(joinpath(@__DIR__,"..","scripts","consistency_control.jl"))

# Closed-form targets exercise the production bridge and rejuvenation kernel independently of the
# cooking model and of the transport's own implementation of the factor-graph likelihood.
struct PosteriorKernelOracle
    n::Int
    center::Vector{Float64}
    V::Matrix{Float64}
    a::Vector{Float64}
    zm::Vector{Float64}
    model::NamedTuple
    bimodal::Bool
end
function E._transport_draw(ref::PosteriorKernelOracle,u)
    if ref.bimodal
        a=-(u[1]-3)^2/(2*.5^2);b=-(u[1]+3)^2/(2*.5^2);hi=max(a,b)
        potential=-hi-log(exp(a-hi)+exp(b-hi))
    else
        potential=(u[1]-2)^2/(2*.4)
    end
    return (potential=potential,draw=copy(u))
end

@testset "posterior kernel analytic oracles" begin
    for bimodal in (false,true)
        ref=PosteriorKernelOracle(1,[0.0],zeros(1,0),Float64[],Float64[],(bl=(n=1,),),bimodal)
        result=E._posterior_particle_run(ref;n=4096,moves=4,rng=MersenneTwister(bimodal ? 701 : 702),maxstages=100)
        x=vec(result.draws)
        @test last(result.stages).beta==1
        if bimodal
            @test abs(count(>(0),x)/length(x)-.5)<.06
            @test abs(mean(x))<.25
            @test abs(var(x)-9.25)<.75
        else
            @test abs(mean(x)-2)<.06
            @test abs(var(x)/.4-1)<.08
        end
    end
end

struct PosteriorBlockOracle
    n::Int
    center::Vector{Float64}
    V::Matrix{Float64}
    a::Vector{Float64}
    zm::Vector{Float64}
    model::NamedTuple
    kind::Symbol
end
function E._transport_draw(ref::PosteriorBlockOracle,u)
    potential=ref.kind===:gaussian ? (sum(abs2,u)+(dot([5.,-2.,1.],u)-3)^2)/2 :
        u[1]^2/2+(u[2]-.4*(u[1]^2-1))^2/(2*.2^2)
    return (potential=potential,draw=copy(u))
end

@testset "posterior weak block preserves joint targets" begin
    # A rotated correlated Gaussian has a known independent weak direction. The curved target then
    # checks a direction that appears uninformed to a local Jacobian but is not independent globally.
    H=[5.,-2.,1.];C=inv(Matrix{Float64}(I,3,3)+H*H');mu=C*H*3
    ref=PosteriorBlockOracle(3,mu,reshape(H/norm(H),3,1),[1/sqrt(31.)],[0.],(bl=(n=2,),),:gaussian)
    result=E._posterior_particle_run(ref;n=8192,moves=4,rng=MersenneTwister(721),maxstages=100)
    @test norm(vec(mean(result.draws;dims=2))-mu)<.08
    @test norm(cov(result.draws;dims=2)-C)<.1
    weak=vec([0.,1.,2.]'*result.draws)/sqrt(5)
    @test abs(mean(weak))<.06
    @test abs(var(weak)-1)<.1
    ref=PosteriorBlockOracle(2,[0.,-.4],reshape([0.,1.],2,1),[.2],Float64[],(bl=(n=2,),),:banana)
    result=E._posterior_particle_run(ref;n=8192,moves=4,rng=MersenneTwister(722),maxstages=100)
    @test maximum(abs.(vec(mean(result.draws;dims=2))))<.1
    @test abs(var(result.draws[1,:])-1)<.12
    @test abs(var(result.draws[2,:])-.36)<.12
    @test abs(mean(result.draws[1,:].^2 .* result.draws[2,:])-.8)<.25
end

@testset "tempered posterior particles" begin
    model=consistency_model(nx=4,suite=:pit,stop=600.0)
    sm=online_init(model;lag=typemax(Int),record_limit=0)
    @test_throws ArgumentError sample_posterior_smc(sm;replicates=1)
    @test_throws ArgumentError sample_posterior_smc(sm;particles=1)
    @test_throws ArgumentError sample_posterior_smc(sm;moves=0)
    small=sample_posterior_smc(sm;particles=32,replicates=4,moves=1,rng=MersenneTwister(913))
    repeated=sample_posterior_smc(sm;particles=32,replicates=4,moves=1,rng=MersenneTwister(913))
    @test small.draws==repeated.draws
    @test !small.diagnostics.mc_precision_ok
    @test_throws ArgumentError belief_at(small)
    @test small.diagnostics.ess_kind===:replicate_mean
    @test all(last(stages).beta==1 for stages in small.diagnostics.stages)

    # One linear pit reading has an analytic marginal after integrating the stationary discrepancy.
    ky=sm.nodes[1][2];ip=first(E.belief_index(model.bl,:pit,:T_pit))
    prior=only(f for f in factors_of(sm.graph,ky) if f isa GaussianPrior)
    mean0=prior.m[ip];C0=inv(prior.R'*prior.R);v0=C0[ip,ip]
    sensor=model.log.sensors[1];R=sensor.noise.ar_sd^2+sensor.noise.white_sd^2
    reading=mean0+2.0;expected_mean=mean0+v0/(v0+R)*2.0;expected_var=v0*R/(v0+R)
    result=assimilate!(sm,MeasurementRecord(:pit,0.0,reading));@test result.committed
    original=deepcopy(sm.graph.values)
    p=sample_posterior_smc(sm;particles=2048,replicates=4,moves=2,rng=MersenneTwister(919))
    @test sm.graph.values==original
    @test isempty(p.diagnostics.numerical_failures)
    i=length(model.blk)+ip
    @test abs(p.mean[i]-expected_mean)<6sqrt(expected_var/p.diagnostics.n)
    @test abs(p.covariance[i,i]/expected_var-1)<.08
    @test p.diagnostics.ess<=p.diagnostics.n
    @test minimum(p.diagnostics.mean_mcse)>0
    @test all(x->0<x<=p.diagnostics.n,p.diagnostics.tail_ess)
    depth=E.latent_index(model.blk,:core_depth_fraction)
    @test abs(p.mean[depth])<6/sqrt(p.diagnostics.n)
    @test abs(p.covariance[depth,depth]-1)<.08
    @test abs(p.covariance[depth,i])<6sqrt(expected_var/p.diagnostics.n)
    @test posterior_quantile(p)[i,1]<expected_mean<posterior_quantile(p)[i,3]

    # Repeated readings of one probe share its OU discrepancy. More white-noise samples must not
    # falsely divide the systematic variance by the number of readings.
    repeated_sm=online_init(model;lag=typemax(Int),record_limit=0)
    records=fill(MeasurementRecord(:pit,0.0,reading),8)
    @test assimilate!(repeated_sm,records).committed
    repeated_p=sample_posterior_smc(repeated_sm;particles=2048,replicates=4,moves=1,rng=MersenneTwister(921))
    effective_R=sensor.noise.ar_sd^2+sensor.noise.white_sd^2/length(records)
    repeated_var=v0*effective_R/(v0+effective_R)
    @test abs(repeated_p.covariance[i,i]/repeated_var-1)<.08
    @test repeated_p.covariance[i,i]>.8expected_var
end
