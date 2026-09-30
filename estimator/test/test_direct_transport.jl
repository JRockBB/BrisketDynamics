using Test, LinearAlgebra, SpecialFunctions
include(joinpath(@__DIR__,"../src/direct_quadrature.jl"))
using .DirectQuadrature
include(joinpath(@__DIR__,"../scripts/direct_transport.jl"))

@testset "full-target nonlinear map and determinant" begin
    coefficients=reshape(.03sin.(collect(1:54)),9,6)
    chart=DirectConditionalChart(zeros(5),[3,1],[2,4,5],[1. 0.;.3 .8],
        [.1 .2;-.2 .1;.3 -.1],[.5 0. 0.;.1 1. 0.;.2 .1 2.],coefficients)
    ref=(zm=zeros(3),zs=ones(3))
    for q in ([.2,-.3,.1,.4,-.6],[-2.,1.,.5,.7,.8])
        mapped=direct_map(chart,q);J=zeros(5,5)
        for j in 1:5
            delta=zeros(5);delta[j]=1e-5
            J[:,j]=(direct_map(chart,q+delta).u-direct_map(chart,q-delta).u)/(2e-5)
        end
        @test logabsdet(J)[1]≈mapped.logjac atol=1e-8
        for index in 1:3
            conditional=direct_conditioning(chart,ref,index,q)
            @test conditional.rotation'*conditional.rotation≈I atol=1e-12
            w=copy(q);w[conditional.target]=0.
            origin=direct_map(chart,conditional.rotation*w).u[index]
            for t in (-.7,.8)
                w[conditional.target]=t
                @test direct_map(chart,conditional.rotation*w).u[index]≈origin+conditional.width*t atol=1e-12
            end
        end
    end
end

@testset "exact target correction removes nonlinear proposal bias" begin
    coefficients=zeros(5,3) # two means, then lower triangular log-root
    coefficients[1,3]=.4;coefficients[2,2]=.2;coefficients[5,2]=.05
    chart=DirectConditionalChart(zeros(3),[1],[2,3],ones(1,1),zeros(2,1),Matrix{Float64}(I,2,2),coefficients)
    function integrand(q)
        mapped=direct_map(chart,q);u=mapped.u
        ratio=exp((sum(abs2,q)-sum(abs2,u))/2+mapped.logjac)
        ratio.*vcat(1.,u,u.^2,u[1]*u[2],u[2]*u[3])
    end
    result=integrate_normal(integrand,3;atol=2e-6,rtol=2e-6,max_evaluations=50000,max_level=7)
    @test result.diagnostics.resolved
    @test result.value≈[1.,0.,0.,0.,1.,1.,1.,0.,0.] atol=2e-5
    # A parameter CDF conditioned in the proposal must include the full map's
    # Jacobian and target correction, including its variable conditional scale.
    ref=(zm=zeros(3),zs=ones(3));threshold=.25
    function probability(q)
        c=direct_conditioning(chart,ref,2,q);w=copy(q);w[c.target]=0.
        u0=direct_map(chart,c.rotation*w).u[2]
        mass=erfc(-(threshold-u0)/c.width/sqrt(2))/2
        mass==0 && return [0.]
        w[c.target]=-sqrt(2)*erfcinv(mass*erfc(-q[c.target]/sqrt(2)))
        v=c.rotation*w;m=direct_map(chart,v)
        [mass*exp((sum(abs2,v)-sum(abs2,m.u))/2+m.logjac)]
    end
    cdf=integrate_normal(probability,3;atol=2e-6,rtol=2e-6,max_evaluations=50000,max_level=7)
    @test cdf.diagnostics.resolved
    @test only(cdf.value)≈erfc(-threshold/sqrt(2))/2 atol=1e-5
end

@testset "curvature coordinates preserve every direction" begin
    design(q)=[(q[1]+2q[2])^2,.1q[3]^2]
    r=direct_curvature_rotation(design,3,3)
    @test r.rotation'*r.rotation≈I atol=1e-12
    @test abs(dot(r.rotation[:,1],[1.,2.,0.]/sqrt(5)))≈1 atol=1e-10
    @test r.diagnostics.dimension_dropped==0
    @test r.diagnostics.dimension_retained==3
end