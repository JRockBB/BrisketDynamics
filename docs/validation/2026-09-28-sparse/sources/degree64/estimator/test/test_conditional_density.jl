if !isdefined(Main,:BrisketEstimator)
    include(joinpath(@__DIR__,"..","src","BrisketEstimator.jl"))
end
using .BrisketEstimator, LinearAlgebra, Test
isdefined(Main,:E) || (const E=BrisketEstimator)

@testset "density discrepancy separates integration error from posterior error" begin
    k=VarKey(:params,:distance,0)
    function density(mean)
        g=FactorGraph();add_variable!(g,k,mean)
        add_factor!(g,GaussianPrior(k,mean,Matrix{Float64}(I,2,2)))
        p=E._conditional_rule(g,[(k,1)],[0.],ones(1,1),5)
        return conditional_density(p;center=[0.],scale=ones(1,1),order=5)
    end
    a=density([0.,0.]);b=density([0.,.4])
    result=conditional_density_distance(a,b;atol=1e-12)
    # Same axis marginal; the conditional KL is .4^2/2 everywhere.
    @test result.tv_bound_estimate≈.2 atol=1e-10
    @test result.integration_change<1e-10
    @test result.tv_bound_estimate>=2E._conditional_phi(.2)-1
    @test isinf(conditional_density_distance(a,b;orders=(5,)).integration_change)
end
@testset "continuous density: correlated Gaussian between and beyond nodes" begin
    k=VarKey(:params,:density,0);g=FactorGraph();add_variable!(g,k,zeros(3))
    A=[1.0 0 0;0 1 0;0 0 1;2 3 -1];b=[.3,-.8,1.2,2.0]
    add_factor!(g,LinearFactor([k],[A],b));gauss_newton!(g)
    C=inv(A'*A);mu=C*A'*b
    for selected in ([(k,1)],[(k,1),(k,3)])
        n=length(selected);center=zeros(n);scale=Matrix{Float64}(I,n,n)
        ps=[E._conditional_rule(g,selected,center,scale,o) for o in (3,5)]
        ds=[conditional_density(p;center,scale,order=o) for (p,o) in zip(ps,(3,5))]
        for d in ds
            @test d.normalization_error<1e-12
            @test conditional_density_distance(d,d).tv_bound_estimate<1e-10
            for x in ([0.,0.,0.],[.7,-.8,.9],[8.,-3.,2.])
                expected=-.5dot(x-mu,C\(x-mu))-.5logdet(C)-1.5log(2pi)
                @test conditional_logpdf(d,x)≈expected atol=1e-8
            end
            for i in 1:3, v in (-1.,0.,1.)
                @test conditional_cdf(d,k,i,v)≈E._conditional_phi((v-mu[i])/sqrt(C[i,i])) atol=1e-8
            end
        end
        @test conditional_density_distance(ds...).tv_bound_estimate<1e-9
        # Accumulated evidence may contain large constants and displayed masses
        # may underflow. The reconstruction must use the finite log weights.
        qs,lw=E._tensor_quadrature(5,n)
        logs=[l+.5sum(abs2,q)+c.log_integral for (l,q,c) in zip(lw,qs,ps[2].nodes)]
        shifted=conditional_density(ps[2];center,scale,order=5,logweights=logs.-10000)
        @test conditional_logpdf(shifted,mu)≈conditional_logpdf(ds[2],mu) atol=1e-8
        bad=deepcopy(ps[2]);c=bad.nodes[1]
        bad.nodes[1]=E.ConditionalNode(c.graph,c.coordinates,c.mean,c.root,c.log_integral,merge(c.solve,(converged=false,)))
        @test_throws ArgumentError conditional_density(bad;center,scale,order=5)
    end
end

struct DensityCurvedFactor <: Factor
    key::VarKey
end
E.vars(f::DensityCurvedFactor)=[f.key]
E.ndim(f::DensityCurvedFactor)=1
E.residual(f::DensityCurvedFactor,v)=[exp(v[1][1]/2)*(v[1][2]-v[1][1]^2)]
function E.jacobians(f::DensityCurvedFactor,v)
    a,y,b=v[1];s=exp(a/2)
    return [reshape([s*((y-a*a)/2-2a),s,0.0],1,3)]
end

@testset "continuous density retains curvature and heteroscedasticity" begin
    k=VarKey(:params,:curved_density,0);g=FactorGraph();add_variable!(g,k,[0.,0.,3.])
    add_factor!(g,GaussianPrior(k,[0.,0.,3.],[1.0 0 0;0 0 .5]))
    add_factor!(g,DensityCurvedFactor(k))
    p=E._conditional_rule(g,[(k,1)],[0.],ones(1,1),5)
    d=conditional_density(p;center=[0.],scale=ones(1,1),order=5)
    # Independent normalized target: a~N(-.5,1), y|a~N(a^2,exp(-a)), b~N(3,4).
    for a in (-1.1,-.23,.37,1.2), y in (-.4,.6,2.3), b in (1.,3.)
        expected=-.5*(a+.5)^2-.5exp(a)*(y-a*a)^2+.5a-.5*((b-3)/2)^2-log(2)-1.5log(2pi)
        @test conditional_logpdf(d,[a,y,b])≈expected atol=1e-8
        m,L=E._density_geometry(d,[a])
        @test isposdef(Symmetric(L*L'))
    end
    @test conditional_cdf(d,k,1,.2)≈E._conditional_phi(.7) atol=1e-8
    @test conditional_cdf(d,k,3,4.)≈E._conditional_phi(.5) atol=1e-8
end
