if !isdefined(Main,:BrisketEstimator)
    include(joinpath(@__DIR__,"..","src","BrisketEstimator.jl"))
end
using .BrisketEstimator, LinearAlgebra, Test
isdefined(Main,:E) || (const E=BrisketEstimator)

# The y|a variance changes with a. Its determinant contributes to the marginal
# probability of a, which distinguishes true nested integration from profiling.
struct CurvedConditionalOracle <: Factor
    key::VarKey
end
E.vars(f::CurvedConditionalOracle)=[f.key]
E.ndim(f::CurvedConditionalOracle)=1
E.residual(f::CurvedConditionalOracle,v)=[exp(v[1][1]/2)*(v[1][2]-v[1][1]^2)]
function E.jacobians(f::CurvedConditionalOracle,v)
    a,y,_=v[1];s=exp(a/2)
    return [reshape([s*((y-a*a)/2-2a),s,0.0],1,3)]
end

@testset "conditional integration: curved joint density and volume" begin
    k=VarKey(:params,:oracle,0);g=FactorGraph()
    add_variable!(g,k,[0.0,0.0,3.0])
    add_factor!(g,LinearFactor([k],[[1.0 0 0;0 0 .5]],[0.0,1.5]))
    add_factor!(g,CurvedConditionalOracle(k))
    saved=deepcopy(g.values)
    p=conditional_fit(g;axes=[(k,1)],orders=(5,9,13),mean_tol=1e-5,covariance_tol=1e-5)
    @test g.values==saved
    @test p.diagnostics.resolved
    @test !p.diagnostics.conditional_gaussian_validated
    @test p.mean ≈ [-.5,1.25,3.0] atol=1e-9
    @test p.covariance ≈ [1.0 -1 0;-1 3+exp(1) 0;0 0 4] atol=1e-7
    @test all(w->w>0,p.weights)
    @test sum(p.weights)≈1.0
    @test all(c->c.root[1,:]==zeros(size(c.root,2)),p.nodes)
    @test all(c->isapprox(c.mean[2],c.mean[1]^2;atol=1e-10),p.nodes)
    @test maximum(c.root[2,:]'*c.root[2,:] |> tr for c in p.nodes)>100
    # Preserve a nonlinear joint decision, not just the marginal state widths.
    @test conditional_expectation(x->(x[2]-x[1]^2)^2,p)≈exp(1) atol=1e-7
    @test conditional_expectation(x->x[1]*x[2],p)≈-1.625 atol=1e-7
    q=conditional_points(p)
    @test q.points*q.weights≈p.mean atol=1e-9
    d=q.points.-p.mean
    @test (d.*q.weights')*d'≈p.covariance atol=1e-8
    @test_throws ArgumentError conditional_fit(g;axes=[(k,1)],orders=(5,))
    @test_throws ArgumentError E._condition_graph(g,[(k,1),(k,1)],[0.0,1.0])
    @test_throws ArgumentError E._tensor_quadrature(9,4;max_nodes=100)
end

@testset "conditional integration: general linear graph and tiny contrasts" begin
    k=VarKey(:params,:linear,0);g=FactorGraph()
    A=[1.0 0 0;0 1 0;0 0 1;2 3 -1]
    b=[.3,-.8,1.2,2.0]
    add_variable!(g,k,zeros(3));add_factor!(g,LinearFactor([k],[A],b))
    C=inv(A'*A);m=C*A'*b
    for axes in ([(k,1)],[(k,2),(k,3)])
        p=conditional_fit(g;axes,orders=(3,5),mean_tol=1e-9,covariance_tol=1e-9)
        @test p.diagnostics.resolved
        @test p.mean≈m atol=1e-12
        @test p.covariance≈C atol=1e-12
        for c in p.nodes
            @test c.solve.converged
        end
    end
    # Covariance summaries can round away a contrast; the resolution check uses
    # the conditional square roots, so a linear graph still agrees to precision.
    tiny=FactorGraph();add_variable!(tiny,k,zeros(3))
    T=[1.0 0 0;0 1 0;0 -1e8 1e8]
    add_factor!(tiny,LinearFactor([k],[T],zeros(3)))
    p=conditional_fit(tiny;axes=[(k,1)],orders=(3,5),mean_tol=1e-7,covariance_tol=1e-7)
    @test p.diagnostics.resolved
    @test p.mean≈zeros(3) atol=1e-10
    # Conditioning must not let an arbitrary constant objective change a mode.
    c1=E._conditional_node!(deepcopy(g),[(k,1)],[.7])
    shifted=deepcopy(g)
    add_factor!(shifted,LinearFactor([k],[zeros(1,3)],[1e6]))
    c2=E._conditional_node!(shifted,[(k,1)],[.7])
    @test c1.mean==c2.mean
    @test c1.root==c2.root
    @test c2.log_integral-c1.log_integral≈-5e11
end

@testset "conditional evidence survives elimination" begin
    a=VarKey(:params,:static,0);x=VarKey(:state,:linear,0)
    g=FactorGraph();add_variable!(g,a,[0.0]);add_variable!(g,x,[0.0])
    add_factor!(g,GaussianPrior(a,[0.0],ones(1,1)))
    add_factor!(g,LinearFactor([a,x],[reshape([-1.0],1,1),ones(1,1)],[0.0]))
    axes=[(a,1)];qs,lw=E._tensor_quadrature(31,1)
    nodes=[E._conditional_node!(deepcopy(g),axes,q) for q in qs]
    logs=[l+sum(abs2,q)/2+c.log_integral for (l,q,c) in zip(lw,qs,nodes)]
    # y1=x1+noise and y2=x2+noise, xi=a+process noise: analytically,
    # a | y=[2,-1] ~ N(1/4,1/2). Compare after discarding x1.
    for (step,observed) in enumerate((2.0,-1.0))
        for j in eachindex(nodes)
            old=nodes[j];gc=old.graph
            key=step==1 ? x : VarKey(:state,:linear,step)
            if step==2
                add_variable!(gc,key,[gc.values[a][1]])
                add_factor!(gc,LinearFactor([a,key],[reshape([-1.0],1,1),ones(1,1)],[0.0]))
            end
            add_factor!(gc,LinearFactor([key],[ones(1,1)],[observed]))
            c=E._conditional_node!(gc,axes,qs[j])
            logs[j]+=c.log_integral-old.log_integral
            marginalize!(gc,[key])
            nodes[j]=E._conditional_node!(gc,axes,qs[j];iters=0)
        end
    end
    p=E._conditional_posterior(nodes,logs,axes)
    @test p.mean≈[.25] atol=1e-9
    @test p.covariance≈reshape([.5],1,1) atol=1e-8
end
