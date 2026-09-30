using Test,LinearAlgebra
isdefined(@__MODULE__,:AdaptiveDensities) || include(joinpath(@__DIR__,"../src/adaptive_density.jl"))
isdefined(@__MODULE__,:SparseDensities) || include(joinpath(@__DIR__,"../src/sparse_density.jl"))
using .AdaptiveDensities,.SparseDensities

@testset "sparse joint density" begin
    # Analytically polynomial log marginal AND nonlinear conditional geometry.
    calls=Ref(0)
    function oracle(q)
        calls[]+=1
        (logmass=-sum(abs2,q)/2,mean=[q[1]^2+.4q[1]*q[2],q[2]],
            root=[exp(.1q[2]) 0.;.9 1e-6exp(.2q[1])])
    end
    p=sparse_density(oracle,fill(-2.,2),fill(2.,2);tolerance=1e-5,max_nodes=65,orders=(7,11))
    @test p.diagnostics.resolved
    @test calls[]==p.diagnostics.nodes<=65
    @test p.diagnostics.nodes<3^2+4*3
    @test !p.diagnostics.tail_validated && !p.diagnostics.target_validated
    q=[.317,-.481];n=oracle(q);g=adaptive_geometry(p,q)
    @test g.mean≈n.mean atol=1e-12
    @test g.root≈n.root atol=1e-12
    @test adaptive_expectation((q,n)->1.,p)≈1. atol=1e-10
    @test adaptive_marginal_logpdf(p,[3.,0.])==-Inf
    @test isfinite(adaptive_logpdf(p,q,n.mean))
    @test_throws DomainError adaptive_geometry(p,[3.,0.])

    # Axis and pairwise probes are blind to a pure four-way interaction.
    four(q)=(logmass=-sum(abs2,q)/2,mean=[8prod(q)],root=ones(1,1))
    limited=sparse_density(four,fill(-1.,4),fill(1.,4);max_nodes=75,tolerance=1e-5,orders=(3,5))
    @test !limited.diagnostics.resolved
    @test any(h.validation_error>1e-2 && isfinite(h.validation_error) for h in limited.diagnostics.history)
    @test limited.diagnostics.nodes<=75
    recovered=sparse_density(four,fill(-1.,4),fill(1.,4);max_nodes=257,tolerance=1e-5,orders=(5,7))
    @test recovered.diagnostics.resolved
    @test adaptive_geometry(recovered,[.2,.3,-.4,.5]).mean≈[-.096] atol=1e-10
    @test recovered.diagnostics.nodes<3^4+8*3^3

    # No interpolation error can excuse unresolved positive cubature.
    spike(q)=(logmass=-(q[1]-.341)^2/(2*.001^2),mean=Float64[],root=zeros(0,0))
    narrow=sparse_density(spike,[-1.],[1.];max_nodes=20,max_cells=1,orders=(3,5))
    @test !narrow.diagnostics.resolved && narrow.diagnostics.reason==:quadrature_budget
    @test narrow.diagnostics.integration_error>1
    @test_throws ArgumentError sparse_density(oracle,[-1.],[1.];max_nodes=0)
    @test_throws ArgumentError sparse_density(oracle,[-1.],[1.];tolerance=NaN)
    @test_throws ErrorException sparse_density(oracle,[-1.],[1.];checkpoint=()->error("deadline"))
end

@testset "sparse resource limits and peak bounds" begin
    calls=Ref(0)
    flat(q)=(calls[]+=1;(logmass=0.,mean=Float64[],root=zeros(0,0)))
    @test_throws ArgumentError sparse_density(flat,[-1.],[1.];max_nodes=1.5)
    @test_throws ArgumentError sparse_density(flat,[-1.],[1.];max_degree=65)
    @test_throws ArgumentError sparse_density(flat,fill(-1.,20),fill(1.,20);max_quadrature_evaluations=20)
    @test calls[]==0
    screen=sparse_density(flat,fill(-1.,20),fill(1.,20);screen_only=true,max_nodes=1)
    @test calls[]==1 && screen.diagnostics.nodes==1
    @test screen.diagnostics.reason==:shape_screen && !screen.diagnostics.resolved
    spike(q)=(logmass=-(q[1]-.341)^2/(2*.001^2),mean=Float64[],root=zeros(0,0))
    capped=sparse_density(spike,[-1.],[1.];orders=(3,5),max_quadrature_evaluations=8)
    @test !capped.diagnostics.resolved
    @test capped.diagnostics.quadrature_reason==:evaluation_budget
    @test capped.diagnostics.quadrature_evaluations==8
    xs=SparseDensities.leja(33)
    for (degree,elevated) in ((2,5),(16,20),(32,32))
        b=SparseDensities.basis_bernstein(xs,degree,.1,.2,elevated)
        for t in (0.,.13,.47,.81,1.)
            actual=SparseDensities.basis(xs,(degree,),[.1+.1t])
            value=sum(b[i+1]*binomial(big(elevated),i)*t^i*(1-t)^(elevated-i) for i in 0:elevated)
            @test value≈actual atol=1e-10 rtol=1e-8
            @test minimum(b)-1e-12<=actual<=maximum(b)+1e-12
        end
    end
end
