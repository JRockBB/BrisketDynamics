using Test,LinearAlgebra,SpecialFunctions
isdefined(@__MODULE__,:AdaptiveDensities) || include(joinpath(@__DIR__,"../src/adaptive_density.jl"))
isdefined(@__MODULE__,:SparseDensities) || include(joinpath(@__DIR__,"../src/sparse_density.jl"))
isdefined(@__MODULE__,:LocalSparseDensities) || include(joinpath(@__DIR__,"../src/local_sparse_density.jl"))
using .AdaptiveDensities,.SparseDensities,.LocalSparseDensities

@testset "local sparse mixed geometry and normalization" begin
    seen=Set{Tuple}();calls=Ref(0)
    function oracle(q)
        push!(seen,Tuple(q));calls[]+=1
        (logmass=-sum(abs2,q)/2,mean=[q[1]^2+.3q[1]*q[2],q[2]],
            root=[exp(.1q[2]) 0.;.9 1e-6exp(.2q[1])])
    end
    p=local_sparse_density(oracle,fill(-2.,2),fill(2.,2);max_nodes=65,tolerance=1e-5,orders=(7,11))
    @test p.diagnostics.resolved && p.diagnostics.cells==1
    @test calls[]==length(seen)==p.diagnostics.nodes<=65
    @test p.log_normalizer≈2log(sqrt(2pi)*erf(sqrt(2.))) atol=1e-10
    q=[.213,-.719];n=adaptive_geometry(p,q);truth=oracle(q)
    @test n.mean≈truth.mean atol=1e-12
    @test n.root≈truth.root atol=1e-12
    @test adaptive_expectation((q,n)->1.,p)≈1. atol=1e-10
    @test isfinite(adaptive_logpdf(p,q,n.mean))
    @test adaptive_marginal_logpdf(p,[3.,0.])==-Inf
    @test_throws DomainError adaptive_geometry(p,[3.,0.])
    @test !p.diagnostics.target_validated && !p.diagnostics.tail_validated
    four(q)=(logmass=-sum(abs2,q)/2,mean=[8prod(q)],root=ones(1,1))
    limited=local_sparse_density(four,fill(-1.,4),fill(1.,4);max_nodes=75,tolerance=1e-5,orders=(3,5))
    @test !limited.diagnostics.resolved
    @test any(h.validation_error>.01 for h in limited.diagnostics.history)
    recovered=local_sparse_density(four,fill(-1.,4),fill(1.,4);max_nodes=257,tolerance=1e-5,orders=(5,7))
    @test recovered.diagnostics.resolved
    @test adaptive_geometry(recovered,[.2,.3,-.4,.5]).mean≈[-.096] atol=1e-10
    @test recovered.diagnostics.nodes<297
end

@testset "coupled spatial refinement and shared budget" begin
    calls=Ref(0);seen=Set{Tuple}()
    function kink(q)
        calls[]+=1;push!(seen,Tuple(q))
        (logmass=-sum(abs2,q)/2,mean=[max(q[1]-.137,0.)^2*(1+.2q[2])],root=reshape([.02],1,1))
    end
    p=local_sparse_density(kink,fill(-2.,2),fill(2.,2);tolerance=.002,max_nodes=1025,orders=(7,11))
    @test p.diagnostics.resolved
    @test p.diagnostics.cells>1
    @test calls[]==length(seen)==p.diagnostics.nodes<=1025
    @test p.diagnostics.quadrature_evaluations<=2_000_000
    # Independent grid points, absent from fitting and both stopping panels.
    errors=[abs(adaptive_geometry(p,[x,y]).mean[1]-max(x-.137,0.)^2*(1+.2y))/.02
        for x in range(-1.93,1.87;length=61),y in range(-1.89,1.91;length=19)]
    @test maximum(errors)<.01
    @test adaptive_expectation((q,n)->1.,p)≈1. atol=1e-10
    @test p.log_normalizer≈2log(sqrt(2pi)*erf(sqrt(2.))) atol=1e-9
    plain(q)=(logmass=-sum(abs2,q)/2,mean=[max(q[1]-.137,0.)^2*(1+.2q[2])],root=reshape([.02],1,1))
    capped=local_sparse_density(plain,fill(-2.,2),fill(2.,2);tolerance=.002,max_nodes=30,orders=(7,11))
    @test !capped.diagnostics.resolved && capped.diagnostics.nodes<=30
    @test isfinite(adaptive_logpdf(capped,[1.91,-1.89],[0.]))
    @test isfinite(adaptive_logpdf(capped,[-1.91,1.89],[0.]))
    @test adaptive_expectation((q,n)->1.,capped)≈1. atol=1e-10
end

@testset "local sparse failure and resource limits" begin
    calls=Ref(0)
    flat(q)=(calls[]+=1;(logmass=0.,mean=Float64[],root=zeros(0,0)))
    @test_throws ArgumentError local_sparse_density(flat,[-1.],[1.];max_degree=2)
    @test_throws ArgumentError local_sparse_density(flat,[-1.],[1.];max_nodes=1.5)
    @test_throws ArgumentError local_sparse_density(flat,fill(-1.,20),fill(1.,20);max_quadrature_evaluations=20)
    @test calls[]==0
    @test_throws ErrorException local_sparse_density(flat,[-1.],[1.];checkpoint=()->error("deadline"))
    @test_throws ArgumentError local_sparse_density(q->(logmass=NaN,mean=[],root=zeros(0,0)),[-1.],[1.])
    p=local_sparse_density(flat,[-1.],[1.];max_nodes=1,orders=(3,5),max_quadrature_evaluations=8)
    @test !p.diagnostics.resolved && p.diagnostics.nodes==1
    @test p.diagnostics.quadrature_evaluations==8
    @test isfinite(adaptive_marginal_logpdf(p,[.4]))
    spike(q)=(logmass=-(q[1]-.341)^2/(2*.001^2),mean=Float64[],root=zeros(0,0))
    p=local_sparse_density(spike,[-1.],[1.];orders=(3,5),max_quadrature_evaluations=8)
    @test !p.diagnostics.resolved && p.diagnostics.reason==:quadrature_budget
    @test p.diagnostics.integration_error>1
end
