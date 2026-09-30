using Test,LinearAlgebra
if !isdefined(@__MODULE__,:E)
    include(joinpath(@__DIR__,"../src/BrisketEstimator.jl"));const E=BrisketEstimator
end
isdefined(@__MODULE__,:SparseConditionalCandidate) || include(joinpath(@__DIR__,"../scripts/sparse_conditional.jl"))
isdefined(@__MODULE__,:target_state_fingerprint) || include(joinpath(@__DIR__,"../scripts/target_ownership.jl"))

struct LocalCoupledResidual <: E.Factor
    key::E.VarKey
end
E.vars(f::LocalCoupledResidual)=[f.key]
E.ndim(::LocalCoupledResidual)=3
E.residual(f::LocalCoupledResidual,v)=[v[1][1],v[1][2],(v[1][3]-v[1][1]^2-.3v[1][1]*v[1][2])/.2]
E.jacobians(f::LocalCoupledResidual,v)=[E.ForwardDiff.jacobian(z->E.residual(f,[z]),v[1])]

@testset "local sparse original graph and chart" begin
    key=E.VarKey(:params,:local_coupled,0);g=E.FactorGraph()
    E.add_variable!(g,key,zeros(3));E.add_factor!(g,LocalCoupledResidual(key))
    before=target_state_fingerprint(g)
    p=expanding_sparse_fit(g;axes=[(key,1),(key,2)],shape_key=key,local_refinement=true,
        radius=2.,max_nodes=65,tolerance=1e-5,orders=(7,11))
    @test p isa LocalSparseConditionalCandidate
    @test p.diagnostics.resolved && p.diagnostics.shape_pass
    @test p.diagnostics.total_graph_solves==1+p.diagnostics.graph_solves<=65
    @test p.diagnostics.graph_solves==p.density.diagnostics.nodes
    @test target_state_fingerprint(g)==before
    q=[.27,-.63]
    @test adaptive_geometry(p.density,q).mean≈[q[1]^2+.3prod(q)] atol=1e-9
    @test adaptive_geometry(p.density,q).root≈reshape([.2],1,1) atol=1e-9
    @test isfinite(adaptive_conditional_logpdf(p,[q;.1]))
    @test !p.diagnostics.committed && !p.diagnostics.consistent
    chart=TriangularChart(AffineChart(zeros(2),Matrix{Float64}(I,2,2)),[[0.,0.,.6]],[[log(.8),0.,0.]],2.)
    mapped=sparse_conditional_fit(g;axes=[(key,1),(key,2)],chart,lower=fill(-2.,2),upper=fill(2.,2),
        local_refinement=true,max_nodes=129,tolerance=1e-5,orders=(7,11),shape_key=key)
    @test mapped.diagnostics.resolved && mapped.diagnostics.shape_pass
    x=[chart_forward(chart,q);.1]
    @test adaptive_conditional_logpdf(mapped,x)≈adaptive_logpdf(mapped.density,q,[.1])-chart_logjac(chart,q) atol=1e-10
    @test target_state_fingerprint(g)==before
    @test_throws ErrorException expanding_sparse_fit(g;axes=[(key,1),(key,2)],shape_key=key,
        local_refinement=true,checkpoint=()->error("deadline"))
    @test target_state_fingerprint(g)==before
end
