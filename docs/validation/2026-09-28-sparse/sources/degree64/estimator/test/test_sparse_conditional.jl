using Test,LinearAlgebra
if !isdefined(@__MODULE__,:E)
    include(joinpath(@__DIR__,"../src/BrisketEstimator.jl"));const E=BrisketEstimator
end
isdefined(@__MODULE__,:SparseConditionalCandidate) || include(joinpath(@__DIR__,"../scripts/sparse_conditional.jl"))
isdefined(@__MODULE__,:target_state_fingerprint) || include(joinpath(@__DIR__,"../scripts/target_ownership.jl"))

struct SparseBananaResidual <: E.Factor
    key::E.VarKey
end
E.vars(f::SparseBananaResidual)=[f.key]
E.ndim(::SparseBananaResidual)=3
E.residual(f::SparseBananaResidual,v)=[v[1][1],v[1][2],(v[1][3]-v[1][2]^2)/.4]
E.jacobians(f::SparseBananaResidual,v)=[E.ForwardDiff.jacobian(z->E.residual(f,[z]),v[1])]

# A linearization cache is mutable even for an exactly Gaussian factor.
mutable struct SparseCountingResidual <: E.Factor
    key::E.VarKey
    evaluations::Int
end
E.vars(f::SparseCountingResidual)=[f.key]
E.ndim(::SparseCountingResidual)=2
E.residual(f::SparseCountingResidual,v)=copy(v[1])
function E.jacobians(f::SparseCountingResidual,v)
    f.evaluations+=1
    [Matrix{Float64}(I,2,2)]
end

@testset "sparse parameter promotion" begin
    key=E.VarKey(:params,:sparse_test,0);g=E.FactorGraph()
    E.add_variable!(g,key,zeros(3));E.add_factor!(g,SparseBananaResidual(key))
    saved=target_state_fingerprint(g)
    p=expanding_sparse_fit(g;axes=[(key,1)],shape_key=key,max_nodes=65,max_axes=3,
        radius=2.,tolerance=1e-5,orders=(7,11))
    @test p.axes==[(key,1),(key,2)]
    @test p.diagnostics.resolved && p.diagnostics.shape_pass
    @test !p.diagnostics.committed && !p.diagnostics.consistent
    @test length(p.diagnostics.axis_stages)==3
    @test p.diagnostics.total_graph_solves==sum(s.fits for s in p.diagnostics.axis_stages)<=65
    @test p.diagnostics.axis_stages[1].shape_defect>3
    @test p.diagnostics.axis_stages[2].shape_pass
    @test target_state_fingerprint(g)==saved
    @test adaptive_geometry(p.density,[.3,.7]).mean≈[.49] atol=1e-8
    @test adaptive_geometry(p.density,[.3,.7]).root≈reshape([.4],1,1) atol=1e-8
    @test isfinite(adaptive_conditional_logpdf(p,[.3,.7,.5]))
    capped=expanding_sparse_fit(g;axes=[(key,1)],shape_key=key,max_nodes=65,max_axes=1)
    @test !capped.diagnostics.resolved && capped.diagnostics.reason==:axis_budget
    @test capped.diagnostics.total_graph_solves==1
    exhausted=expanding_sparse_fit(g;axes=[(key,1)],shape_key=key,max_nodes=1)
    @test !exhausted.diagnostics.resolved && exhausted.diagnostics.reason==:node_budget
    @test exhausted.diagnostics.total_graph_solves==1
    @test target_state_fingerprint(g)==saved
end

@testset "promotion owns initial linearization" begin
    key=E.VarKey(:params,:sparse_cache,0);g=E.FactorGraph()
    E.add_variable!(g,key,zeros(2));factor=SparseCountingResidual(key,0);E.add_factor!(g,factor)
    before=target_state_fingerprint(g)
    p=expanding_sparse_fit(g;axes=[(key,1)],shape_key=key,max_nodes=20,radius=2.)
    @test p.diagnostics.resolved
    @test factor.evaluations==0
    @test target_state_fingerprint(g)==before
    @test_throws ErrorException expanding_sparse_fit(g;axes=[(key,1)],shape_key=key,
        checkpoint=()->error("deadline"))
    @test factor.evaluations==0 && target_state_fingerprint(g)==before
end

@testset "sparse nonlinear map Jacobian" begin
    key=E.VarKey(:params,:sparse_map,0);g=E.FactorGraph()
    E.add_variable!(g,key,zeros(3));E.add_factor!(g,SparseBananaResidual(key))
    map=TriangularChart(AffineChart(zeros(2),Matrix{Float64}(I,2,2)),[[0.,0.,4.]],[[log(.4),0.,0.]],2.)
    p=sparse_conditional_fit(g;axes=[(key,2),(key,3)],chart=map,lower=fill(-4.,2),upper=fill(4.,2),
        max_nodes=65,tolerance=.002,orders=(7,11),shape_key=key)
    @test p.diagnostics.resolved && p.diagnostics.shape_pass
    @test adaptive_geometry(p.density,[.3,.7]).mean≈[0.] atol=1e-9
    x=[.3,.7,.7^2-.4*.2];q=[.7,-.2]
    # Difference from the q-coordinate density is exactly the inverse Jacobian.
    @test adaptive_conditional_logpdf(p,x)≈adaptive_logpdf(p.density,q,[.3])-log(.4) atol=1e-10
    @test all(n.point≈chart_forward(map,n.q) for n in p.diagnostics.nodes)
end

isdefined(@__MODULE__,:validate_sparse_mapped) || include(joinpath(@__DIR__,"../scripts/validate_sparse_mapped.jl"))
@testset "sparse report budgets and settings" begin
    key=E.VarKey(:params,:sparse_report,0);g=E.FactorGraph()
    E.add_variable!(g,key,zeros(2));E.add_factor!(g,E.GaussianPrior(key,zeros(2),Matrix{Float64}(I,2,2)))
    for method in (:tensor,:sparse,:expanding,:mapped)
        mktemp() do path,io
            close(io)
            kw=(axes=[(key,1)],center=[0.],scale=ones(1,1),shape_key=key,
                max_nodes=20,radius=2.,max_degree=4,max_quadrature_evaluations=10000)
            out=method==:mapped ? validate_sparse_mapped(g,path;kw...) : validate_sparse_graph(g,path;method,kw...)
            @test out.report.graph_unchanged && out.report.source_unchanged
            @test out.report.completed_fits<=20
            @test out.candidate.diagnostics.resolved
            @test out.report.settings.max_degree==(method==:tensor ? nothing : 4)
        end
    end
end
