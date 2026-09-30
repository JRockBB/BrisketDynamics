if !isdefined(Main,:BrisketEstimator)
    include(joinpath(@__DIR__,"..","src","BrisketEstimator.jl"))
end
using .BrisketEstimator, LinearAlgebra, Test
isdefined(Main,:E) || (const E=BrisketEstimator)
include(joinpath(@__DIR__,"..","scripts","curvature_audit.jl"))

struct AuditQuadraticResidual <: Factor
    key::VarKey
    a::Vector{Float64}
end
E.vars(f::AuditQuadraticResidual)=[f.key]
E.ndim(::AuditQuadraticResidual)=1
E.residual(f::AuditQuadraticResidual,v)=[sqrt(1+dot(f.a,v[1])^2)]
E.jacobians(f::AuditQuadraticResidual,v)=[reshape(f.a*dot(f.a,v[1])/sqrt(1+dot(f.a,v[1])^2),1,:)]

struct AuditQuarticResidual <: Factor
    key::VarKey
end
E.vars(f::AuditQuarticResidual)=[f.key]
E.ndim(::AuditQuarticResidual)=1
E.residual(f::AuditQuarticResidual,v)=[1+v[1][1]^2]
E.jacobians(f::AuditQuarticResidual,v)=[reshape([2v[1][1],0.],1,:)]

function audit_test_ref(f,k)
    g=FactorGraph();add_variable!(g,k,zeros(2))
    add_factor!(g,GaussianPrior(k,zeros(2),Matrix{Float64}(I,2,2)))
    add_factor!(g,f)
    c=E._conditional_node!(g,Tuple{VarKey,Int}[],Float64[])
    return E._conditional_transport(c)
end

@testset "observed curvature distinguishes quadratic error from non-Gaussian shape" begin
    k=VarKey(:params,:audit,0)
    ref=audit_test_ref(AuditQuadraticResidual(k,[1.,2.]),k)
    a=innovation_curvature(ref;key=k,basis=:full)
    @test a.hessian≈[2. 2.;2. 5.] atol=1e-10
    @test last(a.checks).change<1e-10
    @test last(a.checks).asymmetry<1e-10
    @test a.gradient≈zeros(2)
    @test a.gn_whitening_error<1e-12
    @test E.JSON.parse(E.json_document(a.checks))[1]["change"]===nothing
    @test -.5logdet(a.hessian)≈-.5log(6)
    @test innovation_target(ref,[.2,.3];gradient=true).gradient≈a.hessian*[.2,.3]
    d=curvature_direction_report(ref,a;key=k)
    @test maximum(v.observed_defect for r in d for v in r.values)<1e-10
    @test maximum(v.gn_defect for r in d for v in r.values)>1.
    corrected=curvature_corrected_report(ref,a;key=k)
    @test corrected.positive
    @test corrected.log_volume_correction≈-.5log(6)
    @test maximum(v.observed_defect for r in corrected.directions for v in r.values)<1e-10
    rotated=innovation_curvature(ref;key=k,basis=[1. -1.;1. 1.]/sqrt(2))
    rotated_correction=curvature_corrected_report(ref,rotated;key=k)
    @test rotated_correction.log_volume_correction≈corrected.log_volume_correction
    @test maximum(v.observed_defect for r in rotated_correction.directions for v in r.values)<1e-10
    @test_throws ArgumentError innovation_curvature(ref;key=k,basis=ones(2,2))
    q=innovation_curvature(ref;key=k,basis=reshape([1.,0.],2,1))
    @test last(q.checks).cross_curvature≈2.
    @test !q.full
    @test_throws ArgumentError curvature_corrected_report(ref,q;key=k)

    ref=audit_test_ref(AuditQuarticResidual(k),k)
    a=innovation_curvature(ref;key=k,basis=:full,steps=(.01,.005,.0025))
    @test a.hessian≈[3. 0.;0. 1.] atol=2e-5
    @test a.checks[3].change<a.checks[2].change/3
    d=curvature_direction_report(ref,a;key=k)
    v=only(v for v in d[1].values if v.radius==1.)
    @test v.observed_defect≈.5 atol=1e-5
    @test v.gn_defect≈1.5 atol=1e-10
    @test abs(v.odd_remainder)<1e-10
    corrected=curvature_corrected_report(ref,a;key=k)
    v=only(v for v in corrected.directions[1].values if v.radius==1.)
    @test v.observed_defect≈1/18 atol=1e-5
end
