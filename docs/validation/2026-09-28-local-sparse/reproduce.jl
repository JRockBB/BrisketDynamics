# julia --project=estimator docs/validation/2026-09-28-local-sparse/reproduce.jl CASE METHOD OUTPUT.json [initial|final]
# CASE: core, two_core, two_core_coupled, late_two, full
# METHOD: local, expanding_local, mapped. Default source version: final.
using SHA,JSON,Serialization,LinearAlgebra
length(ARGS) in (3,4) || error("usage: reproduce.jl CASE METHOD OUTPUT.json [initial|final]")
const ROOT=normpath(joinpath(@__DIR__,"../../.."))
const LOCAL_REPLAY_VERSION=length(ARGS)==4 ? ARGS[4] : "final"
LOCAL_REPLAY_VERSION in ("initial","final") || error("unknown source version")
const SOURCE_DIR=LOCAL_REPLAY_VERSION=="final" ? "sources-final" : "sources"
const SOURCE_MANIFEST=LOCAL_REPLAY_VERSION=="final" ? "final-source-manifest.json" : "source-manifest.json"
for entry in JSON.parsefile(joinpath(@__DIR__,"../2026-09-28-curvature/source-manifest.json"))
    bytes2hex(sha256(read(joinpath(ROOT,entry["path"]))))==entry["sha256"] || error("production/calibration baseline changed")
end
for entry in JSON.parsefile(joinpath(@__DIR__,SOURCE_MANIFEST))
    bytes2hex(sha256(read(joinpath(@__DIR__,entry["archive"]))))==entry["sha256"] || error("source archive changed")
end
include(joinpath(ROOT,"estimator/src/BrisketEstimator.jl"))
const E=BrisketEstimator
module LocalReplay
const E=Main.E
include(joinpath(@__DIR__,Main.SOURCE_DIR,"estimator/scripts/validate_sparse.jl"))
include(joinpath(@__DIR__,Main.SOURCE_DIR,"estimator/scripts/validate_sparse_mapped.jl"))
end
case,method_name,output=ARGS[1:3];method=Symbol(method_name)
case in ("core","two_core","two_core_coupled","late_two","full") || error("unknown case")
method in (:local,:expanding_local,:mapped) || error("unknown method")
basecase=case=="two_core_coupled" ? "two_core" : case
input=basecase=="late_two" ? "scratchpad/transport_work/late_two_target.jls" : "scratchpad/curvature_audit/$(basecase).jls"
entry=only(filter(e->e["path"]==input,JSON.parsefile(joinpath(@__DIR__,"input-manifest.json"))))
bytes2hex(sha256(read(joinpath(ROOT,input))))==entry["sha256"] || error("frozen input changed")
snapshot=deserialize(joinpath(ROOT,input))
if basecase=="late_two"
    anchor=snapshot.anchor;axes=snapshot.axes;center=snapshot.center;scale=snapshot.scale
else
    anchor=snapshot.failed.bank.anchor;old=snapshot.result.candidate;source=old.transported.source;axes=copy(source.axes)
    if case=="two_core_coupled" || (case=="full" && method!=:expanding_local)
        shape=E.conditional_shape_diagnostic(old.transported;key=anchor.key_z)
        remaining=setdiff(eachindex(shape.scores),last.(axes))
        push!(axes,(anchor.key_z,remaining[argmax(shape.scores[remaining])]))
    end
    rows=[source.offsets[k][j] for (k,j) in axes]
    center=source.mean[rows];scale=Matrix(cholesky(Symmetric(source.covariance[rows,rows])).L)
end
settings=(axes=axes,center=center,scale=scale,shape_key=anchor.key_z,
    max_nodes=basecase=="full" ? 729 : basecase=="late_two" ? 257 : 129,
    tolerance=basecase=="full" ? .01 : .002,orders=basecase=="full" ? (3,5) : basecase=="late_two" ? (5,9) : (7,11),
    step_bounds=Dict(anchor.key_z=>anchor.latent_step),newton_tol=E.ONLINE_NEWTON_TOL,max_seconds=600.)
LocalReplay.sparse_conditional_fit(anchor.graph;axes,center,scale,lower=fill(-4.,length(axes)),
    upper=fill(4.,length(axes)),max_nodes=1,screen_only=true,
    curvature=LocalReplay.polished_gn_conditional_curvature,
    step_bounds=settings.step_bounds,newton_tol=settings.newton_tol)
if method==:mapped
    LocalReplay.validate_sparse_mapped(anchor.graph,output;local_refinement=true,settings...)
else
    LocalReplay.validate_sparse_graph(anchor.graph,output;method,settings...)
end
