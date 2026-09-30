# julia --project=estimator docs/validation/2026-09-28-sparse/reproduce.jl core sparse OUTPUT.json [VERSION]
# Cooks: core, two_core, late_two, full. Methods: tensor, sparse, expanding, mapped.
# Versions: benchmark (default), degree64, online, final. The final version
# includes bounded cubature, owned initial linearization and shared-budget local fallback.
using SHA,JSON,Serialization,LinearAlgebra
const ROOT=normpath(joinpath(@__DIR__,"../../.."))
const REPLAY_VERSION=length(ARGS)==4 ? ARGS[4] : "benchmark"
REPLAY_VERSION in ("benchmark","degree64","online","final") || error("unknown source version")
for entry in JSON.parsefile(joinpath(@__DIR__,"../2026-09-28-curvature/source-manifest.json"))
    bytes2hex(sha256(read(joinpath(ROOT,entry["path"]))))==entry["sha256"] || error("baseline source differs: $(entry["path"])")
end
archive_manifest=REPLAY_VERSION=="benchmark" ? "benchmark-source-manifest.json" : REPLAY_VERSION*"-source-archive-manifest.json"
for entry in JSON.parsefile(joinpath(@__DIR__,archive_manifest))
    bytes2hex(sha256(read(joinpath(@__DIR__,entry["archive"]))))==entry["sha256"] || error("archived source differs")
end
include(joinpath(ROOT,"estimator/src/BrisketEstimator.jl"))
const E=BrisketEstimator
module SparseReplay
const E=Main.E
const SOURCE=joinpath(@__DIR__,"sources",Main.REPLAY_VERSION,"estimator/scripts")
include(joinpath(SOURCE,"validate_sparse.jl"))
include(joinpath(SOURCE,"validate_sparse_mapped.jl"))
end
length(ARGS) in (3,4) || error("usage: reproduce.jl CASE METHOD OUTPUT.json [VERSION]")
case,method_name,output=ARGS[1:3];method=Symbol(method_name)
case in ("core","two_core","late_two","full") || error("unknown case")
method in (:tensor,:sparse,:expanding,:mapped) || error("unknown method")
input=case=="late_two" ? "scratchpad/transport_work/late_two_target.jls" : "scratchpad/curvature_audit/$(case).jls"
entry=only(filter(e->e["path"]==input,JSON.parsefile(joinpath(@__DIR__,"input-manifest.json"))))
bytes2hex(sha256(read(joinpath(ROOT,input))))==entry["sha256"] || error("input snapshot differs")
snapshot=deserialize(joinpath(ROOT,input))
if case=="late_two"
    anchor=snapshot.anchor;axes=snapshot.axes;center=snapshot.center;scale=snapshot.scale
else
    anchor=snapshot.failed.bank.anchor;old=snapshot.result.candidate;source=old.transported.source;axes=copy(source.axes)
    if case=="full" && method!=:expanding
        shape=E.conditional_shape_diagnostic(old.transported;key=anchor.key_z)
        remaining=setdiff(eachindex(shape.scores),last.(axes))
        push!(axes,(anchor.key_z,remaining[argmax(shape.scores[remaining])]))
    end
    rows=[source.offsets[k][j] for (k,j) in axes]
    center=source.mean[rows];scale=Matrix(cholesky(Symmetric(source.covariance[rows,rows])).L)
end
settings=(axes=axes,center=center,scale=scale,shape_key=anchor.key_z,
    max_nodes=case=="full" ? 729 : case=="late_two" ? 257 : 129,
    tolerance=case=="full" ? .01 : .002,orders=case=="full" ? (3,5) : case=="late_two" ? (5,9) : (7,11),
    step_bounds=Dict(anchor.key_z=>anchor.latent_step),newton_tol=E.ONLINE_NEWTON_TOL)
# Compile the physics/conditional solve before starting the timed comparison.
SparseReplay.sparse_conditional_fit(anchor.graph;axes,center,scale,lower=fill(-4.,length(axes)),
    upper=fill(4.,length(axes)),max_nodes=1,screen_only=true,
    curvature=SparseReplay.polished_gn_conditional_curvature,
    step_bounds=settings.step_bounds,newton_tol=settings.newton_tol)
if method==:mapped
    SparseReplay.validate_sparse_mapped(anchor.graph,output;settings...)
else
    SparseReplay.validate_sparse_graph(anchor.graph,output;method,settings...)
end
