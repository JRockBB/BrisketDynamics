# Run from the repository's estimator environment. Snapshots must be available.
# julia --threads=4 --project=estimator docs/validation/2026-09-28-transport/reproduce.jl [snapshot_dir] [output_dir]
using LinearAlgebra,Serialization,SHA,JSON
const REPOSITORY=normpath(joinpath(@__DIR__,"../../.."))
const SNAPSHOTS=length(ARGS)>=1 ? abspath(ARGS[1]) : joinpath(REPOSITORY,"scratchpad/curvature_audit")
const OUTPUT=length(ARGS)>=2 ? abspath(ARGS[2]) : joinpath(REPOSITORY,"scratchpad/transport_replay")
mkpath(OUTPUT)
# The historical production, physics, and calibration source must still match.
for item in JSON.parsefile(joinpath(@__DIR__,"../2026-09-28-curvature/source-manifest.json"))
    bytes2hex(sha256(read(joinpath(REPOSITORY,item["path"]))))==item["sha256"] ||
        error("historical source mismatch: $(item["path"])")
end
for manifest in ("benchmark-source-manifest.json","reporting-fix-source-manifest.json")
    for item in JSON.parsefile(joinpath(@__DIR__,manifest))
        bytes2hex(sha256(read(joinpath(@__DIR__,item["preserved"]))))==item["sha256"] ||
            error("archived comparison source mismatch")
    end
end
include(joinpath(REPOSITORY,"estimator/src/BrisketEstimator.jl"))
const E=BrisketEstimator
module MappedRun
const E=Main.E
using LinearAlgebra
include(joinpath(Main.REPOSITORY,"estimator/scripts/curvature_audit.jl"))
for filename in ("adaptive_density.jl","conditional_charts.jl","adaptive_conditional.jl",
        "adaptive_curvature.jl","fit_conditional_chart.jl","mapped_diagnostics.jl")
    include(joinpath(@__DIR__,"sources/benchmark",filename))
end
include(joinpath(Main.REPOSITORY,"estimator/scripts/validate_adaptive.jl"))
include(joinpath(@__DIR__,"sources/reporting-fix/target_ownership.jl"))
include(joinpath(@__DIR__,"sources/reporting-fix/validate_transport.jl"))
# The archived driver's original @__DIR__ no longer points into estimator/.
# Report the verified archived hashes, rather than current working source hashes.
function transport_sources()
    [(path=r["path"],sha256=r["sha256"]) for r in
        E.JSON.parsefile(joinpath(@__DIR__,"reporting-fix-source-manifest.json"))]
end
end

let F=MappedRun
    snap=deserialize(joinpath(SNAPSHOTS,"two_core.jls"));anchor=deepcopy(snap.failed.bank.anchor);started=time()
    for batch in snap.batches[snap.arrival+1:end]
        E.assimilate!(anchor,batch).committed || error("continuation failed")
    end
    c=E._conditional_node!(deepcopy(anchor.graph),Tuple{E.VarKey,Int}[],Float64[];
        iters=80,tol=1e-6,ftol=1e-8,step_bounds=Dict(anchor.key_z=>anchor.latent_step),newton_tol=E.ONLINE_NEWTON_TOL)
    c.solve.converged || error("selection solve failed")
    p=E._conditional_posterior([c],[0.],Tuple{E.VarKey,Int}[])
    shape=E.conditional_shape_diagnostic(E.conditional_transport(p);key=anchor.key_z)
    indices=sortperm(shape.scores;rev=true)[1:2];axes=[(anchor.key_z,i) for i in indices]
    rows=[p.offsets[key][i] for (key,i) in axes]
    center=p.mean[rows];scale=Matrix(cholesky(Symmetric(p.covariance[rows,rows])).L)
    F.compare_transport_graph(anchor.graph,joinpath(OUTPUT,"late_two");axes,center,scale,
        shape_key=anchor.key_z,step_bounds=Dict(anchor.key_z=>anchor.latent_step),newton_tol=E.ONLINE_NEWTON_TOL,
        max_nodes=257,tolerance=.002,orders=(5,9),max_seconds=600.,methods=(:triangular,:affine),
        metadata=(case=snap.case,t=1800.,selection_scores=shape.scores,setup_seconds=time()-started,selection_fits=1))
    for suite in (:core,:two_core)
        F.compare_transport_snapshot(joinpath(SNAPSHOTS,"$(suite).jls"),joinpath(OUTPUT,string(suite));
            max_nodes=129,tolerance=.002,orders=(7,11),max_seconds=600.)
    end
    for suite in (:full,:pit_core)
        F.compare_transport_snapshot(joinpath(SNAPSHOTS,"$(suite).jls"),joinpath(OUTPUT,string(suite));
            add_axis=true,max_nodes=729,tolerance=.01,orders=(3,5),max_seconds=600.,methods=(:triangular,:affine))
    end
end
