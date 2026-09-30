# Reproducible offline / online-target qualification, never a deployment command.
if !isdefined(@__MODULE__,:E)
    include(joinpath(@__DIR__,"../src/BrisketEstimator.jl"))
    const E=BrisketEstimator
end
using Serialization, SHA, LinearAlgebra
isdefined(@__MODULE__,:innovation_curvature) || include(joinpath(@__DIR__,"curvature_audit.jl"))
isdefined(@__MODULE__,:AdaptiveConditionalCandidate) || include(joinpath(@__DIR__,"adaptive_conditional.jl"))
isdefined(@__MODULE__,:observed_conditional_curvature) || include(joinpath(@__DIR__,"adaptive_curvature.jl"))

function adaptive_strict_json(value)
    # Preserve the reason/status of unavailable diagnostics; JSON null replaces
    # nonfinite numbers, rather than emitting nonstandard NaN/Infinity tokens.
    function clean(x)
        x isa AbstractDict && return Dict(k=>clean(v) for (k,v) in x)
        x isa AbstractArray && return map(clean,x)
        x isa Real && !isfinite(x) && return nothing
        return x
    end
    parsed=E.JSON.parse(E.json_document(value);allownan=true)
    return E.JSON.json(clean(parsed);allownan=false,pretty=true)
end

function validate_adaptive_snapshot(snapshot_path,report_path;mode=:observed,radius=4.,tolerance=.002,
                                    max_nodes=129,orders=(7,11),add_axis=false)
    mode in (:gn,:polished_gn,:observed) || throw(ArgumentError("mode must be gn, polished_gn or observed"))
    isfinite(radius) && radius>0 || throw(ArgumentError("radius must be positive and finite"))
    snapshot=deserialize(snapshot_path)
    snapshot.backend==E.backend_fingerprint() || error("snapshot backend differs from current source")
    snapshot.physics==E.physics_fingerprint() || error("snapshot physics differs from current source")
    old=snapshot.result.candidate;source=old.transported.source;anchor=snapshot.failed.bank.anchor
    selected=copy(source.axes)
    if add_axis
        shape=E.conditional_shape_diagnostic(old.transported;key=anchor.key_z)
        remaining=setdiff(eachindex(shape.scores),[i for (_,i) in selected])
        isempty(remaining) && throw(ArgumentError("no parameter axis remains"))
        push!(selected,(anchor.key_z,remaining[argmax(shape.scores[remaining])]))
        ia=[source.offsets[k][i] for (k,i) in selected]
        center=source.mean[ia];scale=Matrix(cholesky(Symmetric(source.covariance[ia,ia])).L)
    else
        center=old.center+old.scale*old.base_mean;scale=old.scale*old.base_root
    end
    correction=mode===:gn ? nothing : mode===:polished_gn ? polished_gn_conditional_curvature :
        (node,ref)->observed_conditional_curvature(node,ref;polish=true)
    files=["src/adaptive_density.jl","scripts/adaptive_conditional.jl","scripts/adaptive_curvature.jl",
           "scripts/curvature_audit.jl","scripts/validate_adaptive.jl"]
    sources=[(path="estimator/"*p,sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"..",p))))) for p in files]
    start=time();candidate=nothing;outcome=nothing
    try
        candidate=adaptive_conditional_fit(anchor.graph;axes=selected,center,scale,
            lower=fill(-radius,length(center)),upper=fill(radius,length(center)),tolerance,max_nodes,orders,
            curvature=correction,step_bounds=Dict(anchor.key_z=>anchor.latent_step),newton_tol=E.ONLINE_NEWTON_TOL,
            shape_key=anchor.key_z,progress=h->(println(h);flush(stdout)))
        outcome=(status=:candidate,diagnostics=candidate.diagnostics)
    catch err
        err isa InterruptException && rethrow()
        outcome=(status=:failed,error=sprint(showerror,err))
    end
    write(report_path,adaptive_strict_json((backend=snapshot.backend,physics=snapshot.physics,
        snapshot_sha256=bytes2hex(sha256(read(snapshot_path))),case=snapshot.case,sources=sources,
        settings=(mode=mode,radius=radius,tolerance=tolerance,max_nodes=max_nodes,orders=orders,axes=selected),
        wall=time()-start,outcome=outcome,committed=false,consistent=false)))
    return candidate
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)>=2 || error("usage: validate_adaptive.jl SNAPSHOT.jls REPORT.json [observed|polished_gn|gn] [radius=4] [tolerance=.002]")
    result=validate_adaptive_snapshot(ARGS[1],ARGS[2];mode=length(ARGS)>=3 ? Symbol(ARGS[3]) : :observed,
        radius=length(ARGS)>=4 ? parse(Float64,ARGS[4]) : 4.,tolerance=length(ARGS)>=5 ? parse(Float64,ARGS[5]) : .002)
    result===nothing && error("adaptive qualification failed; see the saved report")
    result.density.diagnostics.resolved || error("adaptive density is unresolved; see the saved report")
    result.diagnostics.shape_pass || error("conditional shape check failed; see the saved report")
end
