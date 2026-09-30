let F=SharedDecoderVerification
for suite in (:core,:two_core)
    snap=deserialize(joinpath(@__DIR__,"../curvature_audit/$(suite).jls"));anchor=deepcopy(snap.failed.bank.anchor)
    for batch in snap.batches[snap.arrival+1:end]
        r=E.assimilate!(anchor,batch);r.committed || error("full-history continuation failed")
    end
    component=E._conditional_node!(deepcopy(anchor.graph),Tuple{E.VarKey,Int}[],Float64[];
        iters=80,tol=1e-6,ftol=1e-8,step_bounds=Dict(anchor.key_z=>anchor.latent_step),newton_tol=E.ONLINE_NEWTON_TOL)
    component.solve.converged || error("axis-selection solve failed")
    reference=E._conditional_posterior([component],[0.],Tuple{E.VarKey,Int}[])
    shape=E.conditional_shape_diagnostic(E.conditional_transport(reference);key=anchor.key_z)
    selected=[(anchor.key_z,argmax(shape.scores))]
    for dimension in 1:1
        rows=[reference.offsets[key][index] for (key,index) in selected]
        center=reference.mean[rows];scale=Matrix(cholesky(Symmetric(reference.covariance[rows,rows])).L)
        t=time();candidate=nothing;outcome=nothing
        println("GUARDED FULL ",suite," axes ",anchor.model.blk.names[last.(selected)]);flush(stdout)
        try
            candidate=F.adaptive_conditional_fit(anchor.graph;axes=selected,center,scale,
                lower=fill(-4.,dimension),upper=fill(4.,dimension),tolerance=.002,max_nodes=dimension==1 ? 129 : 257,
                orders=dimension==1 ? (7,11) : (5,9),curvature=F.polished_gn_conditional_curvature,
                step_bounds=Dict(anchor.key_z=>anchor.latent_step),newton_tol=E.ONLINE_NEWTON_TOL,
                shape_key=anchor.key_z,progress=adaptive_progress)
            outcome=(status=:candidate,diagnostics=candidate.diagnostics)
            serialize(joinpath(@__DIR__,"$(suite)_decoder_full_$(dimension)d.jls"),candidate)
        catch err
            err isa InterruptException && rethrow()
            outcome=(status=:failed,error=sprint(showerror,err))
        end
        write(joinpath(@__DIR__,"$(suite)_decoder_full_$(dimension)d.json"),F.adaptive_strict_json((suite=suite,
            revision=:shared_decoder,source_sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"../../estimator/src/adaptive_density.jl")))),t=last(anchor.nodes)[1],wall=time()-t,axes=selected,axis_names=anchor.model.blk.names[last.(selected)],
            outcome=outcome,committed=false,consistent=false)))
        candidate===nothing && break
        candidate.diagnostics.shape_complete || break
        candidate.diagnostics.shape_pass && break
        remaining=setdiff(eachindex(shape.scores),last.(selected))
        scores=reduce((a,b)->max.(a,b),[check.scores for check in candidate.diagnostics.shape_checks])
        push!(selected,(anchor.key_z,remaining[argmax(scores[remaining])]))
    end
end
end