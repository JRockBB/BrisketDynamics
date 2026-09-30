let F=SharedDecoderVerification
    p=deserialize(joinpath(@__DIR__,"core_decoder_prefix.jls"))
    snap=deserialize(joinpath(@__DIR__,"../curvature_audit/core.jls"))
    original=sort([f for f in snap.failed.bank.anchor.graph.factors if f isa E.DynamicsFactor];by=f->f.t1)
    q=first(p.diagnostics.shape_checks).q;n=F.AdaptiveDensities.adaptive_geometry(p.density,q)
    u=copy(p.decoder.mean);u[p.axisrows]=p.center+p.scale*q;u[p.free]=n.mean
    x=E._conditional_decode(p.decoder,u);roundtrip=E._conditional_encode(p.decoder,x)
    target=F.innovation_target(p.decoder,u)
    physical=E._objective_at(p.decoder.graph,x,p.decoder.offsets)
    @testset "corrected dynamics map is shared by inference and decoding" begin
        @test all(a.model.abstol==b.model.abstol*.0001 && a.model.reltol==b.model.reltol*.0001 for (a,b) in zip(p.decoder.dynamics,original))
        @test all(a.cache.W==b.cache.W && a.cache.wy==b.cache.wy for (a,b) in zip(p.decoder.dynamics,original))
        @test norm(LowerTriangular(n.root)\(roundtrip-u)[p.free])<1e-6
        @test abs(target-physical)<1e-6
        @test !E._same_conditional_decoder(p.decoder,F.curvature_accuracy_ref(p.decoder,.1))
    end
    write(joinpath(@__DIR__,"shared_decoder_check.json"),F.adaptive_strict_json((
        innovation_roundtrip_norm=norm(LowerTriangular(n.root)\(roundtrip-u)[p.free]),
        target_roundtrip_error=abs(target-physical),solver_factor=.0001,process_noise_changed=false)))
end