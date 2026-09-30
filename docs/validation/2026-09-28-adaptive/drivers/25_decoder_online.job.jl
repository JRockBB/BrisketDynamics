let F=SharedDecoderVerification
    snap=deserialize(joinpath(@__DIR__,"../curvature_audit/core.jls"))
    offline=deserialize(joinpath(@__DIR__,"core_decoder_prefix.jls"))
    anchor=snap.accepted.bank.anchor;saved=deepcopy(anchor);t=time()
    online=F.adaptive_arrival_candidate(anchor,snap.batches[snap.arrival];axes=offline.axes,
        center=offline.center,scale=offline.scale,lower=[-4.],upper=[4.],tolerance=.002,max_nodes=129,
        orders=(7,11),shape_key=anchor.key_z,curvature=F.polished_gn_conditional_curvature,progress=adaptive_progress)
    comparison=F.AdaptiveDensities.adaptive_density_distance(offline.density,online.candidate.density)
    noise_equal=all(a.cache.W==b.cache.W && a.cache.wy==b.cache.wy for (a,b) in zip(offline.decoder.dynamics,online.candidate.decoder.dynamics))
    unchanged=E._same_decoder_value(saved,anchor)
    @testset "final adaptive online target and ownership" begin
        @test !online.committed
        @test online.candidate.density.diagnostics.resolved
        @test online.candidate.diagnostics.shape_pass
        @test unchanged
        @test noise_equal
        @test comparison.tv_bound_estimate<.001
        @test last(anchor.nodes)[1]==75.
        @test last(online.anchor.nodes)[1]==510.
        @test_throws ArgumentError F.adaptive_arrival_candidate(anchor,E.MeasurementRecord[])
        short=deepcopy(anchor);short.lag=3
        @test_throws ArgumentError F.adaptive_arrival_candidate(short,snap.batches[snap.arrival])
    end
    compressed=deepcopy(anchor);compressed.lag=0;E.marginalize_old!(compressed);compressed.lag=typemax(Int)
    @testset "final compressed-history rejection" begin
        @test any(f isa E.LinearFactor for f in compressed.graph.factors)
        @test_throws ArgumentError F.adaptive_arrival_candidate(compressed,snap.batches[snap.arrival])
    end
    @testset "final strict adaptive reports" begin
        parsed=E.JSON.parse(F.adaptive_strict_json((bad=Inf,nan=NaN,good=.1,reason=:node_budget)))
        @test parsed["bad"]===nothing
        @test parsed["nan"]===nothing
        @test parsed["good"]==.1
        @test parsed["reason"]=="node_budget"
    end
    write(joinpath(@__DIR__,"decoder_online_replay.json"),F.adaptive_strict_json((wall=time()-t,comparison=comparison,
        noise_equal=noise_equal,anchor_unchanged=unchanged,committed=false,consistent=false,diagnostics=online.candidate.diagnostics)))
end