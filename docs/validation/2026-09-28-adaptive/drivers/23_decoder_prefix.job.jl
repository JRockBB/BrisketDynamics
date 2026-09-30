for suite in (:core,:two_core)
    F=SharedDecoderVerification
    path=joinpath(@__DIR__,"$(suite)_decoder_prefix.json")
    p=F.validate_adaptive_snapshot(joinpath(@__DIR__,"../curvature_audit/$(suite).jls"),path;mode=:polished_gn)
    p===nothing || serialize(joinpath(@__DIR__,"$(suite)_decoder_prefix.jls"),p)
end