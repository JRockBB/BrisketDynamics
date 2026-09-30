@testset "retained posterior forecasts survive later accepted readings" begin
    model=consistency_model(nx=4,suite=:pit,stop=600.)
    sm=sampled_init(model;rng=MersenneTwister(5611));p=sm.posterior;ref=p.history.reference
    original_z=copy(ref.g.values[ref.kz]);original_y=copy(ref.g.values[last(ref.ks)])
    before=posterior_predict(p,[60.];n=32,rng=MersenneTwister(5612))
    ip=first(E.belief_index(model.bl,:pit,:T_pit))
    @test assimilate!(sm,MeasurementRecord(:pit,0.,original_y[ip]+2.)).committed
    @test ref.g.values[last(ref.ks)]!=original_y
    after=posterior_predict(p,[60.];n=32,rng=MersenneTwister(5612))
    @test before.draws==after.draws
    @test before.physical==after.physical
    @test ref.zanchor==original_z && ref.yanchor==original_y
    @test ref.zanchor!==ref.g.values[ref.kz] && ref.yanchor!==ref.g.values[last(ref.ks)]
    @test sm.posterior.history.reference.yanchor==sm.graph_backend.graph.values[last(ref.ks)]
    checkpoint=smoother_checkpoint(sm);restored=restore_checkpoint(checkpoint)
    first_forecast=posterior_predict(sm,[60.];n=32,rng=MersenneTwister(5613))
    second_forecast=posterior_predict(restored,[60.];n=32,rng=MersenneTwister(5613))
    @test first_forecast.draws==second_forecast.draws
    @test restored.posterior.history.reference.yanchor!==sm.posterior.history.reference.yanchor
end
