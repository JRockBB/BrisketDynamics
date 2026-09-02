using BrisketDynamics:
    brisket, chicken, ribs, sous_vide, pibil_pv, electric_pit, with_config,
    MeatConfig, p_sat

@testset "meat and cooker presets" begin
    b = brisket()
    @test length(b.θ) == 8
    @test b.cfg.geom_m == 0.0 && isinf(b.cfg.k_skin)
    c = chicken()
    @test isfinite(c.cfg.k_skin) && c.cfg.geom_m == 1.0
    r = ribs()
    @test r.θ[4] < b.θ[4]  # thin slab
    sv = sous_vide(; T_bath=331.15)
    @test sv.cfg.η_rest == 1.0 && sv.cfg.T_rest == 331.15 && sv.t_pull < 0
    @test pibil_pv(373.15, 5) == fill(p_sat(373.15), 5)
    @test electric_pit(; p_heater=3000.0).p_heater == 3000.0
    c2 = with_config(MeatConfig(); nx=21)
    @test c2.nx == 21 && c2.W0 == MeatConfig().W0
end
