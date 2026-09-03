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

@testset "pork butt preset + Newman quasi-3D" begin
    pb = pork_butt()
    @test pb.cfg.geom_m == 2.0 && pb.cfg.F0 > MeatConfig().F0
    @test length(pb.θ) == 8 && pb.θ[4] ≈ 0.05

    # Newman product rule on two identical slab axes: more faces heat the
    # center FASTER than either 1-D solution, never above the pit, and an
    # axis held at θ = 1 (infinite dimension) reduces to the other axis
    hr = 3600.0
    tknots = collect(0.0:600.0:6hr)
    Tpit = fill(383.15, length(tknots))
    o = meat_solve(MEAT_θ_DEFAULT, Tpit, tknots, tknots)
    T0 = o.Tcore[1]
    T2 = BrisketDynamics.newman_center((o.Tcore, o.Tcore), Tpit, T0)
    @test all(T2 .>= o.Tcore .- 1e-9)
    @test all(T2 .<= Tpit .+ 1e-9)
    T_inf = fill(T0, length(tknots))  # an axis that never heats: θ ≡ 1
    T1 = BrisketDynamics.newman_center((o.Tcore, T_inf), Tpit, T0)
    @test maximum(abs.(T1 .- o.Tcore)) < 1e-6
end
