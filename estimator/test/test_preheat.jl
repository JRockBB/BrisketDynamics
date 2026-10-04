using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, JSON, Test, LinearAlgebra
const E = BrisketEstimator
include(joinpath(@__DIR__, "..", "scripts", "stream_fixture.jl"))
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

@testset "asynchronous slices preserve executed control corners" begin
    tg = collect(0.0:120.0:600.0)
    ug = [0.1, 0.1, 0.8, 0.2, 0.6, 0.6]
    log = CookLog(tg, ug)
    for (a, b) in ((100.0, 250.0), (239.0, 241.0), (120.0, 360.0), (599.0, 600.0))
        tk, uk = damper_knots(log, a, b)
        @test all(lininterp(τ, tk[1], tk[2] - tk[1], uk) ≈ lininterp(τ, tg[1], 120.0, ug)
                  for τ in range(a, b; length=31))
    end
    partial = CookLog(tg, [0.1, 0.1, 0.8, NaN, NaN, NaN])
    @test last(damper_knots(partial, 100.0, 240.0)[1]) == 240.0
    @test_throws ArgumentError damper_knots(partial, 100.0, 241.0)
end

@testset "the streaming lighting fixture uses the refit and an explicit load" begin
    legacy = stream_fixture(REPO)
    fx = stream_fixture(REPO; start=:lighting)
    @test fx.pit.Tpit0 == fx.pit.T_amb
    @test fx.events.t_load == 5400.0
    @test fx.events.t_wrap == legacy.events.t_wrap + fx.events.t_load
    @test length(fx.t) == length(fx.u)
    @test all(isapprox.(diff(fx.t), fx.t[2] - fx.t[1]))
    a = fx.sites[findfirst(==("a0"), fx.names)]
    @test a.p1 / (a.p1 + a.p2) ≈ 0.15
    @test sqrt(a.p1 * a.p2 / ((a.p1 + a.p2)^2 * (a.p1 + a.p2 + 1))) ≈ 0.05
    @test fx.sites[findfirst(==("L"), fx.names)].p1 ≈ log(JSON.parsefile(joinpath(REPO, "data/calibration/map_real.json"))["L"])
    @test_throws ArgumentError stream_fixture(REPO; start=:unknown)
end

@testset "the estimator carries cold-start preheat through loading and restart" begin
    fx = stream_fixture(REPO; start=:lighting)
    cfg = MeatConfig(; nx=8, T0=fx.cfg.T0)
    sys = CookSystem(Slab1D(:brisket; cfg); pit=fx.pit)
    blk = system_latents(sys, fx.sites, fx.names; area=false)
    z = [z_moments(s)[1] for s in blk.sites]
    x = from_z(z, blk.sites)
    event = CookEvent(:load, :brisket, fx.events.t_load)
    model = SystemModel(sys, CookLog(fx.t, fx.u; events=[event]), blk)
    p, play, _ = E._slice_params(model, z, 0.0, 10800.0)
    y0 = belief_encode(belief_from_system(initial_state(sys, p, play), model.bl), model.bl)
    save = [0.0, 3000.0, 5400.0, 5700.0, 10800.0] # 20 blend widths before loading
    y, o = E.propagate(model, y0, z, 0.0, last(save); first_slice=true, output_times=save)
    reference = monolith_solve(x[1:8], x[9:15], fx.u, fx.t, save; pit=fx.pit, cfg,
        t_load=5400.0, abstol=1e-10, reltol=1e-8)
    @test maximum(abs.(o.Tpit .- reference.Tpit)) < 0.01
    @test maximum(abs.(o.piece[:brisket].T_field .- reference.T_field)) < 0.01
    @test maximum(abs.(o.piece[:brisket].T_field[:, 2] .- cfg.T0)) < 1e-5
    @test o.Tpit[2] > fx.pit.Tpit0 + 30
    @test o.ignited[2] > x[15]
    ym, _ = E.propagate(model, y0, z, 0.0, 5700.0; first_slice=true)
    yr, _ = E.propagate(model, ym, z, 5700.0, 10800.0)
    @test maximum(abs.(belief_decode(yr, model.bl) .- belief_decode(y, model.bl))) < 0.01
    @test handoff(merge(reference, (ignited=ones(length(save)),)), 2)[2] == 1.0
    @test handoff(merge(reference, (ignited=fill(1e-10, length(save)),)), 2)[2] == 1e-10
    xb = belief_decode(y, model.bl)
    ia = first(E.belief_index(model.bl, :pit, :ignited))
    xb[ia] = 1.0
    @test belief_handoff(xb, model.bl).ignited == 1.0
    xb[ia] = 1e-10
    @test belief_handoff(xb, model.bl).ignited == 1e-10
end
