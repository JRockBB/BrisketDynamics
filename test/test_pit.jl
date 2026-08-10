# Milestone A2: WSM-class sanity anchor — a full charcoal ring at a fixed
# moderate damper holds the 225–275 °F band for 8+ hours.

@testset "pit forward physics" begin
    hr = 3600.0
    tsave = collect(0.0:300.0:12hr)
    u_tknots = collect(range(0, 12hr, length = 16))
    φ = PIT_φ_DEFAULT

    # moderate fixed damper, no meat load
    pit = pit_solve(φ, fill(0.4, 16), u_tknots, zeros(length(tsave)), tsave, tsave)
    TpitF = (pit.Tpit .- 273.15) .* 9 / 5 .+ 32
    dt_h = 300.0 / hr

    @test count(f -> 225 <= f <= 275, TpitF) * dt_h > 8.0     # WSM hold
    @test issorted(pit.m_fuel, rev = true)                     # fuel only burns
    @test pit.m_fuel[end] > 0.0                                # doesn't run dry at u=0.4

    # damper authority: wide-open runs hotter than nearly-closed
    hot = pit_solve(φ, fill(1.0, 16), u_tknots, zeros(length(tsave)), tsave, tsave)
    cool = pit_solve(φ, fill(0.0, 16), u_tknots, zeros(length(tsave)), tsave, tsave)
    @test maximum(hot.Tpit) > maximum(cool.Tpit) + 30          # ≥30 K of control authority

    # meat load pulls the pit down
    loaded = pit_solve(φ, fill(0.4, 16), u_tknots, fill(150.0, length(tsave)), tsave, tsave)
    @test loaded.Tpit[end] < pit.Tpit[end]
end
