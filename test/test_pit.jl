# Milestone A2: WSM-class sanity anchor — a full charcoal ring at a fixed
# moderate damper holds the 225–275 °F band for 8+ hours.

@testset "pit forward physics" begin
    hr = 3600.0
    tsave = collect(0.0:300.0:12hr)
    u_tknots = collect(range(0, 12hr, length=16))
    φ = PIT_φ_DEFAULT

    # moderate fixed damper, no meat load
    pit = pit_solve(φ, fill(0.4, 16), u_tknots, zeros(length(tsave)), tsave, tsave)
    TpitF = (pit.Tpit .- 273.15) .* 9 / 5 .+ 32
    dt_h = 300.0 / hr

    @test count(f -> 225 <= f <= 275, TpitF) * dt_h > 8.0     # WSM hold
    @test issorted(pit.m_fuel, rev=true)                     # fuel only burns
    @test pit.m_fuel[end] > 0.0                                # doesn't run dry at u=0.4

    # damper authority: wide-open runs hotter than nearly-closed
    hot = pit_solve(φ, fill(1.0, 16), u_tknots, zeros(length(tsave)), tsave, tsave)
    cool = pit_solve(φ, fill(0.0, 16), u_tknots, zeros(length(tsave)), tsave, tsave)
    @test maximum(hot.Tpit) > maximum(cool.Tpit) + 30          # ≥30 K of control authority

    # meat load pulls the pit down
    loaded = pit_solve(φ, fill(0.4, 16), u_tknots, fill(150.0, length(tsave)), tsave, tsave)
    @test loaded.Tpit[end] < pit.Tpit[end]

    # paired-length mismatch must fail loudly (silent p-slicing is a C3 trap)
    @test_throws ArgumentError pit_solve(
        φ, fill(0.4, 8), u_tknots, zeros(length(tsave)), tsave, tsave
    )
    @test_throws ArgumentError pit_solve(
        φ, fill(0.4, 16), u_tknots, zeros(10), tsave, tsave
    )
end

@testset "mid-cook fuel addition" begin
    hr = 3600.0
    tsave = collect(0.0:300.0:8hr)
    u_tknots = collect(range(0, 8hr, length=16))
    φ = PIT_φ_DEFAULT
    # a small ring burns out by mid-cook; a chimney at 3 h keeps the pit up
    cfg = PitConfig(; m0=1.0)
    dry = pit_solve(φ, fill(0.4, 16), u_tknots, zeros(length(tsave)), tsave, tsave; cfg)
    fed = pit_solve(φ, fill(0.4, 16), u_tknots, zeros(length(tsave)), tsave, tsave; cfg, t_add=3hr)
    i2, i4 = findfirst(t -> t >= 2hr, tsave), findfirst(t -> t >= 4hr, tsave)
    i3q = findfirst(t -> t >= 3.25hr, tsave)
    @test abs(fed.m_fuel[i2] - dry.m_fuel[i2]) < 1e-3            # nothing before the add (pulse tail)
    @test fed.m_fuel[i3q] - dry.m_fuel[i3q] > 0.8 * cfg.m_chimney  # the chimney arrived
    @test fed.m_fuel[i4] < fed.m_fuel[i3q]                       # and burns
    @test fed.Tpit[i4] > dry.Tpit[i4] + 10                       # the fire lived on
    # the lid came off for two minutes to dump it: right after, the pit is
    # below what the same addition through a closed lid would give
    sealed = pit_solve(φ, fill(0.4, 16), u_tknots, zeros(length(tsave)), tsave, tsave; cfg=PitConfig(; m0=1.0, UA_open=0.0), t_add=3hr)
    i3s = findfirst(t -> t >= 3hr + 300, tsave)
    @test fed.Tpit[i3s] < sealed.Tpit[i3s] - 5
    @test all(0 .<= fed.ignited .<= 1 + 1e-9)
    # never = the default, bitwise
    never = pit_solve(φ, fill(0.4, 16), u_tknots, zeros(length(tsave)), tsave, tsave; cfg, t_add=BrisketDynamics.ADD_NEVER)
    @test never.m_fuel == dry.m_fuel
end
