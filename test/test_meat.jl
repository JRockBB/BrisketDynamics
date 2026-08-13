# Milestone A1: the stall must appear at literature-plausible parameters,
# and basic physics sanity must hold.

@testset "meat forward physics" begin
    hr = 3600.0
    tknots = collect(0.0:300.0:16hr)               # a stalled cook at 230 °F needs ~16 h
    tsave = tknots
    θ = MEAT_θ_DEFAULT
    Tpit = fill(383.15, length(tknots))            # constant 110 °C pit

    out = meat_solve(θ, Tpit, tknots, tsave)
    TcoreC = out.Tcore .- 273.15

    # sanity: temperatures bounded by pit temperature, moisture non-increasing at surface
    @test all(out.Tcore .<= 383.15 + 0.1)
    @test all(out.Tsurf .<= 383.15 + 0.1)
    @test issorted(out.W_field[end, :], rev = true)         # surface moisture monotone ↓
    @test all(out.j_evap .>= 0)

    # the cook makes progress: core well past 80 °C by 16 h, and the stall BROKE
    # (post-stall climb resumed — the finite moisture reservoir mechanism)
    @test TcoreC[end] > 80

    # THE STALL: longest contiguous window with core rise < 1.5 °C/h while
    # core is between 60 and 82 °C must exceed 2 h
    dt_h = (tsave[2] - tsave[1]) / hr
    rate = diff(TcoreC) ./ dt_h
    in_stall = [rate[i] < 1.5 && 60 <= TcoreC[i] <= 82 for i in eachindex(rate)]
    best = 0; cur = 0
    for b in in_stall
        cur = b ? cur + 1 : 0
        best = max(best, cur)
    end
    @test best * dt_h > 2.0

    # evaporative cooling is the mechanism: peak evaporative heat flux is
    # comparable to (>30% of) the peak convective+radiative input
    q_evap_peak = maximum(2.3e6 .* out.j_evap) * MeatConfig().A_meat
    @test q_evap_peak > 0.3 * maximum(out.q_meat)

    # doneness: collagen conversion happened but is not complete at the core
    @test 0.05 < out.C_field_final[1] < 1.0

    # juiciness proxy: bounded, and losing moisture over the cook
    @test 0.0 < out.w_retained < 1.0
    @test out.w_retained < 0.999        # some moisture actually left
end
