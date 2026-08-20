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

@testset "texas crutch (wrap blend) physics" begin
    hr = 3600.0
    tknots = collect(0.0:600.0:16hr)
    tsave = tknots
    θ = MEAT_θ_DEFAULT
    Tpit = fill(383.15, length(tknots))

    unwrapped = meat_solve(θ, Tpit, tknots, tsave)
    default_tw = meat_solve(θ, Tpit, tknots, tsave; t_wrap = WRAP_NEVER)
    wrapped = meat_solve(θ, Tpit, tknots, tsave; t_wrap = 5hr)

    # WRAP_NEVER default is the unwrapped model exactly (blend ≡ 0 bitwise)
    @test default_tw.Tcore == unwrapped.Tcore
    @test default_tw.j_evap == unwrapped.j_evap
    @test default_tw.w_retained == unwrapped.w_retained

    dt = tsave[2] - tsave[1]
    ∫j(o) = sum(o.j_evap) * dt
    # wrapping suppresses evaporation → less total moisture flux, more retained
    @test ∫j(wrapped) < 0.7 * ∫j(unwrapped)
    @test wrapped.w_retained > unwrapped.w_retained

    # the crutch does its job where it matters: it breaks the stall EARLY
    # (~14 K core advantage near 9 h with wrap at 5 h). With expulsion
    # modeled, the unwrapped surface dries and self-breaks its stall late
    # in the cook, so the advantage peaks mid-cook and CONVERGES by the
    # 16 h asymptote — where the foil's lower effective h and ε can even
    # cost ~0.5 K (the trapped-steam braise channel is unmodeled; see
    # docs/model_gaps.md). Assert the mid-cook stall-break, after t_wrap.
    adv = wrapped.Tcore .- unwrapped.Tcore
    @test maximum(adv) > 8.0
    @test tsave[argmax(adv)] > 5hr

    # post-wrap, evaporative flux collapses: the 90% suppression must beat
    # the (wetter, hotter) wrapped surface's raised raw demand by a wide
    # margin, so compare against the wrapped cook's own pre-wrap peak
    cfg = MeatConfig()
    pre = findall(t -> t < 5hr - 2 * cfg.δ_wrap, tsave)
    post = findall(t -> t > 5hr + 2 * cfg.δ_wrap, tsave)
    @test maximum(wrapped.j_evap[post]) < 0.25 * maximum(wrapped.j_evap[pre])
end

@testset "bark model physics" begin
    hr = 3600.0
    tk = collect(0.0:600.0:14hr)
    θ = MEAT_θ_DEFAULT
    smoke = fill(1.2e-4, length(tk))

    lo = meat_solve(θ, fill(383.15, length(tk)), tk, tk; smoke_knots = smoke)
    hi = meat_solve(θ, fill(408.15, length(tk)), tk, tk; smoke_knots = smoke)
    wr = meat_solve(θ, fill(383.15, length(tk)), tk, tk; smoke_knots = smoke,
                    t_wrap = 5hr)
    dry = meat_solve(θ, fill(383.15, length(tk)), tk, tk)  # no smoke input

    # states bounded and monotone; L* monotone darkening
    for o in (lo, hi, wr)
        @test all(0 .<= o.bark_b .< 1)
        @test all(0 .<= o.bark_s .< 1)
        # monotone up to solver interpolation noise (post-wrap dS ~ 0 and
        # FBDF dense output wiggles at ~1e-12)
        @test all(diff(o.bark_b) .> -1e-8) && all(diff(o.bark_s) .> -1e-8)
        @test issorted(o.l_star, rev = true)
    end
    # no smoke -> no deposition, browning unaffected
    @test all(dry.bark_s .== 0)
    # rtol 1e-6, not 1e-10: with S dynamics active the solver's step
    # sequence differs, so B matches to solver tolerance, not bitwise
    @test isapprox(dry.bark_b[end], lo.bark_b[end]; rtol = 1e-6)
    # hot cook Maillard-saturates; with expulsion drying the surface the
    # low-slow cook browns substantially too (deeper into the a_w bell),
    # so a fixed hi/lo ratio is the wrong invariant (hi pins near 1) —
    # assert saturation vs a clear sub-saturation margin instead
    @test hi.bark_b[end] > 0.95
    @test lo.bark_b[end] < 0.6 * hi.bark_b[end]
    @test lo.bark_s[end] > hi.bark_s[end]
    # wrapping freezes smoke deposition (the known wrap cost)
    @test wr.bark_s[end] < 0.75 * lo.bark_s[end]
    @test isapprox(wr.bark_b[end], lo.bark_b[end]; rtol = 0.3)
    # bark window: most deposition happens in the first half of the cook
    ihalf = findfirst(t -> t >= 7hr, tk)
    @test lo.bark_s[ihalf] > 0.6 * lo.bark_s[end]
end

@testset "moisture physics: fat rendering + expulsion" begin
    hr = 3600.0
    tk = collect(0.0:600.0:12hr)
    θ = MEAT_θ_DEFAULT
    cfg = MeatConfig()

    lo = meat_solve(θ, fill(383.15, length(tk)), tk, tk)   # 110 C pit
    hi = meat_solve(θ, fill(408.15, length(tk)), tk, tk)   # 135 C pit

    # mass closure: water lost = evaporated + dripped, algebraically exact
    for o in (lo, hi)
        water_lost = cfg.W0 * (1 - o.w_retained)
        @test isapprox(water_lost, o.evap_w + o.drip_w; atol = 1e-8)
        @test 0 <= o.fat_rendered <= 1
        @test o.drip_f ≈ cfg.η_drain * cfg.F0 * o.fat_rendered atol = 1e-10
        # juiciness = water + β·retained rendered fat (exceeds water-only)
        @test o.juiciness > o.w_retained
        @test o.juiciness ≈ o.w_retained +
              cfg.β_fat * (1 - cfg.η_drain) * cfg.F0 * o.fat_rendered / cfg.W0 atol = 1e-10
    end

    # hotter cook: more fat rendered, more expulsion, less water retained
    @test hi.fat_rendered > lo.fat_rendered
    @test hi.w_retained < lo.w_retained

    # total wet-basis mass loss in the realistic band for a long unwrapped
    # cook (~35-55%; packers commonly yield ~half of raw weight)
    for o in (lo, hi)
        wet0 = 1.0 + cfg.W0 + cfg.F0     # dry matter + water + fat
        loss = cfg.W0 * (1 - o.w_retained) + o.drip_f
        @test 0.30 < loss / wet0 < 0.60
    end

    # expulsion is one-sided: a cool cook (60 C pit) barely purges
    cool = meat_solve(θ, fill(333.15, length(tk)), tk, tk)
    @test cool.w_retained > lo.w_retained
    @test cool.fat_rendered < 0.5
end
