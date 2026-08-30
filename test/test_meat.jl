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
    @test issorted(out.W_field[end, :], rev=true)         # surface moisture monotone ↓
    # fridge-cold meat below the pit dew point condenses first, then evaporates
    @test out.j_evap[1] <= 0 && all(out.j_evap[(end - 10):end] .>= 0)

    # the cook makes progress: core well past 80 °C by 16 h, and the stall BROKE
    # (post-stall climb resumed — the finite moisture reservoir mechanism)
    @test TcoreC[end] > 80

    # THE STALL: longest contiguous window with core rise < 1.5 °C/h while
    # core is between 60 and 82 °C must exceed 2 h
    dt_h = (tsave[2] - tsave[1]) / hr
    rate = diff(TcoreC) ./ dt_h
    in_stall = [rate[i] < 1.5 && 60 <= TcoreC[i] <= 82 for i in eachindex(rate)]
    best = 0;
    cur = 0
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
    default_tw = meat_solve(θ, Tpit, tknots, tsave; t_wrap=WRAP_NEVER)
    wrapped = meat_solve(θ, Tpit, tknots, tsave; t_wrap=5hr)

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
    # docs/roadmap.md). Assert the mid-cook stall-break, after t_wrap.
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

    lo = meat_solve(θ, fill(383.15, length(tk)), tk, tk; smoke_knots=smoke)
    hi = meat_solve(θ, fill(408.15, length(tk)), tk, tk; smoke_knots=smoke)
    wr = meat_solve(θ, fill(383.15, length(tk)), tk, tk; smoke_knots=smoke, t_wrap=5hr)
    dry = meat_solve(θ, fill(383.15, length(tk)), tk, tk)  # no smoke input

    # states bounded and monotone; L* monotone darkening
    for o in (lo, hi, wr)
        @test all(0 .<= o.bark_b .< 1)
        @test all(0 .<= o.bark_s .< 1)
        # monotone up to solver interpolation noise (post-wrap dS ~ 0 and
        # FBDF dense output wiggles at ~1e-12)
        @test all(diff(o.bark_b) .> -1e-8) && all(diff(o.bark_s) .> -1e-8)
        @test issorted(o.l_star, rev=true)
    end
    # no smoke -> no deposition, browning unaffected
    @test all(dry.bark_s .== 0)
    # rtol 1e-6, not 1e-10: with S dynamics active the solver's step
    # sequence differs, so B matches to solver tolerance, not bitwise
    @test isapprox(dry.bark_b[end], lo.bark_b[end]; rtol=1e-6)
    # hot cook Maillard-saturates; with expulsion drying the surface the
    # low-slow cook browns substantially too (deeper into the a_w bell),
    # so a fixed hi/lo ratio is the wrong invariant (hi pins near 1) —
    # assert saturation vs a clear sub-saturation margin instead
    @test hi.bark_b[end] > 0.95
    @test lo.bark_b[end] < 0.6 * hi.bark_b[end]
    @test lo.bark_s[end] > hi.bark_s[end]
    # wrapping freezes smoke deposition (the known wrap cost)
    @test wr.bark_s[end] < 0.75 * lo.bark_s[end]
    @test isapprox(wr.bark_b[end], lo.bark_b[end]; rtol=0.3)
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

    # mass closure is DEFINITIONAL (drip_w := water_lost - evap_w); this
    # checks the algebra, not PDE conservation of the W field.
    for o in (lo, hi)
        water_lost = cfg.W0 * (1 - o.w_retained)
        @test isapprox(water_lost, o.evap_w + o.drip_w; atol=1e-8)
        @test 0 <= o.fat_rendered <= 1
        @test o.drip_f ≈ cfg.η_drain * cfg.F0 * o.fat_rendered atol = 1e-10
        # juiciness = water + β·retained rendered fat (exceeds water-only)
        @test o.juiciness > o.w_retained
        @test o.juiciness ≈
            o.w_retained +
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

@testset "state handoff continuity (MPC replan invariant)" begin
    # A cook split at 6 h with a FULL state handoff (T/W/C/F fields + bark
    # scalars) must reproduce the straight 12 h cook: since the juiciness
    # build, fat and bark restarting at defaults would re-render fat and
    # zero bark every replanned segment, silently corrupting the terms the
    # default objective optimizes. Tolerances cover the solver restart
    # transient only.
    hr = 3600.0
    tk = collect(0.0:600.0:12hr)
    θ = MEAT_θ_DEFAULT
    Tpit = collect(range(378.0, 388.0; length=length(tk)))
    smoke = fill(1.2e-4, length(tk))
    whole = meat_solve(θ, Tpit, tk, tk; smoke_knots=smoke)

    isplit = findfirst(t -> t >= 6hr, tk)
    tk1 = tk[1:isplit]
    seg1 = meat_solve(θ, Tpit[1:isplit], tk1, tk1; smoke_knots=smoke[1:isplit])
    # second segment: shift times to start at 0, hand off the full state
    tk2 = tk[isplit:end] .- tk[isplit]
    seg2 = meat_solve(
        θ,
        Tpit[isplit:end],
        tk2,
        tk2;
        smoke_knots=smoke[isplit:end],
        T0f=seg1.T_field[:, end],
        W0f=seg1.W_field[:, end],
        C0f=seg1.C_field[:, end],
        F0f=seg1.F_field[:, end],
        B0=seg1.bark_b[end],
        S0=seg1.bark_s[end],
        R0=seg1.ring[end],
    )

    @test abs(seg2.Tcore[end] - whole.Tcore[end]) < 0.05        # K
    @test abs(seg2.ring[end] - whole.ring[end]) < 1e-4          # m
    @test abs(seg2.w_retained - whole.w_retained) < 1e-3
    @test abs(seg2.fat_rendered - whole.fat_rendered) < 1e-3
    @test abs(seg2.juiciness - whole.juiciness) < 2e-3
    @test abs(seg2.bark_b[end] - whole.bark_b[end]) < 1e-3
    @test abs(seg2.bark_s[end] - whole.bark_s[end]) < 1e-3

    # the OLD behavior (fat/bark restart at defaults) must NOT match —
    # this is the bug the handoff fixes, kept as a tripwire
    seg2_old = meat_solve(
        θ,
        Tpit[isplit:end],
        tk2,
        tk2;
        smoke_knots=smoke[isplit:end],
        T0f=seg1.T_field[:, end],
        W0f=seg1.W_field[:, end],
        C0f=seg1.C_field[:, end],
    )
    # the tripwire invariant is DISCONTINUITY, not a specific ratio: both
    # bark integrators partially rebuild on their own (measured: B 0.218
    # vs 0.251, S 0.635 vs 0.891), so assert the defaults-restart error
    # exceeds the handoff's continuity tolerance by two orders of
    # magnitude — that gap is what the handoff fix closes
    @test seg2_old.fat_rendered < whole.fat_rendered - 0.05
    @test abs(seg2_old.bark_s[end] - whole.bark_s[end]) > 0.1
end

@testset "rest / carryover (pull blend) physics" begin
    # docs/model.md, rest: after t_pull the surface sees the rest
    # environment; the interior equations are unchanged and must produce
    # carryover, continued collagen conversion, and surface rewetting.
    hr = 3600.0
    tk = collect(0.0:600.0:14hr)
    θ = MEAT_θ_DEFAULT
    Tpit = fill(383.15, length(tk))
    smoke = fill(1.2e-4, length(tk))
    cfg = MeatConfig()                       # rest defaults = faux cambro

    inpit = meat_solve(θ, Tpit, tk, tk; smoke_knots=smoke)
    never = meat_solve(θ, Tpit, tk, tk; smoke_knots=smoke, t_pull=PULL_NEVER)
    # PULL_NEVER: blend ≡ 0 exactly
    @test never.Tcore == inpit.Tcore
    @test never.j_evap == inpit.j_evap
    @test never.q_meat == inpit.q_meat
    @test never.C_field_final == inpit.C_field_final
    @test never.bark_s == inpit.bark_s

    t_pull = 10hr                            # 4 h rest follows
    ip = findfirst(t -> t >= t_pull, tk)
    cambro = meat_solve(θ, Tpit, tk, tk; smoke_knots=smoke, t_pull=t_pull)
    board = meat_solve(
        θ,
        Tpit,
        tk,
        tk;
        smoke_knots=smoke,
        t_pull=t_pull,
        cfg=MeatConfig(h_rest=7.0, ε_rest=0.85, η_rest=0.0),
    )

    # unchanged cook before the blend opens (pl < 4e-4 before t_pull - 8δ);
    # the core cannot see the last minutes of the surface change anyway
    ipre = findlast(t -> t < t_pull - 8cfg.δ_pull, tk)
    @test maximum(abs.(cambro.Tcore[1:ipre] .- inpit.Tcore[1:ipre])) < 1e-2
    @test cambro.C_field[1, ipre] ≈ inpit.C_field[1, ipre] rtol = 1e-4

    # carryover scales with the internal gradient at the pull: a hot-fast
    # cook pulled early carries over at least as much as the low-slow stall
    # (whose interior is nearly isothermal); afterwards the core cools, the
    # bare board much faster than the cambro
    rest = ip:length(tk)
    rise_lo = maximum(cambro.Tcore[rest]) - cambro.Tcore[ip]
    ip_hf = findfirst(t -> t >= 6hr, tk)
    hotfast = meat_solve(θ, fill(408.15, length(tk)), tk, tk; t_pull=6hr)
    rise_hf = maximum(hotfast.Tcore[ip_hf:end]) - hotfast.Tcore[ip_hf]
    @test rise_hf >= rise_lo
    @test cambro.Tcore[end] < cambro.Tcore[ip] - 0.2
    @test board.Tcore[end] < cambro.Tcore[end] - 5.0

    # doneness keeps converting through the hold — the reason the deadline
    # formulation leans on the rest
    @test cambro.C_field[1, end] > cambro.C_field[1, ip] + 0.02

    # juice redistribution: the wrapped rest rewets the surface from the
    # interior (evaporation blocked, diffusion continues)
    @test cambro.W_field[end, end] > cambro.W_field[end, ip]

    # once pulled the pit sees no load and no smoke deposits
    ipost = findfirst(t -> t >= t_pull + 8cfg.δ_pull, tk)
    @test maximum(abs.(cambro.q_meat[ipost:end])) < 0.01 * maximum(abs.(inpit.q_meat))
    @test cambro.bark_s[end] < cambro.bark_s[ipost] + 1e-3
end

@testset "shrinkage (material coordinate) physics" begin
    # docs/model.md, shrinkage: dry-matter coordinate, thickness stretch
    # st = (v/v0)^n_shrink, exposed area A(t)/A0 = (v̄/v0)^(1 - n_shrink).
    hr = 3600.0
    tk = collect(0.0:600.0:16hr)
    θ = MEAT_θ_DEFAULT
    L0 = θ[4]
    Tpit = fill(383.15, length(tk))
    cfg0 = MeatConfig()
    # the density bookkeeping closes with the fixed constants at the raw state
    v0 = BrisketDynamics.specific_volume(cfg0.W0, cfg0.F0, cfg0)
    @test isapprox((1 + cfg0.W0) / v0, BrisketDynamics.RHO_MEAT; rtol=0.03)
    @test isapprox(1 / v0, BrisketDynamics.RHO_DRY; rtol=0.03)

    iso = meat_solve(θ, Tpit, tk, tk)                             # default: 1/3
    fixed = meat_solve(θ, Tpit, tk, tk; cfg=MeatConfig(n_shrink=0.0))  # old slab
    thick = meat_solve(θ, Tpit, tk, tk; cfg=MeatConfig(n_shrink=1.0))

    # n_shrink = 0: the slab never moves
    @test all(fixed.l_thick .== L0)
    # the thickness shrinks monotonically (to interpolation noise) into the
    # plausible band, more so when the thickness takes all the volume loss
    @test all(diff(iso.l_thick) .< 1e-9)
    @test 0.65 < iso.l_thick[end] / L0 < 0.90
    @test thick.l_thick[end] < iso.l_thick[end]
    @test iso.l_thick[1] ≈ L0 rtol = 1e-12          # raw meat: no shrinkage yet
    # the stall at the same θ: the contracted dry layer resupplies the
    # surface ~1/ς² faster and the exposed area shrinks, so the surface
    # stays wet longer — the plateau is LONGER than the fixed slab's;
    # once the surface dries, the thinner slab climbs faster
    function stall_h(o)
        TcC = o.Tcore .- 273.15
        dt_h = (tk[2] - tk[1]) / hr
        rate = diff(TcC) ./ dt_h
        best = 0
        cur = 0
        for i in eachindex(rate)
            cur = (rate[i] < 1.5 && 60 <= TcC[i] <= 82) ? cur + 1 : 0
            best = max(best, cur)
        end
        return best * dt_h
    end
    @test stall_h(iso) > stall_h(fixed)
    @test stall_h(iso) > 2.0                        # and it is still a stall
    @test iso.Tcore[end] > 273.15 + 80              # that still breaks by 16 h
    # dry-mass bookkeeping is unchanged: closure and the budgets still hold
    for o in (iso, thick)
        @test isapprox(cfg0.W0 * (1 - o.w_retained), o.evap_w + o.drip_w; atol=1e-8)
        @test 0 < o.w_retained < 1
    end
    # the pit sees a shrinking exposed area: early in the cook (identical
    # surface states) the shrunken slab already draws slightly less heat
    i2h = findfirst(t -> t >= 2hr, tk)
    @test iso.q_meat[i2h] < fixed.q_meat[i2h]
    # state handoff stays exact in the material coordinate: split at 6 h
    cfg = MeatConfig()
    isplit = findfirst(t -> t >= 6hr, tk)
    tk1 = tk[1:isplit]
    seg1 = meat_solve(θ, Tpit[1:isplit], tk1, tk1; cfg)
    tk2 = tk[isplit:end] .- tk[isplit]
    seg2 = meat_solve(
        θ, Tpit[isplit:end], tk2, tk2; cfg,
        T0f=seg1.T_field[:, end], W0f=seg1.W_field[:, end],
        C0f=seg1.C_field[:, end], F0f=seg1.F_field[:, end],
        B0=seg1.bark_b[end], S0=seg1.bark_s[end],
    )
    @test abs(seg2.Tcore[end] - iso.Tcore[end]) < 0.05
    @test abs(seg2.l_thick[end] - iso.l_thick[end]) < 1e-5
end

@testset "humidity feedback physics" begin
    # docs/model.md, pit humidity: p_v,pit = pv_base + c·j, j in closed form.
    hr = 3600.0
    tk = collect(0.0:600.0:14hr)
    n = length(tk)
    θ = MEAT_θ_DEFAULT
    Tpit = fill(383.15, n)
    cfg = MeatConfig()

    # the two-step Newton root vs bisection on f(j) = j − a·sp(D − c j);
    # over the cooking range the clamp is inactive (p_sat ≫ p_v) and the
    # closed form is exact, so Newton is a rounding-level correction
    for (Ts, Ws, c) in ((340.0, 1.2, 0.0), (345.0, 0.6, 5e6), (350.0, 0.45, 2e7), (335.0, 2.0, 1e8))
        a = θ[2] * BrisketDynamics.logistic((Ws - cfg.W_crit) / cfg.δW) / (BrisketDynamics.R_V * Ts)
        D = p_sat(Ts) - 3000.0
        f(j) = j - a * BrisketDynamics.softclamp(D - c * j, cfg.δp)
        lo, hi = 0.0, a * BrisketDynamics.softclamp(D, cfg.δp) + 1e-12
        for _ in 1:200
            mid = (lo + hi) / 2
            f(mid) > 0 ? (hi = mid) : (lo = mid)
        end
        jfb = BrisketDynamics.evap_flux_fb(Ts, Ws, θ[2], 3000.0, c, cfg)
        @test isapprox(jfb, (lo + hi) / 2; rtol=1e-9)
    end

    # a humid pit evaporates less and its plateau sits higher
    dry = meat_solve(θ, Tpit, tk, tk; pv_base_knots=fill(1000.0, n))
    humid = meat_solve(θ, Tpit, tk, tk; pv_base_knots=fill(6000.0, n))
    i4 = findfirst(t -> t >= 4hr, tk)
    @test sum(humid.j_evap) < sum(dry.j_evap)
    @test humid.Tsurf[i4] > dry.Tsurf[i4]
    @test humid.Tcore[i4] > dry.Tcore[i4]

    # self-humidification: the meat's own vapor raises the pit humidity and
    # throttles its own evaporation (ṁ_air = 2.5e-3 kg/s: a WSM at low fan duty)
    fb = meat_solve(
        θ, Tpit, tk, tk; pv_base_knots=fill(3000.0, n), imair_knots=fill(1 / 2.5e-3, n)
    )
    nofb = meat_solve(θ, Tpit, tk, tk; pv_base_knots=fill(3000.0, n))
    @test sum(fb.j_evap) < sum(nofb.j_evap)
    @test all((fb.pv_pit .- 3000.0) .* sign.(fb.j_evap) .>= -1e-9)  # vapor follows the flux sign
    @test maximum(fb.pv_pit) > 3500.0
    @test all(nofb.pv_pit .== 3000.0)
end

@testset "condensation physics (two-sided surface flux)" begin
    θ = MEAT_θ_DEFAULT
    cfg = MeatConfig()
    k_m = θ[2]
    # a 10 °C surface under 3 kPa of pit vapor is below the dew point:
    # the flux is condensation at the ungated mass-transfer rate
    Ts, pv = 283.15, 3000.0
    j_dry = BrisketDynamics.evap_flux(Ts, 0.2, k_m, pv, cfg)
    j_wet = BrisketDynamics.evap_flux(Ts, 2.0, k_m, pv, cfg)
    j_ref = -k_m * (pv - p_sat(Ts)) / (BrisketDynamics.R_V * Ts)
    @test j_dry < 0
    @test isapprox(j_dry, j_ref; rtol=1e-6)
    @test isapprox(j_dry, j_wet; rtol=1e-6)  # availability gates evaporation only
    # above the dew point the old one-sided evaporation is recovered
    @test BrisketDynamics.evap_flux(340.0, 1.2, k_m, pv, cfg) ≈
          k_m * BrisketDynamics.logistic((1.2 - cfg.W_crit) / cfg.δW) *
          (p_sat(340.0) - pv) / (BrisketDynamics.R_V * 340.0) rtol = 1e-9
    # with the pit feedback, condensation lowers the pit humidity: the root
    # is self-limiting and satisfies j = F(D0 - c j)
    c = 5e6
    jfb = BrisketDynamics.evap_flux_fb(Ts, 0.2, k_m, pv, c, cfg)
    @test jfb < 0 && abs(jfb) < abs(j_dry)
    D = p_sat(Ts) - pv - c * jfb
    F = -k_m * BrisketDynamics.softclamp(-D, cfg.δp) / (BrisketDynamics.R_V * Ts) +
        k_m * BrisketDynamics.logistic((0.2 - cfg.W_crit) / cfg.δW) *
        BrisketDynamics.softclamp(D, cfg.δp) / (BrisketDynamics.R_V * Ts)
    @test isapprox(jfb, F; rtol=1e-9)

    # fridge-cold meat into a humid pit (dew point ≈ 36 °C): the surface
    # gains water and warms by condensation over the first steps; in a dry
    # pit (dew point below the meat) it loses water from the start
    hr = 3600.0
    tk = collect(0.0:60.0:2hr)
    Tpit = fill(383.15, length(tk))
    humid = meat_solve(θ, Tpit, tk, tk; pv_base_knots=fill(6000.0, length(tk)))
    dry = meat_solve(θ, Tpit, tk, tk; pv_base_knots=fill(1000.0, length(tk)))
    @test humid.j_evap[1] < 0
    @test humid.W_field[end, 3] > humid.W_field[end, 1]
    @test dry.W_field[end, 3] < dry.W_field[end, 1]
    @test humid.Tsurf[5] > dry.Tsurf[5]
    # and it is a transient: once the surface passes the dew point the pit
    # dries it again
    @test humid.j_evap[end] > 0
end

@testset "draft convection h_c(u), k_m(u)" begin
    θ = MEAT_θ_DEFAULT
    cfg = MeatConfig()
    hr = 3600.0
    tk = collect(0.0:600.0:6hr)
    n = length(tk)
    Tpit = fill(383.15, n)
    # the reference airflow reproduces the unscaled coefficients exactly
    ref = meat_solve(θ, Tpit, tk, tk)
    at_ref = meat_solve(θ, Tpit, tk, tk; airflow_knots=fill(cfg.g_ref, n))
    @test ref.Tsurf == at_ref.Tsurf
    # more draft: stronger convection heats the surface faster and dries it more
    open = meat_solve(θ, Tpit, tk, tk; airflow_knots=fill(1.0, n))
    closed = meat_solve(θ, Tpit, tk, tk; airflow_knots=fill(0.25, n))
    i1 = findfirst(t -> t >= 1hr, tk)
    @test open.Tsurf[i1] > closed.Tsurf[i1]
    @test sum(open.j_evap) > sum(closed.j_evap)
    # m_conv = 0 switches the dependence off
    θ0 = copy(θ)
    θ0[8] = 0.0
    a = meat_solve(θ0, Tpit, tk, tk; airflow_knots=fill(1.0, n))
    b = meat_solve(θ0, Tpit, tk, tk; airflow_knots=fill(0.25, n))
    @test a.Tsurf == b.Tsurf
end

@testset "smoke ring (NO penetration front)" begin
    θ = MEAT_θ_DEFAULT
    cfg = MeatConfig()
    hr = 3600.0
    tk = collect(0.0:600.0:8hr)
    n = length(tk)
    Tpit = fill(383.15, n)
    lit = fill(1.2e-4, n)
    out = meat_solve(θ, Tpit, tk, tk; smoke_knots=lit)
    r = out.ring
    @test r[1] == 0.0 && all(diff(r) .>= -1e-12)          # monotone growth from zero
    @test 1e-3 < r[end] < 1.5e-2                          # millimetres, not metres
    # the front stalls once the meat at its depth passes the denaturation gate
    i4, i8 = findfirst(t -> t >= 4hr, tk), n
    @test r[i8] - r[i4] < 0.1 * r[i4]
    # no fire, foil from the start, or a wrapped cook: no ring
    @test all(meat_solve(θ, Tpit, tk, tk).ring .== 0.0)
    @test maximum(meat_solve(θ, Tpit, tk, tk; smoke_knots=lit, t_wrap=-6hr).ring) < 1e-4
    # the surface must be wet for uptake: a dried-out surface stops it
    cfg_dry = MeatConfig(; a_dry=1.5)  # the wet switch never opens
    @test maximum(meat_solve(θ, Tpit, tk, tk; cfg=cfg_dry, smoke_knots=lit).ring) < 1e-4
end

@testset "spritz policy" begin
    θ = MEAT_θ_DEFAULT
    cfg = MeatConfig()
    hr = 3600.0
    tk = collect(0.0:300.0:8hr)
    n = length(tk)
    Tpit = fill(383.15, n)
    lit = fill(1.2e-4, n)
    dry = meat_solve(θ, Tpit, tk, tk; smoke_knots=lit)
    wet = meat_solve(θ, Tpit, tk, tk; smoke_knots=lit, t_spritz=2hr, dt_spritz=1800.0)
    # every pulse wets the surface: across the 2 h pulse the surface water
    # gains on the un-spritzed run (the absorbed fraction; the rest flashes off)
    Ws = wet.W_field[end, :]
    Wd = dry.W_field[end, :]
    i = findfirst(t -> t >= 2hr + 5 * 60, tk)
    @test Ws[i] - Ws[i - 2] > Wd[i] - Wd[i - 2]
    @test wet.spritz_w > 0 && dry.spritz_w == 0
    # the water it costs: a cooler surface, a longer stall, more smoke on a wet surface
    i6 = findfirst(t -> t >= 6hr, tk)
    @test wet.Tsurf[i6] < dry.Tsurf[i6]
    @test wet.Tcore[end] < dry.Tcore[end]
    @test wet.bark_s[end] > dry.bark_s[end]
    # spritzing stops at the wrap
    wrapped = meat_solve(θ, Tpit, tk, tk; smoke_knots=lit, t_spritz=2hr, dt_spritz=1800.0, t_wrap=4hr)
    Ww = wrapped.W_field[end, :]
    i5 = findfirst(t -> t >= 5hr, tk)
    @test maximum(diff(Ww[i5:end])) < 0.5 * maximum(diff(Ws[1:i5]))
    # never = the default, bitwise
    @test meat_solve(θ, Tpit, tk, tk; smoke_knots=lit, t_spritz=BrisketDynamics.SPRITZ_NEVER).Tsurf == dry.Tsurf
end
