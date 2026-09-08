using BrisketDynamics: monolith_solve, monolith_solve_tf, monolith_jacobian, picard_cook, Humidity,
    monolith_stack, monolith_stack_jvp, monolith_stack_jvp_tf, FRAC_NEVER

const MONO_T_F = 8 * 3600.0
const MONO_TSAVE = collect(0.0:600.0:MONO_T_F)
const MONO_UTK = collect(range(0.0, MONO_T_F; length=8))
const MONO_U = fill(0.5, 8)
const MONO_PIT = PitConfig(; T_amb=300.0)

@testset "monolith: one coupled solve vs the Picard exchange" begin
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    mono = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT)
    @test length(mono.Tpit) == length(MONO_TSAVE)
    @test all(isfinite, mono.Tcore) && all(isfinite, mono.Tpit)
    @test mono.Tcore[end] > mono.Tcore[1] + 30
    @test mono.m_fuel[end] < mono.m_fuel[1]
    # the meat adds vapor while evaporating and removes it while condensing
    @test all((mono.pv_pit .- mono.pv_base) .* sign.(mono.j_evap) .>= -1e-9)

    # Picard converges TOWARD the monolith as K grows; the residual left at
    # K=8 is the exchange-grid interpolation, not the coupling truncation
    d = Dict{Int,NTuple{2,Float64}}()
    for K in (1, 2, 4, 8)
        pc = picard_cook(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; K, pit=MONO_PIT)
        d[K] = (
            maximum(abs.(pc.pit.Tpit .- mono.Tpit)),
            maximum(abs.(pc.meat.Tcore .- mono.Tcore)),
        )
    end
    @info "monolith vs Picard(K): max|ΔT_pit|, max|ΔT_core| [K]" d[1] d[2] d[4] d[8]
    @test d[8][1] < d[2][1]
    @test d[8][2] < d[1][2]  # the core sits on the interpolation floor by K = 2
    @test d[8][1] < 2.0
    # on a fine exchange grid the interpolation floor drops and K = 8 must
    # match the monolith to hundredths of a kelvin (measured: millikelvin)
    tsf = collect(0.0:60.0:MONO_TSAVE[end])
    mono_f = monolith_solve(θ, φ, MONO_U, MONO_UTK, tsf; pit=MONO_PIT)
    pc8 = picard_cook(θ, φ, MONO_U, MONO_UTK, tsf; K=8, pit=MONO_PIT)
    d8f = (
        maximum(abs.(pc8.pit.Tpit .- mono_f.Tpit)),
        maximum(abs.(pc8.meat.Tcore .- mono_f.Tcore)),
    )
    @info "fine exchange grid (60 s), K = 8: max|ΔT_pit|, max|ΔT_core| [K]" d8f
    @test d8f[1] < 0.02 && d8f[2] < 0.02
    @test d[8][2] < 0.5

    # cost: one coupled solve vs the two-pass exchange (both warm)
    t_mono = @elapsed monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT)
    t_pic = @elapsed picard_cook(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; K=2, pit=MONO_PIT)
    @info "wall: monolith $(round(t_mono; digits=3)) s, Picard K=2 $(round(t_pic; digits=3)) s"
end

@testset "monolith: forward-mode Jacobian vs finite differences" begin
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    f = o -> vcat(o.Tpit, o.Tcore)
    t_j = @elapsed J = monolith_jacobian(f, θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT)
    @test size(J) == (2 * length(MONO_TSAVE), 15)
    @test all(isfinite, J)
    @info "wall: 14-column forward Jacobian $(round(t_j; digits=2)) s (incl. compile)"
    x = vcat(θ, φ)
    for (j, name) in ((1, "D_eff"), (3, "h_c"), (8, "m_conv"), (10, "UA"), (13, "u_leak"))
        h = 1e-3 * abs(x[j])
        xp, xm = copy(x), copy(x)
        xp[j] += h
        xm[j] -= h
        fp = f(monolith_solve(xp[1:8], xp[9:15], MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT))
        fm = f(monolith_solve(xm[1:8], xm[9:15], MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT))
        fd = (fp .- fm) ./ (2h)
        err = maximum(abs.(J[:, j] .- fd)) / (maximum(abs.(fd)) + 1e-12)
        @info "FD gate $name: relative error $(round(err; sigdigits=3))"
        @test err < 2e-2
    end
end

@testset "monolith: stacked Jacobian and JVP (the Tesseract's derivative endpoints)" begin
    using BrisketDynamics: monolith_stack, monolith_stack_jacobian, monolith_stack_jvp
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    nu = length(MONO_U)
    tw, tp = 3 * 3600.0, 7 * 3600.0
    ta, ts, dts, msp = 5 * 3600.0, 2 * 3600.0, 1800.0, 0.05
    sel = [1, 10, 16, 15 + nu + 1, 15 + nu + 2, 15 + nu + 3, 15 + nu + 4, 15 + nu + 5, 15 + nu + 6]  # D_eff, UA, u_knots[1], t_wrap, t_pull, t_add, t_spritz, dt_spritz, m_spritz
    J = monolith_stack_jacobian(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE, tw, tp, ta, ts, dts, msp, sel; pit=MONO_PIT)
    o = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT, t_wrap=tw, t_pull=tp, t_add=ta, t_spritz=ts, dt_spritz=dts, m_spritz=msp)
    @test size(J) == (length(monolith_stack(o)), length(sel))
    @test all(isfinite, J)
    dv = zeros(15 + nu + 6)
    dv[sel] .= [1e-9, 2.0, 0.1, 600.0, 600.0, 600.0, 600.0, 300.0, 0.01]
    d = monolith_stack_jvp(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE, tw, tp, ta, ts, dts, msp, dv; pit=MONO_PIT)
    ref = J * dv[sel]
    # a 1-partial and a 5-partial Dual solve take slightly different adaptive
    # steps (the error norm includes the partials); the pit / heat-load rows
    # differ by ~1e-3 of the largest entry
    @test maximum(abs.(d .- ref)) / (maximum(abs.(ref)) + 1e-12) < 5e-3
    h = 60.0
    fp = monolith_stack(monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT, t_wrap=tw, t_pull=tp + h, t_add=ta, t_spritz=ts, dt_spritz=dts, m_spritz=msp))
    fm = monolith_stack(monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT, t_wrap=tw, t_pull=tp - h, t_add=ta, t_spritz=ts, dt_spritz=dts, m_spritz=msp))
    fd = (fp .- fm) ./ (2h)
    @test maximum(abs.(J[:, 5] .- fd)) / (maximum(abs.(fd)) + 1e-12) < 2e-2
    @test abs(J[end - 1, 5]) > 0  # juiciness at serve moves with the pull
end

@testset "monolith: planner-exposed decisions (t_salt, wrap trigger, split interval)" begin
    using BrisketDynamics: monolith_stack, monolith_stack_jacobian
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    hr = 3600.0
    base = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT)
    # defaults exactly inert: Inf trigger / config dt_add / zero salt
    same = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT,
                          T_wrap_trigger=Inf, dt_add=Inf, t_salt=0.0)
    @test same.Tcore == base.Tcore && same.w_retained == base.w_retained

    # the trigger fires the latch in the MONOLITH: wrap engages near the
    # crossing and the surface stops drying against the never-wrapped run
    trig = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT,
                          T_wrap_trigger=330.0)
    @test trig.w_retained > base.w_retained + 0.005

    # a finite split interval feeds the fire repeatedly
    splits = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT,
                            t_add=2hr, dt_add=1.5hr)
    single = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT,
                            t_add=2hr)
    @test splits.m_fuel[end] > single.m_fuel[end] + 2 * MONO_PIT.m_chimney

    # dry-brine hours pull water retention up through the WHC coupling
    salted = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT,
                            t_salt=12hr)
    @test salted.w_retained > base.w_retained

    # the three new stack columns agree with finite differences
    nu = length(MONO_U)
    b = 15 + nu
    sel = [b + 7, b + 8, b + 9]  # t_salt, T_wrap_trigger, dt_add
    kw = (; pit=MONO_PIT)
    args = (θ, φ, MONO_U, MONO_UTK, MONO_TSAVE, 1.0e9, 1.0e9, 2hr, 1.0e9, 1800.0, 0.05)
    J = monolith_stack_jacobian(args..., sel;
                                t_salt=12hr, T_wrap_trigger=330.0, dt_add=1.5hr, kw...)
    @test all(isfinite, J)
    # t_salt's temperature derivative is ~1e-15 K/s (noise floor for a
    # central difference through the adaptive solver) — its live signal
    # is water retention, so that column is checked on w_retained only
    nrow = 0
    for (k, name, hstep, tol) in
        ((1, :t_salt, 1800.0, 0.05), (2, :T_wrap_trigger, 0.25, 0.10),
         (3, :dt_add, 60.0, 0.10))
        f = x -> monolith_stack(monolith_solve(args[1:5]...; t_add=2hr,
                                               dt_spritz=1800.0, m_spritz=0.05,
                                               t_salt=(name == :t_salt ? x : 12hr),
                                               T_wrap_trigger=(name == :T_wrap_trigger ? x : 330.0),
                                               dt_add=(name == :dt_add ? x : 1.5hr), kw...))
        x0 = name == :t_salt ? 12hr : (name == :T_wrap_trigger ? 330.0 : 1.5hr)
        fd = (f(x0 + hstep) .- f(x0 - hstep)) ./ (2hstep)
        nrow = length(fd)
        rows = name == :t_salt ? [nrow - 2] : collect(1:nrow)  # w_retained
        scale = maximum(abs.(fd[rows])) + 1e-12
        @test maximum(abs.(J[rows, k] .- fd[rows])) / scale < tol
        @test maximum(abs.(fd[rows])) > 0  # the column is live
    end

    # disabled sentinels seeded by the planner's VJP sweep: exact-zero
    # columns (the eps-aware Dual comparison once sent t_salt = 0 into
    # the sqrt(t) brine singularity and killed the whole Dual solve)
    J0 = monolith_stack_jacobian(args..., sel;
                                 t_salt=0.0, T_wrap_trigger=Inf, dt_add=Inf, kw...)
    @test all(iszero, J0)
end

@testset "monolith: fuel addition, spritz policy and multi-meat load" begin
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    hr = 3600.0
    base = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT)
    # a chimney at 3 h: the bed gains its mass over the dump and nothing else
    # changes while the fire is nowhere near burning out
    add = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT, t_add=3hr)
    i2, i4 = findfirst(t -> t >= 2hr, MONO_TSAVE), findfirst(t -> t >= 4hr, MONO_TSAVE)
    @test abs(add.m_fuel[i2] - base.m_fuel[i2]) < 1e-3  # pulse tail
    @test isapprox(add.m_fuel[i4] - base.m_fuel[i4], MONO_PIT.m_chimney; rtol=2e-2)  # a lit chimney burns a little faster
    @test add.ignited[end] >= base.ignited[end] - 1e-9
    # the spritz policy wets the surface at each pulse and cools it
    sp = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT, t_spritz=2hr, dt_spritz=1800.0)
    @test sp.spritz_w > 0 && base.spritz_w == 0
    @test sp.Tsurf[end] < base.Tsurf[end]
    # every lid lift costs the pit: the save point after the first spritz sits
    # below the un-spritzed pit; the chimney's two minutes cost more
    i2s = findfirst(t -> t >= 2hr + 600, MONO_TSAVE)
    @test sp.Tpit[i2s] < base.Tpit[i2s] - 2.0
    i3s = findfirst(t -> t >= 3hr + 600, MONO_TSAVE)
    @test add.Tpit[i3s] < base.Tpit[i3s] - 5.0
    # a heavier dose wets more
    heavy = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT, t_spritz=2hr, dt_spritz=1800.0, m_spritz=0.1)
    @test heavy.spritz_w > 1.5 * sp.spritz_w
    # multi-meat: two briskets pull the pit down and humidify it
    two = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT, cfg=MeatConfig(; n_meat=2.0))
    @test two.Tpit[i4] < base.Tpit[i4]
    @test two.pv_pit[i4] > base.pv_pit[i4]
end

@testset "monolith: differentiable duration (s = t/t_f)" begin
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    t_f = MONO_T_F
    s_save = MONO_TSAVE ./ t_f
    s_u = MONO_UTK ./ t_f
    fw, fp = 0.5, 0.8
    # tf mode reproduces seconds mode (same physics, s-parameterized solve)
    o_s = monolith_solve(
        θ, φ, MONO_U, MONO_UTK, MONO_TSAVE;
        pit=MONO_PIT, t_wrap=fw * t_f, t_pull=fp * t_f,
    )
    o_tf = monolith_solve_tf(
        θ, φ, MONO_U, t_f, s_u, s_save; pit=MONO_PIT, frac_wrap=fw, frac_pull=fp,
    )
    @test maximum(abs.(o_tf.Tpit .- o_s.Tpit)) < 0.05
    @test maximum(abs.(o_tf.Tcore .- o_s.Tcore)) < 0.05
    @test abs(o_tf.w_retained - o_s.w_retained) < 1e-4
    @test abs(o_tf.juiciness - o_s.juiciness) < 1e-4

    # d(stack)/d(t_f): one Dual solve vs central FD over the duration
    nu = length(MONO_U)
    nv = 8 + 7 + nu + 7
    dv = zeros(nv)
    dv[8 + 7 + nu + 1] = 1.0  # t_f entry of [θ; φ; u; t_f; fracs...; m_spritz]
    g = monolith_stack_jvp_tf(
        θ, φ, MONO_U, t_f, s_u, s_save, fw, fp, FRAC_NEVER, FRAC_NEVER, 0.5, NaN, dv;
        pit=MONO_PIT,
    )
    h = 90.0
    stack_at(tf) = monolith_stack(
        monolith_solve_tf(
            θ, φ, MONO_U, tf, s_u, s_save; pit=MONO_PIT, frac_wrap=fw, frac_pull=fp,
        ),
    )
    fd = (stack_at(t_f + h) .- stack_at(t_f - h)) ./ (2h)
    big = abs.(fd) .> 1e-5
    @test count(big) > 100
    rel = abs.(g[big] .- fd[big]) ./ abs.(fd[big])
    @test sort(rel)[ceil(Int, 0.95 * end)] < 8e-3  # was 5e-3: the salt/pan states shift adaptive steps and this FD-vs-forward check carries solver noise (docs/verification.md, last-bit sensitivity)
    @test maximum(rel) < 5e-2

    # chain rule: d/d(frac_pull) in tf mode == t_f · d/d(t_pull) in seconds mode
    dv2 = zeros(nv)
    dv2[8 + 7 + nu + 3] = 1.0  # frac_pull
    g_fp = monolith_stack_jvp_tf(
        θ, φ, MONO_U, t_f, s_u, s_save, fw, fp, FRAC_NEVER, FRAC_NEVER, 0.5, NaN, dv2;
        pit=MONO_PIT,
    )
    nvs = 8 + 7 + nu + 6
    dv3 = zeros(nvs)
    dv3[8 + 7 + nu + 2] = 1.0  # t_pull in the seconds-mode vector
    g_tp = monolith_stack_jvp(
        θ, φ, MONO_U, MONO_UTK, MONO_TSAVE, fw * t_f, fp * t_f, 1.0e9, 1.0e9, 3600.0,
        NaN, dv3; pit=MONO_PIT,
    )
    big2 = abs.(g_tp .* t_f) .> 1e-3
    rel2 = abs.(g_fp[big2] .- t_f .* g_tp[big2]) ./ abs.(t_f .* g_tp[big2])
    @test sort(rel2)[ceil(Int, 0.9 * end)] < 1e-2
end

@testset "monolith: two-zone offset + wood splits" begin
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    one_zone = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT)
    @test maximum(abs.(one_zone.T_firebox .- one_zone.T_firebox[1])) < 1e-6  # inert by default
    pit2 = offset_smoker(; T_amb=300.0, dt_add=Inf)
    two = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=pit2)
    mid = div(length(MONO_TSAVE), 2)
    @test all(two.T_firebox[mid:end] .> two.Tpit[mid:end])   # firebox runs hotter
    @test two.Tpit[div(mid, 4)] < one_zone.Tpit[div(mid, 4)] # the duct lags startup
    @test all(isfinite, two.Tcore) && two.Tcore[end] > two.Tcore[1] + 20
    # hourly wood splits: the pulse train keeps more fuel in the box than
    # a single addition of the same chimney size
    hr = 3600.0
    pit_split = offset_smoker(; T_amb=300.0, dt_add=1.0hr)
    splits = monolith_solve(
        θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=pit_split, t_add=2.0hr,
    )
    single = monolith_solve(
        θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=pit2, t_add=2.0hr,
    )
    @test splits.m_fuel[end] > single.m_fuel[end] + 0.5
end

@testset "monolith: water pan" begin
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    dry = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT)
    @test all(iszero, dry.m_pan)                    # no pan: states ride inert
    pit_pan = PitConfig(; T_amb=300.0, m_pan0=4.0)
    wet = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=pit_pan)
    @test wet.m_pan[end] < wet.m_pan[1]             # the pan evaporates
    @test maximum(wet.T_pan) < 380.0                # latent sink caps near boiling
    # the pan buffers the startup and feeds the humidity
    i_mid = div(length(MONO_TSAVE), 3)
    @test wet.Tpit[i_mid] < dry.Tpit[i_mid]
    @test sum(wet.pv_base) > sum(dry.pv_base)
    @test all(wet.pan_evap[(end - 5):end] .>= -1e-9)  # steady pan evaporates, not condenses
    # the closed pan balance: the pan drives against the pit vapor that
    # includes its own contribution, so while it evaporates the pit vapor
    # stays below the pan's saturation (it can never push the pit past it)
    @test all(i -> wet.pan_evap[i] <= 0 ||
                   wet.pv_base[i] < BrisketDynamics.p_sat(wet.T_pan[i]),
              eachindex(MONO_TSAVE))
    # closed form vs the open-loop (pre-pan) flux: ṁ = g·taper·(p_sat − pv0)/(1 + g·taper·K)
    hum0 = BrisketDynamics.Humidity()
    q, ṁc, pvb = BrisketDynamics._pan_algebra(380.0, 2.0, 360.0, 0.01, 2000.0, pit_pan, hum0)
    K = hum0.p_atm / 0.622 / 0.01
    gt = pit_pan.g_pan * tanh(2.0 / pit_pan.m_pan_taper)
    ṁ_open = gt * (BrisketDynamics.p_sat(360.0) - 2000.0)
    @test ṁc ≈ ṁ_open / (1 + gt * K)
    @test 0 < ṁc < ṁ_open
    @test pvb ≈ 2000.0 + K * ṁc && pvb < BrisketDynamics.p_sat(360.0)
    @test q ≈ pit_pan.UA_pan * 20.0
    # a small pan DRIES OUT and the pit warms back toward the dry trace
    tiny = monolith_solve(
        θ, φ, MONO_U, MONO_UTK, MONO_TSAVE;
        pit=PitConfig(; T_amb=300.0, m_pan0=0.25),
    )
    @test tiny.m_pan[end] < 0.05
    @test abs(tiny.Tpit[end] - dry.Tpit[end]) < abs(wet.Tpit[end] - dry.Tpit[end]) + 3.0
end

@testset "monolith: smoke quality (burn cleanliness, bitterness)" begin
    # same fire, choked vs open damper: the choked burn smolders (lower
    # cleanliness) and a larger FRACTION of what deposits is dirty; the
    # totals differ too, so the fraction is the invariant, not the mass
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    mo = monolith_solve(θ, φ, fill(0.7, 8), MONO_UTK, MONO_TSAVE; pit=MONO_PIT)
    mc = monolith_solve(θ, φ, fill(0.05, 8), MONO_UTK, MONO_TSAVE; pit=MONO_PIT)
    for o in (mo, mc)
        @test all(0.0 .<= o.smoke_clean .<= 1.0)
        @test o.bitterness >= 0
    end
    n = length(MONO_TSAVE)
    @test sum(mc.smoke_clean) / n < sum(mo.smoke_clean) / n
    @test mc.bitterness / (mc.bark_s[end] + 1e-9) >
          mo.bitterness / (mo.bark_s[end] + 1e-9)
    # electric: nothing burns -> nothing deposits -> nothing bitter
    o_e = monolith_solve(
        θ, PIT_φ_DEFAULT, MONO_U, MONO_UTK, MONO_TSAVE;
        pit=PitConfig(; T_amb=300.0, p_heater=3000.0),
    )
    @test o_e.bitterness < 1e-12
end

@testset "monolith: electric source (p_heater)" begin
    # the element heats, nothing burns: fuel constant, no smoke, no
    # deposition — bark goes Maillard-only, which is what an electric
    # cooker does. a0 = 1 (nothing to ignite).
    θ = MEAT_θ_DEFAULT
    φ = PIT_φ_DEFAULT  # a0 = 1
    pit_e = PitConfig(; T_amb=300.0, p_heater=3000.0)
    o = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=pit_e)
    @test o.Tpit[end] > 340.0                       # the heater holds a cook temperature
    @test abs(o.m_fuel[end] - o.m_fuel[1]) < 1e-9   # nothing burned
    @test maximum(o.mdot) < 1e-12                   # no smoke flux
    @test o.bark_s[end] < 1e-9                      # no smoke deposition
    @test o.Tcore[end] > o.Tcore[1] + 25            # the meat still cooks
end

@testset "monolith: runtime exposed area (the estimator's latent)" begin
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    base = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT)
    same = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT,
                          A_meat=MeatConfig().A_meat)
    @test same.Tpit == base.Tpit && same.Tcore == base.Tcore && same.q_meat == base.q_meat
    @test same.pv_pit == base.pv_pit
    # a bigger cut draws more heat and humidifies the pit more
    big = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT, A_meat=0.24)
    mid = div(length(MONO_TSAVE), 2)
    @test big.Tpit[mid] < base.Tpit[mid]
    @test big.pv_pit[mid] > base.pv_pit[mid]
    @test big.q_meat[mid] > base.q_meat[mid]
    # forward mode through the area matches central differences
    function g(a)
        o = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT, A_meat=a)
        return vcat(o.Tpit, o.Tcore)
    end
    a0 = 0.12
    da = ForwardDiff.derivative(g, a0)
    h = 1e-3 * a0
    fd = (g(a0 + h) .- g(a0 - h)) ./ (2h)
    err = maximum(abs.(da .- fd)) / (maximum(abs.(fd)) + 1e-12)
    @info "FD gate A_meat: relative error $(round(err; sigdigits=3))"
    @test all(isfinite, da) && err < 2e-2
end

@testset "monolith_solve_tf: runtime exposed area (the estimator's latent)" begin
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    t_f = MONO_T_F
    s_u = MONO_UTK ./ t_f
    s_save = MONO_TSAVE ./ t_f
    base = monolith_solve_tf(θ, φ, MONO_U, t_f, s_u, s_save; pit=MONO_PIT, frac_pull=0.8)
    same = monolith_solve_tf(θ, φ, MONO_U, t_f, s_u, s_save; pit=MONO_PIT, frac_pull=0.8,
                             A_meat=MeatConfig().A_meat)
    @test same.Tpit == base.Tpit && same.Tcore == base.Tcore && same.pv_pit == base.pv_pit
    big = monolith_solve_tf(θ, φ, MONO_U, t_f, s_u, s_save; pit=MONO_PIT, frac_pull=0.8, A_meat=0.24)
    mid = div(length(s_save), 2)
    @test big.Tpit[mid] < base.Tpit[mid]
    @test big.pv_pit[mid] > base.pv_pit[mid]
    # the area and the duration are two forward-mode directions of one solve
    function g(v)
        o = monolith_solve_tf(θ, φ, MONO_U, v[2], s_u, s_save; pit=MONO_PIT, frac_pull=0.8, A_meat=v[1])
        return vcat(o.Tpit[mid], o.Tcore[end], o.w_retained)
    end
    v0 = [0.12, t_f]
    Jv = ForwardDiff.jacobian(g, v0)
    h = [1e-3 * 0.12, 1e-3 * t_f]
    for k in 1:2
        e = zeros(2); e[k] = h[k]
        fd = (g(v0 .+ e) .- g(v0 .- e)) ./ (2h[k])
        @test maximum(abs.(Jv[:, k] .- fd)) / (maximum(abs.(fd)) + 1e-12) < 2e-2   # the FD-vs-forward gate of the area test above: the adaptive solve carries last-bit noise (docs/verification.md)
    end
end
