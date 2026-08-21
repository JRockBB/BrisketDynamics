# Julia-level gradient correctness: VJPs vs central finite differences.
# Tight solver tolerances so FD truncation, not solver noise, dominates.

"""
Central-difference partial of scalar f at x along e_i, with a RELATIVE step —
parameters here span 14 orders of magnitude (D_eff ~ 1e-8, C_th ~ 1e4), so an
absolute step either drowns in truncation error or perturbs by 30%.
"""
function fd_partial(f, x, i; hrel=1e-3)
    hi = x[i] == 0 ? 1e-6 : hrel * abs(x[i])
    xp = copy(x);
    xp[i] += hi
    xm = copy(x);
    xm[i] -= hi
    return (f(xp) - f(xm)) / (2hi)
end

relerr(a, b) = abs(a - b) / max(abs(a), abs(b), 1e-12)

@testset "meat adjoint vs FD" begin
    hr = 3600.0
    # 8 h horizon: reaches mid-stall, where D_eff (moisture resupply) is a live
    # parameter — on shorter horizons the surface is demand-limited and the
    # loss is insensitive to D_eff, so FD cannot resolve it against solver noise
    tknots = collect(0.0:600.0:8hr)
    tsave = tknots
    θ = MEAT_θ_DEFAULT
    Tpit = collect(range(380.0, 390.0, length=length(tknots)))  # sloped forcing
    tol = (abstol=1e-9, reltol=1e-9)

    # scalar probe loss mixing every output head
    wts = (Tsurf=0.3, Tcore=1.0, j_evap=1e3, q_meat=1e-2, C_field_final=10.0)
    function loss(θv, Tpitv)
        o = meat_solve(θv, Tpitv, tknots, tsave; tol...)
        return wts.Tsurf * sum(abs2, o.Tsurf .- 350) / length(o.Tsurf) +
               wts.Tcore * sum(abs2, o.Tcore .- 340) / length(o.Tcore) +
               wts.j_evap * sum(o.j_evap) +
               wts.q_meat * sum(o.q_meat) +
               wts.C_field_final * sum(o.C_field_final)
    end
    # matching cotangents for the VJP
    o = meat_solve(θ, Tpit, tknots, tsave; tol...)
    cot = (
        Tsurf=wts.Tsurf .* 2 .* (o.Tsurf .- 350) ./ length(o.Tsurf),
        Tcore=wts.Tcore .* 2 .* (o.Tcore .- 340) ./ length(o.Tcore),
        j_evap=fill(wts.j_evap, length(o.j_evap)),
        q_meat=fill(wts.q_meat, length(o.q_meat)),
        C_field_final=fill(wts.C_field_final, length(o.C_field_final)),
    )
    g = meat_vjp(θ, Tpit, tknots, tsave, cot; tol...)

    # per-parameter FD steps and tolerances (validated by convergence study):
    # D_eff needs a 1% step (its loss response below that is under solver
    # noise). Its FD value drifts each time the state count changes (2.4%
    # after the bark states joined) while the ADJOINT value is stable to 5
    # digits across model versions (composed check: -74672/-74673/-74677) —
    # the tolerance covers FD conditioning, not adjoint error.
    fd_plan = (
        (1, 1e-2, 4e-2), (2, 1e-3, 1e-3), (3, 1e-3, 1e-3), (4, 1e-3, 1e-3), (5, 1e-3, 1e-3)
    )
    for (i, hrel, tol_i) in fd_plan
        fd = fd_partial(v -> loss(v, Tpit), θ, i; hrel)
        @test relerr(g.dθ[i], fd) < tol_i
    end
    for i in (1, 13, 25, 40)                      # spot-check trajectory knots
        fd = fd_partial(v -> loss(θ, v), Tpit, i; hrel=1e-4)
        # 5e-3: knot FD is noise-limited and drifts with the state count
        # (0.2% observed after the bark states); adjoint stable — see above
        @test relerr(g.dTpit_knots[i], fd) < 5e-3
    end
end

@testset "wrap-time adjoint vs FD" begin
    hr = 3600.0
    tknots = collect(0.0:600.0:8hr)
    tsave = tknots
    θ = MEAT_θ_DEFAULT
    Tpit = collect(range(380.0, 390.0, length=length(tknots)))
    tol = (abstol=1e-9, reltol=1e-9)
    t_wrap = 4hr   # blend live mid-horizon so the gradient is nonzero

    wts = (Tsurf=0.3, Tcore=1.0, j_evap=1e3, q_meat=1e-2, C_field_final=10.0)
    function loss(θv, Tpitv, tw)
        o = meat_solve(θv, Tpitv, tknots, tsave; t_wrap=tw, tol...)
        return wts.Tsurf * sum(abs2, o.Tsurf .- 350) / length(o.Tsurf) +
               wts.Tcore * sum(abs2, o.Tcore .- 340) / length(o.Tcore) +
               wts.j_evap * sum(o.j_evap) +
               wts.q_meat * sum(o.q_meat) +
               wts.C_field_final * sum(o.C_field_final)
    end
    o = meat_solve(θ, Tpit, tknots, tsave; t_wrap, tol...)
    cot = (
        Tsurf=wts.Tsurf .* 2 .* (o.Tsurf .- 350) ./ length(o.Tsurf),
        Tcore=wts.Tcore .* 2 .* (o.Tcore .- 340) ./ length(o.Tcore),
        j_evap=fill(wts.j_evap, length(o.j_evap)),
        q_meat=fill(wts.q_meat, length(o.q_meat)),
        C_field_final=fill(wts.C_field_final, length(o.C_field_final)),
    )
    g = meat_vjp(θ, Tpit, tknots, tsave, cot; t_wrap, tol...)

    # FD on t_wrap: hrel = 1e-3 → ~14 s step, well inside δ_wrap = 600 s so
    # the central difference samples the smooth blend, not a switch
    fd_tw =
        (loss(θ, Tpit, t_wrap * (1 + 1e-3)) - loss(θ, Tpit, t_wrap * (1 - 1e-3))) /
        (2 * t_wrap * 1e-3)
    @test relerr(g.dt_wrap, fd_tw) < 1e-3

    # θ and last-knot directions with the wrap ACTIVE: catches contamination
    # of the existing slices by the appended p-tail entry (off-by-one guard)
    fd = fd_partial(v -> loss(v, Tpit, t_wrap), θ, 3)
    @test relerr(g.dθ[3], fd) < 1e-3
    # 5e-3, not 1e-3: with the wrap suppressing end-of-horizon evaporation
    # the last knot's gradient is small (~0.046) and FD noise-limited (0.2%
    # observed). As the off-by-one guard this is ample — tail contamination
    # would swap in dt_wrap's slot (−0.021), a ~100% relative error.
    fd = fd_partial(v -> loss(θ, v, t_wrap), Tpit, length(Tpit); hrel=1e-4)
    @test relerr(g.dTpit_knots[end], fd) < 5e-3
end

@testset "pit ForwardDiff VJP vs FD" begin
    hr = 3600.0
    tsave = collect(0.0:600.0:6hr)
    u_tknots = collect(range(0, 6hr, length=8))
    φ = PIT_φ_GRAD   # a0 < 1: ignition dynamics active, no clamp kink
    u = collect(range(0.3, 0.6, length=8))
    q = fill(150.0, length(tsave))

    tol = (abstol=1e-10, reltol=1e-10)        # keep solver noise below FD signal
    wT, wm = 1e-2, 1.0
    function loss(φv, uv, qv)
        o = pit_solve(φv, uv, u_tknots, qv, tsave, tsave; tol...)
        return wT * sum(abs2, o.Tpit .- 380) + wm * sum(o.m_fuel)
    end
    o = pit_solve(φ, u, u_tknots, q, tsave, tsave; tol...)
    cot = (Tpit=wT .* 2 .* (o.Tpit .- 380), m_fuel=fill(wm, length(o.m_fuel)))
    g = pit_vjp(φ, u, u_tknots, q, tsave, tsave, cot; tol...)

    for i in 1:7
        fd = fd_partial(v -> loss(v, u, q), φ, i)
        @test relerr(g.dφ[i], fd) < 1e-4
    end
    for i in (1, 4, 8)
        fd = fd_partial(v -> loss(φ, v, q), u, i)
        @test relerr(g.du_knots[i], fd) < 1e-4
    end
    for i in (1, 19)
        fd = fd_partial(v -> loss(φ, u, v), q, i)
        # looser tolerance: per-knot heat-load sensitivity is tiny, so central FD
        # is noise-limited here; the VJP itself is an exact ForwardDiff Jacobian
        @test relerr(g.dqmeat_knots[i], fd) < 1e-3
    end
end

@testset "bark adjoint vs FD" begin
    hr = 3600.0
    tknots = collect(0.0:600.0:8hr)
    tsave = tknots
    θ = MEAT_θ_DEFAULT
    Tpit = collect(range(380.0, 390.0, length=length(tknots)))
    smoke = fill(1.2e-4, length(tknots))
    tol = (abstol=1e-9, reltol=1e-9)
    t_wrap = 5hr
    wL = 0.1

    loss(θv, sk, tw) =
        wL *
        sum(meat_solve(θv, Tpit, tknots, tsave; smoke_knots=sk, t_wrap=tw, tol...).l_star)
    o = meat_solve(θ, Tpit, tknots, tsave; smoke_knots=smoke, t_wrap, tol...)
    cot = (l_star=fill(wL, length(o.l_star)),)
    g = meat_vjp(θ, Tpit, tknots, tsave, cot; smoke_knots=smoke, t_wrap, tol...)

    # smoke-knot direction (mid-cook, inside the wet window)
    i = 13
    fd = fd_partial(v -> loss(θ, v, t_wrap), smoke, i; hrel=1e-3)
    # 5e-3: forcing-knot FD noise floor (0.23% observed; θ/t_wrap directions
    # agree to 1e-5/1e-6 — the adjoint is fine, FD is the limit)
    @test relerr(g.dsmoke_knots[i], fd) < 5e-3
    # h_c direction through the bark path (surface temperature -> Maillard)
    fd = fd_partial(v -> loss(v, smoke, t_wrap), θ, 3)
    @test relerr(g.dθ[3], fd) < 2e-3
    # wrap-time direction through the deposition cutoff
    h = 1e-3 * t_wrap
    fd = (loss(θ, smoke, t_wrap + h) - loss(θ, smoke, t_wrap - h)) / (2h)
    @test relerr(g.dt_wrap, fd) < 2e-3
end

@testset "moisture adjoint vs FD" begin
    hr = 3600.0
    tknots = collect(0.0:600.0:8hr)
    tsave = tknots
    θ = MEAT_θ_DEFAULT
    Tpit = collect(range(380.0, 390.0, length=length(tknots)))
    tol = (abstol=1e-9, reltol=1e-9)
    wj, wf = 10.0, 5.0

    loss(θv, Tpitv) =
        wj * meat_solve(θv, Tpitv, tknots, tsave; tol...).juiciness +
        wf * meat_solve(θv, Tpitv, tknots, tsave; tol...).fat_rendered
    o = meat_solve(θ, Tpit, tknots, tsave; tol...)
    cot = (juiciness=wj, fat_rendered=wf)
    g = meat_vjp(θ, Tpit, tknots, tsave, cot; tol...)

    # h_c direction: drives surface T -> expulsion + render + evaporation
    fd = fd_partial(v -> loss(v, Tpit), θ, 3)
    @test relerr(g.dθ[3], fd) < 2e-3
    # k_mult (conduction): moves interior T history -> expulsion field
    fd = fd_partial(v -> loss(v, Tpit), θ, 5)
    @test relerr(g.dθ[5], fd) < 2e-3
    # mid trajectory knot through the new outputs
    fd = fd_partial(v -> loss(θ, v), Tpit, 13; hrel=1e-4)
    @test relerr(g.dTpit_knots[13], fd) < 5e-3

    # the calibrated expulsion latents (theta 5->7, 2026-08-21): k_e scales
    # the purge everywhere, a_myosin moves the W_eq step the stall entry
    # rides on — both reach juiciness through the W field
    fd = fd_partial(v -> loss(v, Tpit), θ, 6)
    @test relerr(g.dθ[6], fd) < 2e-3
    fd = fd_partial(v -> loss(v, Tpit), θ, 7)
    @test relerr(g.dθ[7], fd) < 2e-3

    # k_e also alters surface moisture -> evaporative cooling -> Tcore: check
    # the coupled path through a trajectory head as well
    cotT = (Tcore=ones(length(tsave)),)
    gT = meat_vjp(θ, Tpit, tknots, tsave, cotT; tol...)
    fd = fd_partial(v -> sum(meat_solve(v, Tpit, tknots, tsave; tol...).Tcore), θ, 6)
    @test relerr(gT.dθ[6], fd) < 2e-3
end
