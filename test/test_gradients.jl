# Julia-level gradient correctness: VJPs vs central finite differences.
# Tight solver tolerances so FD truncation, not solver noise, dominates.

"""
Central-difference partial of scalar f at x along e_i, with a RELATIVE step —
parameters here span 14 orders of magnitude (D_eff ~ 1e-8, C_th ~ 1e4), so an
absolute step either drowns in truncation error or perturbs by 30%.
"""
function fd_partial(f, x, i; hrel = 1e-3)
    hi = x[i] == 0 ? 1e-6 : hrel * abs(x[i])
    xp = copy(x); xp[i] += hi
    xm = copy(x); xm[i] -= hi
    (f(xp) - f(xm)) / (2hi)
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
    Tpit = collect(range(380.0, 390.0, length = length(tknots)))  # sloped forcing
    tol = (abstol = 1e-9, reltol = 1e-9)

    # scalar probe loss mixing every output head
    wts = (Tsurf = 0.3, Tcore = 1.0, j_evap = 1e3, q_meat = 1e-2, C_field_final = 10.0)
    function loss(θv, Tpitv)
        o = meat_solve(θv, Tpitv, tknots, tsave; tol...)
        wts.Tsurf * sum(abs2, o.Tsurf .- 350) / length(o.Tsurf) +
        wts.Tcore * sum(abs2, o.Tcore .- 340) / length(o.Tcore) +
        wts.j_evap * sum(o.j_evap) +
        wts.q_meat * sum(o.q_meat) +
        wts.C_field_final * sum(o.C_field_final)
    end
    # matching cotangents for the VJP
    o = meat_solve(θ, Tpit, tknots, tsave; tol...)
    cot = (Tsurf = wts.Tsurf .* 2 .* (o.Tsurf .- 350) ./ length(o.Tsurf),
           Tcore = wts.Tcore .* 2 .* (o.Tcore .- 340) ./ length(o.Tcore),
           j_evap = fill(wts.j_evap, length(o.j_evap)),
           q_meat = fill(wts.q_meat, length(o.q_meat)),
           C_field_final = fill(wts.C_field_final, length(o.C_field_final)))
    g = meat_vjp(θ, Tpit, tknots, tsave, cot; tol...)

    # per-parameter FD steps and tolerances (validated by convergence study):
    # D_eff needs a 1% step (its loss response below that is under solver
    # noise) and carries ~0.5% FD truncation at that step
    fd_plan = ((1, 1e-2, 2e-2), (2, 1e-3, 1e-3), (3, 1e-3, 1e-3),
               (4, 1e-3, 1e-3), (5, 1e-3, 1e-3))
    for (i, hrel, tol_i) in fd_plan
        fd = fd_partial(v -> loss(v, Tpit), θ, i; hrel)
        @test relerr(g.dθ[i], fd) < tol_i
    end
    for i in (1, 13, 25, 40)                      # spot-check trajectory knots
        fd = fd_partial(v -> loss(θ, v), Tpit, i; hrel = 1e-4)
        @test relerr(g.dTpit_knots[i], fd) < 1e-3
    end
end

@testset "pit ForwardDiff VJP vs FD" begin
    hr = 3600.0
    tsave = collect(0.0:600.0:6hr)
    u_tknots = collect(range(0, 6hr, length = 8))
    φ = PIT_φ_DEFAULT
    u = collect(range(0.3, 0.6, length = 8))
    q = fill(150.0, length(tsave))

    tol = (abstol = 1e-10, reltol = 1e-10)        # keep solver noise below FD signal
    wT, wm = 1e-2, 1.0
    function loss(φv, uv, qv)
        o = pit_solve(φv, uv, u_tknots, qv, tsave, tsave; tol...)
        wT * sum(abs2, o.Tpit .- 380) + wm * sum(o.m_fuel)
    end
    o = pit_solve(φ, u, u_tknots, q, tsave, tsave; tol...)
    cot = (Tpit = wT .* 2 .* (o.Tpit .- 380), m_fuel = fill(wm, length(o.m_fuel)))
    g = pit_vjp(φ, u, u_tknots, q, tsave, tsave, cot; tol...)

    for i in 1:5
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
