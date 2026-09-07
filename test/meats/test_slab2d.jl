# 2-D axisymmetric core: limit validations against the 1-D solver and
# the finite-size sanity Newman predicts. Shrink is ON (the default
# n_shrink = 1/3, where the per-direction 2-D stretch coincides with
# each 1-D limit's resolved-direction metric) — everything matches by
# construction (same property functions, purge, surface fluxes and
# material-coordinate metrics).

@testset "2-D axisymmetric core" begin
    hr = 3600.0
    tknots = collect(0.0:600.0:6hr)
    Tpit = fill(383.15, length(tknots))
    θ = MEAT_θ_DEFAULT
    θ2 = [θ[1], θ[2], θ[3], θ[5], θ[6], θ[7]]
    cfg0 = MeatConfig()

    # long-cylinder limit: a tall 2-D cylinder's mid-height axis vs the
    # 1-D radial solve (geom_m = 1, L = R)
    R = 0.03
    two = meat_solve_2d(θ2, Tpit, tknots, tknots;
                        R=R, Hz=0.40, nr=24, nz=28, cfg=cfg0, h=20.0)
    θ1 = [θ[1], θ[2], θ[3], R, θ[5], θ[6], θ[7], θ[8]]
    one = meat_solve(θ1, Tpit, tknots, tknots;
                     cfg=BrisketDynamics.with_config(cfg0; geom_m=1.0))
    err_cyl = maximum(abs.(two.Tcore .- one.Tcore))
    @test err_cyl < 1.5

    # thick-slab limit: a very wide short cylinder's axis vs the 1-D slab
    # through the half-height (geom_m = 0, L = Hz/2)
    Hz = 0.06
    wide = meat_solve_2d(θ2, Tpit, tknots, tknots;
                         R=0.50, Hz=Hz, nr=10, nz=30, cfg=cfg0, h=20.0)
    θs = [θ[1], θ[2], θ[3], Hz / 2, θ[5], θ[6], θ[7], θ[8]]
    slab = meat_solve(θs, Tpit, tknots, tknots; cfg=cfg0)
    err_slab = maximum(abs.(wide.Tcore .- slab.Tcore))
    @test err_slab < 1.5

    # finite-size sanity (what Newman predicts): a finite cylinder's
    # center heats FASTER than either infinite limit
    fin = meat_solve_2d(θ2, Tpit, tknots, tknots;
                        R=R, Hz=0.06, nr=24, nz=20, cfg=cfg0, h=20.0)
    mid = div(length(tknots), 2)
    @test fin.Tcore[mid] > two.Tcore[mid]
    @test fin.Tcore[mid] > slab.Tcore[mid]
    # moisture leaves through every exposed face
    @test fin.w_retained < 1.0
    @test all(isfinite, fin.T_field) && all(isfinite, fin.W_field)

    # the layers above the core, each exactly inert at its default:
    # wrap on every face, grain anisotropy in-plane, point-over-flat
    fin2 = meat_solve_2d(θ2, Tpit, tknots, tknots;
                         R=R, Hz=0.06, nr=24, nz=20, cfg=cfg0, h=20.0)
    @test fin2.Tcore == fin.Tcore  # kwargs defaulted = same solve

    wrapped = meat_solve_2d(θ2, Tpit, tknots, tknots;
                            R=R, Hz=0.06, nr=24, nz=20, cfg=cfg0, h=20.0,
                            t_wrap=1.5hr)
    @test wrapped.w_retained > fin.w_retained + 0.02

    # grain anisotropy directions, each on the physics it drives: with
    # evaporation off (k_m = 0) faster in-plane conduction must heat the
    # core strictly faster (with evap on, the hotter rim also dries
    # harder and the net sign is marginal); faster in-plane moisture
    # transport feeds the faces and lowers retention
    θ2dry = [θ[1], 0.0, θ[3], θ[5], θ[6], θ[7]]
    base_dry = meat_solve_2d(θ2dry, Tpit, tknots, tknots;
                             R=R, Hz=0.06, nr=24, nz=20, cfg=cfg0, h=20.0)
    grain_dry = meat_solve_2d(θ2dry, Tpit, tknots, tknots;
                              R=R, Hz=0.06, nr=24, nz=20, cfg=cfg0, h=20.0,
                              k_grain=1.3)
    @test grain_dry.Tcore[mid] > base_dry.Tcore[mid] + 0.1
    grD = meat_solve_2d(θ2, Tpit, tknots, tknots;
                        R=R, Hz=0.06, nr=24, nz=20, cfg=cfg0, h=20.0,
                        D_grain=2.0)
    @test grD.w_retained < fin.w_retained - 1e-4

    # a uniform cut is z-symmetric to roundoff; a fattier point muscle
    # above z_split breaks the symmetry (fat insulates and carries less
    # water) and the whole cut retains differently
    isym = maximum(abs.(fin.T_field[1, 3, :] .- fin.T_field[1, 18, :]))
    @test isym < 1e-9
    pf = meat_solve_2d(θ2, Tpit, tknots, tknots;
                       R=R, Hz=0.06, nr=24, nz=20, cfg=cfg0, h=20.0,
                       F_top=0.45, z_split=0.5)
    asym = maximum(abs.(pf.T_field[1, 3, :] .- pf.T_field[1, 18, :]))
    @test asym > 0.05
    @test pf.w_retained != fin.w_retained

    # shrink: the cut contracts (current dimensions below raw) and
    # n_shrink = 0 is exactly inert (unit metrics)
    @test fin.R_eff < R - 1e-4 && fin.Hz_eff < 0.06 - 1e-4
    ns0 = meat_solve_2d(θ2, Tpit, tknots, tknots;
                        R=R, Hz=0.06, nr=24, nz=20,
                        cfg=BrisketDynamics.with_config(MeatConfig(); n_shrink=0.0),
                        h=20.0)
    @test ns0.R_eff ≈ R atol = 1e-12
    @test ns0.Hz_eff ≈ 0.06 atol = 1e-12

    # quality fields ride along: collagen at the wide cylinder's axis
    # matches the 1-D center kinetics; bark grows on the faces and the
    # wrap suppresses it (cooler, wetter surface); the cold spot is a
    # real grid minimum below the axis temperature
    @test maximum(abs.(wide.C_core .- slab.C_field[1, :])) < 0.02
    @test 0 < fin.bark_mean < 1 && fin.bark_std > 0
    @test wrapped.bark_mean < fin.bark_mean
    @test all(fin.Tcold .<= fin.Tcore .+ 1e-9)
    @test 0 <= fin.C_min <= minimum(fin.C_field[:, :, end])

    # salt on the faces: t_salt = 0 exactly inert; a dry brine loads
    # every face (corners saltiest — the union of fronts), diffuses
    # inward, and retains water through the WHC coupling
    salted2 = meat_solve_2d(θ2, Tpit, tknots, tknots;
                            R=R, Hz=0.06, nr=24, nz=20, cfg=cfg0, h=20.0,
                            t_salt=12hr)
    @test salted2.salt_mean > 0
    jmid = 10
    @test salted2.salt_field[2, 1, end] <= salted2.salt_field[24, 1, end] + 1e-12
    @test salted2.salt_field[24, 1, end] > salted2.salt_field[24, jmid, end] - 1e-12
    @test salted2.salt_field[1, jmid, end] < 1e-3  # core untouched by the rub
    @test salted2.w_retained > fin.w_retained
    zero_salt = meat_solve_2d(θ2, Tpit, tknots, tknots;
                              R=R, Hz=0.06, nr=24, nz=20, cfg=cfg0, h=20.0,
                              t_salt=0.0)
    @test zero_salt.Tcore == fin.Tcore && zero_salt.w_retained == fin.w_retained

    # spritz on the faces: wets the cut and cools the crust
    sprz = meat_solve_2d(θ2, Tpit, tknots, tknots;
                         R=R, Hz=0.06, nr=24, nz=20, cfg=cfg0, h=20.0,
                         t_spritz=1.5hr, dt_spritz=1800.0)
    @test sprz.w_retained > fin.w_retained
    @test sprz.bark_mean < fin.bark_mean

    # pit coupling: with n_meat = 0 (and humidity constants pinned so
    # the base vapor pressure equals the standalone constant) the pit
    # never sees the meat, so replaying the coupled run's own per-step
    # pit trace through the prescribed-forcing solver reproduces the
    # meat fields exactly — same stepper, same order of operations
    φ = [15e3, 20.0, 3e-4, 1.2e7, 0.2, 3e-4, 0.85]
    utk = collect(range(0.0, 6hr; length=8))
    uk = fill(0.5, 8)
    hum0 = BrisketDynamics.Humidity(y_w=0.0,
                                    omega_amb=cfg0.p_v_pit * 0.622 / 101325.0)
    cpl0 = coupled_solve_2d(θ2, φ, uk, utk, tknots;
                            R=R, Hz=0.06, nr=16, nz=14, cfg=cfg0, h=20.0,
                            hum=hum0, n_meat=0.0)
    nstep = length(cpl0.Tpit_steps)
    tk_replay = collect(range(0.0, 20.0 * (nstep - 1); length=nstep))
    replay = meat_solve_2d(θ2, cpl0.Tpit_steps, tk_replay, tknots;
                           R=R, Hz=0.06, nr=16, nz=14, cfg=cfg0, h=20.0)
    @test maximum(abs.(cpl0.Tcore .- replay.Tcore)) < 1e-6
    @test maximum(abs.(cpl0.W_field .- replay.W_field)) < 1e-8

    # the coupling direction: briskets pull the pit down and humidify it
    cpl2 = coupled_solve_2d(θ2, φ, uk, utk, tknots;
                            R=R, Hz=0.06, nr=16, nz=14, cfg=cfg0, h=20.0,
                            hum=hum0, n_meat=2.0)
    # the pull-down direction holds while the meat is COLD (early); late
    # in a declining burn the hot meat legitimately returns heat
    i1 = findfirst(t -> t >= 1hr, tknots)
    i4 = findfirst(t -> t >= 4hr, tknots)
    @test cpl2.Tpit[i1] < cpl0.Tpit[i1] - 0.5
    @test cpl2.pv_pit[i4] > cpl0.pv_pit[i4]
    @test cpl2.j_evap[i4] > 0 && cpl2.q_meat[i1] > 0
    @test all(diff(cpl2.m_fuel) .<= 1e-12)  # fuel only burns down

    # differentiable: ForwardDiff through the ADI sweeps matches FD
    f(km) = meat_solve_2d([θ[1], km, θ[3], θ[5], θ[6], θ[7]], Tpit, tknots,
                          tknots; R=R, Hz=0.06, nr=12, nz=10, cfg=cfg0,
                          h=30.0).w_retained
    g = ForwardDiff.derivative(f, θ[2])
    fd = (f(θ[2] * 1.02) - f(θ[2] * 0.98)) / (0.04 * θ[2])
    @test isfinite(g) && abs(g - fd) / (abs(fd) + 1e-12) < 0.05
end
