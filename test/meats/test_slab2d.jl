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
    θ2 = [θ[1], θ[2], θ[3], θ[5], θ[6], θ[7], θ[8]]
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

    # smoke on the faces: a burn-rate forcing deposits smoke and drives the
    # ring on every face cell without touching the transport (bitwise);
    # no forcing is exactly zero; in the thick-slab limit the end-face
    # centre carries the 1-D surface's deposition and ring depth
    smk = fill(3e-4, length(tknots))
    wides = meat_solve_2d(θ2, Tpit, tknots, tknots;
                          R=0.50, Hz=Hz, nr=10, nz=30, cfg=cfg0, h=20.0,
                          smoke_knots=smk)
    @test wides.T_field == wide.T_field && wides.W_field == wide.W_field
    @test wide.smoke_mean == 0 && wide.ring_mean == 0 && wide.bitterness == 0
    slabs = meat_solve(θs, Tpit, tknots, tknots; cfg=cfg0, smoke_knots=smk)
    @test maximum(abs.(wides.smoke_ends_t[1, 1, :] .- slabs.bark_s)) < 2e-3
    @test maximum(abs.(wides.ring_ends_t[1, 1, :] .- slabs.ring)) < 5e-5
    # the corners dry first, and a dry face takes less smoke and less ring
    @test wides.smoke_rim[1] < wides.smoke_rim[15]
    @test wides.ring_rim[1] < wides.ring_rim[15]
    @test 0 < wides.ring_min <= wides.ring_mean
    @test wides.l_star_mean < wide.l_star_mean

    # fat renders per node: the thick-slab axis carries the 1-D fat field,
    # and the cut's rendered share and juiciness are the 1-D scores
    jw = div(30, 2) + 1
    @test maximum(abs.(wide.F_field[1, jw, :] .- slab.F_field[1, :])) < 2e-3
    @test 0 < wide.fat_rendered < 1 && wide.F_mean < cfg0.F0
    @test abs(wide.fat_rendered - slab.fat_rendered) < 0.02
    # (the juiciness's water term differs by the wide cylinder's drying rim;
    # its fat term is the 1-D one)
    @test abs((wide.juiciness - wide.w_retained) - (slab.juiciness - slab.w_retained)) < 2e-3

    # bone at the core, on the depth below the nearest exposed face: the
    # thick-slab limit tracks the 1-D bone-in slab, and the bone moves it
    cfgb = BrisketDynamics.with_config(cfg0; bone_frac=0.3)
    wideb = meat_solve_2d(θ2, Tpit, tknots, tknots;
                          R=0.50, Hz=Hz, nr=10, nz=30, cfg=cfgb, h=20.0)
    slabb = meat_solve(θs, Tpit, tknots, tknots; cfg=cfgb)
    @test maximum(abs.(wideb.Tcore .- slabb.Tcore)) < 1.5
    @test maximum(abs.(wideb.Tcore .- wide.Tcore)) > 0.2

    # the draft scales h_c and k_m as (g/g_ref)^m_conv, as in 1-D
    air = fill(2 * cfg0.g_ref, length(tknots))
    widea = meat_solve_2d(θ2, Tpit, tknots, tknots;
                          R=0.50, Hz=Hz, nr=10, nz=30, cfg=cfg0, h=20.0,
                          airflow_knots=air)
    slaba = meat_solve(θs, Tpit, tknots, tknots; cfg=cfg0, airflow_knots=air)
    @test maximum(abs.(widea.Tcore .- slaba.Tcore)) < 1.5
    @test widea.Tcore[div(length(tknots), 3)] > wide.Tcore[div(length(tknots), 3)] + 0.2

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
    θ2dry = [θ[1], 0.0, θ[3], θ[5], θ[6], θ[7], θ[8]]
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

    # pull / rest on the faces: never pulled is bitwise the pit-only
    # solve; a pull mid-cook swaps every face to the cambro — the rim
    # cools, most evaporation stops, the cut retains more water, and
    # nothing before the pull changes
    never = meat_solve_2d(θ2, Tpit, tknots, tknots;
                          R=R, Hz=0.06, nr=24, nz=20, cfg=cfg0, h=20.0,
                          t_pull=PULL_NEVER)
    @test never.Tcore == fin.Tcore && never.W_field == fin.W_field
    pulled = meat_solve_2d(θ2, Tpit, tknots, tknots;
                           R=R, Hz=0.06, nr=24, nz=20, cfg=cfg0, h=20.0,
                           t_pull=3hr)
    @test pulled.T_field[24, 10, end] < fin.T_field[24, 10, end] - 5.0
    @test pulled.w_retained > fin.w_retained + 0.01
    @test maximum(pulled.Tcold) <= maximum(fin.Tcold) + 1e-9
    ipre = findfirst(t -> t >= 1hr, tknots)
    @test maximum(abs.(pulled.T_field[:, :, 1:ipre] .- fin.T_field[:, :, 1:ipre])) < 1e-6

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
                           R=R, Hz=0.06, nr=16, nz=14, cfg=cfg0, h=20.0,
                           airflow_knots=cpl0.airflow_steps)
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

    # the pull in the pit loop: after t_pull the meat's heat and vapor
    # exchange vanish (the pit sees no load), the pit humidity returns to
    # the no-meat base exactly (same fuel state), and the cut rests
    cplp = coupled_solve_2d(θ2, φ, uk, utk, tknots;
                            R=R, Hz=0.06, nr=16, nz=14, cfg=cfg0, h=20.0,
                            hum=hum0, n_meat=2.0, t_pull=3hr)
    @test abs(cplp.q_meat[end]) < 1e-6 && abs(cplp.j_evap[end]) < 1e-12
    @test abs(cplp.pv_pit[end] - cpl0.pv_pit[end]) < 1e-9
    @test cplp.q_meat[i1] ≈ cpl2.q_meat[i1] rtol = 1e-6
    @test cplp.Tcore[end] < cpl2.Tcore[end]

    # the burn smokes the meat in the pit loop: deposition, ring and the
    # dirty share on the faces; the foil cuts the deposition
    @test cpl2.smoke_mean > 0.1 && cpl2.ring_mean > 0 && cpl2.bitterness >= 0
    @test all(cpl2.mdot .> 0) && all(0 .< cpl2.smoke_clean .<= 1)
    @test all(cpl2.lid .== 0)  # no event, no lid lift
    cplw = coupled_solve_2d(θ2, φ, uk, utk, tknots;
                            R=R, Hz=0.06, nr=16, nz=14, cfg=cfg0, h=20.0,
                            hum=hum0, n_meat=2.0, t_wrap=2hr)
    @test cplw.smoke_mean < cpl2.smoke_mean - 0.05

    # pit events: ADD_NEVER is bitwise the default; a chimney at 2 h feeds
    # the fuel (about m_chimney net of its own burn) and lights the bed,
    # and its lid lift drops the pit, which UA_open = 0 removes; spritz
    # pulses lift the lid too
    kwc = (; R=R, Hz=0.06, nr=16, nz=14, cfg=cfg0, h=20.0, hum=hum0, n_meat=2.0)
    never_add = coupled_solve_2d(θ2, φ, uk, utk, tknots; kwc...,
                                 t_add=BrisketDynamics.ADD_NEVER)
    @test never_add.Tpit == cpl2.Tpit && never_add.T_field == cpl2.T_field
    added = coupled_solve_2d(θ2, φ, uk, utk, tknots; kwc..., t_add=2hr)
    i2 = findfirst(t -> t >= 2hr, tknots)
    mch = PitConfig().m_chimney
    @test added.m_fuel[i2 + 1] - cpl2.m_fuel[i2 + 1] > 0.8 * mch
    @test added.m_fuel[i2 + 1] - cpl2.m_fuel[i2 + 1] < mch
    @test added.ignited[i2 + 1] > cpl2.ignited[i2 + 1]
    @test added.lid[i2] > 0.5
    @test added.Tpit[i2] < cpl2.Tpit[i2] - 10.0
    nolid = coupled_solve_2d(θ2, φ, uk, utk, tknots; kwc..., t_add=2hr,
                             pit=PitConfig(UA_open=0.0))
    @test nolid.Tpit[i2] > added.Tpit[i2] + 10.0
    # the chimney pulse is a logistic derivative (τ_add wide): an hour
    # before it the pit has not moved
    @test maximum(abs.(added.Tpit[1:(i2 - 6)] .- cpl2.Tpit[1:(i2 - 6)])) < 1e-3
    sprp = coupled_solve_2d(θ2, φ, uk, utk, tknots; kwc...,
                            t_spritz=1.5hr, dt_spritz=1800.0)
    @test maximum(sprp.lid) > 0 && minimum(sprp.Tpit .- cpl2.Tpit) < -1.0

    # the 2-D stack: smoke scalars appended last, the pit events appended
    # to the differentiable inputs; d(final fuel)/d(t_add) matches FD
    @test length(slab2d_stack(added)) == 4 * length(tknots) + 13
    kws = (; R=R, Hz=0.06, nr=10, nz=8, cfg=cfg0, h=20.0, hum=hum0, n_meat=2.0)
    fa(ta) = coupled_solve_2d(θ2, φ, uk, utk, tknots; kws..., t_add=ta).m_fuel[end]
    ga = ForwardDiff.derivative(fa, 2hr)
    fda = (fa(2hr + 60.0) - fa(2hr - 60.0)) / 120.0
    @test isfinite(ga) && abs(ga - fda) / (abs(fda) + 1e-12) < 1e-3
    nb = length(θ2) + length(φ) + length(uk)
    Jev = slab2d_stack_jacobian(θ2, φ, uk, utk, tknots, WRAP_NEVER, NaN,
                                BrisketDynamics.SPRITZ_NEVER, 3600.0, PULL_NEVER,
                                [nb + 6, nb + 7]; kws..., t_add=2hr, dt_add=1800.0)
    @test all(isfinite, Jev) && sum(abs, Jev[:, 1]) > 0 && sum(abs, Jev[:, 2]) > 0

    # differentiable: ForwardDiff through the ADI sweeps matches FD
    f(km) = meat_solve_2d([θ[1], km, θ[3], θ[5], θ[6], θ[7], θ[8]], Tpit, tknots,
                          tknots; R=R, Hz=0.06, nr=12, nz=10, cfg=cfg0,
                          h=30.0).w_retained
    g = ForwardDiff.derivative(f, θ[2])
    fd = (f(θ[2] * 1.02) - f(θ[2] * 0.98)) / (0.04 * θ[2])
    @test isfinite(g) && abs(g - fd) / (abs(fd) + 1e-12) < 0.05
    # ... and w.r.t. the pull time (later pull = longer in the pit = drier)
    fp(tp) = meat_solve_2d(θ2, Tpit, tknots, tknots; R=R, Hz=0.06, nr=12,
                           nz=10, cfg=cfg0, h=30.0, t_pull=tp).w_retained
    gp = ForwardDiff.derivative(fp, 3hr)
    fdp = (fp(3hr + 60.0) - fp(3hr - 60.0)) / 120.0
    @test isfinite(gp) && gp < 0 && abs(gp - fdp) / (abs(fdp) + 1e-12) < 0.05
end
