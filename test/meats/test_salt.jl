# Salt physics (dry brine; structural — literature constants). Three
# invariants: t_salt = 0 is EXACTLY unsalted (both couplings inert), the
# brine profile behaves like the erfc solution it is, and a salted cook
# holds more water while evaporating less.

@testset "salt physics" begin
    hr = 3600.0
    tknots = collect(0.0:300.0:10hr)
    tsave = tknots
    θ = MEAT_θ_DEFAULT
    Tpit = fill(383.15, length(tknots))

    # --- unsalted inertness: with s ≡ 0 the couplings cannot act, so even
    # absurd coupling constants must not move a single output
    base = meat_solve(θ, Tpit, tknots, tsave)
    wild = meat_solve(
        θ, Tpit, tknots, tsave;
        cfg=BrisketDynamics.with_config(MeatConfig(); β_whc=50.0, k_aw=10.0),
    )
    @test maximum(abs.(wild.Tcore .- base.Tcore)) < 1e-8
    @test abs(wild.w_retained - base.w_retained) < 1e-10
    @test all(iszero, base.salt_field)

    # --- the brine profile: surface-hot, center-cold, deeper with time
    cfg24 = BrisketDynamics.with_config(MeatConfig(); t_salt=24hr)
    cfg96 = BrisketDynamics.with_config(MeatConfig(); t_salt=96hr)
    s24 = BrisketDynamics.salt_profile(θ[4], cfg24)
    s96 = BrisketDynamics.salt_profile(θ[4], cfg96)
    @test issorted(s24)                       # node 1 = center … node nx = surface
    @test s24[end] ≈ cfg24.S_surf atol=1e-3   # erfc(0) = 1 at the surface
    @test s24[1] < 1e-6                       # 24 h does not reach a 3 cm center
    @test s96[1] > s24[1] && s96[div(end, 2)] > s24[div(end, 2)]

    # --- a salted cook: salt diffuses inward, total salt is conserved
    # (zero-flux BCs), more water is held (WHC) and less evaporates (a_w)
    salted = meat_solve(θ, Tpit, tknots, tsave; cfg=cfg96)
    @test salted.salt_field[1, end] > salted.salt_field[1, 1]  # center gains
    @test salted.salt_field[end, end] < salted.salt_field[end, 1]  # surface relaxes
    @test isapprox(salted.salt_mean[end], salted.salt_mean[1], rtol=1e-3)
    @test salted.w_retained > base.w_retained
    @test salted.evap_w < base.evap_w

    # --- restart continuity: handing the salt field back reproduces the
    # single-solve trajectory (the state-restart contract covers salt)
    mid = div(length(tsave), 2)
    seg1 = salted
    seg2 = meat_solve(
        θ, Tpit[mid:end], tknots[mid:end], tsave[mid:end];
        cfg=cfg96,
        T0f=seg1.T_field[:, mid], W0f=seg1.W_field[:, mid],
        C0f=seg1.C_field[:, mid], F0f=seg1.F_field[:, mid],
        Ssalt0f=seg1.salt_field[:, mid],
        B0=seg1.bark_b[mid], S0=seg1.bark_s[mid], R0=seg1.ring[mid],
    )
    @test maximum(abs.(seg2.Tcore .- seg1.Tcore[mid:end])) < 0.05
    @test maximum(abs.(seg2.salt_field[:, end] .- seg1.salt_field[:, end])) < 1e-4
end

@testset "bone region (thermal blend)" begin
    hr = 3600.0
    tknots = collect(0.0:300.0:6hr)
    Tpit = fill(383.15, length(tknots))
    θ = MEAT_θ_DEFAULT

    # boneless default: weights exactly zero, absurd bone constants inert
    base = meat_solve(θ, Tpit, tknots, tknots)
    wild = meat_solve(
        θ, Tpit, tknots, tknots;
        cfg=BrisketDynamics.with_config(MeatConfig(); k_bone=50.0, ρc_bone=1.0e4),
    )
    @test maximum(abs.(wild.Tcore .- base.Tcore)) < 1e-8
    @test all(iszero, BrisketDynamics.bone_weights(MeatConfig()))

    # a bone-in cylinder: low-capacity core heats FASTER mid-cook
    cfgb = BrisketDynamics.with_config(MeatConfig(); geom_m=1.0, bone_frac=0.3)
    cfg0 = BrisketDynamics.with_config(MeatConfig(); geom_m=1.0)
    b = meat_solve(θ, Tpit, tknots, tknots; cfg=cfgb)
    n = meat_solve(θ, Tpit, tknots, tknots; cfg=cfg0)
    mid = div(length(tknots), 2)
    @test b.Tcore[mid] > n.Tcore[mid]
    w = BrisketDynamics.bone_weights(cfgb)
    @test w[1] > 0.9 && w[end] < 1e-3 && issorted(w, rev=true)
end

@testset "t_salt as a differentiable input (M-goal)" begin
    hr = 3600.0
    tknots = collect(0.0:300.0:8hr)
    Tpit = fill(383.15, length(tknots))
    θ = MEAT_θ_DEFAULT

    # override beats config, NaN defers to it
    a = meat_solve(θ, Tpit, tknots, tknots; t_salt=48hr)
    b = meat_solve(
        θ, Tpit, tknots, tknots;
        cfg=BrisketDynamics.with_config(MeatConfig(); t_salt=48hr),
    )
    @test maximum(abs.(a.salt_field .- b.salt_field)) < 1e-12

    # d(w_retained)/d(t_salt) exists through the analytic brine profile,
    # matches finite differences, and is positive (longer brine, more
    # water held) — the planner-facing sensitivity
    wret(ts) = meat_solve(θ, Tpit, tknots, tknots; t_salt=ts).w_retained
    g = ForwardDiff.derivative(wret, 48hr)
    fd = (wret(50hr) - wret(46hr)) / (4hr)
    @test isfinite(g) && g > 0
    @test abs(g - fd) / abs(fd) < 0.05
end
