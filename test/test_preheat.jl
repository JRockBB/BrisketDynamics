# The preheat and the load (src/preheat.jl; the load blend in src/monolith.jl, src/system.jl, src/meats/slab2d.jl),
# gated:
#   the load      before it the meat holds its raw state and the pit draws nothing from it; the default (in from the
#                 start) is the cook as before
#   the crossing  preheat_time finds when the empty pit reaches T_load, and its derivatives match finite differences
#   the lid       the door open while the meat goes on cools the pit
#   two pieces    loaded at different times, each holds its state until its own load

const P_HR = 3600.0
const P_θ = [3.8e-9, 0.0018, 10.0, 0.03, 1.0, 5.6e-4, 0.20, 0.6]
const P_φ = [15e3, 20.0, 3e-4, 1.2e7, 0.2, 3e-4, 0.85]
const P_PIT = PitConfig(; T_amb=300.0, Tpit0=300.0)

@testset "preheat: monolith and system share the partial-load vapor closure" begin
    cfg = MeatConfig(nx=8)
    sys = CookSystem(Slab1D(:brisket; cfg); pit=P_PIT)
    u = [0.4, 0.7, 0.3]
    tload = 3600.0
    ps, play = pack_params(sys, [P_θ], [cfg.A_meat], P_φ, u;
        events=[CookEvent(:load, :brisket, tload)])
    x = initial_state(sys, ps, play)
    x[state_index(sys.layout, :pit, :T_pit)] .= 380.0
    mono! = BrisketDynamics.make_monolith_rhs(cfg, P_PIT, Humidity(), 0.0, P_HR, length(u))
    system!, _ = make_system_rhs(sys, 0.0, P_HR, length(u))
    pm = vcat(P_θ, P_φ, u, WRAP_NEVER, PULL_NEVER, BrisketDynamics.ADD_NEVER,
        BrisketDynamics.SPRITZ_NEVER, 3600.0, cfg.m_spritz, P_PIT.dt_add, tload)
    for t in (0.0, tload - 600.0, tload - 120.0, tload, tload + 120.0, tload + 600.0)
        dm = similar(x); ds = similar(x)
        mono!(dm, x, pm, t); system!(ds, x, ps, t)
        @test dm ≈ ds rtol=1e-11 atol=1e-12
    end
end

@testset "preheat: six-state RHS equals the empty monolith" begin
    cfg = MeatConfig()
    hum = Humidity()
    nm = 5cfg.nx + 3
    u = [0.3, 0.7, 0.5]
    for pit in (P_PIT,
                PitConfig(T_amb=300.0, Tpit0=300.0, m_pan0=1.0),
                PitConfig(T_amb=300.0, Tpit0=300.0, m_pan0=1.0, c_duct=40.0),
                PitConfig(T_amb=300.0, Tpit0=300.0, p_heater=1500.0))
        xpit = [355.0, 5.0, 0.7, 0.8, 320.0, 450.0]
        x = vcat(BrisketDynamics._meat_u0(cfg, Float64[], Float64[], Float64[]), xpit)
        p = vcat(P_θ, P_φ, u, WRAP_NEVER, PULL_NEVER, BrisketDynamics.ADD_NEVER,
                 BrisketDynamics.SPRITZ_NEVER, 3600.0, cfg.m_spritz, pit.dt_add, 1e9)
        full! = BrisketDynamics.make_monolith_rhs(cfg, pit, hum, 0.0, P_HR, length(u))
        small! = BrisketDynamics._preheat_rhs(pit, hum, 0.0, P_HR, length(u))
        dx = zeros(length(x)); dpit = zeros(6)
        full!(dx, x, p, 1000.0)
        small!(dpit, xpit, vcat(P_φ, u), 1000.0)
        @test dpit ≈ dx[(nm + 1):end] rtol=1e-14
        @test all(iszero, dx[1:nm])
        t, reached, _ = preheat_time(330.0, P_φ, u, [0.0, P_HR, 2P_HR]; pit)
        @test reached
        oracle = monolith_solve(P_θ, P_φ, u, [0.0, P_HR, 2P_HR], [0.0, t];
                                pit, t_load=1e9, abstol=1e-10, reltol=1e-10)
        @test oracle.Tpit[end] ≈ 330.0 atol=1e-3
    end
end

@testset "preheat: the load holds the meat and the pit draws nothing before it" begin
    ts = collect(0.0:300.0:(4P_HR)); u = fill(0.5, 4); utk = collect(range(0.0, 4P_HR; length=4))
    o = monolith_solve(P_θ, P_φ, u, utk, ts; pit=P_PIT, t_load=1.5P_HR)
    early = ts .<= 1.5P_HR - 2400                                  # 20 blend widths before the load
    @test maximum(abs.(o.Tcore[early] .- MeatConfig().T0)) < 1e-6 && maximum(abs.(o.q_meat[early])) < 1e-4   # the solver's tolerance, the blend's 2e-9 tail
    @test o.Tcore[end] > 320.0                                      # and it cooks after
    o0 = monolith_solve(P_θ, P_φ, u, utk, ts; pit=P_PIT)
    o1 = monolith_solve(P_θ, P_φ, u, utk, ts; pit=P_PIT, t_load=LOAD_AT_START)
    @test o0.Tcore == o1.Tcore && o0.Tpit == o1.Tpit               # the default is in from the start
    # the 2-D co-integrator: the meat not stepped before its load
    o2 = coupled_solve_2d(P_θ[[1, 2, 3, 5, 6, 7, 8]], P_φ, u, utk, ts; pit=P_PIT, t_load=1.5P_HR, nr=8, nz=8)
    @test all(o2.Tcore[early] .== MeatConfig().T0) && all(o2.q_meat[early] .== 0)
end

@testset "preheat: the time the empty pit reaches the load temperature, and its derivatives" begin
    utk = collect(range(0.0, 12P_HR; length=13)); u = fill(0.6, 13)
    t, ok = preheat_time(330.0, P_φ, u, utk; pit=P_PIT)
    @test ok && 0 < t < 3P_HR
    # the empty pit at that time is at the load temperature
    o = monolith_solve(P_θ, P_φ, u, utk, [0.0, t]; pit=P_PIT, t_load=1e9,
                       abstol=1e-10, reltol=1e-10)
    @test abs(o.Tpit[end] - 330.0) < 0.1
    t2, ok2, sf2 = preheat_time(600.0, P_φ, u, utk; pit=P_PIT, t_max=P_HR, abstol=1e-10, reltol=1e-10)
    @test !ok2 && t2 == P_HR                                        # out of reach: flagged, not an error
    # the shortfall is how far short the empty pit stayed at the cap, and it has a gradient (the damper)
    oc = monolith_solve(P_θ, P_φ, u, utk, [0.0, P_HR]; pit=P_PIT, t_load=1e9, abstol=1e-10, reltol=1e-10)
    @test abs(sf2 - (600.0 - oc.Tpit[end])) < 0.05
    @test preheat_time(330.0, P_φ, u, utk; pit=P_PIT)[3] == 0.0     # reached: no shortfall
    fsf(v) = preheat_time(600.0, P_φ, v, utk; pit=P_PIT, t_max=P_HR, abstol=1e-10, reltol=1e-10)[3]
    gsf = ForwardDiff.gradient(fsf, u)
    up, um = copy(u), copy(u); up[1] += 1e-3; um[1] -= 1e-3
    fdsf = (fsf(up) - fsf(um)) / 2e-3
    @test gsf[1] < 0 && abs(gsf[1] - fdsf) <= 1e-4 * abs(fdsf)       # opening the damper closes the gap
    # T_load in the monolith: the same time, its partials through the preheat into the cook
    ts = collect(0.0:600.0:(5P_HR))
    oT = monolith_solve(P_θ, P_φ, u, utk, ts; pit=P_PIT, T_load=330.0)
    ot = monolith_solve(P_θ, P_φ, u, utk, ts; pit=P_PIT, t_load=t)
    @test oT.load_reached && abs(oT.t_load - t) < 1e-6 && oT.Tcore == ot.Tcore
    fT(x) = monolith_solve(P_θ, P_φ, u, utk, ts; pit=P_PIT, T_load=x[1]).Tcore[end]
    gT = ForwardDiff.gradient(fT, [330.0])[1]
    fdT = (fT([330.5]) - fT([329.5])) / 1.0
    @test abs(gT - fdT) <= 0.02 * abs(fdT) + 1e-6
    f(v) = preheat_time(v[1], v[2:8], v[9:end], utk; pit=P_PIT, abstol=1e-10, reltol=1e-10)[1]
    v = vcat(330.0, P_φ, u)
    g = ForwardDiff.gradient(f, v)
    for i in (1, 3, 7, 9, 10)                                       # T_load, UA, k_spread, two damper knots
        h = 1e-3 * abs(v[i]); vp = copy(v); vp[i] += h; vm = copy(v); vm[i] -= h
        fd = (f(vp) - f(vm)) / 2h
        @test abs(g[i] - fd) <= 1e-4 * abs(fd)
    end
    # duration mode: the knots at s·t_f (a varying damper, so the stretch matters) and the pit's start, as partials
    s = utk ./ 12P_HR; u2 = collect(range(0.9, 0.3; length=13))
    fs(x) = preheat_time(330.0, P_φ, u2, s .* x[1]; pit=P_PIT, Tpit0=x[2], abstol=1e-10, reltol=1e-10)[1]
    x0 = [12P_HR, 300.0]
    gs = ForwardDiff.gradient(fs, x0)
    for (i, h) in ((1, 10.0), (2, 0.05))
        xp = copy(x0); xp[i] += h; xm = copy(x0); xm[i] -= h
        fd = (fs(xp) - fs(xm)) / 2h
        @test abs(gs[i] - fd) <= 1e-4 * abs(fd)
    end
end

@testset "preheat: the lid open at the load cools the pit" begin
    ts = collect(0.0:60.0:(2P_HR)); u = fill(0.5, 3); utk = collect(range(0.0, 2P_HR; length=3))
    shut = PitConfig(; T_amb=300.0, Tpit0=300.0, τ_open_load=1e-3)
    oa = monolith_solve(P_θ, P_φ, u, utk, ts; pit=P_PIT, t_load=P_HR)
    ob = monolith_solve(P_θ, P_φ, u, utk, ts; pit=shut, t_load=P_HR)
    k = findfirst(>=(P_HR + 120), ts)
    @test oa.Tpit[k] < ob.Tpit[k] - 0.5
end

@testset "preheat: two pieces loaded at different times" begin
    ts = collect(0.0:600.0:(4P_HR)); u = fill(0.5, 4); utk = collect(range(0.0, 4P_HR; length=4))
    sys = CookSystem([Slab1D(:a), Slab1D(:b)]; pit=P_PIT)
    o = system_solve(sys, [P_θ, P_θ], [1.0, 1.0], P_φ, u, utk, ts;
                     events=[CookEvent(:load, :a, P_HR), CookEvent(:load, :b, 2P_HR)])
    @test all(abs.(o.piece[:a].Tcore[ts .<= P_HR - 2400] .- MeatConfig().T0) .< 1e-6)
    @test all(abs.(o.piece[:b].Tcore[ts .<= 2P_HR - 2400] .- MeatConfig().T0) .< 1e-6)
    @test o.piece[:a].Tcore[end] > o.piece[:b].Tcore[end] > MeatConfig().T0 + 10
    # the monolith and the system agree on one piece loaded at 1 h
    om = monolith_solve(P_θ, P_φ, u, utk, ts; pit=P_PIT, t_load=P_HR)
    os = system_solve(CookSystem(Slab1D(:a); pit=P_PIT), [P_θ], [MeatConfig().A_meat], P_φ, u, utk, ts;
                      events=[CookEvent(:load, :a, P_HR)])
    @test maximum(abs.(om.Tcore .- os.piece[:a].Tcore)) < 0.01 && maximum(abs.(om.Tpit .- os.Tpit)) < 0.01
end

@testset "preheat: the pit alone is the pit before the load" begin
    # Before the load the meat holds its raw state and the pit draws nothing from it (up to the load
    # blend's logistic tail), so a system with NO pieces must reproduce the full system's pit there.
    # The estimator learns the fire on this pit-only system while the meat is not yet on.
    ts = collect(0.0:60.0:(2P_HR)); u = fill(0.5, 3); utk = collect(range(0.0, 2P_HR; length=3))
    tload = 1.5P_HR
    sys0 = CookSystem(PieceModel[]; pit=P_PIT)
    @test isempty(sys0.pieces) && sys0.layout.n == 6
    # At tight tolerances: the two systems take different step sequences, and at the solver's default
    # 1e-6 their pit temperatures differ by ~1e-3 K from integration alone (falling ~100x per 100x
    # tolerance: 1.2e-3, 3.7e-5, 3.8e-7 K at 1e-6, 1e-8, 1e-10).
    o0 = system_solve(sys0, Vector{Float64}[], Float64[], P_φ, u, utk, ts; abstol=1e-10, reltol=1e-10)
    os = system_solve(CookSystem(Slab1D(:a); pit=P_PIT), [P_θ], [MeatConfig().A_meat], P_φ, u, utk, ts;
                      events=[CookEvent(:load, :a, tload)], abstol=1e-10, reltol=1e-10)
    before = ts .<= tload - 1800                       # the logistic tail is ~3e-7 here
    @test maximum(abs.(o0.Tpit[before] .- os.Tpit[before])) < 1e-5
    @test maximum(abs.(o0.m_fuel[before] .- os.m_fuel[before])) < 1e-6
    @test maximum(abs.(o0.ignited[before] .- os.ignited[before])) < 1e-6
    @test all(iszero, o0.q_meat) && all(iszero, o0.mass) && isempty(o0.pieces)
    # after the load the meat draws heat, so the two must part
    @test maximum(abs.(o0.Tpit[ts .>= tload + 1800] .- os.Tpit[ts .>= tload + 1800])) > 1.0
end
