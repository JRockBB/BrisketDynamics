# The cook as a system (src/system.jl), gated against the monolith:
#   one-piece parity        a system of one slab reproduces monolith_solve (the closure reduces exactly)
#   identical-piece parity  n explicit copies reproduce n_meat = n
#   permutation             the order of heterogeneous pieces changes nothing but the layout
#   conservation            the pit's heat load and vapor load are the sums over the pieces
#   heterogeneity           two different cuts on one pit see one chamber and pull it differently
#   restart                 a cook in two segments through system_handoff equals the whole cook
#   derivatives             the ForwardDiff Jacobian through the system equals the monolith's for one piece
#   layouts and events      the generated named layouts index the right entries; events land on their targets
using LinearAlgebra

const SYS_HR = 3600.0
const SYS_TSAVE = collect(0.0:600.0:(6SYS_HR))
const SYS_U = [0.15, 0.3, 0.5, 0.4, 0.3, 0.3, 0.25]
const SYS_UTK = collect(range(0.0, 6SYS_HR; length=length(SYS_U)))
const SYS_PIT = PitConfig(; T_amb=300.0, m_pan0=1.0, Tpit0=320.0, λ_pan=1.7e6)

# the meat-side fields of a monolith output against a piece's, at the tolerances a solver difference allows
function _piece_vs_monolith(po, mo; T, W, bark=1e-4, ring=1e-6, j=1e-7, thick=1e-7, scalar=1e-5)
    @test maximum(abs.(po.Tcore .- mo.Tcore)) < T
    @test maximum(abs.(po.Tsurf .- mo.Tsurf)) < T
    @test maximum(abs.(po.T_field .- mo.T_field)) < T
    @test maximum(abs.(po.W_field .- mo.W_field)) < W
    @test maximum(abs.(po.bark_b .- mo.bark_b)) < bark
    @test maximum(abs.(po.bark_s .- mo.bark_s)) < bark
    @test maximum(abs.(po.ring .- mo.ring)) < ring
    @test maximum(abs.(po.j_evap .- mo.j_evap)) < j
    @test abs(po.w_retained - mo.w_retained) < scalar
    @test abs(po.juiciness - mo.juiciness) < scalar
    @test abs(po.evap_w - mo.evap_w) < scalar
    @test abs(po.spritz_w - mo.spritz_w) < scalar / 10
    @test abs(po.bitterness - mo.bitterness) < scalar / 10
    @test maximum(abs.(po.l_thick .- mo.l_thick)) < thick
end

@testset "system: layouts, events and the parameter vector" begin
    a = Slab1D(:flat; cfg=MeatConfig(; nx=12))
    b = Slab1D(:butt; cfg=MeatConfig(; nx=9, geom_m=2.0, bone_frac=0.2))
    sys = CookSystem([a, b]; pit=SYS_PIT)
    lay = sys.layout
    @test lay.pieces == [:flat, :butt] && n_pieces(sys) == 2
    @test n_state(sys) == (5 * 12 + 3) + (5 * 9 + 3) + 6
    @test state_index(lay, :flat, :T) == 1:12 && state_index(lay, :flat, :ring) == (5 * 12 + 3):(5 * 12 + 3)
    @test_throws ArgumentError state_index(lay, :flat, :latch)   # the wrap is the planner's time, not a state
    @test state_index(lay, :butt, :W) == (5 * 12 + 3 + 9 + 1):(5 * 12 + 3 + 18)
    @test state_index(lay, :pit, :T_pit) == (n_state(sys) - 5):(n_state(sys) - 5) && state_index(lay, :pit, :T_fb) == n_state(sys):n_state(sys)
    ks = coordinate_kinds(lay)
    @test length(ks) == n_state(sys) && ks[1] == :positive && ks[state_index(lay, :flat, :C)[1]] == :unit && ks[state_index(lay, :pit, :ignited)[1]] == :unit
    @test_throws ArgumentError state_index(lay, :ribs, :T)
    @test_throws ArgumentError state_index(lay, :flat, :bone)
    @test_throws ArgumentError CookSystem([a, Slab1D(:flat)])
    @test_throws ArgumentError CookSystem(Slab1D(:pit))
    @test_throws ArgumentError CookSystem(Slab1D(Symbol("")))
    @test_throws ArgumentError CookSystem(Slab1D[])
    play = ParamLayout(sys, 7)
    @test n_params(play) == 2 * (8 + 7) + 7 + 7 + 2                        # per piece: θ and seven scalars (t_load the seventh)
    @test play[(:flat, :theta)] == 1:8 && play[(:butt, :theta)] == 16:23 && play[(:pit, :phi)] == 31:37 && play[(:pit, :u_knots)] == 38:44
    @test_throws ArgumentError play[(:ribs, :theta)]
    # events: each piece reads its own, the pit its addition; unnamed events are never
    ev = [CookEvent(:wrap, :flat, 4SYS_HR), CookEvent(:spritz, :butt, 2SYS_HR; dt=1800.0, m=0.06), CookEvent(:add, :pit, 3SYS_HR; dt_add=Inf)]
    ef = piece_events(ev, :flat); eb = piece_events(ev, :butt; cfg=b.cfg)
    @test ef.t_wrap == 4SYS_HR && ef.t_pull == PULL_NEVER && ef.t_spritz == BrisketDynamics.SPRITZ_NEVER
    @test eb.t_wrap == WRAP_NEVER && eb.t_spritz == 2SYS_HR && eb.dt_spritz == 1800.0 && eb.m_spritz == 0.06
    @test pit_events(ev; pit=SYS_PIT).t_add == 3SYS_HR && pit_events(CookEvent[]).t_add == BrisketDynamics.ADD_NEVER
    @test_throws ArgumentError CookEvent(:add, :flat, 1.0)
    @test_throws ArgumentError CookEvent(:wrap, :pit, 1.0)
    @test_throws ArgumentError CookEvent(:baste, :flat, 1.0)
    @test_throws ArgumentError CookEvent(:spritz, :flat, 1.0; dt=0.0)
    @test_throws ArgumentError CookEvent(:spritz, :flat, 1.0, (dt=0.0, m=0.1))
    @test_throws ArgumentError CookEvent(:spritz, :flat, 1.0; dt=Inf)
    @test_throws ArgumentError CookEvent(:spritz, :flat, NaN; dt=1.0)
    @test_throws ArgumentError CookEvent(:spritz, :flat, 1.0; m=-0.1)
    @test_throws ArgumentError CookEvent(:spritz, :flat, 1.0, (dt="hourly",))
    θ = MEAT_θ_DEFAULT; φ = PIT_φ_GRAD
    p, pl2 = pack_params(sys, [θ, θ], [0.12, 0.08], φ, SYS_U; events=ev)
    @test pl2.n == play.n && p[play[(:flat, :t_wrap)]] == [4SYS_HR] && p[play[(:butt, :m_spritz)]] == [0.06] && p[play[(:pit, :t_add)]] == [3SYS_HR]
    up = unpack_params(play, p)
    @test up[(:butt, :A_meat)] == 0.08 && up[(:pit, :u_knots)] == SYS_U && up[(:flat, :theta)] == θ
    shared_train = [CookEvent(:spritz, :flat, 120.0; dt=120.0, m=0.02),
                    CookEvent(:spritz, :butt, 120.0; dt=120.0, m=0.02)]
    coincident_trains = [CookEvent(:spritz, :flat, 120.0; dt=120.0, m=0.02),
                         CookEvent(:spritz, :butt, 120.0; dt=240.0, m=0.02)]
    ps, pls = pack_params(sys, [θ, θ], [0.12, 0.08], φ, SYS_U; events=shared_train)
    pd, _ = pack_params(sys, [θ, θ], [0.12, 0.08], φ, SYS_U; events=coincident_trains)
    xs = initial_state(sys, ps, pls); ds = similar(xs); dd = similar(xs)
    rhs, _ = make_system_rhs(sys, SYS_UTK[1], SYS_UTK[2] - SYS_UTK[1], length(SYS_U))
    rhs(ds, xs, ps, 120.0); rhs(dd, xs, pd, 120.0)
    @test ds ≈ dd                                                        # one physical lid opening at a coincident pulse
    @test_throws ArgumentError pack_params(sys, [θ], [0.12, 0.08], φ, SYS_U)
    @test_throws ArgumentError pack_params(sys, [θ, θ], [0.12, 0.08], φ[1:6], SYS_U)
    @test_throws ArgumentError pack_params(sys, [θ, θ], [0.12, 0.08], φ, SYS_U; events=[CookEvent(:pull, :unknown, 1.0)])
    @test_throws ArgumentError pack_params(sys, [θ, θ], [0.12, 0.08], φ, SYS_U;
                                           events=[CookEvent(:pull, :flat, 1.0), CookEvent(:pull, :flat, 2.0)])
    x0 = initial_state(sys, p, play)
    @test length(x0) == n_state(sys) && x0[state_index(lay, :pit, :ignited)[1]] == φ[7] && x0[state_index(lay, :pit, :T_pit)[1]] == SYS_PIT.Tpit0
    @test all(x0[state_index(lay, :butt, :T)] .== b.cfg.T0) && all(x0[state_index(lay, :flat, :C)] .== 0)
    bad_t = copy(SYS_UTK); bad_t[3] += 1.0
    @test_throws ArgumentError system_solve(sys, [θ, θ], [0.12, 0.08], φ, SYS_U, bad_t, SYS_TSAVE; events=ev)
    @test_throws ArgumentError initial_state(sys, p, play; pit_ics=(mfuel0=-1.0,))
    @test_throws ArgumentError initial_state(a, θ; ics=(T0f=fill(NaN, a.cfg.nx),))
end

@testset "system: one slab reproduces the monolith" begin
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    cfg = MeatConfig(; nx=16, t_salt=6SYS_HR)
    kw = (t_wrap=3.5SYS_HR, t_spritz=SYS_HR, dt_spritz=1800.0, m_spritz=0.04, t_add=2SYS_HR)
    mo = monolith_solve(θ, φ, SYS_U, SYS_UTK, SYS_TSAVE; cfg, pit=SYS_PIT, A_meat=0.11, kw...)
    sys = CookSystem(Slab1D(:brisket; cfg); pit=SYS_PIT)
    ev = [CookEvent(:wrap, :brisket, kw.t_wrap), CookEvent(:spritz, :brisket, kw.t_spritz; dt=kw.dt_spritz, m=kw.m_spritz),
          CookEvent(:add, :pit, kw.t_add)]
    so = system_solve(sys, [θ], [0.11], φ, SYS_U, SYS_UTK, SYS_TSAVE; events=ev)
    po = so.pieces[1]
    @test so.piece[:brisket] === po && po.id == :brisket
    # the closure reduces to the monolith's: the same trajectories to the solver's tolerance
    @test maximum(abs.(so.Tpit .- mo.Tpit)) < 1e-3
    @test maximum(abs.(so.pv_pit .- mo.pv_pit)) < 1e-2
    @test maximum(abs.(so.pv_base .- mo.pv_base)) < 1e-6
    @test maximum(abs.(so.q_meat .- mo.q_meat)) < 1e-2
    @test maximum(abs.(so.m_fuel .- mo.m_fuel)) < 1e-6
    @test maximum(abs.(so.m_pan .- mo.m_pan)) < 1e-6 && maximum(abs.(so.T_pan .- mo.T_pan)) < 1e-3
    @test maximum(abs.(so.smoke_clean .- mo.smoke_clean)) < 1e-6
    _piece_vs_monolith(po, mo; T=1e-3, W=1e-6)
    # the state array holds the trajectory the outputs were read from, in the layout's order
    lay = sys.layout
    @test so.X[state_index(lay, :pit, :T_pit)[1], :] == so.Tpit && so.X[state_index(lay, :brisket, :T), :] == po.T_field
    @test size(so.X) == (n_state(sys), length(SYS_TSAVE))
    # the total mass on a scale is the piece's, and it is the estimator's bookkeeping
    @test so.mass == po.mass && abs(po.mass[1] - BrisketDynamics.RHO_DRY * 0.11 * θ[4] * (1 + cfg.W0)) < 1e-9
    # the humidity observable at a state equals the trajectory's chamber vapor pressure
    p, play = pack_params(sys, [θ], [0.11], φ, SYS_U; events=ev)
    @test all(abs(chamber_vapor(sys, so.X[:, k], p, SYS_UTK, SYS_TSAVE[k]) - so.pv_pit[k]) < 1e-9 for k in (1, 7, 19, 37))
    # A reduced-model exchange correction participates in the same humidity fixed point and heat accounting.
    # It is not an output-only shift: the corrected pressure changes the meat-state derivative.
    correction = t -> (pv=25.0, q=10.0)
    soc = system_outputs(sys, so.X, p, SYS_UTK, SYS_TSAVE; correction)
    @test maximum(abs.(soc.pv_pit .- (so.pv_pit .- 25.0))) < 1e-9
    @test maximum(abs.(soc.q_meat .- (so.q_meat .- 10.0))) < 1e-9
    @test soc.q_meat == soc.piece[:brisket].q_meat
    @test soc.vapor_load ≈ soc.piece[:brisket].vapor_load .+ soc.vapor_correction
    rhs_c, _ = make_system_rhs(sys, SYS_UTK[1], SYS_UTK[2] - SYS_UTK[1], length(SYS_U); correction)
    rhs_0, _ = make_system_rhs(sys, SYS_UTK[1], SYS_UTK[2] - SYS_UTK[1], length(SYS_U))
    dc = zeros(n_state(sys)); d0 = similar(dc)
    rhs_c(dc, so.X[:, 7], p, SYS_TSAVE[7]); rhs_0(d0, so.X[:, 7], p, SYS_TSAVE[7])
    @test norm(dc[state_index(sys.layout, :brisket, :W)] - d0[state_index(sys.layout, :brisket, :W)]) > 0
    # the RHS at one instant: the system's derivative equals the monolith's at the same state and parameters
    rhs_s, _ = make_system_rhs(sys, SYS_UTK[1], SYS_UTK[2] - SYS_UTK[1], length(SYS_U))
    rhs_m = BrisketDynamics.make_monolith_rhs(cfg, SYS_PIT, Humidity(), SYS_UTK[1], SYS_UTK[2] - SYS_UTK[1], length(SYS_U))
    pm = vcat(θ, φ, SYS_U, kw.t_wrap, PULL_NEVER, kw.t_add, kw.t_spritz, kw.dt_spritz, kw.m_spritz, SYS_PIT.dt_add, LOAD_AT_START, 0.11)
    nm = 5 * cfg.nx + 3
    for k in (2, 13, 25)
        xs = so.X[:, k]
        xm = vcat(xs[1:nm], xs[nm + 1], xs[nm + 2], xs[nm + 3], xs[nm + 4], xs[nm + 5], xs[nm + 6])   # the same order for one piece
        ds = zeros(length(xs)); dm = zeros(length(xm))
        rhs_s(ds, xs, p, SYS_TSAVE[k]); rhs_m(dm, xm, pm, SYS_TSAVE[k])
        @test maximum(abs.(ds .- dm) ./ (abs.(dm) .+ 1e-12)) < 1e-9
    end
end

@testset "system: n explicit copies reproduce n_meat = n" begin
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    cfg1 = MeatConfig(; nx=12); cfg3 = MeatConfig(; nx=12, n_meat=3.0)
    kw = (t_spritz=1.5SYS_HR, dt_spritz=SYS_HR, m_spritz=0.05, t_wrap=4SYS_HR)
    # a system three times the monolith's size steps differently: at the default tolerance the two differ by 1e-2 K on
    # the pit and 5e-5 on the water (measured 2026-09-24), thirtyfold less per two decades of tolerance, so both run tight
    tol = 1e-9
    mo = monolith_solve(θ, φ, SYS_U, SYS_UTK, SYS_TSAVE; cfg=cfg3, pit=SYS_PIT, abstol=tol, reltol=tol, kw...)
    ids = (:one, :two, :three)
    sys = CookSystem([Slab1D(id; cfg=cfg1) for id in ids]; pit=SYS_PIT)
    ev = reduce(vcat, [[CookEvent(:spritz, id, kw.t_spritz; dt=kw.dt_spritz, m=kw.m_spritz), CookEvent(:wrap, id, kw.t_wrap)] for id in ids])
    so = system_solve(sys, [θ, θ, θ], fill(cfg1.A_meat, 3), φ, SYS_U, SYS_UTK, SYS_TSAVE; events=ev, abstol=tol, reltol=tol)
    @test maximum(abs.(so.Tpit .- mo.Tpit)) < 5e-4                    # three copies share one spritz train: one lid lift per pulse
    @test maximum(abs.(so.pv_pit .- mo.pv_pit)) < 5e-2                # measured 1.6e-3 to 1.3e-2 Pa across solver builds (a sensor's noise is 100 Pa)
    @test maximum(abs.(so.q_meat .- mo.q_meat)) < 1e-2
    for po in so.pieces
        _piece_vs_monolith(po, mo; T=1e-4, W=1e-6, bark=1e-5, ring=1e-7, j=1e-9, thick=2e-9, scalar=1e-6)
    end
    @test maximum(abs.(so.pieces[1].Tcore .- so.pieces[3].Tcore)) < 1e-9   # identical copies stay identical
    @test so.mass ≈ 3 .* so.pieces[1].mass
    # the same three copies WITHOUT the shared train (one spritzed) differ from n_meat = 3: the closure is per piece
    ev1 = [CookEvent(:spritz, :one, kw.t_spritz; dt=kw.dt_spritz, m=kw.m_spritz)]
    s1 = system_solve(sys, [θ, θ, θ], fill(cfg1.A_meat, 3), φ, SYS_U, SYS_UTK, SYS_TSAVE; events=ev1)
    @test s1.pieces[1].spritz_w > 0 && s1.pieces[2].spritz_w == 0 && maximum(abs.(s1.pieces[1].Tsurf .- s1.pieces[2].Tsurf)) > 0.1
end

@testset "system: heterogeneous pieces, permutation and conservation" begin
    θb = MEAT_θ_DEFAULT
    θr = vcat(θb[1:3], 0.018, θb[5:8])                                # a thinner cut
    φ = PIT_φ_GRAD
    flat = Slab1D(:flat; cfg=MeatConfig(; nx=12))
    rack = Slab1D(:rack; cfg=MeatConfig(; nx=9, geom_m=1.0, bone_frac=0.25, n_meat=2.0, λ_vap=2.0e6))
    ev = [CookEvent(:wrap, :flat, 4SYS_HR), CookEvent(:pull, :rack, 3SYS_HR), CookEvent(:spritz, :rack, 1SYS_HR; dt=1800.0, m=0.05),
          CookEvent(:add, :pit, 2SYS_HR)]
    s12 = system_solve(CookSystem([flat, rack]; pit=SYS_PIT), [θb, θr], [0.12, 0.05], φ, SYS_U, SYS_UTK, SYS_TSAVE; events=ev)
    s21 = system_solve(CookSystem([rack, flat]; pit=SYS_PIT), [θr, θb], [0.05, 0.12], φ, SYS_U, SYS_UTK, SYS_TSAVE; events=ev)
    # permutation: the same cook, the pieces found by name
    @test maximum(abs.(s12.Tpit .- s21.Tpit)) < 1e-6 && maximum(abs.(s12.pv_pit .- s21.pv_pit)) < 1e-4
    for id in (:flat, :rack)
        @test maximum(abs.(s12.piece[id].Tcore .- s21.piece[id].Tcore)) < 1e-6
        @test maximum(abs.(s12.piece[id].W_field .- s21.piece[id].W_field)) < 1e-9
    end
    # conservation: the pit's load is the sum of the pieces' (each with its copies), the chamber vapor is the base plus every load
    @test maximum(abs.(s12.q_meat .- (s12.piece[:flat].q_meat .+ s12.piece[:rack].q_meat))) < 1e-9
    @test s12.mass ≈ s12.piece[:flat].mass .+ 2 .* s12.piece[:rack].mass
    @test all(s12.pv_pit .>= s12.pv_base .- 1e-9) || any(s12.piece[:flat].j_pit .< 0)      # condensation lowers it below the base early
    p12, _ = pack_params(CookSystem([flat, rack]; pit=SYS_PIT), [θb, θr], [0.12, 0.05], φ, SYS_U; events=ev)
    @test all(abs(chamber_vapor(CookSystem([flat, rack]; pit=SYS_PIT), s12.X[:, k], p12, SYS_UTK, SYS_TSAVE[k]) - s12.pv_pit[k]) < 1e-9
              for k in eachindex(SYS_TSAVE))                              # includes the rack's partial-pull blend
    # At the middle of the rack's finite pull blend, the solved pressure must close against the
    # *gated* vapor load. This fails if (1 - pull_blend) is omitted from the fixed-point denominator.
    kpull = findfirst(==(3SYS_HR), SYS_TSAVE)
    plpull = BrisketDynamics.pull_blend(SYS_TSAVE[kpull], 3SYS_HR, rack.cfg)
    mair = max(CookSystem([flat, rack]; pit=SYS_PIT).hum.afr * φ[3] * s12.airflow[kpull], 1e-6)
    vapor_load = s12.piece[:flat].vapor_load[kpull] + s12.piece[:rack].vapor_load[kpull]
    @test 0.01 < plpull < 0.99
    @test s12.pv_pit[kpull] ≈ s12.pv_base[kpull] +
          (CookSystem([flat, rack]; pit=SYS_PIT).hum.p_atm / BrisketDynamics.EPS_W) * vapor_load / mair atol=1e-8
    # the rack is pulled at 3 h: an hour on (the pull blend's tail is 6e-6 there) it draws nothing and adds no vapor,
    # and the flat cooks on
    kp = findfirst(SYS_TSAVE .>= 4SYS_HR)
    @test maximum(abs.(s12.piece[:rack].q_meat[kp:end])) < 1e-4 * maximum(abs.(s12.piece[:rack].q_meat))
    @test maximum(abs.(s12.piece[:rack].vapor_load[kp:end])) < 1e-4 * maximum(abs.(s12.piece[:rack].vapor_load))
    @test s12.piece[:flat].q_meat[end] != 0
    # heterogeneity: two pieces on one pit see one chamber; the thin rack's core leads the flat's; two racks pull the pit lower than one
    @test s12.piece[:rack].Tcore[10] > s12.piece[:flat].Tcore[10]
    one_rack = system_solve(CookSystem([flat, Slab1D(:rack; cfg=with_config(rack.cfg; n_meat=1.0))]; pit=SYS_PIT),
                            [θb, θr], [0.12, 0.05], φ, SYS_U, SYS_UTK, SYS_TSAVE; events=ev)
    @test one_rack.Tpit[10] > s12.Tpit[10] && one_rack.pv_pit[6] < s12.pv_pit[6]
    # the flat alone on the pit runs hotter than beside the rack (the rack's load), and sees drier air
    alone = system_solve(CookSystem(flat; pit=SYS_PIT), [θb], [0.12], φ, SYS_U, SYS_UTK, SYS_TSAVE; events=ev[[1, 4]])
    @test alone.Tpit[10] > s12.Tpit[10] && alone.pv_pit[6] < s12.pv_pit[6]
    # the pieces' surfaces do not see the pit's base vapor but the shared closure: the flat's j_evap beside the rack is
    # smaller early (the rack humidifies the chamber)
    @test alone.piece[:flat].j_evap[4] > s12.piece[:flat].j_evap[4]
end

@testset "system: restart through the hand-off equals the whole cook" begin
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    flat = Slab1D(:flat; cfg=MeatConfig(; nx=12)); rack = Slab1D(:rack; cfg=MeatConfig(; nx=9, geom_m=1.0))
    sys = CookSystem([flat, rack]; pit=SYS_PIT)
    ev = [CookEvent(:wrap, :flat, 3.5SYS_HR), CookEvent(:spritz, :rack, 1SYS_HR; dt=1800.0), CookEvent(:add, :pit, 1.4SYS_HR)]
    whole = system_solve(sys, [θ, θ], [0.12, 0.05], φ, SYS_U, SYS_UTK, SYS_TSAVE; events=ev)
    k = findfirst(SYS_TSAVE .>= 2SYS_HR)
    h = system_handoff(whole, k)
    @test length(h.pieces) == 2 && h.pit.Tpit0 == whole.Tpit[k] && h.ignited ≈ whole.ignited[k] && h.x == whole.X[:, k]
    @test h.pieces[1].R0 == whole.piece[:flat].ring[k] && length(h.pieces[2].T0f) == 9
    tt = SYS_TSAVE[k:end]
    # the named restart: the pieces' fields and the pit's keywords, the ignited fraction in place of a0
    second = system_solve(sys, [θ, θ], [0.12, 0.05], φ, SYS_U, SYS_UTK, tt; events=ev, ics=h.pieces,
                          pit_ics=(; h.pit..., ignited=h.ignited), history=h.history)
    @test maximum(abs.(second.Tpit .- whole.Tpit[k:end])) < 0.05
    @test maximum(abs.(second.piece[:flat].Tcore .- whole.piece[:flat].Tcore[k:end])) < 0.05
    @test maximum(abs.(second.piece[:rack].W_field .- whole.piece[:rack].W_field[:, k:end])) < 1e-4
    @test maximum(abs.(second.pv_pit .- whole.pv_pit[k:end])) < 2.0
    @test abs(second.piece[:flat].evap_w - whole.piece[:flat].evap_w) < 1e-4
    @test abs(second.piece[:rack].spritz_w - whole.piece[:rack].spritz_w) < 1e-4
    @test abs(second.piece[:flat].bitterness - whole.piece[:flat].bitterness) < 1e-4
    # the raw restart: the state vector itself
    p, _ = pack_params(sys, [θ, θ], [0.12, 0.05], φ, SYS_U; events=ev)
    raw = system_solve(sys, p, SYS_UTK, tt; x0=h.x)
    @test maximum(abs.(raw.Tpit .- second.Tpit)) < 1e-6 && maximum(abs.(raw.piece[:rack].Tcore .- second.piece[:rack].Tcore)) < 1e-6
end

@testset "system: forward-mode derivatives through the solve" begin
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    cfg = MeatConfig(; nx=10)
    ts = collect(0.0:900.0:(3SYS_HR))
    u = [0.2, 0.4, 0.3, 0.3]; utk = collect(range(0.0, 3SYS_HR; length=4))
    sys = CookSystem(Slab1D(:b; cfg); pit=SYS_PIT)
    f(o) = [o.Tpit[end], o.pieces[1].Tcore[end], o.pv_pit[end]]
    Js = system_jacobian(f, sys, [θ], [0.12], φ, u, utk, ts)
    # the monolith's Jacobian over (θ, φ) at the same point, the area column by a separate solve
    Jm = monolith_jacobian(o -> [o.Tpit[end], o.Tcore[end], o.pv_pit[end]], θ, φ, u, utk, ts; cfg, pit=SYS_PIT, A_meat=0.12)   # the monolith's, over (θ, φ)
    @test size(Js) == (3, 8 + 1 + 7)
    sel = vcat(1:8, 10:16)
    @test maximum(abs.(Js[:, sel] .- Jm) ./ (abs.(Jm) .+ 1e-3)) < 1e-3
    fm(o) = [o.Tpit[end], o.Tcore[end], o.pv_pit[end]]
    JA = ForwardDiff.derivative(a -> fm(monolith_solve(θ, φ, u, utk, ts; cfg, pit=SYS_PIT, A_meat=a)), 0.12)
    @test maximum(abs.(Js[:, 9] .- JA) ./ (abs.(JA) .+ 1e-3)) < 1e-3
    # the JVP along a tangent on the parameters and the initial state, against the Jacobian
    p, play = pack_params(sys, [θ], [0.12], φ, u)
    x0 = initial_state(sys, p, play)
    dp = zeros(length(p)); dp[play[(:b, :theta)][3]] = 1.0                   # along h_c
    S0 = system_stack(system_solve(sys, p, utk, ts; x0))
    dS = system_jvp(sys, p, x0, utk, ts, dp, zeros(length(x0)))
    @test length(dS) == length(S0) && all(isfinite, dS)
    nt = length(ts)
    @test abs(dS[nt] - Jm[1, 3]) < 1e-3 * (abs(Jm[1, 3]) + 1e-3)             # d Tpit[end] / d h_c
    dx0 = zeros(length(x0)); dx0[state_index(sys.layout, :pit, :T_pit)[1]] = 1.0
    dSx = system_jvp(sys, p, x0, utk, ts, zeros(length(p)), dx0)
    @test dSx[1] ≈ 1.0 && 0 < dSx[nt] < 1                                     # a warmer start: the pit at the end remembers a fraction of it
end
