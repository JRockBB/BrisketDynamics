# The online square-root fixed-lag backend (online.jl) on the current 1-D physics through the system adapter,
# on the real-cook mock's first arrivals:
#   the belief layout of a one-slab system is the filter's FilterLayout, entry for entry
#   the dynamics factor's propagation is the monolith's hand-off; the semantic observables are the channels'
#   updates on arrival: convergence, physical beliefs, the innovations, the lag respected
#   fixed lag against the full history on the same readings; a delayed reading against the in-order run
#   sensor subsets, an observable the system cannot supply, a censored reading, a reading outside the lag
#   the restart contract: the snapshot plans as the filter's state does
#   parity with the RxInfer filter at the tangent (n_sigma = 0) over the first arrivals
#   julia --project=estimator estimator/test/test_online.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test, ForwardDiff, Printf
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
blk_old = latent_block(joinpath(REPO, "data", "calibration", "prior_next_cook.json"), cfg)
ev = m["events"]
events_nt = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
events = [CookEvent(:wrap, :brisket, ev["t_wrap"]), CookEvent(:pull, :brisket, ev["t_pull"]), CookEvent(:add, :pit, ev["t_add"]),
          CookEvent(:spritz, :brisket, ev["t_spritz"]; dt=ev["dt_spritz"], m=ev["m_spritz"])]
np = realcook_noise_priors(joinpath(REPO, "data", "calibration", "noise_prior_real.json"), dt)
tau(npr) = -dt / log(clamp(npr.ρ_mean, 1e-6, 0.999))
nc = RealCookNoise()
sys = CookSystem(Slab1D(:brisket; cfg); pit)
blk = system_latents(sys, blk_old.prior_phys, blk_old.phys_names; probes=[:brisket])
sensors = [SensorSpec(:pit, PitTemperature(), NoiseModel(white_sd=0.1, ar_sd=np[1].σ_mean, tau=tau(np[1]))),
           SensorSpec(:core, TemperatureAt(:brisket; depth_latent=:core_depth_fraction), NoiseModel(white_sd=0.1, ar_sd=np[2].σ_mean, tau=tau(np[2]))),
           SensorSpec(:humidity, ChamberVapor(), NoiseModel(white_sd=10.0, ar_sd=np[3].σ_mean, tau=tau(np[3]), cap=nc.hum_cap)),
           SensorSpec(:mass, TotalMass(), NoiseModel(white_sd=0.005, ar_sd=np[4].σ_mean, tau=tau(np[4])))]
rd = m["readings"]; mk = m["masks"]
readings = (pit=Float64.(rd["pit"]), core=Float64.(rd["core"]), humidity=Float64.(rd["humidity"]), mass=Float64.(rd["mass"]))
masks = (pit=Bool.(mk["pit"]), core=Bool.(mk["core"]), humidity=Bool.(mk["humidity"]), mass=Bool.(mk["mass"]))
"The readings of the mock up to grid point `kmax` as records, in arrival order."
function records_to(kmax; ids=(:pit, :core, :humidity, :mass))
    recs = MeasurementRecord[]
    for k in 1:kmax, id in ids
        getfield(masks, id)[k] && push!(recs, MeasurementRecord(id, t[k], getfield(readings, id)[k]))
    end
    return recs
end
"The arrivals grouped by time."
arrivals(recs) = [[r for r in recs if r.t == tk] for tk in unique([r.t for r in recs])]
K_END = 8                                                      # the first eight grid points: fourteen minutes
# TEST_ONLINE_ONLY="parity,subsets" runs those testsets alone (their short names below); empty runs every one
const ONLY = filter(!isempty, split(get(ENV, "TEST_ONLINE_ONLY", ""), ","))
run_set(name) = isempty(ONLY) || name in ONLY
recs = records_to(K_END)
clog = CookLog(t, u; events, sensors, measurements=recs)
model = SystemModel(sys, clog, blk; sn=StateNoise())
z0 = [z_moments(p)[1] for p in blk.sites]
x0 = from_z(z0[1:N_PHYS], blk.sites[1:N_PHYS])
AREA = latent_index(blk, "A_meat"); DEPTH = latent_index(blk, "core_depth_fraction")

run_set("layout") && @testset "the belief layout of one slab is the filter's; the latents are the block's" begin
    bl = model.bl
    o0 = monolith_solve(x0[1:N_THETA], x0[(N_THETA + 1):N_PHYS], u[1:2], t[1:2], t[1:2]; pit, cfg, A_meat=exp(z0[AREA]), events_nt...)
    ics, a = handoff(o0, 1)
    lay = FilterLayout(length(ics.T0f), length(blk), 4)
    @test bl.n == lay.n_state && bl.kinds == E.state_kinds(lay)
    @test first.(bl.groups) == vcat(["brisket." * g for g in ("T", "W", "C", "F", "S", "bark")], collect(STATE_GROUPS[7:end]))
    @test [r for (_, r) in bl.groups] == [r for (_, r) in state_groups(lay)]
    # the same physical state in both packings
    p, play = pack_params(sys, [x0[1:N_THETA]], [exp(z0[AREA])], x0[(N_THETA + 1):N_PHYS], u[1:2]; events)
    xs = initial_state(sys, p, play)
    y_sys = belief_encode(belief_from_system(xs, bl), bl)
    y_fil = pack_state(ics, a, lay)
    big = [i for i in 1:bl.n if abs(xs[bl.sys_index][i]) > 1e-3]                      # away from the floors the coordinates are the filter's
    @test maximum(abs.(y_sys[big] .- y_fil[big])) < 1e-9 && length(big) >= 3 * cfg.nx + 5
    xdec = belief_decode(y_sys, bl); xfil = state_from_coord(y_fil, lay)
    small_pos = [i for i in 1:bl.n if !(i in big) && bl.kinds[i] === :positive]; small_unit = [i for i in 1:bl.n if !(i in big) && bl.kinds[i] === :unit]
    @test maximum(abs.(xdec[big] .- xfil[big])) < 1e-9
    @test maximum(abs.(xdec[small_pos] .- xfil[small_pos])) < 2e-6                            # at a positive floor the belief holds 1e-6 off it
    @test maximum(abs.(xdec[small_unit] .- xfil[small_unit])) < 2e-3                           # at a fraction's floor a thousandth
    xb = xdec
    @test maximum(abs.(system_from_belief(xb, bl)[bl.sys_index] .- xs[bl.sys_index])[big]) < 1e-9
    @test maximum(abs.(belief_encode(xb, bl) .- y_sys)) < 1e-9                             # a fixed point: the floors hold
    h = belief_handoff(xb, bl)
    @test h.pieces[1].T0f ≈ ics.T0f && h.pit.Tpit0 ≈ ics.Tpit0 && h.pieces[1].LATCH0 == 0.0
    @test isapprox(h.ignited, a; atol=2e-6)
    # the latent block: the physics sites, the area, the depth, in latent_block's order
    @test length(blk) == 17 && blk.names == blk_old.names && AREA == 16 && DEPTH == 17
    @test blk.bindings[1] == ((:brisket, :theta), 1) && blk.bindings[9] == ((:pit, :phi), 1) && blk.bindings[AREA] == ((:brisket, :A_meat), 1) && !haskey(blk.bindings, DEPTH)
    @test latent_key(blk, 1) == LatentKey(:piece, :brisket, :D_eff)
    @test latent_key(blk, DEPTH) == LatentKey(:sensor, :brisket, :core_depth_fraction)
    dotted = Symbol("rack.a")
    dsys = CookSystem([Slab1D(dotted; cfg), Slab1D(:rack; cfg)]; pit)
    dblk = system_latents(dsys, blk_old.prior_phys, blk_old.phys_names; probes=[dotted])
    @test latent_index(dblk, LatentKey(:sensor, dotted, :core_depth_fraction)) == length(dblk)
    @test length(unique(dblk.keys)) == length(dblk)                         # dotted display names cannot collide semantically
    dbl = belief_layout(dsys)
    Qdot = belief_process_cov(dbl, Dict(dotted => StateNoise(T=1.0), :rack => StateNoise(T=0.5), :pit => StateNoise()), 1.0)
    rd = dbl.groups[findfirst(g -> first(g) == "rack.a.T", dbl.groups)][2]
    rr = dbl.groups[findfirst(g -> first(g) == "rack.T", dbl.groups)][2]
    @test Qdot[first(rd), first(rd)] ≈ 4 * Qdot[first(rr), first(rr)]
    q = apply_latents(blk, z0, p, play)
    @test q[play[(:brisket, :theta)]] ≈ x0[1:N_THETA] && q[play[(:pit, :phi)]] ≈ x0[(N_THETA + 1):N_PHYS] && q[first(play[(:brisket, :A_meat)])] ≈ exp(z0[AREA])
    # the process covariance: the filter's, group by group
    sn = StateNoise()
    Qb = belief_process_cov(bl, sn, 0.5)
    Qf = state_noise_cov(lay, sn, E.state_rates(sn, lay) .* sqrt(0.5))
    @test maximum(abs.(Qb .- Qf)) < 1e-12
    @test maximum(abs.(belief_init_cov(bl, sn) .- state_noise_cov(lay, sn, E.state_init_sd(sn, lay)))) < 1e-12
    Qe = belief_process_cov(bl, sn, 0.5; events=[:add])
    ip = first(E.belief_index(bl, :pit, :T_pit)); ifu = first(E.belief_index(bl, :pit, :m_fuel))
    @test Qe[ip, ip] - Qb[ip, ip] ≈ sn.T_event^2 && Qe[ifu, ifu] - Qb[ifu, ifu] ≈ sn.m_add^2
    # the coordinate map: the filter's decode Jacobians under the belief's smooth cap (the filter's own cap is a hard minimum;
    # the two agree where the cap is far)
    d_f = inv.(max.(E.state_decode_jacobian(y_fil, lay), eps(Float64)))
    sd_f = sqrt.(max.(diag(Qf) .* d_f .^ 2, 0.0))
    e_f = (1 .+ (sd_f ./ 3.0) .^ 8) .^ (-1 / 8) .* d_f
    Qc_b = belief_cov_to_coord(Qb, y_sys, bl)
    @test maximum(abs.(Qc_b[big, big] .- Matrix(Symmetric((e_f .* Qf) .* e_f'))[big, big])) < 1e-12
    far = [i for i in big if sd_f[i] < 0.05 * 3.0]
    @test length(far) > 10 && maximum(abs.(Qc_b[far, far] .- E._state_cov_to_coord(Qf, y_fil, lay)[far, far])) < 1e-9
end

run_set("observables") && @testset "the dynamics factor propagates as the monolith hands off; the observables read as the channels" begin
    bl = model.bl
    o0 = monolith_solve(x0[1:N_THETA], x0[(N_THETA + 1):N_PHYS], u[1:2], t[1:2], t[1:2]; pit, cfg, A_meat=exp(z0[AREA]), events_nt...)
    ics, a = handoff(o0, 1)
    lay = FilterLayout(length(ics.T0f), length(blk), 4)
    y0 = pack_state(ics, a, lay)
    y1, o = E.propagate(model, y0, z0, t[1], t[3]; first_slice=true)
    om = monolith_solve(x0[1:N_THETA], x0[(N_THETA + 1):N_PHYS], u[1:3], t[1:3], t[1:3]; pit, cfg, A_meat=exp(z0[AREA]), events_nt..., ics...,
                        abstol=model.abstol, reltol=model.reltol)                                      # the reference at the propagator's tolerances
    ics2, a2 = handoff(om, 3)
    xr = state_from_coord(pack_state(ics2, a2, lay), lay); xd = belief_decode(y1, bl)
    away = [i for i in 1:bl.n if abs(xr[i]) > 1e-2]
    @test maximum(abs.(xd[away] .- xr[away])) < 1e-4
    @test maximum(abs.(xd .- xr)) < 2e-3                                             # at a fraction's floor the belief holds a thousandth off it
    @test length(o.t) == 2 && abs(o.Tpit[end] - om.Tpit[end]) < 1e-4
    # the observables against the channel registry on the stacked output
    d = estimator_setup(t[1:3], u[1:3], pit, cfg, blk_old.prior_phys, blk_old.phys_names)
    ph = PhysicsState(AREA, events_nt, lay, true)
    S, layS = physics_stack(vcat(z0, y0, zeros(4)), d, ph)
    xs = system_from_belief(belief_decode(y0, bl), bl)
    p, _, tk = E._slice_params(model, z0, t[1], t[2])
    hT = compile_observable(TemperatureAt(:brisket; depth_latent=:core_depth_fraction), sys, bl, blk, clog)
    hP = compile_observable(PitTemperature(), sys, bl, blk, clog)
    hV = compile_observable(ChamberVapor(), sys, bl, blk, clog)
    hM = compile_observable(TotalMass(), sys, bl, blk, clog)
    hS = compile_observable(SurfaceTemperature(:brisket), sys, bl, blk, clog)
    hB = compile_observable(BarkL(:brisket), sys, bl, blk, clog)
    hW = compile_observable(WetBulbTemperature(), sys, bl, blk, clog)
    hL = compile_observable(LatentValue(:L), sys, bl, blk, clog)
    probe = MeatProbe(1.0, 0.0, [true, false, false]; depth_latent=DEPTH)
    rT, _ = E.predict(probe, S, zeros(length(S), 17), z0, blk.sites, layS)
    @test abs(hT(xs, z0, p, tk, t[1]) - rT[1]) < 1e-9                                        # a uniform field: the cubic is the linear
    @test abs(hP(xs, z0, p, tk, t[1]) - S[layS.oTpit + 1]) < 1e-9
    @test abs(hV(xs, z0, p, tk, t[1]) - S[layS.opv + 1]) < 1e-6
    @test abs(hM(xs, z0, p, tk, t[1]) - S[layS.oMass + 1]) < 1e-9
    @test abs(hS(xs, z0, p, tk, t[1]) - S[layS.oTsurf + 1]) < 1e-9 && abs(hB(xs, z0, p, tk, t[1]) - S[layS.oL + 1]) < 1e-9
    Tw, _ = E.wet_bulb(S[layS.oTpit + 1], S[layS.opv + 1], 6.62e-4 * BrisketDynamics.P_ATM)
    @test abs(hW(xs, z0, p, tk, t[1]) - Tw) < 1e-9 && abs(hL(xs, z0, p, tk, t[1]) - exp(z0[4])) < 1e-12
    hum_low = Humidity(p_atm=80_000.0)
    sys_low = CookSystem(Slab1D(:brisket; cfg); pit, hum=hum_low)
    bl_low = belief_layout(sys_low)
    hW_low = compile_observable(WetBulbTemperature(), sys_low, bl_low, blk, clog)
    hW_explicit = compile_observable(WetBulbTemperature(p_atm=BrisketDynamics.P_ATM), sys_low, bl_low, blk, clog)
    pv_low = chamber_vapor(sys_low, xs, p, tk, t[1])
    Tw_low, _ = E.wet_bulb(xs[first(state_index(sys_low.layout, :pit, :T_pit))], pv_low, 6.62e-4 * hum_low.p_atm)
    @test abs(hW_low(xs, z0, p, tk, t[1]) - Tw_low) < 1e-9 &&
          abs(hW_low(xs, z0, p, tk, t[1]) - hW_explicit(xs, z0, p, tk, t[1])) > 1e-3
    @test_throws ArgumentError WetBulbTemperature(p_atm=-1.0)
    @test_throws ArgumentError WetBulbTemperature(a_psy=NaN)
    # the observables differentiate in the state and the latents (at the propagated state: the initial field is uniform,
    # where the probe's depth moves nothing)
    xs1 = system_from_belief(belief_decode(y1, bl), bl)
    g = ForwardDiff.gradient(v -> hT(v[1:length(xs1)], v[(length(xs1) + 1):end], p, tk, t[3]), vcat(xs1, z0))
    @test all(isfinite, g) && g[length(xs1) + DEPTH] != 0 && 2 <= count(!=(0), g[1:cfg.nx]) <= 4      # the cubic reads four nodes
    gw = ForwardDiff.gradient(v -> hW(v, z0, p, tk, t[1]), xs)
    @test all(isfinite, gw) && gw[first(state_index(sys.layout, :pit, :T_pit))] > 0
    # what the system cannot supply is an explicit error
    @test_throws ObservableUnavailable compile_observable(TemperatureAt(:point; s=0.0), sys, bl, blk, clog)
    @test_throws ObservableUnavailable compile_observable(TemperatureAt(:brisket), sys, bl, blk, clog)
    @test_throws ArgumentError compile_observable(LatentValue(:radius), sys, bl, blk, clog)
    # the record's censoring
    @test censored_at(sensors[3], MeasurementRecord(:humidity, 0.0, nc.hum_cap)) && !censored_at(sensors[3], MeasurementRecord(:humidity, 0.0, 6000.0))
    @test_throws ArgumentError CookLog(t, u; sensors, measurements=[MeasurementRecord(:scale, 0.0, 1.0)])
    @test_throws ArgumentError CookLog(t, u; sensors, measurements=[MeasurementRecord(:pit, 0.0, NaN)])
    @test_throws ArgumentError CookLog([0.0, 60.0, 121.0], [0.2, 0.2, 0.2]; sensors)
    @test_throws CanonicalIndexError setindex!(clog.u, 0.9, 1)                 # the log owns immutable vectors
    @test length(with_measurements(clog, MeasurementRecord(:pit, 5000.0, 400.0)).measurements) == length(recs) + 1 && length(clog.measurements) == length(recs)
    ev_to_add = sensor_events(clog, 0.0, events_nt.t_add)                                   # the wrap and two spritz pulses precede the add in this mock
    @test :add in ev_to_add && :wrap in ev_to_add && count(==(:spritz), ev_to_add) == 2 && isempty(sensor_events(clog, 0.0, 600.0))
    @test sensor_events(clog, events_nt.t_wrap + 1.0, events_nt.t_add) == [:add] && :pull in sensor_events(clog, events_nt.t_add, events_nt.t_pull)
    tk, uk = damper_knots(clog, t[2], t[5])
    @test tk == t[2:5] && uk == u[2:5]
    tk2, uk2 = damper_knots(clog, 100.0, 250.0)
    @test tk2[1] == 100.0 && tk2[end] == 250.0 && length(tk2) == 3 && abs(uk2[1] - lininterp(100.0, t[1], dt, u[1:3])) < 1e-12
end

run_set("transactions") && @testset "failed assimilation rolls back; checkpoints and diagnostics are bounded" begin
    sm = online_init(model; lag=1, iters=0, record_limit=2)
    keys0 = copy(sm.graph.keys); nodes0 = window_nodes(sm); z0s = copy(variable_value(sm.graph, key_params(sm)))
    failed = assimilate!(sm, MeasurementRecord(:pit, t[2], readings.pit[2] + 50.0))
    @test !failed.converged && !failed.committed && sm.graph.keys == keys0 && window_nodes(sm) == nodes0
    @test variable_value(sm.graph, key_params(sm)) == z0s && isempty(sm.obs) && sm.committed_t == t[1]
    # Batch validation happens before insertion: the valid first record is not left behind when the second is invalid.
    @test_throws ArgumentError assimilate!(sm, [MeasurementRecord(:pit, t[2], readings.pit[2]), MeasurementRecord(:unknown, t[2], 1.0)])
    @test sm.graph.keys == keys0 && isempty(sm.obs)
    cp = smoother_checkpoint(sm)
    variable_value(sm.graph, key_params(sm))[1] += 1
    restore_checkpoint!(sm, cp)
    @test variable_value(sm.graph, key_params(sm)) == z0s && sm.record_count == 1
    sm2 = restore_checkpoint(cp)
    @test sm2 !== sm && sm2.graph !== sm.graph && variable_value(sm2.graph, key_params(sm2)) == z0s
    sm3 = restore_checkpoint(cp)
    k2 = E._node_at!(sm2, t[2]); k3 = E._node_at!(sm3, t[2])
    @test sm2.graph.keys == sm3.graph.keys && variable_value(sm2.graph, k2) == variable_value(sm3.graph, k3)
    @test E.residual(sm2.dyn[(sm2.nodes[1][2], k2)],
                     [variable_value(sm2.graph, sm2.key_z), variable_value(sm2.graph, sm2.nodes[1][2]), variable_value(sm2.graph, k2)]) ==
          E.residual(sm3.dyn[(sm3.nodes[1][2], k3)],
                     [variable_value(sm3.graph, sm3.key_z), variable_value(sm3.graph, sm3.nodes[1][2]), variable_value(sm3.graph, k3)])
    @test_throws ArgumentError restore_checkpoint(SmootherCheckpoint(3, cp.physics, cp.backend, cp.smoother))
    @test_throws ArgumentError restore_checkpoint(SmootherCheckpoint(2, "not-the-current-physics", cp.backend, cp.smoother))
    @test_throws ArgumentError restore_checkpoint(SmootherCheckpoint(2, cp.physics, "not-the-current-backend", cp.smoother))

    # An exception after the first record has inserted a node/factors restores the exact pre-batch graph:
    # the execution record is intentionally unset on the second record's slice.
    u_partial = copy(u); u_partial[3:end] .= NaN
    bad_log = CookLog(t, u_partial; events, sensors=clog.sensors)
    sm_bad = online_init(SystemModel(sys, bad_log, blk; sn=StateNoise()); lag=1, iters=2)
    bad_keys = copy(sm_bad.graph.keys); bad_factors = copy(sm_bad.graph.factors)
    @test_throws ArgumentError assimilate!(sm_bad,
        [MeasurementRecord(:pit, t[2], readings.pit[2]), MeasurementRecord(:pit, t[3], readings.pit[3])])
    @test sm_bad.graph.keys == bad_keys && sm_bad.graph.factors == bad_factors &&
          length(sm_bad.nodes) == 1 && isempty(sm_bad.obs)

    # Rollback preserves a shared graph object and the other smoother's factor identities and cache.
    shared = FactorGraph()
    sa = online_init(model; graph=shared, cook=:shared_a, lag=1, iters=0)
    sb = online_init(model; graph=shared, cook=:shared_b, lag=1, iters=2)
    kb = E._node_at!(sb, t[2]); pairb = (sb.nodes[1][2], kb); fb = sb.dyn[pairb]
    E.residual(fb, [variable_value(shared, k) for k in vars(fb)])
    cacheb = deepcopy(fb.cache); shared_keys = copy(shared.keys); shared_values = Dict(k => copy(v) for (k, v) in shared.values)
    rolled = assimilate!(sa, MeasurementRecord(:pit, t[2], readings.pit[2]))
    @test !rolled.committed && sa.graph === sb.graph === shared && shared.keys == shared_keys
    @test sb.dyn[pairb] === fb && fb in shared.factors && fb.cache.z == cacheb.z && fb.cache.y0 == cacheb.y0
    @test all(shared.values[k] == shared_values[k] for k in shared_keys)

    # A checkpoint resumes through a real solve and marginalization with identical values, covariance and past.
    sc = online_init(model; lag=0, iters=12, record_limit=4)
    cc = smoother_checkpoint(sc)
    sc1, sc2 = restore_checkpoint(cc), restore_checkpoint(cc)
    batches = arrivals(recs)[1:2]
    rc1 = [assimilate!(sc1, b) for b in batches]
    rc2 = [assimilate!(sc2, b) for b in batches]
    @test all(x -> x.committed, rc1) && all(x -> x.committed, rc2)
    @test sc1.graph.keys == sc2.graph.keys && sc1.nodes == sc2.nodes && sc1.committed_t == sc2.committed_t
    @test all(sc1.graph.values[k] == sc2.graph.values[k] for k in sc1.graph.keys)
    bc1, bc2 = belief_at(sc1), belief_at(sc2)
    @test bc1.z == bc2.z && bc1.y == bc2.y && bc1.zy_cov == bc2.zy_cov
    @test all(getfield(sc1.past[:brisket], f) == getfield(sc2.past[:brisket], f)
              for f in fieldnames(Past))

    # Execution advances by replacing the immutable log for future factors, never mutating old factors' model.
    sx = online_init(model; lag=2, iters=2)
    kx2 = E._node_at!(sx, t[2]); fold = sx.dyn[(sx.nodes[1][2], kx2)]; model_old = sx.model
    ux = collect(sx.model.log.u); ux[2] = 0.33
    update_execution!(sx; u=ux, events=[CookEvent(:add, :pit, t[3])])
    kx3 = E._node_at!(sx, t[3]); fnew = sx.dyn[(kx2, kx3)]
    @test sx.model !== model_old && model_old.log.u[2] == u[2] && sx.model.log.u[2] == 0.33
    @test fold.model === model_old && fnew.model === sx.model && only(sx.model.log.events).kind === :add

    for i in 1:4; E._push_record!(sm, (synthetic=i,)); end
    @test length(sm.records) == 2 && sm.record_count == 5 && [r.synthetic for r in sm.records] == [3, 4]

    # Many successful window rotations keep every owned collection bounded, not only the live graph's nodes.
    # A four-node slab keeps this acceptance gate cheap while exercising dynamics, observations, an OU chain,
    # marginal factors, committed integrals and the diagnostic ring through twenty arrivals.
    tl = collect(0.0:60.0:1200.0); ul = fill(0.2, length(tl))
    sysl = CookSystem(Slab1D(:long; cfg=MeatConfig(; nx=4, T0=299.8)); pit)
    blkl = system_latents(sysl, blk_old.prior_phys, blk_old.phys_names)
    zl = [z_moments(site)[1] for site in blkl.sites]
    p0l, playl = pack_params(sysl, [zeros(8)], [sysl.pieces[1].cfg.A_meat], vcat(zeros(6), 0.5), ul)
    pl = apply_latents(blkl, zl, p0l, playl)
    truthl = system_solve(sysl, pl, tl, tl)
    spl = SensorSpec(:pit_long, PitTemperature(), NoiseModel(white_sd=0.5, ar_sd=0.5, tau=600.0))
    modell = SystemModel(sysl, CookLog(tl, ul; sensors=[spl]), blkl; sn=StateNoise())
    sml = online_init(modell; lag=2, iters=8, tol=1e-4, ftol=1e-4, record_limit=3)
    sizes = NamedTuple[]
    for k in eachindex(tl)
        rr = assimilate!(sml, MeasurementRecord(:pit_long, tl[k], truthl.Tpit[k]))
        @test rr.committed && rr.converged
        push!(sizes, (keys=length(sml.graph.keys), factors=length(sml.graph.factors),
                      nodes=length(sml.nodes), dyn=length(sml.dyn), obs=length(sml.obs),
                      disc=length(sml.disc[:pit_long]), records=length(sml.records)))
    end
    steady = sizes[5:end]
    @test all(s -> s == first(steady), steady)
    @test first(steady).nodes == 3 && first(steady).dyn == 2 && first(steady).records == 3
end

# the online run over the arrivals; the belief per arrival
function run_online(recs; lag, kw...)
    sm = online_init(SystemModel(sys, CookLog(t, u; events, sensors, measurements=recs), blk; sn=StateNoise()); lag, kw...)
    out = NamedTuple[]
    for grp in arrivals(recs)
        push!(out, assimilate!(sm, grp))
    end
    return sm, out
end

run_set("arrivals") && @testset "updates on arrival: the belief moves toward the readings, the lag is kept, the beliefs are physical" begin
    t0 = time()
    sm, out = run_online(recs; lag=2)
    println("  the fixed-lag run: ", length(out), " arrivals in ", round(Int, time() - t0), " s; iterations ", [r.iters for r in out], "; nodes ", [r.n_nodes for r in out])
    @test length(out) == length(arrivals(recs)) && all(r.converged for r in out)
    @test all(r.n_nodes <= 3 for r in out) && length(sm.nodes) == 3 && sm.committed_t == t[K_END - 3]
    @test all(r.T_pit[2] > 0 && r.pieces[:brisket].T_centre[2] > 0 && r.pieces[:brisket].T_surface[2] > 0 for r in out)
    r = out[end]
    @test abs(r.T_pit[1] - readings.pit[K_END]) < 6.0                          # the belief's pit sits with the pit readings
    @test all(isfinite(v) for (_, v) in r.innovation) && abs(r.innovation[:pit]) < 15.0
    @test all(isfinite, r.z) && all(sqrt.(diag(r.z_cov)) .<= [sqrt(z_moments(p)[2]) for p in blk.sites] .* (1 + 1e-6))   # no latent wider than its prior
    kinds = model.bl.kinds
    @test all(r.x[i] > 0 for i in eachindex(r.x) if kinds[i] == :positive) && all(0 < r.x[i] < 1 for i in eachindex(r.x) if kinds[i] == :unit)
    # the sensors' discrepancy chains: one live variable per sensor at least (the newest is never marginalized)
    @test all(length(sm.disc[id]) >= 1 for id in (:pit, :core)) && sm.disc_committed_t[:pit] > -Inf
    @test isempty(sm.disc[:mass]) || length(sm.disc[:mass]) == 1                  # the scale read once, at the start
    # the innovations at the predicted belief are the readings less the prediction: the scale's first reading sits kilograms
    # from the prior's (the mock's area and thickness are not the prior's centre), and the update absorbs most of it
    @test haskey(out[1].innovation, :mass) && isfinite(out[1].innovation[:mass]) && abs(out[1].innovation[:mass]) > 0.1
    sm1 = online_init(SystemModel(sys, clog, blk; sn=StateNoise()); lag=2)
    r1 = assimilate!(sm1, arrivals(recs)[1])                                          # the first arrival alone: its node is still in the window
    fm = only(f for f in sm1.obs if f.spec.id === :mass && f.t == t[1])
    @test abs(E.innovation(fm, [E.variable_value(sm1.graph, k) for k in vars(fm)])) < 0.1 * abs(r1.innovation[:mass])
    # the Jacobian cache: no factor was linearized more often than the solves it took
    @test all(f.n_jacobians <= f.n_solves for f in values(sm.dyn))
    # the snapshot and the restart contract: the plan from the snapshot starts where the belief is
    snap = belief_snapshot(sm)
    @test snap.t == t[K_END] && snap.provenance.n_nodes == 3 && length(snap.z) == 17 && size(snap.zy_cov) == (17 + model.bl.n, 17 + model.bl.n)
    @test all(f.cache !== nothing &&
              f.cache.z == variable_value(sm.graph, f.key_z) &&
              f.cache.y0 == variable_value(sm.graph, f.key_y0) &&
              f.cache.jz == variable_value(sm.graph, f.key_z) &&
              f.cache.jy0 == variable_value(sm.graph, f.key_y0)
              for f in values(sm.dyn))                               # covariance is linearized at its reported mean
    @test snap.past[:brisket].t == t[K_END] && snap.past[:brisket].leth_s > 0 && snap.past[:brisket].m0 ≈ pit.m0
    s = Schedule(t; u0=u[1]); s.u[1:K_END] = u[1:K_END]; s.t_now = t[K_END]
    st = to_plan_state(snap, s)
    @test st.k == K_END && st.t == t[K_END] && st.ignited == snap.handoff.ignited && st.lay.nx == cfg.nx && st.past === snap.past[:brisket]
    d = estimator_setup(t, fill(0.15, length(t)), pit, cfg, blk_old.prior_phys, blk_old.phys_names)
    f = free_set(s); mζ, _ = plan_prior(f, s)
    o_s, pl, _, tg = plan_solve(snap.z, mζ, f, s, d; state=st)
    @test tg[1] == t[K_END] && abs(o_s.Tcore[1] - r.pieces[:brisket].T_centre[1]) < 1e-6 && abs(o_s.Tpit[1] - r.T_pit[1]) < 1e-6
    # a reading outside the lag is refused
    @test_throws ArgumentError add_measurement!(sm, MeasurementRecord(:pit, t[2], readings.pit[2]))
end

run_set("lag") && @testset "the fixed lag against the full history; a delayed reading against the in-order run" begin
    # the two full-history solves below are compared to 1e-3: pinned by tight tolerances so the gate tests the insertion, not
    # the solver's slack (at the default ftol of 1e-2 nats two paths to the optimum differ by a tenth of a sd)
    sm_full, out_full = run_online(recs; lag=typemax(Int), tol=1e-6, ftol=1e-6)
    sm_lag, out_lag = run_online(recs; lag=1)
    @test length(sm_full.nodes) == K_END && length(sm_lag.nodes) == 2
    rf, rl = out_full[end], out_lag[end]
    @test abs(rf.T_pit[1] - rl.T_pit[1]) < 0.1 && abs(rf.pieces[:brisket].T_centre[1] - rl.pieces[:brisket].T_centre[1]) < 0.05
    @test maximum(abs.(rf.z .- rl.z) ./ sqrt.(diag(rf.z_cov))) < 0.1
    @test maximum(abs.(sqrt.(diag(rf.z_cov)) ./ sqrt.(diag(rl.z_cov)) .- 1)) < 0.1
    # a delayed reading: the pit's sample at the sixth point withheld and assimilated after the eighth; the full-history
    # result is the in-order one, the fixed-lag one within the linearization of what left the window
    late = MeasurementRecord(:pit, t[6], readings.pit[6])
    recs_wo = [r for r in recs if !(r.sensor === :pit && r.t == t[6])]
    sm_d, _ = run_online(recs_wo; lag=typemax(Int), tol=1e-6, ftol=1e-6)
    rd_ = assimilate!(sm_d, [late])
    @test rd_.converged
    @test abs(rd_.T_pit[1] - rf.T_pit[1]) < 1e-3 && maximum(abs.(rd_.z .- rf.z) ./ sqrt.(diag(rf.z_cov))) < 1e-3
    # the delayed node was found, not made again
    @test length(sm_d.nodes) == K_END
    sm_dl, _ = run_online(recs_wo; lag=3)
    rdl = assimilate!(sm_dl, [late])
    @test rdl.converged && abs(rdl.T_pit[1] - rl.T_pit[1]) < 0.2
    # a delayed reading before the window's oldest node, or before a sensor's committed discrepancy, is refused
    @test_throws ArgumentError assimilate!(sm_lag, [MeasurementRecord(:pit, t[3], readings.pit[3])])
end

run_set("subsets") && @testset "sensor subsets, dropouts and a censored reading" begin
    # the pit and the core alone
    sub = records_to(K_END; ids=(:pit, :core))
    sm_s, out_s = run_online(sub; lag=2)
    @test all(r.converged for r in out_s) && isempty(sm_s.disc[:humidity]) && isempty(sm_s.disc[:mass])
    # a dropout: no readings at the third and fourth points; the belief is only propagated there
    drop = [r for r in recs if !(r.t in (t[3], t[4]))]
    sm_g, out_g = run_online(drop; lag=2)
    @test all(r.converged for r in out_g) && length(out_g) == length(arrivals(recs)) - 2
    @test any(r.t == t[5] for r in out_g)
    # a censored humidity reading: the sensor's cap set thirty pascals above the reading it would have made, so the
    # reading at the cap says "at least this", consistent with the cook (a cap far above the chamber's vapor pressure
    # is a reading no model can meet); the factor's loss is the survival likelihood, finite, with the site's gradient
    cap_val = readings.humidity[2] + 30.0
    sensors_c = [sensors[1], sensors[2], SensorSpec(:humidity, ChamberVapor(), NoiseModel(white_sd=10.0, ar_sd=np[3].σ_mean, tau=tau(np[3]), cap=cap_val)), sensors[4]]
    cap_rec = MeasurementRecord(:humidity, t[2], cap_val)
    sm_c = online_init(SystemModel(sys, CookLog(t, u; events, sensors=sensors_c, measurements=vcat(records_to(2; ids=(:pit, :core)), [cap_rec])), blk); lag=2)
    for grp in arrivals(records_to(2; ids=(:pit, :core))); assimilate!(sm_c, grp); end
    fc = add_measurement!(sm_c, cap_rec)
    @test fc.censored
    vals = [E.variable_value(sm_c.graph, k) for k in vars(fc)]
    L = E.loss(fc, vals); rc = E.residual(fc, vals)
    @test isfinite(L) && L >= 0 && length(rc) == 1 && isfinite(rc[1]) && isnan(E.innovation(fc, vals))
    Js = E.jacobians(fc, vals)
    @test length(Js) == 3 && all(all(isfinite, J) for J in Js) && Js[3] == fill(E._censored_normal_site(E._obs_pred(fc, reduce(vcat, vals)), cap_val, 10.0).root_precision, 1, 1)
    r = solve_window!(sm_c)
    @test r.converged
end

run_set("parity") && @testset "parity with the RxInfer filter at the tangent over the first arrivals" begin
    # the same prior, the same noise, the same initial widths, the tangent linearization on both sides: the fixed-lag
    # smoother at lag zero is the iterated filter; the two must agree on the temperatures to a fraction of a kelvin
    # and on the latents to a fraction of a sd (the filter stops its iterations coarsely: iters 4, ftol 0.05)
    K_PAR = 5
    recs_p = records_to(K_PAR)
    sm, out = run_online(recs_p; lag=0)
    chs = realcook_channels(readings, masks, np; cap=nc.hum_cap, depth_latent=DEPTH)
    for ch in chs; ch.mask[(K_PAR + 1):end] .= false; end
    noise = [np[i] for i in 1:4]
    recs_f = sequential_filter(chs, t, u, pit, cfg, blk_old.sites, blk_old.names, events_nt; noise, area_latent=AREA, q=zeros(17),
                               sn=StateNoise(), n_sigma=0, t_end=t[K_PAR], adapt_level=false)
    @test length(out) == length(recs_f) + 1                      # the filter's records start at its first transition; the smoother's at the initial arrival
    for rf in recs_f
        ro = out[findfirst(r -> r.t == rf.t, out)]
        @printf("  t %5.0f s: pit %.2f/%.2f (sd %.2f/%.2f)  centre %.2f/%.2f (sd %.3f/%.3f)  surface %.2f/%.2f\n", ro.t,
                ro.T_pit[1], rf.T_pit[1], ro.T_pit[2], rf.T_pit[2], ro.pieces[:brisket].T_centre[1], rf.T_centre[1],
                ro.pieces[:brisket].T_centre[2], rf.T_centre[2], ro.pieces[:brisket].T_surface[1], rf.T_surface[1])
        @test ro.t == rf.t
        @test abs(ro.T_pit[1] - rf.T_pit[1]) < 0.5 && abs(ro.pieces[:brisket].T_centre[1] - rf.T_centre[1]) < 0.2 && abs(ro.pieces[:brisket].T_surface[1] - rf.T_surface[1]) < 0.5
        @test 0.5 < ro.T_pit[2] / rf.T_pit[2] < 2.0 && 0.5 < ro.pieces[:brisket].T_centre[2] / rf.T_centre[2] < 2.0
        @test maximum(abs.(ro.z .- rf.z) ./ rf.z_sd) < 0.5
    end
end
println("test_online.jl: done")
