# Two pieces on one pit assimilated end to end by the online backend: a flat and a bone-in rack, each with its
# own latents, read by the pit probe, a probe in each piece (the flat's at a latent depth, the rack's at its
# centre), the humidity sensor and one scale under both. Gates:
#   the layouts   a two-piece system's belief, latents, noise groups and observables are generated per piece
#   tracking      both pieces' centre temperatures track their planted truths within their widths as the arrivals come
#   coupling      the scale's observable moves with both pieces' water; the pit's belief sees both loads
#   permutation   the same cook with the pieces in the other order gives the same beliefs, by name
#   unobserved    the offline problem names the quality groups of both pieces that nothing reads
#   julia --project=estimator estimator/test/test_multi.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test, ForwardDiff, Printf
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
prior_file = JSON.parsefile(joinpath(REPO, "data", "calibration", "prior_next_cook.json"))
prior_phys = read_sites(prior_file["prior"]); phys_names = String.(prior_file["names"])
pit = PitConfig(; T_amb=300.0, Tpit0=362.4, m0=7.0)
t = collect(0.0:120.0:1800.0)                                   # a half hour on the two-minute grid
u = fill(0.2, length(t))
flat = Slab1D(:flat; cfg=MeatConfig(; nx=10, T0=299.8))
rack = Slab1D(:rack; cfg=MeatConfig(; nx=8, geom_m=1.0, bone_frac=0.25, T0=301.0, A_meat=0.05))
function build(pieces)
    sys = CookSystem(pieces; pit)
    blk = system_latents(sys, prior_phys, phys_names; probes=[:flat])
    return sys, blk
end
sys, blk = build([flat, rack])
z0 = [z_moments(p)[1] for p in blk.sites]
# the planted truth: the flat thicker than the prior's centre, the rack thinner, the pit's leak wider, the flat's probe deep
z_true = copy(z0)
z_true[latent_index(blk, "flat.L")] += 0.15
z_true[latent_index(blk, "rack.L")] -= 0.4
z_true[latent_index(blk, "u_leak")] += 0.3
z_true[latent_index(blk, "flat.core_depth_fraction")] += 0.4
sensors = [SensorSpec(:pit, PitTemperature(), NoiseModel(white_sd=0.1, ar_sd=1.0, tau=3000.0)),
           SensorSpec(:flat_core, TemperatureAt(:flat; depth_latent=Symbol("flat.core_depth_fraction")), NoiseModel(white_sd=0.1, ar_sd=0.5, tau=6000.0)),
           SensorSpec(:rack_core, TemperatureAt(:rack; s=0.0), NoiseModel(white_sd=0.1, ar_sd=0.5, tau=6000.0)),
           SensorSpec(:humidity, ChamberVapor(), NoiseModel(white_sd=10.0, ar_sd=60.0, tau=2700.0, cap=6500.0)),
           SensorSpec(:scale, TotalMass(), NoiseModel(white_sd=0.005, ar_sd=0.01, tau=2600.0))]
"The truth's cook and its readings on the grid (every sensor at every point; the scale every ten minutes)."
function truth_readings(sys, blk; seed=3)
    rng = MersenneTwister(seed)
    tk, uk = t, u
    p0, play = pack_params(sys, [zeros(8) for _ in sys.pieces], [pc.cfg.A_meat for pc in sys.pieces], vcat(zeros(6), 0.5), uk)
    p = apply_latents(blk, z_true, p0, play)
    o = system_solve(sys, p, tk, t)
    bl = belief_layout(sys)
    clog0 = CookLog(t, u; sensors)
    hs = Dict(s.id => compile_observable(s.observable, sys, bl, blk, clog0) for s in sensors)
    recs = MeasurementRecord[]
    for k in eachindex(t), s in sensors
        (s.id === :scale && (k - 1) % 5 != 0) && continue
        v = hs[s.id](o.X[:, k], z_true, p, tk, t[k]) + s.noise.white_sd * randn(rng)
        push!(recs, MeasurementRecord(s.id, t[k], min(v, s.noise.cap)))
    end
    return recs, o
end
recs, o_true = truth_readings(sys, blk)
arrivals(recs) = [[r for r in recs if r.t == tk] for tk in unique([r.t for r in recs])]
function run(sys, blk, recs; lag=2, kmax=length(t))
    clog = CookLog(t, u; sensors, measurements=recs)
    sm = online_init(SystemModel(sys, clog, blk; sn=StateNoise()); lag)
    out = NamedTuple[]
    for grp in arrivals(recs)
        grp[1].t <= t[kmax] || break
        push!(out, assimilate!(sm, grp; verbose=get(ENV, "TEST_MULTI_VERBOSE", "") == "1"))
    end
    return sm, out
end
const ONLY = filter(!isempty, split(get(ENV, "TEST_MULTI_ONLY", ""), ","))
run_set(name) = isempty(ONLY) || name in ONLY

run_set("layout") && @testset "two pieces: the layouts, the latents and the observables are generated per piece" begin
    bl = belief_layout(sys)
    @test bl.n == (5 * 10 + 3) + (5 * 8 + 3) + 6 && bl.sys.pieces == [:flat, :rack]
    @test first.(bl.groups) == vcat(["flat." * g for g in ("T", "W", "C", "F", "S", "bark")], ["rack." * g for g in ("T", "W", "C", "F", "S", "bark")], collect(STATE_GROUPS[7:end]))
    @test length(blk) == 2 * 8 + 7 + 2 + 1 && blk.names[1] == "flat.D_eff" && blk.names[9] == "rack.D_eff" && blk.names[17] == "C_th"
    @test blk.names[end - 2:end] == ["flat.A_meat", "rack.A_meat", "flat.core_depth_fraction"]
    @test blk.bindings[9] == ((:rack, :theta), 1) && blk.bindings[latent_index(blk, "rack.A_meat")] == ((:rack, :A_meat), 1)
    # a StateNoise per piece, or one for all
    Qd = belief_process_cov(bl, Dict(:flat => StateNoise(), :rack => StateNoise(T=1.0), :pit => StateNoise()), 1.0)
    Qs = belief_process_cov(bl, StateNoise(), 1.0)
    rT = bl.groups[findfirst(x -> first(x) == "rack.T", bl.groups)][2]; fT = bl.groups[1][2]
    @test Qd[rT[1], rT[1]] ≈ 4 * Qs[rT[1], rT[1]] && Qd[fT[1], fT[1]] == Qs[fT[1], fT[1]]
    @test_throws ArgumentError belief_process_cov(bl, Dict(:flat => StateNoise()), 1.0)
    # the scale's observable moves with both pieces' water, the rack's twice (its multiplicity is one here, its area smaller)
    clog0 = CookLog(t, u; sensors)
    hM = compile_observable(TotalMass(), sys, bl, blk, clog0)
    p0, play = pack_params(sys, [zeros(8), zeros(8)], [0.12, 0.05], vcat(zeros(6), 0.5), u[1:2])
    p = apply_latents(blk, z0, p0, play)
    xs = o_true.X[:, 1]
    g = ForwardDiff.gradient(x -> hM(x, z0, p, t[1:2], t[1]), xs)
    @test all(g[state_index(sys.layout, :flat, :W)] .> 0) && all(g[state_index(sys.layout, :rack, :W)] .> 0) && all(g[state_index(sys.layout, :flat, :T)] .== 0)
    @test_throws ObservableUnavailable compile_observable(TemperatureAt(:point; s=0.0), sys, bl, blk, clog0)
    @test_throws ArgumentError compile_observable(TemperatureAt(:rack; depth_latent=Symbol("rack.core_depth_fraction")), sys, bl, blk, clog0)   # no such latent: the rack's probe is at its centre
    shared_spritz = [CookEvent(:spritz, :flat, 120.0; dt=120.0, m=0.02),
                     CookEvent(:spritz, :rack, 120.0; dt=120.0, m=0.02)]
    @test sensor_events(CookLog(t, u; events=shared_spritz), 0.0, 360.0) == fill(:spritz, 3)
    distinct_spritz = [CookEvent(:spritz, :flat, 120.0; dt=120.0, m=0.02),
                       CookEvent(:spritz, :rack, 120.0; dt=240.0, m=0.02)]
    @test sensor_events(CookLog(t, u; events=distinct_spritz), 0.0, 360.0) == fill(:spritz, 3)
end

run_set("assimilation") && @testset "two pieces assimilated: both beliefs track their truths, the pit sees both loads" begin
    t0 = time()
    sm, out = run(sys, blk, recs; lag=2)
    println("  two pieces: ", length(out), " arrivals in ", round(Int, time() - t0), " s; iterations ", [r.iters for r in out],
            "; gradients ", [round(r.gradient; sigdigits=3) for r in out], "; committed ", [r.committed for r in out])
    @test length(out) == length(t) && all(r.converged for r in out)
    for (k, r) in enumerate(out)
        @test r.t == t[k]
        for (id, po) in ((:flat, o_true.piece[:flat]), (:rack, o_true.piece[:rack]))
            m, s = r.pieces[id].T_centre
            @test abs(m - po.Tcore[k]) < 3 * s + 0.3 && s > 0
        end
        @test abs(r.T_pit[1] - o_true.Tpit[k]) < 3 * r.T_pit[2] + 0.3
    end
    r = out[end]
    # the latents moved toward the planted truths where the readings speak: the rack's thickness and the leak
    for nm in ("rack.L", "u_leak")
        i = latent_index(blk, nm)
        @test abs(r.z[i] - z_true[i]) < abs(z0[i] - z_true[i]) + 1e-9
    end
    @test all(sqrt.(diag(r.z_cov)) .<= [sqrt(z_moments(p)[2]) for p in blk.sites] .* (1 + 1e-6))
    kinds = sm.model.bl.kinds
    @test all(r.x[i] > 0 for i in eachindex(r.x) if kinds[i] == :positive) && all(0 < r.x[i] < 1 for i in eachindex(r.x) if kinds[i] == :unit)
    # the snapshot carries both pieces' hand-offs and pasts, and refuses the one-slab planner contract
    snap = belief_snapshot(sm)
    @test length(snap.handoff.pieces) == 2 && length(snap.handoff.pieces[2].T0f) == 8 && Set(keys(snap.past)) == Set([:flat, :rack])
    @test snap.past[:flat].t == t[end] && snap.past[:rack].leth_c > 0
    s = Schedule(t; u0=u[1]); s.t_now = t[end]; s.u .= u
    @test_throws ArgumentError to_plan_state(snap, s)
    # the pieces' hand-off restarts the system where the belief is
    o_r = system_solve(sys, [from_z(r.z[1:8], blk.sites[1:8]), from_z(r.z[9:16], blk.sites[9:16])], [exp(r.z[latent_index(blk, "flat.A_meat")]), exp(r.z[latent_index(blk, "rack.A_meat")])],
                       from_z(r.z[17:23], blk.sites[17:23]), u[end - 1:end], t[end - 1:end], t[end - 1:end];
                       ics=snap.handoff.pieces, pit_ics=(; snap.handoff.pit..., ignited=snap.handoff.ignited))
    @test abs(o_r.piece[:flat].Tcore[1] - r.pieces[:flat].T_centre[1]) < 1e-6 && abs(o_r.piece[:rack].Tcore[1] - r.pieces[:rack].T_centre[1]) < 1e-6
end

run_set("permutation") && @testset "permutation: the pieces in the other order give the same beliefs, by name" begin
    sys2, blk2 = build([rack, flat])
    @test blk2.names[1] == "rack.D_eff" && blk2.names[9] == "flat.D_eff"
    sm1, out1 = run(sys, blk, recs; lag=2, kmax=5)
    sm2, out2 = run(sys2, blk2, recs; lag=2, kmax=5)
    println("  permutation gradients: ", [round(r.gradient; sigdigits=3) for r in out1], " / ",
            [round(r.gradient; sigdigits=3) for r in out2], "; attempts ",
            [r.attempts for r in out1], " / ", [r.attempts for r in out2])
    temperature_differences = [abs(getproperty(r1.pieces[id], field)[1] - getproperty(r2.pieces[id], field)[1])
                               for (r1, r2) in zip(out1, out2), id in (:flat, :rack),
                                   field in (:T_centre, :T_surface)]
    latent_differences = [abs(r1.z[latent_index(blk, nm)] - r2.z[latent_index(blk2, nm)])
                          for (r1, r2) in zip(out1, out2),
                              nm in ("flat.L", "rack.L", "u_leak", "flat.core_depth_fraction")]
    println("  permutation max differences: temperature ", maximum(temperature_differences),
            " K; latent coordinate ", maximum(latent_differences))
    for (r1, r2) in zip(out1, out2)
        @test abs(r1.T_pit[1] - r2.T_pit[1]) < 1e-3 && abs(r1.T_pit[2] - r2.T_pit[2]) < 1e-3
        for id in (:flat, :rack)
            # The accepted adaptive-ODE stationary points may differ by its measured resolution
            # envelope, but an identity/index swap is orders of magnitude larger (the pieces start
            # 1.2 K apart and their planted thickness coordinates differ by 0.55).
            @test abs(r1.pieces[id].T_centre[1] - r2.pieces[id].T_centre[1]) < 0.03 &&
                  abs(r1.pieces[id].T_surface[1] - r2.pieces[id].T_surface[1]) < 0.03
        end
        for nm in ("flat.L", "rack.L", "u_leak", "flat.core_depth_fraction")
            @test abs(r1.z[latent_index(blk, nm)] - r2.z[latent_index(blk2, nm)]) < 0.005
        end
    end
end

run_set("whole") && @testset "whole multi-piece cook with distinct events assimilates through both pulls" begin
    tc = collect(0.0:600.0:(6 * 3600.0))
    uc = [0.20 + 0.08 * sin(π * τ / last(tc)) for τ in tc]
    evc = [
        CookEvent(:spritz, :flat, 900.0; dt=1800.0, m=0.03),
        CookEvent(:wrap, :flat, 2 * 3600.0),
        CookEvent(:pull, :flat, 5 * 3600.0),
        CookEvent(:spritz, :rack, 1200.0; dt=2400.0, m=0.02),
        CookEvent(:wrap, :rack, 1.5 * 3600.0),
        CookEvent(:pull, :rack, 3.5 * 3600.0),
        CookEvent(:add, :pit, 3600.0),
    ]
    p0, play = pack_params(sys, [zeros(8), zeros(8)], [flat.cfg.A_meat, rack.cfg.A_meat],
                           vcat(zeros(6), 0.5), uc; events=evc)
    p = apply_latents(blk, z0, p0, play)
    truth = system_solve(sys, p, tc, tc)
    ss = [
        SensorSpec(:whole_pit, PitTemperature(), NoiseModel(white_sd=0.5)),
        SensorSpec(:whole_flat, TemperatureAt(:flat; s=0.0), NoiseModel(white_sd=0.4)),
        SensorSpec(:whole_rack, TemperatureAt(:rack; s=0.0), NoiseModel(white_sd=0.4)),
        SensorSpec(:whole_humidity, ChamberVapor(), NoiseModel(white_sd=100.0)),
    ]
    log0 = CookLog(tc, uc; events=evc, sensors=ss)
    hs = Dict(sp.id => compile_observable(sp.observable, sys, belief_layout(sys), blk, log0) for sp in ss)
    rec_whole = MeasurementRecord[]
    for k in 1:6:length(tc), sp in ss
        push!(rec_whole, MeasurementRecord(sp.id, tc[k], hs[sp.id](truth.X[:, k], z0, p, tc, tc[k])))
    end
    model_whole = SystemModel(sys, CookLog(tc, uc; events=evc, sensors=ss, measurements=rec_whole), blk; sn=StateNoise())
    sm_whole = online_init(model_whole; lag=2)
    updates = [assimilate!(sm_whole, batch) for batch in arrivals(rec_whole)]
    @test length(updates) == 7 && all(r -> r.committed && r.converged, updates)
    @test belief_at(sm_whole).t == last(tc)
    @test maximum(abs.(truth.piece[:rack].q_meat[findall(>=(4.5 * 3600.0), tc)])) <
          1e-3 * maximum(abs.(truth.piece[:rack].q_meat))
    @test maximum(abs.(truth.piece[:flat].q_meat[findall(>=(6 * 3600.0), tc)])) <
          1e-3 * maximum(abs.(truth.piece[:flat].q_meat))
    @test Set((e.target, e.kind) for e in model_whole.log.events) ==
          Set([(:flat, :spritz), (:flat, :wrap), (:flat, :pull),
               (:rack, :spritz), (:rack, :wrap), (:rack, :pull), (:pit, :add)])
end

run_set("offline") && @testset "the offline problem names both pieces' unobserved quality states" begin
    # Resolve the strict offline stopping gate without changing the state-noise
    # model. The online fixtures above retain their production default accuracy.
    m = SystemModel(sys, CookLog(t, u; sensors, measurements=[r for r in recs if r.t <= t[4]]), blk; sn=StateNoise(),
                    abstol=1e-12, reltol=1e-10, floor_abstol=1e-10, floor_reltol=1e-8)
    prob = offline_problem([m]; ids=[:two], iters=15, tol=1e-4, ftol=1e-3)
    r = solve!(prob; sparse=true)
    @test r.converged
    ug = unobserved_groups(prob, 1)
    println("  unobserved groups: ", ug)
    # Over this short prefix the probes and humidity inform temperature, water, collagen and bark through
    # the dynamics. The single initial total-mass row cannot separate either piece's fat or inert salt.
    @test all(g in ug for g in ("flat.F", "flat.S", "rack.F", "rack.S"))
    @test !any(g in ug for g in ("flat.T", "rack.T", "T_pit", "flat.W", "rack.W", "flat.C", "rack.C", "flat.bark", "rack.bark"))
end
println("test_multi.jl: done")
