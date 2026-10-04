# Reduced-order artifacts (rom.jl): a half-thickness brick with its top and sides exposed as the full order, the
# slab of the same thickness and top area as the reduced order, both in one pit:
#   training     the error model's bias and spread per observable from a design of three draws; the envelope
#   held-out     a fourth draw's errors sit inside the model's bands on the observables a probe and a scale read
#   the artifact round-trips through JSON; a stale fingerprint is refused; projection and reconstruction agree
#   online       an observation factor under the artifact carries the bias and the widened floor
#   julia --project=estimator estimator/test/test_rom.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test, Printf
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
prior_file = JSON.parsefile(joinpath(REPO, "data", "calibration", "prior_next_cook.json"))
prior_phys = read_sites(prior_file["prior"]); phys_names = String.(prior_file["names"])
pit = PitConfig(; T_amb=300.0, Tpit0=362.4, m0=7.0)
L, Ly, Lz = 0.03, 0.10, 0.16
cfg = MeatConfig(; nx=8, n_shrink=0.0, T0=282.0, A_meat=Ly * Lz)
t = collect(0.0:600.0:(2 * 3600.0)); u = fill(0.25, length(t))
fom = Brick3D(:b; Lx=L, Ly, Lz, nx=6, ny=3, nz=4, exposed=(false, true, true, true, true, true), cfg)
rom_piece = Slab1D(:b; cfg)
sys_fom = CookSystem(fom; pit); sys_rom = CookSystem(rom_piece; pit)
# This artifact represents one fixed FOM geometry, so its online slab prior is centred on that geometry.
# The design still varies the surrogate's effective thickness around it and learns that mismatch as ROM error;
# it must not silently attach to the calibration prior's different nominal thickness.
prior_rom = copy(prior_phys)
prior_rom[4] = PriorSite("lognormal", log(L), prior_phys[4].p2)
blk = system_latents(sys_rom, prior_rom, phys_names; area=false)
z0 = [z_moments(p)[1] for p in blk.sites]
rng = MersenneTwister(4)
"A draw of the design: the convective coefficient, the diffusivity and the leak moved, the thickness held."
function draw()
    z = copy(z0)
    for nm in ("h_c", "D_eff", "k_m", "u_leak")
        i = latent_index(blk, nm); z[i] += 0.5 * sqrt(z_moments(blk.sites[i])[2]) * randn(rng)
    end
    iL = latent_index(blk, "L")
    z[iL] += 2 * sqrt(z_moments(blk.sites[iL])[2])                         # cover posterior planning draws in effective thickness
    return z
end
d1 = draw()
design = [d1, 2 .* z0 .- d1, copy(z0)]                                 # bracket the online prior centre in every coordinate
heldout = [0.25 .* design[1] .+ 0.75 .* design[2]]                      # unseen interior point: validation, not extrapolation
u_train = 0.20 .+ 0.12 .* sin.(range(0, π; length=length(t)))
train_scenarios = [
    (u_knots=u, u_tknots=t, tsave=t, events=CookEvent[]),
    (u_knots=u_train, u_tknots=t, tsave=t,
     events=[CookEvent(:spritz, :b, 1800.0; dt=2400.0, m=0.03), CookEvent(:wrap, :b, 4200.0)]),
]
u_heldout = 0.22 .+ 0.09 .* sin.(range(0, 1.3π; length=length(t)))
heldout_scenarios = [
    (u_knots=u_heldout, u_tknots=t, tsave=t,
     events=[CookEvent(:spritz, :b, 1500.0; dt=2700.0, m=0.025), CookEvent(:wrap, :b, 3900.0)]),
]

@testset "the reduced-order artifact: trained, validated, versioned, read online" begin
    @test length(blk) == 15 && !any(occursin("A_meat", n) for n in blk.names)
    t0 = time()
    r = train_rom("half-brick top-and-sides", sys_fom, sys_rom, blk, design, u, t, t;
                  scenarios=train_scenarios, state_noise=StateNoise(T=0.123, T_pit=0.456))
    println("  trained on ", length(design), " draws in ", round(Int, time() - t0), " s; Tcore bias at the end ", round(r.errors[:Tcore].bias[end]; digits=3),
            " K ± ", round(r.errors[:Tcore].sd[end]; digits=3), "; mass bias ", round(r.errors[:mass].bias[end] * 1000; digits=1), " g")
    @test Set(rom_observables(r)) == Set(collect(ROM_OBSERVABLES)) && r.fingerprint == rom_fingerprint()
    @test r.fom["kind"] == "brick3d" && r.fom["N"] == 72 && r.rom["nx"] == 8 && length(r.line) == 6
    @test r.training["n_scenarios"] == 2 && haskey(r.training, "latent_schema") &&
          all(Set(keys(s)) == Set(("family", "coord", "p1", "p2", "p3", "p4"))
              for s in values(r.training["latent_schema"]))
    t_nonuniform = copy(t); t_nonuniform[3] += 1.0
    @test_throws ArgumentError train_rom("bad control grid", sys_fom, sys_rom, blk, design, u, t_nonuniform, t)
    @test all(all(r.errors[obs].sd .>= 0) for obs in ROM_OBSERVABLES) && all(e.t == t for e in values(r.errors))
    @test all(r.envelope[nm][1] <= r.envelope[nm][2] for nm in blk.names)
    @test r.envelope["L"][1] < log(L) < r.envelope["L"][2]
    # the reduction error is real: a brick with exposed sides heats faster than the slab reads, so the slab's core lags (a negative bias)
    @test r.errors[:Tcore].bias[end] < 0 && r.errors[:Tcore].sd[end] > 0
    @test rom_error(r, :Tcore, t[end]).bias == r.errors[:Tcore].bias[end] && rom_error(r, :ring, 0.0) == (bias=0.0, sd=0.0)
    e_mid = rom_error(r, :Tcore, t[3] + 300.0)
    @test abs(e_mid.bias - (r.errors[:Tcore].bias[3] + r.errors[:Tcore].bias[4]) / 2) < 1e-12
    # held out: inside the bands
    v = validate_rom(r, sys_fom, sys_rom, blk, heldout, u, t, t; width=4.0)
    println("  held-out coverage: ", join(["$obs $(round(v[obs].coverage; digits=2)) (corrected/raw rms $(round(v[obs].rms; sigdigits=2))/$(round(v[obs].raw_rms; sigdigits=2)))" for obs in ROM_OBSERVABLES], ", "))
    @test v[:Tcore].coverage >= 0.8 && v[:mass].coverage >= 0.8 && v[:Tsurf].coverage >= 0.7
    @test all(obs -> isfinite(v[obs].rms) && isfinite(v[obs].raw_rms) && v[obs].raw_rms >= 0, ROM_OBSERVABLES)
    @test v[:in_envelope].coverage == 1.0
    fom_bad_cfg = Brick3D(:b; Lx=L, Ly, Lz, nx=6, ny=3, nz=4,
                          exposed=(false, true, true, true, true, true), cfg=with_config(cfg; T0=cfg.T0 + 1))
    @test_throws ArgumentError validate_rom(r, CookSystem(fom_bad_cfg; pit), sys_rom, blk, heldout, u, t, t)
    # Controls and piece events not used by the fit are held out as a separate regime.
    vh = validate_rom(r, sys_fom, sys_rom, blk, heldout, u, t, t;
                      scenarios=heldout_scenarios, width=4.0)
    @test vh[:Tcore].coverage >= 0.7 && vh[:Tsurf].coverage >= 0.6 && vh[:mass].coverage >= 0.7
    # the artifact round-trips; a stale fingerprint is refused
    path = tempname() * ".json"
    save_rom(path, r)
    r2 = load_rom(path)
    @test r2.archetype == r.archetype && r2.line == r.line && r2.errors[:mass].sd == r.errors[:mass].sd && r2.envelope == r.envelope
    d = JSON.parsefile(path); d["fingerprint"] = "000000000000"
    open(path, "w") do io; JSON.print(io, d); end
    @test_throws ErrorException load_rom(path)
    @test load_rom(path; strict=false).fingerprint == "000000000000"
    d["version"] = 1
    open(path, "w") do io; JSON.print(io, d); end
    @test_throws ArgumentError load_rom(path; strict=false)
    d["version"] = 999
    open(path, "w") do io; JSON.print(io, d); end
    @test_throws ArgumentError load_rom(path; strict=false)
    d["version"] = r.training["artifact_version"]; d["unknown"] = true
    open(path, "w") do io; JSON.print(io, d); end
    @test_throws ArgumentError load_rom(path; strict=false)
    rm(path)
    # projection and reconstruction: the line of a reconstructed line is the line; a transverse-uniform field is exact
    o = system_solve(sys_fom, [from_z(z0[1:8], blk.sites[1:8])], [1.0], from_z(z0[9:15], blk.sites[9:15]), u, t, t)
    Tf = o.piece[:b].T_field[:, end]
    line = project_to_line(fom, Tf)
    rec = reconstruct_from_line(fom, line)
    @test project_to_line(fom, rec) ≈ line && length(line) == 6
    @test maximum(abs.(rec .- Tf)) > 0.1                                    # the brick's sides heat: not transverse-uniform
    axis_distance = vec(fom.mesh.coords[:, 1] .- fom.mesh.coords[fom.mesh.core, 1])
    rom_line = project_to_line(r, fom, axis_distance)
    @test length(rom_line) == cfg.nx && rom_line ≈ collect(range(0.0, L; length=cfg.nx))
    @test reconstruct_from_line(r, fom, rom_line) ≈ axis_distance             # explicit six-node FOM ↔ eight-node ROM map
    flat = Brick3D(:f; Lx=L, Ly=1.0, Lz=1.0, nx=6, ny=1, nz=1, exposed=(false, true, false, false, false, false), cfg)
    @test reconstruct_from_line(flat, collect(1.0:6.0)) ≈ collect(1.0:6.0)
    # online: a probe's observation factor under the artifact carries the bias and the widened floor
    sensors = [SensorSpec(:core, TemperatureAt(:b; s=0.0), NoiseModel(white_sd=0.1)), SensorSpec(:pit, PitTemperature(), NoiseModel(white_sd=0.1))]
    clog = CookLog(t, u; sensors)
    blk_p = system_latents(sys_rom, prior_rom, phys_names; area=false)
    @test_throws ArgumentError SystemModel(sys_rom, clog, system_latents(sys_rom, prior_phys, phys_names); rom=r)
    m_rom = SystemModel(sys_rom, clog, blk_p; rom=r)
    m_plain = SystemModel(sys_rom, clog, blk_p)
    for obs in (WetBulbTemperature(), BarkL(:b), TemperatureAt(:b; s=0.5))
        unsupported_log = CookLog(t, u; sensors=[SensorSpec(:unsupported, obs, NoiseModel(white_sd=0.1))])
        @test_throws ObservableUnavailable SystemModel(sys_rom, unsupported_log, blk_p; rom=r)
    end
    sm = online_init(m_rom; lag=2); sm0 = online_init(m_plain; lag=2)
    z_init = variable_value(sm.graph, key_params(sm))
    y_init = variable_value(sm.graph, sm.nodes[1][2])
    z_outside = copy(z_init)
    i_hc = latent_index(blk_p, "h_c")
    z_outside[i_hc] = r.envelope["h_c"][2] + 0.1
    @test_throws DomainError E.propagate(m_rom, y_init, z_outside, t[1], t[2]; first_slice=true)
    @test_throws DomainError E._validate_rom_use(r, blk_p, z_init, t[1], t[end] + 1)
    y_corr, o_corr = E.propagate(m_rom, y_init, z_init, t[1], t[2]; first_slice=true)
    y_plain, o_plain = E.propagate(m_plain, y_init, z_init, t[1], t[2]; first_slice=true)
    @test norm(y_corr - y_plain) > 1e-9
    @test o_corr.pv_pit[end] != o_plain.pv_pit[end] && o_corr.q_meat[end] != o_plain.q_meat[end] # exchange-port bias reaches coupled propagation
    @test o_corr.q_meat == o_corr.piece[:b].q_meat &&
          o_corr.vapor_load ≈ o_corr.piece[:b].vapor_load .+ o_corr.vapor_correction
    # The snapshot retains the artifact, so planning propagates with the same corrected dynamics as estimation.
    sr = SystemSchedule(t, [:b])
    fr_plan = system_free_set(sr)
    ζr = system_plan_prior(fr_plan, sr)[1]
    snap_rom = belief_snapshot(sm); snap_plain = belief_snapshot(sm0)
    op_rom, plan_rom, tg_rom = system_plan_solve(snap_rom, snap_rom.z, ζr, fr_plan, sr)
    op_plain = system_plan_solve(snap_plain, snap_plain.z, ζr, fr_plan, sr)[1]
    @test snap_rom.rom === r && norm(op_rom.X - op_plain.X) > 1e-9
    po_plan = E._planning_piece_output(snap_rom, rom_piece, op_rom.piece[:b], tg_rom)
    @test po_plan.Tcore[end] ≈ op_rom.piece[:b].Tcore[end] - rom_error(r, :Tcore, tg_rom[end], snap_rom.z, snap_rom.blk).bias
    Czero = zeros(size(snap_rom.zy_cov))
    snap_rom_zero = BeliefSnapshot(snap_rom.t, snap_rom.sys, snap_rom.bl, snap_rom.blk, snap_rom.z, snap_rom.y,
                                   Czero, snap_rom.x, zeros(size(snap_rom.x_cov)), snap_rom.handoff,
                                   snap_rom.past, snap_rom.provenance, snap_rom.rom)
    pred_rom = system_plan_predictive(snap_rom_zero, ζr, fr_plan, sr, Dict(:b => PlanGoals()))
    @test maximum(pred_rom[:b].sd) > 0                                  # artifact spread survives even with zero posterior covariance
    et = rom_error(r, :Tcore, t[2], z_init, blk_p)                      # the bias at the latents the record is made at
    x_corr = system_from_belief(belief_decode(y_corr, m_rom.bl), m_rom.bl)
    p_corr, _, tk_corr = E._point_params(m_rom, z_init, t[2])
    h_core = compile_observable(sensors[1].observable, sys_rom, m_rom.bl, blk_p, clog)
    rec_online = [
        MeasurementRecord(:core, t[2], h_core(x_corr, z_init, p_corr, tk_corr, t[2]) - et.bias),
        MeasurementRecord(:pit, t[2], x_corr[first(state_index(sys_rom.layout, :pit, :T_pit))]),
    ]
    fr = add_measurement!(sm, rec_online[1]); fp = add_measurement!(sm, rec_online[2])
    f0 = add_measurement!(sm0, rec_online[1])
    @test fr.rom_key === :Tcore && abs(fr.σ^2 - (0.1^2 + et.sd^2)) < 1e-12 && fp.rom_key === nothing && fp.σ == 0.1 &&
          f0.rom_key === nothing && f0.σ == 0.1
    vals = [E.variable_value(sm.graph, k) for k in vars(fr)]
    @test abs(E.residual(fr, vals)[1] * fr.σ - (rec_online[1].value - E._obs_pred(fr, reduce(vcat, vals)))) < 1e-9
    zv, yv = vals[1], vals[2]
    pp, _, ttk = E._point_params(m_rom, zv, fr.t)
    xx = system_from_belief(belief_decode(yv, m_rom.bl), m_rom.bl)
    raw_pred = fr.h(xx, zv, pp, ttk, fr.t)
    @test E._obs_pred(fr, reduce(vcat, vals)) ≈ raw_pred - rom_error(r, :Tcore, fr.t, zv, blk_p).bias   # the bias is ROM − FOM at the latents: the correction subtracts it
    # Streaming slices must carry the same ROM bias and its latent derivative as node observations.
    reading = E.SliceReading(m_rom, sensors[1], rec_online[1])
    sf = E.SliceObservationFactor(m_rom, [reading], fr.key_z, fr.key_y, fr.t, false;
        kd0=[nothing], kd1=[nothing], G=reshape(E.slice_sensitivity(m_rom, reading, yv, zv), 1, :))
    slice_vals = [copy(v) for v in vals]
    slice_vals[1][i_hc] += 0.02
    @test residual(sf, slice_vals) ≈ residual(fr, slice_vals) rtol=1e-9 atol=1e-9
    @test reduce(hcat, jacobians(sf, slice_vals)) ≈ reduce(hcat, jacobians(fr, slice_vals)) rtol=1e-8 atol=1e-8
    @test E.propagate_states(m_rom, y_init, z_init, t[1], [t[2]]; first_slice=true)[:, 1] ≈ y_corr rtol=1e-8 atol=1e-8
    @test r.state_noise["rates"]["T"] == 0.123 && m_rom.sn.T == 0.123         # the artifact's process noise is actually loaded
    sys_reg = CookSystem([Slab1D(:a; cfg), Slab1D(:b; cfg)]; pit)
    blk_reg = system_latents(sys_reg, prior_rom, phys_names; area=false)
    reg = SystemModel(sys_reg, CookLog(t, u), blk_reg; rom=Dict(:a => r, :b => r))
    @test reg.sn[:a].T == 0.123 && reg.sn[:b].T == 0.123 && reg.sn[:pit].T_pit == 0.456
    @test_throws ArgumentError SystemModel(sys_reg, CookLog(t, u), blk_reg; rom=Dict(:a => r, :b => r, :ghost => r))
    sm_assim = online_init(m_rom; lag=2, iters=20)
    aout = assimilate!(sm_assim, rec_online)
    @test aout.committed && aout.converged && belief_at(sm_assim).t == t[2]   # artifact-backed online assimilation, not only factor construction
    @test_throws ArgumentError SystemModel(CookSystem(Slab1D(:b; cfg=MeatConfig(; nx=7)); pit), clog,
                                           system_latents(CookSystem(Slab1D(:b; cfg=MeatConfig(; nx=7)); pit), prior_phys, phys_names); rom=r)
    cfg_mismatch = with_config(cfg; T0=cfg.T0 + 1)
    sys_mismatch = CookSystem(Slab1D(:b; cfg=cfg_mismatch); pit)
    @test_throws ArgumentError SystemModel(sys_mismatch, clog, system_latents(sys_mismatch, prior_phys, phys_names; area=false); rom=r)
    bad_sites = copy(blk_p.sites)
    s1 = bad_sites[1]
    bad_sites[1] = PriorSite(s1.family, s1.p1, s1.p2, s1.p3 + 1.0, s1.p4; coord=s1.coord)
    bad_blk = LatentBlock(bad_sites, copy(blk_p.names), copy(blk_p.keys), copy(blk_p.bindings), copy(blk_p.index))
    @test_throws ArgumentError SystemModel(sys_rom, clog, bad_blk; rom=r)      # same display names, incompatible transform/units
    bad_sites[1] = PriorSite(s1.family, s1.p1 + 0.1, s1.p2, s1.p3, s1.p4; coord=s1.coord)
    bad_shape_blk = LatentBlock(bad_sites, copy(blk_p.names), copy(blk_p.keys), copy(blk_p.bindings), copy(blk_p.index))
    @test_throws ArgumentError SystemModel(sys_rom, clog, bad_shape_blk; rom=r) # same transform and units, incompatible prior shape
    bad_env = copy(r.envelope); bad_env[first(keys(bad_env))] = (NaN, 1.0)
    malformed = ReducedOrderModel(r.archetype, r.fom, r.rom, r.line, bad_env, r.errors,
                                  r.state_noise, r.fingerprint, r.training)
    @test_throws ArgumentError SystemModel(sys_rom, clog, blk_p; rom=malformed)

    # The same semantic requests read the full-order grid at its actual core and area-weighted surface;
    # inactive fixed-geometry thickness is not exposed as an estimable grid latent.
    blk_f = system_latents(sys_fom, prior_phys, phys_names; area=false)
    @test !("L" in blk_f.names) && length(blk_f) == 14
    gsensors = [SensorSpec(:gc, TemperatureAt(:b; s=0.0), NoiseModel(white_sd=0.1)),
                SensorSpec(:gs, SurfaceTemperature(:b), NoiseModel(white_sd=0.1)),
                SensorSpec(:gm, PieceMass(:b), NoiseModel(white_sd=0.01)),
                SensorSpec(:gb, BarkL(:b), NoiseModel(white_sd=1.0)),
                SensorSpec(:gloc, TemperatureAt(:b; location=Tuple(fom.mesh.coords[fom.mesh.core, :])), NoiseModel(white_sd=0.1))]
    glog = CookLog(t, u; sensors=gsensors)
    gm = SystemModel(sys_fom, glog, blk_f)
    θ = from_z(z0[1:8], blk.sites[1:8]); φ = from_z(z0[9:15], blk.sites[9:15])
    pg, plg = pack_params(sys_fom, [θ], [1.0], φ, u)
    xg = o.X[:, end]; zg = [z_moments(s)[1] for s in blk_f.sites]
    hs = [compile_observable(sp.observable, sys_fom, gm.bl, blk_f, glog) for sp in gsensors]
    got = [h(xg, zg, pg, t, t[end]) for h in hs]
    @test got[1] ≈ o.piece[:b].Tcore[end] && got[2] ≈ o.piece[:b].Tsurf[end]
    @test got[3] ≈ o.piece[:b].mass[end] && got[4] ≈ o.piece[:b].l_star[end]
    @test got[5] ≈ xg[first(state_index(sys_fom.layout, :b, :T)) + fom.mesh.core - 1]
    @test_throws ObservableUnavailable compile_observable(TemperatureAt(:b; location=(2L, 0.0, 0.0)),
                                                          sys_fom, gm.bl, blk_f, glog)
    @test_throws ArgumentError TemperatureAt(:b; s=0.0, location=(0.0, 0.0, 0.0))
end
println("test_rom.jl: done")
