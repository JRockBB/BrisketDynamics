# The process noise of a state the physics does not move, and of a positive state at its floor (system_state.jl,
# factors.jl). An inert group — the firebox of a cooker without a duct, the pan water and its temperature when there
# is no pan, the salt of an unsalted piece — carries no process noise: a model cannot err in a state it does not
# evolve, and a random walk there only widened the belief without bound (the snapshot replay of 2026-10-14: the
# firebox 4.5 → 9.7 K and the salt 8.5 → 26 nats over fifteen hours, nothing to pull them back). A positive state
# next to its floor is boundary-damped in the estimator's own whitener, as it already was for drawing the process:
# a floored salt or pan no longer takes the coordinate cap of noise every slice. Algebra, then a short slice stream
# on the real-cook mock at a coarse slab (no duct, no pan, unsalted: the shipped bench's cooker).
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; nx=6, T0=Float64(meta["tcore0_K"]))
blk_old = latent_block(joinpath(REPO, "data", "calibration", "prior_next_cook.json"), cfg)
sys = CookSystem(Slab1D(:brisket; cfg); pit)
blk = system_latents(sys, blk_old.prior_phys, blk_old.phys_names; probes=[:brisket])
nc = RealCookNoise()
sensors = [SensorSpec(:pit, PitTemperature(), NoiseModel(white_sd=0.1, ar_sd=nc.pit_sd, tau=nc.pit_tau)),
           SensorSpec(:core, TemperatureAt(:brisket; depth_latent=:core_depth_fraction), NoiseModel(white_sd=0.1, ar_sd=nc.core_sd, tau=nc.core_tau)),
           SensorSpec(:mass, TotalMass(), NoiseModel(white_sd=0.005, ar_sd=nc.mass_sd, tau=nc.mass_tau))]   # the scale reads the salt's water coupling: the salt sits off its floor's guard, as on the shipped bench
rd = m["readings"]
interp(v, τ) = (k = clamp(searchsortedlast(t, τ), 1, length(t) - 1); w = (τ - t[k]) / (t[k + 1] - t[k]); (1 - w) * v[k] + w * v[k + 1])
function stream(tmax; pit_s=37.0, core_s=53.0, mass_s=120.0)
    recs = MeasurementRecord[]
    for τ in pit_s:pit_s:tmax; push!(recs, MeasurementRecord(:pit, τ, interp(Float64.(rd["pit"]), τ))); end
    for τ in core_s:core_s:tmax; push!(recs, MeasurementRecord(:core, τ, interp(Float64.(rd["core"]), τ))); end
    for τ in mass_s:mass_s:tmax; push!(recs, MeasurementRecord(:mass, τ, interp(Float64.(rd["mass"]), τ))); end
    sort!(recs; by=r -> r.t)
    return recs
end
sn = StateNoise()
model = SystemModel(sys, CookLog(t, u; sensors), blk; sn)
bl = model.bl
grp(bl_, g) = bl_.groups[findfirst(p -> first(p) == g, bl_.groups)][2]
d1(Q, bl_, g) = (i = first(grp(bl_, g)); Q[i, i])
const INERT = ("T_fb", "m_pan", "T_pan", "brisket.S")

@testset "process noise" begin

@testset "a state the physics does not move carries no process noise" begin
    @test Set(E.inert_groups(bl)) == Set(INERT)
    Q = belief_process_cov(bl, sn, 1.0)
    for g in INERT; @test all(iszero, Q[grp(bl, g), :]); end
    for g in ("T_pit", "m_fuel", "ignited", "brisket.T", "brisket.W", "brisket.C", "brisket.bark"); @test d1(Q, bl, g) > 0; end
    # the initial widths stay: an inert state's belief is its prior
    Q0 = belief_init_cov(bl, sn)
    for g in INERT; @test d1(Q0, bl, g) > 0; end
    # with a duct, a pan and a brine the same groups move, and carry their rates
    sys2 = CookSystem(Slab1D(:brisket; cfg=MeatConfig(; nx=6, T0=cfg.T0, t_salt=3600.0));
                      pit=PitConfig(; T_amb=pit.T_amb, Tpit0=pit.Tpit0, m0=pit.m0, m_pan0=1.0, c_duct=60.0))
    bl2 = belief_layout(sys2)
    @test isempty(E.inert_groups(bl2))
    Q2 = belief_process_cov(bl2, sn, 1.0)
    for (g, r) in (("T_fb", sn.T_fb), ("m_pan", sn.m_pan), ("T_pan", sn.T_pan)); @test d1(Q2, bl2, g) ≈ r^2; end
    @test d1(Q2, bl2, "brisket.S") > 0
    # a lid-open event disturbs the chamber, not a firebox that is a constant (the 2026-10-16 replay: 3 K of width per
    # event on the inert firebox); with a duct the firebox takes the event's disturbance as before
    Qe = belief_process_cov(bl, sn, 1.0; events=[:add, :spritz])
    @test d1(Qe, bl, "T_pit") ≈ d1(Q, bl, "T_pit") + sn.T_event^2 + sn.T_spritz^2
    @test d1(Qe, bl, "T_fb") == 0.0 && d1(Qe, bl, "m_fuel") ≈ d1(Q, bl, "m_fuel") + sn.m_add^2
    Qe2 = belief_process_cov(bl2, sn, 1.0; events=[:add])
    @test d1(Qe2, bl2, "T_fb") ≈ sn.T_fb^2 + sn.T_event^2
    # the rates' sensitivity (the bench accumulator's dQ) is zero for an inert group
    @test E._group_rate(bl, sn, "T_fb", false) == 0.0
    @test E._group_rate(bl, sn, "T_fb", true) == sn.T_fb0
end

sm0 = online_init(model; attach=:slice, lag=3, node_ratio=Inf, max_slice_s=120.0)
k1 = sm0.nodes[1][2]
y0 = copy(E.variable_value(sm0.graph, k1))
fdyn = E.DynamicsFactor(model, sm0.key_z, k1, E.VarKey(:state, sm0.cook, 2), 0.0, 600.0)
Δh = 600.0 / 3600

@testset "a positive state at its floor is boundary-damped in the whitener" begin
    Qc = E._process_cov_coord(fdyn, y0, belief_process_cov(bl, sn, Δh))
    # the floored salt and pan (inert here): only the integrator's floor remains, damped at the boundary (at most a
    # quarter of a nat of variance per slice, where the cap gave nine plus one)
    for g in ("brisket.S", "m_pan"); @test 0 < d1(Qc, bl, g) <= 0.3; end
    # a moving positive state next to its floor (the fuel nearly spent): noise of its own order, not the cap's
    ifuel = first(grp(bl, "m_fuel"))
    ylow = copy(y0); ylow[ifuel] = E._belief_enc(1e-3, :positive)
    Ql = E._process_cov_coord(fdyn, ylow, belief_process_cov(bl, sn, Δh))
    @test 1e-8 < Ql[ifuel, ifuel] < 1e-2
    # away from any boundary the whitener is the undamped one
    Qu = belief_cov_to_coord(belief_process_cov(bl, sn, Δh), y0, bl)
    for g in ("T_pit", "m_fuel", "brisket.T", "brisket.W")
        i = first(grp(bl, g))
        @test isapprox(Qc[i, i], Qu[i, i]; rtol=1e-4)      # the damping is (s/δ)², a few 1e-6 for a temperature, plus the integrator's floor
        @test abs(Qc[i, i] - Qu[i, i]) <= 1e-4 * Qu[i, i] + 1e-4
    end
    # the rates' sensitivity path (Dual-valued) takes the same damping and stays finite; next to the floor the damped
    # coordinate variance FALLS with the rate (its sd is δ/s there), so the sensitivity is negative and small
    S = E._process_cov_sens(fdyn, ylow, ["m_fuel", "T_pit"])
    @test all(M -> all(isfinite, M), S)
    @test S[1][ifuel, ifuel] != 0 && abs(S[1][ifuel, ifuel]) < 1.0
    Sfar = E._process_cov_sens(fdyn, y0, ["m_fuel", "T_pit"])
    @test Sfar[1][ifuel, ifuel] > 0
end

@testset "an unobserved inert group's belief does not random-walk over a slice stream" begin
    recs = stream(7200.0)
    sm = online_init(model; attach=:slice, lag=3, node_ratio=Inf, max_slice_s=120.0)
    sd = Dict{String,Vector{Float64}}(g => Float64[] for g in INERT)
    last_out = nothing
    for r in recs
        out = assimilate!(sm, r)
        out.committed || continue
        last_out = out
        for g in INERT; i = first(grp(bl, g)); push!(sd[g], sqrt(max(out.y_cov[i, i], 0.0))); end
    end
    @test sm.n_dropped == 0 && isempty(sm.retry)
    @test last_out !== nothing
    for g in ("T_fb", "T_pan")                      # a temperature nobody measures and nothing moves: its prior width, all stream
        @test last(sd[g]) <= first(sd[g]) + 0.05
        println("  ", g, ": coordinate sd ", round(first(sd[g]); digits=3), " → ", round(last(sd[g]); digits=3))
    end
    for (g, lim) in (("brisket.S", 1e-3), ("m_pan", 1e-3))   # the floored salt and pan: a two-sd draw decodes to nothing
        i = first(grp(bl, g)); y = last_out.y[i]; s = last(sd[g])
        hi = E._belief_dec(y + 2s, bl.kinds[i])
        println("  ", g, ": coordinate sd ", round(first(sd[g]); digits=3), " → ", round(s; digits=3), ", +2 sd decodes to ", hi)
        @test hi < lim
    end
end
end
