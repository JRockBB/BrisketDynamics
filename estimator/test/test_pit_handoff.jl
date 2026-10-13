# Learn the fire on the pit alone, hand off to the whole cook (pit_handoff.jl): the pit-only model is the
# full model's cooker block and pit sensors; a handed-off smoother carries the pit phase's posterior
# exactly; the phased estimator runs through the hand-off with every component and its weight. Coarse
# slab, model-generated readings: the mechanism, not statistical consistency.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Test
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "..", "scripts", "stream_fixture.jl"))
fx = stream_fixture(REPO; start=:lighting)
cfg = MeatConfig(; nx=4, T0=fx.cfg.T0)
sys = CookSystem(Slab1D(:brisket; cfg); pit=fx.pit)
blk = system_latents(sys, fx.sites, fx.names; probes=[:brisket])
ev = fx.events
events = [CookEvent(:load, :brisket, ev.t_load), CookEvent(:wrap, :brisket, ev.t_wrap), CookEvent(:add, :pit, ev.t_add)]
nc = RealCookNoise()
sensors = [SensorSpec(:pit, PitTemperature(), NoiseModel(white_sd=0.1, ar_sd=nc.pit_sd, tau=nc.pit_tau)),
           SensorSpec(:core, TemperatureAt(:brisket; depth_latent=:core_depth_fraction), NoiseModel(white_sd=0.1, ar_sd=nc.core_sd, tau=nc.core_tau)),
           SensorSpec(:humidity, ChamberVapor(), NoiseModel(white_sd=10.0, ar_sd=nc.hum_sd, tau=nc.hum_tau, cap=nc.hum_cap)),
           SensorSpec(:mass, TotalMass(), NoiseModel(white_sd=0.005, ar_sd=nc.mass_sd, tau=nc.mass_tau))]
model = SystemModel(sys, CookLog(fx.t, fx.u; events, sensors), blk; sn=StateNoise())
iφ = E._phi_indices(blk)

# readings from the model itself at the prior centre: the pit's temperature along the propagated state
function pit_readings(times)
    sm0 = online_init(model)
    y0 = variable_value(sm0.graph, sm0.nodes[1][2]); z0 = variable_value(sm0.graph, sm0.key_z)
    S = E.propagate_states(model, y0, z0, sm0.nodes[1][1], times; first_slice=true)
    ip = first(E.belief_index(model.bl, :pit, :T_pit))
    return [MeasurementRecord(:pit, t, E.belief_decode(S[:, j], model.bl)[ip]) for (j, t) in enumerate(times)]
end

@testset "the pit-only model is the full model's cooker" begin
    pm = pit_only_model(model)
    @test isempty(pm.sys.pieces) && pm.bl.n == 6
    @test pm.blk.names == blk.names[iφ] && pm.blk.sites == blk.sites[iφ]
    @test length(iφ) == 7 && "a0" in pm.blk.names && "k_spread" in pm.blk.names
    @test Set(s.id for s in pm.log.sensors) == Set([:pit, :humidity])      # the probe and the scale read the meat
    @test all(e -> e.target === :pit, pm.log.events) && any(e -> e.kind === :add, pm.log.events)
end

@testset "a handed-off smoother carries the pit posterior exactly" begin
    pm = pit_only_model(model)
    ps = online_init(pm; attach=:slice, max_slice_s=600.0)
    for r in pit_readings(collect(300.0:300.0:1800.0)); @test assimilate!(ps, r).committed; end
    t0, kn = last(ps.nodes)
    C, _ = E.joint_cov(ps.graph; keys=[ps.key_z, kn])
    zφ = copy(variable_value(ps.graph, ps.key_z)); yp = copy(variable_value(ps.graph, kn))
    sm = handoff_smoother(model, zφ, yp, C, trues(7); t0, attach=:slice, max_slice_s=600.0)
    @test !sm.ignite_first && sm.nodes[1][1] == t0
    k1 = sm.nodes[1][2]; pr = model.bl.pit_range
    @test variable_value(sm.graph, sm.key_z)[iφ] == zφ && variable_value(sm.graph, k1)[pr] == yp
    Cf, off = E.joint_cov(sm.graph; keys=[sm.key_z, k1])
    sel = vcat(collect(off[sm.key_z])[iφ], collect(off[k1])[pr])
    @test maximum(abs.(Cf[sel, sel] .- C)) <= 1e-8 * maximum(abs.(C))     # the pit posterior, exactly
    # the meat's latents keep their exact site priors, untouched by the hand-off
    @test any(f -> f isa E.SitePriorFactor && f.key == sm.key_z && sort(f.indices) == setdiff(eachindex(blk.sites), iφ), sm.graph.factors)
    # a full slice after the hand-off keeps the learned ignited state (no reset from a0)
    r = assimilate!(sm, MeasurementRecord(:pit, t0 + 120.0, E.belief_decode(yp, pm.bl)[1]))
    @test r.committed
end

@testset "the hand-off carries every quadrature component and its weight" begin
    # The mechanism, from the pit phase at its prior: the data-driven pit phase does not yet commit (the
    # fire posterior's non-Gaussianity outruns feasible tensor rules; docs/estimator-preheat-consistency.md),
    # so this checks the hand-off itself, not a learned chart.
    ph = phased_init(model; axes=[:a0, :k_spread], orders=(3, 5), hold_lag=2, attach=:slice, node_ratio=Inf, max_slice_s=600.0)
    @test ph.t_handoff == ev.t_load - 600.0 && ph.full === nothing
    weights = [copy(b.logweights) for b in ph.pit.bank.banks]
    E._handoff!(ph)
    full = ph.full
    @test full !== nothing && full.frozen && ph.t_handed == last(ph.pit.bank.anchor.nodes)[1]
    @test [length(b.nodes) for b in full.bank.banks] == [length(b.nodes) for b in ph.pit.bank.banks]
    @test [b.logweights for b in full.bank.banks] == weights                     # the evidence carried, unchanged
    @test all(s.lag == 2 for b in full.bank.banks for s in b.smoothers)
    @test all(!s.ignite_first for b in full.bank.banks for s in b.smoothers) && !full.bank.anchor.ignite_first
    @test [i for (_, i) in full.bank.axes] == iφ[[findfirst(==(n), pit_only_model(model).blk.names) for n in ("a0", "k_spread")]]
    # the full anchor holds the pit anchor's Gaussian over the cooker's latents and the pit state
    pa = ph.pit.bank.anchor; kn = last(pa.nodes)[2]
    Cp, _ = E.joint_cov(pa.graph; keys=[pa.key_z, kn])
    fa = full.bank.anchor; k1 = fa.nodes[1][2]
    Cf, off = E.joint_cov(fa.graph; keys=[fa.key_z, k1])
    sel = vcat(collect(off[fa.key_z])[iφ], collect(off[k1])[model.bl.pit_range])
    @test maximum(abs.(Cf[sel, sel] .- Cp)) <= 1e-8 * maximum(abs.(Cp))
    # every component's graph has the anchor's structure
    @test all(first.(s.nodes) == first.(fa.nodes) for b in full.bank.banks for s in b.smoothers)
end
