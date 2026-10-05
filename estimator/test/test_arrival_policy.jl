# The arrival contract of assimilate! (online.jl): the record's belief time (t_belief), the duplicate policy,
# the retry queue for uncommitted solves, chunk continuation, the complete per-reading innovation rows, and
# the solver's trial-failure classification (factors.jl _trial_failure). Algebra on the real-cook mock at a
# coarse slab; these check the arrival bookkeeping, not statistical consistency.
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
           SensorSpec(:core, TemperatureAt(:brisket; depth_latent=:core_depth_fraction), NoiseModel(white_sd=0.1, ar_sd=nc.core_sd, tau=nc.core_tau))]
rd = m["readings"]
interp(v, τ) = (k = clamp(searchsortedlast(t, τ), 1, length(t) - 1); w = (τ - t[k]) / (t[k + 1] - t[k]); (1 - w) * v[k] + w * v[k + 1])
pit_at(τ) = MeasurementRecord(:pit, τ, interp(Float64.(rd["pit"]), τ))
core_at(τ) = MeasurementRecord(:core, τ, interp(Float64.(rd["core"]), τ))
model = SystemModel(sys, CookLog(t, u; sensors), blk; sn=StateNoise())

@testset "the record carries the belief's own time beside the arrival's" begin
    sm = online_init(model; attach=:slice, node_ratio=Inf, max_slice_s=100.0)
    r1 = assimilate!(sm, pit_at(300.0))                  # age >= max_slice_s: its own node at 300
    @test r1.committed
    @test r1.t == 300.0 && r1.t_belief == 300.0
    r2 = assimilate!(sm, pit_at(150.0))                  # delayed: joins the closed slice; the belief stays at the newest node
    @test r2.committed
    @test r2.t == 150.0 && r2.t_belief == 300.0
    r3 = assimilate!(sm, pit_at(340.0))                  # an open slice: the belief is carried to the arrival
    @test r3.committed
    @test r3.t == 340.0 && r3.t_belief == 340.0
end

@testset "an uncommitted arrival reports the pre-arrival belief at its own time and queues its readings" begin
    sm = online_init(model; attach=:slice, node_ratio=Inf, max_slice_s=Inf, iters=0, restarts=0)
    r = assimilate!(sm, pit_at(37.0))
    @test !r.committed && !r.converged
    @test r.t == 37.0 && r.t_belief == sm.nodes[1][1]    # the first node's time: nothing committed
    @test length(sm.retry) == 1 && sm.retry[1][2] == 1
    @test sm.n_dropped == 0
end

@testset "queued readings are re-attempted and committed on a later arrival" begin
    sm = online_init(model; attach=:slice, node_ratio=Inf, max_slice_s=Inf)
    sm.iters = 0                                          # force the first solve to fail
    r1 = assimilate!(sm, pit_at(37.0))
    @test !r1.committed && length(sm.retry) == 1
    sm.iters = 40                                         # the queued reading joins the next batch
    r2 = assimilate!(sm, pit_at(74.0))
    @test r2.committed
    @test r2.sensors == [:pit, :pit] && isempty(sm.retry)
    @test haskey(sm.seen, (:pit, 37.0)) && haskey(sm.seen, (:pit, 74.0))
    @test r2.dof == 2 && length(r2.innov_rows) == 2
end

@testset "retries are bounded and the drops are counted" begin
    sm = online_init(model; attach=:slice, node_ratio=Inf, max_slice_s=Inf, iters=0, restarts=0)
    assimilate!(sm, pit_at(37.0))                         # attempt 1, queued
    assimilate!(sm, pit_at(74.0))                         # 37 s: attempt 2, queued; 74 s: attempt 1
    @test sort([n for (_, n) in sm.retry]) == [1, 2]
    assimilate!(sm, pit_at(111.0))                        # 37 s exceeds RETRY_MAX and is given up on
    @test E.RETRY_MAX == 2
    @test sm.n_dropped == 1
    @test sort([rec.t for (rec, _) in sm.retry]) == [74.0, 111.0]
end

@testset "every chunk of a batch is attempted even when an earlier chunk fails" begin
    sm = online_init(model; attach=:slice, node_ratio=Inf, max_slice_s=100.0, iters=0, restarts=0)
    before = sm.record_count
    r = assimilate!(sm, [pit_at(90.0), pit_at(210.0)])    # two chunks: 210 s forces its own node
    @test r.chunks == 2
    @test sm.record_count - before == 2                   # both chunks produced an (uncommitted) record
    @test !r.committed
    @test sort([rec.t for (rec, _) in sm.retry]) == [90.0, 210.0]
end

@testset "duplicate readings: exact repeats are ignored, conflicts are dropped with a warning" begin
    sm = online_init(model; attach=:slice, node_ratio=Inf, max_slice_s=Inf)
    r0 = pit_at(37.0)
    r = assimilate!(sm, [r0, MeasurementRecord(:pit, 37.0, r0.value)])
    @test r.committed && r.dof == 1 && length(r.innov_rows) == 1
    # an exact repeat of an assimilated reading leaves nothing to do
    @test_throws ArgumentError assimilate!(sm, [MeasurementRecord(:pit, 37.0, r0.value)])
    # a conflicting value at the same (sensor, time) is dropped; the batch's good reading proceeds
    n0 = sm.n_dropped
    r2 = @test_logs (:warn, r"conflicting duplicate") match_mode=:any assimilate!(
        sm, [MeasurementRecord(:pit, 37.0, r0.value + 1.0), pit_at(74.0)])
    @test r2.committed && r2.dof == 1 && sm.n_dropped == n0 + 1
    @test [rr.t for rr in r2.innov_rows] == [74.0]
end

@testset "innovation rows carry every reading; the per-sensor dicts keep the last" begin
    sm = online_init(model; attach=:slice, node_ratio=Inf, max_slice_s=Inf)
    r = assimilate!(sm, [pit_at(37.0), pit_at(74.0), core_at(53.0)])
    @test r.committed
    @test length(r.innov_rows) == r.dof == 3
    @test sort([rr.t for rr in r.innov_rows]) == [37.0, 53.0, 74.0]
    @test count(rr -> rr.sensor === :pit, r.innov_rows) == 2
    @test all(isfinite(rr.std) for rr in r.innov_rows)
    @test Set(keys(r.innov_std)) == Set([:pit, :core])
    # `innovation` is raw in slice mode: measurement units, not the standardized value
    @test Set(keys(r.innovation)) == Set([:pit, :core])
    @test all(isfinite, values(r.innovation))
end

@testset "the trial-failure classification separates domain failures from defects" begin
    @test E._trial_failure(ErrorException("system solve failed: retcode=Unstable, saved 1/3 points"))
    @test !E._trial_failure(ErrorException("an unrelated error"))
    @test E._trial_failure(DomainError(-1.0))
    @test E._trial_failure(LinearAlgebra.SingularException(2))
    @test E._trial_failure(LinearAlgebra.PosDefException(1))
    @test !E._trial_failure(BoundsError([1.0], 2))
    @test !E._trial_failure(InterruptException())
    @test !E._trial_failure(ArgumentError("a coding error"))
    tsk = Task(() -> throw(DomainError(0.0))); schedule(tsk)
    wrapped = try; wait(tsk); nothing; catch err; err; end
    @test wrapped isa TaskFailedException && E._trial_failure(wrapped)
    @test !E._trial_failure(CompositeException())
    ce = CompositeException(); push!(ce, wrapped)
    @test E._trial_failure(ce)
    push!(ce, ArgumentError("a defect among failures"))
    @test !E._trial_failure(ce)
end

@testset "the reported process noise keeps the filter's coordinate cap" begin
    bl = model.bl
    sm0 = online_init(model)
    y0 = variable_value(sm0.graph, sm0.nodes[1][2])
    Qbig = Matrix(1.0e12I, bl.n, bl.n)
    Qc = E.belief_cov_to_coord(Qbig, y0, bl)
    @test all(diag(Qc) .<= 9.0 * (1 + 1e-6))              # max_sd = 3: no entry's coordinate sd passes the cap
    # far under the cap in EVERY coordinate (even a boundary entry's d of ~1/eps maps 1e-36 under
    # an eighth of the cap): the factors are the raw decode Jacobian's inverse to rounding
    Qsmall = Matrix(1.0e-36I, bl.n, bl.n)
    d = inv.(max.(E.belief_decode_jacobian(y0, bl), eps(Float64)))
    e = E.belief_coord_factors(Qsmall, y0, bl)
    @test maximum(abs.(e .- d) ./ d) <= 1e-12
end
