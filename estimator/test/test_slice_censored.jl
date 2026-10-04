using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, JSON, Test, LinearAlgebra
const E = BrisketEstimator
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
prior = JSON.parsefile(joinpath(REPO, "data", "calibration", "prior_next_cook.json"))
sys = CookSystem(Slab1D(:brisket; cfg=MeatConfig(nx=4, T0=299.8)); pit=PitConfig(Tpit0=362.4))
blk = system_latents(sys, read_sites(prior["prior"]), String.(prior["names"]))
sp = SensorSpec(:pit, PitTemperature(), NoiseModel(white_sd=0.5, ar_sd=2.0, tau=600.0, cap=500.0))
model = SystemModel(sys, CookLog(collect(0.0:30.0:180.0), fill(0.2, 7); sensors=[sp]), blk)
new_smoother() = online_init(model; attach=:slice, node_ratio=Inf, lag=10)

@testset "capped readings close and split existing observation slices" begin
    sm = new_smoother()
    E._add_slice_readings!(sm, [MeasurementRecord(:pit, 30.0, 360.0), MeasurementRecord(:pit, 60.0, 361.0)])
    ka = first(sm.nodes)[2]
    old = sm.slices[ka]
    @test old.key_y1 === nothing
    capped = add_measurement!(sm, MeasurementRecord(:pit, 90.0, 500.0))
    kb = capped.key_y
    @test capped.censored
    @test sm.slices[ka].key_y1 == kb && sm.slices[ka].t1 == 90.0
    @test sm.dyn[(ka, kb)].peer[] === sm.slices[ka]
    @test !(old in sm.graph.factors)
    delayed = add_measurement!(sm, MeasurementRecord(:pit, 45.0, 500.0))
    km = delayed.key_y
    @test getfield.(sm.slice_recs[ka], :t) == [30.0]
    @test getfield.(get(sm.slice_recs, km, MeasurementRecord[]), :t) == [60.0]
    @test sm.slices[ka].key_y1 == km
    @test haskey(sm.slices, km) && sm.slices[km].key_y1 == kb
    @test sm.dyn[(ka, km)].peer[] === sm.slices[ka]
    @test haskey(sm.slices, km) && sm.dyn[(km, kb)].peer[] === sm.slices[km]
    @test add_measurement!(sm, MeasurementRecord(:pit, 0.0, 500.0)).key_y == ka
end

@testset "a mixed batch's capped reading cannot inform its forecast covariance" begin
    sm = new_smoother()
    E._add_slice_readings!(sm, [MeasurementRecord(:pit, 30.0, 360.0)])
    capped = add_measurement!(sm, MeasurementRecord(:pit, 90.0, 500.0))
    reading = MeasurementRecord(:pit, 75.0, 361.0)
    E._add_slice_readings!(sm, [reading])
    ka = first(sm.nodes)[2]
    added = [(ka, [reading])]
    before = copy(sm.graph.factors)
    pc = E._predicted_consistency_slice(sm, added; exclude=[capped])
    @test all(a === b for (a, b) in zip(before, sm.graph.factors)) && length(before) == length(sm.graph.factors)
    pos = findfirst(f -> f === capped, sm.graph.factors)
    deleteat!(sm.graph.factors, pos)
    reference = try
        E._predicted_consistency_slice(sm, added)
    finally
        insert!(sm.graph.factors, pos, capped)
    end
    @test pc.sdev ≈ reference.sdev && pc.nu ≈ reference.nu && pc.nis ≈ reference.nis
    contaminated = E._predicted_consistency_slice(sm, added)
    @test maximum(abs.(contaminated.sdev .- pc.sdev)) > 1e-5
    @test_throws ArgumentError E._predicted_consistency_slice(sm, added; exclude=[capped, capped])
    @test all(a === b for (a, b) in zip(before, sm.graph.factors)) && length(before) == length(sm.graph.factors)
end
