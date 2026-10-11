# Learn-then-freeze in the deterministic (adaptive quadrature) backend: full history while the chart may
# still re-centre, then every smoother compacts to `hold_lag` once `freeze_stable` arrivals needed no
# adaptation; slice attachment mirrors the anchor's structure in every component; the record carries the
# mixture's belief and the phase. Mechanism checks on the consistency fixture, not statistical consistency.
if !isdefined(Main,:BrisketEstimator)
    include(joinpath(@__DIR__,"..","src","BrisketEstimator.jl"))
end
using .BrisketEstimator, BrisketDynamics, LinearAlgebra, Test
isdefined(Main,:E) || (const E=BrisketEstimator)
isdefined(Main,:consistency_model) || include(joinpath(@__DIR__,"..","scripts","consistency_control.jl"))

smoothers(sm)=vcat([sm.bank.anchor],[inner for b in sm.bank.banks for inner in b.smoothers])

@testset "the chart freezes after stable arrivals and compaction begins" begin
    model=consistency_model(nx=4,suite=:pit,stop=600.0)
    # pit data carry no information about insertion depth: the chart never needs to adapt
    sm=deterministic_init(model;axes=[:core_depth_fraction],orders=(3,5),freeze_stable=3,hold_lag=1)
    recs=[MeasurementRecord(:pit,t,294.0+0.02t) for t in 10.0:10.0:80.0]
    phases=Symbol[]
    for (i,rec) in enumerate(recs)
        r=assimilate!(sm,rec)
        @test r.committed
        push!(phases,r.phase)
        @test haskey(r,:z) && haskey(r,:z_cov) && haskey(r.pieces,:brisket) && r.t_belief==r.t
        if !sm.frozen
            # learning: full history, nothing marginalized anywhere
            @test all(s.lag==typemax(Int) for s in smoothers(sm))
            @test !any(f->f isa E.LinearFactor,sm.bank.anchor.graph.factors)
        end
    end
    @test phases[1:2]==[:learn,:learn]
    @test phases[3]==:hold                       # froze on the third stable arrival
    @test all(==(:hold),phases[3:end])
    @test sm.frozen && all(s.lag==1 for s in smoothers(sm))
    @test length(sm.bank.anchor.nodes)<=1+1      # compacted to the hold lag
    @test all(length(inner.nodes)==length(sm.bank.anchor.nodes) for b in sm.bank.banks for inner in b.smoothers)
    @test sm.n_recentres==0 && sm.n_refines==0
    # the default keeps full history for the whole cook
    never=deterministic_init(model;axes=[:core_depth_fraction],orders=(3,5))
    for rec in recs;@test assimilate!(never,rec).phase==:learn;end
    @test !never.frozen && length(never.bank.anchor.nodes)==length(recs)+1
    @test_throws ArgumentError deterministic_init(model;axes=[:core_depth_fraction],orders=(3,5),freeze_stable=0)
end

@testset "slice attachment: components mirror the anchor through freezing" begin
    model=consistency_model(nx=4,suite=:pit,stop=600.0)
    # The integrated axis (depth) is untouched by pit data while pit data inform the OTHER latents
    # nonlinearly: their shape defect passes the default guard (0.25) after ~3 minutes. This test checks
    # structure mirroring through the freeze, not the shape guard, so the guard is opened here.
    sm=deterministic_init(model;axes=[:core_depth_fraction],orders=(3,5),freeze_stable=4,hold_lag=2,
                          attach=:slice,node_ratio=Inf,max_slice_s=90.0,shape_tol=10.0)
    recs=[MeasurementRecord(:pit,t,294.0+0.02t) for t in 30.0:30.0:420.0]
    for rec in recs
        r=assimilate!(sm,rec)
        @test r.committed
        times=first.(sm.bank.anchor.nodes)
        @test all(first.(inner.nodes)==times for b in sm.bank.banks for inner in b.smoothers)
    end
    @test sm.frozen
    @test length(sm.bank.anchor.nodes)<=2+1
    # rejection after freezing reports the standing belief, not a crash
    before=deepcopy(sm.bank.posterior.mean)
    sm.bank.banks[end].smoothers[end].iters=0
    r=assimilate!(sm,MeasurementRecord(:pit,450.0,320.0))
    @test !r.committed && haskey(r,:z) && haskey(r,:reason)
    @test sm.bank.posterior.mean==before
end
