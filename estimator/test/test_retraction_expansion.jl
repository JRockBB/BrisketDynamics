using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, Test
const E = BrisketEstimator

# At x=0 the GN step is exactly 1. With target=2, x=2 is worse
# than x=1; with target=4, x=2 improves the objective but x=4 is worse.
# A live dynamics factor exercises the curved expansion path even though
# this independent coordinate does not perturb the physical trajectory.
struct ExpansionRejectionProbe <: Factor
    key::VarKey
    target::Float64
end
E.vars(f::ExpansionRejectionProbe) = [f.key]
E.ndim(::ExpansionRejectionProbe) = 2
E.residual(f::ExpansionRejectionProbe, vals) = [vals[1][1]-1, vals[1][1]^2-f.target]
E.jacobians(::ExpansionRejectionProbe, vals) = [reshape([1.,2vals[1][1]],2,1)]

@testset "rejected curved expansions preserve the accepted state and objective" begin
    cfg=MeatConfig(nx=3)
    sys=CookSystem(Slab1D(:brisket;cfg))
    prior=latent_block(joinpath(@__DIR__,"../../data/calibration/prior_next_cook.json"),cfg)
    blk=system_latents(sys,prior.prior_phys,prior.phys_names;probes=[:brisket])
    model=SystemModel(sys,CookLog([0.,120.],[0.5,0.5]),blk;
        abstol=1e-13,reltol=1e-11,floor_abstol=1e-10,floor_reltol=1e-8)
    sm=online_init(model;lag=10)
    E._node_at!(sm,90.)
    @test solve_window!(sm;newton_tol=0.).converged
    key=VarKey(:state,:expansion_probe,1)
    for (target,accepted) in ((2.,1.),(4.,2.))
        g=deepcopy(sm.graph)
        add_variable!(g,key,[0.])
        add_factor!(g,ExpansionRejectionProbe(key,target))
        before=objective(g)
        result=gauss_newton!(g;iters=1,tol=1e-6,retract_dynamics=true,expand=4,soc=0)
        @test variable_value(g,key)[1] ≈ accepted atol=1e-8
        @test result.hist[1].alpha ≈ accepted atol=1e-8
        @test objective(g) ≈ result.hist[1].F atol=1e-7
        @test result.F ≈ objective(g) atol=1e-7
        @test result.F < before
    end
end
