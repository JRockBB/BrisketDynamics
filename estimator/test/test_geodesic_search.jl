using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, LinearAlgebra, Test
const E = BrisketEstimator

# The Rosenbrock valley has an independently known curved full step:
# (2,4) + (-1,-4) + (0,1) = (1,1), the exact zero-residual solution.
struct CurvatureProbe <: Factor
    key::VarKey
    mode::Symbol
end
E.vars(f::CurvatureProbe) = [f.key]
E.ndim(::CurvatureProbe) = 2
E._search_residual_curvature(::CurvatureProbe) = true
function E.residual(f::CurvatureProbe, vals)
    x,y = vals[1]
    if abs(x-1.9) < 1e-7 || abs(x-2.1) < 1e-7
        f.mode === :nonfinite && return [NaN, NaN]
        f.mode === :failure && error("invalid curvature probe")
        f.mode === :interrupt && throw(InterruptException())
    end
    return [10(y-x^2), 1-x]
end
E.jacobians(::CurvatureProbe, vals) = [[-20vals[1][1] 10.; -1. 0.]]

@testset "safeguarded curvature search preserves the original problem" begin
    cfg = MeatConfig(nx=3)
    sys = CookSystem(Slab1D(:brisket; cfg))
    prior = latent_block(joinpath(@__DIR__,"../../data/calibration/prior_next_cook.json"),cfg)
    blk = system_latents(sys,prior.prior_phys,prior.phys_names;probes=[:brisket])
    model = SystemModel(sys,CookLog([0.,120.],[0.5,0.5]),blk;
        abstol=1e-13,reltol=1e-11,floor_abstol=1e-10,floor_reltol=1e-8)
    sm = online_init(model;lag=10)
    E._node_at!(sm,90.)
    @test solve_window!(sm;newton_tol=0.).converged
    key = VarKey(:state,:curvature_probe,1)
    function graph(mode=:normal)
        g = deepcopy(sm.graph)
        add_variable!(g,key,[2.,4.])
        add_factor!(g,CurvatureProbe(key,mode))
        return g
    end

    straight, curved = graph(), graph()
    rs = gauss_newton!(straight;iters=1,soc=0,retract_dynamics=true)
    rc = gauss_newton!(curved;iters=1,soc=0,retract_dynamics=true,geodesic_search=true,joint_tol=1e-5)
    @test rs.hist[1].alpha < 1
    @test !rs.converged
    @test rc.converged
    @test rc.hist[1].alpha == 1
    @test variable_value(curved,key) ≈ [1.,1.] atol=1e-7
    @test rc.F ≈ objective(curved) atol=1e-8
    @test rc.newton_joint_sd < 1e-5
    @test rc.F < rs.F

    bounded = graph()
    rb = gauss_newton!(bounded;iters=1,soc=0,retract_dynamics=true,geodesic_search=true,
        step_bounds=Dict(key=>0.5),expand=4,joint_tol=1e-5)
    @test maximum(abs.(variable_value(bounded,key).- [2.,4.])) <= 0.5 + 1e-10
    @test !rb.converged
    @test rb.hist[1].bounded
    @test rb.F ≈ objective(bounded) atol=1e-8
    @test rb.F < objective(graph())

    for mode in (:normal,:nonfinite,:failure,:interrupt)
        g = graph(mode)
        offsets,n = E._offsets(g)
        x = E._stacked(g,g.keys,offsets,n)
        J,r,_,_ = linearize(g)
        d = E._ls_step(J,r,0.)
        transport = E._dynamics_retraction_data(g)
        if mode === :interrupt
            @test_throws InterruptException E._geodesic_search_data(g,x,d,J,r,offsets,transport)
        else
            a,ratio = E._geodesic_search_data(g,x,d,J,r,offsets,transport)
            if mode === :normal
                @test a[offsets[key]] ≈ [0.,1.] atol=1e-7
                @test 0 < ratio < 0.75
            else
                @test iszero(a)
                @test iszero(ratio)
            end
            @test E._stacked(g,g.keys,offsets,n) == x
        end
    end
    @test_throws ArgumentError gauss_newton!(graph();geodesic_search=true)
    @test_throws ArgumentError gauss_newton!(graph();geodesic_search=true,retract_dynamics=true,sparse=true)
end
