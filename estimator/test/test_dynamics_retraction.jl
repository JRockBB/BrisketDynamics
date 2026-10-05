using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, LinearAlgebra, Random, Test
using ForwardDiff
const E = BrisketEstimator

@testset "floor guard joins continuously with a continuous derivative" begin
    δ = E.BELIEF_ENC_MIN
    width = δ * E.BELIEF_ENC_GUARD
    floor_at(x) = E._belief_floor(x, oftype(x, δ))
    lo, hi = δ + width, δ + 2width
    for (edge, slope) in ((lo, 0.), (hi, 1.))
        @test abs(floor_at(nextfloat(edge))-floor_at(prevfloat(edge))) <= 5eps(edge)
        @test ForwardDiff.derivative(floor_at, prevfloat(edge)) ≈ slope atol=2e-8
        @test ForwardDiff.derivative(floor_at, nextfloat(edge)) ≈ slope atol=2e-8
    end
    mid = (lo+hi)/2
    h = width*1e-3
    @test ForwardDiff.derivative(floor_at, mid) ≈ (floor_at(mid+h)-floor_at(mid-h))/(2h) rtol=5e-6
    @test floor_at(δ) == δ
    @test floor_at(2δ) == 2δ
end

@testset "floor derivatives are stable under restart roundoff" begin
    for kind in (:positive,:unit,:ignited)
        y=E._belief_enc(0.,kind)
        x=E._belief_dec(y,kind)
        for fraction in (-1e-9,0.,1e-9)
            v=x+fraction*E.BELIEF_ENC_MIN
            @test E._belief_enc(v,kind)==y
            @test ForwardDiff.derivative(v->E._belief_enc(v,kind),v)==0.
        end
        # Both ends of bounded coordinates need the same protection.
        if kind!==:positive
            y=E._belief_enc(1.,kind);x=E._belief_dec(y,kind)
            for v in (prevfloat(x),x,nextfloat(x))
                @test E._belief_enc(v,kind)==y
                @test ForwardDiff.derivative(v->E._belief_enc(v,kind),v)==0.
            end
        end
    end
end

@testset "nonlinear dynamics search preserves the graph and its tangent" begin
    cfg = MeatConfig(nx=3)
    sys = CookSystem(Slab1D(:brisket; cfg))
    prior = latent_block(joinpath(@__DIR__, "../../data/calibration/prior_next_cook.json"), cfg)
    blk = system_latents(sys, prior.prior_phys, prior.phys_names; probes=[:brisket])
    sensors=[SensorSpec(:pit,PitTemperature(),NoiseModel(white_sd=1.0))]
    model = SystemModel(sys, CookLog([0., 120., 240.], [0.5,0.5,0.5]; sensors), blk;
                        abstol=1e-13, reltol=1e-11, floor_abstol=1e-10, floor_reltol=1e-8)
    sm = online_init(model; lag=10)
    @test_throws ArgumentError solve_window!(sm;newton_tol=-1.)
    @test_throws ArgumentError assimilate!(sm,MeasurementRecord(:pit,1.,300.);newton_tol=NaN)
    @test sm.model === model
    strict = online_init(model; tol=1e-6)
    @test strict.model === model
    @test strict.model.floor_abstol == model.floor_abstol && strict.model.floor_reltol == model.floor_reltol
    @test E._online_resolution_gtol(strict) ≈ 0.01 * E._online_resolution_gtol(sm)
    # The online API honors the caller's accuracy budget, including disabling
    # it and requesting a tighter bound. No hidden stricter cap adds work.
    @test solve_window!(sm).joint_tol == E.ONLINE_NEWTON_TOL
    for budget in (0., 1e-4, 1e-2, 2e-2)
        @test solve_window!(sm;newton_tol=budget).joint_tol == budget
    end
    # A linear pit reading requests exactly 0.005 joint posterior SD of motion.
    # The default budget may retain the mean; a strict caller must update it.
    probe=deepcopy(sm)
    pit_mean,pit_sd=belief_at(probe).T_pit
    target_shift=0.005sqrt((pit_sd^2+1)/pit_sd^2)
    add_measurement!(probe,MeasurementRecord(:pit,0.,pit_mean+target_shift))
    loose,tight=deepcopy(probe),deepcopy(probe)
    loose_result=solve_window!(loose)
    tight_result=solve_window!(tight;newton_tol=1e-4)
    @test loose_result.converged && loose_result.gradient > loose.tol
    @test loose_result.newton_joint_sd ≈ 0.005 atol=1e-7
    @test loose.graph.values == probe.graph.values
    @test tight_result.converged && abs(belief_at(tight).T_pit[1]-pit_mean)>1e-6
    E._node_at!(sm, 90.); E._node_at!(sm, 180.)
    g=sm.graph
    E._ensure_current_dynamics!(g)
    J,r,off,n=linearize(g)
    x=E._stacked(g,g.keys,off,n)
    data=E._dynamics_retraction_data(g)
    @test length(data)==2
    @test E._retract_dynamics(x,off,data) ≈ x atol=1e-12
    rng=MersenneTwister(937)
    d=randn(rng,n)
    # Away from state boundaries, a perturbation checks the identity tangent.
    for k in g.keys
        k==sm.key_z && continue
        d[off[k]] .*= 0.01
    end
    h=1e-3
    xn=E._retract_dynamics(x+h*d,off,data)
    @test norm(xn-x-h*d,Inf) < 0.02h*norm(d,Inf)
    # Each retracted process defect is exactly the tangent model's proposed
    # defect, including the propagated correction from the preceding node.
    for q in data
        f=q.f
        z,y0,y1=E._dyn_unpack(f,[xn[off[k]] for k in vars(f)])
        zt,yt,y1t=E._dyn_unpack(f,[(x+h*d)[off[k]] for k in vars(f)])
        expected=q.defect+(y1t-q.y1)-q.Jy*(yt-q.y0)-q.Jz*(zt-q.z)
        actual=y1-E._dyn_eval!(f,z,y0;jac=false).f
        @test actual ≈ expected atol=2e-12
    end
    @test E._stacked(g,g.keys,off,n)==x
    @test E._within_step_bounds(d,off,2norm(d,Inf),Dict(sm.key_z=>2norm(d[off[sm.key_z]],Inf)))
    @test !E._within_step_bounds(d,off,0.5norm(d,Inf),Dict())
    @test !E._within_step_bounds(d,off,Inf,Dict(sm.key_z=>0.5norm(d[off[sm.key_z]],Inf)))

    # A sensor-driven nonlinear solve must converge to the same posterior mode,
    # including when curved expansion is enabled. No change of noise or prior.
    ylast=g.values[last(sm.nodes)[2]]
    pitindex=first(E.belief_index(model.bl,:pit,:T_pit))
    target=belief_decode(ylast,model.bl)[pitindex]+0.4
    add_measurement!(sm,MeasurementRecord(:pit,180.,target))
    ga,gb=deepcopy(g),deepcopy(g)
    ra=gauss_newton!(ga;iters=25,tol=1e-5,step_bounds=Dict(sm.key_z=>0.25))
    rb=gauss_newton!(gb;iters=25,tol=1e-5,step_bounds=Dict(sm.key_z=>0.25),retract_dynamics=true,expand=4)
    @test ra.converged && rb.converged
    @test isapprox(ra.F,rb.F;atol=1e-7,rtol=1e-6)
    ca=joint_cov(ga)[1];cb=joint_cov(gb)[1]
    xa=E._stacked(ga,ga.keys,off,n);xb=E._stacked(gb,gb.keys,off,n)
    @test maximum(abs.(xa-xb)./sqrt.(diag(ca)))<1e-3
    if maximum(abs.(sqrt.(diag(cb)./diag(ca)).-1))>=1e-3
        for i in sortperm(abs.(sqrt.(diag(cb)./diag(ca)).-1);rev=true)[1:8]
            k=only(k for k in g.keys if i in off[k]); j=i-first(off[k])+1
            kind=k==sm.key_z ? blk.names[j] : string(model.bl.kinds[j])
            println("covariance mismatch ",k," coordinate ",j," ",kind," means ",(xa[i],xb[i])," SD ",(sqrt(ca[i,i]),sqrt(cb[i,i])))
        end
    end
    @test maximum(abs.(sqrt.(diag(cb)./diag(ca)).-1))<1e-3

    # The transport assembles the latent blocks belonging to each interval.
    sd=online_init(model;drift=(moving=[1],sd_per_hour=[0.01],jumps=Tuple{Int,Float64,Float64}[]))
    E._node_at!(sd,90.)
    E._ensure_current_dynamics!(sd.graph)
    linearize(sd.graph)
    od,nd=E._offsets(sd.graph);xd=E._stacked(sd.graph,sd.graph.keys,od,nd)
    dd=E._dynamics_retraction_data(sd.graph)
    @test E._retract_dynamics(xd,od,dd) ≈ xd atol=1e-12
    add_factor!(sd.graph,deepcopy(only(values(sd.dyn))))
    @test_throws ArgumentError E._dynamics_retraction_data(sd.graph)

    # With no nonlinear dynamics, the option retains the exact linear oracle.
    k=VarKey(:state,:linear,1); g0=FactorGraph(); add_variable!(g0,k,[0.,0.])
    add_factor!(g0,LinearFactor([k],[[1e6 -1e6; 1. 1.]],[0.,2.]))
    rc=gauss_newton!(g0;retract_dynamics=true,soc=0)
    @test rc.converged
    @test variable_value(g0,k) ≈ [1.,1.]
end
