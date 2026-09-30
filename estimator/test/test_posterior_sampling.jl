using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
include(joinpath(@__DIR__,"..","src","BrisketEstimator.jl"))
using .BrisketEstimator,BrisketDynamics,LinearAlgebra,Random,Statistics,Test
const E=BrisketEstimator
include(joinpath(@__DIR__,"..","scripts","consistency_control.jl"))

@testset "nonlinear posterior sampling" begin
    model=consistency_model(nx=4,suite=:pit,stop=600.0)
    sm=online_init(model;lag=typemax(Int),record_limit=0)
    original=deepcopy(sm.graph.values)
    p=sample_posterior(sm;n=8192,rng=MersenneTwister(173))
    @test sm.graph.values==original
    @test isempty(p.diagnostics.numerical_failures)
    @test p.diagnostics.ess≈8192 rtol=1e-10
    @test maximum(p.weights)-minimum(p.weights)<1e-12
    mean0=[isscore(s) ? 0.0 : s.p1 for s in model.blk.sites]
    sd0=[isscore(s) ? 1.0 : s.p2 for s in model.blk.sites]
    nz=length(mean0)
    @test maximum(abs.((p.mean[1:nz]-mean0)./sd0))<6/sqrt(8192)
    @test maximum(abs.(diag(p.covariance)[1:nz]./sd0.^2 .- 1))<.07
    quantiles=posterior_quantile(p)
    @test maximum(abs.((quantiles[1:nz,1]-mean0)./sd0 .+ 1.959963984540054))<.12
    @test maximum(abs.((quantiles[1:nz,3]-mean0)./sd0 .- 1.959963984540054))<.12
    @test size(posterior_quantile(p;physical=true))==(model.bl.n,3)
    @test belief_at(p).x≈p.physical*p.weights
    @test all(isfinite,belief_at(p).x_cov)
    @test_throws ArgumentError posterior_quantile(p,[-.1,.5])
    @test E._weighted_quantiles([-10.0,1.0,2.0,10.0],[0.0,.5,.5,0.0],[0.0,1.0])==[1.0,2.0]
    # A tiny spatial contrast beside a large common mode must remain visible to precision checks.
    rng=MersenneTwister(271);base=randn(rng,2,4096)
    correlated=vcat(permutedims(base[1,:]),permutedims(base[1,:]+1e-8*base[2,:]))
    weights=fill(1/4096,4096);center=correlated*weights
    white=E._posterior_whiten(correlated,weights,center,correlated.-center)
    @test white*white'/4096≈Matrix{Float64}(I,2,2) atol=1e-6
    @test_throws ArgumentError sample_posterior(sm;n=1)
    insufficient=sample_posterior(sm;n=16,rng=MersenneTwister(174))
    @test !insufficient.diagnostics.mc_precision_ok
    @test_throws ArgumentError belief_at(insufficient)

    saved=Ref{Any}()
    control_cook(model,3004;times=[0.0,120.0,600.0],lag=typemax(Int),inspect=(state,r,truth)->(saved[]=deepcopy(state)))
    sm=saved[];ref=E._posterior_transport(sm)
    # Unobserved insertion depth is independent of pit-only data, exactly.
    depth=E.latent_index(model.blk,:core_depth_fraction)
    u0=copy(ref.center);u0[depth]=0.0;u1=copy(u0);u1[depth]=2.0
    @test E._transport_draw(ref,u1).potential-E._transport_draw(ref,u0).potential≈2.0 atol=1e-8
    @test norm(ref.V[depth,:].*(ref.a.-1))<1e-10

    # Independently eliminate the discrepancy variables from the original factor objective.
    # This checks the target density, rather than agreement between two samplers sharing a target.
    for seed in 1:3
        eta=randn(MersenneTwister(seed),ref.n)
        u=ref.center+eta+ref.V*((ref.a.-1).*(ref.V'*eta))
        g=deepcopy(ref.g);z=ref.zm+ref.zs.*u[ref.off[ref.kz]];set_value!(g,ref.kz,z)
        for k in ref.ks[2:end]
            f=ref.generators[k]
            y=f isa GaussianPrior ? f.m+ref.roots[k]*u[ref.off[k]] :
                E.propagate(f.model,g.values[f.key_y0],z,f.t0,f.t1;first_slice=f.first)[1]+ref.roots[k]*u[ref.off[k]]
            set_value!(g,k,y)
        end
        predictions=[E._obs_pred(f,vcat(z,g.values[f.key_y],f.key_d===nothing ? Float64[] : [0.0])) for f in ref.fs]
        residual=[f.value for f in ref.fs]-predictions-ref.obsbias
        d=ref.dm+ref.D*ref.Dobs'*ref.W'*ref.W*residual
        for (i,k) in enumerate(ref.dkeys);set_value!(g,k,[d[i]]);end
        @test objective(g)≈E._transport_draw(ref,u).potential rtol=1e-9 atol=1e-7
    end
    original=deepcopy(sm.graph.values)
    p=sample_posterior(sm;n=4096,rng=MersenneTwister(179))
    @test sm.graph.values==original
    @test isempty(p.diagnostics.numerical_failures)
    @test p.diagnostics.ess>100
    @test all(isfinite,p.covariance)
    @test isposdef(Symmetric(p.covariance))
    sm.lag=0;marginalize_old!(sm)
    @test_throws ArgumentError sample_posterior(sm;n=2)

    for suite in (:core,:two_core,:pit_core,:full)
        m=consistency_model(nx=4,suite=suite,stop=600.0)
        s=online_init(m;lag=typemax(Int))
        ref=E._posterior_transport(s)
        @test length(E._transport_draw(ref,zeros(ref.n)).draw)==length(m.blk)+m.bl.n
    end
end
