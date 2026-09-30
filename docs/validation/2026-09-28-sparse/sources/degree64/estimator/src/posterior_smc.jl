# Sequential Monte Carlo from a defensive proposal mixture to the exact nonlinear posterior.
# Del Moral, Doucet & Jasra (2006), doi:10.1111/j.1467-9868.2006.00553.x.
# The rejuvenation kernel augments the bridge with a Gaussian mixture component, then uses an
# elliptical slice move (Murray, Adams & MacKay, 2010) conditional on that component.

export sample_posterior_smc

# Refresh weak reference directions separately. Conditional on B'eta, an elliptical rotation in
# its orthogonal complement preserves the selected Gaussian reference component. The slice test
# still uses the full nonlinear target; the .99 cutoff only selects a computational proposal.
function _weak_moves!(ref,U,X,ratio,beta,mixture_density,rng,moves)
    nz=length(ref.zm);Z=zeros(ref.n,nz);Z[1:nz,:]=Matrix{Float64}(I,nz,nz)
    A=hcat(Z,ref.V[:,ref.a.<.99]);B=if isempty(A)
        zeros(ref.n,0)
    else
        F=svd(A;full=false);F.U[:,F.S.>1e-10]
    end
    size(B,2)==ref.n && return 0
    root(v)=v+ref.V*((ref.a.-1).*(ref.V'*v))
    invroot(v)=v+ref.V*((1 ./ ref.a.-1).*(ref.V'*v))
    n=size(U,2);seeds=rand(rng,UInt64,n);counts=zeros(Int,n)
    Threads.@threads for j in 1:n
        localrng=MersenneTwister(seeds[j]);u=copy(U[:,j]);r=ratio[j]
        for move in 1:moves
            global_component=rand(localrng)<mixture_density(u)[2]
            eta=global_component ? copy(u) : invroot(u-ref.center)
            fixed=B*(B'*eta);weak=eta-fixed;nu=randn(localrng,ref.n);nu-=B*(B'*nu)
            level=beta*r+log(rand(localrng));angle=2pi*rand(localrng);lo=angle-2pi;hi=angle
            accepted=false
            for attempt in 1:512
                candidate_eta=fixed+weak*cos(angle)+nu*sin(angle)
                candidate=global_component ? candidate_eta : ref.center+root(candidate_eta)
                result=_transport_draw(ref,candidate);counts[j]+=1
                isfinite(result.potential) && all(isfinite,result.draw) || error("non-finite weak-block proposal")
                candidate_r=-result.potential-mixture_density(candidate)[1]
                if beta*candidate_r>=level
                    u=candidate;r=candidate_r;X[:,j]=result.draw;accepted=true;break
                end
                if angle<0;lo=angle;else;hi=angle;end
                angle=lo+rand(localrng)*(hi-lo)
            end
            accepted || error("weak-block slice failed")
        end
        U[:,j]=u;ratio[j]=r
    end
    return sum(counts)
end

function _systematic_resample(weights,rng)
    n=length(weights);indices=zeros(Int,n);start=rand(rng)/n;j=1;total=weights[1]
    for i in 1:n
        value=start+(i-1)/n
        while value>total && j<n;j+=1;total+=weights[j];end
        indices[i]=j
    end
    return indices
end

function _elliptical_move(logratio,eta,current,rng;maxeval=512)
    nu=randn(rng,length(eta));level=current+log(rand(rng))
    angle=2pi*rand(rng);lo=angle-2pi;hi=angle
    for attempt in 1:maxeval
        proposal=eta*cos(angle)+nu*sin(angle);value=logratio(proposal)
        if isfinite(value) && value>=level;return proposal,value,attempt;end
        if angle<0;lo=angle;else;hi=angle;end
        angle=lo+rand(rng)*(hi-lo)
    end
    error("posterior rejuvenation failed to find a valid slice point")
end

function _posterior_particle_run(ref;n,moves,rng,maxstages,verbose=false)
    # Fixed settings across all sensor suites and cooks. The prior component gives global support;
    # these weights specify a computational proposal, never a change to the probability model.
    global_weight=.2;logdet=sum(log,ref.a;init=0.0)
    root(v)=v+ref.V*((ref.a.-1).*(ref.V'*v))
    invroot(v)=v+ref.V*((1 ./ ref.a.-1).*(ref.V'*v))
    function mixture_density(u)
        eta=invroot(u-ref.center)
        a=log1p(-global_weight)-logdet-sum(abs2,eta)/2
        b=log(global_weight)-sum(abs2,u)/2
        hi=max(a,b);total=hi+log(exp(a-hi)+exp(b-hi))
        return total,exp(b-total)
    end
    U=randn(rng,ref.n,n)
    for j in 1:n
        rand(rng)>=global_weight && (U[:,j]=ref.center+root(U[:,j]))
    end
    d=length(ref.zm)+ref.model.bl.n;X=zeros(d,n);ratio=zeros(n);ancestors=collect(1:n)
    Threads.@threads for j in 1:n
        result=_transport_draw(ref,view(U,:,j))
        isfinite(result.potential) && all(isfinite,result.draw) || error("non-finite initial posterior particle")
        X[:,j]=result.draw;ratio[j]=-result.potential-mixture_density(view(U,:,j))[1]
    end
    beta=0.0;stages=NamedTuple[]
    for stage in 1:maxstages
        function weights_at(b)
            logw=(b-beta).*ratio;w=exp.(logw.-maximum(logw));return w./sum(w)
        end
        lo=beta;hi=1.0
        if inv(sum(abs2,weights_at(hi)))<.8n
            for _ in 1:45
                mid=(lo+hi)/2
                if inv(sum(abs2,weights_at(mid)))<.8n;hi=mid;else;lo=mid;end
            end
        else
            lo=1.0
        end
        nextbeta=lo;nextbeta>beta || error("posterior tempering stalled")
        weights=weights_at(nextbeta);ess=inv(sum(abs2,weights));indices=_systematic_resample(weights,rng)
        U=U[:,indices];X=X[:,indices];ratio=ratio[indices];ancestors=ancestors[indices]
        beta=nextbeta;seeds=rand(rng,UInt32,n);counts=zeros(Int,n)
        Threads.@threads for j in 1:n
            localrng=MersenneTwister(seeds[j]);u=copy(U[:,j]);value=ratio[j]
            accepted=Ref((u=u,x=copy(X[:,j]),r=value))
            for move in 1:(beta==1.0 ? 2moves : moves)
                useglobal=rand(localrng)<mixture_density(u)[2]
                eta=useglobal ? copy(u) : invroot(u-ref.center)
                function logratio(v)
                    candidate=useglobal ? v : ref.center+root(v)
                    result=_transport_draw(ref,candidate)
                    isfinite(result.potential) && all(isfinite,result.draw) || error("non-finite posterior rejuvenation trajectory")
                    r=-result.potential-mixture_density(candidate)[1]
                    accepted[]=(u=candidate,x=result.draw,r=r)
                    return beta*r
                end
                _,_,attempts=_elliptical_move(logratio,eta,beta*value,localrng)
                counts[j]+=attempts;u=accepted[].u;value=accepted[].r
            end
            U[:,j]=u;X[:,j]=accepted[].x;ratio[j]=value
        end
        weak_evals=_weak_moves!(ref,U,X,ratio,beta,mixture_density,rng,beta==1.0 ? 2 : 1)
        push!(stages,(beta=beta,weight_ess=ess,evaluations=sum(counts)+weak_evals,initial_ancestors=length(unique(ancestors)),weak_evaluations=weak_evals))
        if verbose
            @printf("posterior SMC stage %d: beta %.6f, weight ESS %.0f, %d evaluations\n",stage,beta,ess,sum(counts)+weak_evals);flush(stdout)
        end
        beta==1.0 && return (draws=X,coordinates=U,stages=stages)
    end
    error("posterior tempering did not reach the full likelihood")
end

"""
    sample_posterior_smc(sm; particles=1024, replicates=8, moves=4, rng=MersenneTwister(0))

An expressive joint posterior on static latents and states, using the original nonlinear factor model.
Each independent particle system gradually bridges a local-Gaussian/prior mixture to the full posterior,
resamples, and applies invariant full-space and weak-complement elliptical moves. Full history and
uncensored readings are currently required. The separate weak refresh improves exploration without
changing the posterior target or adding information to poorly observed state directions.
Sensor composition is supplied only through the model's factors.

Unlike a final weight ESS, Monte Carlo errors estimated across independent particle systems account for
dependence created by resampling. Diagnostics report estimated mean and tail precision, all stage ESS
values, and ancestry. `mc_precision_ok` is a numerical precision check, not proof that all modes were found
or that the physical model is calibrated. Increasing particles/replicates and independent calibration tests
remain necessary. Insufficient precision prevents `belief_at(result)` from returning a belief summary.

This is a full-history reference algorithm; its cost grows with trajectory length. It does not change the
live smoother or silently replace its fixed-lag Gaussian approximation.
"""
function sample_posterior_smc(sm::FixedLagSmoother;particles::Int=1024,replicates::Int=8,moves::Int=4,
                              rng::AbstractRNG=MersenneTwister(0),maxstages::Int=100,verbose::Bool=false)
    particles>=2 && replicates>=4 && moves>=1 && maxstages>=1 ||
        throw(ArgumentError("SMC requires at least 2 particles, 4 independent replicates, 1 move and 1 stage"))
    ref=_posterior_transport(sm);d=length(ref.zm)+sm.model.bl.n;n=particles*replicates
    X=zeros(d,n);coordinates=zeros(ref.n,n);stages=Vector{NamedTuple}[]
    for replicate in 1:replicates
        result=_posterior_particle_run(ref;n=particles,moves,rng=MersenneTwister(rand(rng,UInt32)),maxstages,verbose)
        block=(replicate-1)*particles+1:replicate*particles
        X[:,block]=result.draws;coordinates[:,block]=result.coordinates
        push!(stages,result.stages)
    end
    return _replicate_posterior(ref,X,fill(1/n,n),replicates;
        history=PosteriorHistory(ref,coordinates),extra=(proposal=:defensive_tempered_smc,particles=particles,
            replicates=replicates,moves=moves,stages=stages))
end

# Independent systems remain the units of Monte Carlo replication after sequential reweighting.
# Each system has total weight 1/R; the unequal-weight floors prevent a shared few particles from
# masquerading as N independent draws. This is estimated precision, not a guarantee against missed modes.
function _replicate_posterior(ref,X,w,replicates;history=nothing,extra=NamedTuple())
    d,n=size(X);particles=n÷replicates
    particles*replicates==n && replicates>=4 || throw(ArgumentError("equal-size independent particle systems required"))
    means=zeros(d,replicates)
    for replicate in 1:replicates
        block=(replicate-1)*particles+1:replicate*particles
        isapprox(sum(w[block]),1/replicates;atol=1e-12) || throw(ArgumentError("each particle system must have equal total weight"))
        means[:,replicate]=X[:,block]*(replicates.*w[block])
    end
    mu=X*w;delta=X.-mu;C=Matrix(Symmetric((delta.*w')*delta'))
    mean_variance=vec(sum((means.-sum(means;dims=2)./replicates).^2;dims=2))./(replicates*(replicates-1))
    mean_variance=max.(mean_variance,vec(sum((delta.*w').^2;dims=2)),max.(diag(C),eps(Float64))./n)
    mcse=sqrt.(mean_variance);mean_ess=diag(C)./mean_variance
    joint_mc_error=try
        max(sum(abs2,_posterior_whiten(X,w,mu,means.-mu))/(replicates*(replicates-1)),
            sum(abs2,_posterior_whiten(X,w,mu,delta.*w')))
    catch err
        err isa ArgumentError || rethrow()
        Inf
    end
    tail_ess=zeros(d,2)
    for i in 1:d
        qs=_weighted_quantiles(view(X,i,:),w,[.025,.975])
        for (j,p) in enumerate((.025,.975))
            probabilities=[replicates*sum(w[k]*(X[i,k]<=qs[j]) for k in (replicate-1)*particles+1:replicate*particles) for replicate in 1:replicates]
            average=sum(probabilities)/replicates
            variance=sum(abs2,probabilities.-average)/(replicates*(replicates-1))
            floor=sum(abs2,w.*((view(X,i,:).<=qs[j]).-p))
            tail_ess[i,j]=min(n,p*(1-p)/max(variance,floor,eps(Float64)))
        end
    end
    nz=length(ref.zm);physical=zeros(ref.model.bl.n,n)
    for j in 1:n;physical[:,j]=belief_decode(view(X,nz+1:d,j),ref.model.bl);end
    diagnostics=merge((n=n,ess=minimum(mean_ess),ess_kind=:replicate_mean,mean_ess=mean_ess,max_weight=maximum(w),
                 tail_ess=tail_ess,mean_mcse=mcse,
                 joint_mean_mc_error=joint_mc_error,
                 mc_precision_ok=minimum(mean_ess)>=max(400,20d) && minimum(tail_ess)>=400 && joint_mc_error<=.05,
                 numerical_failures=String[],particles=particles,replicates=replicates),extra)
    return SampledPosterior(ref.t,ref.model,X,physical,w,mu,C,diagnostics,history)
end
