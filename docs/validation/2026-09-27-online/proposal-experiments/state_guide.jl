# Candidate computation proposal. The exact nonlinear importance correction preserves the target.
function _guided_state(ref,k,z,mean_y,eta,predictions,m;prior_weight=0.,use_prior=false)
    rows=findall(i->i>m && ref.fs[i].key_y==k,eachindex(ref.fs))
    root=ref.roots[k]
    isempty(rows) && return (y=mean_y+root*eta,u=copy(eta),correction=0.)
    function observations(y)
        [_obs_pred(ref.fs[i],vcat(z,y,ref.fs[i].key_d===nothing ? Float64[] : [0.0])) for i in rows]
    end
    predicted=copy(predictions);predicted[rows]=observations(mean_y)
    b=ref.W[rows,:]*(predicted+ref.obsbias-[f.value for f in ref.fs])
    H=ref.W[rows,rows]*ForwardDiff.jacobian(observations,mean_y)*root
    # q(u) is the Gaussian conditional for the local observation tangent and N(0,I) process prior.
    # Compute its square root through the small observation subspace, without subtracting covariances.
    decomposition=svd(H;full=false);V=decomposition.V;s=decomposition.S
    a=1 ./ sqrt.(1 .+ s.^2)
    center=-V*((s./(1 .+ s.^2)).*(decomposition.U'*b))
    u=use_prior ? copy(eta) : center+eta+V*((a.-1).*(V'*eta))
    logdet=sum(log,a;init=0.)
    correction=(sum(abs2,eta)-sum(abs2,u))/2+logdet
    if prior_weight>0
        # A defensive prior component retains broad support when the observation tangent is poor.
        # Correct against the complete mixture density, not just the selected component's density.
        centered=u-center;eta_q=centered+V*((1 ./ a.-1).*(V'*centered))
        logprior=-sum(abs2,u)/2
        logguide=-sum(abs2,eta_q)/2-logdet
        left=log1p(-prior_weight)+logguide;right=log(prior_weight)+logprior;hi=max(left,right)
        correction=logprior-hi-log(exp(left-hi)+exp(right-hi))
    end
    return (y=mean_y+root*u,u=u,correction=correction)
end

function _guided_transport_extend(old,ref,u,draw,predictions;prior_weight=0.,prior_choices=nothing)
    nz=length(ref.zm);z=copy(draw[1:nz]);y=copy(draw[nz+1:end]);m=length(old.fs)
    pred=vcat(predictions,zeros(length(ref.fs)-m));correction=0.
    function observe!(k)
        for i in m+1:length(ref.fs)
            obs=ref.fs[i];obs.key_y==k || continue
            pred[i]=_obs_pred(obs,vcat(z,y,obs.key_d===nothing ? Float64[] : [0.0]))
        end
    end
    # Before any observations, the retained initial-state coordinates are independent normals.
    # They may be transformed into an observation-conditioned proposal with the same exact correction.
    if m==0 && length(old.ks)==2
        k=last(old.ks);f=ref.generators[k]
        result=_guided_state(ref,k,z,f.m,view(u,ref.off[k]),pred,m;prior_weight,
            use_prior=prior_choices!==nothing && prior_choices[1])
        y=result.y;u[ref.off[k]]=result.u;correction+=result.correction
    end
    observe!(last(old.ks))
    for (step,k) in enumerate(ref.ks[length(old.ks)+1:end])
        f=ref.generators[k];mean_y=propagate(f.model,y,z,f.t0,f.t1;first_slice=f.first)[1]
        result=_guided_state(ref,k,z,mean_y,view(u,ref.off[k]),pred,m;prior_weight,
            use_prior=prior_choices!==nothing && prior_choices[step+1])
        y=result.y;u[ref.off[k]]=result.u;correction+=result.correction
        observe!(k)
    end
    wr=ref.W[m+1:end,:]*(pred+ref.obsbias-[f.value for f in ref.fs])
    return (draw=vcat(z,y),predictions=pred,log_increment=-sum(abs2,wr)/2+correction)
end

function _incremental_posterior(sm::ParticleSmoother,ref)
    old=sm.posterior;history=old.history;previous=history.reference
    _validate_particle_extension(previous,ref)
    n=length(old.weights);count=n÷sm.replicates
    U=vcat(history.coordinates,randn(sm.rng,ref.n-previous.n,n))
    prior_weight=.2;choices=rand(sm.rng,length(ref.ks)-length(previous.ks)+1,n).<prior_weight
    X=similar(old.draws);P=zeros(length(ref.fs),n);logw=log.(old.weights)
    Threads.@threads for j in 1:n
        r=_guided_transport_extend(previous,ref,view(U,:,j),view(old.draws,:,j),view(sm.predictions,:,j);
            prior_weight,prior_choices=view(choices,:,j))
        all(isfinite,r.draw) && all(isfinite,r.predictions) && isfinite(r.log_increment) ||
            error("non-finite guided posterior trajectory")
        X[:,j]=r.draw;P[:,j]=r.predictions;logw[j]+=r.log_increment
    end
    w=zeros(n)
    for replicate in 1:sm.replicates
        block=(replicate-1)*count+1:replicate*count;hi=maximum(logw[block])
        isfinite(hi) || error("all sequential particle weights vanished")
        w[block]=exp.(logw[block].-hi);w[block]./=sum(w[block])*sm.replicates
    end
    p=_replicate_posterior(ref,X,w,sm.replicates;history=PosteriorHistory(ref,U),
        extra=(proposal=:defensive_observation_conditioned_importance,prior_proposal_weight=prior_weight,
            update=:incremental,refreshed=false,weight_ess=inv(sum(abs2,w)),
            propagated_intervals=n*(length(ref.ks)-length(previous.ks))))
    return p,P
end
