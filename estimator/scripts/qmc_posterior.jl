# Experimental randomized quasi-Monte Carlo importance integration of the same
# original nonlinear target used by the independent posterior reference.
push!(LOAD_PATH,joinpath(@__DIR__,"../qmc"))
isdefined(@__MODULE__,:QMCNormalRules) || include(joinpath(@__DIR__,"../src/qmc_rules.jl"))
using .QMCNormalRules, LinearAlgebra, Random, Statistics

struct QMCTimeBudget <: Exception end
Base.showerror(io::IO,::QMCTimeBudget)=print(io,"posterior integration time budget exhausted")

function _qmc_summary(ref,X,rawlogweights,replicates;history,extra=NamedTuple())
    d,n=size(X);count=n÷replicates
    w,logs=E._normalize_particle_logweights(rawlogweights,replicates)
    mu=X*w;delta=X.-mu;C=Matrix(Symmetric((delta.*w')*delta'))
    # QR of the weighted design protects tiny nonlinear state contrasts.
    white=E._posterior_whiten(X,w,mu,delta)
    means=zeros(d,replicates);covs=zeros(d,d,replicates)
    for r in 1:replicates
        b=(r-1)*count+1:r*count;wb=replicates*w[b]
        means[:,r]=white[:,b]*wb
        covs[:,:,r]=(white[:,b].*wb')*white[:,b]'-means[:,r]*means[:,r]'
    end
    joint=sum(abs2,means)/(replicates*(replicates-1))
    cm=dropdims(mean(covs;dims=3);dims=3)
    covariance_error=sqrt(sum(abs2,covs.-cm)/(replicates*(replicates-1)*d))
    mean_se=zeros(d)
    for i in 1:d
        mr=[replicates*dot(w[(r-1)*count+1:r*count],X[i,(r-1)*count+1:r*count]) for r in 1:replicates]
        mean_se[i]=std(mr)/sqrt(replicates)
    end
    quantiles=[.025,.5,.975];tail_se=zeros(d,length(quantiles))
    for i in 1:d
        qs=E._weighted_quantiles(view(X,i,:),w,quantiles)
        for (j,q) in enumerate(qs)
            probabilities=[replicates*sum(w[k]*(X[i,k]<=q) for k in (r-1)*count+1:r*count) for r in 1:replicates]
            tail_se[i,j]=std(probabilities)/sqrt(replicates)
        end
    end
    ess=inv(sum(abs2,w))
    ok=ess>=max(200,4d) && joint<=.05 && covariance_error<=.1 && maximum(tail_se)<=.005 &&
       all(mean_se .<= .05sqrt.(max.(diag(C),eps(Float64))))
    nz=length(ref.zm);physical=zeros(ref.model.bl.n,n)
    for j in 1:n;physical[:,j]=E.belief_decode(view(X,nz+1:d,j),ref.model.bl);end
    diagnostics=merge((n=n,points=count,replicates=replicates,weight_ess=ess,ess=ess,
        max_weight=maximum(w),mean_mcse=mean_se,joint_mean_mc_error=joint,
        covariance_mcse=covariance_error,rank_mcse=tail_se,
        precision_before_refinement=ok,mc_precision_ok=false,refinement_checked=false,
        error_method=:independent_scrambles,numerical_failures=String[],statistically_consistent=false),extra)
    E.SampledPosterior(ref.t,ref.model,X,physical,w,mu,C,diagnostics,history),logs
end

function _qmc_level(ref,n,seeds;cache=nothing,deadline=Inf,method=:rqmc,batch=256)
    method in (:rqmc,:mc) || throw(ArgumentError("unknown importance point rule"))
    R=length(seeds);nz=length(ref.zm);d=nz+ref.model.bl.n;total=n*R
    X=zeros(d,total);U=zeros(ref.n,total);P=zeros(length(ref.fs),total);lw=zeros(total)
    Vtail=ref.V[:,ref.a.^2 .<.8];r=size(Vtail,2);nu=5;mixture=.25
    extra=r>0 ? 1+nu : 0;nd=ref.n+extra
    oldn=cache===nothing ? 0 : cache.points
    oldn<n || throw(ArgumentError("refinement must add points"))
    for replicate in 1:R
        time()<deadline || throw(QMCTimeBudget())
        rng=MersenneTwister(seeds[replicate])
        uniforms=if method===:rqmc
            uniform_net(n,nd;rng)
        else
            # Fixed-size calls preserve the prefix when n is doubled.
            hcat([rand(rng,nd) for _ in 1:n]...)
        end
        eta=-sqrt(2).*E.erfcinv.(2 .* uniforms[1:ref.n,:]);correction=zeros(n)
        if r>0
            for j in 1:n
                if uniforms[ref.n+1,j]<mixture
                    g=-sqrt(2).*E.erfcinv.(2 .*uniforms[ref.n+2:nd,j])
                    scale=sqrt(nu/sum(abs2,g))
                    eta[:,j]+=Vtail*((scale-1)*(Vtail'*eta[:,j]))
                end
                radial=sum(abs2,Vtail'*eta[:,j])
                ratio=E.loggamma((nu+r)/2)-E.loggamma(nu/2)+r/2*log(2/nu)-(nu+r)/2*log1p(radial/nu)+radial/2
                a=log1p(-mixture);b=log(mixture)+ratio;c=max(a,b)
                correction[j]=c+log(exp(a-c)+exp(b-c))
            end
        end
        b=(replicate-1)*n+1:replicate*n
        U[:,b]=ref.center .+ eta+ref.V*((ref.a.-1).*(ref.V'*eta))
        if oldn>0
            old=(replicate-1)*oldn+1:replicate*oldn;firsts=first(b):first(b)+oldn-1
            U[:,firsts]==cache.U[:,old] || error("importance refinement changed a retained coordinate")
            X[:,firsts]=cache.X[:,old];P[:,firsts]=cache.P[:,old];lw[firsts]=cache.logweights[old]
        end
        for firstpoint in oldn+1:batch:n
            time()<deadline || throw(QMCTimeBudget())
            lastpoint=min(n,firstpoint+batch-1)
            Threads.@threads for j in firstpoint:lastpoint
                k=first(b)+j-1;result=E._transport_draw(ref,view(U,:,k))
                isfinite(result.potential)&&all(isfinite,result.draw) || error("nonfinite importance trajectory")
                X[:,k]=result.draw;P[:,k]=result.predictions
                lw[k]=-result.potential+sum(abs2,view(eta,:,j))/2-correction[j]
            end
        end
    end
    (points=n,seeds=seeds,X=X,U=U,P=P,logweights=lw,method=method,uniform_dimensions=nd)
end

function _qmc_refinement(old,new)
    diff=E._posterior_whiten(new.draws,new.weights,new.mean,old.mean-new.mean)
    joint=sum(abs2,diff)
    oldwhite=E._posterior_whiten(new.draws,new.weights,new.mean,old.draws.-old.mean)
    C=(oldwhite.*old.weights')*oldwhite'
    covariance=norm(C-I)/sqrt(length(new.mean))
    rankdifference=0.
    for i in eachindex(new.mean)
        qs=E._weighted_quantiles(view(new.draws,i,:),new.weights,[.025,.5,.975])
        for q in qs
            a=sum(old.weights.*(view(old.draws,i,:).<=q));b=sum(new.weights.*(view(new.draws,i,:).<=q))
            rankdifference=max(rankdifference,abs(a-b))
        end
    end
    (joint_mean_change=joint,covariance_change=covariance,max_rank_change=rankdifference,
     passed=joint<=.05 && covariance<=.1 && rankdifference<=.01)
end

"""
Numerical full-target integration with independently scrambled Sobol nets (or an
IID baseline). Each replicate has a separate self-normalized importance estimate.
At least one doubling checks finite-net bias as well as between-replicate error.
The Gaussian/Student mixture is only a full-support proposal. No covariance or
noise calibration is fitted here, and numerical precision is not statistical SBC.
"""
function qmc_posterior_fit(sm;points=256,max_points=4096,replicates=8,
        rng=MersenneTwister(0),max_seconds=60.,method=:rqmc,progress=nothing)
    ispow2(points)&&ispow2(max_points)&&2<=points<max_points&&replicates>=8&&max_seconds>0 ||
        throw(ArgumentError("use nested power-of-two point budgets and at least eight replicates"))
    start=time();ref=E._posterior_transport(sm);seeds=rand(rng,UInt64,replicates)
    cache=nothing;previous=nothing;posterior=nothing;logs=nothing;levels=NamedTuple[];n=points
    while true
        try
            cache=_qmc_level(ref,n,seeds;cache,deadline=start+max_seconds,method)
        catch err
            err isa QMCTimeBudget || rethrow()
            posterior===nothing && rethrow()
            break
        end
        p,logs=_qmc_summary(ref,cache.X,cache.logweights,replicates;
            history=E.PosteriorHistory(ref,cache.U),extra=(proposal=:defensive_rqmc_importance,method=method,))
        refinement=previous===nothing ? nothing : _qmc_refinement(previous,p)
        ok=p.diagnostics.precision_before_refinement && refinement!==nothing && refinement.passed
        diagnostic=merge(p.diagnostics,(mc_precision_ok=ok,refinement_checked=refinement!==nothing,
            refinement=refinement,wall=time()-start,evaluations=n*replicates))
        posterior=E.SampledPosterior(p.t,p.model,p.draws,p.physical,p.weights,p.mean,p.covariance,diagnostic,p.history)
        record=(points=n,wall=time()-start,passed=ok,weight_ess=diagnostic.weight_ess,
            joint_mean_error=diagnostic.joint_mean_mc_error,covariance_error=diagnostic.covariance_mcse,
            max_rank_error=maximum(diagnostic.rank_mcse),refinement=refinement)
        push!(levels,record);progress===nothing || progress(record)
        (ok || n>=max_points || time()>=start+max_seconds) && break
        previous=posterior;n*=2
    end
    (posterior=posterior,predictions=cache.P,logweights=logs,cache=cache,
        diagnostics=(resolved=posterior.diagnostics.mc_precision_ok,levels=levels,wall=time()-start))
end