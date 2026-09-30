# Nonlinear posterior reference. The Gaussian solve supplies an importance proposal only; every draw
# is evaluated under the original dynamics and observation likelihood. No calibration truth is used.
using SpecialFunctions: loggamma

export SampledPosterior, sample_posterior, posterior_quantile

"Independent prior/process coordinates and the frozen generative graph needed to replay joint histories."
struct PosteriorHistory
    reference::NamedTuple
    coordinates::Matrix{Float64}
end

"""
    SampledPosterior

Weighted joint draws of the static latents and the latest state, with decoded physical-state draws.
`diagnostics` describes numerical sampling quality, not validation of the physical/noise model.
Unlike a Gaussian covariance, `draws` retains nonlinear dependence and non-Gaussian marginal shapes.
"""
struct SampledPosterior
    t::Float64
    model::SystemModel
    draws::Matrix{Float64}             # [latent coordinates; belief coordinates], columns are joint draws
    physical::Matrix{Float64}
    weights::Vector{Float64}
    mean::Vector{Float64}
    covariance::Matrix{Float64}
    diagnostics::NamedTuple
    history::Union{Nothing,PosteriorHistory}
end
SampledPosterior(t,model,draws,physical,weights,mean,covariance,diagnostics)=
    SampledPosterior(t,model,draws,physical,weights,mean,covariance,diagnostics,nothing)

function _weighted_quantiles(x, w, probabilities)
    positive=findall(>(0),w);isempty(positive) && throw(ArgumentError("quantiles require positive posterior weights"))
    order=positive[sortperm(x[positive])]; cumulative=cumsum(w[order]); cumulative[end]=1.0
    return [x[order[searchsortedfirst(cumulative,p)]] for p in probabilities]
end

# Whiten sample contrasts through the rectangular weighted design rather than inverting its Gram
# matrix. State fields can have tiny shape variance alongside large common-mode uncertainty.
function _posterior_whiten(draws,weights,center,values)
    d,n=size(draws);n>d || throw(ArgumentError("too few posterior draws for a joint covariance"))
    delta=draws.-center;scales=sqrt.(vec(sum(delta.^2 .* weights';dims=2)))
    all(>(0),scales) || throw(ArgumentError("a posterior coordinate has zero sampled variance"))
    design=Matrix((delta./scales)') .* sqrt.(weights)
    R=Matrix(qr(design).R)
    minimum(abs.(diag(R)))>eps(Float64)*max(size(design)...)*maximum(abs.(diag(R))) ||
        throw(ArgumentError("posterior sample covariance is numerically rank deficient"))
    return LowerTriangular(R')\(values./scales)
end

"Posterior quantiles from the actual marginal draws, not a Gaussian interval from their moments."
function posterior_quantile(p::SampledPosterior, probabilities=[0.025,0.5,0.975]; physical::Bool=false)
    all(q -> 0<=q<=1, probabilities) || throw(ArgumentError("quantile probabilities must lie in [0,1]"))
    X=physical ? p.physical : p.draws
    return reduce(vcat,[permutedims(_weighted_quantiles(view(X,i,:),p.weights,probabilities)) for i in axes(X,1)])
end

# Integrate the OU chains out analytically. Their conditional covariance is independent of the
# physical trajectory, so this is exact for uncensored Gaussian sensors at arbitrary sample times.
function _posterior_transport(sm::FixedLagSmoother)
    sm.split===nothing || throw(ArgumentError("posterior sampling currently requires static latents"))
    any(f->f.censored,sm.obs) && throw(ArgumentError("posterior sampling currently requires uncensored readings"))
    any(f->f isa LinearFactor || f isa HierarchyFactor,sm.graph.factors) &&
        throw(ArgumentError("posterior sampling requires the full nonlinear history; initialize with lag=typemax(Int)"))
    # Work on a copy: posterior extraction must not alter the live graph or its anchored noise model.
    g=deepcopy(sm.graph); kz=sm.key_z; ks=[kz;[k for (_,k) in sm.nodes]]
    off=Dict{VarKey,UnitRange{Int}}(); n=0
    for k in ks; off[k]=n+1:n+g.dims[k]; n+=g.dims[k]; end
    generators=Dict{VarKey,Factor}()
    for f in g.factors
        if f isa SitePriorFactor && f.key==kz
            all(s->isscore(s) || !_betalike(s),f.sites) ||
                throw(ArgumentError("posterior sampling requires normal or normal-score latent coordinates"))
            f.indices==collect(1:g.dims[kz]) || throw(ArgumentError("posterior sampling requires one complete latent prior"))
            haskey(generators,kz) && throw(ArgumentError("multiple latent priors are not supported"))
            generators[kz]=f
        elseif f isa GaussianPrior && f.key in ks
            haskey(generators,f.key) && throw(ArgumentError("multiple state generators are not supported"))
            generators[f.key]=f
        elseif f isa DynamicsFactor
            haskey(generators,f.key_y1) && throw(ArgumentError("multiple state generators are not supported"))
            generators[f.key_y1]=f
        elseif !(f isa ObservationFactor || f isa DiscrepancyFactor || (f isa GaussianPrior && f.key.entity===:disc))
            throw(ArgumentError("unsupported factor in posterior sampling: $(typeof(f))"))
        end
    end
    length(generators)==length(ks) && haskey(generators,kz) ||
        throw(ArgumentError("posterior sampling requires the complete generative history"))
    prior=generators[kz]
    prior isa SitePriorFactor || throw(ArgumentError("posterior sampling requires a site prior on latents"))
    zm=[isscore(s) ? 0.0 : s.p1 for s in prior.sites]
    zs=[isscore(s) ? 1.0 : s.p2 for s in prior.sites]
    center=zeros(n);center[off[kz]]=(g.values[kz]-zm)./zs
    roots=Dict{VarKey,Matrix{Float64}}()
    for (node,k) in enumerate(ks[2:end])
        f=generators[k]
        if f isa GaussianPrior
            center[off[k]]=residual(f,[g.values[k]]);roots[k]=inv(f.R)
        else
            f.key_z==kz && f.key_m===nothing || throw(ArgumentError("unsupported dynamics latent structure"))
            node>1 && f.key_y0==ks[node] || throw(ArgumentError("posterior sampling requires a chronological state chain"))
            f.relin_tol=0.0;f.relin_tol_state=0.0
            c=_dyn_eval!(f,g.values[kz],g.values[f.key_y0];jac=true)
            center[off[k]]=c.W*(g.values[k]-c.f);roots[k]=inv(c.W)
        end
    end
    fs=ObservationFactor[f for f in g.factors if f isa ObservationFactor]
    # Derive the discrepancy law from its graph factors, including its initial mean/variance.
    # This also verifies that no extra prior or OU factor is silently omitted from the target.
    dkeys=[k for k in g.keys if k.entity===:disc];dindex=Dict(k=>i for (i,k) in enumerate(dkeys))
    dg=Dict{VarKey,Factor}()
    for f in g.factors
        child=f isa DiscrepancyFactor ? f.key_next : f isa GaussianPrior && f.key.entity===:disc ? f.key : nothing
        child===nothing && continue
        haskey(dg,child) && throw(ArgumentError("multiple discrepancy generators are not supported"))
        dg[child]=f
    end
    length(dg)==length(dkeys) || throw(ArgumentError("incomplete discrepancy history"))
    D=zeros(length(dkeys),length(dkeys));dm=zeros(length(dkeys));pending=copy(dkeys);done=Set{VarKey}()
    while !isempty(pending)
        before=length(pending)
        for k in copy(pending)
            g.dims[k]==1 || throw(ArgumentError("discrepancy variables must be scalar"))
            f=dg[k];i=dindex[k]
            if f isa GaussianPrior
                size(f.R)==(1,1) || throw(ArgumentError("discrepancy priors must be scalar"))
                dm[i]=only(f.m);D[i,i]=1/only(f.R)
            else
                f.key_prev in done || continue
                ip=dindex[f.key_prev];dm[i]=f.ρ*dm[ip];D[i,:]=f.ρ*D[ip,:];D[i,i]=f.σ_inn
            end
            deleteat!(pending,findfirst(==(k),pending));push!(done,k)
        end
        length(pending)<before || throw(ArgumentError("discrepancy graph is cyclic or disconnected"))
    end
    Dobs=zeros(length(fs),length(dkeys));obsbias=zeros(length(fs))
    for (i,f) in enumerate(fs)
        if f.key_d!==nothing
            row=dindex[f.key_d];Dobs[i,:]=D[row,:];obsbias[i]=dm[row]
        end
    end
    K=Dobs*Dobs'+Diagonal([f.σ^2 for f in fs])
    W=isempty(fs) ? zeros(0,0) : Matrix(inv(cholesky(Symmetric(K)).L))
    # Build the likelihood Jacobian in independent process coordinates. The prior precision here
    # is exactly I, so all extra Gaussian information is low rank in the number of observations.
    nz=length(zm); ny=sm.model.bl.n; Z=zeros(nz,n);Z[:,off[kz]]=Diagonal(zs)
    Y=zeros(ny,n);H=zeros(length(fs),n)
    for k in ks[2:end]
        f=generators[k]
        if f isa GaussianPrior
            fill!(Y,0);Y[:,off[k]]=roots[k]
        else
            c=f.cache;Y=c.Jy*Y+c.Jz*Z;Y[:,off[k]]+=roots[k]
        end
        for (i,fobs) in enumerate(fs)
            fobs.key_y==k || continue
            v=vcat(g.values[kz],g.values[k],fobs.key_d===nothing ? Float64[] : [0.0])
            grad=ForwardDiff.gradient(w->_obs_pred(fobs,w),v)
            H[i,:]=vec(grad[1:nz]'*Z+grad[nz+1:nz+ny]'*Y)
        end
    end
    if isempty(fs)
        V=zeros(n,0);a=Float64[]
    else
        decomposition=svd(W*H;full=false);V=Matrix(decomposition.V)
        a=1 ./ sqrt.(1 .+ decomposition.S.^2)
    end
    return (;g,kz,ks,off,n,center,generators,roots,zm,zs,fs,W,obsbias,D,Dobs,dm,dkeys,V,a,model=sm.model,t=sm.nodes[end][1])
end

"The sampled posterior's means/covariances in the same layout as `belief_at(sm)`; requires adequate estimated Monte Carlo precision."
function belief_at(p::SampledPosterior)
    p.diagnostics.mc_precision_ok || throw(ArgumentError("posterior sampling has not met its Monte Carlo precision checks; inspect diagnostics and increase or improve sampling"))
    nz=length(p.model.blk);ry=nz+1:length(p.mean);bl=p.model.bl
    x=p.physical*p.weights;dx=p.physical.-x;Px=Matrix(Symmetric((dx.*p.weights')*dx'))
    pieces=Dict{Symbol,NamedTuple}()
    for id in bl.sys.pieces
        pc=piece(p.model.sys,id);rT=belief_index(bl,id,:T)
        if pc isa GridPiece
            ic=rT[pc.mesh.core];weights=zeros(length(rT))
            for (node,area) in pc.mesh.boundary;weights[node]+=area/pc.mesh.area;end
            surface=dot(weights,view(x,rT));variance=dot(weights,view(Px,rT,rT)*weights)
            pieces[id]=(T_centre=(x[ic],sqrt(max(Px[ic,ic],0.0))),T_surface=(surface,sqrt(max(variance,0.0))))
        else
            ic,is=first(rT),last(rT)
            pieces[id]=(T_centre=(x[ic],sqrt(max(Px[ic,ic],0.0))),T_surface=(x[is],sqrt(max(Px[is,is],0.0))))
        end
    end
    ip=first(belief_index(bl,:pit,:T_pit))
    return (t=p.t,z=p.mean[1:nz],z_cov=p.covariance[1:nz,1:nz],y=p.mean[ry],y_cov=p.covariance[ry,ry],
            zy_cov=copy(p.covariance),x=x,x_cov=Px,T_pit=(x[ip],sqrt(max(Px[ip,ip],0.0))),pieces=pieces)
end

function _transport_draw(ref,u)
    z=ref.zm+ref.zs.*view(u,ref.off[ref.kz]);y=Float64[]
    predictions=zeros(length(ref.fs))
    for k in ref.ks[2:end]
        f=ref.generators[k]
        if f isa GaussianPrior
            y=f.m+ref.roots[k]*view(u,ref.off[k])
        else
            y=propagate(f.model,y,z,f.t0,f.t1;first_slice=f.first)[1]+ref.roots[k]*view(u,ref.off[k])
        end
        for (i,obs) in enumerate(ref.fs)
            obs.key_y==k || continue
            v=vcat(z,y,obs.key_d===nothing ? Float64[] : [0.0])
            predictions[i]=_obs_pred(obs,v)
        end
    end
    wr=ref.W*(predictions+ref.obsbias-[f.value for f in ref.fs])
    return (potential=(sum(abs2,u)+sum(abs2,wr))/2,draw=vcat(z,y),predictions=predictions)
end

"""
    sample_posterior(sm; n=8192, rng=MersenneTwister(0), batch=512)

Sample the full nonlinear posterior of a static-latent, uncensored `FixedLagSmoother` which has retained
its full history (`lag=typemax(Int)`). The graph's current point supplies a computational proposal; the
weights evaluate the original nonlinear likelihood, with OU sensor paths integrated out exactly.
Sensor type, count, placement, noise and timestamps come from the ordinary `SensorSpec` declarations.

The proposal is a fixed 80/20 mixture of a local Gaussian and a Student-t proposal in strongly informed
directions, with exact mixture-density correction. The posterior itself is not assumed Student-t or
Gaussian. Nonlinear dynamics map independent process disturbances into each joint state/parameter draw.

This is an offline reference with cost proportional to `n` times the retained trajectory length. Inspect
`diagnostics`, especially tail effective sample sizes and numerical failures. Sampling quality alone does
not establish calibration under physical model mismatch. It does not change the live smoother.
"""
function sample_posterior(sm::FixedLagSmoother; n::Int=8192,rng::AbstractRNG=MersenneTwister(0),batch::Int=512)
    n>=2 && batch>0 || throw(ArgumentError("sample count must be at least two and batch size positive"))
    ref=_posterior_transport(sm);d=length(ref.zm)+sm.model.bl.n
    X=zeros(d,n);coordinates=zeros(ref.n,n);logw=fill(-Inf,n);failures=String[]
    Vtail=ref.V[:,ref.a.^2 .< 0.8];r=size(Vtail,2);nu=5;mixture=0.2
    for first in 1:batch:n
        last=min(n,first+batch-1);nb=last-first+1
        eta=randn(rng,ref.n,nb);correction=zeros(nb)
        for j in 1:nb
            if r>0 && rand(rng)<mixture
                scale=sqrt(nu/sum(abs2,randn(rng,nu)))
                eta[:,j]+=Vtail*((scale-1)*(Vtail'*eta[:,j]))
            end
            if r>0
                radial=sum(abs2,Vtail'*eta[:,j])
                ratio=loggamma((nu+r)/2)-loggamma(nu/2)+r/2*log(2/nu)-(nu+r)/2*log1p(radial/nu)+radial/2
                a=log1p(-mixture);b=log(mixture)+ratio;c=max(a,b)
                correction[j]=c+log(exp(a-c)+exp(b-c))
            end
        end
        U=ref.center .+ eta+ref.V*((ref.a.-1).*(ref.V'*eta))
        coordinates[:,first:last]=U
        errors=fill("",nb)
        Threads.@threads for j in 1:nb
            try
                result=_transport_draw(ref,view(U,:,j))
                isfinite(result.potential) && all(isfinite,result.draw) || error("non-finite trajectory or likelihood")
                X[:,first+j-1]=result.draw
                logw[first+j-1]=-result.potential+sum(abs2,view(eta,:,j))/2-correction[j]
            catch err
                err isa InterruptException && rethrow()
                errors[j]=sprint(showerror,err)
            end
        end
        append!(failures,filter(!isempty,errors))
    end
    all(!isfinite,logw) && error("no finite posterior samples; numerical failures: $(failures[1:min(3,length(failures))])")
    w=exp.(logw.-maximum(logw));w./=sum(w);mu=X*w;delta=X.-mu
    C=Matrix(Symmetric((delta.*w')*delta'))
    tail_ess=zeros(d,2)
    for i in 1:d
        qs=_weighted_quantiles(view(X,i,:),w,[.025,.975])
        for (j,p) in enumerate((.025,.975))
            tail_ess[i,j]=p*(1-p)/sum(abs2,w.*((view(X,i,:).<=qs[j]).-p))
        end
    end
    mcse=sqrt.(vec(sum((delta.*w').^2;dims=2)))
    joint_mc_error=try
        sum(abs2,_posterior_whiten(X,w,mu,delta.*w'))
    catch err
        err isa ArgumentError || rethrow()
        Inf
    end
    nz=length(ref.zm);physical=zeros(sm.model.bl.n,n)
    for j in 1:n;physical[:,j]=belief_decode(view(X,nz+1:d,j),sm.model.bl);end
    diagnostics=(n=n,ess=inv(sum(abs2,w)),max_weight=maximum(w),tail_ess=tail_ess,mean_mcse=mcse,
                 joint_mean_mc_error=joint_mc_error,
                 mc_precision_ok=isempty(failures) && inv(sum(abs2,w))>=max(400,20d) && minimum(tail_ess)>=400 && joint_mc_error<=.05 &&
                     all(mcse .<= .05sqrt.(max.(diag(C),eps(Float64)))),
                 numerical_failures=failures,proposal=:defensive_transport_importance)
    return SampledPosterior(ref.t,sm.model,X,physical,w,mu,C,diagnostics,PosteriorHistory(ref,coordinates))
end
