# Invertible joint-trajectory proposal; weights correct the exact nonlinear densities.
_reference_root(ref,v)=v+ref.V*((ref.a.-1).*(ref.V'*v))
_reference_invroot(ref,v)=v+ref.V*((1 ./ ref.a.-1).*(ref.V'*v))
function _affine_coordinates(old,ref,u)
    eta=vcat(_reference_invroot(old,view(u,1:old.n)-old.center),u[old.n+1:end])
    return ref.center+_reference_root(ref,eta)
end

function _affine_transport_extend(old,ref,u,predictions)
    residuals=predictions+old.obsbias-[f.value for f in old.fs]
    base_potential=(sum(abs2,u)+sum(abs2,old.W*residuals))/2
    candidate=_affine_coordinates(old,ref,u)
    result=_transport_draw(ref,candidate)
    logjac=sum(log,ref.a;init=0.)-sum(log,old.a;init=0.)
    return (draw=result.draw,predictions=result.predictions,coordinates=candidate,
        log_increment=base_potential-result.potential+logjac)
end

function _incremental_posterior(sm::ParticleSmoother,ref)
    old=sm.posterior;history=old.history;previous=history.reference
    _validate_particle_extension(previous,ref)
    n=length(old.weights);count=n÷sm.replicates
    U=vcat(history.coordinates,randn(sm.rng,ref.n-previous.n,n))
    X=similar(old.draws);P=zeros(length(ref.fs),n);logw=log.(old.weights)
    Threads.@threads for j in 1:n
        r=_affine_transport_extend(previous,ref,view(U,:,j),view(sm.predictions,:,j))
        all(isfinite,r.draw) && all(isfinite,r.predictions) && isfinite(r.log_increment) ||
            error("non-finite affine posterior trajectory")
        X[:,j]=r.draw;P[:,j]=r.predictions;U[:,j]=r.coordinates;logw[j]+=r.log_increment
    end
    w=zeros(n)
    for replicate in 1:sm.replicates
        block=(replicate-1)*count+1:replicate*count;hi=maximum(logw[block])
        isfinite(hi) || error("all sequential particle weights vanished")
        w[block]=exp.(logw[block].-hi);w[block]./=sum(w[block])*sm.replicates
    end
    p=_replicate_posterior(ref,X,w,sm.replicates;history=PosteriorHistory(ref,U),
        extra=(proposal=:affine_joint_history_importance,update=:incremental,refreshed=false,
            weight_ess=inv(sum(abs2,w)),propagated_intervals=n*(length(ref.ks)-2)))
    return p,P
end
