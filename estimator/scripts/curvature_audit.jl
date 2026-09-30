# Offline diagnostic only. No inference defaults, priors, or noise scales change.
# Include after BrisketEstimator has been loaded.
using LinearAlgebra

"Resolve the ODE more accurately while preserving the frozen process-noise law and chart."
function curvature_accuracy_ref(ref,factor)
    isfinite(factor) && 0<factor<=1 || throw(ArgumentError("solver accuracy factor must be in (0,1]"))
    out=deepcopy(ref)
    for f in out.dynamics
        m=f.model
        f.model=E.SystemModel(m.sys,m.log,m.blk,m.bl,m.sn,m.abstol*factor,m.reltol*factor,m.rom)
        c=f.cache
        # Invalidate primal and tangent solves, retaining W and its original anchor.
        f.cache=merge(c,(z=fill(NaN,length(c.z)),y0=fill(NaN,length(c.y0)),
            jz=nothing,jy0=nothing,Jy=nothing,Jz=nothing))
    end
    return out
end

"Evaluate the exact target in innovations without subtracting nearly equal states."
function innovation_target(ref,u;gradient=false,with_state=false)
    x=copy(u);off=ref.offsets
    for f in ref.dynamics
        f.key_m===nothing || throw(ArgumentError("curvature audit currently requires static latents"))
        c=E._dyn_eval!(f,x[off[f.key_z]],x[off[f.key_y0]];jac=gradient)
        x[off[f.key_y1]]=c.f+c.W\u[off[f.key_y1]]
    end
    value=0.0;gx=gradient ? zeros(length(u)) : nothing
    for f in ref.graph.factors
        if f isa E.DynamicsFactor
            value+=sum(abs2,u[off[f.key_y1]])/2
        else
            vals=[x[off[k]] for k in E.vars(f)]
            value+=E.loss(f,vals)
            if gradient
                ends=cumsum(length.(vals));starts=vcat(1,ends[1:end-1].+1)
                ranges=[a:b for (a,b) in zip(starts,ends)]
                # Differentiate loss itself, including non-least-squares sites.
                # A factor's optimizer surrogate need not encode its exact loss.
                gf=E.ForwardDiff.gradient(v->E.loss(f,[v[r] for r in ranges]),reduce(vcat,vals))
                for (k,r) in zip(E.vars(f),ranges)
                    gx[off[k]]+=gf[r]
                end
            end
        end
    end
    gradient || return with_state ? (value=value,x=x) : value
    gu=copy(gx)
    for f in reverse(ref.dynamics)
        c=f.cache;g1=copy(gx[off[f.key_y1]])
        gx[off[f.key_z]]+=c.Jz'*g1
        gx[off[f.key_y0]]+=c.Jy'*g1
        gu[off[f.key_y1]]=c.W'\g1+u[off[f.key_y1]]
    end
    generated=Set(f.key_y1 for f in ref.dynamics)
    for k in ref.graph.keys
        k in generated || (gu[off[k]]=gx[off[k]])
    end
    return (value=value,gradient=gu,x=x)
end

"Check that the supplied root actually whitens the optimizer's local metric."
function innovation_gn_metric(ref,u)
    base=innovation_target(ref,u;gradient=true);D=copy(ref.root);off=ref.offsets
    for f in ref.dynamics
        c=f.cache
        D[off[f.key_y1],:]=c.Jy*D[off[f.key_y0],:]+c.Jz*D[off[f.key_z],:]+c.W\ref.root[off[f.key_y1],:]
    end
    H=zeros(size(ref.root,2),size(ref.root,2))
    for f in ref.graph.factors
        if f isa E.DynamicsFactor
            A=ref.root[off[f.key_y1],:]
        else
            lin=E.factor_linearization(f,[base.x[off[k]] for k in E.vars(f)])
            A=sum(J*D[off[k],:] for (k,J) in zip(E.vars(f),lin.Js))
        end
        H+=A'*A
    end
    return H
end

"Orthonormal span of parameter directions in conditional GN-whitened coordinates."
function curvature_parameter_basis(ref,key)
    A=ref.root[ref.offsets[key],:]
    scales=vec(sqrt.(sum(abs2,A;dims=2)));keep=findall(>(0),scales)
    isempty(keep) && return zeros(size(A,2),0)
    s=svd(A[keep,:]./scales[keep])
    rank=count(>(maximum(s.S)*1e-10),s.S)
    return Matrix(s.V[:,1:rank])
end

"Central differences of AD-derived scores; retain unsymmetrized columns and check numerical stability."
function innovation_curvature(ref;key,basis=:parameters,steps=(.04,.02,.01))
    all(>(0),steps) && length(steps)>=2 || throw(ArgumentError("at least two positive steps required"))
    n=size(ref.root,2)
    Q=basis===:full ? Matrix{Float64}(I,n,n) : basis===:parameters ? curvature_parameter_basis(ref,key) : Matrix(basis)
    size(Q,1)==n && opnorm(Q'*Q-I)<1e-8 || throw(ArgumentError("basis must be orthonormal"))
    # Fresh tangents everywhere, while keeping each process whitener fixed.
    for f in ref.dynamics;f.relin_tol=0.;f.relin_tol_state=0.;end
    base=innovation_target(ref,ref.mean;gradient=true)
    raw_objective_difference=E._objective_at(ref.graph,base.x,ref.offsets)-base.value
    gn=innovation_gn_metric(ref,ref.mean)
    g0=ref.root'*base.gradient
    columns=Matrix{Float64}[];checks=NamedTuple[]
    for h in steps
        H=zeros(n,size(Q,2));slope_error=0.
        for j in axes(Q,2)
            du=ref.root*Q[:,j]
            p=innovation_target(ref,ref.mean+h*du;gradient=true)
            m=innovation_target(ref,ref.mean-h*du;gradient=true)
            H[:,j]=ref.root'*(p.gradient-m.gradient)/(2h)
            slope_error=max(slope_error,abs((p.value-m.value)/(2h)-dot(g0,Q[:,j])))
        end
        projected=Q'*H
        change=isempty(columns) ? nothing : opnorm(H-last(columns))
        push!(columns,H)
        push!(checks,(step=h,change=change,asymmetry=opnorm(projected-projected'),
            slope_error=slope_error,gn_difference=opnorm(H-Q),
            cross_curvature=norm(H-Q*projected),eigenvalues=eigvals(Symmetric((projected+projected')/2))))
    end
    H=Matrix(Symmetric((Q'*last(columns)+last(columns)'*Q)/2))
    return (basis=Q,hessian=H,columns=columns,gradient=g0,objective=base.value,checks=checks,
        gn_metric=gn,gn_whitening_error=opnorm(gn-I),raw_objective_difference=raw_objective_difference,
        dimension=n,rank=size(Q,2),full=size(Q,2)==n)
end

"Separate local curvature, odd departures, and even higher-order departures."
function curvature_direction_report(ref,audit;key,radii=(.5,1.,2.))
    result=NamedTuple[]
    for (i,row) in enumerate(ref.offsets[key])
        sd=norm(ref.root[row,:]);sd>0 || continue
        q=ref.root[row,:]/sd
        norm(q-audit.basis*(audit.basis'*q))<1e-8 || continue
        v=audit.basis'*q;h=dot(v,audit.hessian*v);slope=dot(audit.gradient,q)
        values=NamedTuple[]
        for t in radii
            p=innovation_target(ref,ref.mean+t*ref.root*q;with_state=true)
            ep=E._objective_at(ref.graph,p.x,ref.offsets)-p.value
            m=innovation_target(ref,ref.mean-t*ref.root*q;with_state=true)
            em=E._objective_at(ref.graph,m.x,ref.offsets)-m.value
            fp=p.value-audit.objective;fm=m.value-audit.objective
            odd=(fp-fm)/2-t*slope
            even=(fp+fm)/2-t*t*h/2
            push!(values,(radius=t,plus=fp,minus=fm,roundtrip_error=max(abs(ep),abs(em)),odd_remainder=odd,even_remainder=even,
                gn_defect=max(abs(fp-t*slope-t*t/2),abs(fm+t*slope-t*t/2)),
                observed_defect=max(abs(fp-t*slope-t*t*h/2),abs(fm+t*slope-t*t*h/2))))
        end
        push!(result,(index=i,curvature=h,values=values))
    end
    return result
end

"Counterfactual local Laplace shape, using the FULL observed Hessian without clipping."
function curvature_corrected_report(ref,audit;key)
    audit.full || throw(ArgumentError("a projected Hessian is not a full covariance correction"))
    minimum(eigvals(Symmetric(audit.hessian)))>0 || return (positive=false,)
    U=cholesky(Symmetric(audit.hessian)).U
    corrected=E._ConditionalTransport(ref.graph,ref.offsets,ref.mean,ref.root*audit.basis/U,ref.dynamics)
    n=size(ref.root,2)
    localmodel=(basis=Matrix{Float64}(I,n,n),hessian=Matrix{Float64}(I,n,n),
        gradient=U'\(audit.basis'*audit.gradient),objective=audit.objective)
    directions=curvature_direction_report(corrected,localmodel;key)
    return (positive=true,log_volume_correction=-sum(log,diag(U)),directions=directions,
        qualification="counterfactual local quadratic only; derivative stability and posterior shape remain unchecked")
end

"Audit a frozen candidate. Returns numerical evidence, never an acceptance flag."
function audit_candidate(candidate;key,max_components=3,basis=:parameters,steps=(.04,.02,.01),solver_factor=1.)
    p=candidate.transported.source
    ids=sortperm(p.weights;rev=true)[1:min(max_components,length(p.weights))]
    reports=NamedTuple[]
    for j in ids
        start=time();ref=curvature_accuracy_ref(candidate.transported.components[j],solver_factor)
        audit=innovation_curvature(ref;key,basis,steps)
        directions=curvature_direction_report(ref,audit;key)
        spectrum=eigvals(Symmetric(audit.hessian))
        push!(reports,(component=j,weight=p.weights[j],coordinates=p.nodes[j].coordinates,
            dimension=audit.dimension,rank=audit.rank,full=audit.full,
            gradient_norm=norm(audit.gradient),gn_whitening_error=audit.gn_whitening_error,
            raw_objective_difference=audit.raw_objective_difference,checks=audit.checks,
            log_volume_correction=minimum(spectrum)>0 ? -sum(log,spectrum)/2 : nothing,
            directions=directions,corrected=audit.full ? curvature_corrected_report(ref,audit;key) : nothing,
            wall=time()-start))
        check=last(audit.checks)
        println("CURVATURE component ",j," weight ",p.weights[j]," seconds ",time()-start,
            " step change ",check.change," asymmetry ",check.asymmetry,
            " spectrum ",extrema(check.eigenvalues));flush(stdout)
    end
    return (components=reports,checked_mass=sum(p.weights[ids]),basis=basis,solver_factor=solver_factor,
        purpose="local curvature audit; not posterior coverage or a validated Hessian correction")
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)>=2 || error("usage: curvature_audit.jl SNAPSHOT.jls REPORT.json [parameters|full] [solver_factor=1]")
    include(joinpath(@__DIR__,"..","src","BrisketEstimator.jl"))
    using .BrisketEstimator, Serialization
    const E=BrisketEstimator
    snapshot=deserialize(ARGS[1])
    snapshot.backend==backend_fingerprint() || error("snapshot backend differs from the loaded source")
    snapshot.physics==physics_fingerprint() || error("snapshot physics differs from the loaded source")
    report=audit_candidate(snapshot.result.candidate;key=snapshot.accepted.bank.anchor.key_z,
        basis=length(ARGS)>=3 ? Symbol(ARGS[3]) : :parameters,
        solver_factor=length(ARGS)>=4 ? parse(Float64,ARGS[4]) : 1.)
    write(ARGS[2],E.json_document((backend=snapshot.backend,case=snapshot.case,report=report)))
end
