# A nonlinear importance coordinate map for DIRECT full-target integration.
# Conditional mode/curvature fits build a proposal, never posterior evidence.
using LinearAlgebra

function _direction_rotation(a)
    d=length(a);b=a/norm(a);v=copy(b);v[1]-=1.
    norm(v)<1e-14 && return Matrix{Float64}(I,d,d)
    Matrix{Float64}(I,d,d)-2*(v*v')/dot(v,v)
end

abstract type AbstractDirectChart end
struct DirectAffineChart <: AbstractDirectChart
    center::Vector{Float64}
    root::Matrix{Float64}
end
direct_map(c::DirectAffineChart,v)=(u=c.center+c.root*v,logjac=sum(log,diag(c.root)))

struct DirectConditionalChart <: AbstractDirectChart
    center::Vector{Float64}
    axes::Vector{Int}
    remainder::Vector{Int}
    axisroot::Matrix{Float64}
    slope::Matrix{Float64}
    restroot::Matrix{Float64}
    coefficients::Matrix{Float64}
end

function _direct_features(q)
    x=tanh.(q./2)
    vcat(1.,x,[x[i]*x[j] for j in eachindex(x) for i in j:length(x)])
end

function _direct_geometry(c::DirectConditionalChart,q)
    k=length(c.axes);r=length(c.remainder);values=c.coefficients*_direct_features(q)
    mean=values[1:r];root=zeros(r,r);at=r
    for j in 1:r,i in j:r
        at+=1;root[i,j]=i==j ? exp(values[at]) : values[at]
    end
    all(isfinite,root)&&all(>(0),diag(root)) || error("nonrepresentable direct transport geometry")
    mean,root
end

function direct_map(c::DirectConditionalChart,v)
    k=length(c.axes);q=v[1:k];m,L=_direct_geometry(c,q)
    shift=c.axisroot*q;u=copy(c.center);u[c.axes]+=shift
    u[c.remainder]+=c.slope*shift+c.restroot*(m+L*v[k+1:end])
    logjac=sum(log,diag(c.axisroot))+sum(log,diag(c.restroot))+sum(log,diag(L))
    (u=u,logjac=logjac)
end

function _direct_conditional_root(ref,remainder)
    H=Diagonal(sqrt.(max.(inv.(ref.a.^2).-1,0.)))*ref.V'
    d=length(remainder)
    A=Matrix{Float64}(I,d,d)
    if !isempty(ref.a)
        s=svd(H[:,remainder];full=false)
        a=1 ./sqrt.(1 .+s.S.^2)
        A+=s.V*Diagonal(a.-1)*s.V'
    end
    L=Matrix(LowerTriangular(Matrix(qr(A').R)'))
    L*Diagonal(sign.(diag(L)))
end

function _direct_graph_start!(g,ref,u)
    z=ref.zm+ref.zs.*u[ref.off[ref.kz]];g.values[ref.kz].=z
    y=Float64[]
    for key in ref.ks[2:end]
        f=ref.generators[key]
        y=f isa E.GaussianPrior ? f.m+ref.roots[key]*u[ref.off[key]] :
            E.propagate(f.model,y,z,f.t0,f.t1;first_slice=f.first)[1]+ref.roots[key]*u[ref.off[key]]
        g.values[key].=y
    end
    prediction=[E._obs_pred(f,vcat(z,g.values[f.key_y],f.key_d===nothing ? Float64[] : [0.])) for f in ref.fs]
    discrepancy=ref.dm+ref.D*ref.Dobs'*(ref.W'*(ref.W*([f.value for f in ref.fs]-prediction-ref.obsbias)))
    for (j,key) in enumerate(ref.dkeys);g.values[key].=[discrepancy[j]];end
    g
end

"""
Fit a globally invertible conditional map from the original factor objective.
Directions are ranked by departures from the local quadratic, without truth or
sensor labels. Bounded polynomial features prevent uncontrolled extrapolation;
each conditional root has a positive diagonal. The map covers the full space.

All density and moment integrals STILL evaluate the exact target and the exact
map determinant. An inaccurate conditional fit affects efficiency, not which
posterior is being integrated. There is no retained Gaussian remainder.
"""
function fit_direct_transport(sm;max_axes=2,progress=nothing,max_seconds=Inf)
    1<=max_axes<=3 || throw(ArgumentError("direct transport pilot budget supports one to three axes"))
    start=time();ref,center,root=direct_chart(sm);C=root*root';nz=length(ref.zm)
    origin=E._transport_draw(ref,center).potential;scores=zeros(nz)
    for j in 1:nz,s in (-2.,-1.,1.,2.)
        direction=C[:,j]/sqrt(C[j,j])
        score=origin-E._transport_draw(ref,center+s*direction).potential+s*s/2
        scores[j]+=abs(score)
    end
    selected=sortperm(scores;rev=true)[1:min(max_axes,nz)]
    remainder=setdiff(collect(1:ref.n),selected)
    A=Matrix(cholesky(Symmetric(C[selected,selected])).L)
    slope=C[remainder,selected]/C[selected,selected]
    B=_direct_conditional_root(ref,remainder);k=length(selected);r=length(remainder)
    points=[zeros(k)]
    for j in 1:k,s in (-3.,-1.5,1.5,3.)
        q=zeros(k);q[j]=s;push!(points,q)
    end
    for j in 2:k,i in 1:j-1,s in (-1.5,1.5),t in (-1.5,1.5)
        q=zeros(k);q[i]=s;q[j]=t;push!(points,q)
    end
    values=Vector{Float64}[];records=NamedTuple[]
    for q in points
        time()-start<=max_seconds || error("direct transport pilot time budget exhausted")
        target=center[selected]+A*q
        zvalues=ref.zm[selected]+ref.zs[selected].*target
        gc=deepcopy(sm.graph)
        _direct_graph_start!(gc,ref,center+C[:,selected]*(C[selected,selected]\(target-center[selected])))
        node=E._conditional_node!(gc,[(sm.key_z,j) for j in selected],zvalues;
            iters=max(sm.iters,80),tol=sm.tol,ftol=sm.ftol,
            step_bounds=Dict(sm.key_z=>sm.latent_step),newton_tol=E.ONLINE_NEWTON_TOL)
        node.solve.converged || error("direct transport conditional pilot did not converge")
        temporary=deepcopy(sm);temporary.graph=node.graph
        pilot=E._posterior_transport(temporary)
        conditional_root=_direct_conditional_root(pilot,remainder)
        mean=LowerTriangular(B)\(pilot.center[remainder]-center[remainder]-slope*(target-center[selected]))
        relative=LowerTriangular(B)\conditional_root
        packed=vcat(mean,[i==j ? log(relative[i,j]) : relative[i,j] for j in 1:r for i in j:r])
        push!(values,packed);push!(records,(point=q,objective=node.solve.F,converged=node.solve.converged))
        progress===nothing || progress((event=:transport_pilot,completed=length(records),required=length(points)))
    end
    design=reduce(vcat,[permutedims(_direct_features(q)) for q in points])
    Y=reduce(vcat,permutedims.(values));coefficients=Matrix((design\Y)')
    c=DirectConditionalChart(center,selected,remainder,A,slope,B,coefficients)
    (reference=ref,chart=c,base_root=root,diagnostics=(axes=selected,scores=scores,pilots=records,
        fits=length(points),full_target_correction=true,finite_support=false,
        max_pilot_residual=maximum(abs,design*coefficients'-Y)))
end

# A rotation within a conditional-normal block may depend on earlier coordinates:
# it still preserves the standard-normal measure and has unit absolute determinant.
function direct_conditioning(c::DirectAffineChart,ref,index,q)
    nz=length(ref.zm);n=length(q);B=Matrix{Float64}(I,n,n)
    if index<=nz
        a=ref.zs[index].*vec(c.root[index,:]);B=_direction_rotation(a)
        return (rotation=B,target=1,width=norm(a))
    end
    block=ref.off[last(ref.ks)]
    a=vec(ref.roots[last(ref.ks)][index-nz,:]'*c.root[block,block])
    B[block,block]=_direction_rotation(a)
    (rotation=B,target=first(block),width=norm(a))
end

function direct_conditioning(c::DirectConditionalChart,ref,index,q)
    nz=length(ref.zm);n=length(q);k=length(c.axes);rotation=Matrix{Float64}(I,n,n)
    axis=index<=nz ? findfirst(==(index),c.axes) : nothing
    if axis!==nothing
        a=ref.zs[index].*vec(c.axisroot[axis,:]);rotation[1:k,1:k]=_direction_rotation(a)
        return (rotation=rotation,target=1,width=norm(a))
    end
    _,L=_direct_geometry(c,q[1:k])
    if index<=nz
        row=findfirst(==(index),c.remainder)
        a=ref.zs[index].*vec(c.restroot[row,:]'*L)
        rotation[k+1:end,k+1:end]=_direction_rotation(a)
        return (rotation=rotation,target=k+1,width=norm(a))
    end
    block=ref.off[last(ref.ks)]
    rows=[findfirst(==(j),c.remainder) for j in block]
    a=vec(ref.roots[last(ref.ks)][index-nz,:]'*c.restroot[rows,rows]*L[rows,rows])
    columns=k .+rows;rotation[columns,columns]=_direction_rotation(a)
    (rotation=rotation,target=first(columns),width=norm(a))
end

"""
Orthogonal integration coordinates informed by changes in the Jacobian of the
exact likelihood correction and requested outputs. All directions are retained;
this is a rotation of N(0,I), not dimension truncation or a posterior assumption.
"""
function direct_curvature_rotation(design,dimension,parameter_dimension;step=.03,max_seconds=Inf)
    start=time();calls=Ref(0)
    function jacobian(q)
        points=Vector{Vector{Float64}}()
        for j in 1:dimension
            d=zeros(dimension);d[j]=step;push!(points,q+d,q-d)
        end
        values_=Vector{Vector{Float64}}(undef,length(points))
        Threads.@threads for j in eachindex(points);values_[j]=design(points[j]);end
        calls[]+=length(points)
        reduce(hcat,[(values_[2j-1]-values_[2j])/(2step) for j in 1:dimension])
    end
    zero=zeros(dimension);base=jacobian(zero);noutputs=size(base,1)
    grams=[zeros(dimension,dimension) for _ in 1:noutputs]
    for j in 1:parameter_dimension,s in (-1.,1.)
        time()-start<=max_seconds || error("direct curvature rotation time budget exhausted")
        q=zeros(dimension);q[j]=s;difference=jacobian(q)-base
        for i in 1:noutputs
            a=view(difference,i,:);grams[i].+=a*a'
        end
    end
    G=zeros(dimension,dimension)
    for gram in grams
        # Output whitening already specifies meaningful units. Normalize only
        # the rows whose curvature is large, so a single weakly observed field
        # cannot determine every numerical integration direction.
        G+=gram/max(tr(gram),1.)
    end
    eig=eigen(Symmetric(G));order=sortperm(eig.values;rev=true);R=eig.vectors[:,order]
    for j in 1:dimension
        i=argmax(abs.(R[:,j]));R[:,j].*=sign(R[i,j])
    end
    (rotation=R,diagnostics=(evaluations=calls[],eigenvalues=eig.values[order],
        orthogonality_error=norm(R'*R-I),dimension_retained=dimension,
        dimension_dropped=0,wall=time()-start,step=step))
end
