# Development oracle: all static parameters integrated explicitly. The state
# conditional approximation is separately labelled and must be validated.
module AllParameterRun
const E=Main.E
using Main.BrisketEstimator, LinearAlgebra, Statistics, Test
include(joinpath(@__DIR__,"../../estimator/scripts/direct_posterior.jl"))
function fit(sm;max_evaluations=3000,max_seconds=600.,progress=nothing)
    start=time();nz=length(sm.model.blk);ny=sm.model.bl.n;g=nz+ny
    off,_=E._offsets(sm.graph);rows=vcat(collect(off[sm.key_z]),collect(off[last(sm.nodes)[2]]))
    Cz=E.joint_cov(sm.graph;keys=[sm.key_z])[1];A=Matrix(cholesky(Symmetric(Cz)).L);mu=copy(sm.graph.values[sm.key_z])
    ref,cu,Lu=direct_chart(sm);Cu=Lu*Lu';slope=Cu[:,1:nz]/Cu[1:nz,1:nz]
    initial_node=E._conditional_node!(deepcopy(sm.graph),[(sm.key_z,j) for j in 1:nz],mu;
        iters=80,tol=sm.tol,ftol=sm.ftol,step_bounds=Dict(sm.key_z=>sm.latent_step),newton_tol=E.ONLINE_NEWTON_TOL)
    initial_node.solve.converged || error("initial conditional solve")
    log0=initial_node.log_integral;base=initial_node.mean[rows]
    fullroot,_,_=E._conditional_root(deepcopy(sm.graph));S=Matrix(qr(fullroot[rows,:]').R)';Sinv=inv(LowerTriangular(S))
    pairs=[(i,j) for j in 1:g for i in j:g];nodes=Dict{Tuple,NamedTuple}();guard=ReentrantLock()
    function oracle(q)
        point=mu+A*q;gc=deepcopy(sm.graph)
        target=(point-ref.zm)./ref.zs
        _direct_graph_start!(gc,ref,cu+slope*(target-cu[1:nz]))
        node=E._conditional_node!(gc,[(sm.key_z,j) for j in 1:nz],point;
            iters=80,tol=sm.tol,ftol=sm.ftol,step_bounds=Dict(sm.key_z=>sm.latent_step),newton_tol=E.ONLINE_NEWTON_TOL)
        node.solve.converged || error("conditional solve unresolved at $q")
        mean=node.mean[rows];root=node.root[rows,:]
        record=(logmass=node.log_integral,mean=mean,root=root)
        lock(guard) do;nodes[Tuple(q)]=record;end
        z=Sinv*(mean-base);R=Sinv*root;second=R*R'+z*z'
        weight=exp(node.log_integral-log0+sum(abs2,q)/2)
        weight.*vcat(1.,z,[second[i,j] for (i,j) in pairs])
    end
    function scales(v)
        Z=max(abs(v[1]),eps());m=v[2:1+g]/Z;d=zeros(g)
        for ((i,j),x) in zip(pairs,v[2+g:end]/Z);i==j && (d[i]=x);end
        s=sqrt.(max.(abs.(d-m.^2),eps()))
        Z.*vcat(1.,s./(1 .+2abs.(m)./s),[s[i]*s[j] for (i,j) in pairs])
    end
    quadrature=integrate_normal(oracle,nz;atol=.002,rtol=.002,max_evaluations,max_seconds,
        evaluate_batch=_direct_batch(oracle),progress,error_scale=scales)
    Z=quadrature.value[1];z=quadrature.value[2:g+1]/Z;M=zeros(g,g)
    for ((i,j),x) in zip(pairs,quadrature.value[2+g:end]/Z);M[i,j]=M[j,i]=x;end
    Cwhite=Matrix(Symmetric(M-z*z'));C=Matrix(Symmetric(S*Cwhite*S'));m=base+S*z
    (mean=m,covariance=C,valid=isposdef(Symmetric(Cwhite)),diagnostics=quadrature.diagnostics,
        conditional_gaussian_validated=false,wall=time()-start,nodes=nodes,mu=mu,A=A,log0=log0,Z=Z)
end
end