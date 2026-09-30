# Experimental chart learning from the original graph. No simulated truth,
# measurement labels, altered likelihoods, or recycled posterior factors.
isdefined(@__MODULE__,:ConditionalCharts) || include(joinpath(@__DIR__,"../src/conditional_charts.jl"))
using .ConditionalCharts
using LinearAlgebra
isdefined(@__MODULE__,:innovation_target) || include(joinpath(@__DIR__,"curvature_audit.jl"))
isdefined(@__MODULE__,:polished_gn_conditional_curvature) || include(joinpath(@__DIR__,"adaptive_curvature.jl"))

struct ChartFitFailure <: Exception
    cause
    completed::Int
    optimizer_attempts::Int
end
Base.showerror(io::IO,e::ChartFitFailure)=print(io,"chart fit failed after ",e.completed,
    " pilots / ",e.optimizer_attempts," optimizer attempts: ",sprint(showerror,e.cause))

"Learn a triangular pilot chart; its cost is charged before integration starts."
function fit_conditional_chart(graph;axes,center,scale,radius=2.,max_fits=100,
        curvature=polished_gn_conditional_curvature,iters=80,tol=1e-6,ftol=1e-8,
        step_bounds=Dict(),newton_tol=0.,progress=nothing,checkpoint=nothing)
    selected=collect(Tuple{E.VarKey,Int},axes);base=AffineChart(center,scale);k=length(selected)
    k==chart_dimension(base) && length(unique(selected))==k || throw(ArgumentError("invalid chart axes"))
    required=sum(1+2p+2p*p for p in 1:k-1;init=0)
    required<=max_fits || throw(ArgumentError("insufficient map-construction budget"))
    g=deepcopy(graph);off,n=E._offsets(g);ia=[off[key][i] for (key,i) in selected]
    generated=Set(f.key_y1 for f in g.factors if f isa E.DynamicsFactor)
    any(key in generated for (key,_) in selected) && throw(ArgumentError("map axes must not be generated states"))
    C=E.joint_cov(g)[1];mu=E._stacked(g,g.keys,off,n)
    records=NamedTuple[];attempts=Ref(0);common=Ref{Union{Nothing,E._ConditionalTransport}}(nothing)
    starts=Dict{Int,Vector{NamedTuple}}()
    function oracle(point)
        checkpoint===nothing || checkpoint()
        p=length(point);rows=ia[1:p];slope=C[:,rows]/C[rows,rows]
        previous=get!(starts,p,NamedTuple[])
        start=if isempty(previous)
            mu+slope*(point-mu[rows])
        else
            near=previous[argmin([norm(base.scale[1:p,1:p]\(v.point-point)) for v in previous])]
            near.state+slope*(point-near.point)
        end
        gc=deepcopy(g)
        for key in gc.keys;gc.values[key].=start[off[key]];end
        attempts[]+=1
        node=E._conditional_node!(gc,selected[1:p],point;iters,tol,ftol,step_bounds,newton_tol)
        if !node.solve.converged
            start=mu+slope*(point-mu[rows])
            for key in gc.keys;gc.values[key].=start[off[key]];end
            attempts[]+=1
            node=E._conditional_node!(gc,selected[1:p],point;iters,tol,ftol,step_bounds,newton_tol)
        end
        ref=E._conditional_transport(node);corrected=curvature(node,ref)
        node.solve.converged || get(corrected,:mode_converged,false) || error("unresolved chart pilot")
        decoder=get(corrected,:decoder,ref)
        if common[]===nothing
            common[]=deepcopy(decoder)
        else
            E._same_conditional_decoder(common[],decoder) || error("chart pilots use different numerical decoders")
        end
        mean=get(corrected,:mean,ref.mean)
        push!(previous,(point=copy(point),state=copy(get(corrected,:state_mean,node.mean))))
        push!(records,(prefix=copy(point),initial_solve_converged=node.solve.converged,curvature=corrected.diagnostics))
        progress===nothing || progress((event=:chart_pilot,completed=length(records),required=required,prefix=copy(point)))
        return (mean=mean[ia],root=corrected.root[ia,:])
    end
    fitted=try
        fit_triangular_chart(base,oracle;radius,max_fits)
    catch err
        err isa InterruptException && rethrow()
        throw(ChartFitFailure(err,length(records),attempts[]))
    end
    return (chart=fitted.chart,diagnostics=merge(fitted.diagnostics,
        (optimizer_attempts=attempts[],solves=records,committed=false)),decoder=common[])
end
