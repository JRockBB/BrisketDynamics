"""
Deterministic integration against a standard normal measure. The integrand is
evaluated directly: there is no fitted conditional Gaussian or density surrogate.
Sparse-grid coefficients are signed integration weights, NEVER probabilities.
"""
module DirectQuadrature
using LinearAlgebra
using SpecialFunctions: erfc, erfcinv
export integrate_normal, QuadratureResult, normal_rule

struct QuadratureResult
    value::Vector{Float64}
    error::Vector{Float64}
    indices::Vector{Tuple}
    frontier::Vector{Tuple}
    diagnostics::NamedTuple
end

function normal_rule(level::Int)
    level>=0 || throw(ArgumentError("negative quadrature level"))
    level==0 && return ([0.],[1.])
    n=2^(level+1)-1
    e=eigen(SymTridiagonal(zeros(n),sqrt.(Float64.(1:n-1))))
    x=e.values;w=vec(e.vectors[1,:].^2)
    # Preserve exact symmetry and a shared zero across nonnested odd rules.
    for j in 1:n÷2
        a=(abs(x[j])+abs(x[n+1-j]))/2;b=(w[j]+w[n+1-j])/2
        x[j]=-a;x[n+1-j]=a;w[j]=w[n+1-j]=b
    end
    x[n÷2+1]=0.;w./=sum(w)
    x,w
end

function composite_normal_rule(level::Int)
    level<=1 && return normal_rule(level)
    # Fixed positive Gauss-Legendre rules on successively subdivided intervals.
    # The two infinite tails are integrated in Gaussian probability coordinates;
    # no finite support is silently imposed on the target.
    j=Float64.(1:8);e=eigen(SymTridiagonal(zeros(9),j./sqrt.(4j.^2 .-1)))
    gx=e.values;gw=vec(2 .* e.vectors[1,:].^2)
    base=[-8.,-2.,0.,2.,8.];subdivisions=2^(level-2)
    xs=Float64[];ws=Float64[]
    for k in 1:length(base)-1
        edges=range(base[k],base[k+1];length=subdivisions+1)
        for i in 1:subdivisions
            half=(edges[i+1]-edges[i])/2;mid=(edges[i+1]+edges[i])/2
            for (a,w) in zip(gx,gw)
                x=mid+half*a;push!(xs,x);push!(ws,half*w*exp(-x*x/2)/sqrt(2pi))
            end
        end
    end
    tail=erfc(8/sqrt(2))/2
    for (a,w) in zip(gx,gw)
        x=-sqrt(2)*erfcinv(tail*(a+1))
        push!(xs,x,-x);push!(ws,tail*w/2,tail*w/2)
    end
    # The same scalar Gaussian measure is used at every level. Correct only
    # the scalar rule's numerical mass, never posterior moments or covariance.
    ws./=sum(ws)
    xs,ws
end

function difference_rule(level,rule=:hermite)
    generate=rule===:hermite ? normal_rule : composite_normal_rule
    x,w=generate(level)
    level==0 && return x,w
    old,v=generate(level-1);d=Dict(zip(x,w))
    for (a,b) in zip(old,v);d[a]=get(d,a,0.)-b;end
    keys_=sort(collect(keys(d)))
    keys_,[d[k] for k in keys_]
end

function points(index,rules)
    active=findall(>(0),collect(index));d=length(index)
    isempty(active) && return [(ntuple(_->0.,d),1.)]
    selected=[rules[index[i]+1] for i in active]
    out=Tuple{Tuple,Float64}[]
    for ix in Iterators.product((eachindex(r[1]) for r in selected)...)
        q=zeros(d);weight=1.
        for (j,i) in enumerate(active)
            q[i]=selected[j][1][ix[j]];weight*=selected[j][2][ix[j]]
        end
        weight==0 || push!(out,(Tuple(q),weight))
    end
    out
end

function children(index,accepted,max_level)
    result=Tuple[];d=length(index)
    for j in 1:d
        index[j]<max_level || continue
        next=ntuple(k->index[k]+(j==k),d)
        next in accepted && continue
        all(k->next[k]==0 || ntuple(a->next[a]-(a==k),d) in accepted,1:d) || continue
        push!(result,next)
    end
    result
end
"""
    integrate_normal(f, dimension; ...)

Dimension-adaptive Smolyak integration of a vector-valued function under N(0,I).
Every dimension is eligible for refinement. The complete degree-two index shell
(univariate refinement and all pairs) is checked before resolution is allowed.
The error indicator is the sum of absolute frontier contributions, component by
component. It is an estimate, not a rigorous tail or missed-interaction bound.
Failure/budget exhaustion returns unresolved; evaluation failures are not dropped.

The callback `evaluate_batch(points)` may provide parallel, ordered evaluations.
It receives only numerical points; statistical truth and sensor labels are absent.
"""
function integrate_normal(f,dimension::Int;atol=1e-5,rtol=1e-3,max_evaluations=100_000,
        max_level=5,min_shell=2,max_seconds=Inf,evaluate_batch=nothing,progress=nothing,
        priority=nothing,rule=:hermite,error_scale=nothing,verify_rotation=true)
    dimension>0 && max_evaluations>0 && 1<=max_level<=8 && min_shell in (1,2) &&
        isfinite(rtol) && rtol>=0 && all(x->isfinite(x)&&x>=0,atol isa Real ? [atol] : atol) &&
        max_seconds>0 && rule in (:hermite,:composite) || throw(ArgumentError("invalid direct quadrature settings"))
    start=time();rules=[difference_rule(i,rule) for i in 0:max_level]
    cache=Dict{Tuple,Vector{Float64}}();deltas=Dict{Tuple,Vector{Float64}}()
    costs=Dict{Tuple,Int}();accepted=Set{Tuple}();active=Set{Tuple}();history=NamedTuple[]
    total=Float64[];baseline=Float64[];last_error=Float64[];reason=:evaluation_budget
    last_complete=nothing
    rotation_check=nothing;rotation_evaluations=0;force_breadth=false
    evaluations=0;complete=true;depth_two_checked=false;next_progress=1000
    function evaluate_index(index)
        grid=points(index,rules)
        missing=[q for (q,_) in grid if !haskey(cache,q)]
        evaluations+length(missing)<=max_evaluations || return false
        time()-start<=max_seconds || return false
        if !isempty(missing)
            # A whole index is evaluated transactionally; no partial delta enters
            # either the integral or its frontier error estimate.
            values_=evaluate_batch===nothing ? [f(collect(q)) for q in missing] : evaluate_batch(missing)
            length(values_)==length(missing) || throw(DimensionMismatch("batch returned wrong number of values"))
            for (q,v) in zip(missing,values_)
                vv=v isa Real ? [Float64(v)] : collect(Float64,v)
                all(isfinite,vv) || error("nonfinite direct quadrature integrand at $q")
                if isempty(baseline);baseline=copy(vv);total=zeros(length(vv));last_error=fill(Inf,length(vv));end
                length(vv)==length(baseline) || throw(DimensionMismatch("integrand size changed"))
                cache[q]=vv
            end
            evaluations+=length(missing)
            if progress!==nothing && evaluations>=next_progress
                progress((event=:evaluations,evaluations=evaluations,indices=length(accepted),
                    frontier=length(active),error_ratio=Inf,normalizer=first(total),wall=time()-start))
                next_progress=evaluations+1000
            end
        end
        delta=zeros(length(baseline));compensation=zeros(length(baseline))
        subtract=any(>(0),index)
        for (q,w) in grid
            term=w.*(subtract ? cache[q]-baseline : cache[q])
            correction=term-compensation;next=delta+correction
            compensation=(next-delta)-correction;delta=next
        end
        deltas[index]=delta;costs[index]=length(missing)
        true
    end
    zero_index=ntuple(_->0,dimension)
    evaluate_index(zero_index) || error("unable to evaluate quadrature origin")
    push!(accepted,zero_index);total+=deltas[zero_index]
    # Evaluate and accept all first-order directions. This prevents coordinates
    # which only affect reported moments from being silently frozen at the mode.
    initial=[ntuple(k->Int(k==j),dimension) for j in 1:dimension]
    for index in initial
        if !evaluate_index(index);complete=false;break;end
        push!(accepted,index);total+=deltas[index]
    end
    if complete
        for index in sort(collect(accepted);by=i->(sum(i),i))
            for next in children(index,accepted,max_level)
                next in accepted || push!(active,next)
            end
        end
        # A complete initial pair shell catches interactions invisible to all
        # axial evaluations; a failed shell can never produce resolved=true.
        for index in sort(collect(active);by=i->(sum(i),i))
            if !evaluate_index(index);complete=false;break;end
        end
        depth_two_checked=complete
    end
    while complete
        last_error=zeros(length(total))
        fine=copy(total)
        for index in sort(collect(active);by=i->(sum(i),i))
            last_error+=abs.(deltas[index]);fine+=deltas[index]
        end
        last_complete=(value=fine,error=copy(last_error),indices=union(accepted,active))
        tolerance=if error_scale===nothing
            (atol isa Real ? fill(Float64(atol),length(total)) : collect(Float64,atol))+rtol.*abs.(fine)
        else
            scale=collect(Float64,error_scale(fine))
            length(scale)==length(fine)&&all(x->isfinite(x)&&x>0,scale) || error("invalid quadrature error scale")
            ((atol isa Real ? fill(Float64(atol),length(total)) : collect(Float64,atol)).+rtol).*scale
        end
        length(tolerance)==length(total) || throw(DimensionMismatch("absolute tolerance size"))
        error=max(maximum(last_error./max.(tolerance,eps(Float64))),0.)
        item=(evaluations=evaluations,indices=length(accepted),frontier=length(active),
              error_ratio=error,worst_component=argmax(last_error./max.(tolerance,eps(Float64))),
              normalizer=first(fine),wall=time()-start)
        push!(history,item);progress===nothing || progress(item)
        if all(last_error.<=tolerance) && (min_shell==1 || depth_two_checked)
            if verify_rotation && dimension>1
                if rotation_check===nothing
                    direction=sin.(Float64.(1:dimension));direction/=norm(direction)
                    rotate(q)=q-2direction*dot(direction,q)
                    batch=evaluate_batch===nothing ? nothing : qs->evaluate_batch([Tuple(rotate(collect(q))) for q in qs])
                    remaining=max_evaluations-evaluations
                    if remaining<1 || time()-start>=max_seconds
                        reason=:rotation_budget;break
                    end
                    rotation_check=integrate_normal(q->f(rotate(q)),dimension;atol,rtol,
                        max_evaluations=remaining,max_level,min_shell,
                        max_seconds=max_seconds-(time()-start),evaluate_batch=batch,
                        priority,rule,error_scale,verify_rotation=false)
                    rotation_evaluations=rotation_check.diagnostics.evaluations
                    evaluations+=rotation_evaluations
                end
                if !rotation_check.diagnostics.resolved
                    reason=:rotation_precision;break
                end
                if all(abs.(fine-rotation_check.value).<=tolerance)
                    reason=:resolution;break
                end
                # No new truth or reference posterior is used. A fixed rotation
                # of the same Gaussian integration measure exposes interactions
                # that can vanish on the original axial and pair panels.
                force_breadth=true
            else
                reason=:resolution;break
            end
        end
        isempty(active) && (reason=:level_budget;break)
        time()-start<=max_seconds || (reason=:time_budget;break)
        candidates=sort(collect(active);by=i->(sum(i),i))
        score(i)=priority===nothing ? maximum(abs.(deltas[i])./max.(tolerance,eps(Float64)))/max(costs[i],1) : priority(deltas[i],fine,costs[i])
        chosen=force_breadth ? first(candidates) : candidates[argmax(score.(candidates))]
        force_breadth=false
        delete!(active,chosen);push!(accepted,chosen);total+=deltas[chosen]
        for next in children(chosen,accepted,max_level)
            next in accepted || push!(active,next)
        end
        for index in sort([i for i in active if !haskey(deltas,i)];by=i->(sum(i),i))
            if !evaluate_index(index);complete=false;break;end
        end
    end
    if !complete
        reason=time()-start>max_seconds ? :time_budget : :evaluation_budget
        last_error=fill(Inf,length(total))
    end
    limited=any(i->any(==(max_level),i),accepted)
    # Reaching an imposed level limit cannot be called resolution merely because
    # that direction no longer has a successor in the active frontier.
    if reason==:resolution && limited;reason=:level_budget;end
    if last_complete!==nothing
        total=last_complete.value
        # A partially evaluated successor never replaces a complete estimate.
        # Its unresolved status is retained even if the old indicator was small.
        last_error=last_complete.error
    end
    diagnostics=(resolved=reason==:resolution,reason=reason,evaluations=evaluations,
        complete_frontier=complete,pair_shell_checked=depth_two_checked,
        max_level_reached=limited,wall=time()-start,history=history,
        error_is_bound=false,gaussian_remainder=false,monte_carlo=false,
        includes_evaluated_frontier=last_complete!==nothing,rule=rule)
    diagnostics=merge(diagnostics,(rotation_evaluations=rotation_evaluations,
        rotation_checked=rotation_check!==nothing,
        rotation_resolved=rotation_check===nothing ? false : rotation_check.diagnostics.resolved,
        rotation_difference=rotation_check===nothing ? nothing : maximum(abs.(total-rotation_check.value))))
    result_indices=last_complete===nothing ? accepted : last_complete.indices
    QuadratureResult(total,last_error,sort(collect(result_indices);by=i->(sum(i),i)),
        sort(collect(active);by=i->(sum(i),i)),diagnostics)
end
end
