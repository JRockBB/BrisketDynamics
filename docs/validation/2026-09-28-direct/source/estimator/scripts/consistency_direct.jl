# Statistical endpoint checks for direct deterministic inference. Randomness is
# confined to the independent declared-model cook generator, never the estimator.
isdefined(@__MODULE__,:consistency_model) || include(joinpath(@__DIR__,"consistency_control.jl"))
isdefined(@__MODULE__,:DirectPosterior) || include(joinpath(@__DIR__,"direct_posterior.jl"))
using Statistics

# JSON has no NaN/Inf values. Preserve unresolved status and use null for an
# unavailable number, rather than emitting a nonstandard JSON document.
_direct_json_value(x::AbstractFloat)=isfinite(x) ? x : nothing
_direct_json_value(x::NamedTuple)=NamedTuple{keys(x)}(map(_direct_json_value,values(x)))
_direct_json_value(x::AbstractArray)=map(_direct_json_value,x)
_direct_json_value(x::AbstractDict)=Dict(k=>_direct_json_value(v) for (k,v) in x)
_direct_json_value(x::Tuple)=map(_direct_json_value,x)
_direct_json_value(x)=x
function direct_consistency_summary(records,requested;families=5,alpha=.05,min_cooks=64)
    diagnostics_mean(rs,key)=begin
        values=[getproperty(r,key) for r in rs if getproperty(r,key)!==nothing]
        isempty(values) ? nothing : mean(values)
    end
    complete=filter(r->get(r,:completed,false),records);n=length(complete)
    n==0 && return (completed=0,requested=requested,eligible=false,passed=false,gates=NamedTuple[])
    d=length(first(complete).ranks);tests=families*2d
    bound=sqrt(log(2tests/alpha)/(2n));tail=alpha/(2tests)
    count_lo=binomial_quantile(tail,n,.95);count_hi=binomial_quantile(1-tail,n,.95)
    gates=NamedTuple[]
    for i in 1:d
        lower=sort([max(0.,r.ranks[i]-r.rank_errors[i]) for r in complete])
        upper=sort([min(1.,r.ranks[i]+r.rank_errors[i]) for r in complete])
        ks_upper=max(maximum((1:n)./n-lower),maximum(upper-(0:n-1)./n))
        ks_lower=max(0.,maximum((1:n)./n-upper),maximum(lower-(0:n-1)./n))
        push!(gates,(metric="rank CDF $i",lower=ks_lower,upper=ks_upper,limit=bound,passed=ks_upper<=bound))
        definite=sum(r.ranks[i]-r.rank_errors[i]>=.025 && r.ranks[i]+r.rank_errors[i]<=.975 for r in complete)
        possible=sum(r.ranks[i]+r.rank_errors[i]>=.025 && r.ranks[i]-r.rank_errors[i]<=.975 for r in complete)
        push!(gates,(metric="95% coverage $i",lower=definite/n,upper=possible/n,
            limits=[count_lo/n,count_hi/n],passed=count_lo<=definite && possible<=count_hi))
    end
    eligible=length(records)==requested && n==requested && n>=min_cooks
    (completed=n,requested=requested,eligible=eligible,passed=false,marginal_checks_passed=eligible&&all(g.passed for g in gates),
     qualification_missing=[:joint_data_dependent_ranks,:predictive_checks,:online_dynamic_acceptance,:held_out_cook_suites],
     gates=gates,numerical_errors_are_estimates=true,
     mean_nees_z=diagnostics_mean(complete,:nees_z),mean_nees_y=diagnostics_mean(complete,:nees_y),
     mean_gaussian_nees_y=mean(r.gaussian_nees_y for r in complete))
end

"Run exposed development cooks by default; held-out seeds require an explicit frozen run."
function run_direct_consistency(;n=4,seed0=3001,suite=:core,nx=4,
        times=[0.,120.,600.,1800.],output="direct_consistency.json",
        rank_indices=:all,query_atol=.002,query_rtol=.002,kwargs...)
    model=consistency_model(;nx,suite,stop=maximum(times));records=NamedTuple[]
    for seed in seed0:seed0+n-1
        start=time();saved=Ref{Any}()
        println("DIRECT COOK ",suite," seed=",seed);flush(stdout)
        record=try
            control_cook(model,seed;times,lag=typemax(Int),
                inspect=(sm,r,truth)->(saved[]=(sm=deepcopy(sm),gaussian=r,truth=truth)))
            frozen=saved[]
            p=direct_posterior_fit(frozen.sm;kwargs...)
            nz=length(model.blk);target=vcat(frozen.truth.z,frozen.truth.y)
            indices=rank_indices===:all ? collect(eachindex(target)) : collect(rank_indices)
            # Missing numerical summaries stay null in serialized reports.
            gaussian_error=frozen.gaussian.y-frozen.truth.y
            ranks=Float64[];rank_errors=Float64[];queries=NamedTuple[]
            if p.diagnostics.resolved
                for i in indices
                    c=direct_cdf(p,i,target[i];atol=query_atol,rtol=query_rtol)
                    push!(ranks,c.value);push!(rank_errors,c.error)
                    push!(queries,(index=i,resolved=c.resolved,error=c.error,diagnostics=c.diagnostics))
                    println("DIRECT RANK ",suite," seed=",seed," coordinate=",i," resolved=",c.resolved);flush(stdout)
                end
            end
            (seed=seed,completed=p.diagnostics.resolved&&length(ranks)==length(indices)&&all(q.resolved for q in queries),
             posterior_resolved=p.diagnostics.resolved,diagnostics=p.diagnostics,indices=indices,
             ranks=ranks,rank_errors=rank_errors,queries=queries,
             mean=p.diagnostics.moments_valid ? p.mean : nothing,
             covariance=p.diagnostics.moments_valid ? p.covariance : nothing,truth=target,
             nees_z=p.diagnostics.moments_valid ? direct_radius(p,target;indices=1:nz,allow_unresolved=true)/nz : nothing,
             nees_y=p.diagnostics.moments_valid ? direct_radius(p,target;indices=nz+1:length(target),allow_unresolved=true)/model.bl.n : nothing,
             gaussian_nees_y=dot(gaussian_error,frozen.gaussian.y_cov\gaussian_error)/model.bl.n,wall=time()-start)
        catch err
            err isa InterruptException && rethrow()
            (seed=seed,completed=false,error=sprint(showerror,err,catch_backtrace()),wall=time()-start)
        end
        push!(records,record)
        report=(backend=E.backend_fingerprint(),suite=suite,nx=nx,times=times,requested=n,seed0=seed0,
            settings=(;kwargs...),records=records,summary=direct_consistency_summary(records,n),
            purpose="declared-model statistical consistency; development unless independently frozen",
            monte_carlo_inference=false)
        write(output,E.JSON.json(_direct_json_value(report)))
        println("DIRECT FINISH ",suite," seed=",seed," completed=",record.completed," wall=",record.wall);flush(stdout)
    end
    records
end

if abspath(PROGRAM_FILE)==@__FILE__
    records=run_direct_consistency(n=length(ARGS)>0 ? parse(Int,ARGS[1]) : 4,
        seed0=length(ARGS)>1 ? parse(Int,ARGS[2]) : 3001,
        suite=length(ARGS)>2 ? Symbol(ARGS[3]) : :core,
        output=length(ARGS)>3 ? ARGS[4] : "direct_consistency.json")
    all(r.completed for r in records) || error("direct consistency study has unresolved or failed cooks")
end
